// DX11 cubemap reflection lookup and HDR composition, independently authored.
// Six perspective captures approximate radiance at one camera-centered probe.
// See Greene, "Environment Mapping and Other Applications of World Projections"
// (1986). Native cube face selection follows the D3D texture cube convention.
// The CPU keeps right-handed cameras; world-Z reflection aligns sampling axes.
// Roughness selects ordinary averaged mips, not a GGX-convolved environment.
// Capture excludes shadows, transparent surfaces and recursive reflections.
cbuffer CubemapReflectionConstants : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 Viewport;
    float4 Sampling; // intensity, last mip, reserved
    uint4 Options; // phase, shading model, reserved
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
    float4 AtmospherePlanet, AtmosphereRayleigh, AtmosphereMie, AtmosphereSun, AtmosphereSolar, AtmosphereQuality;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> NormalRoughness : register(t2);
Texture2D<float4> SpecularShininess : register(t3);
Texture2D<float4> DiffuseOpacity : register(t4);
Texture2D<float4> AmbientMetallic : register(t5);
Texture2D<float4> ReflectionRadiance : register(t6);
TextureCube<float4> EnvironmentRadiance : register(t7);
SamplerState TrilinearClamp : register(s0);

static const uint CompositePhase = 1u;
static const uint PhysicalShading = 3u;
static const float BackgroundDepth = 1.0f;
static const float PixelCenterOffset = 0.5f;
static const float InteriorDepth = 0.5f; // Avoid the singular far endpoint of infinite projections.
static const float ProjectionEpsilon = 1.0e-7f;
static const float MinimumSquaredLength = 1.0e-12f;
static const float DielectricF0 = 0.04f;
static const float FresnelExponent = 5.0f;
static const float MaximumHalfRadiance = 65504.0f;
static const float HalfRoundingGuard = 32.0f;

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint index : SV_VertexID)
{
    float2 corner = float2((index << 1u) & 2u, index & 2u);
    FullscreenVertex result;
    result.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
    return result;
}

bool NormalizeDirection(inout float3 direction)
{
    float lengthSquared = dot(direction, direction);
    if (!all(isfinite(direction)) || lengthSquared <= MinimumSquaredLength) return false;
    direction *= rsqrt(lengthSquared);
    return true;
}
bool CameraPoint(float2 uv, float depth, out float3 worldPosition)
{
    float2 clipXY = float2(uv.x * 2 - 1, 1 - uv.y * 2);
    float4 projected = mul(InverseViewProjection, float4(clipXY, depth, 1));
    worldPosition = 0;
    if (abs(projected.w) <= ProjectionEpsilon) return false;
    worldPosition = projected.xyz / projected.w;
    return all(isfinite(worldPosition));
}


// Spherical single scattering: Coding Adventure: Atmosphere; GPU Gems 2, ch. 16.
// Metres become kilometres before sphere intersections to retain local precision.
float2 AtmosphereInterval(float3 origin,float3 ray,float radius)
{
    float b=dot(origin,ray),c=dot(origin,origin)-radius*radius;
    float discriminant=b*b-c;
    if (discriminant<0.0) return float2(-1.0,-1.0);
    float root=sqrt(discriminant);
    // The product of the roots avoids subtracting almost equal values near ground.
    float farRoot=-b+root;
    float nearRoot=abs(farRoot)>0.000001 ? c/farRoot : -b-root;
    return float2(nearRoot,farRoot);
}
float2 AtmosphereDensity(float3 position)
{
    float height=max(0.0,length(position)-AtmospherePlanet.x);
    float taper=clamp(1.0-height/AtmospherePlanet.y,0.0,1.0);
    return exp(-height/float2(AtmosphereRayleigh.w,AtmosphereMie.z))*taper;
}
float3 AtmosphereSunTransmission(float3 position)
{
    float2 ground=AtmosphereInterval(position,AtmosphereSun.xyz,AtmospherePlanet.x);
    if (ground.x>0.00001) return float3(0.0,0.0,0.0);
    float distance=max(0.0,AtmosphereInterval(position,AtmosphereSun.xyz,AtmospherePlanet.x+AtmospherePlanet.y).y);
    int count=int(AtmosphereQuality.x);
    float2 depth=float2(0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float begin=distance*float(i*i)/float(count*count);
        float end=distance*float((i+1)*(i+1))/float(count*count);
        depth+=AtmosphereDensity(position+AtmosphereSun.xyz*((begin+end)*0.5))*(end-begin);
    }
    return exp(-min(AtmosphereRayleigh.rgb*depth.x+AtmosphereMie.y*depth.y,float3(80.0,80.0,80.0)));
}
float3 AtmosphereSurface(float3 radiance,float3 world,float3 eye)
{
    if (AtmospherePlanet.w<0.5) return radiance;
    float3 direction=(world-eye)*0.001;
    float distance=length(direction);
    if (distance<0.000001) return radiance;
    direction/=distance;
    float3 origin=eye*0.001;
    origin.y+=AtmospherePlanet.x-AtmospherePlanet.z*0.001;
    if (length(origin)<AtmospherePlanet.x+0.0001) origin=normalize(origin)*(AtmospherePlanet.x+0.0001);
    float2 interval=AtmosphereInterval(origin,direction,AtmospherePlanet.x+AtmospherePlanet.y);
    float begin=max(0.0,interval.x),end=min(distance,interval.y);
    float2 ground=AtmosphereInterval(origin,direction,AtmospherePlanet.x);
    if (ground.x>0.0) end=min(end,ground.x);
    if (end<=begin) return radiance;
    float mu=dot(direction,AtmosphereSun.xyz),g=AtmosphereMie.w;
    float rayleighPhase=0.0596831037*(1.0+mu*mu);
    float miePhase=0.0795774715*(1.0-g*g)/pow(max(0.001,1.0+g*g-2.0*g*mu),1.5);
    int count=int(AtmosphereSolar.w);
    float stepLength=(end-begin)/float(count);
    float3 transmission=float3(1.0,1.0,1.0),scattered=float3(0.0,0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float3 position=origin+direction*(begin+(float(i)+0.5)*stepLength);
        float2 density=AtmosphereDensity(position);
        float3 extinction=AtmosphereRayleigh.rgb*density.x+AtmosphereMie.y*density.y;
        float3 segmentT=exp(-min(extinction*stepLength,float3(80.0,80.0,80.0)));
        float3 integral=(float3(1.0,1.0,1.0)-segmentT)/max(extinction,float3(0.0000001,0.0000001,0.0000001));
        float3 source=(AtmosphereRayleigh.rgb*density.x*rayleighPhase+AtmosphereMie.x*density.y*miePhase)*
            AtmosphereSunTransmission(position)*AtmosphereSolar.rgb*AtmosphereSun.w;
        scattered+=transmission*source*integral;
        transmission*=segmentT;
    }
    return radiance*transmission+scattered;
}

float4 PSMain(FullscreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Options.x == CompositePhase) return float4(ReflectionRadiance.Load(int3(pixel, 0)).rgb, 0);
    if (Sampling.w > 0)
    {
        if (OpaqueDepth.Load(int3(pixel,0)) < BackgroundDepth)
        {
            if (Sampling.x<=0.0 && AtmospherePlanet.w<0.5) return 0;
            float4 receiver=NormalRoughness.Load(int3(pixel,0));
            float3 normal=normalize(receiver.xyz+float3(0.0,0.000001,0.0));
            float3 axis=abs(normal.y)<0.99 ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
            float3 irradiance=float3(0,0,0);
            // Cosine-weighted hemisphere quadrature estimates E/pi directly.
            if (Sampling.x>0.0) for (int i=0;i<16;++i)
            {
                float radius=sqrt((float(i)+0.5)/16.0),phi=float(i)*2.39996323;
                float3 direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(1.0-radius*radius);
                direction.z=-direction.z;
                irradiance+=EnvironmentRadiance.SampleLevel(TrilinearClamp, direction, min(Sampling.y, 3.0)).rgb/16.0;
            }
            float3 albedo=DiffuseOpacity.Load(int3(pixel,0)).rgb;
            float metallic=clamp(AmbientMetallic.Load(int3(pixel,0)).a,0.0,1.0);
            float3 contribution=receiver.a<0.0 ? float3(0,0,0) : irradiance*albedo*(1.0-metallic)*0.96*Sampling.x;
            float2 surfaceUV=(float2(pixel)+PixelCenterOffset)*Viewport.zw;
            float3 world,eye;
            if (CameraPoint(surfaceUV,OpaqueDepth.Load(int3(pixel,0)),world) && CameraPoint(surfaceUV,0,eye))
            { float3 scene=SceneColor.Load(int3(pixel,0)).rgb; contribution=AtmosphereSurface(scene+contribution,world,eye)-scene; }
            return float4(contribution,0);
        }
        float2 uv=(float2(pixel)+PixelCenterOffset)*Viewport.zw;
        float3 nearSky,farSky;
        if (!CameraPoint(uv,0,nearSky) || !CameraPoint(uv,InteriorDepth,farSky)) return 0;
        float3 ray=farSky-nearSky; if (!NormalizeDirection(ray)) return 0;
        float3 sky=EnvironmentRadiance.SampleLevel(TrilinearClamp,float3(ray.xy,-ray.z),0).rgb*Sampling.z;
        return float4(clamp(sky,0,MaximumHalfRadiance)-SceneColor.Load(int3(pixel,0)).rgb,0);
    }
    float4 surface = NormalRoughness.Load(int3(pixel, 0));
    if (Sampling.x <= 0 || OpaqueDepth.Load(int3(pixel, 0)) >= BackgroundDepth || surface.w < 0) return 0;
    float3 normal = surface.xyz;
    if (!NormalizeDirection(normal)) return 0;
    float2 uv = (float2(pixel) + PixelCenterOffset) * Viewport.zw;
    if (Options.w != 0u)
    {
        float3 receiver;
        bool onMirror = CameraPoint(uv, OpaqueDepth.Load(int3(pixel, 0)), receiver) &&
            abs(dot(ReceiverPlane, float4(receiver, 1))) <= ReceiverMinimum.w &&
            abs(dot(normal, ReceiverPlane.xyz)) >= ReceiverMaximum.w;
        const uint FiniteReceiver = 2u;
        if (onMirror && (Options.w != FiniteReceiver ||
            (all(receiver >= ReceiverMinimum.xyz) && all(receiver <= ReceiverMaximum.xyz)))) return 0;
    }
    float3 nearPoint, middlePoint;
    if (!CameraPoint(uv, 0, nearPoint) || !CameraPoint(uv, InteriorDepth, middlePoint)) return 0;
    float3 incident = middlePoint - nearPoint;
    if (!NormalizeDirection(incident)) return 0;
    float3 reflected = reflect(incident, normal);
    float3 nativeDirection = float3(reflected.xy, -reflected.z);
    float roughness = saturate(surface.w);
    float3 radiance = max(EnvironmentRadiance.SampleLevel(TrilinearClamp, nativeDirection, roughness * Sampling.y).rgb, 0);

    float3 f0 = saturate(SpecularShininess.Load(int3(pixel, 0)).rgb);
    if (Options.y == PhysicalShading)
    {
        float metallic = saturate(AmbientMetallic.Load(int3(pixel, 0)).a);
        float3 diffuse = saturate(DiffuseOpacity.Load(int3(pixel, 0)).rgb);
        f0 = lerp(f0, diffuse, metallic);
    }
    float cosine = saturate(dot(normal, -incident));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = Options.y == PhysicalShading?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1 - f0) * pow(1 - cosine, FresnelExponent);
    float3 contribution = radiance * fresnel * Sampling.x;
    // Leave room for half-float rounding in the later additive composition.
    float3 headroom = max(MaximumHalfRadiance - HalfRoundingGuard - SceneColor.Load(int3(pixel, 0)).rgb, 0);
    return float4(min(contribution, headroom), 0);
}
