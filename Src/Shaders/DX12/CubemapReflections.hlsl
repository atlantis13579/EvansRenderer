// Independent DX12 cubemap technique: box mip reduction, receiver lookup and
// additive HDR composition. Six scene views implement the environment-mapping
// approximation described by Greene (1986). World Z is reflected at lookup to
// match native cube storage while capture cameras retain right-handed winding.
// Per-face box mips approximate roughness; no GGX convolution.
// The producer excludes shadowing, transparent objects and recursive reflections.
cbuffer CubeParameters : register(b0)
{
    row_major float4x4 ClipToWorld;
    float4 TargetExtent; // width, height, inverse width, inverse height
    float4 ReflectionScale; // intensity, final mip level, reserved
    uint4 Operation; // composition phase, surface model, reduction face, reserved
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
    float4 AtmospherePlanet, AtmosphereRayleigh, AtmosphereMie, AtmosphereSun, AtmosphereSolar, AtmosphereQuality;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> ReceiverDepth : register(t1);
Texture2D<float4> ReceiverNormal : register(t2);
Texture2D<float4> ReceiverSpecular : register(t3);
Texture2D<float4> ReceiverDiffuse : register(t4);
Texture2D<float4> ReceiverAmbient : register(t5);
Texture2D<float4> ReflectedRadiance : register(t6);
TextureCube<float4> CapturedRadiance : register(t7);
Texture2DArray<float4> ParentFaces : register(t8); // A persistent one-mip SRV.
SamplerState CubeFilter : register(s0);

static const uint AddReflection = 1u;
static const uint MetallicRoughnessModel = 3u;
static const int ReductionWidth = 2;
static const float ReductionWeight = 1.0f / (ReductionWidth * ReductionWidth);
static const float ClearDepthValue = 1.0f;
static const float CenterOffset = 0.5f;
static const float RayInteriorDepth = 0.5f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float MinimumDirectionLengthSquared = 1.0e-12f;
static const float NonmetalReflectance = 0.04f;
static const float SchlickPower = 5.0f;
static const float HalfStorageMaximum = 65504.0f;
static const float HalfStorageRoundingMargin = 32.0f;

struct ScreenTriangle { float4 RasterPosition : SV_POSITION; };
ScreenTriangle VSMain(uint vertex : SV_VertexID)
{
    ScreenTriangle output;
    float2 uv = float2((vertex << 1u) & 2u, vertex & 2u);
    output.RasterPosition = float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1);
    return output;
}

float4 PSMip(ScreenTriangle input) : SV_TARGET
{
    // Faces have power-of-two dimensions, so every destination texel owns
    // exactly four parent texels. Accumulate in float before RGBA16F storage.
    int2 parentOrigin = int2(input.RasterPosition.xy) * ReductionWidth;
    float4 sum = 0;
    for (int row = 0; row < ReductionWidth; ++row)
        for (int column = 0; column < ReductionWidth; ++column)
            sum += ParentFaces.Load(int4(parentOrigin + int2(column, row), Operation.z, 0));
    return sum * ReductionWeight;
}

bool WorldPosition(float2 clipXY, float depth, out float3 world)
{
    float4 h = mul(ClipToWorld, float4(clipXY, depth, 1));
    world = 0;
    if (abs(h.w) < MinimumHomogeneousW) return false;
    world = h.xyz / h.w;
    return all(isfinite(world));
}
bool NormalizeFinite(inout float3 value)
{
    float squaredLength = dot(value, value);
    if (!all(isfinite(value)) || squaredLength <= MinimumDirectionLengthSquared) return false;
    value *= rsqrt(squaredLength);
    return true;
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

float4 PSMain(ScreenTriangle input) : SV_TARGET
{
    int3 address = int3(int2(input.RasterPosition.xy), 0);
    if (Operation.x == AddReflection) return float4(ReflectedRadiance.Load(address).rgb, 0);
    if (ReflectionScale.w > 0)
    {
        if (ReceiverDepth.Load(address) < ClearDepthValue)
        {
            if (ReflectionScale.x<=0.0 && AtmospherePlanet.w<0.5) return 0;
            float4 receiver=ReceiverNormal.Load(address);
            float3 normal=normalize(receiver.xyz+float3(0.0,0.000001,0.0));
            float3 axis=abs(normal.y)<0.99 ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
            float3 irradiance=float3(0,0,0);
            // Cosine-weighted hemisphere quadrature estimates E/pi directly.
            if (ReflectionScale.x>0.0) for (int i=0;i<16;++i)
            {
                float radius=sqrt((float(i)+0.5)/16.0),phi=float(i)*2.39996323;
                float3 direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(1.0-radius*radius);
                direction.z=-direction.z;
                irradiance+=CapturedRadiance.SampleLevel(CubeFilter, direction, min(ReflectionScale.y, 3.0)).rgb/16.0;
            }
            float3 albedo=ReceiverDiffuse.Load(address).rgb;
            float metallic=clamp(ReceiverAmbient.Load(address).a,0.0,1.0);
            float3 contribution=receiver.a<0.0 ? float3(0,0,0) : irradiance*albedo*(1.0-metallic)*0.96*ReflectionScale.x;
            float2 surfaceUV=(float2(address.xy)+CenterOffset)*TargetExtent.zw;
            float2 surfaceClip=float2(surfaceUV.x*2-1,1-surfaceUV.y*2);
            float3 world,eye;
            if (WorldPosition(surfaceClip,ReceiverDepth.Load(address),world) && WorldPosition(surfaceClip,0,eye))
            { float3 scene=OpaqueRadiance.Load(address).rgb; contribution=AtmosphereSurface(scene+contribution,world,eye)-scene; }
            return float4(contribution,0);
        }
        float2 skyUV=(float2(address.xy)+CenterOffset)*TargetExtent.zw;
        float2 skyClip=float2(skyUV.x*2-1,1-skyUV.y*2);
        float3 nearSky,farSky;
        if (!WorldPosition(skyClip,0,nearSky) || !WorldPosition(skyClip,RayInteriorDepth,farSky)) return 0;
        float3 ray=farSky-nearSky; if (!NormalizeFinite(ray)) return 0;
        float3 sky=CapturedRadiance.SampleLevel(CubeFilter,float3(ray.xy,-ray.z),0).rgb*ReflectionScale.z;
        return float4(clamp(sky,0,HalfStorageMaximum)-OpaqueRadiance.Load(address).rgb,0);
    }
    float4 surface = ReceiverNormal.Load(address);
    float3 normal = surface.xyz;
    if (ReflectionScale.x <= 0 || ReceiverDepth.Load(address) >= ClearDepthValue || surface.w < 0 || !NormalizeFinite(normal)) return 0;

    float2 uv = (float2(address.xy) + CenterOffset) * TargetExtent.zw;
    float2 clipXY = float2(uv.x * 2 - 1, 1 - uv.y * 2);
    if (Operation.w != 0u)
    {
        float3 receiver;
        bool planarSurface = WorldPosition(clipXY, ReceiverDepth.Load(address), receiver) &&
            abs(dot(ReceiverPlane.xyz, receiver) + ReceiverPlane.w) <= ReceiverMinimum.w &&
            abs(dot(ReceiverPlane.xyz, normal)) >= ReceiverMaximum.w;
        const uint BoundedPlanarReceiver = 2u;
        bool inside = Operation.w != BoundedPlanarReceiver ||
            (all(receiver >= ReceiverMinimum.xyz) && all(receiver <= ReceiverMaximum.xyz));
        if (planarSurface && inside) return 0;
    }
    float3 nearWorld, interiorWorld;
    if (!WorldPosition(clipXY, 0, nearWorld) || !WorldPosition(clipXY, RayInteriorDepth, interiorWorld)) return 0;
    // Two finite points also support orthographic and infinite-far cameras.
    float3 incoming = interiorWorld - nearWorld;
    if (!NormalizeFinite(incoming)) return 0;
    float3 worldReflection = incoming - 2 * dot(incoming, normal) * normal;
    float3 cubeDirection = float3(worldReflection.x, worldReflection.y, -worldReflection.z);
    float level = saturate(surface.w) * ReflectionScale.y;
    float3 incidentRadiance = max(CapturedRadiance.SampleLevel(CubeFilter, cubeDirection, level).rgb, 0);

    float3 baseReflectance = saturate(ReceiverSpecular.Load(address).rgb);
    if (Operation.y == MetallicRoughnessModel)
        baseReflectance = lerp(baseReflectance, saturate(ReceiverDiffuse.Load(address).rgb),
            saturate(ReceiverAmbient.Load(address).a));
    float viewCosine = saturate(dot(normal, -incoming));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*viewCosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = Operation.y == MetallicRoughnessModel?clamp(baseReflectance*integrated.x+integrated.y,0.0,1.0):baseReflectance + (1 - baseReflectance) * pow(1 - viewCosine, SchlickPower);
    float3 contribution = ReflectionScale.x * fresnel * incidentRadiance;
    float3 available = max(HalfStorageMaximum - HalfStorageRoundingMargin - OpaqueRadiance.Load(address).rgb, 0);
    return float4(min(contribution, available), 0);
}
