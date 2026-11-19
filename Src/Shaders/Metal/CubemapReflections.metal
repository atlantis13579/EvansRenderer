#include <metal_stdlib>
using namespace metal;

// Independent Metal cube lookup and additive HDR composition. Greene,
// Environment Mapping and Other Applications of World Projections (1986),
// approximates directional radiance by six views around one probe. Captures
// reuse opaque/cutout scene materials, direct light, ambient light and emission.
// Native mip filtering approximates roughness without GGX convolution; nearby
// geometry retains parallax error. Capture excludes shadows and transparency.
struct CubeSettings
{
    float4 WorldFromClipRows[4];
    float4 TargetSize; // extent and reciprocal extent
    float4 ReflectionScale; // intensity, highest mip, reserved
    uint4 Operation; // composition phase, surface model, reserved
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
    float4 AtmospherePlanet, AtmosphereRayleigh, AtmosphereMie, AtmosphereSun, AtmosphereSolar, AtmosphereQuality;
};
struct CubeVertex { float4 PixelPosition [[position]]; };

constant uint CompositeOperation = 1;
constant uint MetallicRoughnessModel = 3;
constant float ClearDepth = 1.0;
constant float InteriorRayDepth = 0.5;
constant float HomogeneousTolerance = 1e-7;
constant float MinimumDirectionSquared = 1e-12;
constant float DielectricF0 = 0.04;
constant float SchlickExponent = 5.0;
constant float MaximumHalfRadiance = 65504.0;
constant float HalfRoundingMargin = 32.0;

vertex CubeVertex CubeVertexMain(uint index [[vertex_id]])
{
    const float OversizedTriangle = 3.0;
    float2 corner = index == 0 ? float2(-1, -1) :
        index == 1 ? float2(OversizedTriangle, -1) : float2(-1, OversizedTriangle);
    return {float4(corner, 0, 1)};
}

bool CubeWorldPoint(float2 pixelCenter, float depth, constant CubeSettings& settings, thread float3& world)
{
    // Explicit rows match the CPU matrix. Metal's top-left fragment origin needs
    // Y inversion when reconstructing canonical clip coordinates; depth is [0,1].
    float2 uv = pixelCenter * settings.TargetSize.zw;
    float4 clip(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    float4 h(dot(settings.WorldFromClipRows[0], clip), dot(settings.WorldFromClipRows[1], clip),
        dot(settings.WorldFromClipRows[2], clip), dot(settings.WorldFromClipRows[3], clip));
    world = float3(0);
    if (abs(h.w) < HomogeneousTolerance) return false;
    world = h.xyz / h.w;
    return all(isfinite(world));
}
bool CubeUnitVector(thread float3& vector)
{
    float squared = dot(vector, vector);
    if (!all(isfinite(vector)) || squared <= MinimumDirectionSquared) return false;
    vector *= rsqrt(squared);
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
float2 AtmosphereDensity(float3 position,constant CubeSettings& settings)
{
    float height=max(0.0,length(position)-settings.AtmospherePlanet.x);
    float taper=clamp(1.0-height/settings.AtmospherePlanet.y,0.0,1.0);
    return exp(-height/float2(settings.AtmosphereRayleigh.w,settings.AtmosphereMie.z))*taper;
}
float3 AtmosphereSunTransmission(float3 position,constant CubeSettings& settings)
{
    float2 ground=AtmosphereInterval(position,settings.AtmosphereSun.xyz,settings.AtmospherePlanet.x);
    if (ground.x>0.00001) return float3(0.0,0.0,0.0);
    float distance=max(0.0,AtmosphereInterval(position,settings.AtmosphereSun.xyz,settings.AtmospherePlanet.x+settings.AtmospherePlanet.y).y);
    int count=int(settings.AtmosphereQuality.x);
    float2 depth=float2(0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float begin=distance*float(i*i)/float(count*count);
        float end=distance*float((i+1)*(i+1))/float(count*count);
        depth+=AtmosphereDensity(position+settings.AtmosphereSun.xyz*((begin+end)*0.5),settings)*(end-begin);
    }
    return exp(-min(settings.AtmosphereRayleigh.rgb*depth.x+settings.AtmosphereMie.y*depth.y,float3(80.0,80.0,80.0)));
}
float3 AtmosphereSurface(float3 radiance,float3 world,float3 eye,constant CubeSettings& settings)
{
    if (settings.AtmospherePlanet.w<0.5) return radiance;
    float3 direction=(world-eye)*0.001;
    float distance=length(direction);
    if (distance<0.000001) return radiance;
    direction/=distance;
    float3 origin=eye*0.001;
    origin.y+=settings.AtmospherePlanet.x-settings.AtmospherePlanet.z*0.001;
    if (length(origin)<settings.AtmospherePlanet.x+0.0001) origin=normalize(origin)*(settings.AtmospherePlanet.x+0.0001);
    float2 interval=AtmosphereInterval(origin,direction,settings.AtmospherePlanet.x+settings.AtmospherePlanet.y);
    float begin=max(0.0,interval.x),end=min(distance,interval.y);
    float2 ground=AtmosphereInterval(origin,direction,settings.AtmospherePlanet.x);
    if (ground.x>0.0) end=min(end,ground.x);
    if (end<=begin) return radiance;
    float mu=dot(direction,settings.AtmosphereSun.xyz),g=settings.AtmosphereMie.w;
    float rayleighPhase=0.0596831037*(1.0+mu*mu);
    float miePhase=0.0795774715*(1.0-g*g)/pow(max(0.001,1.0+g*g-2.0*g*mu),1.5);
    int count=int(settings.AtmosphereSolar.w);
    float stepLength=(end-begin)/float(count);
    float3 transmission=float3(1.0,1.0,1.0),scattered=float3(0.0,0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float3 position=origin+direction*(begin+(float(i)+0.5)*stepLength);
        float2 density=AtmosphereDensity(position,settings);
        float3 extinction=settings.AtmosphereRayleigh.rgb*density.x+settings.AtmosphereMie.y*density.y;
        float3 segmentT=exp(-min(extinction*stepLength,float3(80.0,80.0,80.0)));
        float3 integral=(float3(1.0,1.0,1.0)-segmentT)/max(extinction,float3(0.0000001,0.0000001,0.0000001));
        float3 source=(settings.AtmosphereRayleigh.rgb*density.x*rayleighPhase+settings.AtmosphereMie.x*density.y*miePhase)*
            AtmosphereSunTransmission(position,settings)*settings.AtmosphereSolar.rgb*settings.AtmosphereSun.w;
        scattered+=transmission*source*integral;
        transmission*=segmentT;
    }
    return radiance*transmission+scattered;
}

fragment float4 CubeFragmentMain(CubeVertex input [[stage_in]],
    constant CubeSettings& settings [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> normalRoughness [[texture(2)]],
    texture2d<float, access::read> materialSpecular [[texture(3)]],
    texture2d<float, access::read> materialDiffuse [[texture(4)]],
    texture2d<float, access::read> ambientMetallic [[texture(5)]],
    texture2d<float, access::read> reflection [[texture(6)]],
    texturecube<float, access::sample> environment [[texture(7)]])
{
    uint2 pixel(input.PixelPosition.xy);
    if (settings.Operation.x == CompositeOperation) return float4(reflection.read(pixel).rgb, 0);
    if (settings.ReflectionScale.w > 0)
    {
        if (depth.read(pixel) < ClearDepth)
        {
            if (settings.ReflectionScale.x<=0.0 && settings.AtmospherePlanet.w<0.5) return float4(0);
            float4 receiver=normalRoughness.read(pixel);
            float3 normal=normalize(receiver.xyz+float3(0.0,0.000001,0.0));
            float3 axis=abs(normal.y)<0.99 ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
            float3 irradiance=float3(0,0,0);
            constexpr sampler IrradianceFilter(coord::normalized,address::clamp_to_edge,filter::linear,mip_filter::linear);
            // Cosine-weighted hemisphere quadrature estimates E/pi directly.
            if (settings.ReflectionScale.x>0.0) for (int i=0;i<16;++i)
            {
                float radius=sqrt((float(i)+0.5)/16.0),phi=float(i)*2.39996323;
                float3 direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(1.0-radius*radius);
                direction.z=-direction.z;
                irradiance+=environment.sample(IrradianceFilter,direction,level(min(settings.ReflectionScale.y,3.0f))).rgb/16.0;
            }
            float3 albedo=materialDiffuse.read(pixel).rgb;
            float metallic=clamp(ambientMetallic.read(pixel).a,0.0,1.0);
            float3 contribution=receiver.a<0.0 ? float3(0,0,0) : irradiance*albedo*(1.0-metallic)*0.96*settings.ReflectionScale.x;
            float3 world,eye;
            if (CubeWorldPoint(input.PixelPosition.xy,depth.read(pixel),settings,world) && CubeWorldPoint(input.PixelPosition.xy,0,settings,eye))
            { float3 sceneRadiance=scene.read(pixel).rgb; contribution=AtmosphereSurface(sceneRadiance+contribution,world,eye,settings)-sceneRadiance; }
            return float4(contribution,0);
        }
        float3 nearSky,farSky;
        if (!CubeWorldPoint(input.PixelPosition.xy,0,settings,nearSky) ||
            !CubeWorldPoint(input.PixelPosition.xy,InteriorRayDepth,settings,farSky)) return float4(0);
        float3 ray=farSky-nearSky; if (!CubeUnitVector(ray)) return float4(0);
        constexpr sampler SkyFilter(coord::normalized,address::clamp_to_edge,filter::linear,mip_filter::linear);
        float3 sky=environment.sample(SkyFilter,float3(ray.xy,-ray.z),level(0.0f)).rgb*settings.ReflectionScale.z;
        return float4(clamp(sky,0.0f,MaximumHalfRadiance)-scene.read(pixel).rgb,0);
    }
    float4 surface = normalRoughness.read(pixel);
    float3 normal = surface.xyz;
    if (settings.ReflectionScale.x <= 0 || depth.read(pixel) >= ClearDepth || surface.a < 0 || !CubeUnitVector(normal)) return float4(0);

    float3 nearPoint, interiorPoint;
    if (settings.Operation.w != 0)
    {
        float3 receiver;
        bool onPlane = CubeWorldPoint(input.PixelPosition.xy, depth.read(pixel), settings, receiver) &&
            abs(dot(settings.ReceiverPlane, float4(receiver, 1))) <= settings.ReceiverMinimum.w &&
            abs(dot(normal, settings.ReceiverPlane.xyz)) >= settings.ReceiverMaximum.w;
        constexpr uint FiniteReceiver = 2;
        bool bounded = settings.Operation.w == FiniteReceiver;
        if (onPlane && (!bounded || (all(receiver >= settings.ReceiverMinimum.xyz) &&
            all(receiver <= settings.ReceiverMaximum.xyz)))) return float4(0);
    }
    if (!CubeWorldPoint(input.PixelPosition.xy, 0, settings, nearPoint) ||
        !CubeWorldPoint(input.PixelPosition.xy, InteriorRayDepth, settings, interiorPoint)) return float4(0);
    // Two finite positions support orthographic and infinite-far projections.
    float3 cameraRay = interiorPoint - nearPoint;
    if (!CubeUnitVector(cameraRay)) return float4(0);
    float3 reflected = reflect(cameraRay, normal);
    float3 nativeDirection(reflected.x, reflected.y, -reflected.z);
    constexpr sampler CubeFilter(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    float mipLevel = saturate(surface.a) * settings.ReflectionScale.y;
    float3 incoming = max(environment.sample(CubeFilter, nativeDirection, level(mipLevel)).rgb, float3(0));

    float3 f0 = saturate(materialSpecular.read(pixel).rgb);
    if (settings.Operation.y == MetallicRoughnessModel)
        f0 = mix(f0, saturate(materialDiffuse.read(pixel).rgb), saturate(ambientMetallic.read(pixel).a));
    float viewCosine = saturate(dot(normal, -cameraRay));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*viewCosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = settings.Operation.y == MetallicRoughnessModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1 - f0) * pow(1 - viewCosine, SchlickExponent);
    float3 available = max(float3(MaximumHalfRadiance - HalfRoundingMargin) - scene.read(pixel).rgb, float3(0));
    return float4(min(settings.ReflectionScale.x * fresnel * incoming, available), 0);
}
