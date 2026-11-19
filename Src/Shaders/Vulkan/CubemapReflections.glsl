#version 450
// Independent Vulkan environment reflection technique. Six scene views follow
// Greene, Environment Mapping and Other Applications of World Projections (1986).
// Per-face 2x2 mip reduction approximates roughness without GGX convolution.
// Captures contain opaque/cutout direct, ambient and emitted light; nearby
// receivers retain single-probe parallax error. No authored cube is required.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 Vertices[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(Vertices[gl_VertexIndex], 0, 1);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform EnvironmentParameters
{
    mat4 ClipToWorld;
    vec4 PixelExtent;
    vec4 FilterScale; // intensity, last mip, reserved
    uvec4 Operation; // lookup/composition, shading model, reduction face, reserved
    vec4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
    vec4 AtmospherePlanet, AtmosphereRayleigh, AtmosphereMie, AtmosphereSun, AtmosphereSolar, AtmosphereQuality;
} Parameters;
layout(set = 0, binding = 8) uniform sampler PointFilter;
layout(location = 0) out vec4 OutputRadiance;

#ifdef EVANS_CUBE_MIP
layout(set = 0, binding = 11) uniform texture2DArray PreviousMipFaces;
void main()
{
    const int MipFootprint = 2;
    const float MipWeight = 1.0 / float(MipFootprint * MipFootprint);
    ivec2 origin = ivec2(gl_FragCoord.xy) * MipFootprint;
    vec4 total = vec4(0);
    // Each view exposes one parent level. A separate render target prevents
    // overlapping sampled and writable subresources; copies publish each face.
    for (int y = 0; y < MipFootprint; ++y)
        for (int x = 0; x < MipFootprint; ++x)
            total += texelFetch(sampler2DArray(PreviousMipFaces, PointFilter),
                ivec3(origin + ivec2(x, y), Parameters.Operation.z), 0);
    OutputRadiance = total * MipWeight;
}
#else
layout(set = 0, binding = 1) uniform texture2D SceneRadiance;
layout(set = 0, binding = 2) uniform texture2D SceneDepth;
layout(set = 0, binding = 3) uniform texture2D NormalRoughness;
layout(set = 0, binding = 4) uniform texture2D MaterialSpecular;
layout(set = 0, binding = 5) uniform texture2D MaterialDiffuse;
layout(set = 0, binding = 6) uniform texture2D AmbientMetallic;
layout(set = 0, binding = 7) uniform texture2D ReflectionContribution;
layout(set = 0, binding = 9) uniform sampler TrilinearFilter;
layout(set = 0, binding = 10) uniform textureCube EnvironmentFaces;

const uint AddRadiance = 1u;
const uint PbrSurface = 3u;
const float BackgroundDepth = 1.0;
const float InteriorClipDepth = 0.5;
const float HomogeneousThreshold = 1e-7;
const float DirectionThresholdSquared = 1e-12;
const float DielectricReflectance = 0.04;
const float SchlickExponent = 5.0;
const float Binary16Maximum = 65504.0;
const float Binary16RoundingMargin = 32.0;

bool FinitePosition(vec2 screen, float depth, out vec3 world)
{
    // Raster shaders invert canonical clip Y for Vulkan's top-left viewport.
    // Lookup reverses that adaptation; CPU clip depth already lies in [0,1].
    vec2 ndc = screen * Parameters.PixelExtent.zw * vec2(2, -2) + vec2(-1, 1);
    vec4 projected = Parameters.ClipToWorld * vec4(ndc, depth, 1);
    world = vec3(0);
    if (abs(projected.w) < HomogeneousThreshold) return false;
    world = projected.xyz / projected.w;
    return !any(isnan(world)) && !any(isinf(world));
}
bool UnitVector(inout vec3 value)
{
    float magnitudeSquared = dot(value, value);
    if (any(isnan(value)) || any(isinf(value)) || magnitudeSquared <= DirectionThresholdSquared) return false;
    value *= inversesqrt(magnitudeSquared);
    return true;
}

// Spherical single scattering: Coding Adventure: Atmosphere; GPU Gems 2, ch. 16.
// Metres become kilometres before sphere intersections to retain local precision.
vec2 AtmosphereInterval(vec3 origin,vec3 ray,float radius)
{
    float b=dot(origin,ray),c=dot(origin,origin)-radius*radius;
    float discriminant=b*b-c;
    if (discriminant<0.0) return vec2(-1.0,-1.0);
    float root=sqrt(discriminant);
    // The product of the roots avoids subtracting almost equal values near ground.
    float farRoot=-b+root;
    float nearRoot=abs(farRoot)>0.000001 ? c/farRoot : -b-root;
    return vec2(nearRoot,farRoot);
}
vec2 AtmosphereDensity(vec3 position)
{
    float height=max(0.0,length(position)-Parameters.AtmospherePlanet.x);
    float taper=clamp(1.0-height/Parameters.AtmospherePlanet.y,0.0,1.0);
    return exp(-height/vec2(Parameters.AtmosphereRayleigh.w,Parameters.AtmosphereMie.z))*taper;
}
vec3 AtmosphereSunTransmission(vec3 position)
{
    vec2 ground=AtmosphereInterval(position,Parameters.AtmosphereSun.xyz,Parameters.AtmospherePlanet.x);
    if (ground.x>0.00001) return vec3(0.0,0.0,0.0);
    float distance=max(0.0,AtmosphereInterval(position,Parameters.AtmosphereSun.xyz,Parameters.AtmospherePlanet.x+Parameters.AtmospherePlanet.y).y);
    int count=int(Parameters.AtmosphereQuality.x);
    vec2 depth=vec2(0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float begin=distance*float(i*i)/float(count*count);
        float end=distance*float((i+1)*(i+1))/float(count*count);
        depth+=AtmosphereDensity(position+Parameters.AtmosphereSun.xyz*((begin+end)*0.5))*(end-begin);
    }
    return exp(-min(Parameters.AtmosphereRayleigh.rgb*depth.x+Parameters.AtmosphereMie.y*depth.y,vec3(80.0,80.0,80.0)));
}
vec3 AtmosphereSurface(vec3 radiance,vec3 world,vec3 eye)
{
    if (Parameters.AtmospherePlanet.w<0.5) return radiance;
    vec3 direction=(world-eye)*0.001;
    float distance=length(direction);
    if (distance<0.000001) return radiance;
    direction/=distance;
    vec3 origin=eye*0.001;
    origin.y+=Parameters.AtmospherePlanet.x-Parameters.AtmospherePlanet.z*0.001;
    if (length(origin)<Parameters.AtmospherePlanet.x+0.0001) origin=normalize(origin)*(Parameters.AtmospherePlanet.x+0.0001);
    vec2 interval=AtmosphereInterval(origin,direction,Parameters.AtmospherePlanet.x+Parameters.AtmospherePlanet.y);
    float begin=max(0.0,interval.x),end=min(distance,interval.y);
    vec2 ground=AtmosphereInterval(origin,direction,Parameters.AtmospherePlanet.x);
    if (ground.x>0.0) end=min(end,ground.x);
    if (end<=begin) return radiance;
    float mu=dot(direction,Parameters.AtmosphereSun.xyz),g=Parameters.AtmosphereMie.w;
    float rayleighPhase=0.0596831037*(1.0+mu*mu);
    float miePhase=0.0795774715*(1.0-g*g)/pow(max(0.001,1.0+g*g-2.0*g*mu),1.5);
    int count=int(Parameters.AtmosphereSolar.w);
    float stepLength=(end-begin)/float(count);
    vec3 transmission=vec3(1.0,1.0,1.0),scattered=vec3(0.0,0.0,0.0);
    for (int i=0;i<count;++i)
    {
        vec3 position=origin+direction*(begin+(float(i)+0.5)*stepLength);
        vec2 density=AtmosphereDensity(position);
        vec3 extinction=Parameters.AtmosphereRayleigh.rgb*density.x+Parameters.AtmosphereMie.y*density.y;
        vec3 segmentT=exp(-min(extinction*stepLength,vec3(80.0,80.0,80.0)));
        vec3 integral=(vec3(1.0,1.0,1.0)-segmentT)/max(extinction,vec3(0.0000001,0.0000001,0.0000001));
        vec3 source=(Parameters.AtmosphereRayleigh.rgb*density.x*rayleighPhase+Parameters.AtmosphereMie.x*density.y*miePhase)*
            AtmosphereSunTransmission(position)*Parameters.AtmosphereSolar.rgb*Parameters.AtmosphereSun.w;
        scattered+=transmission*source*integral;
        transmission*=segmentT;
    }
    return radiance*transmission+scattered;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Parameters.Operation.x == AddRadiance)
    {
        OutputRadiance = vec4(texelFetch(sampler2D(ReflectionContribution, PointFilter), pixel, 0).rgb, 0);
        return;
    }
    OutputRadiance = vec4(0);
    if (Parameters.FilterScale.w > 0)
    {
        if (texelFetch(sampler2D(SceneDepth,PointFilter),pixel,0).r < BackgroundDepth)
        {
            if (Parameters.FilterScale.x<=0.0 && Parameters.AtmospherePlanet.w<0.5) return;
            vec4 receiver=texelFetch(sampler2D(NormalRoughness,PointFilter),pixel,0);
            vec3 normal=normalize(receiver.xyz+vec3(0.0,0.000001,0.0));
            vec3 axis=abs(normal.y)<0.99 ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
            vec3 irradiance=vec3(0,0,0);
            // Cosine-weighted hemisphere quadrature estimates E/pi directly.
            if (Parameters.FilterScale.x>0.0) for (int i=0;i<16;++i)
            {
                float radius=sqrt((float(i)+0.5)/16.0),phi=float(i)*2.39996323;
                vec3 direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(1.0-radius*radius);
                direction.z=-direction.z;
                irradiance+=textureLod(samplerCube(EnvironmentFaces,TrilinearFilter),direction,min(Parameters.FilterScale.y,3.0)).rgb/16.0;
            }
            vec3 albedo=texelFetch(sampler2D(MaterialDiffuse,PointFilter),pixel,0).rgb;
            float metallic=clamp(texelFetch(sampler2D(AmbientMetallic,PointFilter),pixel,0).a,0.0,1.0);
            vec3 contribution=receiver.a<0.0 ? vec3(0,0,0) : irradiance*albedo*(1.0-metallic)*0.96*Parameters.FilterScale.x;
            vec3 world,eye;
            if (FinitePosition(gl_FragCoord.xy,texelFetch(sampler2D(SceneDepth,PointFilter),pixel,0).r,world) && FinitePosition(gl_FragCoord.xy,0,eye))
            { vec3 scene=texelFetch(sampler2D(SceneRadiance,PointFilter),pixel,0).rgb; contribution=AtmosphereSurface(scene+contribution,world,eye)-scene; }
            OutputRadiance=vec4(contribution,0); return;
        }
        vec3 nearSky,farSky;
        if (!FinitePosition(gl_FragCoord.xy,0,nearSky) || !FinitePosition(gl_FragCoord.xy,InteriorClipDepth,farSky)) return;
        vec3 ray=farSky-nearSky; if (!UnitVector(ray)) return;
        ray.z=-ray.z;
        vec3 sky=textureLod(samplerCube(EnvironmentFaces,TrilinearFilter),ray,0).rgb*Parameters.FilterScale.z;
        OutputRadiance=vec4(clamp(sky,0,Binary16Maximum)-texelFetch(sampler2D(SceneRadiance,PointFilter),pixel,0).rgb,0);
        return;
    }
    vec4 surface = texelFetch(sampler2D(NormalRoughness, PointFilter), pixel, 0);
    vec3 normal = surface.xyz;
    if (Parameters.FilterScale.x <= 0 || surface.a < 0 || !UnitVector(normal) ||
        texelFetch(sampler2D(SceneDepth, PointFilter), pixel, 0).r >= BackgroundDepth) return;
    vec3 start, interior;
    if (Parameters.Operation.w != 0u)
    {
        vec3 receiver;
        float depth = texelFetch(sampler2D(SceneDepth, PointFilter), pixel, 0).r;
        bool planarSurface = FinitePosition(gl_FragCoord.xy, depth, receiver) &&
            abs(dot(Parameters.ReceiverPlane, vec4(receiver, 1))) <= Parameters.ReceiverMinimum.w &&
            abs(dot(normal, Parameters.ReceiverPlane.xyz)) >= Parameters.ReceiverMaximum.w;
        const uint BoundedReceiver = 2u;
        bool inside = Parameters.Operation.w != BoundedReceiver ||
            (all(greaterThanEqual(receiver, Parameters.ReceiverMinimum.xyz)) &&
             all(lessThanEqual(receiver, Parameters.ReceiverMaximum.xyz)));
        if (planarSurface && inside) return;
    }
    if (!FinitePosition(gl_FragCoord.xy, 0, start) || !FinitePosition(gl_FragCoord.xy, InteriorClipDepth, interior)) return;
    vec3 cameraRay = interior - start;
    if (!UnitVector(cameraRay)) return;
    vec3 direction = reflect(cameraRay, normal);
    direction.z = -direction.z; // Native cube axes versus right-handed scene axes.
    float roughnessLod = clamp(surface.a, 0, 1) * Parameters.FilterScale.y;
    vec3 radiance = max(textureLod(samplerCube(EnvironmentFaces, TrilinearFilter), direction, roughnessLod).rgb, vec3(0));
    vec3 f0 = clamp(texelFetch(sampler2D(MaterialSpecular, PointFilter), pixel, 0).rgb, 0, 1);
    if (Parameters.Operation.y == PbrSurface)
        f0 = mix(f0, clamp(texelFetch(sampler2D(MaterialDiffuse, PointFilter), pixel, 0).rgb, 0, 1),
            clamp(texelFetch(sampler2D(AmbientMetallic, PointFilter), pixel, 0).a, 0, 1));
    float viewCosine = clamp(dot(normal, -cameraRay), 0, 1);
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    vec4 brdfFit=clamp(surface.a,0.0,1.0)*vec4(-1.0,-0.0275,-0.572,0.022)+vec4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*viewCosine))*brdfFit.x+brdfFit.y;
    vec2 integrated=vec2(-1.04,1.04)*grazing+brdfFit.zw;
    vec3 weight = Parameters.Operation.y == PbrSurface?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1 - f0) * pow(1 - viewCosine, SchlickExponent);
    vec3 remaining = max(vec3(Binary16Maximum - Binary16RoundingMargin) - texelFetch(sampler2D(SceneRadiance, PointFilter), pixel, 0).rgb, vec3(0));
    OutputRadiance = vec4(min(Parameters.FilterScale.x * weight * radiance, remaining), 0);
}
#endif
#endif
