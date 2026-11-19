#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp samplerCube;

// Independent ES environment lookup and HDR composition. Greene, Environment
// Mapping and Other Applications of World Projections (1986), approximates
// incident radiance with six views around one probe. Capture includes the scene's
// opaque/cutout materials and lights, without shadows or recursive reflections.
// Native mip reduction supplies a basic roughness approximation, not GGX IBL.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 Corners[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(Corners[gl_VertexID], 0, 1);
}
#else
uniform mat4 InverseViewProjection;
uniform vec4 Viewport; // extent XY, reciprocal extent ZW
uniform vec4 Sampling; // intensity, last mip, reserved ZW
uniform uvec4 Options; // phase, material model, reserved ZW
uniform vec4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
    uniform vec4 AtmospherePlanet, AtmosphereRayleigh, AtmosphereMie, AtmosphereSun, AtmosphereSolar, AtmosphereQuality;
uniform sampler2D SceneColor;
uniform sampler2D SceneDepth;
uniform sampler2D NormalRoughness;
uniform sampler2D SpecularShininess;
uniform sampler2D DiffuseOpacity;
uniform sampler2D AmbientMetallic;
uniform sampler2D ReflectionRadiance;
uniform samplerCube EnvironmentRadiance;
out vec4 OutColor;

const uint CompositionPhase = 1u;
const uint PbrModel = 3u;
const float EmptyDepth = 1.0;
const float FiniteRayDepth = 0.5;
const float MinimumProjectionW = 1e-7;
const float MinimumVectorLengthSquared = 1e-12;
const float DielectricF0 = 0.04;
const float FresnelExponent = 5.0;
const float HalfFloatLimit = 65504.0;
const float HalfFloatRoundingGuard = 32.0;

bool NormalizeDirection(inout vec3 direction)
{
    float lengthSquared = dot(direction, direction);
    if (any(isnan(direction)) || any(isinf(direction)) || lengthSquared <= MinimumVectorLengthSquared) return false;
    direction *= inversesqrt(lengthSquared);
    return true;
}

bool Unproject(vec2 ndc, float depth, out vec3 position)
{
    // Geometry maps canonical [0,1] clip Z to GL [-1,1]; sampled window depth
    // already reverses that mapping. The inverse CPU matrix needs [0,1] here.
    vec4 homogeneous = InverseViewProjection * vec4(ndc, depth, 1);
    position = vec3(0);
    if (abs(homogeneous.w) < MinimumProjectionW) return false;
    position = homogeneous.xyz / homogeneous.w;
    return !any(isnan(position)) && !any(isinf(position));
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
    float height=max(0.0,length(position)-AtmospherePlanet.x);
    float taper=clamp(1.0-height/AtmospherePlanet.y,0.0,1.0);
    return exp(-height/vec2(AtmosphereRayleigh.w,AtmosphereMie.z))*taper;
}
vec3 AtmosphereSunTransmission(vec3 position)
{
    vec2 ground=AtmosphereInterval(position,AtmosphereSun.xyz,AtmospherePlanet.x);
    if (ground.x>0.00001) return vec3(0.0,0.0,0.0);
    float distance=max(0.0,AtmosphereInterval(position,AtmosphereSun.xyz,AtmospherePlanet.x+AtmospherePlanet.y).y);
    int count=int(AtmosphereQuality.x);
    vec2 depth=vec2(0.0,0.0);
    for (int i=0;i<count;++i)
    {
        float begin=distance*float(i*i)/float(count*count);
        float end=distance*float((i+1)*(i+1))/float(count*count);
        depth+=AtmosphereDensity(position+AtmosphereSun.xyz*((begin+end)*0.5))*(end-begin);
    }
    return exp(-min(AtmosphereRayleigh.rgb*depth.x+AtmosphereMie.y*depth.y,vec3(80.0,80.0,80.0)));
}
vec3 AtmosphereSurface(vec3 radiance,vec3 world,vec3 eye)
{
    if (AtmospherePlanet.w<0.5) return radiance;
    vec3 direction=(world-eye)*0.001;
    float distance=length(direction);
    if (distance<0.000001) return radiance;
    direction/=distance;
    vec3 origin=eye*0.001;
    origin.y+=AtmospherePlanet.x-AtmospherePlanet.z*0.001;
    if (length(origin)<AtmospherePlanet.x+0.0001) origin=normalize(origin)*(AtmospherePlanet.x+0.0001);
    vec2 interval=AtmosphereInterval(origin,direction,AtmospherePlanet.x+AtmospherePlanet.y);
    float begin=max(0.0,interval.x),end=min(distance,interval.y);
    vec2 ground=AtmosphereInterval(origin,direction,AtmospherePlanet.x);
    if (ground.x>0.0) end=min(end,ground.x);
    if (end<=begin) return radiance;
    float mu=dot(direction,AtmosphereSun.xyz),g=AtmosphereMie.w;
    float rayleighPhase=0.0596831037*(1.0+mu*mu);
    float miePhase=0.0795774715*(1.0-g*g)/pow(max(0.001,1.0+g*g-2.0*g*mu),1.5);
    int count=int(AtmosphereSolar.w);
    float stepLength=(end-begin)/float(count);
    vec3 transmission=vec3(1.0,1.0,1.0),scattered=vec3(0.0,0.0,0.0);
    for (int i=0;i<count;++i)
    {
        vec3 position=origin+direction*(begin+(float(i)+0.5)*stepLength);
        vec2 density=AtmosphereDensity(position);
        vec3 extinction=AtmosphereRayleigh.rgb*density.x+AtmosphereMie.y*density.y;
        vec3 segmentT=exp(-min(extinction*stepLength,vec3(80.0,80.0,80.0)));
        vec3 integral=(vec3(1.0,1.0,1.0)-segmentT)/max(extinction,vec3(0.0000001,0.0000001,0.0000001));
        vec3 source=(AtmosphereRayleigh.rgb*density.x*rayleighPhase+AtmosphereMie.x*density.y*miePhase)*
            AtmosphereSunTransmission(position)*AtmosphereSolar.rgb*AtmosphereSun.w;
        scattered+=transmission*source*integral;
        transmission*=segmentT;
    }
    return radiance*transmission+scattered;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Options.x == CompositionPhase)
    {
        OutColor = vec4(texelFetch(ReflectionRadiance, pixel, 0).rgb, 0);
        return;
    }
    OutColor = vec4(0);
    if (Sampling.w > 0.0)
    {
        if (texelFetch(SceneDepth,pixel,0).r < EmptyDepth)
        {
            if (Sampling.x<=0.0 && AtmospherePlanet.w<0.5) return;
            vec4 receiver=texelFetch(NormalRoughness,pixel,0);
            vec3 normal=normalize(receiver.xyz+vec3(0.0,0.000001,0.0));
            vec3 axis=abs(normal.y)<0.99 ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
            vec3 irradiance=vec3(0,0,0);
            // Cosine-weighted hemisphere quadrature estimates E/pi directly.
            if (Sampling.x>0.0) for (int i=0;i<16;++i)
            {
                float radius=sqrt((float(i)+0.5)/16.0),phi=float(i)*2.39996323;
                vec3 direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(1.0-radius*radius);
                direction.z=-direction.z;
                irradiance+=textureLod(EnvironmentRadiance,direction,min(Sampling.y,3.0)).rgb/16.0;
            }
            vec3 albedo=texelFetch(DiffuseOpacity,pixel,0).rgb;
            float metallic=clamp(texelFetch(AmbientMetallic,pixel,0).a,0.0,1.0);
            vec3 contribution=receiver.a<0.0 ? vec3(0,0,0) : irradiance*albedo*(1.0-metallic)*0.96*Sampling.x;
            vec2 surfaceClip=gl_FragCoord.xy*Viewport.zw*2.0-1.0;
            vec3 world,eye;
            if (Unproject(surfaceClip,texelFetch(SceneDepth,pixel,0).r,world) && Unproject(surfaceClip,0.0,eye))
            { vec3 scene=texelFetch(SceneColor,pixel,0).rgb; contribution=AtmosphereSurface(scene+contribution,world,eye)-scene; }
            OutColor=vec4(contribution,0); return;
        }
        vec2 clip=gl_FragCoord.xy*Viewport.zw*2.0-1.0;
        vec3 nearSky,farSky;
        if (!Unproject(clip,0.0,nearSky) || !Unproject(clip,FiniteRayDepth,farSky)) return;
        vec3 ray=farSky-nearSky; if (!NormalizeDirection(ray)) return;
        vec3 sky=textureLod(EnvironmentRadiance,vec3(ray.xy,-ray.z),0.0).rgb*Sampling.z;
        OutColor=vec4(clamp(sky,0.0,HalfFloatLimit)-texelFetch(SceneColor,pixel,0).rgb,0.0); return;
    }
    vec4 material = texelFetch(NormalRoughness, pixel, 0);
    vec3 normal = material.xyz;
    if (Sampling.x <= 0.0 || material.a < 0.0 || texelFetch(SceneDepth, pixel, 0).r >= EmptyDepth ||
        !NormalizeDirection(normal)) return;

    // Native fragment Y points up, matching canonical clip Y. Capture faces
    // separately invert raster Y to match cube T; lookup needs only world Z flip.
    vec2 ndc = gl_FragCoord.xy * Viewport.zw * 2.0 - 1.0;
    if (Options.w != 0u)
    {
        vec3 receiver;
        bool onMirror = Unproject(ndc, texelFetch(SceneDepth, pixel, 0).r, receiver) &&
            abs(dot(ReceiverPlane.xyz, receiver) + ReceiverPlane.w) <= ReceiverMinimum.w &&
            abs(dot(normal, ReceiverPlane.xyz)) >= ReceiverMaximum.w;
        const uint FiniteMirror = 2u;
        bool inAperture = Options.w != FiniteMirror ||
            (all(greaterThanEqual(receiver, ReceiverMinimum.xyz)) && all(lessThanEqual(receiver, ReceiverMaximum.xyz)));
        if (onMirror && inAperture) return;
    }
    vec3 nearPoint, interiorPoint;
    if (!Unproject(ndc, 0.0, nearPoint) || !Unproject(ndc, FiniteRayDepth, interiorPoint)) return;
    vec3 eyeRay = interiorPoint - nearPoint;
    if (!NormalizeDirection(eyeRay)) return;
    vec3 reflected = reflect(eyeRay, normal);
    vec3 direction = vec3(reflected.xy, -reflected.z);
    float mip = clamp(material.a, 0.0, 1.0) * Sampling.y;
    vec3 radiance = max(textureLod(EnvironmentRadiance, direction, mip).rgb, vec3(0));

    vec3 f0 = clamp(texelFetch(SpecularShininess, pixel, 0).rgb, 0.0, 1.0);
    if (Options.y == PbrModel)
    {
        vec3 baseColor = clamp(texelFetch(DiffuseOpacity, pixel, 0).rgb, 0.0, 1.0);
        float metallic = clamp(texelFetch(AmbientMetallic, pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, baseColor, metallic);
    }
    float cosine = clamp(dot(normal, -eyeRay), 0.0, 1.0);
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    vec4 brdfFit=clamp(material.a,0.0,1.0)*vec4(-1.0,-0.0275,-0.572,0.022)+vec4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    vec2 integrated=vec2(-1.04,1.04)*grazing+brdfFit.zw;
    vec3 fresnel = Options.y == PbrModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1.0 - f0) * pow(1.0 - cosine, FresnelExponent);
    vec3 headroom = max(vec3(HalfFloatLimit - HalfFloatRoundingGuard) - texelFetch(SceneColor, pixel, 0).rgb, vec3(0));
    OutColor = vec4(min(Sampling.x * fresnel * radiance, headroom), 0);
}
#endif
