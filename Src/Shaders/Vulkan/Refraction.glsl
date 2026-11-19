#version 450
#ifndef EVANS_VERTEX_STAGE
// Low-frequency Lumen translucency. This native GLSL consumer reconstructs the
// nine real SH terms after clamped trilinear interpolation of the froxel grid.
layout(set=3,binding=4,std140,row_major) uniform LumenTranslucency
{
    mat4 LumenVolumeCamera,LumenVolumeInverse;
    uvec4 LumenVolumeCells;
    vec4 LumenVolumeDepth,LumenVolumeExtent;
};
layout(set=3,binding=5) uniform sampler2D LumenVolumeSH;
const float LumenDielectricReflectance=0.04;
vec3 LumenTranslucentIrradiance(vec3 position,vec3 normal)
{
    if (LumenVolumeCells.w==0u) return vec3(0);
    vec4 projected=LumenVolumeCamera*vec4(position,1);
    if (projected.w<=LumenVolumeExtent.z) return vec3(0);
    vec3 clip=projected.xyz/projected.w;
    if (any(greaterThan(abs(clip.xy),vec2(1))) || clip.z<0 || clip.z>1) return vec3(0);
    vec4 startH=LumenVolumeInverse*vec4(clip.xy,0,1),stopH=LumenVolumeInverse*vec4(clip.xy,1,1);
    if (min(abs(startH.w),abs(stopH.w))<=LumenVolumeExtent.z) return vec3(0);
    vec3 start=startH.xyz/startH.w;
    float span=min(length(stopH.xyz/stopH.w-start),LumenVolumeDepth.x),distanceToStart=length(position-start);
    if (span<=LumenVolumeExtent.z || distanceToStart>span) return vec3(0);
    vec2 uv=vec2(clip.x+1,1-clip.y)*0.5;
    vec3 grid=vec3(uv*vec2(LumenVolumeCells.xy),pow(clamp(distanceToStart/span,0,1),1/LumenVolumeDepth.y)*float(LumenVolumeCells.z))-0.5;
    grid=clamp(grid,vec3(0),vec3(LumenVolumeCells.xyz)-1);
    float first=floor(grid.z),second=min(first+1,float(LumenVolumeCells.z-1u));
    vec3 n=normal*inversesqrt(max(dot(normal,normal),LumenVolumeExtent.w));
    const float ShPi=3.14159265358979323846;
    const int Coefficients=9;
    float basis[Coefficients]=float[](sqrt(1/(4*ShPi)),sqrt(3/(4*ShPi))*n.y,sqrt(3/(4*ShPi))*n.z,sqrt(3/(4*ShPi))*n.x,
        sqrt(15/(4*ShPi))*n.x*n.y,sqrt(15/(4*ShPi))*n.y*n.z,sqrt(5/(16*ShPi))*(3*n.z*n.z-1),
        sqrt(15/(4*ShPi))*n.x*n.z,sqrt(15/(16*ShPi))*(n.x*n.x-n.y*n.y));
    vec3 incoming=vec3(0);

    const uint VolumeVisibilityBit=2u;
    if ((LumenVolumeCells.w&VolumeVisibilityBit)!=0u)
    {
        uvec3 base=uvec3(floor(grid)); vec3 fraction=fract(grid);
        // Evaluate certificates before interpolating SH. Rejected corners keep
        // their original kernel mass at zero, preventing light across surfaces.
        for (uint corner=0u;corner<8u;++corner)
        {
            uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=min(base+bit,LumenVolumeCells.xyz-1u);
            vec3 weights=mix(1.0-fraction,fraction,notEqual(bit,uvec3(0u))); float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
            uint row=p.z*LumenVolumeCells.y+p.y; float support=texelFetch(LumenVolumeSH,ivec2(p.x,row),0).a; if (support<=0.0) continue;
            vec2 clipXY=vec2(2.0*(float(p.x)+0.5)/float(LumenVolumeCells.x)-1.0,1.0-2.0*(float(p.y)+0.5)/float(LumenVolumeCells.y));
            vec4 a=LumenVolumeInverse*vec4(clipXY,0,1),b=LumenVolumeInverse*vec4(clipXY,1,1);
            if (min(abs(a.w),abs(b.w))<=LumenVolumeExtent.z) continue;
            vec3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; float span=length(axis);
            if (isnan(span) || isinf(span) || span<=LumenVolumeExtent.z) continue;
            float depth=pow((float(p.z)+0.5)/float(LumenVolumeCells.z),LumenVolumeDepth.y);
            vec3 probe=begin+axis*(depth*min(span,LumenVolumeDepth.x)/span); float separation=length(position-probe);
            if (isnan(separation) || isinf(separation) || separation>support) continue;
            for (int coefficient=0;coefficient<Coefficients;++coefficient)
                incoming+=weight*basis[coefficient]*texelFetch(LumenVolumeSH,ivec2(p.x+uint(coefficient)*LumenVolumeCells.x,row),0).xyz;
        }
        return max(incoming,vec3(0))*LumenVolumeDepth.z/(ShPi*LumenVolumeDepth.w);
    }
    for (int coefficient=0;coefficient<Coefficients;++coefficient)
    {
        float x=float(coefficient*int(LumenVolumeCells.x))+grid.x+0.5;
        vec2 a=vec2(x,first*float(LumenVolumeCells.y)+grid.y+0.5)/LumenVolumeExtent.xy;
        vec2 b=vec2(x,second*float(LumenVolumeCells.y)+grid.y+0.5)/LumenVolumeExtent.xy;
        incoming+=basis[coefficient]*mix(textureLod(LumenVolumeSH,a,0).xyz,textureLod(LumenVolumeSH,b,0).xyz,fract(grid.z));
    }
    return max(incoming,vec3(0))*LumenVolumeDepth.z/(ShPi*LumenVolumeDepth.w);
}

#endif

// Native Vulkan GLSL, compiled in memory at runtime. Capture flips clip Y;
// depth and screen reconstruction retain the renderer's [0,1] clip convention.
// PBRT 4e: Specular Reflection and Transmission; Transmittance.
// Perspective-correct DDA: https://jcgt.org/published/0003/04/04/paper.pdf
// Nearest local slab only: internal reflection energy is summed per polarization,
// with the first exit as its spatial representative. Offscreen radiance uses the
// original background for transmission and uniform environment for reflection.
layout(set=0,binding=0,std140,row_major) uniform RefractionUniforms
{
    mat4 World, NormalWorld, ViewProjection, InverseViewProjection;
    vec4 Color, Material, Optics, Absorption, Viewport, Trace, Appearance, Environment;
    uvec4 Textures, Options;
};
const float VectorTolerance = 1e-12, ClipTolerance = 1e-7, IntersectionTolerance = 1e-5;
const float TexelCenter = 0.5, ClearDepth = 1.0, RadianceLimit = 65504.0, TintLimit = 0.0001;
const float UnpolarizedWeight = 0.5;
bool NormalizeVector(inout vec3 value)
{
    if (any(isnan(value)) || any(isinf(value))) return false;
    float scale = max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (scale == 0.0) return false;
    value /= scale;
    float squared = dot(value,value);
    if (any(isnan(value)) || any(isinf(value)) || squared <= VectorTolerance) return false;
    value *= inversesqrt(squared); return true;
}
#if defined(EVANS_VERTEX_STAGE)
#if defined(EVANS_REFRACTION_CAPTURE)
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 VertexColor;
layout(location=3) in vec2 Coordinates;
layout(location=0) out vec3 WorldPosition;
layout(location=1) out vec3 WorldNormal;
layout(location=2) out vec4 LinearColor;
layout(location=3) out vec2 UV;
const float ColorBreakpoint = 0.04045, LinearScale = 12.92;
const float PowerOffset = 0.055, PowerScale = 1.055, PowerExponent = 2.4;
void main()
{
    vec4 world = World*vec4(Position,1.0);
    gl_Position = ViewProjection*world; gl_Position.y = -gl_Position.y;
    WorldPosition = world.xyz/world.w; WorldNormal = mat3(NormalWorld)*Normal;
    vec3 encoded = clamp(VertexColor.rgb,0.0,1.0);
    vec3 decoded = mix(pow((encoded+PowerOffset)/PowerScale,vec3(PowerExponent)),encoded/LinearScale,
        lessThanEqual(encoded,vec3(ColorBreakpoint)));
    LinearColor = vec4(decoded,VertexColor.a); UV = vec2(Coordinates.x,1.0-Coordinates.y);
}
#else
void main()
{
    vec2 corner = vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position = vec4(corner*2.0-1.0,0.0,1.0);
}
#endif
#else
layout(set=0,binding=1) uniform sampler MaterialSampler;
layout(set=0,binding=2) uniform texture2D OpaqueDepth;
#if defined(EVANS_REFRACTION_CAPTURE)
layout(set=0,binding=3) uniform texture2D DiffuseMap;
layout(set=0,binding=4) uniform texture2D NormalMap;
layout(set=0,binding=5) uniform texture2D OpacityMap;
layout(location=0) in vec3 WorldPosition;
layout(location=1) in vec3 WorldNormal;
layout(location=2) in vec4 LinearColor;
layout(location=3) in vec2 UV;
layout(location=0) out vec4 CapturedNormal;
layout(location=1) out vec4 CapturedTint;
const float UvTolerance = 1e-10;
void main()
{
    if (gl_FragCoord.z >= texelFetch(sampler2D(OpaqueDepth,MaterialSampler),ivec2(gl_FragCoord.xy),0).r) discard;
    vec4 material = LinearColor*Color;
    if (Textures.x != 0u) material *= texture(sampler2D(DiffuseMap,MaterialSampler),UV);
    float opacity = clamp(material.a*(Textures.z != 0u ? texture(sampler2D(OpacityMap,MaterialSampler),UV).r : 1.0),0.0,1.0);
    if (opacity <= 0.0 || opacity < Material.x) discard;
    vec3 normal = WorldNormal;
    if (!NormalizeVector(normal)) { normal = cross(dFdx(WorldPosition),dFdy(WorldPosition)); if (!NormalizeVector(normal)) discard; }
    if (Textures.y != 0u)
    {
        vec3 horizontal = dFdx(WorldPosition), vertical = dFdy(WorldPosition);
        vec2 u = dFdx(UV), v = dFdy(UV); float determinant = u.x*v.y-u.y*v.x;
        if (abs(determinant) > UvTolerance)
        {
            vec3 tangent = (horizontal*v.y-vertical*u.y)/determinant;
            vec3 sourceB = (vertical*u.x-horizontal*v.x)/determinant;
            tangent -= normal*dot(normal,tangent);
            if (NormalizeVector(tangent))
            {
                vec3 bitangent = cross(normal,tangent); if (dot(bitangent,sourceB) < 0.0) bitangent = -bitangent;
                vec3 map = texture(sampler2D(NormalMap,MaterialSampler),UV).xyz*2.0-1.0;
                vec3 mapped = map.x*tangent+map.y*bitangent+map.z*normal;
                if (NormalizeVector(mapped)) normal = mapped;
            }
        }
    }
    CapturedNormal = vec4(normal,1.0);
    CapturedTint = vec4(Textures.w == 0u ? vec3(1.0) : clamp(material.rgb,0.0,1.0),Material.y > 0.0 ? 1.0 : opacity);
}
#else
layout(set=0,binding=6) uniform texture2D SceneColor;
layout(set=0,binding=7) uniform texture2D SurfaceDepth;
layout(set=0,binding=8) uniform texture2D SurfaceNormal;
layout(set=0,binding=9) uniform texture2D SurfaceTint;
layout(location=0) out vec4 OutColor;
bool Reconstruct(vec2 uv,float depth,out vec3 point)
{
    vec4 homogeneous = InverseViewProjection*vec4(2.0*uv.x-1.0,1.0-2.0*uv.y,depth,1.0);
    point = vec3(0.0); if (abs(homogeneous.w) < ClipTolerance) return false;
    point = homogeneous.xyz/homogeneous.w; return !any(isnan(point)) && !any(isinf(point));
}
bool ViewingDirection(vec2 uv,out vec3 direction)
{
    vec3 nearPoint, insidePoint; direction = vec3(0.0);
    if (!Reconstruct(uv,0.0,nearPoint) || !Reconstruct(uv,TexelCenter,insidePoint)) return false;
    direction = insidePoint-nearPoint; return NormalizeVector(direction);
}
bool SurfaceAt(ivec2 pixel,out vec3 point,out vec3 camera)
{
    point = camera = vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Viewport.xy)))) return false;
    float depth = texelFetch(sampler2D(OpaqueDepth,MaterialSampler),pixel,0).r;
    vec2 uv = (vec2(pixel)+TexelCenter)*Viewport.zw;
    return depth < ClearDepth && Reconstruct(uv,depth,point) && ViewingDirection(uv,camera);
}
bool PlaneInterval(float begin,float end,inout vec2 interval)
{
    if (begin < 0.0 && end < 0.0) return false;
    if ((begin < 0.0) != (end < 0.0))
    { float root = begin/(begin-end); if (begin < 0.0) interval.x = max(interval.x,root); else interval.y = min(interval.y,root); }
    return interval.x < interval.y;
}
bool TraceOpaque(vec3 origin,vec3 direction,out vec2 uv)
{
    uv = vec2(0.0); vec3 end = origin+direction*Trace.x;
    vec4 startClip = ViewProjection*vec4(origin,1.0), endClip = ViewProjection*vec4(end,1.0);
    if (any(isnan(startClip)) || any(isinf(startClip)) || any(isnan(endClip)) || any(isinf(endClip))) return false;
    vec2 limits = vec2(0.0,1.0);
    if (!PlaneInterval(startClip.w-ClipTolerance,endClip.w-ClipTolerance,limits) ||
        !PlaneInterval(startClip.w+startClip.x,endClip.w+endClip.x,limits) || !PlaneInterval(startClip.w-startClip.x,endClip.w-endClip.x,limits) ||
        !PlaneInterval(startClip.w+startClip.y,endClip.w+endClip.y,limits) || !PlaneInterval(startClip.w-startClip.y,endClip.w-endClip.y,limits) ||
        !PlaneInterval(startClip.z,endClip.z,limits) || !PlaneInterval(startClip.w-startClip.z,endClip.w-endClip.z,limits)) return false;
    vec3 delta = end-origin, first = origin+limits.x*delta, last = origin+limits.y*delta;
    startClip = ViewProjection*vec4(first,1.0); endClip = ViewProjection*vec4(last,1.0);
    float k0 = 1.0/startClip.w, k1 = 1.0/endClip.w;
    vec3 q0 = first*k0, q1 = last*k1;
    vec2 p0 = (startClip.xy*k0*vec2(TexelCenter,-TexelCenter)+TexelCenter)*Viewport.xy;
    vec2 p1 = (endClip.xy*k1*vec2(TexelCenter,-TexelCenter)+TexelCenter)*Viewport.xy;
    vec2 extent = abs(p1-p0); float increment = min(1.0,Trace.w/max(max(extent.x,extent.y),Trace.w));
    // Degenerate projected segments still test one full world-space interval.
    for (uint stepIndex = 0u; stepIndex < Options.x; ++stepIndex)
    {
        float begin = float(stepIndex)*increment; if (begin >= 1.0) break;
        float finish = min(begin+increment,1.0), midpoint = (begin+finish)*TexelCenter;
        vec3 surface, camera;
        if (!SurfaceAt(ivec2(floor(mix(p0,p1,midpoint))),surface,camera)) continue;
        float n0 = dot(q0-surface*k0,camera), n1 = dot(q1-surface*k1,camera);
        float d0 = mix(n0,n1,begin)/mix(k0,k1,begin), d1 = mix(n0,n1,finish)/mix(k0,k1,finish);
        if (max(d0,d1) < 0.0 || min(d0,d1) > Trace.y) continue;
        float root = clamp(abs(n1-n0) > ClipTolerance ? -n0/(n1-n0) : midpoint,begin,finish);
        vec2 pixel = mix(p0,p1,root);
        if (!SurfaceAt(ivec2(floor(pixel)),surface,camera)) continue;
        vec3 position = mix(q0,q1,root)/mix(k0,k1,root); float separation = dot(position-surface,camera);
        if (separation < -IntersectionTolerance || separation > Trace.y) continue;
        uv = pixel*Viewport.zw; return true;
    }
    return false;
}
vec3 SceneSample(vec2 uv)
{
    vec2 pixel = uv*Viewport.xy-TexelCenter; ivec2 origin = ivec2(floor(pixel)); vec2 blend = fract(pixel); vec3 result = vec3(0.0);
    for (int row = 0; row < 2; ++row) for (int column = 0; column < 2; ++column)
    {
        ivec2 cell = clamp(origin+ivec2(column,row),ivec2(0),ivec2(Viewport.xy)-1);
        result += texelFetch(sampler2D(SceneColor,MaterialSampler),cell,0).rgb*
            (column == 0 ? 1.0-blend.x : blend.x)*(row == 0 ? 1.0-blend.y : blend.y);
    }
    return max(result,vec3(0.0));
}
vec3 IncomingRadiance(vec3 origin,vec3 direction,vec3 fallback)
{
    vec2 uv; if (!TraceOpaque(origin,direction,uv)) return fallback;
    vec2 border = min(uv,1.0-uv)*Viewport.xy;
    float fade = Appearance.x > 0.0 ? clamp(min(border.x,border.y)/Appearance.x,0.0,1.0) : 1.0;
    return mix(fallback,SceneSample(uv),fade);
}
void PolarizedPaths(float f,vec3 attenuation,out vec3 reflection,out vec3 transmission)
{
    reflection = vec3(1.0); transmission = vec3(0.0); if (f >= 1.0) return;
    float pass = 1.0-f;
    transmission = attenuation*pass*pass/((pass+f*(1.0-attenuation))*(1.0+attenuation*f));
    reflection = f+attenuation*f*transmission;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 scene = texelFetch(sampler2D(SceneColor,MaterialSampler),pixel,0);
    vec4 tint = texelFetch(sampler2D(SurfaceTint,MaterialSampler),pixel,0);
    float depth = texelFetch(sampler2D(SurfaceDepth,MaterialSampler),pixel,0).r; OutColor = scene;
    if (depth >= ClearDepth || tint.a <= 0.0) return;
    vec3 point, incident, normal = texelFetch(sampler2D(SurfaceNormal,MaterialSampler),pixel,0).xyz;
    vec2 uv = gl_FragCoord.xy*Viewport.zw;
    if (!Reconstruct(uv,depth,point) || !ViewingDirection(uv,incident) || !NormalizeVector(normal)) return;
    if (dot(incident,normal) > 0.0) normal = -normal;
    float ci = clamp(-dot(incident,normal),0.0,1.0), ratio = Optics.x/Optics.y;
    vec3 crossDirection = cross(incident,normal); float sin2 = ratio*ratio*dot(crossDirection,crossDirection);
    vec3 r = vec3(1.0), t = vec3(0.0), exitPoint = point; bool matched = Optics.x == Optics.y;
    if (matched || sin2 < 1.0)
    {
        float ct = matched ? ci : sqrt(max(0.0,1.0-sin2));
        if (ct == 0.0 && Optics.z > 0.0) return;
        float path = Optics.z == 0.0 ? 0.0 : Optics.z/ct;
        exitPoint += path*(matched ? incident : ratio*cross(normal,crossDirection)-ct*normal);
        if (isnan(path) || isinf(path) || any(isnan(exitPoint)) || any(isinf(exitPoint))) return;
        vec3 a = exp(-(Absorption.rgb-log(max(tint.rgb,vec3(TintLimit)))/Optics.w)*path);
        float s = 0.0, p = 0.0;
        if (!matched)
        {
            s = (Optics.x*ci-Optics.y*ct)/(Optics.x*ci+Optics.y*ct);
            p = (Optics.y*ci-Optics.x*ct)/(Optics.y*ci+Optics.x*ct);
        }
        vec3 sr,st,pr,pt; PolarizedPaths(s*s,a,sr,st); PolarizedPaths(p*p,a,pr,pt);
        r = (sr+pr)*UnpolarizedWeight; t = (st+pt)*UnpolarizedWeight;
    }
    vec3 radiance = vec3(0.0);
    // Low-frequency volume radiance replaces only the unresolved reflection.
    vec3 reflected=reflect(incident,normal);
    vec3 fallback=LumenVolumeCells.w!=0u ? LumenTranslucentIrradiance(point,reflected) : Environment.rgb;
    if (any(greaterThan(r,vec3(0.0)))) radiance += r*IncomingRadiance(point+normal*Trace.z,reflected,fallback);
    if (any(greaterThan(t,vec3(0.0)))) radiance += t*IncomingRadiance(exitPoint+incident*Trace.z,incident,scene.rgb);
    OutColor = vec4(clamp(mix(scene.rgb,radiance,clamp(tint.a,0.0,1.0)),0.0,RadianceLimit),scene.a);
}
#endif
#endif
