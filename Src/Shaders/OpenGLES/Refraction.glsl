#version 300 es
#ifndef EVANS_VERTEX_STAGE
precision highp float;
precision highp int;
// GLES material shaders consume the GPU lighting volume as one filtered texture;
// storage buffers are needed only by the separate probe integration producer.
uniform highp mat4 LumenVolumeCamera,LumenVolumeInverse;
uniform highp uvec4 LumenVolumeCells;
uniform highp vec4 LumenVolumeDepth,LumenVolumeExtent;
uniform highp sampler2D LumenVolumeSH;
const highp float LumenDielectricReflectance=0.04;
highp vec3 LumenTranslucentIrradiance(highp vec3 position,highp vec3 normal)
{
    if (LumenVolumeCells.w==0u) return vec3(0.0);
    highp vec4 clip=LumenVolumeCamera*vec4(position,1.0);
    if (clip.w<=LumenVolumeExtent.z) return vec3(0.0);
    highp vec3 ndc=clip.xyz/clip.w;
    if (any(greaterThan(abs(ndc.xy),vec2(1.0))) || ndc.z<0.0 || ndc.z>1.0) return vec3(0.0);
    highp vec4 nearH=LumenVolumeInverse*vec4(ndc.xy,0.0,1.0),farH=LumenVolumeInverse*vec4(ndc.xy,1.0,1.0);
    if (min(abs(nearH.w),abs(farH.w))<=LumenVolumeExtent.z) return vec3(0.0);
    highp vec3 start=nearH.xyz/nearH.w;
    highp float extent=min(length(farH.xyz/farH.w-start),LumenVolumeDepth.x),distanceToStart=length(position-start);
    if (extent<=LumenVolumeExtent.z || distanceToStart>extent) return vec3(0.0);
    highp vec2 uv=vec2(ndc.x+1.0,1.0-ndc.y)*0.5;
    highp vec3 cell=vec3(uv*vec2(LumenVolumeCells.xy),pow(clamp(distanceToStart/extent,0.0,1.0),1.0/LumenVolumeDepth.y)*float(LumenVolumeCells.z))-0.5;
    cell=clamp(cell,vec3(0.0),vec3(LumenVolumeCells.xyz)-1.0);
    highp float z0=floor(cell.z),z1=min(z0+1.0,float(LumenVolumeCells.z-1u));
    highp vec3 n=normal*inversesqrt(max(dot(normal,normal),LumenVolumeExtent.w));
    const highp float IrradiancePi=3.14159265358979323846;
    const int TermCount=9;
    highp float terms[TermCount]=float[](sqrt(1.0/(4.0*IrradiancePi)),sqrt(3.0/(4.0*IrradiancePi))*n.y,
        sqrt(3.0/(4.0*IrradiancePi))*n.z,sqrt(3.0/(4.0*IrradiancePi))*n.x,sqrt(15.0/(4.0*IrradiancePi))*n.x*n.y,
        sqrt(15.0/(4.0*IrradiancePi))*n.y*n.z,sqrt(5.0/(16.0*IrradiancePi))*(3.0*n.z*n.z-1.0),
        sqrt(15.0/(4.0*IrradiancePi))*n.x*n.z,sqrt(15.0/(16.0*IrradiancePi))*(n.x*n.x-n.y*n.y));
    highp vec3 sum=vec3(0.0);

    const uint VolumeVisibilityBit=2u;
    if ((LumenVolumeCells.w&VolumeVisibilityBit)!=0u)
    {
        uvec3 base=uvec3(floor(cell)); highp vec3 fraction=fract(cell);
        // Explicit corners retain each probe's visibility certificate. Missing
        // corners stay missing instead of renormalizing SH from another side.
        for (uint corner=0u;corner<8u;++corner)
        {
            uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=min(base+bit,LumenVolumeCells.xyz-1u);
            highp vec3 weights=mix(1.0-fraction,fraction,notEqual(bit,uvec3(0u))); highp float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
            uint row=p.z*LumenVolumeCells.y+p.y; highp float support=texelFetch(LumenVolumeSH,ivec2(p.x,row),0).a; if (support<=0.0) continue;
            highp vec2 clipXY=vec2(2.0*(float(p.x)+0.5)/float(LumenVolumeCells.x)-1.0,1.0-2.0*(float(p.y)+0.5)/float(LumenVolumeCells.y));
            highp vec4 a=LumenVolumeInverse*vec4(clipXY,0.0,1.0),b=LumenVolumeInverse*vec4(clipXY,1.0,1.0);
            if (min(abs(a.w),abs(b.w))<=LumenVolumeExtent.z) continue;
            highp vec3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; highp float span=length(axis);
            if (isnan(span) || isinf(span) || span<=LumenVolumeExtent.z) continue;
            highp float depth=pow((float(p.z)+0.5)/float(LumenVolumeCells.z),LumenVolumeDepth.y);
            highp vec3 probe=begin+axis*(depth*min(span,LumenVolumeDepth.x)/span); highp float separation=length(position-probe);
            if (isnan(separation) || isinf(separation) || separation>support) continue;
            for (int band=0;band<TermCount;++band) sum+=weight*terms[band]*texelFetch(LumenVolumeSH,ivec2(p.x+uint(band)*LumenVolumeCells.x,row),0).rgb;
        }
        return max(sum,vec3(0.0))*LumenVolumeDepth.z/(IrradiancePi*LumenVolumeDepth.w);
    }
    for (int band=0;band<TermCount;++band)
    {
        highp float column=float(band)*float(LumenVolumeCells.x)+cell.x+0.5;
        highp vec2 first=vec2(column,z0*float(LumenVolumeCells.y)+cell.y+0.5)/LumenVolumeExtent.xy;
        highp vec2 second=vec2(column,z1*float(LumenVolumeCells.y)+cell.y+0.5)/LumenVolumeExtent.xy;
        sum+=terms[band]*mix(textureLod(LumenVolumeSH,first,0.0).rgb,textureLod(LumenVolumeSH,second,0.0).rgb,cell.z-z0);
    }
    return max(sum,vec3(0.0))*LumenVolumeDepth.z/(IrradiancePi*LumenVolumeDepth.w);
}

#endif

precision highp float;
precision highp int;
precision highp sampler2D;
// Independent ES refraction. GL images retain bottom-left raster coordinates;
// projection depth alone converts from the renderer's [0,1] to GL [-1,1].
// Snell/Fresnel/Beer: PBRT 4e, Specular Reflection and Transmission; Transmittance.
// Projected traversal: https://jcgt.org/published/0003/04/04/paper.pdf
// Single captured layer, local parallel slab, first spatial exit, polarized
// internal energy sum. Screen misses use unwarped transmission or uniform reflection.
uniform mat4 World, NormalWorld, ViewProjection, InverseViewProjection;
uniform vec4 Color, Material, Optics, Absorption, Viewport, Trace, Appearance, Environment;
uniform uvec4 Textures, Options;
const float DirectionThreshold = 1e-12, ProjectionThreshold = 1e-7, HitTolerance = 1e-5;
const float PixelCenter = 0.5, FarDepth = 1.0, HalfFloatMaximum = 65504.0, MinimumTint = 0.0001;
const float UnpolarizedWeight = 0.5;
bool Unit(inout vec3 v)
{
    if (any(isnan(v)) || any(isinf(v))) return false;
    float scale = max(abs(v.x),max(abs(v.y),abs(v.z)));
    if (scale == 0.0) return false;
    v /= scale;
    float squared = dot(v,v);
    if (any(isnan(v)) || any(isinf(v)) || squared <= DirectionThreshold) return false;
    v *= inversesqrt(squared); return true;
}
#ifdef EVANS_VERTEX_STAGE
#ifdef EVANS_REFRACTION_CAPTURE
layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
out vec3 WorldPosition, WorldNormal;
out vec4 VertexColor;
out vec2 UV;
const float SRGBThreshold = 0.04045, SRGBLinearDivisor = 12.92;
const float SRGBOffset = 0.055, SRGBScale = 1.055, SRGBExponent = 2.4;
vec3 Decode(vec3 encoded)
{
    encoded = clamp(encoded,0.0,1.0);
    return mix(pow((encoded+SRGBOffset)/SRGBScale,vec3(SRGBExponent)),encoded/SRGBLinearDivisor,
        lessThanEqual(encoded,vec3(SRGBThreshold)));
}
void main()
{
    vec4 position = World*vec4(InPosition,1.0);
    gl_Position = ViewProjection*position; gl_Position.z = 2.0*gl_Position.z-gl_Position.w;
    WorldPosition = position.xyz/position.w; WorldNormal = mat3(NormalWorld)*InNormal;
    VertexColor = vec4(Decode(InColor.rgb),InColor.a); UV = vec2(InUV.x,1.0-InUV.y);
}
#else
void main()
{
    vec2 corner = vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position = vec4(corner*2.0-1.0,0.0,1.0);
}
#endif
#else
uniform sampler2D OpaqueDepth;
#ifdef EVANS_REFRACTION_CAPTURE
in vec3 WorldPosition, WorldNormal;
in vec4 VertexColor;
in vec2 UV;
uniform sampler2D DiffuseMap, NormalMap, OpacityMap;
layout(location=0) out vec4 OutNormal;
layout(location=1) out vec4 OutTint;
const float TangentThreshold = 1e-10;
void main()
{
    if (gl_FragCoord.z >= texelFetch(OpaqueDepth,ivec2(gl_FragCoord.xy),0).r) discard;
    vec4 material = VertexColor*Color;
    if (Textures.x != 0u) material *= texture(DiffuseMap,UV);
    float opacity = clamp(material.a*(Textures.z != 0u ? texture(OpacityMap,UV).r : 1.0),0.0,1.0);
    if (opacity <= 0.0 || opacity < Material.x) discard;
    vec3 normal = WorldNormal;
    if (!Unit(normal)) { normal = cross(dFdx(WorldPosition),dFdy(WorldPosition)); if (!Unit(normal)) discard; }
    if (Textures.y != 0u)
    {
        vec3 dx = dFdx(WorldPosition), dy = dFdy(WorldPosition);
        vec2 ux = dFdx(UV), uy = dFdy(UV); float determinant = ux.x*uy.y-ux.y*uy.x;
        if (abs(determinant) > TangentThreshold)
        {
            vec3 tangent = (dx*uy.y-dy*ux.y)/determinant;
            vec3 reference = (dy*ux.x-dx*uy.x)/determinant;
            tangent -= normal*dot(normal,tangent);
            if (Unit(tangent))
            {
                vec3 bitangent = cross(normal,tangent); if (dot(bitangent,reference) < 0.0) bitangent = -bitangent;
                vec3 encoded = texture(NormalMap,UV).xyz*2.0-1.0;
                vec3 mapped = tangent*encoded.x+bitangent*encoded.y+normal*encoded.z;
                if (Unit(mapped)) normal = mapped;
            }
        }
    }
    OutNormal = vec4(normal,1.0);
    OutTint = vec4(Textures.w != 0u ? clamp(material.rgb,0.0,1.0) : vec3(1.0),Material.y > 0.0 ? 1.0 : opacity);
}
#else
uniform sampler2D SceneColor, SurfaceDepth, SurfaceNormal, SurfaceTint;
layout(location=0) out vec4 OutColor;
bool WorldPoint(vec2 uv,float depth,out vec3 point)
{
    point = vec3(0.0); vec4 h = InverseViewProjection*vec4(uv*2.0-1.0,depth,1.0);
    if (abs(h.w) < ProjectionThreshold) return false;
    point = h.xyz/h.w; return !any(isnan(point)) && !any(isinf(point));
}
bool CameraRay(vec2 uv,out vec3 direction)
{
    direction = vec3(0.0); vec3 nearPoint, interior;
    if (!WorldPoint(uv,0.0,nearPoint) || !WorldPoint(uv,PixelCenter,interior)) return false;
    direction = interior-nearPoint; return Unit(direction);
}
bool ReadOpaque(ivec2 cell,out vec3 point,out vec3 ray)
{
    point = ray = vec3(0.0);
    if (any(lessThan(cell,ivec2(0))) || any(greaterThanEqual(cell,ivec2(Viewport.xy)))) return false;
    float depth = texelFetch(OpaqueDepth,cell,0).r; vec2 uv = (vec2(cell)+PixelCenter)*Viewport.zw;
    return depth < FarDepth && WorldPoint(uv,depth,point) && CameraRay(uv,ray);
}
bool ClipPlane(float a,float b,inout vec2 interval)
{
    if (a < 0.0 && b < 0.0) return false;
    if ((a < 0.0) != (b < 0.0))
    { float crossing = a/(a-b); if (a < 0.0) interval.x = max(interval.x,crossing); else interval.y = min(interval.y,crossing); }
    return interval.x < interval.y;
}
bool BackgroundHit(vec3 origin,vec3 direction,out vec2 coordinate)
{
    coordinate = vec2(0.0); vec3 end = origin+direction*Trace.x;
    vec4 a = ViewProjection*vec4(origin,1.0), b = ViewProjection*vec4(end,1.0);
    if (any(isnan(a)) || any(isinf(a)) || any(isnan(b)) || any(isinf(b))) return false;
    vec2 interval = vec2(0.0,1.0);
    if (!ClipPlane(a.w-ProjectionThreshold,b.w-ProjectionThreshold,interval) ||
        !ClipPlane(a.w+a.x,b.w+b.x,interval) || !ClipPlane(a.w-a.x,b.w-b.x,interval) ||
        !ClipPlane(a.w+a.y,b.w+b.y,interval) || !ClipPlane(a.w-a.y,b.w-b.y,interval) ||
        !ClipPlane(a.z,b.z,interval) || !ClipPlane(a.w-a.z,b.w-b.z,interval)) return false;
    vec3 segment = end-origin, start = origin+segment*interval.x, finish = origin+segment*interval.y;
    a = ViewProjection*vec4(start,1.0); b = ViewProjection*vec4(finish,1.0);
    float k0 = 1.0/a.w, k1 = 1.0/b.w; vec3 q0 = start*k0, q1 = finish*k1;
    vec2 p0 = (a.xy*k0*PixelCenter+PixelCenter)*Viewport.xy, p1 = (b.xy*k1*PixelCenter+PixelCenter)*Viewport.xy;
    vec2 span = abs(p1-p0); float stepSize = min(1.0,Trace.w/max(max(span.x,span.y),Trace.w));
    // Keep the start pixel: a camera-aligned transmitted ray projects to a point.
    for (uint index = 0u; index < Options.x; ++index)
    {
        float begin = float(index)*stepSize; if (begin >= 1.0) break;
        float finishT = min(begin+stepSize,1.0), midpoint = (begin+finishT)*PixelCenter;
        vec3 surface, camera;
        if (!ReadOpaque(ivec2(floor(mix(p0,p1,midpoint))),surface,camera)) continue;
        float n0 = dot(q0-surface*k0,camera), n1 = dot(q1-surface*k1,camera);
        float z0 = mix(n0,n1,begin)/mix(k0,k1,begin), z1 = mix(n0,n1,finishT)/mix(k0,k1,finishT);
        if (max(z0,z1) < 0.0 || min(z0,z1) > Trace.y) continue;
        float root = clamp(abs(n1-n0) > ProjectionThreshold ? -n0/(n1-n0) : midpoint,begin,finishT);
        vec2 pixel = mix(p0,p1,root);
        if (!ReadOpaque(ivec2(floor(pixel)),surface,camera)) continue;
        vec3 point = mix(q0,q1,root)/mix(k0,k1,root); float separation = dot(point-surface,camera);
        if (separation < -HitTolerance || separation > Trace.y) continue;
        coordinate = pixel*Viewport.zw; return true;
    }
    return false;
}
vec3 BilinearBackground(vec2 uv)
{
    vec2 location = uv*Viewport.xy-PixelCenter; ivec2 first = ivec2(floor(location)); vec2 weight = fract(location); vec3 color = vec3(0.0);
    // Explicit reads avoid requiring float linear-filter extensions on ES.
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x)
        color += texelFetch(SceneColor,clamp(first+ivec2(x,y),ivec2(0),ivec2(Viewport.xy)-1),0).rgb*
            (x == 0 ? 1.0-weight.x : weight.x)*(y == 0 ? 1.0-weight.y : weight.y);
    return max(color,vec3(0.0));
}
vec3 Incoming(vec3 origin,vec3 direction,vec3 missing)
{
    vec2 hit; if (!BackgroundHit(origin,direction,hit)) return missing;
    vec2 border = min(hit,1.0-hit)*Viewport.xy;
    float fade = Appearance.x > 0.0 ? clamp(min(border.x,border.y)/Appearance.x,0.0,1.0) : 1.0;
    return mix(missing,BilinearBackground(hit),fade);
}
void SlabEnergy(float f,vec3 a,out vec3 r,out vec3 t)
{
    r = vec3(1.0); t = vec3(0.0); if (f >= 1.0) return;
    float pass = 1.0-f; t = a*pass*pass/((pass+f*(1.0-a))*(1.0+a*f)); r = f+a*f*t;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy); vec4 scene = texelFetch(SceneColor,pixel,0), material = texelFetch(SurfaceTint,pixel,0);
    OutColor = scene; float depth = texelFetch(SurfaceDepth,pixel,0).r;
    if (depth >= FarDepth || material.a <= 0.0) return;
    vec2 uv = gl_FragCoord.xy*Viewport.zw; vec3 position, incident, normal = texelFetch(SurfaceNormal,pixel,0).xyz;
    if (!WorldPoint(uv,depth,position) || !CameraRay(uv,incident) || !Unit(normal)) return;
    if (dot(normal,incident) > 0.0) normal = -normal;
    float ci = clamp(-dot(normal,incident),0.0,1.0), eta = Optics.x/Optics.y;
    vec3 perpendicular = cross(incident,normal); float sinT2 = eta*eta*dot(perpendicular,perpendicular);
    vec3 reflection = vec3(1.0), transmission = vec3(0.0), exitPoint = position;
    bool matched = Optics.x == Optics.y;
    if (matched || sinT2 < 1.0)
    {
        float ct = matched ? ci : sqrt(max(0.0,1.0-sinT2));
        if (ct == 0.0 && Optics.z > 0.0) return;
        float path = Optics.z == 0.0 ? 0.0 : Optics.z/ct;
        exitPoint += path*(matched ? incident : eta*cross(normal,perpendicular)-ct*normal);
        if (isnan(path) || isinf(path) || any(isnan(exitPoint)) || any(isinf(exitPoint))) return;
        vec3 attenuation = exp(-(Absorption.rgb-log(max(material.rgb,vec3(MinimumTint)))/Optics.w)*path);
        float s = 0.0, p = 0.0;
        if (!matched)
        {
            s = (Optics.x*ci-Optics.y*ct)/(Optics.x*ci+Optics.y*ct);
            p = (Optics.y*ci-Optics.x*ct)/(Optics.y*ci+Optics.x*ct);
        }
        vec3 sr,st,pr,pt; SlabEnergy(s*s,attenuation,sr,st); SlabEnergy(p*p,attenuation,pr,pt);
        reflection = (sr+pr)*UnpolarizedWeight; transmission = (st+pt)*UnpolarizedWeight;
    }
    vec3 radiance = vec3(0.0);
    // Low-frequency volume radiance replaces only the unresolved reflection.
    vec3 reflected=reflect(incident,normal);
    vec3 missing=LumenVolumeCells.w!=0u ? LumenTranslucentIrradiance(position,reflected) : Environment.rgb;
    if (any(greaterThan(reflection,vec3(0.0)))) radiance += reflection*Incoming(position+normal*Trace.z,reflected,missing);
    if (any(greaterThan(transmission,vec3(0.0)))) radiance += transmission*Incoming(exitPoint+incident*Trace.z,incident,scene.rgb);
    OutColor = vec4(clamp(mix(scene.rgb,radiance,clamp(material.a,0.0,1.0)),0.0,HalfFloatMaximum),scene.a);
}
#endif
#endif
