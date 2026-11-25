#include <metal_stdlib>
using namespace metal;
// Independent Metal receiver for the Lumen froxel SH atlas. Fixed angular
// samples make the low-frequency volume stable without borrowing opaque history.
struct LumenVolumeParameters { float4 Camera[4],Inverse[4]; uint4 Grid; float4 Distribution,Atlas; };
float4 lumenVolumeProject(constant float4* rows,float4 value)
{ return float4(dot(rows[0],value),dot(rows[1],value),dot(rows[2],value),dot(rows[3],value)); }
constant float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 point,float3 direction,
    constant LumenVolumeParameters& volume,texture2d<float> image)
{
    if (volume.Grid.w==0) return 0;
    float4 projected=lumenVolumeProject(volume.Camera,float4(point,1));
    if (projected.w<=volume.Atlas.z) return 0;
    float3 ndc=projected.xyz/projected.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,0,1)),farH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=volume.Atlas.z) return 0;
    float3 origin=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-origin),volume.Distribution.x),travel=length(point-origin);
    if (extent<=volume.Atlas.z || travel>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(volume.Grid.xy),pow(saturate(travel/extent),1/volume.Distribution.y)*volume.Grid.z)-0.5;
    cell=clamp(cell,float3(0),float3(volume.Grid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(volume.Grid.z-1));
    float3 n=direction*rsqrt(max(dot(direction,direction),volume.Atlas.w));
    constexpr float SHPi=3.14159265358979323846;
    constexpr uint SHCount=9;
    float basis[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    constexpr sampler volumeFilter(coord::normalized,address::clamp_to_edge,filter::linear);
    float3 sum=0;

    constexpr uint VolumeVisibilityBit=2;
    if ((volume.Grid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=fract(cell);
        // A corner may contribute only inside its surface-free support sphere.
        // Preserve the original weights so rejected corners cannot add energy.
        for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,volume.Grid.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*volume.Grid.y+p.y; float support=image.read(uint2(p.x,row)).w; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5f)/volume.Grid.x-1,1-2*(float(p.y)+0.5f)/volume.Grid.y);
            float4 a=lumenVolumeProject(volume.Inverse,float4(clipXY,0,1)),b=lumenVolumeProject(volume.Inverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=volume.Atlas.z) continue;
            float3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; float span=length(axis);
            if (!isfinite(span) || span<=volume.Atlas.z) continue;
            float depth=pow((float(p.z)+0.5f)/volume.Grid.z,volume.Distribution.y);
            float3 probe=begin+axis*(depth*min(span,volume.Distribution.x)/span); float separation=length(point-probe);
            if (!isfinite(separation) || separation>support) continue;
            for (uint coefficient=0;coefficient<SHCount;++coefficient)
                sum+=weight*basis[coefficient]*image.read(uint2(p.x+coefficient*volume.Grid.x,row)).xyz;
        }
        return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
    }
    for (uint coefficient=0;coefficient<SHCount;++coefficient)
    {
        float x=coefficient*volume.Grid.x+cell.x+0.5;
        float2 a=float2(x,lower*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        float2 b=float2(x,upper*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        sum+=basis[coefficient]*mix(image.sample(volumeFilter,a,level(0)).xyz,image.sample(volumeFilter,b,level(0)).xyz,cell.z-lower);
    }
    return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
}


// Metal's independent dielectric implementation. A private D32 nearest-layer
// capture precedes composition from frozen HDR; the output is never sampled.
// Snell/Fresnel/Beer: PBRT 4e, Specular Reflection and Transmission; Transmittance.
// Perspective-correct tracing: https://jcgt.org/published/0003/04/04/paper.pdf
// The local parallel slab sums polarized energy but uses one spatial exit.
// Missing screen information falls back to unwarped transmission or uniform
// environment reflection, without claiming hidden geometry or recursive transport.
struct RefractionUniforms
{
    float4 World[4], NormalWorld[4], ViewProjection[4], InverseViewProjection[4];
    float4 Color, Material, Optics, Absorption, Viewport, Trace, Appearance, Environment;
    uint4 Textures, Options;
};
constant float DirectionTolerance = 1e-12, ProjectionTolerance = 1e-7, DepthTolerance = 1e-5;
constant float TexelCenter = 0.5, FarDepth = 1, MaximumRadiance = 65504, MinimumTint = 0.0001;
constant float UnpolarizedWeight = 0.5;
float4 Transform(constant float4* rows,float4 value)
{ return float4(dot(rows[0],value),dot(rows[1],value),dot(rows[2],value),dot(rows[3],value)); }
bool Unit(thread float3& value)
{
    if (!all(isfinite(value))) return false;
    float scale = max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (scale == 0) return false;
    value /= scale;
    float squared = dot(value,value);
    if (!all(isfinite(value)) || squared <= DirectionTolerance) return false;
    value *= rsqrt(squared); return true;
}
#ifdef EVANS_REFRACTION_CAPTURE
struct MeshVertex { packed_float3 Position, Normal; packed_float4 Color; packed_float2 UV; };
struct SurfaceVertex { float4 Position [[position]]; float3 WorldPosition, Normal; float4 Color; float2 UV; };
struct CaptureResult { float4 Normal [[color(0)]]; float4 Tint [[color(1)]]; };
constant float ColorThreshold = 0.04045, LinearDivisor = 12.92, ColorOffset = 0.055;
constant float ColorScale = 1.055, ColorPower = 2.4, TangentTolerance = 1e-10;
vertex SurfaceVertex VSMain(uint id [[vertex_id]],device const MeshVertex* vertices [[buffer(0)]],
    constant RefractionUniforms& data [[buffer(1)]])
{
    const MeshVertex source = vertices[id]; SurfaceVertex result;
    float4 world = Transform(data.World,float4(float3(source.Position),1));
    result.Position = Transform(data.ViewProjection,world); result.WorldPosition = world.xyz/world.w;
    result.Normal = Transform(data.NormalWorld,float4(float3(source.Normal),0)).xyz;
    float3 encoded = saturate(float4(source.Color).rgb);
    result.Color = float4(select(pow((encoded+ColorOffset)/ColorScale,float3(ColorPower)),encoded/LinearDivisor,encoded <= ColorThreshold),source.Color.w);
    result.UV = float2(source.UV.x,1-source.UV.y); return result;
}
fragment CaptureResult PSMain(SurfaceVertex input [[stage_in]],constant RefractionUniforms& data [[buffer(0)]],
    depth2d<float,access::read> opaqueDepth [[texture(0)]],texture2d<float> diffuse [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],texture2d<float> opacityMap [[texture(3)]])
{
    constexpr sampler materialSampler(coord::normalized,address::repeat,filter::linear);
    if (input.Position.z >= opaqueDepth.read(uint2(input.Position.xy))) discard_fragment();
    float4 material = input.Color*data.Color;
    if (data.Textures.x != 0) material *= diffuse.sample(materialSampler,input.UV);
    float coverage = saturate(material.a*(data.Textures.z != 0 ? opacityMap.sample(materialSampler,input.UV).r : 1));
    if (coverage <= 0 || coverage < data.Material.x) discard_fragment();
    float3 normal = input.Normal;
    if (!Unit(normal)) { normal = cross(dfdx(input.WorldPosition),dfdy(input.WorldPosition)); if (!Unit(normal)) discard_fragment(); }
    if (data.Textures.y != 0)
    {
        float3 dx = dfdx(input.WorldPosition), dy = dfdy(input.WorldPosition);
        float2 ux = dfdx(input.UV), uy = dfdy(input.UV); float determinant = ux.x*uy.y-ux.y*uy.x;
        if (abs(determinant) > TangentTolerance)
        {
            float3 tangent = (dx*uy.y-dy*ux.y)/determinant, reference = (dy*ux.x-dx*uy.x)/determinant;
            tangent -= normal*dot(normal,tangent);
            if (Unit(tangent))
            {
                float3 bitangent = cross(normal,tangent); if (dot(bitangent,reference) < 0) bitangent = -bitangent;
                float3 map = normalMap.sample(materialSampler,input.UV).xyz*2-1;
                float3 mapped = map.x*tangent+map.y*bitangent+map.z*normal;
                if (Unit(mapped)) normal = mapped;
            }
        }
    }
    CaptureResult result; result.Normal = float4(normal,1);
    result.Tint = float4(data.Textures.w != 0 ? saturate(material.rgb) : float3(1),data.Material.y > 0 ? 1 : coverage); return result;
}
#else
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex VSMain(uint id [[vertex_id]])
{ float2 corner = float2((id<<1)&2,id&2); return {float4(corner*2-1,0,1)}; }
bool Unproject(float2 uv,float depth,constant RefractionUniforms& data,thread float3& point)
{
    point = float3(0); float4 h = Transform(data.InverseViewProjection,float4(2*uv.x-1,1-2*uv.y,depth,1));
    if (abs(h.w) < ProjectionTolerance) return false;
    point = h.xyz/h.w; return all(isfinite(point));
}
bool ViewRay(float2 uv,constant RefractionUniforms& data,thread float3& ray)
{
    float3 nearPoint, interiorPoint; ray = float3(0);
    if (!Unproject(uv,0,data,nearPoint) || !Unproject(uv,TexelCenter,data,interiorPoint)) return false;
    ray = interiorPoint-nearPoint; return Unit(ray);
}
bool ReadSurface(int2 pixel,depth2d<float,access::read> depth,constant RefractionUniforms& data,
    thread float3& point,thread float3& ray)
{
    point = ray = float3(0);
    if (any(pixel < 0) || any(pixel >= int2(data.Viewport.xy))) return false;
    float sampleDepth = depth.read(uint2(pixel)); float2 uv = (float2(pixel)+TexelCenter)*data.Viewport.zw;
    return sampleDepth < FarDepth && Unproject(uv,sampleDepth,data,point) && ViewRay(uv,data,ray);
}
bool ClipPlane(float start,float end,thread float2& interval)
{
    if (start < 0 && end < 0) return false;
    if ((start < 0) != (end < 0))
    { float crossing = start/(start-end); if (start < 0) interval.x = max(interval.x,crossing); else interval.y = min(interval.y,crossing); }
    return interval.x < interval.y;
}
bool TraceScene(float3 origin,float3 direction,depth2d<float,access::read> depth,
    constant RefractionUniforms& data,thread float2& hit)
{
    hit = float2(0); float3 end = origin+direction*data.Trace.x;
    float4 h0 = Transform(data.ViewProjection,float4(origin,1)), h1 = Transform(data.ViewProjection,float4(end,1));
    if (!all(isfinite(h0)) || !all(isfinite(h1))) return false;
    float2 interval(0,1);
    if (!ClipPlane(h0.w-ProjectionTolerance,h1.w-ProjectionTolerance,interval) ||
        !ClipPlane(h0.w+h0.x,h1.w+h1.x,interval) || !ClipPlane(h0.w-h0.x,h1.w-h1.x,interval) ||
        !ClipPlane(h0.w+h0.y,h1.w+h1.y,interval) || !ClipPlane(h0.w-h0.y,h1.w-h1.y,interval) ||
        !ClipPlane(h0.z,h1.z,interval) || !ClipPlane(h0.w-h0.z,h1.w-h1.z,interval)) return false;
    float3 segment = end-origin, first = origin+segment*interval.x, last = origin+segment*interval.y;
    h0 = Transform(data.ViewProjection,float4(first,1)); h1 = Transform(data.ViewProjection,float4(last,1));
    float k0 = 1/h0.w, k1 = 1/h1.w; float3 q0 = first*k0, q1 = last*k1;
    float2 p0 = (h0.xy*k0*float2(TexelCenter,-TexelCenter)+TexelCenter)*data.Viewport.xy;
    float2 p1 = (h1.xy*k1*float2(TexelCenter,-TexelCenter)+TexelCenter)*data.Viewport.xy;
    float2 extent = abs(p1-p0); float step = min(1.0f,data.Trace.w/max(max(extent.x,extent.y),data.Trace.w));
    // Unlike reflected opaque rays, transmission must keep its origin pixel.
    for (uint index = 0; index < data.Options.x; ++index)
    {
        float begin = index*step; if (begin >= 1) break;
        float finish = min(begin+step,1.0f), midpoint = (begin+finish)*TexelCenter;
        float3 surface, cameraRay;
        if (!ReadSurface(int2(floor(mix(p0,p1,midpoint))),depth,data,surface,cameraRay)) continue;
        float n0 = dot(q0-surface*k0,cameraRay), n1 = dot(q1-surface*k1,cameraRay);
        float d0 = mix(n0,n1,begin)/mix(k0,k1,begin), d1 = mix(n0,n1,finish)/mix(k0,k1,finish);
        if (max(d0,d1) < 0 || min(d0,d1) > data.Trace.y) continue;
        float root = clamp(abs(n1-n0) > ProjectionTolerance ? -n0/(n1-n0) : midpoint,begin,finish);
        float2 pixel = mix(p0,p1,root);
        if (!ReadSurface(int2(floor(pixel)),depth,data,surface,cameraRay)) continue;
        float3 point = mix(q0,q1,root)/mix(k0,k1,root); float separation = dot(point-surface,cameraRay);
        if (separation < -DepthTolerance || separation > data.Trace.y) continue;
        hit = pixel*data.Viewport.zw; return true;
    }
    return false;
}
float3 SceneRadiance(float2 uv,texture2d<float,access::read> scene,constant RefractionUniforms& data)
{
    float2 location = uv*data.Viewport.xy-TexelCenter; int2 first = int2(floor(location)); float2 fraction = fract(location); float3 result(0);
    for (int row = 0; row < 2; ++row) for (int column = 0; column < 2; ++column)
    {
        uint2 cell = uint2(clamp(first+int2(column,row),int2(0),int2(data.Viewport.xy)-1));
        result += scene.read(cell).rgb*(column == 0 ? 1-fraction.x : fraction.x)*(row == 0 ? 1-fraction.y : fraction.y);
    }
    return max(result,float3(0));
}
float3 Incoming(float3 origin,float3 direction,float3 fallback,depth2d<float,access::read> depth,
    texture2d<float,access::read> scene,constant RefractionUniforms& data)
{
    float2 hit; if (!TraceScene(origin,direction,depth,data,hit)) return fallback;
    float2 border = min(hit,1-hit)*data.Viewport.xy;
    float fade = data.Appearance.x > 0 ? saturate(min(border.x,border.y)/data.Appearance.x) : 1;
    return mix(fallback,SceneRadiance(hit,scene,data),fade);
}
void SlabPaths(float f,float3 a,thread float3& r,thread float3& t)
{
    r = float3(1); t = float3(0); if (f >= 1) return;
    float pass = 1-f; t = a*pass*pass/((pass+f*(1-a))*(1+a*f)); r = f+a*f*t;
}
fragment float4 PSMain(ScreenVertex input [[stage_in]],constant RefractionUniforms& data [[buffer(0)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(11)]],texture2d<float> lumenAtlas [[texture(19)]],
    depth2d<float,access::read> opaqueDepth [[texture(0)]],texture2d<float,access::read> scene [[texture(4)]],
    depth2d<float,access::read> surfaceDepth [[texture(5)]],texture2d<float,access::read> normals [[texture(6)]],
    texture2d<float,access::read> tintMap [[texture(7)]])
{
    uint2 pixel = uint2(input.Position.xy); float4 background = scene.read(pixel), tint = tintMap.read(pixel);
    float depth = surfaceDepth.read(pixel); if (depth >= FarDepth || tint.a <= 0) return background;
    float2 uv = input.Position.xy*data.Viewport.zw; float3 point, incident, normal = normals.read(pixel).xyz;
    if (!Unproject(uv,depth,data,point) || !ViewRay(uv,data,incident) || !Unit(normal)) return background;
    if (dot(incident,normal) > 0) normal = -normal;
    float ci = saturate(-dot(normal,incident)), ratio = data.Optics.x/data.Optics.y;
    float3 perpendicular = cross(incident,normal); float sin2 = ratio*ratio*dot(perpendicular,perpendicular);
    float3 reflection(1), transmission(0), exitPoint = point; bool matched = data.Optics.x == data.Optics.y;
    if (matched || sin2 < 1)
    {
        float ct = matched ? ci : sqrt(max(0.0f,1-sin2));
        if (ct == 0 && data.Optics.z > 0) return background;
        float path = data.Optics.z == 0 ? 0 : data.Optics.z/ct;
        exitPoint += path*(matched ? incident : ratio*cross(normal,perpendicular)-ct*normal);
        if (!isfinite(path) || !all(isfinite(exitPoint))) return background;
        float3 a = exp(-(data.Absorption.rgb-log(max(tint.rgb,float3(MinimumTint)))/data.Optics.w)*path);
        float s = 0, p = 0;
        if (!matched)
        {
            s = (data.Optics.x*ci-data.Optics.y*ct)/(data.Optics.x*ci+data.Optics.y*ct);
            p = (data.Optics.y*ci-data.Optics.x*ct)/(data.Optics.y*ci+data.Optics.x*ct);
        }
        float3 sr,st,pr,pt; SlabPaths(s*s,a,sr,st); SlabPaths(p*p,a,pr,pt);
        reflection = (sr+pr)*UnpolarizedWeight; transmission = (st+pt)*UnpolarizedWeight;
    }
    float3 radiance(0);
    // Low-frequency volume radiance replaces only the unresolved reflection.
    float3 reflected=reflect(incident,normal);
    float3 fallback=lumenVolume.Grid.w!=0 ? LumenTranslucentIrradiance(point,reflected,lumenVolume,lumenAtlas) : data.Environment.rgb;
    if (any(reflection > 0)) radiance += reflection*Incoming(point+normal*data.Trace.z,reflected,fallback,opaqueDepth,scene,data);
    if (any(transmission > 0)) radiance += transmission*Incoming(exitPoint+incident*data.Trace.z,incident,background.rgb,opaqueDepth,scene,data);
    return float4(clamp(mix(background.rgb,radiance,saturate(tint.a)),0.0f,MaximumRadiance),background.a);
}
#endif
