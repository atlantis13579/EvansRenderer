// Native D3D12 translucent Lumen receiver. The atlas contains signed, cosine-
// convolved SH; division by pi below applies the diffuse BRDF exactly once.
cbuffer LumenTranslucencyParameters : register(b9)
{
    row_major float4x4 TranslucentCamera,TranslucentInverse;
    uint4 TranslucentCells;
    float4 TranslucentDepthScale,TranslucentImageSize;
};
Texture2D<float4> TranslucentSH : register(t19);
SamplerState TranslucentFilter : register(s2);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 position,float3 n)
{
    if (TranslucentCells.w==0) return 0;
    float4 homogeneous=mul(TranslucentCamera,float4(position,1));
    if (homogeneous.w<=TranslucentImageSize.z) return 0;
    float3 screen=homogeneous.xyz/homogeneous.w;
    if (screen.z<0 || screen.z>1 || any(abs(screen.xy)>1)) return 0;
    float4 originH=mul(TranslucentInverse,float4(screen.xy,0,1)),endH=mul(TranslucentInverse,float4(screen.xy,1,1));
    if (min(abs(originH.w),abs(endH.w))<=TranslucentImageSize.z) return 0;
    float3 origin=originH.xyz/originH.w;
    float span=min(length(endH.xyz/endH.w-origin),TranslucentDepthScale.x),travel=length(position-origin);
    if (span<=TranslucentImageSize.z || travel>span) return 0;
    float2 screenUV=float2(screen.x*0.5+0.5,0.5-screen.y*0.5);
    float3 location=float3(screenUV*TranslucentCells.xy,pow(saturate(travel/span),rcp(TranslucentDepthScale.y))*TranslucentCells.z)-0.5;
    location=clamp(location,0,float3(TranslucentCells.xyz)-1);
    float z0=floor(location.z),z1=min(z0+1,float(TranslucentCells.z-1));
    n*=rsqrt(max(dot(n,n),TranslucentImageSize.w));
    static const float SHPi=3.14159265358979323846;
    static const uint SHCount=9;
    float sh[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    float3 result=0;

    static const uint VolumeVisibilityBit=2;
    if ((TranslucentCells.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(location)); float3 fraction=frac(location);
        // Unsupported corners keep zero weight in the original kernel; a
        // surviving bright corner must not acquire the missing corner's mass.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,TranslucentCells.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*TranslucentCells.y+p.y; float radius=TranslucentSH.Load(int3(p.x,row,0)).w; if (radius<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/TranslucentCells.x-1,1-2*(float(p.y)+0.5)/TranslucentCells.y);
            float4 a=mul(TranslucentInverse,float4(clipXY,0,1)),b=mul(TranslucentInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=TranslucentImageSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float spanToFar=length(axis);
            if (!isfinite(spanToFar) || spanToFar<=TranslucentImageSize.z) continue;
            float depth=pow((float(p.z)+0.5)/TranslucentCells.z,TranslucentDepthScale.y);
            float3 probe=start+axis*(depth*min(spanToFar,TranslucentDepthScale.x)/spanToFar);
            float separation=length(position-probe); if (!isfinite(separation) || separation>radius) continue;
            [unroll] for (uint band=0;band<SHCount;++band)
                result+=weight*sh[band]*TranslucentSH.Load(int3(p.x+band*TranslucentCells.x,row,0)).xyz;
        }
        return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
    }
    [unroll] for (uint band=0;band<SHCount;++band)
    {
        float column=band*TranslucentCells.x+location.x+0.5;
        float2 low=float2(column,z0*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        float2 high=float2(column,z1*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        result+=sh[band]*lerp(TranslucentSH.SampleLevel(TranslucentFilter,low,0).xyz,
            TranslucentSH.SampleLevel(TranslucentFilter,high,0).xyz,location.z-z0);
    }
    return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
}

// Native DX12 two-stage refraction. Capture owns nearest dielectric geometry;
// composition reads separate frozen opaque radiance and captured D32 depth.
// PBRT 4e: Specular Reflection and Transmission; Transmittance (Snell/Fresnel/Beer).
// McGuire & Mara, JCGT 2014: https://jcgt.org/published/0003/04/04/paper.pdf
// Parallel-slab energy includes polarized internal bounces. Only the first exit
// is spatially traced. A screen-space miss retains the original background for
// transmission or a uniform environment for reflection; no hidden layer is known.
cbuffer DielectricData : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, InverseViewProjection;
    float4 Color, Material, Optics, Absorption, Viewport, Trace, Appearance, Environment;
    uint4 Textures, Options;
};
Texture2D<float> OpaqueDepth:register(t0);
static const float MinimumLengthSquared = 1e-12, HomogeneousTolerance = 1e-7;
static const float SurfaceTolerance = 1e-5, CenterOffset = 0.5, BackgroundDepth = 1;
static const float MaxHalf = 65504, TintFloor = 0.0001;
static const float UnpolarizedWeight = 0.5;
bool NormalizeDirection(inout float3 direction)
{
    if (!all(isfinite(direction))) return false;
    float scale = max(abs(direction.x),max(abs(direction.y),abs(direction.z)));
    if (scale == 0) return false;
    direction /= scale;
    float squared = dot(direction,direction);
    if (!all(isfinite(direction)) || squared <= MinimumLengthSquared) return false;
    direction *= rsqrt(squared); return true;
}
#if defined(EVANS_REFRACTION_CAPTURE)
Texture2D<float4> DiffuseMap:register(t1), NormalMap:register(t2), OpacityMap:register(t3);
SamplerState MaterialSampler:register(s0);
static const float TransferThreshold = 0.04045, LinearDivisor = 12.92, TransferOffset = 0.055;
static const float TransferScale = 1.055, TransferExponent = 2.4, UvDeterminantTolerance = 1e-10;
float Decode(float value)
{ value = saturate(value); return value <= TransferThreshold ? value/LinearDivisor : pow((value+TransferOffset)/TransferScale,TransferExponent); }
struct InputVertex { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR0; float2 UV:TEXCOORD0; };
struct Surface { float4 Position:SV_POSITION; float3 Point:TEXCOORD0; float3 Normal:TEXCOORD1; float4 Color:COLOR0; float2 UV:TEXCOORD2; };
Surface VSMain(InputVertex vertex)
{
    Surface result; float4 worldPoint = mul(World,float4(vertex.Position,1));
    result.Position = mul(ViewProjection,worldPoint); result.Point = worldPoint.xyz/worldPoint.w;
    result.Normal = mul((float3x3)NormalWorld,vertex.Normal);
    result.Color = float4(Decode(vertex.Color.r),Decode(vertex.Color.g),Decode(vertex.Color.b),vertex.Color.a);
    result.UV = float2(vertex.UV.x,1-vertex.UV.y); return result;
}
struct Capture { float4 Normal:SV_TARGET0; float4 Tint:SV_TARGET1; };
Capture PSMain(Surface fragment)
{
    if (fragment.Position.z >= OpaqueDepth.Load(int3(int2(fragment.Position.xy),0))) discard;
    float4 material = fragment.Color*Color;
    if (Textures.x != 0) material *= DiffuseMap.Sample(MaterialSampler,fragment.UV);
    float opacity = saturate(material.a*(Textures.z != 0 ? OpacityMap.Sample(MaterialSampler,fragment.UV).r : 1));
    if (opacity <= 0 || opacity < Material.x) discard;
    float3 normal = fragment.Normal;
    if (!NormalizeDirection(normal)) { normal = cross(ddx(fragment.Point),ddy(fragment.Point)); if (!NormalizeDirection(normal)) discard; }
    if (Textures.y != 0)
    {
        float3 horizontal = ddx(fragment.Point), vertical = ddy(fragment.Point);
        float2 u = ddx(fragment.UV), v = ddy(fragment.UV); float det = u.x*v.y-u.y*v.x;
        if (abs(det) > UvDeterminantTolerance)
        {
            float3 tangent = (horizontal*v.y-vertical*u.y)/det;
            float3 referenceB = (vertical*u.x-horizontal*v.x)/det;
            tangent -= dot(tangent,normal)*normal;
            if (NormalizeDirection(tangent))
            {
                float3 bitangent = cross(normal,tangent);
                if (dot(bitangent,referenceB) < 0) bitangent = -bitangent;
                float3 map = NormalMap.Sample(MaterialSampler,fragment.UV).xyz*2-1;
                float3 mapped = map.x*tangent+map.y*bitangent+map.z*normal;
                if (NormalizeDirection(mapped)) normal = mapped;
            }
        }
    }
    Capture result; result.Normal = float4(normal,1);
    result.Tint = float4(Textures.w == 0 ? 1 : saturate(material.rgb),Material.y > 0 ? 1 : opacity); return result;
}
#else
Texture2D<float4> SceneColor:register(t4), SurfaceNormal:register(t6), SurfaceTint:register(t7);
Texture2D<float> SurfaceDepth:register(t5);
struct Screen { float4 Position:SV_POSITION; };
Screen VSMain(uint index:SV_VertexID)
{
    Screen result; float2 corner = float2((index<<1)&2,index&2);
    result.Position = float4(corner*2-1,0,1); return result;
}
float2 ToPixels(float4 clip)
{ return (clip.xy/clip.w*float2(CenterOffset,-CenterOffset)+CenterOffset)*Viewport.xy; }
bool Reconstruct(float2 uv,float depth,out float3 worldPoint)
{
    worldPoint = 0; float4 h = mul(InverseViewProjection,float4(2*uv.x-1,1-2*uv.y,depth,1));
    if (abs(h.w) < HomogeneousTolerance) return false;
    worldPoint = h.xyz/h.w; return all(isfinite(worldPoint));
}
bool CameraDirection(float2 uv,out float3 direction)
{
    float3 nearPoint, interiorPoint; direction = 0;
    if (!Reconstruct(uv,0,nearPoint) || !Reconstruct(uv,CenterOffset,interiorPoint)) return false;
    direction = interiorPoint-nearPoint; return NormalizeDirection(direction);
}
bool DepthSurface(int2 pixel,out float3 worldPoint,out float3 direction)
{
    worldPoint = direction = 0;
    if (any(pixel < 0) || any(pixel >= int2(Viewport.xy))) return false;
    float depth = OpaqueDepth.Load(int3(pixel,0)); float2 uv = (float2(pixel)+CenterOffset)*Viewport.zw;
    return depth < BackgroundDepth && Reconstruct(uv,depth,worldPoint) && CameraDirection(uv,direction);
}
bool LimitPlane(float first,float last,inout float2 interval)
{
    if (first < 0 && last < 0) return false;
    if ((first < 0) != (last < 0))
    { float crossing = first/(first-last); if (first < 0) interval.x = max(interval.x,crossing); else interval.y = min(interval.y,crossing); }
    return interval.x < interval.y;
}
bool FindBackground(float3 origin,float3 direction,out float2 coordinate)
{
    coordinate = 0; float3 end = origin+Trace.x*direction;
    float4 h0 = mul(ViewProjection,float4(origin,1)), h1 = mul(ViewProjection,float4(end,1));
    if (!all(isfinite(h0)) || !all(isfinite(h1))) return false;
    float2 interval = float2(0,1);
    if (!LimitPlane(h0.w-HomogeneousTolerance,h1.w-HomogeneousTolerance,interval) ||
        !LimitPlane(h0.w+h0.x,h1.w+h1.x,interval) || !LimitPlane(h0.w-h0.x,h1.w-h1.x,interval) ||
        !LimitPlane(h0.w+h0.y,h1.w+h1.y,interval) || !LimitPlane(h0.w-h0.y,h1.w-h1.y,interval) ||
        !LimitPlane(h0.z,h1.z,interval) || !LimitPlane(h0.w-h0.z,h1.w-h1.z,interval)) return false;
    float3 delta = end-origin, start = origin+interval.x*delta, finish = origin+interval.y*delta;
    h0 = mul(ViewProjection,float4(start,1)); h1 = mul(ViewProjection,float4(finish,1));
    float2 reciprocalW = float2(rcp(h0.w),rcp(h1.w));
    float3 q0 = start*reciprocalW.x, q1 = finish*reciprocalW.y;
    float2 p0 = ToPixels(h0), p1 = ToPixels(h1), span = abs(p1-p0);
    float increment = min(1,Trace.w/max(max(span.x,span.y),Trace.w));
    // No origin-pixel exclusion: camera-parallel transmission may stay in it.
    [loop] for (uint sampleIndex = 0; sampleIndex < Options.x; ++sampleIndex)
    {
        float lower = sampleIndex*increment; if (lower >= 1) break;
        float upper = min(lower+increment,1), midpoint = (lower+upper)*CenterOffset;
        float3 surface, camera;
        if (!DepthSurface(int2(floor(lerp(p0,p1,midpoint))),surface,camera)) continue;
        float numerator0 = dot(q0-surface*reciprocalW.x,camera), numerator1 = dot(q1-surface*reciprocalW.y,camera);
        float d0 = lerp(numerator0,numerator1,lower)/lerp(reciprocalW.x,reciprocalW.y,lower);
        float d1 = lerp(numerator0,numerator1,upper)/lerp(reciprocalW.x,reciprocalW.y,upper);
        if (max(d0,d1) < 0 || min(d0,d1) > Trace.y) continue;
        float slope = numerator1-numerator0;
        float root = clamp(abs(slope) > HomogeneousTolerance ? -numerator0/slope : midpoint,lower,upper);
        float2 hit = lerp(p0,p1,root);
        if (!DepthSurface(int2(floor(hit)),surface,camera)) continue;
        float3 position = lerp(q0,q1,root)/lerp(reciprocalW.x,reciprocalW.y,root);
        float separation = dot(position-surface,camera);
        if (separation < -SurfaceTolerance || separation > Trace.y) continue;
        coordinate = hit*Viewport.zw; return true;
    }
    return false;
}
float3 SampleRadiance(float2 uv)
{
    float2 p = uv*Viewport.xy-CenterOffset; int2 first = int2(floor(p)); float2 fraction = frac(p); float3 sum = 0;
    [unroll] for (int row = 0; row < 2; ++row) [unroll] for (int column = 0; column < 2; ++column)
    {
        int2 pixel = clamp(first+int2(column,row),0,int2(Viewport.xy)-1);
        sum += SceneColor.Load(int3(pixel,0)).rgb*(column == 0 ? 1-fraction.x : fraction.x)*(row == 0 ? 1-fraction.y : fraction.y);
    }
    return max(sum,0);
}
float3 TraceRadiance(float3 origin,float3 direction,float3 miss)
{
    float2 hit; if (!FindBackground(origin,direction,hit)) return miss;
    float2 border = min(hit,1-hit)*Viewport.xy;
    float visibility = Appearance.x > 0 ? saturate(min(border.x,border.y)/Appearance.x) : 1;
    return lerp(miss,SampleRadiance(hit),visibility);
}
void SumPaths(float reflectance,float3 attenuation,out float3 reflected,out float3 transmitted)
{
    reflected = 1; transmitted = 0; if (reflectance >= 1) return;
    float interfaceTransmission = 1-reflectance;
    transmitted = attenuation*interfaceTransmission*interfaceTransmission/((interfaceTransmission+reflectance*(1-attenuation))*(1+attenuation*reflectance));
    reflected = reflectance+attenuation*reflectance*transmitted;
}
float4 PSMain(Screen fragment):SV_TARGET
{
    int3 pixel = int3(int2(fragment.Position.xy),0); float4 background = SceneColor.Load(pixel), tint = SurfaceTint.Load(pixel);
    float depth = SurfaceDepth.Load(pixel);
    if (depth >= BackgroundDepth || tint.a <= 0) return background;
    float3 worldPoint, incident, normal = SurfaceNormal.Load(pixel).xyz; float2 uv = fragment.Position.xy*Viewport.zw;
    if (!Reconstruct(uv,depth,worldPoint) || !CameraDirection(uv,incident) || !NormalizeDirection(normal)) return background;
    if (dot(incident,normal) > 0) normal = -normal;
    float ci = saturate(-dot(incident,normal)), ratio = Optics.x/Optics.y;
    float3 crossDirection = cross(incident,normal); float sineSquared = ratio*ratio*dot(crossDirection,crossDirection);
    float3 reflection = 1, transmission = 0, exitPoint = worldPoint;
    bool matched = Optics.x == Optics.y;
    if (matched || sineSquared < 1)
    {
        float ct = matched ? ci : sqrt(max(0,1-sineSquared));
        if (ct == 0 && Optics.z > 0) return background;
        float length = Optics.z == 0 ? 0 : Optics.z/ct;
        exitPoint += length*(matched ? incident : ratio*cross(normal,crossDirection)-ct*normal);
        if (!isfinite(length) || !all(isfinite(exitPoint))) return background;
        float3 attenuation = exp(-(Absorption.rgb-log(max(tint.rgb,TintFloor))/Optics.w)*length);
        float s = 0, p = 0;
        if (!matched)
        {
            s = (Optics.x*ci-Optics.y*ct)/(Optics.x*ci+Optics.y*ct);
            p = (Optics.y*ci-Optics.x*ct)/(Optics.y*ci+Optics.x*ct);
        }
        float3 sr,st,pr,pt; SumPaths(s*s,attenuation,sr,st); SumPaths(p*p,attenuation,pr,pt);
        reflection = (sr+pr)*UnpolarizedWeight; transmission = (st+pt)*UnpolarizedWeight;
    }
    float3 radiance = 0;
    // Low-frequency volume radiance replaces only the unresolved reflection.
    float3 reflected=reflect(incident,normal);
    float3 fallback=TranslucentCells.w!=0 ? LumenTranslucentIrradiance(worldPoint,reflected) : Environment.rgb;
    if (any(reflection > 0)) radiance += reflection*TraceRadiance(worldPoint+normal*Trace.z,reflected,fallback);
    if (any(transmission > 0)) radiance += transmission*TraceRadiance(exitPoint+incident*Trace.z,incident,background.rgb);
    return float4(clamp(lerp(background.rgb,radiance,saturate(tint.a)),0,MaxHalf),background.a);
}
#endif
