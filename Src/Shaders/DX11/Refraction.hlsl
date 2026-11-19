// Lumen froxel irradiance for transparent receivers. Real SH through band two;
// XYZ interpolation is clamped within coefficient bands, never across them.
cbuffer LumenTranslucentReceiver : register(b8)
{
    row_major float4x4 LumenVolumeCamera,LumenVolumeInverse;
    uint4 LumenVolumeGrid;
    float4 LumenVolumeDistribution,LumenVolumeAtlasSize;
};
Texture2D<float4> LumenVolumeImage : register(t19);
SamplerState LumenVolumeSampler : register(s4);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 worldPosition,float3 normal)
{
    if (LumenVolumeGrid.w==0) return 0;
    float4 clip=mul(LumenVolumeCamera,float4(worldPosition,1));
    if (clip.w<=LumenVolumeAtlasSize.z) return 0;
    float3 ndc=clip.xyz/clip.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=mul(LumenVolumeInverse,float4(ndc.xy,0,1)),farH=mul(LumenVolumeInverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=LumenVolumeAtlasSize.z) return 0;
    float3 nearPoint=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-nearPoint),LumenVolumeDistribution.x);
    float distance=length(worldPosition-nearPoint);
    if (extent<=LumenVolumeAtlasSize.z || distance>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(LumenVolumeGrid.xy),pow(saturate(distance/extent),1/LumenVolumeDistribution.y)*LumenVolumeGrid.z)-0.5;
    cell=clamp(cell,0,float3(LumenVolumeGrid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(LumenVolumeGrid.z-1));
    normal*=rsqrt(max(dot(normal,normal),LumenVolumeAtlasSize.w));
    static const float SpherePi=3.14159265358979323846;
    static const uint CoefficientCount=9;
    float basis[CoefficientCount]={sqrt(1/(4*SpherePi)),sqrt(3/(4*SpherePi))*normal.y,sqrt(3/(4*SpherePi))*normal.z,sqrt(3/(4*SpherePi))*normal.x,
        sqrt(15/(4*SpherePi))*normal.x*normal.y,sqrt(15/(4*SpherePi))*normal.y*normal.z,
        sqrt(5/(16*SpherePi))*(3*normal.z*normal.z-1),sqrt(15/(4*SpherePi))*normal.x*normal.z,
        sqrt(15/(16*SpherePi))*(normal.x*normal.x-normal.y*normal.y)};
    float3 irradiance=0;

    static const uint VolumeVisibilityBit=2;
    if ((LumenVolumeGrid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=frac(cell);
        // Keep the original trilinear mass. Rejected support spheres contribute
        // no light, rather than renormalizing bright probes across an occluder.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,LumenVolumeGrid.xyz-1);
            float3 weights=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
            float weight=weights.x*weights.y*weights.z; if (weight<=0) continue;
            uint row=p.z*LumenVolumeGrid.y+p.y; float support=LumenVolumeImage.Load(int3(p.x,row,0)).a; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/LumenVolumeGrid.x-1,1-2*(float(p.y)+0.5)/LumenVolumeGrid.y);
            float4 a=mul(LumenVolumeInverse,float4(clipXY,0,1)),b=mul(LumenVolumeInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=LumenVolumeAtlasSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float span=length(axis);
            if (!isfinite(span) || span<=LumenVolumeAtlasSize.z) continue;
            float depth=pow((float(p.z)+0.5)/LumenVolumeGrid.z,LumenVolumeDistribution.y);
            float3 probe=start+axis*(depth*min(span,LumenVolumeDistribution.x)/span);
            float separation=length(worldPosition-probe); if (!isfinite(separation) || separation>support) continue;
            [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
                irradiance+=weight*basis[coefficient]*LumenVolumeImage.Load(int3(p.x+coefficient*LumenVolumeGrid.x,row,0)).rgb;
        }
        return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
    }
    [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
    {
        float x=coefficient*LumenVolumeGrid.x+cell.x+0.5;
        float2 a=float2(x,lower*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        float2 b=float2(x,upper*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        irradiance+=lerp(LumenVolumeImage.SampleLevel(LumenVolumeSampler,a,0).rgb,
            LumenVolumeImage.SampleLevel(LumenVolumeSampler,b,0).rgb,frac(cell.z))*basis[coefficient];
    }
    return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
}

// DX11 nearest-layer dielectric capture and frozen-background composition.
// Snell/Fresnel/Beer: PBRT 4e, Specular Reflection and Transmission; Transmittance.
// Screen traversal: McGuire and Mara, Efficient GPU Screen-Space Ray Tracing (2014).
// https://jcgt.org/published/0003/04/04/paper.pdf
// A parallel slab sums internal energy per polarization but traces its first exit.
// Missing transmission uses the unwarped background; missing reflection uses a
// uniform environment. Hidden layers and directional incoming radiance are absent.
cbuffer RefractionUniforms : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, InverseViewProjection;
    float4 Color, Material, Optics, Absorption, Viewport, Trace, Appearance, Environment;
    uint4 Textures, Options;
};
Texture2D<float> OpaqueDepth : register(t0);
static const float VectorEpsilon = 1e-12f, ProjectionEpsilon = 1e-7f, HitTolerance = 1e-5f;
static const float PixelCenter = 0.5f, ClearDepth = 1, InteriorDepth = 0.5f;
static const float MaximumRadiance = 65504, MinimumTint = 0.0001f;
static const float UnpolarizedWeight = 0.5f;
bool Unit(inout float3 value)
{
    if (!all(isfinite(value))) return false;
    float scale = max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (scale == 0) return false;
    value /= scale;
    float lengthSquared = dot(value,value);
    if (!all(isfinite(value)) || lengthSquared <= VectorEpsilon) return false;
    value *= rsqrt(lengthSquared); return true;
}
#ifdef EVANS_REFRACTION_CAPTURE
Texture2D<float4> DiffuseMap : register(t1), NormalMap : register(t2), OpacityMap : register(t3);
SamplerState MaterialSampler : register(s0);
static const float SRGBBreakpoint = 0.04045f, SRGBDivisor = 12.92f;
static const float SRGBOffset = 0.055f, SRGBScale = 1.055f, SRGBPower = 2.4f;
static const float TangentTolerance = 1e-10f;
struct MeshVertex { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR0; float2 UV:TEXCOORD0; };
struct SurfaceVertex { float4 Position:SV_POSITION; float3 WorldPosition:TEXCOORD0; float3 Normal:TEXCOORD1; float4 Color:COLOR0; float2 UV:TEXCOORD2; };
float3 LinearColor(float3 encoded)
{
    encoded = saturate(encoded);
    float3 low = encoded/SRGBDivisor, high = pow((encoded+SRGBOffset)/SRGBScale,SRGBPower);
    return float3(encoded.r <= SRGBBreakpoint ? low.r : high.r,encoded.g <= SRGBBreakpoint ? low.g : high.g,encoded.b <= SRGBBreakpoint ? low.b : high.b);
}
SurfaceVertex VSMain(MeshVertex vertex)
{
    SurfaceVertex output;
    float4 position = mul(World,float4(vertex.Position,1));
    output.Position = mul(ViewProjection,position); output.WorldPosition = position.xyz/position.w;
    output.Normal = mul((float3x3)NormalWorld,vertex.Normal);
    output.Color = float4(LinearColor(vertex.Color.rgb),vertex.Color.a);
    output.UV = float2(vertex.UV.x,1-vertex.UV.y); return output;
}
struct CaptureOutput { float4 Normal:SV_TARGET0; float4 TintCoverage:SV_TARGET1; };
CaptureOutput PSMain(SurfaceVertex input)
{
    int2 pixel = int2(input.Position.xy);
    if (input.Position.z >= OpaqueDepth.Load(int3(pixel,0))) discard;
    float4 material = input.Color*Color;
    if (Textures.x != 0) material *= DiffuseMap.Sample(MaterialSampler,input.UV);
    float coverage = saturate(material.a*(Textures.z != 0 ? OpacityMap.Sample(MaterialSampler,input.UV).r : 1));
    if (coverage <= 0 || coverage < Material.x) discard;
    float3 normal = input.Normal;
    if (!Unit(normal)) { normal = cross(ddx(input.WorldPosition),ddy(input.WorldPosition)); if (!Unit(normal)) discard; }
    if (Textures.y != 0)
    {
        float3 dx = ddx(input.WorldPosition), dy = ddy(input.WorldPosition);
        float2 tx = ddx(input.UV), ty = ddy(input.UV);
        float determinant = tx.x*ty.y-tx.y*ty.x;
        if (abs(determinant) > TangentTolerance)
        {
            float3 tangent = (dx*ty.y-dy*tx.y)/determinant;
            float3 sourceBitangent = (dy*tx.x-dx*ty.x)/determinant;
            tangent -= normal*dot(normal,tangent);
            if (Unit(tangent))
            {
                float3 bitangent = cross(normal,tangent)*(dot(cross(normal,tangent),sourceBitangent) < 0 ? -1 : 1);
                float3 mapped = NormalMap.Sample(MaterialSampler,input.UV).xyz*2-1;
                mapped = tangent*mapped.x+bitangent*mapped.y+normal*mapped.z;
                if (Unit(mapped)) normal = mapped;
            }
        }
    }
    CaptureOutput output; output.Normal = float4(normal,1);
    output.TintCoverage = float4(Textures.w != 0 ? saturate(material.rgb) : 1,Material.y > 0 ? 1 : coverage);
    return output;
}
#else
Texture2D<float4> SceneColor : register(t4);
Texture2D<float> SurfaceDepth : register(t5);
Texture2D<float4> SurfaceNormal : register(t6), SurfaceTint : register(t7);
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint id:SV_VertexID)
{
    ScreenVertex output; float2 corner = float2((id<<1)&2,id&2);
    output.Position = float4(corner*2-1,0,1); return output;
}
bool Unproject(float2 uv,float depth,out float3 worldPoint)
{
    float4 h = mul(InverseViewProjection,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    worldPoint = 0; if (abs(h.w) < ProjectionEpsilon) return false;
    worldPoint = h.xyz/h.w; return all(isfinite(worldPoint));
}
bool ViewRay(float2 uv,out float3 ray)
{
    float3 nearPoint, farPoint; ray = 0;
    if (!Unproject(uv,0,nearPoint) || !Unproject(uv,InteriorDepth,farPoint)) return false;
    ray = farPoint-nearPoint; return Unit(ray);
}
bool Plane(float a,float b,inout float enter,inout float leave)
{
    if (a < 0 && b < 0) return false;
    if ((a < 0) != (b < 0)) { float t = a/(a-b); if (a < 0) enter = max(enter,t); else leave = min(leave,t); }
    return enter < leave;
}
bool Opaque(int2 pixel,out float3 worldPoint,out float3 cameraRay)
{
    worldPoint = cameraRay = 0;
    if (any(pixel < 0) || any(pixel >= int2(Viewport.xy))) return false;
    float depth = OpaqueDepth.Load(int3(pixel,0)); float2 uv = (float2(pixel)+PixelCenter)*Viewport.zw;
    return depth < ClearDepth && Unproject(uv,depth,worldPoint) && ViewRay(uv,cameraRay);
}
bool TraceBackground(float3 origin,float3 direction,out float2 hit)
{
    hit = 0; float3 end = origin+direction*Trace.x;
    float4 a = mul(ViewProjection,float4(origin,1)), b = mul(ViewProjection,float4(end,1));
    if (!all(isfinite(a)) || !all(isfinite(b))) return false;
    float enter = 0, leave = 1;
    if (!Plane(a.w-ProjectionEpsilon,b.w-ProjectionEpsilon,enter,leave) ||
        !Plane(a.w+a.x,b.w+b.x,enter,leave) || !Plane(a.w-a.x,b.w-b.x,enter,leave) ||
        !Plane(a.w+a.y,b.w+b.y,enter,leave) || !Plane(a.w-a.y,b.w-b.y,enter,leave) ||
        !Plane(a.z,b.z,enter,leave) || !Plane(a.w-a.z,b.w-b.z,enter,leave)) return false;
    float3 segment = end-origin, first = origin+segment*enter, last = origin+segment*leave;
    a = mul(ViewProjection,float4(first,1)); b = mul(ViewProjection,float4(last,1));
    float k0 = rcp(a.w), k1 = rcp(b.w); float3 q0 = first*k0, q1 = last*k1;
    float2 p0 = (a.xy*k0*float2(PixelCenter,-PixelCenter)+PixelCenter)*Viewport.xy;
    float2 p1 = (b.xy*k1*float2(PixelCenter,-PixelCenter)+PixelCenter)*Viewport.xy;
    float major = max(abs(p1.x-p0.x),abs(p1.y-p0.y));
    // Refractors are absent from opaque depth. Include the origin pixel, even
    // for a camera-aligned ray; SSR's usual pixel exclusion would lose this case.
    float step = min(1,Trace.w/max(major,Trace.w));
    [loop] for (uint index = 0; index < Options.x; ++index)
    {
        float begin = index*step; if (begin >= 1) break;
        float finish = min(begin+step,1), middle = (begin+finish)*PixelCenter;
        float3 surface, cameraRay;
        if (!Opaque(int2(floor(lerp(p0,p1,middle))),surface,cameraRay)) continue;
        float d0 = dot(q0-surface*k0,cameraRay), d1 = dot(q1-surface*k1,cameraRay);
        float z0 = lerp(d0,d1,begin)/lerp(k0,k1,begin), z1 = lerp(d0,d1,finish)/lerp(k0,k1,finish);
        if (max(z0,z1) < 0 || min(z0,z1) > Trace.y) continue;
        float crossing = abs(d1-d0) > ProjectionEpsilon ? -d0/(d1-d0) : middle;
        crossing = clamp(crossing,begin,finish);
        float2 projected = lerp(p0,p1,crossing);
        if (!Opaque(int2(floor(projected)),surface,cameraRay)) continue;
        float3 worldPoint = lerp(q0,q1,crossing)/lerp(k0,k1,crossing);
        float depth = dot(worldPoint-surface,cameraRay);
        if (depth < -HitTolerance || depth > Trace.y) continue;
        hit = projected*Viewport.zw; return true;
    }
    return false;
}
float3 Background(float2 uv)
{
    float2 texel = uv*Viewport.xy-PixelCenter; int2 base = int2(floor(texel)); float2 blend = frac(texel); float3 color = 0;
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x)
        color += SceneColor.Load(int3(clamp(base+int2(x,y),0,int2(Viewport.xy)-1),0)).rgb*
            (x == 0 ? 1-blend.x : blend.x)*(y == 0 ? 1-blend.y : blend.y);
    return max(color,0);
}
float3 Incoming(float3 origin,float3 direction,float3 fallback)
{
    float2 uv; if (!TraceBackground(origin,direction,uv)) return fallback;
    float2 edge = min(uv,1-uv)*Viewport.xy;
    float fade = Appearance.x > 0 ? saturate(min(edge.x,edge.y)/Appearance.x) : 1;
    return lerp(fallback,Background(uv),fade);
}
void Energy(float f,float3 a,out float3 r,out float3 t)
{
    r = 1; t = 0; if (f >= 1) return;
    float through = 1-f;
    t = a*through*through/((through+f*(1-a))*(1+a*f)); r = f+a*f*t;
}
float4 PSMain(ScreenVertex input):SV_TARGET
{
    int3 pixel = int3(int2(input.Position.xy),0); float4 scene = SceneColor.Load(pixel);
    float depth = SurfaceDepth.Load(pixel); float4 material = SurfaceTint.Load(pixel);
    if (depth >= ClearDepth || material.a <= 0) return scene;
    float2 uv = input.Position.xy*Viewport.zw; float3 position, incident, normal = SurfaceNormal.Load(pixel).xyz;
    if (!Unproject(uv,depth,position) || !ViewRay(uv,incident) || !Unit(normal)) return scene;
    if (dot(normal,incident) > 0) normal = -normal;
    float ci = saturate(-dot(normal,incident)), eta = Optics.x/Optics.y;
    float3 perpendicular = cross(incident,normal); float sinT2 = eta*eta*dot(perpendicular,perpendicular);
    float3 r = 1, t = 0, exitPoint = position;
    if (Optics.x == Optics.y || sinT2 < 1)
    {
        bool matched = Optics.x == Optics.y;
        float ct = matched ? ci : sqrt(max(0,1-sinT2));
        if (ct == 0 && Optics.z > 0) return scene;
        float path = Optics.z == 0 ? 0 : Optics.z/ct;
        float3 internalRay = matched ? incident : eta*cross(normal,perpendicular)-ct*normal;
        exitPoint += path*internalRay;
        if (!isfinite(path) || !all(isfinite(exitPoint))) return scene;
        float3 sigma = Absorption.rgb-log(max(material.rgb,MinimumTint))/Optics.w;
        float3 attenuation = exp(-sigma*path);
        float fs = 0, fp = 0;
        if (!matched)
        {
            fs = (Optics.x*ci-Optics.y*ct)/(Optics.x*ci+Optics.y*ct);
            fp = (Optics.y*ci-Optics.x*ct)/(Optics.y*ci+Optics.x*ct);
        }
        float3 rs,ts,rp,tp; Energy(fs*fs,attenuation,rs,ts); Energy(fp*fp,attenuation,rp,tp);
        r = (rs+rp)*UnpolarizedWeight; t = (ts+tp)*UnpolarizedWeight;
    }
    float3 radiance = 0;
    // Low-frequency volume radiance replaces only the unresolved reflection.
    float3 reflectedDirection=reflect(incident,normal);
    float3 missingReflection=LumenVolumeGrid.w!=0 ? LumenTranslucentIrradiance(position,reflectedDirection) : Environment.rgb;
    if (any(r > 0)) radiance += r*Incoming(position+normal*Trace.z,reflectedDirection,missingReflection);
    if (any(t > 0)) radiance += t*Incoming(exitPoint+incident*Trace.z,incident,scene.rgb);
    return float4(clamp(lerp(scene.rgb,radiance,saturate(material.a)),0,MaximumRadiance),scene.a);
}
#endif
