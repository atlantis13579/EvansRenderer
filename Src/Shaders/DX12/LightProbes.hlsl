// D3D12 opaque indirect diffuse gather. Moment-based weights reject hidden grid
// neighbors while SH evaluates their low-frequency irradiance at the normal.
// World-space data survives camera motion; no velocity/history input is sampled.
// https://jcgt.org/published/0008/02/01/ ; https://graphics.stanford.edu/papers/envmap/
cbuffer ProbeUniforms : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uint4 Dimensions, Options;
};
Texture2D<float4> IrradianceSH : register(t1), ProbeMoments : register(t2), OpaqueColor : register(t3);
Texture2D<float> OpaqueDepth : register(t4);
Texture2D<float4> SurfaceNormal : register(t5), SurfaceDiffuse : register(t6), SurfaceAmbient : register(t7);
static const float Pi = 3.14159265358979323846f, SHDecodeScale = 4;
static const float MinimumW = 1e-7f, MinimumSquaredLength = 1e-12f, MinimumWeightSum = 1e-6f;
static const float WrapWeightFloor = 0.05f, NonmetalF0 = 0.04f, MaximumBinary16 = 65504, HalfTexel = 0.5f;
static const uint SHCount = 9, MetallicRoughnessModel = 3;
struct FullscreenOutput { float4 Position : SV_Position; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput output; float2 p = float2((id << 1) & 2, id & 2);
    output.Position = float4(p * 2 - 1, 0, 1); return output;
}
float RealBasis(uint i, float3 w)
{
    switch (i)
    {
    case 0: return sqrt(1.0f/(4*Pi));
    case 1: return w.y*sqrt(3.0f/(4*Pi));
    case 2: return w.z*sqrt(3.0f/(4*Pi));
    case 3: return w.x*sqrt(3.0f/(4*Pi));
    case 4: return w.x*w.y*sqrt(15.0f/(4*Pi));
    case 5: return w.y*w.z*sqrt(15.0f/(4*Pi));
    case 6: return (3*w.z*w.z-1)*sqrt(5.0f/(16*Pi));
    case 7: return w.x*w.z*sqrt(15.0f/(4*Pi));
    default: return (w.x*w.x-w.y*w.y)*sqrt(15.0f/(16*Pi));
    }
}
int2 MirrorOctBorder(int2 texel)
{
    int size = int(Options.z);
    if (texel.x < 0) { texel.x = -1-texel.x; texel.y = size-1-texel.y; }
    if (texel.x >= size) { texel.x = size*2-1-texel.x; texel.y = size-1-texel.y; }
    if (texel.y < 0) { texel.y = -1-texel.y; texel.x = size-1-texel.x; }
    if (texel.y >= size) { texel.y = size*2-1-texel.y; texel.x = size-1-texel.x; }
    return texel;
}
float2 FilterMoments(uint index, float3 direction)
{
    direction /= dot(abs(direction), float3(1,1,1));
    float2 oct = direction.xy;
    if (direction.z < 0) oct = (1-abs(oct.yx))*float2(oct.x >= 0 ? 1 : -1, oct.y >= 0 ? 1 : -1);
    float2 position = (oct*0.5f+0.5f)*Options.z-HalfTexel;
    int2 first = int2(floor(position)); float2 blend = frac(position);
    uint columns = uint(Visibility.w);
    int2 origin = int2(index%columns,index/columns)*int(Options.z);
    float2 result = 0;
    [unroll] for (uint neighbor=0; neighbor<4; ++neighbor)
    {
        int2 offset = int2(neighbor&1,neighbor>>1);
        float weight = (offset.x == 0 ? 1-blend.x : blend.x)*(offset.y == 0 ? 1-blend.y : blend.y);
        result += ProbeMoments.Load(int3(origin+MirrorOctBorder(first+offset),0)).xy*weight;
    }
    return result;
}
float4 PSMain(FullscreenOutput input) : SV_Target0
{
    int2 pixel = int2(input.Position.xy);
    float4 color = OpaqueColor.Load(int3(pixel,0)), material = SurfaceNormal.Load(int3(pixel,0));
    float depth = OpaqueDepth.Load(int3(pixel,0));
    if (depth >= 1 || material.w < 0 || dot(material.xyz,material.xyz) <= MinimumSquaredLength || !all(isfinite(material))) return color;
    float2 uv = input.Position.xy*Viewport.zw;
    float4 p = mul(InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!all(isfinite(p)) || abs(p.w) <= MinimumW) return color;
    float3 world = p.xyz/p.w, normal = normalize(material.xyz);
    float3 coordinate = (world-GridMinimum.xyz)/(GridMaximum.xyz-GridMinimum.xyz)*float3(Dimensions.xyz-1);
    if (any(coordinate<0) || any(coordinate>float3(Dimensions.xyz-1))) return color;
    uint3 first = min(uint3(floor(coordinate)),Dimensions.xyz-2);
    float3 alpha = coordinate-float3(first), receiver = world+normal*Appearance.y;
    float3 sum = 0; float totalWeight = 0;
    [unroll] for (uint neighbor=0; neighbor<8; ++neighbor)
    {
        uint3 offset = uint3(neighbor&1,(neighbor>>1)&1,neighbor>>2), cell = first+offset;
        uint index = cell.x+Dimensions.x*(cell.y+Dimensions.y*cell.z);
        if (IrradianceSH.Load(int3(0,index,0)).a <= 0) continue;
        float3 probe = lerp(GridMinimum.xyz,GridMaximum.xyz,float3(cell)/float3(Dimensions.xyz-1));
        float3 fromProbe = receiver-probe; float distance = length(fromProbe);
        float3 direction = distance > MinimumW ? fromProbe/distance : normal;
        float2 moments = FilterMoments(index,direction);
        float delta = max((distance-Appearance.z)/Visibility.z-moments.x,0);
        float variance = max(moments.y-moments.x*moments.x,Visibility.x);
        float visibility = pow(saturate(variance/max(variance+delta*delta,MinimumSquaredLength)),Appearance.w);
        float3 toProbe = probe-world;
        float facing = dot(toProbe,toProbe) > MinimumSquaredLength ? dot(normal,normalize(toProbe)) : 1;
        float wrap = max((facing+1)*0.5f,WrapWeightFloor);
        float3 barycentric = float3(offset.x == 0 ? 1-alpha.x : alpha.x,offset.y == 0 ? 1-alpha.y : alpha.y,offset.z == 0 ? 1-alpha.z : alpha.z);
        float weight = barycentric.x*barycentric.y*barycentric.z*wrap*wrap*visibility;
        float3 irradiance = 0;
        [unroll] for (uint coefficient=0; coefficient<SHCount; ++coefficient)
            irradiance += IrradianceSH.Load(int3(coefficient,index,0)).rgb*RealBasis(coefficient,normal);
        sum += max(irradiance*SHDecodeScale,0)*weight; totalWeight += weight;
    }
    if (totalWeight <= MinimumWeightSum) return color;
    float3 reflectance = saturate(SurfaceDiffuse.Load(int3(pixel,0)).rgb);
    if (Options.y == MetallicRoughnessModel) reflectance *= (1-NonmetalF0)*(1-saturate(SurfaceAmbient.Load(int3(pixel,0)).w));
    return float4(clamp(color.rgb+(sum/totalWeight)*reflectance*Appearance.x,0,MaximumBinary16),color.a);
}
