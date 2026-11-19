// DX12 surface-atlas GI. An opaque-depth-equal draw writes one indirect
// contribution per covered sample; a separate fullscreen draw adds the resolved
// contribution. Keeping overwrite semantics prevents coplanar draws from adding
// lighting twice. Both stages compile directly from this native HLSL source.
cbuffer LightmapData : register(b0)
{
    row_major float4x4 World, ViewProjection;
    float4 DiffuseColor, Appearance;
    uint4 Options;
};
Texture2D<float4> FrontAtlas:register(t0), BackAtlas:register(t1);
Texture2D<float4> DiffuseMap:register(t2), OpacityMap:register(t3);
Texture2D<float4> SceneColor:register(t4), Contribution:register(t5);
SamplerState MaterialSampler:register(s0);
static const uint Compose = 1;
static const float MaximumRadiance = 65504, TexelCenter = 0.5f;
static const float ColorThreshold = 0.04045f, LinearDivisor = 12.92f;
static const float PowerOffset = 0.055f, PowerDivisor = 1.055f, ColorPower = 2.4f;
float DecodeComponent(float encoded)
{
    encoded = saturate(encoded);
    return encoded <= ColorThreshold ? encoded/LinearDivisor : pow((encoded+PowerOffset)/PowerDivisor,ColorPower);
}
struct MeshInput { float3 Position:POSITION; float2 MaterialUV:TEXCOORD0; float2 AtlasUV:TEXCOORD1; float4 Color:COLOR0; };
struct FragmentInput { float4 Position:SV_POSITION; float2 MaterialUV:TEXCOORD0; float2 AtlasUV:TEXCOORD1; float4 Color:COLOR0; };
FragmentInput VSMain(MeshInput vertex, uint id:SV_VertexID)
{
    FragmentInput result = (FragmentInput)0;
    if (Options.w == Compose)
    {
        float2 corner = float2((id<<1)&2,id&2);
        result.Position = float4(corner*2-1,0,1); return result;
    }
    float4 position = mul(World,float4(vertex.Position,1));
    result.Position = mul(ViewProjection,position);
    result.MaterialUV = float2(vertex.MaterialUV.x,1-vertex.MaterialUV.y);
    result.AtlasUV = vertex.AtlasUV;
    result.Color = float4(DecodeComponent(vertex.Color.r),DecodeComponent(vertex.Color.g),DecodeComponent(vertex.Color.b),vertex.Color.a);
    return result;
}
float3 AtlasRadiance(float2 coordinate, bool isFront)
{
    if (!isFront && Options.z == 0) return 0;
    float2 location = coordinate*Appearance.w-TexelCenter;
    int2 origin = int2(floor(location)); float2 fraction = frac(location);
    float3 filtered = 0;
    [unroll] for (int row=0;row<2;++row) [unroll] for (int column=0;column<2;++column)
    {
        int2 sampleCell = clamp(origin+int2(column,row),0,int(Appearance.w)-1);
        float3 sampleValue = isFront ? FrontAtlas.Load(int3(sampleCell,0)).xyz : BackAtlas.Load(int3(sampleCell,0)).xyz;
        float horizontal = column == 0 ? 1-fraction.x : fraction.x;
        float vertical = row == 0 ? 1-fraction.y : fraction.y;
        filtered += sampleValue*horizontal*vertical;
    }
    return max(filtered,0);
}
float4 PSMain(FragmentInput fragment, bool isFront:SV_IsFrontFace):SV_TARGET
{
    if (Options.w == Compose)
    {
        int3 pixel = int3(int2(fragment.Position.xy),0);
        float4 color = SceneColor.Load(pixel);
        float visibility = Options.x != 0 ? saturate(FrontAtlas.Load(pixel).r) : 1;
        return float4(clamp(color.rgb+Contribution.Load(pixel).rgb*visibility,0,MaximumRadiance),color.a);
    }
    float4 reflectance = fragment.Color*DiffuseColor;
    if (Options.x != 0) reflectance *= DiffuseMap.Sample(MaterialSampler,fragment.MaterialUV);
    float alpha = reflectance.a;
    float3 surfaceMap = Options.y != 0 ? OpacityMap.Sample(MaterialSampler,fragment.MaterialUV).rgb : float3(1, 1, 1);
    float diffuseWeight = Options.y > 1 ? 0.96 * (1 - Appearance.z * surfaceMap.b) : Appearance.z;
    alpha *= surfaceMap.r;
    if (saturate(alpha) < Appearance.y) discard;
    // The producer already divided irradiance by pi. Apply only receiver
    // reflectance and the dielectric/metallic energy factor supplied by the CPU.
    float3 indirect = AtlasRadiance(fragment.AtlasUV,isFront)*saturate(reflectance.rgb)*Appearance.x*diffuseWeight;
    return float4(clamp(indirect,0,MaximumRadiance),1);
}
