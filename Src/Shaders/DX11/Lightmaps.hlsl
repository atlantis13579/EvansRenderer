// DX11 lightmap receiver and HDR composition. The receiver replays exact local
// geometry against read-only opaque depth. Overwrite the contribution target,
// then resolve its MSAA samples before adding it once to the scene. UV2 belongs
// to a particular instance/LOD; material UVs and vertex colors remain independent.
// Atlas RGB stores E/pi. Gutters permit base-level bilinear reads without mips.
cbuffer LightmapUniforms : register(b0)
{
    row_major float4x4 World, ViewProjection;
    float4 DiffuseColor, Appearance;
    uint4 Options;
};
Texture2D<float4> FrontAtlas : register(t0);
Texture2D<float4> BackAtlas : register(t1);
Texture2D<float4> DiffuseMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
Texture2D<float4> SceneColor : register(t4);
Texture2D<float4> Contribution : register(t5);
SamplerState MaterialSampler : register(s0);
static const uint CompositePass = 1;
static const float HalfFloatMaximum = 65504, PixelCenter = 0.5f;
static const float SRGBThreshold = 0.04045f, SRGBLinearScale = 12.92f;
static const float SRGBOffset = 0.055f, SRGBScale = 1.055f, SRGBExponent = 2.4f;
float3 DecodeVertex(float3 value)
{
    value = saturate(value);
    return float3(value.r <= SRGBThreshold ? value.r/SRGBLinearScale : pow((value.r+SRGBOffset)/SRGBScale,SRGBExponent),
        value.g <= SRGBThreshold ? value.g/SRGBLinearScale : pow((value.g+SRGBOffset)/SRGBScale,SRGBExponent),
        value.b <= SRGBThreshold ? value.b/SRGBLinearScale : pow((value.b+SRGBOffset)/SRGBScale,SRGBExponent));
}
struct ReceiverInput { float3 Position:POSITION; float2 MaterialUV:TEXCOORD0; float2 AtlasUV:TEXCOORD1; float4 Color:COLOR0; };
struct ReceiverVertex { float4 Position:SV_POSITION; float2 MaterialUV:TEXCOORD0; float2 AtlasUV:TEXCOORD1; float4 Color:COLOR0; };
ReceiverVertex VSMain(ReceiverInput input, uint vertexId:SV_VertexID)
{
    ReceiverVertex output = (ReceiverVertex)0;
    if (Options.w == CompositePass)
    {
        float2 corner = float2((vertexId << 1) & 2,vertexId & 2);
        output.Position = float4(corner*2-1,0,1); return output;
    }
    float4 worldPosition = mul(World,float4(input.Position,1));
    output.Position = mul(ViewProjection,worldPosition);
    output.MaterialUV = float2(input.MaterialUV.x,1-input.MaterialUV.y);
    output.AtlasUV = input.AtlasUV;
    output.Color = float4(DecodeVertex(input.Color.rgb),input.Color.a); return output;
}
float3 ReadIrradiance(float2 uv, bool front)
{
    if (!front && Options.z == 0) return 0;
    float2 texel = uv*Appearance.w-PixelCenter;
    int2 base = int2(floor(texel)); float2 blend = frac(texel); float3 value = 0;
    [unroll] for (int y=0;y<2;++y) [unroll] for (int x=0;x<2;++x)
    {
        int2 cell = clamp(base+int2(x,y),0,int(Appearance.w)-1);
        float weight = (x == 0 ? 1-blend.x : blend.x)*(y == 0 ? 1-blend.y : blend.y);
        value += (front ? FrontAtlas.Load(int3(cell,0)).rgb : BackAtlas.Load(int3(cell,0)).rgb)*weight;
    }
    return max(value,0);
}
float4 PSMain(ReceiverVertex input, bool front:SV_IsFrontFace):SV_TARGET
{
    if (Options.w == CompositePass)
    {
        int2 pixel = int2(input.Position.xy); float4 scene = SceneColor.Load(int3(pixel,0));
        // Screen AO modulates indirect irradiance, preserving direct lights and emission.
        float visibility = Options.x != 0 ? saturate(FrontAtlas.Load(int3(pixel,0)).r) : 1;
        return float4(clamp(scene.rgb+Contribution.Load(int3(pixel,0)).rgb*visibility,0,HalfFloatMaximum),scene.a);
    }
    float4 material = DiffuseColor*input.Color;
    if (Options.x != 0) material *= DiffuseMap.Sample(MaterialSampler,input.MaterialUV);
    float3 surfaceMap = Options.y != 0 ? OpacityMap.Sample(MaterialSampler,input.MaterialUV).rgb : float3(1, 1, 1);
    float diffuseWeight = Options.y > 1 ? 0.96 * (1 - Appearance.z * surfaceMap.b) : Appearance.z;
    float opacity = material.a*surfaceMap.r;
    if (saturate(opacity) < Appearance.y) discard;
    float3 radiance = ReadIrradiance(input.AtlasUV,front)*saturate(material.rgb)*diffuseWeight*Appearance.x;
    return float4(clamp(radiance,0,HalfFloatMaximum),1);
}
