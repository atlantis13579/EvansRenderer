// DX12 CSM depth rendering, independently authored for its native draw contract.
// One draw stream is recorded for each stable directional-light projection.
// Receiver selection is evaluated in the native lighting shaders; this stage
// preserves material coverage while producing the individual depth resources.
// Zhang, Sun and Nyman: Parallel-Split Shadow Maps on Programmable GPUs (2007).
static const uint LightSlots = 4u;
static const float NoiseMultiplier = 52.9829189f;
static const float2 NoiseProjection = float2(0.06711056f, 0.00583715f);
struct PackedLight
{
    float4 PositionAndType;
    float4 DirectionAndIntensity;
    float4 ColorAndShadow;
    float4 EmitterSize;
};
cbuffer GeometryUniforms : register(b0)
{
    row_major float4x4 ObjectToWorld;
    row_major float4x4 NormalTransformUnused;
    row_major float4x4 WorldToCascade;
    row_major float4x4 SingleMapTransformUnused;
    float4 DrawTintUnused;
    float4 DiffuseUnused;
    float4 AmbientUnused;
    float4 SpecularUnused;
    float4 EmissionUnused;
    float4 MaterialCoverage; // shininess, opacity, alpha cutoff, outline width
    float4 EnvironmentUnused;
    float4 CameraUnused;
    float4 DrawOptions; // single shadow, render mode, light index, alpha blend
    uint4 ImageOptions;
    PackedLight SceneLightsUnused[LightSlots];
    float4 SurfaceUnused;
};
Texture2D<float4> BaseTexture : register(t1);
Texture2D<float4> CoverageTexture : register(t3);
SamplerState CoverageSampler : register(s0);
struct MeshInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD0;
};
struct ShadowFragment
{
    float4 Clip : SV_POSITION;
    float2 UV : TEXCOORD0;
    float VertexOpacity : TEXCOORD1;
};
ShadowFragment VSMain(MeshInput vertex)
{
    ShadowFragment result;
    result.Clip = mul(WorldToCascade, mul(ObjectToWorld, float4(vertex.Position, 1.0f)));
    result.UV = float2(vertex.UV.x, 1.0f - vertex.UV.y);
    result.VertexOpacity = vertex.Color.a;
    return result;
}
void PSMain(ShadowFragment fragment)
{
    float coverage = fragment.VertexOpacity * MaterialCoverage.y;
    if (ImageOptions.x != 0u) coverage *= BaseTexture.Sample(CoverageSampler, fragment.UV).a;
    if (ImageOptions.z != 0u) coverage *= CoverageTexture.Sample(CoverageSampler, fragment.UV).r;
    coverage = saturate(coverage);
    if (MaterialCoverage.z > 0.0f)
        clip(coverage - MaterialCoverage.z);
    else if (DrawOptions.w > 0.5f)
    {
        if (coverage <= 0.0f) discard;
        // Stable interleaved gradient noise converts partial opacity to coverage.
        float threshold = frac(NoiseMultiplier * frac(dot(floor(fragment.Clip.xy), NoiseProjection)));
        clip(coverage - threshold);
    }
}
