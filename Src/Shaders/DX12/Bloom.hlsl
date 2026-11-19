// Independent DX12 HDR bloom stages. A normalized sigma-two, nine-tap
// Gaussian is separable, reducing the two-dimensional blur from 81 to 18 taps.
// RGBA16F has a finite representable range; saturation is not tone mapping.
static const float MaximumHalfFloatRadiance = 65504.0f;
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
};
Texture2D<float4> SourceColor : register(t0);
Texture2D<float4> BlurredBloom : register(t1);
SamplerState ClampSampler : register(s0);
static const uint BrightExtraction = 0u;
static const uint HorizontalBlur = 1u;
static const uint VerticalBlur = 2u;
static const int BlurRadius = 4;
static const float GaussianKernel[5] = {0.20416369f, 0.18017382f, 0.12383154f, 0.06628225f, 0.02763055f};
static const float DownsampleNormalization = 0.25f;
struct FullscreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput result;
    result.UV = float2((id << 1u) & 2u, id & 2u);
    result.Position = float4(result.UV * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return result;
}
float4 PSMain(FullscreenOutput fragment) : SV_TARGET
{
    float2 uv = fragment.UV;
    float4 source = SourceColor.SampleLevel(ClampSampler, uv, 0);
    if (PassToneMapping.x == BrightExtraction)
    {
        float2 delta = InverseSourceExtentBloom.xy * 0.5f;
        float3 total = max(SourceColor.SampleLevel(ClampSampler, uv - delta, 0).rgb - InverseSourceExtentBloom.z, 0.0f);
        total += max(SourceColor.SampleLevel(ClampSampler, uv + delta * float2(1.0f, -1.0f), 0).rgb - InverseSourceExtentBloom.z, 0.0f);
        total += max(SourceColor.SampleLevel(ClampSampler, uv + delta * float2(-1.0f, 1.0f), 0).rgb - InverseSourceExtentBloom.z, 0.0f);
        total += max(SourceColor.SampleLevel(ClampSampler, uv + delta, 0).rgb - InverseSourceExtentBloom.z, 0.0f);
        return float4(min(total * DownsampleNormalization, MaximumHalfFloatRadiance), source.a);
    }
    if (PassToneMapping.x == HorizontalBlur || PassToneMapping.x == VerticalBlur)
    {
        float2 axis = PassToneMapping.x == HorizontalBlur ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
        float2 stepUV = axis * InverseSourceExtentBloom.xy;
        float3 filtered = source.rgb * GaussianKernel[0];
        [unroll] for (int index = 1; index <= BlurRadius; ++index)
        {
            float3 negative = SourceColor.SampleLevel(ClampSampler, uv - stepUV * index, 0).rgb;
            float3 positive = SourceColor.SampleLevel(ClampSampler, uv + stepUV * index, 0).rgb;
            filtered += (negative + positive) * GaussianKernel[index];
        }
        return float4(min(filtered, MaximumHalfFloatRadiance), source.a);
    }
    float3 bloom = BlurredBloom.SampleLevel(ClampSampler, uv, 0).rgb;
    return float4(min(source.rgb + bloom * InverseSourceExtentBloom.w, MaximumHalfFloatRadiance), source.a);
}
