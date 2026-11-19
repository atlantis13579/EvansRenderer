// Linear HDR bloom: threshold before downsampling, separable Gaussian blur,
// then additive reconstruction. One technique owns every stage in this file.
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
SamplerState LinearClamp : register(s0);
static const uint ExtractPass = 0u;
static const uint HorizontalPass = 1u;
static const uint VerticalPass = 2u;
static const int GaussianRadius = 4;
// exp(-offset^2 / (2*sigma^2)), sigma=2, normalized over [-4,4].
static const float GaussianWeights[5] = {0.20416369f, 0.18017382f, 0.12383154f, 0.06628225f, 0.02763055f};
static const float FourTapWeight = 0.25f;
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((id << 1u) & 2u, id & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
float3 BrightSample(float2 uv)
{
    return max(SourceColor.SampleLevel(LinearClamp, uv, 0).rgb - InverseSourceExtentBloom.z, 0.0f);
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    float4 center = SourceColor.SampleLevel(LinearClamp, input.UV, 0);
    float2 halfTexel = InverseSourceExtentBloom.xy * 0.5f;
    if (PassToneMapping.x == ExtractPass)
    {
        float3 bright = BrightSample(input.UV + halfTexel * float2(-1.0f, -1.0f))
            + BrightSample(input.UV + halfTexel * float2(1.0f, -1.0f))
            + BrightSample(input.UV + halfTexel * float2(-1.0f, 1.0f))
            + BrightSample(input.UV + halfTexel);
        return float4(min(bright * FourTapWeight, MaximumHalfFloatRadiance), center.a);
    }
    if (PassToneMapping.x == HorizontalPass || PassToneMapping.x == VerticalPass)
    {
        float2 direction = PassToneMapping.x == HorizontalPass
            ? float2(InverseSourceExtentBloom.x, 0.0f) : float2(0.0f, InverseSourceExtentBloom.y);
        float3 color = center.rgb * GaussianWeights[0];
        [unroll] for (int offset = 1; offset <= GaussianRadius; ++offset)
        {
            color += GaussianWeights[offset] * (
                SourceColor.SampleLevel(LinearClamp, input.UV + direction * offset, 0).rgb
                + SourceColor.SampleLevel(LinearClamp, input.UV - direction * offset, 0).rgb);
        }
        return float4(min(color, MaximumHalfFloatRadiance), center.a);
    }
    return float4(min(center.rgb + InverseSourceExtentBloom.w * BlurredBloom.SampleLevel(LinearClamp, input.UV, 0).rgb, MaximumHalfFloatRadiance), center.a);
}
