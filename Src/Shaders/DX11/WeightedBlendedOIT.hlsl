// McGuire and Bavoil, Weighted Blended Order-Independent Transparency (2013).
// https://jcgt.org/published/0002/02/09/
// Surface consumers produce sum(C * alpha * weight, alpha * weight) and
// product(1 - alpha). This independent technique resolves those statistics;
// the output merger composites straight radiance with coverage over HDR color.
Texture2D<float4> Accumulation : register(t0);
Texture2D<float> Revealage : register(t1);
Texture2DMS<float4> SampleAccumulation : register(t2);
Texture2DMS<float> SampleRevealage : register(t3);
struct FullscreenVertex { float4 Position : SV_Position; };
FullscreenVertex VSMain(uint vertexIndex : SV_VertexID)
{
    const float FullscreenExtent = 3.0;
    FullscreenVertex output;
    float2 corner = vertexIndex == 0 ? float2(-1.0, -1.0) :
        vertexIndex == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    output.Position = float4(corner, 0.0, 1.0);
    return output;
}
float4 ResolveStatistics(float4 accumulated, float revealage)
{
    const float MaximumHalfFloatRadiance = 65504.0;
    if (accumulated.a <= 0.0) return 0.0;
    float3 radiance = clamp(accumulated.rgb / accumulated.a, 0.0, MaximumHalfFloatRadiance);
    return float4(radiance, 1.0 - saturate(revealage));
}
float4 PSMain(FullscreenVertex input) : SV_Target
{
    int3 pixel = int3(input.Position.xy, 0);
    return ResolveStatistics(Accumulation.Load(pixel), Revealage.Load(pixel));
}
float4 PSResolveSamples(FullscreenVertex input, uint sampleIndex : SV_SampleIndex) : SV_Target
{
    // Sample-frequency composition preserves the opaque sample colors/depth
    // and the exact coverage of intersecting transparent silhouettes.
    int2 pixel = int2(input.Position.xy);
    return ResolveStatistics(SampleAccumulation.Load(pixel, sampleIndex), SampleRevealage.Load(pixel, sampleIndex));
}
