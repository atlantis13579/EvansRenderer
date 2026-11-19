#include <metal_stdlib>
using namespace metal;

// McGuire and Bavoil, Weighted Blended Order-Independent Transparency (2013).
// https://jcgt.org/published/0002/02/09/
// Native MSL composition: normalize weighted statistics, recover coverage from
// revealage, and source-over blend into the original opaque HDR attachment.
struct TransparencyVertex { float4 Position [[position]]; };
vertex TransparencyVertex TransparencyResolveVertex(uint vertexIndex [[vertex_id]])
{
    const float FullscreenCorner = 3.0;
    TransparencyVertex output;
    output.Position = float4(vertexIndex == 1 ? FullscreenCorner : -1.0,
        vertexIndex == 2 ? FullscreenCorner : -1.0, 0.0, 1.0);
    return output;
}
float4 NormalizeTransparency(float4 sum, float transmission)
{
    if (sum.a <= 0.0) return float4(0.0);
    const float HalfFloatSceneLimit = 65504.0;
    return float4(clamp(sum.rgb / sum.a, 0.0, HalfFloatSceneLimit),
        1.0 - clamp(transmission, 0.0, 1.0));
}
fragment float4 TransparencyResolveFragment(TransparencyVertex input [[stage_in]],
    texture2d<float, access::read> radianceSum [[texture(0)]],
    texture2d<float, access::read> uncoveredFraction [[texture(1)]])
{
    uint2 pixel = uint2(input.Position.xy);
    return NormalizeTransparency(radianceSum.read(pixel), uncoveredFraction.read(pixel).r);
}
fragment float4 TransparencyResolveSamplesFragment(TransparencyVertex input [[stage_in]],
    uint sampleIndex [[sample_id]],
    texture2d_ms<float, access::read> radianceSum [[texture(0)]],
    texture2d_ms<float, access::read> uncoveredFraction [[texture(1)]])
{
    // sample_id requests per-sample execution and selects the output sample.
    // Averaging A and R before this nonlinear division would break coverage.
    uint2 pixel = uint2(input.Position.xy);
    return NormalizeTransparency(radianceSum.read(pixel, sampleIndex),
        uncoveredFraction.read(pixel, sampleIndex).r);
}
