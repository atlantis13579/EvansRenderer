#include <metal_stdlib>
using namespace metal;
// Independent Metal opaque temporal supersampling at native resolution.
// Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014:
// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
// Metal fragment positions and texture rows use the shared top-left convention.
// Explicit matrix rows preserve the host's 128-byte, row-major uniform contract.
struct TemporalUniforms
{
    float4 PreviousInverseCamera[4];
    float4 PreviousOpticalPlane;
    float4 ExtentFeedbackTolerance;
    float4 JitterDifference;
    uint4 HistoryState;
};
struct TemporalVertex { float4 Position [[position]]; };
struct TemporalOutput
{
    float4 Radiance [[color(0)]];
    float HardwareDepth [[color(1)]];
};
constant float PixelCenter = 0.5;
constant float HalfWeight = 0.5;
constant float QuarterWeight = 0.25;
constant float ClearDepth = 1.0;
constant float MinimumHomogeneousW = 1.0e-7;
constant float MinimumDepthTolerance = 1.0e-4;
constant float MinimumBoxRadius = 1.0e-6;
constant float MaximumHalfFloatRadiance = 65504.0;
constant int NeighborhoodRadius = 1;
constant int DepthFootprintWidth = 2;

vertex TemporalVertex TemporalVertexMain(uint vertexId [[vertex_id]])
{
    const float TriangleExtent = 3.0;
    float2 point = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(TriangleExtent, -1.0) : float2(-1.0, TriangleExtent);
    return {float4(point, 0.0, 1.0)};
}
float3 EncodeColor(float3 rgb)
{
    return float3((rgb.r + 2.0 * rgb.g + rgb.b) * QuarterWeight,
        (rgb.r - rgb.b) * HalfWeight, (2.0 * rgb.g - rgb.r - rgb.b) * QuarterWeight);
}
float3 DecodeColor(float3 ycocg)
{
    float averageRedBlue = ycocg.x - ycocg.z;
    return float3(averageRedBlue + ycocg.y, ycocg.x + ycocg.z, averageRedBlue - ycocg.y);
}
float PreviousOpticalDistance(float2 position, float hardwareDepth, constant TemporalUniforms& uniforms)
{
    float2 ndc = position / uniforms.ExtentFeedbackTolerance.xy * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float4 clip(ndc, hardwareDepth, 1.0);
    float4 world(dot(uniforms.PreviousInverseCamera[0], clip), dot(uniforms.PreviousInverseCamera[1], clip),
        dot(uniforms.PreviousInverseCamera[2], clip), dot(uniforms.PreviousInverseCamera[3], clip));
    if (abs(world.w) <= MinimumHomogeneousW) return -1.0;
    return dot(uniforms.PreviousOpticalPlane, float4(world.xyz / world.w, 1.0));
}
bool AcceptHistoryDepth(float2 position, float predictedDepth, bool background,
    constant TemporalUniforms& uniforms, texture2d<float, access::read> previousDepth)
{
    int2 first = int2(floor(position - PixelCenter));
    int2 extent = int2(uniforms.ExtentFeedbackTolerance.xy);
    float expected = PreviousOpticalDistance(position, predictedDepth, uniforms);
    float tolerance = max(MinimumDepthTolerance, abs(expected) * uniforms.ExtentFeedbackTolerance.w);
    // Compare four real point depths. Interpolating depth at an edge would
    // manufacture a surface between foreground and background.
    for (int y = 0; y < DepthFootprintWidth; ++y)
        for (int x = 0; x < DepthFootprintWidth; ++x)
        {
            int2 candidate = first + int2(x, y);
            if (any(candidate < int2(0)) || any(candidate >= extent)) continue;
            float depth = previousDepth.read(uint2(candidate)).r;
            if (background)
            {
                if (depth >= ClearDepth) return true;
            }
            else if (depth < ClearDepth)
            {
                float observed = PreviousOpticalDistance(float2(candidate) + PixelCenter, depth, uniforms);
                if (isfinite(expected) && isfinite(observed) && abs(observed - expected) <= tolerance) return true;
            }
        }
    return false;
}
float3 ClipHistoryColor(int2 pixel, float3 centerColor, float3 historyColor,
    constant TemporalUniforms& uniforms, texture2d<float, access::read> currentColor)
{
    float3 lower = EncodeColor(centerColor), upper = lower;
    int2 last = int2(uniforms.ExtentFeedbackTolerance.xy) - 1;
    for (int y = -NeighborhoodRadius; y <= NeighborhoodRadius; ++y)
        for (int x = -NeighborhoodRadius; x <= NeighborhoodRadius; ++x)
        {
            uint2 neighbor = uint2(clamp(pixel + int2(x, y), int2(0), last));
            float3 color = EncodeColor(currentColor.read(neighbor).rgb);
            lower = min(lower, color); upper = max(upper, color);
        }
    float3 center = (lower + upper) * HalfWeight;
    float3 radius = max((upper - lower) * HalfWeight, float3(MinimumBoxRadius));
    float3 offset = EncodeColor(historyColor) - center;
    float3 relative = abs(offset / radius);
    float scale = max(1.0, max(relative.x, max(relative.y, relative.z)));
    return DecodeColor(center + offset / scale);
}
fragment TemporalOutput TemporalFragmentMain(TemporalVertex input [[stage_in]],
    constant TemporalUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> currentColor [[texture(0)]],
    depth2d<float, access::read> currentDepth [[texture(1)]],
    texture2d<float, access::read> velocity [[texture(2)]],
    texture2d<float, access::sample> previousColor [[texture(3)]],
    texture2d<float, access::read> previousDepth [[texture(4)]])
{
    uint2 pixel = uint2(input.Position.xy);
    TemporalOutput output {currentColor.read(pixel), currentDepth.read(pixel)};
    if (uniforms.HistoryState.x == 0 || uniforms.ExtentFeedbackTolerance.z <= 0.0) return output;
    bool background = output.HardwareDepth >= ClearDepth;
    float4 motion = velocity.read(pixel);
    if (!background && motion.w <= 0.0) return output;
    float2 colorPosition = input.Position.xy - (background ? float2(0.0) : motion.xy);
    // Accumulated color follows physical motion. Raw depth also follows the
    // previous raster grid, whose camera jitter differs from this frame's grid.
    float2 depthPosition = colorPosition - uniforms.JitterDifference.xy;
    float2 extent = uniforms.ExtentFeedbackTolerance.xy;
    if (any(colorPosition < PixelCenter) || any(colorPosition > extent - PixelCenter) ||
        any(depthPosition < 0.0) || any(depthPosition >= extent)) return output;
    if (!AcceptHistoryDepth(depthPosition, background ? ClearDepth : 1-motion.z, background, uniforms, previousDepth)) return output;
    constexpr sampler HistorySampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 oldColor = previousColor.sample(HistorySampler, colorPosition / extent, level(0.0)).rgb;
    float3 clipped = ClipHistoryColor(int2(pixel), output.Radiance.rgb, oldColor, uniforms, currentColor);
    output.Radiance.rgb = clamp(mix(output.Radiance.rgb, clipped, uniforms.ExtentFeedbackTolerance.z),
        float3(0.0), float3(MaximumHalfFloatRadiance));
    return output; // Alpha and hardware depth always describe the current frame.
}
