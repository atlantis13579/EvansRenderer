// Temporal supersampling of opaque HDR radiance. Halton jitter is applied by
// geometry; physical motion reprojects accumulated color, while jitter-adjusted
// motion reprojects raw depth. Reject disocclusions, clip history to the current
// 3x3 YCoCg neighborhood, and blend before exposure and tone mapping.
// Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014.
// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
// This teaching implementation has no transparency/reactive mask or upscaling.
cbuffer TemporalConstants : register(b0)
{
    row_major float4x4 PreviousInverseViewProjection;
    float4 PreviousDepthPlane;
    float4 ExtentWeightTolerance; // dimensions, history feedback, relative depth tolerance
    float4 JitterDelta; // current minus previous top-left jitter, in pixels
    uint4 Options; // valid committed history
};
Texture2D<float4> CurrentColor : register(t0);
Texture2D<float> CurrentDepth : register(t1);
Texture2D<float4> MotionVectors : register(t2);
Texture2D<float4> HistoryColor : register(t3);
Texture2D<float> HistoryDepth : register(t4);
SamplerState LinearClamp : register(s0);

static const float PixelCenter = 0.5f;
static const float ClearDepth = 1.0f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float AbsoluteDepthTolerance = 1.0e-4f;
static const float MinimumColorExtent = 1.0e-6f;
static const float MaximumHalfFloat = 65504.0f;
static const int NeighborhoodRadius = 1;
// YCoCg separates luminance from chroma for a compact neighborhood bound.
static const float3 LuminanceWeights = float3(0.25f, 0.5f, 0.25f);
static const float3 OrangeWeights = float3(0.5f, 0.0f, -0.5f);
static const float3 GreenWeights = float3(-0.25f, 0.5f, -0.25f);
float3 ToYCoCg(float3 value)
{
    return float3(dot(value, LuminanceWeights), dot(value, OrangeWeights), dot(value, GreenWeights));
}
float3 ToRGB(float3 value)
{
    return float3(value.x + value.y - value.z, value.x + value.z, value.x - value.y - value.z);
}
float PreviousDistance(float2 pixel, float depth)
{
    float2 uv = pixel / ExtentWeightTolerance.xy;
    float4 world = mul(PreviousInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(world.w) <= HomogeneousEpsilon) return -1.0f;
    return dot(PreviousDepthPlane, float4(world.xyz / world.w, 1));
}
bool DepthAgrees(float2 previousPixel, float expected, bool background)
{
    // Test the four texels surrounding the previous geometric sample. Point
    // depth avoids inventing a surface by bilinear interpolation across an edge.
    int2 extent = int2(ExtentWeightTolerance.xy);
    int2 base = int2(floor(previousPixel - PixelCenter));
    float expectedDistance = PreviousDistance(previousPixel, expected);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x)
    {
        int2 pixel = base + int2(x, y);
        if (any(pixel < 0) || any(pixel >= extent)) continue;
        float depth = HistoryDepth.Load(int3(pixel, 0));
        if (background) { if (depth >= ClearDepth) return true; else continue; }
        if (depth >= ClearDepth) continue;
        float distance = PreviousDistance(float2(pixel) + PixelCenter, depth);
        float tolerance = max(AbsoluteDepthTolerance, abs(expectedDistance) * ExtentWeightTolerance.w);
        if (isfinite(distance) && isfinite(expectedDistance) && abs(distance - expectedDistance) <= tolerance) return true;
    }
    return false;
}
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    float2 corner = float2((id << 1) & 2, id & 2);
    ScreenVertex output;
    output.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}
struct TemporalOutput { float4 Color : SV_TARGET0; float Depth : SV_TARGET1; };
TemporalOutput PSMain(ScreenVertex input)
{
    int2 pixel = int2(input.Position.xy), extent = int2(ExtentWeightTolerance.xy);
    float4 current = CurrentColor.Load(int3(pixel, 0));
    TemporalOutput output;
    output.Color = current;
    output.Depth = CurrentDepth.Load(int3(pixel, 0));
    if (Options.x == 0 || ExtentWeightTolerance.z <= 0) return output;
    bool background = output.Depth >= ClearDepth;
    float4 motion = MotionVectors.Load(int3(pixel, 0));
    if (!background && motion.w <= 0) return output;
    float2 displacement = background ? float2(0, 0) : motion.xy;
    float2 previousColorPixel = input.Position.xy - displacement;
    float2 previousDepthPixel = previousColorPixel - JitterDelta.xy;
    if (any(previousColorPixel < PixelCenter) || any(previousColorPixel > ExtentWeightTolerance.xy - PixelCenter) ||
        any(previousDepthPixel < 0) || any(previousDepthPixel >= ExtentWeightTolerance.xy)) return output;
    if (!DepthAgrees(previousDepthPixel, background ? ClearDepth : 1-motion.z, background)) return output;
    float3 minimumColor = ToYCoCg(current.rgb), maximumColor = minimumColor;
    for (int y = -NeighborhoodRadius; y <= NeighborhoodRadius; ++y)
        for (int x = -NeighborhoodRadius; x <= NeighborhoodRadius; ++x)
        {
            int2 samplePixel = clamp(pixel + int2(x, y), int2(0, 0), extent - 1);
            float3 sampleColor = ToYCoCg(CurrentColor.Load(int3(samplePixel, 0)).rgb);
            minimumColor = min(minimumColor, sampleColor);
            maximumColor = max(maximumColor, sampleColor);
        }
    float3 history = ToYCoCg(HistoryColor.SampleLevel(LinearClamp, previousColorPixel / ExtentWeightTolerance.xy, 0).rgb);
    float3 center = (minimumColor + maximumColor) * 0.5f;
    float3 halfExtent = max((maximumColor - minimumColor) * 0.5f, MinimumColorExtent);
    float3 delta = history - center;
    float3 ratio = abs(delta / halfExtent);
    float scale = max(1.0f, max(ratio.x, max(ratio.y, ratio.z)));
    history = ToRGB(center + delta / scale);
    output.Color.rgb = clamp(lerp(current.rgb, history, ExtentWeightTolerance.z), 0, MaximumHalfFloat);
    return output; // alpha and hardware depth belong to this frame, never history
}
