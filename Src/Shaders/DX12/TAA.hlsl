// DX12 opaque temporal reconstruction. The host records one immutable constant
// snapshot, five texture tables, and two MRT outputs into its command list.
// Color history follows physical motion; raw depth also follows raster jitter.
// Optical-depth rejection and a current YCoCg box bound constrain old radiance.
// Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014.
// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
// This variant accumulates native-resolution rigid surfaces before transparency.
cbuffer TemporalFrameParameters : register(b0)
{
    row_major float4x4 PreviousClipToWorld;
    float4 OpticalDistancePlane;
    float4 FrameSizeFeedbackTolerance;
    float4 SamplingOffset; // current-minus-previous jitter in top-left pixels
    uint4 HistoryState;
};
Texture2D<float4> SceneRadiance : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> SurfaceMotion : register(t2);
Texture2D<float4> CommittedRadiance : register(t3);
Texture2D<float> CommittedDepth : register(t4);
SamplerState HistoryFilter : register(s0);

static const float HalfPixel = 0.5f;
static const float HalfWeight = 0.5f;
static const float QuarterWeight = 0.25f;
static const float BackgroundDepth = 1.0f;
static const float MinimumClipW = 1.0e-7f;
static const float MinimumDepthAgreement = 1.0e-4f;
static const float MinimumBoxExtent = 1.0e-6f;
static const float HalfFloatRadianceLimit = 65504.0f;
static const int ColorNeighborhoodRadius = 1;

float3 EncodeColor(float3 rgb)
{
    return float3((rgb.r + 2 * rgb.g + rgb.b) * QuarterWeight,
        (rgb.r - rgb.b) * HalfWeight, (2 * rgb.g - rgb.r - rgb.b) * QuarterWeight);
}
float3 DecodeColor(float3 ycocg)
{
    float redBlueMean = ycocg.x - ycocg.z;
    return float3(redBlueMean + ycocg.y, ycocg.x + ycocg.z, redBlueMean - ycocg.y);
}
float OpticalDistance(float2 rasterPosition, float depth)
{
    float2 ndc = rasterPosition / FrameSizeFeedbackTolerance.xy * float2(2, -2) + float2(-1, 1);
    float4 homogeneousWorld = mul(PreviousClipToWorld, float4(ndc, depth, 1));
    if (abs(homogeneousWorld.w) <= MinimumClipW) return -1;
    return dot(OpticalDistancePlane, float4(homogeneousWorld.xyz / homogeneousWorld.w, 1));
}
bool HasCorrespondingDepth(float2 projectedPixel, float expectedDepth, bool background)
{
    int2 footprintOrigin = int2(floor(projectedPixel - HalfPixel));
    int2 dimensions = int2(FrameSizeFeedbackTolerance.xy);
    float predictedDistance = OpticalDistance(projectedPixel, expectedDepth);
    float tolerance = max(MinimumDepthAgreement, abs(predictedDistance) * FrameSizeFeedbackTolerance.w);
    // Point comparisons against a 2x2 footprint do not interpolate a fictitious
    // depth between foreground and background. Any corresponding tap suffices.
    for (int tapY = 0; tapY < 2; ++tapY)
        for (int tapX = 0; tapX < 2; ++tapX)
        {
            int2 texel = footprintOrigin + int2(tapX, tapY);
            if (any(texel < 0) || any(texel >= dimensions)) continue;
            float oldDepth = CommittedDepth.Load(int3(texel, 0));
            if (background)
            {
                if (oldDepth >= BackgroundDepth) return true;
            }
            else if (oldDepth < BackgroundDepth)
            {
                float oldDistance = OpticalDistance(float2(texel) + HalfPixel, oldDepth);
                if (isfinite(predictedDistance) && isfinite(oldDistance) &&
                    abs(oldDistance - predictedDistance) <= tolerance) return true;
            }
        }
    return false;
}
float3 ClipOldRadiance(int2 centerPixel, float3 centerRadiance, float3 oldRadiance)
{
    float3 low = EncodeColor(centerRadiance), high = low;
    int2 lastTexel = int2(FrameSizeFeedbackTolerance.xy) - 1;
    for (int row = -ColorNeighborhoodRadius; row <= ColorNeighborhoodRadius; ++row)
        for (int column = -ColorNeighborhoodRadius; column <= ColorNeighborhoodRadius; ++column)
        {
            int2 neighbor = clamp(centerPixel + int2(column, row), int2(0, 0), lastTexel);
            float3 radiance = EncodeColor(SceneRadiance.Load(int3(neighbor, 0)).rgb);
            low = min(low, radiance); high = max(high, radiance);
        }
    float3 boxCenter = (low + high) * HalfWeight;
    float3 extent = max((high - low) * HalfWeight, MinimumBoxExtent);
    float3 offset = EncodeColor(oldRadiance) - boxCenter;
    float3 normalizedOffset = abs(offset / extent);
    float scale = max(1, max(normalizedOffset.r, max(normalizedOffset.g, normalizedOffset.b)));
    return DecodeColor(boxCenter + offset / scale);
}
struct ScreenPoint { float4 Position : SV_POSITION; };
ScreenPoint VSMain(uint vertexIndex : SV_VertexID)
{
    float2 triangleCorner = float2((vertexIndex << 1) & 2, vertexIndex & 2);
    ScreenPoint result;
    result.Position = float4(triangleCorner * float2(2, -2) + float2(-1, 1), 0, 1);
    return result;
}
struct HistoryOutput
{
    float4 Radiance : SV_TARGET0;
    float HardwareDepth : SV_TARGET1;
};
HistoryOutput PSMain(ScreenPoint input)
{
    int2 texel = int2(input.Position.xy);
    HistoryOutput result;
    result.Radiance = SceneRadiance.Load(int3(texel, 0));
    result.HardwareDepth = SceneDepth.Load(int3(texel, 0));
    if (HistoryState.x == 0 || FrameSizeFeedbackTolerance.z <= 0) return result;
    bool background = result.HardwareDepth >= BackgroundDepth;
    float4 correspondence = SurfaceMotion.Load(int3(texel, 0));
    if (!background && correspondence.w <= 0) return result;
    float2 physicalTravel = background ? float2(0, 0) : correspondence.xy;
    float2 colorPosition = input.Position.xy - physicalTravel;
    float2 depthPosition = colorPosition - SamplingOffset.xy;
    float2 dimensions = FrameSizeFeedbackTolerance.xy;
    if (any(colorPosition < HalfPixel) || any(colorPosition > dimensions - HalfPixel) ||
        any(depthPosition < 0) || any(depthPosition >= dimensions)) return result;
    if (!HasCorrespondingDepth(depthPosition, background ? BackgroundDepth : 1-correspondence.z, background)) return result;
    float3 oldRadiance = CommittedRadiance.SampleLevel(HistoryFilter, colorPosition / dimensions, 0).rgb;
    float3 boundedHistory = ClipOldRadiance(texel, result.Radiance.rgb, oldRadiance);
    result.Radiance.rgb = clamp(lerp(result.Radiance.rgb, boundedHistory, FrameSizeFeedbackTolerance.z), 0, HalfFloatRadianceLimit);
    return result; // source alpha/depth never accumulate from earlier frames
}
