#version 300 es
// Native-resolution opaque temporal supersampling, before transparency.
// Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014:
// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
// GL images have a bottom-left origin; motion and camera reconstruction use
// top-left pixels. Convert only at image access, not in the motion contract.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenExtent = 3.0;
    vec2 corner = gl_VertexID == 0 ? vec2(-1.0) :
        gl_VertexID == 1 ? vec2(FullscreenExtent, -1.0) : vec2(-1.0, FullscreenExtent);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
uniform mat4 PreviousInverseCamera;
uniform vec4 PreviousOpticalPlane;
uniform vec4 ExtentFeedbackTolerance;
uniform vec4 JitterDifference;
uniform uvec4 HistoryOptions;
uniform highp sampler2D CurrentColor;
uniform highp sampler2D CurrentDepth;
uniform highp sampler2D PhysicalMotion;
uniform highp sampler2D HistoryColor;
uniform highp sampler2D HistoryDepth;
layout(location = 0) out vec4 AccumulatedColor;
layout(location = 1) out float CapturedDepth;

const float PixelCenter = 0.5;
const float HalfWeight = 0.5;
const float ClearDepth = 1.0;
const float MinimumHomogeneousW = 1.0e-7;
const float MinimumOpticalTolerance = 1.0e-4;
const float MinimumColorExtent = 1.0e-6;
const float MaximumHalfFloatRadiance = 65504.0;
const int NeighborhoodRadius = 1;
const int DepthFootprintWidth = 2;

ivec2 ImageTexel(ivec2 topLeftTexel)
{
    return ivec2(topLeftTexel.x, int(ExtentFeedbackTolerance.y) - 1 - topLeftTexel.y);
}
vec2 ImageUV(vec2 topLeftPosition)
{
    return vec2(topLeftPosition.x, ExtentFeedbackTolerance.y - topLeftPosition.y) / ExtentFeedbackTolerance.xy;
}
vec3 ToLuminanceChroma(vec3 rgb)
{
    float redBlueMean = (rgb.r + rgb.b) * HalfWeight;
    return vec3((redBlueMean + rgb.g) * HalfWeight, (rgb.r - rgb.b) * HalfWeight,
        (rgb.g - redBlueMean) * HalfWeight);
}
vec3 ToRGB(vec3 ycocg)
{
    return vec3(ycocg.x + ycocg.y - ycocg.z, ycocg.x + ycocg.z, ycocg.x - ycocg.y - ycocg.z);
}
float PreviousDistance(vec2 position, float depth)
{
    vec2 ndc = position / ExtentFeedbackTolerance.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    // The geometry shader remaps clip Z for GL. Stored depth is already [0,1],
    // matching the canonical previous camera matrix; no second remap is needed.
    vec4 world = PreviousInverseCamera * vec4(ndc, depth, 1.0);
    if (abs(world.w) <= MinimumHomogeneousW) return -1.0;
    return dot(PreviousOpticalPlane, vec4(world.xyz / world.w, 1.0));
}
bool Finite(float value) { return !isnan(value) && !isinf(value); }
bool DepthCorresponds(vec2 position, float expectedDepth, bool sky)
{
    ivec2 first = ivec2(floor(position - PixelCenter));
    ivec2 extent = ivec2(ExtentFeedbackTolerance.xy);
    float expectedDistance = PreviousDistance(position, expectedDepth);
    float tolerance = max(MinimumOpticalTolerance, abs(expectedDistance) * ExtentFeedbackTolerance.w);
    // Test point depths separately: bilinear depth would invent a surface
    // between silhouette foreground and background. Any matching tap suffices.
    for (int y = 0; y < DepthFootprintWidth; ++y)
        for (int x = 0; x < DepthFootprintWidth; ++x)
        {
            ivec2 pixel = first + ivec2(x, y);
            if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, extent))) continue;
            float depth = texelFetch(HistoryDepth, ImageTexel(pixel), 0).r;
            if (sky)
            {
                if (depth >= ClearDepth) return true;
            }
            else if (depth < ClearDepth)
            {
                float distance = PreviousDistance(vec2(pixel) + PixelCenter, depth);
                if (Finite(expectedDistance) && Finite(distance) && abs(distance - expectedDistance) <= tolerance) return true;
            }
        }
    return false;
}
vec3 BoundHistory(ivec2 pixel, vec3 previous)
{
    vec3 lower = ToLuminanceChroma(AccumulatedColor.rgb), upper = lower;
    ivec2 last = ivec2(ExtentFeedbackTolerance.xy) - 1;
    for (int y = -NeighborhoodRadius; y <= NeighborhoodRadius; ++y)
        for (int x = -NeighborhoodRadius; x <= NeighborhoodRadius; ++x)
        {
            ivec2 neighbor = clamp(pixel + ivec2(x, y), ivec2(0), last);
            vec3 sampleColor = ToLuminanceChroma(texelFetch(CurrentColor, ImageTexel(neighbor), 0).rgb);
            lower = min(lower, sampleColor); upper = max(upper, sampleColor);
        }
    vec3 center = (lower + upper) * HalfWeight;
    vec3 extent = max((upper - lower) * HalfWeight, vec3(MinimumColorExtent));
    vec3 offset = ToLuminanceChroma(previous) - center;
    vec3 relative = abs(offset / extent);
    float divisor = max(1.0, max(relative.x, max(relative.y, relative.z)));
    return ToRGB(center + offset / divisor);
}
void main()
{
    ivec2 nativePixel = ivec2(gl_FragCoord.xy);
    ivec2 pixel = ImageTexel(nativePixel); // The integer origin flip is its own inverse.
    vec2 position = vec2(pixel) + PixelCenter;
    AccumulatedColor = texelFetch(CurrentColor, nativePixel, 0);
    CapturedDepth = texelFetch(CurrentDepth, nativePixel, 0).r;
    if (HistoryOptions.x == 0u || ExtentFeedbackTolerance.z <= 0.0) return;
    bool sky = CapturedDepth >= ClearDepth;
    vec4 motion = texelFetch(PhysicalMotion, nativePixel, 0);
    if (!sky && motion.w <= 0.0) return;
    vec2 oldColorPosition = position - (sky ? vec2(0.0) : motion.xy);
    vec2 oldDepthPosition = oldColorPosition - JitterDifference.xy;
    if (any(lessThan(oldColorPosition, vec2(PixelCenter))) ||
        any(greaterThan(oldColorPosition, ExtentFeedbackTolerance.xy - PixelCenter)) ||
        any(lessThan(oldDepthPosition, vec2(0.0))) ||
        any(greaterThanEqual(oldDepthPosition, ExtentFeedbackTolerance.xy))) return;
    if (!DepthCorresponds(oldDepthPosition, sky ? ClearDepth : 1.0-motion.z, sky)) return;
    vec3 previous = textureLod(HistoryColor, ImageUV(oldColorPosition), 0.0).rgb;
    vec3 bounded = BoundHistory(pixel, previous);
    AccumulatedColor.rgb = clamp(mix(AccumulatedColor.rgb, bounded, ExtentFeedbackTolerance.z),
        vec3(0.0), vec3(MaximumHalfFloatRadiance));
    // Alpha and hardware depth describe this frame only; neither is accumulated.
}
#endif
