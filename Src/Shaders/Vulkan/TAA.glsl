#version 450
// Independent Vulkan temporal supersampling of opaque, scene-linear radiance.
// Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014:
// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
// Geometry flips clip Y for Vulkan's positive-height viewport. Fragment/image
// coordinates therefore already match the shared top-left motion convention.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleExtent = 3.0;
    vec2 position = gl_VertexIndex == 0 ? vec2(-1.0) :
        gl_VertexIndex == 1 ? vec2(TriangleExtent, -1.0) : vec2(-1.0, TriangleExtent);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform TemporalParameters
{
    mat4 PreviousInverseViewProjection;
    vec4 PreviousDepthPlane;
    vec4 ExtentWeightTolerance;
    vec4 JitterDelta;
    uvec4 Options;
} Temporal;
// Only RadianceHistory uses a linear sampler. In particular, D32 depth must
// not require linear-filter support merely because color history needs it.
layout(set = 0, binding = 1) uniform sampler2D OpaqueRadiance;
layout(set = 0, binding = 2) uniform sampler2D OpaqueDepth;
layout(set = 0, binding = 3) uniform sampler2D RigidVelocity;
layout(set = 0, binding = 4) uniform sampler2D RadianceHistory;
layout(set = 0, binding = 5) uniform sampler2D DepthHistory;
layout(location = 0) out vec4 NextRadiance;
layout(location = 1) out float NextDepth;

const float PixelCenterOffset = 0.5;
const float Half = 0.5;
const float Quarter = 0.25;
const float ClearDepth = 1.0;
const float MinimumProjectionW = 1.0e-7;
const float AbsoluteDepthTolerance = 1.0e-4;
const float MinimumNeighborhoodExtent = 1.0e-6;
const float MaximumHalfFloatRadiance = 65504.0;
const int NeighborhoodRadius = 1;
const int DepthFootprintWidth = 2;

vec3 EncodeYCoCg(vec3 color)
{
    return vec3(dot(color, vec3(Quarter, Half, Quarter)),
        (color.r - color.b) * Half, dot(color, vec3(-Quarter, Half, -Quarter)));
}
vec3 DecodeYCoCg(vec3 color)
{
    float redBlueAverage = color.x - color.z;
    return vec3(redBlueAverage + color.y, color.x + color.z, redBlueAverage - color.y);
}
float OpticalDepth(vec2 pixelPosition, float hardwareDepth)
{
    vec2 clipXY = pixelPosition / Temporal.ExtentWeightTolerance.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    vec4 world = Temporal.PreviousInverseViewProjection * vec4(clipXY, hardwareDepth, 1.0);
    if (abs(world.w) <= MinimumProjectionW) return -1.0;
    return dot(Temporal.PreviousDepthPlane, vec4(world.xyz / world.w, 1.0));
}
bool IsFinite(float value) { return !isnan(value) && !isinf(value); }
bool AcceptPreviousSurface(vec2 previousPixel, float predictedHardwareDepth, bool background)
{
    ivec2 dimensions = ivec2(Temporal.ExtentWeightTolerance.xy);
    ivec2 firstTexel = ivec2(floor(previousPixel - PixelCenterOffset));
    float predictedDistance = OpticalDepth(previousPixel, predictedHardwareDepth);
    float tolerance = max(AbsoluteDepthTolerance, abs(predictedDistance) * Temporal.ExtentWeightTolerance.w);
    // A 2x2 point footprint tests real surfaces instead of interpolating an
    // artificial depth at a silhouette. Any corresponding surface accepts history.
    for (int row = 0; row < DepthFootprintWidth; ++row)
        for (int column = 0; column < DepthFootprintWidth; ++column)
        {
            ivec2 candidate = firstTexel + ivec2(column, row);
            if (any(lessThan(candidate, ivec2(0))) || any(greaterThanEqual(candidate, dimensions))) continue;
            float depth = texelFetch(DepthHistory, candidate, 0).r;
            if (background)
            {
                if (depth >= ClearDepth) return true;
            }
            else if (depth < ClearDepth)
            {
                float distance = OpticalDepth(vec2(candidate) + PixelCenterOffset, depth);
                if (IsFinite(distance) && IsFinite(predictedDistance) && abs(distance - predictedDistance) <= tolerance) return true;
            }
        }
    return false;
}
vec3 ClipToNeighborhood(ivec2 centerPixel, vec3 currentColor, vec3 previousColor)
{
    vec3 lower = EncodeYCoCg(currentColor), upper = lower;
    ivec2 lastTexel = ivec2(Temporal.ExtentWeightTolerance.xy) - 1;
    for (int row = -NeighborhoodRadius; row <= NeighborhoodRadius; ++row)
        for (int column = -NeighborhoodRadius; column <= NeighborhoodRadius; ++column)
        {
            ivec2 pixel = clamp(centerPixel + ivec2(column, row), ivec2(0), lastTexel);
            vec3 value = EncodeYCoCg(texelFetch(OpaqueRadiance, pixel, 0).rgb);
            lower = min(lower, value); upper = max(upper, value);
        }
    vec3 center = (lower + upper) * Half;
    vec3 radius = max((upper - lower) * Half, vec3(MinimumNeighborhoodExtent));
    vec3 displacement = EncodeYCoCg(previousColor) - center;
    vec3 relative = abs(displacement / radius);
    float scale = max(1.0, max(relative.x, max(relative.y, relative.z)));
    return DecodeYCoCg(center + displacement / scale);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    NextRadiance = texelFetch(OpaqueRadiance, pixel, 0);
    NextDepth = texelFetch(OpaqueDepth, pixel, 0).r;
    if (Temporal.Options.x == 0u || Temporal.ExtentWeightTolerance.z <= 0.0) return;
    bool background = NextDepth >= ClearDepth;
    vec4 velocity = texelFetch(RigidVelocity, pixel, 0);
    if (!background && velocity.w <= 0.0) return;
    // Color has accumulated on the output grid; depth retains raster jitter.
    vec2 previousColorPixel = gl_FragCoord.xy - (background ? vec2(0.0) : velocity.xy);
    vec2 previousDepthPixel = previousColorPixel - Temporal.JitterDelta.xy;
    vec2 dimensions = Temporal.ExtentWeightTolerance.xy;
    if (any(lessThan(previousColorPixel, vec2(PixelCenterOffset))) ||
        any(greaterThan(previousColorPixel, dimensions - PixelCenterOffset)) ||
        any(lessThan(previousDepthPixel, vec2(0.0))) || any(greaterThanEqual(previousDepthPixel, dimensions))) return;
    if (!AcceptPreviousSurface(previousDepthPixel, background ? ClearDepth : 1.0-velocity.z, background)) return;
    vec3 previous = textureLod(RadianceHistory, previousColorPixel / dimensions, 0.0).rgb;
    vec3 clipped = ClipToNeighborhood(pixel, NextRadiance.rgb, previous);
    NextRadiance.rgb = clamp(mix(NextRadiance.rgb, clipped, Temporal.ExtentWeightTolerance.z),
        vec3(0.0), vec3(MaximumHalfFloatRadiance));
    // Current alpha and depth remain exact inputs, not temporal averages.
}
#endif
