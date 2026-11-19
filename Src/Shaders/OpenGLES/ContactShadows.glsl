#version 310 es
#ifdef EVANS_VERTEX_STAGE

// Native line primitives have no face classification. Reject all endpoints of a
// back-facing original triangle, retaining the native line coverage and clipping.
invariant gl_Position;
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}
#endif
#ifdef EVANS_CONTACT_DEPTH
// Camera depth uses the same matrix association, UV orientation and alpha
// coverage as the visible ES surfaces. Instancing supplies the existing ES
// Forward vertex interface; no lighting or normal reconstruction is required.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
// Match camera-depth producers and consumers across separately linked programs.
// GLSL ES 3.00, section 4.6: invariant raster positions prevent replay holes.

layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World, ViewProjection;
out vec4 VertexColor;
out vec2 TexCoord;
void main()
{
    vec4 world = World * vec4(InPosition, 1.0);
    vec4 clip = ViewProjection * world;
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    VertexColor = InColor;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    CullWireFace();
}
#else
in vec4 VertexColor;
in vec2 TexCoord;
uniform sampler2D DiffuseMap, OpacityMap;
uniform vec4 MaterialParameters, TextureFlags;
const float EnabledThreshold = 0.5;
void main()
{
    float alpha = VertexColor.a * MaterialParameters.y;
    if (TextureFlags.x > EnabledThreshold) alpha *= texture(DiffuseMap, TexCoord).a;
    if (TextureFlags.z > EnabledThreshold) alpha *= texture(OpacityMap, TexCoord).r;
    if (MaterialParameters.z > 0.0 && clamp(alpha, 0.0, 1.0) < MaterialParameters.z) discard;
}
#endif
#else
// Independent ES implementation of finite-depth-slab contact visibility.
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing" (2014):
// https://jcgt.org/published/0003/04/04/paper.pdf
// Traverse exact screen-cell intervals with axial z/W and reciprocal W.
// The CPU contract uses top-left pixels and [0,1] depth. ES storage uses
// bottom-left pixels; convert only at texture accesses and the output store.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DMS;
precision highp image2DArray;

#define CONTACT_GROUP_WIDTH 8
layout(local_size_x = CONTACT_GROUP_WIDTH, local_size_y = CONTACT_GROUP_WIDTH, local_size_z = 1) in;
layout(r32f, binding = 0) writeonly uniform image2DArray Visibility;
#ifdef EVANS_MULTISAMPLE
uniform sampler2DMS CameraDepth;
#else
uniform sampler2D CameraDepth;
#endif
uniform mat4 Camera, InverseCamera;
uniform vec4 DepthMetric; // near, far, perspective, positive W guard
uniform vec4 LightPositionType, LightDirection;
uniform vec4 TraceParameters; // maximum world length, start bias, axial thickness
uniform uvec4 ImageAndBudget; // width, height, visited-cell budget, samples
const int MaximumDepthSamples = 8;
uniform vec4 SampleLocations[MaximumDepthSamples]; // actual ES [0,1] positions, fixed per image

const float ClearDepth = 1.0;
const float Half = 0.5;
const float DirectionEpsilon = 1.0e-12;
const float FloatEpsilon = 1.1920928955078125e-7;
const float CrossingTolerance = 4.0 * FloatEpsilon;
const float DepthRoundoff = 4.0 * FloatEpsilon;
const float BeyondSegment = 2.0;

ivec2 NativePixel(ivec2 canonical)
{ return ivec2(canonical.x, int(ImageAndBudget.y) - 1 - canonical.y); }
float ReadDepth(ivec2 pixel, int sampleIndex)
{
#ifdef EVANS_MULTISAMPLE
    return texelFetch(CameraDepth, NativePixel(pixel), sampleIndex).r;
#else
    return texelFetch(CameraDepth, NativePixel(pixel), 0).r;
#endif
}
float AxialDepth(float stored)
{
    float nearPlane = DepthMetric.x, farPlane = DepthMetric.y;
    return DepthMetric.z > Half ? nearPlane * farPlane / (farPlane - stored * (farPlane - nearPlane))
        : mix(nearPlane, farPlane, stored);
}
bool RestrictSegment(float first, float last, inout vec2 interval)
{
    if (first < 0.0 && last < 0.0) return false;
    if ((first < 0.0) != (last < 0.0))
    {
        float crossing = first / (first - last);
        if (first < 0.0) interval.x = max(interval.x, crossing);
        else interval.y = min(interval.y, crossing);
    }
    return interval.x < interval.y;
}
bool ClipRay(vec4 a, vec4 b, out vec2 interval)
{
    interval = vec2(0.0, 1.0);
    return RestrictSegment(a.w - DepthMetric.w, b.w - DepthMetric.w, interval)
        && RestrictSegment(a.w + a.x, b.w + b.x, interval)
        && RestrictSegment(a.w - a.x, b.w - b.x, interval)
        && RestrictSegment(a.w + a.y, b.w + b.y, interval)
        && RestrictSegment(a.w - a.y, b.w - b.y, interval)
        && RestrictSegment(a.z, b.z, interval)
        && RestrictSegment(a.w - a.z, b.w - b.z, interval);
}
vec2 ScreenPixel(vec4 projected)
{ return (projected.xy / projected.w * vec2(Half, -Half) + Half) * vec2(ImageAndBudget.xy); }

float TraceVisibility(vec3 receiver, ivec2 receiverCell, int sampleIndex)
{
    bool directional = LightPositionType.w < Half;
    vec3 towardLight = directional ? LightDirection.xyz : LightPositionType.xyz - receiver;
    float lightDistance = length(towardLight);
    if (lightDistance <= DirectionEpsilon) return 1.0;
    float rayLength = directional ? TraceParameters.x : min(TraceParameters.x, lightDistance);
    if (rayLength <= TraceParameters.y) return 1.0;
    // Directional vectors are normalized by the host contract. Normalizing a
    // second time perturbs corner ownership without adding useful precision.
    vec3 direction = directional ? towardLight : towardLight / lightDistance;
    vec3 firstWorld = receiver + direction * TraceParameters.y;
    vec3 lastWorld = receiver + direction * rayLength;
    vec4 first = Camera * vec4(firstWorld, 1.0);
    vec4 last = Camera * vec4(lastWorld, 1.0);
    vec2 clipInterval;
    if (!ClipRay(first, last, clipInterval)) return 1.0;
    // Reproject clipped world endpoints; avoid cancellation from interpolating
    // already projected endpoints close to the positive-W guard.
    first = Camera * vec4(mix(firstWorld, lastWorld, clipInterval.x), 1.0);
    last = Camera * vec4(mix(firstWorld, lastWorld, clipInterval.y), 1.0);
    vec2 screenStart = ScreenPixel(first), delta = ScreenPixel(last) - screenStart;
    if (max(abs(delta.x), abs(delta.y)) <= DirectionEpsilon) return 1.0;
    // Projected coordinates retain viewport-scale roundoff even when only a
    // fraction of a pixel survives clipping. Convert that error to progress.
    float pixelUncertainty = CrossingTolerance * float(max(ImageAndBudget.x, ImageAndBudget.y));
    float intervalTolerance = pixelUncertainty / max(abs(delta.x), abs(delta.y));
    // A near-corner comparison combines X and Y crossing-time uncertainty.
    float minorDelta = min(abs(delta.x), abs(delta.y));
    float cornerTolerance = intervalTolerance + (minorDelta > DirectionEpsilon ? pixelUncertainty / minorDelta : 0.0);
    vec2 reciprocalW = vec2(1.0 / first.w, 1.0 / last.w);
    vec2 axialOverW = vec2(AxialDepth(first.z * reciprocalW.x), AxialDepth(last.z * reciprocalW.y)) * reciprocalW;

    ivec2 extent = ivec2(ImageAndBudget.xy), cell = ivec2(floor(screenStart));
    ivec2 directionCell = ivec2(1);
    vec2 crossing = vec2(BeyondSegment), increment = vec2(BeyondSegment);
    for (int axis = 0; axis < 2; ++axis)
    {
        if (delta[axis] < 0.0)
        {
            directionCell[axis] = -1;
            if (screenStart[axis] == floor(screenStart[axis])) --cell[axis];
        }
        cell[axis] = clamp(cell[axis], 0, extent[axis] - 1);
        if (abs(delta[axis]) > DirectionEpsilon)
        {
            float boundary = float(cell[axis] + (directionCell[axis] > 0 ? 1 : 0));
            crossing[axis] = max(0.0, (boundary - screenStart[axis]) / delta[axis]);
            increment[axis] = 1.0 / abs(delta[axis]);
        }
    }
    float begin = 0.0;
    for (uint visited = 0u; visited < ImageAndBudget.z && begin < 1.0; ++visited)
    {
        if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, extent))) return 1.0;
        float end = min(1.0, min(crossing.x, crossing.y));
        float stored = ReadDepth(cell, sampleIndex);
        if (end > begin + intervalTolerance && any(notEqual(cell, receiverCell)) && stored < ClearDepth)
        {
            float front = AxialDepth(stored), back = front + TraceParameters.z;
            float entryDepth = mix(axialOverW.x, axialOverW.y, begin) / mix(reciprocalW.x, reciprocalW.y, begin);
            float exitDepth = mix(axialOverW.x, axialOverW.y, end) / mix(reciprocalW.x, reciprocalW.y, end);
            // Positive W makes axial depth monotonic on this cell interval.
            // Resolvable interval overlap proves a hit without solving its position.
            float depthUncertainty = DepthRoundoff * max(1.0,
                max(max(abs(front), abs(back)), max(abs(entryDepth), abs(exitDepth))));
            if (max(entryDepth, exitDepth) >= front + depthUncertainty &&
                min(entryDepth, exitDepth) <= back - depthUncertainty) return 0.0;
        }
        if (end >= 1.0) return 1.0;
        begin = max(begin, end);
        for (int axis = 0; axis < 2; ++axis)
            if (crossing[axis] <= end + cornerTolerance)
            { begin = max(begin, crossing[axis]); cell[axis] += directionCell[axis]; crossing[axis] += increment[axis]; }
    }
    return 1.0; // Missing/off-screen geometry and exhausted budgets remain lit.
}
void main()
{
    uvec3 invocation = gl_GlobalInvocationID;
    if (any(greaterThanEqual(invocation.xy, ImageAndBudget.xy)) || invocation.z >= ImageAndBudget.w) return;
    ivec2 pixel = ivec2(invocation.xy);
    int sampleIndex = int(invocation.z);
    float stored = ReadDepth(pixel, sampleIndex), visibility = 1.0;
    if (stored >= 0.0 && stored < ClearDepth)
    {
        vec2 samplePosition = SampleLocations[sampleIndex].xy;
        vec2 canonicalPosition = vec2(pixel) + vec2(samplePosition.x, 1.0 - samplePosition.y);
        vec2 ndc = canonicalPosition / vec2(ImageAndBudget.xy) * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
        vec4 homogeneous = InverseCamera * vec4(ndc, stored, 1.0);
        vec3 receiver = homogeneous.xyz / homogeneous.w;
        if (!any(isnan(receiver)) && !any(isinf(receiver))) visibility = TraceVisibility(receiver, pixel, sampleIndex);
    }
    imageStore(Visibility, ivec3(NativePixel(pixel), sampleIndex), vec4(visibility));
}
#endif
