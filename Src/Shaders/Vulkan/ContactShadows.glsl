#version 450

#if defined(EVANS_CONTACT_DEPTH)
// The camera-depth prerequisite preserves visible alpha coverage without light
// bias. Locations match the independent instanced Forward vertex interface.
layout(set = 0, binding = 0, std140, row_major) uniform CameraDepthDraw
{
    mat4 World;
    mat4 NormalWorld;
    mat4 ViewProjection;
    mat4 LightViewProjection;
    vec4 PipelineColor;
    vec4 MaterialDiffuseColor;
    vec4 MaterialAmbientColor;
    vec4 MaterialSpecularColor;
    vec4 MaterialEmissiveColor;
    vec4 MaterialParameters;
    vec4 AmbientColorStrength;
    vec4 CameraLightCount;
    vec4 RenderParameters;
    uvec4 TextureFlags;
} Draw;
#ifdef EVANS_VERTEX_STAGE
// Match material replay arithmetic so read-only depth testing retains coverage.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 2) out vec4 VertexColor;
layout(location = 4) out vec2 TexCoord;
void main()
{
    vec4 world = Draw.World * vec4(InPosition, 1.0);
    gl_Position = Draw.ViewProjection * world;
    gl_Position.y = -gl_Position.y;
    VertexColor = InColor;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
#else
layout(location = 2) in vec4 VertexColor;
layout(location = 4) in vec2 TexCoord;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
void main()
{
    float diffuseAlpha = Draw.TextureFlags.x != 0u
        ? texture(sampler2D(DiffuseImage, MaterialSampler), TexCoord).a : 1.0;
    float opacity = Draw.TextureFlags.z != 0u
        ? texture(sampler2D(OpacityImage, MaterialSampler), TexCoord).r : 1.0;
    float coverage = clamp(VertexColor.a * Draw.MaterialParameters.y * diffuseAlpha * opacity, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0 && coverage < Draw.MaterialParameters.z) discard;
}
#endif
#else
// Independent Vulkan visibility producer. Perspective interpolation follows
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing" (2014),
// https://jcgt.org/published/0003/04/04/paper.pdf . Exact cell intervals intersect
// finite camera-axis depth slabs; background, self cells and uncertain tangencies
// stay lit.
// Camera matrices use canonical top-left pixels and [0,1] depth. Vulkan image
// rows already follow that convention; sample offsets come from the caller's
// actual raster pattern, in pixel units measured from the pixel centre.
#define CONTACT_WORKGROUP_EDGE 8
#define CONTACT_SAMPLE_CAPACITY 8
layout(local_size_x = CONTACT_WORKGROUP_EDGE, local_size_y = CONTACT_WORKGROUP_EDGE, local_size_z = 1) in;
layout(set = 0, binding = 0, std140, row_major) uniform TraceData
{
    mat4 Camera;
    mat4 InverseCamera;
    vec4 DepthMetric; // near distance, far distance, perspective, positive W guard
    vec4 Emitter; // raw direction or position; W selects directional (0) or finite (1)
    vec4 LightDirection; // normalized direction, reserved by the shared constant layout
    vec4 Trace; // world ray length, world start bias, axial slab thickness
    uvec4 ExtentBudget; // width, height, cell budget, depth samples
    vec4 SampleOffsets[CONTACT_SAMPLE_CAPACITY];
};
#ifdef EVANS_MULTISAMPLE
layout(set = 0, binding = 1) uniform sampler2DMS CameraDepth;
#else
layout(set = 0, binding = 1) uniform sampler2D CameraDepth;
#endif
layout(set = 0, binding = 2, r32f) writeonly uniform image2DArray ContactVisibility;

const float PixelCentre = 0.5;
const float DirectionFloor = 1.0e-12;
const float FloatEpsilon = 1.1920928955078125e-7;
const float RoundoffUlps = 4.0;
const float InfiniteCrossing = 3.402823466e38;

vec3 ProjectiveDivide(vec4 homogeneous)
{
    // Refine the hardware quotient with its fused residual. A reciprocal-based
    // divide may otherwise shift a receiver by an ULP at a depth-slab boundary.
    vec3 position = homogeneous.xyz / homogeneous.w;
    return position + fma(-position, vec3(homogeneous.w), homogeneous.xyz) / homogeneous.w;
}

float ReadDepth(ivec2 pixel, int sampleIndex)
{
#ifdef EVANS_MULTISAMPLE
    return texelFetch(CameraDepth, pixel, sampleIndex).r;
#else
    return texelFetch(CameraDepth, pixel, 0).r;
#endif
}
float AxialDepth(float deviceDepth)
{
    float nearDistance = DepthMetric.x, farDistance = DepthMetric.y;
    // Reciprocal interpolation avoids overflowing a near*far product.
    return DepthMetric.z != 0.0 ? 1.0 /
        ((1.0 - deviceDepth) / nearDistance + deviceDepth / farDistance) : deviceDepth * farDistance;
}
bool ClipHalfSpace(float first, float last, inout vec2 interval)
{
    if (first < 0.0 && last < 0.0) return false;
    if ((first < 0.0) != (last < 0.0))
    {
        float crossing = first / (first - last);
        if (first < 0.0) interval.x = max(interval.x, crossing);
        else interval.y = min(interval.y, crossing);
    }
    return interval.x <= interval.y;
}
float Visibility(ivec2 receiver, int sampleIndex)
{
    float storedReceiver = ReadDepth(receiver, sampleIndex);
    if (storedReceiver >= 1.0) return 1.0;
    // Form the centred numerator before division. Subtracting one from a
    // rounded UV loses precision near the image centre and at slab tangencies.
    vec2 ndc = ((vec2(receiver) + PixelCentre + SampleOffsets[sampleIndex].xy) * 2.0 - vec2(ExtentBudget.xy)) / vec2(ExtentBudget.xy);
    vec4 world = InverseCamera * vec4(ndc.x, -ndc.y, storedReceiver, 1.0);
    vec3 origin = ProjectiveDivide(world);
    vec3 toward = Emitter.w == 0.0 ? Emitter.xyz : Emitter.xyz - origin;
    // Keep a scaled, non-unit ray vector. One scalar converts world distances;
    // normalizing each component separately can perturb a pixel-corner slope.
    // Scaling also avoids overflow in the squared length of a distant light.
    float scale = max(max(abs(toward.x), abs(toward.y)), abs(toward.z));
    if (scale == 0.0 || isinf(scale) || isnan(scale)) return 1.0;
    vec3 direction = ProjectiveDivide(vec4(toward, scale));
    float rayMetric = length(direction);
    float lightDistance = scale * rayMetric;
    float rayLength = Emitter.w == 0.0 ? Trace.x : min(Trace.x, lightDistance);
    if (rayLength <= Trace.y) return 1.0;
    // Retain the known receiver clip coordinate instead of projecting an
    // already rounded world position back into its own pixel. Project only the
    // ray direction, then apply the bias/length in homogeneous coordinates.
    float receiverW = ProjectiveDivide(vec4(1.0, 1.0, 1.0, world.w)).x;
    vec4 receiverClip = vec4(ndc.x, -ndc.y, storedReceiver, 1.0) * receiverW;
    vec4 projectedDirection = Camera * vec4(direction, 0.0);
    vec2 rayParameters = vec2(Trace.y, rayLength) / rayMetric;
    vec4 first = receiverClip + projectedDirection * rayParameters.x;
    vec4 last = receiverClip + projectedDirection * rayParameters.y;
    vec2 clipping = vec2(0.0, 1.0);
    if (!ClipHalfSpace(first.w - DepthMetric.w, last.w - DepthMetric.w, clipping) ||
        !ClipHalfSpace(first.x + first.w, last.x + last.w, clipping) ||
        !ClipHalfSpace(first.w - first.x, last.w - last.x, clipping) ||
        !ClipHalfSpace(first.y + first.w, last.y + last.w, clipping) ||
        !ClipHalfSpace(first.w - first.y, last.w - last.y, clipping) ||
        !ClipHalfSpace(first.z, last.z, clipping) ||
        !ClipHalfSpace(first.w - first.z, last.w - last.z, clipping)) return 1.0;
    vec4 a = clipping.x == 0.0 ? first : mix(first, last, clipping.x);
    vec4 b = clipping.y == 1.0 ? last : mix(first, last, clipping.y);
    vec2 reciprocalW = 1.0 / vec2(a.w, b.w);
    vec2 pixelScale = vec2(ExtentBudget.xy) * vec2(PixelCentre, -PixelCentre);
    vec2 receiverPixel = vec2(receiver) + PixelCentre + SampleOffsets[sampleIndex].xy;
    vec2 centredPixel = receiverPixel - vec2(ExtentBudget.xy) * PixelCentre;
    // Differentiate the projected ray from its known raster origin. Computing
    // the minor-axis displacement as two rounded endpoints would lose the
    // corner slope. The FMA residual preserves cancellation in this determinant.
    vec2 product = centredPixel * projectedDirection.w;
    vec2 residual = fma(centredPixel, vec2(projectedDirection.w), -product);
    vec2 pixelDirection = fma(projectedDirection.xy, pixelScale, -product) - residual;
    vec2 clippedParameters = mix(vec2(rayParameters.x), vec2(rayParameters.y), clipping);
    vec2 start = fma(pixelDirection, vec2(clippedParameters.x / a.w), receiverPixel);
    vec2 delta = pixelDirection * (receiverW * (clippedParameters.y - clippedParameters.x) / (a.w * b.w));
    float majorDelta = max(abs(delta.x), abs(delta.y));
    if (majorDelta <= DirectionFloor) return 1.0;
    float intervalUncertainty = RoundoffUlps * FloatEpsilon * float(max(ExtentBudget.x, ExtentBudget.y)) / majorDelta;
    // Pixel error maps to a different crossing-time error on each axis. Their
    // sum prevents a steep ray's uncertain corner from becoming a thin blocker.
    float minorDelta = min(abs(delta.x), abs(delta.y));
    float cornerUncertainty = intervalUncertainty + (minorDelta > DirectionFloor ?
        RoundoffUlps * FloatEpsilon * float(max(ExtentBudget.x, ExtentBudget.y)) / minorDelta : 0.0);
    ivec2 cell = ivec2(floor(start)), step = ivec2(1);
    vec2 nextBoundary = vec2(InfiniteCrossing), stride = vec2(InfiniteCrossing);
    for (int axis = 0; axis < 2; ++axis)
    {
        if (delta[axis] < 0.0)
        {
            step[axis] = -1;
            if (start[axis] == floor(start[axis])) --cell[axis];
        }
        cell[axis] = clamp(cell[axis], 0, int(ExtentBudget[axis]) - 1);
        if (abs(delta[axis]) > DirectionFloor)
        {
            float edge = float(cell[axis] + (step[axis] > 0 ? 1 : 0));
            nextBoundary[axis] = max(0.0, (edge - start[axis]) / delta[axis]);
            stride[axis] = 1.0 / abs(delta[axis]);
        }
    }
    // A normalized perspective camera has W equal to axial world distance, so
    // z/W is exactly one. Reconstructing it through device depth adds avoidable
    // cancellation at the conservative slab-boundary uncertainty threshold.
    vec2 axialOverW = DepthMetric.z != 0.0 ? vec2(1.0) :
        vec2(AxialDepth(a.z * reciprocalW.x), AxialDepth(b.z * reciprocalW.y)) * reciprocalW;
    float enter = 0.0;
    for (uint visited = 0; visited < ExtentBudget.z && enter < 1.0; ++visited)
    {
        if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, ivec2(ExtentBudget.xy)))) break;
        float leave = min(min(nextBoundary.x, nextBoundary.y), 1.0);
        float surface = ReadDepth(cell, sampleIndex);
        if (leave > enter + intervalUncertainty && any(notEqual(cell, receiver)) && surface < 1.0)
        {
            float front = AxialDepth(surface), back = front + Trace.z;
            float zEnter = (axialOverW.x + (axialOverW.y - axialOverW.x) * enter) /
                (reciprocalW.x + (reciprocalW.y - reciprocalW.x) * enter);
            float zLeave = (axialOverW.x + (axialOverW.y - axialOverW.x) * leave) /
                (reciprocalW.x + (reciprocalW.y - reciprocalW.x) * leave);
            float uncertainty = RoundoffUlps * FloatEpsilon * max(1.0,
                max(max(abs(front), abs(back)), max(abs(zEnter), abs(zLeave))));
            // A monotone projective depth interval intersects the slab iff its
            // endpoints straddle it. A genuine crossing survives zero thickness.
            // Compare the small overlap directly. Adding its uncertainty to a
            // large axial depth can round the threshold back onto an endpoint.
            if (max(zEnter, zLeave) - front >= uncertainty && back - min(zEnter, zLeave) >= uncertainty) return 0.0;
        }
        if (leave >= 1.0) break;
        enter = max(enter, leave);
        for (int axis = 0; axis < 2; ++axis)
            if (nextBoundary[axis] <= leave + cornerUncertainty)
            { enter = max(enter, nextBoundary[axis]); cell[axis] += step[axis]; nextBoundary[axis] += stride[axis]; }
    }
    return 1.0;
}
void main()
{
    uvec3 index = gl_GlobalInvocationID;
    if (index.x >= ExtentBudget.x || index.y >= ExtentBudget.y || index.z >= ExtentBudget.w) return;
    imageStore(ContactVisibility, ivec3(index), vec4(Visibility(ivec2(index.xy), int(index.z))));
}

#endif // EVANS_CONTACT_DEPTH
