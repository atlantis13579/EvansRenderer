#version 450
// Independent Vulkan thin-lens depth of field. Signed circle-of-confusion
// radii distinguish near and far layers. A bounded integer disk gathers each
// source pixel's footprint; inverse-area weights spread foreground silhouettes.
// GPU Gems, chapter 23, "Depth of Field: A Survey of Techniques" (Demers, 2004):
// https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques
// GPU Gems 3, chapter 28, "Practical Post-Process Depth of Field" (Hammon, 2007):
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
// Full-resolution deterministic gathering favors readable coverage over sparse
// sampling. Nearby far-field color approximates hidden background under a near
// silhouette; a single depth image cannot recover truly occluded geometry.
// Transparent draws follow this opaque-only effect. No history or art is needed.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform DepthOfFieldConstants
{
    vec4 ViewportRadius;
    vec4 CoCScaleBias;
    uvec4 Options;
} Lens;
layout(set = 0, binding = 1) uniform texture2D SourceColor;
layout(set = 0, binding = 2) uniform texture2D SceneDepth;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutColor;

const uint CompositePass = 1u;
const float Pi = 3.141592653589793;
const float ClearDepth = 1.0;
const float PixelFootprintRadius = 0.5;
const float MinimumKernelArea = 1.0;
const float FocusTransitionStart = 0.5;
const float FocusTransitionEnd = 1.5;
const float MinimumWeight = 1.0e-7;
const float MaximumHalfFloatRadiance = 65504.0;

float CircleOfConfusion(ivec2 pixel)
{
    float depth = texelFetch(sampler2D(SceneDepth, PointSampler), pixel, 0).r;
    float radius = depth >= ClearDepth ? Lens.CoCScaleBias.z :
        depth * Lens.CoCScaleBias.x + Lens.CoCScaleBias.y;
    return clamp(radius, -Lens.ViewportRadius.z, Lens.ViewportRadius.z);
}
float DiskWeight(float radius, float distance)
{
    float coverage = clamp(radius + PixelFootprintRadius - distance, 0.0, 1.0);
    return coverage / max(Pi * radius * radius, MinimumKernelArea);
}
float BlurAmount(float radius)
{
    return smoothstep(FocusTransitionStart, FocusTransitionEnd, radius);
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(sampler2D(SourceColor, PointSampler), pixel, 0);
    // The second stage copies RGB into a loaded scene attachment with alpha
    // writes disabled. Orthographic/pinhole cameras and zero radius are identities.
    if (Lens.Options.x == CompositePass || Lens.ViewportRadius.z <= 0.0 ||
        Lens.CoCScaleBias.w <= 0.0 || all(equal(Lens.CoCScaleBias.xy, vec2(0.0))))
    {
        OutColor = center;
        return;
    }

    float centerRadius = CircleOfConfusion(pixel);
    float diskExtent = Lens.ViewportRadius.z + PixelFootprintRadius;
    int kernelExtent = int(ceil(diskExtent));
    ivec2 extent = textureSize(sampler2D(SourceColor, PointSampler), 0);
    vec3 nearRadiance = vec3(0.0);
    vec3 farRadiance = vec3(0.0);
    float nearWeight = 0.0;
    float farWeight = 0.0;
    for (int y = -kernelExtent; y <= kernelExtent; ++y)
    for (int x = -kernelExtent; x <= kernelExtent; ++x)
    {
        float distance = length(vec2(x, y));
        if (distance > diskExtent) continue;
        // Clamp the image address, retaining the original disk distance. This
        // extends edge colors without collapsing the sampling footprint.
        ivec2 neighbor = clamp(pixel + ivec2(x, y), ivec2(0), extent - 1);
        float radius = CircleOfConfusion(neighbor);
        vec3 radiance = texelFetch(sampler2D(SourceColor, PointSampler), neighbor, 0).rgb;
        if (radius < 0.0)
        {
            float weight = DiskWeight(-radius, distance) * BlurAmount(-radius);
            nearRadiance += radiance * weight;
            nearWeight += weight;
        }
        else
        {
            float gatherRadius = centerRadius < 0.0 ? max(radius, -centerRadius) : radius;
            float weight = DiskWeight(gatherRadius, distance);
            farRadiance += radiance * weight;
            farWeight += weight;
        }
    }
    vec3 result = center.rgb;
    if (farWeight > MinimumWeight)
        result = mix(result, farRadiance / farWeight, BlurAmount(abs(centerRadius)));
    if (nearWeight > MinimumWeight)
        result = mix(result, nearRadiance / nearWeight, clamp(nearWeight, 0.0, 1.0));
    // A normalized gather may round just above its brightest half-float input.
    OutColor = vec4(clamp(result, vec3(0.0), vec3(MaximumHalfFloatRadiance)), center.a);
}
#endif
