#version 300 es
// Thin-lens signed circles of confusion, followed by source-footprint gathering.
// GPU Gems, chapter 23, "Depth of Field: A Survey of Techniques", and GPU Gems 3,
// chapter 28, "Practical Post-Process Depth of Field":
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
// The bounded integer disk favors a readable reference over sparse-kernel speed.
// Opaque radiance is processed before transparency; no temporal history is used.
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
uniform vec4 ViewportRadius; // Inverse dimensions, maximum circle radius in pixels.
uniform vec4 CoCScaleBias; // Hardware-depth slope, intercept, infinity radius, perspective flag.
uniform uvec4 Options;
uniform sampler2D SourceColor;
uniform sampler2D OpaqueDepth;
out vec4 OutColor;
const uint CompositePass = 1u;
const float Pi = 3.141592653589793;
const float ClearDepth = 1.0;
const float PixelFootprintRadius = 0.5;
const float BlurTransitionStart = 0.5;
const float BlurTransitionEnd = 1.5;
const float MinimumDiskArea = 1.0;
const float MinimumGatherWeight = 1.0e-7;
const float MaximumHalfFloatRadiance = 65504.0;

ivec2 NativePixel(ivec2 pixel)
{
    return ivec2(pixel.x, textureSize(SourceColor, 0).y - 1 - pixel.y);
}
float SignedCircle(ivec2 pixel)
{
    float depth = texelFetch(OpaqueDepth, NativePixel(pixel), 0).r;
    // Hardware depth is affine in inverse optical distance. The CPU derives
    // these lens coefficients; clear depth uses the infinite-distance limit.
    float radius = depth >= ClearDepth ? CoCScaleBias.z : depth * CoCScaleBias.x + CoCScaleBias.y;
    return clamp(radius, -ViewportRadius.z, ViewportRadius.z);
}
float DiskWeight(float radius, float distance)
{
    float coverage = clamp(radius + PixelFootprintRadius - distance, 0.0, 1.0);
    return coverage / max(Pi * radius * radius, MinimumDiskArea);
}
float BlurAmount(float radius)
{
    return smoothstep(BlurTransitionStart, BlurTransitionEnd, radius);
}
void main()
{
    ivec2 extent = textureSize(SourceColor, 0);
    ivec2 pixel = ivec2(int(gl_FragCoord.x), extent.y - 1 - int(gl_FragCoord.y));
    vec4 center = texelFetch(SourceColor, NativePixel(pixel), 0);
    if (Options.x == CompositePass || ViewportRadius.z <= 0.0 || CoCScaleBias.w <= 0.0 ||
        (CoCScaleBias.x == 0.0 && CoCScaleBias.y == 0.0))
    {
        OutColor = center;
        return;
    }
    float centerCircle = SignedCircle(pixel);
    float maximumFootprint = ViewportRadius.z + PixelFootprintRadius;
    int searchExtent = int(ceil(maximumFootprint));
    vec3 nearRadiance = vec3(0.0);
    vec3 farRadiance = vec3(0.0);
    float nearWeight = 0.0;
    float farWeight = 0.0;
    for (int y = -searchExtent; y <= searchExtent; ++y)
        for (int x = -searchExtent; x <= searchExtent; ++x)
        {
            ivec2 offset = ivec2(x, y);
            float distance = length(vec2(offset));
            if (distance > maximumFootprint) continue;
            // Clamp the fetched texel, not the disk distance: extend edge color.
            ivec2 neighbor = clamp(pixel + offset, ivec2(0), extent - 1);
            float circle = SignedCircle(neighbor);
            vec3 radiance = texelFetch(SourceColor, NativePixel(neighbor), 0).rgb;
            if (circle < 0.0)
            {
                // Foreground coverage follows each source circle across silhouettes.
                float weight = DiskWeight(-circle, distance) * BlurAmount(-circle);
                nearRadiance += radiance * weight;
                nearWeight += weight;
            }
            else
            {
                // A single visible depth layer cannot recover hidden background.
                // Nearby far samples approximate the fill beneath blurred foreground.
                float radius = centerCircle < 0.0 ? max(circle, -centerCircle) : circle;
                float weight = DiskWeight(radius, distance);
                farRadiance += radiance * weight;
                farWeight += weight;
            }
        }
    vec3 result = center.rgb;
    if (farWeight > MinimumGatherWeight)
        result = mix(result, farRadiance / farWeight, BlurAmount(abs(centerCircle)));
    if (nearWeight > MinimumGatherWeight)
        result = mix(result, nearRadiance / nearWeight, clamp(nearWeight, 0.0, 1.0));
    OutColor = vec4(min(result, vec3(MaximumHalfFloatRadiance)), center.a);
}
#endif
