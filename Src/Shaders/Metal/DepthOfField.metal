#include <metal_stdlib>
using namespace metal;
// Signed thin-lens CoC separates foreground coverage from background blur.
// See GPU Gems chapter 23, "Depth of Field: A Survey of Techniques", and
// GPU Gems 3 chapter 28, "Practical Post-Process Depth of Field":
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
// This deterministic integer-disk reference processes opaque HDR before transparent
// draws. Its single depth layer approximates hidden background using nearby pixels.
struct DepthOfFieldUniforms
{
    float4 ViewportRadius; // Inverse dimensions and maximum pixel radius.
    float4 CoCScaleBias; // Depth slope, intercept, infinite-depth radius, perspective flag.
    uint4 Options;
};
struct DepthOfFieldVertex { float4 Position [[position]]; };
constant uint CompositePass = 1;
constant float Pi = 3.141592653589793;
constant float ClearDepth = 1.0;
constant float PixelFootprintRadius = 0.5;
constant float MinimumBlurRadius = 0.5;
constant float FullBlurRadius = 1.5;
constant float MinimumDiskArea = 1.0;
constant float MinimumGatherWeight = 1.0e-7;
constant float MaximumHalfFloatRadiance = 65504.0;

vertex DepthOfFieldVertex DepthOfFieldVertexMain(uint index [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 point = index == 0 ? float2(-1.0) :
        index == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(point, 0.0, 1.0)};
}
float CircleRadius(uint2 pixel, constant DepthOfFieldUniforms& uniforms, depth2d<float, access::read> depth)
{
    float hardwareDepth = depth.read(pixel);
    // Inverse optical distance is affine in hardware depth. Coefficients come
    // from the physical lens; background depth has a finite infinite-distance limit.
    float signedRadius = hardwareDepth >= ClearDepth ? uniforms.CoCScaleBias.z :
        hardwareDepth * uniforms.CoCScaleBias.x + uniforms.CoCScaleBias.y;
    return clamp(signedRadius, -uniforms.ViewportRadius.z, uniforms.ViewportRadius.z);
}
float FootprintWeight(float radius, float separation)
{
    float footprint = saturate(radius + PixelFootprintRadius - separation);
    return footprint / max(Pi * radius * radius, MinimumDiskArea);
}
float DefocusBlend(float radius)
{
    return smoothstep(MinimumBlurRadius, FullBlurRadius, radius);
}
fragment float4 DepthOfFieldFragmentMain(DepthOfFieldVertex input [[stage_in]],
    constant DepthOfFieldUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> source [[texture(0)]], depth2d<float, access::read> depth [[texture(1)]])
{
    uint2 pixel = uint2(input.Position.xy);
    float4 center = source.read(pixel);
    if (uniforms.Options.x == CompositePass || uniforms.ViewportRadius.z <= 0.0 || uniforms.CoCScaleBias.w <= 0.0 ||
        (uniforms.CoCScaleBias.x == 0.0 && uniforms.CoCScaleBias.y == 0.0)) return center;
    int2 dimensions(source.get_width(), source.get_height());
    float centerRadius = CircleRadius(pixel, uniforms, depth);
    float footprintLimit = uniforms.ViewportRadius.z + PixelFootprintRadius;
    int bound = int(ceil(footprintLimit));
    float3 foregroundSum = float3(0.0);
    float3 backgroundSum = float3(0.0);
    float foregroundWeight = 0.0;
    float backgroundWeight = 0.0;
    for (int row = -bound; row <= bound; ++row)
        for (int column = -bound; column <= bound; ++column)
        {
            int2 offset(column, row);
            float separation = length(float2(offset));
            if (separation > footprintLimit) continue;
            // Edge extension repeats source texels while retaining the disk offset.
            uint2 neighbor = uint2(clamp(int2(pixel) + offset, int2(0), dimensions - 1));
            float radius = CircleRadius(neighbor, uniforms, depth);
            float3 radiance = source.read(neighbor).rgb;
            if (radius < 0.0)
            {
                // Source foreground footprints spread coverage beyond their silhouettes.
                float weight = FootprintWeight(-radius, separation) * DefocusBlend(-radius);
                foregroundSum += radiance * weight;
                foregroundWeight += weight;
            }
            else
            {
                float gatherRadius = centerRadius < 0.0 ? max(radius, -centerRadius) : radius;
                float weight = FootprintWeight(gatherRadius, separation);
                backgroundSum += radiance * weight;
                backgroundWeight += weight;
            }
        }
    float3 radiance = center.rgb;
    if (backgroundWeight > MinimumGatherWeight)
        radiance = mix(radiance, backgroundSum / backgroundWeight, DefocusBlend(abs(centerRadius)));
    if (foregroundWeight > MinimumGatherWeight)
        radiance = mix(radiance, foregroundSum / foregroundWeight, saturate(foregroundWeight));
    return float4(min(radiance, MaximumHalfFloatRadiance), center.a);
}
