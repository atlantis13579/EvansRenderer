#include <metal_stdlib>
using namespace metal;

// Independent Metal screen-space light shafts. A depth-tested analytic emitter
// feeds normalized radial quadrature; this is an image effect, not medium transport.
// Kenny Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems 3.
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process
struct ShaftParameters
{
    float4 EmitterUVDepthRadius;
    float4 EmitterRadianceStrength;
    float4 DecayDepthBiasImageSize;
    uint4 SampleCountPhaseVisible;
};
struct ShaftRasterVertex { float4 Position [[position]]; };
constant uint ShaftMaskPhase = 0;
constant float ShaftPixelCenter = 0.5;
constant float ShaftHalfFloatLimit = 65504.0;

vertex ShaftRasterVertex ShaftVertexMain(uint vertexIndex [[vertex_id]])
{
    const float OverscanCorner = 3.0;
    float2 corner = vertexIndex == 0 ? float2(-1.0, -1.0) :
        vertexIndex == 1 ? float2(OverscanCorner, -1.0) : float2(-1.0, OverscanCorner);
    return {float4(corner, 0.0, 1.0)};
}

float ShaftReadMask(texture2d<float, access::read> mask, int2 pixel)
{
    int2 extent = int2(mask.get_width(), mask.get_height());
    if (any(pixel < int2(0)) || any(pixel >= extent)) return 0.0;
    return mask.read(uint2(pixel)).r;
}

float ShaftBilinearMask(texture2d<float, access::read> mask, float2 coordinate)
{
    // Read access needs no sampler. A zero border clips the actual source disk
    // instead of replicating its edge into a bright streak beyond the viewport.
    float2 pixel = coordinate * float2(mask.get_width(), mask.get_height()) - ShaftPixelCenter;
    int2 base = int2(floor(pixel));
    float2 weight = fract(pixel);
    float upper = mix(ShaftReadMask(mask, base), ShaftReadMask(mask, base + int2(1, 0)), weight.x);
    float lower = mix(ShaftReadMask(mask, base + int2(0, 1)), ShaftReadMask(mask, base + int2(1, 1)), weight.x);
    return mix(upper, lower, weight.y);
}

fragment float4 ShaftFragmentMain(ShaftRasterVertex raster [[stage_in]],
    constant ShaftParameters& parameters [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> opaqueDepth [[texture(1)]],
    texture2d<float, access::read> mask [[texture(2)]])
{
    // Metal raster positions, texture rows and the host emitter projection all
    // start at the top-left; canonical depth already occupies [0,1].
    uint2 pixel = uint2(raster.Position.xy);
    float2 uv = raster.Position.xy / parameters.DecayDepthBiasImageSize.zw;
    bool sourceVisible = parameters.SampleCountPhaseVisible.z != 0;
    if (parameters.SampleCountPhaseVisible.y == ShaftMaskPhase)
    {
        float coverage = 0.0;
        if (sourceVisible && opaqueDepth.read(pixel) + parameters.DecayDepthBiasImageSize.y >= parameters.EmitterUVDepthRadius.z)
        {
            float2 separation = uv - parameters.EmitterUVDepthRadius.xy;
            separation.x *= parameters.DecayDepthBiasImageSize.z / parameters.DecayDepthBiasImageSize.w;
            float disk = saturate(1.0 - length(separation) / parameters.EmitterUVDepthRadius.w);
            coverage = disk * disk;
        }
        return float4(coverage, 0.0, 0.0, 1.0);
    }

    float4 background = scene.read(pixel);
    if (!sourceVisible) return background;
    uint sampleCount = parameters.SampleCountPhaseVisible.x;
    float integral = 0.0;
    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        float parameter = (float(sampleIndex) + ShaftPixelCenter) / float(sampleCount);
        float2 sampleUV = mix(uv, parameters.EmitterUVDepthRadius.xy, parameter);
        integral += ShaftBilinearMask(mask, sampleUV) * exp(-parameters.DecayDepthBiasImageSize.x * parameter);
    }
    // Midpoint integration spans [0,1], so quality changes retain exposure.
    float strength = integral * parameters.EmitterRadianceStrength.a / float(sampleCount);
    float3 radiance = background.rgb + parameters.EmitterRadianceStrength.rgb * strength;
    return float4(min(radiance, float3(ShaftHalfFloatLimit)), background.a);
}
