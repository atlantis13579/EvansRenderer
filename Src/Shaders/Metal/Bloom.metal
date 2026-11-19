#include <metal_stdlib>
using namespace metal;
// Thresholded scene radiance is reduced to half resolution, convolved with a
// separable Gaussian (sigma two, radius four), and added before display mapping.
struct HdrUniforms { float4 InverseSourceExtentBloom; float4 GradeExposure; uint4 PassToneMapping; };
struct HdrVertex { float4 Position [[position]]; float2 UV; };
vertex HdrVertex HdrVertexMain(uint vertexId [[vertex_id]])
{
    const float TriangleExtent = 3.0;
    float2 point = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(TriangleExtent, -1.0) : float2(-1.0, TriangleExtent);
    return {float4(point, 0.0, 1.0), point * float2(0.5, -0.5) + 0.5};
}
fragment float4 HdrFragmentMain(HdrVertex input [[stage_in]],
    constant HdrUniforms& uniforms [[buffer(0)]], texture2d<float> scene [[texture(0)]],
    texture2d<float> bloom [[texture(1)]])
{
    constexpr sampler LinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    const uint Extract = 0;
    const float MaximumHalfFloatRadiance = 65504.0;
    const uint BlurX = 1;
    const uint BlurY = 2;
    const int GaussianRadius = 4;
    const float Kernel[5] = {0.20416369, 0.18017382, 0.12383154, 0.06628225, 0.02763055};
    float4 source = scene.sample(LinearClamp, input.UV);
    if (uniforms.PassToneMapping.x == Extract)
    {
        float3 bright = 0.0;
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                bright += max(scene.sample(LinearClamp,
                    input.UV + (float2(x, y) - 0.5) * uniforms.InverseSourceExtentBloom.xy).rgb -
                    uniforms.InverseSourceExtentBloom.z, 0.0);
        const float BoxNormalization = 0.25;
        return float4(min(bright * BoxNormalization, MaximumHalfFloatRadiance), 1.0);
    }
    if (uniforms.PassToneMapping.x == BlurX || uniforms.PassToneMapping.x == BlurY)
    {
        float2 stepUV = uniforms.PassToneMapping.x == BlurX ?
            float2(uniforms.InverseSourceExtentBloom.x, 0.0) : float2(0.0, uniforms.InverseSourceExtentBloom.y);
        float3 filtered = source.rgb * Kernel[0];
        for (int offset = 1; offset <= GaussianRadius; ++offset)
            filtered += (scene.sample(LinearClamp, input.UV + float(offset) * stepUV).rgb +
                scene.sample(LinearClamp, input.UV - float(offset) * stepUV).rgb) * Kernel[offset];
        return float4(min(filtered, MaximumHalfFloatRadiance), 1.0);
    }
    float3 composed = source.rgb + uniforms.InverseSourceExtentBloom.w * bloom.sample(LinearClamp, input.UV).rgb;
    return float4(min(composed, MaximumHalfFloatRadiance), source.a);
}
