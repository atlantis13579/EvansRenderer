#include <metal_stdlib>
using namespace metal;
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// Reinhard et al. (2002), Photographic Tone Reproduction; Narkowicz (2016),
// ACES Filmic Tone Mapping Curve. ACES here denotes that rational curve fit,
// not an ACES reference rendering transform. Hardware performs sRGB encoding.
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
    constant HdrUniforms& uniforms [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler LinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    const uint AcesCurve = 2;
    const float NumeratorQuadratic = 2.51;
    const float NumeratorLinear = 0.03;
    const float DenominatorQuadratic = 2.43;
    const float DenominatorLinear = 0.59;
    const float DenominatorConstant = 0.14;
    float4 source = scene.sample(LinearClamp, input.UV);
    float3 radiance = max(source.rgb, 0.0) * uniforms.GradeExposure.z;
    float3 mapped = radiance / (1.0 + radiance);
    if (uniforms.PassToneMapping.y == AcesCurve)
        mapped = radiance * (NumeratorQuadratic * radiance + NumeratorLinear) /
            (radiance * (DenominatorQuadratic * radiance + DenominatorLinear) + DenominatorConstant);
    return float4(saturate(mapped), source.a);
}
