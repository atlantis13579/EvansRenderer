#include <metal_stdlib>
using namespace metal;
// Asset-free analytic grading in scene-linear Rec.709: saturation followed by
// contrast around middle gray. Tone mapping and display encoding happen later.
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
    const float3 LuminanceRec709(0.2126, 0.7152, 0.0722);
    const float PhotographicMiddleGray = 0.18;
    const float MaximumHalfFloatRadiance = 65504.0;
    float4 source = scene.sample(LinearClamp, input.UV);
    float luma = dot(source.rgb, LuminanceRec709);
    float3 saturated = mix(float3(luma), source.rgb, uniforms.GradeExposure.y);
    float3 result = (saturated - PhotographicMiddleGray) * uniforms.GradeExposure.x + PhotographicMiddleGray;
    return float4(clamp(result, 0.0, MaximumHalfFloatRadiance), source.a);
}
