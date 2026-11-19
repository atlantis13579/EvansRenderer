#include <metal_stdlib>
using namespace metal;
// Metal 2.1 adds the attribute; guaranteed cross-pass invariance also needs
// the runtime preserveInvariance option (macOS 11 / iOS 14 or newer).
#if __METAL_VERSION__ >= 210
#define EVANS_POSITION [[position, invariant]]
#else
#define EVANS_POSITION [[position]]
#endif
// Independent Metal stages for front depth peeling. Depth-only capture is
// followed by material lighting at exactly that depth, then ordered composition.
// RGB stores premultiplied radiance; accumulated alpha stores transmittance.
// Everitt (2001), Interactive Order-Independent Transparency; Bavoil/Myers (2008),
// section 2.2: https://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf
struct PeelingConstants
{
    float4 World[4], ViewProjection[4];
    float4 Material, Viewport;
    uint4 Options;
};
constant float FarDepth = 1, MaximumHalfRadiance = 65504;
constant float DiagnosticAlpha = 0.75;
constant float3 DiagnosticColor = float3(1,0,1);
float4 TransformRows(constant float4* rows,float4 value)
{ return float4(dot(rows[0],value),dot(rows[1],value),dot(rows[2],value),dot(rows[3],value)); }

#ifdef EVANS_OIT_CAPTURE
struct MeshVertex { packed_float3 Position, Normal; packed_float4 Color; packed_float2 UV; };
struct LayerVertex { float4 Position EVANS_POSITION; float Alpha; float2 UV; };
vertex LayerVertex VSMain(uint index [[vertex_id]],device const MeshVertex* vertices [[buffer(0)]],
    constant PeelingConstants& data [[buffer(1)]])
{
    const MeshVertex input = vertices[index]; LayerVertex output;
    float4 world = TransformRows(data.World,float4(float3(input.Position),1));
    output.Position = TransformRows(data.ViewProjection,world);
    output.Alpha = input.Color.w;
    output.UV = float2(input.UV.x,1-input.UV.y);
    return output;
}
fragment void PSMain(LayerVertex input [[stage_in]],constant PeelingConstants& data [[buffer(0)]],
    depth2d<float,access::read> opaqueDepth [[texture(0)]],depth2d<float,access::read> previousDepth [[texture(1)]],
    texture2d<float> diffuse [[texture(2)]],texture2d<float> opacity [[texture(3)]],sampler materialSampler [[sampler(0)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (input.Position.z >= opaqueDepth.read(pixel)) discard_fragment();
    if (data.Options.x == 0 && input.Position.z <= previousDepth.read(pixel)) discard_fragment();
    float coverage = input.Alpha*data.Material.x;
    if (data.Options.y != 0) coverage *= diffuse.sample(materialSampler,input.UV).a;
    if (data.Options.z != 0) coverage *= opacity.sample(materialSampler,input.UV).r;
    if (!isfinite(coverage)) discard_fragment();
    coverage = saturate(coverage);
    if (coverage <= 0 || coverage < data.Material.y) discard_fragment();
}
#else
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex VSMain(uint vertexId [[vertex_id]])
{
    ScreenVertex output;
    float2 corner = float2((vertexId<<1)&2,vertexId&2);
    output.Position = float4(corner*2-1,0,1); return output;
}
fragment float4 PSMain(ScreenVertex input [[stage_in]],constant PeelingConstants& data [[buffer(0)]],
    texture2d<float,access::read> layerColor [[texture(4)]],texture2d<float,access::read> previousAccumulation [[texture(5)]],
    texture2d<float,access::read> sceneColor [[texture(6)]],depth2d<float,access::read> capturedDepth [[texture(7)]])
{
    uint2 pixel = uint2(input.Position.xy);
    float4 accumulated = previousAccumulation.read(pixel);
#ifdef EVANS_OIT_ACCUMULATE
    if (capturedDepth.read(pixel) >= FarDepth) return accumulated;
    float4 layer = layerColor.read(pixel);
    float remaining = saturate(accumulated.a);
    return float4(accumulated.rgb+remaining*layer.rgb,remaining*(1-saturate(layer.a)));
#else
    float4 background = sceneColor.read(pixel);
    float remaining = saturate(accumulated.a);
    float3 radiance = accumulated.rgb+remaining*background.rgb;
    if (data.Options.w != 0 && capturedDepth.read(pixel) < FarDepth)
        radiance = mix(radiance,DiagnosticColor,DiagnosticAlpha);
    return float4(clamp(radiance,0.0f,MaximumHalfRadiance),1-remaining+remaining*saturate(background.a));
#endif
}
#endif
