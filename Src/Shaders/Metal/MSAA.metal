#include <metal_stdlib>
using namespace metal;
// Native MSL multisample reconstruction. Radiance is averaged in linear space;
// depth, material and motion retain one foreground surface at coverage edges.
// See Metal Shading Language Specification: multisample textures and sample_id.
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
struct SampleResolveVertex { float4 Position [[position]]; };
vertex SampleResolveVertex MSAAResolveVertex(uint vertexIndex [[vertex_id]])
{
    float2 corner(float((vertexIndex << 1) & 2), float(vertexIndex & 2));
    SampleResolveVertex output;
    output.Position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
struct ForegroundSample { uint Index; float Depth; };
ForegroundSample SelectForeground(uint2 pixel, depth2d_ms<float, access::read> depths)
{
    ForegroundSample result {0, depths.read(pixel, 0)};
    for (uint index = 1; index < depths.get_num_samples(); ++index)
    {
        float depth = depths.read(pixel, index);
        if (depth < result.Depth) { result.Index = index; result.Depth = depth; }
    }
    return result;
}
fragment float4 MSAAColorFragment(SampleResolveVertex input [[stage_in]],
    texture2d_ms<float, access::read> radiance [[texture(0)]])
{
    float4 sum = float4(0.0);
    uint count = radiance.get_num_samples();
    for (uint index = 0; index < count; ++index)
        sum += radiance.read(uint2(input.Position.xy), index);
    // sRGB reads decode and the sRGB destination encodes the linear average.
    return sum / float(count);
}
struct DepthResolveResult { float Depth [[depth(any)]]; };
fragment DepthResolveResult MSAADepthFragment(SampleResolveVertex input [[stage_in]],
    depth2d_ms<float, access::read> depths [[texture(1)]])
{
    DepthResolveResult output;
    output.Depth = SelectForeground(uint2(input.Position.xy), depths).Depth;
    return output;
}
struct MaterialResolveResult
{
    float4 Diffuse [[color(0)]];
    float4 Normal [[color(1)]];
    float4 Specular [[color(2)]];
    float4 Ambient [[color(3)]];
    float4 Emission [[color(4)]];
    float Depth [[depth(any)]];
};
fragment MaterialResolveResult MSAAMaterialFragment(SampleResolveVertex input [[stage_in]],
    depth2d_ms<float, access::read> depths [[texture(1)]],
    texture2d_ms<float, access::read> diffuse [[texture(2)]],
    texture2d_ms<float, access::read> normal [[texture(3)]],
    texture2d_ms<float, access::read> specular [[texture(4)]],
    texture2d_ms<float, access::read> ambient [[texture(5)]],
    texture2d_ms<float, access::read> emission [[texture(6)]])
{
    uint2 pixel = uint2(input.Position.xy);
    ForegroundSample foreground = SelectForeground(pixel, depths);
    MaterialResolveResult output;
    output.Depth = foreground.Depth;
    output.Diffuse = diffuse.read(pixel, foreground.Index);
    output.Normal = normal.read(pixel, foreground.Index);
    output.Specular = specular.read(pixel, foreground.Index);
    output.Ambient = ambient.read(pixel, foreground.Index);
    output.Emission = emission.read(pixel, foreground.Index);
    return output;
}
fragment float4 MSAAVelocityFragment(SampleResolveVertex input [[stage_in]],
    depth2d_ms<float, access::read> depths [[texture(1)]],
    texture2d_ms<float, access::read> velocity [[texture(7)]])
{
    uint2 pixel = uint2(input.Position.xy);
    return velocity.read(pixel, SelectForeground(pixel, depths).Index);
}
fragment float4 MSAARestoreFragment(SampleResolveVertex input [[stage_in]],
    texture2d<float, access::read> filteredRadiance [[texture(8)]])
{
    // Broadcast the processed opaque layer to color samples. The original
    // multisample depth is retained separately for later transparent coverage.
    // This approximates filtered backgrounds at mixed-coverage edges.
    return filteredRadiance.read(uint2(input.Position.xy));
}
