#include <metal_stdlib>
using namespace metal;
// Native Metal projection of baked ray samples. L2 SH captures the smooth
// clamped-cosine response; directional distance moments support visibility-aware
// interpolation. No recursive irradiance feedback is baked by this producer.
// https://graphics.stanford.edu/papers/envmap/ ; https://jcgt.org/published/0008/02/01/
struct ProbeUniforms
{
    float4 InverseViewProjection[4];
    float4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uint4 Dimensions, Options;
};
constant float Pi = 3.14159265358979323846, CoefficientStorageScale = 0.25;
constant float HalfFloatMaximum = 65504, MinimumWeightSum = 1e-12, PixelCenter = 0.5;
constant uint SampleBorder = 1, ProjectIrradiance = 0;
struct ProbeVertex { float4 Position [[position]]; };
vertex ProbeVertex VSMain(uint vertexId [[vertex_id]])
{
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    return {float4(uv*2-1,0,1)};
}
float SHBasis(uint index, float3 n)
{
    if (index == 0) return sqrt(1.0f/(4*Pi));
    if (index == 1) return n.y*sqrt(3.0f/(4*Pi));
    if (index == 2) return n.z*sqrt(3.0f/(4*Pi));
    if (index == 3) return n.x*sqrt(3.0f/(4*Pi));
    if (index == 4) return n.x*n.y*sqrt(15.0f/(4*Pi));
    if (index == 5) return n.y*n.z*sqrt(15.0f/(4*Pi));
    if (index == 6) return (3*n.z*n.z-1)*sqrt(5.0f/(16*Pi));
    if (index == 7) return n.x*n.z*sqrt(15.0f/(4*Pi));
    return (n.x*n.x-n.y*n.y)*sqrt(15.0f/(16*Pi));
}
float3 OctDirection(uint2 cell, uint size)
{
    float2 square = (float2(cell)+PixelCenter)/float(size)*2-1;
    float3 direction(square,1-abs(square.x)-abs(square.y));
    float fold = saturate(-direction.z);
    direction.x += direction.x >= 0 ? -fold : fold;
    direction.y += direction.y >= 0 ? -fold : fold;
    return normalize(direction);
}
fragment float4 PSMain(ProbeVertex input [[stage_in]], constant ProbeUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> samples [[texture(0)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (uniforms.Options.x == ProjectIrradiance)
    {
        float3 value(0);
        for (uint ray=0; ray<uniforms.Dimensions.w; ++ray)
        {
            float4 direction = samples.read(uint2(ray+SampleBorder,0));
            float3 radiance = samples.read(uint2(ray+SampleBorder,pixel.y+SampleBorder)).rgb;
            value += radiance*(SHBasis(pixel.x,direction.xyz)*direction.w);
        }
        float convolution = pixel.x == 0 ? 1.0f : pixel.x < 4 ? 2.0f/3.0f : 1.0f/4.0f;
        value *= convolution*CoefficientStorageScale;
        return float4(clamp(value,-HalfFloatMaximum,HalfFloatMaximum),samples.read(uint2(0,pixel.y+SampleBorder)).a);
    }
    uint tileSize = uniforms.Options.z, columns = uint(uniforms.Visibility.w);
    uint2 tile = pixel/tileSize;
    uint probe = tile.x+tile.y*columns;
    if (probe >= uniforms.Options.w) return float4(1,1,0,0);
    float3 axis = OctDirection(pixel-tile*tileSize,tileSize);
    float2 moments(0); float weightSum=0;
    for (uint ray=0; ray<uniforms.Dimensions.w; ++ray)
    {
        float3 direction = samples.read(uint2(ray+SampleBorder,0)).xyz;
        float d = saturate(samples.read(uint2(ray+SampleBorder,probe+SampleBorder)).a/uniforms.Visibility.z);
        float weight = pow(saturate(dot(direction,axis)),uniforms.Visibility.y);
        moments += float2(d,d*d)*weight; weightSum += weight;
    }
    return float4(weightSum > MinimumWeightSum ? moments/weightSum : float2(1),0,1);
}
