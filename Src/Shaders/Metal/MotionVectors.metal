#include <metal_stdlib>
using namespace metal;
// Rigid per-pixel motion vectors from current/previous homogeneous positions.
// Interpolate the previous clip position before dividing by W, replay opaque
// alpha coverage against read-only depth, and output top-left pixel displacement,
// reversed previous hardware depth and validity. This producer applies no shutter blur.
// Rosado, GPU Gems 3, "Motion Blur as a Post-Processing Effect" (2007), sec. 27.4.
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
struct MotionUniforms
{
    float4 World[4];
    float4 ViewProjection[4];
    float4 PreviousWorldViewProjection[4];
    float4 InverseViewProjection[4];
    float4 DepthPlane;
    float4 ViewportExposure; // dimensions, shutter / (2 * interval), maximum radius
    float4 Material; // object/material opacity, alpha cutoff
    uint4 Options; // filter pass, diffuse texture, opacity texture, valid history
};
constant float PixelCenter = 0.5;
constant float MinimumClipW = 1.0e-7;
constant float FarDepth = 1.0;
constant float MaximumHalfFloat = 65504.0;
constant uint VertexFloatCount = 12;
constant uint PositionOffset = 0;
constant uint AlphaOffset = 9;
constant uint TexCoordOffset = 10;

float4 MultiplyRows(constant float4* rows, float4 position)
{
    return float4(dot(rows[0], position), dot(rows[1], position),
        dot(rows[2], position), dot(rows[3], position));
}
struct MotionVertexInput { float Values[VertexFloatCount]; };
struct VelocityVaryings
{
    float4 Position [[position]];
    float4 PreviousClip;
    float2 UV;
    float Alpha;
};
vertex VelocityVaryings MotionVelocityVertex(device const MotionVertexInput* vertices [[buffer(0)]],
    constant MotionUniforms& uniforms [[buffer(1)]], uint vertexId [[vertex_id]])
{
    const device MotionVertexInput& input = vertices[vertexId];
    float4 local(input.Values[PositionOffset], input.Values[PositionOffset + 1], input.Values[PositionOffset + 2], 1.0);
    float4 world = MultiplyRows(uniforms.World, local);
    VelocityVaryings output;
    // Preserve the opaque vertex path's world-then-camera arithmetic. Metal
    // projects Y-up clip coordinates to top-left framebuffer pixels directly.
    output.Position = MultiplyRows(uniforms.ViewProjection, world);
    output.PreviousClip = MultiplyRows(uniforms.PreviousWorldViewProjection, local);
    output.UV = float2(input.Values[TexCoordOffset], 1.0 - input.Values[TexCoordOffset + 1]);
    output.Alpha = input.Values[AlphaOffset] * uniforms.Material.x;
    return output;
}
fragment float4 MotionVelocityFragment(VelocityVaryings input [[stage_in]],
    constant MotionUniforms& uniforms [[buffer(1)]],
    texture2d<float> diffuse [[texture(1)]], texture2d<float> opacity [[texture(3)]], sampler materialSampler [[sampler(0)]])
{
    float alpha = input.Alpha;
    if (uniforms.Options.y != 0) alpha *= diffuse.sample(materialSampler, input.UV).a;
    if (uniforms.Options.z != 0) alpha *= opacity.sample(materialSampler, input.UV).r;
    if (uniforms.Material.y > 0.0 && saturate(alpha) < uniforms.Material.y) discard_fragment();
    if (uniforms.Options.w == 0 || input.PreviousClip.w <= MinimumClipW) return float4(0.0);
    // Perspective interpolation must precede the previous-position divide.
    float3 previousNdc = input.PreviousClip.xyz / input.PreviousClip.w;
    if (!all(isfinite(previousNdc)) || previousNdc.z < 0.0 || previousNdc.z > FarDepth) return float4(0.0);
    float2 previousPixel = (previousNdc.xy * float2(PixelCenter, -PixelCenter) + PixelCenter) * uniforms.ViewportExposure.xy;
    // Camera jitter belongs to sampling, not physical shutter displacement.
    float2 displacement = input.Position.xy - previousPixel - uniforms.Material.zw;
    if (any(abs(displacement) > MaximumHalfFloat)) return float4(0.0);
    // Reversed hardware depth avoids half-float precision loss near the far plane.
    return float4(displacement, 1.0-previousNdc.z, 1.0);
}
