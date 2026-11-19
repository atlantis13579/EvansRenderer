// Rigid per-pixel motion vectors from current/previous homogeneous positions.
// Interpolate the previous clip position before dividing by W, replay opaque
// alpha coverage against read-only depth, and output top-left pixel displacement,
// reversed previous hardware depth and validity. This producer applies no shutter blur.
// Rosado, GPU Gems 3, "Motion Blur as a Post-Processing Effect" (2007), sec. 27.4.
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
cbuffer MotionConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 ViewProjection;
    row_major float4x4 PreviousWorldViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 DepthPlane;
    float4 ViewportExposure;
    float4 Material;
    uint4 Options;
};
Texture2D<float4> Auxiliary : register(t3);
Texture2D<float4> OpacityMap : register(t4);
SamplerState SurfaceSampler : register(s0);
static const float ClipWEpsilon = 1.0e-7f;
static const float MaximumHalfFloat = 65504.0f;
static const float ClearDepth = 1.0f;

struct GeometryInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct GeometryVertex
{
    float4 Position : SV_POSITION;
    float4 PreviousClip : TEXCOORD0;
    float2 TexCoord : TEXCOORD1;
    float Alpha : TEXCOORD2;
};
GeometryVertex VSVelocity(GeometryInput input)
{
    GeometryVertex output;
    // Match opaque geometry's arithmetic order for the read-only depth replay.
    output.Position = mul(ViewProjection, mul(World, float4(input.Position, 1.0f)));
    output.PreviousClip = mul(PreviousWorldViewProjection, float4(input.Position, 1.0f));
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    output.Alpha = input.Color.a * Material.x;
    return output;
}
float4 PSVelocity(GeometryVertex input) : SV_TARGET
{
    float alpha = input.Alpha;
    if (Options.y != 0u) alpha *= Auxiliary.Sample(SurfaceSampler, input.TexCoord).a;
    if (Options.z != 0u) alpha *= OpacityMap.Sample(SurfaceSampler, input.TexCoord).r;
    if (Material.y > 0.0f) clip(saturate(alpha) - Material.y);
    if (Options.w == 0u || input.PreviousClip.w <= ClipWEpsilon) return 0.0f;
    // Divide after perspective interpolation, never interpolate divided velocity.
    float3 previous = input.PreviousClip.xyz / input.PreviousClip.w;
    if (previous.z < 0.0f || previous.z > ClearDepth || !all(isfinite(previous))) return 0.0f;
    float2 previousPixel = (previous.xy * float2(0.5f, -0.5f) + 0.5f) * ViewportExposure.xy;
    // Sampling jitter is not physical motion and must not create shutter blur.
    float2 displacement = input.Position.xy - previousPixel - Material.zw;
    if (any(abs(displacement) > MaximumHalfFloat)) return 0.0f;
    // Half-float motion targets need the small distance from far depth; storing
    // depth near one loses the precision required for optical-depth rejection.
    return float4(displacement, 1.0f-previous.z, 1.0f);
}
