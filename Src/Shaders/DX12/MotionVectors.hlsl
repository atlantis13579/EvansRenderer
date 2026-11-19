// Rigid per-pixel motion vectors from current/previous homogeneous positions.
// Interpolate the previous clip position before dividing by W, replay opaque
// alpha coverage against read-only depth, and output top-left pixel displacement,
// reversed previous hardware depth and validity. This producer applies no shutter blur.
// Rosado, GPU Gems 3, "Motion Blur as a Post-Processing Effect" (2007), sec. 27.4.
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
cbuffer ShutterParameters : register(b0)
{
    row_major float4x4 ObjectToWorld;
    row_major float4x4 CameraTransform;
    row_major float4x4 PreviousObjectToClip;
    row_major float4x4 ClipToWorld;
    float4 OpticalDepthPlane;
    float4 ExtentExposureRadius;
    float4 OpacityCutoff;
    uint4 StageFlags;
};
Texture2D<float4> LocalMaximumOrDiffuse : register(t3);
Texture2D<float4> OpacityTexture : register(t4);
SamplerState MaterialSampler : register(s0);
static const float HomogeneousEpsilon = 1.0e-7f;
static const float HalfFloatMaximum = 65504.0f;
static const float FarDepth = 1.0f;

struct MeshVertex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD0;
};
struct VelocityVertex
{
    float4 RasterPosition : SV_POSITION;
    float4 PreviousPosition : TEXCOORD0;
    float2 UV : TEXCOORD1;
    float Coverage : TEXCOORD2;
};
VelocityVertex VSVelocity(MeshVertex vertex)
{
    VelocityVertex result;
    // The separate transforms match the opaque pass's depth arithmetic.
    result.RasterPosition = mul(CameraTransform, mul(ObjectToWorld, float4(vertex.Position, 1)));
    result.PreviousPosition = mul(PreviousObjectToClip, float4(vertex.Position, 1));
    result.UV = float2(vertex.UV.x, 1 - vertex.UV.y);
    result.Coverage = vertex.Color.a * OpacityCutoff.x;
    return result;
}
float4 PSVelocity(VelocityVertex vertex) : SV_TARGET
{
    float coverage = vertex.Coverage;
    if (StageFlags.y != 0u) coverage *= LocalMaximumOrDiffuse.Sample(MaterialSampler, vertex.UV).a;
    if (StageFlags.z != 0u) coverage *= OpacityTexture.Sample(MaterialSampler, vertex.UV).r;
    if (OpacityCutoff.y > 0) clip(saturate(coverage) - OpacityCutoff.y);
    if (StageFlags.w == 0u || vertex.PreviousPosition.w <= HomogeneousEpsilon) return 0;
    float3 oldNdc = vertex.PreviousPosition.xyz / vertex.PreviousPosition.w;
    if (!all(isfinite(oldNdc)) || oldNdc.z < 0 || oldNdc.z > FarDepth) return 0;
    float2 oldPixel = (oldNdc.xy * float2(0.5f, -0.5f) + 0.5f) * ExtentExposureRadius.xy;
    // Remove camera sampling offsets; the shutter consumes physical motion.
    float2 travel = vertex.RasterPosition.xy - oldPixel - OpacityCutoff.zw;
    if (any(abs(travel) > HalfFloatMaximum)) return 0;
    // Raw displacement stays reusable; MotionBlur owns shutter scaling.
    // Store distance from far depth to retain perspective precision in RGBA16F.
    return float4(travel, 1-oldNdc.z, 1);
}
