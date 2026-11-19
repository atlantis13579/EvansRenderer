// Lumen translucent lighting volume, Wright et al., SIGGRAPH 2022.
// Store signed cosine-convolved SH in nine horizontal bands. Scaling by the
// sphere measure keeps ordinary HDR radiance representable in a half texture.
cbuffer VolumeFrame : register(b0)
{
    row_major float4x4 ViewProjection,InverseCamera;
    uint4 Grid;
    float4 Distribution,Atlas;
};
static const uint CoefficientCount=9;
static const float MaximumHalf=65504;
struct IrradianceProbe
{
    float4 Coefficients[CoefficientCount],Coverage;
    uint4 TraceCounts,CacheCounts,Metadata;
};
StructuredBuffer<IrradianceProbe> Probes : register(t0);
StructuredBuffer<float> Support : register(t1);
static const uint VisibilitySupportBit=2,HalfFractionBits=10,HalfExponentBias=15;
static const float SupportPackingMargin=2.0/float(1u<<HalfFractionBits);
static const float HalfSubnormalQuantum=1.0/float(1u<<(HalfFractionBits+HalfExponentBias-1));
float4 VSVolume(uint vertex:SV_VertexID):SV_POSITION
{
    float2 corner=float2((vertex<<1)&2,vertex&2);
    return float4(corner*2-1,0,1);
}
float4 PSVolume(float4 position:SV_POSITION):SV_TARGET
{
    uint2 pixel=uint2(position.xy);
    uint coefficient=pixel.x/Grid.x;
    uint probe=pixel.x%Grid.x+pixel.y*Grid.x;
    float3 packed=Probes[probe].Coefficients[coefficient].rgb*Distribution.w;
    // Bias downward by more than a half ULP before the RGBA16F conversion;
    // rounding must not enlarge a certified surface-free support sphere.
    float support=(Grid.w&VisibilitySupportBit)!=0?min(Support[probe],MaximumHalf)*(1-SupportPackingMargin)-HalfSubnormalQuantum:1;
    return float4(clamp(packed,-MaximumHalf,MaximumHalf),max(0,support));
}
