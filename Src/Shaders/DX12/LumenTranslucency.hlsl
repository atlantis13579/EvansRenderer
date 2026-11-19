// Independently authored D3D12 packing stage for the translucent Lumen volume.
// GPU integrated SH remains signed; each Z slice occupies Grid.y atlas rows.
// Reference: Wright et al., "Lumen", SIGGRAPH 2022, translucency volume lighting.
cbuffer FroxelParameters : register(b0)
{
    row_major float4x4 Camera,Inverse;
    uint4 Cells;
    float4 DepthAndScale,ImageExtent;
};
static const uint SHBandsSquared=9;
static const float HalfFiniteLimit=65504;
struct ProbeSH
{
    float4 SH[SHBandsSquared];
    float4 Coverage;
    uint4 Trace,Cache,Identity;
};
StructuredBuffer<ProbeSH> Irradiance : register(t0);
StructuredBuffer<float> VisibilityRadius : register(t1);
static const uint VisibilitySupportBit=2,HalfFractionBits=10,HalfExponentBias=15;
static const float SupportPackingMargin=2.0/float(1u<<HalfFractionBits);
static const float HalfSubnormalQuantum=1.0/float(1u<<(HalfFractionBits+HalfExponentBias-1));
float4 VSPack(uint id:SV_VertexID):SV_POSITION
{
    float2 uv=float2((id<<1)&2,id&2);
    return float4(uv.x*2-1,1-uv.y*2,0,1);
}
float4 PSPack(float4 fragment:SV_POSITION):SV_TARGET
{
    uint x=uint(fragment.x),row=uint(fragment.y);
    uint band=x/Cells.x,source=row*Cells.x+x%Cells.x;
    float3 coefficient=Irradiance[source].SH[band].xyz*DepthAndScale.w;
    // The half conversion may round upward; shrink the bound first so its
    // encoded radius cannot cross the conservative SDF support boundary.
    float support=(Cells.w&VisibilitySupportBit)!=0?min(VisibilityRadius[source],HalfFiniteLimit)*(1-SupportPackingMargin)-HalfSubnormalQuantum:1;
    return float4(clamp(coefficient,-HalfFiniteLimit,HalfFiniteLimit),max(0,support));
}
