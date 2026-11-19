#include <metal_stdlib>
using namespace metal;

// Metal implementation of the low-frequency Lumen translucency volume.
// Reference: Wright et al., "Lumen", SIGGRAPH 2022. Signed SH coefficients are
// packed into horizontal bands, with depth slices stacked vertically.
constant uint SHCoefficientCount=9;
constant float FiniteHalfMaximum=65504;
constant uint VisibilitySupportBit=2,HalfFractionBits=10,HalfExponentBias=15;
constant float SupportPackingMargin=2.0f/float(1u<<HalfFractionBits);
constant float HalfSubnormalQuantum=1.0f/float(1u<<(HalfFractionBits+HalfExponentBias-1));
struct VolumeFrame { float4 Camera[4],Inverse[4]; uint4 Grid; float4 Distribution,Atlas; };
struct IrradianceProbe { float4 SH[SHCoefficientCount],Coverage; uint4 Trace,Cache,Metadata; };
struct PackVertex { float4 Position [[position]]; };
vertex PackVertex lumenVolumeVertex(uint id [[vertex_id]])
{
    float2 corner=float2((id<<1)&2,id&2);
    return {float4(corner.x*2-1,1-corner.y*2,0,1)};
}
fragment float4 lumenVolumeFragment(PackVertex input [[stage_in]],
    constant VolumeFrame& frame [[buffer(0)]],device const IrradianceProbe* probes [[buffer(1)]],device const float* radii [[buffer(2)]])
{
    uint2 pixel=uint2(input.Position.xy);
    uint coefficient=pixel.x/frame.Grid.x;
    uint source=pixel.y*frame.Grid.x+pixel.x%frame.Grid.x;
    float3 value=probes[source].SH[coefficient].rgb*frame.Distribution.w;
    // Shrink before half conversion; rounding must not grow the support sphere.
    float support=(frame.Grid.w&VisibilitySupportBit)!=0?min(radii[source],FiniteHalfMaximum)*(1-SupportPackingMargin)-HalfSubnormalQuantum:1;
    return float4(clamp(value,-FiniteHalfMaximum,FiniteHalfMaximum),max(0.0f,support));
}
