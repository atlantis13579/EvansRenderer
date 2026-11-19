#version 450
// Native GLSL Lumen translucency packing. Nine coefficient bands share one
// filtered image; no cross-API shader translation or embedded SPIR-V is used.
// Wright et al., "Lumen", SIGGRAPH 2022: froxel lighting for translucent surfaces.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=0,std140,row_major) uniform Volume
{
    mat4 Camera,Inverse;
    uvec4 Grid;
    vec4 Distribution,Atlas;
};
const uint Coefficients=9u;
const float HalfMaximum=65504.0;
struct Probe { vec4 SH[Coefficients]; vec4 Coverage; uvec4 Trace,Cache,Metadata; };
layout(set=0,binding=1,std430) readonly buffer Integrated { Probe Values[]; };
layout(set=0,binding=2,std430) readonly buffer VisibilityBounds { float Radii[]; };
const uint VisibilitySupportBit=2u,HalfFractionBits=10u,HalfExponentBias=15u;
const float SupportPackingMargin=2.0/float(1u<<HalfFractionBits);
const float HalfSubnormalQuantum=1.0/float(1u<<(HalfFractionBits+HalfExponentBias-1u));
layout(location=0) out vec4 Packed;
void main()
{
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint coefficient=pixel.x/Grid.x;
    uint index=pixel.y*Grid.x+pixel.x%Grid.x;
    // Conservative half packing keeps the quantized sphere inside its SDF bound.
    float support=(Grid.w&VisibilitySupportBit)!=0u?min(Radii[index],HalfMaximum)*(1.0-SupportPackingMargin)-HalfSubnormalQuantum:1.0;
    Packed=vec4(clamp(Values[index].SH[coefficient].xyz*Distribution.w,-HalfMaximum,HalfMaximum),max(0.0,support));
}
#endif
