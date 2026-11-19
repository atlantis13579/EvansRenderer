#version 310 es
precision highp float;
precision highp int;
// GLES 3.1 producer, ordinary 2D-texture consumer. The bottom atlas row is
// logical Y=0 on this API, so receiver sampling needs no framebuffer Y flip.
// Wright et al., "Lumen", SIGGRAPH 2022, low-frequency translucency lighting.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 vertex=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(vertex*2.0-1.0,0.0,1.0);
}
#else
layout(std140,binding=0,row_major) uniform Volume
{
    mat4 Camera,Inverse;
    uvec4 Grid;
    vec4 Distribution,Atlas;
};
const int SHCount=9;
const float HalfLimit=65504.0;
struct Irradiance { vec4 SH[SHCount]; vec4 Coverage; uvec4 Trace,Cache,Metadata; };
layout(std430,binding=0) readonly buffer Source { Irradiance Probes[]; };
layout(std430,binding=1) readonly buffer Bounds { float Radii[]; };
const uint VisibilitySupportBit=2u,HalfFractionBits=10u,HalfExponentBias=15u;
const float SupportPackingMargin=2.0/float(1u<<HalfFractionBits);
const float HalfSubnormalQuantum=1.0/float(1u<<(HalfFractionBits+HalfExponentBias-1u));
layout(location=0) out vec4 Color;
void main()
{
    ivec2 texel=ivec2(gl_FragCoord.xy);
    int columns=int(Grid.x),band=texel.x/columns;
    int index=texel.y*columns+texel.x%columns;
    vec3 packed=Probes[index].SH[band].rgb*Distribution.w;
    // Leave more than a half ULP below the true support before half packing.
    float support=(Grid.w&VisibilitySupportBit)!=0u?min(Radii[index],HalfLimit)*(1.0-SupportPackingMargin)-HalfSubnormalQuantum:1.0;
    Color=vec4(clamp(packed,-HalfLimit,HalfLimit),max(0.0,support));
}
#endif
