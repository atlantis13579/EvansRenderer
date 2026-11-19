#version 310 es
// ES 3.1 compute: immutable source data, exact-position displacement directions,
// and CSR face gathering. Between the two dispatches issue a shader-storage
// barrier, and include the vertex-attribute barrier before raster consumption.
precision highp float;
precision highp int;
#define DISPLACEMENT_GROUP_WIDTH 64
layout(local_size_x=DISPLACEMENT_GROUP_WIDTH) in;
layout(std140,binding=0) uniform DisplacementParameters { vec4 ObjectBounds,HeightShape; uvec4 Geometry; } Parameters;
layout(std430,binding=1) readonly buffer BaseMesh { uint Data[]; } Mesh;
layout(std430,binding=2) readonly buffer SurfaceDirections { vec4 Data[]; } Direction;
layout(std430,binding=3) readonly buffer MeshIndices { uint Data[]; } Index;
layout(std430,binding=4) readonly buffer NeighborRanges { uvec4 Data[]; } Range;
layout(std430,binding=5) readonly buffer NeighborFaces { uint Data[]; } Neighbor;
layout(std430,binding=6) buffer DisplacedMesh { uint Data[]; } OutputMesh;
const uint PackedWords=12u,NormalComponent=3u,CoordinatesXY=1u,CoordinatesXZ=2u;
const float Pi=3.14159265358979323846,Midpoint=0.5,FourierWeight=0.25;
vec3 BasePosition(uint index)
{ uint address=index*PackedWords; return uintBitsToFloat(uvec3(Mesh.Data[address],Mesh.Data[address+1u],Mesh.Data[address+2u])); }
vec3 DisplacedPosition(uint index)
{ uint address=index*PackedWords; return uintBitsToFloat(uvec3(OutputMesh.Data[address],OutputMesh.Data[address+1u],OutputMesh.Data[address+2u])); }
vec3 NormalizeOr(vec3 value,vec3 fallback)
{ float largest=max(abs(value.x),max(abs(value.y),abs(value.z))); return largest>0.0?normalize(value/largest):fallback; }
void WriteVector(uint address,vec3 value)
{ uvec3 words=floatBitsToUint(value); OutputMesh.Data[address]=words.x; OutputMesh.Data[address+1u]=words.y; OutputMesh.Data[address+2u]=words.z; }
float ScalarHeight(vec3 point)
{
    vec3 relative=(point-Parameters.ObjectBounds.xyz)*Parameters.ObjectBounds.w;
    vec2 uv=Parameters.Geometry.z==CoordinatesXY?relative.xy:Parameters.Geometry.z==CoordinatesXZ?relative.xz:relative.yz;
    uv*=Parameters.HeightShape.y;
    float phasePlus=2.0*Pi*fract(uv.x+uv.y),phaseMinus=2.0*Pi*fract(uv.x-uv.y);
    return Midpoint+FourierWeight*(cos(phasePlus)+cos(phaseMinus));
}
void main()
{
    uint index=gl_GlobalInvocationID.x; if (index>=Parameters.Geometry.x) return;
#if EVANS_DISPLACEMENT_PASS == 0
    for (uint component=0u;component<PackedWords;++component) OutputMesh.Data[index*PackedWords+component]=Mesh.Data[index*PackedWords+component];
    if (Parameters.HeightShape.x==0.0) return;
    vec3 base=BasePosition(index);
    WriteVector(index*PackedWords,base+Direction.Data[index].xyz*(Parameters.HeightShape.x*(ScalarHeight(base)-Parameters.HeightShape.z)));
#else
    if (Parameters.HeightShape.x==0.0) return;
    uint address=index*PackedWords+NormalComponent;
    vec3 original=uintBitsToFloat(uvec3(Mesh.Data[address],Mesh.Data[address+1u],Mesh.Data[address+2u]));
    vec3 reference=NormalizeOr(original,vec3(0.0)),total=vec3(0.0); uvec2 neighborhood=Range.Data[index].xy;
    for (uint entry=0u;entry<neighborhood.y;++entry)
    {
        uint face=Neighbor.Data[neighborhood.x+entry];
        uint a=Index.Data[face*3u],b=Index.Data[face*3u+1u],c=Index.Data[face*3u+2u];
        vec3 oldArea=cross((BasePosition(b)-BasePosition(a))*Parameters.ObjectBounds.w,(BasePosition(c)-BasePosition(a))*Parameters.ObjectBounds.w);
        vec3 newArea=cross((DisplacedPosition(b)-DisplacedPosition(a))*Parameters.ObjectBounds.w,(DisplacedPosition(c)-DisplacedPosition(a))*Parameters.ObjectBounds.w);
        total+=dot(oldArea,reference)<0.0?-newArea:newArea;
    }
    WriteVector(address,NormalizeOr(total,original));
#endif
}
