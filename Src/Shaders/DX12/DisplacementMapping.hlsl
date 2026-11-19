// D3D12 owns these two compute programs. A UAV barrier separates deformation
// from normal gathering; transition the result to vertex input only afterwards.
// Shared geometric directions avoid UV/hard-normal position cracks. Authored
// normal groups still choose distinct CSR neighborhoods for shading creases.
static const uint GroupWidth=64,PackedVertexWords=12,FloatBytes=4,NormalOffset=3;
static const uint CoordinatesXY=1,CoordinatesXZ=2;
static const float Pi=3.14159265358979323846,MeanHeight=0.5,WaveAmplitude=0.25;
cbuffer Parameters:register(b0) { float4 Bounds,HeightShape; uint4 GeometryCount; };
ByteAddressBuffer Mesh:register(t0);
StructuredBuffer<float4> WeldedDirection:register(t1);
StructuredBuffer<uint> TriangleIndices:register(t2);
StructuredBuffer<uint4> NeighborRanges:register(t3);
StructuredBuffer<uint> NeighborTriangles:register(t4);
RWByteAddressBuffer Deformed:register(u0);
uint VertexAddress(uint index,uint component=0) { return (index*PackedVertexWords+component)*FloatBytes; }
float3 BasePosition(uint index) { return asfloat(Mesh.Load3(VertexAddress(index))); }
float3 NewPosition(uint index) { return asfloat(Deformed.Load3(VertexAddress(index))); }
float3 DirectionOr(float3 value,float3 fallback)
{
    float largest=max(abs(value.x),max(abs(value.y),abs(value.z)));
    return largest>0?normalize(value/largest):fallback;
}
float SampleHeight(float3 position)
{
    float3 relative=(position-Bounds.xyz)*Bounds.w;
    float2 chart=GeometryCount.z==CoordinatesXY?relative.xy:GeometryCount.z==CoordinatesXZ?relative.xz:relative.yz;
    chart*=HeightShape.y;
    float sum=2*Pi*frac(chart.x+chart.y),difference=2*Pi*frac(chart.x-chart.y);
    return MeanHeight+WaveAmplitude*(cos(sum)+cos(difference));
}
[numthreads(GroupWidth,1,1)] void CSDisplace(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x; if (index>=GeometryCount.x) return;
    [unroll] for (uint component=0;component<PackedVertexWords;++component)
        Deformed.Store(VertexAddress(index,component),Mesh.Load(VertexAddress(index,component)));
    if (HeightShape.x==0) return;
    float3 position=BasePosition(index);
    position+=WeldedDirection[index].xyz*(HeightShape.x*(SampleHeight(position)-HeightShape.z));
    Deformed.Store3(VertexAddress(index),asuint(position));
}
[numthreads(GroupWidth,1,1)] void CSNormals(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x; if (index>=GeometryCount.x || HeightShape.x==0) return;
    float3 original=asfloat(Mesh.Load3(VertexAddress(index,NormalOffset)));
    float3 reference=DirectionOr(original,0),areaNormal=0;
    uint2 neighborhood=NeighborRanges[index].xy;
    for (uint entry=0;entry<neighborhood.y;++entry)
    {
        uint face=NeighborTriangles[neighborhood.x+entry];
        uint a=TriangleIndices[face*3],b=TriangleIndices[face*3+1],c=TriangleIndices[face*3+2];
        float3 originalArea=cross((BasePosition(b)-BasePosition(a))*Bounds.w,(BasePosition(c)-BasePosition(a))*Bounds.w);
        float3 displacedArea=cross((NewPosition(b)-NewPosition(a))*Bounds.w,(NewPosition(c)-NewPosition(a))*Bounds.w);
        areaNormal+=dot(originalArea,reference)<0?-displacedArea:displacedArea;
    }
    Deformed.Store3(VertexAddress(index,NormalOffset),asuint(DirectionOr(areaNormal,original)));
}
