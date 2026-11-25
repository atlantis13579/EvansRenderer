#include <metal_stdlib>
using namespace metal;
// MSL owns the complete deformation and gather kernels. End or synchronize the
// producer encoder before normal gathering, then make its vertex writes visible
// to raster encoders. No shader source from another technique/API is included.
constant uint VertexWordCount=12,NormalWordOffset=3,ProjectionXY=1,ProjectionXZ=2;
constant float Pi=3.14159265358979323846,HeightMean=0.5,WaveWeight=0.25;
struct DisplacementParameters { float4 Bounds,Shape; uint4 Counts; };
float3 ReadPosition(const device uint* mesh,uint vertexIndex)
{ uint offset=vertexIndex*VertexWordCount; return as_type<float3>(uint3(mesh[offset],mesh[offset+1],mesh[offset+2])); }
void WriteVector(device uint* mesh,uint offset,float3 value)
{ uint3 bits=as_type<uint3>(value); mesh[offset]=bits.x; mesh[offset+1]=bits.y; mesh[offset+2]=bits.z; }
float3 UnitOr(float3 value,float3 fallback)
{ float largest=max(abs(value.x),max(abs(value.y),abs(value.z))); return largest>0?normalize(value/largest):fallback; }
float HeightAt(float3 point,constant DisplacementParameters& frame)
{
    float3 local=(point-frame.Bounds.xyz)*frame.Bounds.w;
    float2 coordinate=(frame.Counts.z==ProjectionXY?local.xy:frame.Counts.z==ProjectionXZ?local.xz:local.yz)*frame.Shape.y;
    float2 phase=2*Pi*fract(float2(coordinate.x+coordinate.y,coordinate.x-coordinate.y));
    return HeightMean+WaveWeight*(cos(phase.x)+cos(phase.y));
}
kernel void Displace(const device uint* source [[buffer(0)]],const device float4* directions [[buffer(1)]],
    device uint* destination [[buffer(5)]],constant DisplacementParameters& frame [[buffer(6)]],uint vertexIndex [[thread_position_in_grid]])
{
    if (vertexIndex>=frame.Counts.x) return;
    for (uint word=0;word<VertexWordCount;++word) destination[vertexIndex*VertexWordCount+word]=source[vertexIndex*VertexWordCount+word];
    if (frame.Shape.x==0) return;
    float3 position=ReadPosition(source,vertexIndex);
    position+=directions[vertexIndex].xyz*((HeightAt(position,frame)-frame.Shape.z)*frame.Shape.x);
    WriteVector(destination,vertexIndex*VertexWordCount,position);
}
kernel void RebuildNormals(const device uint* source [[buffer(0)]],const device uint* indices [[buffer(2)]],
    const device uint4* ranges [[buffer(3)]],const device uint* adjacency [[buffer(4)]],device uint* destination [[buffer(5)]],
    constant DisplacementParameters& frame [[buffer(6)]],uint vertexIndex [[thread_position_in_grid]])
{
    if (vertexIndex>=frame.Counts.x || frame.Shape.x==0) return;
    uint offset=vertexIndex*VertexWordCount+NormalWordOffset;
    float3 original=as_type<float3>(uint3(source[offset],source[offset+1],source[offset+2]));
    float3 reference=UnitOr(original,float3(0)),sum=float3(0); uint2 range=ranges[vertexIndex].xy;
    for (uint neighbor=0;neighbor<range.y;++neighbor)
    {
        uint triangle=adjacency[range.x+neighbor];
        uint a=indices[triangle*3],b=indices[triangle*3+1],c=indices[triangle*3+2];
        float3 sourceArea=cross((ReadPosition(source,b)-ReadPosition(source,a))*frame.Bounds.w,(ReadPosition(source,c)-ReadPosition(source,a))*frame.Bounds.w);
        float3 displacedArea=cross((ReadPosition(destination,b)-ReadPosition(destination,a))*frame.Bounds.w,(ReadPosition(destination,c)-ReadPosition(destination,a))*frame.Bounds.w);
        sum+=dot(sourceArea,reference)<0?-displacedArea:displacedArea;
    }
    WriteVector(destination,offset,UnitOr(sum,original));
}
