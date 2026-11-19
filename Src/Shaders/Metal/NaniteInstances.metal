#include <metal_stdlib>
using namespace metal;
// Encoders separate GPU count/prefix production from indirect geometry assembly.
constant uint Threads=64,InstanceStride=44,RangeWord=32,DispatchWord=36,QueryWord=40,DispatchColumns=64;
constant uint TotalGroupWord=3,VectorWords=4,NormalMatrixWord=16,MainVisible=2,Deferred=3;
uint4 readInstance(device const uint* words,uint at) { return uint4(words[at],words[at+1],words[at+2],words[at+3]); }
float4 readVertex(device const uint* words,uint at) { return as_type<float4>(readInstance(words,at)); }
kernel void EvansInstanceCount(constant uint4& schedule [[buffer(0)]],device const uint* decisions [[buffer(1)]],
    device uint* words [[buffer(4)]],uint id [[thread_position_in_grid]])
{
    if (id>=schedule.x) return;
    uint record=schedule.z+id*InstanceStride; uint4 query=readInstance(words,record+QueryWord); bool active=query.z!=0;
    if (active && schedule.w!=0)
    {
        active=false;
        for (uint i=0;i<query.y;++i) { uint state=decisions[query.x+i]; active=active || state==MainVisible || state==Deferred; }
    }
    uint count=words[record+RangeWord+1]; words[schedule.y+id]=active?(count+Threads-1)/Threads:0;
}
kernel void EvansInstanceScan(constant uint4& schedule [[buffer(0)]],device uint* words [[buffer(4)]])
{
    uint total=0;
    for (uint i=0;i<schedule.x;++i) { total+=words[schedule.y+i]; words[schedule.y+i]=total; }
    uint columns=max(1u,min(total,DispatchColumns));
    words[0]=columns; words[1]=max(1u,(total+columns-1)/columns); words[2]=1; words[TotalGroupWord]=total;
}
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
// Native decoded streams are expanded once on the GPU for coverage and shading.
// Explicit matrix rows preserve the C++ ABI without transposing the source data.
struct GeometryAssembly { float4 World[4],Normal[4]; uint4 Range,Dispatch; };
constant uint AssemblyVertexWords=12,AssemblyCorners=3,AssemblyNormalWord=3,AssemblyColorWord=6,AssemblyUVWord=10;
constant float AssemblyNormalLimit=1.0e-6f,AssemblySRGBBreak=0.04045f,AssemblySRGBSlope=12.92f;
constant float AssemblySRGBBias=0.055f,AssemblySRGBScale=1.055f,AssemblySRGBPower=2.4f;
float assemblyLinear(float value)
{ value=clamp(value,0.0f,1.0f); return value<=AssemblySRGBBreak?value/AssemblySRGBSlope:pow((value+AssemblySRGBBias)/AssemblySRGBScale,AssemblySRGBPower); }
float3 assemblyNormal(float3 n)
{
    float scale=max(abs(n.x),max(abs(n.y),abs(n.z))); float3 unit=scale>0?n/scale:float3(0,1,0);
    float magnitude=length(unit); return scale*magnitude>AssemblyNormalLimit?unit/magnitude:float3(0,1,0);
}
kernel void EvansInstanceAssemble(constant uint4& schedule [[buffer(0)]],device const uint* vertices [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device WorldTriangle* output [[buffer(3)]],device const uint* words [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint job=group.y*words[0]+group.x; if (job>=words[TotalGroupWord]) return;
    uint low=0,high=schedule.x;
    while (low<high) { uint middle=(low+high)/2; if (words[schedule.y+middle]<=job) low=middle+1; else high=middle; }
    uint previous=0; if (low>0) previous=words[schedule.y+low-1];
    uint local=(job-previous)*Threads+lane,record=schedule.z+low*InstanceStride;
    GeometryAssembly frame;
    for (uint row=0;row<4;++row) { frame.World[row]=readVertex(words,record+row*VectorWords); frame.Normal[row]=readVertex(words,record+NormalMatrixWord+row*VectorWords); }
    frame.Range=readInstance(words,record+RangeWord); frame.Dispatch=readInstance(words,record+DispatchWord);
    if (local>=frame.Range.y) return;
    uint first=frame.Range.z+local*AssemblyCorners;
    uint3 corners(indices[first],indices[first+1],indices[first+2]); bool valid=all(corners<frame.Range.w);
    WorldTriangle result={}; float2 uv[3]={};
    if (valid) for (uint c=0;c<AssemblyCorners;++c)
    {
        uint at=corners[c]*AssemblyVertexWords;
        float4 p(readVertex(vertices,at).xyz,1),n(readVertex(vertices,at+AssemblyNormalWord).xyz,0);
        float4 color=readVertex(vertices,at+AssemblyColorWord);
        uv[c]=as_type<float2>(uint2(vertices[at+AssemblyUVWord],vertices[at+AssemblyUVWord+1]));
        result.Position[c]=float4(dot(frame.World[0],p),dot(frame.World[1],p),dot(frame.World[2],p),1);
        float3 transformed(dot(frame.Normal[0],n),dot(frame.Normal[1],n),dot(frame.Normal[2],n));
        valid=valid && all(isfinite(result.Position[c])) && all(isfinite(transformed)) && all(isfinite(color)) && all(isfinite(uv[c]));
        result.Normal[c]=float4(assemblyNormal(transformed),0);
        result.Color[c]=float4(assemblyLinear(color.r),assemblyLinear(color.g),assemblyLinear(color.b),color.a);
    }
    if (!valid) { result=WorldTriangle{}; for (uint c=0;c<AssemblyCorners;++c) uv[c]=0; }
    for (uint c=0;c<AssemblyCorners;++c) result.Position[c].w=1;
    result.UVAB=float4(uv[0],uv[1]); result.UVC=float4(uv[2],0,0); result.Source=uint4(frame.Dispatch.z,local,0,0);
    output[frame.Range.x+local]=result;
}
