// GPU instance count, prefix allocation and indirect assembly.
// Deferred clusters remain eligible for the second occlusion phase.
// Decode consumers read the native indexed streams directly. The resulting
// world-space table is shared by coverage and deferred attribute reconstruction.
cbuffer InstanceFrame : register(b0) { uint4 Schedule; };
ByteAddressBuffer InstanceTable : register(t2);
RWByteAddressBuffer InstanceWork : register(u1);
ByteAddressBuffer Vertices : register(t0);
Buffer<uint> Indices : register(t1);
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
RWStructuredBuffer<WorldTriangle> Triangles : register(u0);
static const uint Threads=64,VertexWords=12,WordBytes=4,NormalWord=3,ColorWord=6,UVWord=10,Corners=3;
static const float NormalMinimumLength=1.0e-6f,SRGBBreak=0.04045f,SRGBLinearScale=12.92f;
static const float SRGBOffset=0.055f,SRGBScale=1.055f,SRGBExponent=2.4f;
float LinearColor(float encoded)
{ encoded=saturate(encoded); return encoded<=SRGBBreak?encoded/SRGBLinearScale:pow((encoded+SRGBOffset)/SRGBScale,SRGBExponent); }
float3 UnitNormal(float3 value)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    float3 direction=scale>0?value/scale:float3(0,1,0); float magnitude=length(direction);
    return scale*magnitude>NormalMinimumLength?direction/magnitude:float3(0,1,0);
}
// Inclusive ends support empty sources without losing the following source.
static const uint MatrixRowWords=4,NormalMatrixWord=16,TotalGroupWord=3,RecordWords=44,RangeWord=32,DispatchWord=36,QueryWord=40,DispatchColumns=64;
static const uint MainVisible=2,Deferred=3;
[numthreads(Threads,1,1)] void CSCount(uint id:SV_DispatchThreadID)
{
    if (id>=Schedule.x) return;
    uint base=Schedule.z+id*RecordWords;
    uint4 query=InstanceWork.Load4((base+QueryWord)*WordBytes);
    bool active=query.z!=0;
    [branch] if (active && Schedule.w!=0)
    {
        active=false;
        for (uint i=0;i<query.y;++i)
        {
            uint state=Vertices.Load((query.x+i)*WordBytes);
            active=active || state==MainVisible || state==Deferred;
        }
    }
    uint triangles=InstanceWork.Load((base+RangeWord+1)*WordBytes);
    InstanceWork.Store((Schedule.y+id)*WordBytes,active?(triangles+Threads-1)/Threads:0);
}
[numthreads(1,1,1)] void CSScan()
{
    uint total=0;
    for (uint i=0;i<Schedule.x;++i) { total+=InstanceWork.Load((Schedule.y+i)*WordBytes); InstanceWork.Store((Schedule.y+i)*WordBytes,total); }
    uint columns=max(1u,min(total,DispatchColumns));
    InstanceWork.Store4(0,uint4(columns,max(1u,(total+columns-1)/columns),1,total));
}
float4 InstanceRow(uint word) { return asfloat(InstanceTable.Load4(word*WordBytes)); }
[numthreads(Threads,1,1)] void CSAssemble(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint groupIndex=group.y*InstanceTable.Load(0)+group.x;
    [branch] if (groupIndex>=InstanceTable.Load(TotalGroupWord*WordBytes)) return;
    uint low=0,high=Schedule.x;
    while (low<high) { uint middle=(low+high)/2; if (InstanceTable.Load((Schedule.y+middle)*WordBytes)<=groupIndex) low=middle+1; else high=middle; }
    uint previous=0;
    [branch] if (low!=0) previous=InstanceTable.Load((Schedule.y+low-1)*WordBytes);
    uint local=(groupIndex-previous)*Threads+lane,base=Schedule.z+low*RecordWords;
    uint4 Range=InstanceTable.Load4((base+RangeWord)*WordBytes),Dispatch=InstanceTable.Load4((base+DispatchWord)*WordBytes);
    [branch] if (local>=Range.y) return;
    float4x4 World=float4x4(InstanceRow(base),InstanceRow(base+1*MatrixRowWords),InstanceRow(base+2*MatrixRowWords),InstanceRow(base+3*MatrixRowWords));
    float4x4 Normal=float4x4(InstanceRow(base+NormalMatrixWord),InstanceRow(base+NormalMatrixWord+1*MatrixRowWords),InstanceRow(base+NormalMatrixWord+2*MatrixRowWords),InstanceRow(base+NormalMatrixWord+3*MatrixRowWords));
    uint baseIndex=Range.z+local*Corners;
    uint3 indices=uint3(Indices[baseIndex],Indices[baseIndex+1],Indices[baseIndex+2]);
    WorldTriangle output=(WorldTriangle)0; bool valid=all(indices<Range.w);
    [branch] if (valid)
    {
        float2 uv[Corners];
        [unroll] for (uint corner=0;corner<Corners;++corner)
        {
            uint address=indices[corner]*VertexWords*WordBytes;
            float3 p=asfloat(Vertices.Load3(address)),n=asfloat(Vertices.Load3(address+NormalWord*WordBytes));
            float4 color=asfloat(Vertices.Load4(address+ColorWord*WordBytes)); uv[corner]=asfloat(Vertices.Load2(address+UVWord*WordBytes));
            float4 position=mul(World,float4(p,1)); float3 normal=mul(Normal,float4(n,0)).xyz;
            valid=valid && all(isfinite(position)) && all(isfinite(normal)) && all(isfinite(color)) && all(isfinite(uv[corner]));
            output.Position[corner]=position; output.Normal[corner]=float4(UnitNormal(normal),0);
            output.Color[corner]=float4(LinearColor(color.r),LinearColor(color.g),LinearColor(color.b),color.a);
        }
        output.UVAB=float4(uv[0],uv[1]); output.UVC=float4(uv[2],0,0);
    }
    if (!valid) output=(WorldTriangle)0;
    [unroll] for (uint corner=0;corner<Corners;++corner) output.Position[corner].w=1;
    output.Source=uint4(Dispatch.z,local,0,0); Triangles[Range.x+local]=output;
}
