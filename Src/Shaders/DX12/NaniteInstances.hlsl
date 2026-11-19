// D3D12 GPU-produced dispatch arguments select instance work without a count readback.
// D3D12 assembly preserves the decoded buffers' native GPU addresses. Each
// invocation expands one indexed triangle into the material reconstruction ABI.
cbuffer InstanceParameters : register(b0) { uint4 Schedule; };
ByteAddressBuffer InstanceRecords : register(t2);
RWByteAddressBuffer InstanceSchedule : register(u1);
ByteAddressBuffer VertexStream : register(t0),IndexStream : register(t1);
RWByteAddressBuffer WorldTable : register(u0);
static const uint GroupWidth=64,WordSize=4,VertexStride=12,TriangleStride=48,VectorStride=4,CornerCount=3;
static const uint VertexNormal=3,VertexColor=6,VertexUV=10,TableNormal=12,TableColor=24,TableUV=36,TableSource=44;
static const float NormalLengthLimit=1.0e-6f,SRGBThreshold=0.04045f,SRGBSlope=12.92f;
static const float SRGBBias=0.055f,SRGBDenominator=1.055f,SRGBPower=2.4f;
float DecodeChannel(float value)
{ value=saturate(value); return value<=SRGBThreshold?value/SRGBSlope:pow((value+SRGBBias)/SRGBDenominator,SRGBPower); }
float3 NormalizeSource(float3 n)
{
    float bound=max(abs(n.x),max(abs(n.y),abs(n.z)));
    float3 scaled=bound>0?n/bound:float3(0,1,0); float distance=length(scaled);
    return bound*distance>NormalLengthLimit?scaled/distance:float3(0,1,0);
}
static const uint MatrixRowWords=4,NormalMatrixWord=16,TotalGroupWord=3;
static const uint RecordStride=44,RangeOffset=32,DispatchOffset=36,QueryOffset=40,MaximumColumns=64;
static const uint VisibleState=2,DeferredState=3;
[numthreads(GroupWidth,1,1)] void CSCount(uint id:SV_DispatchThreadID)
{
    if (id>=Schedule.x) return;
    uint record=Schedule.z+id*RecordStride;
    uint4 query=InstanceSchedule.Load4((record+QueryOffset)*WordSize);
    bool eligible=query.z!=0;
    [branch] if (eligible && Schedule.w!=0)
    {
        eligible=false;
        for (uint i=0;i<query.y;++i)
        {
            uint state=VertexStream.Load((query.x+i)*WordSize);
            eligible=eligible || state==VisibleState || state==DeferredState;
        }
    }
    uint count=InstanceSchedule.Load((record+RangeOffset+1)*WordSize);
    InstanceSchedule.Store((Schedule.y+id)*WordSize,eligible?(count+GroupWidth-1)/GroupWidth:0);
}
[numthreads(1,1,1)] void CSScan()
{
    uint groups=0;
    for (uint i=0;i<Schedule.x;++i) { groups+=InstanceSchedule.Load((Schedule.y+i)*WordSize); InstanceSchedule.Store((Schedule.y+i)*WordSize,groups); }
    uint columns=max(1u,min(groups,MaximumColumns));
    InstanceSchedule.Store4(0,uint4(columns,max(1u,(groups+columns-1)/columns),1,groups));
}
float4 InstanceVector(uint word) { return asfloat(InstanceRecords.Load4(word*WordSize)); }
[numthreads(GroupWidth,1,1)] void CSAssemble(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint group=(block.y*InstanceRecords.Load(0)+block.x);
    [branch] if (group>=InstanceRecords.Load(TotalGroupWord*WordSize)) return;
    uint first=0,last=Schedule.x;
    while (first<last) { uint middle=(first+last)/2; if (InstanceRecords.Load((Schedule.y+middle)*WordSize)>group) last=middle; else first=middle+1; }
    uint start=0;
    [branch] if (first>0) start=InstanceRecords.Load((Schedule.y+first-1)*WordSize);
    uint local=(group-start)*GroupWidth+lane,record=Schedule.z+first*RecordStride;
    uint4 GeometryRange=InstanceRecords.Load4((record+RangeOffset)*WordSize),Work=InstanceRecords.Load4((record+DispatchOffset)*WordSize);
    [branch] if (local>=GeometryRange.y) return;
    float4x4 WorldMatrix=float4x4(InstanceVector(record),InstanceVector(record+1*MatrixRowWords),InstanceVector(record+2*MatrixRowWords),InstanceVector(record+3*MatrixRowWords));
    float4x4 NormalMatrix=float4x4(InstanceVector(record+NormalMatrixWord),InstanceVector(record+NormalMatrixWord+1*MatrixRowWords),InstanceVector(record+NormalMatrixWord+2*MatrixRowWords),InstanceVector(record+NormalMatrixWord+3*MatrixRowWords));
    uint3 indices=IndexStream.Load3((GeometryRange.z+local*CornerCount)*WordSize);
    float4 positions[CornerCount],normals[CornerCount],colors[CornerCount]; float2 uv[CornerCount];
    [unroll] for (uint c=0;c<CornerCount;++c) { positions[c]=float4(0,0,0,1); normals[c]=colors[c]=0; uv[c]=0; }
    bool valid=all(indices<GeometryRange.w);
    [branch] if (valid)
    {
        [unroll] for (uint sourceCorner=0;sourceCorner<CornerCount;++sourceCorner)
        {
            uint address=indices[sourceCorner]*VertexStride*WordSize;
            float3 p=asfloat(VertexStream.Load3(address)),n=asfloat(VertexStream.Load3(address+VertexNormal*WordSize));
            float4 color=asfloat(VertexStream.Load4(address+VertexColor*WordSize)); uv[sourceCorner]=asfloat(VertexStream.Load2(address+VertexUV*WordSize));
            positions[sourceCorner]=mul(WorldMatrix,float4(p,1)); float3 transformed=mul(NormalMatrix,float4(n,0)).xyz;
            valid=valid && all(isfinite(positions[sourceCorner])) && all(isfinite(transformed)) && all(isfinite(color)) && all(isfinite(uv[sourceCorner]));
            normals[sourceCorner]=float4(NormalizeSource(transformed),0); colors[sourceCorner]=float4(DecodeChannel(color.r),DecodeChannel(color.g),DecodeChannel(color.b),color.a);
        }
    }
    uint destination=(GeometryRange.x+local)*TriangleStride;
    [unroll] for (uint destinationCorner=0;destinationCorner<CornerCount;++destinationCorner)
    {
        if (!valid) { positions[destinationCorner]=float4(0,0,0,1); normals[destinationCorner]=colors[destinationCorner]=0; uv[destinationCorner]=0; }
        WorldTable.Store4((destination+destinationCorner*VectorStride)*WordSize,asuint(positions[destinationCorner]));
        WorldTable.Store4((destination+TableNormal+destinationCorner*VectorStride)*WordSize,asuint(normals[destinationCorner]));
        WorldTable.Store4((destination+TableColor+destinationCorner*VectorStride)*WordSize,asuint(colors[destinationCorner]));
    }
    WorldTable.Store4((destination+TableUV)*WordSize,asuint(float4(uv[0],uv[1])));
    WorldTable.Store4((destination+TableUV+VectorStride)*WordSize,asuint(float4(uv[2],0,0)));
    WorldTable.Store4((destination+TableSource)*WordSize,uint4(Work.z,local,0,0));
}
