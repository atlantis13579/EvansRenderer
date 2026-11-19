// D3D11: surface-normal geometric displacement followed by deterministic CSR
// normal reconstruction. Dispatch positions, unbind/rebind UAV state as needed,
// then dispatch normals; never read positions from an incomplete producer.
static const uint ThreadCount=64,VertexWords=12,WordBytes=4,NormalWord=3;
static const uint ProjectionXY=1,ProjectionXZ=2;
static const float Pi=3.14159265358979323846,ModeWeight=0.25,HeightMidpoint=0.5;
cbuffer DisplacementParameters:register(b0) { float4 ObjectBounds,Shape; uint4 Counts; };
ByteAddressBuffer Source:register(t0);
StructuredBuffer<float4> Direction:register(t1);
StructuredBuffer<uint> Indices:register(t2);
StructuredBuffer<uint4> Neighborhood:register(t3);
StructuredBuffer<uint> Adjacent:register(t4);
RWByteAddressBuffer Result:register(u0);
uint Address(uint vertex,uint word=0) { return (vertex*VertexWords+word)*WordBytes; }
float3 SourcePosition(uint vertex) { return asfloat(Source.Load3(Address(vertex))); }
float3 ResultPosition(uint vertex) { return asfloat(Result.Load3(Address(vertex))); }
float3 UnitOr(float3 value,float3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    return scale>0?normalize(value/scale):fallback;
}
float Height(float3 position)
{
    float3 object=(position-ObjectBounds.xyz)*ObjectBounds.w;
    float2 uv=(Counts.z==ProjectionXY?object.xy:Counts.z==ProjectionXZ?object.xz:object.yz)*Shape.y;
    float2 phase=2*Pi*frac(float2(uv.x+uv.y,uv.x-uv.y));
    return HeightMidpoint+ModeWeight*(cos(phase.x)+cos(phase.y));
}
[numthreads(ThreadCount,1,1)] void CSDisplace(uint3 thread:SV_DispatchThreadID)
{
    uint vertex=thread.x; if (vertex>=Counts.x) return;
    [unroll] for (uint word=0;word<VertexWords;++word) Result.Store(Address(vertex,word),Source.Load(Address(vertex,word)));
    if (Shape.x==0) return;
    float3 position=SourcePosition(vertex)+Direction[vertex].xyz*((Height(SourcePosition(vertex))-Shape.z)*Shape.x);
    Result.Store3(Address(vertex),asuint(position));
}
[numthreads(ThreadCount,1,1)] void CSNormals(uint3 thread:SV_DispatchThreadID)
{
    uint vertex=thread.x; if (vertex>=Counts.x || Shape.x==0) return;
    float3 original=asfloat(Source.Load3(Address(vertex,NormalWord))),orientation=UnitOr(original,0);
    uint2 range=Neighborhood[vertex].xy; float3 sum=0;
    for (uint neighbor=0;neighbor<range.y;++neighbor)
    {
        uint face=Adjacent[range.x+neighbor];
        uint a=Indices[face*3],b=Indices[face*3+1],c=Indices[face*3+2];
        float3 sourceFace=cross((SourcePosition(b)-SourcePosition(a))*ObjectBounds.w,(SourcePosition(c)-SourcePosition(a))*ObjectBounds.w);
        float3 deformedFace=cross((ResultPosition(b)-ResultPosition(a))*ObjectBounds.w,(ResultPosition(c)-ResultPosition(a))*ObjectBounds.w);
        sum+=dot(sourceFace,orientation)<0?-deformedFace:deformedFace;
    }
    Result.Store3(Address(vertex,NormalWord),asuint(UnitOr(sum,original)));
}
