// Vlachos et al., "Curved PN Triangles", I3D 2001.
// https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
// Independent D3D12 PN tessellation. Edge controls are shared across duplicate
// attribute seams, while a straight-crease flag resolves conflicting tangent
// planes. A UAV dependency makes edge writes visible to vertex sampling.
static const float CenterElevation=1.5;
static const uint Threads=64,PackedWords=12,BytesPerWord=4,NormalOffset=3,ColorOffset=6,UVOffset=10;
static const uint LinearCrease=1,ReversedGeometry=1,DegeneratePatch=1;
cbuffer TessellationFrame:register(b0) { uint4 SourceCount,ResultCount; float4 Surface,ObjectBounds; };
ByteAddressBuffer Mesh:register(t0);
StructuredBuffer<uint4> Patches:register(t1),PatchEdges:register(t2),AttributeEdges:register(t3),SharedEdges:register(t4);
RWStructuredBuffer<float4> EdgePoints:register(u0);
RWByteAddressBuffer ResultVertices:register(u1);
RWStructuredBuffer<uint> ResultIndices:register(u2);
uint ByteOffset(uint index,uint component=0) { return (index*PackedWords+component)*BytesPerWord; }
float3 UnitDirection(float3 value,float3 fallback)
{ float largest=max(abs(value.x),max(abs(value.y),abs(value.z))); return largest>0?normalize(value/largest):fallback; }
float3 LocalPosition(uint index) { return (asfloat(Mesh.Load3(ByteOffset(index)))-ObjectBounds.xyz)*ObjectBounds.w; }
float3 LocalNormal(uint index) { return UnitDirection(asfloat(Mesh.Load3(ByteOffset(index,NormalOffset))),0); }
float LinearColor(float encoded)
{
    const float Split=0.04045,Divisor=12.92,Bias=0.055,Scale=1.055,Power=2.4;
    encoded=saturate(encoded); return encoded<=Split?encoded/Divisor:pow((encoded+Bias)/Scale,Power);
}
float EncodedColor(float value)
{
    const float Split=0.0031308,Multiplier=12.92,Bias=0.055,Scale=1.055,Power=2.4;
    value=saturate(value); return value<=Split?value*Multiplier:Scale*pow(value,1/Power)-Bias;
}
float InterpolatedColor(float a,float b,float c,float3 barycentric)
{ if (a==b && b==c) return a; return EncodedColor(dot(float3(LinearColor(a),LinearColor(b),LinearColor(c)),barycentric)); }
// Avoid subtracting two positions that already contain a shared origin offset.
float3 SourceEdge(uint first,uint second) { return (asfloat(Mesh.Load3(ByteOffset(second)))-asfloat(Mesh.Load3(ByteOffset(first))))*ObjectBounds.w; }
float3 MidNormal(uint first,uint second,float3 na,float3 nb)
{
    float3 edge=SourceEdge(first,second),sum=na+nb; float squared=dot(edge,edge);
    if (squared>0) sum-=edge*(2*dot(edge,sum)/squared); return UnitDirection(sum,na);
}
void WriteVertex(uint index,uint3 corner,float3 barycentric,float3 position,float3 normal)
{
    float4 a=asfloat(Mesh.Load4(ByteOffset(corner.x,ColorOffset))),b=asfloat(Mesh.Load4(ByteOffset(corner.y,ColorOffset))),c=asfloat(Mesh.Load4(ByteOffset(corner.z,ColorOffset)));
    float4 color=float4(InterpolatedColor(a.x,b.x,c.x,barycentric),InterpolatedColor(a.y,b.y,c.y,barycentric),InterpolatedColor(a.z,b.z,c.z,barycentric),dot(float3(a.w,b.w,c.w),barycentric));
    float2 uv=asfloat(Mesh.Load2(ByteOffset(corner.x,UVOffset)))*barycentric.x+asfloat(Mesh.Load2(ByteOffset(corner.y,UVOffset)))*barycentric.y+asfloat(Mesh.Load2(ByteOffset(corner.z,UVOffset)))*barycentric.z;
    ResultVertices.Store3(ByteOffset(index),asuint(ObjectBounds.xyz+position/ObjectBounds.w));
    ResultVertices.Store3(ByteOffset(index,NormalOffset),asuint(UnitDirection(normal,UnitDirection(LocalNormal(corner.x),float3(0,0,1)))));
    ResultVertices.Store4(ByteOffset(index,ColorOffset),asuint(color)); ResultVertices.Store2(ByteOffset(index,UVOffset),asuint(uv));
}
uint AttributeSample(uint edgeId,uint start,uint numerator)
{ uint4 edge=AttributeEdges[edgeId]; return SourceCount.x+edgeId*(ResultCount.z-1)+(edge.x==start?numerator:ResultCount.z-numerator)-1; }
uint SampleAddress(uint faceId,uint i,uint j)
{
    uint4 p=Patches[faceId],e=PatchEdges[faceId]; uint n=ResultCount.z;
    if (i==0 && j==0) return p.x; if (i==n) return p.y; if (j==n) return p.z;
    if (j==0) return AttributeSample(e.x,p.x,i); if (i==0) return AttributeSample(e.z,p.x,j); if (i+j==n) return AttributeSample(e.y,p.y,j);
    return SourceCount.x+SourceCount.z*(n-1)+faceId*ResultCount.w+(i-1)*(n-1)-(i-1)*i/2+j-1;
}
void OrientedControls(uint edgeId,uint start,out float3 first,out float3 second)
{
    uint4 edge=AttributeEdges[edgeId]; bool forward=(edge.x==start)!=((edge.w&ReversedGeometry)!=0);
    first=EdgePoints[edge.z*2+(forward?0:1)].xyz; second=EdgePoints[edge.z*2+(forward?1:0)].xyz;
}
[numthreads(Threads,1,1)] void CSEdges(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x; if (index>=SourceCount.w) return; uint4 edge=SharedEdges[index];
    float3 a=LocalPosition(edge.x),direction=SourceEdge(edge.x,edge.y);
    float3 first=a+direction/3,second=a+direction*(2.0/3.0);
    if ((edge.z&LinearCrease)==0)
    { float3 na=LocalNormal(edge.x),nb=LocalNormal(edge.y); first-=na*(dot(direction,na)/3); second+=nb*(dot(direction,nb)/3); }
    EdgePoints[index*2]=float4(first,0); EdgePoints[index*2+1]=float4(second,0);
}
[numthreads(Threads,1,1)] void CSVertices(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x,n=ResultCount.z; if (index>=ResultCount.x) return;
    if (index<SourceCount.x)
    { [unroll] for (uint component=0;component<PackedWords;++component) ResultVertices.Store(ByteOffset(index,component),Mesh.Load(ByteOffset(index,component))); return; }
    uint interiorStart=SourceCount.x+SourceCount.z*(n-1);
    if (index<interiorStart)
    {
        uint ordinal=index-SourceCount.x,edgeId=ordinal/(n-1),step=ordinal%(n-1)+1; uint4 attribute=AttributeEdges[edgeId],edge=SharedEdges[attribute.z];
        uint along=(attribute.w&ReversedGeometry)!=0?n-step:step;
        float t=float(along)/n,s=float(n-along)/n,v=float(step)/n,u=float(n-step)/n;
        float3 a=LocalPosition(edge.x),b=LocalPosition(edge.y),planar=a*s+b*t;
        float3 curve=a*(s*s*s)+EdgePoints[attribute.z*2].xyz*(3*s*s*t)+EdgePoints[attribute.z*2+1].xyz*(3*s*t*t)+b*(t*t*t);
        float3 na=LocalNormal(attribute.x),nb=LocalNormal(attribute.y),mid=MidNormal(attribute.x,attribute.y,na,nb);
        WriteVertex(index,uint3(attribute.x,attribute.y,attribute.x),float3(u,v,0),lerp(planar,curve,Surface.x),lerp(na*u+nb*v,na*(u*u)+mid*(2*u*v)+nb*(v*v),Surface.x)); return;
    }
    uint offset=index-interiorStart,faceId=offset/ResultCount.w,residual=offset%ResultCount.w,i=1;
    while (residual>=n-i-1) { residual-=n-i-1; ++i; } uint j=residual+1;
    float3 barycentric=float3(float(n-i-j),float(i),float(j))/n; float u=barycentric.x,v=barycentric.y,w=barycentric.z;
    uint3 corner=Patches[faceId].xyz,e=PatchEdges[faceId].xyz;
    float3 a=LocalPosition(corner.x),b=LocalPosition(corner.y),c=LocalPosition(corner.z),ab,ba,bc,cb,ca,ac;
    OrientedControls(e.x,corner.x,ab,ba); OrientedControls(e.y,corner.y,bc,cb); OrientedControls(e.z,corner.z,ca,ac);
    float3 center=(ab+ba+bc+cb+ca+ac)*(CenterElevation/6.0)-(a+b+c)/6;
    float3 curved=a*(u*u*u)+b*(v*v*v)+c*(w*w*w)+ab*(3*u*u*v)+ba*(3*u*v*v)+bc*(3*v*v*w)+cb*(3*v*w*w)+ca*(3*w*w*u)+ac*(3*w*u*u)+center*(6*u*v*w);
    float3 na=LocalNormal(corner.x),nb=LocalNormal(corner.y),nc=LocalNormal(corner.z);
    float3 quadratic=na*(u*u)+nb*(v*v)+nc*(w*w)+MidNormal(corner.x,corner.y,na,nb)*(2*u*v)+MidNormal(corner.y,corner.z,nb,nc)*(2*v*w)+MidNormal(corner.z,corner.x,nc,na)*(2*w*u);
    WriteVertex(index,corner,barycentric,lerp(a*u+b*v+c*w,curved,Surface.x),lerp(na*u+nb*v+nc*w,quadratic,Surface.x));
}
[numthreads(Threads,1,1)] void CSIndices(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x,n=ResultCount.z; if (index>=ResultCount.y/3) return; uint faceId=index/(n*n),ordinal=index%(n*n);
    if (n>1 && (Patches[faceId].w&DegeneratePatch)!=0) { ResultIndices[index*3]=ResultIndices[index*3+1]=ResultIndices[index*3+2]=Patches[faceId].x; return; }
    uint remaining=n*n-ordinal,root=(uint)sqrt(float(remaining)); if (root*root<remaining) ++root;
    uint i=n-root,inRow=ordinal-i*(2*n-i),j=inRow/2; bool upper=(inRow&1)!=0;
    ResultIndices[index*3]=SampleAddress(faceId,upper?i+1:i,j); ResultIndices[index*3+1]=upper?SampleAddress(faceId,i+1,j+1):SampleAddress(faceId,i+1,j);
    ResultIndices[index*3+2]=SampleAddress(faceId,i,j+1);
}
