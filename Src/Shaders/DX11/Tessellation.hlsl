// Vlachos et al., "Curved PN Triangles", I3D 2001.
// https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
// D3D11 compute PN tessellation: canonical cubic edge controls, packed vertex
// evaluation, then indexed lattice generation. These are GPU geometry producers;
// fixed-function hull/domain execution is a separate path.
static const float CenterElevation=1.5;
static const uint GroupWidth=64,VertexWords=12,WordBytes=4,NormalWord=3,ColorWord=6,UVWord=10;
static const uint StraightEdge=1,ReverseEdge=1,DegenerateFace=1;
cbuffer Parameters:register(b0) { uint4 Inputs,Outputs; float4 Shape,Bounds; };
ByteAddressBuffer Source:register(t0);
StructuredBuffer<uint4> Faces:register(t1),FaceEdges:register(t2),Attributes:register(t3),GeometryEdges:register(t4);
RWStructuredBuffer<float4> Controls:register(u0);
RWByteAddressBuffer Vertices:register(u1);
// A typed R32_UINT UAV also serves as the native index stream.
RWBuffer<uint> Indices:register(u2);
uint Address(uint id,uint word=0) { return (id*VertexWords+word)*WordBytes; }
float3 Position(uint id) { return (asfloat(Source.Load3(Address(id)))-Bounds.xyz)*Bounds.w; }
float3 UnitOr(float3 value,float3 fallback)
{ float magnitude=max(abs(value.x),max(abs(value.y),abs(value.z))); return magnitude>0?normalize(value/magnitude):fallback; }
float3 Normal(uint id) { return UnitOr(asfloat(Source.Load3(Address(id,NormalWord))),0); }
float Decode(float value)
{
    const float Breakpoint=0.04045,Slope=12.92,Offset=0.055,Scale=1.055,Exponent=2.4;
    value=saturate(value); return value<=Breakpoint?value/Slope:pow((value+Offset)/Scale,Exponent);
}
float Encode(float value)
{
    const float Breakpoint=0.0031308,Slope=12.92,Offset=0.055,Scale=1.055,Exponent=2.4;
    value=saturate(value); return value<=Breakpoint?value*Slope:Scale*pow(value,1/Exponent)-Offset;
}
float ColorBlend(float a,float b,float c,float3 weight)
{ return a==b && b==c?a:Encode(dot(float3(Decode(a),Decode(b),Decode(c)),weight)); }
// Subtract source endpoints before normalization to retain thin-edge directions.
float3 EdgeDelta(uint from,uint to) { return (asfloat(Source.Load3(Address(to)))-asfloat(Source.Load3(Address(from))))*Bounds.w; }
float3 NormalControl(uint from,uint to,float3 na,float3 nb)
{
    float3 delta=EdgeDelta(from,to),sum=na+nb; float squared=dot(delta,delta);
    if (squared>0) sum-=delta*(2*dot(delta,sum)/squared); return UnitOr(sum,na);
}
void StoreVertex(uint id,uint3 corners,float3 weight,float3 position,float3 normal)
{
    float4 a=asfloat(Source.Load4(Address(corners.x,ColorWord))),b=asfloat(Source.Load4(Address(corners.y,ColorWord))),c=asfloat(Source.Load4(Address(corners.z,ColorWord)));
    float4 color=float4(ColorBlend(a.r,b.r,c.r,weight),ColorBlend(a.g,b.g,c.g,weight),ColorBlend(a.b,b.b,c.b,weight),dot(float3(a.a,b.a,c.a),weight));
    float2 uv=asfloat(Source.Load2(Address(corners.x,UVWord)))*weight.x+asfloat(Source.Load2(Address(corners.y,UVWord)))*weight.y+asfloat(Source.Load2(Address(corners.z,UVWord)))*weight.z;
    Vertices.Store3(Address(id),asuint(Bounds.xyz+position/Bounds.w)); Vertices.Store3(Address(id,NormalWord),asuint(UnitOr(normal,UnitOr(Normal(corners.x),float3(0,0,1)))));
    Vertices.Store4(Address(id,ColorWord),asuint(color)); Vertices.Store2(Address(id,UVWord),asuint(uv));
}
void EdgeControls(uint attribute,uint from,out float3 nearFrom,out float3 nearTo)
{
    uint4 record=Attributes[attribute]; bool forward=(record.x==from)!=((record.w&ReverseEdge)!=0);
    nearFrom=Controls[record.z*2+(forward?0:1)].xyz; nearTo=Controls[record.z*2+(forward?1:0)].xyz;
}
uint EdgeVertex(uint attribute,uint from,uint step)
{
    uint4 edge=Attributes[attribute]; uint n=Outputs.z;
    return Inputs.x+attribute*(n-1)+(edge.x==from?step:n-step)-1;
}
uint LatticeVertex(uint face,uint i,uint j)
{
    uint4 corners=Faces[face],edges=FaceEdges[face]; uint n=Outputs.z;
    if (i==0 && j==0) return corners.x; if (i==n) return corners.y; if (j==n) return corners.z;
    if (j==0) return EdgeVertex(edges.x,corners.x,i); if (i==0) return EdgeVertex(edges.z,corners.x,j); if (i+j==n) return EdgeVertex(edges.y,corners.y,j);
    uint before=(i-1)*(n-1)-(i-1)*i/2;
    return Inputs.x+Inputs.z*(n-1)+face*Outputs.w+before+j-1;
}
[numthreads(GroupWidth,1,1)] void CSEdges(uint3 invocation:SV_DispatchThreadID)
{
    uint id=invocation.x; if (id>=Inputs.w) return; uint4 edge=GeometryEdges[id];
    float3 a=Position(edge.x),delta=EdgeDelta(edge.x,edge.y),nearA=a+delta/3,nearB=a+2*delta/3;
    if ((edge.z&StraightEdge)==0) { float3 na=Normal(edge.x),nb=Normal(edge.y); nearA-=na*(dot(na,delta)/3); nearB+=nb*(dot(nb,delta)/3); }
    Controls[id*2]=float4(nearA,0); Controls[id*2+1]=float4(nearB,0);
}
[numthreads(GroupWidth,1,1)] void CSVertices(uint3 invocation:SV_DispatchThreadID)
{
    uint id=invocation.x,n=Outputs.z; if (id>=Outputs.x) return;
    if (id<Inputs.x) { [unroll] for (uint word=0;word<VertexWords;++word) Vertices.Store(Address(id,word),Source.Load(Address(id,word))); return; }
    uint edgeEnd=Inputs.x+Inputs.z*(n-1);
    if (id<edgeEnd)
    {
        uint ordinal=id-Inputs.x,attribute=ordinal/(n-1),step=ordinal%(n-1)+1;
        uint4 edge=Attributes[attribute],geometric=GeometryEdges[edge.z]; uint canonical=(edge.w&ReverseEdge)!=0?n-step:step;
        float t=float(canonical)/n,s=float(n-canonical)/n,v=float(step)/n,u=float(n-step)/n;
        float3 a=Position(geometric.x),b=Position(geometric.y);
        float3 planar=a*s+b*t,curved=a*(s*s*s)+Controls[edge.z*2].xyz*(3*s*s*t)+Controls[edge.z*2+1].xyz*(3*s*t*t)+b*(t*t*t);
        float3 na=Normal(edge.x),nb=Normal(edge.y),middle=NormalControl(edge.x,edge.y,na,nb);
        StoreVertex(id,uint3(edge.x,edge.y,edge.x),float3(u,v,0),lerp(planar,curved,Shape.x),lerp(na*u+nb*v,na*(u*u)+middle*(2*u*v)+nb*(v*v),Shape.x)); return;
    }
    uint local=id-edgeEnd,face=local/Outputs.w,remaining=local%Outputs.w,i=1;
    while (remaining>=n-i-1) { remaining-=n-i-1; ++i; } uint j=remaining+1;
    float u=float(n-i-j)/n,v=float(i)/n,w=float(j)/n; uint3 corners=Faces[face].xyz; uint3 edges=FaceEdges[face].xyz;
    float3 a=Position(corners.x),b=Position(corners.y),c=Position(corners.z),ab,ba,bc,cb,ca,ac;
    EdgeControls(edges.x,corners.x,ab,ba); EdgeControls(edges.y,corners.y,bc,cb); EdgeControls(edges.z,corners.z,ca,ac);
    float3 average=(ab+ba+bc+cb+ca+ac)/6,center=average*CenterElevation-(a+b+c)/6;
    float3 curved=a*(u*u*u)+b*(v*v*v)+c*(w*w*w)+ab*(3*u*u*v)+ba*(3*u*v*v)+bc*(3*v*v*w)+cb*(3*v*w*w)+ca*(3*w*w*u)+ac*(3*w*u*u)+center*(6*u*v*w);
    float3 na=Normal(corners.x),nb=Normal(corners.y),nc=Normal(corners.z);
    float3 shaded=na*(u*u)+nb*(v*v)+nc*(w*w)+NormalControl(corners.x,corners.y,na,nb)*(2*u*v)+NormalControl(corners.y,corners.z,nb,nc)*(2*v*w)+NormalControl(corners.z,corners.x,nc,na)*(2*w*u);
    StoreVertex(id,corners,float3(u,v,w),lerp(a*u+b*v+c*w,curved,Shape.x),lerp(na*u+nb*v+nc*w,shaded,Shape.x));
}
[numthreads(GroupWidth,1,1)] void CSIndices(uint3 invocation:SV_DispatchThreadID)
{
    uint id=invocation.x,n=Outputs.z; if (id>=Outputs.y/3) return; uint face=id/(n*n),local=id%(n*n);
    if (n>1 && (Faces[face].w&DegenerateFace)!=0) { Indices[id*3]=Indices[id*3+1]=Indices[id*3+2]=Faces[face].x; return; }
    uint remainder=n*n-local,r=(uint)sqrt(float(remainder)); if (r*r<remainder) ++r;
    uint i=n-r,within=local-i*(2*n-i),j=within/2;
    Indices[id*3]=LatticeVertex(face,(within&1)!=0?i+1:i,j);
    Indices[id*3+1]=(within&1)!=0?LatticeVertex(face,i+1,j+1):LatticeVertex(face,i+1,j);
    Indices[id*3+2]=LatticeVertex(face,i,j+1);
}
