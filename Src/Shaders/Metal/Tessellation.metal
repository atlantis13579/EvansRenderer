#include <metal_stdlib>
using namespace metal;

// Independent Metal compute tessellation: shared cubic PN edge controls,
// exclusive vertex writes and a uniformly indexed triangular lattice.
// Vlachos et al., "Curved PN Triangles", I3D 2001:
// https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
// Quadratic shading normals are separate from the cubic position derivative.
constant uint VertexStride=12,NormalOffset=3,ColorOffset=6,UVOffset=10;
constant uint LinearCrease=1,ReverseOrientation=1,DegeneratePatch=1;
constant float CenterElevation=1.5;
struct TessellationFrame { uint4 Inputs,Outputs; float4 Shape,Bounds; };
float3 ReadVector(device const uint* vertices,uint word)
{ return as_type<float3>(uint3(vertices[word],vertices[word+1],vertices[word+2])); }
float3 SafeUnit(float3 value,float3 fallback)
{ float magnitude=max(abs(value.x),max(abs(value.y),abs(value.z))); return magnitude>0?normalize(value/magnitude):fallback; }
float3 PatchPosition(device const uint* vertices,uint index,constant TessellationFrame& frame)
{ return (ReadVector(vertices,index*VertexStride)-frame.Bounds.xyz)*frame.Bounds.w; }
float3 PatchNormal(device const uint* vertices,uint index)
{ return SafeUnit(ReadVector(vertices,index*VertexStride+NormalOffset),float3(0)); }
float LinearChannel(float value)
{
    const float Breakpoint=0.04045,Divisor=12.92,Bias=0.055,Scale=1.055,Exponent=2.4;
    value=clamp(value,0.0f,1.0f); return value<=Breakpoint?value/Divisor:pow((value+Bias)/Scale,Exponent);
}
float EncodedChannel(float value)
{
    const float Breakpoint=0.0031308,Multiplier=12.92,Bias=0.055,Scale=1.055,Exponent=2.4;
    value=clamp(value,0.0f,1.0f); return value<=Breakpoint?value*Multiplier:Scale*pow(value,1/Exponent)-Bias;
}
float ColorChannel(float a,float b,float c,float3 weight)
{ return a==b && b==c?a:EncodedChannel(dot(float3(LinearChannel(a),LinearChannel(b),LinearChannel(c)),weight)); }
// The original endpoint difference retains short edges under large translations.
float3 SourceEdge(device const uint* source,constant TessellationFrame& frame,uint a,uint b)
{ return (ReadVector(source,b*VertexStride)-ReadVector(source,a*VertexStride))*frame.Bounds.w; }
float3 QuadraticControl(device const uint* source,constant TessellationFrame& frame,uint a,uint b,float3 na,float3 nb)
{
    float3 edge=SourceEdge(source,frame,a,b),coefficient=na+nb; float squared=dot(edge,edge);
    if (squared>0) coefficient-=edge*(2*dot(edge,coefficient)/squared); return SafeUnit(coefficient,na);
}
void StoreVector(device uint* vertices,uint word,float3 value)
{ uint3 bits=as_type<uint3>(value); vertices[word]=bits.x; vertices[word+1]=bits.y; vertices[word+2]=bits.z; }
float4 ReadColor(device const uint* source,uint vertexIndex)
{
    uint word=vertexIndex*VertexStride+ColorOffset;
    return as_type<float4>(uint4(source[word],source[word+1],source[word+2],source[word+3]));
}
float2 ReadUV(device const uint* source,uint vertexIndex)
{ uint word=vertexIndex*VertexStride+UVOffset; return as_type<float2>(uint2(source[word],source[word+1])); }
void EmitVertex(device const uint* source,device uint* vertices,constant TessellationFrame& frame,
    uint vertexIndex,uint3 corner,float3 weight,float3 position,float3 normal)
{
    float4 a=ReadColor(source,corner.x),b=ReadColor(source,corner.y),c=ReadColor(source,corner.z);
    float4 color=float4(ColorChannel(a.r,b.r,c.r,weight),ColorChannel(a.g,b.g,c.g,weight),
        ColorChannel(a.b,b.b,c.b,weight),dot(float3(a.a,b.a,c.a),weight));
    float2 uv=ReadUV(source,corner.x)*weight.x+ReadUV(source,corner.y)*weight.y+ReadUV(source,corner.z)*weight.z;
    uint word=vertexIndex*VertexStride;
    StoreVector(vertices,word,frame.Bounds.xyz+position/frame.Bounds.w);
    StoreVector(vertices,word+NormalOffset,SafeUnit(normal,SafeUnit(PatchNormal(source,corner.x),float3(0,0,1))));
    uint4 channels=as_type<uint4>(color); for (uint channel=0;channel<4;++channel) vertices[word+ColorOffset+channel]=channels[channel];
    uint2 coordinates=as_type<uint2>(uv); vertices[word+UVOffset]=coordinates.x; vertices[word+UVOffset+1]=coordinates.y;
}
float3 EdgeControl(device const uint4* edges,device const float4* controls,uint edge,uint from,bool nearStart)
{
    uint4 record=edges[edge]; bool forward=(record.x==from)!=((record.w&ReverseOrientation)!=0);
    return controls[record.z*2+uint(forward!=nearStart)].xyz;
}
uint BoundaryIndex(device const uint4* edges,constant TessellationFrame& frame,uint edge,uint from,uint step)
{
    uint4 record=edges[edge]; uint n=frame.Outputs.z;
    return frame.Inputs.x+edge*(n-1)+(record.x==from?step:n-step)-1;
}
uint LatticeIndex(device const uint4* patches,device const uint4* boundaries,device const uint4* edges,
    constant TessellationFrame& frame,uint face,uint i,uint j)
{
    uint4 p=patches[face],e=boundaries[face]; uint n=frame.Outputs.z;
    if (i==0 && j==0) return p.x; if (i==n) return p.y; if (j==n) return p.z;
    if (j==0) return BoundaryIndex(edges,frame,e.x,p.x,i);
    if (i==0) return BoundaryIndex(edges,frame,e.z,p.x,j);
    if (i+j==n) return BoundaryIndex(edges,frame,e.y,p.y,j);
    return frame.Inputs.x+frame.Inputs.z*(n-1)+face*frame.Outputs.w+(i-1)*(n-1)-(i-1)*i/2+j-1;
}
kernel void BuildEdges(device const uint* source [[buffer(0)]],device const uint4* edges [[buffer(4)]],
    device float4* controls [[buffer(5)]],constant TessellationFrame& frame [[buffer(8)]],uint index [[thread_position_in_grid]])
{
    if (index>=frame.Inputs.w) return; uint4 edge=edges[index];
    float3 a=PatchPosition(source,edge.x,frame),delta=SourceEdge(source,frame,edge.x,edge.y);
    float3 first=a+delta/3,second=a+delta*(2.0f/3.0f);
    if ((edge.z&LinearCrease)==0)
    { float3 na=PatchNormal(source,edge.x),nb=PatchNormal(source,edge.y); first-=na*(dot(na,delta)/3); second+=nb*(dot(nb,delta)/3); }
    controls[index*2]=float4(first,0); controls[index*2+1]=float4(second,0);
}
kernel void BuildVertices(device const uint* source [[buffer(0)]],device const uint4* patches [[buffer(1)]],
    device const uint4* boundaries [[buffer(2)]],device const uint4* edges [[buffer(3)]],device const uint4* geometric [[buffer(4)]],
    device const float4* controls [[buffer(5)]],device uint* vertices [[buffer(6)]],
    constant TessellationFrame& frame [[buffer(8)]],uint vertexIndex [[thread_position_in_grid]])
{
    uint n=frame.Outputs.z; if (vertexIndex>=frame.Outputs.x) return;
    if (vertexIndex<frame.Inputs.x)
    { for (uint word=0;word<VertexStride;++word) vertices[vertexIndex*VertexStride+word]=source[vertexIndex*VertexStride+word]; return; }
    uint interiorStart=frame.Inputs.x+frame.Inputs.z*(n-1);
    if (vertexIndex<interiorStart)
    {
        uint ordinal=vertexIndex-frame.Inputs.x,edgeId=ordinal/(n-1),step=ordinal%(n-1)+1;
        uint4 record=edges[edgeId],edge=geometric[record.z]; uint along=(record.w&ReverseOrientation)!=0?n-step:step;
        float t=float(along)/n,s=float(n-along)/n,v=float(step)/n,u=float(n-step)/n;
        float3 a=PatchPosition(source,edge.x,frame),b=PatchPosition(source,edge.y,frame);
        float3 cubic=a*(s*s*s)+controls[record.z*2].xyz*(3*s*s*t)+controls[record.z*2+1].xyz*(3*s*t*t)+b*(t*t*t);
        float3 na=PatchNormal(source,record.x),nb=PatchNormal(source,record.y);
        float3 middle=QuadraticControl(source,frame,record.x,record.y,na,nb);
        EmitVertex(source,vertices,frame,vertexIndex,uint3(record.x,record.y,record.x),float3(u,v,0),
            mix(a*s+b*t,cubic,frame.Shape.x),mix(na*u+nb*v,na*(u*u)+middle*(2*u*v)+nb*(v*v),frame.Shape.x)); return;
    }
    uint ordinal=vertexIndex-interiorStart,face=ordinal/frame.Outputs.w,remainder=ordinal%frame.Outputs.w,i=1;
    while (remainder>=n-i-1) { remainder-=n-i-1; ++i; } uint j=remainder+1;
    float3 weight=float3(n-i-j,i,j)/n; float u=weight.x,v=weight.y,w=weight.z;
    uint3 p=patches[face].xyz,e=boundaries[face].xyz;
    float3 a=PatchPosition(source,p.x,frame),b=PatchPosition(source,p.y,frame),c=PatchPosition(source,p.z,frame);
    float3 ab=EdgeControl(edges,controls,e.x,p.x,true),ba=EdgeControl(edges,controls,e.x,p.x,false);
    float3 bc=EdgeControl(edges,controls,e.y,p.y,true),cb=EdgeControl(edges,controls,e.y,p.y,false);
    float3 ca=EdgeControl(edges,controls,e.z,p.z,true),ac=EdgeControl(edges,controls,e.z,p.z,false);
    float3 center=(ab+ba+bc+cb+ca+ac)*(CenterElevation/6)-(a+b+c)/6;
    float3 cubic=a*(u*u*u)+b*(v*v*v)+c*(w*w*w)+ab*(3*u*u*v)+ba*(3*u*v*v)+bc*(3*v*v*w)+cb*(3*v*w*w)+
        ca*(3*w*w*u)+ac*(3*w*u*u)+center*(6*u*v*w);
    float3 na=PatchNormal(source,p.x),nb=PatchNormal(source,p.y),nc=PatchNormal(source,p.z);
    float3 quadratic=na*(u*u)+nb*(v*v)+nc*(w*w)+QuadraticControl(source,frame,p.x,p.y,na,nb)*(2*u*v)+
        QuadraticControl(source,frame,p.y,p.z,nb,nc)*(2*v*w)+QuadraticControl(source,frame,p.z,p.x,nc,na)*(2*w*u);
    EmitVertex(source,vertices,frame,vertexIndex,p,weight,mix(a*u+b*v+c*w,cubic,frame.Shape.x),
        mix(na*u+nb*v+nc*w,quadratic,frame.Shape.x));
}
kernel void BuildIndices(device const uint4* patches [[buffer(1)]],device const uint4* boundaries [[buffer(2)]],
    device const uint4* edges [[buffer(3)]],device uint* indices [[buffer(7)]],
    constant TessellationFrame& frame [[buffer(8)]],uint faceIndex [[thread_position_in_grid]])
{
    uint n=frame.Outputs.z; if (faceIndex>=frame.Outputs.y/3) return;
    uint face=faceIndex/(n*n),ordinal=faceIndex%(n*n);
    if (n>1 && (patches[face].w&DegeneratePatch)!=0)
    { indices[faceIndex*3]=indices[faceIndex*3+1]=indices[faceIndex*3+2]=patches[face].x; return; }
    // Rows contain 2*(n-i)-1 triangles. Correct sqrt rounding in integer space.
    uint residual=n*n-ordinal,root=uint(sqrt(float(residual))); if (root*root<residual) ++root;
    uint i=n-root,inRow=ordinal-i*(2*n-i),j=inRow/2; bool upper=(inRow&1)!=0;
    indices[faceIndex*3]=LatticeIndex(patches,boundaries,edges,frame,face,upper?i+1:i,j);
    indices[faceIndex*3+1]=LatticeIndex(patches,boundaries,edges,frame,face,i+1,upper?j+1:j);
    indices[faceIndex*3+2]=LatticeIndex(patches,boundaries,edges,frame,face,i,j+1);
}
