#version 310 es
// Vlachos et al., "Curved PN Triangles", I3D 2001.
// https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
// ES compute geometry amplification: shared PN edge data, uniquely owned vertex
// samples and deterministic triangle indices. This does not require ES 3.2
// tessellation-evaluation stages. Storage and vertex/index barriers are required.
precision highp float;
precision highp int;
#define PN_GROUP_WIDTH 64
layout(local_size_x=PN_GROUP_WIDTH) in;
layout(std140,binding=0) uniform TessellationParameters { uvec4 SourceCount,ResultCount; vec4 Surface,Box; } Parameters;
layout(std430,binding=0) readonly buffer BaseVertices { uint Data[]; } Mesh;
layout(std430,binding=1) readonly buffer SourcePatches { uvec4 Data[]; } PatchData;
layout(std430,binding=2) readonly buffer PatchBoundaries { uvec4 Data[]; } Boundary;
layout(std430,binding=3) readonly buffer AttributeBoundaries { uvec4 Data[]; } Attribute;
layout(std430,binding=4) readonly buffer GeometricBoundaries { uvec4 Data[]; } Shared;
layout(std430,binding=5) buffer EdgeControlPoints { vec4 Data[]; } Curve;
layout(std430,binding=6) writeonly buffer GeneratedVertices { uint Data[]; } VertexData;
layout(std430,binding=7) writeonly buffer GeneratedIndices { uint Data[]; } IndexData;
const float CenterElevation=1.5;
const uint PackedWords=12u,NormalComponent=3u,ColorComponent=6u,UVComponent=10u;
const uint StraightBoundary=1u,ReverseBoundary=1u,CollapsedPatch=1u;
vec3 ReadTriple(uint start) { return uintBitsToFloat(uvec3(Mesh.Data[start],Mesh.Data[start+1u],Mesh.Data[start+2u])); }
vec4 ReadColor(uint vertex)
{ uint start=vertex*PackedWords+ColorComponent; return uintBitsToFloat(uvec4(Mesh.Data[start],Mesh.Data[start+1u],Mesh.Data[start+2u],Mesh.Data[start+3u])); }
vec2 ReadUV(uint vertex)
{ uint start=vertex*PackedWords+UVComponent; return uintBitsToFloat(uvec2(Mesh.Data[start],Mesh.Data[start+1u])); }
vec3 NormalizeOr(vec3 value,vec3 fallback)
{ float maximum=max(abs(value.x),max(abs(value.y),abs(value.z))); return maximum>0.0?normalize(value/maximum):fallback; }
vec3 SourcePosition(uint vertex) { return (ReadTriple(vertex*PackedWords)-Parameters.Box.xyz)*Parameters.Box.w; }
vec3 SourceNormal(uint vertex) { return NormalizeOr(ReadTriple(vertex*PackedWords+NormalComponent),vec3(0.0)); }
float ToLinear(float encoded)
{
    const float Split=0.04045,Divisor=12.92,Bias=0.055,Multiplier=1.055,Exponent=2.4;
    encoded=clamp(encoded,0.0,1.0); return encoded<=Split?encoded/Divisor:pow((encoded+Bias)/Multiplier,Exponent);
}
float ToEncoded(float value)
{
    const float Split=0.0031308,Multiplier=12.92,Bias=0.055,Scale=1.055,Exponent=2.4;
    value=clamp(value,0.0,1.0); return value<=Split?value*Multiplier:Scale*pow(value,1.0/Exponent)-Bias;
}
float BlendColor(float a,float b,float c,vec3 weights)
{ return a==b && b==c?a:ToEncoded(dot(vec3(ToLinear(a),ToLinear(b),ToLinear(c)),weights)); }
// A direct endpoint difference avoids cancellation of normalized coordinates.
vec3 SourceEdge(uint a,uint b) { return (ReadTriple(b*PackedWords)-ReadTriple(a*PackedWords))*Parameters.Box.w; }
vec3 EdgeNormal(uint a,uint b,vec3 na,vec3 nb)
{
    vec3 direction=SourceEdge(a,b),mid=na+nb; float squared=dot(direction,direction);
    if (squared>0.0) mid-=direction*(2.0*dot(direction,mid)/squared); return NormalizeOr(mid,na);
}
void WriteTriple(uint start,vec3 value)
{ uvec3 words=floatBitsToUint(value); VertexData.Data[start]=words.x; VertexData.Data[start+1u]=words.y; VertexData.Data[start+2u]=words.z; }
void EmitVertex(uint vertex,uvec3 corner,vec3 weights,vec3 position,vec3 normal)
{
    vec4 a=ReadColor(corner.x),b=ReadColor(corner.y),c=ReadColor(corner.z);
    vec4 color=vec4(BlendColor(a.x,b.x,c.x,weights),BlendColor(a.y,b.y,c.y,weights),BlendColor(a.z,b.z,c.z,weights),dot(vec3(a.w,b.w,c.w),weights));
    vec2 uv=ReadUV(corner.x)*weights.x+ReadUV(corner.y)*weights.y+ReadUV(corner.z)*weights.z; uint start=vertex*PackedWords;
    WriteTriple(start,Parameters.Box.xyz+position/Parameters.Box.w); WriteTriple(start+NormalComponent,NormalizeOr(normal,NormalizeOr(SourceNormal(corner.x),vec3(0,0,1))));
    uvec4 colors=floatBitsToUint(color); for (uint channel=0u;channel<4u;++channel) VertexData.Data[start+ColorComponent+channel]=colors[channel];
    uvec2 coordinates=floatBitsToUint(uv); VertexData.Data[start+UVComponent]=coordinates.x; VertexData.Data[start+UVComponent+1u]=coordinates.y;
}
void ControlsFrom(uint attributeId,uint from,out vec3 nearStart,out vec3 nearEnd)
{
    uvec4 edge=Attribute.Data[attributeId]; bool forward=(edge.x==from)!=((edge.w&ReverseBoundary)!=0u);
    nearStart=Curve.Data[edge.z*2u+(forward?0u:1u)].xyz; nearEnd=Curve.Data[edge.z*2u+(forward?1u:0u)].xyz;
}
uint BoundaryVertex(uint attributeId,uint from,uint numerator)
{ uvec4 edge=Attribute.Data[attributeId]; uint n=Parameters.ResultCount.z; return Parameters.SourceCount.x+attributeId*(n-1u)+(edge.x==from?numerator:n-numerator)-1u; }
uint GridVertex(uint face,uint i,uint j)
{
    uvec4 corners=PatchData.Data[face],edges=Boundary.Data[face]; uint n=Parameters.ResultCount.z;
    if (i==0u && j==0u) return corners.x; if (i==n) return corners.y; if (j==n) return corners.z;
    if (j==0u) return BoundaryVertex(edges.x,corners.x,i); if (i==0u) return BoundaryVertex(edges.z,corners.x,j); if (i+j==n) return BoundaryVertex(edges.y,corners.y,j);
    return Parameters.SourceCount.x+Parameters.SourceCount.z*(n-1u)+face*Parameters.ResultCount.w+(i-1u)*(n-1u)-(i-1u)*i/2u+j-1u;
}
void main()
{
    uint vertex=gl_GlobalInvocationID.x,n=Parameters.ResultCount.z;
#if EVANS_TESSELLATION_PASS == 0
    if (vertex>=Parameters.SourceCount.w) return; uvec4 edge=Shared.Data[vertex];
    vec3 a=SourcePosition(edge.x),delta=SourceEdge(edge.x,edge.y),nearA=a+delta/3.0,nearB=a+delta*(2.0/3.0);
    if ((edge.z&StraightBoundary)==0u) { vec3 na=SourceNormal(edge.x),nb=SourceNormal(edge.y); nearA-=na*(dot(na,delta)/3.0); nearB+=nb*(dot(nb,delta)/3.0); }
    Curve.Data[vertex*2u]=vec4(nearA,0); Curve.Data[vertex*2u+1u]=vec4(nearB,0);
#elif EVANS_TESSELLATION_PASS == 1
    if (vertex>=Parameters.ResultCount.x) return;
    if (vertex<Parameters.SourceCount.x) { for (uint component=0u;component<PackedWords;++component) VertexData.Data[vertex*PackedWords+component]=Mesh.Data[vertex*PackedWords+component]; return; }
    uint firstInterior=Parameters.SourceCount.x+Parameters.SourceCount.z*(n-1u);
    if (vertex<firstInterior)
    {
        uint ordinal=vertex-Parameters.SourceCount.x,attributeId=ordinal/(n-1u),step=ordinal%(n-1u)+1u; uvec4 record=Attribute.Data[attributeId],edge=Shared.Data[record.z];
        uint along=(record.w&ReverseBoundary)!=0u?n-step:step;
        float t=float(along)/float(n),s=float(n-along)/float(n),v=float(step)/float(n),u=float(n-step)/float(n);
        vec3 a=SourcePosition(edge.x),b=SourcePosition(edge.y),cubic=a*(s*s*s)+Curve.Data[record.z*2u].xyz*(3.0*s*s*t)+Curve.Data[record.z*2u+1u].xyz*(3.0*s*t*t)+b*(t*t*t);
        vec3 na=SourceNormal(record.x),nb=SourceNormal(record.y),mid=EdgeNormal(record.x,record.y,na,nb);
        EmitVertex(vertex,uvec3(record.x,record.y,record.x),vec3(u,v,0),mix(a*s+b*t,cubic,Parameters.Surface.x),mix(na*u+nb*v,na*(u*u)+mid*(2.0*u*v)+nb*(v*v),Parameters.Surface.x)); return;
    }
    uint ordinal=vertex-firstInterior,face=ordinal/Parameters.ResultCount.w,remainder=ordinal%Parameters.ResultCount.w,i=1u;
    while (remainder>=n-i-1u) { remainder-=n-i-1u; ++i; } uint j=remainder+1u;
    vec3 weights=vec3(n-i-j,i,j)/float(n); float u=weights.x,v=weights.y,w=weights.z;
    uvec3 corners=PatchData.Data[face].xyz,edges=Boundary.Data[face].xyz; vec3 a=SourcePosition(corners.x),b=SourcePosition(corners.y),c=SourcePosition(corners.z),ab,ba,bc,cb,ca,ac;
    ControlsFrom(edges.x,corners.x,ab,ba); ControlsFrom(edges.y,corners.y,bc,cb); ControlsFrom(edges.z,corners.z,ca,ac);
    vec3 center=(ab+ba+bc+cb+ca+ac)*(CenterElevation/6.0)-(a+b+c)/6.0;
    vec3 cubic=a*(u*u*u)+b*(v*v*v)+c*(w*w*w)+ab*(3.0*u*u*v)+ba*(3.0*u*v*v)+bc*(3.0*v*v*w)+cb*(3.0*v*w*w)+ca*(3.0*w*w*u)+ac*(3.0*w*u*u)+center*(6.0*u*v*w);
    vec3 na=SourceNormal(corners.x),nb=SourceNormal(corners.y),nc=SourceNormal(corners.z);
    vec3 quadratic=na*(u*u)+nb*(v*v)+nc*(w*w)+EdgeNormal(corners.x,corners.y,na,nb)*(2.0*u*v)+EdgeNormal(corners.y,corners.z,nb,nc)*(2.0*v*w)+EdgeNormal(corners.z,corners.x,nc,na)*(2.0*w*u);
    EmitVertex(vertex,corners,weights,mix(a*u+b*v+c*w,cubic,Parameters.Surface.x),mix(na*u+nb*v+nc*w,quadratic,Parameters.Surface.x));
#else
    if (vertex>=Parameters.ResultCount.y/3u) return; uint face=vertex/(n*n),ordinal=vertex%(n*n);
    if (n>1u && (PatchData.Data[face].w&CollapsedPatch)!=0u) { IndexData.Data[vertex*3u]=PatchData.Data[face].x; IndexData.Data[vertex*3u+1u]=PatchData.Data[face].x; IndexData.Data[vertex*3u+2u]=PatchData.Data[face].x; return; }
    uint residual=n*n-ordinal,root=uint(sqrt(float(residual))); if (root*root<residual) ++root;
    uint i=n-root,inRow=ordinal-i*(2u*n-i),j=inRow/2u; bool upper=(inRow&1u)!=0u;
    IndexData.Data[vertex*3u]=GridVertex(face,upper?i+1u:i,j); IndexData.Data[vertex*3u+1u]=upper?GridVertex(face,i+1u,j+1u):GridVertex(face,i+1u,j); IndexData.Data[vertex*3u+2u]=GridVertex(face,i,j+1u);
#endif
}
