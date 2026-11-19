#version 450
// Vlachos et al., "Curved PN Triangles", I3D 2001.
// https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
// Vulkan owns this complete three-dispatch PN producer. Edge construction must
// finish before vertex evaluation; vertex/index storage then needs host and
// vertex/index-input visibility. GLSL is compiled directly in memory.
#define TESSELLATION_GROUP_WIDTH 64
layout(local_size_x=TESSELLATION_GROUP_WIDTH) in;
layout(std140,set=0,binding=0) uniform Parameters { uvec4 Inputs,Outputs; vec4 Shape,Bounds; } Frame;
layout(std430,set=0,binding=1) readonly buffer SourceBuffer { uint Words[]; } Source;
layout(std430,set=0,binding=2) readonly buffer FaceBuffer { uvec4 Values[]; } Faces;
layout(std430,set=0,binding=3) readonly buffer FaceEdgeBuffer { uvec4 Values[]; } FaceEdges;
layout(std430,set=0,binding=4) readonly buffer AttributeBuffer { uvec4 Values[]; } Attributes;
layout(std430,set=0,binding=5) readonly buffer GeometryBuffer { uvec4 Values[]; } GeometryEdges;
layout(std430,set=0,binding=6) buffer ControlBuffer { vec4 Values[]; } Controls;
layout(std430,set=0,binding=7) writeonly buffer VertexBuffer { uint Words[]; } Vertices;
layout(std430,set=0,binding=8) writeonly buffer IndexBuffer { uint Values[]; } Indices;
const float CenterElevation=1.5;
const uint VertexWords=12u,NormalWord=3u,ColorWord=6u,UVWord=10u;
const uint LinearEdge=1u,ReverseEdge=1u,DegenerateFace=1u;
vec3 Read3(uint address) { return uintBitsToFloat(uvec3(Source.Words[address],Source.Words[address+1u],Source.Words[address+2u])); }
vec4 Read4(uint address) { return uintBitsToFloat(uvec4(Source.Words[address],Source.Words[address+1u],Source.Words[address+2u],Source.Words[address+3u])); }
vec2 Read2(uint address) { return uintBitsToFloat(uvec2(Source.Words[address],Source.Words[address+1u])); }
vec3 UnitOr(vec3 value,vec3 fallback)
{ float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); return scale>0?normalize(value/scale):fallback; }
vec3 Position(uint index) { return (Read3(index*VertexWords)-Frame.Bounds.xyz)*Frame.Bounds.w; }
vec3 Normal(uint index) { return UnitOr(Read3(index*VertexWords+NormalWord),vec3(0)); }
float Decode(float value)
{
    const float Threshold=0.04045,Slope=12.92,Offset=0.055,Scale=1.055,Power=2.4;
    value=clamp(value,0,1); return value<=Threshold?value/Slope:pow((value+Offset)/Scale,Power);
}
float Encode(float value)
{
    const float Threshold=0.0031308,Slope=12.92,Offset=0.055,Scale=1.055,Power=2.4;
    value=clamp(value,0,1); return value<=Threshold?value*Slope:Scale*pow(value,1/Power)-Offset;
}
float ColorBlend(float a,float b,float c,vec3 weights)
{ return a==b && b==c?a:Encode(dot(vec3(Decode(a),Decode(b),Decode(c)),weights)); }
// Preserve short-edge directions before the object-space normalization.
vec3 SourceEdge(uint a,uint b) { return (Read3(b*VertexWords)-Read3(a*VertexWords))*Frame.Bounds.w; }
vec3 NormalControl(uint a,uint b,vec3 na,vec3 nb)
{ vec3 edge=SourceEdge(a,b),sum=na+nb; float squared=dot(edge,edge); if (squared>0) sum-=edge*(2*dot(edge,sum)/squared); return UnitOr(sum,na); }
void Store3(uint address,vec3 value)
{ uvec3 bits=floatBitsToUint(value); Vertices.Words[address]=bits.x; Vertices.Words[address+1u]=bits.y; Vertices.Words[address+2u]=bits.z; }
void StoreVertex(uint index,uvec3 corners,vec3 weights,vec3 position,vec3 normal)
{
    vec4 a=Read4(corners.x*VertexWords+ColorWord),b=Read4(corners.y*VertexWords+ColorWord),c=Read4(corners.z*VertexWords+ColorWord);
    vec4 color=vec4(ColorBlend(a.r,b.r,c.r,weights),ColorBlend(a.g,b.g,c.g,weights),ColorBlend(a.b,b.b,c.b,weights),dot(vec3(a.a,b.a,c.a),weights));
    vec2 uv=Read2(corners.x*VertexWords+UVWord)*weights.x+Read2(corners.y*VertexWords+UVWord)*weights.y+Read2(corners.z*VertexWords+UVWord)*weights.z;
    uint address=index*VertexWords; Store3(address,Frame.Bounds.xyz+position/Frame.Bounds.w); Store3(address+NormalWord,UnitOr(normal,UnitOr(Normal(corners.x),vec3(0,0,1))));
    uvec4 colorBits=floatBitsToUint(color); for (uint component=0u;component<4u;++component) Vertices.Words[address+ColorWord+component]=colorBits[component];
    uvec2 uvBits=floatBitsToUint(uv); Vertices.Words[address+UVWord]=uvBits.x; Vertices.Words[address+UVWord+1u]=uvBits.y;
}
void OrientedControls(uint attributeId,uint from,out vec3 first,out vec3 second)
{
    uvec4 record=Attributes.Values[attributeId]; bool forward=(record.x==from)!=((record.w&ReverseEdge)!=0u);
    first=Controls.Values[record.z*2u+(forward?0u:1u)].xyz; second=Controls.Values[record.z*2u+(forward?1u:0u)].xyz;
}
uint EdgeVertex(uint attributeId,uint from,uint step)
{ uvec4 edge=Attributes.Values[attributeId]; uint n=Frame.Outputs.z; return Frame.Inputs.x+attributeId*(n-1u)+(edge.x==from?step:n-step)-1u; }
uint Lattice(uint face,uint i,uint j)
{
    uvec4 p=Faces.Values[face],e=FaceEdges.Values[face]; uint n=Frame.Outputs.z;
    if (i==0u && j==0u) return p.x; if (i==n) return p.y; if (j==n) return p.z;
    if (j==0u) return EdgeVertex(e.x,p.x,i); if (i==0u) return EdgeVertex(e.z,p.x,j); if (i+j==n) return EdgeVertex(e.y,p.y,j);
    return Frame.Inputs.x+Frame.Inputs.z*(n-1u)+face*Frame.Outputs.w+(i-1u)*(n-1u)-(i-1u)*i/2u+j-1u;
}
void main()
{
    uint index=gl_GlobalInvocationID.x,n=Frame.Outputs.z;
#if EVANS_TESSELLATION_PASS == 0
    if (index>=Frame.Inputs.w) return; uvec4 edge=GeometryEdges.Values[index];
    vec3 a=Position(edge.x),delta=SourceEdge(edge.x,edge.y),first=a+delta/3,second=a+delta*(2.0/3.0);
    if ((edge.z&LinearEdge)==0u) { vec3 na=Normal(edge.x),nb=Normal(edge.y); first-=na*(dot(na,delta)/3); second+=nb*(dot(nb,delta)/3); }
    Controls.Values[index*2u]=vec4(first,0); Controls.Values[index*2u+1u]=vec4(second,0);
#elif EVANS_TESSELLATION_PASS == 1
    if (index>=Frame.Outputs.x) return;
    if (index<Frame.Inputs.x) { for (uint word=0u;word<VertexWords;++word) Vertices.Words[index*VertexWords+word]=Source.Words[index*VertexWords+word]; return; }
    uint edgeEnd=Frame.Inputs.x+Frame.Inputs.z*(n-1u);
    if (index<edgeEnd)
    {
        uint ordinal=index-Frame.Inputs.x,attributeId=ordinal/(n-1u),step=ordinal%(n-1u)+1u; uvec4 record=Attributes.Values[attributeId],edge=GeometryEdges.Values[record.z];
        uint canonical=(record.w&ReverseEdge)!=0u?n-step:step; float t=float(canonical)/float(n),s=float(n-canonical)/float(n),v=float(step)/float(n),u=float(n-step)/float(n);
        vec3 a=Position(edge.x),b=Position(edge.y),curved=a*(s*s*s)+Controls.Values[record.z*2u].xyz*(3*s*s*t)+Controls.Values[record.z*2u+1u].xyz*(3*s*t*t)+b*(t*t*t);
        vec3 na=Normal(record.x),nb=Normal(record.y),middle=NormalControl(record.x,record.y,na,nb);
        StoreVertex(index,uvec3(record.x,record.y,record.x),vec3(u,v,0),mix(a*s+b*t,curved,Frame.Shape.x),mix(na*u+nb*v,na*(u*u)+middle*(2*u*v)+nb*(v*v),Frame.Shape.x)); return;
    }
    uint local=index-edgeEnd,face=local/Frame.Outputs.w,residual=local%Frame.Outputs.w,i=1u;
    while (residual>=n-i-1u) { residual-=n-i-1u; ++i; } uint j=residual+1u;
    vec3 weights=vec3(n-i-j,i,j)/float(n); float u=weights.x,v=weights.y,w=weights.z;
    uvec3 p=Faces.Values[face].xyz,e=FaceEdges.Values[face].xyz; vec3 a=Position(p.x),b=Position(p.y),c=Position(p.z),ab,ba,bc,cb,ca,ac;
    OrientedControls(e.x,p.x,ab,ba); OrientedControls(e.y,p.y,bc,cb); OrientedControls(e.z,p.z,ca,ac);
    vec3 center=(ab+ba+bc+cb+ca+ac)*(CenterElevation/6.0)-(a+b+c)/6;
    vec3 curved=a*(u*u*u)+b*(v*v*v)+c*(w*w*w)+ab*(3*u*u*v)+ba*(3*u*v*v)+bc*(3*v*v*w)+cb*(3*v*w*w)+ca*(3*w*w*u)+ac*(3*w*u*u)+center*(6*u*v*w);
    vec3 na=Normal(p.x),nb=Normal(p.y),nc=Normal(p.z),quadratic=na*(u*u)+nb*(v*v)+nc*(w*w)+NormalControl(p.x,p.y,na,nb)*(2*u*v)+NormalControl(p.y,p.z,nb,nc)*(2*v*w)+NormalControl(p.z,p.x,nc,na)*(2*w*u);
    StoreVertex(index,p,weights,mix(a*u+b*v+c*w,curved,Frame.Shape.x),mix(na*u+nb*v+nc*w,quadratic,Frame.Shape.x));
#else
    if (index>=Frame.Outputs.y/3u) return; uint face=index/(n*n),ordinal=index%(n*n);
    if (n>1u && (Faces.Values[face].w&DegenerateFace)!=0u) { Indices.Values[index*3u]=Faces.Values[face].x; Indices.Values[index*3u+1u]=Faces.Values[face].x; Indices.Values[index*3u+2u]=Faces.Values[face].x; return; }
    uint remaining=n*n-ordinal,root=uint(sqrt(float(remaining))); if (root*root<remaining) ++root;
    uint i=n-root,inRow=ordinal-i*(2u*n-i),j=inRow/2u; bool upper=(inRow&1u)!=0u;
    Indices.Values[index*3u]=Lattice(face,upper?i+1u:i,j); Indices.Values[index*3u+1u]=upper?Lattice(face,i+1u,j+1u):Lattice(face,i+1u,j); Indices.Values[index*3u+2u]=Lattice(face,i,j+1u);
#endif
}
