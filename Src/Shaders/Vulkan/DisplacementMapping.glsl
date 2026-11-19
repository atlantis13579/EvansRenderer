#version 450
// Independent Vulkan compute source, compiled directly from GLSL at runtime.
// Pass 0 writes all positions; a compute write -> read/write buffer barrier is
// required before pass 1 gathers area normals. No float atomic extension is used.
#define DISPLACEMENT_GROUP_WIDTH 64
layout(local_size_x=DISPLACEMENT_GROUP_WIDTH) in;
layout(std140,set=0,binding=0) uniform Parameters { vec4 Bounds,Shape; uvec4 Counts; } Frame;
layout(std430,set=0,binding=1) readonly buffer SourceBuffer { uint Words[]; } Source;
layout(std430,set=0,binding=2) readonly buffer DirectionBuffer { vec4 Values[]; } Directions;
layout(std430,set=0,binding=3) readonly buffer IndexBuffer { uint Values[]; } Indices;
layout(std430,set=0,binding=4) readonly buffer RangeBuffer { uvec4 Values[]; } Neighborhoods;
layout(std430,set=0,binding=5) readonly buffer AdjacencyBuffer { uint Values[]; } Adjacency;
layout(std430,set=0,binding=6) buffer ResultBuffer { uint Words[]; } Result;
const uint VertexWords=12u,NormalWord=3u,ProjectionXY=1u,ProjectionXZ=2u;
const float Pi=3.14159265358979323846,MeanHeight=0.5,ModeAmplitude=0.25;
vec3 SourcePosition(uint vertex)
{ uint start=vertex*VertexWords; return uintBitsToFloat(uvec3(Source.Words[start],Source.Words[start+1u],Source.Words[start+2u])); }
vec3 ResultPosition(uint vertex)
{ uint start=vertex*VertexWords; return uintBitsToFloat(uvec3(Result.Words[start],Result.Words[start+1u],Result.Words[start+2u])); }
vec3 UnitOr(vec3 value,vec3 fallback)
{ float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); return scale>0?normalize(value/scale):fallback; }
void StoreVector(uint start,vec3 value)
{ uvec3 bits=floatBitsToUint(value); Result.Words[start]=bits.x; Result.Words[start+1u]=bits.y; Result.Words[start+2u]=bits.z; }
float Height(vec3 position)
{
    vec3 object=(position-Frame.Bounds.xyz)*Frame.Bounds.w;
    vec2 chart=(Frame.Counts.z==ProjectionXY?object.xy:Frame.Counts.z==ProjectionXZ?object.xz:object.yz)*Frame.Shape.y;
    vec2 phase=2*Pi*fract(vec2(chart.x+chart.y,chart.x-chart.y));
    return MeanHeight+ModeAmplitude*(cos(phase.x)+cos(phase.y));
}
void main()
{
    uint vertex=gl_GlobalInvocationID.x; if (vertex>=Frame.Counts.x) return;
#if EVANS_DISPLACEMENT_PASS == 0
    for (uint word=0;word<VertexWords;++word) Result.Words[vertex*VertexWords+word]=Source.Words[vertex*VertexWords+word];
    if (Frame.Shape.x==0) return;
    vec3 point=SourcePosition(vertex);
    StoreVector(vertex*VertexWords,point+Directions.Values[vertex].xyz*((Height(point)-Frame.Shape.z)*Frame.Shape.x));
#else
    if (Frame.Shape.x==0) return;
    uint start=vertex*VertexWords+NormalWord;
    vec3 original=uintBitsToFloat(uvec3(Source.Words[start],Source.Words[start+1u],Source.Words[start+2u]));
    vec3 orientation=UnitOr(original,vec3(0)),sum=vec3(0); uvec2 range=Neighborhoods.Values[vertex].xy;
    for (uint neighbor=0;neighbor<range.y;++neighbor)
    {
        uint triangle=Adjacency.Values[range.x+neighbor];
        uint a=Indices.Values[triangle*3u],b=Indices.Values[triangle*3u+1u],c=Indices.Values[triangle*3u+2u];
        vec3 sourceFace=cross((SourcePosition(b)-SourcePosition(a))*Frame.Bounds.w,(SourcePosition(c)-SourcePosition(a))*Frame.Bounds.w);
        vec3 deformedFace=cross((ResultPosition(b)-ResultPosition(a))*Frame.Bounds.w,(ResultPosition(c)-ResultPosition(a))*Frame.Bounds.w);
        sum+=dot(sourceFace,orientation)<0?-deformedFace:deformedFace;
    }
    StoreVector(start,UnitOr(sum,original));
#endif
}
