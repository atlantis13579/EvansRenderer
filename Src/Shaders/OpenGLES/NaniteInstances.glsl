#version 310 es
// One GPU schedule per decoded stream family; stable IDs survive compaction.
precision highp float;
precision highp int;
// ES assembly reads the decoded streams directly; no CPU-expanded triangle
// table is uploaded. Matrices retain the row-major renderer ABI.
#if EVANS_INSTANCE_PASS == 1
layout(local_size_x=1) in;
#else
layout(local_size_x=64) in;
#endif
layout(std140,binding=0) uniform InstanceFrame { uvec4 schedule; };
layout(std430,binding=3) buffer InstanceWords { uint instances[]; };
layout(std430,binding=0) readonly buffer Vertices { uint vertexWords[]; };
layout(std430,binding=1) readonly buffer Indices { uint indices[]; };
struct WorldTriangle { vec4 position[3],normal[3],color[3],uvAB,uvC; uvec4 source; };
layout(std430,binding=2) writeonly buffer Triangles { uint triangleWords[]; };
#if defined(EVANS_VISIBILITY_BANK_WORDS)
layout(std430,binding=4) writeonly buffer TriangleOverflow { uint overflowWords[]; };
#endif
const uint GroupWidth=64u,VertexWords=12u,Corners=3u,NormalWord=3u,ColorWord=6u,UVWord=10u;
const uint TriangleWords=48u,VectorWords=4u,TableNormal=12u,TableColor=24u,TableUV=36u,TableSource=44u;
const float NormalLimit=1.0e-6,SRGBThreshold=0.04045,SRGBSlope=12.92,SRGBBias=0.055,SRGBScale=1.055,SRGBPower=2.4;
vec3 readThree(uint at) { return uintBitsToFloat(uvec3(vertexWords[at],vertexWords[at+1u],vertexWords[at+2u])); }
void writeVector(uint at,uvec4 value)
{
#if defined(EVANS_VISIBILITY_BANK_WORDS)
    // Bank boundaries are vector-aligned, so one vector never straddles blocks.
    if (at>=EVANS_VISIBILITY_BANK_WORDS)
    {
        at-=EVANS_VISIBILITY_BANK_WORDS;
        overflowWords[at]=value.x; overflowWords[at+1u]=value.y; overflowWords[at+2u]=value.z; overflowWords[at+3u]=value.w;
        return;
    }
#endif
    triangleWords[at]=value.x; triangleWords[at+1u]=value.y; triangleWords[at+2u]=value.z; triangleWords[at+3u]=value.w;
}
bool finiteVector(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 stableNormal(vec3 value)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    vec3 unit=scale>0.0?value/scale:vec3(0.0,1.0,0.0); float magnitude=length(unit);
    return scale*magnitude>NormalLimit?unit/magnitude:vec3(0.0,1.0,0.0);
}
float linearChannel(float value)
{ value=clamp(value,0.0,1.0); return value<=SRGBThreshold?value/SRGBSlope:pow((value+SRGBBias)/SRGBScale,SRGBPower); }
const uint MatrixRowWords=4u,NormalMatrixWord=16u,TotalGroupWord=3u;
const uint RecordStride=44u,RangeWord=32u,DispatchWord=36u,QueryWord=40u,MaximumColumns=64u;
const uint VisibleState=2u,DeferredState=3u;
uvec4 loadInstance(uint at) { return uvec4(instances[at],instances[at+1u],instances[at+2u],instances[at+3u]); }
void main()
{
#if EVANS_INSTANCE_PASS == 0
    uint id=gl_GlobalInvocationID.x; if (id>=schedule.x) return;
    uint at=schedule.z+id*RecordStride; uvec4 query=loadInstance(at+QueryWord); bool includeSource=query.z!=0u;
    if (includeSource && schedule.w!=0u)
    {
        includeSource=false;
        for (uint i=0u;i<query.y;++i) { uint state=vertexWords[query.x+i]; includeSource=includeSource || state==VisibleState || state==DeferredState; }
    }
    uint triangles=instances[at+RangeWord+1u]; instances[schedule.y+id]=includeSource?(triangles+GroupWidth-1u)/GroupWidth:0u;
#elif EVANS_INSTANCE_PASS == 1
    uint end=0u;
    for (uint i=0u;i<schedule.x;++i) { end+=instances[schedule.y+i]; instances[schedule.y+i]=end; }
    uint columns=max(1u,min(end,MaximumColumns));
    instances[0]=columns; instances[1]=max(1u,(end+columns-1u)/columns); instances[2]=1u; instances[TotalGroupWord]=end;
#else
    uint group=gl_WorkGroupID.y*instances[0]+gl_WorkGroupID.x; if (group>=instances[TotalGroupWord]) return;
    uint begin=0u,end=schedule.x;
    while (begin<end) { uint middle=(begin+end)/2u; if (instances[schedule.y+middle]>group) end=middle; else begin=middle+1u; }
    uint previous=0u; if (begin!=0u) previous=instances[schedule.y+begin-1u];
    uint local=(group-previous)*GroupWidth+gl_LocalInvocationIndex,record=schedule.z+begin*RecordStride;
    uvec4 range=loadInstance(record+RangeWord),work=loadInstance(record+DispatchWord); if (local>=range.y) return;
    mat4 world=transpose(mat4(uintBitsToFloat(loadInstance(record)),uintBitsToFloat(loadInstance(record+1u*MatrixRowWords)),uintBitsToFloat(loadInstance(record+2u*MatrixRowWords)),uintBitsToFloat(loadInstance(record+3u*MatrixRowWords))));
    mat4 normal=transpose(mat4(uintBitsToFloat(loadInstance(record+NormalMatrixWord)),uintBitsToFloat(loadInstance(record+NormalMatrixWord+1u*MatrixRowWords)),uintBitsToFloat(loadInstance(record+NormalMatrixWord+2u*MatrixRowWords)),uintBitsToFloat(loadInstance(record+NormalMatrixWord+3u*MatrixRowWords))));
    uint first=range.z+local*Corners; uvec3 corner=uvec3(indices[first],indices[first+1u],indices[first+2u]);
    bool valid=all(lessThan(corner,uvec3(range.w))); WorldTriangle result; vec2 uv[3];
    for (uint c=0u;c<Corners;++c)
    {
        result.position[c]=vec4(0.0,0.0,0.0,1.0); result.normal[c]=result.color[c]=vec4(0.0); uv[c]=vec2(0.0);
    }
    if (valid) for (uint c=0u;c<Corners;++c)
    {
        uint at=corner[c]*VertexWords; vec3 p=readThree(at),n=readThree(at+NormalWord);
        vec4 color=vec4(readThree(at+ColorWord),uintBitsToFloat(vertexWords[at+ColorWord+3u]));
        uv[c]=uintBitsToFloat(uvec2(vertexWords[at+UVWord],vertexWords[at+UVWord+1u]));
        result.position[c]=world*vec4(p,1.0); vec3 transformed=(normal*vec4(n,0.0)).xyz;
        valid=valid && finiteVector(result.position[c]) && finiteVector(vec4(transformed,0.0)) && finiteVector(color) && finiteVector(vec4(uv[c],0.0,0.0));
        result.normal[c]=vec4(stableNormal(transformed),0.0);
        result.color[c]=vec4(linearChannel(color.r),linearChannel(color.g),linearChannel(color.b),color.a);
    }
    if (!valid) for (uint c=0u;c<Corners;++c)
    { result.position[c]=vec4(0.0,0.0,0.0,1.0); result.normal[c]=result.color[c]=vec4(0.0); uv[c]=vec2(0.0); }
    result.uvAB=vec4(uv[0],uv[1]); result.uvC=vec4(uv[2],0.0,0.0); result.source=uvec4(work.z,local,0u,0u);
    // Explicit vector stores avoid array-bearing struct copies in ES translators.
    uint destination=(range.x+local)*TriangleWords;
    for (uint c=0u;c<Corners;++c)
    {
        writeVector(destination+c*VectorWords,floatBitsToUint(result.position[c]));
        writeVector(destination+TableNormal+c*VectorWords,floatBitsToUint(result.normal[c]));
        writeVector(destination+TableColor+c*VectorWords,floatBitsToUint(result.color[c]));
    }
    writeVector(destination+TableUV,floatBitsToUint(result.uvAB)); writeVector(destination+TableUV+VectorWords,floatBitsToUint(result.uvC));
    writeVector(destination+TableSource,result.source);
#endif
}
