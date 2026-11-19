#version 450
// GPU instance schedule: count eligible work, scan, then assemble indirectly.
// Vulkan's decoded vertex/index storage is consumed in place. The output is
// the same world-space ABI used by visibility projection and material resolve.
#if EVANS_INSTANCE_PASS == 1
layout(local_size_x=1) in;
#else
layout(local_size_x=64) in;
#endif
layout(std140,set=0,binding=0) uniform InstanceParameters { uvec4 schedule; };
layout(std430,set=0,binding=4) buffer InstanceStorage { uint instanceWords[]; };
layout(std430,set=0,binding=1) readonly buffer VertexStorage { uint vertices[]; };
layout(std430,set=0,binding=2) readonly buffer IndexStorage { uint indices[]; };
layout(std430,set=0,binding=3) writeonly buffer TriangleStorage { uint triangles[]; };
const uint Threads=64u,VertexStride=12u,TriangleStride=48u,VectorStride=4u,CornerCount=3u;
const uint VertexNormal=3u,VertexColor=6u,VertexUV=10u,TableNormal=12u,TableColor=24u,TableUV=36u,TableSource=44u;
const float NormalThreshold=1.0e-6,SRGBBreak=0.04045,SRGBDivisor=12.92,SRGBOffset=0.055,SRGBRange=1.055,SRGBExponent=2.4;
vec4 loadAttribute(uint at)
{ return uintBitsToFloat(uvec4(vertices[at],vertices[at+1u],vertices[at+2u],vertices[at+3u])); }
void storeVector(uint at,uvec4 value)
{ triangles[at]=value.x; triangles[at+1u]=value.y; triangles[at+2u]=value.z; triangles[at+3u]=value.w; }
bool finiteAttribute(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
float decodeSRGB(float value)
{ value=clamp(value,0.0,1.0); return value<=SRGBBreak?value/SRGBDivisor:pow((value+SRGBOffset)/SRGBRange,SRGBExponent); }
vec3 sourceNormal(vec3 n)
{
    float largest=max(abs(n.x),max(abs(n.y),abs(n.z))); vec3 scaled=largest>0.0?n/largest:vec3(0.0,1.0,0.0);
    float magnitude=length(scaled); return largest*magnitude>NormalThreshold?scaled/magnitude:vec3(0.0,1.0,0.0);
}
const uint MatrixRowWords=4u,NormalMatrixWord=16u,TotalGroupWord=3u;
const uint RecordWords=44u,RangeOffset=32u,DispatchOffset=36u,QueryOffset=40u,DispatchColumns=64u;
const uint MainVisible=2u,Deferred=3u;
uvec4 instanceVector(uint at) { return uvec4(instanceWords[at],instanceWords[at+1u],instanceWords[at+2u],instanceWords[at+3u]); }
void main()
{
#if EVANS_INSTANCE_PASS == 0
    uint source=gl_GlobalInvocationID.x; if (source>=schedule.x) return;
    uint record=schedule.z+source*RecordWords; uvec4 query=instanceVector(record+QueryOffset);
    bool eligible=query.z!=0u;
    if (eligible && schedule.w!=0u)
    {
        eligible=false;
        for (uint i=0u;i<query.y;++i) { uint state=vertices[query.x+i]; eligible=eligible || state==MainVisible || state==Deferred; }
    }
    uint count=instanceWords[record+RangeOffset+1u]; instanceWords[schedule.y+source]=eligible?(count+Threads-1u)/Threads:0u;
#elif EVANS_INSTANCE_PASS == 1
    uint total=0u;
    for (uint i=0u;i<schedule.x;++i) { total+=instanceWords[schedule.y+i]; instanceWords[schedule.y+i]=total; }
    uint columns=max(1u,min(total,DispatchColumns));
    instanceWords[0]=columns; instanceWords[1]=max(1u,(total+columns-1u)/columns); instanceWords[2]=1u; instanceWords[TotalGroupWord]=total;
#else
    uint workgroup=gl_WorkGroupID.y*instanceWords[0]+gl_WorkGroupID.x;
    if (workgroup>=instanceWords[TotalGroupWord]) return;
    uint low=0u,high=schedule.x;
    while (low<high) { uint middle=(low+high)/2u; if (instanceWords[schedule.y+middle]<=workgroup) low=middle+1u; else high=middle; }
    uint start=0u; if (low>0u) start=instanceWords[schedule.y+low-1u];
    uint id=(workgroup-start)*Threads+gl_LocalInvocationIndex,record=schedule.z+low*RecordWords;
    uvec4 geometryRange=instanceVector(record+RangeOffset),dispatchInfo=instanceVector(record+DispatchOffset);
    if (id>=geometryRange.y) return;
    mat4 worldMatrix=transpose(mat4(uintBitsToFloat(instanceVector(record)),uintBitsToFloat(instanceVector(record+1u*MatrixRowWords)),uintBitsToFloat(instanceVector(record+2u*MatrixRowWords)),uintBitsToFloat(instanceVector(record+3u*MatrixRowWords))));
    mat4 normalMatrix=transpose(mat4(uintBitsToFloat(instanceVector(record+NormalMatrixWord)),uintBitsToFloat(instanceVector(record+NormalMatrixWord+1u*MatrixRowWords)),uintBitsToFloat(instanceVector(record+NormalMatrixWord+2u*MatrixRowWords)),uintBitsToFloat(instanceVector(record+NormalMatrixWord+3u*MatrixRowWords))));
    uint base=geometryRange.z+id*CornerCount;
    uvec3 refs=uvec3(indices[base],indices[base+1u],indices[base+2u]); bool valid=all(lessThan(refs,uvec3(geometryRange.w)));
    vec4 position[3],normal[3],color[3]; vec2 uv[3];
    for (uint c=0u;c<CornerCount;++c) { position[c]=vec4(0.0,0.0,0.0,1.0); normal[c]=color[c]=vec4(0.0); uv[c]=vec2(0.0); }
    if (valid) for (uint c=0u;c<CornerCount;++c)
    {
        uint at=refs[c]*VertexStride; position[c]=worldMatrix*vec4(loadAttribute(at).xyz,1.0);
        vec3 transformed=(normalMatrix*vec4(loadAttribute(at+VertexNormal).xyz,0.0)).xyz; vec4 authored=loadAttribute(at+VertexColor);
        uv[c]=uintBitsToFloat(uvec2(vertices[at+VertexUV],vertices[at+VertexUV+1u]));
        valid=valid && finiteAttribute(position[c]) && finiteAttribute(vec4(transformed,0.0)) && finiteAttribute(authored) && finiteAttribute(vec4(uv[c],0.0,0.0));
        normal[c]=vec4(sourceNormal(transformed),0.0); color[c]=vec4(decodeSRGB(authored.r),decodeSRGB(authored.g),decodeSRGB(authored.b),authored.a);
    }
    uint destination=(geometryRange.x+id)*TriangleStride;
    for (uint c=0u;c<CornerCount;++c)
    {
        if (!valid) { position[c]=vec4(0.0,0.0,0.0,1.0); normal[c]=color[c]=vec4(0.0); uv[c]=vec2(0.0); }
        storeVector(destination+c*VectorStride,floatBitsToUint(position[c]));
        storeVector(destination+TableNormal+c*VectorStride,floatBitsToUint(normal[c]));
        storeVector(destination+TableColor+c*VectorStride,floatBitsToUint(color[c]));
    }
    storeVector(destination+TableUV,floatBitsToUint(vec4(uv[0],uv[1])));
    storeVector(destination+TableUV+VectorStride,floatBitsToUint(vec4(uv[2],0.0,0.0)));
    storeVector(destination+TableSource,uvec4(dispatchInfo.z,id,0u,0u));
#endif
}
