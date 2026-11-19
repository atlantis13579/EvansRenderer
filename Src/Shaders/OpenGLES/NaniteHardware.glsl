#version 310 es
#if defined(EVANS_HARDWARE_GRAPHICS)
precision highp float;
precision highp int;
// Independent ES hardware/export stages. GPU route and projected-corner buffers
// are shared with compute; only final visibility attachments feed materials.
layout(std140,binding=0) uniform DrawParameters { uvec4 counts,offsets,dispatchInfo,settings,range; vec4 material; uvec4 images,schedule; };
#if defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS != 2
layout(std430,binding=0) readonly buffer GeometryTable { uint geometryWords[]; };
// Separate blocks retain global addresses beyond the per-block device limit.
#if defined(EVANS_VISIBILITY_BANK_WORDS)
layout(std430,binding=4) readonly buffer GeometryOverflow { uint overflowWords[]; };
#endif
uint readGeometryWord(uint word)
{
#if defined(EVANS_VISIBILITY_BANK_WORDS)
    const uint BankWords=EVANS_VISIBILITY_BANK_WORDS;
    if (word>=BankWords) return overflowWords[word-BankWords];
#endif
    return geometryWords[word];
}
#endif
#if (defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS != 2) || (!defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS == 2)
layout(std430,binding=1) readonly buffer ClipTable { uint clipWords[]; };
#endif
uniform highp sampler2D HardwareOwners;
uniform highp usampler2D HardwareQueue;
uint readQueue(uint word) { return texelFetch(HardwareQueue,ivec2(word%schedule.w,word/schedule.w),0).r; }
uniform highp sampler2D RasterDiffuse;
uniform highp sampler2D RasterOpacity;
const uint ClipStride=16u,GeometryStride=48u,Corners=3u,HardwareRoute=2u,BackFace=0x80000000u;
const uint ColorOffset=24u,UVOffset=36u,VectorWords=4u;
const float SmallestNormal=1.175494351e-38,InverseTolerance=1.0e-12;
#if defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS != 2
vec4 geometryVector(uint id,uint word)
{ uint at=id*GeometryStride+word; return uintBitsToFloat(uvec4(readGeometryWord(at),readGeometryWord(at+1u),readGeometryWord(at+2u),readGeometryWord(at+3u))); }
#endif
#if (defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS != 2) || (!defined(EVANS_VERTEX_STAGE) && EVANS_HARDWARE_PASS == 2)
vec4 cornerClip(uint id,uint corner)
{ uint at=offsets.z+id*ClipStride+corner*VectorWords; return uintBitsToFloat(uvec4(clipWords[at],clipWords[at+1u],clipWords[at+2u],clipWords[at+3u])); }
#endif
bool finiteClip(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool inClip(vec4 p)
{ return finiteClip(p) && p.w>=SmallestNormal && all(lessThanEqual(abs(p.xy),vec2(p.w))) && p.z>=0.0 && p.z<=p.w; }
vec4 snapCoverage(vec4 p)
{
    float scale=float(1u<<settings.y); vec2 size=vec2(counts.xy);
    vec2 pixel=floor((p.xy/p.w*vec2(0.5,-0.5)+0.5)*size*scale+0.5)/scale;
    p.xy=(pixel/size*vec2(2.0,-2.0)+vec2(-1.0,1.0))*p.w; return p;
}
bool weightsFromClip(vec4 clipA,vec4 clipB,vec4 clipC,vec2 pixel,out vec3 weights,out vec3 dx,out vec3 dy)
{
    weights=vec3(1.0,0.0,0.0); dx=dy=vec3(0.0);
    vec3 a=clipA.xyw,b=clipB.xyw,c=clipC.xyw;
    vec3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
    if (scale==0.0) return false; a/=scale; b/=scale; c/=scale;
    vec3 row0=cross(b,c),row1=cross(c,a),row2=cross(a,b); float determinant=dot(a,row0);
    if (abs(determinant)<=InverseTolerance) return false; row0/=determinant; row1/=determinant; row2/=determinant;
    vec3 position=vec3(pixel/vec2(counts.xy)*vec2(2.0,-2.0)+vec2(-1.0,1.0),1.0);
    vec3 h=vec3(dot(row0,position),dot(row1,position),dot(row2,position)); float sum=h.x+h.y+h.z;
    if (abs(sum)<=InverseTolerance) return false; weights=h/sum;
    vec3 hx=vec3(row0.x,row1.x,row2.x)*(2.0/float(counts.x)),hy=vec3(row0.y,row1.y,row2.y)*(-2.0/float(counts.y));
    dx=(hx-weights*(hx.x+hx.y+hx.z))/sum; dy=(hy-weights*(hy.x+hy.y+hy.z))/sum;
    return !any(isnan(weights)) && !any(isinf(weights)) && !any(isnan(dx)) && !any(isinf(dx)) && !any(isnan(dy)) && !any(isinf(dy));
}
#if defined(EVANS_VERTEX_STAGE)
#if !(EVANS_HARDWARE_PASS == 2)
flat out uint reference,priority;
// Passing original corners avoids simultaneous vertex/fragment storage reads
// of one allocation on ES translation layers, while retaining analytic weights.
flat out vec4 primitiveA,primitiveB,primitiveC,primitiveUVAB;
flat out vec2 primitiveUVC;
flat out vec3 primitiveAlpha;
#endif
void main()
{
#if (EVANS_HARDWARE_PASS == 2)
    uvec2 corner=uvec2((uint(gl_VertexID)<<1u)&2u,uint(gl_VertexID)&2u); gl_Position=vec4(vec2(corner)*2.0-1.0,0.0,1.0);
#else
    gl_Position=vec4(0.0); reference=priority=0u;
    primitiveA=primitiveB=primitiveC=primitiveUVAB=vec4(0.0); primitiveUVC=vec2(0.0); primitiveAlpha=vec3(0.0);
    priority=readQueue(readQueue(schedule.x)+uint(gl_VertexID)/Corners); if (priority>=schedule.z) return;
    uint id=readQueue(schedule.y+priority); if (id>=counts.z) return;
    vec4 a=cornerClip(id,0u),b=cornerClip(id,1u),c=cornerClip(id,2u);
    primitiveA=a; primitiveB=b; primitiveC=c;
    primitiveUVAB=geometryVector(id,UVOffset); primitiveUVC=geometryVector(id,UVOffset+VectorWords).xy;
    primitiveAlpha=vec3(geometryVector(id,ColorOffset).a,geometryVector(id,ColorOffset+VectorWords).a,geometryVector(id,ColorOffset+2u*VectorWords).a);
    uint corner=uint(gl_VertexID)%Corners; vec4 clip=corner==0u?a:corner==1u?b:c;
    // Shared inside corners retain the compute grid across clipping.
    if (inClip(clip)) clip=snapCoverage(clip);
    gl_Position=vec4(clip.xy,clip.z*2.0-clip.w,clip.w); reference=id+1u;
#endif
}
#else
#if EVANS_HARDWARE_PASS == 2
layout(location=0) out uint outputReference;
layout(location=1) out vec2 outputBary;
#elif EVANS_HARDWARE_PASS == 1
// Keys stay below 2^24, so float MIN blending preserves exact integer order.
layout(location=0) out highp float ownershipKey;
#endif
#if !(EVANS_HARDWARE_PASS == 2)
flat in uint reference,priority;
flat in vec4 primitiveA,primitiveB,primitiveC,primitiveUVAB;
flat in vec2 primitiveUVC;
flat in vec3 primitiveAlpha;
#endif
void main()
{
    vec2 pixel=vec2(gl_FragCoord.x,float(counts.y)-gl_FragCoord.y);
#if (EVANS_HARDWARE_PASS == 2)
    const uint EmptyOwner=1u<<24u;
    uint key=uint(texelFetch(HardwareOwners,ivec2(gl_FragCoord.xy),0).r),rank=key>>1u;
    if (key==EmptyOwner || rank==0u || rank>schedule.z) discard;
    uint safePriority=min(rank>0u?rank-1u:0u,max(schedule.z,1u)-1u);
    uint id=readQueue(schedule.y+safePriority); if (id>=counts.z) discard;
    id=min(id,max(counts.z,1u)-1u);
    vec3 w,dx,dy; if (!weightsFromClip(cornerClip(id,0u),cornerClip(id,1u),cornerClip(id,2u),pixel,w,dx,dy)) discard;
    outputReference=(id+1u)|((key&1u)!=0u?BackFace:0u); outputBary=w.yz;
#else
    if (reference==0u || reference>counts.z || gl_FragCoord.z>=1.0) discard;
    vec3 w,dx,dy;
    if (!weightsFromClip(primitiveA,primitiveB,primitiveC,pixel,w,dx,dy)) discard;
    vec4 uv01=primitiveUVAB; vec2 uv2=primitiveUVC;
    vec2 a=vec2(uv01.x,1.0-uv01.y),b=vec2(uv01.z,1.0-uv01.w),c=vec2(uv2.x,1.0-uv2.y);
    vec2 uv=a*w.x+b*w.y+c*w.z,uvDx=(b-a)*dx.y+(c-a)*dx.z,uvDy=(b-a)*dy.y+(c-a)*dy.z;
    vec3 alpha=primitiveAlpha;
    float coverage=dot(alpha,w)*material.x;
    if (images.x!=0u) coverage*=textureGrad(RasterDiffuse,uv,uvDx,uvDy).a;
    if (images.y!=0u) coverage*=textureGrad(RasterOpacity,uv,uvDx,uvDy).r;
    if (material.y>0.0 && clamp(coverage,0.0,1.0)<material.y) discard;
#if EVANS_HARDWARE_PASS == 1
    ownershipKey=float(((priority+1u)<<1u)|(gl_FrontFacing?0u:1u));
#endif
#endif
}
#endif
#else
precision highp float;
precision highp int;
#if EVANS_HARDWARE_COMPUTE == 2
layout(local_size_x=1) in;
#else
layout(local_size_x=64) in;
#endif
layout(std140,binding=0) uniform HardwareParameters { uvec4 counts,layoutInfo,images; };
layout(std430,binding=0) readonly buffer Ranges { uvec4 ranges[]; };
layout(std430,binding=1) readonly buffer Routes { uint routes[]; };
layout(std430,binding=2) buffer Arguments { uint arguments[]; };
layout(binding=0,r32ui) writeonly uniform highp uimage2D queueImage;
const uint GroupWidth=64u,BinStride=8u,StartWord=4u,CursorWord=5u,CountWord=6u,HardwareRoute=2u,Corners=3u;
void storeQueue(uint word,uint value) { imageStore(queueImage,ivec2(word%images.z,word/images.z),uvec4(value,0u,0u,0u)); }
#if EVANS_HARDWARE_COMPUTE == 1 || EVANS_HARDWARE_COMPUTE == 3
uvec2 candidate(uint priority)
{
    uint first=0u,last=counts.y;
    while (first<last) { uint middle=(first+last)/2u; uvec4 record=ranges[middle]; if (record.w+record.y>priority) last=middle; else first=middle+1u; }
    uvec4 record=ranges[first]; return uvec2(record.x+priority-record.w,record.z);
}
#endif
void main()
{
    uint id=(gl_WorkGroupID.y*layoutInfo.w+gl_WorkGroupID.x)*GroupWidth+gl_LocalInvocationIndex;
#if EVANS_HARDWARE_COMPUTE == 0
    if (id<counts.z) for (uint offset=0u;offset<BinStride;++offset) arguments[id*BinStride+offset]=0u;
#elif EVANS_HARDWARE_COMPUTE == 1
    if (id>=counts.x) return; uvec2 item=candidate(id); if (routes[layoutInfo.x+item.x]!=HardwareRoute) return;
    atomicAdd(arguments[item.y*BinStride],1u);
#elif EVANS_HARDWARE_COMPUTE == 2
    uint first=0u;
    for (uint bin=0u;bin<counts.z;++bin)
    {
        uint header=bin*BinStride,count=arguments[header]; arguments[header]=count*Corners; arguments[header+1u]=1u;
        arguments[header+StartWord]=first; arguments[header+CountWord]=count; storeQueue(bin,layoutInfo.z+first); first+=count;
    }
#else
    if (id>=counts.x) return; uvec2 item=candidate(id); if (routes[layoutInfo.x+item.x]!=HardwareRoute) return;
    uint header=item.y*BinStride,slot=atomicAdd(arguments[header+CursorWord],1u);
    storeQueue(layoutInfo.y+id,item.x); storeQueue(layoutInfo.z+arguments[header+StartWord]+slot,id);
#endif
}
#endif
