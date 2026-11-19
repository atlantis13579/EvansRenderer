#version 450
#if defined(EVANS_HARDWARE_MESH)
#extension GL_EXT_mesh_shader : require
#endif
#if defined(EVANS_HARDWARE_GRAPHICS)
// Vulkan hardware and depth export stages use the same GPU clip table as
// compute. The original homogeneous corners determine material interpolation;
// a common snapped grid determines coverage along hardware/compute boundaries.
layout(std140,set=0,binding=0) uniform DrawFrame
{ uvec4 counts,layoutInfo,dispatchInfo,quality,range; vec4 material; uvec4 images,schedule; };
layout(std430,set=0,binding=1) readonly buffer WorldTable { uint worldWords[]; };
layout(std430,set=0,binding=2) readonly buffer ProjectedTable { uint projectedWords[]; };
layout(std430,set=0,binding=3) readonly buffer QueueTable { uint queueWords[]; };
layout(std430,set=0,binding=4) buffer OwnerTable { uint ownerWords[]; };
layout(set=0,binding=5) uniform sampler2D diffuseMap;
layout(set=0,binding=6) uniform sampler2D opacityMap;
const uint WorldStride=48u,ClipStride=16u,CornerCount=3u,HardwareRoute=2u,FaceBit=0x80000000u;
const uint VectorWords=4u,ColorWord=24u,UVWord=36u;
const float NormalMinimum=1.175494351e-38,InverseEpsilon=1.0e-12;
vec4 worldVector(uint id,uint offset)
{ uint at=id*WorldStride+offset; return uintBitsToFloat(uvec4(worldWords[at],worldWords[at+1u],worldWords[at+2u],worldWords[at+3u])); }
vec4 projectedCorner(uint id,uint corner)
{ uint at=layoutInfo.z+id*ClipStride+corner*VectorWords; return uintBitsToFloat(uvec4(projectedWords[at],projectedWords[at+1u],projectedWords[at+2u],projectedWords[at+3u])); }
bool safeClip(vec4 p)
{
    return !any(isnan(p)) && !any(isinf(p)) && p.w>=NormalMinimum &&
        all(lessThanEqual(abs(p.xy),vec2(p.w))) && p.z>=0.0 && p.z<=p.w;
}
vec4 snappedPosition(vec4 clip)
{
    float grid=float(1u<<quality.y); vec2 dimensions=vec2(counts.xy);
    vec2 pixel=floor((clip.xy/clip.w*vec2(0.5,-0.5)+0.5)*dimensions*grid+0.5)/grid;
    clip.xy=(pixel/dimensions*vec2(2.0,-2.0)+vec2(-1.0,1.0))*clip.w; return clip;
}
bool interpolate(uint id,vec2 pixel,out vec3 weights,out vec3 dx,out vec3 dy)
{
    weights=vec3(1.0,0.0,0.0); dx=dy=vec3(0.0);
    vec3 a=projectedCorner(id,0u).xyw,b=projectedCorner(id,1u).xyw,c=projectedCorner(id,2u).xyw;
    vec3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
    if (scale==0.0) return false; a/=scale; b/=scale; c/=scale;
    vec3 row0=cross(b,c),row1=cross(c,a),row2=cross(a,b); float determinant=dot(a,row0);
    if (abs(determinant)<=InverseEpsilon) return false; row0/=determinant; row1/=determinant; row2/=determinant;
    vec3 position=vec3(pixel/vec2(counts.xy)*vec2(2.0,-2.0)+vec2(-1.0,1.0),1.0);
    vec3 h=vec3(dot(row0,position),dot(row1,position),dot(row2,position)); float sum=h.x+h.y+h.z;
    if (abs(sum)<=InverseEpsilon) return false; weights=h/sum;
    vec3 hx=vec3(row0.x,row1.x,row2.x)*(2.0/float(counts.x)),hy=vec3(row0.y,row1.y,row2.y)*(-2.0/float(counts.y));
    dx=(hx-weights*(hx.x+hx.y+hx.z))/sum; dy=(hy-weights*(hy.x+hy.y+hy.z))/sum;
    return !any(isnan(weights)) && !any(isinf(weights)) && !any(isnan(dx)) && !any(isinf(dx)) && !any(isnan(dy)) && !any(isinf(dy));
}
#if defined(EVANS_VERTEX_STAGE)
#if !(EVANS_HARDWARE_PASS == 2)
layout(location=0) flat out uint rasterReference;
layout(location=1) flat out uint rasterPriority;
#endif
void main()
{
#if (EVANS_HARDWARE_PASS == 2)
    uvec2 corner=uvec2((uint(gl_VertexIndex)<<1u)&2u,uint(gl_VertexIndex)&2u);
    gl_Position=vec4(vec2(corner)*2.0-1.0,0.0,1.0);
#else
    gl_Position=vec4(0.0); rasterReference=rasterPriority=0u;
    uint priority=queueWords[queueWords[schedule.x]+uint(gl_VertexIndex)/CornerCount]; if (priority>=schedule.z) return;
    uint id=queueWords[schedule.y+priority]; if (id>=counts.z) return; rasterPriority=priority;
    vec4 a=projectedCorner(id,0u),b=projectedCorner(id,1u),c=projectedCorner(id,2u);
    uint corner=uint(gl_VertexIndex)%CornerCount; vec4 clip=corner==0u?a:corner==1u?b:c;
    // Clip crossings must not change the grid on their shared inside edge.
    if (safeClip(clip)) clip=snappedPosition(clip);
    gl_Position=vec4(clip.x,-clip.y,clip.zw); rasterReference=id+1u;
#endif
}
#else
#if EVANS_HARDWARE_PASS == 2
layout(location=0) out uint visibilityReference;
layout(location=1) out vec2 visibilityBary;
#elif EVANS_HARDWARE_PASS == 1
layout(early_fragment_tests) in;
#endif
#if !(EVANS_HARDWARE_PASS == 2)
#if defined(EVANS_HARDWARE_MESH)
layout(location=0) perprimitiveEXT flat in uint rasterReference;
layout(location=1) perprimitiveEXT flat in uint rasterPriority;
#else
layout(location=0) flat in uint rasterReference;
layout(location=1) flat in uint rasterPriority;
#endif
#endif
void main()
{
    vec2 pixel=gl_FragCoord.xy;
#if (EVANS_HARDWARE_PASS == 2)
    const uint EmptyOwner=0xffffffffu;
    uint address=uint(pixel.y)*counts.x+uint(pixel.x),owner=ownerWords[address],priority=owner>>1u;
    if (owner==EmptyOwner || priority==0u || priority>schedule.z) discard;
    uint safePriority=min(priority>0u?priority-1u:0u,max(schedule.z,1u)-1u);
    uint id=queueWords[schedule.y+safePriority]; if (id>=counts.z) discard;
    id=min(id,max(counts.z,1u)-1u);
    vec3 w,dx,dy; if (!interpolate(id,pixel,w,dx,dy)) discard;
    visibilityReference=(id+1u)|((owner&1u)!=0u?FaceBit:0u); visibilityBary=w.yz;
#else
    if (rasterReference==0u || rasterReference>counts.z || gl_FragCoord.z>=1.0) discard;
    uint id=min(rasterReference>0u?rasterReference-1u:0u,max(counts.z,1u)-1u); vec3 w,dx,dy;
    if (!interpolate(id,pixel,w,dx,dy)) discard;
    vec4 uv01=worldVector(id,UVWord),uv2=worldVector(id,UVWord+VectorWords);
    vec2 a=vec2(uv01.x,1.0-uv01.y),b=vec2(uv01.z,1.0-uv01.w),c=vec2(uv2.x,1.0-uv2.y);
    vec2 uv=a*w.x+b*w.y+c*w.z,uvDx=(b-a)*dx.y+(c-a)*dx.z,uvDy=(b-a)*dy.y+(c-a)*dy.z;
    vec3 alpha=vec3(worldVector(id,ColorWord).a,worldVector(id,ColorWord+VectorWords).a,worldVector(id,ColorWord+2u*VectorWords).a);
    float coverage=dot(alpha,w)*material.x;
    if (images.x!=0u) coverage*=textureGrad(diffuseMap,uv,uvDx,uvDy).a;
    if (images.y!=0u) coverage*=textureGrad(opacityMap,uv,uvDx,uvDy).r;
    if (material.y>0.0 && clamp(coverage,0.0,1.0)<material.y) discard;
#if EVANS_HARDWARE_PASS == 1
    uint address=uint(pixel.y)*counts.x+uint(pixel.x);
    atomicMin(ownerWords[address],((rasterPriority+1u)<<1u)|(gl_FrontFacing?0u:1u));
#endif
#endif
}
#endif
#else
// Native Vulkan queue construction. Clear/count/prefix/scatter use four
// storage bindings; the clear pass aliases range storage with pixel owners.
#if EVANS_HARDWARE_COMPUTE == 2
layout(local_size_x=1) in;
#else
layout(local_size_x=64) in;
#endif
layout(std140,set=0,binding=0) uniform ScheduleFrame { uvec4 counts,offsets,images; };
#if EVANS_HARDWARE_COMPUTE == 0
layout(std430,set=0,binding=1) buffer OwnerStorage { uint owners[]; };
#else
layout(std430,set=0,binding=1) readonly buffer RangeStorage { uvec4 ranges[]; };
#endif
layout(std430,set=0,binding=2) readonly buffer RouteStorage { uint routes[]; };
layout(std430,set=0,binding=3) buffer ArgumentStorage { uint arguments[]; };
layout(std430,set=0,binding=4) buffer QueueStorage { uint queue[]; };
const uint Threads=64u,BinWords=8u,OffsetWord=4u,CursorWord=5u,CountWord=6u,HardwareRoute=2u,Corners=3u,EmptyOwner=0xffffffffu;
#if EVANS_HARDWARE_COMPUTE == 1 || EVANS_HARDWARE_COMPUTE == 3
uvec2 sourceFor(uint priority)
{
    uint low=0u,high=counts.y;
    while (low<high) { uint middle=(low+high)/2u; uvec4 item=ranges[middle]; if (item.w+item.y<=priority) low=middle+1u; else high=middle; }
    uvec4 item=ranges[low]; return uvec2(item.x+priority-item.w,item.z);
}
#endif
void main()
{
    uint id=(gl_WorkGroupID.y*offsets.w+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
#if EVANS_HARDWARE_COMPUTE == 0
    if (id<counts.w) owners[id]=EmptyOwner;
    if (id<counts.z) for (uint word=0u;word<BinWords;++word) arguments[id*BinWords+word]=0u;
#elif EVANS_HARDWARE_COMPUTE == 1
    if (id>=counts.x) return; uvec2 source=sourceFor(id); if (routes[offsets.x+source.x]!=HardwareRoute) return;
    atomicAdd(arguments[source.y*BinWords],1u);
#elif EVANS_HARDWARE_COMPUTE == 2
    uint start=0u;
    for (uint bin=0u;bin<counts.z;++bin)
    {
        uint base=bin*BinWords,count=arguments[base]; arguments[base]=count*Corners; arguments[base+1u]=1u;
        arguments[base+OffsetWord]=start; arguments[base+CountWord]=count; queue[bin]=offsets.z+start; start+=count;
    }
#else
    if (id>=counts.x) return; uvec2 source=sourceFor(id); if (routes[offsets.x+source.x]!=HardwareRoute) return;
    uint base=source.y*BinWords,slot=atomicAdd(arguments[base+CursorWord],1u);
    queue[offsets.y+id]=source.x; queue[offsets.z+arguments[base+OffsetWord]+slot]=id;
#endif
}
#endif
