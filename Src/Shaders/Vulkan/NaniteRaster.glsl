#version 450
#if defined(EVANS_NANITE_VISIBILITY_GEOMETRY)
// Vulkan's decoded vertex/index storage is consumed in place. The output is
// the same world-space ABI used by visibility projection and material resolve.
layout(local_size_x=64) in;
layout(std140,row_major,set=0,binding=0) uniform GeometryParameters { mat4 worldMatrix,normalMatrix; uvec4 geometryRange,dispatchInfo; };
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
void main()
{
    uint id=(gl_WorkGroupID.y*dispatchInfo.x+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
    if (id>=dispatchInfo.y) return;
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
}
#elif defined(EVANS_NANITE_GRAPHICS)
// Vulkan hardware and depth export stages use the same GPU clip table as
// compute. The original homogeneous corners determine material interpolation;
// a common snapped grid determines coverage along hardware/compute boundaries.
layout(std140,set=0,binding=0) uniform DrawFrame
{ uvec4 counts,layoutInfo,dispatchInfo,quality,range; vec4 material; uvec4 images; };
layout(std430,set=0,binding=1) readonly buffer WorldTable { uint worldWords[]; };
layout(std430,set=0,binding=2) readonly buffer ProjectedTable { uint projectedWords[]; };
layout(std430,set=0,binding=3) readonly buffer ResultTable { uint resultWords[]; };
layout(set=0,binding=4) uniform sampler2D diffuseMap;
layout(set=0,binding=5) uniform sampler2D opacityMap;
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
#if !defined(EVANS_NANITE_EXPORT)
layout(location=0) flat out uint rasterReference;
#endif
void main()
{
#if defined(EVANS_NANITE_EXPORT)
    uvec2 corner=uvec2((uint(gl_VertexIndex)<<1u)&2u,uint(gl_VertexIndex)&2u);
    gl_Position=vec4(vec2(corner)*2.0-1.0,0.0,1.0);
#else
    gl_Position=vec4(0.0); rasterReference=0u;
    uint local=uint(gl_VertexIndex)/CornerCount,id=range.x+local;
    if (local>=range.y || id>=counts.z) return;
    const uint PreserveMain=2u; if ((quality.w&PreserveMain)!=0u) id=range.x+range.y-1u-local;
    if (resultWords[layoutInfo.w+id]!=HardwareRoute) return;
    vec4 a=projectedCorner(id,0u),b=projectedCorner(id,1u),c=projectedCorner(id,2u);
    uint corner=uint(gl_VertexIndex)%CornerCount; vec4 clip=corner==0u?a:corner==1u?b:c;
    // Clip crossings must not change the grid on their shared inside edge.
    if (safeClip(clip)) clip=snappedPosition(clip);
    gl_Position=vec4(clip.x,-clip.y,clip.zw); rasterReference=id+1u;
#endif
}
#else
layout(location=0) out uint visibilityReference;
layout(location=1) out vec2 visibilityBary;
#if !defined(EVANS_NANITE_EXPORT)
layout(location=0) flat in uint rasterReference;
#endif
void main()
{
    vec2 pixel=gl_FragCoord.xy;
#if defined(EVANS_NANITE_EXPORT)
    const uint SoftwarePriorityBit=0x80000000u;
    uint address=uint(pixel.y)*counts.x+uint(pixel.x),owner=resultWords[counts.w+address],reference=(owner&~SoftwarePriorityBit)>>1u;
    gl_FragDepth=uintBitsToFloat(resultWords[address]); visibilityReference=reference|((owner&1u)!=0u?FaceBit:0u); visibilityBary=vec2(0.0);
    if (reference>0u && reference<=counts.z) { vec3 w,dx,dy; interpolate(reference-1u,pixel,w,dx,dy); visibilityBary=w.yz; }
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
    visibilityReference=rasterReference|(gl_FrontFacing?0u:FaceBit); visibilityBary=w.yz;
#endif
}
#endif
#elif defined(EVANS_NANITE_FRAME_PASS)
// Vulkan frame input producers. Explicit barriers publish storage writes to
// the raster kernels and hardware stages; the camera attachments stay on GPU.
layout(local_size_x=64) in;
layout(std140,set=0,binding=0,row_major) uniform BridgeFrame
{ uvec4 counts,layoutInfo,dispatchInfo,quality; mat4 camera; uvec4 range; };
layout(std430,set=0,binding=1) readonly buffer WorldStorage { uint worldWords[]; };
layout(std430,set=0,binding=2) buffer PackedInput { uint inputWords[]; };
const uint Threads=64u,ClipStride=16u,WorldStride=48u,VectorWords=4u,OcclusionRejectedFlag=8u;
#if EVANS_NANITE_FRAME_PASS == 4
layout(std430,set=0,binding=5) readonly buffer ClusterVisibility { uint decisions[]; };
#endif
#if EVANS_NANITE_FRAME_PASS == 5
layout(set=0,binding=3) uniform sampler2D cameraDepth;
layout(set=0,binding=4) uniform usampler2D cameraReference;
#endif
vec4 sourcePosition(uint word)
{ return uintBitsToFloat(uvec4(worldWords[word],worldWords[word+1u],worldWords[word+2u],worldWords[word+3u])); }
void storeClip(uint word,vec4 clip)
{ uvec4 bits=floatBitsToUint(clip); inputWords[word]=bits.x; inputWords[word+1u]=bits.y; inputWords[word+2u]=bits.z; inputWords[word+3u]=bits.w; }
void main()
{
    uint invocation=(gl_WorkGroupID.y*dispatchInfo.x+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
    if (invocation>=dispatchInfo.y) return;
#if EVANS_NANITE_FRAME_PASS == 4
    uint id=range.x+invocation,outputBase=layoutInfo.z+id*ClipStride;
    const uint FrustumRejectedState=1u,ClipFlagsWord=12u;
    if (range.w!=0u) if (decisions[range.w-1u]==FrustumRejectedState)
    { inputWords[outputBase+ClipFlagsWord]=range.z|OcclusionRejectedFlag; return; }
    for (uint corner=0u;corner<3u;++corner) storeClip(outputBase+corner*VectorWords,camera*sourcePosition(id*WorldStride+corner*VectorWords));
    uint flags=range.z;
    if (range.w!=0u && decisions[range.w-1u]!=quality.z) flags|=OcclusionRejectedFlag;
    inputWords[outputBase+12u]=flags; inputWords[outputBase+13u]=0u; inputWords[outputBase+14u]=0u; inputWords[outputBase+15u]=0u;
#elif EVANS_NANITE_FRAME_PASS == 5
    ivec2 pixel=ivec2(invocation%counts.x,invocation/counts.x);
    inputWords[layoutInfo.x+invocation]=floatBitsToUint(texelFetch(cameraDepth,pixel,0).r);
    inputWords[layoutInfo.y+invocation]=texelFetch(cameraReference,pixel,0).r;
#else
    inputWords[invocation]=0u;
#endif
}
#else
// Vulkan half-space micropolygon rasterizer. Separate depth/owner epochs use
// core uint atomics. The two raster epochs deliberately compile the same entry
// with a runtime phase, keeping depth arithmetic identical between dispatches.
// Nanite reference: Karis, Stubbe and Wihlidal, SIGGRAPH 2021, slides 83--86.
layout(local_size_x=64) in;
layout(std140,set=0,binding=0) uniform RasterFrame { uvec4 counts,layoutInfo,dispatchInfo,quality; };
layout(std430,set=0,binding=1) readonly buffer Inputs { uint inputWords[]; };
layout(std430,set=0,binding=2) buffer Outputs { uint outputWords[]; };
const uint Threads=64u,TriangleStride=16u,Reject=0u,Software=1u,Hardware=2u;
const uint PriorityOrdering=1u,PreserveMain=2u,SoftwarePriorityBit=0x80000000u;
const uint CullBack=1u,ClockwiseFront=2u,ForceHardware=4u,ReferenceMask=0x7fffffffu,Unowned=0xffffffffu;
const float MinimumNormal=1.175494351e-38;
vec4 clipVertex(uint address)
{ return uintBitsToFloat(uvec4(inputWords[address],inputWords[address+1u],inputWords[address+2u],inputWords[address+3u])); }
int edgeValue(ivec2 a,ivec2 b,ivec2 p)
{ ivec2 edge=b-a,delta=p-a; return edge.x*delta.y-edge.y*delta.x; }
bool insideEdge(int e,ivec2 a,ivec2 b)
{ ivec2 edge=b-a; return e>0 || (e==0 && (edge.y<0 || (edge.y==0 && edge.x>0))); }
struct TriangleSetup { ivec2 a,b,c,first,last; vec3 z; int area; uint key; };
uint setupTriangle(uint triangle,out TriangleSetup t)
{
    uint base=layoutInfo.z+triangle*TriangleStride;
    vec4 a=clipVertex(base),b=clipVertex(base+4u),c=clipVertex(base+8u); uint flags=inputWords[base+12u];
    const uint OcclusionRejectedFlag=8u;
    if ((flags&OcclusionRejectedFlag)!=0u) return Reject;
    if ((flags&ForceHardware)!=0u || any(isnan(a)) || any(isnan(b)) || any(isnan(c)) || any(isinf(a)) || any(isinf(b)) || any(isinf(c)) || min(a.w,min(b.w,c.w))<MinimumNormal) return Hardware;
    if (any(greaterThan(abs(a.xy),vec2(a.w))) || any(greaterThan(abs(b.xy),vec2(b.w))) || any(greaterThan(abs(c.xy),vec2(c.w))) ||
        a.z<0.0 || b.z<0.0 || c.z<0.0 || a.z>a.w || b.z>b.w || c.z>c.w) return Hardware;
    float scale=float(1u<<quality.y); vec2 viewport=vec2(counts.xy);
    t.a=ivec2(floor((a.xy/a.w*vec2(0.5,-0.5)+0.5)*viewport*scale+0.5));
    t.b=ivec2(floor((b.xy/b.w*vec2(0.5,-0.5)+0.5)*viewport*scale+0.5));
    t.c=ivec2(floor((c.xy/c.w*vec2(0.5,-0.5)+0.5)*viewport*scale+0.5));
    ivec2 lo=min(t.a,min(t.b,t.c)),hi=max(t.a,max(t.b,t.c));
    if (any(greaterThan(hi-lo,ivec2(int(quality.x)*int(scale))))) return Hardware;
    t.area=edgeValue(t.a,t.b,t.c); if (t.area==0) return Reject;
    bool back=(t.area>0)!=((flags&ClockwiseFront)!=0u);
    if (back && (flags&CullBack)!=0u) return Reject;
    t.key=((triangle+1u)<<1u)|(back?1u:0u); t.z=vec3(a.z/a.w,b.z/b.w,c.z/c.w);
    if (t.area<0) { ivec2 point=t.b; t.b=t.c; t.c=point; float z=t.z.y; t.z.y=t.z.z; t.z.z=z; t.area=-t.area; }
    t.first=max(ivec2(0),ivec2(ceil((vec2(lo)-scale*0.5)/scale)));
    t.last=min(ivec2(counts.xy)-1,ivec2(floor((vec2(hi)-scale*0.5)/scale)));
    return any(greaterThan(t.first,t.last))?Reject:Software;
}
void rasterTriangle(uint id)
{
    bool ownership=dispatchInfo.z==3u; TriangleSetup t; uint route=setupTriangle(id,t);
    if (!ownership) outputWords[layoutInfo.w+id]=route;
    if (dispatchInfo.z==4u || route!=Software) return;
    int scale=int(1u<<quality.y);
    for (int y=t.first.y;y<=t.last.y;++y) for (int x=t.first.x;x<=t.last.x;++x)
    {
        ivec2 samplePosition=ivec2(x,y)*scale+scale/2;
        int a=edgeValue(t.b,t.c,samplePosition),b=edgeValue(t.c,t.a,samplePosition),c=edgeValue(t.a,t.b,samplePosition);
        if (insideEdge(a,t.b,t.c) && insideEdge(b,t.c,t.a) && insideEdge(c,t.a,t.b))
        {
            precise float depth=t.z.x+(t.z.y-t.z.x)*(float(b)/float(t.area))+(t.z.z-t.z.x)*(float(c)/float(t.area));
            depth=clamp(depth,0.0,1.0); uint bits=depth<=0.0?0u:floatBitsToUint(depth),pixel=uint(y)*counts.x+uint(x);
            if ((quality.w&PriorityOrdering)!=0u?bits<floatBitsToUint(1.0):bits<inputWords[layoutInfo.x+pixel])
            {
                if (!ownership) atomicMin(outputWords[pixel],bits);
                else if (bits==outputWords[pixel]) atomicMin(outputWords[counts.w+pixel],t.key|((quality.w&PriorityOrdering)!=0u?SoftwarePriorityBit:0u));
            }
        }
    }
}
void main()
{
    uint id=(gl_WorkGroupID.y*dispatchInfo.x+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
    if (id>=dispatchInfo.y) return;
#if EVANS_NANITE_RASTER_PASS == 0
    uint depth=inputWords[layoutInfo.x+id];
    if ((quality.w&PreserveMain)!=0u)
    { uint previous=outputWords[id]; outputWords[layoutInfo.w+counts.z+id]=previous; depth=min(depth,previous); }
    outputWords[id]=depth;
#elif EVANS_NANITE_RASTER_PASS == 2
    uint encoded=inputWords[layoutInfo.y+id],key=((encoded&ReferenceMask)<<1u)|(encoded>>31u);
    uint minimum=outputWords[id],captured=inputWords[layoutInfo.x+id];
    if ((quality.w&PreserveMain)!=0u)
    {
        uint previous=outputWords[counts.w+id],previousDepth=outputWords[layoutInfo.w+counts.z+id];
        if (previousDepth==captured && (previous&~SoftwarePriorityBit)==key) key=previous;
        if (minimum!=captured) key=Unowned;
        if (previousDepth==minimum) key=min(key,previous);
    }
    else if (minimum!=captured) key=Unowned;
    outputWords[counts.w+id]=key;
#else
    rasterTriangle(id);
#endif
}
#endif
