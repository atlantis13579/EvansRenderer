#version 310 es
#if defined(EVANS_NANITE_VISIBILITY_GEOMETRY)
precision highp float;
precision highp int;
// ES assembly reads the decoded streams directly; no CPU-expanded triangle
// table is uploaded. Matrices retain the row-major renderer ABI.
layout(local_size_x=64) in;
layout(std140,row_major,binding=0) uniform AssemblyFrame { mat4 world,normal; uvec4 range,work; };
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
void main()
{
    uint local=(gl_WorkGroupID.y*work.x+gl_WorkGroupID.x)*GroupWidth+gl_LocalInvocationIndex;
    if (local>=work.y) return;
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
}
#elif defined(EVANS_NANITE_GRAPHICS)
precision highp float;
precision highp int;
// Independent ES hardware/export stages. GPU route and projected-corner buffers
// are shared with compute; only final visibility attachments feed materials.
layout(std140,binding=0) uniform DrawParameters { uvec4 counts,offsets,dispatchInfo,settings,range; vec4 material; uvec4 images; };
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
layout(std430,binding=1) readonly buffer ClipTable { uint clipWords[]; };
layout(std430,binding=2) readonly buffer WinnerTable { uint winnerWords[]; };
uniform highp sampler2D RasterDiffuse;
uniform highp sampler2D RasterOpacity;
const uint ClipStride=16u,GeometryStride=48u,Corners=3u,HardwareRoute=2u,BackFace=0x80000000u;
const uint ColorOffset=24u,UVOffset=36u,VectorWords=4u;
const float SmallestNormal=1.175494351e-38,InverseTolerance=1.0e-12;
vec4 geometryVector(uint id,uint word)
{ uint at=id*GeometryStride+word; return uintBitsToFloat(uvec4(readGeometryWord(at),readGeometryWord(at+1u),readGeometryWord(at+2u),readGeometryWord(at+3u))); }
vec4 cornerClip(uint id,uint corner)
{ uint at=offsets.z+id*ClipStride+corner*VectorWords; return uintBitsToFloat(uvec4(clipWords[at],clipWords[at+1u],clipWords[at+2u],clipWords[at+3u])); }
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
#if !defined(EVANS_NANITE_EXPORT)
flat out uint reference;
// Passing original corners avoids simultaneous vertex/fragment storage reads
// of one allocation on ES translation layers, while retaining analytic weights.
flat out vec4 primitiveA,primitiveB,primitiveC,primitiveUVAB;
flat out vec2 primitiveUVC;
flat out vec3 primitiveAlpha;
#endif
void main()
{
#if defined(EVANS_NANITE_EXPORT)
    uvec2 corner=uvec2((uint(gl_VertexID)<<1u)&2u,uint(gl_VertexID)&2u); gl_Position=vec4(vec2(corner)*2.0-1.0,0.0,1.0);
#else
    gl_Position=vec4(0.0); reference=0u;
    primitiveA=primitiveB=primitiveC=primitiveUVAB=vec4(0.0); primitiveUVC=vec2(0.0); primitiveAlpha=vec3(0.0);
    uint local=uint(gl_VertexID)/Corners,id=range.x+local;
    if (local>=range.y || id>=counts.z) return;
    const uint PreserveMain=2u; if ((settings.w&PreserveMain)!=0u) id=range.x+range.y-1u-local;
    if (winnerWords[offsets.w+id]!=HardwareRoute) return;
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
layout(location=0) out uint outputReference;
layout(location=1) out vec2 outputBary;
#if !defined(EVANS_NANITE_EXPORT)
flat in uint reference;
flat in vec4 primitiveA,primitiveB,primitiveC,primitiveUVAB;
flat in vec2 primitiveUVC;
flat in vec3 primitiveAlpha;
#endif
void main()
{
    vec2 pixel=vec2(gl_FragCoord.x,float(counts.y)-gl_FragCoord.y);
#if defined(EVANS_NANITE_EXPORT)
    const uint SoftwarePriorityBit=0x80000000u;
    uint at=uint(pixel.y)*counts.x+uint(pixel.x),key=winnerWords[counts.w+at],ref=(key&~SoftwarePriorityBit)>>1u;
    gl_FragDepth=uintBitsToFloat(winnerWords[at]); outputReference=ref|((key&1u)!=0u?BackFace:0u); outputBary=vec2(0.0);
    if (ref>0u && ref<=counts.z) { vec3 w,dx,dy; weightsFromClip(cornerClip(ref-1u,0u),cornerClip(ref-1u,1u),cornerClip(ref-1u,2u),pixel,w,dx,dy); outputBary=w.yz; }
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
    outputReference=reference|(gl_FrontFacing?0u:BackFace); outputBary=w.yz;
#endif
}
#endif
#elif defined(EVANS_NANITE_FRAME_PASS)
precision highp float;
precision highp int;
layout(local_size_x=64) in;
layout(std140,binding=0) uniform FrameInput { uvec4 counts,offsets,dispatchInfo,settings; vec4 cameraRows[4]; uvec4 range; };
layout(std430,binding=0) readonly buffer GeometryStorage { uint geometryWords[]; };
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
layout(std430,binding=1) buffer InputStorage { uint inputWords[]; };
const uint Threads=64u,ClipWords=16u,WorldWords=48u,OcclusionRejectedFlag=8u;
#if EVANS_NANITE_FRAME_PASS == 4
layout(std430,binding=2) readonly buffer ClusterVisibility { uint decisions[]; };
#endif
#if EVANS_NANITE_FRAME_PASS == 5
uniform highp sampler2D RasterDepth;
uniform highp usampler2D RasterVisibility;
#endif
vec4 readWorldPosition(uint word)
{ return uintBitsToFloat(uvec4(readGeometryWord(word),readGeometryWord(word+1u),readGeometryWord(word+2u),readGeometryWord(word+3u))); }
void writeVector(uint word,vec4 value)
{ uvec4 bits=floatBitsToUint(value); inputWords[word]=bits.x; inputWords[word+1u]=bits.y; inputWords[word+2u]=bits.z; inputWords[word+3u]=bits.w; }
void main()
{
    uint lane=(gl_WorkGroupID.y*dispatchInfo.x+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
    if (lane>=dispatchInfo.y) return;
#if EVANS_NANITE_FRAME_PASS == 4
    uint id=range.x+lane,base=offsets.z+id*ClipWords;
    const uint FrustumRejectedState=1u,ClipFlagsWord=12u;
    if (range.w!=0u) if (decisions[range.w-1u]==FrustumRejectedState)
    { inputWords[base+ClipFlagsWord]=range.z|OcclusionRejectedFlag; return; }
    for (uint corner=0u;corner<3u;++corner)
    {
        vec4 p=readWorldPosition(id*WorldWords+corner*4u);
        writeVector(base+corner*4u,vec4(dot(cameraRows[0],p),dot(cameraRows[1],p),dot(cameraRows[2],p),dot(cameraRows[3],p)));
    }
    uint flags=range.z;
    if (range.w!=0u && decisions[range.w-1u]!=settings.z) flags|=OcclusionRejectedFlag;
    inputWords[base+12u]=flags; inputWords[base+13u]=0u; inputWords[base+14u]=0u; inputWords[base+15u]=0u;
#elif EVANS_NANITE_FRAME_PASS == 5
    ivec2 pixel=ivec2(lane%counts.x,counts.y-1u-lane/counts.x);
    inputWords[offsets.x+lane]=floatBitsToUint(texelFetch(RasterDepth,pixel,0).r);
    inputWords[offsets.y+lane]=texelFetch(RasterVisibility,pixel,0).r;
#else
    inputWords[lane]=0u;
#endif
}
#else
// OpenGL ES compute rasterization uses buffer atomics and top-left integer
// coverage. Screen coordinates are top-left even though GL images are not;
// the hardware bridge supplies canonical seed buffers. Nanite, SIGGRAPH 2021.
// Both raster epochs use this same binary and a runtime ownership flag. This
// keeps depth bits repeatable without requiring the ES 3.2 precise qualifier.
precision highp float;
precision highp int;
layout(local_size_x=64) in;
layout(std140,binding=0) uniform Parameters { uvec4 counts,offsets,dispatchInfo,settings; };
layout(std430,binding=0) readonly buffer Source { uint sourceWords[]; };
layout(std430,binding=1) buffer Destination { uint destinationWords[]; };
const uint LaneCount=64u,PrimitiveWords=16u,Rejected=0u,ComputeRoute=1u,HardwareRoute=2u;
const uint PriorityOrdering=1u,PreserveMain=2u,SoftwarePriorityBit=0x80000000u;
const uint CullFlag=1u,ClockwiseFlag=2u,ForceHardware=4u,ReferenceMask=0x7fffffffu,NoOwner=0xffffffffu;
const float NormalMinimum=1.175494351e-38;
vec4 readClip(uint at)
{ return uintBitsToFloat(uvec4(sourceWords[at],sourceWords[at+1u],sourceWords[at+2u],sourceWords[at+3u])); }
int halfSpace(ivec2 a,ivec2 b,ivec2 position)
{ ivec2 direction=b-a,relative=position-a; return direction.x*relative.y-direction.y*relative.x; }
bool accepts(int e,ivec2 a,ivec2 b)
{
    ivec2 d=b-a;
    // ES framebuffer Y inversion changes horizontal edge ownership. Match
    // native hardware when a snapped edge passes exactly through sample centers.
    bool horizontal=d.x<0;
    return e>0 || (e==0 && (d.y<0 || (d.y==0 && horizontal)));
}
struct Primitive { ivec2 p0,p1,p2,minimum,maximum; vec3 depths; int area; uint owner; };
uint preparePrimitive(uint index,out Primitive p)
{
    uint at=offsets.z+index*PrimitiveWords;
    vec4 a=readClip(at),b=readClip(at+4u),c=readClip(at+8u); uint flags=sourceWords[at+12u];
    const uint OcclusionRejectedFlag=8u;
    if ((flags&OcclusionRejectedFlag)!=0u) return Rejected;
    if ((flags&ForceHardware)!=0u || any(isnan(a)) || any(isnan(b)) || any(isnan(c)) || any(isinf(a)) || any(isinf(b)) || any(isinf(c)) || a.w<NormalMinimum || b.w<NormalMinimum || c.w<NormalMinimum) return HardwareRoute;
    if (any(greaterThan(abs(a.xy),vec2(a.w))) || any(greaterThan(abs(b.xy),vec2(b.w))) || any(greaterThan(abs(c.xy),vec2(c.w))) ||
        a.z<0.0 || a.z>a.w || b.z<0.0 || b.z>b.w || c.z<0.0 || c.z>c.w) return HardwareRoute;
    float grid=float(1u<<settings.y); vec2 size=vec2(counts.xy);
    p.p0=ivec2(floor((a.xy/a.w*vec2(0.5,-0.5)+0.5)*size*grid+0.5));
    p.p1=ivec2(floor((b.xy/b.w*vec2(0.5,-0.5)+0.5)*size*grid+0.5));
    p.p2=ivec2(floor((c.xy/c.w*vec2(0.5,-0.5)+0.5)*size*grid+0.5));
    ivec2 low=min(p.p0,min(p.p1,p.p2)),high=max(p.p0,max(p.p1,p.p2));
    if (any(greaterThan(high-low,ivec2(int(settings.x)*int(grid))))) return HardwareRoute;
    p.area=halfSpace(p.p0,p.p1,p.p2); if (p.area==0) return Rejected;
    bool back=(p.area>0)!=((flags&ClockwiseFlag)!=0u);
    if (back && (flags&CullFlag)!=0u) return Rejected;
    p.owner=((index+1u)<<1u)|uint(back); p.depths=vec3(a.z/a.w,b.z/b.w,c.z/c.w);
    if (p.area<0)
    { ivec2 point=p.p1; p.p1=p.p2; p.p2=point; float value=p.depths.y; p.depths.y=p.depths.z; p.depths.z=value; p.area=-p.area; }
    p.minimum=max(ivec2(0),ivec2(ceil((vec2(low)-grid*0.5)/grid)));
    p.maximum=min(ivec2(counts.xy)-1,ivec2(floor((vec2(high)-grid*0.5)/grid)));
    return any(greaterThan(p.minimum,p.maximum))?Rejected:ComputeRoute;
}
void rasterPrimitive(uint index)
{
    bool ownerPass=dispatchInfo.z==3u; Primitive p; uint route=preparePrimitive(index,p);
    if (!ownerPass) destinationWords[offsets.w+index]=route;
    if (dispatchInfo.z==4u || route!=ComputeRoute) return;
    int grid=int(1u<<settings.y);
    for (int y=p.minimum.y;y<=p.maximum.y;++y) for (int x=p.minimum.x;x<=p.maximum.x;++x)
    {
        ivec2 position=ivec2(x,y)*grid+grid/2;
        int a=halfSpace(p.p1,p.p2,position),b=halfSpace(p.p2,p.p0,position),c=halfSpace(p.p0,p.p1,position);
        if (accepts(a,p.p1,p.p2) && accepts(b,p.p2,p.p0) && accepts(c,p.p0,p.p1))
        {
            float z=p.depths.x+(p.depths.y-p.depths.x)*(float(b)/float(p.area))+(p.depths.z-p.depths.x)*(float(c)/float(p.area));
            z=clamp(z,0.0,1.0); uint depth=z<=0.0?0u:floatBitsToUint(z),pixel=uint(y)*counts.x+uint(x);
            if ((settings.w&PriorityOrdering)!=0u?depth<floatBitsToUint(1.0):depth<sourceWords[offsets.x+pixel])
            {
                if (!ownerPass) atomicMin(destinationWords[pixel],depth);
                else if (depth==destinationWords[pixel]) atomicMin(destinationWords[counts.w+pixel],p.owner|((settings.w&PriorityOrdering)!=0u?SoftwarePriorityBit:0u));
            }
        }
    }
}
void main()
{
    uint index=(gl_WorkGroupID.y*dispatchInfo.x+gl_WorkGroupID.x)*LaneCount+gl_LocalInvocationIndex;
    if (index>=dispatchInfo.y) return;
#if EVANS_NANITE_RASTER_PASS == 0
    uint depth=sourceWords[offsets.x+index];
    if ((settings.w&PreserveMain)!=0u)
    { uint previous=destinationWords[index]; destinationWords[offsets.w+counts.z+index]=previous; depth=min(depth,previous); }
    destinationWords[index]=depth;
#elif EVANS_NANITE_RASTER_PASS == 2
    uint reference=sourceWords[offsets.y+index],owner=((reference&ReferenceMask)<<1u)|(reference>>31u);
    uint minimum=destinationWords[index],captured=sourceWords[offsets.x+index];
    if ((settings.w&PreserveMain)!=0u)
    {
        uint previous=destinationWords[counts.w+index],previousDepth=destinationWords[offsets.w+counts.z+index];
        if (previousDepth==captured && (previous&~SoftwarePriorityBit)==owner) owner=previous;
        if (minimum!=captured) owner=NoOwner;
        if (previousDepth==minimum) owner=min(owner,previous);
    }
    else if (minimum!=captured) owner=NoOwner;
    destinationWords[counts.w+index]=owner;
#else
    rasterPrimitive(index);
#endif
}
#endif
