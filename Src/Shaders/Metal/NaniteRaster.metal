#include <metal_stdlib>
using namespace metal;
// Metal micropolygon visibility. Tracked buffer dependencies between encoders
// separate minimum-depth and minimum-owner epochs. Relaxed uint atomics are
// sufficient within each epoch; no threadgroup/tile lock serializes triangles.
// Karis et al., A Deep Dive into Nanite Virtualized Geometry, SIGGRAPH 2021.
struct RasterFrame { uint4 counts,layout,dispatch,quality; };
constant uint Threads=64,TriangleWords=16,Rejected=0,ComputeRoute=1,HardwareRoute=2;
constant uint PriorityOrdering=1,PreserveMain=2,SoftwarePriorityBit=0x80000000u;
constant uint CullBack=1,FrontClockwise=2,ForceHardware=4,OcclusionRejected=8,ReferenceMask=0x7fffffffu,NoOwner=0xffffffffu;
constant float MinimumNormal=1.175494351e-38f;
float4 readVertex(device const uint* words,uint base)
{ return as_type<float4>(uint4(words[base],words[base+1],words[base+2],words[base+3])); }
int edgeFunction(int2 from,int2 to,int2 pixel)
{ int2 direction=to-from,relative=pixel-from; return direction.x*relative.y-direction.y*relative.x; }
bool edgeCovers(int edge,int2 from,int2 to)
{ int2 direction=to-from; return edge>0 || (edge==0 && (direction.y<0 || (direction.y==0 && direction.x>0))); }
struct RasterPrimitive { int2 a,b,c,first,last; float3 depth; int area; uint owner; };
uint prepare(uint triangle,constant RasterFrame& frame,device const uint* words,thread RasterPrimitive& p)
{
    uint base=frame.layout.z+triangle*TriangleWords;
    float4 a=readVertex(words,base),b=readVertex(words,base+4),c=readVertex(words,base+8); uint flags=words[base+12];
    if ((flags&OcclusionRejected)!=0) return Rejected;
    if ((flags&ForceHardware)!=0 || !all(isfinite(a)) || !all(isfinite(b)) || !all(isfinite(c)) || min(a.w,min(b.w,c.w))<MinimumNormal) return HardwareRoute;
    if (any(abs(a.xy)>a.w) || any(abs(b.xy)>b.w) || any(abs(c.xy)>c.w) ||
        a.z<0 || a.z>a.w || b.z<0 || b.z>b.w || c.z<0 || c.z>c.w) return HardwareRoute;
    float grid=float(1u<<frame.quality.y); float2 size=float2(frame.counts.xy);
    p.a=int2(floor((a.xy/a.w*float2(0.5f,-0.5f)+0.5f)*size*grid+0.5f));
    p.b=int2(floor((b.xy/b.w*float2(0.5f,-0.5f)+0.5f)*size*grid+0.5f));
    p.c=int2(floor((c.xy/c.w*float2(0.5f,-0.5f)+0.5f)*size*grid+0.5f));
    int2 low=min(p.a,min(p.b,p.c)),high=max(p.a,max(p.b,p.c));
    if (any(high-low>int(frame.quality.x)*int(grid))) return HardwareRoute;
    p.area=edgeFunction(p.a,p.b,p.c); if (p.area==0) return Rejected;
    bool back=(p.area>0)!=((flags&FrontClockwise)!=0);
    if (back && (flags&CullBack)!=0) return Rejected;
    p.owner=((triangle+1u)<<1u)|uint(back); p.depth=float3(a.z/a.w,b.z/b.w,c.z/c.w);
    if (p.area<0)
    { int2 point=p.b; p.b=p.c; p.c=point; float z=p.depth.y; p.depth.y=p.depth.z; p.depth.z=z; p.area=-p.area; }
    p.first=max(int2(0),int2(ceil((float2(low)-grid*0.5f)/grid)));
    p.last=min(int2(frame.counts.xy)-1,int2(floor((float2(high)-grid*0.5f)/grid)));
    return any(p.first>p.last)?Rejected:ComputeRoute;
}
kernel void EvansNaniteSeedDepth(constant RasterFrame& frame [[buffer(0)]],device const uint* input [[buffer(1)]],
    device atomic_uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint pixel=(group.y*frame.dispatch.x+group.x)*Threads+lane; if (pixel>=frame.dispatch.y) return;
    uint depth=input[frame.layout.x+pixel];
    if ((frame.quality.w&PreserveMain)!=0)
    {
        uint previous=atomic_load_explicit(output+pixel,memory_order_relaxed);
        atomic_store_explicit(output+frame.layout.w+frame.counts.z+pixel,previous,memory_order_relaxed); depth=min(depth,previous);
    }
    atomic_store_explicit(output+pixel,depth,memory_order_relaxed);
}
kernel void EvansNaniteSeedOwner(constant RasterFrame& frame [[buffer(0)]],device const uint* input [[buffer(1)]],
    device atomic_uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint pixel=(group.y*frame.dispatch.x+group.x)*Threads+lane; if (pixel>=frame.dispatch.y) return;
    uint encoded=input[frame.layout.y+pixel],owner=((encoded&ReferenceMask)<<1u)|(encoded>>31u);
    uint minimum=atomic_load_explicit(output+pixel,memory_order_relaxed),captured=input[frame.layout.x+pixel];
    if ((frame.quality.w&PreserveMain)!=0)
    {
        uint previous=atomic_load_explicit(output+frame.counts.w+pixel,memory_order_relaxed);
        uint previousDepth=atomic_load_explicit(output+frame.layout.w+frame.counts.z+pixel,memory_order_relaxed);
        if (previousDepth==captured && (previous&~SoftwarePriorityBit)==owner) owner=previous;
        if (minimum!=captured) owner=NoOwner;
        if (previousDepth==minimum) owner=min(owner,previous);
    }
    else if (minimum!=captured) owner=NoOwner;
    atomic_store_explicit(output+frame.counts.w+pixel,owner,memory_order_relaxed);
}
kernel void EvansNaniteRaster(constant RasterFrame& frame [[buffer(0)]],device const uint* input [[buffer(1)]],
    device atomic_uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint triangle=(group.y*frame.dispatch.x+group.x)*Threads+lane; if (triangle>=frame.dispatch.y) return;
    // One entry point for both epochs prevents independently optimized depth
    // expressions from disagreeing in their low bits. Compile with fast math off.
    bool ownership=frame.dispatch.z==3; RasterPrimitive p; uint route=prepare(triangle,frame,input,p);
    if (!ownership) atomic_store_explicit(output+frame.layout.w+triangle,route,memory_order_relaxed);
    if (frame.dispatch.z==4 || route!=ComputeRoute) return;
    int grid=int(1u<<frame.quality.y);
    for (int y=p.first.y;y<=p.last.y;++y) for (int x=p.first.x;x<=p.last.x;++x)
    {
        int2 sample=int2(x,y)*grid+grid/2;
        int a=edgeFunction(p.b,p.c,sample),b=edgeFunction(p.c,p.a,sample),c=edgeFunction(p.a,p.b,sample);
        if (edgeCovers(a,p.b,p.c) && edgeCovers(b,p.c,p.a) && edgeCovers(c,p.a,p.b))
        {
            float z=p.depth.x+(p.depth.y-p.depth.x)*(float(b)/float(p.area))+(p.depth.z-p.depth.x)*(float(c)/float(p.area));
            z=clamp(z,0.0f,1.0f); uint depth=z<=0?0u:as_type<uint>(z),pixel=uint(y)*frame.counts.x+uint(x);
            if ((frame.quality.w&PriorityOrdering)!=0?depth<as_type<uint>(1.0f):depth<input[frame.layout.x+pixel])
            {
                if (!ownership) atomic_fetch_min_explicit(output+pixel,depth,memory_order_relaxed);
                else if (depth==atomic_load_explicit(output+pixel,memory_order_relaxed))
                    atomic_fetch_min_explicit(output+frame.counts.w+pixel,p.owner|((frame.quality.w&PriorityOrdering)!=0?SoftwarePriorityBit:0u),memory_order_relaxed);
            }
        }
    }
}

// Native frame producers and graphics stages share these GPU resources. The
// command encoders form explicit producer/consumer boundaries; no host readback
// participates in hardware/compute selection or material reconstruction.
struct FrameInput { RasterFrame Raster; float4 CameraRows[4]; uint4 Range; };
struct DrawParameters { RasterFrame Raster; uint4 Range; float4 Material; uint4 Images; };
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
// Native decoded streams are expanded once on the GPU for coverage and shading.
// Explicit matrix rows preserve the C++ ABI without transposing the source data.
struct GeometryAssembly { float4 World[4],Normal[4]; uint4 Range,Dispatch; };
constant uint AssemblyVertexWords=12,AssemblyCorners=3,AssemblyNormalWord=3,AssemblyColorWord=6,AssemblyUVWord=10;
constant float AssemblyNormalLimit=1.0e-6f,AssemblySRGBBreak=0.04045f,AssemblySRGBSlope=12.92f;
constant float AssemblySRGBBias=0.055f,AssemblySRGBScale=1.055f,AssemblySRGBPower=2.4f;
float assemblyLinear(float value)
{ value=clamp(value,0.0f,1.0f); return value<=AssemblySRGBBreak?value/AssemblySRGBSlope:pow((value+AssemblySRGBBias)/AssemblySRGBScale,AssemblySRGBPower); }
float3 assemblyNormal(float3 n)
{
    float scale=max(abs(n.x),max(abs(n.y),abs(n.z))); float3 unit=scale>0?n/scale:float3(0,1,0);
    float magnitude=length(unit); return scale*magnitude>AssemblyNormalLimit?unit/magnitude:float3(0,1,0);
}
kernel void EvansNaniteVisibilityGeometry(constant GeometryAssembly& frame [[buffer(0)]],device const uint* vertices [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device WorldTriangle* output [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint local=(group.y*frame.Dispatch.x+group.x)*Threads+lane; if (local>=frame.Dispatch.y) return;
    uint first=frame.Range.z+local*AssemblyCorners;
    uint3 corners(indices[first],indices[first+1],indices[first+2]); bool valid=all(corners<frame.Range.w);
    WorldTriangle result={}; float2 uv[3]={};
    if (valid) for (uint c=0;c<AssemblyCorners;++c)
    {
        uint at=corners[c]*AssemblyVertexWords;
        float4 p(readVertex(vertices,at).xyz,1),n(readVertex(vertices,at+AssemblyNormalWord).xyz,0);
        float4 color=readVertex(vertices,at+AssemblyColorWord);
        uv[c]=as_type<float2>(uint2(vertices[at+AssemblyUVWord],vertices[at+AssemblyUVWord+1]));
        result.Position[c]=float4(dot(frame.World[0],p),dot(frame.World[1],p),dot(frame.World[2],p),1);
        float3 transformed(dot(frame.Normal[0],n),dot(frame.Normal[1],n),dot(frame.Normal[2],n));
        valid=valid && all(isfinite(result.Position[c])) && all(isfinite(transformed)) && all(isfinite(color)) && all(isfinite(uv[c]));
        result.Normal[c]=float4(assemblyNormal(transformed),0);
        result.Color[c]=float4(assemblyLinear(color.r),assemblyLinear(color.g),assemblyLinear(color.b),color.a);
    }
    if (!valid) { result=WorldTriangle{}; for (uint c=0;c<AssemblyCorners;++c) uv[c]=0; }
    for (uint c=0;c<AssemblyCorners;++c) result.Position[c].w=1;
    result.UVAB=float4(uv[0],uv[1]); result.UVC=float4(uv[2],0,0); result.Source=uint4(frame.Dispatch.z,local,0,0);
    output[frame.Range.x+local]=result;
}
float4 cameraPosition(float4 p,constant FrameInput& frame)
{ return float4(dot(frame.CameraRows[0],p),dot(frame.CameraRows[1],p),dot(frame.CameraRows[2],p),dot(frame.CameraRows[3],p)); }
void writeClip(device uint* words,uint base,float4 value)
{ uint4 bits=as_type<uint4>(value); words[base]=bits.x; words[base+1]=bits.y; words[base+2]=bits.z; words[base+3]=bits.w; }
kernel void EvansNaniteClearInputs(constant FrameInput& frame [[buffer(0)]],device uint* output [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint word=(group.y*frame.Raster.dispatch.x+group.x)*Threads+lane;
    if (word<frame.Raster.dispatch.y) output[word]=0;
}
kernel void EvansNaniteProjectInputs(constant FrameInput& frame [[buffer(0)]],device const WorldTriangle* geometry [[buffer(1)]],
    device uint* output [[buffer(2)]],device const uint* decisions [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint local=(group.y*frame.Raster.dispatch.x+group.x)*Threads+lane; if (local>=frame.Raster.dispatch.y) return;
    uint id=frame.Range.x+local,base=frame.Raster.layout.z+id*TriangleWords;
    constexpr uint FrustumRejectedState=1,ClipFlagsWord=12;
    if (frame.Range.w!=0) if (decisions[frame.Range.w-1]==FrustumRejectedState)
    { output[base+ClipFlagsWord]=frame.Range.z|OcclusionRejected; return; }
    for (uint corner=0;corner<3;++corner) writeClip(output,base+corner*4,cameraPosition(geometry[id].Position[corner],frame));
    uint flags=frame.Range.z;
    if (frame.Range.w!=0 && decisions[frame.Range.w-1]!=frame.Raster.quality.z) flags|=OcclusionRejected;
    output[base+12]=flags; output[base+13]=output[base+14]=output[base+15]=0;
}
kernel void EvansNaniteCaptureInputs(constant FrameInput& frame [[buffer(0)]],device uint* output [[buffer(2)]],
    depth2d<float,access::read> depth [[texture(0)]],texture2d<uint,access::read> reference [[texture(1)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint pixel=(group.y*frame.Raster.dispatch.x+group.x)*Threads+lane; if (pixel>=frame.Raster.dispatch.y) return;
    uint2 location(pixel%frame.Raster.counts.x,pixel/frame.Raster.counts.x);
    output[frame.Raster.layout.x+pixel]=as_type<uint>(depth.read(location)); output[frame.Raster.layout.y+pixel]=reference.read(location).x;
}
float4 projectedCorner(device const uint* input,uint id,uint corner,constant DrawParameters& p)
{ return readVertex(input,p.Raster.layout.z+id*TriangleWords+corner*4); }
bool safeCoverageCorner(float4 p)
{ return all(isfinite(p)) && p.w>=MinimumNormal && all(abs(p.xy)<=p.w) && p.z>=0 && p.z<=p.w; }
float4 snapCoverage(float4 clip,constant DrawParameters& p)
{
    float grid=float(1u<<p.Raster.quality.y); float2 size=float2(p.Raster.counts.xy);
    float2 pixel=floor((clip.xy/clip.w*float2(0.5f,-0.5f)+0.5f)*size*grid+0.5f)/grid;
    clip.xy=(pixel/size*float2(2,-2)+float2(-1,1))*clip.w; return clip;
}
bool materialWeights(uint id,float2 pixel,constant DrawParameters& p,device const uint* input,
    thread float3& weights,thread float3& dx,thread float3& dy)
{
    const float InverseTolerance=1.0e-12f;
    weights=float3(1,0,0); dx=dy=float3(0);
    float3 a=projectedCorner(input,id,0,p).xyw,b=projectedCorner(input,id,1,p).xyw,c=projectedCorner(input,id,2,p).xyw;
    float3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
    if (scale==0) return false; a/=scale; b/=scale; c/=scale;
    float3 r0=cross(b,c),r1=cross(c,a),r2=cross(a,b); float determinant=dot(a,r0);
    if (abs(determinant)<=InverseTolerance) return false; r0/=determinant; r1/=determinant; r2/=determinant;
    float3 position(pixel/float2(p.Raster.counts.xy)*float2(2,-2)+float2(-1,1),1);
    float3 h(dot(r0,position),dot(r1,position),dot(r2,position)); float sum=h.x+h.y+h.z;
    if (abs(sum)<=InverseTolerance) return false; weights=h/sum;
    float3 hx=float3(r0.x,r1.x,r2.x)*(2.0f/p.Raster.counts.x),hy=float3(r0.y,r1.y,r2.y)*(-2.0f/p.Raster.counts.y);
    dx=(hx-weights*(hx.x+hx.y+hx.z))/sum; dy=(hy-weights*(hy.x+hy.y+hy.z))/sum;
    return all(isfinite(weights)) && all(isfinite(dx)) && all(isfinite(dy));
}
struct HardwareVertex { float4 Position [[position]]; uint Reference [[flat]]; };
vertex HardwareVertex EvansNaniteHardwareVertex(uint vertex [[vertex_id]],constant DrawParameters& p [[buffer(0)]],
    device const uint* input [[buffer(2)]],device const uint* result [[buffer(3)]])
{
    HardwareVertex output{}; const uint Corners=3; uint local=vertex/Corners,id=p.Range.x+local;
    if (local>=p.Range.y || id>=p.Raster.counts.z) return output;
    if ((p.Raster.quality.w&PreserveMain)!=0) id=p.Range.x+p.Range.y-1u-local;
    if (result[p.Raster.layout.w+id]!=HardwareRoute) return output;
    float4 a=projectedCorner(input,id,0,p),b=projectedCorner(input,id,1,p),c=projectedCorner(input,id,2,p);
    uint corner=vertex%Corners; output.Position=corner==0?a:corner==1?b:c;
    // Preserve the common edge grid beside hardware-clipped triangles.
    if (safeCoverageCorner(output.Position)) output.Position=snapCoverage(output.Position,p);
    output.Reference=id+1; return output;
}
struct RasterPayload { uint Reference [[color(0)]]; float2 Bary [[color(1)]]; };
fragment RasterPayload EvansNaniteHardwareFragment(HardwareVertex vertex [[stage_in]],bool front [[front_facing]],
    constant DrawParameters& p [[buffer(0)]],device const WorldTriangle* geometry [[buffer(1)]],device const uint* input [[buffer(2)]],
    texture2d<float> diffuse [[texture(0)]],texture2d<float> opacity [[texture(1)]],sampler surfaceSampler [[sampler(0)]])
{
    const uint FaceBit=0x80000000u;
    if (vertex.Reference==0 || vertex.Reference>p.Raster.counts.z || vertex.Position.z>=1) discard_fragment();
    uint id=min(vertex.Reference>0?vertex.Reference-1u:0u,max(p.Raster.counts.z,1u)-1u); float3 w,dx,dy;
    if (!materialWeights(id,vertex.Position.xy,p,input,w,dx,dy)) discard_fragment();
    WorldTriangle item=geometry[id]; float2 a(item.UVAB.x,1-item.UVAB.y),b(item.UVAB.z,1-item.UVAB.w),c(item.UVC.x,1-item.UVC.y);
    float2 uv=a*w.x+b*w.y+c*w.z,uvDx=(b-a)*dx.y+(c-a)*dx.z,uvDy=(b-a)*dy.y+(c-a)*dy.z;
    float coverage=dot(float3(item.Color[0].a,item.Color[1].a,item.Color[2].a),w)*p.Material.x;
    if (p.Images.x) coverage*=diffuse.sample(surfaceSampler,uv,gradient2d(uvDx,uvDy)).a;
    if (p.Images.y) coverage*=opacity.sample(surfaceSampler,uv,gradient2d(uvDx,uvDy)).r;
    if (p.Material.y>0 && saturate(coverage)<p.Material.y) discard_fragment();
    return {vertex.Reference|(front?0u:FaceBit),w.yz};
}
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex EvansNaniteExportVertex(uint vertex [[vertex_id]])
{ uint2 p((vertex<<1u)&2u,vertex&2u); return {float4(float2(p)*float2(2,-2)+float2(-1,1),0,1)}; }
struct ExportPayload { uint Reference [[color(0)]]; float2 Bary [[color(1)]]; float Depth [[depth(any)]]; };
fragment ExportPayload EvansNaniteExportFragment(ScreenVertex vertex [[stage_in]],constant DrawParameters& p [[buffer(0)]],
    device const uint* input [[buffer(2)]],device const uint* result [[buffer(3)]])
{
    const uint FaceBit=0x80000000u;
    uint pixel=uint(vertex.Position.y)*p.Raster.counts.x+uint(vertex.Position.x),owner=result[p.Raster.counts.w+pixel],reference=(owner&~SoftwarePriorityBit)>>1u;
    ExportPayload output{reference|((owner&1u)?FaceBit:0u),float2(0),as_type<float>(result[pixel])};
    if (reference>0 && reference<=p.Raster.counts.z) { float3 w,dx,dy; materialWeights(reference-1u,vertex.Position.xy,p,input,w,dx,dy); output.Bary=w.yz; }
    return output;
}
