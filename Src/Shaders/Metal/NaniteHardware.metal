#include <metal_stdlib>
using namespace metal;
// Native Metal count/prefix/scatter and separate depth/ownership epochs.
// Fragment ownership uses early tests with depth writes disabled (MSL 5.1.2).
struct HardwareSchedule { uint4 Counts,Layout,Images; };
struct RasterFrame { uint4 counts,layout,dispatch,quality; };
struct DrawParameters { RasterFrame Raster; uint4 Range; float4 Material; uint4 Images,Schedule; };
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
constant uint HardwareThreads=64,BinWords=8,OffsetWord=4,CursorWord=5,CountWord=6,HardwareRoute=2,Corners=3;
constant uint EmptyOwner=0xffffffffu,FaceBit=0x80000000u,TriangleWords=16,VectorWords=4;
constant float MinimumNormal=1.175494351e-38f;
uint laneIndex(uint3 group,uint lane,constant HardwareSchedule& p) { return (group.y*p.Layout.w+group.x)*HardwareThreads+lane; }
uint2 sourceFor(uint priority,constant HardwareSchedule& p,device const uint4* ranges)
{
    uint first=0,last=p.Counts.y;
    while (first<last) { uint middle=(first+last)/2; uint4 record=ranges[middle]; if (record.w+record.y>priority) last=middle; else first=middle+1; }
    uint4 record=ranges[first]; return uint2(record.x+priority-record.w,record.z);
}
kernel void EvansHardwareClear(constant HardwareSchedule& p [[buffer(0)]],device atomic_uint* arguments [[buffer(3)]],
    device atomic_uint* owners [[buffer(5)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint id=laneIndex(group,lane,p);
    if (id<p.Counts.w) atomic_store_explicit(&owners[id],EmptyOwner,memory_order_relaxed);
    if (id<p.Counts.z) for (uint word=0;word<BinWords;++word) atomic_store_explicit(&arguments[id*BinWords+word],0,memory_order_relaxed);
}
kernel void EvansHardwareCount(constant HardwareSchedule& p [[buffer(0)]],device const uint4* ranges [[buffer(1)]],device const uint* routes [[buffer(2)]],
    device atomic_uint* arguments [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint id=laneIndex(group,lane,p); if (id>=p.Counts.x) return;
    uint2 source=sourceFor(id,p,ranges); if (routes[p.Layout.x+source.x]!=HardwareRoute) return;
    atomic_fetch_add_explicit(&arguments[source.y*BinWords],1,memory_order_relaxed);
}
kernel void EvansHardwareScan(constant HardwareSchedule& p [[buffer(0)]],device atomic_uint* arguments [[buffer(3)]],device uint* queue [[buffer(4)]])
{
    uint first=0;
    for (uint bin=0;bin<p.Counts.z;++bin)
    {
        uint header=bin*BinWords,count=atomic_load_explicit(&arguments[header],memory_order_relaxed);
        atomic_store_explicit(&arguments[header],count*Corners,memory_order_relaxed); atomic_store_explicit(&arguments[header+1],1,memory_order_relaxed);
        atomic_store_explicit(&arguments[header+OffsetWord],first,memory_order_relaxed); atomic_store_explicit(&arguments[header+CountWord],count,memory_order_relaxed);
        queue[bin]=p.Layout.z+first; first+=count;
    }
}
kernel void EvansHardwareScatter(constant HardwareSchedule& p [[buffer(0)]],device const uint4* ranges [[buffer(1)]],device const uint* routes [[buffer(2)]],
    device atomic_uint* arguments [[buffer(3)]],device uint* queue [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint id=laneIndex(group,lane,p); if (id>=p.Counts.x) return;
    uint2 source=sourceFor(id,p,ranges); if (routes[p.Layout.x+source.x]!=HardwareRoute) return;
    uint header=source.y*BinWords,slot=atomic_fetch_add_explicit(&arguments[header+CursorWord],1,memory_order_relaxed);
    uint first=atomic_load_explicit(&arguments[header+OffsetWord],memory_order_relaxed);
    queue[p.Layout.y+id]=source.x; queue[p.Layout.z+first+slot]=id;
}
float4 readVertex(device const uint* words,uint at) { return as_type<float4>(uint4(words[at],words[at+1],words[at+2],words[at+3])); }
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
struct HardwareVertex { float4 Position [[position]]; uint Reference [[flat]]; uint Priority [[flat]]; };
vertex HardwareVertex EvansHardwareQueueVertex(uint vertex [[vertex_id]],constant DrawParameters& p [[buffer(0)]],
    device const uint* input [[buffer(2)]],device const uint* queue [[buffer(3)]])
{
    HardwareVertex output{}; uint priority=queue[queue[p.Schedule.x]+vertex/Corners]; if (priority>=p.Schedule.z) return output;
    uint id=queue[p.Schedule.y+priority]; if (id>=p.Raster.counts.z) return output;
    output.Position=projectedCorner(input,id,vertex%Corners,p);
    if (safeCoverageCorner(output.Position)) output.Position=snapCoverage(output.Position,p);
    output.Reference=id+1; output.Priority=priority; return output;
}
void coverage(HardwareVertex vertex,constant DrawParameters& p,device const WorldTriangle* geometry,device const uint* input,
    texture2d<float> diffuse,texture2d<float> opacity,sampler surfaceSampler)
{
    if (vertex.Reference==0 || vertex.Reference>p.Raster.counts.z || vertex.Position.z>=1) discard_fragment();
    uint id=min(vertex.Reference>0?vertex.Reference-1u:0u,max(p.Raster.counts.z,1u)-1u); float3 w,dx,dy;
    if (!materialWeights(id,vertex.Position.xy,p,input,w,dx,dy)) discard_fragment();
    WorldTriangle item=geometry[id]; float2 a(item.UVAB.x,1-item.UVAB.y),b(item.UVAB.z,1-item.UVAB.w),c(item.UVC.x,1-item.UVC.y);
    float2 uv=a*w.x+b*w.y+c*w.z,uvDx=(b-a)*dx.y+(c-a)*dx.z,uvDy=(b-a)*dy.y+(c-a)*dy.z;
    float coverage=dot(float3(item.Color[0].a,item.Color[1].a,item.Color[2].a),w)*p.Material.x;
    if (p.Images.x) coverage*=diffuse.sample(surfaceSampler,uv,gradient2d(uvDx,uvDy)).a;
    if (p.Images.y) coverage*=opacity.sample(surfaceSampler,uv,gradient2d(uvDx,uvDy)).r;
    if (p.Material.y>0 && saturate(coverage)<p.Material.y) discard_fragment();
}
fragment void EvansHardwareDepth(HardwareVertex vertex [[stage_in]],constant DrawParameters& p [[buffer(0)]],
    device const WorldTriangle* geometry [[buffer(1)]],device const uint* input [[buffer(2)]],
    texture2d<float> diffuse [[texture(0)]],texture2d<float> opacity [[texture(1)]],sampler surfaceSampler [[sampler(0)]])
{ coverage(vertex,p,geometry,input,diffuse,opacity,surfaceSampler); }
[[early_fragment_tests]]
fragment void EvansHardwareOwner(HardwareVertex vertex [[stage_in]],bool front [[front_facing]],constant DrawParameters& p [[buffer(0)]],
    device const WorldTriangle* geometry [[buffer(1)]],device const uint* input [[buffer(2)]],device atomic_uint* owners [[buffer(4)]],
    texture2d<float> diffuse [[texture(0)]],texture2d<float> opacity [[texture(1)]],sampler surfaceSampler [[sampler(0)]])
{
    coverage(vertex,p,geometry,input,diffuse,opacity,surfaceSampler);
    uint pixel=uint(vertex.Position.y)*p.Raster.counts.x+uint(vertex.Position.x);
    atomic_fetch_min_explicit(&owners[pixel],((vertex.Priority+1u)<<1u)|(front?0u:1u),memory_order_relaxed);
}
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex EvansHardwareResolveVertex(uint vertex [[vertex_id]])
{ uint2 p((vertex<<1u)&2u,vertex&2u); return {float4(float2(p)*float2(2,-2)+float2(-1,1),0,1)}; }
struct RasterPayload { uint Reference [[color(0)]]; float2 Bary [[color(1)]]; };
fragment RasterPayload EvansHardwareResolve(ScreenVertex vertex [[stage_in]],constant DrawParameters& p [[buffer(0)]],
    device const uint* input [[buffer(2)]],device const uint* queue [[buffer(3)]],device const uint* owners [[buffer(4)]])
{
    uint pixel=uint(vertex.Position.y)*p.Raster.counts.x+uint(vertex.Position.x),owner=owners[pixel],priority=owner>>1u;
    if (owner==EmptyOwner || priority==0 || priority>p.Schedule.z) discard_fragment();
    uint safePriority=min(priority>0?priority-1u:0u,max(p.Schedule.z,1u)-1u);
    uint id=queue[p.Schedule.y+safePriority]; if (id>=p.Raster.counts.z) discard_fragment();
    id=min(id,max(p.Raster.counts.z,1u)-1u);
    float3 w,dx,dy; if (!materialWeights(id,vertex.Position.xy,p,input,w,dx,dy)) discard_fragment();
    return {(id+1u)|((owner&1u)?FaceBit:0u),w.yz};
}
