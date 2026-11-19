#if defined(EVANS_HARDWARE_GRAPHICS)
// Native hardware and export stages share the compute-projected clip table.
// Material reconstruction continues to use original homogeneous positions;
// snapping changes coverage only. Clipped triangles remain hardware-owned.
cbuffer RasterDraw : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images,Schedule; };
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
StructuredBuffer<WorldTriangle> Geometry : register(t0);
ByteAddressBuffer Projected : register(t1);
ByteAddressBuffer Queue : register(t2),OwnerRead : register(t5);
RWByteAddressBuffer Owners : register(u2);
Texture2D<float4> DiffuseMap : register(t3),OpacityMap : register(t4);
SamplerState SurfaceSampler : register(s0);
static const uint WordBytes=4,RecordWords=16,TriangleCorners=3,HardwareRoute=2,BackFaceBit=0x80000000u;
static const uint SoftwarePriorityBit=0x80000000u,PreserveMain=2u;
static const float MinimumNormal=1.175494351e-38f,InverseTolerance=1.0e-12f;
void ClipCorners(uint id,out float4 a,out float4 b,out float4 c)
{
    uint base=(Layout.z+id*RecordWords)*WordBytes;
    a=asfloat(Projected.Load4(base)); b=asfloat(Projected.Load4(base+4*WordBytes)); c=asfloat(Projected.Load4(base+8*WordBytes));
}
bool Projectable(float4 corner)
{
    return all(isfinite(corner)) && corner.w>=MinimumNormal && all(abs(corner.xy)<=corner.w) && corner.z>=0 && corner.z<=corner.w;
}
float4 Snap(float4 clip)
{
    float grid=float(1u<<Quality.y); float2 pixel=floor((clip.xy/clip.w*float2(0.5,-0.5)+0.5)*float2(Counts.xy)*grid+0.5)/grid;
    clip.xy=(pixel/float2(Counts.xy)*float2(2,-2)+float2(-1,1))*clip.w; return clip;
}
bool Weights(uint id,float2 pixel,out float3 weights,out float3 dx,out float3 dy)
{
    float4 a,b,c; ClipCorners(id,a,b,c); weights=float3(1,0,0); dx=dy=0;
    float3 bound=max(abs(a.xyw),max(abs(b.xyw),abs(c.xyw))); float scale=max(bound.x,max(bound.y,bound.z));
    if (scale==0) return false;
    float3 ca=a.xyw/scale,cb=b.xyw/scale,cc=c.xyw/scale;
    float3 r0=cross(cb,cc),r1=cross(cc,ca),r2=cross(ca,cb); float determinant=dot(ca,r0);
    if (abs(determinant)<=InverseTolerance) return false;
    r0/=determinant; r1/=determinant; r2/=determinant;
    float3 position=float3(pixel/float2(Counts.xy)*float2(2,-2)+float2(-1,1),1);
    float3 h=float3(dot(r0,position),dot(r1,position),dot(r2,position)); float total=h.x+h.y+h.z;
    if (abs(total)<=InverseTolerance) return false;
    weights=h/total;
    float3 hx=float3(r0.x,r1.x,r2.x)*(2.0/float(Counts.x)),hy=float3(r0.y,r1.y,r2.y)*(-2.0/float(Counts.y));
    dx=(hx-weights*(hx.x+hx.y+hx.z))/total; dy=(hy-weights*(hy.x+hy.y+hy.z))/total;
    return all(isfinite(weights)) && all(isfinite(dx)) && all(isfinite(dy));
}
struct HardwareVertex { float4 Position:SV_POSITION; nointerpolation uint Reference:TEXCOORD0; nointerpolation uint Priority:TEXCOORD1; };
HardwareVertex VSQueue(uint vertex:SV_VertexID)
{
    HardwareVertex result=(HardwareVertex)0;
    uint first=Queue.Load(Schedule.x*WordBytes),priority=Queue.Load((first+vertex/TriangleCorners)*WordBytes);
    [branch] if (priority>=Schedule.z) return result;
    uint id=Queue.Load((Schedule.y+priority)*WordBytes); [branch] if (id>=Counts.z) return result;
    result.Priority=priority;
    float4 a,b,c; ClipCorners(id,a,b,c); uint corner=vertex%TriangleCorners;
    result.Position=corner==0?a:corner==1?b:c;
    // Snap eligible shared corners even when the third corner needs clipping.
    if (Projectable(result.Position)) result.Position=Snap(result.Position);
    result.Reference=id+1u; return result;
}
struct RasterTargets { uint Reference:SV_TARGET0; float2 Bary:SV_TARGET1; };
void Coverage(HardwareVertex input)
{
    if (input.Reference==0 || input.Reference>Counts.z || input.Position.z>=1) discard;
    uint id=min(input.Reference>0?input.Reference-1u:0u,max(Counts.z,1u)-1u);
    float3 weights,dx,dy; if (!Weights(id,input.Position.xy,weights,dx,dy)) discard;
    WorldTriangle item=Geometry[id];
    float2 a=float2(item.UVAB.x,1-item.UVAB.y),b=float2(item.UVAB.z,1-item.UVAB.w),c=float2(item.UVC.x,1-item.UVC.y);
    float2 uv=a*weights.x+b*weights.y+c*weights.z,uvDx=(b-a)*dx.y+(c-a)*dx.z,uvDy=(b-a)*dy.y+(c-a)*dy.z;
    float alpha=(item.Color[0].a*weights.x+item.Color[1].a*weights.y+item.Color[2].a*weights.z)*Material.x;
    if (Images.x!=0) alpha*=DiffuseMap.SampleGrad(SurfaceSampler,uv,uvDx,uvDy).a;
    if (Images.y!=0) alpha*=OpacityMap.SampleGrad(SurfaceSampler,uv,uvDx,uvDy).r;
    if (Material.y>0 && saturate(alpha)<Material.y) discard;

}
// Ownership runs only after a complete depth epoch. Early tests are forced,
// while depth writes are disabled, so alpha rejection cannot corrupt depth.
void PSDepth(HardwareVertex input) { Coverage(input); }
[earlydepthstencil] void PSOwner(HardwareVertex input,bool front:SV_IsFrontFace)
{
    Coverage(input); uint pixel=uint(input.Position.y)*Counts.x+uint(input.Position.x),ignored;
    Owners.InterlockedMin(pixel*WordBytes,((input.Priority+1u)<<1u)|(front?0u:1u),ignored);
}
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSResolve(uint vertex:SV_VertexID)
{ float2 p=float2((vertex<<1u)&2u,vertex&2u); ScreenVertex result; result.Position=float4(p*float2(2,-2)+float2(-1,1),0,1); return result; }
RasterTargets PSResolve(ScreenVertex input)
{
    static const uint EmptyOwner=0xffffffffu;
    uint pixel=uint(input.Position.y)*Counts.x+uint(input.Position.x),key=OwnerRead.Load(pixel*WordBytes),priority=key>>1u;
    if (key==EmptyOwner || priority==0 || priority>Schedule.z) discard;
    // Discarded helper lanes must also retain valid raw-buffer addresses.
    uint safePriority=min(priority>0?priority-1u:0u,max(Schedule.z,1u)-1u);
    uint id=Queue.Load((Schedule.y+safePriority)*WordBytes); if (id>=Counts.z) discard;
    id=min(id,max(Counts.z,1u)-1u);
    float3 w,dx,dy; if (!Weights(id,input.Position.xy,w,dx,dy)) discard;
    RasterTargets result; result.Reference=(id+1u)|((key&1u)!=0?BackFaceBit:0u); result.Bary=w.yz; return result;
}
#else
// Count/prefix/scatter build native indirect arguments and a compact queue.
// Queue entries keep original submission priorities; atomic ownership makes
// scatter order and nonadjacent material batching irrelevant to equal depths.
cbuffer ScheduleFrame : register(b0) { uint4 Counts,Layout,Images; };
ByteAddressBuffer Ranges : register(t0),Routes : register(t1);
RWByteAddressBuffer Arguments : register(u0),Queue : register(u1),Owners : register(u2);
static const uint Threads=64,WordBytes=4,RangeWords=4,BinWords=8,HardwareRoute=2,TriangleCorners=3;
static const uint BinOffset=4,BinCursor=5,EmptyOwner=0xffffffffu;
uint Invocation(uint3 group,uint lane) { return (group.y*Layout.w+group.x)*Threads+lane; }
uint4 Candidate(uint id)
{
    uint low=0,high=Counts.y;
    while (low<high) { uint middle=(low+high)/2; uint4 range=Ranges.Load4(middle*RangeWords*WordBytes); if (range.w+range.y<=id) low=middle+1; else high=middle; }
    uint4 source=Ranges.Load4(low*RangeWords*WordBytes); return uint4(source.x+id-source.w,source.z,0,0);
}
[numthreads(Threads,1,1)] void CSClear(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Invocation(group,lane);
    if (id<Counts.w) Owners.Store(id*WordBytes,EmptyOwner);
    if (id<Counts.z) { Arguments.Store4(id*BinWords*WordBytes,0); Arguments.Store4((id*BinWords+BinOffset)*WordBytes,0); }
}
[numthreads(Threads,1,1)] void CSCount(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Invocation(group,lane); [branch] if (id>=Counts.x) return;
    uint4 work=Candidate(id); if (Routes.Load((Layout.x+work.x)*WordBytes)!=HardwareRoute) return;
    uint ignored; Arguments.InterlockedAdd(work.y*BinWords*WordBytes,1,ignored);
}
[numthreads(1,1,1)] void CSScan()
{
    uint offset=0;
    for (uint bin=0;bin<Counts.z;++bin)
    {
        uint count=Arguments.Load(bin*BinWords*WordBytes);
        Arguments.Store4(bin*BinWords*WordBytes,uint4(count*TriangleCorners,1,0,0));
        Arguments.Store4((bin*BinWords+BinOffset)*WordBytes,uint4(offset,0,count,0));
        Queue.Store(bin*WordBytes,Layout.z+offset); offset+=count;
    }
}
[numthreads(Threads,1,1)] void CSScatter(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Invocation(group,lane); [branch] if (id>=Counts.x) return;
    uint4 work=Candidate(id); if (Routes.Load((Layout.x+work.x)*WordBytes)!=HardwareRoute) return;
    uint slot; Arguments.InterlockedAdd((work.y*BinWords+BinCursor)*WordBytes,1,slot);
    uint start=Arguments.Load((work.y*BinWords+BinOffset)*WordBytes);
    Queue.Store((Layout.y+id)*WordBytes,work.x); Queue.Store((Layout.z+start+slot)*WordBytes,id);
}
#endif
