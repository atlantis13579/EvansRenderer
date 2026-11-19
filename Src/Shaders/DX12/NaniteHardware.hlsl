#if defined(EVANS_HARDWARE_GRAPHICS)
// D3D12 hardware coverage and visibility export, using the GPU's projected
// corners and preclassified routes. Analytic gradients preserve alpha testing
// without differentiating across primitive IDs (Schied and Dachsbacher, 2015).
cbuffer DrawParameters : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images,Schedule; };
ByteAddressBuffer Geometry : register(t0),ClipTable : register(t1),WorkQueue : register(t2),OwnerRead : register(t5);
RWByteAddressBuffer OwnerWrite : register(u0);
Texture2D<float4> Albedo : register(t3),Opacity : register(t4);
SamplerState RepeatSampler : register(s0);
static const uint BytesPerWord=4,ClipStride=16,GeometryStride=48,CornerCount=3,HardwareRoute=2,BackFace=0x80000000u;
static const uint VectorWords=4,ColorWord=24,UVWord=36;
static const float NormalMinimum=1.175494351e-38f,MatrixTolerance=1.0e-12f;
float4 GeometryVector(uint id,uint word) { return asfloat(Geometry.Load4((id*GeometryStride+word)*BytesPerWord)); }
void ReadCorners(uint id,out float4 p0,out float4 p1,out float4 p2)
{
    uint address=(Layout.z+id*ClipStride)*BytesPerWord;
    p0=asfloat(ClipTable.Load4(address)); p1=asfloat(ClipTable.Load4(address+4*BytesPerWord)); p2=asfloat(ClipTable.Load4(address+8*BytesPerWord));
}
bool CanSnap(float4 corner)
{
    return all(isfinite(corner)) && corner.w>=NormalMinimum && all(abs(corner.xy)<=corner.w) && corner.z>=0 && corner.z<=corner.w;
}
float4 CoveragePosition(float4 p)
{
    float subpixels=float(1u<<Quality.y); float2 extent=float2(Counts.xy);
    float2 snapped=floor((p.xy/p.w*float2(0.5,-0.5)+0.5)*extent*subpixels+0.5)/subpixels;
    p.xy=(snapped/extent*float2(2,-2)+float2(-1,1))*p.w; return p;
}
bool Interpolation(uint id,float2 pixel,out float3 w,out float3 ddxW,out float3 ddyW)
{
    w=float3(1,0,0); ddxW=ddyW=0; float4 p0,p1,p2; ReadCorners(id,p0,p1,p2);
    float3 magnitude=max(abs(p0.xyw),max(abs(p1.xyw),abs(p2.xyw))); float scale=max(magnitude.x,max(magnitude.y,magnitude.z));
    if (scale==0) return false;
    float3 a=p0.xyw/scale,b=p1.xyw/scale,c=p2.xyw/scale;
    float3 row0=cross(b,c),row1=cross(c,a),row2=cross(a,b); float determinant=dot(a,row0);
    if (abs(determinant)<=MatrixTolerance) return false;
    row0/=determinant; row1/=determinant; row2/=determinant;
    float3 samplePosition=float3(pixel/float2(Counts.xy)*float2(2,-2)+float2(-1,1),1);
    float3 numerator=float3(dot(row0,samplePosition),dot(row1,samplePosition),dot(row2,samplePosition));
    float denominator=numerator.x+numerator.y+numerator.z; if (abs(denominator)<=MatrixTolerance) return false;
    w=numerator/denominator;
    float3 nx=float3(row0.x,row1.x,row2.x)*(2.0/Counts.x),ny=float3(row0.y,row1.y,row2.y)*(-2.0/Counts.y);
    ddxW=(nx-w*(nx.x+nx.y+nx.z))/denominator; ddyW=(ny-w*(ny.x+ny.y+ny.z))/denominator;
    return all(isfinite(w)) && all(isfinite(ddxW)) && all(isfinite(ddyW));
}
static const uint SoftwarePriorityBit=0x80000000u,PreserveMain=2u;
struct HardwareVertex { float4 Position:SV_POSITION; nointerpolation uint Ref:TEXCOORD0; nointerpolation uint Priority:TEXCOORD1; };
HardwareVertex VSQueue(uint vertex:SV_VertexID)
{
    HardwareVertex output=(HardwareVertex)0;
    uint start=WorkQueue.Load(Schedule.x*BytesPerWord),priority=WorkQueue.Load((start+vertex/CornerCount)*BytesPerWord);
    [branch] if (priority>=Schedule.z) return output;
    uint id=WorkQueue.Load((Schedule.y+priority)*BytesPerWord); [branch] if (id>=Counts.z) return output;
    output.Priority=priority;
    float4 p0,p1,p2; ReadCorners(id,p0,p1,p2); uint corner=vertex%CornerCount;
    output.Position=corner==0?p0:corner==1?p1:p2;
    // Shared inside corners keep one grid across clipped hardware neighbors.
    if (CanSnap(output.Position)) output.Position=CoveragePosition(output.Position);
    output.Ref=id+1u; return output;
}
struct RasterPayload { uint Ref:SV_TARGET0; float2 Bary:SV_TARGET1; };
void Coverage(HardwareVertex input)
{
    if (input.Ref==0 || input.Ref>Counts.z || input.Position.z>=1) discard;
    uint id=min(input.Ref>0?input.Ref-1u:0u,max(Counts.z,1u)-1u);
    float3 w,wx,wy; if (!Interpolation(id,input.Position.xy,w,wx,wy)) discard;
    float4 uv01=GeometryVector(id,UVWord),uv2=GeometryVector(id,UVWord+VectorWords);
    float2 a=float2(uv01.x,1-uv01.y),b=float2(uv01.z,1-uv01.w),c=float2(uv2.x,1-uv2.y);
    float2 uv=a*w.x+b*w.y+c*w.z,dx=(b-a)*wx.y+(c-a)*wx.z,dy=(b-a)*wy.y+(c-a)*wy.z;
    float3 alpha=float3(GeometryVector(id,ColorWord).a,GeometryVector(id,ColorWord+VectorWords).a,GeometryVector(id,ColorWord+2*VectorWords).a);
    float coverage=dot(alpha,w)*Material.x;
    if (Images.x!=0) coverage*=Albedo.SampleGrad(RepeatSampler,uv,dx,dy).a;
    if (Images.y!=0) coverage*=Opacity.SampleGrad(RepeatSampler,uv,dx,dy).r;
    if (Material.y>0 && saturate(coverage)<Material.y) discard;

}
void PSDepth(HardwareVertex input) { Coverage(input); }
// Read-only EQUAL depth testing precedes side effects. Cutout discard occurs
// before atomic ownership; the separate depth epoch uses ordinary late tests.
[earlydepthstencil] void PSOwner(HardwareVertex input,bool front:SV_IsFrontFace)
{
    Coverage(input); uint address=(uint(input.Position.y)*Counts.x+uint(input.Position.x))*BytesPerWord,ignored;
    OwnerWrite.InterlockedMin(address,((input.Priority+1u)<<1u)|(front?0u:1u),ignored);
}
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSResolve(uint vertex:SV_VertexID)
{ float2 corner=float2((vertex<<1u)&2u,vertex&2u); ScreenVertex output; output.Position=float4(corner*float2(2,-2)+float2(-1,1),0,1); return output; }
RasterPayload PSResolve(ScreenVertex input)
{
    static const uint InvalidOwner=0xffffffffu;
    uint pixel=uint(input.Position.y)*Counts.x+uint(input.Position.x),owner=OwnerRead.Load(pixel*BytesPerWord),priority=owner>>1u;
    if (owner==InvalidOwner || priority==0 || priority>Schedule.z) discard;
    // Discarded helper lanes must also retain valid raw-buffer addresses.
    uint safePriority=min(priority>0?priority-1u:0u,max(Schedule.z,1u)-1u);
    uint id=WorkQueue.Load((Schedule.y+safePriority)*BytesPerWord); if (id>=Counts.z) discard;
    id=min(id,max(Counts.z,1u)-1u);
    float3 w,dx,dy; if (!Interpolation(id,input.Position.xy,w,dx,dy)) discard;
    RasterPayload output; output.Ref=(id+1u)|((owner&1u)!=0?BackFace:0u); output.Bary=w.yz; return output;
}
#else
// Independent D3D12 hardware queue construction. GPU-written draw arguments
// refer to compact candidate priorities; no count is consumed by the CPU.
cbuffer HardwareSchedule : register(b0) { uint4 Counts,Layout,Images; };
ByteAddressBuffer SourceRanges : register(t0),RasterRoutes : register(t1);
RWByteAddressBuffer DrawArguments : register(u0),CompactQueue : register(u1),PixelOwners : register(u2);
static const uint GroupWidth=64,WordSize=4,RangeStride=4,HeaderStride=8,HardwareRoute=2,Corners=3;
static const uint HeaderStart=4,HeaderCursor=5,InvalidOwner=0xffffffffu;
uint LaneId(uint3 block,uint lane) { return (block.y*Layout.w+block.x)*GroupWidth+lane; }
uint2 Source(uint priority)
{
    uint first=0,last=Counts.y;
    while (first<last) { uint middle=(first+last)/2; uint4 record=SourceRanges.Load4(middle*RangeStride*WordSize); if (record.w+record.y>priority) last=middle; else first=middle+1; }
    uint4 item=SourceRanges.Load4(first*RangeStride*WordSize); return uint2(item.x+priority-item.w,item.z);
}
[numthreads(GroupWidth,1,1)] void CSClear(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=LaneId(block,lane);
    if (id<Counts.w) PixelOwners.Store(id*WordSize,InvalidOwner);
    if (id<Counts.z) { DrawArguments.Store4(id*HeaderStride*WordSize,0); DrawArguments.Store4((id*HeaderStride+HeaderStart)*WordSize,0); }
}
[numthreads(GroupWidth,1,1)] void CSCount(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint priority=LaneId(block,lane); [branch] if (priority>=Counts.x) return;
    uint2 source=Source(priority); if (RasterRoutes.Load((Layout.x+source.x)*WordSize)!=HardwareRoute) return;
    uint unused; DrawArguments.InterlockedAdd(source.y*HeaderStride*WordSize,1,unused);
}
[numthreads(1,1,1)] void CSScan()
{
    uint first=0;
    for (uint bin=0;bin<Counts.z;++bin)
    {
        uint triangles=DrawArguments.Load(bin*HeaderStride*WordSize);
        DrawArguments.Store4(bin*HeaderStride*WordSize,uint4(triangles*Corners,1,0,0));
        DrawArguments.Store4((bin*HeaderStride+HeaderStart)*WordSize,uint4(first,0,triangles,0));
        CompactQueue.Store(bin*WordSize,Layout.z+first); first+=triangles;
    }
}
[numthreads(GroupWidth,1,1)] void CSScatter(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint priority=LaneId(block,lane); [branch] if (priority>=Counts.x) return;
    uint2 source=Source(priority); if (RasterRoutes.Load((Layout.x+source.x)*WordSize)!=HardwareRoute) return;
    uint ordinal; DrawArguments.InterlockedAdd((source.y*HeaderStride+HeaderCursor)*WordSize,1,ordinal);
    uint start=DrawArguments.Load((source.y*HeaderStride+HeaderStart)*WordSize);
    CompactQueue.Store((Layout.y+priority)*WordSize,source.x); CompactQueue.Store((Layout.z+start+ordinal)*WordSize,priority);
}
#endif
