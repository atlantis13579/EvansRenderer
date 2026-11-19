#if defined(EVANS_NANITE_VISIBILITY_GEOMETRY)
// D3D12 assembly preserves the decoded buffers' native GPU addresses. Each
// invocation expands one indexed triangle into the material reconstruction ABI.
cbuffer NativeGeometryFrame : register(b0)
{ row_major float4x4 WorldMatrix,NormalMatrix; uint4 GeometryRange,Work; };
ByteAddressBuffer VertexStream : register(t0),IndexStream : register(t1);
RWByteAddressBuffer WorldTable : register(u0);
static const uint GroupWidth=64,WordSize=4,VertexStride=12,TriangleStride=48,VectorStride=4,CornerCount=3;
static const uint VertexNormal=3,VertexColor=6,VertexUV=10,TableNormal=12,TableColor=24,TableUV=36,TableSource=44;
static const float NormalLengthLimit=1.0e-6f,SRGBThreshold=0.04045f,SRGBSlope=12.92f;
static const float SRGBBias=0.055f,SRGBDenominator=1.055f,SRGBPower=2.4f;
float DecodeChannel(float value)
{ value=saturate(value); return value<=SRGBThreshold?value/SRGBSlope:pow((value+SRGBBias)/SRGBDenominator,SRGBPower); }
float3 NormalizeSource(float3 n)
{
    float bound=max(abs(n.x),max(abs(n.y),abs(n.z)));
    float3 scaled=bound>0?n/bound:float3(0,1,0); float distance=length(scaled);
    return bound*distance>NormalLengthLimit?scaled/distance:float3(0,1,0);
}
[numthreads(GroupWidth,1,1)] void CSVisibilityGeometry(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint local=(block.y*Work.x+block.x)*GroupWidth+lane;
    [branch] if (local>=Work.y) return;
    uint3 indices=IndexStream.Load3((GeometryRange.z+local*CornerCount)*WordSize);
    float4 positions[CornerCount],normals[CornerCount],colors[CornerCount]; float2 uv[CornerCount];
    [unroll] for (uint c=0;c<CornerCount;++c) { positions[c]=float4(0,0,0,1); normals[c]=colors[c]=0; uv[c]=0; }
    bool valid=all(indices<GeometryRange.w);
    [branch] if (valid)
    {
        [unroll] for (uint sourceCorner=0;sourceCorner<CornerCount;++sourceCorner)
        {
            uint address=indices[sourceCorner]*VertexStride*WordSize;
            float3 p=asfloat(VertexStream.Load3(address)),n=asfloat(VertexStream.Load3(address+VertexNormal*WordSize));
            float4 color=asfloat(VertexStream.Load4(address+VertexColor*WordSize)); uv[sourceCorner]=asfloat(VertexStream.Load2(address+VertexUV*WordSize));
            positions[sourceCorner]=mul(WorldMatrix,float4(p,1)); float3 transformed=mul(NormalMatrix,float4(n,0)).xyz;
            valid=valid && all(isfinite(positions[sourceCorner])) && all(isfinite(transformed)) && all(isfinite(color)) && all(isfinite(uv[sourceCorner]));
            normals[sourceCorner]=float4(NormalizeSource(transformed),0); colors[sourceCorner]=float4(DecodeChannel(color.r),DecodeChannel(color.g),DecodeChannel(color.b),color.a);
        }
    }
    uint destination=(GeometryRange.x+local)*TriangleStride;
    [unroll] for (uint destinationCorner=0;destinationCorner<CornerCount;++destinationCorner)
    {
        if (!valid) { positions[destinationCorner]=float4(0,0,0,1); normals[destinationCorner]=colors[destinationCorner]=0; uv[destinationCorner]=0; }
        WorldTable.Store4((destination+destinationCorner*VectorStride)*WordSize,asuint(positions[destinationCorner]));
        WorldTable.Store4((destination+TableNormal+destinationCorner*VectorStride)*WordSize,asuint(normals[destinationCorner]));
        WorldTable.Store4((destination+TableColor+destinationCorner*VectorStride)*WordSize,asuint(colors[destinationCorner]));
    }
    WorldTable.Store4((destination+TableUV)*WordSize,asuint(float4(uv[0],uv[1])));
    WorldTable.Store4((destination+TableUV+VectorStride)*WordSize,asuint(float4(uv[2],0,0)));
    WorldTable.Store4((destination+TableSource)*WordSize,uint4(Work.z,local,0,0));
}
#elif defined(EVANS_NANITE_GRAPHICS)
// D3D12 hardware coverage and visibility export, using the GPU's projected
// corners and preclassified routes. Analytic gradients preserve alpha testing
// without differentiating across primitive IDs (Schied and Dachsbacher, 2015).
cbuffer DrawParameters : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images; };
ByteAddressBuffer Geometry : register(t0),ClipTable : register(t1),VisibilityResult : register(t2);
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
struct HardwareVertex { float4 Position:SV_POSITION; nointerpolation uint Ref:TEXCOORD0; };
HardwareVertex VSHardware(uint vertex:SV_VertexID)
{
    HardwareVertex output=(HardwareVertex)0; uint local=vertex/CornerCount,id=Range.x+local;
    [branch] if (local>=Range.y || id>=Counts.z) return output;
    if ((Quality.w&PreserveMain)!=0) id=Range.x+Range.y-1u-local;
    [branch] if (VisibilityResult.Load((Layout.w+id)*BytesPerWord)!=HardwareRoute) return output;
    float4 p0,p1,p2; ReadCorners(id,p0,p1,p2); uint corner=vertex%CornerCount;
    output.Position=corner==0?p0:corner==1?p1:p2;
    // Shared inside corners keep one grid across clipped hardware neighbors.
    if (CanSnap(output.Position)) output.Position=CoveragePosition(output.Position);
    output.Ref=id+1u; return output;
}
struct RasterPayload { uint Ref:SV_TARGET0; float2 Bary:SV_TARGET1; };
RasterPayload PSHardware(HardwareVertex input,bool front:SV_IsFrontFace)
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
    RasterPayload output; output.Ref=input.Ref|(front?0u:BackFace); output.Bary=w.yz; return output;
}
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSExport(uint vertex:SV_VertexID)
{ float2 corner=float2((vertex<<1u)&2u,vertex&2u); ScreenVertex output; output.Position=float4(corner*float2(2,-2)+float2(-1,1),0,1); return output; }
struct ExportPayload { uint Ref:SV_TARGET0; float2 Bary:SV_TARGET1; float Depth:SV_DEPTH; };
ExportPayload PSExport(ScreenVertex input)
{
    uint pixel=uint(input.Position.y)*Counts.x+uint(input.Position.x),owner=VisibilityResult.Load((Counts.w+pixel)*BytesPerWord),ref=(owner&~SoftwarePriorityBit)>>1u;
    ExportPayload output; output.Depth=asfloat(VisibilityResult.Load(pixel*BytesPerWord)); output.Ref=ref|((owner&1u)!=0?BackFace:0u); output.Bary=0;
    [branch] if (ref>0 && ref<=Counts.z) { float3 w,wx,wy; Interpolation(ref-1u,input.Position.xy,w,wx,wy); output.Bary=w.yz; }
    return output;
}
#elif defined(EVANS_NANITE_FRAME_PASS)
// Projection and hardware-depth capture consume GPU buffers and attachments.
cbuffer FrameInput : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality; row_major float4x4 Camera; uint4 Range; };
ByteAddressBuffer WorldGeometry : register(t0);
RWByteAddressBuffer InputTable : register(u0);
static const uint BytesPerWord=4,ClipRecordWords=16,WorldRecordWords=48,Threads=64;
#if EVANS_NANITE_FRAME_PASS == 4
ByteAddressBuffer ClusterDecisions : register(t3);
static const uint OcclusionRejectedFlag=8,FrustumRejectedState=1,ClipFlagsWord=12;
[numthreads(Threads,1,1)] void CSProject(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint local=(block.y*Dispatch.x+block.x)*Threads+lane;
    [branch] if (local>=Dispatch.y) return;
    uint id=Range.x+local,source=id*WorldRecordWords*BytesPerWord,target=(Layout.z+id*ClipRecordWords)*BytesPerWord;
    // Only definitive frustum rejection skips geometry; deferred/main winners
    // retain their projected corners through the second visibility phase.
    [branch] if (Range.w!=0) { [branch] if (ClusterDecisions.Load((Range.w-1)*BytesPerWord)==FrustumRejectedState)
    { InputTable.Store4(target+ClipFlagsWord*BytesPerWord,uint4(Range.z|OcclusionRejectedFlag,0,0,0)); return; } }
    float4 a=asfloat(WorldGeometry.Load4(source)),b=asfloat(WorldGeometry.Load4(source+4*BytesPerWord)),c=asfloat(WorldGeometry.Load4(source+8*BytesPerWord));
    InputTable.Store4(target,asuint(mul(Camera,a))); InputTable.Store4(target+4*BytesPerWord,asuint(mul(Camera,b)));
    InputTable.Store4(target+8*BytesPerWord,asuint(mul(Camera,c))); uint flags=Range.z;
    [branch] if (Range.w!=0) { if (ClusterDecisions.Load((Range.w-1)*BytesPerWord)!=Quality.z) flags|=OcclusionRejectedFlag; }
    InputTable.Store4(target+12*BytesPerWord,uint4(flags,0,0,0));
}
#elif EVANS_NANITE_FRAME_PASS == 5
Texture2D<float> CameraDepth : register(t1);
Texture2D<uint> CameraReferences : register(t2);
[numthreads(Threads,1,1)] void CSCapture(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint pixel=(block.y*Dispatch.x+block.x)*Threads+lane;
    [branch] if (pixel>=Dispatch.y) return;
    int3 at=int3(pixel%Counts.x,pixel/Counts.x,0);
    InputTable.Store((Layout.x+pixel)*BytesPerWord,asuint(CameraDepth.Load(at)));
    InputTable.Store((Layout.y+pixel)*BytesPerWord,CameraReferences.Load(at));
}
#else
[numthreads(Threads,1,1)] void CSClear(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{ uint word=(block.y*Dispatch.x+block.x)*Threads+lane; [branch] if (word<Dispatch.y) InputTable.Store(word*BytesPerWord,0); }
#endif
#else
// D3D12 small-triangle visibility, independently authored for raw root buffers.
// The four dispatches are seed depth, atomic depth, seed ownership, atomic owner.
// UAV barriers freeze the minimum before payload selection; no spinlocks or
// optional 64-bit atomic instructions are needed. Karis et al., SIGGRAPH 2021.
cbuffer RasterParameters : register(b0) { uint4 Counts,Layout,Dispatch,Quality; };
ByteAddressBuffer Source : register(t0);
RWByteAddressBuffer Result : register(u0);
static const uint BytesPerWord=4,WordsPerTriangle=16,Threads=64;
static const uint RejectRoute=0,SoftwareRoute=1,HardwareRoute=2;
static const uint PriorityOrdering=1u,PreserveMain=2u,SoftwarePriorityBit=0x80000000u;
static const uint CullFlag=1,ClockwiseFlag=2,ForceHardwareFlag=4,OcclusionRejected=8,TriangleMask=0x7fffffffu,InvalidOwner=0xffffffffu;
static const float SmallestNormal=1.175494351e-38f;
struct RasterTriangle { int2 P0,P1,P2,Lo,Hi; float3 Depth; int TwiceArea; uint Key,Route; };
int SignedEdge(int2 start,int2 end,int2 pixelPosition)
{ int2 edge=end-start,delta=pixelPosition-start; return edge.x*delta.y-edge.y*delta.x; }
bool Covered(int value,int2 start,int2 end)
{ int2 edge=end-start; return value>0 || (value==0 && (edge.y<0 || (edge.y==0 && edge.x>0))); }
RasterTriangle TriangleSetup(uint id)
{
    RasterTriangle t=(RasterTriangle)0; t.Route=HardwareRoute; t.TwiceArea=1;
    uint base=(Layout.z+id*WordsPerTriangle)*BytesPerWord;
    float4 v0=asfloat(Source.Load4(base)),v1=asfloat(Source.Load4(base+4*BytesPerWord)),v2=asfloat(Source.Load4(base+8*BytesPerWord));
    uint mode=Source.Load(base+12*BytesPerWord);
    bool projectable=(mode&(ForceHardwareFlag|OcclusionRejected))==0 && all(isfinite(v0)) && all(isfinite(v1)) && all(isfinite(v2)) && v0.w>=SmallestNormal && v1.w>=SmallestNormal && v2.w>=SmallestNormal &&
        all(abs(v0.xy)<=v0.w) && all(abs(v1.xy)<=v1.w) && all(abs(v2.xy)<=v2.w) &&
        v0.z>=0 && v1.z>=0 && v2.z>=0 && v0.z<=v0.w && v1.z<=v1.w && v2.z<=v2.w;
    [branch] if (projectable)
    {
        float grid=float(1u<<Quality.y); float2 dimensions=float2(Counts.xy);
        t.P0=int2(floor((v0.xy/v0.w*float2(0.5,-0.5)+0.5)*dimensions*grid+0.5));
        t.P1=int2(floor((v1.xy/v1.w*float2(0.5,-0.5)+0.5)*dimensions*grid+0.5));
        t.P2=int2(floor((v2.xy/v2.w*float2(0.5,-0.5)+0.5)*dimensions*grid+0.5));
        int2 minimum=min(t.P0,min(t.P1,t.P2)),maximum=max(t.P0,max(t.P1,t.P2));
        // At most 32 * 256 local subpixels in either direction prevents overflow.
        [branch] if (all(maximum-minimum<=int(Quality.x)*int(grid)))
        {
            t.TwiceArea=SignedEdge(t.P0,t.P1,t.P2); t.Route=RejectRoute;
            bool reverse=(t.TwiceArea>0)!=((mode&ClockwiseFlag)!=0);
            [branch] if (t.TwiceArea!=0 && !(reverse && (mode&CullFlag)!=0))
            {
                t.Key=((id+1u)<<1u)|uint(reverse); t.Depth=float3(v0.z/v0.w,v1.z/v1.w,v2.z/v2.w);
                if (t.TwiceArea<0)
                { int2 temporaryPosition=t.P1; t.P1=t.P2; t.P2=temporaryPosition; float z=t.Depth.y; t.Depth.y=t.Depth.z; t.Depth.z=z; t.TwiceArea=-t.TwiceArea; }
                t.Lo=max(int2(0,0),int2(ceil((float2(minimum)-grid*0.5)/grid)));
                t.Hi=min(int2(Counts.xy)-1,int2(floor((float2(maximum)-grid*0.5)/grid)));
                t.Route=any(t.Lo>t.Hi)?RejectRoute:SoftwareRoute;
            }
        }
    }
    if ((mode&OcclusionRejected)!=0) t.Route=RejectRoute;
    return t;
}
void VisitTriangle(uint id,bool selectOwner)
{
    RasterTriangle t=TriangleSetup(id); uint route=t.Route;
    if (!selectOwner) Result.Store((Layout.w+id)*BytesPerWord,route);
    [branch] if (Dispatch.z==4u || route!=SoftwareRoute) return;
    int grid=int(1u<<Quality.y);
    for (int row=t.Lo.y;row<=t.Hi.y;++row) for (int column=t.Lo.x;column<=t.Hi.x;++column)
    {
        int2 pixelPosition=int2(column,row)*grid+(grid>>1);
        int a=SignedEdge(t.P1,t.P2,pixelPosition),b=SignedEdge(t.P2,t.P0,pixelPosition),c=SignedEdge(t.P0,t.P1,pixelPosition);
        [branch] if (Covered(a,t.P1,t.P2) && Covered(b,t.P2,t.P0) && Covered(c,t.P0,t.P1))
        {
            float z=t.Depth.x+(t.Depth.y-t.Depth.x)*(float(b)/float(t.TwiceArea))+(t.Depth.z-t.Depth.x)*(float(c)/float(t.TwiceArea));
            z=saturate(z); uint depth=z<=0?0u:asuint(z),pixel=uint(row)*Counts.x+uint(column);
            [branch] if ((Quality.w&PriorityOrdering)!=0?depth<asuint(1.0f):depth<Source.Load((Layout.x+pixel)*BytesPerWord))
            {
                if (!selectOwner) Result.InterlockedMin(pixel*BytesPerWord,depth);
                else if (depth==Result.Load(pixel*BytesPerWord)) Result.InterlockedMin((Counts.w+pixel)*BytesPerWord,t.Key|((Quality.w&PriorityOrdering)!=0?SoftwarePriorityBit:0u));
            }
        }
    }
}
uint LinearLane(uint3 block,uint lane) { return (block.y*Dispatch.x+block.x)*Threads+lane; }
[numthreads(Threads,1,1)] void CSSeedDepth(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=LinearLane(block,lane); [branch] if (id>=Dispatch.y) return;
    uint depth=Source.Load((Layout.x+id)*BytesPerWord);
    [branch] if ((Quality.w&PreserveMain)!=0)
    { uint previous=Result.Load(id*BytesPerWord); Result.Store((Layout.w+Counts.z+id)*BytesPerWord,previous); depth=min(depth,previous); }
    Result.Store(id*BytesPerWord,depth);
}
[numthreads(Threads,1,1)] void CSRaster(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{ uint id=LinearLane(block,lane); [branch] if (id<Dispatch.y) VisitTriangle(id,Dispatch.z==3u); }
// Keep optional scratch reads inside real control flow: root UAV views have
// no bounds metadata, and flattened loads would exceed single-phase storage.
[numthreads(Threads,1,1)] void CSSeedOwner(uint3 block:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=LinearLane(block,lane); [branch] if (id>=Dispatch.y) return;
    uint ref=Source.Load((Layout.y+id)*BytesPerWord),owner=((ref&TriangleMask)<<1u)|(ref>>31u);
    uint minimum=Result.Load(id*BytesPerWord),captured=Source.Load((Layout.x+id)*BytesPerWord);
    [branch] if ((Quality.w&PreserveMain)!=0)
    {
        uint previousOwner=Result.Load((Counts.w+id)*BytesPerWord),previousDepth=Result.Load((Layout.w+Counts.z+id)*BytesPerWord);
        if (previousDepth==captured && (previousOwner&~SoftwarePriorityBit)==owner) owner=previousOwner;
        if (minimum!=captured) owner=InvalidOwner;
        if (previousDepth==minimum) owner=min(owner,previousOwner);
    }
    else if (minimum!=captured) owner=InvalidOwner;
    Result.Store((Counts.w+id)*BytesPerWord,owner);
}
#endif
