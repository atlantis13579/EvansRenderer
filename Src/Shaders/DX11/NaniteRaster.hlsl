#if defined(EVANS_NANITE_VISIBILITY_GEOMETRY)
// Decode consumers read the native indexed streams directly. The resulting
// world-space table is shared by coverage and deferred attribute reconstruction.
cbuffer AssemblyFrame : register(b0)
{ row_major float4x4 World,Normal; uint4 Range,Dispatch; };
ByteAddressBuffer Vertices : register(t0);
Buffer<uint> Indices : register(t1);
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
RWStructuredBuffer<WorldTriangle> Triangles : register(u0);
static const uint Threads=64,VertexWords=12,WordBytes=4,NormalWord=3,ColorWord=6,UVWord=10,Corners=3;
static const float NormalMinimumLength=1.0e-6f,SRGBBreak=0.04045f,SRGBLinearScale=12.92f;
static const float SRGBOffset=0.055f,SRGBScale=1.055f,SRGBExponent=2.4f;
float LinearColor(float encoded)
{ encoded=saturate(encoded); return encoded<=SRGBBreak?encoded/SRGBLinearScale:pow((encoded+SRGBOffset)/SRGBScale,SRGBExponent); }
float3 UnitNormal(float3 value)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    float3 direction=scale>0?value/scale:float3(0,1,0); float magnitude=length(direction);
    return scale*magnitude>NormalMinimumLength?direction/magnitude:float3(0,1,0);
}
[numthreads(Threads,1,1)] void CSVisibilityGeometry(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint local=(group.y*Dispatch.x+group.x)*Threads+lane;
    [branch] if (local>=Dispatch.y) return;
    uint baseIndex=Range.z+local*Corners;
    uint3 indices=uint3(Indices[baseIndex],Indices[baseIndex+1],Indices[baseIndex+2]);
    WorldTriangle output=(WorldTriangle)0; bool valid=all(indices<Range.w);
    [branch] if (valid)
    {
        float2 uv[Corners];
        [unroll] for (uint corner=0;corner<Corners;++corner)
        {
            uint address=indices[corner]*VertexWords*WordBytes;
            float3 p=asfloat(Vertices.Load3(address)),n=asfloat(Vertices.Load3(address+NormalWord*WordBytes));
            float4 color=asfloat(Vertices.Load4(address+ColorWord*WordBytes)); uv[corner]=asfloat(Vertices.Load2(address+UVWord*WordBytes));
            float4 position=mul(World,float4(p,1)); float3 normal=mul(Normal,float4(n,0)).xyz;
            valid=valid && all(isfinite(position)) && all(isfinite(normal)) && all(isfinite(color)) && all(isfinite(uv[corner]));
            output.Position[corner]=position; output.Normal[corner]=float4(UnitNormal(normal),0);
            output.Color[corner]=float4(LinearColor(color.r),LinearColor(color.g),LinearColor(color.b),color.a);
        }
        output.UVAB=float4(uv[0],uv[1]); output.UVC=float4(uv[2],0,0);
    }
    if (!valid) output=(WorldTriangle)0;
    [unroll] for (uint corner=0;corner<Corners;++corner) output.Position[corner].w=1;
    output.Source=uint4(Dispatch.z,local,0,0); Triangles[Range.x+local]=output;
}
#elif defined(EVANS_NANITE_GRAPHICS)
// Native hardware and export stages share the compute-projected clip table.
// Material reconstruction continues to use original homogeneous positions;
// snapping changes coverage only. Clipped triangles remain hardware-owned.
cbuffer RasterDraw : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images; };
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
StructuredBuffer<WorldTriangle> Geometry : register(t0);
ByteAddressBuffer Projected : register(t1);
ByteAddressBuffer Winners : register(t2);
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
struct HardwareVertex { float4 Position:SV_POSITION; nointerpolation uint Reference:TEXCOORD0; };
HardwareVertex VSHardware(uint vertex:SV_VertexID)
{
    HardwareVertex result=(HardwareVertex)0; uint local=vertex/TriangleCorners,id=Range.x+local;
    [branch] if (local>=Range.y || id>=Counts.z) return result;
    if ((Quality.w&PreserveMain)!=0) id=Range.x+Range.y-1u-local;
    [branch] if (Winners.Load((Layout.w+id)*WordBytes)!=HardwareRoute) return result;
    float4 a,b,c; ClipCorners(id,a,b,c); uint corner=vertex%TriangleCorners;
    result.Position=corner==0?a:corner==1?b:c;
    // Snap eligible shared corners even when the third corner needs clipping.
    if (Projectable(result.Position)) result.Position=Snap(result.Position);
    result.Reference=id+1u; return result;
}
struct RasterTargets { uint Reference:SV_TARGET0; float2 Bary:SV_TARGET1; };
RasterTargets PSHardware(HardwareVertex input,bool front:SV_IsFrontFace)
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
    RasterTargets result; result.Reference=input.Reference|(front?0u:BackFaceBit); result.Bary=weights.yz; return result;
}
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSExport(uint vertex:SV_VertexID)
{ float2 p=float2((vertex<<1u)&2u,vertex&2u); ScreenVertex result; result.Position=float4(p*float2(2,-2)+float2(-1,1),0,1); return result; }
struct ExportTargets { uint Reference:SV_TARGET0; float2 Bary:SV_TARGET1; float Depth:SV_DEPTH; };
ExportTargets PSExport(ScreenVertex input)
{
    uint pixel=uint(input.Position.y)*Counts.x+uint(input.Position.x),key=Winners.Load((Counts.w+pixel)*WordBytes),reference=(key&~SoftwarePriorityBit)>>1u;
    // Bound depth explicitly, avoiding a raw bytecode load directly into oDepth.
    ExportTargets result; result.Depth=saturate(asfloat(Winners.Load(pixel*WordBytes))); result.Reference=reference|((key&1u)!=0?BackFaceBit:0u); result.Bary=0;
    [branch] if (reference>0 && reference<=Counts.z) { float3 w,dx,dy; Weights(reference-1u,input.Position.xy,w,dx,dy); result.Bary=w.yz; }
    return result;
}
#elif defined(EVANS_NANITE_FRAME_PASS)
// GPU frame bridge: clear packed inputs, project the immutable world-space
// table, and capture hardware winners without mapping either resource to CPU.
cbuffer FrameBridge : register(b0)
{
    uint4 Counts,Layout,Dispatch,Quality;
    row_major float4x4 Camera;
    uint4 Range;
};
RWByteAddressBuffer Destination : register(u0);
static const uint WordBytes=4,RecordWords=16,Threads=64;
#if EVANS_NANITE_FRAME_PASS == 4
struct WorldTriangle { float4 Position[3],Normal[3],Color[3]; float4 UVAB,UVC; uint4 Source; };
StructuredBuffer<WorldTriangle> Geometry : register(t0);
ByteAddressBuffer ClusterDecisions : register(t3);
static const uint OcclusionRejectedFlag=8,FrustumRejectedState=1,ClipFlagsWord=12;
[numthreads(Threads,1,1)] void CSProject(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint local=(group.y*Dispatch.x+group.x)*Threads+lane;
    [branch] if (local>=Dispatch.y) return;
    uint id=Range.x+local,base=(Layout.z+id*RecordWords)*WordBytes;
    // Frustum-rejected instances may have no assembled world-table record.
    [branch] if (Range.w!=0) { [branch] if (ClusterDecisions.Load((Range.w-1)*WordBytes)==FrustumRejectedState)
    { Destination.Store4(base+ClipFlagsWord*WordBytes,uint4(Range.z|OcclusionRejectedFlag,0,0,0)); return; } }
    WorldTriangle item=Geometry[id];
    Destination.Store4(base,asuint(mul(Camera,item.Position[0])));
    Destination.Store4(base+4*WordBytes,asuint(mul(Camera,item.Position[1])));
    Destination.Store4(base+8*WordBytes,asuint(mul(Camera,item.Position[2])));
    uint flags=Range.z;
    // Preserve projected corners for first-phase winners during post export.
    [branch] if (Range.w!=0) { if (ClusterDecisions.Load((Range.w-1)*WordBytes)!=Quality.z) flags|=OcclusionRejectedFlag; }
    Destination.Store4(base+12*WordBytes,uint4(flags,0,0,0));
}
#elif EVANS_NANITE_FRAME_PASS == 5
Texture2D<float> HardwareDepth : register(t1);
Texture2D<uint> HardwareReference : register(t2);
[numthreads(Threads,1,1)] void CSCapture(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint pixel=(group.y*Dispatch.x+group.x)*Threads+lane;
    [branch] if (pixel>=Dispatch.y) return;
    int3 location=int3(pixel%Counts.x,pixel/Counts.x,0);
    Destination.Store((Layout.x+pixel)*WordBytes,asuint(HardwareDepth.Load(location)));
    Destination.Store((Layout.y+pixel)*WordBytes,HardwareReference.Load(location));
}
#else
[numthreads(Threads,1,1)] void CSClear(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{ uint word=(group.y*Dispatch.x+group.x)*Threads+lane; if (word<Dispatch.y) Destination.Store(word*WordBytes,0); }
#endif
#else
// D3D11 micropolygon rasterization: integer half spaces and two 32-bit atomic
// epochs replace Nanite's 64-bit depth/reference atomic (Karis et al., 2021).
// Depth is frozen before ownership. Hardware wins equal depth; compute ties
// choose the first triangle. Each kernel requires a dispatch boundary.
cbuffer RasterFrame : register(b0) { uint4 Counts, Layout, Dispatch, Quality; };
ByteAddressBuffer Input : register(t0);
RWByteAddressBuffer Output : register(u0);
static const uint WordBytes=4, TriangleWords=16, GroupWidth=64;
static const uint Rejected=0, Compute=1, Hardware=2, CullBack=1, ClockwiseFront=2, ForceHardware=4, OcclusionRejected=8;
static const uint PriorityOrdering=1u,PreserveMain=2u,SoftwarePriorityBit=0x80000000u;
static const uint FaceBit=0x80000000u, ReferenceMask=0x7fffffffu, NoOwner=0xffffffffu;
static const float MinimumNormal=1.175494351e-38f;
struct Setup { int2 A,B,C,Minimum,Maximum; float3 Z; int Area; uint Owner,Route; };
int Edge(int2 a,int2 b,int2 p) { int2 d=b-a,q=p-a; return d.x*q.y-d.y*q.x; }
bool Inclusive(int2 a,int2 b) { int2 d=b-a; return d.y<0 || (d.y==0 && d.x>0); }
bool Inside(int value,int2 a,int2 b) { return value>0 || (value==0 && Inclusive(a,b)); }
Setup Prepare(uint primitiveIndex)
{
    Setup s=(Setup)0; s.Route=Hardware; s.Area=1;
    uint at=(Layout.z+primitiveIndex*TriangleWords)*WordBytes;
    float4 a=asfloat(Input.Load4(at)),b=asfloat(Input.Load4(at+4*WordBytes)),c=asfloat(Input.Load4(at+8*WordBytes));
    uint flags=Input.Load(at+12*WordBytes);
    bool clipped=(flags&(ForceHardware|OcclusionRejected))!=0 || !all(isfinite(a)) || !all(isfinite(b)) || !all(isfinite(c)) || min(a.w,min(b.w,c.w))<MinimumNormal || any(abs(a.xy)>a.w) || any(abs(b.xy)>b.w) || any(abs(c.xy)>c.w) ||
        a.z<0 || a.z>a.w || b.z<0 || b.z>b.w || c.z<0 || c.z>c.w;
    // A single exit keeps FXC's aggregate definite-assignment analysis intact.
    // Branches protect projection and bounded edge products, including dead lanes.
    [branch] if (!clipped)
    {
        float scale=float(1u<<Quality.y); float2 viewport=float2(Counts.xy);
        s.A=int2(floor((a.xy/a.w*float2(0.5,-0.5)+0.5)*viewport*scale+0.5));
        s.B=int2(floor((b.xy/b.w*float2(0.5,-0.5)+0.5)*viewport*scale+0.5));
        s.C=int2(floor((c.xy/c.w*float2(0.5,-0.5)+0.5)*viewport*scale+0.5));
        int2 low=min(s.A,min(s.B,s.C)),high=max(s.A,max(s.B,s.C));
        [branch] if (all(high-low<=int(Quality.x)*int(scale)))
        {
            s.Area=Edge(s.A,s.B,s.C); s.Route=Rejected;
            bool back=(s.Area>0)!=((flags&ClockwiseFront)!=0);
            [branch] if (s.Area!=0 && !(back && (flags&CullBack)!=0))
            {
                s.Owner=((primitiveIndex+1u)<<1u)|(back?1u:0u); s.Z=float3(a.z/a.w,b.z/b.w,c.z/c.w);
                if (s.Area<0) { int2 temporary=s.B; s.B=s.C; s.C=temporary; float z=s.Z.y; s.Z.y=s.Z.z; s.Z.z=z; s.Area=-s.Area; }
                s.Minimum=max(int2(0,0),int2(ceil((float2(low)-scale*0.5)/scale)));
                s.Maximum=min(int2(Counts.xy)-1,int2(floor((float2(high)-scale*0.5)/scale)));
                s.Route=any(s.Minimum>s.Maximum)?Rejected:Compute;
            }
        }
    }
    if ((flags&OcclusionRejected)!=0) s.Route=Rejected;
    return s;
}
void Raster(uint primitiveIndex,bool ownership)
{
    Setup s=Prepare(primitiveIndex); uint route=s.Route;
    if (!ownership) Output.Store((Layout.w+primitiveIndex)*WordBytes,route);
    [branch] if (Dispatch.z==4u || route!=Compute) return;
    int scale=int(1u<<Quality.y),halfSample=scale>>1;
    for (int y=s.Minimum.y;y<=s.Maximum.y;++y) for (int x=s.Minimum.x;x<=s.Maximum.x;++x)
    {
        int2 p=int2(x,y)*scale+halfSample;
        int ea=Edge(s.B,s.C,p),eb=Edge(s.C,s.A,p),ec=Edge(s.A,s.B,p);
        [branch] if (Inside(ea,s.B,s.C) && Inside(eb,s.C,s.A) && Inside(ec,s.A,s.B))
        {
            float depth=s.Z.x+(s.Z.y-s.Z.x)*(float(eb)/float(s.Area))+(s.Z.z-s.Z.x)*(float(ec)/float(s.Area));
            depth=saturate(depth); uint bits=depth<=0?0u:asuint(depth),pixel=uint(y)*Counts.x+uint(x);
            // Standalone seeds retain strict LESS. Frame keys arbitrate equal
            // depth across phases; the far clear never acquires an owner.
            [branch] if ((Quality.w&PriorityOrdering)!=0?bits<asuint(1.0f):bits<Input.Load((Layout.x+pixel)*WordBytes))
            {
                if (!ownership) Output.InterlockedMin(pixel*WordBytes,bits);
                else if (bits==Output.Load(pixel*WordBytes)) Output.InterlockedMin((Counts.w+pixel)*WordBytes,s.Owner|((Quality.w&PriorityOrdering)!=0?SoftwarePriorityBit:0u));
            }
        }
    }
}
uint Invocation(uint3 group,uint lane) { return (group.y*Dispatch.x+group.x)*GroupWidth+lane; }
[numthreads(GroupWidth,1,1)] void CSSeedDepth(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint i=Invocation(group,lane); if (i>=Dispatch.y) return;
    uint depth=Input.Load((Layout.x+i)*WordBytes);
    [branch] if ((Quality.w&PreserveMain)!=0)
    { uint previous=Output.Load(i*WordBytes); Output.Store((Layout.w+Counts.z+i)*WordBytes,previous); depth=min(depth,previous); }
    Output.Store(i*WordBytes,depth);
}
[numthreads(GroupWidth,1,1)] void CSRaster(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{ uint i=Invocation(group,lane); [branch] if (i<Dispatch.y) Raster(i,Dispatch.z==3u); }
// Keep optional scratch reads inside real control flow: root UAV views have
// no bounds metadata, and flattened loads would exceed single-phase storage.
[numthreads(GroupWidth,1,1)] void CSSeedOwner(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint i=Invocation(group,lane); [branch] if (i>=Dispatch.y) return;
    uint encoded=Input.Load((Layout.y+i)*WordBytes),key=((encoded&ReferenceMask)<<1u)|(encoded>>31u);
    uint depth=Output.Load(i*WordBytes),captured=Input.Load((Layout.x+i)*WordBytes);
    [branch] if ((Quality.w&PreserveMain)!=0)
    {
        uint previous=Output.Load((Counts.w+i)*WordBytes),previousDepth=Output.Load((Layout.w+Counts.z+i)*WordBytes);
        if (previousDepth==captured && (previous&~SoftwarePriorityBit)==key) key=previous;
        if (depth!=captured) key=NoOwner;
        if (previousDepth==depth) key=min(key,previous);
    }
    else if (depth!=captured) key=NoOwner;
    Output.Store((Counts.w+i)*WordBytes,key);
}
#endif
