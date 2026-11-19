#if defined(EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE)
// Every filter tap resolves its own page and coarse fallback. Comparing first
// and then interpolating avoids leaking unrelated physical neighbors at seams.
cbuffer LookupFrame : register(b0)
{
    row_major float4x4 LightViewProjection;
    uint4 VirtualExtent,PhysicalExtent,Filtering;
    float4 Reserved;
};
Texture2D<float> PhysicalDepth : register(t0);
Texture2D<float> PageDirectory : register(t1);
StructuredBuffer<float4> Queries : register(t2);
RWStructuredBuffer<float4> Results : register(u0);
static const uint LookupThreads=64,PageBorder=1;
static const float TexelCenter=0.5;
float3 LookupDepth(float2 uv,uint requestedMip)
{
    uint side=VirtualExtent.x/VirtualExtent.y,offset=0;
    [loop] for (uint mip=0;mip<VirtualExtent.z;++mip)
    {
        if (mip>=requestedMip)
        {
            uint resolution=side*VirtualExtent.y;
            uint2 pixel=min(uint2(uv*resolution),resolution-1),page=pixel/VirtualExtent.y;
            uint address=offset+page.y*side+page.x;
            uint entry=uint(round(PageDirectory.Load(int3(address%VirtualExtent.w,address/VirtualExtent.w,0))*PhysicalExtent.w));
            if (entry!=0)
            {
                uint slot=entry-1,stride=VirtualExtent.y+PageBorder*2;
                uint2 physical=uint2(slot%PhysicalExtent.z,slot/PhysicalExtent.z)*stride+PageBorder+pixel%VirtualExtent.y;
                return float3(PhysicalDepth.Load(int3(physical,0)),mip,entry);
            }
        }
        offset+=side*side; side/=2;
    }
    return float3(1,VirtualExtent.z-1,0);
}
float CompareTap(int2 pixel,uint mip,float receiver)
{
    uint resolution=VirtualExtent.x>>mip;
    float2 uv=(float2(clamp(pixel,0,int(resolution)-1))+TexelCenter)/resolution;
    return receiver<=LookupDepth(uv,mip).x?1:0;
}
[numthreads(LookupThreads,1,1)]
void CSLookup(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*Filtering.w; if (index>=Filtering.z) return;
    float4 query=Queries[index];
    if (any(query.xy<0) || any(query.xy>1) || query.z<0 || query.z>1) { Results[index]=float4(1,1,-1,0); return; }
    float3 center=LookupDepth(query.xy,uint(query.w)); float visibility=query.z<=center.x?1:0;
    if (Filtering.x!=0)
    {
        uint mip=uint(center.y); float2 texel=query.xy*(VirtualExtent.x>>mip)-TexelCenter;
        int2 origin=int2(floor(texel)); float2 weight=frac(texel); visibility=0;
        [loop] for (int y=-int(Filtering.x);y<=int(Filtering.x);++y)
            [loop] for (int x=-int(Filtering.x);x<=int(Filtering.x);++x)
            {
                int2 p=origin+int2(x,y);
                float a=CompareTap(p,mip,query.z),b=CompareTap(p+int2(1,0),mip,query.z);
                float c=CompareTap(p+int2(0,1),mip,query.z),d=CompareTap(p+1,mip,query.z);
                visibility+=lerp(lerp(a,b,weight.x),lerp(c,d,weight.x),weight.y);
            }
        visibility/=Filtering.y;
    }
    Results[index]=float4(visibility,center);
}
#elif defined(EVANS_VIRTUAL_SHADOW_RASTER_STAGE)
// Copy-on-write depth tiles. Coverage noise uses logical mip texels, so cache
// repacking never changes alpha coverage along a tile or its sampled gutter.
cbuffer AtlasDraw : register(b0)
{
    row_major float4x4 WorldViewProjection;
    float4 Material;
    uint4 Textures;
    float4 Origins;
    uint4 Extent;
    row_major float4x4 VegetationWorld, VegetationNormalWorld;
    float4 Wind, Gust, VegetationHistory;
};
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
float3 VegetationOffset(float3 world, float height, float4 wind, float4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return float3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
float4 VegetationPoint(float3 position,float2 uv)
{
    if (Gust.w<0.5) return float4(position,1);
    float3 world=mul(VegetationWorld,float4(position,1)).xyz;
    float height=max(0.0,Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,Wind,Gust,VegetationHistory.y,Wind.z);
    return float4(position+mul(transpose((float3x3)VegetationNormalWorld),offset),1);
}
Texture2D<float4> DiffuseImage : register(t0);
Texture2D<float4> OpacityImage : register(t1);
Texture2D<float> ParentDepth : register(t2);
SamplerState MaterialSampler : register(s0);
RWStructuredBuffer<float> AtlasReadback : register(u0);
static const float EnabledThreshold=0.5,NoiseScale=52.9829189;
static const float2 NoiseDirection=float2(0.06711056,0.00583715);
static const float FullscreenReach=3;
static const uint AtlasReadThreads=64;
struct DepthVertex { float3 Position:POSITION; float4 Color:COLOR0; float2 UV:TEXCOORD0; };
struct DepthVarying { float4 Position:SV_POSITION; float2 UV:TEXCOORD0; float Alpha:COLOR0; };
DepthVarying VSDepth(DepthVertex vertex)
{
    DepthVarying result; result.Position=mul(WorldViewProjection,VegetationPoint(vertex.Position,vertex.UV));
    result.UV=float2(vertex.UV.x,1-vertex.UV.y); result.Alpha=vertex.Color.a; return result;
}
void PSDepth(DepthVarying input)
{
    float coverage=input.Alpha*Material.x;
    [branch] if (Textures.x!=0) coverage*=DiffuseImage.Sample(MaterialSampler,input.UV).a;
    [branch] if (Textures.y!=0) coverage*=OpacityImage.Sample(MaterialSampler,input.UV).r;
    coverage=saturate(coverage);
    if (Material.y>0) clip(coverage-Material.y);
    else if (Material.z>EnabledThreshold)
    {
        float2 virtualPixel=floor(input.Position.xy-Origins.xy)+Origins.zw;
        float threshold=frac(NoiseScale*frac(dot(virtualPixel,NoiseDirection)));
        if (coverage<=0 || coverage<threshold) discard;
    }
}
float4 VSCopy(uint vertex:SV_VertexID):SV_POSITION
{ return float4(vertex==0?float2(-1,-1):vertex==1?float2(-1,FullscreenReach):float2(FullscreenReach,-1),0,1); }
float PSCopy(float4 position:SV_POSITION):SV_Depth
{
    int2 local=int2(floor(position.xy)-Origins.xy);
    return ParentDepth.Load(int3(int2(Origins.zw)+local,0));
}
[numthreads(AtlasReadThreads,1,1)]
void CSRead(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*Extent.z; if (index>=Extent.x*Extent.y) return;
    AtlasReadback[index]=ParentDepth.Load(int3(index%Extent.x,index/Extent.x,0));
}
#else
// Paged depth demand, independent of variance shadow moments and dense CSM.
// Pixel-plane derivatives select virtual mip resolution; every marked page
// requests its ancestors so a bounded physical pool always has a coarse fallback.
// Karis, "A Deep Dive into Nanite Virtualized Geometry", SIGGRAPH 2021,
// Virtual Shadow Maps section. No Unreal shader implementation is included.
cbuffer VirtualShadowFrame : register(b0)
{
    row_major float4x4 CameraInverse,LightViewProjection;
    uint4 Extent,Limits;
    float4 Quality,Sampling;
};
StructuredBuffer<float> ReceiverDepth : register(t0);
RWStructuredBuffer<uint> RequestedPages : register(u0);
static const uint DemandGroupWidth=64;
static const float PixelCenter=0.5,ClearDepth=1;
bool ProjectReceiver(float2 pixel,float depth,out float3 ndc)
{
    float2 uv=(float2(pixel)+PixelCenter)/float2(Extent.xy);
    float4 world=mul(CameraInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    ndc=0; if (abs(world.w)<=Quality.y) return false;
    float4 light=mul(LightViewProjection,float4(world.xyz/world.w,1));
    if (light.w<=Quality.y) return false;
    ndc=light.xyz/light.w; return all(isfinite(ndc));
}
uint LevelOffset(uint mip)
{
    uint side=Extent.z/Extent.w,offset=0;
    for (uint level=0;level<mip;++level) { offset+=side*side; side/=2; }
    return offset;
}
[numthreads(DemandGroupWidth,1,1)]
void CSClear(uint3 thread : SV_DispatchThreadID)
{ uint index=thread.x+thread.y*Limits.w; if (index<Limits.y) RequestedPages[index]=0; }
[numthreads(DemandGroupWidth,1,1)]
void CSMark(uint3 thread : SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*Limits.w; if (index>=Extent.x*Extent.y) return;
    uint2 pixel=uint2(index%Extent.x,index/Extent.x); float depth=ReceiverDepth[index];
    if (depth>=ClearDepth) return;
    float3 center,horizontal,vertical;
    if (!ProjectReceiver(pixel,depth,center) || any(abs(center.xy)>1) || center.z<0 || center.z>1 ||
        !ProjectReceiver(float2(pixel)+float2(Sampling.x,0),depth,horizontal) || !ProjectReceiver(float2(pixel)+float2(0,Sampling.y),depth,vertical)) return;
    float footprint=max(length(horizontal.xy-center.xy),length(vertical.xy-center.xy))*PixelCenter*Extent.z;
    uint mip=uint(clamp(floor(log2(max(1,footprint))+Quality.x+Quality.w),0,float(Limits.x-1)));
    uint side=(Extent.z/Extent.w)>>mip;
    float2 uv=center.xy*float2(PixelCenter,-PixelCenter)+PixelCenter;
    float padding=(Limits.z+PixelCenter)/Extent.w;
    int2 first=clamp(int2(floor(uv*side-padding-Quality.z)),int2(0,0),int2(side-1,side-1));
    int2 last=clamp(int2(floor(uv*side+padding+Quality.z)),int2(0,0),int2(side-1,side-1));
    for (int y=first.y;y<=last.y;++y) for (int x=first.x;x<=last.x;++x)
    {
        uint2 page=uint2(x,y); uint levelSide=side,offset=LevelOffset(mip);
        for (uint level=mip;level<Limits.x;++level)
        {
            InterlockedOr(RequestedPages[offset+page.y*levelSide+page.x],1);
            offset+=levelSide*levelSide; page/=2; levelSide/=2;
        }
    }
}

#endif
