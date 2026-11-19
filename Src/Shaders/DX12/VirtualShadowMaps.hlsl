#if defined(EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE)
// The page directory shares a depth-typed binding with mutually exclusive CSM.
// Bounded normalized entries round back to exact one-based physical slots.
cbuffer VirtualLookupParameters : register(b0)
{
    row_major float4x4 LightProjection;
    uint4 VirtualSize,PhysicalSize,Filter;
    float4 Padding;
};
Texture2D<float> AtlasDepth : register(t0);
Texture2D<float> DirectoryDepth : register(t1);
StructuredBuffer<float4> LookupSamples : register(t2);
RWStructuredBuffer<float4> LookupValues : register(u0);
static const uint LookupGroupWidth=64,TileGutter=1;
static const float PixelCenter=0.5;
float3 ResolveDepth(float2 coordinates,uint firstMip)
{
    uint pages=VirtualSize.x/VirtualSize.y,base=0;
    [loop] for (uint level=0;level<VirtualSize.z;++level)
    {
        if (level>=firstMip)
        {
            uint2 texel=min(uint2(coordinates*(pages*VirtualSize.y)),pages*VirtualSize.y-1);
            uint2 page=texel/VirtualSize.y; uint id=base+page.y*pages+page.x;
            uint mapping=uint(round(DirectoryDepth.Load(int3(id%VirtualSize.w,id/VirtualSize.w,0))*PhysicalSize.w));
            if (mapping>0)
            {
                uint slot=mapping-1,pitch=VirtualSize.y+TileGutter*2;
                uint2 location=uint2(slot%PhysicalSize.z,slot/PhysicalSize.z)*pitch+TileGutter+texel%VirtualSize.y;
                return float3(AtlasDepth.Load(int3(location,0)),level,mapping);
            }
        }
        base+=pages*pages; pages>>=1;
    }
    return float3(1,VirtualSize.z-1,0);
}
float ResolveComparison(int2 texel,uint level,float comparison)
{
    uint extent=VirtualSize.x>>level;
    float2 coordinates=(float2(clamp(texel,0,int(extent)-1))+PixelCenter)/extent;
    return comparison<=ResolveDepth(coordinates,level).x?1:0;
}
[numthreads(LookupGroupWidth,1,1)]
void CSLookup(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Filter.w; if (index>=Filter.z) return;
    float4 query=LookupSamples[index];
    if (any(query.xy<0) || any(query.xy>1) || query.z<0 || query.z>1) { LookupValues[index]=float4(1,1,-1,0); return; }
    float3 selected=ResolveDepth(query.xy,uint(query.w)); float lit=query.z<=selected.x?1:0;
    if (Filter.x>0)
    {
        uint level=uint(selected.y); float2 position=query.xy*(VirtualSize.x>>level)-PixelCenter;
        int2 base=int2(floor(position)); float2 blend=frac(position); lit=0;
        [loop] for (int row=-int(Filter.x);row<=int(Filter.x);++row)
            [loop] for (int column=-int(Filter.x);column<=int(Filter.x);++column)
            {
                int2 p=base+int2(column,row);
                float lower=lerp(ResolveComparison(p,level,query.z),ResolveComparison(p+int2(1,0),level,query.z),blend.x);
                float upper=lerp(ResolveComparison(p+int2(0,1),level,query.z),ResolveComparison(p+1,level,query.z),blend.x);
                lit+=lerp(lower,upper,blend.y);
            }
        lit/=Filter.y;
    }
    LookupValues[index]=float4(lit,selected);
}
#elif defined(EVANS_VIRTUAL_SHADOW_RASTER_STAGE)
cbuffer AtlasParameters : register(b0)
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
Texture2D<float4> DiffuseCoverage : register(t0);
Texture2D<float4> OpacityCoverage : register(t1);
Texture2D<float> SourceAtlas : register(t2);
SamplerState CoverageSampler : register(s0);
RWStructuredBuffer<float> DepthFeedback : register(u0);
static const float FlagEnabled=0.5,InterleavedScale=52.9829189;
static const float2 InterleavedDirection=float2(0.06711056,0.00583715);
static const float TriangleReach=3;
static const uint ReadbackThreads=64;
struct CasterVertex { float3 Position:POSITION; float4 Color:COLOR0; float2 TexCoord:TEXCOORD0; };
struct CasterFragment { float4 Position:SV_POSITION; float2 UV:TEXCOORD0; float Coverage:COLOR0; };
CasterFragment VSDepth(CasterVertex vertex)
{
    CasterFragment output; output.Position=mul(WorldViewProjection,VegetationPoint(vertex.Position,vertex.TexCoord));
    output.UV=float2(vertex.TexCoord.x,1-vertex.TexCoord.y); output.Coverage=vertex.Color.a; return output;
}
void PSDepth(CasterFragment fragment)
{
    float opacity=fragment.Coverage*Material.x;
    [branch] if (Textures.x!=0) opacity*=DiffuseCoverage.Sample(CoverageSampler,fragment.UV).a;
    [branch] if (Textures.y!=0) opacity*=OpacityCoverage.Sample(CoverageSampler,fragment.UV).r;
    opacity=saturate(opacity);
    if (Material.y>0) { if (opacity<Material.y) discard; }
    else if (Material.z>FlagEnabled)
    {
        // Atlas slot indices are deliberately absent from the coverage seed.
        float2 virtualTexel=floor(fragment.Position.xy-Origins.xy)+Origins.zw;
        float threshold=frac(InterleavedScale*frac(dot(virtualTexel,InterleavedDirection)));
        if (opacity<=0 || opacity<threshold) discard;
    }
}
float4 VSCopy(uint vertex:SV_VertexID):SV_POSITION
{
    float2 corner=vertex==0?float2(-1,-1):vertex==1?float2(-1,TriangleReach):float2(TriangleReach,-1);
    return float4(corner,0,1);
}
float PSCopy(float4 fragment:SV_POSITION):SV_Depth
{
    int2 source=int2(Origins.zw)+int2(floor(fragment.xy)-Origins.xy);
    return SourceAtlas.Load(int3(source,0));
}
[numthreads(ReadbackThreads,1,1)]
void CSRead(uint3 invocation:SV_DispatchThreadID)
{
    uint address=invocation.x+invocation.y*Extent.z; if (address>=Extent.x*Extent.y) return;
    DepthFeedback[address]=SourceAtlas.Load(int3(address%Extent.x,address/Extent.x,0));
}
#else
// Native D3D12 virtual-depth demand. Bounded structured views supply source
// hardware depth and an independent page directory. Clear and marking are
// separate dispatches with an explicit UAV ordering dependency in the host.
cbuffer PageDemandFrame : register(b0)
{
    row_major float4x4 CameraInverse,LightViewProjection;
    uint4 Extent,Limits;
    float4 Quality,Sampling;
};
StructuredBuffer<float> HardwareDepth : register(t0);
RWStructuredBuffer<uint> Demand : register(u0);
static const uint PageDemandThreads=64;
static const float HalfPixel=0.5,FarDepth=1;
bool LightCoordinates(float2 rasterPixel,float depth,out float3 position)
{
    float2 sample=(float2(rasterPixel)+HalfPixel)/float2(Extent.xy);
    float4 world=mul(CameraInverse,float4(sample*float2(2,-2)+float2(-1,1),depth,1));
    position=0;
    if (abs(world.w)<=Quality.y) return false;
    float4 clip=mul(LightViewProjection,float4(world.xyz/world.w,1));
    if (clip.w<=Quality.y) return false;
    position=clip.xyz/clip.w; return all(isfinite(position));
}
[numthreads(PageDemandThreads,1,1)]
void CSClear(uint3 invocation : SV_DispatchThreadID)
{ uint index=invocation.x+invocation.y*Limits.w; if (index<Limits.y) Demand[index]=0; }
[numthreads(PageDemandThreads,1,1)]
void CSMark(uint3 invocation : SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Limits.w; if (index>=Extent.x*Extent.y) return;
    float depth=HardwareDepth[index]; if (depth>=FarDepth) return;
    uint2 rasterPixel=uint2(index%Extent.x,index/Extent.x); float3 center,alongX,alongY;
    if (!LightCoordinates(rasterPixel,depth,center) || any(abs(center.xy)>1) || center.z<0 || center.z>1) return;
    if (!LightCoordinates(float2(rasterPixel)+float2(Sampling.x,0),depth,alongX) || !LightCoordinates(float2(rasterPixel)+float2(0,Sampling.y),depth,alongY)) return;
    float projectedPixel=max(length(alongX.xy-center.xy),length(alongY.xy-center.xy))*HalfPixel*Extent.z;
    uint chosen=uint(clamp(floor(log2(max(1,projectedPixel))+Quality.x+Quality.w),0,float(Limits.x-1)));
    uint pageSide=Extent.z/Extent.w,tableStart=0;
    for (uint level=0;level<chosen;++level) { tableStart+=pageSide*pageSide; pageSide/=2; }
    float2 pagePosition=(center.xy*float2(HalfPixel,-HalfPixel)+HalfPixel)*pageSide;
    float radius=(Limits.z+HalfPixel)/Extent.w;
    int2 minimum=clamp(int2(floor(pagePosition-radius-Quality.z)),int2(0,0),int2(pageSide-1,pageSide-1));
    int2 maximum=clamp(int2(floor(pagePosition+radius+Quality.z)),int2(0,0),int2(pageSide-1,pageSide-1));
    for (int row=minimum.y;row<=maximum.y;++row) for (int column=minimum.x;column<=maximum.x;++column)
    {
        uint2 location=uint2(column,row); uint side=pageSide,offset=tableStart;
        for (uint mip=chosen;mip<Limits.x;++mip)
        {
            InterlockedOr(Demand[offset+location.y*side+location.x],1);
            offset+=side*side; side/=2; location/=2;
        }
    }
}

#endif
