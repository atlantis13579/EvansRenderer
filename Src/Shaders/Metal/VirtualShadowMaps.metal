#include <metal_stdlib>
using namespace metal;

struct VirtualLookupFrame
{
    float4 LightRows[4];
    uint4 VirtualExtent,PhysicalExtent,Filtering;
    float4 Reserved;
};
constant uint VirtualTileBorder=1;
constant float VirtualPixelCenter=0.5;
float3 virtualStoredDepth(float2 uv,uint requestedMip,constant VirtualLookupFrame& frame,
    depth2d<float,access::read> atlas,depth2d<float,access::read> directory)
{
    uint side=frame.VirtualExtent.x/frame.VirtualExtent.y,offset=0;
    for (uint mip=0;mip<frame.VirtualExtent.z;++mip)
    {
        if (mip>=requestedMip)
        {
            uint resolution=side*frame.VirtualExtent.y;
            uint2 pixel=min(uint2(uv*float(resolution)),uint2(resolution-1)),page=pixel/frame.VirtualExtent.y;
            uint address=offset+page.y*side+page.x;
            uint entry=uint(round(directory.read(uint2(address%frame.VirtualExtent.w,address/frame.VirtualExtent.w))*float(frame.PhysicalExtent.w)));
            if (entry!=0)
            {
                uint slot=entry-1,stride=frame.VirtualExtent.y+VirtualTileBorder*2;
                uint2 physical=uint2(slot%frame.PhysicalExtent.z,slot/frame.PhysicalExtent.z)*stride+VirtualTileBorder+pixel%frame.VirtualExtent.y;
                return float3(atlas.read(physical),float(mip),float(entry));
            }
        }
        offset+=side*side; side/=2;
    }
    return float3(1,float(frame.VirtualExtent.z-1),0);
}
float virtualComparison(int2 texel,uint mip,float comparison,constant VirtualLookupFrame& frame,
    depth2d<float,access::read> atlas,depth2d<float,access::read> directory)
{
    uint resolution=frame.VirtualExtent.x>>mip;
    float2 uv=(float2(clamp(texel,int2(0),int2(resolution-1)))+VirtualPixelCenter)/float(resolution);
    return comparison<=virtualStoredDepth(uv,mip,frame,atlas,directory).x?1.0:0.0;
}
kernel void virtualShadowLookup(constant VirtualLookupFrame& frame [[buffer(0)]],
    device const float4* queries [[buffer(1)]],device float4* results [[buffer(2)]],
    depth2d<float,access::read> atlas [[texture(0)]],depth2d<float,access::read> directory [[texture(1)]],
    uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Filtering.w; if (index>=frame.Filtering.z) return;
    float4 query=queries[index];
    if (any(query.xy<0) || any(query.xy>1) || query.z<0 || query.z>1) { results[index]=float4(1,1,-1,0); return; }
    float3 center=virtualStoredDepth(query.xy,uint(query.w),frame,atlas,directory);
    float visibility=query.z<=center.x?1.0:0.0;
    if (frame.Filtering.x!=0)
    {
        uint mip=uint(center.y); float2 texel=query.xy*float(frame.VirtualExtent.x>>mip)-VirtualPixelCenter;
        int2 origin=int2(floor(texel)); float2 weight=fract(texel); visibility=0;
        for (int y=-int(frame.Filtering.x);y<=int(frame.Filtering.x);++y)
            for (int x=-int(frame.Filtering.x);x<=int(frame.Filtering.x);++x)
            {
                int2 p=origin+int2(x,y);
                float a=virtualComparison(p,mip,query.z,frame,atlas,directory),b=virtualComparison(p+int2(1,0),mip,query.z,frame,atlas,directory);
                float c=virtualComparison(p+int2(0,1),mip,query.z,frame,atlas,directory),d=virtualComparison(p+1,mip,query.z,frame,atlas,directory);
                visibility+=mix(mix(a,b,weight.x),mix(c,d,weight.x),weight.y);
            }
        visibility/=float(frame.Filtering.y);
    }
    results[index]=float4(visibility,center);
}

// Independent MSL receiver-plane demand. Atomics are relaxed because native
// encoder ordering separates clear, marking, and the directory's next consumer.
struct VirtualShadowRequestFrame
{
    float4 CameraInverseRows[4],LightViewProjectionRows[4];
    uint4 Extent,Limits;
    float4 Quality,Sampling;
};
constant float PixelCenter=0.5f,ClearDepth=1.0f;
bool projectReceiver(constant VirtualShadowRequestFrame& frame,float2 pixel,float depth,thread float3& ndc)
{
    float2 uv=(float2(pixel)+PixelCenter)/float2(frame.Extent.xy);
    float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 world=float4(dot(frame.CameraInverseRows[0],clip),dot(frame.CameraInverseRows[1],clip),dot(frame.CameraInverseRows[2],clip),dot(frame.CameraInverseRows[3],clip));
    ndc=0; if (abs(world.w)<=frame.Quality.y) return false; world=float4(world.xyz/world.w,1);
    float4 light=float4(dot(frame.LightViewProjectionRows[0],world),dot(frame.LightViewProjectionRows[1],world),dot(frame.LightViewProjectionRows[2],world),dot(frame.LightViewProjectionRows[3],world));
    if (light.w<=frame.Quality.y) return false;
    ndc=light.xyz/light.w; return all(isfinite(ndc));
}
kernel void virtualShadowClear(constant VirtualShadowRequestFrame& frame [[buffer(0)]],device atomic_uint* requests [[buffer(2)]],uint2 invocation [[thread_position_in_grid]])
{ uint index=invocation.x+invocation.y*frame.Limits.w; if (index<frame.Limits.y) atomic_store_explicit(requests+index,0u,memory_order_relaxed); }
kernel void virtualShadowMark(constant VirtualShadowRequestFrame& frame [[buffer(0)]],device const float* depths [[buffer(1)]],
    device atomic_uint* requests [[buffer(2)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Limits.w; if (index>=frame.Extent.x*frame.Extent.y) return;
    float depth=depths[index]; if (depth>=ClearDepth) return;
    uint2 pixel=uint2(index%frame.Extent.x,index/frame.Extent.x); float3 center,neighborX,neighborY;
    if (!projectReceiver(frame,float2(pixel),depth,center) || any(abs(center.xy)>1) || center.z<0 || center.z>1) return;
    if (!projectReceiver(frame,float2(pixel)+float2(frame.Sampling.x,0),depth,neighborX) || !projectReceiver(frame,float2(pixel)+float2(0,frame.Sampling.y),depth,neighborY)) return;
    float footprint=max(length(neighborX.xy-center.xy),length(neighborY.xy-center.xy))*PixelCenter*float(frame.Extent.z);
    uint mip=uint(clamp(floor(log2(max(1.0f,footprint))+frame.Quality.x+frame.Quality.w),0.0f,float(frame.Limits.x-1)));
    uint side=frame.Extent.z/frame.Extent.w,offset=0;
    for (uint level=0;level<mip;++level) { offset+=side*side; side/=2; }
    float2 point=(center.xy*float2(PixelCenter,-PixelCenter)+PixelCenter)*float(side);
    float padding=(float(frame.Limits.z)+PixelCenter)/float(frame.Extent.w);
    int2 first=clamp(int2(floor(point-padding-frame.Quality.z)),int2(0),int2(side-1));
    int2 last=clamp(int2(floor(point+padding+frame.Quality.z)),int2(0),int2(side-1));
    for (int y=first.y;y<=last.y;++y) for (int x=first.x;x<=last.x;++x)
    {
        uint2 coordinate=uint2(x,y); uint parentSide=side,parentOffset=offset;
        for (uint level=mip;level<frame.Limits.x;++level)
        {
            atomic_fetch_or_explicit(requests+parentOffset+coordinate.y*parentSide+coordinate.x,1u,memory_order_relaxed);
            parentOffset+=parentSide*parentSide; parentSide/=2; coordinate/=2;
        }
    }
}

// Native immutable atlas rendering and point depth transfer. The raster block
// matches the host's vegetation-aware draw ABI and carries logical noise origins.
struct VirtualAtlasDraw
{
    float4 ProjectionRows[4];
    float4 Material;
    uint4 Textures;
    float4 Origins;
    uint4 Extent;
    float4 VegetationWorld[4], VegetationNormalWorld[4];
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
float4 VegetationPoint(float3 position,float2 uv,constant VirtualAtlasDraw& draw)
{
    if (draw.Gust.w<0.5) return float4(position,1);
    float4 local=float4(position,1);
    float3 world=float3(dot(draw.VegetationWorld[0],local),dot(draw.VegetationWorld[1],local),dot(draw.VegetationWorld[2],local));
    float height=max(0.0f,draw.Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,draw.Wind,draw.Gust,draw.VegetationHistory.y,draw.Wind.z);
    return float4(position+draw.VegetationNormalWorld[0].xyz*offset.x+draw.VegetationNormalWorld[1].xyz*offset.y+draw.VegetationNormalWorld[2].xyz*offset.z,1);
}
struct VirtualCasterVertex { packed_float3 Position,Normal; packed_float4 Color; packed_float2 UV; };
struct VirtualCasterVarying { float4 Position [[position]]; float2 UV; float Alpha; };
constant float AtlasFlagThreshold=0.5f,AtlasNoiseScale=52.9829189f;
constant float2 AtlasNoiseDirection=float2(0.06711056f,0.00583715f);
vertex VirtualCasterVarying virtualAtlasDepthVertex(device const VirtualCasterVertex* vertices [[buffer(0)]],
    constant VirtualAtlasDraw& draw [[buffer(1)]],uint id [[vertex_id]])
{
    float4 point=VegetationPoint(float3(vertices[id].Position),float2(vertices[id].UV),draw); VirtualCasterVarying result;
    result.Position=float4(dot(draw.ProjectionRows[0],point),dot(draw.ProjectionRows[1],point),dot(draw.ProjectionRows[2],point),dot(draw.ProjectionRows[3],point));
    result.UV=float2(vertices[id].UV.x,1.0f-vertices[id].UV.y); result.Alpha=vertices[id].Color.w; return result;
}
fragment void virtualAtlasDepthFragment(VirtualCasterVarying input [[stage_in]],constant VirtualAtlasDraw& draw [[buffer(1)]],
    texture2d<float> diffuse [[texture(0)]],texture2d<float> opacity [[texture(1)]],sampler materialSampler [[sampler(0)]])
{
    float coverage=input.Alpha*draw.Material.x;
    if (draw.Textures.x!=0) coverage*=diffuse.sample(materialSampler,input.UV).a;
    if (draw.Textures.y!=0) coverage*=opacity.sample(materialSampler,input.UV).r;
    coverage=saturate(coverage);
    if (draw.Material.y>0) { if (coverage<draw.Material.y) discard_fragment(); }
    else if (draw.Material.z>AtlasFlagThreshold)
    {
        float2 logicalTexel=floor(input.Position.xy-draw.Origins.xy)+draw.Origins.zw;
        float threshold=fract(AtlasNoiseScale*fract(dot(logicalTexel,AtlasNoiseDirection)));
        if (coverage<=0 || coverage<threshold) discard_fragment();
    }
}
struct VirtualCopyVarying { float4 Position [[position]]; };
vertex VirtualCopyVarying virtualAtlasCopyVertex(uint id [[vertex_id]])
{
    constexpr float TriangleReach=3;
    float2 corner=id==0?float2(-1):id==1?float2(-1,TriangleReach):float2(TriangleReach,-1);
    VirtualCopyVarying result; result.Position=float4(corner,0,1); return result;
}
struct VirtualCopyDepth { float Depth [[depth(any)]]; };
fragment VirtualCopyDepth virtualAtlasCopyFragment(VirtualCopyVarying input [[stage_in]],constant VirtualAtlasDraw& draw [[buffer(1)]],
    depth2d<float,access::read> source [[texture(2)]])
{
    uint2 texel=uint2(int2(draw.Origins.zw)+int2(floor(input.Position.xy)-draw.Origins.xy));
    VirtualCopyDepth result; result.Depth=source.read(texel); return result;
}
kernel void virtualAtlasRead(constant VirtualAtlasDraw& draw [[buffer(0)]],device float* feedback [[buffer(1)]],
    depth2d<float,access::read> source [[texture(0)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*draw.Extent.z;
    if (index>=draw.Extent.x*draw.Extent.y) return;
    feedback[index]=source.read(uint2(index%draw.Extent.x,index/draw.Extent.x));
}
