#version 310 es
precision highp float;
precision highp int;
#if defined(EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE)
layout(std140,binding=0) uniform LookupFrame
{
    vec4 LightRows[4];
    uvec4 VirtualExtent,PhysicalExtent,Filtering;
    vec4 Reserved;
};
uniform highp sampler2D PhysicalDepth;
uniform highp sampler2D PageDirectory;
layout(std430,binding=0) readonly buffer Queries { vec4 Values[]; } Input;
layout(std430,binding=1) writeonly buffer Results { vec4 Values[]; } Output;
#define VIRTUAL_LOOKUP_THREADS 64
layout(local_size_x=VIRTUAL_LOOKUP_THREADS,local_size_y=1,local_size_z=1) in;
const uint PageBorder=1u;
const float PixelCenter=0.5;
vec3 lookupDepth(vec2 uv,uint requestedMip)
{
    uint side=VirtualExtent.x/VirtualExtent.y,offset=0u;
    for (uint mip=0u;mip<VirtualExtent.z;++mip)
    {
        if (mip>=requestedMip)
        {
            uint resolution=side*VirtualExtent.y;
            uvec2 pixel=min(uvec2(uv*float(resolution)),uvec2(resolution-1u)),page=pixel/VirtualExtent.y;
            uint address=offset+page.y*side+page.x;
            // Directory row zero is metadata row zero; only raster depth flips Y.
            uint entry=uint(round(texelFetch(PageDirectory,ivec2(address%VirtualExtent.w,address/VirtualExtent.w),0).r*float(PhysicalExtent.w)));
            if (entry!=0u)
            {
                uint slot=entry-1u,stride=VirtualExtent.y+PageBorder*2u;
                uvec2 physical=uvec2(slot%PhysicalExtent.z,slot/PhysicalExtent.z)*stride+PageBorder+pixel%VirtualExtent.y;
                physical.y=PhysicalExtent.y-1u-physical.y;
                return vec3(texelFetch(PhysicalDepth,ivec2(physical),0).r,float(mip),float(entry));
            }
        }
        offset+=side*side; side/=2u;
    }
    return vec3(1.0,float(VirtualExtent.z-1u),0.0);
}
float compareTap(ivec2 pixel,uint mip,float receiver)
{
    uint resolution=VirtualExtent.x>>mip;
    vec2 uv=(vec2(clamp(pixel,ivec2(0),ivec2(resolution-1u)))+PixelCenter)/float(resolution);
    return receiver<=lookupDepth(uv,mip).x?1.0:0.0;
}
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Filtering.w; if (index>=Filtering.z) return;
    vec4 query=Input.Values[index];
    if (any(lessThan(query.xy,vec2(0))) || any(greaterThan(query.xy,vec2(1))) || query.z<0.0 || query.z>1.0)
    { Output.Values[index]=vec4(1,1,-1,0); return; }
    vec3 center=lookupDepth(query.xy,uint(query.w)); float visibility=query.z<=center.x?1.0:0.0;
    if (Filtering.x!=0u)
    {
        uint mip=uint(center.y); vec2 texel=query.xy*float(VirtualExtent.x>>mip)-PixelCenter;
        ivec2 origin=ivec2(floor(texel)); vec2 weight=fract(texel); visibility=0.0;
        for (int y=-int(Filtering.x);y<=int(Filtering.x);++y) for (int x=-int(Filtering.x);x<=int(Filtering.x);++x)
        {
            ivec2 p=origin+ivec2(x,y);
            float a=compareTap(p,mip,query.z),b=compareTap(p+ivec2(1,0),mip,query.z);
            float c=compareTap(p+ivec2(0,1),mip,query.z),d=compareTap(p+ivec2(1),mip,query.z);
            visibility+=mix(mix(a,b,weight.x),mix(c,d,weight.x),weight.y);
        }
        visibility/=float(Filtering.y);
    }
    Output.Values[index]=vec4(visibility,center);
}
#elif defined(EVANS_VIRTUAL_SHADOW_RASTER_STAGE) || defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE) || defined(EVANS_VIRTUAL_SHADOW_READ_STAGE)
layout(std140,binding=0) uniform AtlasFrame
{
    vec4 WorldViewProjectionRows[4];
    vec4 Material;
    uvec4 Textures;
    vec4 Origins;
    uvec4 Extent;
    vec4 VegetationWorld[4], VegetationNormalWorld[4];
    vec4 Wind, Gust, VegetationHistory;
};
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
vec3 VegetationOffset(vec3 world, float height, vec4 wind, vec4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return vec3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
vec4 VegetationPoint(vec3 position,vec2 uv)
{
    if (Gust.w<0.5) return vec4(position,1);
    vec4 local=vec4(position,1);
    vec3 world=vec3(dot(VegetationWorld[0],local),dot(VegetationWorld[1],local),dot(VegetationWorld[2],local));
    float height=max(0.0,Gust.w<1.5?uv.y:position.y);
    vec3 offset=VegetationOffset(world,height,Wind,Gust,VegetationHistory.y,Wind.z);
    return vec4(position+VegetationNormalWorld[0].xyz*offset.x+VegetationNormalWorld[1].xyz*offset.y+VegetationNormalWorld[2].xyz*offset.z,1);
}
#if defined(EVANS_VIRTUAL_SHADOW_READ_STAGE)
#define VIRTUAL_ATLAS_READ_THREADS 64
layout(local_size_x=VIRTUAL_ATLAS_READ_THREADS,local_size_y=1,local_size_z=1) in;
uniform highp sampler2D ParentDepth;
layout(std430,binding=1) writeonly buffer DepthFeedback { float Values[]; } Feedback;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Extent.z;
    if (index>=Extent.x*Extent.y) return;
    Feedback.Values[index]=texelFetch(ParentDepth,ivec2(index%Extent.x,Extent.y-1u-index/Extent.x),0).r;
}
#elif defined(EVANS_VERTEX_STAGE)
#if !defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
layout(location=0) in vec3 InPosition;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
out vec2 CoverageUV;
out float CoverageAlpha;
#endif
void main()
{
#if defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
    const float FullscreenExtent=3.0;
    vec2 corner=gl_VertexID==0?vec2(-1):gl_VertexID==1?vec2(-1,FullscreenExtent):vec2(FullscreenExtent,-1);
    gl_Position=vec4(corner,0,1);
#else
    vec4 point=VegetationPoint(InPosition,InUV);
    vec4 clip=vec4(dot(WorldViewProjectionRows[0],point),dot(WorldViewProjectionRows[1],point),dot(WorldViewProjectionRows[2],point),dot(WorldViewProjectionRows[3],point));
    clip.z=clip.z*2.0-clip.w; gl_Position=clip;
    CoverageUV=vec2(InUV.x,1.0-InUV.y); CoverageAlpha=InColor.a;
#endif
}
#else
#if defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
uniform highp sampler2D ParentDepth;
void main()
{
    vec2 topLeft=vec2(gl_FragCoord.x,float(Extent.y)-gl_FragCoord.y);
    ivec2 source=ivec2(Origins.zw)+ivec2(floor(topLeft)-Origins.xy);
    source.y=int(Extent.w)-1-source.y;
    gl_FragDepth=texelFetch(ParentDepth,source,0).r;
}
#else
in vec2 CoverageUV;
in float CoverageAlpha;
uniform highp sampler2D DiffuseImage,OpacityImage;
const float FlagThreshold=0.5,NoiseScale=52.9829189;
const vec2 NoiseDirection=vec2(0.06711056,0.00583715);
void main()
{
    float alpha=CoverageAlpha*Material.x;
    if (Textures.x!=0u) alpha*=texture(DiffuseImage,CoverageUV).a;
    if (Textures.y!=0u) alpha*=texture(OpacityImage,CoverageUV).r;
    alpha=clamp(alpha,0.0,1.0);
    if (Material.y>0.0) { if (alpha<Material.y) discard; }
    else if (Material.z>FlagThreshold)
    {
        vec2 topLeft=vec2(gl_FragCoord.x,float(Extent.y)-gl_FragCoord.y);
        vec2 logicalTexel=floor(topLeft-Origins.xy)+Origins.zw;
        float threshold=fract(NoiseScale*fract(dot(logicalTexel,NoiseDirection)));
        if (alpha<=0.0 || alpha<threshold) discard;
    }
}
#endif
#endif
#else
precision highp float;
precision highp int;
// Native ES demand uses canonical top-left depth records. Image packing flips
// native GL rows at its boundary, keeping this page-address algorithm portable.
// Clear precedes marking with GL_SHADER_STORAGE_BARRIER_BIT ordering.
#ifndef EVANS_VIRTUAL_SHADOW_PASS
#define EVANS_VIRTUAL_SHADOW_PASS 0
#endif
#define VIRTUAL_SHADOW_GROUP_WIDTH 64
layout(local_size_x=VIRTUAL_SHADOW_GROUP_WIDTH,local_size_y=1,local_size_z=1) in;
layout(std140,binding=0) uniform PageRequestFrame
{
    vec4 CameraInverseRows[4],LightViewProjectionRows[4];
    uvec4 Extent,Limits;
    vec4 Quality,Sampling;
};
layout(std430,binding=0) readonly buffer DepthRecords { float Values[]; } Depth;
layout(std430,binding=1) buffer RequestRecords { uint Values[]; } Requests;
const float PixelCenter=0.5,ClearDepth=1.0;
bool lightPosition(vec2 pixel,float depth,out vec3 position)
{
    vec2 uv=(vec2(pixel)+PixelCenter)/vec2(Extent.xy);
    vec4 clip=vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1);
    vec4 world=vec4(dot(CameraInverseRows[0],clip),dot(CameraInverseRows[1],clip),dot(CameraInverseRows[2],clip),dot(CameraInverseRows[3],clip));
    position=vec3(0); if (abs(world.w)<=Quality.y) return false; world=vec4(world.xyz/world.w,1);
    vec4 light=vec4(dot(LightViewProjectionRows[0],world),dot(LightViewProjectionRows[1],world),dot(LightViewProjectionRows[2],world),dot(LightViewProjectionRows[3],world));
    if (light.w<=Quality.y) return false;
    position=light.xyz/light.w; return !any(isnan(position)) && !any(isinf(position));
}
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Limits.w;
#if EVANS_VIRTUAL_SHADOW_PASS == 0
    if (index<Limits.y) Requests.Values[index]=0u;
#else
    if (index>=Extent.x*Extent.y) return;
    float depth=Depth.Values[index]; if (depth>=ClearDepth) return;
    uvec2 pixel=uvec2(index%Extent.x,index/Extent.x); vec3 center,adjacentX,adjacentY;
    if (!lightPosition(vec2(pixel),depth,center) || any(greaterThan(abs(center.xy),vec2(1))) || center.z<0.0 || center.z>1.0) return;
    if (!lightPosition(vec2(pixel)+vec2(Sampling.x,0),depth,adjacentX) || !lightPosition(vec2(pixel)+vec2(0,Sampling.y),depth,adjacentY)) return;
    float footprint=max(length(adjacentX.xy-center.xy),length(adjacentY.xy-center.xy))*PixelCenter*float(Extent.z);
    uint mip=uint(clamp(floor(log2(max(1.0,footprint))+Quality.x+Quality.w),0.0,float(Limits.x-1u)));
    uint side=Extent.z/Extent.w,offset=0u;
    for (uint level=0u;level<mip;++level) { offset+=side*side; side/=2u; }
    vec2 page=(center.xy*vec2(PixelCenter,-PixelCenter)+PixelCenter)*float(side);
    float padding=(float(Limits.z)+PixelCenter)/float(Extent.w);
    ivec2 first=clamp(ivec2(floor(page-padding-Quality.z)),ivec2(0),ivec2(side-1u));
    ivec2 last=clamp(ivec2(floor(page+padding+Quality.z)),ivec2(0),ivec2(side-1u));
    for (int y=first.y;y<=last.y;++y) for (int x=first.x;x<=last.x;++x)
    {
        uvec2 coordinate=uvec2(x,y); uint parentSide=side,parentOffset=offset;
        for (uint level=mip;level<Limits.x;++level)
        {
            atomicOr(Requests.Values[parentOffset+coordinate.y*parentSide+coordinate.x],1u);
            parentOffset+=parentSide*parentSide; parentSide/=2u; coordinate/=2u;
        }
    }
#endif
}

#endif
