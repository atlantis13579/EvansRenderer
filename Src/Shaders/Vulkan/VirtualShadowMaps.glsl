#version 450
#if defined(EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE)
layout(std140,set=0,binding=0,row_major) uniform LookupParameters
{
    mat4 LightProjection;
    uvec4 VirtualSize,PhysicalSize,Filter;
    vec4 Reserved;
} Lookup;
layout(set=0,binding=1) uniform sampler2D AtlasDepth;
layout(set=0,binding=2) uniform sampler2D DirectoryDepth;
layout(std430,set=0,binding=3) readonly buffer SampleQueries { vec4 Values[]; } Queries;
layout(std430,set=0,binding=4) writeonly buffer SampleResults { vec4 Values[]; } Results;
#define VIRTUAL_LOOKUP_GROUP_WIDTH 64
layout(local_size_x=VIRTUAL_LOOKUP_GROUP_WIDTH,local_size_y=1,local_size_z=1) in;
const uint TileGutter=1u;
const float TexelCenter=0.5;
vec3 resolveDepth(vec2 coordinates,uint minimumLevel)
{
    uint pages=Lookup.VirtualSize.x/Lookup.VirtualSize.y,first=0u;
    for (uint level=0u;level<Lookup.VirtualSize.z;++level)
    {
        if (level>=minimumLevel)
        {
            uint extent=pages*Lookup.VirtualSize.y;
            uvec2 pixel=min(uvec2(coordinates*float(extent)),uvec2(extent-1u)),page=pixel/Lookup.VirtualSize.y;
            uint id=first+page.y*pages+page.x;
            uint mapping=uint(round(texelFetch(DirectoryDepth,ivec2(id%Lookup.VirtualSize.w,id/Lookup.VirtualSize.w),0).r*float(Lookup.PhysicalSize.w)));
            if (mapping>0u)
            {
                uint tile=mapping-1u,stride=Lookup.VirtualSize.y+TileGutter*2u;
                uvec2 location=uvec2(tile%Lookup.PhysicalSize.z,tile/Lookup.PhysicalSize.z)*stride+TileGutter+pixel%Lookup.VirtualSize.y;
                return vec3(texelFetch(AtlasDepth,ivec2(location),0).r,float(level),float(mapping));
            }
        }
        first+=pages*pages; pages>>=1u;
    }
    return vec3(1.0,float(Lookup.VirtualSize.z-1u),0.0);
}
float compareVirtual(ivec2 pixel,uint level,float comparison)
{
    uint extent=Lookup.VirtualSize.x>>level;
    vec2 uv=(vec2(clamp(pixel,ivec2(0),ivec2(extent-1u)))+TexelCenter)/float(extent);
    return comparison<=resolveDepth(uv,level).x?1.0:0.0;
}
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Lookup.Filter.w; if (index>=Lookup.Filter.z) return;
    vec4 query=Queries.Values[index];
    if (any(lessThan(query.xy,vec2(0))) || any(greaterThan(query.xy,vec2(1))) || query.z<0.0 || query.z>1.0)
    { Results.Values[index]=vec4(1,1,-1,0); return; }
    vec3 selected=resolveDepth(query.xy,uint(query.w)); float lit=query.z<=selected.x?1.0:0.0;
    if (Lookup.Filter.x>0u)
    {
        uint level=uint(selected.y); vec2 position=query.xy*float(Lookup.VirtualSize.x>>level)-TexelCenter;
        ivec2 origin=ivec2(floor(position)); vec2 fraction=fract(position); lit=0.0;
        for (int row=-int(Lookup.Filter.x);row<=int(Lookup.Filter.x);++row) for (int column=-int(Lookup.Filter.x);column<=int(Lookup.Filter.x);++column)
        {
            ivec2 p=origin+ivec2(column,row);
            float upper=mix(compareVirtual(p,level,query.z),compareVirtual(p+ivec2(1,0),level,query.z),fraction.x);
            float lower=mix(compareVirtual(p+ivec2(0,1),level,query.z),compareVirtual(p+ivec2(1),level,query.z),fraction.x);
            lit+=mix(upper,lower,fraction.y);
        }
        lit/=float(Lookup.Filter.y);
    }
    Results.Values[index]=vec4(lit,selected);
}
#elif defined(EVANS_VIRTUAL_SHADOW_RASTER_STAGE) || defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE) || defined(EVANS_VIRTUAL_SHADOW_READ_STAGE)
layout(std140,set=0,binding=0,row_major) uniform AtlasFrame
{
    mat4 WorldViewProjection;
    vec4 Material;
    uvec4 Textures;
    vec4 Origins;
    uvec4 Extent;
    vec4 VegetationWorld[4], VegetationNormalWorld[4];
    vec4 Wind, Gust, VegetationHistory;
} Draw;
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
    if (Draw.Gust.w<0.5) return vec4(position,1);
    vec4 local=vec4(position,1);
    vec3 world=vec3(dot(Draw.VegetationWorld[0],local),dot(Draw.VegetationWorld[1],local),dot(Draw.VegetationWorld[2],local));
    float height=max(0.0,Draw.Gust.w<1.5?uv.y:position.y);
    vec3 offset=VegetationOffset(world,height,Draw.Wind,Draw.Gust,Draw.VegetationHistory.y,Draw.Wind.z);
    return vec4(position+Draw.VegetationNormalWorld[0].xyz*offset.x+Draw.VegetationNormalWorld[1].xyz*offset.y+Draw.VegetationNormalWorld[2].xyz*offset.z,1);
}
#if defined(EVANS_VIRTUAL_SHADOW_READ_STAGE)
#define VIRTUAL_ATLAS_READ_THREADS 64
layout(local_size_x=VIRTUAL_ATLAS_READ_THREADS,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=3) uniform sampler2D SourceAtlas;
layout(std430,set=0,binding=4) writeonly buffer AtlasFeedback { float Values[]; } Feedback;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Draw.Extent.z;
    if (index>=Draw.Extent.x*Draw.Extent.y) return;
    Feedback.Values[index]=texelFetch(SourceAtlas,ivec2(index%Draw.Extent.x,index/Draw.Extent.x),0).r;
}
#elif defined(EVANS_VERTEX_STAGE)
#if !defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
layout(location=0) in vec3 InPosition;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
layout(location=0) out vec2 UV;
layout(location=1) out float Alpha;
#endif
void main()
{
#if defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
    const float TriangleExtent=3.0;
    vec2 corner=gl_VertexIndex==0?vec2(-1):gl_VertexIndex==1?vec2(-1,TriangleExtent):vec2(TriangleExtent,-1);
    gl_Position=vec4(corner,0,1); gl_Position.y=-gl_Position.y;
#else
    gl_Position=Draw.WorldViewProjection*VegetationPoint(InPosition,InUV); gl_Position.y=-gl_Position.y;
    UV=vec2(InUV.x,1.0-InUV.y); Alpha=InColor.a;
#endif
}
#else
#if defined(EVANS_VIRTUAL_SHADOW_COPY_STAGE)
layout(set=0,binding=3) uniform sampler2D SourceAtlas;
void main()
{
    ivec2 coordinate=ivec2(Draw.Origins.zw)+ivec2(floor(gl_FragCoord.xy)-Draw.Origins.xy);
    gl_FragDepth=texelFetch(SourceAtlas,coordinate,0).r;
}
#else
layout(location=0) in vec2 UV;
layout(location=1) in float Alpha;
layout(set=0,binding=1) uniform sampler2D DiffuseImage;
layout(set=0,binding=2) uniform sampler2D OpacityImage;
const float EnabledThreshold=0.5,InterleavedScale=52.9829189;
const vec2 InterleavedDirection=vec2(0.06711056,0.00583715);
void main()
{
    float coverage=Alpha*Draw.Material.x;
    if (Draw.Textures.x!=0u) coverage*=texture(DiffuseImage,UV).a;
    if (Draw.Textures.y!=0u) coverage*=texture(OpacityImage,UV).r;
    coverage=clamp(coverage,0.0,1.0);
    if (Draw.Material.y>0.0) { if (coverage<Draw.Material.y) discard; }
    else if (Draw.Material.z>EnabledThreshold)
    {
        vec2 logicalPixel=floor(gl_FragCoord.xy-Draw.Origins.xy)+Draw.Origins.zw;
        float threshold=fract(InterleavedScale*fract(dot(logicalPixel,InterleavedDirection)));
        if (coverage<=0.0 || coverage<threshold) discard;
    }
}
#endif
#endif
#else
// Virtual shadow page requests are separate from both depth rasterization and
// immutable physical residency. The host publishes clear writes to marking and
// marking writes to the directory consumer through storage-buffer dependencies.
#ifndef EVANS_VIRTUAL_SHADOW_PASS
#define EVANS_VIRTUAL_SHADOW_PASS 0
#endif
#define VIRTUAL_SHADOW_GROUP_WIDTH 64
layout(local_size_x=VIRTUAL_SHADOW_GROUP_WIDTH,local_size_y=1,local_size_z=1) in;
layout(std140,set=0,binding=0) uniform RequestFrame
{
    vec4 CameraInverseRows[4],LightViewProjectionRows[4];
    uvec4 Extent,Limits;
    vec4 Quality,Sampling;
};
layout(std430,set=0,binding=1) readonly buffer CameraDepth { float Values[]; } Depth;
layout(std430,set=0,binding=2) buffer PageDemand { uint Values[]; } Demand;
const float HalfPixel=0.5,FarDepth=1.0;
bool projectToLight(vec2 pixel,float depth,out vec3 ndc)
{
    vec2 pixelUV=(vec2(pixel)+HalfPixel)/vec2(Extent.xy);
    vec4 clip=vec4(pixelUV*vec2(2,-2)+vec2(-1,1),depth,1);
    vec4 homogeneous=vec4(dot(CameraInverseRows[0],clip),dot(CameraInverseRows[1],clip),dot(CameraInverseRows[2],clip),dot(CameraInverseRows[3],clip));
    ndc=vec3(0); if (abs(homogeneous.w)<=Quality.y) return false;
    vec4 world=vec4(homogeneous.xyz/homogeneous.w,1);
    vec4 light=vec4(dot(LightViewProjectionRows[0],world),dot(LightViewProjectionRows[1],world),dot(LightViewProjectionRows[2],world),dot(LightViewProjectionRows[3],world));
    if (light.w<=Quality.y) return false;
    ndc=light.xyz/light.w; return !any(isnan(ndc)) && !any(isinf(ndc));
}
void main()
{
    uint receiver=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Limits.w;
#if EVANS_VIRTUAL_SHADOW_PASS == 0
    if (receiver<Limits.y) Demand.Values[receiver]=0u;
#else
    if (receiver>=Extent.x*Extent.y) return;
    float depth=Depth.Values[receiver]; if (depth>=FarDepth) return;
    uvec2 raster=uvec2(receiver%Extent.x,receiver/Extent.x); vec3 point,dx,dy;
    if (!projectToLight(vec2(raster),depth,point) || any(greaterThan(abs(point.xy),vec2(1))) || point.z<0.0 || point.z>1.0) return;
    if (!projectToLight(vec2(raster)+vec2(Sampling.x,0),depth,dx) || !projectToLight(vec2(raster)+vec2(0,Sampling.y),depth,dy)) return;
    float projectedSize=max(length(dx.xy-point.xy),length(dy.xy-point.xy))*HalfPixel*float(Extent.z);
    uint firstMip=uint(clamp(floor(log2(max(1.0,projectedSize))+Quality.x+Quality.w),0.0,float(Limits.x-1u)));
    uint pages=Extent.z/Extent.w,start=0u;
    for (uint mip=0u;mip<firstMip;++mip) { start+=pages*pages; pages/=2u; }
    vec2 pagePoint=(point.xy*vec2(HalfPixel,-HalfPixel)+HalfPixel)*float(pages);
    float extent=(float(Limits.z)+HalfPixel)/float(Extent.w);
    ivec2 lo=clamp(ivec2(floor(pagePoint-extent-Quality.z)),ivec2(0),ivec2(pages-1u));
    ivec2 hi=clamp(ivec2(floor(pagePoint+extent+Quality.z)),ivec2(0),ivec2(pages-1u));
    for (int row=lo.y;row<=hi.y;++row) for (int column=lo.x;column<=hi.x;++column)
    {
        uvec2 page=uvec2(column,row); uint levelPages=pages,offset=start;
        for (uint mip=firstMip;mip<Limits.x;++mip)
        {
            atomicOr(Demand.Values[offset+page.y*levelPages+page.x],1u);
            offset+=levelPages*levelPages; page/=2u; levelPages/=2u;
        }
    }
#endif
}

#endif
