#include <metal_stdlib>
using namespace metal;
float3 lumenEnvironmentRadiance(float3 baseSky,float4 axisPower,float3 peak,float3 direction)
{
    if (!any(peak>0)) return baseSky;
    float cosine=saturate(dot(axisPower.xyz,direction));
    return baseSky+peak*pow(cosine,axisPower.w);
}
// Lumen (Wright et al., SIGGRAPH 2022): cache distant directional radiance,
// then validate near visibility with the mesh SDF. Equal-area square filtering
// follows Clarberg (JGT 2008); unknown taps never become normalized light.
enum { WorldCascadeLimit=4 };
struct WorldGrid { float4 OriginCell; uint4 Dimensions; float4 Distances; };
struct WorldFrame { WorldGrid Grids[WorldCascadeLimit]; uint4 Work; float4 Sky,Query,Transition,EnvironmentAxis,EnvironmentPeak; };
struct WorldHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct WorldSample { WorldHit Hit; float4 RadianceCoverage; uint4 Cache; };
struct WorldOrigin { float4 PositionMinimum; uint4 Metadata; };
constant uint WorldMissing=0xffffffffu,WorldMiss=0,WorldSurface=1,WorldUnknown=2,WorldScreen=3,WorldCached=4,WorldComplete=1;
constant float WorldPi=3.14159265358979323846,WorldPhaseUnit=1.0/4294967296.0;
uint worldReverseBits(uint value)
{
    constexpr uint BitCount=32; uint result=0;
    for (uint bit=0;bit<BitCount;++bit) { result=(result<<1)|(value&1); value>>=1; }
    return result;
}
float3 worldQueryDirection(uint ray,uint id,constant WorldFrame& frame)
{
    uint sample=ray%frame.Work.y,phase=worldReverseBits(sample)+worldReverseBits(id^frame.Work.z);
    float z=1-2*(float(sample)+0.5f)/frame.Work.y,angle=2*WorldPi*(float(phase)*WorldPhaseUnit);
    float ring=sqrt(max(0.0f,1-z*z)); return float3(ring*cos(angle),ring*sin(angle),z);
}
float2 worldAngularUV(float3 direction)
{
    float3 n=normalize(direction); float radius=sqrt(max(0.0f,1-abs(n.z)));
    float angle=abs(n.x)+abs(n.y)>0?atan2(abs(n.y),abs(n.x))*2/WorldPi:0;
    float2 p=radius*float2(1-angle,angle); if (n.z<0) p=1-p.yx;
    return (1+p*float2(n.x<0?-1:1,n.y<0?-1:1))*0.5f;
}
int2 worldMirror(int2 pixel,int side)
{
    if (pixel.x<0) { pixel.x=-pixel.x-1; pixel.y=side-1-pixel.y; }
    if (pixel.x>=side) { pixel.x=2*side-1-pixel.x; pixel.y=side-1-pixel.y; }
    if (pixel.y<0) { pixel.y=-pixel.y-1; pixel.x=side-1-pixel.x; }
    if (pixel.y>=side) { pixel.y=2*side-1-pixel.y; pixel.x=side-1-pixel.x; }
    return clamp(pixel,int2(0),int2(side-1));
}
bool worldFilter(device const WorldSample* samples,constant WorldFrame& frame,WorldGrid grid,uint probe,float3 delta,float3 ray,thread float4& answer)
{
    answer=0; float3 projected=ray; float correction=1;
    if (frame.Sky.w>0)
    {
        float radius=grid.Distances.x,b=dot(delta,ray),root=b*b+radius*radius-dot(delta,delta);
        if (root<=0) return false;
        float t=-b+sqrt(root); projected=delta+t*ray; float area=radius*dot(projected,ray);
        if (t<=0 || area<=frame.Query.z*radius*radius) return false;
        correction=t*t/area;
    }
    if (!all(isfinite(projected)) || !isfinite(correction)) return false;
    float2 pixel=worldAngularUV(projected)*grid.Dimensions.w-0.5f,fraction=fract(pixel); int2 lower=int2(floor(pixel));
    int radius=int(frame.Transition.y); float extent=float(radius+1),normalization=1/(extent*extent*extent*extent);
    int2 center=worldMirror(int2(floor(pixel+0.5f)),int(grid.Dimensions.w));
    uint first=probe*uint(frame.Transition.z)*grid.Dimensions.w*grid.Dimensions.w;
    WorldSample anchor=samples[first+uint(center.y)*grid.Dimensions.w+uint(center.x)];
    constexpr float FilterNormalCosine=0.95f;
    // Unit-mass angular tent. Unknown rays, separate instances and nonplanar
    // depth changes reject the proposal; the local/full trace remains in charge.
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(float2(0),extent-abs(float2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0) continue;
        int2 p=worldMirror(lower+int2(x,y),int(grid.Dimensions.w));
        WorldSample source=samples[first+uint(p.y)*grid.Dimensions.w+uint(p.x)]; float4 incoming;
        if (radius>0)
        {
            if (source.Hit.Metadata.z!=anchor.Hit.Metadata.z) return false;
            if (source.Hit.Metadata.z==WorldSurface)
            {
                float3 separation=source.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz;
                float roundoff=frame.Query.z*max(1.0f,max(length(source.Hit.PositionDistance.xyz),length(anchor.Hit.PositionDistance.xyz)));
                float footprint=source.Hit.Uncertainty.y+anchor.Hit.Uncertainty.y+roundoff;
                if (!isfinite(footprint)) return false;
                if (any(source.Hit.Metadata.xy!=anchor.Hit.Metadata.xy) || dot(source.Hit.NormalResidual.xyz,anchor.Hit.NormalResidual.xyz)<FilterNormalCosine ||
                    abs(dot(separation,source.Hit.NormalResidual.xyz))>footprint || abs(dot(separation,anchor.Hit.NormalResidual.xyz))>footprint) return false;
            }
        }
        if (source.Hit.Metadata.z==WorldMiss) incoming=float4(lumenEnvironmentRadiance(frame.Sky.xyz,frame.EnvironmentAxis,frame.EnvironmentPeak.xyz,ray),1);
        else if (source.Hit.Metadata.z==WorldSurface && source.Cache.z==WorldComplete) incoming=float4(source.RadianceCoverage.xyz*correction,source.RadianceCoverage.w);
        else return false;
        if (!all(isfinite(incoming)) || any(incoming<0) || incoming.w<=0) return false;
        answer+=weight*incoming;
    }
    return true;
}
constant float WorldLayerNormalCosine=0.95f,WorldLayerRayCosine=0.05f;
float worldLayerPrecision(float3 a,float3 b,constant WorldFrame& frame)
{ return frame.Query.z*max(1.0f,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))); }
bool worldSameLayer(WorldHit a,WorldHit b,constant WorldFrame& frame)
{
    if (a.Metadata.z!=b.Metadata.z) return false;
    if (a.Metadata.z==WorldMiss) return true;
    float width=a.Uncertainty.z+b.Uncertainty.z+worldLayerPrecision(a.PositionDistance.xyz,b.PositionDistance.xyz,frame);
    float3 separation=a.PositionDistance.xyz-b.PositionDistance.xyz;
    return a.Metadata.z==WorldSurface && all(a.Metadata.xy==b.Metadata.xy) && dot(a.NormalResidual.xyz,b.NormalResidual.xyz)>=WorldLayerNormalCosine &&
        isfinite(width) && abs(dot(separation,a.NormalResidual.xyz))<=width && abs(dot(separation,b.NormalResidual.xyz))<=width;
}
bool worldValidLayer(WorldSample sample)
{ return sample.Hit.Metadata.z==WorldSurface && sample.Cache.z==WorldComplete && all(isfinite(sample.RadianceCoverage)) && all(sample.RadianceCoverage>=0) && sample.RadianceCoverage.w>0; }
bool worldDepthSurface(device const WorldSample* samples,constant WorldFrame& frame,WorldGrid grid,uint probe,
    WorldSample anchor,uint ordinal,float4 origin,float3 delta,float3 ray,thread WorldSample& answer)
{
    answer=anchor; float cosine=dot(anchor.Hit.NormalResidual.xyz,ray);
    if (!worldValidLayer(anchor) || abs(cosine)<WorldLayerRayCosine) return false;
    float travel=dot(anchor.Hit.PositionDistance.xyz-origin.xyz,anchor.Hit.NormalResidual.xyz)/cosine;
    if (!isfinite(travel) || travel<=origin.w || travel>=frame.Query.x) return false;
    // Radiance along a clear ray is unchanged. Reproject the surface address,
    // without assigning a distant surface the old near-sphere area correction.
    float3 projected=frame.Sky.w>0?delta+travel*ray:ray; float square=dot(projected,projected);
    if (!all(isfinite(projected)) || !isfinite(square) || square<=0) return false;
    uint side=grid.Dimensions.w,rays=side*side,layers=uint(frame.Transition.z),first=probe*layers*rays;
    float2 pixel=worldAngularUV(projected)*side-0.5f,fraction=fract(pixel); int2 lower=int2(floor(pixel));
    int radius=int(frame.Transition.y); float extent=float(radius+1),normalization=1/(extent*extent*extent*extent);
    float footprint=anchor.Hit.Uncertainty.z; float4 light=0;
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(float2(0),extent-abs(float2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0) continue;
        int2 p=worldMirror(lower+int2(x,y),int(side)); uint texel=uint(p.y)*side+uint(p.x);
        bool found=false; float best=0; WorldSample selected=anchor; uint selectedDepth=0;
        for (uint depth=0;depth<layers;++depth)
        {
            WorldSample candidate=samples[first+depth*rays+texel];
            if (!worldValidLayer(candidate) || !worldSameLayer(anchor.Hit,candidate.Hit,frame)) continue;
            float plane=abs(dot(candidate.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz,anchor.Hit.NormalResidual.xyz));
            if (!found || plane<best) { found=true; best=plane; selected=candidate; selectedDepth=depth; }
        }
        if (!found) return false;
        light+=weight*selected.RadianceCoverage; footprint=max(footprint,selected.Hit.Uncertainty.z+best); ordinal=max(ordinal,selectedDepth);
    }
    if (!all(isfinite(light)) || any(light<0) || light.w<=0 || !isfinite(footprint)) return false;
    answer.Hit.PositionDistance=float4(origin.xyz+travel*ray,travel); answer.Hit.NormalResidual=float4(anchor.Hit.NormalResidual.xyz,0);
    answer.Hit.Uncertainty=float4(0,0,footprint,0); answer.RadianceCoverage=light; answer.Cache=uint4(ordinal,0,WorldComplete,0); return true;
}
bool worldDepthProbe(device const WorldSample* samples,constant WorldFrame& frame,WorldGrid grid,uint probe,
    float4 origin,float3 delta,float3 ray,thread WorldSample& answer)
{
    uint side=grid.Dimensions.w,rays=side*side,layers=uint(frame.Transition.z),first=probe*layers*rays;
    float2 address=worldAngularUV(ray)*side-0.5f; int2 center=worldMirror(int2(floor(address+0.5f)),int(side));
    answer=samples[first]; bool found=false; float closest=frame.Query.x;
    for (uint depth=0;depth<layers;++depth)
    {
        WorldSample anchor=samples[first+depth*rays+uint(center.y)*side+uint(center.x)],candidate;
        if (worldDepthSurface(samples,frame,grid,probe,anchor,depth,origin,delta,ray,candidate) && candidate.Hit.PositionDistance.w<closest)
        { answer=candidate; closest=candidate.Hit.PositionDistance.w; found=true; }
    }
    if (found) return true;
    int2 lower=int2(floor(address)); float2 fraction=fract(address); int radius=int(frame.Transition.y); float extent=float(radius+1);
    // Only complete primary miss footprints propose sky. The next stage must
    // prove a full query miss before retaining the layered environment value.
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(float2(0),extent-abs(float2(x,y)-fraction)); if (tent.x*tent.y<=0) continue;
        int2 p=worldMirror(lower+int2(x,y),int(side)); if (samples[first+uint(p.y)*side+uint(p.x)].Hit.Metadata.z!=WorldMiss) return false;
    }
    answer.Hit.PositionDistance=0; answer.Hit.NormalResidual=0; answer.Hit.Uncertainty=0; answer.Hit.Metadata=uint4(WorldMissing,WorldMissing,WorldMiss,0);
    answer.RadianceCoverage=float4(lumenEnvironmentRadiance(frame.Sky.xyz,frame.EnvironmentAxis,frame.EnvironmentPeak.xyz,ray),1);
    answer.Cache=uint4(WorldMissing,0,WorldComplete,0); return all(isfinite(answer.RadianceCoverage));
}
bool worldFilterLevel(device const WorldSample* samples,constant WorldFrame& frame,uint level,WorldOrigin origin,float3 direction,
    thread float4& sum,thread float& border)
{
    WorldGrid grid=frame.Grids[level]; sum=0; border=0;
    float3 coordinate=(origin.PositionMinimum.xyz-grid.OriginCell.xyz)*grid.Distances.w;
    if (!all(isfinite(coordinate)) || any(coordinate<0) || any(coordinate>float3(grid.Dimensions.xyz)-1) ||
        origin.PositionMinimum.w>=grid.Distances.y || grid.Distances.y>=frame.Query.x) return false;
    float3 faces=min(coordinate,float3(grid.Dimensions.xyz)-1-coordinate); border=min(faces.x,min(faces.y,faces.z));
    uint3 cell=min(uint3(floor(coordinate)),grid.Dimensions.xyz-2); float3 alpha=clamp(coordinate-float3(cell),0.0f,1.0f);
    uint first=level*grid.Dimensions.x*grid.Dimensions.y*grid.Dimensions.z;
    for (uint tap=0;tap<8;++tap)
    {
        uint3 offset=uint3(tap&1,(tap>>1)&1,tap>>2),p=cell+offset;
        float weight=(offset.x?alpha.x:1-alpha.x)*(offset.y?alpha.y:1-alpha.y)*(offset.z?alpha.z:1-alpha.z); if (weight<=0) continue;
        float3 delta=origin.PositionMinimum.xyz-grid.OriginCell.xyz-float3(p)*grid.OriginCell.w; float4 value;
        if (!worldFilter(samples,frame,grid,first+(p.z*grid.Dimensions.y+p.y)*grid.Dimensions.x+p.x,delta,direction,value)) return false;
        sum+=weight*value;
    }
    return all(isfinite(sum)) && sum.w>0;
}
bool worldDepthLevel(device const WorldSample* samples,constant WorldFrame& frame,uint level,WorldOrigin origin,float3 ray,
    thread WorldSample& answer,thread float& border)
{
    answer=WorldSample{}; border=0; WorldGrid grid=frame.Grids[level];
    float3 location=(origin.PositionMinimum.xyz-grid.OriginCell.xyz)*grid.Distances.w;
    if (!all(isfinite(location)) || any(location<0) || any(location>float3(grid.Dimensions.xyz)-1) ||
        origin.PositionMinimum.w>=grid.Distances.y || grid.Distances.y>=frame.Query.x) return false;
    float3 faces=min(location,float3(grid.Dimensions.xyz)-1-location); border=min(faces.x,min(faces.y,faces.z));
    uint3 cell=min(uint3(floor(location)),grid.Dimensions.xyz-2); float3 fraction=saturate(location-float3(cell));
    uint first=level*grid.Dimensions.x*grid.Dimensions.y*grid.Dimensions.z; bool initialized=false; float4 light=0;
    for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=cell+bit;
        float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
        float3 delta=origin.PositionMinimum.xyz-grid.OriginCell.xyz-float3(p)*grid.OriginCell.w; WorldSample candidate; candidate.Hit.Medium=float4(0,0,0,1);
        if (!worldDepthProbe(samples,frame,grid,first+(p.z*grid.Dimensions.y+p.y)*grid.Dimensions.x+p.x,origin.PositionMinimum,delta,ray,candidate)) return false;
        if (!initialized) { answer=candidate; initialized=true; }
        else
        {
            if (!worldSameLayer(answer.Hit,candidate.Hit,frame)) return false;
            float plane=abs(dot(answer.Hit.PositionDistance.xyz-candidate.Hit.PositionDistance.xyz,answer.Hit.NormalResidual.xyz));
            answer.Hit.Uncertainty.z=max(answer.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z+plane); answer.Cache.x=max(answer.Cache.x,candidate.Cache.x);
        }
        light+=weight*candidate.RadianceCoverage;
    }
    answer.RadianceCoverage=light; return initialized && all(isfinite(light)) && light.w>0;
}
kernel void lumenWorldLookup(constant WorldFrame& frame [[buffer(0)]],device const WorldSample* samples [[buffer(1)]],
    device const WorldOrigin* origins [[buffer(2)]],device WorldHit* hits [[buffer(3)]],device WorldSample* results [[buffer(4)]],
    uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Work.w; if (index>=frame.Work.x) return;
    if (frame.Query.y>0 && hits[index].Metadata.z==WorldScreen) return;
    WorldHit empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0; empty.Metadata=uint4(WorldMissing,WorldMissing,WorldUnknown,0);
    WorldSample result; result.Hit.Medium=float4(0,0,0,1); result.Hit=empty; result.RadianceCoverage=0; result.Cache=uint4(WorldMissing,0,0,0);
    hits[index]=empty; results[index]=result;
    WorldOrigin origin=origins[index/frame.Work.y]; if (origin.Metadata.y) return;
    float3 direction=worldQueryDirection(index,origin.Metadata.x,frame); uint count=min(uint(frame.Query.w),uint(WorldCascadeLimit));
    if (frame.Transition.z>1)
    {
        for (uint level=0;level<count;++level)
        {
            WorldSample candidate; candidate.Hit.Medium=float4(0,0,0,1); float border; if (!worldDepthLevel(samples,frame,level,origin,direction,candidate,border)) continue;
            uint selected=level;
            if (frame.Transition.x>0 && border<frame.Transition.x && level+1<count)
            {
                WorldSample coarse; coarse.Hit.Medium=float4(0,0,0,1); float ignored;
                if (worldDepthLevel(samples,frame,level+1,origin,direction,coarse,ignored) && worldSameLayer(candidate.Hit,coarse.Hit,frame))
                {
                    float fineWeight=saturate(border/frame.Transition.x); candidate.RadianceCoverage=mix(coarse.RadianceCoverage,candidate.RadianceCoverage,fineWeight);
                    if (fineWeight==0) { candidate.Hit=coarse.Hit; candidate.Cache=coarse.Cache; selected=level+1; }
                    else
                    {
                        float plane=abs(dot(candidate.Hit.PositionDistance.xyz-coarse.Hit.PositionDistance.xyz,candidate.Hit.NormalResidual.xyz));
                        candidate.Hit.Uncertainty.z=max(candidate.Hit.Uncertainty.z,coarse.Hit.Uncertainty.z+plane); candidate.Cache.x=max(candidate.Cache.x,coarse.Cache.x);
                    }
                }
            }
            if (candidate.Hit.Metadata.z==WorldSurface) candidate.Hit.Uncertainty=float4(candidate.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z,0,0);
            else candidate.Hit.PositionDistance=float4(origin.PositionMinimum.xyz+direction*frame.Query.x,frame.Query.x);
            candidate.Hit.Metadata.z=WorldCached; candidate.Hit.Metadata.w=selected; candidate.RadianceCoverage.w=saturate(candidate.RadianceCoverage.w);
            hits[index]=candidate.Hit; results[index]=candidate; return;
        }
        return;
    }
    for (uint level=0;level<count;++level)
    {
        float4 sum; float border; if (!worldFilterLevel(samples,frame,level,origin,direction,sum,border)) continue;
        float localDistance=frame.Grids[level].Distances.y; uint selected=level;
        if (frame.Transition.x>0 && border<frame.Transition.x && level+1<count)
        {
            float4 coarse; float unusedBorder;
            if (worldFilterLevel(samples,frame,level+1,origin,direction,coarse,unusedBorder))
            {
                float fineWeight=clamp(border/frame.Transition.x,0.0f,1.0f); sum=mix(coarse,sum,fineWeight);
                localDistance=frame.Grids[level+1].Distances.y; if (fineWeight==0) selected=level+1;
            }
        }
        result.Hit.PositionDistance=float4(origin.PositionMinimum.xyz+direction*localDistance,localDistance); result.Hit.Metadata.z=WorldCached; result.Hit.Metadata.w=selected;
        result.RadianceCoverage=float4(sum.xyz,clamp(sum.w,0.0f,1.0f)); result.Cache=uint4(WorldMissing,0,WorldComplete,0);
        hits[index]=result.Hit; results[index]=result; return;
    }
}
