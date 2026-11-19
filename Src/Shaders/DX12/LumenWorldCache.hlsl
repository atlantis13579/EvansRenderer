// Directional world-probe reconstruction, following Lumen (Wright et al. 2022).
// A candidate survives only when the following mesh-SDF pass proves local
// visibility. Unknown angular taps reject the whole interpolation footprint.
static const uint CascadeLimit=4;
float3 LumenEnvironmentRadiance(float3 baseSky,float4 axisPower,float3 peak,float3 direction)
{
    if (!any(peak>0)) return baseSky;
    float cosine=saturate(dot(axisPower.xyz,direction));
    return baseSky+peak*pow(cosine,axisPower.w);
}
struct Grid { float4 Lattice; uint4 Extent; float4 Interval; };
cbuffer Reconstruction:register(b0)
{
    Grid Grids[CascadeLimit];
    uint4 Dispatch; float4 Environment,Options,Transition,EnvironmentAxis,EnvironmentPeak;
};
struct GeometryValue { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct RadianceValue { GeometryValue Hit; float4 ColorCoverage; uint4 Cache; };
struct ProbeSeed { float4 PositionMinimum; uint4 Metadata; };
StructuredBuffer<RadianceValue> CacheValues:register(t0);
StructuredBuffer<ProbeSeed> Seeds:register(t1);
RWStructuredBuffer<GeometryValue> Intersections:register(u0);
RWStructuredBuffer<RadianceValue> Transport:register(u1);
static const uint Threads=64,Absent=0xffffffffu,MissStatus=0,SurfaceStatus=1,UnknownStatus=2,ScreenStatus=3,CacheStatus=4,Ready=1;
static const float Pi=3.14159265358979323846,PhaseReciprocal=1.0/4294967296.0;
float3 directionForRay(uint index,uint id)
{
    uint sampleIndex=index%Dispatch.y,phase=reversebits(sampleIndex)+reversebits(id^Dispatch.z);
    float height=1-2*(float(sampleIndex)+0.5)/Dispatch.y,azimuth=2*Pi*(float(phase)*PhaseReciprocal);
    float ring=sqrt(max(0,1-height*height)); return float3(ring*cos(azimuth),ring*sin(azimuth),height);
}
float2 equalAreaSquare(float3 direction)
{
    direction=normalize(direction); float r=sqrt(max(0,1-abs(direction.z)));
    float phi=abs(direction.x)+abs(direction.y)>0?atan2(abs(direction.y),abs(direction.x))*2/Pi:0;
    float2 square=float2(r*(1-phi),r*phi); if (direction.z<0) square=1-square.yx;
    return 0.5*(1+square*float2(direction.x<0?-1:1,direction.y<0?-1:1));
}
int2 reflectedPixel(int2 pixel,int size)
{
    if (pixel.x<0) { pixel.x=-1-pixel.x; pixel.y=size-1-pixel.y; }
    else if (pixel.x>=size) { pixel.x=2*size-1-pixel.x; pixel.y=size-1-pixel.y; }
    if (pixel.y<0) { pixel.y=-1-pixel.y; pixel.x=size-1-pixel.x; }
    else if (pixel.y>=size) { pixel.y=2*size-1-pixel.y; pixel.x=size-1-pixel.x; }
    return clamp(pixel,int2(0,0),int2(size-1,size-1));
}
bool reconstructDirection(Grid grid,uint probe,float3 offset,float3 ray,out float4 answer)
{
    answer=0; float3 mapped=ray; float jacobian=1;
    if (Environment.w!=0)
    {
        float radius=grid.Interval.x,along=dot(offset,ray);
        float root=along*along+radius*radius-dot(offset,offset); if (root<=0) return false;
        float travel=sqrt(root)-along; mapped=offset+travel*ray;
        float projectedArea=radius*dot(mapped,ray);
        if (travel<=0 || projectedArea<=Options.z*radius*radius) return false;
        jacobian=travel*travel/projectedArea;
    }
    if (!all(isfinite(mapped)) || !isfinite(jacobian)) return false;
    // Clarberg's equal-area square needs reflected edges, not clamped filtering.
    float2 texel=equalAreaSquare(mapped)*float(grid.Extent.w)-0.5,ratio=frac(texel);
    int2 lower=int2(floor(texel));
    int radius=int(Transition.y); float support=float(radius+1),normalization=1/(support*support*support*support);
    int2 center=reflectedPixel(int2(floor(texel+0.5)),int(grid.Extent.w));
    uint first=probe*uint(Transition.z)*grid.Extent.w*grid.Extent.w;
    RadianceValue reference=CacheValues[first+uint(center.y)*grid.Extent.w+uint(center.x)];
    // Keep the fixed tent mass. Incomplete or discontinuous footprints fall
    // back to tracing instead of borrowing energy from a different depth.
    static const float FilterNormalAgreement=0.95;
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,support-abs(float2(x,y)-ratio)); float weight=tent.x*tent.y*normalization; if (weight<=0) continue;
        int2 p=reflectedPixel(lower+int2(x,y),int(grid.Extent.w)); uint address=first+uint(p.y)*grid.Extent.w+uint(p.x);
        RadianceValue data=CacheValues[address]; float4 light=0;
        if (radius>0)
        {
            if (data.Hit.Metadata.z!=reference.Hit.Metadata.z) return false;
            if (data.Hit.Metadata.z==SurfaceStatus)
            {
                float3 separation=data.Hit.PositionDistance.xyz-reference.Hit.PositionDistance.xyz;
                float roundoff=Options.z*max(1,max(length(data.Hit.PositionDistance.xyz),length(reference.Hit.PositionDistance.xyz)));
                float tolerance=data.Hit.Uncertainty.y+reference.Hit.Uncertainty.y+roundoff;
                if (!isfinite(tolerance)) return false;
                if (any(data.Hit.Metadata.xy!=reference.Hit.Metadata.xy) || dot(data.Hit.NormalResidual.xyz,reference.Hit.NormalResidual.xyz)<FilterNormalAgreement ||
                    abs(dot(separation,data.Hit.NormalResidual.xyz))>tolerance || abs(dot(separation,reference.Hit.NormalResidual.xyz))>tolerance) return false;
            }
        }
        if (data.Hit.Metadata.z==MissStatus) light=float4(LumenEnvironmentRadiance(Environment.xyz,EnvironmentAxis,EnvironmentPeak.xyz,ray),1);
        else if (data.Hit.Metadata.z==SurfaceStatus && data.Cache.z==Ready) light=float4(data.ColorCoverage.xyz*jacobian,data.ColorCoverage.w);
        else return false;
        if (!all(isfinite(light)) || any(light<0) || light.w<=0) return false;
        answer+=weight*light;
    }
    return true;
}
static const float LayerPlaneAgreement=0.95,LayerRayMinimumCosine=0.05;
// Inspect the IEEE-754 exponent directly: SM5 can warn about isfinite() on
// loop-merged values even with strict arithmetic and warnings-as-errors.
static const uint DepthExponentMask=0x7f800000u;
bool depthFinite(float value) { return (asuint(value)&DepthExponentMask)!=DepthExponentMask; }
bool depthFinite4(float4 value) { return all((asuint(value)&DepthExponentMask)!=DepthExponentMask); }
float worldLayerPrecision(float3 a,float3 b)
{ return Options.z*max(1,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))); }
bool compatibleLayer(GeometryValue a,GeometryValue b)
{
    if (a.Metadata.z!=b.Metadata.z) return false;
    if (a.Metadata.z==MissStatus) return true;
    float width=a.Uncertainty.z+b.Uncertainty.z+worldLayerPrecision(a.PositionDistance.xyz,b.PositionDistance.xyz);
    float3 separation=a.PositionDistance.xyz-b.PositionDistance.xyz;
    return a.Metadata.z==SurfaceStatus && all(a.Metadata.xy==b.Metadata.xy) && dot(a.NormalResidual.xyz,b.NormalResidual.xyz)>=LayerPlaneAgreement &&
        depthFinite(width) && abs(dot(separation,a.NormalResidual.xyz))<=width && abs(dot(separation,b.NormalResidual.xyz))<=width;
}
bool validLayerLight(RadianceValue value)
{ return value.Hit.Metadata.z==SurfaceStatus && value.Cache.z==Ready && depthFinite4(value.ColorCoverage) && all(value.ColorCoverage>=0) && value.ColorCoverage.w>0; }
bool reconstructSurfaceLayer(Grid grid,uint probe,RadianceValue anchor,uint depth,float4 query,float3 delta,float3 ray,out RadianceValue result)
{
    result=anchor; float cosine=dot(anchor.Hit.NormalResidual.xyz,ray);
    if (!validLayerLight(anchor) || abs(cosine)<LayerRayMinimumCosine) return false;
    float travel=dot(anchor.Hit.PositionDistance.xyz-query.xyz,anchor.Hit.NormalResidual.xyz)/cosine;
    if (!depthFinite(travel) || travel<=query.w || travel>=Options.x) return false;
    // Transport radiance is conserved. Reproject the surface address instead
    // of applying a near-sphere solid-angle correction to this depth layer.
    float3 mapped=Environment.w>0?delta+travel*ray:ray; float squared=dot(mapped,mapped);
    if (!depthFinite4(float4(mapped,squared)) || squared<=0) return false;
    uint side=grid.Extent.w,rays=side*side,layers=uint(Transition.z),start=probe*layers*rays;
    float2 texel=equalAreaSquare(mapped)*side-0.5,fraction=frac(texel); int2 lower=int2(floor(texel));
    int radius=int(Transition.y); float support=float(radius+1),normalization=1/(support*support*support*support);
    float width=anchor.Hit.Uncertainty.z; float4 light=0;
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,support-abs(float2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0) continue;
        int2 pixel=reflectedPixel(lower+int2(x,y),int(side)); uint address=uint(pixel.y)*side+uint(pixel.x);
        bool matched=false; float error=0; RadianceValue selected=anchor; uint selectedDepth=0;
        [loop] for (uint layer=0;layer<layers;++layer)
        {
            RadianceValue value=CacheValues[start+layer*rays+address];
            if (!validLayerLight(value) || !compatibleLayer(anchor.Hit,value.Hit)) continue;
            float plane=abs(dot(value.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz,anchor.Hit.NormalResidual.xyz));
            if (!matched || plane<error) { matched=true; error=plane; selected=value; selectedDepth=layer; }
        }
        if (!matched) return false;
        light+=weight*selected.ColorCoverage; width=max(width,selected.Hit.Uncertainty.z+error); depth=max(depth,selectedDepth);
    }
    if (!depthFinite4(light) || any(light<0) || light.w<=0 || !depthFinite(width)) return false;
    result.Hit.PositionDistance=float4(query.xyz+travel*ray,travel); result.Hit.NormalResidual=float4(anchor.Hit.NormalResidual.xyz,0);
    result.Hit.Uncertainty=float4(0,0,width,0); result.ColorCoverage=light; result.Cache=uint4(depth,0,Ready,0); return true;
}
bool reconstructDepthProbe(Grid grid,uint probe,float4 query,float3 delta,float3 ray,out RadianceValue result)
{
    uint side=grid.Extent.w,rays=side*side,layers=uint(Transition.z),start=probe*layers*rays;
    float2 address=equalAreaSquare(ray)*side-0.5; int2 nearest=reflectedPixel(int2(floor(address+0.5)),int(side));
    result=CacheValues[start]; bool found=false; float closest=Options.x;
    [loop] for (uint depth=0;depth<layers;++depth)
    {
        RadianceValue anchor=CacheValues[start+depth*rays+uint(nearest.y)*side+uint(nearest.x)],candidate;
        if (reconstructSurfaceLayer(grid,probe,anchor,depth,query,delta,ray,candidate) && candidate.Hit.PositionDistance.w<closest)
        { result=candidate; closest=candidate.Hit.PositionDistance.w; found=true; }
    }
    if (found) return true;
    int2 lower=int2(floor(address)); float2 fraction=frac(address); int radius=int(Transition.y); float support=float(radius+1);
    // A primary miss footprint may propose sky, but its complete query ray
    // must still miss before the following SDF stage retains that proposal.
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,support-abs(float2(x,y)-fraction)); if (tent.x*tent.y<=0) continue;
        int2 pixel=reflectedPixel(lower+int2(x,y),int(side));
        if (CacheValues[start+uint(pixel.y)*side+uint(pixel.x)].Hit.Metadata.z!=MissStatus) return false;
    }
    result.Hit.PositionDistance=0; result.Hit.NormalResidual=0; result.Hit.Uncertainty=0; result.Hit.Metadata=uint4(Absent,Absent,MissStatus,0);
    result.ColorCoverage=float4(LumenEnvironmentRadiance(Environment.xyz,EnvironmentAxis,EnvironmentPeak.xyz,ray),1);
    result.Cache=uint4(Absent,0,Ready,0); return depthFinite4(result.ColorCoverage);
}
bool reconstructLevel(uint level,ProbeSeed seed,float3 ray,out float4 sum,out float boundary)
{
    Grid grid=Grids[level]; sum=0; boundary=0;
    float3 location=(seed.PositionMinimum.xyz-grid.Lattice.xyz)*grid.Interval.w;
    if (!all(isfinite(location)) || any(location<0) || any(location>float3(grid.Extent.xyz)-1) ||
        grid.Interval.y<=seed.PositionMinimum.w || grid.Interval.y>=Options.x) return false;
    float3 faces=min(location,float3(grid.Extent.xyz)-1-location); boundary=min(faces.x,min(faces.y,faces.z));
    uint3 base=min(uint3(floor(location)),grid.Extent.xyz-2); float3 alpha=saturate(location-float3(base));
    uint range=level*grid.Extent.x*grid.Extent.y*grid.Extent.z;
    [loop] for (uint z=0;z<2;++z) [loop] for (uint y=0;y<2;++y) [loop] for (uint x=0;x<2;++x)
    {
        float weight=(x?alpha.x:1-alpha.x)*(y?alpha.y:1-alpha.y)*(z?alpha.z:1-alpha.z); if (weight<=0) continue;
        uint3 p=base+uint3(x,y,z); float3 delta=seed.PositionMinimum.xyz-grid.Lattice.xyz-float3(p)*grid.Lattice.w; float4 value=0;
        if (!reconstructDirection(grid,range+(p.z*grid.Extent.y+p.y)*grid.Extent.x+p.x,delta,ray,value)) return false;
        sum+=weight*value;
    }
    return all(isfinite(sum)) && sum.w>0;
}
bool reconstructDepthLevel(uint level,ProbeSeed seed,float3 ray,out RadianceValue result,out float boundary)
{
    result=(RadianceValue)0; result.Hit.Medium=float4(0,0,0,1); boundary=0; Grid grid=Grids[level];
    float3 location=(seed.PositionMinimum.xyz-grid.Lattice.xyz)*grid.Interval.w;
    if (!all(isfinite(location)) || any(location<0) || any(location>float3(grid.Extent.xyz)-1) ||
        grid.Interval.y<=seed.PositionMinimum.w || grid.Interval.y>=Options.x) return false;
    float3 face=min(location,float3(grid.Extent.xyz)-1-location); boundary=min(face.x,min(face.y,face.z));
    uint3 base=min(uint3(floor(location)),grid.Extent.xyz-2); float3 fraction=saturate(location-float3(base));
    uint first=level*grid.Extent.x*grid.Extent.y*grid.Extent.z; bool initialized=false; float4 light=0;
    [loop] for (uint z=0;z<2;++z) [loop] for (uint y=0;y<2;++y) [loop] for (uint x=0;x<2;++x)
    {
        float weight=(x?fraction.x:1-fraction.x)*(y?fraction.y:1-fraction.y)*(z?fraction.z:1-fraction.z); if (weight<=0) continue;
        uint3 p=base+uint3(x,y,z); float3 delta=seed.PositionMinimum.xyz-grid.Lattice.xyz-float3(p)*grid.Lattice.w; RadianceValue candidate; candidate.Hit.Medium=float4(0,0,0,1);
        if (!reconstructDepthProbe(grid,first+(p.z*grid.Extent.y+p.y)*grid.Extent.x+p.x,seed.PositionMinimum,delta,ray,candidate)) return false;
        if (!initialized) { result=candidate; initialized=true; }
        else
        {
            if (!compatibleLayer(result.Hit,candidate.Hit)) return false;
            float plane=abs(dot(result.Hit.PositionDistance.xyz-candidate.Hit.PositionDistance.xyz,result.Hit.NormalResidual.xyz));
            result.Hit.Uncertainty.z=max(result.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z+plane); result.Cache.x=max(result.Cache.x,candidate.Cache.x);
        }
        light+=weight*candidate.ColorCoverage;
    }
    result.ColorCoverage=light; return initialized && depthFinite4(light) && light.w>0;
}
[numthreads(Threads,1,1)]
void CSWorldLookup(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*Dispatch.w; if (index>=Dispatch.x) return;
    if (Options.y!=0 && Intersections[index].Metadata.z==ScreenStatus) return;
    GeometryValue empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0; empty.Metadata=uint4(Absent,Absent,UnknownStatus,0);
    RadianceValue result; result.Hit.Medium=float4(0,0,0,1); result.Hit=empty; result.ColorCoverage=0; result.Cache=uint4(Absent,0,0,0);
    Intersections[index]=empty; Transport[index]=result;
    ProbeSeed seed=Seeds[index/Dispatch.y]; if (seed.Metadata.y!=0) return;
    float3 ray=directionForRay(index,seed.Metadata.x); uint count=min(uint(Options.w),CascadeLimit);
    if (Transition.z>1)
    {
        [loop] for (uint level=0;level<count;++level)
        {
            RadianceValue candidate; candidate.Hit.Medium=float4(0,0,0,1); float boundary; if (!reconstructDepthLevel(level,seed,ray,candidate,boundary)) continue;
            uint selected=level;
            if (Transition.x>0 && boundary<Transition.x && level+1<count)
            {
                RadianceValue coarse; coarse.Hit.Medium=float4(0,0,0,1); float ignored;
                if (reconstructDepthLevel(level+1,seed,ray,coarse,ignored) && compatibleLayer(candidate.Hit,coarse.Hit))
                {
                    float fineWeight=saturate(boundary/Transition.x); candidate.ColorCoverage=lerp(coarse.ColorCoverage,candidate.ColorCoverage,fineWeight);
                    if (fineWeight==0) { candidate.Hit=coarse.Hit; candidate.Cache=coarse.Cache; selected=level+1; }
                    else
                    {
                        float plane=abs(dot(candidate.Hit.PositionDistance.xyz-coarse.Hit.PositionDistance.xyz,candidate.Hit.NormalResidual.xyz));
                        candidate.Hit.Uncertainty.z=max(candidate.Hit.Uncertainty.z,coarse.Hit.Uncertainty.z+plane); candidate.Cache.x=max(candidate.Cache.x,coarse.Cache.x);
                    }
                }
            }
            if (candidate.Hit.Metadata.z==SurfaceStatus) candidate.Hit.Uncertainty=float4(candidate.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z,0,0);
            else candidate.Hit.PositionDistance=float4(seed.PositionMinimum.xyz+ray*Options.x,Options.x);
            candidate.Hit.Metadata.z=CacheStatus; candidate.Hit.Metadata.w=selected; candidate.ColorCoverage.w=saturate(candidate.ColorCoverage.w);
            Intersections[index]=candidate.Hit; Transport[index]=candidate; return;
        }
        return;
    }
    [loop] for (uint level=0;level<count;++level)
    {
        float4 sum=0; float boundary=0; if (!reconstructLevel(level,seed,ray,sum,boundary)) continue;
        uint selected=level; float nearLimit=Grids[level].Interval.y;
        if (Transition.x>0 && boundary<Transition.x && level+1<count)
        {
            float4 coarse=0; float coarseBoundary=0;
            if (reconstructLevel(level+1,seed,ray,coarse,coarseBoundary))
            {
                float weight=saturate(boundary/Transition.x); sum=coarse*(1-weight)+sum*weight;
                nearLimit=Grids[level+1].Interval.y; if (weight==0) selected=level+1;
            }
        }
        result.Hit.PositionDistance=float4(seed.PositionMinimum.xyz+ray*nearLimit,nearLimit); result.Hit.Metadata.z=CacheStatus; result.Hit.Metadata.w=selected;
        result.ColorCoverage=float4(sum.xyz,saturate(sum.w)); result.Cache=uint4(Absent,0,Ready,0);
        Intersections[index]=result.Hit; Transport[index]=result; return;
    }
}
