// Wright et al., Lumen (SIGGRAPH 2022): interpolate distant radiance only after
// a local visibility trace. This pass proposes values; the SDF pass validates
// the entire near interval before retaining WorldCacheHit.
// Clarberg, Fast Equal-Area Mapping of the (Hemi)Sphere (JGT 2008).
static const uint MaximumCascades=4;
float3 LumenEnvironmentRadiance(float3 baseSky,float4 axisPower,float3 peak,float3 direction)
{
    if (!any(peak>0)) return baseSky;
    float cosine=saturate(dot(axisPower.xyz,direction));
    return baseSky+peak*pow(cosine,axisPower.w);
}
struct Cascade { float4 OriginCell; uint4 Dimensions; float4 Distances; };
cbuffer LookupFrame:register(b0)
{
    Cascade Cascades[MaximumCascades];
    uint4 Work; float4 Sky,Query,Transition,EnvironmentAxis,EnvironmentPeak;
};
struct Hit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct Sample { Hit Geometry; float4 RadianceCoverage; uint4 Cache; };
struct Origin { float4 PositionMinimum; uint4 Metadata; };
StructuredBuffer<Sample> WorldSamples:register(t0);
StructuredBuffer<Origin> Origins:register(t1);
RWStructuredBuffer<Hit> Hits:register(u0);
RWStructuredBuffer<Sample> Results:register(u1);
static const uint GroupWidth=64,Miss=0,Surface=1,Unknown=2,ScreenHit=3,WorldCacheHit=4,Complete=1,Missing=0xffffffffu;
static const float Pi=3.14159265358979323846,UintPhaseScale=1.0/4294967296.0;
float3 RayDirection(uint index,uint identifier)
{
    uint ray=index%Work.y,phase=reversebits(ray)+reversebits(identifier^Work.z);
    float z=1-2*(float(ray)+0.5)/float(Work.y),angle=2*Pi*(float(phase)*UintPhaseScale);
    float radius=sqrt(max(0,1-z*z)); return float3(radius*cos(angle),radius*sin(angle),z);
}
float2 AngularUV(float3 v)
{
    v=normalize(v); float radius=sqrt(max(0,1-abs(v.z)));
    float angle=(abs(v.x)+abs(v.y)>0)?atan2(abs(v.y),abs(v.x))*(2/Pi):0;
    float2 p=radius*float2(1-angle,angle); if (v.z<0) p=1-p.yx;
    return (p*float2(v.x<0?-1:1,v.y<0?-1:1)+1)*0.5;
}
int2 FoldTexel(int2 p,int side)
{
    // The square's edges join by reflection, including the four south-pole corners.
    if (p.x<0) { p.x=-p.x-1; p.y=side-1-p.y; }
    if (p.x>=side) { p.x=2*side-1-p.x; p.y=side-1-p.y; }
    if (p.y<0) { p.y=-p.y-1; p.x=side-1-p.x; }
    if (p.y>=side) { p.y=2*side-1-p.y; p.x=side-1-p.x; }
    return clamp(p,0,side-1);
}
bool ReadProbe(Cascade layout,uint probe,float3 delta,float3 direction,out float4 value)
{
    value=0; float3 projected=direction; float correction=1;
    if (Sky.w>0)
    {
        float radius=layout.Distances.x,b=dot(delta,direction),discriminant=b*b+radius*radius-dot(delta,delta);
        if (discriminant<=0) return false;
        float distance=-b+sqrt(discriminant); projected=delta+distance*direction;
        float denominator=radius*dot(projected,direction);
        if (distance<=0 || denominator<=Query.z*radius*radius) return false;
        correction=distance*distance/denominator;
    }
    if (!all(isfinite(projected)) || !isfinite(correction)) return false;
    float2 position=AngularUV(projected)*layout.Dimensions.w-0.5,blend=frac(position); int2 base=int2(floor(position));
    int radius=int(Transition.y); float extent=float(radius+1),normalization=1/(extent*extent*extent*extent);
    int2 nearest=FoldTexel(int2(floor(position+0.5)),int(layout.Dimensions.w));
    uint first=probe*uint(Transition.z)*layout.Dimensions.w*layout.Dimensions.w;
    Sample anchor=WorldSamples[first+uint(nearest.y)*layout.Dimensions.w+uint(nearest.x)];
    // A normalized tent broadens the angular footprint without renormalizing
    // missing rays. Depth/normal discontinuities reject the proposal entirely.
    static const float MinimumFilterNormalCosine=0.95;
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,extent-abs(float2(x,y)-blend)); float weight=tent.x*tent.y*normalization;
        if (weight<=0) continue;
        int2 pixel=FoldTexel(base+int2(x,y),int(layout.Dimensions.w));
        Sample s=WorldSamples[first+uint(pixel.y)*layout.Dimensions.w+uint(pixel.x)];
        if (radius>0)
        {
            if (s.Geometry.Metadata.z!=anchor.Geometry.Metadata.z) return false;
            if (s.Geometry.Metadata.z==Surface)
            {
                float3 separation=s.Geometry.PositionDistance.xyz-anchor.Geometry.PositionDistance.xyz;
                float roundoff=Query.z*max(1,max(length(s.Geometry.PositionDistance.xyz),length(anchor.Geometry.PositionDistance.xyz)));
                float footprint=s.Geometry.Uncertainty.y+anchor.Geometry.Uncertainty.y+roundoff;
                if (!isfinite(footprint)) return false;
                if (any(s.Geometry.Metadata.xy!=anchor.Geometry.Metadata.xy) || dot(s.Geometry.NormalResidual.xyz,anchor.Geometry.NormalResidual.xyz)<MinimumFilterNormalCosine ||
                    abs(dot(separation,s.Geometry.NormalResidual.xyz))>footprint || abs(dot(separation,anchor.Geometry.NormalResidual.xyz))>footprint) return false;
            }
        }
        float4 incoming=0;
        if (s.Geometry.Metadata.z==Miss) incoming=float4(LumenEnvironmentRadiance(Sky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction),1);
        else if (s.Geometry.Metadata.z==Surface && s.Cache.z==Complete) incoming=float4(s.RadianceCoverage.xyz*correction,s.RadianceCoverage.w);
        else return false;
        if (!all(isfinite(incoming)) || any(incoming<0) || incoming.w<=0) return false;
        value+=weight*incoming;
    }
    return true;
}
static const float LayerNormalCosine=0.95,MinimumLayerRayCosine=0.05;
// IEEE-754 exponent bits retain explicit NaN/Inf rejection when the SM5
// optimizer diagnoses isfinite() on loop-merged values as a redundant test.
static const uint LayerExponentMask=0x7f800000u;
bool LayerFinite(float value) { return (asuint(value)&LayerExponentMask)!=LayerExponentMask; }
bool LayerFinite4(float4 value) { return all((asuint(value)&LayerExponentMask)!=LayerExponentMask); }
float LayerRoundoff(float3 a,float3 b)
{ return Query.z*max(1,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))); }
bool SameLayerPlane(Hit a,Hit b)
{
    if (a.Metadata.z!=b.Metadata.z) return false;
    if (a.Metadata.z==Miss) return true;
    float tolerance=a.Uncertainty.z+b.Uncertainty.z+LayerRoundoff(a.PositionDistance.xyz,b.PositionDistance.xyz);
    float3 offset=a.PositionDistance.xyz-b.PositionDistance.xyz;
    return a.Metadata.z==Surface && all(a.Metadata.xy==b.Metadata.xy) && dot(a.NormalResidual.xyz,b.NormalResidual.xyz)>=LayerNormalCosine &&
        LayerFinite(tolerance) && abs(dot(offset,a.NormalResidual.xyz))<=tolerance && abs(dot(offset,b.NormalResidual.xyz))<=tolerance;
}
bool LayerLighting(Sample s)
{ return s.Geometry.Metadata.z==Surface && s.Cache.z==Complete && LayerFinite4(s.RadianceCoverage) && all(s.RadianceCoverage>=0) && s.RadianceCoverage.w>0; }
bool FilterLayer(Cascade layout,uint probe,Sample anchor,uint ordinal,float3 queryPosition,float3 delta,float3 direction,float minimum,out Sample answer)
{
    answer=anchor; float cosine=dot(anchor.Geometry.NormalResidual.xyz,direction);
    if (!LayerLighting(anchor) || abs(cosine)<MinimumLayerRayCosine) return false;
    float distance=dot(anchor.Geometry.PositionDistance.xyz-queryPosition,anchor.Geometry.NormalResidual.xyz)/cosine;
    if (!LayerFinite(distance) || distance<=minimum || distance>=Query.x) return false;
    // Radiance remains constant along a clear ray. Surface reprojection changes
    // its atlas address, without the old near-sphere solid-angle approximation.
    float3 projected=Sky.w>0?delta+distance*direction:direction;
    float projectedSquared=dot(projected,projected);
    if (!LayerFinite4(float4(projected,projectedSquared)) || projectedSquared<=0) return false;
    uint side=layout.Dimensions.w,rays=side*side,layers=uint(Transition.z),first=probe*layers*rays;
    float2 position=AngularUV(projected)*side-0.5,fraction=frac(position); int2 lower=int2(floor(position));
    int radius=int(Transition.y); float extent=float(radius+1),normalization=1/(extent*extent*extent*extent);
    float footprint=anchor.Geometry.Uncertainty.z; float4 light=0;
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,extent-abs(float2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0) continue;
        int2 pixel=FoldTexel(lower+int2(x,y),int(side)); uint texel=uint(pixel.y)*side+uint(pixel.x);
        bool found=false; float best=0; Sample chosen=anchor; uint chosenLayer=0;
        [loop] for (uint layer=0;layer<layers;++layer)
        {
            Sample s=WorldSamples[first+layer*rays+texel];
            if (!LayerLighting(s) || !SameLayerPlane(anchor.Geometry,s.Geometry)) continue;
            float plane=abs(dot(s.Geometry.PositionDistance.xyz-anchor.Geometry.PositionDistance.xyz,anchor.Geometry.NormalResidual.xyz));
            if (!found || plane<best) { found=true; best=plane; chosen=s; chosenLayer=layer; }
        }
        if (!found) return false;
        light+=weight*chosen.RadianceCoverage; ordinal=max(ordinal,chosenLayer);
        footprint=max(footprint,chosen.Geometry.Uncertainty.z+best);
    }
    if (!LayerFinite4(light) || any(light<0) || light.w<=0 || !LayerFinite(footprint)) return false;
    answer.Geometry.PositionDistance=float4(queryPosition+distance*direction,distance);
    answer.Geometry.NormalResidual=float4(anchor.Geometry.NormalResidual.xyz,0);
    answer.Geometry.Uncertainty=float4(0,0,footprint,0); answer.RadianceCoverage=light; answer.Cache=uint4(ordinal,0,Complete,0); return true;
}
bool ReadLayeredProbe(Cascade layout,uint probe,float3 position,float3 delta,float3 direction,float minimum,out Sample answer)
{
    uint side=layout.Dimensions.w,rays=side*side,layers=uint(Transition.z),first=probe*layers*rays;
    float2 address=AngularUV(direction)*side-0.5; int2 nearest=FoldTexel(int2(floor(address+0.5)),int(side));
    bool found=false; float nearestDistance=Query.x; answer=WorldSamples[first];
    [loop] for (uint layer=0;layer<layers;++layer)
    {
        Sample anchor=WorldSamples[first+layer*rays+uint(nearest.y)*side+uint(nearest.x)],candidate;
        if (FilterLayer(layout,probe,anchor,layer,position,delta,direction,minimum,candidate) && candidate.Geometry.PositionDistance.w<nearestDistance)
        { nearestDistance=candidate.Geometry.PositionDistance.w; answer=candidate; found=true; }
    }
    if (found) return true;
    // Sky is proposed only from a complete primary miss footprint. Layered
    // queries subsequently validate their full horizon, including this sky.
    int2 lower=int2(floor(address)); float2 fraction=frac(address); int radius=int(Transition.y); float extent=float(radius+1);
    [loop] for (int y=-radius;y<=radius+1;++y) [loop] for (int x=-radius;x<=radius+1;++x)
    {
        float2 tent=max(0,extent-abs(float2(x,y)-fraction)); if (tent.x*tent.y<=0) continue;
        int2 pixel=FoldTexel(lower+int2(x,y),int(side));
        if (WorldSamples[first+uint(pixel.y)*side+uint(pixel.x)].Geometry.Metadata.z!=Miss) return false;
    }
    answer.Geometry.PositionDistance=0; answer.Geometry.NormalResidual=0; answer.Geometry.Uncertainty=0;
    answer.Geometry.Metadata=uint4(Missing,Missing,Miss,0);
    answer.RadianceCoverage=float4(LumenEnvironmentRadiance(Sky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction),1);
    answer.Cache=uint4(Missing,0,Complete,0); return LayerFinite4(answer.RadianceCoverage);
}
// Each octave occupies a contiguous probe range in the shared radiance buffer.
bool ReadCascade(uint level,Origin source,float3 direction,out float4 radiance,out float edge)
{
    radiance=0; edge=0; Cascade layout=Cascades[level];
    float3 grid=(source.PositionMinimum.xyz-layout.OriginCell.xyz)*layout.Distances.w;
    if (!LayerFinite4(float4(grid,0)) || any(grid<0) || any(grid>float3(layout.Dimensions.xyz)-1) ||
        source.PositionMinimum.w>=layout.Distances.y || layout.Distances.y>=Query.x) return false;
    float3 face=min(grid,float3(layout.Dimensions.xyz)-1-grid); edge=min(face.x,min(face.y,face.z));
    uint3 cell=min(uint3(floor(grid)),layout.Dimensions.xyz-2); float3 fraction=saturate(grid-float3(cell));
    uint first=level*layout.Dimensions.x*layout.Dimensions.y*layout.Dimensions.z;
    [loop] for (uint corner=0;corner<8;++corner)
    {
        uint3 offset=uint3(corner&1,(corner>>1)&1,corner>>2),coordinate=cell+offset;
        float3 weights=float3(offset.x?fraction.x:1-fraction.x,offset.y?fraction.y:1-fraction.y,offset.z?fraction.z:1-fraction.z);
        float weight=weights.x*weights.y*weights.z; if (weight<=0) continue;
        float3 delta=source.PositionMinimum.xyz-(layout.OriginCell.xyz+float3(coordinate)*layout.OriginCell.w); float4 value=0;
        if (!ReadProbe(layout,first+(coordinate.z*layout.Dimensions.y+coordinate.y)*layout.Dimensions.x+coordinate.x,delta,direction,value)) return false;
        radiance+=weight*value;
    }
    return all(isfinite(radiance)) && radiance.w>0;
}
bool ReadLayeredCascade(uint level,Origin source,float3 direction,out Sample answer,out float edge)
{
    answer=(Sample)0; edge=0; Cascade layout=Cascades[level];
    float3 grid=(source.PositionMinimum.xyz-layout.OriginCell.xyz)*layout.Distances.w;
    if (!all(isfinite(grid)) || any(grid<0) || any(grid>float3(layout.Dimensions.xyz)-1) ||
        source.PositionMinimum.w>=layout.Distances.y || layout.Distances.y>=Query.x) return false;
    float3 face=min(grid,float3(layout.Dimensions.xyz)-1-grid); edge=min(face.x,min(face.y,face.z));
    uint3 cell=min(uint3(floor(grid)),layout.Dimensions.xyz-2); float3 fraction=saturate(grid-float3(cell));
    uint first=level*layout.Dimensions.x*layout.Dimensions.y*layout.Dimensions.z; bool initialized=false; float4 light=0;
    [loop] for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),coordinate=cell+bit;
        float3 weights=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
        float weight=weights.x*weights.y*weights.z; if (weight<=0) continue;
        float3 delta=source.PositionMinimum.xyz-layout.OriginCell.xyz-float3(coordinate)*layout.OriginCell.w;
        Sample candidate;
        if (!ReadLayeredProbe(layout,first+(coordinate.z*layout.Dimensions.y+coordinate.y)*layout.Dimensions.x+coordinate.x,
            source.PositionMinimum.xyz,delta,direction,source.PositionMinimum.w,candidate)) return false;
        if (!initialized) { answer=candidate; initialized=true; }
        else
        {
            if (!SameLayerPlane(answer.Geometry,candidate.Geometry)) return false;
            float plane=abs(dot(answer.Geometry.PositionDistance.xyz-candidate.Geometry.PositionDistance.xyz,answer.Geometry.NormalResidual.xyz));
            answer.Geometry.Uncertainty.z=max(answer.Geometry.Uncertainty.z,candidate.Geometry.Uncertainty.z+plane);
            answer.Cache.x=max(answer.Cache.x,candidate.Cache.x);
        }
        light+=weight*candidate.RadianceCoverage;
    }
    answer.RadianceCoverage=light; return initialized && LayerFinite4(light) && light.w>0;
}
[numthreads(GroupWidth,1,1)]
void CSWorldLookup(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Work.w; if (index>=Work.x) return;
    if (Query.y>0 && Hits[index].Metadata.z==ScreenHit) return;
    Hit hit; hit.Medium=float4(0,0,0,1); hit.PositionDistance=0; hit.NormalResidual=0; hit.Uncertainty=0; hit.Metadata=uint4(Missing,Missing,Unknown,0);
    Sample result; result.Geometry=hit; result.RadianceCoverage=0; result.Cache=uint4(Missing,0,0,0);
    Hits[index]=hit; Results[index]=result;
    Origin source=Origins[index/Work.y]; if (source.Metadata.y!=0) return;
    float3 direction=RayDirection(index,source.Metadata.x); uint count=min(uint(Query.w),MaximumCascades);
    if (Transition.z>1)
    {
        [loop] for (uint level=0;level<count;++level)
        {
            Sample candidate; float edge;
            if (!ReadLayeredCascade(level,source,direction,candidate,edge)) continue;
            uint selected=level;
            if (Transition.x>0 && edge<Transition.x && level+1<count)
            {
                Sample coarse; float ignored;
                if (ReadLayeredCascade(level+1,source,direction,coarse,ignored) && SameLayerPlane(candidate.Geometry,coarse.Geometry))
                {
                    float fineWeight=saturate(edge/Transition.x);
                    candidate.RadianceCoverage=lerp(coarse.RadianceCoverage,candidate.RadianceCoverage,fineWeight);
                    if (fineWeight==0) { candidate.Geometry=coarse.Geometry; candidate.Cache=coarse.Cache; selected=level+1; }
                    else
                    {
                        float plane=abs(dot(candidate.Geometry.PositionDistance.xyz-coarse.Geometry.PositionDistance.xyz,candidate.Geometry.NormalResidual.xyz));
                        candidate.Geometry.Uncertainty.z=max(candidate.Geometry.Uncertainty.z,coarse.Geometry.Uncertainty.z+plane);
                        candidate.Cache.x=max(candidate.Cache.x,coarse.Cache.x);
                    }
                }
            }
            hit=candidate.Geometry;
            if (hit.Metadata.z==Surface) hit.Uncertainty=float4(hit.Uncertainty.z,hit.Uncertainty.z,0,0);
            else hit.PositionDistance=float4(source.PositionMinimum.xyz+direction*Query.x,Query.x);
            hit.Metadata.z=WorldCacheHit; hit.Metadata.w=selected;
            candidate.Geometry=hit; candidate.RadianceCoverage.w=saturate(candidate.RadianceCoverage.w);
            Hits[index]=hit; Results[index]=candidate; return;
        }
        return;
    }
    [loop] for (uint level=0;level<count;++level)
    {
        float4 radiance=0; float edge=0; if (!ReadCascade(level,source,direction,radiance,edge)) continue;
        float localDistance=Cascades[level].Distances.y; uint selected=level;
        // Blend complete footprints only. The larger near interval protects both.
        if (Transition.x>0 && edge<Transition.x && level+1<count)
        {
            float4 coarse=0; float ignored=0;
            if (ReadCascade(level+1,source,direction,coarse,ignored))
            {
                float fineWeight=saturate(edge/Transition.x); radiance=lerp(coarse,radiance,fineWeight);
                localDistance=Cascades[level+1].Distances.y; if (fineWeight==0) selected=level+1;
            }
        }
        hit.PositionDistance=float4(source.PositionMinimum.xyz+direction*localDistance,localDistance); hit.Metadata.z=WorldCacheHit; hit.Metadata.w=selected;
        result.Geometry=hit; result.RadianceCoverage=float4(radiance.xyz,saturate(radiance.w)); result.Cache=uint4(Missing,0,Complete,0);
        Hits[index]=hit; Results[index]=result; return;
    }
}
