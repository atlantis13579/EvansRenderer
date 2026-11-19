#version 310 es
precision highp float;
precision highp int;
// Independent ES world-cache reconstruction. Four SSBOs fit the ES compute
// minimum. Raw records avoid driver-dependent nested structured-buffer lowering.
// Lumen (SIGGRAPH 2022); Clarberg's equal-area sphere map (JGT 2008).
layout(local_size_x=64) in;
const uint CascadeLimit=4u;
vec3 lumenEnvironmentRadiance(vec3 baseSky,vec4 axisPower,vec3 peak,vec3 direction)
{
    if (!any(greaterThan(peak,vec3(0.0)))) return baseSky;
    float cosine=clamp(dot(axisPower.xyz,direction),0.0,1.0);
    return baseSky+peak*pow(cosine,axisPower.w);
}
struct CascadeGrid { vec4 Lattice; uvec4 Shape; vec4 Range; };
layout(std140,binding=0) uniform Lookup
{
    CascadeGrid Grids[CascadeLimit];
    uvec4 Work; vec4 Sky,Query,Transition,EnvironmentAxis,EnvironmentPeak;
};
layout(std430,binding=0) readonly buffer CachedRays { uvec4 Words[]; } Cache;
layout(std430,binding=1) readonly buffer ProbeOrigins { uvec4 Words[]; } Origins;
layout(std430,binding=2) buffer TraceHits { uvec4 Words[]; } Hits;
layout(std430,binding=3) buffer RayResults { uvec4 Words[]; } Results;
const uint OriginStride=2u,HitStride=5u,ResultStride=7u,HitTag=3u,ColorVector=5u,CacheTag=6u;
const uint Missing=0xffffffffu,Miss=0u,Surface=1u,Unresolved=2u,ScreenHit=3u,WorldHit=4u,Complete=1u;
const float Pi=3.14159265358979323846,PhaseUnit=1.0/4294967296.0;
bool finiteValue(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 queryDirection(uint sampleId,uint identifier)
{
    uint phase=bitfieldReverse(sampleId)+bitfieldReverse(identifier^Work.z);
    float z=1.0-2.0*(float(sampleId)+0.5)/float(Work.y),angle=2.0*Pi*(float(phase)*PhaseUnit);
    float r=sqrt(max(0.0,1.0-z*z)); return vec3(r*cos(angle),r*sin(angle),z);
}
vec2 angularCoordinates(vec3 value)
{
    value=normalize(value); float r=sqrt(max(0.0,1.0-abs(value.z)));
    float angle=abs(value.x)+abs(value.y)>0.0?atan(abs(value.y),abs(value.x))*2.0/Pi:0.0;
    vec2 square=r*vec2(1.0-angle,angle); if (value.z<0.0) square=1.0-square.yx;
    return 0.5*(vec2(1.0)+square*vec2(value.x<0.0?-1.0:1.0,value.y<0.0?-1.0:1.0));
}
ivec2 mirrorAngular(ivec2 p,int edge)
{
    if (p.x<0) { p.x=-1-p.x; p.y=edge-1-p.y; }
    else if (p.x>=edge) { p.x=2*edge-1-p.x; p.y=edge-1-p.y; }
    if (p.y<0) { p.y=-1-p.y; p.x=edge-1-p.x; }
    else if (p.y>=edge) { p.y=2*edge-1-p.y; p.x=edge-1-p.x; }
    return clamp(p,ivec2(0),ivec2(edge-1));
}
bool filterProbe(CascadeGrid grid,uint probe,vec3 delta,vec3 ray,out vec4 light)
{
    light=vec4(0.0); vec3 corrected=ray; float energyScale=1.0;
    if (Sky.w>0.0)
    {
        float radius=grid.Range.x,b=dot(delta,ray),root=b*b+radius*radius-dot(delta,delta);
        if (root<=0.0) return false;
        float distance=sqrt(root)-b; corrected=delta+distance*ray;
        float area=radius*dot(corrected,ray); if (distance<=0.0 || area<=Query.z*radius*radius) return false;
        energyScale=distance*distance/area;
    }
    if (!finiteValue(vec4(corrected,energyScale))) return false;
    vec2 pixel=angularCoordinates(corrected)*float(grid.Shape.w)-0.5,blend=fract(pixel); ivec2 lower=ivec2(floor(pixel));
    int radius=int(Transition.y); float extent=float(radius+1),normalization=1.0/(extent*extent*extent*extent);
    ivec2 center=mirrorAngular(ivec2(floor(pixel+0.5)),int(grid.Shape.w));
    uint first=probe*uint(Transition.z)*grid.Shape.w*grid.Shape.w;
    uint reference=(first+uint(center.y)*grid.Shape.w+uint(center.x))*ResultStride;
    const uint PositionVector=0u,NormalVector=1u,UncertaintyVector=2u;
    const float MinimumFilterNormalCosine=0.95;
    // Fixed tent mass avoids normalizing incomplete rays into artificial light.
    // Widened footprints must retain one surface and its tangent-plane depth.
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),extent-abs(vec2(x,y)-blend)); float weight=tent.x*tent.y*normalization; if (weight<=0.0) continue;
        ivec2 p=mirrorAngular(lower+ivec2(x,y),int(grid.Shape.w)); uint address=(first+uint(p.y)*grid.Shape.w+uint(p.x))*ResultStride;
        uint status=Cache.Words[address+HitTag].z; vec4 incoming;
        if (radius>0)
        {
            uvec4 tag=Cache.Words[address+HitTag],anchorTag=Cache.Words[reference+HitTag];
            if (tag.z!=anchorTag.z) return false;
            if (status==Surface)
            {
                vec3 position=uintBitsToFloat(Cache.Words[address+PositionVector]).xyz,anchorPosition=uintBitsToFloat(Cache.Words[reference+PositionVector]).xyz;
                vec3 normal=uintBitsToFloat(Cache.Words[address+NormalVector]).xyz,anchorNormal=uintBitsToFloat(Cache.Words[reference+NormalVector]).xyz;
                float roundoff=Query.z*max(1.0,max(length(position),length(anchorPosition)));
                float footprint=uintBitsToFloat(Cache.Words[address+UncertaintyVector]).y+uintBitsToFloat(Cache.Words[reference+UncertaintyVector]).y+roundoff;
                if (isnan(footprint) || isinf(footprint)) return false;
                vec3 separation=position-anchorPosition;
                if (any(notEqual(tag.xy,anchorTag.xy)) || dot(normal,anchorNormal)<MinimumFilterNormalCosine ||
                    abs(dot(separation,normal))>footprint || abs(dot(separation,anchorNormal))>footprint) return false;
            }
        }
        if (status==Miss) incoming=vec4(lumenEnvironmentRadiance(Sky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,ray),1.0);
        else if (status==Surface && Cache.Words[address+CacheTag].z==Complete)
        { vec4 value=uintBitsToFloat(Cache.Words[address+ColorVector]); incoming=vec4(value.xyz*energyScale,value.w); }
        else return false;
        if (!finiteValue(incoming) || any(lessThan(incoming,vec4(0.0))) || incoming.w<=0.0) return false;
        light+=weight*incoming;
    }
    return true;
}
// These are local values only: the ES storage interface remains raw uvec4
// records, avoiding ANGLE's nested structured-buffer load limitations.
struct DepthGeometry { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct DepthValue { DepthGeometry Hit; vec4 Light; uvec4 Cache; };
const float DepthNormalAgreement=0.95,DepthRayCosineFloor=0.05;
DepthValue readDepthRecord(uint index)
{
    uint at=index*ResultStride; DepthValue value; value.Hit.Medium=vec4(0,0,0,1);
    value.Hit.PositionDistance=uintBitsToFloat(Cache.Words[at]); value.Hit.NormalResidual=uintBitsToFloat(Cache.Words[at+1u]);
    value.Hit.Uncertainty=uintBitsToFloat(Cache.Words[at+2u]); value.Hit.Metadata=Cache.Words[at+HitTag];
    value.Light=uintBitsToFloat(Cache.Words[at+ColorVector]); value.Cache=Cache.Words[at+CacheTag]; return value;
}
float depthPrecision(vec3 a,vec3 b)
{ return Query.z*max(1.0,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))); }
bool sameDepthSurface(DepthGeometry a,DepthGeometry b)
{
    if (a.Metadata.z!=b.Metadata.z) return false;
    if (a.Metadata.z==Miss) return true;
    float allowance=a.Uncertainty.z+b.Uncertainty.z+depthPrecision(a.PositionDistance.xyz,b.PositionDistance.xyz);
    vec3 delta=a.PositionDistance.xyz-b.PositionDistance.xyz;
    return a.Metadata.z==Surface && all(equal(a.Metadata.xy,b.Metadata.xy)) && dot(a.NormalResidual.xyz,b.NormalResidual.xyz)>=DepthNormalAgreement &&
        !isnan(allowance) && !isinf(allowance) && abs(dot(delta,a.NormalResidual.xyz))<=allowance && abs(dot(delta,b.NormalResidual.xyz))<=allowance;
}
bool validDepthLight(DepthValue value)
{ return value.Hit.Metadata.z==Surface && value.Cache.z==Complete && finiteValue(value.Light) && all(greaterThanEqual(value.Light,vec4(0.0))) && value.Light.w>0.0; }
bool filterDepthPlane(CascadeGrid grid,uint probe,DepthValue anchor,uint ordinal,vec4 origin,vec3 delta,vec3 ray,out DepthValue answer)
{
    answer=anchor; float cosine=dot(anchor.Hit.NormalResidual.xyz,ray);
    if (!validDepthLight(anchor) || abs(cosine)<DepthRayCosineFloor) return false;
    float distance=dot(anchor.Hit.PositionDistance.xyz-origin.xyz,anchor.Hit.NormalResidual.xyz)/cosine;
    if (isnan(distance) || isinf(distance) || distance<=origin.w || distance>=Query.x) return false;
    // A clear ray preserves radiance. The plane changes the angular address;
    // its light is not rescaled by the old near-sphere solid-angle estimate.
    vec3 projected=Sky.w>0.0?delta+distance*ray:ray; float squared=dot(projected,projected);
    if (!finiteValue(vec4(projected,squared)) || squared<=0.0) return false;
    uint side=grid.Shape.w,rays=side*side,layers=uint(Transition.z),first=probe*layers*rays;
    vec2 pixel=angularCoordinates(projected)*float(side)-0.5,fraction=fract(pixel); ivec2 lower=ivec2(floor(pixel));
    int radius=int(Transition.y); float extent=float(radius+1),normalization=1.0/(extent*extent*extent*extent);
    vec4 light=vec4(0.0); float footprint=anchor.Hit.Uncertainty.z;
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),extent-abs(vec2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0.0) continue;
        ivec2 p=mirrorAngular(lower+ivec2(x,y),int(side)); uint texel=uint(p.y)*side+uint(p.x);
        bool found=false; float best=0.0; DepthValue selected=anchor; uint selectedLayer=0u;
        for (uint layer=0u;layer<layers;++layer)
        {
            DepthValue candidate=readDepthRecord(first+layer*rays+texel);
            if (!validDepthLight(candidate) || !sameDepthSurface(anchor.Hit,candidate.Hit)) continue;
            float plane=abs(dot(candidate.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz,anchor.Hit.NormalResidual.xyz));
            if (!found || plane<best) { found=true; best=plane; selected=candidate; selectedLayer=layer; }
        }
        if (!found) return false;
        light+=weight*selected.Light; footprint=max(footprint,selected.Hit.Uncertainty.z+best); ordinal=max(ordinal,selectedLayer);
    }
    if (!finiteValue(light) || any(lessThan(light,vec4(0.0))) || light.w<=0.0 || isnan(footprint) || isinf(footprint)) return false;
    answer.Hit.PositionDistance=vec4(origin.xyz+distance*ray,distance); answer.Hit.NormalResidual=vec4(anchor.Hit.NormalResidual.xyz,0.0);
    answer.Hit.Uncertainty=vec4(0.0,0.0,footprint,0.0); answer.Light=light; answer.Cache=uvec4(ordinal,0u,Complete,0u); return true;
}
bool filterDepthProbe(CascadeGrid grid,uint probe,vec4 origin,vec3 delta,vec3 ray,out DepthValue answer)
{
    uint side=grid.Shape.w,rays=side*side,layers=uint(Transition.z),first=probe*layers*rays;
    vec2 address=angularCoordinates(ray)*float(side)-0.5; ivec2 center=mirrorAngular(ivec2(floor(address+0.5)),int(side));
    answer=readDepthRecord(first); bool found=false; float closest=Query.x;
    for (uint layer=0u;layer<layers;++layer)
    {
        DepthValue anchor=readDepthRecord(first+layer*rays+uint(center.y)*side+uint(center.x)),candidate;
        if (filterDepthPlane(grid,probe,anchor,layer,origin,delta,ray,candidate) && candidate.Hit.PositionDistance.w<closest)
        { answer=candidate; closest=candidate.Hit.PositionDistance.w; found=true; }
    }
    if (found) return true;
    ivec2 lower=ivec2(floor(address)); vec2 fraction=fract(address); int radius=int(Transition.y); float extent=float(radius+1);
    // Miss-only footprints can propose sky. A full-horizon SDF miss remains
    // necessary before the next stage accepts this layered sky proposal.
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),extent-abs(vec2(x,y)-fraction)); if (tent.x*tent.y<=0.0) continue;
        ivec2 p=mirrorAngular(lower+ivec2(x,y),int(side));
        if (Cache.Words[(first+uint(p.y)*side+uint(p.x))*ResultStride+HitTag].z!=Miss) return false;
    }
    answer.Hit.PositionDistance=vec4(0.0); answer.Hit.NormalResidual=vec4(0.0); answer.Hit.Uncertainty=vec4(0.0);
    answer.Hit.Metadata=uvec4(Missing,Missing,Miss,0u); answer.Light=vec4(lumenEnvironmentRadiance(Sky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,ray),1.0);
    answer.Cache=uvec4(Missing,0u,Complete,0u); return finiteValue(answer.Light);
}
bool filterLevel(uint level,vec4 origin,vec3 direction,out vec4 accumulated,out float edge)
{
    CascadeGrid grid=Grids[level]; accumulated=vec4(0.0); edge=0.0;
    vec3 location=(origin.xyz-grid.Lattice.xyz)*grid.Range.w;
    if (!finiteValue(vec4(location,0.0)) || any(lessThan(location,vec3(0.0))) || any(greaterThan(location,vec3(grid.Shape.xyz)-1.0)) ||
        origin.w>=grid.Range.y || grid.Range.y>=Query.x) return false;
    vec3 faces=min(location,vec3(grid.Shape.xyz)-1.0-location); edge=min(faces.x,min(faces.y,faces.z));
    uvec3 cell=min(uvec3(floor(location)),grid.Shape.xyz-2u); vec3 blend=clamp(location-vec3(cell),0.0,1.0);
    uint baseProbe=level*grid.Shape.x*grid.Shape.y*grid.Shape.z;
    for (uint z=0u;z<2u;++z) for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
    {
        float weight=(x!=0u?blend.x:1.0-blend.x)*(y!=0u?blend.y:1.0-blend.y)*(z!=0u?blend.z:1.0-blend.z); if (weight<=0.0) continue;
        uvec3 p=cell+uvec3(x,y,z); vec3 offset=origin.xyz-grid.Lattice.xyz-vec3(p)*grid.Lattice.w; vec4 value;
        if (!filterProbe(grid,baseProbe+(p.z*grid.Shape.y+p.y)*grid.Shape.x+p.x,offset,direction,value)) return false;
        accumulated+=weight*value;
    }
    return finiteValue(accumulated) && accumulated.w>0.0;
}
bool filterDepthLevel(uint level,vec4 origin,vec3 ray,out DepthValue answer,out float edge)
{
    CascadeGrid grid=Grids[level]; answer=readDepthRecord(0u); edge=0.0;
    vec3 location=(origin.xyz-grid.Lattice.xyz)*grid.Range.w;
    if (!finiteValue(vec4(location,0.0)) || any(lessThan(location,vec3(0.0))) || any(greaterThan(location,vec3(grid.Shape.xyz)-1.0)) ||
        origin.w>=grid.Range.y || grid.Range.y>=Query.x) return false;
    vec3 faces=min(location,vec3(grid.Shape.xyz)-1.0-location); edge=min(faces.x,min(faces.y,faces.z));
    uvec3 cell=min(uvec3(floor(location)),grid.Shape.xyz-2u); vec3 fraction=clamp(location-vec3(cell),0.0,1.0);
    uint first=level*grid.Shape.x*grid.Shape.y*grid.Shape.z; bool initialized=false; vec4 light=vec4(0.0);
    for (uint z=0u;z<2u;++z) for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
    {
        float weight=(x!=0u?fraction.x:1.0-fraction.x)*(y!=0u?fraction.y:1.0-fraction.y)*(z!=0u?fraction.z:1.0-fraction.z); if (weight<=0.0) continue;
        uvec3 p=cell+uvec3(x,y,z); vec3 delta=origin.xyz-grid.Lattice.xyz-vec3(p)*grid.Lattice.w; DepthValue candidate; candidate.Hit.Medium=vec4(0,0,0,1);
        if (!filterDepthProbe(grid,first+(p.z*grid.Shape.y+p.y)*grid.Shape.x+p.x,origin,delta,ray,candidate)) return false;
        if (!initialized) { answer=candidate; initialized=true; }
        else
        {
            if (!sameDepthSurface(answer.Hit,candidate.Hit)) return false;
            float plane=abs(dot(answer.Hit.PositionDistance.xyz-candidate.Hit.PositionDistance.xyz,answer.Hit.NormalResidual.xyz));
            answer.Hit.Uncertainty.z=max(answer.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z+plane); answer.Cache.x=max(answer.Cache.x,candidate.Cache.x);
        }
        light+=weight*candidate.Light;
    }
    answer.Light=light; return initialized && finiteValue(light) && light.w>0.0;
}
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Work.w; if (index>=Work.x) return;
    uint hit=index*HitStride,result=index*ResultStride;
    if (Query.y>0.0 && Hits.Words[hit+HitTag].z==ScreenHit) return;
    for (uint field=0u;field<HitTag;++field) { Hits.Words[hit+field]=uvec4(0u); Results.Words[result+field]=uvec4(0u); }
    Hits.Words[hit+HitTag]=uvec4(Missing,Missing,Unresolved,0u); Results.Words[result+HitTag]=uvec4(Missing,Missing,Unresolved,0u);
    Hits.Words[hit+4u]=floatBitsToUint(vec4(0,0,0,1)); Results.Words[result+4u]=floatBitsToUint(vec4(0,0,0,1));
    Results.Words[result+ColorVector]=uvec4(0u); Results.Words[result+CacheTag]=uvec4(Missing,0u,0u,0u);
    uint source=(index/Work.y)*OriginStride; vec4 origin=uintBitsToFloat(Origins.Words[source]); uvec4 tag=Origins.Words[source+1u]; if (tag.y!=0u) return;
    vec3 direction=queryDirection(index%Work.y,tag.x); uint count=min(uint(Query.w),CascadeLimit);
    if (Transition.z>1.0)
    {
        for (uint level=0u;level<count;++level)
        {
            DepthValue candidate; candidate.Hit.Medium=vec4(0,0,0,1); float edge; if (!filterDepthLevel(level,origin,direction,candidate,edge)) continue;
            uint selected=level;
            if (Transition.x>0.0 && edge<Transition.x && level+1u<count)
            {
                DepthValue coarse; coarse.Hit.Medium=vec4(0,0,0,1); float ignored;
                if (filterDepthLevel(level+1u,origin,direction,coarse,ignored) && sameDepthSurface(candidate.Hit,coarse.Hit))
                {
                    float fineWeight=clamp(edge/Transition.x,0.0,1.0); candidate.Light=mix(coarse.Light,candidate.Light,fineWeight);
                    if (fineWeight==0.0) { candidate.Hit=coarse.Hit; candidate.Cache=coarse.Cache; selected=level+1u; }
                    else
                    {
                        float plane=abs(dot(candidate.Hit.PositionDistance.xyz-coarse.Hit.PositionDistance.xyz,candidate.Hit.NormalResidual.xyz));
                        candidate.Hit.Uncertainty.z=max(candidate.Hit.Uncertainty.z,coarse.Hit.Uncertainty.z+plane); candidate.Cache.x=max(candidate.Cache.x,coarse.Cache.x);
                    }
                }
            }
            if (candidate.Hit.Metadata.z==Surface) candidate.Hit.Uncertainty=vec4(candidate.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z,0.0,0.0);
            else candidate.Hit.PositionDistance=vec4(origin.xyz+direction*Query.x,Query.x);
            candidate.Hit.Metadata.z=WorldHit; candidate.Hit.Metadata.w=selected; candidate.Light.w=clamp(candidate.Light.w,0.0,1.0);
            uvec4 positionWord=floatBitsToUint(candidate.Hit.PositionDistance),normalWord=floatBitsToUint(candidate.Hit.NormalResidual),uncertaintyWord=floatBitsToUint(candidate.Hit.Uncertainty);
            Hits.Words[hit]=positionWord; Hits.Words[hit+1u]=normalWord; Hits.Words[hit+2u]=uncertaintyWord; Hits.Words[hit+HitTag]=candidate.Hit.Metadata;
            Results.Words[result]=positionWord; Results.Words[result+1u]=normalWord; Results.Words[result+2u]=uncertaintyWord; Results.Words[result+HitTag]=candidate.Hit.Metadata;
            Results.Words[result+ColorVector]=floatBitsToUint(candidate.Light); Results.Words[result+CacheTag]=candidate.Cache; return;
        }
        return;
    }
    for (uint level=0u;level<count;++level)
    {
        vec4 accumulated; float edge; if (!filterLevel(level,origin,direction,accumulated,edge)) continue;
        float localDistance=Grids[level].Range.y; uint selected=level;
        if (Transition.x>0.0 && edge<Transition.x && level+1u<count)
        {
            vec4 coarse; float unusedEdge;
            if (filterLevel(level+1u,origin,direction,coarse,unusedEdge))
            {
                float fineWeight=clamp(edge/Transition.x,0.0,1.0); accumulated=coarse*(1.0-fineWeight)+accumulated*fineWeight;
                localDistance=Grids[level+1u].Range.y; if (fineWeight==0.0) selected=level+1u;
            }
        }
        uvec4 endpoint=floatBitsToUint(vec4(origin.xyz+direction*localDistance,localDistance)),metadata=uvec4(Missing,Missing,WorldHit,selected);
        Hits.Words[hit]=endpoint; Hits.Words[hit+HitTag]=metadata; Results.Words[result]=endpoint; Results.Words[result+HitTag]=metadata;
        Results.Words[result+ColorVector]=floatBitsToUint(vec4(accumulated.xyz,clamp(accumulated.w,0.0,1.0))); Results.Words[result+CacheTag]=uvec4(Missing,0u,Complete,0u); return;
    }
}
