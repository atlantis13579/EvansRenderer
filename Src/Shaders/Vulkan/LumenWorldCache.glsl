#version 450
// Lumen world radiance reconstruction (Wright et al., SIGGRAPH 2022).
// Trilinear probes and bilinear equal-area directions retain explicit unknown
// data. The subsequent SDF dispatch accepts candidates only after a local miss.
layout(local_size_x=64) in;
const uint MaximumCascades=4u;
vec3 lumenEnvironmentRadiance(vec3 baseSky,vec4 axisPower,vec3 peak,vec3 direction)
{
    if (!any(greaterThan(peak,vec3(0.0)))) return baseSky;
    float cosine=clamp(dot(axisPower.xyz,direction),0.0,1.0);
    return baseSky+peak*pow(cosine,axisPower.w);
}
struct Cascade { vec4 OriginCell; uvec4 Dimensions; vec4 Distances; };
layout(std140,binding=0) uniform LookupParameters
{
    Cascade Cascades[MaximumCascades];
    uvec4 Work; vec4 Sky,Query,Transition,EnvironmentAxis,EnvironmentPeak;
} Frame;
struct Geometry { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct Radiance { Geometry Hit; vec4 ColorCoverage; uvec4 Cache; };
struct Origin { vec4 PositionMinimum; uvec4 Metadata; };
layout(std430,binding=1) readonly buffer CacheStorage { Radiance Samples[]; } World;
layout(std430,binding=2) readonly buffer OriginStorage { Origin Seeds[]; } Input;
layout(std430,binding=3) buffer HitStorage { Geometry Values[]; } Hits;
layout(std430,binding=4) buffer ResultStorage { Radiance Values[]; } Output;
const uint Invalid=0xffffffffu,Miss=0u,Surface=1u,Unknown=2u,Screen=3u,WorldHit=4u,Complete=1u;
const float Pi=3.14159265358979323846,UnsignedReciprocal=1.0/4294967296.0;
bool finite4(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
vec3 probeRay(uint ray,uint id)
{
    uint sampleId=ray%Frame.Work.y,phase=bitfieldReverse(sampleId)+bitfieldReverse(id^Frame.Work.z);
    float z=1.0-2.0*(float(sampleId)+0.5)/float(Frame.Work.y),phi=2.0*Pi*(float(phase)*UnsignedReciprocal);
    float r=sqrt(max(0.0,1.0-z*z)); return vec3(r*cos(phi),r*sin(phi),z);
}
vec2 directionUV(vec3 direction)
{
    vec3 n=normalize(direction); float radius=sqrt(max(0.0,1.0-abs(n.z)));
    float angle=abs(n.x)+abs(n.y)>0.0?atan(abs(n.y),abs(n.x))*2.0/Pi:0.0;
    vec2 p=radius*vec2(1.0-angle,angle); if (n.z<0.0) p=1.0-p.yx;
    return (vec2(1.0)+p*vec2(n.x<0.0?-1.0:1.0,n.y<0.0?-1.0:1.0))*0.5;
}
ivec2 wrapSquare(ivec2 p,int size)
{
    // Clarberg 2008: each equal-area square edge reflects into its partner.
    if (p.x<0) { p.x=-p.x-1; p.y=size-1-p.y; }
    if (p.x>=size) { p.x=2*size-p.x-1; p.y=size-1-p.y; }
    if (p.y<0) { p.y=-p.y-1; p.x=size-1-p.x; }
    if (p.y>=size) { p.y=2*size-p.y-1; p.x=size-1-p.x; }
    return clamp(p,ivec2(0),ivec2(size-1));
}
bool sampleWorld(Cascade cascade,uint id,vec3 displacement,vec3 direction,out vec4 radiance)
{
    radiance=vec4(0.0); vec3 projected=direction; float solidAngleRatio=1.0;
    if (Frame.Sky.w>0.0)
    {
        float radius=cascade.Distances.x,b=dot(displacement,direction);
        float radicand=b*b+radius*radius-dot(displacement,displacement); if (radicand<=0.0) return false;
        float t=-b+sqrt(radicand); projected=displacement+t*direction;
        float area=radius*dot(projected,direction);
        if (t<=0.0 || area<=Frame.Query.z*radius*radius) return false;
        solidAngleRatio=t*t/area;
    }
    if (!finite4(vec4(projected,solidAngleRatio))) return false;
    vec2 texel=directionUV(projected)*float(cascade.Dimensions.w)-0.5;
    ivec2 lower=ivec2(floor(texel)); vec2 fraction=fract(texel);
    int radius=int(Frame.Transition.y); float support=float(radius+1),normalization=1.0/(support*support*support*support);
    ivec2 nearest=wrapSquare(ivec2(floor(texel+0.5)),int(cascade.Dimensions.w));
    uint first=id*uint(Frame.Transition.z)*cascade.Dimensions.w*cascade.Dimensions.w;
    Radiance anchor=World.Samples[first+uint(nearest.y)*cascade.Dimensions.w+uint(nearest.x)];
    // This tent has fixed unit mass. Unknown data and depth discontinuities
    // reject its entire footprint, leaving the mesh-SDF fallback responsible.
    const float MinimumNormalAgreement=0.95;
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),support-abs(vec2(x,y)-fraction));
        float weight=tent.x*tent.y*normalization; if (weight<=0.0) continue;
        ivec2 p=wrapSquare(lower+ivec2(x,y),int(cascade.Dimensions.w));
        Radiance value=World.Samples[first+uint(p.y)*cascade.Dimensions.w+uint(p.x)];
        if (radius>0)
        {
            if (value.Hit.Metadata.z!=anchor.Hit.Metadata.z) return false;
            if (value.Hit.Metadata.z==Surface)
            {
                vec3 separation=value.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz;
                float roundoff=Frame.Query.z*max(1.0,max(length(value.Hit.PositionDistance.xyz),length(anchor.Hit.PositionDistance.xyz)));
                float footprint=value.Hit.Uncertainty.y+anchor.Hit.Uncertainty.y+roundoff;
                if (isnan(footprint) || isinf(footprint)) return false;
                if (any(notEqual(value.Hit.Metadata.xy,anchor.Hit.Metadata.xy)) || dot(value.Hit.NormalResidual.xyz,anchor.Hit.NormalResidual.xyz)<MinimumNormalAgreement ||
                    abs(dot(separation,value.Hit.NormalResidual.xyz))>footprint || abs(dot(separation,anchor.Hit.NormalResidual.xyz))>footprint) return false;
            }
        }
        vec4 light;
        if (value.Hit.Metadata.z==Miss) light=vec4(lumenEnvironmentRadiance(Frame.Sky.xyz,Frame.EnvironmentAxis,Frame.EnvironmentPeak.xyz,direction),1.0);
        else if (value.Hit.Metadata.z==Surface && value.Cache.z==Complete) light=vec4(value.ColorCoverage.xyz*solidAngleRatio,value.ColorCoverage.w);
        else return false;
        if (!finite4(light) || any(lessThan(light,vec4(0.0))) || light.w<=0.0) return false;
        radiance+=weight*light;
    }
    return true;
}
const float LayerNormalCosine=0.95,LayerRayCosineFloor=0.05;
float layerRoundoff(vec3 a,vec3 b)
{ return Frame.Query.z*max(1.0,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))); }
bool sameDepthPlane(Geometry a,Geometry b)
{
    if (a.Metadata.z!=b.Metadata.z) return false;
    if (a.Metadata.z==Miss) return true;
    float allowance=a.Uncertainty.z+b.Uncertainty.z+layerRoundoff(a.PositionDistance.xyz,b.PositionDistance.xyz);
    vec3 separation=a.PositionDistance.xyz-b.PositionDistance.xyz;
    return a.Metadata.z==Surface && all(equal(a.Metadata.xy,b.Metadata.xy)) && dot(a.NormalResidual.xyz,b.NormalResidual.xyz)>=LayerNormalCosine &&
        !isnan(allowance) && !isinf(allowance) && abs(dot(separation,a.NormalResidual.xyz))<=allowance && abs(dot(separation,b.NormalResidual.xyz))<=allowance;
}
bool completeLayer(Radiance value)
{ return value.Hit.Metadata.z==Surface && value.Cache.z==Complete && finite4(value.ColorCoverage) && all(greaterThanEqual(value.ColorCoverage,vec4(0.0))) && value.ColorCoverage.w>0.0; }
bool filterDepthSurface(Cascade grid,uint probe,Radiance anchor,uint ordinal,vec4 query,vec3 delta,vec3 ray,out Radiance answer)
{
    answer=anchor; float cosine=dot(anchor.Hit.NormalResidual.xyz,ray);
    if (!completeLayer(anchor) || abs(cosine)<LayerRayCosineFloor) return false;
    float travel=dot(anchor.Hit.PositionDistance.xyz-query.xyz,anchor.Hit.NormalResidual.xyz)/cosine;
    if (isnan(travel) || isinf(travel) || travel<=query.w || travel>=Frame.Query.x) return false;
    // Surface reprojection conserves radiance; it changes the atlas address,
    // rather than applying a near-sphere area correction to a distant plane.
    vec3 mapped=Frame.Sky.w>0.0?delta+travel*ray:ray; float square=dot(mapped,mapped);
    if (!finite4(vec4(mapped,square)) || square<=0.0) return false;
    uint side=grid.Dimensions.w,rays=side*side,layers=uint(Frame.Transition.z),first=probe*layers*rays;
    vec2 address=directionUV(mapped)*float(side)-0.5,fraction=fract(address); ivec2 lower=ivec2(floor(address));
    int radius=int(Frame.Transition.y); float support=float(radius+1),normalization=1.0/(support*support*support*support);
    float footprint=anchor.Hit.Uncertainty.z; vec4 light=vec4(0.0);
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),support-abs(vec2(x,y)-fraction)); float weight=tent.x*tent.y*normalization; if (weight<=0.0) continue;
        ivec2 p=wrapSquare(lower+ivec2(x,y),int(side)); uint pixel=uint(p.y)*side+uint(p.x);
        bool found=false; float best=0.0; Radiance selected=anchor; uint selectedDepth=0u;
        for (uint depth=0u;depth<layers;++depth)
        {
            Radiance candidate=World.Samples[first+depth*rays+pixel];
            if (!completeLayer(candidate) || !sameDepthPlane(anchor.Hit,candidate.Hit)) continue;
            float plane=abs(dot(candidate.Hit.PositionDistance.xyz-anchor.Hit.PositionDistance.xyz,anchor.Hit.NormalResidual.xyz));
            if (!found || plane<best) { found=true; best=plane; selected=candidate; selectedDepth=depth; }
        }
        if (!found) return false;
        light+=weight*selected.ColorCoverage; ordinal=max(ordinal,selectedDepth); footprint=max(footprint,selected.Hit.Uncertainty.z+best);
    }
    if (!finite4(light) || any(lessThan(light,vec4(0.0))) || light.w<=0.0 || isnan(footprint) || isinf(footprint)) return false;
    answer.Hit.PositionDistance=vec4(query.xyz+travel*ray,travel); answer.Hit.NormalResidual=vec4(anchor.Hit.NormalResidual.xyz,0.0);
    answer.Hit.Uncertainty=vec4(0.0,0.0,footprint,0.0); answer.ColorCoverage=light; answer.Cache=uvec4(ordinal,0u,Complete,0u); return true;
}
bool sampleDepthProbe(Cascade grid,uint probe,vec4 query,vec3 delta,vec3 ray,out Radiance answer)
{
    uint side=grid.Dimensions.w,rays=side*side,layers=uint(Frame.Transition.z),first=probe*layers*rays;
    vec2 address=directionUV(ray)*float(side)-0.5; ivec2 nearest=wrapSquare(ivec2(floor(address+0.5)),int(side));
    answer=World.Samples[first]; bool found=false; float closest=Frame.Query.x;
    for (uint depth=0u;depth<layers;++depth)
    {
        Radiance anchor=World.Samples[first+depth*rays+uint(nearest.y)*side+uint(nearest.x)],candidate;
        if (filterDepthSurface(grid,probe,anchor,depth,query,delta,ray,candidate) && candidate.Hit.PositionDistance.w<closest)
        { answer=candidate; closest=candidate.Hit.PositionDistance.w; found=true; }
    }
    if (found) return true;
    ivec2 lower=ivec2(floor(address)); vec2 fraction=fract(address); int radius=int(Frame.Transition.y); float support=float(radius+1);
    // A complete primary miss footprint can propose sky. The following SDF
    // trace still checks its entire query horizon before accepting that sky.
    for (int y=-radius;y<=radius+1;++y) for (int x=-radius;x<=radius+1;++x)
    {
        vec2 tent=max(vec2(0.0),support-abs(vec2(x,y)-fraction)); if (tent.x*tent.y<=0.0) continue;
        ivec2 p=wrapSquare(lower+ivec2(x,y),int(side)); if (World.Samples[first+uint(p.y)*side+uint(p.x)].Hit.Metadata.z!=Miss) return false;
    }
    answer.Hit=Geometry(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(Invalid,Invalid,Miss,0u),vec4(0,0,0,1));
    answer.ColorCoverage=vec4(lumenEnvironmentRadiance(Frame.Sky.xyz,Frame.EnvironmentAxis,Frame.EnvironmentPeak.xyz,ray),1.0);
    answer.Cache=uvec4(Invalid,0u,Complete,0u); return finite4(answer.ColorCoverage);
}
bool sampleCascade(uint level,Origin origin,vec3 direction,out vec4 sum,out float border)
{
    Cascade cascade=Frame.Cascades[level]; sum=vec4(0.0); border=0.0;
    vec3 grid=(origin.PositionMinimum.xyz-cascade.OriginCell.xyz)*cascade.Distances.w;
    if (!finite4(vec4(grid,0.0)) || any(lessThan(grid,vec3(0.0))) || any(greaterThan(grid,vec3(cascade.Dimensions.xyz)-1.0)) ||
        origin.PositionMinimum.w>=cascade.Distances.y || cascade.Distances.y>=Frame.Query.x) return false;
    vec3 faces=min(grid,vec3(cascade.Dimensions.xyz)-1.0-grid); border=min(faces.x,min(faces.y,faces.z));
    uvec3 cell=min(uvec3(floor(grid)),cascade.Dimensions.xyz-2u); vec3 alpha=clamp(grid-vec3(cell),0.0,1.0);
    uint firstProbe=level*cascade.Dimensions.x*cascade.Dimensions.y*cascade.Dimensions.z;
    for (uint corner=0u;corner<8u;++corner)
    {
        uvec3 offset=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=cell+offset;
        vec3 weights=mix(1.0-alpha,alpha,notEqual(offset,uvec3(0u))); float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
        vec3 delta=origin.PositionMinimum.xyz-(cascade.OriginCell.xyz+vec3(p)*cascade.OriginCell.w); vec4 value;
        if (!sampleWorld(cascade,firstProbe+(p.z*cascade.Dimensions.y+p.y)*cascade.Dimensions.x+p.x,delta,direction,value)) return false;
        sum+=weight*value;
    }
    return finite4(sum) && sum.w>0.0;
}
bool sampleDepthCascade(uint level,Origin origin,vec3 ray,out Radiance answer,out float border)
{
    Cascade grid=Frame.Cascades[level]; answer=Radiance(Geometry(vec4(0),vec4(0),vec4(0),uvec4(0),vec4(0,0,0,1)),vec4(0),uvec4(0)); border=0.0;
    vec3 location=(origin.PositionMinimum.xyz-grid.OriginCell.xyz)*grid.Distances.w;
    if (!finite4(vec4(location,0.0)) || any(lessThan(location,vec3(0.0))) || any(greaterThan(location,vec3(grid.Dimensions.xyz)-1.0)) ||
        origin.PositionMinimum.w>=grid.Distances.y || grid.Distances.y>=Frame.Query.x) return false;
    vec3 faces=min(location,vec3(grid.Dimensions.xyz)-1.0-location); border=min(faces.x,min(faces.y,faces.z));
    uvec3 cell=min(uvec3(floor(location)),grid.Dimensions.xyz-2u); vec3 fraction=clamp(location-vec3(cell),0.0,1.0);
    uint first=level*grid.Dimensions.x*grid.Dimensions.y*grid.Dimensions.z; bool initialized=false; vec4 light=vec4(0.0);
    for (uint corner=0u;corner<8u;++corner)
    {
        uvec3 offset=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=cell+offset;
        vec3 weights=mix(1.0-fraction,fraction,notEqual(offset,uvec3(0u))); float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
        vec3 delta=origin.PositionMinimum.xyz-grid.OriginCell.xyz-vec3(p)*grid.OriginCell.w; Radiance candidate; candidate.Hit.Medium=vec4(0,0,0,1);
        if (!sampleDepthProbe(grid,first+(p.z*grid.Dimensions.y+p.y)*grid.Dimensions.x+p.x,origin.PositionMinimum,delta,ray,candidate)) return false;
        if (!initialized) { answer=candidate; initialized=true; }
        else
        {
            if (!sameDepthPlane(answer.Hit,candidate.Hit)) return false;
            float plane=abs(dot(answer.Hit.PositionDistance.xyz-candidate.Hit.PositionDistance.xyz,answer.Hit.NormalResidual.xyz));
            answer.Hit.Uncertainty.z=max(answer.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z+plane); answer.Cache.x=max(answer.Cache.x,candidate.Cache.x);
        }
        light+=weight*candidate.ColorCoverage;
    }
    answer.ColorCoverage=light; return initialized && finite4(light) && light.w>0.0;
}
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Frame.Work.w; if (index>=Frame.Work.x) return;
    if (Frame.Query.y>0.0 && Hits.Values[index].Metadata.z==Screen) return;
    Geometry geometry; geometry.Medium=vec4(0,0,0,1); geometry.PositionDistance=vec4(0.0); geometry.NormalResidual=vec4(0.0); geometry.Uncertainty=vec4(0.0); geometry.Metadata=uvec4(Invalid,Invalid,Unknown,0u);
    Radiance result; result.Hit.Medium=vec4(0,0,0,1); result.Hit=geometry; result.ColorCoverage=vec4(0.0); result.Cache=uvec4(Invalid,0u,0u,0u);
    Hits.Values[index]=geometry; Output.Values[index]=result;
    Origin origin=Input.Seeds[index/Frame.Work.y]; if (origin.Metadata.y!=0u) return;
    vec3 direction=probeRay(index,origin.Metadata.x); uint count=min(uint(Frame.Query.w),MaximumCascades);
    if (Frame.Transition.z>1.0)
    {
        for (uint level=0u;level<count;++level)
        {
            Radiance candidate; candidate.Hit.Medium=vec4(0,0,0,1); float border; if (!sampleDepthCascade(level,origin,direction,candidate,border)) continue;
            uint selected=level;
            if (Frame.Transition.x>0.0 && border<Frame.Transition.x && level+1u<count)
            {
                Radiance coarse; coarse.Hit.Medium=vec4(0,0,0,1); float ignored;
                if (sampleDepthCascade(level+1u,origin,direction,coarse,ignored) && sameDepthPlane(candidate.Hit,coarse.Hit))
                {
                    float fineWeight=clamp(border/Frame.Transition.x,0.0,1.0); candidate.ColorCoverage=mix(coarse.ColorCoverage,candidate.ColorCoverage,fineWeight);
                    if (fineWeight==0.0) { candidate.Hit=coarse.Hit; candidate.Cache=coarse.Cache; selected=level+1u; }
                    else
                    {
                        float plane=abs(dot(candidate.Hit.PositionDistance.xyz-coarse.Hit.PositionDistance.xyz,candidate.Hit.NormalResidual.xyz));
                        candidate.Hit.Uncertainty.z=max(candidate.Hit.Uncertainty.z,coarse.Hit.Uncertainty.z+plane); candidate.Cache.x=max(candidate.Cache.x,coarse.Cache.x);
                    }
                }
            }
            if (candidate.Hit.Metadata.z==Surface) candidate.Hit.Uncertainty=vec4(candidate.Hit.Uncertainty.z,candidate.Hit.Uncertainty.z,0.0,0.0);
            else candidate.Hit.PositionDistance=vec4(origin.PositionMinimum.xyz+direction*Frame.Query.x,Frame.Query.x);
            candidate.Hit.Metadata.z=WorldHit; candidate.Hit.Metadata.w=selected; candidate.ColorCoverage.w=clamp(candidate.ColorCoverage.w,0.0,1.0);
            Hits.Values[index]=candidate.Hit; Output.Values[index]=candidate; return;
        }
        return;
    }
    for (uint level=0u;level<count;++level)
    {
        vec4 sum; float border; if (!sampleCascade(level,origin,direction,sum,border)) continue;
        float localDistance=Frame.Cascades[level].Distances.y; uint selected=level;
        if (Frame.Transition.x>0.0 && border<Frame.Transition.x && level+1u<count)
        {
            vec4 coarse; float unusedBorder;
            if (sampleCascade(level+1u,origin,direction,coarse,unusedBorder))
            {
                float fineWeight=clamp(border/Frame.Transition.x,0.0,1.0); sum=mix(coarse,sum,fineWeight);
                localDistance=Frame.Cascades[level+1u].Distances.y; if (fineWeight==0.0) selected=level+1u;
            }
        }
        result.Hit.PositionDistance=vec4(origin.PositionMinimum.xyz+direction*localDistance,localDistance); result.Hit.Metadata.z=WorldHit; result.Hit.Metadata.w=selected;
        result.ColorCoverage=vec4(sum.xyz,clamp(sum.w,0.0,1.0)); result.Cache=uvec4(Invalid,0u,Complete,0u);
        Hits.Values[index]=result.Hit; Output.Values[index]=result; return;
    }
}
