#include <metal_stdlib>
using namespace metal;
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
#include <metal_raytracing>
using namespace metal::raytracing;
// Explicitly forward the two native resources through shared transport helpers.
#define LUMEN_RAY_ARGUMENTS ,instance_acceleration_structure acceleration,device const uint* hardware
#define LUMEN_RAY_FORWARD ,acceleration,hardware
#define LUMEN_RAY_BINDINGS ,instance_acceleration_structure acceleration [[buffer(8)]],device const uint* hardware [[buffer(9)]]
#else
#define LUMEN_RAY_ARGUMENTS
#define LUMEN_RAY_FORWARD
#define LUMEN_RAY_BINDINGS
#endif
struct LumenProbeOrigin { float4 PositionMinimum; uint4 Metadata; };
constant uint LumenScreenSurface=3,LumenWorldCache=4,LumenHairPointerShift=2;
float3 lumenEnvironmentRadiance(float3 baseSky,float4 axisPower,float3 peak,float3 direction)
{
    if (!any(peak>0)) return baseSky;
    float cosine=saturate(dot(axisPower.xyz,direction));
    return baseSky+peak*pow(cosine,axisPower.w);
}
constant uint LumenProbeNeighbors=4,LumenLinkWordBits=32;
uint probeLinkWords(uint slots) { return (LumenProbeNeighbors*slots+LumenLinkWordBits-1)/LumenLinkWordBits; }
struct LumenScreenFrame { float4 Inverse[4]; uint4 Viewport,Grid; float4 Parameters,Adaptive; };
// Uniform anchors retain their tiles. Quadtree descendants add probes only for
// planes absent from all ancestors; unused reserved slots remain invalid.
constant uint LumenPlacementMissing=0xffffffffu,LumenPlacementBackground=1,LumenPlacementInvalid=2;
constant uint LumenPlacementLevels=2,LumenPlacementChildren=4;
constant float LumenPlacementClearDepth=1,LumenPlacementMinimumNormal=1e-12;
float lumenPlacementMagnitude(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
bool lumenPlacementSurface(constant LumenScreenFrame& frame,texture2d<float,access::read> depths,
    texture2d<float,access::read> normals,uint2 pixel,thread float3& point,thread float3& normal)
{
    point=normal=0; float2 uv=(float2(pixel)+0.5)/float2(frame.Viewport.xy);
    float4 clip(uv.x*2-1,1-uv.y*2,depths.read(pixel).x,1);
    float4 h(dot(frame.Inverse[0],clip),dot(frame.Inverse[1],clip),dot(frame.Inverse[2],clip),dot(frame.Inverse[3],clip));
    if (!all(isfinite(h)) || abs(h.w)<=frame.Parameters.w) return false;
    point=h.xyz/h.w; normal=normalize(normals.read(pixel).xyz);
    return all(isfinite(point)) && all(abs(point)+frame.Parameters.x<=frame.Parameters.z);
}
uint2 lumenPlacementPixel(constant LumenScreenFrame& frame,texture2d<float,access::read> depths,
    texture2d<float,access::read> normals,uint2 lower,uint2 upper,uint count,
    thread const float3* parents,thread const float3* parentNormals,thread bool& found)
{
    uint best=LumenPlacementMissing; uint2 selected=0;
    for (uint y=lower.y;y<upper.y;++y) for (uint x=lower.x;x<upper.x;++x)
    {
        uint2 pixel(x,y); float depth=depths.read(pixel).x; float4 material=normals.read(pixel);
        if (!isfinite(depth) || depth<0 || depth>=LumenPlacementClearDepth || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=LumenPlacementMinimumNormal) continue;
        if (count!=0)
        {
            float3 point,normal; if (!lumenPlacementSurface(frame,depths,normals,pixel,point,normal)) continue;
            bool covered=false;
            for (uint ancestor=0;ancestor<count;++ancestor)
            {
                float3 delta=point-parents[ancestor]; float magnitude=lumenPlacementMagnitude(delta),distance=magnitude>0?magnitude*length(delta/magnitude):0;
                float error=max(abs(dot(delta,normal)),abs(dot(delta,parentNormals[ancestor])));
                float tolerance=frame.Adaptive.y+frame.Adaptive.z*distance+frame.Adaptive.w*max(lumenPlacementMagnitude(point),lumenPlacementMagnitude(parents[ancestor]));
                covered=covered || (dot(normal,parentNormals[ancestor])>frame.Adaptive.x && error<=tolerance);
            }
            if (covered) continue;
        }
        int2 delta=int2(pixel*2+1)-int2(lower+upper); uint metric=uint(delta.x*delta.x+delta.y*delta.y);
        if (metric<best) { selected=pixel; best=metric; }
    }
    found=best!=LumenPlacementMissing; return selected;
}
kernel void lumenPlaceScreenProbes(constant LumenScreenFrame& frame [[buffer(0)]],
    device LumenProbeOrigin* origins [[buffer(1)]],texture2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::read> normals [[texture(1)]],uint index [[thread_position_in_grid]])
{
    if (index>=frame.Grid.y) return;
    uint count=frame.Grid.w+1,tile=index/count,child=index%count;
    uint2 lower=uint2(tile%frame.Viewport.w,tile/frame.Viewport.w)*frame.Viewport.z;
    uint2 upper=min(lower+frame.Viewport.z,frame.Viewport.xy); bool found;
    float3 parents[LumenPlacementLevels],parentNormals[LumenPlacementLevels];
    for (uint clear=0;clear<LumenPlacementLevels;++clear) { parents[clear]=0; parentNormals[clear]=0; }
    uint path[LumenPlacementLevels],levels=0;
    for (uint node=child;node!=0;node=(node-1)/LumenPlacementChildren) path[levels++]=(node-1)%LumenPlacementChildren;
    uint2 selected=lumenPlacementPixel(frame,depth,normals,lower,upper,0,parents,parentNormals,found);
    LumenProbeOrigin result; result.PositionMinimum=0; result.Metadata=uint4(index,LumenPlacementBackground,0,0);
    float3 point=0,normal=0;
    for (uint level=0;level<levels;++level)
    {
        if (!found || !lumenPlacementSurface(frame,depth,normals,selected,point,normal)) { origins[index]=result; return; }
        parents[level]=point; parentNormals[level]=normal;
        uint quadrant=path[levels-1-level]; uint2 middle=(lower+upper)/2;
        lower=uint2((quadrant&1)?middle.x:lower.x,(quadrant&2)?middle.y:lower.y);
        upper=uint2((quadrant&1)?upper.x:middle.x,(quadrant&2)?upper.y:middle.y);
        selected=lumenPlacementPixel(frame,depth,normals,lower,upper,level+1,parents,parentNormals,found);
    }
    if (found)
    {
        result.Metadata=uint4(index,LumenPlacementInvalid,selected);
        if (lumenPlacementSurface(frame,depth,normals,selected,point,normal))
        { result.PositionMinimum=float4(point+normal*frame.Parameters.x,frame.Parameters.y); result.Metadata.y=0; }
    }
    origins[index]=result;
}
uint lumenReverseProbeBits(uint source)
{
    constexpr uint IntegerBits=32; uint reversed=0;
    for (uint bit=0;bit<IntegerBits;++bit) { reversed=(reversed<<1)|(source&1u); source>>=1; }
    return reversed;
}
float3 lumenProbeDirection(uint sample,uint count,uint identifier,uint sequence)
{
    constexpr float Pi=3.14159265358979323846,UintFraction=1.0/4294967296.0;
    uint phase=lumenReverseProbeBits(sample)+lumenReverseProbeBits(identifier^sequence);
    float z=1-2*(float(sample)+0.5)/float(count),angle=2*Pi*(float(phase)*UintFraction);
    float radius=sqrt(max(0.0f,1-z*z)); return float3(radius*cos(angle),radius*sin(angle),z);
}

struct LumenLookupFrame { uint4 Counts,Atlas; float4 Parameters; };
struct LumenLookupCard { float4 Projection[4],Inverse[4]; float4 Origin,BackwardScale; uint4 Tile; };
struct LumenLookupQuery { float4 PositionRadius,NormalCosine; uint4 Range; };
struct LumenLookupValue { float4 AlbedoMetallic,NormalCoverage,EmissionRoughness; uint4 Metadata; };
constant uint LumenLookupDirections=6,LumenLookupBorder=1,LumenLookupMissing=0xffffffffu;
LumenLookupValue lumenMissingMaterial()
{ LumenLookupValue value; value.AlbedoMetallic=0; value.NormalCoverage=0; value.EmissionRoughness=0; value.Metadata=uint4(LumenLookupMissing,0,0,0); return value; }
float lumenCoordinateSize(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
// Material depths are geometric samples, so bilinear weights are accepted per
// tap in the query's tangent plane before any material values are interpolated.
bool lumenLookupCard(LumenLookupCard card,LumenLookupQuery query,constant LumenLookupFrame& frame,
    texture2d<float,access::read> albedo,texture2d<float,access::read> normalImage,texture2d<float,access::read> emission,
    thread LumenLookupValue& value,thread float& depthError,thread float& facing)
{
    value=lumenMissingMaterial(); depthError=0; facing=dot(query.NormalCosine.xyz,card.BackwardScale.xyz);
    if (facing<frame.Parameters.x) return false;
    float3 relative=query.PositionRadius.xyz-card.Origin.xyz; float4 local=float4(relative,1);
    float3 projected=float3(dot(card.Projection[0],local),dot(card.Projection[1],local),dot(card.Projection[2],local));
    if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 location=float2(card.Tile.xy)+LumenLookupBorder+float2(projected.x+1,1-projected.y)*0.5f*frame.Counts.y-0.5f;
    int2 lower=int2(floor(location)); float2 fraction=fract(location); float coverage=0; float3 normal=0;
    for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,(tap>>1)&1),pixel=uint2(lower+int2(corner));
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y); float4 normalDepth=normalImage.read(pixel);
        if (weight<=0 || dot(normalDepth.xyz,normalDepth.xyz)==0 || dot(normalDepth.xyz,query.NormalCosine.xyz)<query.NormalCosine.w) continue;
        float2 uv=(float2(pixel)-float2(card.Tile.xy)-LumenLookupBorder+0.5f)/frame.Counts.y;
        float4 clip=float4(uv.x*2-1,1-uv.y*2,normalDepth.w,1);
        float3 position=float3(dot(card.Inverse[0],clip),dot(card.Inverse[1],clip),dot(card.Inverse[2],clip));
        float error=abs(dot(relative-position,query.NormalCosine.xyz));
        float roundoff=frame.Parameters.y*max(card.BackwardScale.w,max(lumenCoordinateSize(relative),lumenCoordinateSize(position)))+
            frame.Parameters.z*dot(abs(query.PositionRadius.xyz),abs(query.NormalCosine.xyz));
        if (!isfinite(error) || error>query.PositionRadius.w+roundoff) continue;
        value.AlbedoMetallic+=albedo.read(pixel)*weight; value.EmissionRoughness+=emission.read(pixel)*weight;
        normal+=normalDepth.xyz*weight; coverage+=weight; depthError+=error*weight;
    }
    if (coverage<=0) return false;
    value.AlbedoMetallic/=coverage; value.EmissionRoughness/=coverage; value.NormalCoverage=float4(normal/coverage,saturate(coverage));
    value.Metadata=uint4(card.Tile.w,1,1,0); depthError/=coverage; return true;
}
kernel void lumenLookupSurface(constant LumenLookupFrame& frame [[buffer(0)]],device const LumenLookupCard* cards [[buffer(1)]],
    device const LumenLookupQuery* queries [[buffer(2)]],device LumenLookupValue* results [[buffer(3)]],
    texture2d<float,access::read> albedo [[texture(0)]],texture2d<float,access::read> normalImage [[texture(1)]],
    texture2d<float,access::read> emission [[texture(2)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Atlas.z; if (index>=frame.Counts.x) return;
    LumenLookupQuery query=queries[index]; LumenLookupValue layers[LumenLookupDirections]; float errors[LumenLookupDirections],facing[LumenLookupDirections];
    for (uint group=0;group<LumenLookupDirections;++group) { layers[group]=lumenMissingMaterial(); errors[group]=0; facing[group]=0; }
    for (uint candidate=0;candidate<query.Range.y;++candidate)
    {
        LumenLookupCard card=cards[query.Range.x+candidate]; LumenLookupValue value; float error,cosine;
        if (!lumenLookupCard(card,query,frame,albedo,normalImage,emission,value,error,cosine)) continue;
        uint group=card.Tile.z;
        if (layers[group].Metadata.z==0 || error<errors[group] || (error==errors[group] && value.NormalCoverage.w>layers[group].NormalCoverage.w))
        { layers[group]=value; errors[group]=error; facing[group]=cosine; }
    }
    LumenLookupValue result=lumenMissingMaterial(); float total=0,orientationSum=0;
    for (uint group=0;group<LumenLookupDirections;++group) if (layers[group].Metadata.z!=0)
    { float orientation=facing[group]*facing[group]; total+=orientation*layers[group].NormalCoverage.w; orientationSum+=orientation; }
    if (total>0)
    {
        float3 normal=0; float dominant=0;
        for (uint group=0;group<LumenLookupDirections;++group) if (layers[group].Metadata.z!=0)
        {
            float weight=facing[group]*facing[group]*layers[group].NormalCoverage.w/total;
            result.AlbedoMetallic+=layers[group].AlbedoMetallic*weight; result.EmissionRoughness+=layers[group].EmissionRoughness*weight;
            normal+=layers[group].NormalCoverage.xyz*weight; ++result.Metadata.y;
            if (weight>dominant || (weight==dominant && layers[group].Metadata.x<result.Metadata.x)) { dominant=weight; result.Metadata.x=layers[group].Metadata.x; }
        }
        float squared=dot(normal,normal); result.NormalCoverage=float4(squared>0?normal*rsqrt(squared):query.NormalCosine.xyz,saturate(total/orientationSum)); result.Metadata.z=1;
    }
    result.AlbedoMetallic=saturate(result.AlbedoMetallic); result.EmissionRoughness.w=saturate(result.EmissionRoughness.w);
    results[index]=result;
}

// Sparse mesh SDF tracing is authored directly in MSL. A stackless BVH narrows
// the candidates; exhaustion stays explicit so GI can choose another fallback.
struct LumenDistanceFrame { uint4 Counts,Offsets,Traversal; float4 Parameters,HairSky; };
struct LumenDistanceRay { float4 OriginMinimum,DirectionMaximum; };
struct LumenDistanceHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
// Heitz (JCGT 2018): a visible-normal GGX sample maps the stretched-view
// disk to a reflected direction. Textures and generated rays use top-left rows.
struct LumenReflectionRayFrame { float4 Inverse[4]; uint4 Viewport,Work; float4 Trace,Numeric; };
constant uint LumenSpecularPeriod=64;
constant float LumenSpecularPi=3.14159265358979323846,LumenSpecularUintUnit=1.0/4294967296.0;
constant float LumenSpecularGolden=0.6180339887498948482,LumenSpecularAxisLimit=0.9;
bool lumenSpecularPosition(constant LumenReflectionRayFrame& frame,uint2 pixel,float depth,thread float3& result)
{
    float2 uv=(float2(pixel)+0.5f)/float2(frame.Viewport.xy); float4 clip=float4(uv.x*2-1,1-uv.y*2,depth,1);
    float4 h=float4(dot(frame.Inverse[0],clip),dot(frame.Inverse[1],clip),dot(frame.Inverse[2],clip),dot(frame.Inverse[3],clip));
    result=0; if (!all(isfinite(h)) || abs(h.w)<=frame.Numeric.y) return false;
    result=h.xyz/h.w; return all(isfinite(result)) && all(abs(result)<=frame.Numeric.x);
}
float3 lumenSpecularMicroNormal(float3 view,float alpha,float2 xi)
{
    float3 stretched=normalize(float3(alpha*view.xy,view.z)); float radiusSquared=dot(stretched.xy,stretched.xy);
    float3 right=radiusSquared>0?float3(-stretched.y,stretched.x,0)/sqrt(radiusSquared):float3(1,0,0),up=cross(stretched,right);
    float angle=2*LumenSpecularPi*xi.y; float2 disk=sqrt(xi.x)*float2(cos(angle),sin(angle));
    disk.y=mix(sqrt(max(0.0f,1-disk.x*disk.x)),disk.y,(1+stretched.z)*0.5f);
    float3 h=right*disk.x+up*disk.y+stretched*sqrt(max(0.0f,1-dot(disk,disk)));
    return normalize(float3(alpha*h.xy,max(0.0f,h.z)));
}
kernel void lumenReflectionRays(constant LumenReflectionRayFrame& frame [[buffer(0)]],device LumenDistanceRay* output [[buffer(1)]],
    texture2d<float,access::read> depths [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Work.w; if (index>=frame.Work.y) return;
    LumenDistanceRay ray; ray.OriginMinimum=0; ray.DirectionMaximum=0; output[index]=ray;
    uint2 pixel=min(uint2(index%frame.Viewport.w,index/frame.Viewport.w)*frame.Viewport.z+frame.Viewport.z/2,frame.Viewport.xy-1);
    float depth=depths.read(pixel).x; float4 material=normals.read(pixel);
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || material.w<0 || material.w>=frame.Trace.z || dot(material.xyz,material.xyz)<=frame.Numeric.z) return;
    float3 world,nearWorld,midWorld;
    if (!lumenSpecularPosition(frame,pixel,depth,world) || !lumenSpecularPosition(frame,pixel,0,nearWorld) || !lumenSpecularPosition(frame,pixel,0.5,midWorld)) return;
    float3 normal=normalize(material.xyz),view=normalize(nearWorld-midWorld); if (!all(isfinite(view)) || dot(normal,view)<=0) return;
    float3 axis=abs(normal.z)<LumenSpecularAxisLimit?float3(0,0,1):float3(0,1,0),tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
    float phase=float(lumenReverseProbeBits(index))*LumenSpecularUintUnit;
    float2 xi=fract(float2((float(frame.Work.z%LumenSpecularPeriod)+0.5f)/LumenSpecularPeriod,
        float(lumenReverseProbeBits(frame.Work.z))*LumenSpecularUintUnit)+float2(phase,fract(phase*LumenSpecularGolden)));
    float3 micro=material.w==0?float3(0,0,1):lumenSpecularMicroNormal(float3(dot(view,tangent),dot(view,bitangent),dot(view,normal)),max(material.w*material.w,frame.Trace.w),xi);
    float3 halfVector=tangent*micro.x+bitangent*micro.y+normal*micro.z,direction=reflect(-view,halfVector);
    ray.OriginMinimum=float4(world+normal*frame.Trace.y,0);
    ray.DirectionMaximum=float4(direction,dot(normal,direction)>0?frame.Trace.x:0); output[index]=ray;
}
constant uint SdfFieldStride=20,SdfInstanceStride=20,SdfNodeStride=12,SdfRowStride=4;
constant uint SdfBrickGrid=4,SdfStorage=8,SdfGridMetric=12,SdfErrorMetric=16,SdfInverse=4,SdfTag=16,SdfNodeMax=4,SdfNodeLink=8;
constant uint SdfInterior=6,SdfBorder=1,SdfSide=8,SdfBrickSamples=512,SdfFirstBrick=2;
constant uint SdfInvalid=0xffffffffu,SdfMiss=0,SdfSurface=1,SdfUnresolved=2;
constant float SdfQuantizedMaximum=32767.0f;
uint4 sdfWords(device const uint* scene,uint at) { return uint4(scene[at],scene[at+1],scene[at+2],scene[at+3]); }
float4 sdfFloats(device const uint* scene,uint at) { return as_type<float4>(sdfWords(scene,at)); }
float3 sdfMap(device const uint* scene,uint object,float3 v)
{ return float3(dot(sdfFloats(scene,object+SdfInverse).xyz,v),dot(sdfFloats(scene,object+SdfInverse+SdfRowStride).xyz,v),dot(sdfFloats(scene,object+SdfInverse+SdfRowStride*2).xyz,v)); }
float sdfLength(float3 v)
{ float scale=max(abs(v.x),max(abs(v.y),abs(v.z))); return scale>0?length(v/scale)*scale:0; }
float3 sdfUnit(float3 v,float3 fallback)
{
    if (!all(isfinite(v))) return fallback;
    float scale=max(abs(v.x),max(abs(v.y),abs(v.z))); if (scale==0) return fallback;
    v/=scale; return v*rsqrt(dot(v,v));
}
bool sdfSlabs(float3 o,float3 d,float3 lo,float3 hi,thread float& begin,thread float& end)
{
    for (uint axis=0;axis<3;++axis)
    {
        if (d[axis]==0) { if (o[axis]<lo[axis] || o[axis]>hi[axis]) return false; }
        else
        {
            float a=(lo[axis]-o[axis])/d[axis],b=(hi[axis]-o[axis])/d[axis];
            begin=max(begin,min(a,b)); end=min(end,max(a,b)); if (begin>end) return false;
        }
    }
    return true;
}
float sdfSample(device const uint* scene,uint field,float3 position)
{
    uint3 dimensions=sdfWords(scene,field).xyz,brickDimensions=sdfWords(scene,field+SdfBrickGrid).xyz;
    uint4 addresses=sdfWords(scene,field+SdfStorage); float4 metric=sdfFloats(scene,field+SdfGridMetric);
    float band=sdfFloats(scene,field+SdfErrorMetric).x;
    float3 grid=clamp((position-metric.xyz)/metric.w-0.5f,float3(0),float3(dimensions)-1);
    uint3 lower=uint3(floor(grid)),brick=lower/SdfInterior,local=lower%SdfInterior+SdfBorder; float3 blend=grid-float3(lower);
    uint page=scene[addresses.x+(brick.z*brickDimensions.y+brick.y)*brickDimensions.x+brick.x];
    if (page<SdfFirstBrick) return page==1?-band:band;
    float value=0;
    for (uint z=0;z<2;++z) for (uint y=0;y<2;++y) for (uint x=0;x<2;++x)
    {
        uint element=(page-SdfFirstBrick)*SdfBrickSamples+((local.z+z)*SdfSide+local.y+y)*SdfSide+local.x+x;
        uint pair=scene[addresses.y+element/2]; int encoded=(element&1)?(as_type<int>(pair)>>16):(as_type<int>(pair<<16)>>16);
        float weight=(x?blend.x:1-blend.x)*(y?blend.y:1-blend.y)*(z?blend.z:1-blend.z);
        value+=float(encoded)*weight;
    }
    return value*(band/SdfQuantizedMaximum);
}
float3 sdfNormal(device const uint* scene,uint object,uint field,float3 p,float cell,float3 fallback)
{
    float3 g=float3(sdfSample(scene,field,p+float3(cell,0,0))-sdfSample(scene,field,p-float3(cell,0,0)),
        sdfSample(scene,field,p+float3(0,cell,0))-sdfSample(scene,field,p-float3(0,cell,0)),sdfSample(scene,field,p+float3(0,0,cell))-sdfSample(scene,field,p-float3(0,0,cell)));
    g=sdfUnit(g,float3(0));
    float3 world=sdfFloats(scene,object+SdfInverse).xyz*g.x+sdfFloats(scene,object+SdfInverse+SdfRowStride).xyz*g.y+sdfFloats(scene,object+SdfInverse+SdfRowStride*2).xyz*g.z;
    return sdfUnit(world,fallback);
}
uint sdfTraceObject(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray,uint index,float anchor,float limit,
    thread float& at,thread float4& normal,thread float2& error,thread uint& steps)
{
    at=anchor; normal=0; error=0; steps=0;
    uint object=frame.Offsets.y+index*SdfInstanceStride,field=frame.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
    float3 origin=sdfMap(scene,object,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*anchor-sdfFloats(scene,object).xyz),direction=sdfMap(scene,object,ray.DirectionMaximum.xyz);
    float speed=sdfLength(direction); if (!all(isfinite(origin)) || !all(isfinite(direction)) || !isfinite(speed) || speed<=0) return SdfUnresolved;
    float4 metric=sdfFloats(scene,field+SdfGridMetric),bounds=sdfFloats(scene,field+SdfErrorMetric); float entry=0,exit=max(0.0f,limit-anchor);
    if (!sdfSlabs(origin,direction,metric.xyz,metric.xyz+float3(sdfWords(scene,field).xyz)*metric.w,entry,exit)) return SdfMiss;
    float travel=entry,tolerance=bounds.y+frame.Parameters.x*metric.w;
    for (uint iteration=0;iteration<frame.Traversal.x;++iteration)
    {
        at=min(limit,anchor+travel); float3 p=origin+direction*travel; ++steps;
        if (!all(isfinite(p))) return SdfUnresolved;
        float distance=sdfSample(scene,field,p);
        if (abs(distance)<=tolerance)
        {
            normal=float4(sdfNormal(scene,object,field,p,metric.w,-ray.DirectionMaximum.xyz),distance/speed);
            error=float2(bounds.y,tolerance)/speed; return SdfSurface;
        }
        float next=travel+(abs(distance)-bounds.y)/speed*frame.Parameters.y;
        if (!isfinite(next) || next<=travel) return SdfUnresolved;
        if (next>exit) return SdfMiss; travel=next;
    }
    at=min(limit,anchor+travel); return SdfUnresolved;
}
// The source-plane offset metric is 1/|A^-T n| for world-to-field A.
// Normalize matrix rows together before cofactors to preserve scale range.
float sdfNormalSpeed(device const uint* scene,uint object,float3 normal)
{
    float3 x=sdfFloats(scene,object+SdfInverse).xyz,y=sdfFloats(scene,object+SdfInverse+SdfRowStride).xyz,z=sdfFloats(scene,object+SdfInverse+SdfRowStride*2).xyz;
    float3 extent=max(abs(x),max(abs(y),abs(z))); float scale=max(extent.x,max(extent.y,extent.z)); if (scale==0) return 0;
    x/=scale; y/=scale; z/=scale; float3 cx=cross(y,z),cy=cross(z,x),cz=cross(x,y);
    float inverseLength=sdfLength(float3(dot(cx,normal),dot(cy,normal),dot(cz,normal)));
    float speed=inverseLength>0?scale*(abs(dot(x,cx))/inverseLength):0; return isfinite(speed)?speed:0;
}
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
// Metal intersection queries evaluate cutout coverage inline. A rejected
// candidate is ignored without advancing the ray past other coincident shells.
constant uint LumenHardwareInstanceStride=16,LumenHardwareFaceStride=24,LumenHardwareMaterialStride=12;
// Field offsets are uint words, matching the immutable host records.
constant uint HardwareB=4,HardwareC=8,HardwareUVAB=12,HardwareUVC=16,HardwareTags=20;
constant uint HardwareOpacity=4,HardwareCoverage=8,HardwareNormal0=4,HardwareNormal1=8,HardwareNormal2=12;
float lumenHardwareMap(device const uint* words,uint start,uint4 image,float2 uv)
{
    if (image.w==0) return 1;
    float2 wrapped=fract(uv),pixel=float2(wrapped.x,1-wrapped.y)*float2(image.yz)-0.5f;
    int2 first=int2(floor(pixel)),size=int2(image.yz);float2 f=fract(pixel);float value=0;
    for (uint tap=0;tap<4;++tap)
    {
        int2 bit=int2(tap&1,tap>>1),p=((first+bit)%size+size)%size;
        float2 weight=select(1-f,f,bit!=int2(0));
        value+=as_type<float>(words[start+image.x+uint(p.y)*image.y+uint(p.x)])*weight.x*weight.y;
    }
    return value;
}
bool lumenHardwareCoverage(device const uint* words,uint instance,uint primitive,float2 coordinates,uint flags)
{
    uint4 range=sdfWords(words,instance*LumenHardwareInstanceStride);uint face=range.x+primitive*LumenHardwareFaceStride;
    uint4 metadata=sdfWords(words,face+HardwareTags);if ((metadata.w&flags)!=flags) return false;
    uint material=range.y+metadata.x*LumenHardwareMaterialStride;float4 coverage=sdfFloats(words,material+HardwareCoverage);
    if (coverage.y<=0) return true;
    float3 bary=float3(1-coordinates.x-coordinates.y,coordinates);
    float4 a=sdfFloats(words,face),b=sdfFloats(words,face+HardwareB),c=sdfFloats(words,face+HardwareC),uvab=sdfFloats(words,face+HardwareUVAB);
    float2 uv=uvab.xy*bary.x+uvab.zw*bary.y+sdfFloats(words,face+HardwareUVC).xy*bary.z;
    float alpha=saturate(dot(float3(a.w,b.w,c.w),bary)*coverage.x*lumenHardwareMap(words,range.z,sdfWords(words,material),uv)*lumenHardwareMap(words,range.z,sdfWords(words,material+HardwareOpacity),uv));
    return isfinite(alpha) && alpha>=coverage.y;
}
LumenDistanceHit lumenTraceHardware(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay input,uint flags,
    instance_acceleration_structure acceleration,device const uint* hardware)
{
    LumenDistanceHit hit;hit.Medium=float4(0,0,0,1);hit.NormalResidual=0;hit.Uncertainty=0;hit.Metadata=uint4(SdfInvalid,SdfInvalid,SdfMiss,0);
    hit.PositionDistance=float4(input.OriginMinimum.xyz+input.DirectionMaximum.xyz*input.DirectionMaximum.w,input.DirectionMaximum.w);
    if (frame.Counts.z==0 || input.DirectionMaximum.w<=input.OriginMinimum.w) return hit;
    ray segment(input.OriginMinimum.xyz,input.DirectionMaximum.xyz,input.OriginMinimum.w,input.DirectionMaximum.w);
    intersection_params parameters;parameters.assume_geometry_type(geometry_type::triangle);parameters.force_opacity(forced_opacity::non_opaque);
    constexpr uint GeometryMask=1,ShadowMask=2;
    intersection_query<instancing,triangle_data> query;
    query.reset(segment,acceleration,flags!=0?ShadowMask:GeometryMask,parameters);
    uint selected=SdfInvalid,primitive=SdfInvalid;float nearest=input.DirectionMaximum.w;
    while (query.next())
    {
        if (query.get_candidate_intersection_type()!=intersection_type::triangle) continue;
        uint candidate=query.get_candidate_instance_id(),face=query.get_candidate_primitive_id();float distance=query.get_candidate_triangle_distance();
        if (!lumenHardwareCoverage(hardware,candidate,face,query.get_candidate_triangle_barycentric_coord(),flags)) continue;
        if (selected==SdfInvalid || distance<nearest || (distance==nearest && (candidate<selected || (candidate==selected && face<primitive))))
        {selected=candidate;primitive=face;nearest=distance;query.commit_triangle_intersection();}
    }
    if (selected==SdfInvalid) return hit;
    uint object=selected*LumenHardwareInstanceStride,face=sdfWords(hardware,object).x+primitive*LumenHardwareFaceStride;
    float3 a=sdfFloats(hardware,face).xyz,b=sdfFloats(hardware,face+HardwareB).xyz,c=sdfFloats(hardware,face+HardwareC).xyz,e0=b-a,e1=c-a;
    if (!all(isfinite(e0)) || !all(isfinite(e1)))
    {float scale=max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(max(abs(b.x),max(abs(b.y),abs(b.z))),max(abs(c.x),max(abs(c.y),abs(c.z)))));e0=b/scale-a/scale;e1=c/scale-a/scale;}
    float3 local=cross(sdfUnit(e0,float3(0)),sdfUnit(e1,float3(0)));
    float3 normal=sdfUnit(float3(dot(sdfFloats(hardware,object+HardwareNormal0).xyz,local),dot(sdfFloats(hardware,object+HardwareNormal1).xyz,local),dot(sdfFloats(hardware,object+HardwareNormal2).xyz,local)),float3(0));
    hit.PositionDistance=float4(input.OriginMinimum.xyz+input.DirectionMaximum.xyz*nearest,nearest);hit.NormalResidual=float4(normal,0);
    hit.Metadata=uint4(selected,sdfWords(scene,frame.Offsets.y+selected*SdfInstanceStride+SdfTag).y,dot(normal,normal)>0?SdfSurface:SdfUnresolved,0);return hit;
}
#endif
LumenDistanceHit lumenTraceLocalOpaque(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray,uint requiredFlags LUMEN_RAY_ARGUMENTS)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    return lumenTraceHardware(scene,frame,ray,requiredFlags,acceleration,hardware);
#else
    float nearest=ray.DirectionMaximum.w,unknown=nearest;
    uint selected=SdfInvalid,pending=SdfInvalid,node=0,totalSteps=0; float4 selectedNormal=0; float2 selectedError=0;
    while (node<frame.Counts.y)
    {
        uint address=frame.Offsets.z+node*SdfNodeStride; uint4 link=sdfWords(scene,address+SdfNodeLink);
        float begin=ray.OriginMinimum.w,end=min(nearest,unknown);
        if (!sdfSlabs(ray.OriginMinimum.xyz,ray.DirectionMaximum.xyz,sdfFloats(scene,address).xyz,sdfFloats(scene,address+SdfNodeMax).xyz,begin,end)) { node=link.w; continue; }
        if (link.z==0) { node=link.x; continue; }
        uint flags=sdfWords(scene,frame.Offsets.y+link.y*SdfInstanceStride+SdfTag).z;
        if ((flags&requiredFlags)!=requiredFlags) { node=link.w; continue; }
        float at; float4 normal; float2 error; uint steps;
        uint status=sdfTraceObject(scene,frame,ray,link.y,begin,end,at,normal,error,steps); totalSteps+=steps;
        if (status==SdfSurface && (at<nearest || (at==nearest && link.y<selected))) { nearest=at; selected=link.y; selectedNormal=normal; selectedError=error; }
        if (status==SdfUnresolved && (at<unknown || (at==unknown && link.y<pending))) { unknown=at; pending=link.y; }
        node=link.w;
    }
    LumenDistanceHit hit; hit.Medium=float4(0,0,0,1); hit.NormalResidual=0; hit.Uncertainty=0; hit.Metadata=uint4(SdfInvalid,SdfInvalid,SdfMiss,totalSteps); float at=ray.DirectionMaximum.w;
    if (pending!=SdfInvalid && (selected==SdfInvalid || unknown<=nearest))
    { at=unknown; hit.Metadata.xyz=uint3(pending,sdfWords(scene,frame.Offsets.y+pending*SdfInstanceStride+SdfTag).y,SdfUnresolved); }
    else if (selected!=SdfInvalid)
    { at=nearest; hit.NormalResidual=selectedNormal; hit.Uncertainty.xy=selectedError; hit.Metadata.xyz=uint3(selected,sdfWords(scene,frame.Offsets.y+selected*SdfInstanceStride+SdfTag).y,SdfSurface); }
    hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*at,at); return hit;
#endif
}

LumenDistanceHit lumenTraceLocalOpaque(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray LUMEN_RAY_ARGUMENTS)
{ return lumenTraceLocalOpaque(scene,frame,ray,frame.Traversal.z LUMEN_RAY_FORWARD); }

float lumenGlobalClearance(device const uint* scene,constant LumenDistanceFrame& frame,float3 world)
{
    constexpr uint HeaderPrefix=8,GridWords=12,GridMetrics=4,GridStorage=8,VoxelWords=4;
    uint4 shape=sdfWords(scene,frame.Offsets.w);
    for (uint level=0;level<shape.x;++level)
    {
        uint header=frame.Offsets.w+HeaderPrefix+level*GridWords; float4 grid=sdfFloats(scene,header);
        float3 p=(world-grid.xyz)/grid.w;
        if (!all(isfinite(p)) || any(p<0) || any(p>float(shape.y-1))) continue;
        uint3 lower=min(uint3(floor(p)),uint3(shape.y-2)); float3 f=p-float3(lower); float radius=0;
        uint start=sdfWords(scene,header+GridStorage).x;
        for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),cell=lower+bit;
            float4 value=sdfFloats(scene,start+((cell.z*shape.y+cell.y)*shape.y+cell.x)*VoxelWords);
            float3 weight=select(1-f,f,bit!=uint3(0));
            radius+=(frame.Traversal.z!=0?value.w:value.y)*weight.x*weight.y*weight.z;
        }
        return isfinite(radius)?max(0.0f,radius-sdfFloats(scene,header+GridMetrics).x):0;
    }
    return 0;
}
LumenDistanceHit lumenTraceOpaque(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray LUMEN_RAY_ARGUMENTS)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    return lumenTraceHardware(scene,frame,ray,frame.Traversal.z,acceleration,hardware);
#else
    LumenDistanceHit hit; hit.Medium=float4(0,0,0,1); hit.PositionDistance=0; hit.NormalResidual=0; hit.Uncertainty=0; hit.Metadata=uint4(SdfInvalid,SdfInvalid,SdfMiss,0);
    bool global=frame.Offsets.w!=0; uint stages=global && frame.Parameters.z>0?2:1;
    float distance=ray.OriginMinimum.w; uint nearSteps=0,globalSteps=0;
    for (uint stage=0;stage<stages;++stage)
    {
        bool detailNear=stages==2 && stage==0;
        if (global && !detailNear) for (uint iteration=0;iteration<uint(frame.Parameters.w);++iteration)
        {
            ++globalSteps; float empty=lumenGlobalClearance(scene,frame,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*distance);
            float next=distance+empty*frame.Parameters.y; if (!isfinite(next) || next<=distance) break;
            if (next>ray.DirectionMaximum.w)
            {
                hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
                hit.NormalResidual=0; hit.Uncertainty=float4(0,0,0,float(globalSteps)); hit.Metadata=uint4(SdfInvalid,SdfInvalid,SdfMiss,nearSteps); return hit;
            }
            distance=next;
        }
        LumenDistanceRay detail=ray; detail.OriginMinimum.w=distance;
        if (detailNear) detail.DirectionMaximum.w=min(ray.DirectionMaximum.w,distance+frame.Parameters.z);
        hit=lumenTraceLocalOpaque(scene,frame,detail LUMEN_RAY_FORWARD); hit.Metadata.w+=nearSteps; hit.Uncertainty.w=float(globalSteps);
        if (!detailNear || hit.Metadata.z!=SdfMiss || detail.DirectionMaximum.w==ray.DirectionMaximum.w) return hit;
        distance=detail.DirectionMaximum.w; nearSteps=hit.Metadata.w;
    }
    return hit;
#endif
}
// Integrate the cell-average projected fiber density, retaining a separate
// caster covariance. A homogeneous cell has an analytic Beer solution with
// isotropic low-frequency sky scattering and emitted radiance as its source.
float4 lumenHairMedium(device const uint* words,constant LumenDistanceFrame& frame,LumenDistanceRay ray,float limit,uint requiredFlags)
{
    constexpr uint HairCellWordStride=24,HairVectorWords=4;
    uint header=frame.Traversal.w>>LumenHairPointerShift; float4 medium=float4(0,0,0,1); if (header==0) return medium;
    uint4 grid=sdfWords(words,header); float4 metric=sdfFloats(words,header+HairVectorWords);
    float3 direction=ray.DirectionMaximum.xyz,origin=ray.OriginMinimum.xyz-metric.xyz;
    float first=ray.OriginMinimum.w,last=min(limit,ray.DirectionMaximum.w);
    if (last<=first || !sdfSlabs(origin,direction,float3(0),float3(grid.xyz)*metric.w,first,last) || last<=first) return medium;
    int3 cell=clamp(int3(floor((origin+direction*first)/metric.w)),int3(0),int3(grid.xyz)-1);
    int3 increment=int3(sign(direction)); bool caster=(requiredFlags&1u)!=0;
    for (uint visit=0;visit<grid.x+grid.y+grid.z+1 && first<last;++visit)
    {
        float3 crossing=last;
        for (uint axis=0;axis<3;++axis) if (increment[axis]!=0)
            crossing[axis]=((float(cell[axis])+(increment[axis]>0?1.0f:0.0f))*metric.w-origin[axis])/direction[axis];
        float next=max(first,min(last,min(crossing.x,min(crossing.y,crossing.z))));
        uint at=grid.w+(uint(cell.x)+grid.x*(uint(cell.y)+grid.y*uint(cell.z)))*HairCellWordStride;
        float4 optical=sdfFloats(words,at),emission=sdfFloats(words,at+HairVectorWords);
        float4 m=sdfFloats(words,at+(caster?4:2)*HairVectorWords),n=sdfFloats(words,at+(caster?5:3)*HairVectorWords);
        float aligned=dot(m.xyz,direction*direction)+2*(m.w*direction.x*direction.y+n.x*direction.x*direction.z+n.y*direction.y*direction.z);
        float sigma=(caster?emission.w:optical.x)*sqrt(saturate(1-aligned));
        float transmission=exp(-sigma*(next-first));
        if (!caster) medium.xyz+=medium.w*(1-transmission)*(optical.yzw*frame.HairSky.xyz+emission.xyz);
        medium.w*=transmission; first=next; if (first>=last) break;
        bool outside=false;
        for (uint axis=0;axis<3;++axis) if (increment[axis]!=0 && crossing[axis]<=next)
        { cell[axis]+=increment[axis]; outside=outside || cell[axis]<0 || cell[axis]>=int(grid[axis]); }
        if (outside) break;
    }
    return medium;
}
LumenDistanceHit lumenTraceLocalScene(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray,uint flags LUMEN_RAY_ARGUMENTS)
{
    auto hit=lumenTraceLocalOpaque(scene,frame,ray,flags LUMEN_RAY_FORWARD);
    hit.Medium=lumenHairMedium(scene,frame,ray,hit.Metadata.z==SdfMiss?ray.DirectionMaximum.w:hit.PositionDistance.w,flags); return hit;
}
LumenDistanceHit lumenTraceLocalScene(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray LUMEN_RAY_ARGUMENTS)
{ return lumenTraceLocalScene(scene,frame,ray,frame.Traversal.z LUMEN_RAY_FORWARD); }
LumenDistanceHit lumenTraceScene(device const uint* scene,constant LumenDistanceFrame& frame,LumenDistanceRay ray LUMEN_RAY_ARGUMENTS)
{
    auto hit=lumenTraceOpaque(scene,frame,ray LUMEN_RAY_FORWARD);
    hit.Medium=lumenHairMedium(scene,frame,ray,hit.Metadata.z==SdfMiss?ray.DirectionMaximum.w:hit.PositionDistance.w,frame.Traversal.z); return hit;
}
kernel void lumenTraceDistance(constant LumenDistanceFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const LumenDistanceRay* rays [[buffer(2)]],device LumenDistanceHit* hits [[buffer(3)]] LUMEN_RAY_BINDINGS,uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Traversal.y; if (index>=frame.Counts.x) return;
    hits[index]=lumenTraceScene(scene,frame,rays[index] LUMEN_RAY_FORWARD);
}

float volumeMagnitude(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float volumeInstanceSupport(device const uint* scene,constant LumenDistanceFrame& frame,uint index,float3 world)
{
    uint object=frame.Offsets.y+index*SdfInstanceStride,field=frame.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
    float3 center=sdfFloats(scene,object).xyz,delta=world-center;
    float3 a=sdfFloats(scene,object+SdfInverse).xyz,b=sdfFloats(scene,object+SdfInverse+SdfRowStride).xyz,c=sdfFloats(scene,object+SdfInverse+SdfRowStride*2).xyz;
    float3 p=float3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(volumeMagnitude(a),max(volumeMagnitude(b),volumeMagnitude(c)));
    if (!all(isfinite(p)) || !isfinite(scale) || scale<=0) return 0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    // The induced 1/infinity norm bound converts local SDF clearance into a
    // surface-free world sphere for general invertible instance transforms.
    float rows=max(dot(a,float3(1)),max(dot(b,float3(1)),dot(c,float3(1)))),norm=scale*sqrt(rows*volumeMagnitude(a+b+c));
    if (!isfinite(norm) || norm<=0) return 0;
    float4 grid=sdfFloats(scene,field+SdfGridMetric),errors=sdfFloats(scene,field+SdfErrorMetric);
    float3 end=grid.xyz+float3(sdfWords(scene,field).xyz)*grid.w,sampled=clamp(p,grid.xyz+grid.w*0.5f,end-grid.w*0.5f);
    float outside=sdfLength(p-clamp(p,grid.xyz,end)),extension=sdfLength(p-sampled);
    // A single point contributes one interpolation error. A ray-step bound's
    // extra possible-hit errors are unnecessary for this support sphere.
    float lower=max(outside,abs(sdfSample(scene,field,sampled))-errors.y-frame.Parameters.x*grid.w-extension);
    lower-=frame.Parameters.y*max(volumeMagnitude(p),max(volumeMagnitude(grid.xyz),volumeMagnitude(end)));
    float support=lower/norm-frame.Parameters.y*max(volumeMagnitude(world),volumeMagnitude(center));
    return isfinite(support)?clamp(support,0.0f,frame.Parameters.z):0;
}
kernel void lumenVolumeSupport(constant LumenDistanceFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const LumenProbeOrigin* origins [[buffer(2)]],device float* radii [[buffer(3)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Traversal.y; if (index>=frame.Counts.x) return;
    LumenProbeOrigin probe=origins[index]; float support=probe.Metadata.y==0?frame.Parameters.z:0;
    for (uint object=0;object<frame.Counts.z && support>0;++object) support=min(support,volumeInstanceSupport(scene,frame,object,probe.PositionMinimum.xyz));
    radii[index]=support;
}
// The temporary Z uncertainty is a normal-plane radius used only by the
// following radiance resolve; the public distance-hit contract keeps Z zero.
kernel void lumenTraceRadiance(constant LumenDistanceFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const LumenDistanceRay* rays [[buffer(2)]],device LumenDistanceHit* hits [[buffer(3)]] LUMEN_RAY_BINDINGS,uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Traversal.y; if (index>=frame.Counts.x) return;
    constexpr uint ReflectionGenerated=1,ReflectionScreenFirst=2;
    if ((frame.Traversal.w&ReflectionScreenFirst)!=0 && hits[index].Metadata.z==LumenScreenSurface) { hits[index].Medium=lumenHairMedium(scene,frame,rays[index],hits[index].PositionDistance.w,frame.Traversal.z); return; }
    if ((frame.Traversal.w&ReflectionGenerated)!=0 && rays[index].DirectionMaximum.w<=0)
    {
        LumenDistanceHit empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0;
        empty.Metadata=uint4(SdfInvalid,SdfInvalid,SdfUnresolved,0); hits[index]=empty; return;
    }
    LumenDistanceHit hit=lumenTraceScene(scene,frame,rays[index] LUMEN_RAY_FORWARD);
    if (hit.Metadata.z==SdfSurface)
    {
        uint object=frame.Offsets.y+hit.Metadata.x*SdfInstanceStride,field=frame.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
        float threshold=sdfFloats(scene,field+SdfErrorMetric).y+frame.Parameters.x*sdfFloats(scene,field+SdfGridMetric).w;
#if !defined(EVANS_LUMEN_HARDWARE_RAYS)
        hit.Uncertainty.z=threshold/sdfNormalSpeed(scene,object,hit.NormalResidual.xyz);
#endif
    }
    hits[index]=hit;
}

// Equal-area square atlas, following Clarberg (JGT 2008). World probes retain
// fixed texel directions rather than the screen estimator's azimuthal sequence.
float3 lumenCacheDirection(uint texel,uint side)
{
    constexpr float QuarterPi=3.14159265358979323846/4;
    side=max(side,1u); float2 square=2*(float2(texel%side,texel/side)+0.5)/float(side)-1;
    float fold=1-abs(square.x)-abs(square.y),radius=1-abs(fold);
    float phi=radius>0?QuarterPi*(1+(abs(square.y)-abs(square.x))/radius):0;
    float horizontal=radius*sqrt(max(0.0f,2-radius*radius));
    float2 signXY=float2(square.x<0?-1:1,square.y<0?-1:1);
    return float3(horizontal*abs(float2(cos(phi),sin(phi)))*signXY,(fold<0?-1:1)*(1-radius*radius));
}
struct LumenProbeTraceFrame { LumenDistanceFrame Trace; uint4 Sampling; float4 Range; };
kernel void lumenTraceProbes(constant LumenProbeTraceFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const LumenProbeOrigin* origins [[buffer(2)]],device LumenDistanceHit* hits [[buffer(3)]] LUMEN_RAY_BINDINGS,uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Trace.Traversal.y; if (index>=frame.Trace.Counts.x) return;
    uint probe=index/frame.Sampling.x,sample=index%frame.Sampling.x; LumenProbeOrigin origin=origins[probe];

    if (origin.Metadata.y!=0)
    { LumenDistanceHit empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0; empty.Metadata=uint4(SdfInvalid,SdfInvalid,SdfUnresolved,0); hits[index]=empty; return; }
    LumenDistanceRay ray; ray.OriginMinimum=origin.PositionMinimum;
    float3 direction=frame.Range.y>0?lumenCacheDirection(sample,uint(frame.Range.y)):
        lumenProbeDirection(sample,frame.Sampling.x,origin.Metadata.x,frame.Sampling.z);
    bool cached=(frame.Sampling.w&2)!=0 && hits[index].Metadata.z==LumenWorldCache;
    bool cachedSurface=cached && hits[index].Metadata.x!=SdfInvalid;
    // Without a surface endpoint cached radiance cannot delimit fiber transport.
    if ((frame.Trace.Traversal.w>>LumenHairPointerShift)!=0 && !cachedSurface) cached=false;
    ray.DirectionMaximum=float4(direction,cached && !cachedSurface?hits[index].PositionDistance.w:frame.Range.x);
    if ((frame.Sampling.w&1)!=0 && hits[index].Metadata.z==LumenScreenSurface)
    { hits[index].Medium=lumenHairMedium(scene,frame.Trace,ray,hits[index].PositionDistance.w,frame.Trace.Traversal.z); return; }
    uint layers=uint(frame.Range.z); bool terminal=false; LumenDistanceHit previous{}; previous.Medium=float4(0,0,0,1);
    constexpr float MinimumLayerCosine=0.05f,LayerNormalCosine=0.95f;
    // SDF crossings form an ordered stack. Repeated shells, grazing crossings
    // and exhausted intervals remain unknown instead of inventing background.
    for (uint layer=0;layer<layers;++layer)
    {
    LumenDistanceHit hit=previous; if (!terminal) hit=lumenTraceScene(scene,frame.Trace,ray LUMEN_RAY_FORWARD);
    if (cached && !cachedSurface && hit.Metadata.z==SdfMiss) { hits[index].Medium=hit.Medium; return; }
    if (hit.Metadata.z==SdfSurface)
    {
        uint object=frame.Trace.Offsets.y+hit.Metadata.x*SdfInstanceStride,field=frame.Trace.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
        float threshold=sdfFloats(scene,field+SdfErrorMetric).y+frame.Trace.Parameters.x*sdfFloats(scene,field+SdfGridMetric).w;
#if !defined(EVANS_LUMEN_HARDWARE_RAYS)
        hit.Uncertainty.z=threshold/sdfNormalSpeed(scene,object,hit.NormalResidual.xyz);
#endif
    }
    float roundoff=frame.Range.w*max(1.0f,max(abs(ray.OriginMinimum.x),max(abs(ray.OriginMinimum.y),abs(ray.OriginMinimum.z)))+ray.DirectionMaximum.w);
    if (cachedSurface && hit.Metadata.z==SdfSurface)
    {
        LumenDistanceHit expected=hits[index]; float tolerance=expected.Uncertainty.x+hit.Uncertainty.z+roundoff;
        if (all(hit.Metadata.xy==expected.Metadata.xy) && dot(hit.NormalResidual.xyz,expected.NormalResidual.xyz)>=LayerNormalCosine &&
            isfinite(tolerance) && abs(dot(hit.PositionDistance.xyz-expected.PositionDistance.xyz,expected.NormalResidual.xyz))<=tolerance) { hits[index].Medium=hit.Medium; return; }
    }
    if (layer>0 && !terminal && hit.Metadata.z==SdfSurface && hit.Metadata.x==previous.Metadata.x &&
        abs(dot(hit.NormalResidual.xyz,previous.NormalResidual.xyz))>=LayerNormalCosine &&
        abs(dot(hit.PositionDistance.xyz-previous.PositionDistance.xyz,previous.NormalResidual.xyz))<=hit.Uncertainty.z+previous.Uncertainty.z+roundoff)
    { hit.Metadata.z=SdfUnresolved; hit.NormalResidual=0; hit.Uncertainty.xyz=0; }
    uint destination=(probe*layers+layer)*frame.Sampling.x+sample; hits[destination]=hit;
    if (layer+1==layers) break; previous=hit;
    if (!terminal && hit.Metadata.z==SdfSurface)
    {
        float cosine=abs(dot(hit.NormalResidual.xyz,ray.DirectionMaximum.xyz));
        float next=hit.PositionDistance.w+2*hit.Uncertainty.z/max(cosine,MinimumLayerCosine)+roundoff;
        if (cosine>=MinimumLayerCosine && isfinite(next) && next>hit.PositionDistance.w && next<ray.DirectionMaximum.w) ray.OriginMinimum.w=next;
        else { previous.Metadata.z=SdfUnresolved; previous.NormalResidual=0; previous.Uncertainty.xyz=0; terminal=true; }
    }
    else terminal=true;
    }
}
struct LumenLightingFrame { LumenDistanceFrame Trace; uint4 Atlas,Work; float4 Lighting,Numeric; uint4 Tiles,Bounce; float4 Sky,EnvironmentAxis,EnvironmentPeak; };
struct LumenLightingTexel { float4 RadianceCoverage; uint4 Metadata; };
constant uint LitCardVectors=7,LitEmitterVectors=4,LitOrigin=4,LitBackward=5,LitTile=6,LitBorder=1;
constant uint LitDirectional=0,LitRectangle=2,LitComplete=1,LitUnresolved=2,LitInvalid=3;
constant uint LitEmitterNormal=1,LitEmitterEnergy=2,LitEmitterShadow=3;
float4 litFloat(device const uint4* data,uint at) { return as_type<float4>(data[at]); }
float3 litPosition(device const uint4* data,constant LumenLightingFrame& frame,uint card,uint2 tile,int2 texel,float depth)
{
    float2 uv=(float2(texel)-float2(tile)-LitBorder+0.5f)/frame.Atlas.z;
    float4 clip=float4(uv.x*2-1,1-uv.y*2,depth,1);
    return float3(dot(litFloat(data,card),clip),dot(litFloat(data,card+1),clip),dot(litFloat(data,card+2),clip));
}
bool litNeighbor(device const uint4* data,constant LumenLightingFrame& frame,texture2d<float,access::read> normals,
    uint card,uint2 tile,int2 texel,int2 step,float3 center,thread float3& delta)
{
    int2 adjacent=texel+step; delta=0;
    if (any(adjacent<int2(tile)) || any(adjacent>=int2(tile+frame.Work.w))) return false;
    float4 value=normals.read(uint2(adjacent)); if (dot(value.xyz,value.xyz)==0) return false;
    delta=litPosition(data,frame,card,tile,adjacent,value.w)-center; return all(isfinite(delta));
}
float3 litGeometricNormal(device const uint4* data,constant LumenLightingFrame& frame,texture2d<float,access::read> normals,
    uint card,uint2 tile,int2 pixel,float3 center,float3 backward)
{
    float3 right,left,down,up;
    bool r=litNeighbor(data,frame,normals,card,tile,pixel,int2(1,0),center,right),l=litNeighbor(data,frame,normals,card,tile,pixel,int2(-1,0),center,left);
    bool d=litNeighbor(data,frame,normals,card,tile,pixel,int2(0,1),center,down),u=litNeighbor(data,frame,normals,card,tile,pixel,int2(0,-1),center,up);
    if ((!r && !l) || (!d && !u)) return backward;
    float3 dx=r && (!l || dot(right,right)<=dot(left,left))?right:-left,dy=d && (!u || dot(down,down)<=dot(up,up))?down:-up;
    float3 result=sdfUnit(cross(dx,dy),backward); return dot(result,backward)>=0?result:-result;
}
constant uint BounceDirections=6,BounceCardVectors=11,BounceInverse=4,BounceOrigin=8,BounceBackward=9,BounceTile=10;
constant float BounceFacing=0.1,BounceNormals=0.25,BounceCellRadius=2;
constant float BounceLocalRoundoff=16*1.1920928955078125e-7,BounceWorldRoundoff=0.5*1.1920928955078125e-7;
constant float BouncePi=3.14159265358979323846,BounceFraction=1.0/4294967296.0,BounceAxisLimit=0.9;
float3 bounceTransform(device const uint4* data,uint matrix,float4 value)
{ return float3(dot(litFloat(data,matrix),value),dot(litFloat(data,matrix+1),value),dot(litFloat(data,matrix+2),value)); }
float bounceMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
// Jacobi transport reads one complete, immutable source atlas. Independent
// direct and bounce entry points avoid binding a dummy previous-light buffer.
bool bounceCardLighting(device const uint4* data,device const LumenLightingTexel* previous,constant LumenLightingFrame& frame,
    texture2d<float,access::read> normals,uint card,LumenDistanceHit hit,float radius,
    thread float4& value,thread float& error,thread float& facing,thread uint& status)
{
    uint base=frame.Bounce.z+frame.Bounce.w+card*BounceCardVectors; uint4 tile=data[base+BounceTile];
    float4 backward=litFloat(data,base+BounceBackward); value=0; error=0; status=0;
    facing=dot(backward.xyz,hit.NormalResidual.xyz); if (facing<BounceFacing) return false;
    float3 relative=hit.PositionDistance.xyz-litFloat(data,base+BounceOrigin).xyz,clip=bounceTransform(data,base,float4(relative,1));
    if (!all(isfinite(clip)) || any(abs(clip.xy)>1)) return false;
    float2 uv=float2(clip.x*0.5f+0.5f,0.5f-clip.y*0.5f),coordinate=float2(tile.xy)+LitBorder+uv*frame.Atlas.z-0.5f;
    int2 lower=int2(floor(coordinate)); float2 fraction=coordinate-float2(lower); float coverage=0; status=LitComplete;
    for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,tap>>1); uint2 pixel=uint2(lower+int2(corner));
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y);
        float4 normal=normals.read(pixel);
        if (weight<=0 || dot(normal.xyz,normal.xyz)==0 || dot(normal.xyz,hit.NormalResidual.xyz)<BounceNormals) continue;
        float2 texelUv=(float2(pixel)-float2(tile.xy)-LitBorder+0.5f)/frame.Atlas.z;
        float3 captured=bounceTransform(data,base+BounceInverse,float4(texelUv.x*2-1,1-texelUv.y*2,normal.w,1));
        float separation=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float roundoff=BounceLocalRoundoff*max(backward.w,max(bounceMagnitude(relative),bounceMagnitude(captured)))+
            BounceWorldRoundoff*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(separation) || separation>radius+roundoff) continue;
        LumenLightingTexel cached=previous[pixel.y*frame.Atlas.x+pixel.x]; status=max(status,cached.Metadata.x==0?LitInvalid:cached.Metadata.x);
        value.xyz+=cached.RadianceCoverage.xyz*weight; error+=separation*weight; coverage+=weight;
    }
    if (coverage<=0) { status=0; return false; }
    value=float4(value.xyz/coverage,saturate(coverage)); error/=coverage; return true;
}
uint bounceIncidentLighting(device const uint* scene,device const uint4* data,device const LumenLightingTexel* previous,
    constant LumenLightingFrame& frame,texture2d<float,access::read> normals,LumenDistanceHit hit,thread float3& incoming)
{
    incoming=0; uint4 range=data[frame.Bounce.z+hit.Metadata.x]; if (range.y==0) return 0;
    uint object=frame.Trace.Offsets.y+hit.Metadata.x*SdfInstanceStride,field=frame.Trace.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
    float threshold=sdfFloats(scene,field+SdfErrorMetric).y+frame.Trace.Parameters.x*sdfFloats(scene,field+SdfGridMetric).w;
    float radius=threshold*BounceCellRadius/sdfNormalSpeed(scene,object,hit.NormalResidual.xyz); if (!isfinite(radius)) return LitInvalid;
    float4 values[BounceDirections]; float errors[BounceDirections],facings[BounceDirections]; uint states[BounceDirections];
    for (uint direction=0;direction<BounceDirections;++direction) { values[direction]=0; errors[direction]=0; facings[direction]=0; states[direction]=0; }
    for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=frame.Bounce.z*4+range.x+candidate,card=data[word/4][word%4];
        uint direction=data[frame.Bounce.z+frame.Bounce.w+card*BounceCardVectors+BounceTile].z;
        float4 value; float error,facing; uint status;
        if (!bounceCardLighting(data,previous,frame,normals,card,hit,radius,value,error,facing,status)) continue;
        if (states[direction]==0 || error<errors[direction] || (error==errors[direction] && value.w>values[direction].w))
        { states[direction]=status; values[direction]=value; errors[direction]=error; facings[direction]=facing; }
    }
    float sum=0; uint status=0;
    for (uint direction=0;direction<BounceDirections;++direction) if (states[direction]!=0)
    {
        float weight=facings[direction]*facings[direction]*values[direction].w;
        incoming+=values[direction].xyz*weight; sum+=weight; status=max(status,states[direction]);
    }
    if (sum>0) incoming/=sum;
    if (status==LitInvalid || !all(isfinite(incoming))) { incoming=0; return LitInvalid; }
    return sum>0?status:0;
}
float3 bounceHemisphereDirection(constant LumenLightingFrame& frame,uint ray,uint texel,float3 normal)
{
    float squared=(float(ray)+0.5f)/frame.Bounce.x;
    uint phase=lumenReverseProbeBits(ray)+lumenReverseProbeBits(texel^frame.Bounce.y); float angle=2*BouncePi*(float(phase)*BounceFraction);
    float3 axis=abs(normal.z)<BounceAxisLimit?float3(0,0,1):float3(0,1,0),tangent=normalize(cross(axis,normal));
    return sqrt(squared)*(cos(angle)*tangent+sin(angle)*cross(normal,tangent))+sqrt(1-squared)*normal;
}
void litSurface(constant LumenLightingFrame& frame,device const uint* scene,device const uint4* data,
    device LumenLightingTexel* output,device const LumenLightingTexel* previous,texture2d<float,access::read> albedoImage,
    texture2d<float,access::read> normalImage,texture2d<float,access::read> emissionImage,uint2 invocation LUMEN_RAY_ARGUMENTS)
{
    uint work=invocation.x+invocation.y*frame.Trace.Traversal.y,area=frame.Tiles.w*frame.Tiles.w;
    if (work>=frame.Tiles.y*area) return;
    uint block=frame.Tiles.x+work/area,local=work%area;
    uint2 pixel=uint2(block%frame.Tiles.z,block/frame.Tiles.z)*frame.Tiles.w+uint2(local%frame.Tiles.w,local/frame.Tiles.w);
    if (any(pixel>=frame.Atlas.xy)) return;
    uint index=pixel.y*frame.Atlas.x+pixel.x;
    LumenLightingTexel result; result.RadianceCoverage=0; result.Metadata=0;
    uint cardIndex=(pixel.y/frame.Work.w)*frame.Atlas.w+pixel.x/frame.Work.w;
    if (cardIndex>=frame.Work.x) { output[index]=result; return; }
    float4 normalDepth=normalImage.read(pixel); if (dot(normalDepth.xyz,normalDepth.xyz)==0) { output[index]=result; return; }
    uint card=cardIndex*LitCardVectors; uint4 tile=data[card+LitTile];
    float3 relative=litPosition(data,frame,card,tile.xy,int2(pixel),normalDepth.w),position=relative+litFloat(data,card+LitOrigin).xyz;
    float3 geometric=litGeometricNormal(data,frame,normalImage,card,tile.xy,int2(pixel),relative,litFloat(data,card+LitBackward).xyz);
    float bias=max(frame.Lighting.x,frame.Numeric.x*max(abs(position.x),max(abs(position.y),abs(position.z))));
    if (tile.z!=SdfInvalid)
    {
        uint object=frame.Trace.Offsets.y+tile.z*SdfInstanceStride,field=frame.Trace.Offsets.x+sdfWords(scene,object+SdfTag).x*SdfFieldStride;
        float speed=sdfNormalSpeed(scene,object,geometric); float threshold=sdfFloats(scene,field+SdfErrorMetric).y+frame.Trace.Parameters.x*sdfFloats(scene,field+SdfGridMetric).w;
        bias=max(bias,threshold*frame.Lighting.y/speed);
    }
    float3 origin=position+geometric*bias,irradiance=0; bool invalid=!all(isfinite(origin)) || !all(isfinite(normalDepth));
    for (uint lightIndex=0;lightIndex<frame.Work.y && !invalid;++lightIndex)
    {
        uint first=frame.Work.z+lightIndex*LitEmitterVectors;
        float4 source=litFloat(data,first),normalRange=litFloat(data,first+LitEmitterNormal),energy=litFloat(data,first+LitEmitterEnergy);
        if (!any(energy.xyz>0) || energy.w==0) continue;
        float3 wi=source.xyz; float attenuation=1;
        if (uint(source.w)!=LitDirectional)
        {
            float3 delta=source.xyz-position; float distance=sdfLength(delta); if (distance==0) continue;
            wi=delta/distance; float window=1;
            if (normalRange.w>0)
            {
                if (max(abs(delta.x),max(abs(delta.y),abs(delta.z)))>=normalRange.w) continue;
                float3 scaled=delta/normalRange.w; float squared=dot(scaled,scaled); window=saturate(1-squared*squared); window*=window;
            }
            float emitterCosine=uint(source.w)==LitRectangle?saturate(dot(normalRange.xyz,-wi)):1;
            attenuation=energy.w*emitterCosine*window/max(distance*distance,frame.Lighting.w);
        }
        invalid=!all(isfinite(wi)) || !isfinite(attenuation); if (invalid) continue;
        float cosine=saturate(dot(normalDepth.xyz,wi)); if (cosine==0 || attenuation==0 || dot(geometric,wi)<=0) continue;
        if (data[first+LitEmitterShadow].x!=0)
        {
            LumenDistanceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(wi,frame.Lighting.z);
            if (uint(source.w)!=LitDirectional)
            {
                float3 target=source.xyz-origin; float distance=sdfLength(target);
                if (distance>0) ray.DirectionMaximum=float4(target/distance,min(frame.Lighting.z,max(0.0f,distance-frame.Lighting.x))); else ray.DirectionMaximum.w=0;
            }
            if (ray.DirectionMaximum.w>0)
            {
                ++result.Metadata.y; LumenDistanceHit visibility=lumenTraceScene(scene,frame.Trace,ray LUMEN_RAY_FORWARD);
                if (visibility.Metadata.z==SdfSurface) { ++result.Metadata.z; continue; }
                if (visibility.Metadata.z==SdfUnresolved) { ++result.Metadata.w; continue; }
                attenuation*=visibility.Medium.w;
            }
        }
        irradiance+=energy.xyz*(attenuation*cosine);
    }
    if (previous!=nullptr)
    {
        float3 indirect=0,normal=sdfUnit(normalDepth.xyz,geometric);
        for (uint rayIndex=0;rayIndex<frame.Bounce.x && !invalid;++rayIndex)
        {
            ++result.Metadata.y; float3 direction=bounceHemisphereDirection(frame,rayIndex,index,normal); if (dot(direction,geometric)<=0) continue;
            LumenDistanceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(direction,frame.Sky.w);
            LumenDistanceHit hit=lumenTraceLocalScene(scene,frame.Trace,ray,0 LUMEN_RAY_FORWARD);
            if (hit.Metadata.z==SdfMiss) { indirect+=lumenEnvironmentRadiance(frame.Sky.xyz,frame.EnvironmentAxis,frame.EnvironmentPeak.xyz,direction)*hit.Medium.w+hit.Medium.xyz; continue; }
            if (hit.Metadata.z!=SdfSurface) { ++result.Metadata.w; continue; }
            float3 incoming; uint status=bounceIncidentLighting(scene,data,previous,frame,normalImage,hit,incoming);
            if (status==LitInvalid) { invalid=true; continue; }
            if (status!=LitComplete) ++result.Metadata.w;
            indirect+=incoming*hit.Medium.w+hit.Medium.xyz;
        }
        irradiance+=indirect*(BouncePi/frame.Bounce.x);
    }
    float4 albedo=albedoImage.read(pixel); float3 radiance=emissionImage.read(pixel).xyz+albedo.xyz*(1-albedo.w)*irradiance*frame.Numeric.y;
    invalid=invalid || !all(isfinite(radiance)); result.RadianceCoverage=float4(invalid?float3(0):max(radiance,float3(0)),1);
    result.Metadata.x=invalid?LitInvalid:(result.Metadata.w!=0?LitUnresolved:LitComplete); output[index]=result;
}

kernel void lumenLightSurface(constant LumenLightingFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const uint4* data [[buffer(2)]],device LumenLightingTexel* output [[buffer(3)]],
    texture2d<float,access::read> albedo [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
    texture2d<float,access::read> emission [[texture(2)]] LUMEN_RAY_BINDINGS,uint2 invocation [[thread_position_in_grid]])
{ litSurface(frame,scene,data,output,nullptr,albedo,normals,emission,invocation LUMEN_RAY_FORWARD); }
kernel void lumenBounceSurface(constant LumenLightingFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const uint4* data [[buffer(2)]],device LumenLightingTexel* output [[buffer(3)]],device const LumenLightingTexel* previous [[buffer(4)]],
    texture2d<float,access::read> albedo [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
    texture2d<float,access::read> emission [[texture(2)]] LUMEN_RAY_BINDINGS,uint2 invocation [[thread_position_in_grid]])
{ litSurface(frame,scene,data,output,previous,albedo,normals,emission,invocation LUMEN_RAY_FORWARD); }

// Independent Metal material capture for six-direction surface-cache cards.
// The only include is the platform standard library, never another technique.
struct LumenCaptureFrame
{
    float4 WorldRows[4],NormalRows[4],ClipRows[4];
    float4 BaseColor,EmissionRoughness,Surface;
    uint4 Textures;
};
constant float NormalSquaredEpsilon=1.0e-10f,ProjectionEpsilon=1.0e-7f,UvDeterminantEpsilon=1.0e-8f;
constant float SrgbBreakpoint=0.04045f,SrgbLinearScale=12.92f,SrgbOffset=0.055f,SrgbScale=1.055f,SrgbGamma=2.4f;
float3 lumenLinearVertex(float3 value)
{
    value=saturate(value); float3 linear=value/SrgbLinearScale,curved=pow((value+SrgbOffset)/SrgbScale,float3(SrgbGamma));
    return select(curved,linear,value<=SrgbBreakpoint);
}
float3 lumenUnitNormal(float3 value,float3 fallback)
{ float square=dot(value,value); return square>NormalSquaredEpsilon?value*rsqrt(square):fallback; }
struct LumenVertex
{
    packed_float3 Position,Normal;
    packed_float4 Color;
    packed_float2 UV;
};
struct LumenVarying
{
    float4 Position [[position]];
    float3 World,Normal;
    float4 Color;
    float2 UV;
};
vertex LumenVarying lumenCaptureVertex(device const LumenVertex* vertices [[buffer(0)]],uint index [[vertex_id]],constant LumenCaptureFrame& frame [[buffer(1)]])
{
    LumenVertex vertex=vertices[index]; LumenVarying output; float4 local=float4(float3(vertex.Position),1);
    float4 world=float4(dot(frame.WorldRows[0],local),dot(frame.WorldRows[1],local),dot(frame.WorldRows[2],local),dot(frame.WorldRows[3],local));
    // Project capture-relative world coordinates without a combined matrix.
    output.Position=float4(dot(frame.ClipRows[0],world),dot(frame.ClipRows[1],world),dot(frame.ClipRows[2],world),dot(frame.ClipRows[3],world));
    output.World=world.xyz/max(abs(world.w),ProjectionEpsilon);
    output.Normal=float3(dot(frame.NormalRows[0].xyz,vertex.Normal),dot(frame.NormalRows[1].xyz,vertex.Normal),dot(frame.NormalRows[2].xyz,vertex.Normal));
    output.Color=float4(lumenLinearVertex(vertex.Color.rgb),vertex.Color.a); output.UV=float2(vertex.UV.x,1-vertex.UV.y); return output;
}
float3 lumenMappedNormal(LumenVarying input,bool front,constant LumenCaptureFrame& frame,texture2d<float> image,sampler materialSampler)
{
    float3 n=lumenUnitNormal(input.Normal,float3(0,0,1)); if (!front) n=-n;
    float3 dx=dfdx(input.World),dy=dfdy(input.World); float2 du=dfdx(input.UV),dv=dfdy(input.UV);
    float determinant=du.x*dv.y-du.y*dv.x;
    if (frame.Textures.y==0 || abs(determinant)<UvDeterminantEpsilon) return n;
    float3 tangent=(dx*dv.y-dy*du.y)/determinant,sourceBitangent=(dy*du.x-dx*dv.x)/determinant;
    tangent-=n*dot(n,tangent); if (dot(tangent,tangent)<=NormalSquaredEpsilon) return n;
    tangent=normalize(tangent); float3 bitangent=cross(n,tangent); if (dot(bitangent,sourceBitangent)<0) bitangent=-bitangent;
    float3 mapped=image.sample(materialSampler,input.UV).xyz*2-1;
    return lumenUnitNormal(tangent*mapped.x+bitangent*mapped.y+n*mapped.z,n);
}
struct LumenTargets { float4 AlbedoMetallic [[color(0)]]; float4 NormalDepth [[color(1)]]; float4 EmissionRoughness [[color(2)]]; };
fragment LumenTargets lumenCaptureFragment(LumenVarying input [[stage_in]],bool front [[front_facing]],constant LumenCaptureFrame& frame [[buffer(1)]],
    texture2d<float> diffuse [[texture(0)]],texture2d<float> normalMap [[texture(1)]],texture2d<float> opacityMap [[texture(2)]],sampler materialSampler [[sampler(0)]])
{
    float3 normal=lumenMappedNormal(input,front,frame,normalMap,materialSampler);
    float4 material=frame.Textures.x!=0?diffuse.sample(materialSampler,input.UV):float4(1);
    float opacity=frame.Textures.z!=0?opacityMap.sample(materialSampler,input.UV).r:1;
    float coverage=saturate(input.Color.a*frame.BaseColor.a*material.a*opacity); if (coverage<frame.Surface.y) discard_fragment();
    LumenTargets result; result.AlbedoMetallic=float4(saturate(input.Color.rgb*frame.BaseColor.rgb*material.rgb),frame.Surface.x);
    result.NormalDepth=float4(normal,saturate(input.Position.z)); result.EmissionRoughness=frame.EmissionRoughness; return result;
}
constant uint LumenMaterialPlanes=3;
kernel void lumenReadMaterials(constant uint4& extent [[buffer(0)]],device float4* output [[buffer(1)]],
    texture2d<float,access::read> albedo [[texture(0)]],texture2d<float,access::read> normal [[texture(1)]],
    texture2d<float,access::read> emission [[texture(2)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*extent.z; if (index>=extent.x*extent.y) return;
    uint2 pixel=uint2(index%extent.x,index/extent.x); uint first=index*LumenMaterialPlanes;
    output[first]=albedo.read(pixel); output[first+1]=normal.read(pixel); output[first+2]=emission.read(pixel);
}

// The resolve reads the preceding GPU trace and the immutable direct-radiance
// buffer. Source-normal depth remains the authority for layer/coverage tests.
struct LumenRadianceFrame { uint4 Counts,Atlas,Mapping; float4 Parameters,Precision; };
struct LumenRadianceAnswer { LumenDistanceHit Hit; float4 RadianceCoverage; uint4 Cache; };
struct LumenScreenRayFrame { float4 Projection[4],Inverse[4]; uint4 Work,Viewport; float4 Anchor,Forward,Limits; };
// Perspective-correct DDA: McGuire and Mara (JCGT 2014). The screen layer is
// trusted only inside its depth slab; all other rays retain full SDF fallback.
float4 lumenScreenTransform(constant float4* rows,float4 point)
{ return float4(dot(rows[0],point),dot(rows[1],point),dot(rows[2],point),dot(rows[3],point)); }
struct LumenScreenDepthFrame
{ float4 Inverse[4]; float4 Anchor,Forward; uint4 Source,Destination,Dispatch; float4 Numeric; };
// Independent encoders order each minimum reduction. Clear pixels contribute
// the empty sentinel; singular reconstruction contributes the unknown sentinel.
kernel void lumenBuildScreenDepth(constant LumenScreenDepthFrame& frame [[buffer(0)]],device uint* words [[buffer(1)]],
    texture2d<float,access::read> depth [[texture(0)]],uint2 invocation [[thread_position_in_grid]])
{
    constexpr uint HeaderWords=4;
    uint index=invocation.x+invocation.y*frame.Dispatch.x; if (index>=frame.Destination.x*frame.Destination.y) return;
    if (index==0) for (uint k=0;k<HeaderWords;++k) words[frame.Destination.w*HeaderWords+k]=frame.Destination[k];
    uint2 pixel=uint2(index%frame.Destination.x,index/frame.Destination.x); float minimum=frame.Numeric.y;
    if (frame.Destination.w==0)
    {
        float z=depth.read(pixel).x;
        if (isfinite(z) && z>=0 && z<1)
        {
            float2 uv=(float2(pixel)+0.5)/float2(frame.Destination.xy);
            float4 h=lumenScreenTransform(frame.Inverse,float4(uv.x*2-1,1-uv.y*2,z,1)); minimum=frame.Numeric.z;
            if (all(isfinite(h)) && abs(h.w)>frame.Numeric.x)
            {
                float3 position=h.xyz/h.w;
                float margin=frame.Anchor.w*dot(abs(position)+abs(frame.Anchor.xyz),abs(frame.Forward.xyz));
                float value=dot(position-frame.Anchor.xyz,frame.Forward.xyz)-margin;
                if (isfinite(value)) minimum=value;
            }
        }
    }
    else for (uint child=0;child<4;++child)
    {
        // Repeated edge values do not change the minimum, and bounded
        // addresses remain safe when the compiler predicates child reads.
        uint2 p=min(pixel*2+uint2(child&1,child>>1),max(frame.Source.xy,uint2(1))-1);
        minimum=min(minimum,as_type<float>(words[frame.Source.z+p.y*frame.Source.x+p.x]));
    }
    words[frame.Destination.z+index]=as_type<uint>(minimum);
}
bool lumenClipScreenPlane(float a,float b,thread float& begin,thread float& end)
{
    if (a<0 && b<0) return false;
    if (a<0) begin=max(begin,a/(a-b)); else if (b<0) end=min(end,a/(a-b));
    return begin<=end;
}
bool lumenFindScreenHit(constant LumenScreenRayFrame& frame,LumenProbeOrigin seed,float3 direction,
    texture2d<float,access::read> depth,texture2d<float,access::read> normals,texture2d<float,access::read> color,
    device const uint* tree,thread LumenDistanceHit& hit,thread float3& incoming)
{
    constexpr float NormalFloor=1e-12,ParallelFloor=1e-7,ClearDepth=1,PixelCenter=0.5;
    float3 origin=seed.PositionMinimum.xyz,from=origin+direction*seed.PositionMinimum.w,to=origin+direction*frame.Limits.x;
    float4 a=lumenScreenTransform(frame.Projection,float4(from,1)),b=lumenScreenTransform(frame.Projection,float4(to,1));
    if (!all(isfinite(a)) || !all(isfinite(b))) return false;
    float begin=0,end=1;
    if (!lumenClipScreenPlane(a.w+a.x,b.w+b.x,begin,end) || !lumenClipScreenPlane(a.w-a.x,b.w-b.x,begin,end) ||
        !lumenClipScreenPlane(a.w+a.y,b.w+b.y,begin,end) || !lumenClipScreenPlane(a.w-a.y,b.w-b.y,begin,end) ||
        !lumenClipScreenPlane(a.z,b.z,begin,end) || !lumenClipScreenPlane(a.w-a.z,b.w-b.z,begin,end) ||
        !lumenClipScreenPlane(a.w-frame.Limits.w,b.w-frame.Limits.w,begin,end)) return false;
    float4 h0=mix(a,b,begin),h1=mix(a,b,end); float w0=1/h0.w,w1=1/h1.w;
    float3 q0=mix(from,to,begin)*w0,q1=mix(from,to,end)*w1;
    float2 p0=(h0.xy*w0*float2(0.5,-0.5)+0.5)*float2(frame.Viewport.xy);
    float2 p1=(h1.xy*w1*float2(0.5,-0.5)+0.5)*float2(frame.Viewport.xy);
    float2 size=abs(p1-p0); float pixels=max(size.x,size.y); if (pixels<1) return false;
    float advance=frame.Limits.z/pixels,rate=dot(direction,frame.Forward.xyz);
    uint next=0,level=0;
    for (uint visit=0;visit<frame.Viewport.z;++visit)
    {
        uint step=next++;
        float start=float(step)*advance; if (start>=1) break; float finish=min(start+advance,1.0f);
        int2 pixel=int2(floor(mix(p0,p1,(start+finish)*0.5)));
        if (any(pixel<0) || any(pixel>=int2(frame.Viewport.xy))) return false;
        if (all(uint2(pixel)==seed.Metadata.zw)) continue;
        if (frame.Viewport.w!=0)
        {
            uint2 cell=uint2(pixel)>>level; float2 motion=p1-p0;
            float2 low=float2(cell<<level),high=min(low+float(1u<<level),float2(frame.Viewport.xy));
            float boundary=1;
            if (motion.x!=0) boundary=min(boundary,((motion.x>0?high.x:low.x)-p0.x)/motion.x);
            if (motion.y!=0) boundary=min(boundary,((motion.y>0?high.y:low.y)-p0.y)/motion.y);
            uint stop=uint(clamp(ceil(boundary/advance-PixelCenter),float(step+1),ceil(1/advance)));
            float tailStart=float(stop-1)*advance,tailEnd=min(tailStart+advance,1.0f);
            int2 tail=int2(floor(mix(p0,p1,(tailStart+tailEnd)*0.5)));
            if (any(tail<0) || any(tail>=int2(frame.Viewport.xy)) || any((uint2(tail)>>level)!=cell)) { stop=step+1; tailEnd=finish; }

            {
                constexpr uint HeaderWords=4; uint header=level*HeaderWords;
                float nearest=as_type<float>(tree[tree[header+2]+cell.y*tree[header]+cell.x]);
                float3 aPoint=mix(q0,q1,start)/mix(w0,w1,start),bPoint=mix(q0,q1,tailEnd)/mix(w0,w1,tailEnd);
                float bound=max(dot(aPoint-origin,direction)*rate,dot(bPoint-origin,direction)*rate)+dot(origin-frame.Anchor.xyz,frame.Forward.xyz);
                float error=frame.Anchor.w*dot(abs(aPoint)+abs(bPoint)+2*(abs(origin)+abs(frame.Anchor.xyz)),float3(1));
                if (isfinite(bound+error) && bound+error<nearest) { next=stop; level=min(level+1,frame.Viewport.w-1); continue; }
            }
            if (level>0) { --level; next=step; continue; }
        }
        float z=depth.read(uint2(pixel)).x; if (!isfinite(z) || z<0 || z>=ClearDepth) continue;
        float2 uv=(float2(pixel)+PixelCenter)/float2(frame.Viewport.xy);
        float4 h=lumenScreenTransform(frame.Inverse,float4(uv.x*2-1,1-uv.y*2,z,1));
        if (!all(isfinite(h)) || abs(h.w)<=frame.Limits.w) return false;
        float3 surface=h.xyz/h.w;
        float t0=dot(mix(q0,q1,start)/mix(w0,w1,start)-origin,direction);
        float t1=dot(mix(q0,q1,finish)/mix(w0,w1,finish)-origin,direction);
        float plane=dot(surface-origin,frame.Forward.xyz),farther=max(t0*rate,t1*rate);
        if (!all(isfinite(float4(t0,t1,plane,farther)))) return false;
        if (farther<plane) continue;
        if (min(t0*rate,t1*rate)>plane+frame.Limits.y) return false;
        float3 normal=normals.read(uint2(pixel)).xyz;
        if (!all(isfinite(normal)) || dot(normal,normal)<=NormalFloor || dot(normal,direction)>=0) return false;
        float travel=abs(rate)>ParallelFloor?clamp(plane/rate,min(t0,t1),max(t0,t1)):(t0+t1)*0.5;
        if (travel<=seed.PositionMinimum.w || travel>frame.Limits.x) return false;
        incoming=color.read(uint2(pixel)).xyz; if (!all(isfinite(incoming))) return false;
        incoming=max(incoming,0.0f); hit.PositionDistance=float4(surface,travel); hit.NormalResidual=float4(normalize(normal),0);
        hit.Uncertainty=0; hit.Metadata=uint4(uint2(pixel),LumenScreenSurface,visit+1); return true;
    }
    return false;
}
kernel void lumenTraceScreen(constant LumenScreenRayFrame& frame [[buffer(0)]],device const LumenProbeOrigin* origins [[buffer(1)]],
    device LumenDistanceHit* hits [[buffer(2)]],device LumenRadianceAnswer* answers [[buffer(3)]],device const uint* tree [[buffer(4)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
    texture2d<float,access::read> color [[texture(2)]],uint2 invocation [[thread_position_in_grid]])
{
    constexpr uint MissingCard=0xffffffffu,Complete=1;
    uint index=invocation.x+invocation.y*frame.Work.w; if (index>=frame.Work.x) return;
    LumenProbeOrigin origin=origins[index/frame.Work.y]; LumenRadianceAnswer result; result.Hit.Medium=float4(0,0,0,1);
    result.Hit.PositionDistance=0; result.Hit.NormalResidual=0; result.Hit.Uncertainty=0;
    result.Hit.Metadata=uint4(MissingCard,MissingCard,0,0); result.RadianceCoverage=0; result.Cache=uint4(MissingCard,0,0,0);
    LumenDistanceHit hit; hit.Medium=float4(0,0,0,1); float3 light;
    float3 direction=lumenProbeDirection(index%frame.Work.y,frame.Work.y,origin.Metadata.x,frame.Work.z);
    if (frame.Forward.w!=0)
    { float4 reflected=as_type<float4>(origin.Metadata); direction=reflected.xyz; origin.Metadata.y=reflected.w>0?0:1; }
    if (origin.Metadata.y==0 && lumenFindScreenHit(frame,origin,direction,depth,normals,color,tree,hit,light))
    { result.Hit=hit; result.RadianceCoverage=float4(light,1); result.Cache.z=Complete; }
    hits[index]=result.Hit; answers[index]=result;
}
constant uint RadianceDirections=6,RadianceCardStride=11,RadianceInverse=4,RadianceOrigin=8,RadianceBackward=9,RadianceTile=10,RadianceBorder=1;
constant uint RadianceMissing=0xffffffffu,RadianceComplete=1,RadiancePartial=2,RadianceInvalid=3;
float4 radianceVector(device const uint4* data,uint offset) { return as_type<float4>(data[offset]); }
float radianceMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float3 radianceProject(device const uint4* data,uint base,float4 v)
{ return float3(dot(radianceVector(data,base),v),dot(radianceVector(data,base+1),v),dot(radianceVector(data,base+2),v)); }
bool radianceCard(device const uint4* data,device const LumenLightingTexel* lighting,constant LumenRadianceFrame& frame,
    texture2d<float,access::read> normalImage,uint card,LumenDistanceHit hit,float radius,thread float4& value,thread float& error,thread float& facing,thread uint& state)
{
    value=0; error=0; state=0; uint base=frame.Mapping.x+card*RadianceCardStride; uint4 tile=data[base+RadianceTile];
    float4 backward=radianceVector(data,base+RadianceBackward); facing=dot(hit.NormalResidual.xyz,backward.xyz); if (facing<frame.Parameters.x) return false;
    float3 relative=hit.PositionDistance.xyz-radianceVector(data,base+RadianceOrigin).xyz,projected=radianceProject(data,base,float4(relative,1));
    if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 location=float2(tile.xy)+RadianceBorder+float2(projected.x*0.5f+0.5f,0.5f-projected.y*0.5f)*frame.Counts.y-0.5f;
    int2 lower=int2(floor(location)); float2 fraction=location-float2(lower); float coverage=0; state=RadianceComplete;
    for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,tap>>1); int2 pixel=lower+int2(corner);
        float weight=(corner.x!=0?fraction.x:1-fraction.x)*(corner.y!=0?fraction.y:1-fraction.y); float4 normal=normalImage.read(uint2(pixel));
        if (weight<=0 || dot(normal.xyz,normal.xyz)==0 || dot(normal.xyz,hit.NormalResidual.xyz)<frame.Parameters.y) continue;
        float2 uv=(float2(pixel)-float2(tile.xy)-RadianceBorder+0.5f)/frame.Counts.y;
        float3 captured=radianceProject(data,base+RadianceInverse,float4(uv.x*2-1,1-uv.y*2,normal.w,1));
        float planeError=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float allowance=frame.Precision.x*max(backward.w,max(radianceMagnitude(relative),radianceMagnitude(captured)))+frame.Precision.y*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(planeError) || planeError>radius+allowance) continue;
        LumenLightingTexel light=lighting[uint(pixel.y)*frame.Atlas.x+uint(pixel.x)]; state=max(state,light.Metadata.x!=0?light.Metadata.x:RadianceInvalid);
        value.xyz+=light.RadianceCoverage.xyz*weight; error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0) { state=0; return false; }
    value=float4(value.xyz/coverage,saturate(coverage)); error/=coverage; return true;
}
kernel void lumenResolveRadiance(constant LumenRadianceFrame& frame [[buffer(0)]],device const LumenDistanceHit* hits [[buffer(1)]],
    device const LumenLightingTexel* lighting [[buffer(2)]],device const uint4* data [[buffer(3)]],device LumenRadianceAnswer* answers [[buffer(4)]],
    texture2d<float,access::read> normalImage [[texture(0)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Atlas.w; if (index>=frame.Counts.x) return;
    LumenRadianceAnswer result; result.Hit.Medium=float4(0,0,0,1); result.Hit=hits[index]; result.RadianceCoverage=0; result.Cache=uint4(RadianceMissing,0,0,0);
    if (result.Hit.Metadata.z==LumenScreenSurface || result.Hit.Metadata.z==LumenWorldCache) { answers[index].Hit.Medium=result.Hit.Medium; return; }
    float radius=frame.Parameters.z+result.Hit.Uncertainty.z*frame.Parameters.w;
    if (frame.Mapping.z==0) result.Hit.Uncertainty.z=0;
    if (result.Hit.Metadata.z!=SdfSurface) { answers[index]=result; return; }
    if (!isfinite(radius)) { result.Cache.z=RadianceInvalid; answers[index]=result; return; }
    uint4 range=data[result.Hit.Metadata.x]; if (range.y==0) { answers[index]=result; return; }
    float4 groups[RadianceDirections]; float errors[RadianceDirections],facings[RadianceDirections]; uint states[RadianceDirections],cards[RadianceDirections];
    for (uint group=0;group<RadianceDirections;++group) { groups[group]=0; errors[group]=0; facings[group]=0; states[group]=0; cards[group]=RadianceMissing; }
    for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=range.x+candidate,card=data[word/4][word%4],group=data[frame.Mapping.x+card*RadianceCardStride+RadianceTile].z;
        float4 value; float error,facing; uint state; if (!radianceCard(data,lighting,frame,normalImage,card,result.Hit,radius,value,error,facing,state)) continue;
        if (states[group]==0 || error<errors[group] || (error==errors[group] && value.w>groups[group].w))
        { groups[group]=value; errors[group]=error; facings[group]=facing; states[group]=state; cards[group]=card; }
    }
    float total=0,available=0,dominant=0; uint state=RadianceComplete;
    for (uint group=0;group<RadianceDirections;++group) if (states[group]!=0)
    { float orientation=facings[group]*facings[group]; total+=orientation*groups[group].w; available+=orientation; }
    if (total>0)
    {
        for (uint group=0;group<RadianceDirections;++group) if (states[group]!=0)
        {
            float weight=facings[group]*facings[group]*groups[group].w/total; result.RadianceCoverage.xyz+=groups[group].xyz*weight;
            ++result.Cache.y; state=max(state,states[group]); if (states[group]!=RadianceComplete) ++result.Cache.w;
            if (weight>dominant || (weight==dominant && cards[group]<result.Cache.x)) { dominant=weight; result.Cache.x=cards[group]; }
        }
        result.RadianceCoverage.w=saturate(total/available); result.Cache.z=state;
        if (state==RadianceInvalid || !all(isfinite(result.RadianceCoverage))) { result.RadianceCoverage=0; result.Cache.z=RadianceInvalid; }
    }
    answers[index]=result;
}

struct LumenProbeIntegrationFrame { uint4 Counts; float4 SkySolidAngle,EnvironmentAxis,EnvironmentPeak; };
struct LumenProbeIrradiance { float4 Coefficients[9]; float4 Coverage; uint4 TraceCounts,CacheCounts,Metadata; };
constant uint ProbeHarmonics=9,ProbeComplete=1,ProbePartial=2,ProbeInvalid=3;
void lumenProjectIrradiance(thread LumenProbeIrradiance& result,float3 flux,float3 w)
{
    constexpr float Pi=3.14159265358979323846;
    const float constantBand=Pi*sqrt(1/(4*Pi)),linearBand=(2*Pi/3)*sqrt(3/(4*Pi));
    const float mixedBand=(Pi/4)*sqrt(15/(4*Pi)),polarBand=(Pi/4)*sqrt(5/(16*Pi)),differenceBand=(Pi/4)*sqrt(15/(16*Pi));
    result.Coefficients[0].xyz+=flux*constantBand;
    result.Coefficients[1].xyz+=flux*(w.y*linearBand); result.Coefficients[2].xyz+=flux*(w.z*linearBand); result.Coefficients[3].xyz+=flux*(w.x*linearBand);
    result.Coefficients[4].xyz+=flux*(w.x*w.y*mixedBand); result.Coefficients[5].xyz+=flux*(w.y*w.z*mixedBand);
    result.Coefficients[6].xyz+=flux*((3*w.z*w.z-1)*polarBand); result.Coefficients[7].xyz+=flux*(w.x*w.z*mixedBand);
    result.Coefficients[8].xyz+=flux*((w.x*w.x-w.y*w.y)*differenceBand);
}
kernel void lumenIntegrateProbes(constant LumenProbeIntegrationFrame& frame [[buffer(0)]],
    device const LumenRadianceAnswer* incoming [[buffer(1)]],device const LumenProbeOrigin* origins [[buffer(2)]],
    device LumenProbeIrradiance* irradiance [[buffer(3)]],uint2 invocation [[thread_position_in_grid]])
{
    uint index=invocation.x+invocation.y*frame.Counts.w; if (index>=frame.Counts.x) return;
    LumenProbeIrradiance result; for (uint coefficient=0;coefficient<ProbeHarmonics;++coefficient) result.Coefficients[coefficient]=0;
    result.TraceCounts=0; result.CacheCounts=0; result.Coverage=0;
    uint identifier=origins[index].Metadata.x; float materialCoverage=0,meanScaledDistance=0; bool invalid=origins[index].Metadata.y!=0;
    for (uint sample=0;sample<frame.Counts.y;++sample)
    {
        LumenRadianceAnswer ray=incoming[index*frame.Counts.y+sample]; float3 light=0;
        float3 direction=lumenProbeDirection(sample,frame.Counts.y,identifier,frame.Counts.z);
        if (ray.Hit.Metadata.z==SdfSurface || ray.Hit.Metadata.z==LumenScreenSurface || ray.Hit.Metadata.z==LumenWorldCache)
        {
            if (ray.Hit.Metadata.z==LumenWorldCache) ++result.TraceCounts.w;
            else { ++result.TraceCounts.x; meanScaledDistance+=ray.Hit.PositionDistance.w/float(frame.Counts.y); }
            if (ray.Cache.z==ProbeComplete) { ++result.CacheCounts.x; light=ray.RadianceCoverage.xyz; materialCoverage+=ray.RadianceCoverage.w; }
            else if (ray.Cache.z==ProbePartial) { ++result.CacheCounts.y; light=ray.RadianceCoverage.xyz; materialCoverage+=ray.RadianceCoverage.w; }
            else if (ray.Cache.z==ProbeInvalid) { ++result.CacheCounts.w; invalid=true; }
            else ++result.CacheCounts.z;
        }
        else if (ray.Hit.Metadata.z==SdfMiss) { ++result.TraceCounts.y; light=lumenEnvironmentRadiance(frame.SkySolidAngle.xyz,frame.EnvironmentAxis,frame.EnvironmentPeak.xyz,direction); }
        else ++result.TraceCounts.z;
        light=light*ray.Hit.Medium.w+ray.Hit.Medium.xyz;
        lumenProjectIrradiance(result,light*frame.SkySolidAngle.w,direction);
    }
    result.Coverage=float4(float(result.TraceCounts.y+result.CacheCounts.x)/frame.Counts.y,(result.TraceCounts.x+result.TraceCounts.w)?materialCoverage/(result.TraceCounts.x+result.TraceCounts.w):0,
        float(result.TraceCounts.y)/frame.Counts.y,result.TraceCounts.x?meanScaledDistance*(float(frame.Counts.y)/result.TraceCounts.x):0);
    for (uint coefficient=0;coefficient<ProbeHarmonics;++coefficient) invalid=invalid || !all(isfinite(result.Coefficients[coefficient]));
    invalid=invalid || !all(isfinite(result.Coverage));
    // The average of bounded coverage can round just beyond its upper bound.
    result.Coverage.y=saturate(result.Coverage.y); uint status=(result.TraceCounts.z || result.CacheCounts.y || result.CacheCounts.z)?ProbePartial:ProbeComplete;
    if (invalid) { status=ProbeInvalid; result.Coverage=0; for (uint coefficient=0;coefficient<ProbeHarmonics;++coefficient) result.Coefficients[coefficient]=0; }
    result.Metadata=uint4(identifier,status,frame.Counts.y,frame.Counts.z); irradiance[index]=result;
}

// Local plane compatibility limits interpolation across unrelated receivers.
// L2 coefficients contain cosine-convolved irradiance; Lambert adds albedo/pi.
struct LumenGatherFrame { LumenDistanceFrame Trace; LumenScreenFrame Screen; float4 Appearance,Numeric; uint4 Model; float4 Visibility; };
struct LumenGatherVertex { float4 Position [[position]]; };
vertex LumenGatherVertex lumenGatherVertex(uint vertexID [[vertex_id]])
{
    float2 corner=float2((vertexID<<1)&2,vertexID&2);
    return {float4(corner*float2(2,-2)+float2(-1,1),0,1)};
}
float lumenGatherBasis(uint coefficient,float3 n)
{
    constexpr float Pi=3.14159265358979323846;
    switch (coefficient)
    {
        case 0:return sqrt(1/(4*Pi)); case 1:return sqrt(3/(4*Pi))*n.y;
        case 2:return sqrt(3/(4*Pi))*n.z; case 3:return sqrt(3/(4*Pi))*n.x;
        case 4:return sqrt(15/(4*Pi))*n.x*n.y; case 5:return sqrt(15/(4*Pi))*n.y*n.z;
        case 6:return sqrt(5/(16*Pi))*(3*n.z*n.z-1); case 7:return sqrt(15/(4*Pi))*n.x*n.z;
        case 8:return sqrt(15/(16*Pi))*(n.x*n.x-n.y*n.y);
    }
    return 0;
}
float lumenGatherAxis(float pixel,uint extent,uint tiles,uint tileSide)
{
    float side=float(tileSide),coordinate=pixel/side-0.5;
    if (tiles>1 && extent%tileSide!=0)
    {
        float previous=(float(tiles)-1.5)*side,last=(float(tiles-1)*side+extent)*0.5;
        if (pixel>previous) coordinate=float(tiles-2)+(pixel-previous)/(last-previous);
    }
    return clamp(coordinate,0.0f,float(tiles-1));
}
float lumenGatherMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
// Visibility links share the radiance producer's immutable scene. A completed
// miss admits a probe; a surface or unresolved interval conservatively rejects it.
kernel void lumenProbeVisibility(constant LumenGatherFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const LumenProbeOrigin* origins [[buffer(2)]],device uint* masks [[buffer(3)]],
    depth2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]] LUMEN_RAY_BINDINGS,
    uint2 dispatch [[thread_position_in_grid]])
{
    uint index=dispatch.x+dispatch.y*frame.Trace.Traversal.y; if (index>=frame.Trace.Counts.x) return;
    uint words=probeLinkWords(frame.Screen.Grid.w+1);
    for (uint word=0;word<words;++word) masks[index*words+word]=0; uint2 pixel=uint2(index%frame.Screen.Viewport.x,index/frame.Screen.Viewport.x);
    float z=depth.read(pixel); float4 material=normals.read(pixel);
    if (!isfinite(z) || z<0 || z>=1 || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=frame.Numeric.x) return;
    float2 center=float2(pixel)+0.5,uv=center/float2(frame.Screen.Viewport.xy);
    float4 clip=float4(uv*float2(2,-2)+float2(-1,1),z,1);
    float4 h=float4(dot(frame.Screen.Inverse[0],clip),dot(frame.Screen.Inverse[1],clip),dot(frame.Screen.Inverse[2],clip),dot(frame.Screen.Inverse[3],clip));
    if (!all(isfinite(h)) || abs(h.w)<=frame.Screen.Parameters.w) return;
    float3 world=h.xyz/h.w,normal=normalize(material.xyz);
    if (!all(isfinite(world)) || any(abs(world)>frame.Screen.Parameters.z)) return;
    float2 grid=float2(lumenGatherAxis(center.x,frame.Screen.Viewport.x,frame.Screen.Viewport.w,frame.Screen.Viewport.z),
        lumenGatherAxis(center.y,frame.Screen.Viewport.y,frame.Screen.Grid.x,frame.Screen.Viewport.z));
    uint2 cell=uint2(floor(grid)); float2 fraction=grid-float2(cell); uint3 mask=0;
    uint slots=frame.Screen.Grid.w+1;
    for (uint tap=0;tap<4*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),tile=min(cell+corner,uint2(frame.Screen.Viewport.w,frame.Screen.Grid.x)-1);
        if ((corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y)<=0) continue;
        uint probeIndex=(tile.y*frame.Screen.Viewport.w+tile.x)*slots+tap%slots; LumenProbeOrigin origin=origins[probeIndex];
        if (origin.Metadata.x!=probeIndex || origin.Metadata.y!=0 || any(origin.Metadata.zw>=frame.Screen.Viewport.xy)) continue;
        float4 probeSurface=normals.read(origin.Metadata.zw);
        if (!all(isfinite(probeSurface)) || probeSurface.w<0 || dot(probeSurface.xyz,probeSurface.xyz)<=frame.Numeric.x) continue;
        float3 n=normalize(probeSurface.xyz),anchor=origin.PositionMinimum.xyz-n*frame.Screen.Parameters.x;
        if (dot(n,normal)<=frame.Appearance.y || !all(isfinite(anchor))) continue;
        float bias=frame.Visibility.x+frame.Visibility.y*max(lumenGatherMagnitude(world),lumenGatherMagnitude(anchor));
        float3 start=world+normal*bias,end=anchor+n*bias,delta=end-start; float distance=sdfLength(delta);
        if (!all(isfinite(start)) || !all(isfinite(end)) || !isfinite(distance)) continue;
        if (distance==0) { mask[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits); continue; }
        LumenDistanceRay ray={float4(start,0),float4(sdfUnit(delta,normal),distance)};
        LumenDistanceHit hit=lumenTraceScene(scene,frame.Trace,ray LUMEN_RAY_FORWARD); if (hit.Metadata.z==SdfMiss) mask[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits);
    }
    for (uint word=0;word<words;++word) masks[index*words+word]=mask[word];
}
fragment float4 lumenGatherFragment(LumenGatherVertex input [[stage_in]],constant LumenGatherFrame& frame [[buffer(0)]],
    device const LumenProbeOrigin* origins [[buffer(1)]],device const LumenProbeIrradiance* probes [[buffer(2)]],device const uint* links [[buffer(3)]],
    texture2d<float,access::read> sceneImage [[texture(0)]],depth2d<float,access::read> depthImage [[texture(1)]],
    texture2d<float,access::read> normalImage [[texture(2)]],texture2d<float,access::read> diffuseImage [[texture(3)]],
    texture2d<float,access::read> ambientImage [[texture(4)]])
{
    constexpr float Pi=3.14159265358979323846,CoordinateRoundoff=32*1.1920928955078125e-7;
    constexpr uint PBR=3;
    uint2 pixel=uint2(input.Position.xy); float4 scene=frame.Model.z!=0?float4(0):sceneImage.read(pixel),materialNormal=normalImage.read(pixel);
    float depth=depthImage.read(pixel);
    if (frame.Appearance.x==0 || !isfinite(depth) || depth<0 || depth>=1 || materialNormal.w<0 ||
        !all(isfinite(materialNormal)) || dot(materialNormal.xyz,materialNormal.xyz)<=frame.Numeric.x) return scene;
    float2 uv=input.Position.xy/float2(frame.Screen.Viewport.xy); float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 h=float4(dot(frame.Screen.Inverse[0],clip),dot(frame.Screen.Inverse[1],clip),dot(frame.Screen.Inverse[2],clip),dot(frame.Screen.Inverse[3],clip));
    if (!all(isfinite(h)) || abs(h.w)<=frame.Screen.Parameters.w) return scene;
    float3 world=h.xyz/h.w,normal=normalize(materialNormal.xyz);
    if (!all(isfinite(world)) || any(abs(world)>frame.Screen.Parameters.z)) return scene;
    float2 grid=float2(lumenGatherAxis(input.Position.x,frame.Screen.Viewport.x,frame.Screen.Viewport.w,frame.Screen.Viewport.z),
        lumenGatherAxis(input.Position.y,frame.Screen.Viewport.y,frame.Screen.Grid.x,frame.Screen.Viewport.z));
    uint2 cell=uint2(floor(grid)); float2 fraction=grid-float2(cell); float3 irradiance=0; float totalWeight=0;
    uint slots=frame.Screen.Grid.w+1;
    for (uint tap=0;tap<4*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),tile=min(cell+corner,uint2(frame.Screen.Viewport.w,frame.Screen.Grid.x)-1);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y); if (weight<=0) continue;
        if (frame.Model.y!=0 && (links[(pixel.y*frame.Screen.Viewport.x+pixel.x)*probeLinkWords(slots)+tap/LumenLinkWordBits]&(1u<<(tap%LumenLinkWordBits)))==0) continue;
        uint index=(tile.y*frame.Screen.Viewport.w+tile.x)*slots+tap%slots; LumenProbeOrigin origin=origins[index]; LumenProbeIrradiance probe=probes[index];
        if (origin.Metadata.x!=index || origin.Metadata.y!=0 || any(origin.Metadata.zw>=frame.Screen.Viewport.xy) ||
            probe.Metadata.x!=index || probe.Metadata.y<ProbeComplete || probe.Metadata.y>ProbePartial) continue;
        float4 anchorNormal=normalImage.read(origin.Metadata.zw);
        if (anchorNormal.w<0 || !all(isfinite(anchorNormal)) || dot(anchorNormal.xyz,anchorNormal.xyz)<=frame.Numeric.x) continue;
        float3 n=normalize(anchorNormal.xyz),anchor=origin.PositionMinimum.xyz-n*frame.Screen.Parameters.x;
        float agreement=dot(normal,n); if (agreement<=frame.Appearance.y) continue;
        float3 delta=world-anchor; float magnitude=lumenGatherMagnitude(delta),distance=magnitude>0?magnitude*length(delta/magnitude):0;
        float error=max(abs(dot(delta,normal)),abs(dot(delta,n)));
        float tolerance=frame.Appearance.z+frame.Appearance.w*distance+CoordinateRoundoff*max(lumenGatherMagnitude(world),lumenGatherMagnitude(anchor));
        if (!isfinite(error) || !isfinite(tolerance) || error>tolerance) continue;
        float normalWeight=(agreement-frame.Appearance.y)/(1-frame.Appearance.y),planeWeight=tolerance>0?saturate(1-error/tolerance):1;
        weight*=normalWeight*normalWeight*planeWeight;
        float3 value=0; for (uint coefficient=0;coefficient<ProbeHarmonics;++coefficient) value+=probe.Coefficients[coefficient].xyz*lumenGatherBasis(coefficient,normal);
        if (!all(isfinite(value))) continue; irradiance+=max(value,0.0f)*weight; totalWeight+=weight;
    }
    if (totalWeight<=frame.Numeric.y || !all(isfinite(irradiance))) return scene;
    float3 diffuse=saturate(diffuseImage.read(pixel).xyz);
    if (frame.Model.x==PBR) diffuse*=(1-frame.Numeric.w)*(1-saturate(ambientImage.read(pixel).w));
    float3 indirect=irradiance/totalWeight*diffuse/Pi;
    if (!all(isfinite(indirect))) return scene;
    if (frame.Model.z!=0) return float4(clamp(indirect,0.0f,frame.Numeric.z),1);
    return float4(clamp(scene.xyz+indirect*frame.Appearance.x,0.0f,frame.Numeric.z),scene.w);
}

// Diffuse-only temporal reconstruction. The packed history retains FP32 depth
// alongside binary16 energy/count and signed octahedral normal coordinates.
struct LumenTemporalFrame
{
    float4 Inverse[4],PreviousInverse[4],PreviousPlane;
    uint4 Extent;
    float4 Rules,Clip,Jitter,Numeric;
};
constexpr uint LumenTemporalHalfBits=16,LumenTemporalHalfMask=0xffffu;
constexpr float LumenTemporalNormalScale=32767,LumenTemporalNormalFloor=1e-12;
constexpr int LumenTemporalFilterRadius=1;
float4 lumenTemporalColor(uint4 bits)
{ return float4(float2(as_type<half2>(bits.x)),float2(as_type<half2>(bits.y))); }
float2 lumenTemporalSign(float2 value) { return float2(value.x>=0?1:-1,value.y>=0?1:-1); }
float3 lumenTemporalNormal(uint packed)
{
    float2 oct=float2(int(packed<<LumenTemporalHalfBits)>>LumenTemporalHalfBits,int(packed)>>LumenTemporalHalfBits)/LumenTemporalNormalScale;
    float3 normal=float3(oct,1-abs(oct.x)-abs(oct.y));
    if (normal.z<0) normal.xy=(1-abs(normal.yx))*lumenTemporalSign(normal.xy);
    return normalize(normal);
}
uint4 lumenTemporalPack(float4 color,float3 normal,float depth)
{
    float2 oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
    if (normal.z<0) oct=(1-abs(oct.yx))*lumenTemporalSign(oct);
    int2 encoded=int2(round(clamp(oct,-1.0f,1.0f)*LumenTemporalNormalScale));
    return uint4(as_type<uint>(half2(color.xy)),as_type<uint>(half2(color.zw)),
        (uint(encoded.x)&LumenTemporalHalfMask)|(uint(encoded.y)<<LumenTemporalHalfBits),as_type<uint>(depth));
}
bool lumenTemporalSurface(constant LumenTemporalFrame& frame,uint2 pixel,depth2d<float,access::read> depths,
    texture2d<float,access::read> normals,thread float3& world,thread float3& normal,thread float& depth)
{
    depth=depths.read(pixel); float4 surface=normals.read(pixel); world=0; normal=float3(0,0,1);
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(surface)) || surface.w<0 || dot(surface.xyz,surface.xyz)<=LumenTemporalNormalFloor) return false;
    float2 uv=(float2(pixel)+0.5)/float2(frame.Extent.xy); float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 h=float4(dot(frame.Inverse[0],clip),dot(frame.Inverse[1],clip),dot(frame.Inverse[2],clip),dot(frame.Inverse[3],clip));
    if (!all(isfinite(h)) || abs(h.w)<=frame.Numeric.z) return false;
    world=h.xyz/h.w; normal=normalize(surface.xyz); return all(isfinite(world));
}
float lumenTemporalDistance(constant LumenTemporalFrame& frame,float2 center,float depth)
{
    float2 uv=center/float2(frame.Extent.xy); float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 h=float4(dot(frame.PreviousInverse[0],clip),dot(frame.PreviousInverse[1],clip),dot(frame.PreviousInverse[2],clip),dot(frame.PreviousInverse[3],clip));
    if (!all(isfinite(h)) || abs(h.w)<=frame.Numeric.z) return -1;
    return dot(frame.PreviousPlane,float4(h.xyz/h.w,1));
}
void lumenAccumulateHistory(constant LumenTemporalFrame& frame,device const uint4* committed,device uint4* pending,
    texture2d<float,access::read> current,depth2d<float,access::read> depths,texture2d<float,access::read> normals,
    texture2d<float,access::read> velocities,uint2 dispatch,device const float4* specular,bool reflected)
{
    uint index=dispatch.x+dispatch.y*frame.Extent.w; if (index>=frame.Extent.z) return;
    uint2 pixel=uint2(index%frame.Extent.x,index/frame.Extent.x); float3 world,normal; float depth;
    float4 estimate=reflected?specular[index]:current.read(pixel);
    if (!lumenTemporalSurface(frame,pixel,depths,normals,world,normal,depth) || !all(isfinite(estimate)) || estimate.w<=0)
    { pending[index]=lumenTemporalPack(float4(0),float3(0,0,1),1); return; }
    if (reflected && estimate.w<1) { pending[index]=lumenTemporalPack(float4(clamp(estimate.xyz,0.0f,frame.Numeric.w),estimate.w),normal,depth); return; }
    estimate=float4(clamp(estimate.xyz,0.0f,frame.Numeric.w),1); pending[index]=lumenTemporalPack(estimate,normal,depth);
    if (frame.Jitter.w==0 || frame.Rules.x<=0) return;
    float4 motion=velocities.read(pixel); if (!all(isfinite(motion)) || motion.w<=0) return;
    float2 previous=float2(pixel)+0.5-motion.xy-frame.Jitter.xy;
    if (any(previous<0.5) || any(previous>float2(frame.Extent.xy)-0.5)) return;
    float expected=lumenTemporalDistance(frame,previous,1-motion.z); if (!isfinite(expected) || expected<0) return;
    int2 base=int2(floor(previous-0.5)); float2 fraction=fract(previous-0.5); float4 sum=0; float weights=0;
    for (uint tap=0;tap<4;++tap)
    {
        int2 corner=int2(tap&1,tap>>1),candidate=base+corner;
        uint2 bounded=uint2(clamp(candidate,int2(0),int2(frame.Extent.xy)-1));
        uint4 packed=committed[bounded.y*frame.Extent.x+bounded.x]; float4 old=lumenTemporalColor(packed); float oldDepth=as_type<float>(packed.w);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y);
        if (weight<=0 || any(candidate<0) || any(candidate>=int2(frame.Extent.xy)) || !all(isfinite(old)) || old.w<1 || old.w>frame.Clip.y ||
            !isfinite(oldDepth) || oldDepth<0 || oldDepth>=1 || dot(normal,lumenTemporalNormal(packed.z))<frame.Rules.z) continue;
        float optical=lumenTemporalDistance(frame,float2(candidate)+0.5,oldDepth);
        if (!isfinite(optical) || abs(optical-expected)>max(frame.Numeric.x,abs(expected)*frame.Rules.y)) continue;
        sum+=old*weight; weights+=weight;
    }
    if (weights<=LumenTemporalNormalFloor) return;
    float4 history=sum/weights; float3 minimum=estimate.xyz,maximum=minimum;
    for (int y=-LumenTemporalFilterRadius;y<=LumenTemporalFilterRadius;++y) for (int x=-LumenTemporalFilterRadius;x<=LumenTemporalFilterRadius;++x)
    {
        uint2 neighbor=uint2(clamp(int2(pixel)+int2(x,y),int2(0),int2(frame.Extent.xy)-1)); float4 value=reflected?specular[neighbor.y*frame.Extent.x+neighbor.x]:current.read(neighbor);
        float3 otherPosition,otherNormal; float otherDepth;
        if (!all(isfinite(value)) || value.w<=0 || !lumenTemporalSurface(frame,neighbor,depths,normals,otherPosition,otherNormal,otherDepth) || dot(normal,otherNormal)<frame.Rules.z) continue;
        float error=abs(dot(otherPosition-world,normal)); if (!isfinite(error) || error>frame.Rules.w) continue;
        minimum=min(minimum,value.xyz); maximum=max(maximum,value.xyz);
    }
    float3 center=(minimum+maximum)*0.5,extent=max((maximum-minimum)*0.5+abs(center)*frame.Clip.x,frame.Numeric.y);
    float3 delta=history.xyz-center,ratio=abs(delta/extent); history.xyz=center+delta/max(1.0f,max(ratio.x,max(ratio.y,ratio.z)));
    float count=min(history.w+1,frame.Clip.y),weight=min(frame.Rules.x,(count-1)/count);
    pending[index]=lumenTemporalPack(float4(clamp(mix(estimate.xyz,history.xyz,weight),0.0f,frame.Numeric.w),count),normal,depth);
}
kernel void lumenTemporalResolve(constant LumenTemporalFrame& frame [[buffer(0)]],device const uint4* committed [[buffer(1)]],
    device uint4* pending [[buffer(2)]],texture2d<float,access::read> current [[texture(0)]],depth2d<float,access::read> depths [[texture(1)]],
    texture2d<float,access::read> normals [[texture(2)]],texture2d<float,access::read> velocities [[texture(3)]],uint2 dispatch [[thread_position_in_grid]])
{ lumenAccumulateHistory(frame,committed,pending,current,depths,normals,velocities,dispatch,nullptr,false); }
kernel void lumenReflectionTemporal(constant LumenTemporalFrame& frame [[buffer(0)]],device const uint4* committed [[buffer(1)]],
    device uint4* pending [[buffer(2)]],device const float4* specular [[buffer(3)]],texture2d<float,access::read> current [[texture(0)]],
    depth2d<float,access::read> depths [[texture(1)]],texture2d<float,access::read> normals [[texture(2)]],
    texture2d<float,access::read> velocities [[texture(3)]],uint2 dispatch [[thread_position_in_grid]])
{ lumenAccumulateHistory(frame,committed,pending,current,depths,normals,velocities,dispatch,specular,true); }
vertex LumenGatherVertex lumenTemporalVertex(uint vertexID [[vertex_id]])
{
    float2 corner=float2((vertexID<<1)&2,vertexID&2); return {float4(corner*float2(2,-2)+float2(-1,1),0,1)};
}
fragment float4 lumenTemporalComposite(LumenGatherVertex input [[stage_in]],constant LumenTemporalFrame& frame [[buffer(0)]],
    device const uint4* history [[buffer(1)]],texture2d<float,access::read> sceneImage [[texture(0)]])
{
    uint2 pixel=uint2(input.Position.xy); float4 scene=sceneImage.read(pixel),indirect=lumenTemporalColor(history[pixel.y*frame.Extent.x+pixel.x]);
    float3 contribution=indirect.w>0?indirect.xyz*frame.Jitter.z:float3(0);
    return float4(clamp(scene.xyz+contribution,0.0f,frame.Numeric.w),scene.w);
}

// Native Metal specular reconstruction. Visible-normal sampling cancels the
// GGX distribution, leaving Fresnel times correlated Smith G2/G1(view).
// Unknown cache paths stay in the denominator; partial energy bypasses history.
struct LumenReflectionResolveFrame
{
    LumenReflectionRayFrame Rays;
    float4 Acceptance,Filter,Sky;
    uint4 Dispatch;
    float4 EnvironmentAxis,EnvironmentPeak;
};
struct LumenReflectionIncident { LumenDistanceHit Hit; float4 Radiance; uint4 Cache; };
constant uint LumenReflectionPBR=3,LumenReflectionMiss=0,LumenReflectionSurface=1,LumenReflectionScreen=3;
constant uint LumenReflectionComplete=1,LumenReflectionPartial=2;
constant float LumenReflectionPartialConfidence=0.5;
bool lumenReflectionSurface(constant LumenReflectionResolveFrame& frame,texture2d<float,access::read> depths,
    texture2d<float,access::read> normals,uint2 pixel,thread float3& position,thread float3& normal,thread float& roughness)
{
    float depth=depths.read(pixel).x; float4 material=normals.read(pixel); position=normal=0; roughness=material.w;
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || roughness<0 || roughness>=frame.Rays.Trace.z || dot(material.xyz,material.xyz)<=frame.Rays.Numeric.z) return false;
    normal=normalize(material.xyz); return lumenSpecularPosition(frame.Rays,pixel,depth,position);
}
float lumenReflectionLambda(float cosine,float alpha,float minimumCosine)
{
    float square=max(cosine*cosine,minimumCosine*minimumCosine);
    return (sqrt(1+alpha*alpha*max(0.0f,1-square)/square)-1)*0.5f;
}
kernel void lumenReflectionResolve(constant LumenReflectionResolveFrame& frame [[buffer(0)]],
    device const LumenDistanceRay* paths [[buffer(1)]],device const LumenReflectionIncident* lighting [[buffer(2)]],
    device float4* output [[buffer(3)]],texture2d<float,access::read> depths [[texture(0)]],
    texture2d<float,access::read> normals [[texture(1)]],texture2d<float,access::read> baseColors [[texture(2)]],
    texture2d<float,access::read> metalness [[texture(3)]],texture2d<float,access::read> specularColors [[texture(4)]],uint2 dispatch [[thread_position_in_grid]])
{
    uint index=dispatch.x+dispatch.y*frame.Dispatch.y; if (index>=frame.Dispatch.z) return;
    output[index]=0; uint2 pixel(index%frame.Rays.Viewport.x,index/frame.Rays.Viewport.x);
    float3 position,normal; float roughness; if (!lumenReflectionSurface(frame,depths,normals,pixel,position,normal,roughness)) return;
    float4 base=baseColors.read(pixel),metal=metalness.read(pixel),specular=specularColors.read(pixel);
    if (!all(isfinite(base)) || !all(isfinite(metal)) || !all(isfinite(specular))) return;
    float3 f0=frame.Dispatch.x==LumenReflectionPBR?mix(float3(frame.Filter.w),saturate(base.xyz),saturate(metal.w)):saturate(specular.xyz);
    int2 center=int2(pixel/frame.Rays.Viewport.z); int radius=int(frame.Filter.x); float3 result=0; float weights=0; bool complete=true;
    for (int row=-radius;row<=radius;++row) for (int column=-radius;column<=radius;++column)
    {
        int2 cell=center+int2(column,row); if (any(cell<0) || cell.x>=int(frame.Rays.Viewport.w) || cell.y>=int(frame.Rays.Work.x)) continue;
        uint rayIndex=uint(cell.y)*frame.Rays.Viewport.w+uint(cell.x); LumenDistanceRay path=paths[rayIndex];
        if (!all(isfinite(path.DirectionMaximum)) || dot(path.DirectionMaximum.xyz,path.DirectionMaximum.xyz)<=frame.Rays.Numeric.z) continue;
        uint2 source=min(uint2(cell)*frame.Rays.Viewport.z+frame.Rays.Viewport.z/2,frame.Rays.Viewport.xy-1);
        float3 p,n; float r;
        if (!lumenReflectionSurface(frame,depths,normals,source,p,n,r) || dot(n,normal)<frame.Acceptance.x || abs(r-roughness)>frame.Acceptance.w) continue;
        float3 offset=p-position; float plane=max(abs(dot(offset,n)),abs(dot(offset,normal)));
        if (!isfinite(plane) || plane>frame.Acceptance.y+frame.Acceptance.z*length(offset)) continue;
        float2 distance=(float2(source)-float2(pixel))/float(frame.Rays.Viewport.z); float weight=1/(1+dot(distance,distance)); weights+=weight;
        if (path.DirectionMaximum.w<=0) continue;
        float3 nearPosition,midPosition;
        if (!lumenSpecularPosition(frame.Rays,source,0,nearPosition) || !lumenSpecularPosition(frame.Rays,source,0.5f,midPosition)) { complete=false; continue; }
        float3 eye=normalize(nearPosition-midPosition),outgoing=path.DirectionMaximum.xyz; float noV=dot(n,eye),noL=dot(n,outgoing);
        if (!all(isfinite(eye)) || noV<=0 || noL<=0) { complete=false; continue; }
        LumenReflectionIncident incoming=lighting[rayIndex]; float3 radiance=0;
        if (incoming.Hit.Metadata.z==LumenReflectionMiss) radiance=lumenEnvironmentRadiance(frame.Sky.xyz,frame.EnvironmentAxis,frame.EnvironmentPeak.xyz,outgoing);
        else if ((incoming.Hit.Metadata.z==LumenReflectionSurface || incoming.Hit.Metadata.z==LumenReflectionScreen) &&
            (incoming.Cache.z==LumenReflectionComplete || incoming.Cache.z==LumenReflectionPartial))
        { radiance=incoming.Radiance.xyz; complete=complete && incoming.Cache.z==LumenReflectionComplete; }
        else complete=false;
        radiance=radiance*incoming.Hit.Medium.w+incoming.Hit.Medium.xyz;
        if (!all(isfinite(radiance)) || any(radiance<0)) { complete=false; continue; }
        float alpha=max(r*r,frame.Rays.Trace.w),lambdaV=lumenReflectionLambda(noV,alpha,frame.Filter.y),lambdaL=lumenReflectionLambda(noL,alpha,frame.Filter.y);
        float smith=r==0?1:(1+lambdaV)/(1+lambdaV+lambdaL);
        float complement=1-saturate(dot(eye,normalize(eye+outgoing))),square=complement*complement;
        result+=radiance*(f0+(1-f0)*square*square*complement)*smith*weight;
    }
    if (weights>0 && all(isfinite(result))) output[index]=float4(clamp(result/weights,0.0f,frame.Filter.z),complete?1.0f:LumenReflectionPartialConfidence);
}
