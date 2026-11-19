#include <metal_stdlib>
using namespace metal;
// Independent Metal camera paths. Compile without fast math: traversal keeps
// conservative FP32 edge predicates, and accumulation keeps the FP32 mean.
// Diffuse/GGX mixture continuation and explicit next-event light sampling:
// https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
// Heitz GGX VNDF (2018): https://jcgt.org/published/0007/04/01/
// Ray-aligned predicates: https://jcgt.org/published/0002/01/05/
constant uint Lambertian=1,FlatFlag=1,CasterFlag=1,DirectionalLight=0,PointLight=1;
constant float Pi=3.141592653589793,PixelCenter=0.5,InteriorDepth=0.5,EmptyDepth=1;
constant float MinimumW=1e-7,MinimumSquared=1e-16,TangentEpsilon=1e-8,ParallelAxis=0.999;
constant float FloatEpsilon=1.1920928955078125e-7,EdgeError=3*FloatEpsilon/(1-3*FloatEpsilon);
constant float MinimumAlpha=0.0025,MinimumCosine=1e-6,DielectricF0=0.04,FresnelPower=5;
constant float LightDistanceFloor=0.01,LobeProbabilityFloor=0.05;
constant float RouletteFloor=0.05,RouletteCeiling=0.95,HalfMaximum=65504,HalfSpacing=32;
constant uint RouletteStartDepth=3;
// PCG RXS-M-XS/LCG constants specify the integer permutation and recurrence.
// Jarzynski and Olano (2020): https://jcgt.org/published/0009/03/02/
constant uint PcgMultiplier=747796405u,PcgIncrement=2891336453u,PcgOutputMultiplier=277803737u;
constant uint PcgDynamicShift=28,PcgMinimumShift=4,PcgOutputShift=22,FloatDiscardBits=8;
constant float UnitIntervalScale=1.0/16777216.0;
struct PathFrame
{
    float4 InverseViewProjection[4],ViewProjection[4];
    float4 Viewport,Trace,Background,Material;
    uint4 SceneCounts,ShadingCounts,Sampling,Options;
};
struct BvhNode { float4 Minimum,Maximum; uint4 Links; };
struct Triangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct CoverageMaterial { uint4 Diffuse,Opacity; float4 Coverage; };
struct Surface { float4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct MaterialRecord { float4 Base,AmbientColor,Specular,Emission; uint4 Diffuse,Normal; };
struct LightRecord { float4 PositionType,DirectionIntensity,ColorShadow,Area; };
struct RayContext
{
    constant PathFrame* Frame;
    device const BvhNode* Nodes;
    device const Triangle* Triangles;
    device const CoverageMaterial* Coverage;
    device const float* AlphaTexels;
    device const Surface* Surfaces;
    device const MaterialRecord* Materials;
    device const float4* ColorTexels;
    device const LightRecord* Lights;
};
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex VSMain(uint index [[vertex_id]])
{ float2 corner=float2((index<<1)&2,index&2); return {float4(corner*2-1,0,1)}; }
float3 Unit(float3 value,float3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (!(scale>0) || !isfinite(scale)) return fallback;
    value/=scale; return value*rsqrt(dot(value,value));
}
void TexelAddresses(uint4 image,float2 uv,thread uint4& address,thread float2& blend)
{
    float2 position=fract(float2(uv.x,-uv.y))*float2(image.yz)-PixelCenter;
    uint x=position.x<0 ? image.y-1 : uint(floor(position.x)),y=position.y<0 ? image.z-1 : uint(floor(position.y));
    uint nextX=x+1==image.y ? 0 : x+1,nextY=y+1==image.z ? 0 : y+1;
    address=image.x+uint4(y*image.y+x,y*image.y+nextX,nextY*image.y+x,nextY*image.y+nextX); blend=fract(position);
}
float SampleAlpha(uint4 image,float2 uv,thread const RayContext& context)
{
    if (!image.w) return 1; uint4 a; float2 t; TexelAddresses(image,uv,a,t);
    return mix(mix(context.AlphaTexels[a.x],context.AlphaTexels[a.y],t.x),mix(context.AlphaTexels[a.z],context.AlphaTexels[a.w],t.x),t.y);
}
float3 SampleColor(uint4 image,float2 uv,thread const RayContext& context)
{
    if (!image.w) return float3(1); uint4 a; float2 t; TexelAddresses(image,uv,a,t);
    return mix(mix(context.ColorTexels[a.x].rgb,context.ColorTexels[a.y].rgb,t.x),mix(context.ColorTexels[a.z].rgb,context.ColorTexels[a.w].rgb,t.x),t.y);
}
bool IntersectBounds(BvhNode node,float3 origin,float3 direction,float maximum)
{
    float enter=0,leave=maximum;
    for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(float2 a,float2 b)
{ float forward=a.x*b.y,backward=a.y*b.x,edge=forward-backward; return abs(edge)<=EdgeError*(abs(forward)+abs(backward)) ? 0 : edge; }
bool IntersectTriangle(Triangle triangle,float3 origin,float3 direction,float maximum,thread float& distance,thread float3& weights)
{
    distance=0; weights=0;
    uint z=abs(direction.x)>abs(direction.y) ? 0 : 1; if (abs(direction.z)>abs(direction[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3; float3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    float2 shear=float2(direction[x],direction[y])/direction[z];
    float2 pa=float2(a[x],a[y])-a[z]*shear,pb=float2(b[x],b[y])-b[z]*shear,pc=float2(c[x],c[y])-c[z]*shear;
    float3 edges=float3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(edges<0) && any(edges>0)) return false;
    float determinant=edges.x+edges.y+edges.z; if (determinant==0) return false;
    distance=dot(edges,float3(a[z],b[z],c[z]))/(determinant*direction[z]);
    if (!(distance>0 && distance<=maximum)) return false; weights=edges/determinant; return true;
}
struct Hit { uint Index; float Distance; float3 Weights; };
bool TraceScene(float3 origin,float3 direction,float maximum,bool shadow,thread const RayContext& context,thread Hit& hit)
{
    hit={0,maximum,float3(0)}; uint cursor=0; bool found=false;
    while (cursor<context.Frame->SceneCounts.x)
    {
        BvhNode node=context.Nodes[cursor];
        if (!IntersectBounds(node,origin,direction,hit.Distance)) { cursor=node.Links.w; continue; }
        if (!node.Links.z) { cursor=node.Links.x; continue; }
        for (uint offset=0;offset<node.Links.z;++offset)
        {
            uint index=node.Links.y+offset; Triangle triangle=context.Triangles[index];
            if (shadow && !(triangle.Metadata.w&CasterFlag)) continue;
            float distance; float3 weights;
            if (!IntersectTriangle(triangle,origin,direction,hit.Distance,distance,weights)) continue;
            CoverageMaterial material=context.Coverage[triangle.Metadata.x];
            if (shadow || material.Coverage.y>0.0)
            {
                float2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float alpha=saturate(dot(weights,float3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x*
                    SampleAlpha(material.Diffuse,uv,context)*SampleAlpha(material.Opacity,uv,context));
                if (alpha<material.Coverage.y || (shadow && alpha<=0.0)) continue;
            }
            hit={index,distance,weights}; found=true; if (shadow) return true;
        }
        cursor=node.Links.w;
    }
    return found;
}
struct HitSurface { float3 Position,Normal,Geometric,Diffuse,Specular,AmbientColor,Emission; float Shininess,Opacity,IOR; };
HitSurface ReadHit(Hit hit,float3 origin,float3 direction,thread const RayContext& context)
{
    Triangle triangle=context.Triangles[hit.Index]; Surface surface=context.Surfaces[hit.Index];
    MaterialRecord material=context.Materials[triangle.Metadata.x]; float3 w=hit.Weights;
    float3 ab=triangle.B.xyz-triangle.A.xyz,ac=triangle.C.xyz-triangle.A.xyz;
    float2 uv=triangle.UVAB.xy*w.x+triangle.UVAB.zw*w.y+triangle.UVC.xy*w.z;
    HitSurface result; result.Position=origin+direction*hit.Distance;
    result.Geometric=Unit(cross(Unit(ab,float3(1,0,0)),Unit(ac,float3(0,1,0))),float3(0,0,1));
    float3 normal=(context.Frame->Options.y&FlatFlag) ? result.Geometric : Unit(surface.NormalA.xyz*w.x+surface.NormalB.xyz*w.y+surface.NormalC.xyz*w.z,result.Geometric);
    if (dot(result.Geometric,direction)>0) normal=-normal;
    if (material.Normal.w)
    {
        float2 uvB=float2(triangle.UVAB.z-triangle.UVAB.x,triangle.UVAB.y-triangle.UVAB.w);
        float2 uvC=float2(triangle.UVC.x-triangle.UVAB.x,triangle.UVAB.y-triangle.UVC.y);
        float determinant=uvB.x*uvC.y-uvB.y*uvC.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            float3 tangent=(ab*uvC.y-ac*uvB.y)/determinant,sourceBitangent=(ac*uvB.x-ab*uvC.x)/determinant;
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            tangent=Unit(tangent-normal*dot(normal,tangent),Unit(cross(helper,normal),float3(1,0,0)));
            float3 bitangent=cross(normal,tangent); bitangent*=dot(bitangent,sourceBitangent)<0 ? -1.0f : 1.0f;
            float3 mapped=SampleColor(material.Normal,uv,context)*2-1;
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=max((surface.ColorA.rgb*w.x+surface.ColorB.rgb*w.y+surface.ColorC.rgb*w.z)*
        material.Base.rgb*SampleColor(material.Diffuse,uv,context),float3(0));
    result.Specular=material.Specular.rgb; result.Shininess=material.Specular.w;
    result.AmbientColor=material.AmbientColor.rgb; result.Emission=material.Emission.rgb;
    result.Opacity=clamp(material.Base.w*(triangle.A.w*w.x+triangle.B.w*w.y+triangle.C.w*w.z),0.0,1.0); result.IOR=material.Emission.w;
    return result;
}

uint Hash(uint value)
{
    uint state=value*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return (word>>PcgOutputShift)^word;
}
float Random(thread uint& state)
{
    state=state*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return float(((word>>PcgOutputShift)^word)>>FloatDiscardBits)*UnitIntervalScale;
}
float2 Random2(thread uint& state) { float x=Random(state),y=Random(state); return float2(x,y); }
float3 Fresnel(float3 f0,float cosine)
{ return f0+(1-f0)*pow(1-saturate(cosine),FresnelPower); }
float Smith(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),MinimumCosine); }
float Distribution(float nh,float alpha)
{ float a2=alpha*alpha,d=nh*nh*(a2-1)+1; return a2/(Pi*d*d); }
float SpecularProbability(float3 albedo,constant PathFrame& frame)
{
    if (frame.Options.x==Lambertian) return 0;
    float3 f0=mix(float3(DielectricF0),albedo,frame.Material.y),kd=(1-f0)*(1-frame.Material.y)*albedo;
    float s=max(f0.x,max(f0.y,f0.z)),d=max(kd.x,max(kd.y,kd.z));
    return d<=0 ? 1 : clamp(s/(s+d),LobeProbabilityFloor,1-LobeProbabilityFloor);
}
float3 EvaluateBSDF(float3 albedo,float3 normal,float3 view,float3 incoming,constant PathFrame& frame,thread float& pdf)
{
    pdf=0; float nv=dot(normal,view),nl=dot(normal,incoming);
    if (nv<=0 || nl<=0) return float3(0);
    if (frame.Options.x==Lambertian) { pdf=nl/Pi; return albedo*pdf; }
    float3 h=Unit(view+incoming,normal); float alpha=max(frame.Material.x*frame.Material.x,MinimumAlpha);
    float distribution=Distribution(saturate(dot(normal,h)),alpha),mask=Smith(nv,alpha);
    pdf=mix(nl/Pi,distribution*mask/max(4*nv,MinimumCosine),SpecularProbability(albedo,frame));
    float3 f=Fresnel(mix(float3(DielectricF0),albedo,frame.Material.y),dot(view,h));
    return (1-f)*(1-frame.Material.y)*albedo*(nl/Pi)+f*distribution*mask*Smith(nl,alpha)/max(4*nv,MinimumCosine);
}
void Basis(float3 normal,thread float3& tangent,thread float3& bitangent)
{
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    tangent=Unit(cross(helper,normal),float3(1,0,0)); bitangent=cross(normal,tangent);
}
float3 VisibleNormal(float3 view,float alpha,float2 xi)
{
    float3 stretched=Unit(float3(view.xy*alpha,view.z),float3(0,0,1));
    float len=dot(stretched.xy,stretched.xy);
    float3 t=len>0 ? float3(-stretched.y,stretched.x,0)/sqrt(len) : float3(1,0,0),b=cross(stretched,t);
    float radius=sqrt(xi.x),angle=2*Pi*xi.y,x=radius*cos(angle),y=radius*sin(angle);
    y=mix(sqrt(max(0.0f,1-x*x)),y,(1+stretched.z)/2);
    float3 projected=t*x+b*y+stretched*sqrt(max(0.0f,1-x*x-y*y));
    return Unit(float3(projected.xy*alpha,max(projected.z,0.0f)),float3(0,0,1));
}
bool Scatter(HitSurface surface,float3 view,constant PathFrame& frame,thread uint& state,thread float3& incoming,thread float3& weight)
{
    float3 albedo=saturate(surface.Diffuse),tangent,bitangent; Basis(surface.Normal,tangent,bitangent);
    float choice=Random(state); float2 xi=Random2(state);
    if (choice<SpecularProbability(albedo,frame))
    {
        float3 localView=float3(dot(view,tangent),dot(view,bitangent),dot(view,surface.Normal));
        float3 h=VisibleNormal(localView,max(frame.Material.x*frame.Material.x,MinimumAlpha),xi);
        incoming=Unit(reflect(-view,tangent*h.x+bitangent*h.y+surface.Normal*h.z),surface.Normal);
    }
    else
    {
        float radius=sqrt(xi.x),angle=2*Pi*xi.y;
        incoming=tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+surface.Normal*sqrt(1-xi.x);
    }
    float pdf; float3 value=EvaluateBSDF(albedo,surface.Normal,view,incoming,frame,pdf); weight=float3(0);
    if (!(pdf>0) || dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0) return false;
    weight=value/pdf; return all(isfinite(weight)) && any(weight>0);
}
float3 OffsetOrigin(HitSurface surface,float3 direction,constant PathFrame& frame)
{ return surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0 ? frame.Trace.y : -frame.Trace.y); }
float3 DirectLighting(HitSurface surface,float3 view,thread const RayContext& context,thread uint& state)
{
    constant PathFrame& frame=*context.Frame; float3 sum(0);
    for (uint index=0;index<frame.ShadingCounts.w;++index)
    {
        LightRecord light=context.Lights[index]; uint type=uint(light.PositionType.w);
        bool distant=type==DirectionalLight,rectangle=type!=DirectionalLight && type!=PointLight;
        float3 source=light.PositionType.xyz,normal=Unit(light.DirectionIntensity.xyz,float3(0,-1,0)); float measure=1;
        if (rectangle)
        {
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Unit(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
            float2 uv=Random2(state)-PixelCenter;
            source+=tangent*(uv.x*light.Area.x)+bitangent*(uv.y*light.Area.y); measure=light.Area.w>0 ? abs(light.Area.x*light.Area.y) : 1;
        }
        float3 delta=distant ? Unit(light.DirectionIntensity.xyz,float3(0,1,0))*frame.Trace.z : source-surface.Position;
        float squared=dot(delta,delta); if (!(squared>MinimumSquared)) continue;
        float distance=sqrt(squared); float3 incoming=delta/distance;
        if (dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0) continue;
        float attenuation=1;
        if (!distant)
        {
            float window=1;
            if (light.Area.z>0) { float ratio=distance/(light.Area.z>0 ? light.Area.z : 1.0); window=saturate(1-ratio*ratio*ratio*ratio); window*=window; }
            attenuation=window/max(squared,LightDistanceFloor);
        }
        if (rectangle) attenuation*=measure*max(dot(normal,-incoming),0.0f);
        if (!(attenuation>0)) continue;
        Hit blocker;
        if (frame.Options.z!=0 && light.ColorShadow.w>0 && TraceScene(OffsetOrigin(surface,incoming,frame),incoming,max(distance-frame.Trace.y,0.0f),true,context,blocker)) continue;
        float pdf; sum+=EvaluateBSDF(saturate(surface.Diffuse),surface.Normal,view,incoming,frame,pdf)*
            max(light.ColorShadow.rgb,float3(0))*max(light.DirectionIntensity.w,0.0f)*attenuation;
    }
    return sum;
}
float4 Transform(constant float4* rows,float4 vector)
{ return float4(dot(rows[0],vector),dot(rows[1],vector),dot(rows[2],vector),dot(rows[3],vector)); }
bool Unproject(float2 uv,float depth,constant PathFrame& frame,thread float3& point)
{
    float4 h=Transform(frame.InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1)); point=float3(0);
    if (!(abs(h.w)>MinimumW)) return false;
    point=h.xyz/h.w; return all(isfinite(point));
}
bool PrimaryRay(float2 pixel,constant PathFrame& frame,thread float3& origin,thread float3& direction,thread float& maximum)
{
    float2 uv=pixel*frame.Viewport.zw; float3 middle,farPoint; direction=float3(0); maximum=frame.Trace.x;
    if (!Unproject(uv,0,frame,origin) || !Unproject(uv,InteriorDepth,frame,middle)) return false;
    direction=Unit(middle-origin,float3(0)); if (!(dot(direction,direction)>0)) return false;
    if (Unproject(uv,1,frame,farPoint)) maximum=min(maximum,length(farPoint-origin));
    return maximum>0;
}
float4 PathSample(float2 pixel,thread const RayContext& context,thread uint& state)
{
    constant PathFrame& frame=*context.Frame; float3 origin,direction; float maximum;
    if (!PrimaryRay(pixel,frame,origin,direction,maximum)) return frame.Background;
    float3 radiance(0),throughput(1); float coverage=1;
    for (uint depth=0;depth<frame.Sampling.z;++depth)
    {
        Hit hit;
        if (!TraceScene(origin,direction,maximum,false,context,hit))
        { radiance+=throughput*frame.Background.rgb; if (!depth) coverage=frame.Background.a; break; }
        HitSurface surface=ReadHit(hit,origin,direction,context); float3 view=-direction;
        if (dot(surface.Normal,view)<=0) surface.Normal=dot(surface.Geometric,view)>0 ? surface.Geometric : -surface.Geometric;
        // Fresnel branching traces both dielectric reflection and transmission.
        // Thin opacity uses stochastic coverage, preserving the unweighted expectation.
        if (surface.IOR>1.0)
        {
            bool entering=dot(direction,surface.Geometric)<0.0;
            float3 face=entering?surface.Geometric:-surface.Geometric;
            float eta=entering?1.0/surface.IOR:surface.IOR;
            float f0=(surface.IOR-1.0)/(surface.IOR+1.0);f0*=f0;
            float cosine=clamp(-dot(direction,face),0.0,1.0);
            float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
            float3 transmitted=refract(direction,face,eta);
            if (dot(transmitted,transmitted)<0.000001 || Random(state)<fresnel) direction=reflect(direction,face);
            else { direction=transmitted;throughput*=clamp(surface.Diffuse,0.0,1.0)*(eta*eta); }
            origin=OffsetOrigin(surface,direction,frame);maximum=frame.Trace.x;continue;
        }
        if (surface.Opacity<1.0 && Random(state)>surface.Opacity)
        { origin=OffsetOrigin(surface,direction,frame);maximum=frame.Trace.x;continue; }
        radiance+=throughput*(surface.Emission+DirectLighting(surface,view,context,state));
        if (depth+1==frame.Sampling.z) break;
        float3 weight; if (!Scatter(surface,view,frame,state,direction,weight)) break;
        throughput*=weight;
        if (depth+1>=RouletteStartDepth)
        {
            float survival=clamp(max(throughput.x,max(throughput.y,throughput.z)),RouletteFloor,RouletteCeiling);
            if (Random(state)>=survival) break;
            throughput/=survival;
        }
        origin=OffsetOrigin(surface,direction,frame); maximum=frame.Trace.x;
    }
    return float4(radiance,coverage);
}
struct PathOutput { float4 Mean [[color(0)]]; float Depth [[color(1)]]; };
fragment PathOutput PSMain(ScreenVertex input [[stage_in]],constant PathFrame& frame [[buffer(0)]],
    device const BvhNode* nodes [[buffer(1)]],device const Triangle* triangles [[buffer(2)]],
    device const CoverageMaterial* coverage [[buffer(3)]],device const float* alphaTexels [[buffer(4)]],
    device const Surface* surfaces [[buffer(5)]],device const MaterialRecord* materials [[buffer(6)]],
    device const float4* colorTexels [[buffer(7)]],device const LightRecord* lights [[buffer(8)]],
    texture2d<float,access::read> previous [[texture(0)]])
{
    uint2 pixel=uint2(input.Position.xy); PathOutput output;
    RayContext context {&frame,nodes,triangles,coverage,alphaTexels,surfaces,materials,colorTexels,lights};
    output.Mean=frame.Sampling.x ? previous.read(pixel) : float4(0);
    for (uint path=0;path<frame.Sampling.y;++path)
    {
        uint sequence=frame.Sampling.x+path,state=Hash(pixel.x^Hash(pixel.y^Hash(sequence^frame.Sampling.w)));
        float2 jitter=Random2(state); float4 value=PathSample(float2(pixel)+jitter,context,state);
        output.Mean+=(value-output.Mean)/float(sequence+1);
    }
    float3 origin,direction; float maximum; Hit center; output.Depth=EmptyDepth;
    if (PrimaryRay(float2(pixel)+PixelCenter,frame,origin,direction,maximum) && TraceScene(origin,direction,maximum,false,context,center))
    {
        float4 clip=Transform(frame.ViewProjection,float4(origin+direction*center.Distance,1));
        if (abs(clip.w)>MinimumW) output.Depth=saturate(clip.z/clip.w);
    }
    return output;
}
struct CompositeOutput { float4 Color [[color(0)]]; float Depth [[depth(any)]]; };
fragment CompositeOutput PSCompose(ScreenVertex input [[stage_in]],texture2d<float,access::read> mean [[texture(0)]],
    texture2d<float,access::read> centerDepth [[texture(1)]])
{
    uint2 pixel=uint2(input.Position.xy); float4 value=mean.read(pixel); CompositeOutput output;
    output.Color=float4(clamp(value.rgb,float3(0),float3(HalfMaximum-HalfSpacing)),saturate(value.a));
    output.Depth=centerDepth.read(pixel).r; return output;
}
