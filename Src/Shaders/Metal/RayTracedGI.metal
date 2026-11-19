#include <metal_stdlib>
using namespace metal;
// Metal diffuse GI: scene-material evaluation, cosine-sampled paths, complete
// emitter lighting, then depth/normal/albedo bilateral reconstruction. Compile
// without fast math so conservative scene-edge predicates retain their meaning.
// PBRT 4e: https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection
// https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Simple_Path_Tracer
// Ray-aligned triangle edges: https://jcgt.org/published/0002/01/05/ .
// Ray hits retain the Toon palette, shallow normals and colored light ramp.
constant float ToonShadingModel = 4.0;
constant float ToonAlbedoExponent = 0.8;
constant float ToonPaletteSteps = 16.0;
constant float ToonColorSaturation = 1.12;
constant float ToonNormalStrength = 0.12;
constant float ToonLightScale = 0.45;
constant float ToonBandSoftness = 0.4;
constant float ToonShadowFloor = 0.18;
constant float ToonHighlightThreshold = 0.8;
constant float ToonHighlightStrength = 0.18;
constant float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
constant float3 ToonCoolShadow = float3(0.56, 0.68, 1.0);
constant float3 ToonWarmLight = float3(1.0, 0.94, 0.82);
float3 ToonAlbedo(float3 value, float model)
{
    if (model != ToonShadingModel) return value;
    float3 paint = pow(saturate(value), float3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(mix(float3(dot(paint, ToonLuma)), paint, ToonColorSaturation));
}
float3 ToonResponse(float cosine, float bands)
{
    float angularBand = saturate(cosine) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, fract(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return mix(ToonCoolShadow, ToonWarmLight, level) * mix(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

constant uint TraceStage=0,HorizontalStage=1,VerticalStage=2,CompositeStage=3;
constant uint PBRModel=3,ToonModel=4,FlatFlag=1,CasterFlag=1,DirectionalLight=0,PointLight=1,LegacyGrid=2;
constant float Pi=3.141592653589793,GoldenStep=(3-sqrt(5.0f))/2;
constant float PixelCenter=0.5,EmptyDepth=1,MinimumW=1e-7,MinimumSquared=1e-16;
constant float FloatEpsilon=1.1920928955078125e-7,EdgeError=3*FloatEpsilon/(1-3*FloatEpsilon);
constant float TangentEpsilon=1e-8,ParallelAxis=0.999,MinimumAlpha=0.0025,MinimumCosine=1e-6;
constant float DielectricF0=0.04,FresnelPower=5,LightDistanceFloor=0.01;
constant float ReflectanceSigma=0.05;
constant float HalfMaximum=65504,HalfSpacing=32,SpatialSigma=1.5,NormalPower=32;
constant float BiasSigmaScale=4,FootprintSigmaScale=2,MinimumDepthSigma=1e-5,MinimumWeight=1e-7;
constant int FilterRadius=2;
struct DiffuseGIFrame
{
    float4 InverseViewProjection[4];
    float4 Viewport,Trace,CameraForward,Reserved,Material;
    uint4 SceneCounts,ShadingCounts,Options,Lighting;
};
struct BvhNode { float4 Minimum,Maximum; uint4 Links; };
struct Triangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct CoverageMaterial { uint4 Diffuse,Opacity; float4 Coverage; };
struct Surface { float4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct MaterialRecord { float4 Base,AmbientColor,Specular,Emission; uint4 Diffuse,Normal; };
struct LightRecord { float4 PositionType,DirectionIntensity,ColorShadow,Area; };
struct RayContext
{
    constant DiffuseGIFrame* Frame;
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
            if (material.Coverage.y>0)
            {
                float2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float alpha=saturate(dot(weights,float3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x*
                    SampleAlpha(material.Diffuse,uv,context)*SampleAlpha(material.Opacity,uv,context));
                if (alpha<material.Coverage.y) continue;
            }
            hit={index,distance,weights}; found=true; if (shadow) return true;
        }
        cursor=node.Links.w;
    }
    return found;
}
struct HitSurface { float3 Position,Normal,Geometric,Diffuse,Specular,AmbientColor,Emission; float Shininess; };
HitSurface ReadHit(Hit hit,float3 origin,float3 direction,thread const RayContext& context)
{
    Triangle triangle=context.Triangles[hit.Index]; Surface surface=context.Surfaces[hit.Index];
    MaterialRecord material=context.Materials[triangle.Metadata.x]; float3 w=hit.Weights;
    float3 ab=triangle.B.xyz-triangle.A.xyz,ac=triangle.C.xyz-triangle.A.xyz;
    float2 uv=triangle.UVAB.xy*w.x+triangle.UVAB.zw*w.y+triangle.UVC.xy*w.z;
    HitSurface result; result.Position=origin+direction*hit.Distance;
    result.Geometric=Unit(cross(Unit(ab,float3(1,0,0)),Unit(ac,float3(0,1,0))),float3(0,0,1));
    float3 normal=(context.Frame->Options.w&FlatFlag) ? result.Geometric : Unit(surface.NormalA.xyz*w.x+surface.NormalB.xyz*w.y+surface.NormalC.xyz*w.z,result.Geometric);
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
            if (float(context.Frame->Options.z) == ToonShadingModel) mapped = mix(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=ToonAlbedo(max((surface.ColorA.rgb*w.x+surface.ColorB.rgb*w.y+surface.ColorC.rgb*w.z)*
        material.Base.rgb*SampleColor(material.Diffuse,uv,context),float3(0)), float(context.Frame->Options.z));
    result.Specular=material.Specular.rgb; result.Shininess=material.Specular.w;
    result.AmbientColor=material.AmbientColor.rgb; result.Emission=material.Emission.rgb; return result;
}
float3 Fresnel(float3 f0,float cosine) { return f0+(1-f0)*pow(1-saturate(cosine),FresnelPower); }
float SmithG1(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),MinimumCosine); }
float3 BrdfCosine(HitSurface surface,float3 view,float3 incoming,constant DiffuseGIFrame& frame)
{
    float nl=dot(surface.Normal,incoming),nv=dot(surface.Normal,view); if (nl<=0 || nv<=0) return float3(0);
    float3 halfVector=Unit(view+incoming,surface.Normal); float nh=saturate(dot(surface.Normal,halfVector));
    if (frame.Options.z!=PBRModel)
    {
        float highlight=pow(nh,surface.Shininess);
        if (frame.Options.z==ToonModel) return surface.Diffuse * ToonResponse(max(nl, 0.0), frame.Material.z) + surface.Specular * ToonHighlight(highlight);
        return surface.Diffuse*nl+surface.Specular*highlight;
    }
    float alpha=max(frame.Material.x*frame.Material.x,MinimumAlpha),a2=alpha*alpha;
    float denominator=nh*nh*(a2-1)+1,distribution=a2/(Pi*denominator*denominator);
    float3 f=Fresnel(mix(float3(DielectricF0),saturate(surface.Diffuse),frame.Material.y),dot(view,halfVector));
    float3 specular=f*distribution*SmithG1(nv,alpha)*SmithG1(nl,alpha)/max(4*nv*nl,MinimumCosine);
    return ((1-f)*(1-frame.Material.y)*surface.Diffuse/Pi+specular)*nl;
}
float3 EmitterSample(HitSurface surface,float3 view,float3 emitter,float3 energy,float3 emitterNormal,float measure,
    bool distant,bool rectangle,bool castsShadow,float range,thread const RayContext& context)
{
    constant DiffuseGIFrame& frame=*context.Frame;
    float3 delta=distant ? emitter*frame.CameraForward.w : emitter-surface.Position;
    float squared=dot(delta,delta); if (!(squared>MinimumSquared)) return float3(0);
    float distance=sqrt(squared); float3 incoming=delta/distance; float attenuation=1;
    if (!distant)
    {
        float window=1;
        if (range>0) { float relative=distance/range; window=saturate(1-relative*relative*relative*relative); window*=window; }
        attenuation=measure*window*(rectangle ? max(dot(emitterNormal,-incoming),0.0f) : 1)/max(squared,LightDistanceFloor);
    }
    float3 value=BrdfCosine(surface,view,incoming,frame)*energy*attenuation; if (!any(value>0)) return float3(0);
    if (frame.Lighting.x && castsShadow)
    {
        float3 origin=surface.Position+surface.Geometric*(dot(surface.Geometric,incoming)>=0 ? frame.Trace.y : -frame.Trace.y);
        float3 segment=surface.Position+delta-origin; float extent=length(segment); Hit blocker;
        if (extent>frame.Trace.y && TraceScene(origin,segment/extent,extent-frame.Trace.y,true,context,blocker)) return float3(0);
    }
    return value;
}
float3 Shade(HitSurface surface,float3 view,thread const RayContext& context)
{
    constant DiffuseGIFrame& frame=*context.Frame;
    float3 result=surface.Emission;
    for (uint index=0;index<frame.ShadingCounts.w;++index)
    {
        LightRecord light=context.Lights[index]; uint type=uint(light.PositionType.w); bool shadows=light.ColorShadow.w>0;
        float3 energy=max(light.ColorShadow.rgb,float3(0))*max(light.DirectionIntensity.w,0.0f); if (!any(energy>0)) continue;
        if (type==DirectionalLight)
            result+=EmitterSample(surface,view,Unit(light.DirectionIntensity.xyz,float3(0,1,0)),energy,float3(0),1,true,false,shadows,0,context);
        else if (type==PointLight)
            result+=EmitterSample(surface,view,light.PositionType.xyz,energy,float3(0),1,false,false,shadows,light.Area.z,context);
        else
        {
            float3 normal=Unit(light.DirectionIntensity.xyz,float3(0,-1,0));
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Unit(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
            uint grid=light.Area.w>0 ? uint(light.Area.w) : LegacyGrid;
            float area=light.Area.w>0 ? abs(light.Area.x*light.Area.y) : 1;
            for (uint y=0;y<grid;++y) for (uint x=0;x<grid;++x)
            {
                float2 coordinate=(float2(x,y)+PixelCenter)/float(grid)-PixelCenter;
                float3 emitter=light.PositionType.xyz+tangent*(coordinate.x*light.Area.x)+bitangent*(coordinate.y*light.Area.y);
                result+=EmitterSample(surface,view,emitter,energy,normal,area/float(grid*grid),false,true,shadows,light.Area.z,context);
            }
        }
    }
    return clamp(result,0.0f,HalfMaximum-HalfSpacing);
}
bool Unproject(float2 uv,float depth,constant DiffuseGIFrame& frame,thread float3& point)
{
    float4 clip=float4(uv.x*2-1,1-uv.y*2,depth,1);
    float4 world=float4(dot(frame.InverseViewProjection[0],clip),dot(frame.InverseViewProjection[1],clip),
        dot(frame.InverseViewProjection[2],clip),dot(frame.InverseViewProjection[3],clip));
    point=0; if (!(abs(world.w)>MinimumW)) return false; point=world.xyz/world.w; return all(isfinite(point));
}
bool Receiver(uint2 pixel,constant DiffuseGIFrame& frame,texture2d<float,access::read> depth,texture2d<float,access::read> normals,
    thread float3& point,thread float3& normal,thread float& roughness)
{
    point=normal=0; float4 value=normals.read(pixel); roughness=value.w; float squared=dot(value.xyz,value.xyz);
    float z=depth.read(pixel).r;
    if (!(z<EmptyDepth && roughness>=0 && squared>MinimumSquared && isfinite(squared))) return false;
    normal=value.xyz*rsqrt(squared); return Unproject((float2(pixel)+PixelCenter)*frame.Viewport.zw,z,frame,point);
}
float2 PathSample(uint2 pixel,uint path,uint bounce,constant DiffuseGIFrame& frame)
{
    float phase=fract(dot(float2(pixel)+PixelCenter,float2(sqrt(2.0f),sqrt(3.0f))));
    float2 stratum=float2((float(path)+PixelCenter)/float(frame.Options.y),phase+GoldenStep*float(path));
    return fract(stratum+float(bounce)*float2(sqrt(5.0f),sqrt(7.0f)));
}
float3 HemisphereDirection(float3 normal,float2 xi)
{
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    float3 tangent=Unit(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
    float radius=sqrt(xi.x),azimuth=2*Pi*xi.y;
    return Unit(tangent*(radius*cos(azimuth))+bitangent*(radius*sin(azimuth))+normal*sqrt(1-xi.x),normal);
}
float3 DiffuseWeight(float3 color,float metallic,constant DiffuseGIFrame& frame)
{
    float3 result=saturate(color);
    if (frame.Options.z==PBRModel) result*=(1-saturate(metallic))*(1-DielectricF0);
    return result;
}
float3 ReceiverDiffuse(uint2 pixel,constant DiffuseGIFrame& frame,
    texture2d<float,access::read> diffuse,texture2d<float,access::read> ambient)
{ return DiffuseWeight(diffuse.read(pixel).rgb,ambient.read(pixel).w,frame); }
float3 Estimate(uint2 pixel,float3 point,float3 normal,thread const RayContext& context,
    texture2d<float,access::read> diffuse,texture2d<float,access::read> ambient)
{
    constant DiffuseGIFrame& frame=*context.Frame;
    float3 receiver=ReceiverDiffuse(pixel,frame,diffuse,ambient),result=0;
    if (!any(receiver>0)) return result;
    for (uint path=0;path<frame.Options.y;++path)
    {
        float3 throughput=receiver,origin=point+normal*frame.Trace.y;
        float3 direction=HemisphereDirection(normal,PathSample(pixel,path,0,frame));
        for (uint bounce=0;bounce<frame.Lighting.y;++bounce)
        {
            Hit hit; if (!TraceScene(origin,direction,frame.Trace.x,false,context,hit)) break;
            HitSurface surface=ReadHit(hit,origin,direction,context);
            result+=throughput*min(Shade(surface,-direction,context),float3(HalfMaximum-HalfSpacing));
            if (bounce+1==frame.Lighting.y) break;
            throughput*=DiffuseWeight(surface.Diffuse,frame.Material.y,frame); if (!any(throughput>0)) break;
            direction=HemisphereDirection(surface.Normal,PathSample(pixel,path,bounce+1,frame));
            if (dot(surface.Geometric,direction)*dot(surface.Geometric,surface.Normal)<=0) break;
            origin=surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0 ? frame.Trace.y : -frame.Trace.y);
        }
    }
    return min(result*(frame.Trace.z/float(frame.Options.y)),float3(HalfMaximum-HalfSpacing));
}
float3 Filter(uint2 pixel,float3 point,float3 normal,constant DiffuseGIFrame& frame,
    texture2d<float,access::read> depth,texture2d<float,access::read> normals,texture2d<float,access::read> previous,
    texture2d<float,access::read> diffuse,texture2d<float,access::read> ambient)
{
    int2 axis=frame.Options.x==HorizontalStage ? int2(1,0) : int2(0,1); float3 adjacent;
    if (!Unproject((float2(int2(pixel)+axis)+PixelCenter)*frame.Viewport.zw,depth.read(pixel).r,frame,adjacent)) adjacent=point;
    float sigma=max(max(frame.Trace.y*BiasSigmaScale,length(adjacent-point)*FootprintSigmaScale),MinimumDepthSigma);
    float3 sum=0,centerDiffuse=ReceiverDiffuse(pixel,frame,diffuse,ambient); float total=0;
    for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        int2 q=int2(pixel)+axis*offset; if (any(q<0) || any(q>=int2(frame.Viewport.xy))) continue;
        float3 position,n; float r; if (!Receiver(uint2(q),frame,depth,normals,position,n,r)) continue;
        float dz=dot(position-point,frame.CameraForward.xyz)/sigma;
        float3 df=(ReceiverDiffuse(uint2(q),frame,diffuse,ambient)-centerDiffuse)/ReflectanceSigma;
        float weight=exp(-float(offset*offset)/(2*SpatialSigma*SpatialSigma)-(dz*dz+dot(df,df))/2)*pow(saturate(dot(normal,n)),NormalPower);
        sum+=previous.read(uint2(q)).rgb*weight; total+=weight;
    }
    return sum/max(total,MinimumWeight);
}
fragment float4 PSMain(ScreenVertex input [[stage_in]],constant DiffuseGIFrame& frame [[buffer(0)]],
    device const BvhNode* nodes [[buffer(1)]],device const Triangle* triangles [[buffer(2)]],
    device const CoverageMaterial* coverage [[buffer(3)]],device const float* alphaTexels [[buffer(4)]],
    device const Surface* surfaces [[buffer(5)]],device const MaterialRecord* materials [[buffer(6)]],
    device const float4* colorTexels [[buffer(7)]],device const LightRecord* lights [[buffer(8)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
    texture2d<float,access::read> diffuse [[texture(2)]],
    texture2d<float,access::read> ambient [[texture(3)]],texture2d<float,access::read> previous [[texture(4)]],
    texture2d<float,access::read> opaque [[texture(5)]])
{
    uint2 pixel=uint2(input.Position.xy);
    if (frame.Options.x==CompositeStage) return float4(previous.read(pixel).rgb,0);
    float3 point,normal; float roughness;
    if (!Receiver(pixel,frame,depth,normals,point,normal,roughness)) return float4(0);
    if (frame.Trace.z<=0) return float4(0);
    float3 value;
    if (frame.Options.x==TraceStage)
    {
        RayContext context {&frame,nodes,triangles,coverage,alphaTexels,surfaces,materials,colorTexels,lights};
        value=Estimate(pixel,point,normal,context,diffuse,ambient);
    }
    else value=Filter(pixel,point,normal,frame,depth,normals,previous,diffuse,ambient);
    if (frame.Options.x==VerticalStage) value=min(value,max(float3(HalfMaximum-HalfSpacing)-opaque.read(pixel).rgb,float3(0)));
    return float4(value,0);
}
