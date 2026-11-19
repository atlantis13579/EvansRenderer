// DX12 diffuse scene transport with a bounded number of cosine-sampled bounces.
// Explicit emitter visibility is evaluated at each hit; ambient is not recursive.
// Independent bilateral reconstruction uses receiver depth, normals and albedo.
// PBRT 4e: https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection
// https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Simple_Path_Tracer
// Conservative triangle edges: https://jcgt.org/published/0002/01/05/ .
cbuffer DiffuseGIFrame : register(b0)
{
    row_major float4x4 WorldFromClip;
    float4 FrameSize,RayParameters,EyeForward,Reserved,MaterialParameters;
    uint4 GeometryCounts,SurfaceCounts,Control,LightControl;
};
Texture2D<float> DepthImage : register(t0);
Texture2D<float4> NormalImage : register(t1),DiffuseImage : register(t2);
Texture2D<float4> AmbientImage : register(t3),PreviousRadiance : register(t4),OpaqueColor : register(t5);
struct BvhNode { float4 Minimum,Maximum; uint4 Link; };
struct Primitive { float4 P0,P1,P2,UV01,UV2; uint4 Identity; };
struct AlphaMaterial { uint4 Diffuse,Opacity; float4 Factor; };
struct CornerData { float4 N0,N1,N2,C0,C1,C2; };
struct HitMaterial { float4 Base,Ambient,Specular,Emitted; uint4 ColorMap,NormalMap; };
struct SceneLight { float4 PositionType,DirectionPower,ColorShadow,Shape; };
StructuredBuffer<BvhNode> Hierarchy : register(t6);
StructuredBuffer<Primitive> Geometry : register(t7);
StructuredBuffer<AlphaMaterial> AlphaMaterials : register(t8);
StructuredBuffer<float> AlphaAtlas : register(t9);
StructuredBuffer<CornerData> Attributes : register(t10);
StructuredBuffer<HitMaterial> HitMaterials : register(t11);
StructuredBuffer<float4> ColorAtlas : register(t12);
StructuredBuffer<SceneLight> SceneLights : register(t13);
// Ray hits retain the Toon palette, shallow normals and colored light ramp.
static const float ToonShadingModel = 4.0;
static const float ToonAlbedoExponent = 0.8;
static const float ToonPaletteSteps = 16.0;
static const float ToonColorSaturation = 1.12;
static const float ToonNormalStrength = 0.12;
static const float ToonLightScale = 0.45;
static const float ToonBandSoftness = 0.4;
static const float ToonShadowFloor = 0.18;
static const float ToonHighlightThreshold = 0.8;
static const float ToonHighlightStrength = 0.18;
static const float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
static const float3 ToonCoolShadow = float3(0.56, 0.68, 1.0);
static const float3 ToonWarmLight = float3(1.0, 0.94, 0.82);
float3 ToonAlbedo(float3 color, float model)
{
    if (model != ToonShadingModel) return color;
    float3 paint = pow(saturate(color), ToonAlbedoExponent);
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(lerp(dot(paint, ToonLuma).xxx, paint, ToonColorSaturation));
}
float3 ToonResponse(float cosine, float bands)
{
    float angularBand = saturate(cosine) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, frac(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return lerp(ToonCoolShadow, ToonWarmLight, level) * lerp(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

static const uint TraceStage=0,HorizontalStage=1,VerticalStage=2,AddStage=3;
static const uint PbrShading=3,ToonShading=4,FlatFlag=1,CasterFlag=1,Directional=0,Point=1,LegacyGrid=2;
static const float Pi=3.141592653589793,SequenceStep=(3-sqrt(5.0))/2;
static const float Center=0.5,BackgroundDepth=1,WThreshold=1e-7,VectorThreshold=1e-16;
static const float FloatSpacing=1.1920928955078125e-7,EdgeBound=3*FloatSpacing/(1-3*FloatSpacing);
static const float UvThreshold=1e-8,AxisLimit=0.999,AlphaFloor=0.0025,CosineFloor=1e-6;
static const float Dielectric=0.04,FresnelExponent=5,DistanceSquaredFloor=0.01;
static const float ReflectanceSigma=0.05;
static const float HdrMaximum=65504,HdrSpacing=32,PixelSigma=1.5,NormalExponent=32;
static const float BiasFilterScale=4,FootprintFilterScale=2,DepthFloor=1e-5,WeightFloor=1e-7;
static const int KernelRadius=2;
struct VertexOut { float4 Position:SV_POSITION; };
VertexOut VSMain(uint vertex:SV_VertexID)
{ VertexOut output; float2 corner=float2((vertex<<1)&2,vertex&2); output.Position=float4(corner*2-1,0,1); return output; }
float3 Normalized(float3 v,float3 fallback)
{
    float scale=max(abs(v.x),max(abs(v.y),abs(v.z))); if (!(scale>0) || !isfinite(scale)) return fallback;
    v/=scale; return v*rsqrt(dot(v,v));
}
void AtlasFootprint(uint4 map,float2 uv,out uint4 address,out float2 fraction)
{
    float2 position=frac(float2(uv.x,-uv.y))*map.yz-Center;
    uint x=position.x<0 ? map.y-1 : uint(floor(position.x)),y=position.y<0 ? map.z-1 : uint(floor(position.y));
    uint nextX=x+1==map.y ? 0 : x+1,nextY=y+1==map.z ? 0 : y+1;
    address=map.x+uint4(y*map.y+x,y*map.y+nextX,nextY*map.y+x,nextY*map.y+nextX); fraction=frac(position);
}
float ReadAlpha(uint4 map,float2 uv)
{
    if (map.w==0) return 1; uint4 address; float2 f; AtlasFootprint(map,uv,address,f);
    return lerp(lerp(AlphaAtlas[address.x],AlphaAtlas[address.y],f.x),lerp(AlphaAtlas[address.z],AlphaAtlas[address.w],f.x),f.y);
}
float3 ReadColor(uint4 map,float2 uv)
{
    if (map.w==0) return 1; uint4 address; float2 f; AtlasFootprint(map,uv,address,f);
    return lerp(lerp(ColorAtlas[address.x].rgb,ColorAtlas[address.y].rgb,f.x),lerp(ColorAtlas[address.z].rgb,ColorAtlas[address.w].rgb,f.x),f.y);
}
bool Slabs(BvhNode node,float3 start,float3 direction,float maximum)
{
    float enter=0,leave=maximum;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (start[axis]<node.Minimum[axis] || start[axis]>node.Maximum[axis]) return false; }
        else
        {
            float lower=(node.Minimum[axis]-start[axis])/direction[axis],upper=(node.Maximum[axis]-start[axis])/direction[axis];
            enter=max(enter,min(lower,upper)); leave=min(leave,max(lower,upper)); if (enter>leave) return false;
        }
    }
    return true;
}
float EdgeFunction(float2 a,float2 b)
{ precise float positive=a.x*b.y,negative=a.y*b.x; precise float edge=positive-negative; return abs(edge)<=EdgeBound*(abs(positive)+abs(negative)) ? 0 : edge; }
bool HitPrimitive(Primitive p,float3 start,float3 direction,float maximum,out float distance,out float3 barycentric)
{
    distance=0; barycentric=0; uint major=abs(direction.x)>abs(direction.y) ? 0 : 1;
    if (abs(direction.z)>abs(direction[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 a=p.P0.xyz-start,b=p.P1.xyz-start,c=p.P2.xyz-start;
    float2 shear=float2(direction[horizontal],direction[vertical])/direction[major];
    precise float2 q0=float2(a[horizontal],a[vertical])-a[major]*shear;
    precise float2 q1=float2(b[horizontal],b[vertical])-b[major]*shear,q2=float2(c[horizontal],c[vertical])-c[major]*shear;
    float3 edges=float3(EdgeFunction(q1,q2),EdgeFunction(q2,q0),EdgeFunction(q0,q1));
    if (any(edges<0) && any(edges>0)) return false;
    float sum=edges.x+edges.y+edges.z; if (sum==0) return false;
    distance=dot(edges,float3(a[major],b[major],c[major]))/(sum*direction[major]);
    if (!(distance>0 && distance<=maximum)) return false; barycentric=edges/sum; return true;
}
struct Intersection { uint PrimitiveIndex; float Distance; float3 Barycentric; };
bool Traverse(float3 start,float3 direction,float maximum,bool visibilityOnly,out Intersection closest)
{
    closest.PrimitiveIndex=0; closest.Distance=maximum; closest.Barycentric=0;
    bool accepted=false; uint index=0;
    [loop] while (index<GeometryCounts.x)
    {
        BvhNode node=Hierarchy[index];
        if (!Slabs(node,start,direction,closest.Distance)) { index=node.Link.w; continue; }
        if (node.Link.z==0) { index=node.Link.x; continue; }
        [loop] for (uint local=0;local<node.Link.z;++local)
        {
            uint primitiveIndex=node.Link.y+local; Primitive p=Geometry[primitiveIndex];
            if (visibilityOnly && (p.Identity.w&CasterFlag)==0) continue;
            float distance; float3 barycentric;
            if (!HitPrimitive(p,start,direction,closest.Distance,distance,barycentric)) continue;
            AlphaMaterial material=AlphaMaterials[p.Identity.x];
            if (material.Factor.y>0)
            {
                float2 uv=p.UV01.xy*barycentric.x+p.UV01.zw*barycentric.y+p.UV2.xy*barycentric.z;
                float coverage=saturate(dot(barycentric,float3(p.P0.w,p.P1.w,p.P2.w))*material.Factor.x*
                    ReadAlpha(material.Diffuse,uv)*ReadAlpha(material.Opacity,uv));
                if (coverage<material.Factor.y) continue;
            }
            closest.PrimitiveIndex=primitiveIndex; closest.Distance=distance; closest.Barycentric=barycentric; accepted=true;
            if (visibilityOnly) return true;
        }
        index=node.Link.w;
    }
    return accepted;
}
struct MaterialPoint { float3 Position,N,Ng,Kd,Ks,Ka,Emission; float Exponent; };
MaterialPoint EvaluateMaterial(Intersection hit,float3 start,float3 direction)
{
    Primitive primitive=Geometry[hit.PrimitiveIndex]; CornerData corners=Attributes[hit.PrimitiveIndex];
    HitMaterial material=HitMaterials[primitive.Identity.x]; float3 bary=hit.Barycentric;
    float2 uv=primitive.UV01.xy*bary.x+primitive.UV01.zw*bary.y+primitive.UV2.xy*bary.z;
    float3 ab=primitive.P1.xyz-primitive.P0.xyz,ac=primitive.P2.xyz-primitive.P0.xyz;
    MaterialPoint surfacePoint; surfacePoint.Position=start+direction*hit.Distance;
    surfacePoint.Ng=Normalized(cross(Normalized(ab,float3(1,0,0)),Normalized(ac,float3(0,1,0))),float3(0,0,1));
    surfacePoint.N=(Control.w&FlatFlag) ? surfacePoint.Ng : Normalized(corners.N0.xyz*bary.x+corners.N1.xyz*bary.y+corners.N2.xyz*bary.z,surfacePoint.Ng);
    if (dot(surfacePoint.Ng,direction)>0) surfacePoint.N=-surfacePoint.N;
    if (material.NormalMap.w)
    {
        float2 st1=float2(primitive.UV01.z-primitive.UV01.x,primitive.UV01.y-primitive.UV01.w);
        float2 st2=float2(primitive.UV2.x-primitive.UV01.x,primitive.UV01.y-primitive.UV2.y);
        float determinant=st1.x*st2.y-st1.y*st2.x;
        if (abs(determinant)>=UvThreshold)
        {
            float3 tangent=(ab*st2.y-ac*st1.y)/determinant,sourceBitangent=(ac*st1.x-ab*st2.x)/determinant;
            float3 axis=abs(surfacePoint.N.y)<AxisLimit ? float3(0,1,0) : float3(1,0,0);
            tangent=Normalized(tangent-surfacePoint.N*dot(tangent,surfacePoint.N),Normalized(cross(axis,surfacePoint.N),float3(1,0,0)));
            float3 bitangent=cross(surfacePoint.N,tangent); bitangent*=dot(bitangent,sourceBitangent)<0 ? -1 : 1;
            float3 mapped=ReadColor(material.NormalMap,uv)*2-1;
            if (float(Control.z) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            surfacePoint.N=Normalized(tangent*mapped.x+bitangent*mapped.y+surfacePoint.N*mapped.z,surfacePoint.N);
        }
    }
    surfacePoint.Kd=ToonAlbedo(max((corners.C0.rgb*bary.x+corners.C1.rgb*bary.y+corners.C2.rgb*bary.z)*material.Base.rgb*ReadColor(material.ColorMap,uv),0), float(Control.z));
    surfacePoint.Ks=material.Specular.rgb; surfacePoint.Exponent=material.Specular.w; surfacePoint.Ka=material.Ambient.rgb; surfacePoint.Emission=material.Emitted.rgb;
    return surfacePoint;
}
float3 Schlick(float3 base,float cosine) { return base+(1-base)*pow(1-saturate(cosine),FresnelExponent); }
float Masking(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),CosineFloor); }
float3 Scattering(MaterialPoint surfacePoint,float3 view,float3 incoming)
{
    float nl=dot(surfacePoint.N,incoming),nv=dot(surfacePoint.N,view); if (nl<=0 || nv<=0) return 0;
    float3 h=Normalized(view+incoming,surfacePoint.N); float nh=saturate(dot(surfacePoint.N,h));
    if (Control.z!=PbrShading)
    {
        float highlight=pow(nh,surfacePoint.Exponent);
        if (Control.z==ToonShading) return surfacePoint.Kd * ToonResponse(max(nl, 0.0), MaterialParameters.z) + surfacePoint.Ks * ToonHighlight(highlight);
        return surfacePoint.Kd*nl+surfacePoint.Ks*highlight;
    }
    float alpha=max(MaterialParameters.x*MaterialParameters.x,AlphaFloor),alphaSquared=alpha*alpha;
    float denominator=nh*nh*(alphaSquared-1)+1;
    float distribution=alphaSquared/(Pi*denominator*denominator);
    float3 f=Schlick(lerp(Dielectric.xxx,saturate(surfacePoint.Kd),MaterialParameters.y),dot(view,h));
    float3 specular=f*distribution*Masking(nv,alpha)*Masking(nl,alpha)/max(4*nv*nl,CosineFloor);
    return ((1-f)*(1-MaterialParameters.y)*surfacePoint.Kd/Pi+specular)*nl;
}
float3 EmitterSample(MaterialPoint surfacePoint,float3 view,float3 source,float3 radiance,float3 normal,float measure,bool distant,bool rectangle,bool shadows,float radius)
{
    float3 displacement=distant ? source*EyeForward.w : source-surfacePoint.Position;
    float squared=dot(displacement,displacement); if (!(squared>VectorThreshold)) return 0;
    float distance=sqrt(squared); float3 incoming=displacement/distance; float attenuation=1;
    if (!distant)
    {
        float window=1;
        // FXC also specializes this expression for lights with unbounded range.
        if (radius>0) { float relative=distance/(radius>0 ? radius : 1); window=saturate(1-relative*relative*relative*relative); window*=window; }
        attenuation=measure*window*(rectangle ? max(dot(normal,-incoming),0) : 1)/max(squared,DistanceSquaredFloor);
    }
    float3 contribution=Scattering(surfacePoint,view,incoming)*radiance*attenuation; if (!any(contribution>0)) return 0;
    if (LightControl.x && shadows)
    {
        float3 origin=surfacePoint.Position+surfacePoint.Ng*(dot(surfacePoint.Ng,incoming)>=0 ? RayParameters.y : -RayParameters.y);
        float3 segment=surfacePoint.Position+displacement-origin; float extent=length(segment); Intersection obstruction;
        if (extent>RayParameters.y && Traverse(origin,segment/extent,extent-RayParameters.y,true,obstruction)) return 0;
    }
    return contribution;
}
float3 IncomingRadiance(MaterialPoint surfacePoint,float3 view)
{
    float3 radiance=surfacePoint.Emission;
    [loop] for (uint index=0;index<SurfaceCounts.w;++index)
    {
        SceneLight light=SceneLights[index]; float3 energy=max(light.ColorShadow.rgb,0)*max(light.DirectionPower.w,0);
        if (!any(energy>0)) continue; uint kind=uint(light.PositionType.w); bool shadows=light.ColorShadow.w>0;
        if (kind==Directional)
            radiance+=EmitterSample(surfacePoint,view,Normalized(light.DirectionPower.xyz,float3(0,1,0)),energy,0,1,true,false,shadows,0);
        else if (kind==Point)
            radiance+=EmitterSample(surfacePoint,view,light.PositionType.xyz,energy,0,1,false,false,shadows,light.Shape.z);
        else
        {
            float3 normal=Normalized(light.DirectionPower.xyz,float3(0,-1,0));
            float3 axis=abs(normal.y)<AxisLimit ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Normalized(cross(axis,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
            uint count=light.Shape.w>0 ? uint(light.Shape.w) : LegacyGrid;
            float measure=light.Shape.w>0 ? abs(light.Shape.x*light.Shape.y) : 1;
            [loop] for (uint y=0;y<count;++y) [loop] for (uint x=0;x<count;++x)
            {
                float2 coordinate=(float2(x,y)+Center)/count-Center;
                float3 emitter=light.PositionType.xyz+tangent*(coordinate.x*light.Shape.x)+bitangent*(coordinate.y*light.Shape.y);
                radiance+=EmitterSample(surfacePoint,view,emitter,energy,normal,measure/(count*count),false,true,shadows,light.Shape.z);
            }
        }
    }
    return clamp(radiance,0,HdrMaximum-HdrSpacing);
}
bool WorldPoint(float2 uv,float depth,out float3 result)
{
    result=0; float4 homogeneous=mul(WorldFromClip,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!(abs(homogeneous.w)>WThreshold)) return false; result=homogeneous.xyz/homogeneous.w; return all(isfinite(result));
}
bool ReadReceiver(int2 pixel,out float3 position,out float3 normal,out float roughness)
{
    position=normal=0; float4 value=NormalImage.Load(int3(pixel,0)); roughness=value.w;
    float lengthSquared=dot(value.xyz,value.xyz),depth=DepthImage.Load(int3(pixel,0));
    if (!(depth<BackgroundDepth && roughness>=0 && lengthSquared>VectorThreshold && isfinite(lengthSquared))) return false;
    normal=value.xyz*rsqrt(lengthSquared); return WorldPoint((float2(pixel)+Center)*FrameSize.zw,depth,position);
}
float2 PathSample(int2 pixel,uint path,uint depth)
{
    float phase=frac(dot(float2(pixel)+Center,float2(sqrt(2.0),sqrt(3.0))));
    float2 stratum=float2((path+Center)/Control.y,phase+SequenceStep*path);
    return frac(stratum+depth*float2(sqrt(5.0),sqrt(7.0)));
}
float3 DiffuseDirection(float3 n,float2 xi)
{
    float3 axis=abs(n.x)<abs(n.y) ? float3(1,0,0) : float3(0,1,0);
    float3 tangent=Normalized(cross(axis,n),float3(1,0,0)),bitangent=cross(n,tangent);
    float radial=sqrt(xi.x),azimuth=2*Pi*xi.y;
    return Normalized(tangent*(radial*cos(azimuth))+bitangent*(radial*sin(azimuth))+n*sqrt(1-xi.x),n);
}
float3 DiffuseReflectance(float3 kd,float metal)
{
    float3 weight=saturate(kd);
    if (Control.z==PbrShading) weight*=(1-saturate(metal))*(1-Dielectric);
    return weight;
}
float3 ReceiverDiffuse(int2 pixel)
{ return DiffuseReflectance(DiffuseImage.Load(int3(pixel,0)).rgb,AmbientImage.Load(int3(pixel,0)).w); }
float3 TraceDiffuse(int2 pixel,float3 position,float3 normal)
{
    float3 receiver=ReceiverDiffuse(pixel),radiance=0; if (!any(receiver>0)) return 0;
    [loop] for (uint path=0;path<Control.y;++path)
    {
        float3 weight=receiver,start=position+normal*RayParameters.y;
        float3 ray=DiffuseDirection(normal,PathSample(pixel,path,0));
        [loop] for (uint depth=0;depth<LightControl.y;++depth)
        {
            Intersection hit; if (!Traverse(start,ray,RayParameters.x,false,hit)) break;
            MaterialPoint surfacePoint=EvaluateMaterial(hit,start,ray);
            radiance+=weight*min(IncomingRadiance(surfacePoint,-ray),HdrMaximum-HdrSpacing);
            if (depth+1==LightControl.y) break;
            weight*=DiffuseReflectance(surfacePoint.Kd,MaterialParameters.y); if (!any(weight>0)) break;
            ray=DiffuseDirection(surfacePoint.N,PathSample(pixel,path,depth+1));
            if (dot(surfacePoint.Ng,ray)*dot(surfacePoint.Ng,surfacePoint.N)<=0) break;
            start=surfacePoint.Position+surfacePoint.Ng*(dot(surfacePoint.Ng,ray)>=0 ? RayParameters.y : -RayParameters.y);
        }
    }
    return min(radiance*(RayParameters.z/Control.y),HdrMaximum-HdrSpacing);
}
float3 Reconstruct(int2 pixel,float3 position,float3 normal)
{
    int2 direction=Control.x==HorizontalStage ? int2(1,0) : int2(0,1); float3 neighbor;
    if (!WorldPoint((float2(pixel+direction)+Center)*FrameSize.zw,DepthImage.Load(int3(pixel,0)),neighbor)) neighbor=position;
    float sigma=max(max(RayParameters.y*BiasFilterScale,length(neighbor-position)*FootprintFilterScale),DepthFloor);
    float3 total=0,centerDiffuse=ReceiverDiffuse(pixel); float normalization=0;
    [unroll] for (int offset=-KernelRadius;offset<=KernelRadius;++offset)
    {
        int2 q=pixel+direction*offset; if (any(q<0) || any(q>=int2(FrameSize.xy))) continue;
        float3 other,n; float r; if (!ReadReceiver(q,other,n,r)) continue;
        float depthDelta=dot(other-position,EyeForward.xyz)/sigma;
        float3 reflectanceDelta=(ReceiverDiffuse(q)-centerDiffuse)/ReflectanceSigma;
        float weight=exp(-offset*offset/(2*PixelSigma*PixelSigma)-(depthDelta*depthDelta+dot(reflectanceDelta,reflectanceDelta))/2)*pow(saturate(dot(normal,n)),NormalExponent);
        total+=PreviousRadiance.Load(int3(q,0)).rgb*weight; normalization+=weight;
    }
    return total/max(normalization,WeightFloor);
}
float4 PSMain(VertexOut input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy);
    if (Control.x==AddStage) return float4(PreviousRadiance.Load(int3(pixel,0)).rgb,0);
    float3 position,normal; float roughness;
    [branch] if (!ReadReceiver(pixel,position,normal,roughness)) return 0;
    [branch] if (RayParameters.z<=0) return 0;
    float3 value;
    [branch] if (Control.x==TraceStage) value=TraceDiffuse(pixel,position,normal);
    else value=Reconstruct(pixel,position,normal);
    if (Control.x==VerticalStage) value=min(value,max(HdrMaximum-HdrSpacing-OpaqueColor.Load(int3(pixel,0)).rgb,0));
    return float4(value,0);
}
