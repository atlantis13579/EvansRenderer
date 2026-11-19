// DX11 bounded diffuse GI: cosine-weighted scene paths, explicit secondary
// lighting, two depth/normal/albedo-aware filters and RGB-only composition.
// Bounces exclude recursive ambient and stop at the configured transport depth.
// PBRT 4e: https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection
// https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Simple_Path_Tracer
// Conservative FP32 triangle edges: Woop et al., JCGT 2013,
// https://jcgt.org/published/0002/01/05/ .
cbuffer RayGIParameters : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 Viewport,Trace,CameraForward,Reserved,Material;
    uint4 SceneCounts,ShadingCounts,Options,Lighting;
};
Texture2D<float> SceneDepth : register(t0);
Texture2D<float4> NormalRoughness : register(t1),DiffuseOpacity : register(t2);
Texture2D<float4> AmbientMetallic : register(t3),PreviousRadiance : register(t4),SceneColor : register(t5);
struct Node { float4 Lower,Upper; uint4 Links; };
struct Triangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct CoverageMaterial { uint4 Diffuse,Opacity; float4 Alpha; };
struct Surface { float4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct SurfaceMaterial { float4 Base,AmbientColor,Specular,Emission; uint4 Diffuse,Normal; };
struct Light { float4 PositionType,DirectionIntensity,ColorShadow,Area; };
StructuredBuffer<Node> Nodes : register(t6);
StructuredBuffer<Triangle> Triangles : register(t7);
StructuredBuffer<CoverageMaterial> CoverageMaterials : register(t8);
StructuredBuffer<float> AlphaTexels : register(t9);
StructuredBuffer<Surface> Surfaces : register(t10);
StructuredBuffer<SurfaceMaterial> SurfaceMaterials : register(t11);
StructuredBuffer<float4> ColorTexels : register(t12);
StructuredBuffer<Light> Lights : register(t13);
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

static const uint TracePass=0,HorizontalPass=1,VerticalPass=2,CompositePass=3,PBRModel=3,ToonModel=4;
static const uint FlatNormals=1,ShadowCaster=1,DirectionalLight=0,PointLight=1,LegacyAreaAxis=2;
static const float Pi=3.141592653589793,GoldenFraction=(3-sqrt(5.0))/2;
static const float PixelCenter=0.5,EmptyDepth=1,MinimumW=1e-7,MinimumSquared=1e-16;
static const float Binary32Epsilon=1.1920928955078125e-7,EdgeError=3*Binary32Epsilon/(1-3*Binary32Epsilon);
static const float TangentEpsilon=1e-8,ParallelAxis=0.999,MinimumAlpha=0.0025,MinimumCosine=1e-6;
static const float DielectricF0=0.04,SchlickPower=5,LightDistanceFloor=0.01;
static const float ReflectanceSigma=0.05;
static const float HalfMaximum=65504,HalfSpacing=32,SpatialSigma=1.5,NormalPower=32;
static const float DepthBiasScale=4,DepthFootprintScale=2,MinimumFilterDepth=1e-5,MinimumWeight=1e-7;
static const int FilterRadius=2;
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint index:SV_VertexID)
{ float2 p=float2((index<<1)&2,index&2); ScreenVertex v; v.Position=float4(p*2-1,0,1); return v; }
float3 Unit(float3 value,float3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (!(scale>0) || !isfinite(scale)) return fallback;
    value/=scale; return value*rsqrt(dot(value,value));
}
void TexelCoordinates(uint4 image,float2 uv,out uint4 indices,out float2 blend)
{
    float2 p=frac(float2(uv.x,-uv.y))*image.yz-PixelCenter;
    uint2 a=uint2(p.x<0 ? image.y-1 : uint(floor(p.x)),p.y<0 ? image.z-1 : uint(floor(p.y)));
    uint2 b=uint2(a.x+1==image.y ? 0 : a.x+1,a.y+1==image.z ? 0 : a.y+1);
    indices=image.x+uint4(a.y*image.y+a.x,a.y*image.y+b.x,b.y*image.y+a.x,b.y*image.y+b.x); blend=frac(p);
}
float Alpha(uint4 image,float2 uv)
{
    if (!image.w) return 1; uint4 p; float2 t; TexelCoordinates(image,uv,p,t);
    return lerp(lerp(AlphaTexels[p.x],AlphaTexels[p.y],t.x),lerp(AlphaTexels[p.z],AlphaTexels[p.w],t.x),t.y);
}
float3 TextureRGB(uint4 image,float2 uv)
{
    if (!image.w) return 1; uint4 p; float2 t; TexelCoordinates(image,uv,p,t);
    return lerp(lerp(ColorTexels[p.x].rgb,ColorTexels[p.y].rgb,t.x),lerp(ColorTexels[p.z].rgb,ColorTexels[p.w].rgb,t.x),t.y);
}
bool Box(Node node,float3 origin,float3 ray,float limit)
{
    float nearDistance=0,farDistance=limit;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (ray[axis]==0) { if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false; }
        else
        {
            float a=(node.Lower[axis]-origin[axis])/ray[axis],b=(node.Upper[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(a,b)); farDistance=min(farDistance,max(a,b));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float Edge(float2 a,float2 b)
{ precise float x=a.x*b.y,y=a.y*b.x; precise float e=x-y; return abs(e)<=EdgeError*(abs(x)+abs(y)) ? 0 : e; }
bool Intersect(Triangle tri,float3 origin,float3 ray,float limit,out float distance,out float3 weights)
{
    distance=0; weights=0;
    uint z=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3;
    float3 a=tri.A.xyz-origin,b=tri.B.xyz-origin,c=tri.C.xyz-origin;
    float2 shear=float2(ray[x],ray[y])/ray[z];
    precise float2 pa=float2(a[x],a[y])-shear*a[z],pb=float2(b[x],b[y])-shear*b[z],pc=float2(c[x],c[y])-shear*c[z];
    float3 e=float3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(e<0) && any(e>0)) return false;
    float determinant=e.x+e.y+e.z; if (determinant==0) return false;
    distance=dot(e,float3(a[z],b[z],c[z]))/(determinant*ray[z]);
    if (!(distance>0 && distance<=limit)) return false;
    weights=e/determinant; return true;
}
struct Hit { uint Index; float Distance; float3 Weights; };
bool TraceScene(float3 origin,float3 ray,float limit,bool shadow,out Hit hit)
{
    hit.Index=0; hit.Distance=limit; hit.Weights=0; bool found=false; uint nodeIndex=0;
    [loop] while (nodeIndex<SceneCounts.x)
    {
        Node node=Nodes[nodeIndex];
        if (!Box(node,origin,ray,hit.Distance)) { nodeIndex=node.Links.w; continue; }
        if (!node.Links.z) { nodeIndex=node.Links.x; continue; }
        [loop] for (uint offset=0;offset<node.Links.z;++offset)
        {
            uint index=node.Links.y+offset; Triangle tri=Triangles[index]; float distance; float3 weights;
            if (shadow && !(tri.Metadata.w&ShadowCaster)) continue;
            if (!Intersect(tri,origin,ray,hit.Distance,distance,weights)) continue;
            CoverageMaterial material=CoverageMaterials[tri.Metadata.x];
            if (material.Alpha.y>0)
            {
                float2 uv=tri.UVAB.xy*weights.x+tri.UVAB.zw*weights.y+tri.UVC.xy*weights.z;
                float coverage=saturate(dot(weights,float3(tri.A.w,tri.B.w,tri.C.w))*material.Alpha.x*Alpha(material.Diffuse,uv)*Alpha(material.Opacity,uv));
                if (coverage<material.Alpha.y) continue;
            }
            hit.Index=index; hit.Distance=distance; hit.Weights=weights; found=true;
            if (shadow) return true;
        }
        nodeIndex=node.Links.w;
    }
    return found;
}
struct HitSurface { float3 Position,Normal,Geometric,Diffuse,Specular,Emission,AmbientColor; float Shininess; };
HitSurface ReadHit(Hit hit,float3 origin,float3 ray)
{
    Triangle t=Triangles[hit.Index]; Surface s=Surfaces[hit.Index]; SurfaceMaterial m=SurfaceMaterials[t.Metadata.x];
    float3 w=hit.Weights; float2 uv=t.UVAB.xy*w.x+t.UVAB.zw*w.y+t.UVC.xy*w.z;
    float3 edgeB=t.B.xyz-t.A.xyz,edgeC=t.C.xyz-t.A.xyz;
    HitSurface result; result.Position=origin+ray*hit.Distance;
    result.Geometric=Unit(cross(Unit(edgeB,float3(1,0,0)),Unit(edgeC,float3(0,1,0))),float3(0,0,1));
    float3 normal=(Options.w&FlatNormals) ? result.Geometric : Unit(s.NormalA.xyz*w.x+s.NormalB.xyz*w.y+s.NormalC.xyz*w.z,result.Geometric);
    if (dot(result.Geometric,ray)>0) normal=-normal;
    if (m.Normal.w)
    {
        float2 uvB=float2(t.UVAB.z-t.UVAB.x,t.UVAB.y-t.UVAB.w),uvC=float2(t.UVC.x-t.UVAB.x,t.UVAB.y-t.UVC.y);
        float determinant=uvB.x*uvC.y-uvB.y*uvC.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            float3 tangent=(edgeB*uvC.y-edgeC*uvB.y)/determinant,sourceBitangent=(edgeC*uvB.x-edgeB*uvC.x)/determinant;
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            tangent=Unit(tangent-normal*dot(normal,tangent),Unit(cross(helper,normal),float3(1,0,0)));
            float3 crossed=cross(normal,tangent),bitangent=crossed*(dot(crossed,sourceBitangent)<0 ? -1 : 1);
            float3 mapped=TextureRGB(m.Normal,uv)*2-1;
            if (float(Options.z) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=ToonAlbedo(max((s.ColorA.rgb*w.x+s.ColorB.rgb*w.y+s.ColorC.rgb*w.z)*m.Base.rgb*TextureRGB(m.Diffuse,uv),0), float(Options.z));
    result.Specular=m.Specular.rgb; result.Shininess=m.Specular.w; result.Emission=m.Emission.rgb; result.AmbientColor=m.AmbientColor.rgb;
    return result;
}
float3 Fresnel(float3 f0,float cosine) { return f0+(1-f0)*pow(1-saturate(cosine),SchlickPower); }
float SmithG1(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),MinimumCosine); }
float3 BrdfCosine(HitSurface surface,float3 view,float3 direction)
{
    float nl=dot(surface.Normal,direction),nv=dot(surface.Normal,view); if (nl<=0 || nv<=0) return 0;
    float3 halfVector=Unit(view+direction,surface.Normal); float nh=saturate(dot(surface.Normal,halfVector));
    if (Options.z!=PBRModel)
    {
        float highlight=pow(nh,surface.Shininess);
        if (Options.z==ToonModel) return surface.Diffuse * ToonResponse(max(nl, 0.0), Material.z) + surface.Specular * ToonHighlight(highlight);
        return surface.Diffuse*nl+surface.Specular*highlight;
    }
    float alpha=max(Material.x*Material.x,MinimumAlpha),a2=alpha*alpha;
    float denominator=nh*nh*(a2-1)+1,distribution=a2/(Pi*denominator*denominator);
    float3 f=Fresnel(lerp(DielectricF0.xxx,saturate(surface.Diffuse),Material.y),dot(view,halfVector));
    float3 specular=f*distribution*SmithG1(nv,alpha)*SmithG1(nl,alpha)/max(4*nv*nl,MinimumCosine);
    return ((1-f)*(1-Material.y)*surface.Diffuse/Pi+specular)*nl;
}
float3 LitSample(HitSurface surface,float3 view,float3 emitter,float3 emitted,float3 emitterNormal,float weight,bool directional,bool area,bool castsShadow,float range)
{
    float3 offset=directional ? emitter*CameraForward.w : emitter-surface.Position;
    float squared=dot(offset,offset); if (!(squared>MinimumSquared)) return 0;
    float distance=sqrt(squared); float3 direction=offset/distance;
    float attenuation=1;
    if (!directional)
    {
        float window=1;
        // Keep the denominator defined in FXC's specialized unbounded-light path.
        if (range>0) { float ratio=distance/(range>0 ? range : 1); window=saturate(1-ratio*ratio*ratio*ratio); window*=window; }
        attenuation=weight*window*(area ? max(dot(emitterNormal,-direction),0) : 1)/max(squared,LightDistanceFloor);
    }
    float3 value=BrdfCosine(surface,view,direction)*emitted*attenuation;
    if (!any(value>0)) return 0;
    if (Lighting.x && castsShadow)
    {
        float3 origin=surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0 ? Trace.y : -Trace.y);
        float3 segment=surface.Position+offset-origin; float length=sqrt(dot(segment,segment)); Hit blocker;
        if (length>Trace.y && TraceScene(origin,segment/length,length-Trace.y,true,blocker)) return 0;
    }
    return value;
}
float3 Shade(HitSurface surface,float3 view)
{
    float3 result=surface.Emission;
    [loop] for (uint index=0;index<ShadingCounts.w;++index)
    {
        Light light=Lights[index]; uint type=uint(light.PositionType.w);
        float3 emitted=max(light.ColorShadow.rgb,0)*max(light.DirectionIntensity.w,0);
        if (!any(emitted>0)) continue;
        bool shadows=light.ColorShadow.w>0;
        if (type==DirectionalLight)
            result+=LitSample(surface,view,Unit(light.DirectionIntensity.xyz,float3(0,1,0)),emitted,0,1,true,false,shadows,0);
        else if (type==PointLight)
            result+=LitSample(surface,view,light.PositionType.xyz,emitted,0,1,false,false,shadows,light.Area.z);
        else
        {
            float3 normal=Unit(light.DirectionIntensity.xyz,float3(0,-1,0));
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Unit(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
            uint axis=light.Area.w>0 ? uint(light.Area.w) : LegacyAreaAxis;
            float measure=light.Area.w>0 ? abs(light.Area.x*light.Area.y) : 1;
            [loop] for (uint y=0;y<axis;++y) [loop] for (uint x=0;x<axis;++x)
            {
                float2 uv=(float2(x,y)+PixelCenter)/axis-PixelCenter;
                float3 emitter=light.PositionType.xyz+tangent*(uv.x*light.Area.x)+bitangent*(uv.y*light.Area.y);
                result+=LitSample(surface,view,emitter,emitted,normal,measure/(axis*axis),false,true,shadows,light.Area.z);
            }
        }
    }
    return min(max(result,0),HalfMaximum-HalfSpacing);
}
bool Unproject(float2 uv,float depth,out float3 surfacePoint)
{
    float4 h=mul(InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1)); surfacePoint=0;
    if (!(abs(h.w)>MinimumW)) return false; surfacePoint=h.xyz/h.w; return all(isfinite(surfacePoint));
}
bool Receiver(int2 pixel,out float3 surfacePoint,out float3 normal,out float roughness)
{
    surfacePoint=normal=0; float4 n=NormalRoughness.Load(int3(pixel,0)); roughness=n.w;
    float depth=SceneDepth.Load(int3(pixel,0)); float lengthSquared=dot(n.xyz,n.xyz);
    if (!(depth<EmptyDepth && roughness>=0 && lengthSquared>MinimumSquared && isfinite(lengthSquared))) return false;
    normal=n.xyz*rsqrt(lengthSquared); return Unproject((float2(pixel)+PixelCenter)*Viewport.zw,depth,surfacePoint);
}
// Stratified disk radii and irrational rotations decorrelate path dimensions.
float2 DiffuseSequence(int2 pixel,uint sampleIndex,uint bounce)
{
    float rotation=frac(dot(float2(pixel)+PixelCenter,float2(sqrt(2.0),sqrt(3.0))));
    return frac(float2((sampleIndex+PixelCenter)/Options.y,rotation+GoldenFraction*sampleIndex)+
        bounce*float2(sqrt(5.0),sqrt(7.0)));
}
float3 CosineDirection(float3 normal,float2 xi)
{
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    float3 tangent=Unit(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
    float radius=sqrt(xi.x),angle=2*Pi*xi.y;
    return Unit(tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+normal*sqrt(1-xi.x),normal);
}
float3 DiffuseWeight(float3 color,float metallic)
{
    float3 result=saturate(color);
    if (Options.z==PBRModel) result*=(1-saturate(metallic))*(1-DielectricF0);
    return result;
}
float3 ReceiverDiffuse(int2 pixel)
{ return DiffuseWeight(DiffuseOpacity.Load(int3(pixel,0)).rgb,AmbientMetallic.Load(int3(pixel,0)).w); }
float3 Estimate(int2 pixel,float3 surfacePoint,float3 normal)
{
    float3 receiver=ReceiverDiffuse(pixel); if (!any(receiver>0)) return 0;
    float3 sum=0;
    [loop] for (uint sampleIndex=0;sampleIndex<Options.y;++sampleIndex)
    {
        float3 throughput=receiver,origin=surfacePoint+normal*Trace.y;
        float3 direction=CosineDirection(normal,DiffuseSequence(pixel,sampleIndex,0));
        [loop] for (uint bounce=0;bounce<Lighting.y;++bounce)
        {
            Hit hit; if (!TraceScene(origin,direction,Trace.x,false,hit)) break;
            HitSurface surface=ReadHit(hit,origin,direction);
            sum+=throughput*min(Shade(surface,-direction),HalfMaximum-HalfSpacing);
            if (bounce+1==Lighting.y) break;
            throughput*=DiffuseWeight(surface.Diffuse,Material.y); if (!any(throughput>0)) break;
            direction=CosineDirection(surface.Normal,DiffuseSequence(pixel,sampleIndex,bounce+1));
            if (dot(surface.Geometric,direction)*dot(surface.Geometric,surface.Normal)<=0) break;
            origin=surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0 ? Trace.y : -Trace.y);
        }
    }
    return min(sum*(Trace.z/Options.y),HalfMaximum-HalfSpacing);
}
float3 Filter(int2 pixel,float3 surfacePoint,float3 normal)
{
    int2 axis=Options.x==HorizontalPass ? int2(1,0) : int2(0,1);
    float3 adjacent; float depth=SceneDepth.Load(int3(pixel,0));
    if (!Unproject((float2(pixel+axis)+PixelCenter)*Viewport.zw,depth,adjacent)) adjacent=surfacePoint;
    float sigma=max(max(Trace.y*DepthBiasScale,length(adjacent-surfacePoint)*DepthFootprintScale),MinimumFilterDepth);
    float3 sum=0,centerDiffuse=ReceiverDiffuse(pixel); float weights=0;
    [unroll] for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        int2 q=pixel+axis*offset; if (any(q<0) || any(q>=int2(Viewport.xy))) continue;
        float3 other,n; float r; if (!Receiver(q,other,n,r)) continue;
        float dz=dot(other-surfacePoint,CameraForward.xyz)/sigma;
        float3 df=(ReceiverDiffuse(q)-centerDiffuse)/ReflectanceSigma;
        float weight=exp(-offset*offset/(2*SpatialSigma*SpatialSigma)-(dz*dz+dot(df,df))/2)*pow(saturate(dot(normal,n)),NormalPower);
        sum+=PreviousRadiance.Load(int3(q,0)).rgb*weight; weights+=weight;
    }
    return sum/max(weights,MinimumWeight);
}
float4 PSMain(ScreenVertex input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy);
    if (Options.x==CompositePass) return float4(PreviousRadiance.Load(int3(pixel,0)).rgb,0);
    float3 surfacePoint,normal; float roughness;
    // Keep BVH loops behind a real branch: background normals are invalid rays.
    [branch] if (!Receiver(pixel,surfacePoint,normal,roughness)) return 0;
    [branch] if (Trace.z<=0) return 0;
    float3 value;
    [branch] if (Options.x==TracePass) value=Estimate(pixel,surfacePoint,normal);
    else value=Filter(pixel,surfacePoint,normal);
    if (Options.x==VerticalPass) value=min(value,max(HalfMaximum-HalfSpacing-SceneColor.Load(int3(pixel,0)).rgb,0));
    return float4(value,0);
}
