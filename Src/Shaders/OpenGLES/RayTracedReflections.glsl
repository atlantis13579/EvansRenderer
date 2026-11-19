#version 310 es
// ES 3.1 scene reflections require seven fragment SSBOs. Storage limits are a
// device capability, not an assumption about every ES 3.1 implementation.
// GGX VNDF: Heitz (2018), https://jcgt.org/published/0007/04/01/ .
// Ray-aligned FP32 edges: Woop et al. (2013), https://jcgt.org/published/0002/01/05/ .
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
uniform mat4 WorldFromClip;
uniform vec4 FrameSize,RayParameters,EyeForward,AmbientLight,MaterialParameters;
uniform uvec4 GeometryCounts,SurfaceCounts,Control,LightControl;
uniform highp sampler2D DepthImage,NormalImage,SpecularImage,DiffuseImage,AmbientImage,PreviousRadiance,OpaqueColor;
struct NodeData { vec4 Lower,Upper; uvec4 Link; };
struct PrimitiveData { vec4 A,B,C,UVAB,UVC; uvec4 Identity; };
struct CoverageData { uvec4 Diffuse,Opacity; vec4 Factor; };
struct SurfaceData { vec4 NA,NB,NC,CA,CB,CC; };
struct MaterialData { vec4 Base,Ambient,Specular,Emission; uvec4 Diffuse,Normal; };
struct LightData { vec4 PositionType,DirectionIntensity,ColorShadow,Area; };
layout(std430,binding=0) readonly buffer NodesBlock { NodeData Nodes[]; };
layout(std430,binding=1) readonly buffer PrimitivesBlock { PrimitiveData Primitives[]; };
layout(std430,binding=2) readonly buffer CoverageBlock { CoverageData CoverageMaterials[]; };
layout(std430,binding=3) readonly buffer AlphaBlock { float AlphaTexels[]; };
layout(std430,binding=4) readonly buffer SurfacesBlock { SurfaceData Surfaces[]; };
layout(std430,binding=5) readonly buffer MaterialsBlock { MaterialData Materials[]; };
// RGBA32F texel addressing avoids the much smaller single-SSBO byte limit.
uniform highp sampler2D ColorAtlas;
layout(std430,binding=7) readonly buffer LightsBlock { LightData Lights[]; };
layout(location=0) out vec4 OutColor;
// Ray hits retain the Toon palette, shallow normals and colored light ramp.
precision highp float;
precision highp int;
const float ToonShadingModel = 4.0;
const float ToonAlbedoExponent = 0.8;
const float ToonPaletteSteps = 16.0;
const float ToonColorSaturation = 1.12;
const float ToonNormalStrength = 0.12;
const float ToonLightScale = 0.45;
const float ToonBandSoftness = 0.4;
const float ToonShadowFloor = 0.18;
const float ToonHighlightThreshold = 0.8;
const float ToonHighlightStrength = 0.18;
const vec3 ToonLuma = vec3(0.2126, 0.7152, 0.0722);
const vec3 ToonCoolShadow = vec3(0.56, 0.68, 1.0);
const vec3 ToonWarmLight = vec3(1.0, 0.94, 0.82);
vec3 ToonAlbedo(vec3 value, float model)
{
    if (model != ToonShadingModel) return value;
    vec3 paint = pow(clamp(value, 0.0, 1.0), vec3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return clamp(mix(vec3(dot(paint, ToonLuma)), paint, ToonColorSaturation), 0.0, 1.0);
}
vec3 ToonResponse(float cosine, float bands)
{
    float angularBand = clamp(cosine, 0.0, 1.0) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, fract(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return mix(ToonCoolShadow, ToonWarmLight, level) * mix(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

const uint RayStage=0u,HorizontalStage=1u,VerticalStage=2u,CompositeStage=3u;
const uint PBR=3u,Toon=4u,FlatFlag=1u,CasterFlag=1u,Directional=0u,Point=1u,LegacyGrid=2u;
const float Pi=3.141592653589793,SequenceStep=(3.0-sqrt(5.0))/2.0;
const float PixelCenter=0.5,InteriorDepth=0.5,BackgroundDepth=1.0,ProjectionEpsilon=1e-7,VectorEpsilon=1e-16;
const float FloatSpacing=1.1920928955078125e-7,EdgeError=3.0*FloatSpacing/(1.0-3.0*FloatSpacing);
const float TangentEpsilon=1e-8,ParallelAxis=0.999,AlphaFloor=0.0025,CosineFloor=1e-6;
const float Dielectric=0.04,FresnelPower=5.0,DistanceSquaredFloor=0.01;
const float ReflectanceSigma=0.05;
const float RadianceMaximum=65504.0,RadianceSpacing=32.0,SpatialSigma=1.5,NormalPower=32.0,RoughnessSigma=0.1;
const float BiasSigmaScale=4.0,PixelSigmaScale=2.0,DepthSigmaFloor=1e-5,WeightFloor=1e-7;
const int FilterRadius=2;

// Member loads also support ANGLE drivers whose aggregate SSBO load helpers
// cannot represent a returned structure on every generated HLSL control path.
NodeData ReadNode(uint index)
{ NodeData value; value.Lower=Nodes[index].Lower; value.Upper=Nodes[index].Upper; value.Link=Nodes[index].Link; return value; }
PrimitiveData ReadPrimitive(uint index)
{
    PrimitiveData value; value.A=Primitives[index].A; value.B=Primitives[index].B; value.C=Primitives[index].C;
    value.UVAB=Primitives[index].UVAB; value.UVC=Primitives[index].UVC; value.Identity=Primitives[index].Identity; return value;
}
CoverageData ReadCoverage(uint index)
{
    CoverageData value; value.Diffuse=CoverageMaterials[index].Diffuse; value.Opacity=CoverageMaterials[index].Opacity;
    value.Factor=CoverageMaterials[index].Factor; return value;
}
SurfaceData ReadAttributes(uint index)
{
    SurfaceData value; value.NA=Surfaces[index].NA; value.NB=Surfaces[index].NB; value.NC=Surfaces[index].NC;
    value.CA=Surfaces[index].CA; value.CB=Surfaces[index].CB; value.CC=Surfaces[index].CC; return value;
}
MaterialData ReadMaterial(uint index)
{
    MaterialData value; value.Base=Materials[index].Base; value.Ambient=Materials[index].Ambient;
    value.Specular=Materials[index].Specular; value.Emission=Materials[index].Emission;
    value.Diffuse=Materials[index].Diffuse; value.Normal=Materials[index].Normal; return value;
}
LightData ReadLight(uint index)
{
    LightData value; value.PositionType=Lights[index].PositionType; value.DirectionIntensity=Lights[index].DirectionIntensity;
    value.ColorShadow=Lights[index].ColorShadow; value.Area=Lights[index].Area; return value;
}
vec3 UnitVector(vec3 vector,vec3 fallback)
{
    float largest=max(abs(vector.x),max(abs(vector.y),abs(vector.z)));
    if (!(largest>0.0) || isnan(largest) || isinf(largest)) return fallback;
    vector/=largest; return vector*inversesqrt(dot(vector,vector));
}
void TexelFootprint(uvec4 image,vec2 uv,out uvec4 address,out vec2 blend)
{
    vec2 coordinate=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uint x=coordinate.x<0.0 ? image.y-1u : uint(floor(coordinate.x)),y=coordinate.y<0.0 ? image.z-1u : uint(floor(coordinate.y));
    uint right=x+1u==image.y ? 0u : x+1u,bottom=y+1u==image.z ? 0u : y+1u;
    address=image.x+uvec4(y*image.y+x,y*image.y+right,bottom*image.y+x,bottom*image.y+right); blend=fract(coordinate);
}
float SampleAlpha(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0; uvec4 a; vec2 b; TexelFootprint(image,uv,a,b);
    return mix(mix(AlphaTexels[a.x],AlphaTexels[a.y],b.x),mix(AlphaTexels[a.z],AlphaTexels[a.w],b.x),b.y);
}
vec3 ReadColorTexel(uint index)
{
    uint rowWidth=uint(textureSize(ColorAtlas,0).x);
    return texelFetch(ColorAtlas,ivec2(index%rowWidth,index/rowWidth),0).rgb;
}
vec3 SampleColor(uvec4 image,vec2 uv)
{
    if (image.w==0u) return vec3(1.0); uvec4 a; vec2 b; TexelFootprint(image,uv,a,b);
    return mix(mix(ReadColorTexel(a.x),ReadColorTexel(a.y),b.x),mix(ReadColorTexel(a.z),ReadColorTexel(a.w),b.x),b.y);
}
bool IntersectBounds(NodeData node,vec3 origin,vec3 ray,float maximum)
{
    float enter=0.0,leave=maximum;
    for (uint axis=0u;axis<3u;++axis)
    {
        if (ray[axis]==0.0) { if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false; }
        else
        {
            float lower=(node.Lower[axis]-origin[axis])/ray[axis],upper=(node.Upper[axis]-origin[axis])/ray[axis];
            enter=max(enter,min(lower,upper)); leave=min(leave,max(lower,upper)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(vec2 a,vec2 b)
{
    // ES 3.1 has no core precise qualifier. The conservative band handles the
    // two products' FP32 uncertainty, including drivers that contract them.
    float positive=a.x*b.y,negative=a.y*b.x,edge=positive-negative;
    return abs(edge)<=EdgeError*(abs(positive)+abs(negative)) ? 0.0 : edge;
}
bool IntersectTriangle(PrimitiveData triangle,vec3 origin,vec3 ray,float maximum,out float distance,out vec3 weights)
{
    distance=0.0; weights=vec3(0.0); uint z=abs(ray.x)>abs(ray.y) ? 0u : 1u; if (abs(ray.z)>abs(ray[z])) z=2u;
    uint x=(z+1u)%3u,y=(x+1u)%3u; vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(ray[x],ray[y])/ray[z];
    vec2 pa=vec2(a[x],a[y])-shear*a[z],pb=vec2(b[x],b[y])-shear*b[z],pc=vec2(c[x],c[y])-shear*c[z];
    vec3 edges=vec3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float determinant=edges.x+edges.y+edges.z; if (determinant==0.0) return false;
    distance=dot(edges,vec3(a[z],b[z],c[z]))/(determinant*ray[z]);
    if (!(distance>0.0 && distance<=maximum)) return false; weights=edges/determinant; return true;
}
struct RayHit { uint Index; float Distance; vec3 Weights; };
bool TraceGeometry(vec3 origin,vec3 ray,float maximum,bool shadow,out RayHit hit)
{
    hit=RayHit(0u,maximum,vec3(0.0)); uint cursor=0u; bool found=false;
    while (cursor<GeometryCounts.x)
    {
        NodeData node=ReadNode(cursor);
        if (!IntersectBounds(node,origin,ray,hit.Distance)) { cursor=node.Link.w; continue; }
        if (node.Link.z==0u) { cursor=node.Link.x; continue; }
        for (uint offset=0u;offset<node.Link.z;++offset)
        {
            uint index=node.Link.y+offset; PrimitiveData triangle=ReadPrimitive(index);
            if (shadow && (triangle.Identity.w&CasterFlag)==0u) continue;
            float distance; vec3 weights;
            if (!IntersectTriangle(triangle,origin,ray,hit.Distance,distance,weights)) continue;
            CoverageData material=ReadCoverage(triangle.Identity.x);
            if (material.Factor.y>0.0)
            {
                vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float alpha=clamp(dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Factor.x*
                    SampleAlpha(material.Diffuse,uv)*SampleAlpha(material.Opacity,uv),0.0,1.0);
                if (alpha<material.Factor.y) continue;
            }
            hit=RayHit(index,distance,weights); found=true; if (shadow) return true;
        }
        cursor=node.Link.w;
    }
    return found;
}
struct ShadedPoint { vec3 Position,Normal,Geometric,Diffuse,Specular,Ambient,Emission; float Shininess; };
ShadedPoint HitPoint(RayHit hit,vec3 origin,vec3 ray)
{
    PrimitiveData triangle=ReadPrimitive(hit.Index); SurfaceData surface=ReadAttributes(hit.Index);
    MaterialData material=ReadMaterial(triangle.Identity.x); vec3 weights=hit.Weights;
    vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
    vec3 edgeB=triangle.B.xyz-triangle.A.xyz,edgeC=triangle.C.xyz-triangle.A.xyz;
    ShadedPoint point; point.Position=origin+ray*hit.Distance;
    point.Geometric=UnitVector(cross(UnitVector(edgeB,vec3(1,0,0)),UnitVector(edgeC,vec3(0,1,0))),vec3(0,0,1));
    point.Normal=(Control.w&FlatFlag)!=0u ? point.Geometric : UnitVector(surface.NA.xyz*weights.x+surface.NB.xyz*weights.y+surface.NC.xyz*weights.z,point.Geometric);
    if (dot(point.Geometric,ray)>0.0) point.Normal=-point.Normal;
    if (material.Normal.w!=0u)
    {
        vec2 uvB=vec2(triangle.UVAB.z-triangle.UVAB.x,triangle.UVAB.y-triangle.UVAB.w);
        vec2 uvC=vec2(triangle.UVC.x-triangle.UVAB.x,triangle.UVAB.y-triangle.UVC.y);
        float determinant=uvB.x*uvC.y-uvB.y*uvC.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            vec3 tangent=(edgeB*uvC.y-edgeC*uvB.y)/determinant,sourceBitangent=(edgeC*uvB.x-edgeB*uvC.x)/determinant;
            vec3 helper=abs(point.Normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            tangent=UnitVector(tangent-point.Normal*dot(point.Normal,tangent),UnitVector(cross(helper,point.Normal),vec3(1,0,0)));
            vec3 bitangent=cross(point.Normal,tangent); bitangent*=dot(bitangent,sourceBitangent)<0.0 ? -1.0 : 1.0;
            vec3 mapped=SampleColor(material.Normal,uv)*2.0-1.0;
            if (float(Control.z) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            point.Normal=UnitVector(tangent*mapped.x+bitangent*mapped.y+point.Normal*mapped.z,point.Normal);
        }
    }
    point.Diffuse=ToonAlbedo(max((surface.CA.rgb*weights.x+surface.CB.rgb*weights.y+surface.CC.rgb*weights.z)*material.Base.rgb*SampleColor(material.Diffuse,uv),vec3(0)), float(Control.z));
    point.Specular=material.Specular.rgb; point.Shininess=material.Specular.w; point.Ambient=material.Ambient.rgb; point.Emission=material.Emission.rgb;
    return point;
}
vec3 Schlick(vec3 f0,float cosine) { return f0+(1.0-f0)*pow(1.0-clamp(cosine,0.0,1.0),FresnelPower); }
float SmithG1(float cosine,float alpha)
{ return 2.0*cosine/max(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine),CosineFloor); }
vec3 Response(ShadedPoint point,vec3 view,vec3 incoming)
{
    float nl=dot(point.Normal,incoming),nv=dot(point.Normal,view); if (nl<=0.0 || nv<=0.0) return vec3(0);
    vec3 halfVector=UnitVector(view+incoming,point.Normal); float nh=clamp(dot(point.Normal,halfVector),0.0,1.0);
    if (Control.z!=PBR)
    {
        float highlight=pow(nh,point.Shininess);
        if (Control.z==Toon) return point.Diffuse * ToonResponse(max(nl, 0.0), MaterialParameters.z) + point.Specular * ToonHighlight(highlight);
        return point.Diffuse*nl+point.Specular*highlight;
    }
    float alpha=max(MaterialParameters.x*MaterialParameters.x,AlphaFloor),a2=alpha*alpha;
    float denominator=nh*nh*(a2-1.0)+1.0,distribution=a2/(Pi*denominator*denominator);
    vec3 f=Schlick(mix(vec3(Dielectric),clamp(point.Diffuse,0.0,1.0),MaterialParameters.y),dot(view,halfVector));
    vec3 specular=f*distribution*SmithG1(nv,alpha)*SmithG1(nl,alpha)/max(4.0*nv*nl,CosineFloor);
    return ((1.0-f)*(1.0-MaterialParameters.y)*point.Diffuse/Pi+specular)*nl;
}
vec3 LightSample(ShadedPoint point,vec3 view,vec3 emitter,vec3 emission,vec3 emitterNormal,float measure,bool distant,bool rectangle,bool shadows,float range)
{
    vec3 offset=distant ? emitter*EyeForward.w : emitter-point.Position;
    float squared=dot(offset,offset); if (!(squared>VectorEpsilon)) return vec3(0);
    float distance=sqrt(squared); vec3 incoming=offset/distance; float falloff=1.0;
    if (!distant)
    {
        float window=1.0;
        if (range>0.0) { float relative=distance/range; window=clamp(1.0-relative*relative*relative*relative,0.0,1.0); window*=window; }
        falloff=measure*window*(rectangle ? max(dot(emitterNormal,-incoming),0.0) : 1.0)/max(squared,DistanceSquaredFloor);
    }
    vec3 result=Response(point,view,incoming)*emission*falloff; if (!any(greaterThan(result,vec3(0)))) return vec3(0);
    if (LightControl.x!=0u && shadows)
    {
        vec3 start=point.Position+point.Geometric*(dot(point.Geometric,incoming)>=0.0 ? RayParameters.y : -RayParameters.y);
        vec3 segment=point.Position+offset-start; float extent=length(segment); RayHit blocker;
        if (extent>RayParameters.y && TraceGeometry(start,segment/extent,extent-RayParameters.y,true,blocker)) return vec3(0);
    }
    return result;
}
vec3 Shade(ShadedPoint point,vec3 view)
{
    float diffuseFraction=Control.z==PBR ? 1.0-MaterialParameters.y : 1.0;
    vec3 result=point.Emission+point.Diffuse*max(point.Ambient,vec3(0))*max(AmbientLight.rgb,vec3(0))*diffuseFraction;
    for (uint index=0u;index<SurfaceCounts.w;++index)
    {
        LightData light=ReadLight(index); uint type=uint(light.PositionType.w); bool shadows=light.ColorShadow.w>0.0;
        vec3 emission=max(light.ColorShadow.rgb,vec3(0))*max(light.DirectionIntensity.w,0.0);
        if (!any(greaterThan(emission,vec3(0)))) continue;
        if (type==Directional)
            result+=LightSample(point,view,UnitVector(light.DirectionIntensity.xyz,vec3(0,1,0)),emission,vec3(0),1.0,true,false,shadows,0.0);
        else if (type==Point)
            result+=LightSample(point,view,light.PositionType.xyz,emission,vec3(0),1.0,false,false,shadows,light.Area.z);
        else
        {
            vec3 normal=UnitVector(light.DirectionIntensity.xyz,vec3(0,-1,0));
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=UnitVector(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
            uint grid=light.Area.w>0.0 ? uint(light.Area.w) : LegacyGrid;
            float area=light.Area.w>0.0 ? abs(light.Area.x*light.Area.y) : 1.0;
            for (uint y=0u;y<grid;++y) for (uint x=0u;x<grid;++x)
            {
                vec2 uv=(vec2(x,y)+PixelCenter)/float(grid)-PixelCenter;
                vec3 emitter=light.PositionType.xyz+tangent*(uv.x*light.Area.x)+bitangent*(uv.y*light.Area.y);
                result+=LightSample(point,view,emitter,emission,normal,area/float(grid*grid),false,true,shadows,light.Area.z);
            }
        }
    }
    return clamp(result,0.0,RadianceMaximum-RadianceSpacing);
}
bool Unproject(vec2 uv,float depth,out vec3 point)
{
    point=vec3(0); vec4 homogeneous=WorldFromClip*vec4(uv*2.0-1.0,depth,1.0);
    if (!(abs(homogeneous.w)>ProjectionEpsilon)) return false;
    point=homogeneous.xyz/homogeneous.w; return !any(isnan(point)) && !any(isinf(point));
}
bool Receiver(ivec2 pixel,out vec3 point,out vec3 normal,out float roughness)
{
    point=normal=vec3(0); vec4 n=texelFetch(NormalImage,pixel,0); roughness=n.w;
    float squared=dot(n.xyz,n.xyz),depth=texelFetch(DepthImage,pixel,0).r;
    if (!(depth<BackgroundDepth && roughness>=0.0 && squared>VectorEpsilon) || isnan(squared) || isinf(squared)) return false;
    normal=n.xyz*inversesqrt(squared); return Unproject((vec2(pixel)+PixelCenter)*FrameSize.zw,depth,point);
}
vec3 SampleVisibleNormal(vec3 view,float alpha,vec2 xi)
{
    vec3 stretched=UnitVector(vec3(view.xy*alpha,view.z),vec3(0,0,1));
    vec3 tangent=UnitVector(vec3(-stretched.y,stretched.x,0),vec3(1,0,0)),bitangent=cross(stretched,tangent);
    float radius=sqrt(xi.x),angle=2.0*Pi*xi.y,x=radius*cos(angle),y=radius*sin(angle),blend=(1.0+stretched.z)/2.0;
    y=(1.0-blend)*sqrt(max(0.0,1.0-x*x))+blend*y;
    vec3 hemisphere=tangent*x+bitangent*y+stretched*sqrt(max(0.0,1.0-x*x-y*y));
    return UnitVector(vec3(hemisphere.xy*alpha,max(hemisphere.z,0.0)),vec3(0,0,1));
}
vec3 ReceiverF0(ivec2 pixel)
{
    if (Control.z==PBR) return mix(vec3(Dielectric),clamp(texelFetch(DiffuseImage,pixel,0).rgb,0.0,1.0),clamp(texelFetch(AmbientImage,pixel,0).w,0.0,1.0));
    return clamp(texelFetch(SpecularImage,pixel,0).rgb,0.0,1.0);
}
vec3 Estimate(ivec2 pixel,vec3 point,vec3 normal,float roughness)
{
    if (RayParameters.z<=0.0 || roughness>=RayParameters.w) return vec3(0);
    vec2 uv=(vec2(pixel)+PixelCenter)*FrameSize.zw; vec3 nearPoint,interior;
    if (!Unproject(uv,0.0,nearPoint) || !Unproject(uv,InteriorDepth,interior)) return vec3(0);
    vec3 view=UnitVector(nearPoint-interior,vec3(0)); float nv=dot(normal,view); if (nv<=0.0) return vec3(0);
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    vec3 tangent=UnitVector(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
    vec3 localView=vec3(dot(view,tangent),dot(view,bitangent),nv),f0=ReceiverF0(pixel);
    // Use top-left coordinates only for the sequence, matching the other APIs.
    vec2 canonical=vec2(float(pixel.x),FrameSize.y-1.0-float(pixel.y));
    float alpha=max(roughness*roughness,AlphaFloor),phase=fract(dot(canonical+PixelCenter,vec2(sqrt(2.0),sqrt(3.0))));
    vec3 result=vec3(0),origin=point+normal*RayParameters.y;
    for (uint index=0u;index<Control.y;++index)
    {
        vec2 xi=vec2((float(index)+PixelCenter)/float(Control.y),fract(phase+SequenceStep*float(index)));
        vec3 sampled=SampleVisibleNormal(localView,alpha,xi),halfVector=tangent*sampled.x+bitangent*sampled.y+normal*sampled.z;
        vec3 ray=UnitVector(reflect(-view,halfVector),normal); float nl=dot(normal,ray); if (nl<=0.0) continue;
        RayHit hit; if (!TraceGeometry(origin,ray,RayParameters.x,false,hit)) continue;
        result+=Shade(HitPoint(hit,origin,ray),-ray)*Schlick(f0,dot(view,halfVector))*SmithG1(nl,alpha);
    }
    return min(result*(RayParameters.z/float(Control.y)),vec3(RadianceMaximum-RadianceSpacing));
}
vec3 Filter(ivec2 pixel,vec3 point,vec3 normal,float roughness)
{
    ivec2 axis=Control.x==HorizontalStage ? ivec2(1,0) : ivec2(0,1); vec3 neighbor;
    if (!Unproject((vec2(pixel+axis)+PixelCenter)*FrameSize.zw,texelFetch(DepthImage,pixel,0).r,neighbor)) neighbor=point;
    float sigma=max(max(RayParameters.y*BiasSigmaScale,length(neighbor-point)*PixelSigmaScale),DepthSigmaFloor);
    vec3 result=vec3(0),centerF0=ReceiverF0(pixel); float total=0.0;
    for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        ivec2 q=pixel+axis*offset; if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(FrameSize.xy)))) continue;
        vec3 position,n; float r; if (!Receiver(q,position,n,r)) continue;
        float dz=dot(position-point,EyeForward.xyz)/sigma,dr=(r-roughness)/RoughnessSigma;
        vec3 df=(ReceiverF0(q)-centerF0)/ReflectanceSigma;
        float weight=exp(-float(offset*offset)/(2.0*SpatialSigma*SpatialSigma)-(dz*dz+dr*dr+dot(df,df))/2.0)*pow(clamp(dot(normal,n),0.0,1.0),NormalPower);
        result+=texelFetch(PreviousRadiance,q,0).rgb*weight; total+=weight;
    }
    return result/max(total,WeightFloor);
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (Control.x==CompositeStage) { OutColor=vec4(texelFetch(PreviousRadiance,pixel,0).rgb,0); return; }
    vec3 point,normal; float roughness;
    if (!Receiver(pixel,point,normal,roughness)) { OutColor=vec4(0); return; }
    if (RayParameters.z<=0.0 || roughness>=RayParameters.w) { OutColor=vec4(0); return; }
    vec3 result;
    if (Control.x==RayStage) result=Estimate(pixel,point,normal,roughness);
    else result=Filter(pixel,point,normal,roughness);
    if (Control.x==VerticalStage) result=min(result,max(vec3(RadianceMaximum-RadianceSpacing)-texelFetch(OpaqueColor,pixel,0).rgb,vec3(0)));
    OutColor=vec4(result,0);
}
#endif
