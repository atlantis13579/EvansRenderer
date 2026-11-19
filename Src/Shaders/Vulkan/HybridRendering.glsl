#version 450
// Hybrid primary rasterization and stochastic screen/scene reflections.
// Screen radiance and scene-hit lighting share one sampled GGX direction.
// https://gpuopen.com/fidelityfx-hybrid-reflections/
// Runtime-compiled Vulkan GLSL: scene intersections, hit shading and spatial
// reconstruction. Screen intersections can reuse the opaque camera radiance.
// Heitz, GGX VNDF (2018): https://jcgt.org/published/0007/04/01/ .
// Woop et al., ray-aligned triangles (2013): https://jcgt.org/published/0002/01/05/ .
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=0,std140,row_major) uniform ReflectionParameters
{
    mat4 InverseViewProjection;
    vec4 Viewport,Trace,CameraForward,Ambient,Material;
    uvec4 SceneCounts,ShadingCounts,Options,Lighting;
    mat4 ViewProjection;
    vec4 ScreenTrace; // World thickness, pixel stride, edge fade pixels, reserved.
    uvec4 HybridOptions; // Screen tracing, scene fallback, maximum screen steps, reserved.
} Frame;
layout(set=0,binding=1) uniform texture2D SceneDepth;
layout(set=0,binding=2) uniform texture2D NormalRoughness;
layout(set=0,binding=3) uniform texture2D SpecularShininess;
layout(set=0,binding=4) uniform texture2D DiffuseOpacity;
layout(set=0,binding=5) uniform texture2D AmbientMetallic;
layout(set=0,binding=6) uniform texture2D PreviousRadiance;
layout(set=0,binding=7) uniform texture2D SceneColor;
layout(set=0,binding=8) uniform sampler PointSampler;
struct BvhNode { vec4 Minimum,Maximum; uvec4 Links; };
struct Triangle { vec4 A,B,C,UVAB,UVC; uvec4 Metadata; };
struct CutoutMaterial { uvec4 Diffuse,Opacity; vec4 Coverage; };
struct Surface { vec4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct MaterialRecord { vec4 Base,AmbientColor,Specular,Emission; uvec4 Diffuse,Normal; };
struct LightRecord { vec4 PositionType,DirectionIntensity,ColorShadow,Area; };
layout(set=0,binding=9,std430) readonly buffer NodeStorage { BvhNode Nodes[]; };
layout(set=0,binding=10,std430) readonly buffer TriangleStorage { Triangle Triangles[]; };
layout(set=0,binding=11,std430) readonly buffer CutoutStorage { CutoutMaterial Cutouts[]; };
layout(set=0,binding=12,std430) readonly buffer AlphaStorage { float AlphaTexels[]; };
layout(set=0,binding=13,std430) readonly buffer SurfaceStorage { Surface Surfaces[]; };
layout(set=0,binding=14,std430) readonly buffer MaterialStorage { MaterialRecord Materials[]; };
// Preserve the Float4 atlas exactly without depending on maxStorageBufferRange.
layout(set=0,binding=15) uniform texture2D ColorAtlas;
layout(set=0,binding=16,std430) readonly buffer LightStorage { LightRecord Lights[]; };
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

const uint TracePass=0u,HorizontalPass=1u,VerticalPass=2u,CompositePass=3u,PBRModel=3u,ToonModel=4u;
const uint FlatNormals=1u,ShadowCaster=1u,DirectionalLight=0u,PointLight=1u,LegacyAreaGrid=2u;
const float Pi=3.141592653589793,GoldenFraction=(3.0-sqrt(5.0))/2.0;
const float PixelCenter=0.5,InteriorDepth=0.5,ClearDepth=1.0,MinimumW=1e-7,MinimumSquared=1e-16;
const float FloatEpsilon=1.1920928955078125e-7,EdgeUncertainty=3.0*FloatEpsilon/(1.0-3.0*FloatEpsilon);
const float TangentEpsilon=1e-8,ParallelAxis=0.999,MinimumAlpha=0.0025,MinimumCosine=1e-6;
const float DielectricF0=0.04,SchlickExponent=5.0,LightDistanceFloor=0.01;
const float ReflectanceSigma=0.05;
const float HalfMaximum=65504.0,HalfSpacing=32.0,SpatialSigma=1.5,NormalExponent=32.0,RoughnessSigma=0.1;
const float DepthBiasScale=4.0,DepthFootprintScale=2.0,MinimumDepthSigma=1e-5,MinimumWeight=1e-7;
const int FilterRadius=2;
bool Finite(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 Unit(vec3 value,vec3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (!(scale>0.0) || isinf(scale) || isnan(scale)) return fallback;
    value/=scale; return value*inversesqrt(dot(value,value));
}
void AtlasAddresses(uvec4 image,vec2 uv,out uvec4 addresses,out vec2 weight)
{
    vec2 position=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uvec2 a=uvec2(position.x<0.0 ? image.y-1u : uint(floor(position.x)),position.y<0.0 ? image.z-1u : uint(floor(position.y)));
    uvec2 b=uvec2(a.x+1u==image.y ? 0u : a.x+1u,a.y+1u==image.z ? 0u : a.y+1u);
    addresses=image.x+uvec4(a.y*image.y+a.x,a.y*image.y+b.x,b.y*image.y+a.x,b.y*image.y+b.x); weight=fract(position);
}
float Alpha(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0; uvec4 p; vec2 w; AtlasAddresses(image,uv,p,w);
    return mix(mix(AlphaTexels[p.x],AlphaTexels[p.y],w.x),mix(AlphaTexels[p.z],AlphaTexels[p.w],w.x),w.y);
}
vec3 ReadColorTexel(uint index)
{
    uint rowWidth=uint(textureSize(sampler2D(ColorAtlas,PointSampler),0).x);
    return texelFetch(sampler2D(ColorAtlas,PointSampler),ivec2(index%rowWidth,index/rowWidth),0).rgb;
}
vec3 Color(uvec4 image,vec2 uv)
{
    if (image.w==0u) return vec3(1.0); uvec4 p; vec2 w; AtlasAddresses(image,uv,p,w);
    return mix(mix(ReadColorTexel(p.x),ReadColorTexel(p.y),w.x),mix(ReadColorTexel(p.z),ReadColorTexel(p.w),w.x),w.y);
}
bool BoundsHit(BvhNode node,vec3 origin,vec3 direction,float limit)
{
    float enter=0.0,leave=limit;
    for (uint axis=0u;axis<3u;++axis)
    {
        if (direction[axis]==0.0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(vec2 a,vec2 b)
{ precise float positive=a.x*b.y,negative=a.y*b.x; precise float edge=positive-negative; return abs(edge)<=EdgeUncertainty*(abs(positive)+abs(negative)) ? 0.0 : edge; }
bool Intersect(Triangle triangle,vec3 origin,vec3 direction,float limit,out float distance,out vec3 barycentric)
{
    distance=0.0; barycentric=vec3(0.0);
    uint z=abs(direction.x)>abs(direction.y) ? 0u : 1u; if (abs(direction.z)>abs(direction[z])) z=2u;
    uint x=(z+1u)%3u,y=(x+1u)%3u;
    vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(direction[x],direction[y])/direction[z];
    precise vec2 pa=vec2(a[x],a[y])-a[z]*shear,pb=vec2(b[x],b[y])-b[z]*shear,pc=vec2(c[x],c[y])-c[z]*shear;
    vec3 edge=vec3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(lessThan(edge,vec3(0.0))) && any(greaterThan(edge,vec3(0.0)))) return false;
    float area=edge.x+edge.y+edge.z; if (area==0.0) return false;
    distance=dot(edge,vec3(a[z],b[z],c[z]))/(area*direction[z]);
    if (!(distance>0.0 && distance<=limit)) return false; barycentric=edge/area; return true;
}
struct Hit { uint Index; float Distance; vec3 Weights; };
bool TraceScene(vec3 origin,vec3 direction,float limit,bool shadow,out Hit closest)
{
    closest=Hit(0u,limit,vec3(0.0)); uint nodeIndex=0u; bool found=false;
    while (nodeIndex<Frame.SceneCounts.x)
    {
        BvhNode node=Nodes[nodeIndex];
        if (!BoundsHit(node,origin,direction,closest.Distance)) { nodeIndex=node.Links.w; continue; }
        if (node.Links.z==0u) { nodeIndex=node.Links.x; continue; }
        for (uint offset=0u;offset<node.Links.z;++offset)
        {
            uint index=node.Links.y+offset; Triangle triangle=Triangles[index];
            if (shadow && (triangle.Metadata.w&ShadowCaster)==0u) continue;
            float distance; vec3 weights;
            if (!Intersect(triangle,origin,direction,closest.Distance,distance,weights)) continue;
            CutoutMaterial material=Cutouts[triangle.Metadata.x];
            if (material.Coverage.y>0.0)
            {
                vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float coverage=clamp(dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x*
                    Alpha(material.Diffuse,uv)*Alpha(material.Opacity,uv),0.0,1.0);
                if (coverage<material.Coverage.y) continue;
            }
            closest=Hit(index,distance,weights); found=true; if (shadow) return true;
        }
        nodeIndex=node.Links.w;
    }
    return found;
}
struct HitSurface { vec3 Position,Normal,Geometric,Diffuse,Specular,AmbientColor,Emission; float Shininess; };
HitSurface ReadHit(Hit hit,vec3 origin,vec3 direction)
{
    Triangle triangle=Triangles[hit.Index]; Surface surface=Surfaces[hit.Index]; MaterialRecord material=Materials[triangle.Metadata.x];
    vec3 w=hit.Weights,ab=triangle.B.xyz-triangle.A.xyz,ac=triangle.C.xyz-triangle.A.xyz;
    vec2 uv=triangle.UVAB.xy*w.x+triangle.UVAB.zw*w.y+triangle.UVC.xy*w.z;
    HitSurface result; result.Position=origin+direction*hit.Distance;
    result.Geometric=Unit(cross(Unit(ab,vec3(1,0,0)),Unit(ac,vec3(0,1,0))),vec3(0,0,1));
    vec3 normal=(Frame.Options.w&FlatNormals)!=0u ? result.Geometric : Unit(surface.NormalA.xyz*w.x+surface.NormalB.xyz*w.y+surface.NormalC.xyz*w.z,result.Geometric);
    if (dot(result.Geometric,direction)>0.0) normal=-normal;
    if (material.Normal.w!=0u)
    {
        vec2 du=vec2(triangle.UVAB.z-triangle.UVAB.x,triangle.UVAB.y-triangle.UVAB.w);
        vec2 dv=vec2(triangle.UVC.x-triangle.UVAB.x,triangle.UVAB.y-triangle.UVC.y);
        float determinant=du.x*dv.y-du.y*dv.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            vec3 tangent=(ab*dv.y-ac*du.y)/determinant,sourceBitangent=(ac*du.x-ab*dv.x)/determinant;
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            tangent=Unit(tangent-normal*dot(tangent,normal),Unit(cross(helper,normal),vec3(1,0,0)));
            vec3 crossed=cross(normal,tangent),bitangent=crossed*(dot(crossed,sourceBitangent)<0.0 ? -1.0 : 1.0);
            vec3 mapped=Color(material.Normal,uv)*2.0-1.0;
            if (float(Frame.Options.z) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=ToonAlbedo(max((surface.ColorA.rgb*w.x+surface.ColorB.rgb*w.y+surface.ColorC.rgb*w.z)*material.Base.rgb*Color(material.Diffuse,uv),vec3(0.0)), float(Frame.Options.z));
    result.Specular=material.Specular.rgb; result.Shininess=material.Specular.w; result.AmbientColor=material.AmbientColor.rgb; result.Emission=material.Emission.rgb;
    return result;
}
vec3 Fresnel(vec3 base,float cosine) { return base+(1.0-base)*pow(1.0-clamp(cosine,0.0,1.0),SchlickExponent); }
float Smith(float cosine,float alpha)
{ return 2.0*cosine/max(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine),MinimumCosine); }
vec3 BrdfCosine(HitSurface surface,vec3 view,vec3 direction)
{
    float nl=dot(surface.Normal,direction),nv=dot(surface.Normal,view); if (nl<=0.0 || nv<=0.0) return vec3(0.0);
    vec3 halfVector=Unit(view+direction,surface.Normal); float nh=clamp(dot(surface.Normal,halfVector),0.0,1.0);
    if (Frame.Options.z!=PBRModel)
    {
        float highlight=pow(nh,surface.Shininess);
        if (Frame.Options.z==ToonModel) return surface.Diffuse * ToonResponse(max(nl, 0.0), Frame.Material.z) + surface.Specular * ToonHighlight(highlight);
        return surface.Diffuse*nl+surface.Specular*highlight;
    }
    float alpha=max(Frame.Material.x*Frame.Material.x,MinimumAlpha),a2=alpha*alpha;
    float denominator=nh*nh*(a2-1.0)+1.0,distribution=a2/(Pi*denominator*denominator);
    vec3 f=Fresnel(mix(vec3(DielectricF0),clamp(surface.Diffuse,0.0,1.0),Frame.Material.y),dot(view,halfVector));
    vec3 specular=f*distribution*Smith(nv,alpha)*Smith(nl,alpha)/max(4.0*nv*nl,MinimumCosine);
    return ((1.0-f)*(1.0-Frame.Material.y)*surface.Diffuse/Pi+specular)*nl;
}
vec3 Illuminate(HitSurface surface,vec3 view,vec3 emitter,vec3 energy,vec3 emitterNormal,float weight,bool distant,bool area,bool shadow,float range)
{
    vec3 delta=distant ? emitter*Frame.CameraForward.w : emitter-surface.Position;
    float squared=dot(delta,delta); if (!(squared>MinimumSquared)) return vec3(0.0);
    float distance=sqrt(squared); vec3 direction=delta/distance; float attenuation=1.0;
    if (!distant)
    {
        float window=1.0;
        if (range>0.0) { float relative=distance/range; window=clamp(1.0-relative*relative*relative*relative,0.0,1.0); window*=window; }
        attenuation=weight*window*(area ? max(dot(emitterNormal,-direction),0.0) : 1.0)/max(squared,LightDistanceFloor);
    }
    vec3 radiance=BrdfCosine(surface,view,direction)*energy*attenuation;
    if (!any(greaterThan(radiance,vec3(0.0)))) return vec3(0.0);
    if (Frame.Lighting.x!=0u && shadow)
    {
        vec3 origin=surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0.0 ? Frame.Trace.y : -Frame.Trace.y);
        vec3 segment=surface.Position+delta-origin; float extent=length(segment); Hit blocker;
        if (extent>Frame.Trace.y && TraceScene(origin,segment/extent,extent-Frame.Trace.y,true,blocker)) return vec3(0.0);
    }
    return radiance;
}
vec3 Shade(HitSurface surface,vec3 view)
{
    float diffuseFraction=Frame.Options.z==PBRModel ? 1.0-Frame.Material.y : 1.0;
    vec3 result=surface.Emission+surface.Diffuse*max(surface.AmbientColor,vec3(0))*max(Frame.Ambient.rgb,vec3(0.0))*diffuseFraction;
    for (uint index=0u;index<Frame.ShadingCounts.w;++index)
    {
        LightRecord light=Lights[index]; uint type=uint(light.PositionType.w); bool shadows=light.ColorShadow.w>0.0;
        vec3 energy=max(light.ColorShadow.rgb,vec3(0.0))*max(light.DirectionIntensity.w,0.0);
        if (!any(greaterThan(energy,vec3(0.0)))) continue;
        if (type==DirectionalLight)
            result+=Illuminate(surface,view,Unit(light.DirectionIntensity.xyz,vec3(0,1,0)),energy,vec3(0),1.0,true,false,shadows,0.0);
        else if (type==PointLight)
            result+=Illuminate(surface,view,light.PositionType.xyz,energy,vec3(0),1.0,false,false,shadows,light.Area.z);
        else
        {
            vec3 normal=Unit(light.DirectionIntensity.xyz,vec3(0,-1,0));
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=Unit(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
            uint axis=light.Area.w>0.0 ? uint(light.Area.w) : LegacyAreaGrid;
            float area=light.Area.w>0.0 ? abs(light.Area.x*light.Area.y) : 1.0;
            for (uint y=0u;y<axis;++y) for (uint x=0u;x<axis;++x)
            {
                vec2 uv=(vec2(x,y)+PixelCenter)/float(axis)-PixelCenter;
                vec3 emitter=light.PositionType.xyz+tangent*(uv.x*light.Area.x)+bitangent*(uv.y*light.Area.y);
                result+=Illuminate(surface,view,emitter,energy,normal,area/float(axis*axis),false,true,shadows,light.Area.z);
            }
        }
    }
    return clamp(result,0.0,HalfMaximum-HalfSpacing);
}
bool Unproject(vec2 uv,float depth,out vec3 position)
{
    position=vec3(0.0); vec4 world=Frame.InverseViewProjection*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!(abs(world.w)>MinimumW)) return false; position=world.xyz/world.w; return Finite(position);
}
bool Receiver(ivec2 pixel,out vec3 position,out vec3 normal,out float roughness)
{
    position=normal=vec3(0.0); vec4 value=texelFetch(sampler2D(NormalRoughness,PointSampler),pixel,0); roughness=value.w;
    float squared=dot(value.xyz,value.xyz),depth=texelFetch(sampler2D(SceneDepth,PointSampler),pixel,0).r;
    if (!(depth<ClearDepth && roughness>=0.0 && squared>MinimumSquared) || isnan(squared) || isinf(squared)) return false;
    normal=value.xyz*inversesqrt(squared); return Unproject((vec2(pixel)+PixelCenter)*Frame.Viewport.zw,depth,position);
}
vec3 VisibleNormal(vec3 view,float alpha,vec2 random)
{
    vec3 stretched=Unit(vec3(view.xy*alpha,view.z),vec3(0,0,1));
    vec3 tangent=Unit(vec3(-stretched.y,stretched.x,0.0),vec3(1,0,0)),bitangent=cross(stretched,tangent);
    float radius=sqrt(random.x),angle=2.0*Pi*random.y;
    float x=radius*cos(angle),y=radius*sin(angle),blend=(1.0+stretched.z)/2.0;
    y=(1.0-blend)*sqrt(max(0.0,1.0-x*x))+blend*y;
    vec3 hemisphere=tangent*x+bitangent*y+stretched*sqrt(max(0.0,1.0-x*x-y*y));
    return Unit(vec3(hemisphere.xy*alpha,max(hemisphere.z,0.0)),vec3(0,0,1));
}
vec3 ReceiverF0(ivec2 pixel)
{
    if (Frame.Options.z==PBRModel) return mix(vec3(DielectricF0),clamp(texelFetch(sampler2D(DiffuseOpacity,PointSampler),pixel,0).rgb,0.0,1.0),
        clamp(texelFetch(sampler2D(AmbientMetallic,PointSampler),pixel,0).w,0.0,1.0));
    return clamp(texelFetch(sampler2D(SpecularShininess,PointSampler),pixel,0).rgb,0.0,1.0);
}
const float ScreenStartPixels=2.0,ScreenRootEpsilon=1e-7,ScreenDistanceFadeStart=0.8;
bool ClipScreenPlane(float a,float b,inout float first,inout float last)
{
    if (a<0.0 && b<0.0) return false;
    if ((a<0.0)!=(b<0.0)) { float t=a/(a-b); if (a<0.0) first=max(first,t); else last=min(last,t); }
    return first<last;
}
bool ClipScreenSegment(inout vec3 start,inout vec3 finish)
{
    vec4 a=Frame.ViewProjection*vec4(start,1.0),b=Frame.ViewProjection*vec4(finish,1.0);
    if (any(isnan(a)) || any(isinf(a)) || any(isnan(b)) || any(isinf(b))) return false;
    float first=0.0,last=1.0;
    if (!ClipScreenPlane(a.w-MinimumW,b.w-MinimumW,first,last) ||
        !ClipScreenPlane(a.w+a.x,b.w+b.x,first,last) || !ClipScreenPlane(a.w-a.x,b.w-b.x,first,last) ||
        !ClipScreenPlane(a.w+a.y,b.w+b.y,first,last) || !ClipScreenPlane(a.w-a.y,b.w-b.y,first,last) ||
        !ClipScreenPlane(a.z,b.z,first,last) || !ClipScreenPlane(a.w-a.z,b.w-b.z,first,last)) return false;
    vec3 segment=finish-start; finish=start+segment*last; start+=segment*first; return true;
}
bool ReadScreenSurface(ivec2 pixel,out vec3 point,out vec3 normal,out vec3 eyeRay)
{
    point=normal=eyeRay=vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Frame.Viewport.xy)))) return false;
    float roughness; if (!Receiver(pixel,point,normal,roughness)) return false;
    vec2 uv=(vec2(pixel)+PixelCenter)*Frame.Viewport.zw; vec3 nearPoint,interior;
    if (!Unproject(uv,0.0,nearPoint) || !Unproject(uv,InteriorDepth,interior)) return false;
    eyeRay=Unit(interior-nearPoint,vec3(0.0)); return dot(eyeRay,eyeRay)>0.0;
}
vec3 ReadCameraRadiance(vec2 uv)
{
    vec2 coordinate=uv*Frame.Viewport.xy-PixelCenter,blend=fract(coordinate);
    ivec2 p=ivec2(floor(coordinate)),last=ivec2(Frame.Viewport.xy)-1;
    vec3 a=texelFetch(sampler2D(SceneColor,PointSampler),clamp(p,ivec2(0),last),0).rgb;
    vec3 b=texelFetch(sampler2D(SceneColor,PointSampler),clamp(p+ivec2(1,0),ivec2(0),last),0).rgb;
    vec3 c=texelFetch(sampler2D(SceneColor,PointSampler),clamp(p+ivec2(0,1),ivec2(0),last),0).rgb;
    vec3 d=texelFetch(sampler2D(SceneColor,PointSampler),clamp(p+1,ivec2(0),last),0).rgb;
    return max(mix(mix(a,b,blend.x),mix(c,d,blend.x),blend.y),vec3(0.0));
}
float ScreenIncoming(vec3 origin,vec3 ray,out vec3 radiance)
{
    radiance=vec3(0.0); if (Frame.HybridOptions.x==0u) return 0.0;
    vec3 start=origin,finish=origin+ray*Frame.Trace.x; if (!ClipScreenSegment(start,finish)) return 0.0;
    vec4 a=Frame.ViewProjection*vec4(start,1.0),b=Frame.ViewProjection*vec4(finish,1.0);
    float ka=1.0/a.w,kb=1.0/b.w; vec3 qa=start*ka,qb=finish*kb;
    // Viewport storage is top-left; matrices still use renderer clip coordinates.
    vec2 pa=(a.xy*ka*vec2(0.5,-0.5)+PixelCenter)*Frame.Viewport.xy;
    vec2 pb=(b.xy*kb*vec2(0.5,-0.5)+PixelCenter)*Frame.Viewport.xy;
    float major=max(abs(pb.x-pa.x),abs(pb.y-pa.y)); if (major<=ScreenStartPixels) return 0.0;
    float stride=Frame.ScreenTrace.y/major,first=ScreenStartPixels/major;
    for (uint step=0u;step<Frame.HybridOptions.z;++step)
    {
        float lo=first+float(step)*stride; if (lo>=1.0) break; float hi=min(lo+stride,1.0),mid=(lo+hi)/2.0;
        ivec2 pixel=ivec2(floor(mix(pa,pb,mid))); vec3 point,n,eyeRay;
        if (!ReadScreenSurface(pixel,point,n,eyeRay) || dot(n,-ray)<=0.0) continue;
        float da=dot(qa-point*ka,eyeRay),db=dot(qb-point*kb,eyeRay);
        float za=mix(da,db,lo)/mix(ka,kb,lo),zb=mix(da,db,hi)/mix(ka,kb,hi);
        if (max(za,zb)<0.0 || min(za,zb)>Frame.ScreenTrace.x) continue;
        float slope=db-da,t=abs(slope)>ScreenRootEpsilon ? -da/slope : mid; t=clamp(t,lo,hi);
        vec2 hitPixel=mix(pa,pb,t); ivec2 refined=ivec2(floor(hitPixel));
        if (!ReadScreenSurface(refined,point,n,eyeRay) || dot(n,-ray)<=0.0) continue;
        vec3 hit=mix(qa,qb,t)/mix(ka,kb,t); float dz=dot(hit-point,eyeRay);
        if (dz < -ScreenRootEpsilon || dz>Frame.ScreenTrace.x) continue;
        vec2 uv=hitPixel*Frame.Viewport.zw,edge=min(uv,1.0-uv)*Frame.Viewport.xy;
        float confidence=Frame.ScreenTrace.z>0.0 ? clamp(min(edge.x,edge.y)/Frame.ScreenTrace.z,0.0,1.0) : 1.0;
        confidence*=1.0-smoothstep(Frame.Trace.x*ScreenDistanceFadeStart,Frame.Trace.x,length(hit-origin));
        radiance=ReadCameraRadiance(uv); return confidence;
    }
    return 0.0;
}
vec3 Estimate(ivec2 pixel,vec3 position,vec3 normal,float roughness)
{
    if (Frame.Trace.z<=0.0 || roughness>=Frame.Trace.w) return vec3(0.0);
    vec2 uv=(vec2(pixel)+PixelCenter)*Frame.Viewport.zw; vec3 nearPoint,interior;
    if (!Unproject(uv,0.0,nearPoint) || !Unproject(uv,InteriorDepth,interior)) return vec3(0.0);
    vec3 view=Unit(nearPoint-interior,vec3(0)); float nv=dot(normal,view); if (nv<=0.0) return vec3(0.0);
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    vec3 tangent=Unit(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
    vec3 localView=vec3(dot(view,tangent),dot(view,bitangent),nv);
    vec3 f0=ReceiverF0(pixel);
    float alpha=max(roughness*roughness,MinimumAlpha),phase=fract(dot(vec2(pixel)+PixelCenter,vec2(sqrt(2.0),sqrt(3.0))));
    vec3 sum=vec3(0.0),origin=position+normal*Frame.Trace.y;
    for (uint index=0u;index<Frame.Options.y;++index)
    {
        vec2 random=vec2((float(index)+PixelCenter)/float(Frame.Options.y),fract(phase+GoldenFraction*float(index)));
        vec3 sampled=VisibleNormal(localView,alpha,random),halfVector=tangent*sampled.x+bitangent*sampled.y+normal*sampled.z;
        vec3 direction=Unit(reflect(-view,halfVector),normal); float nl=dot(normal,direction); if (nl<=0.0) continue;
        vec3 screen; float confidence=ScreenIncoming(origin,direction,screen);
        vec3 incoming=screen*confidence;
        if (confidence<1.0 && Frame.HybridOptions.y!=0u)
        {
            Hit hit; if (TraceScene(origin,direction,Frame.Trace.x,false,hit))
                incoming+=Shade(ReadHit(hit,origin,direction),-direction)*(1.0-confidence);
        }
        // VNDF PDF cancels the outgoing masking term of separable Smith G2.
        sum+=incoming*Fresnel(f0,dot(view,halfVector))*Smith(nl,alpha);
    }
    return min(sum*(Frame.Trace.z/float(Frame.Options.y)),vec3(HalfMaximum-HalfSpacing));
}
vec3 Filter(ivec2 pixel,vec3 position,vec3 normal,float roughness)
{
    ivec2 axis=Frame.Options.x==HorizontalPass ? ivec2(1,0) : ivec2(0,1); vec3 adjacent;
    float depth=texelFetch(sampler2D(SceneDepth,PointSampler),pixel,0).r;
    if (!Unproject((vec2(pixel+axis)+PixelCenter)*Frame.Viewport.zw,depth,adjacent)) adjacent=position;
    float sigma=max(max(Frame.Trace.y*DepthBiasScale,length(adjacent-position)*DepthFootprintScale),MinimumDepthSigma);
    vec3 sum=vec3(0.0),centerF0=ReceiverF0(pixel); float total=0.0;
    for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        ivec2 q=pixel+axis*offset; if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(Frame.Viewport.xy)))) continue;
        vec3 other,n; float r; if (!Receiver(q,other,n,r)) continue;
        float dz=dot(other-position,Frame.CameraForward.xyz)/sigma,dr=(r-roughness)/RoughnessSigma;
        vec3 df=(ReceiverF0(q)-centerF0)/ReflectanceSigma;
        float weight=exp(-float(offset*offset)/(2.0*SpatialSigma*SpatialSigma)-(dz*dz+dr*dr+dot(df,df))/2.0)*pow(clamp(dot(normal,n),0.0,1.0),NormalExponent);
        sum+=texelFetch(sampler2D(PreviousRadiance,PointSampler),q,0).rgb*weight; total+=weight;
    }
    return sum/max(total,MinimumWeight);
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (Frame.Options.x==CompositePass) { OutColor=vec4(texelFetch(sampler2D(PreviousRadiance,PointSampler),pixel,0).rgb,0); return; }
    vec3 position,normal; float roughness;
    if (!Receiver(pixel,position,normal,roughness)) { OutColor=vec4(0); return; }
    if (Frame.Trace.z<=0.0 || roughness>=Frame.Trace.w) { OutColor=vec4(0); return; }
    vec3 value;
    if (Frame.Options.x==TracePass) value=Estimate(pixel,position,normal,roughness);
    else value=Filter(pixel,position,normal,roughness);
    if (Frame.Options.x==VerticalPass)
        value=min(value,max(vec3(HalfMaximum-HalfSpacing)-texelFetch(sampler2D(SceneColor,PointSampler),pixel,0).rgb,vec3(0)));
    OutColor=vec4(value,0);
}
#endif
