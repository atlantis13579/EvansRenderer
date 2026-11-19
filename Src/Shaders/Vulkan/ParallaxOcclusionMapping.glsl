#version 450
// Independently authored Vulkan material capture. Tatarchuk, Practical Parallax
// Occlusion Mapping (SIGGRAPH 2006): adaptive layers, bracket refinement and
// final profile interpolation. Base geometry retains raster depth/silhouette.
const uint MaximumLights=4u,FlatModel=1u,PhysicalModel=3u;
const uint AutomaticCoordinates=0u,MeshCoordinates=1u,ObjectCoordinates=2u;
const float Pi=3.14159265358979323846;
const float DirectionEpsilon=1e-12,GramRelativeEpsilon=1e-6;
const float IntersectionEpsilon=1e-7,MaximumCoordinateTravel=0.5;
const float MaximumRadiance=65504.0,MinimumRoughness=0.045,OverlayMode=0.5;
struct SceneLight { vec4 PositionType,DirectionIntensity,ColorCastShadows,AreaSizePadding; };
layout(std140,row_major,set=0,binding=0) uniform DrawConstants
{
    mat4 World,NormalWorld,ViewProjection,LightViewProjection;
    vec4 PipelineColor,MaterialDiffuseColor,MaterialAmbientColor,MaterialSpecularColor,MaterialEmissiveColor;
    vec4 MaterialParameters,AmbientColorStrength,CameraLightCount,RenderParameters;
    uvec4 TextureFlags;
    SceneLight Lights[MaximumLights];
    vec4 ShadingParameters;
} Draw;
layout(push_constant) uniform ParallaxFrame { vec4 Shape; uvec4 Steps; vec4 ObjectBounds; } HeightFrame;
vec3 UnitOr(vec3 value,vec3 fallback)
{ float squared=dot(value,value); return squared>DirectionEpsilon?value*inversesqrt(squared):fallback; }
vec3 DecodeSrgb(vec3 color)
{
    const float Threshold=0.04045,LinearScale=12.92,PowerOffset=0.055,PowerScale=1.055,PowerExponent=2.4;
    return mix(pow((color+PowerOffset)/PowerScale,vec3(PowerExponent)),color/LinearScale,lessThanEqual(color,vec3(Threshold)));
}
bool ValidChart(vec2 dx,vec2 dy)
{
    vec2 product=vec2(length(dx),length(dy));
    float scale=product.x*product.y;
    const float RelativeChartEpsilon=1e-6;
    return scale>0.0 && abs(dx.x*dy.y-dx.y*dy.x)>RelativeChartEpsilon*scale;
}
#if defined(EVANS_VERTEX_STAGE)
invariant gl_Position;
precise gl_Position;
layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
layout(location=0) out vec3 WorldPosition;
layout(location=1) out vec3 WorldNormal;
layout(location=2) out vec4 VertexColor;
layout(location=3) out vec2 MaterialUV;
layout(location=4) out vec3 LocalPosition;
void main()
{
    precise vec4 world=Draw.World*vec4(InPosition,1);
    gl_Position=Draw.ViewProjection*world; gl_Position.y=-gl_Position.y;
    WorldPosition=world.xyz/max(abs(world.w),IntersectionEpsilon);
    WorldNormal=UnitOr(mat3(Draw.NormalWorld)*InNormal,vec3(0,1,0));
    VertexColor=vec4(DecodeSrgb(clamp(InColor.rgb,0,1)),InColor.a);
    MaterialUV=vec2(InUV.x,1-InUV.y); LocalPosition=InPosition;
}
#else
layout(location=0) in vec3 WorldPosition;
layout(location=1) in vec3 WorldNormal;
layout(location=2) in vec4 VertexColor;
layout(location=3) in vec2 MaterialUV;
layout(location=4) in vec3 LocalPosition;
layout(set=0,binding=2) uniform texture2D DiffuseImage;
layout(set=0,binding=3) uniform texture2D NormalImage;
layout(set=0,binding=4) uniform texture2D OpacityImage;
layout(set=0,binding=5) uniform sampler MaterialSampler;
layout(location=0) out vec4 DiffuseOpacity;
layout(location=1) out vec4 NormalRoughness;
layout(location=2) out vec4 SpecularShininess;
layout(location=3) out vec4 AmbientMetallic;
layout(location=4) out vec4 Emission;
vec3 SampleHeight(vec2 coordinate,vec2 dx,vec2 dy)
{
    vec2 phase=2*Pi*fract(vec2(coordinate.x+coordinate.y,coordinate.x-coordinate.y));
    vec2 fx=vec2(dx.x+dx.y,dx.x-dx.y),fy=vec2(dy.x+dy.y,dy.x-dy.y);
    vec2 weights=exp(-2*Pi*Pi*(fx*fx+fy*fy)); const float ModeWeight=0.25;
    vec2 slope=-ModeWeight*2*Pi*weights*sin(phase);
    return vec3(0.5+ModeWeight*dot(weights,cos(phase)),slope.x+slope.y,slope.x-slope.y);
}
vec2 ObjectChart(vec3 coordinate,vec3 dx,vec3 dy)
{
    vec3 axis=abs(cross(dx,dy)); vec3 local=(coordinate-HeightFrame.ObjectBounds.xyz)*HeightFrame.ObjectBounds.w;
    return axis.z>=max(axis.x,axis.y)?local.xy:axis.y>=axis.x?local.xz:local.yz;
}
struct SurfaceFrame { vec3 du,dv; vec2 ray; float thickness,cosine; bool valid; };
SurfaceFrame BuildFrame(vec3 dx,vec3 dy,vec2 hx,vec2 hy,vec3 normal,vec3 eye)
{
    SurfaceFrame frame=SurfaceFrame(vec3(0),vec3(0),vec2(0),0,0,false);
    float determinant=hx.x*hy.y-hx.y*hy.x;
    if (!ValidChart(hx,hy)) return frame;
    frame.du=(dx*hy.y-dy*hx.y)/determinant; frame.dv=(dy*hx.x-dx*hy.x)/determinant;
    frame.du-=normal*dot(normal,frame.du); frame.dv-=normal*dot(normal,frame.dv);
    float uu=dot(frame.du,frame.du),uv=dot(frame.du,frame.dv),vv=dot(frame.dv,frame.dv),gram=uu*vv-uv*uv;
    if (min(uu,vv)<=DirectionEpsilon || gram<=GramRelativeEpsilon*uu*vv) return frame;
    frame.cosine=max(dot(normal,eye),0);
    frame.thickness=HeightFrame.Shape.x*sqrt(min(uu,vv))*smoothstep(HeightFrame.Shape.w,2*HeightFrame.Shape.w,frame.cosine);
    vec3 travel=-eye*frame.thickness/max(frame.cosine,HeightFrame.Shape.w);
    float a=dot(frame.du,travel),b=dot(frame.dv,travel);
    frame.ray=vec2(a*vv-b*uv,b*uu-a*uv)/gram;
    float distance=length(frame.ray),limit=distance>MaximumCoordinateTravel?MaximumCoordinateTravel/distance:1;
    frame.ray*=limit; frame.thickness*=limit; frame.valid=true; return frame;
}
vec3 IntersectHeight(vec2 start,vec2 direction,vec2 dx,vec2 dy,float cosine)
{
    uint count=uint(ceil(mix(float(HeightFrame.Steps.y),float(HeightFrame.Steps.x),clamp(cosine,0,1))));
    float low=0,high=0,lowError=SampleHeight(start,dx,dy).x-1,highError=lowError;
    if (lowError<0)
    {
        for (uint layer=1u;layer<=count;++layer)
        {
            high=float(layer)/float(count); highError=high+SampleHeight(start+direction*high,dx,dy).x-1;
            if (highError>=0) break;
            low=high; lowError=highError;
        }
        for (uint iteration=0u;iteration<HeightFrame.Steps.z;++iteration)
        {
            float middle=(low+high)/2,error=middle+SampleHeight(start+direction*middle,dx,dy).x-1;
            if (error>=0) { high=middle; highError=error; } else { low=middle; lowError=error; }
        }
    }
    float difference=highError-lowError;
    float depth=mix(low,high,abs(difference)>IntersectionEpsilon?clamp(-lowError/difference,0,1):0);
    return vec3(start+direction*depth,depth);
}
void main()
{
    vec3 px=dFdx(WorldPosition),py=dFdy(WorldPosition);
    vec2 ux=dFdx(MaterialUV),uy=dFdy(MaterialUV);
    vec3 normal=UnitOr(WorldNormal,vec3(0,1,0));
    if (Draw.TextureFlags.w==FlatModel) { vec3 face=UnitOr(cross(px,py),normal); normal=dot(face,normal)<0?-face:face; }
    if (!gl_FrontFacing) normal=-normal;
    float materialDeterminant=ux.x*uy.y-ux.y*uy.x;
    vec2 heightUV=MaterialUV*HeightFrame.Shape.y;
    if (HeightFrame.Steps.w==ObjectCoordinates || (HeightFrame.Steps.w==AutomaticCoordinates && !ValidChart(ux,uy)))
        heightUV=ObjectChart(LocalPosition,dFdx(LocalPosition),dFdy(LocalPosition))*HeightFrame.Shape.y;
    vec2 hx=dFdx(heightUV),hy=dFdy(heightUV),uv=MaterialUV;
    SurfaceFrame frame=BuildFrame(px,py,hx,hy,normal,UnitOr(Draw.CameraLightCount.xyz-WorldPosition,normal));
    vec3 hit=vec3(heightUV,0);
    bool heightEnabled=frame.valid && HeightFrame.Shape.x>0 && Draw.TextureFlags.w!=FlatModel && Draw.RenderParameters.y<=OverlayMode;
    if (heightEnabled)
    {
        hit=IntersectHeight(heightUV,frame.ray,hx,hy,frame.cosine);
        float determinant=hx.x*hy.y-hx.y*hy.x;
        vec2 materialDu=(ux*hy.y-uy*hx.y)/determinant,materialDv=(uy*hx.x-ux*hy.x)/determinant;
        uv+=materialDu*(hit.x-heightUV.x)+materialDv*(hit.y-heightUV.y);
        vec2 slope=SampleHeight(hit.xy,hx,hy).yz*frame.thickness*HeightFrame.Shape.z;
        vec3 bumped=cross(frame.du+normal*slope.x,frame.dv+normal*slope.y);
        normal=UnitOr(dot(bumped,normal)<0?-bumped:bumped,normal);
    }
    vec4 albedo=Draw.TextureFlags.x!=0u?textureGrad(sampler2D(DiffuseImage,MaterialSampler),uv,ux,uy):vec4(1);
    vec3 surfaceMap = Draw.TextureFlags.z != 0 ? textureGrad(sampler2D(OpacityImage,MaterialSampler),uv,ux,uy).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Draw.ShadingParameters.xy;
    if (Draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity=surfaceMap.r;
    opacity=clamp(opacity*albedo.a*VertexColor.a*Draw.MaterialParameters.y,0,1);
    if (Draw.MaterialParameters.z>0 && opacity<Draw.MaterialParameters.z) discard;
    if (Draw.TextureFlags.y!=0u && Draw.TextureFlags.w!=FlatModel && ValidChart(ux,uy))
    {
        vec3 tangent=(px*uy.y-py*ux.y)/materialDeterminant,sourceB=(py*ux.x-px*uy.x)/materialDeterminant;
        tangent=UnitOr(tangent-normal*dot(normal,tangent),UnitOr(cross(vec3(0,1,0),normal),vec3(1,0,0)));
        float handedness=dot(cross(normal,tangent),sourceB)<0?-1:1; vec3 bitangent=UnitOr(cross(normal,tangent)*handedness,vec3(0,0,1));
        vec3 mapped=textureGrad(sampler2D(NormalImage,MaterialSampler),uv,ux,uy).xyz*2-1;
        normal=UnitOr(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
    }
    vec3 surface=VertexColor.rgb*max(Draw.PipelineColor.rgb,vec3(0))*max(albedo.rgb,vec3(0));
    float metallic=clamp(roughnessMetallic.y,0,1);
    vec3 ambient=surface*max(Draw.MaterialAmbientColor.rgb,vec3(0))*max(Draw.AmbientColorStrength.rgb,vec3(0))*max(Draw.AmbientColorStrength.w,0);
    if (Draw.TextureFlags.w==PhysicalModel) ambient*=1-metallic;
    DiffuseOpacity=vec4(clamp(surface*max(Draw.MaterialDiffuseColor.rgb,vec3(0)),0,MaximumRadiance),opacity);
    NormalRoughness=vec4(normal,clamp(roughnessMetallic.x,MinimumRoughness,1));
    SpecularShininess=clamp(vec4(Draw.MaterialSpecularColor.rgb,max(Draw.MaterialParameters.x,1)),0,MaximumRadiance);
    AmbientMetallic=vec4(clamp(ambient,0,MaximumRadiance),metallic); Emission=vec4(clamp(Draw.MaterialEmissiveColor.rgb,0,MaximumRadiance),clamp(Draw.MaterialParameters.w,0,1));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Draw.RenderParameters.y > 1.0 && Draw.RenderParameters.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (Draw.RenderParameters.y>OverlayMode) { NormalRoughness=vec4(0,0,0,-1); Emission.rgb=DecodeSrgb(clamp(Draw.PipelineColor.rgb,0,1)); }
}
#endif
