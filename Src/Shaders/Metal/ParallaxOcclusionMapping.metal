#include <metal_stdlib>
using namespace metal;

// Independent MSL material capture. Reverse height tracing with adaptive layers,
// first-crossing refinement and explicit texture gradients; Tatarchuk (2006).
// A periodic analytic field supplies height and slope without an authored asset.
constant uint MaximumLights=4,FlatModel=1,PhysicalModel=3;
constant uint AutomaticCoordinates=0,MeshCoordinates=1,ObjectCoordinates=2;
constant float Pi=3.14159265358979323846;
constant float DirectionEpsilon=1e-12,GramRelativeEpsilon=1e-6,IntersectionEpsilon=1e-7;
constant float MaximumCoordinateTravel=0.5,MaximumRadiance=65504,MinimumRoughness=0.045,OverlayMode=0.5;
struct SceneLight { float4 PositionType,DirectionIntensity,ColorCastShadows,AreaSizePadding; };
struct DrawConstants
{
    float4 World[4],NormalWorld[4],ViewProjection[4],LightViewProjection[4];
    float4 PipelineColor,MaterialDiffuseColor,MaterialAmbientColor,MaterialSpecularColor,MaterialEmissiveColor;
    float4 MaterialParameters,AmbientColorStrength,CameraLightCount,RenderParameters;
    uint4 TextureFlags;
    SceneLight Lights[MaximumLights];
    float4 ShadingParameters;
};
struct ParallaxConstants { float4 Shape; uint4 Steps; float4 ObjectBounds; };
struct MeshVertex { float Values[12]; };
struct SurfaceVertex
{
    float4 Position [[position, invariant]];
    float3 WorldPosition,Normal;
    float4 Color;
    float2 UV;
    float3 LocalPosition;
};
struct MaterialOutputs
{
    float4 DiffuseOpacity [[color(0)]];
    float4 NormalRoughness [[color(1)]];
    float4 SpecularShininess [[color(2)]];
    float4 AmbientMetallic [[color(3)]];
    float4 Emission [[color(4)]];
};
float3 UnitOr(float3 p,float3 fallback)
{ float squared=dot(p,p); return squared>DirectionEpsilon?p*rsqrt(squared):fallback; }
float4 TransformRows(constant float4* matrix,float4 p)
{ return float4(dot(matrix[0],p),dot(matrix[1],p),dot(matrix[2],p),dot(matrix[3],p)); }
float3 DecodeSrgb(float3 encoded)
{
    const float Breakpoint=0.04045,LinearScale=12.92,PowerOffset=0.055,PowerScale=1.055,Exponent=2.4;
    return select(pow((encoded+PowerOffset)/PowerScale,float3(Exponent)),encoded/LinearScale,encoded<=Breakpoint);
}
bool ValidChart(float2 dx,float2 dy)
{
    float2 product=float2(length(dx),length(dy));
    float scale=product.x*product.y;
    const float RelativeChartEpsilon=1e-6;
    return scale>0 && abs(dx.x*dy.y-dx.y*dy.x)>RelativeChartEpsilon*scale;
}
vertex SurfaceVertex VSMain(uint vertexID [[vertex_id]],const device MeshVertex* vertices [[buffer(0)]],constant DrawConstants& draw [[buffer(1)]])
{
    const device float* input=vertices[vertexID].Values;
    float3 local=float3(input[0],input[1],input[2]); float4 world=TransformRows(draw.World,float4(local,1));
    SurfaceVertex output; output.Position=TransformRows(draw.ViewProjection,world);
    output.WorldPosition=world.xyz/max(abs(world.w),IntersectionEpsilon);
    output.Normal=UnitOr(TransformRows(draw.NormalWorld,float4(input[3],input[4],input[5],0)).xyz,float3(0,1,0));
    output.Color=float4(DecodeSrgb(clamp(float3(input[6],input[7],input[8]),0.0f,1.0f)),input[9]);
    output.UV=float2(input[10],1-input[11]); output.LocalPosition=local; return output;
}
float3 Height(float2 uv,float2 dx,float2 dy)
{
    float2 phase=2*Pi*fract(float2(uv.x+uv.y,uv.x-uv.y));
    float2 x= float2(dx.x+dx.y,dx.x-dx.y),y=float2(dy.x+dy.y,dy.x-dy.y);
    float2 weight=exp(-2*Pi*Pi*(x*x+y*y)); const float ModeWeight=0.25;
    float2 gradient=-ModeWeight*2*Pi*weight*sin(phase);
    return float3(0.5+ModeWeight*dot(weight,cos(phase)),gradient.x+gradient.y,gradient.x-gradient.y);
}
float2 ObjectChart(float3 p,float3 dx,float3 dy,constant ParallaxConstants& settings)
{
    float3 axis=abs(cross(dx,dy)),local=(p-settings.ObjectBounds.xyz)*settings.ObjectBounds.w;
    return axis.z>=max(axis.x,axis.y)?local.xy:axis.y>=axis.x?local.xz:local.yz;
}
struct HeightFrame { float3 du,dv; float2 ray; float thickness,cosine; bool valid; };
HeightFrame BuildFrame(float3 px,float3 py,float2 hx,float2 hy,float3 normal,float3 view,constant ParallaxConstants& settings)
{
    HeightFrame result{float3(0),float3(0),float2(0),0,0,false};
    float determinant=hx.x*hy.y-hx.y*hy.x;
    if (!ValidChart(hx,hy)) return result;
    result.du=(px*hy.y-py*hx.y)/determinant; result.dv=(py*hx.x-px*hy.x)/determinant;
    result.du-=normal*dot(normal,result.du); result.dv-=normal*dot(normal,result.dv);
    float uu=dot(result.du,result.du),uv=dot(result.du,result.dv),vv=dot(result.dv,result.dv),gram=uu*vv-uv*uv;
    if (min(uu,vv)<=DirectionEpsilon || gram<=GramRelativeEpsilon*uu*vv) return result;
    result.cosine=max(dot(normal,view),0.0f);
    result.thickness=settings.Shape.x*sqrt(min(uu,vv))*smoothstep(settings.Shape.w,2*settings.Shape.w,result.cosine);
    float3 travel=-view*(result.thickness/max(result.cosine,settings.Shape.w));
    float a=dot(result.du,travel),b=dot(result.dv,travel); result.ray=float2(a*vv-b*uv,b*uu-a*uv)/gram;
    float distance=length(result.ray),scale=distance>MaximumCoordinateTravel?MaximumCoordinateTravel/distance:1;
    result.ray*=scale; result.thickness*=scale; result.valid=true; return result;
}
float3 Trace(float2 start,float2 direction,float2 dx,float2 dy,float cosine,constant ParallaxConstants& settings)
{
    uint layers=uint(ceil(mix(float(settings.Steps.y),float(settings.Steps.x),clamp(cosine,0.0f,1.0f))));
    float low=0,high=0,lowError=Height(start,dx,dy).x-1,highError=lowError;
    if (lowError<0)
    {
        for (uint layer=1;layer<=layers;++layer)
        {
            high=float(layer)/layers; highError=high+Height(start+direction*high,dx,dy).x-1;
            if (highError>=0) break;
            low=high; lowError=highError;
        }
        for (uint refine=0;refine<settings.Steps.z;++refine)
        {
            float middle=(low+high)/2,error=middle+Height(start+direction*middle,dx,dy).x-1;
            if (error>=0) { high=middle; highError=error; } else { low=middle; lowError=error; }
        }
    }
    float difference=highError-lowError;
    float depth=mix(low,high,abs(difference)>IntersectionEpsilon?clamp(-lowError/difference,0.0f,1.0f):0);
    return float3(start+direction*depth,depth);
}
fragment MaterialOutputs PSMain(SurfaceVertex input [[stage_in]],bool front [[front_facing]],constant DrawConstants& draw [[buffer(1)]],
    constant ParallaxConstants& settings [[buffer(2)]],texture2d<float> albedoImage [[texture(1)]],texture2d<float> normalImage [[texture(2)]],
    texture2d<float> opacityImage [[texture(3)]],sampler materialSampler [[sampler(0)]])
{
    float3 px=dfdx(input.WorldPosition),py=dfdy(input.WorldPosition);
    float2 ux=dfdx(input.UV),uy=dfdy(input.UV); float materialDeterminant=ux.x*uy.y-ux.y*uy.x;
    float3 normal=UnitOr(input.Normal,float3(0,1,0));
    if (draw.TextureFlags.w==FlatModel) { float3 geometric=UnitOr(cross(px,py),normal); normal=dot(geometric,normal)<0?-geometric:geometric; }
    if (!front) normal=-normal;
    float2 base=input.UV*settings.Shape.y;
    if (settings.Steps.w==ObjectCoordinates || (settings.Steps.w==AutomaticCoordinates && !ValidChart(ux,uy)))
        base=ObjectChart(input.LocalPosition,dfdx(input.LocalPosition),dfdy(input.LocalPosition),settings)*settings.Shape.y;
    float2 hx=dfdx(base),hy=dfdy(base),uv=input.UV;
    HeightFrame frame=BuildFrame(px,py,hx,hy,normal,UnitOr(draw.CameraLightCount.xyz-input.WorldPosition,normal),settings);
    float3 hit=float3(base,0); bool active=frame.valid && settings.Shape.x>0 && draw.TextureFlags.w!=FlatModel && draw.RenderParameters.y<=OverlayMode;
    if (active)
    {
        hit=Trace(base,frame.ray,hx,hy,frame.cosine,settings);
        float determinant=hx.x*hy.y-hx.y*hy.x;
        float2 materialDu=(ux*hy.y-uy*hx.y)/determinant,materialDv=(uy*hx.x-ux*hy.x)/determinant;
        uv+=materialDu*(hit.x-base.x)+materialDv*(hit.y-base.y);
        float2 gradient=Height(hit.xy,hx,hy).yz*frame.thickness*settings.Shape.z;
        float3 perturbed=cross(frame.du+normal*gradient.x,frame.dv+normal*gradient.y);
        normal=UnitOr(dot(perturbed,normal)<0?-perturbed:perturbed,normal);
    }
    float4 albedo=draw.TextureFlags.x!=0?albedoImage.sample(materialSampler,uv,gradient2d(ux,uy)):float4(1);
    float3 surfaceMap = draw.TextureFlags.z != 0 ? opacityImage.sample(materialSampler,uv,gradient2d(ux,uy)).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = draw.ShadingParameters.xy;
    if (draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float alpha=surfaceMap.r;
    alpha=clamp(alpha*albedo.a*input.Color.a*draw.MaterialParameters.y,0.0f,1.0f);
    if (draw.MaterialParameters.z>0 && alpha<draw.MaterialParameters.z) discard_fragment();
    if (draw.TextureFlags.y!=0 && draw.TextureFlags.w!=FlatModel && ValidChart(ux,uy))
    {
        float3 tangent=(px*uy.y-py*ux.y)/materialDeterminant,sourceB=(py*ux.x-px*uy.x)/materialDeterminant;
        tangent=UnitOr(tangent-normal*dot(normal,tangent),UnitOr(cross(float3(0,1,0),normal),float3(1,0,0)));
        float sign=dot(cross(normal,tangent),sourceB)<0?-1:1; float3 bitangent=UnitOr(cross(normal,tangent)*sign,float3(0,0,1));
        float3 mapped=normalImage.sample(materialSampler,uv,gradient2d(ux,uy)).xyz*2-1;
        normal=UnitOr(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
    }
    float3 surface=input.Color.rgb*max(draw.PipelineColor.rgb,0.0f)*max(albedo.rgb,0.0f); float metallic=clamp(roughnessMetallic.y,0.0f,1.0f);
    float3 ambient=surface*max(draw.MaterialAmbientColor.rgb,0.0f)*max(draw.AmbientColorStrength.rgb,0.0f)*max(draw.AmbientColorStrength.w,0.0f);
    if (draw.TextureFlags.w==PhysicalModel) ambient*=1-metallic;
    MaterialOutputs output;
    output.DiffuseOpacity=float4(clamp(surface*max(draw.MaterialDiffuseColor.rgb,0.0f),0.0f,MaximumRadiance),alpha);
    output.NormalRoughness=float4(normal,clamp(roughnessMetallic.x,MinimumRoughness,1.0f));
    output.SpecularShininess=clamp(float4(draw.MaterialSpecularColor.rgb,max(draw.MaterialParameters.x,1.0f)),0.0f,MaximumRadiance);
    output.AmbientMetallic=float4(clamp(ambient,0.0f,MaximumRadiance),metallic); output.Emission=float4(clamp(draw.MaterialEmissiveColor.rgb,0.0f,MaximumRadiance),saturate(draw.MaterialParameters.w));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (draw.RenderParameters.y > 1.0 && draw.RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    if (draw.RenderParameters.y>OverlayMode) { output.NormalRoughness=float4(0,0,0,-1); output.Emission.rgb=DecodeSrgb(clamp(draw.PipelineColor.rgb,0.0f,1.0f)); }
    return output;
}
