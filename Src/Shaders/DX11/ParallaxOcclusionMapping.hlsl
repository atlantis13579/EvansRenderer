// Tatarchuk, Practical Parallax Occlusion Mapping, SIGGRAPH 2006 course.
// A bounded view-ray search retains the first sampled height crossing, refines
// its bracket and intersects the final linear segment. The analytic periodic
// input needs no asset; all material reads use gradients captured before search.
static const uint MaximumLights=4,FlatModel=1,PhysicalModel=3;
static const uint AutomaticCoordinates=0,MeshCoordinates=1,ObjectCoordinates=2;
static const float Pi=3.14159265358979323846;
static const float DirectionEpsilon=1.0e-12,GramRelativeEpsilon=1.0e-6;
static const float IntersectionEpsilon=1.0e-7,MaximumCoordinateTravel=0.5;
static const float MaximumRadiance=65504,MinimumRoughness=0.045,OverlayMode=0.5;
struct SceneLight { float4 PositionType,DirectionIntensity,ColorCastShadows,AreaSizePadding; };
cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World,NormalWorld,ViewProjection,LightViewProjection;
    float4 PipelineColor,MaterialDiffuseColor,MaterialAmbientColor,MaterialSpecularColor,MaterialEmissiveColor;
    float4 MaterialParameters,AmbientColorStrength,CameraLightCount,RenderParameters;
    uint4 TextureFlags;
    SceneLight Lights[MaximumLights];
    float4 ShadingParameters;
};
cbuffer ParallaxFrame : register(b1)
{
    float4 Shape;
    uint4 Steps;
    float4 ObjectBounds;
};
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> NormalMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
SamplerState MaterialSampler : register(s0);
struct MeshVertex { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR; float2 UV:TEXCOORD0; };
struct SurfaceVertex
{
    precise float4 Position:SV_POSITION;
    float3 WorldPosition:TEXCOORD0;
    float3 Normal:TEXCOORD1;
    float4 Color:COLOR0;
    float2 UV:TEXCOORD2;
    float3 LocalPosition:TEXCOORD3;
};
struct MaterialOutputs
{
    float4 DiffuseOpacity:SV_TARGET0;
    float4 NormalRoughness:SV_TARGET1;
    float4 SpecularShininess:SV_TARGET2;
    float4 AmbientMetallic:SV_TARGET3;
    float4 Emission:SV_TARGET4;
};
float3 UnitOr(float3 value,float3 fallback)
{ float squared=dot(value,value); return squared>DirectionEpsilon?value*rsqrt(squared):fallback; }
float3 DecodeSrgb(float3 value)
{
    const float Threshold=0.04045,LinearScale=12.92,PowerOffset=0.055,PowerScale=1.055,PowerExponent=2.4;
    return lerp(pow((value+PowerOffset)/PowerScale,PowerExponent),value/LinearScale,value<=Threshold);
}
bool ValidChart(float2 dx,float2 dy)
{
    float2 product=float2(length(dx),length(dy));
    float scale=product.x*product.y;
    const float RelativeChartEpsilon=1e-6;
    return scale>0 && abs(dx.x*dy.y-dx.y*dy.x)>RelativeChartEpsilon*scale;
}
SurfaceVertex VSMain(MeshVertex input)
{
    SurfaceVertex output; precise float4 world=mul(World,float4(input.Position,1));
    output.Position=mul(ViewProjection,world); output.WorldPosition=world.xyz/max(abs(world.w),IntersectionEpsilon);
    output.Normal=UnitOr(mul((float3x3)NormalWorld,input.Normal),float3(0,1,0));
    output.Color=float4(DecodeSrgb(saturate(input.Color.rgb)),input.Color.a);
    output.UV=float2(input.UV.x,1-input.UV.y); output.LocalPosition=input.Position; return output;
}
// Return height and its two analytic derivatives. The independent diagonal
// Fourier modes retain anisotropic Gaussian footprint attenuation.
float3 Height(float2 uv,float2 dx,float2 dy)
{
    float2 phase=2*Pi*frac(float2(uv.x+uv.y,uv.x-uv.y));
    float2 fx=float2(dx.x+dx.y,dx.x-dx.y),fy=float2(dy.x+dy.y,dy.x-dy.y);
    float2 weight=exp(-2*Pi*Pi*(fx*fx+fy*fy));
    const float ModeWeight=0.25;
    float2 slope=-ModeWeight*2*Pi*weight*sin(phase);
    return float3(0.5+ModeWeight*dot(weight,cos(phase)),slope.x+slope.y,slope.x-slope.y);
}
float2 ObjectChart(float3 local,float3 localDx,float3 localDy)
{
    float3 axis=abs(cross(localDx,localDy));
    float3 scaled=(local-ObjectBounds.xyz)*ObjectBounds.w;
    return axis.z>=max(axis.x,axis.y)?scaled.xy:axis.y>=axis.x?scaled.xz:scaled.yz;
}
struct HeightFrame { float3 Du,Dv,Normal; float2 Ray; float Thickness,Cosine; bool Valid; };
HeightFrame MakeFrame(float3 dx,float3 dy,float2 hx,float2 hy,float3 normal,float3 eye)
{
    HeightFrame frame=(HeightFrame)0; frame.Normal=normal;
    float determinant=hx.x*hy.y-hx.y*hy.x;
    if (!ValidChart(hx,hy)) return frame;
    frame.Du=(dx*hy.y-dy*hx.y)/determinant; frame.Dv=(dy*hx.x-dx*hy.x)/determinant;
    frame.Du-=normal*dot(normal,frame.Du); frame.Dv-=normal*dot(normal,frame.Dv);
    float uu=dot(frame.Du,frame.Du),uv=dot(frame.Du,frame.Dv),vv=dot(frame.Dv,frame.Dv),gram=uu*vv-uv*uv;
    if (min(uu,vv)<=DirectionEpsilon || gram<=GramRelativeEpsilon*uu*vv) return frame;
    frame.Cosine=max(dot(normal,eye),0);
    frame.Thickness=Shape.x*sqrt(min(uu,vv))*smoothstep(Shape.w,2*Shape.w,frame.Cosine);
    float3 travel=-eye*frame.Thickness/max(frame.Cosine,Shape.w);
    float a=dot(frame.Du,travel),b=dot(frame.Dv,travel);
    frame.Ray=float2(a*vv-b*uv,b*uu-a*uv)/gram;
    float distance=length(frame.Ray),limit=distance>MaximumCoordinateTravel?MaximumCoordinateTravel/distance:1;
    frame.Ray*=limit; frame.Thickness*=limit; frame.Valid=true; return frame;
}
float3 Trace(float2 origin,float2 ray,float2 dx,float2 dy,float cosine)
{
    uint count=(uint)ceil(lerp(float(Steps.y),float(Steps.x),saturate(cosine)));
    float low=0,high=0,lowError=Height(origin,dx,dy).x-1,highError=lowError;
    if (lowError<0)
    {
        for (uint layer=1;layer<=count;++layer)
        {
            high=float(layer)/count; highError=high+Height(origin+ray*high,dx,dy).x-1;
            if (highError>=0) break;
            low=high; lowError=highError;
        }
        for (uint refine=0;refine<Steps.z;++refine)
        {
            float middle=(low+high)/2,error=middle+Height(origin+ray*middle,dx,dy).x-1;
            if (error>=0) { high=middle; highError=error; } else { low=middle; lowError=error; }
        }
    }
    float difference=highError-lowError;
    float depth=lerp(low,high,abs(difference)>IntersectionEpsilon?saturate(-lowError/difference):0);
    return float3(origin+ray*depth,depth);
}
MaterialOutputs PSMain(SurfaceVertex input,bool front:SV_ISFRONTFACE)
{
    float3 px=ddx(input.WorldPosition),py=ddy(input.WorldPosition);
    float2 ux=ddx(input.UV),uy=ddy(input.UV);
    float3 normal=UnitOr(input.Normal,float3(0,1,0));
    if (TextureFlags.w==FlatModel)
    { float3 face=UnitOr(cross(px,py),normal); normal=dot(face,normal)<0?-face:face; }
    if (!front) normal=-normal;
    float2 heightUV=input.UV*Shape.y;
    float uvDeterminant=ux.x*uy.y-ux.y*uy.x;
    if (Steps.w==ObjectCoordinates || (Steps.w==AutomaticCoordinates && !ValidChart(ux,uy)))
        heightUV=ObjectChart(input.LocalPosition,ddx(input.LocalPosition),ddy(input.LocalPosition))*Shape.y;
    float2 hx=ddx(heightUV),hy=ddy(heightUV),materialUV=input.UV;
    HeightFrame frame=MakeFrame(px,py,hx,hy,normal,UnitOr(CameraLightCount.xyz-input.WorldPosition,normal));
    float3 hit=float3(heightUV,0);
    bool active=frame.Valid && Shape.x>0 && TextureFlags.w!=FlatModel && RenderParameters.y<=OverlayMode;
    if (active)
    {
        hit=Trace(heightUV,frame.Ray,hx,hy,frame.Cosine);
        float determinant=hx.x*hy.y-hx.y*hy.x;
        float2 materialDu=(ux*hy.y-uy*hx.y)/determinant,materialDv=(uy*hx.x-ux*hy.x)/determinant;
        materialUV+=materialDu*(hit.x-heightUV.x)+materialDv*(hit.y-heightUV.y);
        float2 gradient=Height(hit.xy,hx,hy).yz*frame.Thickness*Shape.z;
        float3 bumped=cross(frame.Du+normal*gradient.x,frame.Dv+normal*gradient.y);
        normal=UnitOr(dot(bumped,normal)<0?-bumped:bumped,normal);
    }
    float4 albedo=TextureFlags.x!=0?DiffuseMap.SampleGrad(MaterialSampler,materialUV,ux,uy):float4(1,1,1,1);
    float3 surfaceMap = TextureFlags.z != 0 ? OpacityMap.SampleGrad(MaterialSampler,materialUV,ux,uy).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity=surfaceMap.r;
    opacity=saturate(opacity*albedo.a*input.Color.a*MaterialParameters.y);
    if (MaterialParameters.z>0 && opacity<MaterialParameters.z) discard;
    if (TextureFlags.y!=0 && TextureFlags.w!=FlatModel && ValidChart(ux,uy))
    {
        float3 tangent=(px*uy.y-py*ux.y)/uvDeterminant,sourceB=(py*ux.x-px*uy.x)/uvDeterminant;
        tangent=UnitOr(tangent-normal*dot(normal,tangent),UnitOr(cross(float3(0,1,0),normal),float3(1,0,0)));
        float sign=dot(cross(normal,tangent),sourceB)<0?-1:1; float3 bitangent=UnitOr(cross(normal,tangent)*sign,float3(0,0,1));
        float3 mapped=NormalMap.SampleGrad(MaterialSampler,materialUV,ux,uy).xyz*2-1;
        normal=UnitOr(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
    }
    float3 surface=input.Color.rgb*max(PipelineColor.rgb,0)*max(albedo.rgb,0);
    float metallic=saturate(roughnessMetallic.y);
    float3 ambient=surface*max(MaterialAmbientColor.rgb,0)*max(AmbientColorStrength.rgb,0)*max(AmbientColorStrength.w,0);
    if (TextureFlags.w==PhysicalModel) ambient*=1-metallic;
    MaterialOutputs output;
    output.DiffuseOpacity=float4(clamp(surface*max(MaterialDiffuseColor.rgb,0),0,MaximumRadiance),opacity);
    output.NormalRoughness=float4(normal,clamp(roughnessMetallic.x,MinimumRoughness,1));
    output.SpecularShininess=clamp(float4(MaterialSpecularColor.rgb,max(MaterialParameters.x,1)),0,MaximumRadiance);
    output.AmbientMetallic=float4(clamp(ambient,0,MaximumRadiance),metallic);
    output.Emission=float4(clamp(MaterialEmissiveColor.rgb,0,MaximumRadiance),saturate(MaterialParameters.w));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y>OverlayMode)
    { output.NormalRoughness=float4(0,0,0,-1); output.Emission.rgb=DecodeSrgb(saturate(PipelineColor.rgb)); }
    return output;
}
