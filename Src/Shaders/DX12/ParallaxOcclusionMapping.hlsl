// D3D12 owns this height-field material evaluator. A coarse front-to-back ray
// search, bisection and secant interpolation approximate the first visible
// crossing (Tatarchuk, SIGGRAPH 2006). No other technique source is included.
static const uint LightCapacity=4,FlatShading=1,PhysicallyBased=3;
static const uint CoordinatesAutomatic=0,CoordinatesMesh=1,CoordinatesObject=2;
static const float Pi=3.14159265358979323846;
static const float VectorFloor=1e-12,GramFloor=1e-6,RootFloor=1e-7;
static const float TravelLimit=0.5,HalfFloatCeiling=65504,MinimumRoughness=0.045,WireThreshold=0.5;
struct LightRecord { float4 PositionType,DirectionIntensity,ColorCastShadows,AreaSizePadding; };
cbuffer MeshParameters : register(b0)
{
    row_major float4x4 World,NormalWorld,ViewProjection,LightViewProjection;
    float4 PipelineColor,DiffuseColor,AmbientColor,SpecularColor,EmissionColor;
    float4 Material,AmbientLight,EyeLightCount,Render;
    uint4 TextureFlags;
    LightRecord Lights[LightCapacity];
    float4 Shading;
};
cbuffer HeightParameters : register(b1)
{
    float4 Shape;
    uint4 Steps;
    float4 ObjectBounds;
};
Texture2D<float4> Albedo : register(t1);
Texture2D<float4> TangentNormal : register(t2);
Texture2D<float4> Coverage : register(t3);
SamplerState MaterialSampler : register(s0);
struct VertexInput { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR; float2 UV:TEXCOORD0; };
struct Varyings
{
    precise float4 Position:SV_POSITION;
    float3 WorldPosition:TEXCOORD0,Normal:TEXCOORD1;
    float4 Color:COLOR0;
    float2 UV:TEXCOORD2;
    float3 ObjectPosition:TEXCOORD3;
};
struct MaterialResult
{
    float4 DiffuseOpacity:SV_TARGET0,NormalRoughness:SV_TARGET1,SpecularShininess:SV_TARGET2,AmbientMetallic:SV_TARGET3,Emission:SV_TARGET4;
};
float3 DirectionOr(float3 value,float3 fallback)
{ float squared=dot(value,value); return squared>VectorFloor?value*rsqrt(squared):fallback; }
float3 LinearColor(float3 encoded)
{
    const float Breakpoint=0.04045,Slope=12.92,Offset=0.055,Scale=1.055,Exponent=2.4;
    return encoded<=Breakpoint?encoded/Slope:pow((encoded+Offset)/Scale,Exponent);
}
bool ValidChart(float2 dx,float2 dy)
{
    float2 product=float2(length(dx),length(dy));
    float scale=product.x*product.y;
    const float RelativeChartEpsilon=1e-6;
    return scale>0 && abs(dx.x*dy.y-dx.y*dy.x)>RelativeChartEpsilon*scale;
}
Varyings VSMain(VertexInput input)
{
    Varyings output; precise float4 worldPosition=mul(World,float4(input.Position,1));
    output.Position=mul(ViewProjection,worldPosition); output.WorldPosition=worldPosition.xyz/max(abs(worldPosition.w),RootFloor);
    output.Normal=DirectionOr(mul((float3x3)NormalWorld,input.Normal),float3(0,1,0));
    output.Color=float4(LinearColor(saturate(input.Color.rgb)),input.Color.a);
    output.UV=float2(input.UV.x,1-input.UV.y); output.ObjectPosition=input.Position; return output;
}
// Height and gradient of two independently filtered diagonal Fourier modes.
float3 Field(float2 p,float2 dx,float2 dy)
{
    float sum=p.x+p.y,difference=p.x-p.y;
    float sumVariance=dot(float2(dx.x+dx.y,dy.x+dy.y),float2(dx.x+dx.y,dy.x+dy.y));
    float differenceVariance=dot(float2(dx.x-dx.y,dy.x-dy.y),float2(dx.x-dx.y,dy.x-dy.y));
    float2 angles=2*Pi*frac(float2(sum,difference));
    float2 amplitudes=exp(-2*Pi*Pi*float2(sumVariance,differenceVariance));
    const float FourierWeight=0.25;
    float2 derivatives=-FourierWeight*2*Pi*amplitudes*sin(angles);
    return float3(0.5+FourierWeight*dot(amplitudes,cos(angles)),derivatives.x+derivatives.y,derivatives.x-derivatives.y);
}
float2 ProjectionChart(float3 position,float3 dx,float3 dy)
{
    float3 dominant=abs(cross(dx,dy)); float3 normalized=(position-ObjectBounds.xyz)*ObjectBounds.w;
    if (dominant.z>=dominant.x && dominant.z>=dominant.y) return normalized.xy;
    return dominant.y>=dominant.x?normalized.xz:normalized.yz;
}
struct ChartFrame { float3 U,V; float2 Direction; float Thickness,Cosine; bool Valid; };
ChartFrame Parameterize(float3 px,float3 py,float2 hx,float2 hy,float3 normal,float3 view)
{
    ChartFrame result=(ChartFrame)0;
    float jacobian=hx.x*hy.y-hx.y*hy.x;
    if (!ValidChart(hx,hy)) return result;
    result.U=(px*hy.y-py*hx.y)/jacobian; result.V=(py*hx.x-px*hy.x)/jacobian;
    result.U-=normal*dot(normal,result.U); result.V-=normal*dot(normal,result.V);
    float uu=dot(result.U,result.U),uv=dot(result.U,result.V),vv=dot(result.V,result.V),determinant=uu*vv-uv*uv;
    if (min(uu,vv)<=VectorFloor || determinant<=GramFloor*uu*vv) return result;
    result.Cosine=max(dot(normal,view),0);
    result.Thickness=Shape.x*sqrt(min(uu,vv))*smoothstep(Shape.w,2*Shape.w,result.Cosine);
    float3 shift=-view*(result.Thickness/max(result.Cosine,Shape.w));
    float projectionU=dot(shift,result.U),projectionV=dot(shift,result.V);
    result.Direction=float2(projectionU*vv-projectionV*uv,projectionV*uu-projectionU*uv)/determinant;
    float distance=length(result.Direction),scale=distance>TravelLimit?TravelLimit/distance:1;
    result.Direction*=scale; result.Thickness*=scale; result.Valid=true; return result;
}
float3 FirstCrossing(float2 uv,float2 direction,float2 dx,float2 dy,float viewCosine)
{
    uint layers=(uint)ceil(Steps.y+(float(Steps.x)-float(Steps.y))*saturate(viewCosine));
    float nearDepth=0,farDepth=0,nearError=Field(uv,dx,dy).x-1,farError=nearError;
    if (nearError<0)
    {
        for (uint layer=1;layer<=layers;++layer)
        {
            farDepth=float(layer)/layers; farError=farDepth+Field(uv+direction*farDepth,dx,dy).x-1;
            if (farError>=0) break;
            nearDepth=farDepth; nearError=farError;
        }
        for (uint level=0;level<Steps.z;++level)
        {
            float center=(nearDepth+farDepth)/2,value=center+Field(uv+direction*center,dx,dy).x-1;
            if (value<0) { nearDepth=center; nearError=value; } else { farDepth=center; farError=value; }
        }
    }
    float denominator=farError-nearError;
    float fraction=abs(denominator)>RootFloor?saturate(-nearError/denominator):0;
    float depth=lerp(nearDepth,farDepth,fraction); return float3(uv+direction*depth,depth);
}
MaterialResult PSMain(Varyings input,bool front:SV_ISFRONTFACE)
{
    float3 px=ddx(input.WorldPosition),py=ddy(input.WorldPosition);
    float2 tx=ddx(input.UV),ty=ddy(input.UV); float uvDet=tx.x*ty.y-tx.y*ty.x;
    float3 n=DirectionOr(input.Normal,float3(0,1,0));
    if (TextureFlags.w==FlatShading) { float3 geometric=DirectionOr(cross(px,py),n); n=dot(geometric,n)<0?-geometric:geometric; }
    if (!front) n=-n;
    float2 base=input.UV*Shape.y;
    if (Steps.w==CoordinatesObject || (Steps.w==CoordinatesAutomatic && !ValidChart(tx,ty)))
        base=ProjectionChart(input.ObjectPosition,ddx(input.ObjectPosition),ddy(input.ObjectPosition))*Shape.y;
    float2 hx=ddx(base),hy=ddy(base),mappedUV=input.UV;
    ChartFrame frame=Parameterize(px,py,hx,hy,n,DirectionOr(EyeLightCount.xyz-input.WorldPosition,n));
    float3 hit=float3(base,0); bool active=frame.Valid && Shape.x>0 && TextureFlags.w!=FlatShading && Render.y<=WireThreshold;
    if (active)
    {
        hit=FirstCrossing(base,frame.Direction,hx,hy,frame.Cosine);
        float det=hx.x*hy.y-hx.y*hy.x;
        float2 u=(tx*hy.y-ty*hx.y)/det,v=(ty*hx.x-tx*hy.x)/det;
        mappedUV+=u*(hit.x-base.x)+v*(hit.y-base.y);
        float2 derivatives=Field(hit.xy,hx,hy).yz*frame.Thickness*Shape.z;
        float3 perturbed=cross(frame.U+n*derivatives.x,frame.V+n*derivatives.y);
        n=DirectionOr(dot(perturbed,n)<0?-perturbed:perturbed,n);
    }
    float4 albedo=TextureFlags.x?Albedo.SampleGrad(MaterialSampler,mappedUV,tx,ty):float4(1,1,1,1);
    float3 surfaceMap = TextureFlags.z != 0 ? Coverage.SampleGrad(MaterialSampler,mappedUV,tx,ty).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = Shading.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float alpha=(surfaceMap.r)*albedo.a*input.Color.a*Material.y;
    alpha=saturate(alpha); if (Material.z>0 && alpha<Material.z) discard;
    if (TextureFlags.y && TextureFlags.w!=FlatShading && ValidChart(tx,ty))
    {
        float3 u=(px*ty.y-py*tx.y)/uvDet,v=(py*tx.x-px*ty.x)/uvDet;
        u=DirectionOr(u-n*dot(n,u),DirectionOr(cross(float3(0,1,0),n),float3(1,0,0)));
        float orientation=dot(cross(n,u),v)<0?-1:1; v=DirectionOr(cross(n,u)*orientation,float3(0,0,1));
        float3 local=TangentNormal.SampleGrad(MaterialSampler,mappedUV,tx,ty).xyz*2-1;
        n=DirectionOr(u*local.x+v*local.y+n*local.z,n);
    }
    float3 color=input.Color.rgb*max(PipelineColor.rgb,0)*max(albedo.rgb,0);
    float metallic=saturate(roughnessMetallic.y); float3 ambient=color*max(AmbientColor.rgb,0)*max(AmbientLight.rgb,0)*max(AmbientLight.w,0);
    if (TextureFlags.w==PhysicallyBased) ambient*=1-metallic;
    MaterialResult output;
    output.DiffuseOpacity=float4(clamp(color*max(DiffuseColor.rgb,0),0,HalfFloatCeiling),alpha);
    output.NormalRoughness=float4(n,clamp(roughnessMetallic.x,MinimumRoughness,1));
    output.SpecularShininess=clamp(float4(SpecularColor.rgb,max(Material.x,1)),0,HalfFloatCeiling);
    output.AmbientMetallic=float4(clamp(ambient,0,HalfFloatCeiling),metallic); output.Emission=float4(clamp(EmissionColor.rgb,0,HalfFloatCeiling),saturate(Material.w));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Render.y > 1.0 && Render.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (Render.y>WireThreshold) { output.NormalRoughness=float4(0,0,0,-1); output.Emission.rgb=LinearColor(saturate(PipelineColor.rgb)); }
    return output;
}
