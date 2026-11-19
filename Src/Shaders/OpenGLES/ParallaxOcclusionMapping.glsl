#version 310 es
// Standalone ES material stage. Adaptive reverse height tracing follows
// Tatarchuk (SIGGRAPH 2006), with a first-crossing bracket and linear refinement.
// Native material UVs keep their established V flip; gradients precede tracing.
precision highp float;
precision highp int;
const int LightCapacity=4;
const uint FlatShading=1u,PhysicallyBased=3u;
const uint CoordinatesAutomatic=0u,CoordinatesMesh=1u,CoordinatesObject=2u;
const float Pi=3.14159265358979323846;
const float VectorFloor=1e-12,GramFloor=1e-6,RootFloor=1e-7;
const float TravelLimit=0.5,HalfFloatCeiling=65504.0,MinimumRoughness=0.045,WireThreshold=0.5;
struct LightRecord { vec4 PositionType,DirectionIntensity,ColorCastShadows,AreaSizePadding; };
layout(std140,row_major,binding=0) uniform MeshParameters
{
    mat4 World,NormalWorld,ViewProjection,LightViewProjection;
    vec4 PipelineColor,DiffuseColor,AmbientColor,SpecularColor,EmissionColor;
    vec4 Material,AmbientLight,EyeLightCount,Render;
    uvec4 TextureFlags;
    LightRecord Lights[LightCapacity];
    vec4 Shading;
} Draw;
layout(std140,binding=1) uniform HeightParameters { vec4 Shape; uvec4 Steps; vec4 ObjectBounds; } HeightFrame;
vec3 DirectionOr(vec3 value,vec3 fallback)
{ float squared=dot(value,value); return squared>VectorFloor?value*inversesqrt(squared):fallback; }
vec3 LinearColor(vec3 encoded)
{
    const float Breakpoint=0.04045,Slope=12.92,Offset=0.055,Scale=1.055,Exponent=2.4;
    return mix(pow((encoded+Offset)/Scale,vec3(Exponent)),encoded/Slope,lessThanEqual(encoded,vec3(Breakpoint)));
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
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}

layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
out vec3 WorldPosition;
out vec3 WorldNormal;
out vec4 VertexColor;
out vec2 MaterialUV;
out vec3 ObjectPosition;
void main()
{
    vec4 worldPosition=Draw.World*vec4(InPosition,1);
    gl_Position=Draw.ViewProjection*worldPosition; gl_Position.z=gl_Position.z*2.0-gl_Position.w;
    WorldPosition=worldPosition.xyz/max(abs(worldPosition.w),RootFloor);
    WorldNormal=DirectionOr(mat3(Draw.NormalWorld)*InNormal,vec3(0,1,0));
    VertexColor=vec4(LinearColor(clamp(InColor.rgb,0.0,1.0)),InColor.a);
    MaterialUV=vec2(InUV.x,1.0-InUV.y); ObjectPosition=InPosition;
    CullWireFace();
}
#else
in vec3 WorldPosition;
in vec3 WorldNormal;
in vec4 VertexColor;
in vec2 MaterialUV;
in vec3 ObjectPosition;
layout(binding=1) uniform highp sampler2D Albedo;
layout(binding=2) uniform highp sampler2D TangentNormal;
layout(binding=3) uniform highp sampler2D Coverage;
layout(location=0) out vec4 DiffuseOpacity;
layout(location=1) out vec4 NormalRoughness;
layout(location=2) out vec4 SpecularShininess;
layout(location=3) out vec4 AmbientMetallic;
layout(location=4) out vec4 Emission;
vec3 Field(vec2 p,vec2 dx,vec2 dy)
{
    vec2 angles=2.0*Pi*fract(vec2(p.x+p.y,p.x-p.y));
    vec2 dxModes=vec2(dx.x+dx.y,dx.x-dx.y),dyModes=vec2(dy.x+dy.y,dy.x-dy.y);
    vec2 amplitudes=exp(-2.0*Pi*Pi*(dxModes*dxModes+dyModes*dyModes));
    const float FourierWeight=0.25;
    vec2 derivatives=-FourierWeight*2.0*Pi*amplitudes*sin(angles);
    return vec3(0.5+FourierWeight*dot(amplitudes,cos(angles)),derivatives.x+derivatives.y,derivatives.x-derivatives.y);
}
vec2 ProjectionChart(vec3 p,vec3 dx,vec3 dy)
{
    vec3 dominant=abs(cross(dx,dy)),normalized=(p-HeightFrame.ObjectBounds.xyz)*HeightFrame.ObjectBounds.w;
    if (dominant.z>=dominant.x && dominant.z>=dominant.y) return normalized.xy;
    return dominant.y>=dominant.x?normalized.xz:normalized.yz;
}
struct ChartFrame { vec3 u,v; vec2 direction; float thickness,cosine; bool valid; };
ChartFrame Parameterize(vec3 px,vec3 py,vec2 hx,vec2 hy,vec3 n,vec3 view)
{
    ChartFrame result=ChartFrame(vec3(0),vec3(0),vec2(0),0.0,0.0,false);
    float jacobian=hx.x*hy.y-hx.y*hy.x;
    if (!ValidChart(hx,hy)) return result;
    result.u=(px*hy.y-py*hx.y)/jacobian; result.v=(py*hx.x-px*hy.x)/jacobian;
    result.u-=n*dot(n,result.u); result.v-=n*dot(n,result.v);
    float uu=dot(result.u,result.u),uv=dot(result.u,result.v),vv=dot(result.v,result.v),determinant=uu*vv-uv*uv;
    if (min(uu,vv)<=VectorFloor || determinant<=GramFloor*uu*vv) return result;
    result.cosine=max(dot(n,view),0.0);
    result.thickness=HeightFrame.Shape.x*sqrt(min(uu,vv))*smoothstep(HeightFrame.Shape.w,2.0*HeightFrame.Shape.w,result.cosine);
    vec3 shift=-view*result.thickness/max(result.cosine,HeightFrame.Shape.w);
    float a=dot(shift,result.u),b=dot(shift,result.v);
    result.direction=vec2(a*vv-b*uv,b*uu-a*uv)/determinant;
    float distance=length(result.direction),limit=distance>TravelLimit?TravelLimit/distance:1.0;
    result.direction*=limit; result.thickness*=limit; result.valid=true; return result;
}
vec3 FirstCrossing(vec2 uv,vec2 direction,vec2 dx,vec2 dy,float cosine)
{
    uint count=uint(ceil(mix(float(HeightFrame.Steps.y),float(HeightFrame.Steps.x),clamp(cosine,0.0,1.0))));
    float nearDepth=0.0,farDepth=0.0,nearError=Field(uv,dx,dy).x-1.0,farError=nearError;
    if (nearError<0.0)
    {
        for (uint layer=1u;layer<=count;++layer)
        {
            farDepth=float(layer)/float(count); farError=farDepth+Field(uv+direction*farDepth,dx,dy).x-1.0;
            if (farError>=0.0) break;
            nearDepth=farDepth; nearError=farError;
        }
        for (uint level=0u;level<HeightFrame.Steps.z;++level)
        {
            float center=(nearDepth+farDepth)/2.0,value=center+Field(uv+direction*center,dx,dy).x-1.0;
            if (value<0.0) { nearDepth=center; nearError=value; } else { farDepth=center; farError=value; }
        }
    }
    float denominator=farError-nearError,fraction=abs(denominator)>RootFloor?clamp(-nearError/denominator,0.0,1.0):0.0;
    float depth=mix(nearDepth,farDepth,fraction); return vec3(uv+direction*depth,depth);
}
void main()
{
    vec3 px=dFdx(WorldPosition),py=dFdy(WorldPosition);
    vec2 tx=dFdx(MaterialUV),ty=dFdy(MaterialUV); float uvDet=tx.x*ty.y-tx.y*ty.x;
    vec3 n=DirectionOr(WorldNormal,vec3(0,1,0));
    if (Draw.TextureFlags.w==FlatShading) { vec3 geometric=DirectionOr(cross(px,py),n); n=dot(geometric,n)<0.0?-geometric:geometric; }
    if (!gl_FrontFacing) n=-n;
    vec2 base=MaterialUV*HeightFrame.Shape.y;
    if (HeightFrame.Steps.w==CoordinatesObject || (HeightFrame.Steps.w==CoordinatesAutomatic && !ValidChart(tx,ty)))
        base=ProjectionChart(ObjectPosition,dFdx(ObjectPosition),dFdy(ObjectPosition))*HeightFrame.Shape.y;
    vec2 hx=dFdx(base),hy=dFdy(base),uv=MaterialUV;
    ChartFrame frame=Parameterize(px,py,hx,hy,n,DirectionOr(Draw.EyeLightCount.xyz-WorldPosition,n));
    vec3 hit=vec3(base,0); bool heightEnabled=frame.valid && HeightFrame.Shape.x>0.0 && Draw.TextureFlags.w!=FlatShading && Draw.Render.y<=WireThreshold;
    if (heightEnabled)
    {
        hit=FirstCrossing(base,frame.direction,hx,hy,frame.cosine);
        float determinant=hx.x*hy.y-hx.y*hy.x;
        vec2 u=(tx*hy.y-ty*hx.y)/determinant,v=(ty*hx.x-tx*hy.x)/determinant;
        uv+=u*(hit.x-base.x)+v*(hit.y-base.y);
        vec2 slope=Field(hit.xy,hx,hy).yz*frame.thickness*HeightFrame.Shape.z;
        vec3 bumped=cross(frame.u+n*slope.x,frame.v+n*slope.y); n=DirectionOr(dot(bumped,n)<0.0?-bumped:bumped,n);
    }
    vec4 albedo=Draw.TextureFlags.x!=0u?textureGrad(Albedo,uv,tx,ty):vec4(1);
    vec3 surfaceMap = Draw.TextureFlags.z != 0u ? textureGrad(Coverage,uv,tx,ty).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Draw.Shading.xy;
    if (Draw.TextureFlags.z > 1u) roughnessMetallic *= surfaceMap.gb;
    float alpha=surfaceMap.r;
    alpha=clamp(alpha*albedo.a*VertexColor.a*Draw.Material.y,0.0,1.0);
    if (Draw.Material.z>0.0 && alpha<Draw.Material.z) discard;
    if (Draw.TextureFlags.y!=0u && Draw.TextureFlags.w!=FlatShading && ValidChart(tx,ty))
    {
        vec3 u=(px*ty.y-py*tx.y)/uvDet,v=(py*tx.x-px*ty.x)/uvDet;
        u=DirectionOr(u-n*dot(n,u),DirectionOr(cross(vec3(0,1,0),n),vec3(1,0,0)));
        float handedness=dot(cross(n,u),v)<0.0?-1.0:1.0; v=DirectionOr(cross(n,u)*handedness,vec3(0,0,1));
        vec3 local=textureGrad(TangentNormal,uv,tx,ty).xyz*2.0-1.0; n=DirectionOr(u*local.x+v*local.y+n*local.z,n);
    }
    vec3 color=VertexColor.rgb*max(Draw.PipelineColor.rgb,vec3(0))*max(albedo.rgb,vec3(0)); float metallic=clamp(roughnessMetallic.y,0.0,1.0);
    vec3 ambient=color*max(Draw.AmbientColor.rgb,vec3(0))*max(Draw.AmbientLight.rgb,vec3(0))*max(Draw.AmbientLight.w,0.0);
    if (Draw.TextureFlags.w==PhysicallyBased) ambient*=1.0-metallic;
    DiffuseOpacity=vec4(clamp(color*max(Draw.DiffuseColor.rgb,vec3(0)),0.0,HalfFloatCeiling),alpha);
    NormalRoughness=vec4(n,clamp(roughnessMetallic.x,MinimumRoughness,1.0));
    SpecularShininess=clamp(vec4(Draw.SpecularColor.rgb,max(Draw.Material.x,1.0)),0.0,HalfFloatCeiling);
    AmbientMetallic=vec4(clamp(ambient,0.0,HalfFloatCeiling),metallic); Emission=vec4(clamp(Draw.EmissionColor.rgb,0.0,HalfFloatCeiling),clamp(Draw.Material.w,0.0,1.0));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Draw.Render.y > 1.0 && Draw.Render.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (Draw.Render.y>WireThreshold) { NormalRoughness=vec4(0,0,0,-1); Emission.rgb=LinearColor(clamp(Draw.PipelineColor.rgb,0.0,1.0)); }
}
#endif
