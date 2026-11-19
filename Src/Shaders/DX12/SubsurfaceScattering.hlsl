// DX12 direct-diffuse diffusion. The source retains sample coverage separately
// from its radiance; depth/normal guidance prevents cross-silhouette transport.
// Jimenez et al. (2015): https://www.iryoku.com/separable-sss/
// Two projected normalized-diffusion marginals approximate the radial profile.
static const uint TapCapacity=25,CopyStage=0,HorizontalStage=1,CompositeStage=3;
static const float Center=0.5,EmptyDepth=1,WThreshold=1e-7,NormalThreshold=1e-12;
static const float MaskThreshold=1e-6,WeightThreshold=1e-7,DepthSigmaFloor=1e-6,HalfMaximum=65504;
cbuffer DiffusionParameters : register(b0)
{
    row_major float4x4 WorldFromClip;
    float4 FrameSize;
    float4 Scatter; // World distance, strength, maximum pixel support, profile support.
    float4 EdgeControl; // Optical depth scale, normal exponent, reserved, reserved.
    float4 OpticalForward;
    uint4 Control; // Stage, tap count, reserved, reserved.
    float4 Profile[TapCapacity]; // RGB probability masses and signed world-distance scale.
};
Texture2D<float4> DiffuseInput : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> SurfaceNormal : register(t2),OpaqueColor : register(t3),UnfilteredDiffuse : register(t4);
struct Fullscreen { float4 Position:SV_POSITION; };
Fullscreen VSMain(uint index:SV_VertexID)
{
    uint2 corner=uint2((index<<1)&2,index&2); Fullscreen output;
    output.Position=float4(float2(corner)*2-1,0,1); return output;
}
bool WorldPosition(float2 uv,float z,out float3 world)
{
    float4 h=mul(WorldFromClip,float4(uv.x*2-1,1-uv.y*2,z,1));
    world=0; if (abs(h.w)<WThreshold) return false;
    world=h.xyz/h.w; return all(isfinite(world));
}
bool ReadSurface(int2 coordinate,out float3 world,out float3 normal)
{
    world=normal=0;
    if (any(coordinate<0) || any(coordinate>=int2(FrameSize.xy))) return false;
    float depth=OpaqueDepth.Load(int3(coordinate,0)); float4 encoded=SurfaceNormal.Load(int3(coordinate,0));
    float magnitude=dot(encoded.xyz,encoded.xyz);
    if (!(depth<EmptyDepth && encoded.w>=0 && magnitude>NormalThreshold && isfinite(magnitude))) return false;
    normal=encoded.xyz*rsqrt(magnitude); return WorldPosition((float2(coordinate)+Center)*FrameSize.zw,depth,world);
}
void Gather(int2 coordinate,float linearWeight,float3 profile,float3 origin,float3 normal,float depthSigma,
    inout float3 accumulated,inout float3 normalization)
{
    if (linearWeight<=0) return;
    float3 world,n; if (!ReadSurface(coordinate,world,n)) return;
    float4 signal=DiffuseInput.Load(int3(coordinate,0)); if (signal.a<=MaskThreshold) return;
    float delta=dot(world-origin,OpticalForward.xyz)/depthSigma;
    float angular=EdgeControl.y>0 ? pow(saturate(dot(normal,n)),EdgeControl.y) : 1;
    float3 weight=profile*(linearWeight*exp(-delta*delta/2)*angular*signal.a);
    accumulated+=signal.rgb/signal.a*weight; normalization+=weight;
}
float4 PSMain(Fullscreen input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy); float4 signal=DiffuseInput.Load(int3(pixel,0));
    if (Control.x==CopyStage) return signal;
    if (Control.x==CompositeStage)
    {
        float4 scene=OpaqueColor.Load(int3(pixel,0));
        float3 replacement=(signal.rgb-UnfilteredDiffuse.Load(int3(pixel,0)).rgb)*Scatter.y;
        return float4(clamp(scene.rgb+replacement,0,HalfMaximum),scene.a);
    }
    float3 world,normal; if (signal.a<=MaskThreshold || !ReadSurface(pixel,world,normal)) return signal;
    int2 axis=Control.x==HorizontalStage ? int2(1,0) : int2(0,1);
    float3 adjacent; float z=OpaqueDepth.Load(int3(pixel,0));
    if (!WorldPosition((float2(pixel+axis)+Center)*FrameSize.zw,z,adjacent)) return signal;
    float footprint=length(adjacent-world); if (!(footprint>0)) return signal;
    float support=Scatter.x*Scatter.w/footprint;
    float limitScale=support>Scatter.z && support>0 ? Scatter.z/support : 1;
    float pixelScale=Scatter.x*limitScale/footprint,depthSigma=max(Scatter.x*limitScale*EdgeControl.x,DepthSigmaFloor);
    float3 accumulated=0,normalization=0;
    [loop] for (uint tap=0;tap<min(Control.y,TapCapacity);++tap)
    {
        float offset=Profile[tap].w*pixelScale,base=floor(offset),fraction=offset-base;
        int2 q=pixel+axis*int(base);
        Gather(q,1-fraction,Profile[tap].rgb,world,normal,depthSigma,accumulated,normalization);
        Gather(q+axis,fraction,Profile[tap].rgb,world,normal,depthSigma,accumulated,normalization);
    }
    // RGB normalization preserves a constant irradiance independently of the
    // rejected taps. Center coverage/nonmetal weight is restored only once.
    return float4(clamp(accumulated/max(normalization,WeightThreshold)*signal.a,0,HalfMaximum),signal.a);
}
