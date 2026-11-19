// Direct diffuse radiance is produced separately by deferred lighting. These
// stages diffuse that signal and replace its contribution in opaque HDR.
// Jimenez et al. (2015): https://www.iryoku.com/separable-sss/
// The RGB kernel is a separable projection of normalized diffusion; it does not
// transport specular reflection, ambient illumination, emission or transmission.
static const uint KernelSize=25u,SourcePass=0u,HorizontalPass=1u,CompositePass=3u;
static const float PixelCenter=0.5f,ClearDepth=1.0f,MinimumW=1.0e-7f;
static const float VectorTolerance=1.0e-12f,MinimumMask=1.0e-6f,MinimumWeight=1.0e-7f;
static const float MinimumDepthSigma=1.0e-6f,MaximumRadiance=65504.0f;
cbuffer SubsurfaceFrame : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 Viewport;
    float4 Diffusion; // World distance, strength, maximum pixel support, kernel support.
    float4 Rejection; // Optical depth scale, normal exponent, reserved, reserved.
    float4 CameraForward;
    uint4 Options; // Pass, tap count, reserved, reserved.
    float4 Kernel[KernelSize]; // RGB bin masses, signed distance multiplier.
};
Texture2D<float4> DiffuseSignal : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> NormalRoughness : register(t2);
Texture2D<float4> SceneColor : register(t3);
Texture2D<float4> OriginalDiffuse : register(t4);
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint vertex:SV_VertexID)
{
    float2 p=float2((vertex<<1u)&2u,vertex&2u);
    ScreenVertex result; result.Position=float4(p*2-1,0,1); return result;
}
bool Unproject(float2 uv,float depth,out float3 position)
{
    float4 world=mul(InverseViewProjection,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    position=0; if (abs(world.w)<MinimumW) return false;
    position=world.xyz/world.w; return all(isfinite(position));
}
bool Surface(int2 pixel,out float3 position,out float3 normal)
{
    position=normal=0;
    if (any(pixel<0) || any(pixel>=int2(Viewport.xy))) return false;
    float depth=SceneDepth.Load(int3(pixel,0)); float4 n=NormalRoughness.Load(int3(pixel,0));
    float squared=dot(n.xyz,n.xyz);
    if (!(depth<ClearDepth && n.w>=0 && squared>VectorTolerance && isfinite(squared))) return false;
    normal=n.xyz*rsqrt(squared); return Unproject((float2(pixel)+PixelCenter)*Viewport.zw,depth,position);
}
void Accumulate(int2 pixel,float fraction,float3 kernel,float3 center,float3 normal,float sigma,
    inout float3 radiance,inout float3 weights)
{
    if (fraction<=0) return;
    float3 position,n; if (!Surface(pixel,position,n)) return;
    float4 signal=DiffuseSignal.Load(int3(pixel,0)); if (signal.a<=MinimumMask) return;
    float dz=dot(position-center,CameraForward.xyz)/sigma;
    float normalWeight=Rejection.y>0 ? pow(saturate(dot(normal,n)),Rejection.y) : 1;
    float3 weight=kernel*(fraction*exp(-dz*dz/2)*normalWeight*signal.a);
    // The seed is premultiplied by signal coverage and the nonmetal fraction.
    // Unpremultiply only for transport; restore center coverage after filtering.
    radiance+=signal.rgb/signal.a*weight; weights+=weight;
}
float4 PSMain(ScreenVertex input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy); float4 center=DiffuseSignal.Load(int3(pixel,0));
    if (Options.x==SourcePass) return center;
    if (Options.x==CompositePass)
    {
        float4 scene=SceneColor.Load(int3(pixel,0));
        float3 delta=(center.rgb-OriginalDiffuse.Load(int3(pixel,0)).rgb)*Diffusion.y;
        return float4(clamp(scene.rgb+delta,0,MaximumRadiance),scene.a);
    }
    float3 position,normal; if (center.a<=MinimumMask || !Surface(pixel,position,normal)) return center;
    int2 axis=Options.x==HorizontalPass ? int2(1,0) : int2(0,1);
    float depth=SceneDepth.Load(int3(pixel,0)); float3 adjacent;
    if (!Unproject((float2(pixel+axis)+PixelCenter)*Viewport.zw,depth,adjacent)) return center;
    float footprint=length(adjacent-position); if (!(footprint>0)) return center;
    float requestedSupport=Diffusion.x*Diffusion.w/footprint;
    float scale=requestedSupport>Diffusion.z && requestedSupport>0 ? Diffusion.z/requestedSupport : 1;
    float pixelsPerDistance=Diffusion.x*scale/footprint;
    float sigma=max(Diffusion.x*scale*Rejection.x,MinimumDepthSigma);
    float3 radiance=0,weights=0;
    [loop] for (uint tap=0;tap<min(Options.y,KernelSize);++tap)
    {
        float offset=Kernel[tap].w*pixelsPerDistance,lower=floor(offset),blend=offset-lower;
        int2 q=pixel+axis*int(lower);
        Accumulate(q,1-blend,Kernel[tap].rgb,position,normal,sigma,radiance,weights);
        Accumulate(q+axis,blend,Kernel[tap].rgb,position,normal,sigma,radiance,weights);
    }
    return float4(clamp(radiance/max(weights,MinimumWeight)*center.a,0,MaximumRadiance),center.a);
}
