#include <metal_stdlib>
using namespace metal;
// Metal direct-diffuse diffusion. The original diffuse term is replaced once,
// while opaque alpha and the other lighting components remain untouched.
// Jimenez et al. (2015): https://www.iryoku.com/separable-sss/
constant uint KernelCapacity=25,CopyStage=0,HorizontalStage=1,CompositeStage=3;
constant float PixelCenter=0.5,EmptyDepth=1,MinimumW=1e-7,NormalTolerance=1e-12;
constant float MinimumMask=1e-6,MinimumWeight=1e-7,MinimumSigma=1e-6,HalfMaximum=65504;
struct DiffusionFrame
{
    float4 WorldFromClip[4];
    float4 Viewport;
    float4 Diffusion; // World distance, strength, maximum pixel support, kernel support.
    float4 Rejection; // Optical depth scale, normal exponent, reserved, reserved.
    float4 CameraForward;
    uint4 Options; // Stage, tap count, reserved, reserved.
    float4 Kernel[KernelCapacity]; // RGB bin masses and signed profile-distance multiple.
};
struct Fullscreen { float4 Position [[position]]; };
vertex Fullscreen VSMain(uint index [[vertex_id]])
{
    float2 p=float2((index<<1)&2,index&2); return {float4(p*2-1,0,1)};
}
bool Unproject(float2 uv,float depth,constant DiffusionFrame& frame,thread float3& world)
{
    float4 p=float4(uv.x*2-1,1-uv.y*2,depth,1);
    float4 h=float4(dot(frame.WorldFromClip[0],p),dot(frame.WorldFromClip[1],p),
        dot(frame.WorldFromClip[2],p),dot(frame.WorldFromClip[3],p));
    world=float3(0); if (abs(h.w)<MinimumW) return false;
    world=h.xyz/h.w; return all(isfinite(world));
}
bool Surface(int2 pixel,constant DiffusionFrame& frame,texture2d<float,access::read> depth,
    texture2d<float,access::read> normals,thread float3& world,thread float3& normal)
{
    world=normal=float3(0);
    if (any(pixel<0) || any(pixel>=int2(frame.Viewport.xy))) return false;
    uint2 q=uint2(pixel); float z=depth.read(q).r; float4 n=normals.read(q); float squared=dot(n.xyz,n.xyz);
    if (!(z<EmptyDepth && n.w>=0 && squared>NormalTolerance && isfinite(squared))) return false;
    normal=n.xyz*rsqrt(squared); return Unproject((float2(pixel)+PixelCenter)*frame.Viewport.zw,z,frame,world);
}
void Accumulate(int2 pixel,float fraction,float3 kernel,float3 center,float3 normal,float sigma,
    constant DiffusionFrame& frame,texture2d<float,access::read> signal,texture2d<float,access::read> depth,
    texture2d<float,access::read> normals,thread float3& radiance,thread float3& weights)
{
    if (fraction<=0) return;
    float3 world,n; if (!Surface(pixel,frame,depth,normals,world,n)) return;
    float4 value=signal.read(uint2(pixel)); if (value.a<=MinimumMask) return;
    float distance=dot(world-center,frame.CameraForward.xyz)/sigma;
    float angular=frame.Rejection.y>0 ? pow(saturate(dot(normal,n)),frame.Rejection.y) : 1;
    float3 weight=kernel*(fraction*exp(-distance*distance/2)*angular*value.a);
    radiance+=value.rgb/value.a*weight; weights+=weight;
}
fragment float4 PSMain(Fullscreen input [[stage_in]],constant DiffusionFrame& frame [[buffer(0)]],
    texture2d<float,access::read> signal [[texture(0)]],texture2d<float,access::read> depth [[texture(1)]],
    texture2d<float,access::read> normals [[texture(2)]],texture2d<float,access::read> scene [[texture(3)]],
    texture2d<float,access::read> original [[texture(4)]])
{
    int2 pixel=int2(input.Position.xy); float4 center=signal.read(uint2(pixel));
    if (frame.Options.x==CopyStage) return center;
    if (frame.Options.x==CompositeStage)
    {
        float4 color=scene.read(uint2(pixel));
        float3 delta=(center.rgb-original.read(uint2(pixel)).rgb)*frame.Diffusion.y;
        return float4(clamp(color.rgb+delta,0.0f,HalfMaximum),color.a);
    }
    float3 world,normal; if (center.a<=MinimumMask || !Surface(pixel,frame,depth,normals,world,normal)) return center;
    int2 axis=frame.Options.x==HorizontalStage ? int2(1,0) : int2(0,1); float3 adjacent;
    float z=depth.read(uint2(pixel)).r;
    if (!Unproject((float2(pixel+axis)+PixelCenter)*frame.Viewport.zw,z,frame,adjacent)) return center;
    float footprint=length(adjacent-world); if (!(footprint>0)) return center;
    float support=frame.Diffusion.x*frame.Diffusion.w/footprint;
    float scale=support>frame.Diffusion.z && support>0 ? frame.Diffusion.z/support : 1;
    float pixelScale=frame.Diffusion.x*scale/footprint,sigma=max(frame.Diffusion.x*scale*frame.Rejection.x,MinimumSigma);
    float3 radiance=0,weights=0;
    for (uint tap=0;tap<min(frame.Options.y,KernelCapacity);++tap)
    {
        float offset=frame.Kernel[tap].w*pixelScale,lower=floor(offset),blend=offset-lower;
        int2 q=pixel+axis*int(lower);
        Accumulate(q,1-blend,frame.Kernel[tap].rgb,world,normal,sigma,frame,signal,depth,normals,radiance,weights);
        Accumulate(q+axis,blend,frame.Kernel[tap].rgb,world,normal,sigma,frame,signal,depth,normals,radiance,weights);
    }
    return float4(clamp(radiance/max(weights,float3(MinimumWeight))*center.a,0.0f,HalfMaximum),center.a);
}
