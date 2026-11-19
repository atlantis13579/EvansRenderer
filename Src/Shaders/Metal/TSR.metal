#include <metal_stdlib>
using namespace metal;
// GPU image ingress/egress is separate from immutable history storage. A later
// display pass may overwrite its own HDR image without mutating that history.
struct TextureBridgeFrame { uint4 ExtentFlags; };
constant uint PacketVectors=5,FinalField=0,OpaqueField=1,MotionField=2,DepthField=3,OutlineField=4;
constant uint MotionPresent=1,OutlinePresent=2,ReactivePresent=4;
kernel void tsrPack(constant TextureBridgeFrame& frame [[buffer(0)]],device float4* packet [[buffer(1)]],
    texture2d<float,access::read> finalHDR [[texture(0)]],texture2d<float,access::read> opaqueHDR [[texture(1)]],
    texture2d<float,access::read> velocity [[texture(2)]],depth2d<float,access::read> depth [[texture(3)]],
    texture2d<float,access::read> outline [[texture(4)]],texture2d<float,access::read> extraReactive [[texture(5)]],uint2 at [[thread_position_in_grid]])
{
    if (any(at>=frame.ExtentFlags.xy)) return;
    uint first=(at.y*frame.ExtentFlags.x+at.x)*PacketVectors;
    float4 motion=0,overlay=0; float reactive=0;
    if ((frame.ExtentFlags.z&MotionPresent)!=0) motion=velocity.read(at);
    if ((frame.ExtentFlags.z&OutlinePresent)!=0) overlay=outline.read(at);
    if ((frame.ExtentFlags.z&ReactivePresent)!=0) reactive=extraReactive.read(at).r;
    packet[first+FinalField]=finalHDR.read(at);
    packet[first+OpaqueField]=opaqueHDR.read(at);
    packet[first+MotionField]=motion;
    packet[first+DepthField]=float4(depth.read(at),reactive,0,0);
    packet[first+OutlineField]=overlay;
}
struct ExportVertex { float4 Position [[position]]; };
vertex ExportVertex tsrExportVertex(uint vertex [[vertex_id]])
{
    constexpr float TriangleReach=3;
    float2 p=vertex==0?float2(-1):vertex==1?float2(-1,TriangleReach):float2(TriangleReach,-1);
    ExportVertex result; result.Position=float4(p,0,1); return result;
}
fragment float4 tsrExportFragment(ExportVertex vertex [[stage_in]],constant TextureBridgeFrame& frame [[buffer(0)]],
    device const float4* history [[buffer(1)]])
{
    constexpr uint HistoryVectors=2;
    uint2 at=uint2(vertex.Position.xy);
    return history[(at.y*frame.ExtentFlags.x+at.x)*HistoryVectors];
}
// Native Metal HDR temporal reconstruction. Row arrays preserve the host camera
// ABI without relying on a matrix-major convention or another API's source.
// Salvi, "An Excursion in Temporal Supersampling", GDC 2016; AMD, "Temporal
// Upscaling: Past, Present, and Future", GDC 2023. Vendor SDK integrations are outside the project scope.
struct ReconstructionFrame
{
    float4 CurrentInverseRows[4],PreviousInverseRows[4];
    float4 CurrentDepthPlane,PreviousDepthPlane;
    uint4 Extent;
    float4 JitterQuality,Controls;
    uint4 Options;
};
constant uint InputStride=5,HistoryStride=2;
constant int FilterRadius=2,MotionRadius=1;
constant float PixelCenter=0.5,ClearDepth=1,Pi=3.14159265358979323846;
constant float WFloor=1e-7,DepthFloor=1e-4,ColorFloor=1e-4,WeightFloor=1e-6,RadianceLimit=65504;
constant float UnsupportedObservation=0.001;
constant float NeighborhoodSamples=(FilterRadius*2)*(FilterRadius*2);
float4 inputAt(device const float4* input,constant ReconstructionFrame& f,int2 p,uint field)
{
    p=clamp(p,int2(0),int2(f.Extent.xy)-1);
    return input[(uint(p.y)*f.Extent.x+uint(p.x))*InputStride+field];
}
float4 previousAt(device const float4* history,constant ReconstructionFrame& f,int2 p,uint field)
{
    p=clamp(p,int2(0),int2(f.Extent.zw)-1);
    return history[(uint(p.y)*f.Extent.z+uint(p.x))*HistoryStride+field];
}
float maximumComponent(float3 c) { return max(c.x,max(c.y,c.z)); }
float3 toYCoCg(float3 c) { return float3((c.r+2*c.g+c.b)/4,(c.r-c.b)/2,(-c.r+2*c.g-c.b)/4); }
float3 toRGB(float3 c) { return float3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z); }
float lanczos(float distance)
{
    float x=abs(distance); if (x>=FilterRadius) return 0; if (x<WeightFloor) return 1;
    return sin(Pi*x)*sin(Pi*x/FilterRadius)/(Pi*Pi*x*x/FilterRadius);
}
float optical(constant ReconstructionFrame& f,float2 uv,float depth,bool previous)
{
    float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1),world;
    if (previous) world=float4(dot(f.PreviousInverseRows[0],clip),dot(f.PreviousInverseRows[1],clip),dot(f.PreviousInverseRows[2],clip),dot(f.PreviousInverseRows[3],clip));
    else world=float4(dot(f.CurrentInverseRows[0],clip),dot(f.CurrentInverseRows[1],clip),dot(f.CurrentInverseRows[2],clip),dot(f.CurrentInverseRows[3],clip));
    if (abs(world.w)<=WFloor) return -1;
    return dot(previous?f.PreviousDepthPlane:f.CurrentDepthPlane,float4(world.xyz/world.w,1));
}
int2 floorRenderFraction(float2 value)
{
    constexpr float BoundaryRoundoff=1e-6f;
    float2 closest=round(value);
    return int2(floor(select(value,closest,abs(value-closest)<=BoundaryRoundoff)));
}
kernel void tsrReconstruct(constant ReconstructionFrame& f [[buffer(0)]],device const float4* input [[buffer(1)]],
    device const float4* previous [[buffer(2)]],device float4* reconstructed [[buffer(3)]],device float4* auxiliary [[buffer(4)]],uint2 id [[thread_position_in_grid]])
{
    if (any(id>=f.Extent.zw)) return;
    // Pixel-center products fit uint under the host's bounded extent contract.
    uint2 divisor=f.Extent.zw*2u,numerator=(id*2u+1u)*f.Extent.xy;
    uint2 whole=numerator/divisor;
    float2 fraction=float2(numerator%divisor)/float2(divisor)+f.JitterQuality.xy;
    float2 sample=float2(whole)+fraction;
    int2 base=int2(whole)+floorRenderFraction(fraction-PixelCenter);
    float2 footprint=float2(f.Extent.xy)/float2(f.Extent.zw);
    float3 total=0,wideTotal=0,mean=0,squared=0,lo=RadianceLimit,hi=0; float alpha=0,weights=0,wideWeights=0,reactive=0;
    for (int y=1-FilterRadius;y<=FilterRadius;++y) for (int x=1-FilterRadius;x<=FilterRadius;++x)
    {
        int2 p=base+int2(x,y); float4 color=inputAt(input,f,p,0),opaque=inputAt(input,f,p,1);
        float3 rgb=clamp(color.rgb,0.0f,RadianceLimit),yc=toYCoCg(rgb);
        float2 offset=float2(p)+PixelCenter-sample;
        float w=lanczos(offset.x/footprint.x)*lanczos(offset.y/footprint.y),wide=lanczos(offset.x)*lanczos(offset.y);
        total+=rgb*w; wideTotal+=rgb*wide; alpha+=color.a*wide; weights+=w; wideWeights+=wide; mean+=yc; squared+=yc*yc; lo=min(lo,rgb); hi=max(hi,rgb);
        float scale=max(max(maximumComponent(rgb),maximumComponent(opaque.rgb)),ColorFloor);
        reactive=max(reactive,max(inputAt(input,f,p,3).y,maximumComponent(abs(color.rgb-opaque.rgb))/scale*f.Controls.y));
    }
    bool supported=weights>WeightFloor;
    float3 fresh;
    if (supported) fresh=total/weights; else fresh=wideTotal/max(wideWeights,WeightFloor);
    fresh=clamp(fresh,lo,hi); float3 color=fresh;
    alpha=clamp(alpha/max(wideWeights,WeightFloor),0.0f,1.0f); reactive=clamp(reactive,0.0f,1.0f);
    int2 center=int2(whole)+floorRenderFraction(fraction),selected=center; float depth=inputAt(input,f,selected,3).x;
    for (int y=-MotionRadius;y<=MotionRadius;++y) for (int x=-MotionRadius;x<=MotionRadius;++x)
    {
        int2 p=center+int2(x,y); float candidate=inputAt(input,f,p,3).x;
        if (candidate<depth) { depth=candidate; selected=p; }
    }
    selected=clamp(selected,int2(0),int2(f.Extent.xy)-1);
    float4 velocity=inputAt(input,f,selected,2); bool sky=depth>=ClearDepth;
    float2 motion=sky?float2(0):velocity.xy;
    float2 oldPixel=float2(id)+PixelCenter-motion*float2(f.Extent.zw)/float2(f.Extent.xy);
    float2 residual=(float2(center)+PixelCenter-sample)/footprint; float observation=supported?exp2(-dot(residual,residual)):UnsupportedObservation,count=observation;
    bool valid=f.Options.x!=0 && reactive<1 && (sky?f.Options.y!=0:velocity.w>0) && all(oldPixel>=PixelCenter) && all(oldPixel<=float2(f.Extent.zw)-PixelCenter);
    if (valid)
    {
        float expected=optical(f,(float2(selected)+PixelCenter-f.JitterQuality.xy-motion)/float2(f.Extent.xy),sky?ClearDepth:1-velocity.z,true);
        int2 first=int2(floor(oldPixel-PixelCenter)); bool agrees=false;
        for (int y=0;y<2;++y) for (int x=0;x<2;++x)
        {
            float4 meta=previousAt(previous,f,first+int2(x,y),1);
            agrees=agrees || (sky?meta.x>=ClearDepth:meta.x<ClearDepth && expected>0 && meta.z>0 && abs(meta.z-expected)<=max(DepthFloor,abs(expected)*f.JitterQuality.w));
        }
        if (agrees)
        {
            float2 fraction=oldPixel-PixelCenter-float2(first); float3 history=0; float oldCount=0;
            for (int y=0;y<2;++y) for (int x=0;x<2;++x)
            {
                float w=(x?fraction.x:1-fraction.x)*(y?fraction.y:1-fraction.y);
                history+=previousAt(previous,f,first+int2(x,y),0).rgb*w; oldCount+=previousAt(previous,f,first+int2(x,y),1).y*w;
            }
            mean/=NeighborhoodSamples; squared/=NeighborhoodSamples;
            float3 radius=max(sqrt(max(squared-mean*mean,0.0f))*f.Controls.x,ColorFloor);
            float3 delta=toYCoCg(history)-mean; history=toRGB(mean+delta/max(1.0f,maximumComponent(abs(delta/radius)))); oldCount=min(oldCount,f.Controls.z);
            float feedback=min(f.JitterQuality.z,oldCount/(oldCount+observation))*(1-reactive);
            color=mix(fresh,history,feedback); count=min(f.Controls.z,oldCount*(1-reactive)+observation);
        }
    }
    uint address=(id.y*f.Extent.z+id.x)*HistoryStride;
    float distance=sky?-1:optical(f,(float2(selected)+PixelCenter-f.JitterQuality.xy)/float2(f.Extent.xy),depth,false);
    reconstructed[address]=float4(clamp(color,0.0f,RadianceLimit),alpha); reconstructed[address+1]=float4(depth,count,distance,reactive);
    auxiliary[address]=float4(velocity.xy*float2(f.Extent.zw)/float2(f.Extent.xy),velocity.w>0?1-velocity.z:0,velocity.w); auxiliary[address+1]=inputAt(input,f,center,4);
}
