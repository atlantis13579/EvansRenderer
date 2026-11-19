// Global log-average exposure with exponential EV adaptation (Goodnight et al.,
// EGSR 2003). Sum/count reduction preserves odd borders. History belongs to
// successful presents; the first/reset frame snaps to its measured target.
cbuffer ExposureFrame : register(b0)
{
    uint4 PhaseExtentHistory;
    float4 KeyLimitsDelta;
    float4 HalfLivesLimits;
};
Texture2D<float4> Source : register(t0);
Texture2D<float4> PreviousExposure : register(t1);
static const uint MeterPhase=0,ReducePhase=1,AdaptPhase=2,CompositePhase=3;
static const uint ReductionWidth=2;
static const float3 LinearRec709Luminance=float3(0.2126,0.7152,0.0722);
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint index : SV_VertexID)
{
    ScreenVertex output;
    float2 corner=float2((index<<1)&2,index&2);
    output.Position=float4(corner*float2(2,-2)+float2(-1,1),0,1);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint2 pixel=uint2(input.Position.xy);
    const uint phase=PhaseExtentHistory.x;
    if (phase==MeterPhase || phase==ReducePhase)
    {
        float2 total=0;
        for (uint y=0;y<ReductionWidth;++y) for (uint x=0;x<ReductionWidth;++x)
        {
            uint2 samplePixel=pixel*ReductionWidth+uint2(x,y);
            if (any(samplePixel>=PhaseExtentHistory.yz)) continue;
            float4 sampleValue=Source.Load(int3(samplePixel,0));
            if (phase==ReducePhase) total+=sampleValue.xy;
            else
            {
                float3 radiance=all(isfinite(sampleValue.rgb))?clamp(sampleValue.rgb,0,HalfLivesLimits.w):0;
                total+=float2(log2(max(dot(radiance,LinearRec709Luminance),HalfLivesLimits.z)),1);
            }
        }
        return float4(total,0,1);
    }
    if (phase==AdaptPhase)
    {
        float2 meter=Source.Load(int3(0,0,0)).xy;
        float meanLog=meter.x/max(meter.y,1);
        float target=clamp(log2(KeyLimitsDelta.x)-meanLog,KeyLimitsDelta.y,KeyLimitsDelta.z);
        float adapted=target;
        if (PhaseExtentHistory.w!=0)
        {
            float previous=clamp(PreviousExposure.Load(int3(0,0,0)).x,KeyLimitsDelta.y,KeyLimitsDelta.z);
            float halfLife=target>previous?HalfLivesLimits.x:HalfLivesLimits.y;
            float oldWeight=exp2(-KeyLimitsDelta.w/halfLife);
            adapted=lerp(target,previous,oldWeight);
        }
        return float4(adapted,target,meanLog,1);
    }
    float4 color=Source.Load(int3(pixel,0));
    float scale=exp2(PreviousExposure.Load(int3(0,0,0)).x);
    return float4(min(max(color.rgb,0)*scale,HalfLivesLimits.w),color.a);
}
