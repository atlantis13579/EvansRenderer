#include <metal_stdlib>
using namespace metal;

// Independent MSL global exposure. FP32 sum/count images preserve the complete
// scene measure; separate presented/pending EV images make history transactional.
// Goodnight et al., Interactive Time-Dependent Tone Mapping, EGSR 2003.
struct ExposureFrame
{
    uint4 PhaseExtentHistory;
    float4 KeyLimitsDelta;
    float4 HalfLivesLimits;
};
constant uint MeterPhase=0,ReducePhase=1,AdaptPhase=2,CompositePhase=3;
constant uint ReductionWidth=2;
constant float3 LinearRec709Luminance=float3(0.2126,0.7152,0.0722);
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex VSMain(uint index [[vertex_id]])
{
    float2 corner=float2((index<<1)&2,index&2);
    return {float4(corner*float2(2,-2)+float2(-1,1),0,1)};
}
fragment float4 PSMain(ScreenVertex input [[stage_in]],constant ExposureFrame& frame [[buffer(0)]],
    texture2d<float,access::read> source [[texture(0)]],texture2d<float,access::read> previousExposure [[texture(1)]])
{
    uint2 pixel=uint2(input.Position.xy);
    if (frame.PhaseExtentHistory.x==MeterPhase || frame.PhaseExtentHistory.x==ReducePhase)
    {
        float2 total=0;
        for (uint y=0;y<ReductionWidth;++y) for (uint x=0;x<ReductionWidth;++x)
        {
            uint2 coordinate=pixel*ReductionWidth+uint2(x,y);
            if (any(coordinate>=frame.PhaseExtentHistory.yz)) continue;
            float4 value=source.read(coordinate);
            if (frame.PhaseExtentHistory.x==ReducePhase) total+=value.xy;
            else
            {
                float3 radiance=all(isfinite(value.rgb))?clamp(value.rgb,0.0f,frame.HalfLivesLimits.w):float3(0);
                total+=float2(log2(max(dot(radiance,LinearRec709Luminance),frame.HalfLivesLimits.z)),1);
            }
        }
        return float4(total,0,1);
    }
    if (frame.PhaseExtentHistory.x==AdaptPhase)
    {
        float2 meter=source.read(uint2(0)).xy;
        float meanLog=meter.x/max(meter.y,1.0f);
        float target=clamp(log2(frame.KeyLimitsDelta.x)-meanLog,frame.KeyLimitsDelta.y,frame.KeyLimitsDelta.z);
        float adapted=target;
        if (frame.PhaseExtentHistory.w!=0)
        {
            float previous=clamp(previousExposure.read(uint2(0)).x,frame.KeyLimitsDelta.y,frame.KeyLimitsDelta.z);
            float halfLife=target>previous?frame.HalfLivesLimits.x:frame.HalfLivesLimits.y;
            adapted=mix(target,previous,exp2(-frame.KeyLimitsDelta.w/halfLife));
        }
        return float4(adapted,target,meanLog,1);
    }
    float4 color=source.read(pixel);
    float scale=exp2(previousExposure.read(uint2(0)).x);
    return float4(min(max(color.rgb,0.0f)*scale,frame.HalfLivesLimits.w),color.a);
}
