// Independent D3D12 implementation: photographic global key (Reinhard 2002),
// temporal adaptation (Goodnight et al. 2003). The GPU retains sum/count data
// in FP32 and adapts EV in private presentation-transactional history slots.
cbuffer ExposureParameters : register(b0)
{
    uint4 OperationSizeHistory;
    float4 KeyRangeTime;
    float4 RatesRadianceBounds;
};
Texture2D<float4> InputImage : register(t0);
Texture2D<float4> ExposureHistory : register(t1);
static const uint Measure=0,Reduce=1,Adapt=2,Apply=3;
static const uint BlockExtent=2;
static const float3 Rec709Weights=float3(0.2126,0.7152,0.0722);
float4 VSMain(uint vertex : SV_VertexID) : SV_POSITION
{
    float2 uv=float2((vertex<<1)&2,vertex&2);
    return float4(uv*float2(2,-2)+float2(-1,1),0,1);
}
float4 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    uint2 pixel=uint2(position.xy);
    if (OperationSizeHistory.x<=Reduce)
    {
        float2 sums=0;
        for (uint row=0;row<BlockExtent;++row) for (uint column=0;column<BlockExtent;++column)
        {
            uint2 location=pixel*BlockExtent+uint2(column,row);
            if (any(location>=OperationSizeHistory.yz)) continue;
            float4 value=InputImage.Load(int3(location,0));
            if (OperationSizeHistory.x==Reduce) sums+=value.xy;
            else
            {
                float3 bounded=all(isfinite(value.rgb))?clamp(value.rgb,0,RatesRadianceBounds.w):0;
                sums+=float2(log2(max(dot(bounded,Rec709Weights),RatesRadianceBounds.z)),1);
            }
        }
        return float4(sums,0,1);
    }
    if (OperationSizeHistory.x==Adapt)
    {
        float2 aggregate=InputImage.Load(int3(0,0,0)).xy;
        float logAverage=aggregate.x/max(aggregate.y,1);
        float desired=clamp(log2(KeyRangeTime.x)-logAverage,KeyRangeTime.y,KeyRangeTime.z);
        float exposure=desired;
        if (OperationSizeHistory.w!=0)
        {
            float old=clamp(ExposureHistory.Load(int3(0,0,0)).x,KeyRangeTime.y,KeyRangeTime.z);
            float seconds=desired>old?RatesRadianceBounds.x:RatesRadianceBounds.y;
            exposure=desired+(old-desired)*exp2(-KeyRangeTime.w/seconds);
        }
        return float4(exposure,desired,logAverage,1);
    }
    float4 source=InputImage.Load(int3(pixel,0));
    float multiplier=exp2(ExposureHistory.Load(int3(0,0,0)).x);
    return float4(min(max(source.rgb,0)*multiplier,RatesRadianceBounds.w),source.a);
}
