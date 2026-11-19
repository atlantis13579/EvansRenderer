#version 450
// Standalone Vulkan GLSL, compiled at runtime. Photographic log-average
// metering (Reinhard 2002) and an asymmetric exponential EV response. FP32
// sums/counts preserve partial reduction blocks at all image boundaries.
#if defined(EVANS_VERTEX_STAGE)
void main()
{
    vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
layout(std140,set=0,binding=0) uniform ExposureFrame
{
    uvec4 PhaseExtentHistory;
    vec4 KeyLimitsDelta;
    vec4 HalfLivesLimits;
};
layout(set=0,binding=1) uniform sampler2D Source;
layout(set=0,binding=2) uniform sampler2D PreviousExposure;
layout(location=0) out vec4 Result;
const uint MeterPhase=0u,ReducePhase=1u,AdaptPhase=2u,CompositePhase=3u;
const int ReductionWidth=2;
const vec3 LinearRec709Luminance=vec3(0.2126,0.7152,0.0722);
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (PhaseExtentHistory.x==MeterPhase || PhaseExtentHistory.x==ReducePhase)
    {
        vec2 total=vec2(0);
        for (int y=0;y<ReductionWidth;++y) for (int x=0;x<ReductionWidth;++x)
        {
            ivec2 coordinate=pixel*ReductionWidth+ivec2(x,y);
            if (any(greaterThanEqual(coordinate,ivec2(PhaseExtentHistory.yz)))) continue;
            vec4 value=texelFetch(Source,coordinate,0);
            if (PhaseExtentHistory.x==ReducePhase) total+=value.xy;
            else
            {
                bool valid=!any(isnan(value.rgb)) && !any(isinf(value.rgb));
                vec3 radiance=valid?clamp(value.rgb,vec3(0),vec3(HalfLivesLimits.w)):vec3(0);
                total+=vec2(log2(max(dot(radiance,LinearRec709Luminance),HalfLivesLimits.z)),1);
            }
        }
        Result=vec4(total,0,1); return;
    }
    if (PhaseExtentHistory.x==AdaptPhase)
    {
        vec2 meter=texelFetch(Source,ivec2(0),0).xy;
        float meanLog=meter.x/max(meter.y,1.0);
        float target=clamp(log2(KeyLimitsDelta.x)-meanLog,KeyLimitsDelta.y,KeyLimitsDelta.z);
        float adapted=target;
        if (PhaseExtentHistory.w!=0u)
        {
            float previous=clamp(texelFetch(PreviousExposure,ivec2(0),0).x,KeyLimitsDelta.y,KeyLimitsDelta.z);
            float halfLife=target>previous?HalfLivesLimits.x:HalfLivesLimits.y;
            adapted=mix(target,previous,exp2(-KeyLimitsDelta.w/halfLife));
        }
        Result=vec4(adapted,target,meanLog,1); return;
    }
    vec4 color=texelFetch(Source,pixel,0);
    float scale=exp2(texelFetch(PreviousExposure,ivec2(0),0).x);
    Result=vec4(min(max(color.rgb,vec3(0))*scale,vec3(HalfLivesLimits.w)),color.a);
}
#endif
