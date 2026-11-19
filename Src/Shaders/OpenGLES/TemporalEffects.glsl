#version 310 es
// OpenGL ES owns this implementation. Native bottom-left coordinates are used
// consistently for all levels, history and composition; no reprojection occurs.
// See Goodnight et al., Interactive Time-Dependent Tone Mapping (EGSR 2003).
precision highp float;
precision highp int;
#if defined(EVANS_VERTEX_STAGE)
void main()
{
    vec2 triangle=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(triangle*2.0-1.0,0,1);
}
#else
layout(std140,binding=0) uniform ExposureParameters
{
    uvec4 OperationSizeHistory;
    vec4 KeyRangeTime;
    vec4 RatesRadianceBounds;
};
layout(binding=0) uniform highp sampler2D InputImage;
layout(binding=1) uniform highp sampler2D ExposureHistory;
layout(location=0) out vec4 OutputColor;
const uint Measure=0u,Reduce=1u,Adapt=2u,Apply=3u;
const int BlockExtent=2;
const vec3 Rec709Weights=vec3(0.2126,0.7152,0.0722);
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (OperationSizeHistory.x<=Reduce)
    {
        vec2 aggregate=vec2(0);
        for (int row=0;row<BlockExtent;++row) for (int column=0;column<BlockExtent;++column)
        {
            ivec2 location=pixel*BlockExtent+ivec2(column,row);
            if (any(greaterThanEqual(location,ivec2(OperationSizeHistory.yz)))) continue;
            vec4 data=texelFetch(InputImage,location,0);
            if (OperationSizeHistory.x==Reduce) aggregate+=data.xy;
            else
            {
                bool finiteColor=!any(isnan(data.rgb)) && !any(isinf(data.rgb));
                vec3 bounded=finiteColor?clamp(data.rgb,vec3(0),vec3(RatesRadianceBounds.w)):vec3(0);
                aggregate+=vec2(log2(max(dot(bounded,Rec709Weights),RatesRadianceBounds.z)),1);
            }
        }
        OutputColor=vec4(aggregate,0,1); return;
    }
    if (OperationSizeHistory.x==Adapt)
    {
        vec2 sumCount=texelFetch(InputImage,ivec2(0),0).xy;
        float average=sumCount.x/max(sumCount.y,1.0);
        float desired=clamp(log2(KeyRangeTime.x)-average,KeyRangeTime.y,KeyRangeTime.z);
        float exposure=desired;
        if (OperationSizeHistory.w!=0u)
        {
            float old=clamp(texelFetch(ExposureHistory,ivec2(0),0).x,KeyRangeTime.y,KeyRangeTime.z);
            float halfLife=desired>old?RatesRadianceBounds.x:RatesRadianceBounds.y;
            exposure=desired+(old-desired)*exp2(-KeyRangeTime.w/halfLife);
        }
        OutputColor=vec4(exposure,desired,average,1); return;
    }
    vec4 original=texelFetch(InputImage,pixel,0);
    float gain=exp2(texelFetch(ExposureHistory,ivec2(0),0).x);
    OutputColor=vec4(min(max(original.rgb,vec3(0))*gain,vec3(RatesRadianceBounds.w)),original.a);
}
#endif
