#version 310 es
precision highp float;
precision highp int;
#if defined(EVANS_TSR_PACK_STAGE)
// Image access flips Y once. Packet addresses, velocities and reconstruction
// remain in the canonical top-left convention used by every native API.
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(std140,binding=0) uniform BridgeFrame { uvec4 ImageExtentFlags; };
uniform highp sampler2D FinalRadiance,OpaqueRadiance,RenderMotion,RenderDepth,RenderOutline;
uniform highp sampler2D AdditionalReactive;
layout(std430,binding=0) writeonly buffer PackedRecords { vec4 Values[]; } Packed;
const uint RecordStride=5u,FinalMember=0u,OpaqueMember=1u,MotionMember=2u,DepthMember=3u,OutlineMember=4u;
const uint MotionPresent=1u,OutlinePresent=2u,ReactivePresent=4u;
void main()
{
    uvec2 pixel=gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel,ImageExtentFlags.xy))) return;
    ivec2 texel=ivec2(pixel.x,ImageExtentFlags.y-1u-pixel.y);
    uint record=(pixel.y*ImageExtentFlags.x+pixel.x)*RecordStride;
    vec4 velocity=vec4(0),overlay=vec4(0); float reactive=0.0;
    if ((ImageExtentFlags.z&MotionPresent)!=0u) velocity=texelFetch(RenderMotion,texel,0);
    if ((ImageExtentFlags.z&OutlinePresent)!=0u) overlay=texelFetch(RenderOutline,texel,0);
    if ((ImageExtentFlags.z&ReactivePresent)!=0u) reactive=texelFetch(AdditionalReactive,texel,0).r;
    Packed.Values[record+FinalMember]=texelFetch(FinalRadiance,texel,0);
    Packed.Values[record+OpaqueMember]=texelFetch(OpaqueRadiance,texel,0);
    Packed.Values[record+MotionMember]=velocity;
    Packed.Values[record+DepthMember]=vec4(texelFetch(RenderDepth,texel,0).r,reactive,0,0);
    Packed.Values[record+OutlineMember]=overlay;
}
#elif defined(EVANS_TSR_EXPORT_VERTEX_STAGE)
void main()
{
    const float TriangleLimit=3.0;
    vec2 position=gl_VertexID==0?vec2(-1):gl_VertexID==1?vec2(-1,TriangleLimit):vec2(TriangleLimit,-1);
    gl_Position=vec4(position,0,1);
}
#elif defined(EVANS_TSR_EXPORT_FRAGMENT_STAGE)
layout(std140,binding=0) uniform PresentationFrame { uvec4 ImageExtentFlags; };
layout(std430,binding=0) readonly buffer PresentationRecords { vec4 Values[]; } History;
layout(location=0) out vec4 UpscaledHDR;
const uint HistoryRecordStride=2u;
void main()
{
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    pixel.y=ImageExtentFlags.y-1u-pixel.y;
    UpscaledHDR=History.Values[(pixel.y*ImageExtentFlags.x+pixel.x)*HistoryRecordStride];
}
#else
// ES compute implementation of independent HDR temporal upscaling. Four storage
// bindings fit the ES 3.1 minimum; render and presentation grids are explicit.
// Salvi, "An Excursion in Temporal Supersampling", GDC 2016; AMD, "Temporal
// Upscaling: Past, Present, and Future", GDC 2023. No vendor algorithm is claimed.
precision highp float;
precision highp int;
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(std140,binding=0) uniform UpscaleFrame
{
    vec4 CurrentInverseRows[4]; vec4 PreviousInverseRows[4];
    vec4 CurrentDepthPlane,PreviousDepthPlane;
    uvec4 Extent;
    vec4 JitterQuality,Controls;
    uvec4 Options;
};
layout(std430,binding=0) readonly buffer RenderRecords { vec4 Values[]; } Render;
layout(std430,binding=1) readonly buffer PreviousRecords { vec4 Values[]; } Previous;
layout(std430,binding=2) writeonly buffer ReconstructedRecords { vec4 Values[]; } Reconstructed;
layout(std430,binding=3) writeonly buffer CurrentRecords { vec4 Values[]; } Current;
const uint RenderStride=5u,HistoryStride=2u;
const int KernelRadius=2,VelocityRadius=1;
const float HalfPixel=0.5,FarDepth=1.0,Pi=3.14159265358979323846;
const float WFloor=1e-7,DepthFloor=1e-4,ColorFloor=1e-4,KernelFloor=1e-6,RadianceLimit=65504.0;
const float UnsupportedObservation=0.001;
const float KernelSampleCount=float((KernelRadius*2)*(KernelRadius*2));
vec4 renderAt(ivec2 p,uint field)
{
    p=clamp(p,ivec2(0),ivec2(Extent.xy)-1);
    return Render.Values[(uint(p.y)*Extent.x+uint(p.x))*RenderStride+field];
}
vec4 oldAt(ivec2 p,uint field)
{
    p=clamp(p,ivec2(0),ivec2(Extent.zw)-1);
    return Previous.Values[(uint(p.y)*Extent.z+uint(p.x))*HistoryStride+field];
}
float maxChannel(vec3 c) { return max(c.x,max(c.y,c.z)); }
vec3 encodeColor(vec3 c) { return vec3((c.r+2.0*c.g+c.b)/4.0,(c.r-c.b)/2.0,(-c.r+2.0*c.g-c.b)/4.0); }
vec3 decodeColor(vec3 c) { return vec3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z); }
float reconstructionWeight(float distance)
{
    float x=abs(distance); if (x>=float(KernelRadius)) return 0.0; if (x<KernelFloor) return 1.0;
    return sin(Pi*x)*sin(Pi*x/float(KernelRadius))/(Pi*Pi*x*x/float(KernelRadius));
}
float eyeDepth(vec2 uv,float depth,bool old)
{
    vec4 clip=vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1),world;
    if (old) world=vec4(dot(PreviousInverseRows[0],clip),dot(PreviousInverseRows[1],clip),dot(PreviousInverseRows[2],clip),dot(PreviousInverseRows[3],clip));
    else world=vec4(dot(CurrentInverseRows[0],clip),dot(CurrentInverseRows[1],clip),dot(CurrentInverseRows[2],clip),dot(CurrentInverseRows[3],clip));
    if (abs(world.w)<=WFloor) return -1.0;
    return dot(old?PreviousDepthPlane:CurrentDepthPlane,vec4(world.xyz/world.w,1));
}
ivec2 inputCellFloor(vec2 fraction)
{
    const float BoundaryTolerance=1e-6;
    vec2 integral=round(fraction);
    return ivec2(floor(mix(fraction,integral,lessThanEqual(abs(fraction-integral),vec2(BoundaryTolerance)))));
}
void main()
{
    uvec2 pixel=gl_GlobalInvocationID.xy; if (any(greaterThanEqual(pixel,Extent.zw))) return;
    // Integer quotient/remainder keeps exact boundaries stable when a GLSL
    // compiler replaces floating division with multiplication by a reciprocal.
    uvec2 denominator=Extent.zw*2u,numerator=(pixel*2u+1u)*Extent.xy;
    uvec2 cell=numerator/denominator;
    vec2 fraction=vec2(numerator%denominator)/vec2(denominator)+JitterQuality.xy;
    vec2 samplePoint=vec2(cell)+fraction;
    ivec2 base=ivec2(cell)+inputCellFloor(fraction-HalfPixel);
    vec2 footprint=vec2(Extent.xy)/vec2(Extent.zw);
    vec3 sum=vec3(0),wideSum=vec3(0),mean=vec3(0),squared=vec3(0),minimum=vec3(RadianceLimit),maximum=vec3(0);
    float alpha=0.0,weights=0.0,wideWeights=0.0,reactive=0.0;
    for (int y=1-KernelRadius;y<=KernelRadius;++y) for (int x=1-KernelRadius;x<=KernelRadius;++x)
    {
        ivec2 at=base+ivec2(x,y); vec4 finalColor=renderAt(at,0u),opaque=renderAt(at,1u);
        vec3 rgb=clamp(finalColor.rgb,vec3(0),vec3(RadianceLimit)),color=encodeColor(rgb);
        vec2 offset=vec2(at)+HalfPixel-samplePoint;
        float w=reconstructionWeight(offset.x/footprint.x)*reconstructionWeight(offset.y/footprint.y),wide=reconstructionWeight(offset.x)*reconstructionWeight(offset.y);
        sum+=rgb*w; wideSum+=rgb*wide; alpha+=finalColor.a*wide; weights+=w; wideWeights+=wide; mean+=color; squared+=color*color; minimum=min(minimum,rgb); maximum=max(maximum,rgb);
        float normalization=max(max(maxChannel(rgb),maxChannel(opaque.rgb)),ColorFloor);
        reactive=max(reactive,max(renderAt(at,3u).y,maxChannel(abs(finalColor.rgb-opaque.rgb))/normalization*Controls.y));
    }
    bool supported=weights>KernelFloor;
    vec3 spatial;
    if (supported) spatial=sum/weights; else spatial=wideSum/max(wideWeights,KernelFloor);
    spatial=clamp(spatial,minimum,maximum); vec3 resolved=spatial;
    alpha=clamp(alpha/max(wideWeights,KernelFloor),0.0,1.0); reactive=clamp(reactive,0.0,1.0);
    ivec2 center=ivec2(cell)+inputCellFloor(fraction),selected=center; float depth=renderAt(selected,3u).x;
    for (int y=-VelocityRadius;y<=VelocityRadius;++y) for (int x=-VelocityRadius;x<=VelocityRadius;++x)
    {
        ivec2 at=center+ivec2(x,y); float candidate=renderAt(at,3u).x;
        if (candidate<depth) { depth=candidate; selected=at; }
    }
    selected=clamp(selected,ivec2(0),ivec2(Extent.xy)-1);
    vec4 velocity=renderAt(selected,2u); bool sky=depth>=FarDepth;
    vec2 displacement=sky?vec2(0):velocity.xy;
    vec2 reprojected=vec2(pixel)+HalfPixel-displacement*vec2(Extent.zw)/vec2(Extent.xy);
    vec2 residual=(vec2(center)+HalfPixel-samplePoint)/footprint; float currentWeight=supported?exp2(-dot(residual,residual)):UnsupportedObservation,count=currentWeight;
    bool valid=Options.x!=0u && reactive<1.0 && (sky?Options.y!=0u:velocity.w>0.0) &&
        all(greaterThanEqual(reprojected,vec2(HalfPixel))) && all(lessThanEqual(reprojected,vec2(Extent.zw)-HalfPixel));
    if (valid)
    {
        float expected=eyeDepth((vec2(selected)+HalfPixel-JitterQuality.xy-displacement)/vec2(Extent.xy),sky?FarDepth:1.0-velocity.z,true);
        ivec2 lower=ivec2(floor(reprojected-HalfPixel)); bool agrees=false;
        for (int y=0;y<2;++y) for (int x=0;x<2;++x)
        {
            vec4 meta=oldAt(lower+ivec2(x,y),1u);
            agrees=agrees || (sky?meta.x>=FarDepth:meta.x<FarDepth && expected>0.0 && meta.z>0.0 && abs(meta.z-expected)<=max(DepthFloor,abs(expected)*JitterQuality.w));
        }
        if (agrees)
        {
            vec2 f=reprojected-HalfPixel-vec2(lower); vec3 history=vec3(0); float previousWeight=0.0;
            for (int y=0;y<2;++y) for (int x=0;x<2;++x)
            {
                float w=(x!=0?f.x:1.0-f.x)*(y!=0?f.y:1.0-f.y);
                history+=oldAt(lower+ivec2(x,y),0u).rgb*w; previousWeight+=oldAt(lower+ivec2(x,y),1u).y*w;
            }
            mean/=KernelSampleCount; squared/=KernelSampleCount;
            vec3 radius=max(sqrt(max(squared-mean*mean,vec3(0)))*Controls.x,vec3(ColorFloor));
            vec3 delta=encodeColor(history)-mean; history=decodeColor(mean+delta/max(1.0,maxChannel(abs(delta/radius)))); previousWeight=min(previousWeight,Controls.z);
            float feedback=min(JitterQuality.z,previousWeight/(previousWeight+currentWeight))*(1.0-reactive);
            resolved=mix(spatial,history,feedback); count=min(Controls.z,previousWeight*(1.0-reactive)+currentWeight);
        }
    }
    uint address=(pixel.y*Extent.z+pixel.x)*HistoryStride;
    float distance=sky?-1.0:eyeDepth((vec2(selected)+HalfPixel-JitterQuality.xy)/vec2(Extent.xy),depth,false);
    Reconstructed.Values[address]=vec4(clamp(resolved,vec3(0),vec3(RadianceLimit)),alpha);
    Reconstructed.Values[address+1u]=vec4(depth,count,distance,reactive);
    Current.Values[address]=vec4(velocity.xy*vec2(Extent.zw)/vec2(Extent.xy),velocity.w>0.0?1.0-velocity.z:0.0,velocity.w);
    Current.Values[address+1u]=renderAt(center,4u);
}
#endif
