#version 450
#if defined(EVANS_TSR_PACK_STAGE)
// Resolved input images use canonical top-left texel coordinates in Vulkan.
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(std140,set=0,binding=0) uniform TextureFrame { uvec4 ExtentFlags; };
layout(set=0,binding=1) uniform sampler2D FinalHDR;
layout(set=0,binding=2) uniform sampler2D OpaqueHDR;
layout(set=0,binding=3) uniform sampler2D PhysicalMotion;
layout(set=0,binding=4) uniform sampler2D HardwareDepth;
layout(set=0,binding=5) uniform sampler2D OutlineMask;
layout(set=0,binding=6) uniform sampler2D AdditionalReactive;
layout(std430,set=0,binding=7) writeonly buffer RenderPacket { vec4 Values[]; } Packed;
const uint PacketStride=5u,FinalField=0u,OpaqueField=1u,MotionField=2u,DepthField=3u,OutlineField=4u;
const uint HasMotion=1u,HasOutline=2u,HasReactive=4u;
void main()
{
    uvec2 pixel=gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel,ExtentFlags.xy))) return;
    ivec2 at=ivec2(pixel); uint first=(pixel.y*ExtentFlags.x+pixel.x)*PacketStride;
    vec4 motion=vec4(0),outline=vec4(0); float reactive=0;
    if ((ExtentFlags.z&HasMotion)!=0u) motion=texelFetch(PhysicalMotion,at,0);
    if ((ExtentFlags.z&HasOutline)!=0u) outline=texelFetch(OutlineMask,at,0);
    if ((ExtentFlags.z&HasReactive)!=0u) reactive=texelFetch(AdditionalReactive,at,0).r;
    Packed.Values[first+FinalField]=texelFetch(FinalHDR,at,0);
    Packed.Values[first+OpaqueField]=texelFetch(OpaqueHDR,at,0);
    Packed.Values[first+MotionField]=motion;
    Packed.Values[first+DepthField]=vec4(texelFetch(HardwareDepth,at,0).r,reactive,0,0);
    Packed.Values[first+OutlineField]=outline;
}
#elif defined(EVANS_TSR_EXPORT_VERTEX_STAGE)
void main()
{
    const float FullscreenReach=3.0;
    vec2 corner=gl_VertexIndex==0?vec2(-1):gl_VertexIndex==1?vec2(-1,FullscreenReach):vec2(FullscreenReach,-1);
    gl_Position=vec4(corner,0,1);
}
#elif defined(EVANS_TSR_EXPORT_FRAGMENT_STAGE)
layout(std140,set=0,binding=0) uniform ExportFrame { uvec4 ExtentFlags; };
layout(std430,set=0,binding=1) readonly buffer PresentationPacket { vec4 Values[]; } History;
layout(location=0) out vec4 ExportedHDR;
const uint HistoryStride=2u;
void main()
{
    uvec2 at=uvec2(gl_FragCoord.xy);
    ExportedHDR=History.Values[(at.y*ExtentFlags.x+at.x)*HistoryStride];
}
#else
// Vulkan-native HDR temporal reconstruction. No other technique source is
// included or translated. Row arrays match the C++ row-major camera ABI.
// References: Salvi, GDC 2016, "An Excursion in Temporal Supersampling";
// AMD, GDC 2023, "Temporal Upscaling: Past, Present, and Future".
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(std140,set=0,binding=0) uniform ReconstructionFrame
{
    vec4 CurrentInverseRows[4]; vec4 PreviousInverseRows[4];
    vec4 CurrentDepthPlane,PreviousDepthPlane;
    uvec4 Extent;
    vec4 JitterQuality,Controls;
    uvec4 Options;
};
layout(std430,set=0,binding=1) readonly buffer SourceBuffer { vec4 Values[]; } SourcePixels;
layout(std430,set=0,binding=2) readonly buffer HistoryBuffer { vec4 Values[]; } OldPixels;
layout(std430,set=0,binding=3) writeonly buffer CandidateBuffer { vec4 Values[]; } NewPixels;
layout(std430,set=0,binding=4) writeonly buffer AuxiliaryBuffer { vec4 Values[]; } Auxiliary;
const uint SourceStride=5u,HistoryStride=2u;
const int FilterRadius=2,DilationRadius=1;
const float PixelCenter=0.5,ClearDepth=1.0,Pi=3.14159265358979323846;
const float MinimumW=1e-7,DepthFloor=1e-4,ColorFloor=1e-4,WeightFloor=1e-6,MaximumRadiance=65504.0;
const float UnsupportedObservation=0.001;
const float FilterSamples=float((FilterRadius*2)*(FilterRadius*2));
vec4 sourceAt(ivec2 p,uint member)
{
    p=clamp(p,ivec2(0),ivec2(Extent.xy)-1);
    return SourcePixels.Values[(uint(p.y)*Extent.x+uint(p.x))*SourceStride+member];
}
vec4 historyAt(ivec2 p,uint member)
{
    p=clamp(p,ivec2(0),ivec2(Extent.zw)-1);
    return OldPixels.Values[(uint(p.y)*Extent.z+uint(p.x))*HistoryStride+member];
}
float largest(vec3 c) { return max(c.x,max(c.y,c.z)); }
vec3 toYCoCg(vec3 c) { return vec3((c.r+2.0*c.g+c.b)/4.0,(c.r-c.b)/2.0,(-c.r+2.0*c.g-c.b)/4.0); }
vec3 toRGB(vec3 c) { return vec3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z); }
float lanczos(float distance)
{
    float x=abs(distance); if (x>=float(FilterRadius)) return 0.0; if (x<WeightFloor) return 1.0;
    return sin(Pi*x)*sin(Pi*x/float(FilterRadius))/(Pi*Pi*x*x/float(FilterRadius));
}
float opticalDistance(vec2 uv,float depth,bool previous)
{
    vec4 clip=vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1),world;
    if (previous) world=vec4(dot(PreviousInverseRows[0],clip),dot(PreviousInverseRows[1],clip),dot(PreviousInverseRows[2],clip),dot(PreviousInverseRows[3],clip));
    else world=vec4(dot(CurrentInverseRows[0],clip),dot(CurrentInverseRows[1],clip),dot(CurrentInverseRows[2],clip),dot(CurrentInverseRows[3],clip));
    if (abs(world.w)<=MinimumW) return -1.0;
    return dot(previous?PreviousDepthPlane:CurrentDepthPlane,vec4(world.xyz/world.w,1));
}
ivec2 localGridFloor(vec2 value)
{
    const float GridRoundingTolerance=1e-6;
    vec2 closest=round(value);
    vec2 snapped=mix(value,closest,lessThanEqual(abs(value-closest),vec2(GridRoundingTolerance)));
    return ivec2(floor(snapped));
}
void main()
{
    uvec2 id=gl_GlobalInvocationID.xy; if (any(greaterThanEqual(id,Extent.zw))) return;
    // Split the rational pixel center before jitter, keeping floor decisions
    // independent of full-coordinate floating-point cancellation.
    uvec2 doubledExtent=Extent.zw*2u,centerNumerator=(id*2u+1u)*Extent.xy;
    uvec2 integerPart=centerNumerator/doubledExtent;
    vec2 fractionalPart=vec2(centerNumerator%doubledExtent)/vec2(doubledExtent)+JitterQuality.xy;
    vec2 location=vec2(integerPart)+fractionalPart;
    ivec2 base=ivec2(integerPart)+localGridFloor(fractionalPart-PixelCenter);
    vec2 footprint=vec2(Extent.xy)/vec2(Extent.zw);
    vec3 sum=vec3(0),wideSum=vec3(0),mean=vec3(0),moment=vec3(0),lo=vec3(MaximumRadiance),hi=vec3(0);
    float alpha=0.0,weight=0.0,wideWeight=0.0,reactivity=0.0;
    for (int y=1-FilterRadius;y<=FilterRadius;++y) for (int x=1-FilterRadius;x<=FilterRadius;++x)
    {
        ivec2 p=base+ivec2(x,y); vec4 current=sourceAt(p,0u),opaque=sourceAt(p,1u);
        vec3 rgb=clamp(current.rgb,vec3(0),vec3(MaximumRadiance)),yc=toYCoCg(rgb);
        vec2 offset=vec2(p)+PixelCenter-location;
        float w=lanczos(offset.x/footprint.x)*lanczos(offset.y/footprint.y),wide=lanczos(offset.x)*lanczos(offset.y);
        sum+=rgb*w; wideSum+=rgb*wide; alpha+=current.a*wide; weight+=w; wideWeight+=wide; mean+=yc; moment+=yc*yc; lo=min(lo,rgb); hi=max(hi,rgb);
        float scale=max(max(largest(rgb),largest(opaque.rgb)),ColorFloor);
        reactivity=max(reactivity,max(sourceAt(p,3u).y,largest(abs(current.rgb-opaque.rgb))/scale*Controls.y));
    }
    bool supported=weight>WeightFloor;
    vec3 fresh;
    if (supported) fresh=sum/weight; else fresh=wideSum/max(wideWeight,WeightFloor);
    fresh=clamp(fresh,lo,hi); vec3 color=fresh;
    alpha=clamp(alpha/max(wideWeight,WeightFloor),0.0,1.0); reactivity=clamp(reactivity,0.0,1.0);
    ivec2 nearest=ivec2(integerPart)+localGridFloor(fractionalPart),selected=nearest; float depth=sourceAt(selected,3u).x;
    for (int y=-DilationRadius;y<=DilationRadius;++y) for (int x=-DilationRadius;x<=DilationRadius;++x)
    {
        ivec2 p=nearest+ivec2(x,y); float candidate=sourceAt(p,3u).x;
        if (candidate<depth) { depth=candidate; selected=p; }
    }
    selected=clamp(selected,ivec2(0),ivec2(Extent.xy)-1);
    vec4 velocity=sourceAt(selected,2u); bool background=depth>=ClearDepth;
    vec2 motion=background?vec2(0):velocity.xy;
    vec2 oldLocation=vec2(id)+PixelCenter-motion*vec2(Extent.zw)/vec2(Extent.xy);
    vec2 residual=(vec2(nearest)+PixelCenter-location)/footprint; float observation=supported?exp2(-dot(residual,residual)):UnsupportedObservation,count=observation;
    bool valid=Options.x!=0u && reactivity<1.0 && (background?Options.y!=0u:velocity.w>0.0) &&
        all(greaterThanEqual(oldLocation,vec2(PixelCenter))) && all(lessThanEqual(oldLocation,vec2(Extent.zw)-PixelCenter));
    if (valid)
    {
        float expected=opticalDistance((vec2(selected)+PixelCenter-JitterQuality.xy-motion)/vec2(Extent.xy),background?ClearDepth:1.0-velocity.z,true);
        ivec2 first=ivec2(floor(oldLocation-PixelCenter)); bool agrees=false;
        for (int y=0;y<2;++y) for (int x=0;x<2;++x)
        {
            vec4 meta=historyAt(first+ivec2(x,y),1u);
            agrees=agrees || (background?meta.x>=ClearDepth:meta.x<ClearDepth && expected>0.0 && meta.z>0.0 && abs(meta.z-expected)<=max(DepthFloor,abs(expected)*JitterQuality.w));
        }
        if (agrees)
        {
            vec2 f=oldLocation-PixelCenter-vec2(first); vec3 oldColor=vec3(0); float oldCount=0.0;
            for (int y=0;y<2;++y) for (int x=0;x<2;++x)
            {
                float w=(x!=0?f.x:1.0-f.x)*(y!=0?f.y:1.0-f.y);
                oldColor+=historyAt(first+ivec2(x,y),0u).rgb*w; oldCount+=historyAt(first+ivec2(x,y),1u).y*w;
            }
            mean/=FilterSamples; moment/=FilterSamples;
            vec3 extent=max(sqrt(max(moment-mean*mean,vec3(0)))*Controls.x,vec3(ColorFloor));
            vec3 delta=toYCoCg(oldColor)-mean; oldColor=toRGB(mean+delta/max(1.0,largest(abs(delta/extent)))); oldCount=min(oldCount,Controls.z);
            float feedback=min(JitterQuality.z,oldCount/(oldCount+observation))*(1.0-reactivity);
            color=mix(fresh,oldColor,feedback); count=min(Controls.z,oldCount*(1.0-reactivity)+observation);
        }
    }
    uint address=(id.y*Extent.z+id.x)*HistoryStride;
    float optical=background?-1.0:opticalDistance((vec2(selected)+PixelCenter-JitterQuality.xy)/vec2(Extent.xy),depth,false);
    NewPixels.Values[address]=vec4(clamp(color,vec3(0),vec3(MaximumRadiance)),alpha);
    NewPixels.Values[address+1u]=vec4(depth,count,optical,reactivity);
    Auxiliary.Values[address]=vec4(velocity.xy*vec2(Extent.zw)/vec2(Extent.xy),velocity.w>0.0?1.0-velocity.z:0.0,velocity.w);
    Auxiliary.Values[address+1u]=sourceAt(nearest,4u);
}
#endif
