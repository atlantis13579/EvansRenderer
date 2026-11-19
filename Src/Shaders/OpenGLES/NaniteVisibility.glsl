#version 310 es
precision highp float;
precision highp int;
// ES keeps exact uint storage for depth bits and classification identities.
// Previous-frame occlusion defers; only current main-pass depth can reject.
layout(local_size_x=64) in;
layout(std140,binding=0) uniform VisibilityParameters { uvec4 counts,layoutInfo,sourceMip,targetMip; };
#if EVANS_NANITE_VISIBILITY_PASS == 2
layout(binding=0) uniform highp sampler2D cameraDepth;
#else
layout(std430,binding=0) readonly buffer InputData { uint sourceWords[]; };
#endif
layout(std430,binding=1) buffer OutputData { uint feedback[]; };
#if EVANS_NANITE_VISIBILITY_PASS == 3
layout(std430,binding=2) readonly buffer Pyramid { uint depthPyramid[]; };
layout(std430,binding=3) readonly buffer MainDecisions { uint mainDecisions[]; };
#endif
const uint GroupWidth=64u,QueryWords=44u,CurrentRows=8u,PreviousRows=24u,IdentityWords=40u;
const uint OutsideFrustum=1u,DrawMain=2u,Retest=3u,DrawPost=4u,RejectOccluded=5u,FootprintLimit=4u;
const float ErrorScale=32.0*1.1920928955078125e-7,ErrorMinimum=32.0*1.1754943508222875e-38;
#if EVANS_NANITE_VISIBILITY_PASS != 2
vec4 readVector(uint offset)
{ return uintBitsToFloat(uvec4(sourceWords[offset],sourceWords[offset+1u],sourceWords[offset+2u],sourceWords[offset+3u])); }
struct ProjectedBox { vec2 minimum; vec2 maximum; float nearest; bool outside; bool usable; };
ProjectedBox project(uint query,uint rows)
{
    ProjectedBox box=ProjectedBox(vec2(1),vec2(0),1.0,false,true); uint outsideMask=63u; bool finite=true;
    vec3 a=readVector(query).xyz,b=readVector(query+4u).xyz;
    for (uint vertex=0u;vertex<8u;++vertex)
    {
        vec4 position=vec4((vertex&1u)!=0u?b.x:a.x,(vertex&2u)!=0u?b.y:a.y,(vertex&4u)!=0u?b.z:a.z,1);
        vec4 lower,upper;
        for (uint row=0u;row<4u;++row)
        {
            vec4 coefficients=readVector(query+rows+row*4u);
            float sum=dot(coefficients,position),magnitude=dot(abs(coefficients),abs(position));
            float margin=magnitude*ErrorScale+ErrorMinimum; lower[row]=sum-margin; upper[row]=sum+margin;
            finite=finite && !isinf(magnitude) && !isnan(magnitude) && !isinf(lower[row]) && !isnan(lower[row]) && !isinf(upper[row]) && !isnan(upper[row]);
        }
        uint planes=0u;
        if (upper.x+upper.w<0.0) planes|=1u; if (upper.w-lower.x<0.0) planes|=2u;
        if (upper.y+upper.w<0.0) planes|=4u; if (upper.w-lower.y<0.0) planes|=8u;
        if (upper.z<0.0) planes|=16u; if (upper.w-lower.z<0.0) planes|=32u; outsideMask&=planes;
        if (lower.w<=0.0 || lower.z<=0.0) { box.usable=false; continue; }
        vec3 lo=min(lower.xyz/lower.w,lower.xyz/upper.w),hi=max(upper.xyz/lower.w,upper.xyz/upper.w);
        vec3 margin=(max(abs(lo),abs(hi))+1.0)*ErrorScale; lo-=margin; hi+=margin;
        box.minimum=min(box.minimum,vec2(lo.x+1.0,1.0-hi.y)*0.5);
        box.maximum=max(box.maximum,vec2(hi.x+1.0,1.0-lo.y)*0.5); box.nearest=min(box.nearest,lo.z);
    }
    box.outside=finite && outsideMask!=0u;
    box.usable=box.usable && finite && all(greaterThanEqual(box.minimum,vec2(0))) && all(lessThanEqual(box.maximum,vec2(1))) && box.nearest>0.0;
    return box;
}
bool occluded(ProjectedBox box)
{
    if (!box.usable) return false;
    uvec2 size=counts.yz,lo=uvec2(floor(box.minimum*vec2(size))),hi=min(size-1u,uvec2(floor(box.maximum*vec2(size)))); uint offset=0u;
    while (any(greaterThan(hi-lo+1u,uvec2(FootprintLimit)))) { offset+=size.x*size.y; size=(size+1u)/2u; lo/=2u; hi/=2u; }
    for (uint y=lo.y;y<=hi.y;++y) for (uint x=lo.x;x<=hi.x;++x)
    {
#if EVANS_NANITE_VISIBILITY_PASS == 3
        float depth=uintBitsToFloat(depthPyramid[offset+y*size.x+x]);
#else
        float depth=uintBitsToFloat(feedback[offset+y*size.x+x]);
#endif
        if (!(depth<box.nearest)) return false;
    }
    return true;
}
#endif
void main()
{
    uint invocation=(gl_WorkGroupID.y*targetMip.w+gl_WorkGroupID.x)*GroupWidth+gl_LocalInvocationIndex;
#if EVANS_NANITE_VISIBILITY_PASS == 2
    if (invocation>=targetMip.y*targetMip.z) return;
    // GL framebuffer rows are bottom-up; the shared pyramid is top-left.
    ivec2 pixel=ivec2(invocation%targetMip.y,targetMip.z-1u-invocation/targetMip.y);
    feedback[invocation]=floatBitsToUint(texelFetch(cameraDepth,pixel,0).r);
#elif EVANS_NANITE_VISIBILITY_PASS == 0
    if (invocation>=targetMip.y*targetMip.z) return;
    float farthest=0.0;
    if (sourceMip.w==0u) farthest=uintBitsToFloat(sourceWords[invocation]);
    else for (uint dy=0u;dy<2u;++dy) for (uint dx=0u;dx<2u;++dx)
    {
        uvec2 child=uvec2(invocation%targetMip.y,invocation/targetMip.y)*2u+uvec2(dx,dy);
        if (child.x<sourceMip.y && child.y<sourceMip.z) farthest=max(farthest,uintBitsToFloat(feedback[sourceMip.x+child.y*sourceMip.y+child.x]));
    }
    feedback[targetMip.x+invocation]=floatBitsToUint(farthest);
#else
    if (invocation>=counts.x) return;
    uint query=layoutInfo.x+invocation*QueryWords,state=DrawMain;
    if (counts.w==0u)
    {
        if (project(query,CurrentRows).outside) state=OutsideFrustum;
        else
        {
            bool history=sourceWords[query+IdentityWords]!=0u;
#if EVANS_NANITE_VISIBILITY_PASS == 3
            history=history && layoutInfo.w!=0u;
#endif
            if (history && occluded(project(query,PreviousRows))) state=Retest;
        }
    }
    else
    {
#if EVANS_NANITE_VISIBILITY_PASS == 3
        state=mainDecisions[invocation];
#else
        state=sourceWords[query+IdentityWords+1u];
#endif
        if (state==Retest) state=occluded(project(query,CurrentRows))?RejectOccluded:DrawPost;
    }
    feedback[layoutInfo.y+invocation]=state;
#endif
}
