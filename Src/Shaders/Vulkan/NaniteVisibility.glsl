#version 450
// Independent Vulkan implementation of two-pass cluster/HZB visibility.
// Reference: A Deep Dive into Nanite Virtualized Geometry, SIGGRAPH 2021.
layout(local_size_x=64) in;
layout(std140,set=0,binding=0) uniform Parameters { uvec4 counts,layoutInfo,sourceMip,targetMip; };
#if EVANS_NANITE_VISIBILITY_PASS == 2
layout(set=0,binding=1) uniform sampler2D cameraDepth;
#else
layout(std430,set=0,binding=1) readonly buffer Inputs { uint inputWords[]; };
#endif
layout(std430,set=0,binding=2) buffer Outputs { uint outputWords[]; };
#if EVANS_NANITE_VISIBILITY_PASS == 3
layout(std430,set=0,binding=3) readonly buffer Hierarchy { uint hierarchyWords[]; };
layout(std430,set=0,binding=4) readonly buffer MainStates { uint mainStates[]; };
#endif
const uint ThreadCount=64u,QueryStride=44u,CurrentMatrix=8u,HistoryMatrix=24u,IdentityOffset=40u;
const uint FrustumRejected=1u,MainVisible=2u,Deferred=3u,PostVisible=4u,Occluded=5u,MaximumFootprint=4u;
const float ArithmeticError=32.0*1.1920928955078125e-7,UnderflowError=32.0*1.1754943508222875e-38;
#if EVANS_NANITE_VISIBILITY_PASS != 2
vec4 vectorAt(uint word)
{ return uintBitsToFloat(uvec4(inputWords[word],inputWords[word+1u],inputWords[word+2u],inputWords[word+3u])); }
struct Projection { vec4 rectangle; float nearest; bool outside; bool safe; };
Projection projectBox(uint query,uint matrixOffset)
{
    Projection p=Projection(vec4(1,1,0,0),1.0,false,true); uint commonOutside=63u; bool finite=true;
    vec3 minimum=vectorAt(query).xyz,maximum=vectorAt(query+4u).xyz;
    for (uint corner=0u;corner<8u;++corner)
    {
        vec4 position=vec4((corner&1u)!=0u?maximum.x:minimum.x,(corner&2u)!=0u?maximum.y:minimum.y,(corner&4u)!=0u?maximum.z:minimum.z,1);
        vec4 low,high;
        for (uint row=0u;row<4u;++row)
        {
            vec4 coefficients=vectorAt(query+matrixOffset+row*4u);
            float magnitude=dot(abs(coefficients),abs(position)),value=dot(coefficients,position),error=magnitude*ArithmeticError+UnderflowError;
            low[row]=value-error; high[row]=value+error;
            finite=finite && !isinf(magnitude) && !isnan(magnitude) && !isinf(low[row]) && !isnan(low[row]) && !isinf(high[row]) && !isnan(high[row]);
        }
        uint planes=0u;
        if (high.x+high.w<0.0) planes|=1u; if (high.w-low.x<0.0) planes|=2u;
        if (high.y+high.w<0.0) planes|=4u; if (high.w-low.y<0.0) planes|=8u;
        if (high.z<0.0) planes|=16u; if (high.w-low.z<0.0) planes|=32u; commonOutside&=planes;
        if (low.w<=0.0 || low.z<=0.0) { p.safe=false; continue; }
        vec3 lo=min(low.xyz/low.w,low.xyz/high.w),hi=max(high.xyz/low.w,high.xyz/high.w);
        vec3 error=(max(abs(lo),abs(hi))+1.0)*ArithmeticError; lo-=error; hi+=error;
        p.rectangle.xy=min(p.rectangle.xy,vec2(lo.x+1.0,1.0-hi.y)*0.5);
        p.rectangle.zw=max(p.rectangle.zw,vec2(hi.x+1.0,1.0-lo.y)*0.5); p.nearest=min(p.nearest,lo.z);
    }
    p.outside=finite && commonOutside!=0u;
    p.safe=p.safe && finite && all(greaterThanEqual(p.rectangle.xy,vec2(0))) && all(lessThanEqual(p.rectangle.zw,vec2(1))) && p.nearest>0.0;
    return p;
}
bool hidden(Projection p)
{
    if (!p.safe) return false;
    uvec2 extent=counts.yz,first=uvec2(floor(p.rectangle.xy*vec2(extent))),last=min(extent-1u,uvec2(floor(p.rectangle.zw*vec2(extent)))); uint offset=0u;
    while (any(greaterThan(last-first+1u,uvec2(MaximumFootprint))))
    { offset+=extent.x*extent.y; extent=(extent+1u)/2u; first/=2u; last/=2u; }
    for (uint y=first.y;y<=last.y;++y) for (uint x=first.x;x<=last.x;++x)
    {
#if EVANS_NANITE_VISIBILITY_PASS == 3
        float depth=uintBitsToFloat(hierarchyWords[offset+y*extent.x+x]);
#else
        float depth=uintBitsToFloat(outputWords[offset+y*extent.x+x]);
#endif
        if (!(depth<p.nearest)) return false;
    }
    return true;
}
#endif
void main()
{
    uint index=(gl_WorkGroupID.y*targetMip.w+gl_WorkGroupID.x)*ThreadCount+gl_LocalInvocationIndex;
#if EVANS_NANITE_VISIBILITY_PASS == 2
    if (index>=targetMip.y*targetMip.z) return;
    outputWords[index]=floatBitsToUint(texelFetch(cameraDepth,ivec2(index%targetMip.y,index/targetMip.y),0).r);
#elif EVANS_NANITE_VISIBILITY_PASS == 0
    if (index>=targetMip.y*targetMip.z) return;
    float depth=0.0;
    if (sourceMip.w==0u) depth=uintBitsToFloat(inputWords[index]);
    else
    {
        uvec2 origin=uvec2(index%targetMip.y,index/targetMip.y)*2u;
        for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
        { uvec2 child=origin+uvec2(x,y); if (child.x<sourceMip.y && child.y<sourceMip.z) depth=max(depth,uintBitsToFloat(outputWords[sourceMip.x+child.y*sourceMip.y+child.x])); }
    }
    outputWords[targetMip.x+index]=floatBitsToUint(depth);
#else
    if (index>=counts.x) return;
    uint query=layoutInfo.x+index*QueryStride,state=MainVisible;
    if (counts.w==0u)
    {
        if (projectBox(query,CurrentMatrix).outside) state=FrustumRejected;
        else
        {
            bool history=inputWords[query+IdentityOffset]!=0u;
#if EVANS_NANITE_VISIBILITY_PASS == 3
            history=history && layoutInfo.w!=0u;
#endif
            if (history && hidden(projectBox(query,HistoryMatrix))) state=Deferred;
        }
    }
    else
    {
#if EVANS_NANITE_VISIBILITY_PASS == 3
        state=mainStates[index];
#else
        state=inputWords[query+IdentityOffset+1u];
#endif
        if (state==Deferred) state=hidden(projectBox(query,CurrentMatrix))?Occluded:PostVisible;
    }
    outputWords[layoutInfo.y+index]=state;
#endif
}
