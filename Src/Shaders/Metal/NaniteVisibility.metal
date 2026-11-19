#include <metal_stdlib>
using namespace metal;
// Metal's encoder boundaries order MAX-depth pyramid levels. Compile without
// fast math: uncertain projections must stay visible, including overflow.
struct Parameters { uint4 counts,layoutInfo,sourceMip,targetMip; };
constant uint ThreadCount=64,QueryStride=44,CurrentRows=8,PreviousRows=24,IdentityOffset=40;
constant uint FrustumRejected=1,MainVisible=2,Deferred=3,PostVisible=4,Occluded=5,FootprintLimit=4;
constant float ErrorFactor=32.0f*1.1920928955078125e-7f,UnderflowError=32.0f*1.1754943508222875e-38f;
float4 loadVector(device const uint* input,uint word)
{ return as_type<float4>(uint4(input[word],input[word+1],input[word+2],input[word+3])); }
struct Projection { float4 rectangle; float nearest; bool outside,safe; };
Projection projectBounds(device const uint* input,uint base,uint rows)
{
    Projection p{float4(1,1,0,0),1,false,true}; uint excluded=63; bool finite=true;
    float3 minimum=loadVector(input,base).xyz,maximum=loadVector(input,base+4).xyz;
    for (uint corner=0;corner<8;++corner)
    {
        float4 position((corner&1)?maximum.x:minimum.x,(corner&2)?maximum.y:minimum.y,(corner&4)?maximum.z:minimum.z,1);
        float4 low,high;
        for (uint row=0;row<4;++row)
        {
            float4 coefficients=loadVector(input,base+rows+row*4);
            float center=dot(coefficients,position),magnitude=dot(abs(coefficients),abs(position));
            float error=magnitude*ErrorFactor+UnderflowError; low[row]=center-error; high[row]=center+error;
            finite=finite && isfinite(magnitude) && isfinite(low[row]) && isfinite(high[row]);
        }
        uint planes=0;
        if (high.x+high.w<0) planes|=1; if (high.w-low.x<0) planes|=2;
        if (high.y+high.w<0) planes|=4; if (high.w-low.y<0) planes|=8;
        if (high.z<0) planes|=16; if (high.w-low.z<0) planes|=32; excluded&=planes;
        if (low.w<=0 || low.z<=0) { p.safe=false; continue; }
        float3 lo=min(low.xyz/low.w,low.xyz/high.w),hi=max(high.xyz/low.w,high.xyz/high.w);
        float3 error=(max(abs(lo),abs(hi))+1)*ErrorFactor; lo-=error; hi+=error;
        p.rectangle.xy=min(p.rectangle.xy,float2(lo.x+1,1-hi.y)*0.5f);
        p.rectangle.zw=max(p.rectangle.zw,float2(hi.x+1,1-lo.y)*0.5f); p.nearest=min(p.nearest,lo.z);
    }
    p.outside=finite && excluded!=0;
    p.safe=p.safe && finite && all(p.rectangle.xy>=0) && all(p.rectangle.zw<=1) && p.nearest>0;
    return p;
}
bool hidden(Projection p,constant Parameters& frame,device const uint* pyramid)
{
    if (!p.safe) return false;
    uint2 extent=frame.counts.yz,first=uint2(floor(p.rectangle.xy*float2(extent))),last=min(extent-1,uint2(floor(p.rectangle.zw*float2(extent)))); uint offset=0;
    while (any(last-first+1>FootprintLimit)) { offset+=extent.x*extent.y; extent=(extent+1)/2; first/=2; last/=2; }
    for (uint y=first.y;y<=last.y;++y) for (uint x=first.x;x<=last.x;++x)
        if (!(as_type<float>(pyramid[offset+y*extent.x+x])<p.nearest)) return false;
    return true;
}
kernel void EvansNaniteReduce(constant Parameters& frame [[buffer(0)]],device const uint* input [[buffer(1)]],
    device uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint index=(group.y*frame.targetMip.w+group.x)*ThreadCount+lane; if (index>=frame.targetMip.y*frame.targetMip.z) return;
    float depth=0;
    if (frame.sourceMip.w==0) depth=as_type<float>(input[index]);
    else for (uint y=0;y<2;++y) for (uint x=0;x<2;++x)
    {
        uint2 child=uint2(index%frame.targetMip.y,index/frame.targetMip.y)*2+uint2(x,y);
        if (child.x<frame.sourceMip.y && child.y<frame.sourceMip.z) depth=max(depth,as_type<float>(output[frame.sourceMip.x+child.y*frame.sourceMip.y+child.x]));
    }
    output[frame.targetMip.x+index]=as_type<uint>(depth);
}
kernel void EvansNaniteCull(constant Parameters& frame [[buffer(0)]],device const uint* input [[buffer(1)]],
    device uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint index=(group.y*frame.targetMip.w+group.x)*ThreadCount+lane; if (index>=frame.counts.x) return;
    uint query=frame.layoutInfo.x+index*QueryStride,state=MainVisible;
    if (frame.counts.w==0)
    { if (projectBounds(input,query,CurrentRows).outside) state=FrustumRejected; else if (input[query+IdentityOffset]!=0 && hidden(projectBounds(input,query,PreviousRows),frame,output)) state=Deferred; }
    else { state=input[query+IdentityOffset+1]; if (state==Deferred) state=hidden(projectBounds(input,query,CurrentRows),frame,output)?Occluded:PostVisible; }
    output[frame.layoutInfo.y+index]=state;
}
kernel void EvansNaniteCapture(constant Parameters& frame [[buffer(0)]],device uint* pyramid [[buffer(2)]],
    depth2d<float,access::read> cameraDepth [[texture(0)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint index=(group.y*frame.targetMip.w+group.x)*ThreadCount+lane; if (index>=frame.targetMip.y*frame.targetMip.z) return;
    pyramid[index]=as_type<uint>(cameraDepth.read(uint2(index%frame.targetMip.y,index/frame.targetMip.y)));
}
kernel void EvansNaniteFrameCull(constant Parameters& frame [[buffer(0)]],device const uint* queries [[buffer(1)]],
    device uint* decisions [[buffer(2)]],device const uint* hierarchy [[buffer(3)]],device const uint* mainDecisions [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint index=(group.y*frame.targetMip.w+group.x)*ThreadCount+lane; if (index>=frame.counts.x) return;
    uint query=index*QueryStride,state=MainVisible;
    if (frame.counts.w==0)
    {
        if (projectBounds(queries,query,CurrentRows).outside) state=FrustumRejected;
        else if (frame.layoutInfo.w!=0 && queries[query+IdentityOffset]!=0 && hidden(projectBounds(queries,query,PreviousRows),frame,hierarchy)) state=Deferred;
    }
    else { state=mainDecisions[index]; if (state==Deferred) state=hidden(projectBounds(queries,query,CurrentRows),frame,hierarchy)?Occluded:PostVisible; }
    decisions[index]=state;
}
