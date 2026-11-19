// Karis et al., SIGGRAPH 2021: history only defers the current cluster cut.
// Standard depth uses MAX reduction; a clear pixel must prevent occlusion.
cbuffer VisibilityFrame : register(b0) { uint4 Counts,Layout,Source,Target; };
RWByteAddressBuffer Feedback : register(u0);
static const uint Threads=64,QueryWords=44,CurrentRows=8,PreviousRows=24,IdentityWords=40;
static const uint FrustumRejected=1,MainVisible=2,Deferred=3,PostVisible=4,Occluded=5,MaximumFootprint=4;
static const float Roundoff=32.0*1.1920928955078125e-7,Underflow=32.0*1.1754943508222875e-38;
#if defined(EVANS_NANITE_CAPTURE_DEPTH)
Texture2D<float> CameraDepth : register(t0);
[numthreads(Threads,1,1)]
void CSCapture(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint index=(group.y*Target.w+group.x)*Threads+lane; if (index>=Target.y*Target.z) return;
    Feedback.Store(index*4,asuint(CameraDepth.Load(int3(index%Target.y,index/Target.y,0))));
}
#else
ByteAddressBuffer Input : register(t0);
#if defined(EVANS_NANITE_FRAME_CULL)
ByteAddressBuffer DepthHierarchy : register(t1);
ByteAddressBuffer MainDecisions : register(t2);
#endif
uint Read(uint word) { return Input.Load(word*4); }
float4 Vector(uint word) { return asfloat(Input.Load4(word*4)); }
struct Projection { float4 Rectangle; float Near; bool Outside,Safe; };
Projection Project(uint query,uint rows)
{
    Projection p; p.Rectangle=float4(1,1,0,0); p.Near=1; p.Outside=false; p.Safe=true;
    uint outside=63; bool finite=true;
    [unroll] for (uint corner=0;corner<8;++corner)
    {
        float3 minimum=Vector(query).xyz,maximum=Vector(query+4).xyz;
        float4 position=float4((corner&1)?maximum.x:minimum.x,(corner&2)?maximum.y:minimum.y,(corner&4)?maximum.z:minimum.z,1);
        float4 low=0,high=0;
        [unroll] for (uint row=0;row<4;++row)
        {
            float4 coefficients=Vector(query+rows+row*4);
            float magnitude=dot(abs(coefficients),abs(position)),center=dot(coefficients,position);
            float uncertainty=magnitude*Roundoff+Underflow;
            low[row]=center-uncertainty; high[row]=center+uncertainty;
            finite=finite && isfinite(magnitude) && isfinite(low[row]) && isfinite(high[row]);
        }
        uint mask=0;
        if (high.x+high.w<0) mask|=1; if (high.w-low.x<0) mask|=2;
        if (high.y+high.w<0) mask|=4; if (high.w-low.y<0) mask|=8;
        if (high.z<0) mask|=16; if (high.w-low.z<0) mask|=32; outside&=mask;
        if (low.w<=0 || low.z<=0) { p.Safe=false; continue; }
        float3 lower=min(low.xyz/low.w,low.xyz/high.w),upper=max(high.xyz/low.w,high.xyz/high.w);
        float3 margin=(max(abs(lower),abs(upper))+1)*Roundoff; lower-=margin; upper+=margin;
        p.Rectangle.xy=min(p.Rectangle.xy,float2(lower.x+1,1-upper.y)*0.5);
        p.Rectangle.zw=max(p.Rectangle.zw,float2(upper.x+1,1-lower.y)*0.5); p.Near=min(p.Near,lower.z);
    }
    p.Outside=finite && outside!=0;
    p.Safe=p.Safe && finite && all(p.Rectangle.xy>=0) && all(p.Rectangle.zw<=1) && p.Near>0;
    return p;
}
bool Hidden(Projection p)
{
    if (!p.Safe) return false;
    uint2 extent=Counts.yz,first=(uint2)floor(p.Rectangle.xy*extent),last=min(extent-1,(uint2)floor(p.Rectangle.zw*extent));
    uint offset=0;
    while (any(last-first+1>MaximumFootprint))
    { offset+=extent.x*extent.y; extent=(extent+1)/2; first/=2; last/=2; }
    for (uint y=first.y;y<=last.y;++y) for (uint x=first.x;x<=last.x;++x)
    {
#if defined(EVANS_NANITE_FRAME_CULL)
        float depth=asfloat(DepthHierarchy.Load((offset+y*extent.x+x)*4));
#else
        float depth=asfloat(Feedback.Load((offset+y*extent.x+x)*4));
#endif
        if (!(depth<p.Near)) return false;
    }
    return true;
}
[numthreads(Threads,1,1)]
void CSReduce(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint index=(group.y*Target.w+group.x)*Threads+lane; if (index>=Target.y*Target.z) return;
    uint x=index%Target.y,y=index/Target.y; float farthest=0;
    if (Source.w==0) farthest=asfloat(Read(index));
    else [unroll] for (uint dy=0;dy<2;++dy) [unroll] for (uint dx=0;dx<2;++dx)
    {
        uint2 child=uint2(x,y)*2+uint2(dx,dy);
        [branch] if (child.x<Source.y && child.y<Source.z)
            farthest=max(farthest,asfloat(Feedback.Load((Source.x+child.y*Source.y+child.x)*4)));
    }
    Feedback.Store((Target.x+index)*4,asuint(farthest));
}
[numthreads(Threads,1,1)]
void CSCull(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint index=(group.y*Target.w+group.x)*Threads+lane; if (index>=Counts.x) return;
    uint query=Layout.x+index*QueryWords,state=MainVisible;
    if (Counts.w==0)
    {
        if (Project(query,CurrentRows).Outside) state=FrustumRejected;
        else
        {
            bool hasHistory=Read(query+IdentityWords)!=0;
#if defined(EVANS_NANITE_FRAME_CULL)
            hasHistory=hasHistory && Layout.w!=0;
#endif
            [branch] if (hasHistory) { if (Hidden(Project(query,PreviousRows))) state=Deferred; }
        }
    }
    else
    {
#if defined(EVANS_NANITE_FRAME_CULL)
        state=MainDecisions.Load(index*4);
#else
        state=Read(query+IdentityWords+1);
#endif
        if (state==Deferred) state=Hidden(Project(query,CurrentRows))?Occluded:PostVisible;
    }
    Feedback.Store((Layout.y+index)*4,state);
}
#endif
