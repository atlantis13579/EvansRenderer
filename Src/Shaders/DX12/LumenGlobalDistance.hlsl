// Native D3D12 global-field composition. Raw word addresses retain integer IDs;
// no generated source or shader includes are used. See Wright et al., Lumen (2022).
static const uint MaximumCascades=4,ComposeThreads=64,DispatchColumns=1024;
// Three source error bands cover the sampled point, native hit threshold
// and the uncertainty of the SDF at that possible hit.
struct Clipmap { float4 OriginCell,Metrics; uint4 Storage; };
cbuffer ComposeConstants : register(b0)
{
    uint4 SceneCounts,SceneOffsets,Layout;
    float4 Numeric;
    Clipmap Clipmaps[MaximumCascades];
    uint4 Dispatch;
};
ByteAddressBuffer Source : register(t0);
ByteAddressBuffer DirtyVoxels : register(t1);
RWByteAddressBuffer Destination : register(u0);
static const uint WordBytes=4,VectorWords=4,FieldWords=20,InstanceWords=20;
static const uint BrickDimensions=4,FieldStorage=8,FieldGrid=12,FieldErrors=16,InstanceRows=4,InstanceMetadata=16;
static const uint Interior=6,Halo=1,BrickSide=8,BrickSamples=512,FirstBrick=2,ShadowFlag=1,RowStride=ComposeThreads*DispatchColumns;
static const float Quantization=32767.0,SourceErrorBands=3.0;
uint4 loadWords(uint offset) { return Source.Load4(offset*WordBytes); }
float4 loadFloats(uint offset) { return asfloat(loadWords(offset)); }
float maximumAbsolute(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float distanceLength(float3 p) { float m=maximumAbsolute(p); return m>0?length(p/m)*m:0; }
float sampleField(uint address,float3 position)
{
    float4 transform=loadFloats(address+FieldGrid); uint3 extent=loadWords(address).xyz;
    float3 location=clamp((position-transform.xyz)/transform.w-0.5,0,float3(extent)-1);
    uint3 lower=uint3(floor(location)),brick=lower/Interior,local=lower%Interior+Halo;
    uint3 dimensions=loadWords(address+BrickDimensions).xyz; uint4 storage=loadWords(address+FieldStorage);
    uint page=Source.Load((storage.x+(brick.z*dimensions.y+brick.y)*dimensions.x+brick.x)*WordBytes);
    float band=loadFloats(address+FieldErrors).x; if (page<FirstBrick) return page==1?-band:band;
    float3 fraction=location-lower; float value=0;
    for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),at=local+bit;
        uint sample=(page-FirstBrick)*BrickSamples+(at.z*BrickSide+at.y)*BrickSide+at.x;
        uint packed=Source.Load((storage.y+sample/2)*WordBytes);
        int encoded=(sample&1)?int(packed)>>16:int(packed<<16)>>16;
        float3 weight=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
        value+=float(encoded)*weight.x*weight.y*weight.z;
    }
    return value*(band/Quantization);
}
float2 evaluateInstance(uint index,float3 position,float band)
{
    uint instance=SceneOffsets.y+index*InstanceWords;
    uint field=SceneOffsets.x+loadWords(instance+InstanceMetadata).x*FieldWords;
    float3 offset=position-loadFloats(instance).xyz;
    float3 x=loadFloats(instance+InstanceRows).xyz,y=loadFloats(instance+InstanceRows+VectorWords).xyz,z=loadFloats(instance+InstanceRows+VectorWords*2).xyz;
    float3 local=float3(dot(x,offset),dot(y,offset),dot(z,offset));
    float scale=max(maximumAbsolute(x),max(maximumAbsolute(y),maximumAbsolute(z)));
    if (!all(isfinite(local)) || !isfinite(scale) || scale<=0) return 0;
    x=abs(x/scale); y=abs(y/scale); z=abs(z/scale);
    float row=max(dot(x,float3(1,1,1)),max(dot(y,float3(1,1,1)),dot(z,float3(1,1,1))));
    // sqrt(||A||1 ||A||inf) bounds the largest singular value for sheared instances.
    float rate=scale*sqrt(row*maximumAbsolute(x+y+z)); if (!isfinite(rate) || rate<=0) return 0;
    float4 grid=loadFloats(field+FieldGrid),errors=loadFloats(field+FieldErrors);
    float3 upper=grid.xyz+float3(loadWords(field).xyz)*grid.w;
    float3 sampleAt=clamp(local,grid.xyz+grid.w*0.5,upper-grid.w*0.5);
    float sampled=sampleField(field,sampleAt),extension=distanceLength(local-sampleAt);
    float outside=distanceLength(local-clamp(local,grid.xyz,upper));
    float roundoff=Numeric.y*max(maximumAbsolute(local),max(maximumAbsolute(grid.xyz),maximumAbsolute(upper)));
    float signedValue=outside>0?max(outside,sampled-extension):sampled;
    float empty=max(outside,sampled-extension-SourceErrorBands*errors.y-Numeric.x*grid.w)-roundoff;
    if (!isfinite(signedValue) || !isfinite(empty)) return 0;
    return float2(clamp(signedValue/rate,-band,band),clamp(empty/rate,0,band));
}
[numthreads(ComposeThreads,1,1)]
void CSCompose(uint3 invocation : SV_DispatchThreadID)
{
    uint work=invocation.x+invocation.y*RowStride; if (work>=Dispatch.y) return;
    uint index=DirtyVoxels.Load((Dispatch.x+work)*WordBytes);
    uint volume=Layout.y*Layout.y*Layout.y,level=index/volume,voxel=index%volume;
    Clipmap clip=Clipmaps[level];
    uint3 coordinate=uint3(voxel%Layout.y,voxel/Layout.y%Layout.y,voxel/(Layout.y*Layout.y));
    float3 position=clip.OriginCell.xyz+float3(coordinate)*clip.OriginCell.w; float4 result=clip.Metrics.y.xxxx;
    for (uint object=0;object<SceneCounts.y;++object)
    {
        float2 value=evaluateInstance(object,position,clip.Metrics.y); result.xy=min(result.xy,value);
        if ((loadWords(SceneOffsets.y+object*InstanceWords+InstanceMetadata).z&ShadowFlag)!=0) result.zw=min(result.zw,value);
    }
    Destination.Store4((clip.Storage.x+voxel*VectorWords)*WordBytes,asuint(result));
}
