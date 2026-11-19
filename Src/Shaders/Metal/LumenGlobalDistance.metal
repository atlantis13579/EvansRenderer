#include <metal_stdlib>
using namespace metal;
// Independent Metal clipmap composition. Transform norms bound distances under
// shear; signed interiors and uncertain cells never become empty-space evidence.
constant uint MaximumCascades=4,ComposeThreads=64,DispatchColumns=1024;
// Three source error bands cover the sampled point, native hit threshold
// and the uncertainty of the SDF at that possible hit.
struct Clipmap { float4 OriginCell,Metrics; uint4 Storage; };
struct ComposeFrame { uint4 Counts,Offsets,Layout; float4 Numeric; Clipmap Grids[MaximumCascades]; uint4 Dispatch; };
constant uint FieldWords=20,InstanceWords=20,VectorWords=4;
constant uint FieldBricks=4,FieldStorage=8,FieldGrid=12,FieldErrors=16,InstanceRows=4,InstanceMetadata=16;
constant uint BrickInterior=6,BrickHalo=1,BrickSide=8,BrickSamples=512,FirstBrick=2,ShadowFlag=1,DispatchStride=ComposeThreads*DispatchColumns;
constant float SignedRange=32767.0f,SourceErrorBands=3.0f;
uint4 readWords(device const uint* scene,uint i) { return uint4(scene[i],scene[i+1],scene[i+2],scene[i+3]); }
float4 readFloats(device const uint* scene,uint i) { return as_type<float4>(readWords(scene,i)); }
float magnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float stableLength(float3 v) { float scale=magnitude(v); return scale>0?length(v/scale)*scale:0; }
float sampleMesh(device const uint* scene,uint field,float3 p)
{
    float4 grid=readFloats(scene,field+FieldGrid); uint3 dimensions=readWords(scene,field).xyz;
    float3 coordinate=clamp((p-grid.xyz)/grid.w-0.5f,float3(0),float3(dimensions)-1);
    uint3 lower=uint3(floor(coordinate)),brick=lower/BrickInterior,local=lower%BrickInterior+BrickHalo;
    uint3 bricks=readWords(scene,field+FieldBricks).xyz; uint4 storage=readWords(scene,field+FieldStorage);
    uint page=scene[storage.x+(brick.z*bricks.y+brick.y)*bricks.x+brick.x]; float band=readFloats(scene,field+FieldErrors).x;
    if (page<FirstBrick) return page==1?-band:band;
    float3 f=coordinate-float3(lower); float result=0;
    for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),cell=local+bit;
        uint sample=(page-FirstBrick)*BrickSamples+(cell.z*BrickSide+cell.y)*BrickSide+cell.x;
        uint pair=scene[storage.y+sample/2]; int encoded=(sample&1)?int(pair)>>16:int(pair<<16)>>16;
        float3 weight=select(1-f,f,bit!=uint3(0)); result+=float(encoded)*weight.x*weight.y*weight.z;
    }
    return result*band/SignedRange;
}
float2 evaluateInstance(device const uint* scene,constant ComposeFrame& frame,uint id,float3 world,float band)
{
    uint instance=frame.Offsets.y+id*InstanceWords,field=frame.Offsets.x+readWords(scene,instance+InstanceMetadata).x*FieldWords;
    float3 delta=world-readFloats(scene,instance).xyz;
    float3 a=readFloats(scene,instance+InstanceRows).xyz,b=readFloats(scene,instance+InstanceRows+VectorWords).xyz,c=readFloats(scene,instance+InstanceRows+VectorWords*2).xyz;
    float3 local=float3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(magnitude(a),max(magnitude(b),magnitude(c)));
    if (!all(isfinite(local)) || !isfinite(scale) || scale<=0) return float2(0);
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    float row=max(dot(a,float3(1)),max(dot(b,float3(1)),dot(c,float3(1))));
    float norm=scale*sqrt(row*magnitude(a+b+c)); if (!isfinite(norm) || norm<=0) return float2(0);
    float4 grid=readFloats(scene,field+FieldGrid),errors=readFloats(scene,field+FieldErrors);
    float3 upper=grid.xyz+float3(readWords(scene,field).xyz)*grid.w;
    float3 sampleAt=clamp(local,grid.xyz+grid.w*0.5f,upper-grid.w*0.5f);
    float sdf=sampleMesh(scene,field,sampleAt),extension=stableLength(local-sampleAt),outside=stableLength(local-clamp(local,grid.xyz,upper));
    float guard=frame.Numeric.y*max(magnitude(local),max(magnitude(grid.xyz),magnitude(upper)));
    float signedDistance=outside>0?max(outside,sdf-extension):sdf;
    float clearance=max(outside,sdf-extension-SourceErrorBands*errors.y-frame.Numeric.x*grid.w)-guard;
    if (!isfinite(signedDistance) || !isfinite(clearance)) return float2(0);
    return float2(clamp(signedDistance/norm,-band,band),clamp(clearance/norm,0.0f,band));
}
kernel void lumenComposeGlobalDistance(constant ComposeFrame& frame [[buffer(0)]],device const uint* scene [[buffer(1)]],
    device const uint* dirty [[buffer(2)]],device uint* output [[buffer(3)]],uint3 invocation [[thread_position_in_grid]])
{
    uint work=invocation.x+invocation.y*DispatchStride; if (work>=frame.Dispatch.y) return;
    uint index=dirty[frame.Dispatch.x+work];
    uint side=frame.Layout.y,volume=side*side*side,level=index/volume,voxel=index%volume;
    Clipmap grid=frame.Grids[level]; uint3 cell=uint3(voxel%side,voxel/side%side,voxel/(side*side));
    float3 world=grid.OriginCell.xyz+float3(cell)*grid.OriginCell.w; float4 result(grid.Metrics.y);
    for (uint instance=0;instance<frame.Counts.y;++instance)
    {
        float2 distance=evaluateInstance(scene,frame,instance,world,grid.Metrics.y); result.xy=min(result.xy,distance);
        if ((readWords(scene,frame.Offsets.y+instance*InstanceWords+InstanceMetadata).z&ShadowFlag)!=0) result.zw=min(result.zw,distance);
    }
    uint address=grid.Storage.x+voxel*VectorWords; uint4 encoded=as_type<uint4>(result);
    output[address]=encoded.x; output[address+1]=encoded.y; output[address+2]=encoded.z; output[address+3]=encoded.w;
}
