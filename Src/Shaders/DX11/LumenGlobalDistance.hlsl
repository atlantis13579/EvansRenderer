// Camera-centered global distance clipmaps, composed from independent mesh SDFs.
// Signed unions preserve solid interiors. Clearance subtracts source and target
// interpolation error plus the native hit band; uncertain cells require mesh tracing.
static const uint MaximumCascades=4,ComposeThreads=64,DispatchColumns=1024;
// Three source error bands cover the sampled point, native hit threshold
// and the uncertainty of the SDF at that possible hit.
struct Grid { float4 OriginCell,Metrics; uint4 Storage; };
cbuffer Composition : register(b0)
{
    uint4 Counts,Offsets,Layout;
    float4 Numeric;
    Grid Grids[MaximumCascades];
    uint4 Dispatch;
};
StructuredBuffer<uint> Mesh : register(t0);
StructuredBuffer<uint> DirtyVoxels : register(t1);
RWStructuredBuffer<uint> Global : register(u0);
static const uint FieldWords=20,InstanceWords=20,VectorWords=4;
static const uint FieldBricks=4,FieldStorage=8,FieldGrid=12,FieldErrors=16,InstanceRows=4,InstanceMetadata=16;
static const uint BrickInterior=6,BrickHalo=1,BrickSide=8,BrickSamples=512,FirstBrick=2,ShadowCaster=1;
static const uint DispatchStride=ComposeThreads*DispatchColumns;
static const float SignedRange=32767.0,SourceErrorBands=3.0;
uint4 words(uint at) { return uint4(Mesh[at],Mesh[at+1],Mesh[at+2],Mesh[at+3]); }
float4 floats(uint at) { return asfloat(words(at)); }
float magnitude(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float stableLength(float3 p) { float scale=magnitude(p); return scale>0?length(p/scale)*scale:0; }
float meshSample(uint field,float3 position)
{
    uint3 size=words(field).xyz,brickSize=words(field+FieldBricks).xyz;
    uint4 storage=words(field+FieldStorage); float4 grid=floats(field+FieldGrid);
    float band=floats(field+FieldErrors).x;
    float3 coordinate=clamp((position-grid.xyz)/grid.w-0.5,0,float3(size)-1);
    uint3 lower=uint3(floor(coordinate)),brick=lower/BrickInterior,local=lower%BrickInterior+BrickHalo;
    float3 f=coordinate-lower;
    uint page=Mesh[storage.x+(brick.z*brickSize.y+brick.y)*brickSize.x+brick.x];
    if (page<FirstBrick) return page==1?-band:band;
    float sum=0;
    for (uint z=0;z<2;++z) for (uint y=0;y<2;++y) for (uint x=0;x<2;++x)
    {
        uint sample=(page-FirstBrick)*BrickSamples+((local.z+z)*BrickSide+local.y+y)*BrickSide+local.x+x;
        uint pair=Mesh[storage.y+sample/2]; int value=(sample&1)?int(pair)>>16:int(pair<<16)>>16;
        float3 weight=float3(x?f.x:1-f.x,y?f.y:1-f.y,z?f.z:1-f.z);
        sum+=value*weight.x*weight.y*weight.z;
    }
    return sum*band/SignedRange;
}
float2 instanceDistance(uint index,float3 world,float band)
{
    uint instance=Offsets.y+index*InstanceWords,field=Offsets.x+words(instance+InstanceMetadata).x*FieldWords;
    float3 delta=world-floats(instance).xyz;
    float3 a=floats(instance+InstanceRows).xyz,b=floats(instance+InstanceRows+VectorWords).xyz,c=floats(instance+InstanceRows+VectorWords*2).xyz;
    float3 p=float3(dot(a,delta),dot(b,delta),dot(c,delta));
    float scale=max(magnitude(a),max(magnitude(b),magnitude(c)));
    if (!all(isfinite(p)) || !isfinite(scale) || scale<=0) return 0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    float rows=max(dot(a,1.0.xxx),max(dot(b,1.0.xxx),dot(c,1.0.xxx)));
    float norm=scale*sqrt(rows*magnitude(a+b+c));
    if (!isfinite(norm) || norm<=0) return 0;
    float4 grid=floats(field+FieldGrid),errors=floats(field+FieldErrors);
    float3 end=grid.xyz+float3(words(field).xyz)*grid.w;
    float3 samplePosition=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float sdf=meshSample(field,samplePosition),outside=stableLength(p-clamp(p,grid.xyz,end));
    float guard=Numeric.y*max(magnitude(p),max(magnitude(grid.xyz),magnitude(end)));
    float extension=stableLength(p-samplePosition);
    float clearance=max(outside,sdf-SourceErrorBands*errors.y-Numeric.x*grid.w-extension)-guard;
    float signedDistance=outside>0?max(outside,sdf-extension):sdf;
    if (!isfinite(clearance) || !isfinite(signedDistance)) return 0;
    return float2(clamp(signedDistance/norm,-band,band),clamp(clearance/norm,0,band));
}
[numthreads(ComposeThreads,1,1)]
void CSCompose(uint3 thread : SV_DispatchThreadID)
{
    uint work=thread.x+thread.y*DispatchStride; if (work>=Dispatch.y) return;
    uint index=DirtyVoxels[Dispatch.x+work];
    uint perGrid=Layout.y*Layout.y*Layout.y,level=index/perGrid,voxel=index%perGrid;
    Grid grid=Grids[level]; uint3 cell=uint3(voxel%Layout.y,(voxel/Layout.y)%Layout.y,voxel/(Layout.y*Layout.y));
    float3 world=grid.OriginCell.xyz+float3(cell)*grid.OriginCell.w;
    float4 result=grid.Metrics.y.xxxx;
    for (uint instance=0;instance<Counts.y;++instance)
    {
        float2 distance=instanceDistance(instance,world,grid.Metrics.y); result.xy=min(result.xy,distance);
        if ((words(Offsets.y+instance*InstanceWords+InstanceMetadata).z&ShadowCaster)!=0) result.zw=min(result.zw,distance);
    }
    uint address=grid.Storage.x+voxel*VectorWords; uint4 encoded=asuint(result);
    Global[address]=encoded.x; Global[address+1]=encoded.y; Global[address+2]=encoded.z; Global[address+3]=encoded.w;
}
