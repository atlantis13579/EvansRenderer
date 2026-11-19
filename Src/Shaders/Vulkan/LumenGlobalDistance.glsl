#version 450
// GPU composition of mesh SDFs into camera-snapped global clipmaps. Two channels
// per visibility class store signed union distance and a conservative empty radius.
const uint MaximumCascades=4u,ComposeThreads=64u,DispatchColumns=1024u;
// Three source error bands cover the sampled point, native hit threshold
// and the uncertainty of the SDF at that possible hit.
struct Clipmap { vec4 OriginCell,Metrics; uvec4 Storage; };
layout(std140,set=0,binding=0) uniform Composition
{
    uvec4 Counts,Offsets,Layout;
    vec4 Numeric;
    Clipmap Grids[MaximumCascades];
    uvec4 Dispatch;
} Frame;
layout(std430,set=0,binding=1) readonly buffer MeshScene { uint Words[]; } Mesh;
layout(std430,set=0,binding=2) readonly buffer DirtyList { uint Indices[]; } Dirty;
layout(std430,set=0,binding=3) writeonly buffer GlobalScene { uint Words[]; } Global;
const uint FieldWords=20u,InstanceWords=20u,VectorWords=4u;
const uint BrickDimensions=4u,FieldStorage=8u,FieldGrid=12u,FieldErrors=16u,InstanceRows=4u,InstanceData=16u;
const uint BrickInterior=6u,BrickHalo=1u,BrickSide=8u,BrickSamples=512u,FirstBrick=2u,ShadowFlag=1u,DispatchStride=ComposeThreads*DispatchColumns;
const float SignedRange=32767.0,SourceErrorBands=3.0;
uvec4 words(uint at) { return uvec4(Mesh.Words[at],Mesh.Words[at+1u],Mesh.Words[at+2u],Mesh.Words[at+3u]); }
vec4 floats(uint at) { return uintBitsToFloat(words(at)); }
bool finiteScalar(float v) { return !isnan(v) && !isinf(v); }
bool finiteVector(vec3 v) { return !any(isnan(v)) && !any(isinf(v)); }
float magnitude(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float stableLength(vec3 v) { float scale=magnitude(v); return scale>0.0?length(v/scale)*scale:0.0; }
float sampleMesh(uint field,vec3 p)
{
    vec4 grid=floats(field+FieldGrid); uvec3 size=words(field).xyz,bricks=words(field+BrickDimensions).xyz;
    vec3 uv=clamp((p-grid.xyz)/grid.w-0.5,vec3(0.0),vec3(size)-1.0);
    uvec3 lower=uvec3(floor(uv)),brick=lower/BrickInterior,local=lower%BrickInterior+BrickHalo;
    uvec4 storage=words(field+FieldStorage); float band=floats(field+FieldErrors).x;
    uint page=Mesh.Words[storage.x+(brick.z*bricks.y+brick.y)*bricks.x+brick.x];
    if (page<FirstBrick) return page==1u?-band:band;
    vec3 fraction=uv-vec3(lower); float sum=0.0;
    for (uint z=0u;z<2u;++z) for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
    {
        uint sampleIndex=(page-FirstBrick)*BrickSamples+((local.z+z)*BrickSide+local.y+y)*BrickSide+local.x+x;
        uint packed=Mesh.Words[storage.y+sampleIndex/2u];
        int encoded=(sampleIndex&1u)!=0u?int(packed)>>16:int(packed<<16)>>16;
        vec3 weight=mix(vec3(1.0)-fraction,fraction,bvec3(x!=0u,y!=0u,z!=0u));
        sum+=float(encoded)*weight.x*weight.y*weight.z;
    }
    return sum*band/SignedRange;
}
vec2 instanceDistance(uint index,vec3 world,float band)
{
    uint instance=Frame.Offsets.y+index*InstanceWords,field=Frame.Offsets.x+words(instance+InstanceData).x*FieldWords;
    vec3 offset=world-floats(instance).xyz;
    vec3 a=floats(instance+InstanceRows).xyz,b=floats(instance+InstanceRows+VectorWords).xyz,c=floats(instance+InstanceRows+VectorWords*2u).xyz;
    vec3 p=vec3(dot(a,offset),dot(b,offset),dot(c,offset));
    float scale=max(magnitude(a),max(magnitude(b),magnitude(c)));
    if (!finiteVector(p) || !finiteScalar(scale) || scale<=0.0) return vec2(0.0);
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    float rows=max(dot(a,vec3(1.0)),max(dot(b,vec3(1.0)),dot(c,vec3(1.0))));
    float norm=scale*sqrt(rows*magnitude(a+b+c)); if (!finiteScalar(norm) || norm<=0.0) return vec2(0.0);
    vec4 grid=floats(field+FieldGrid),errors=floats(field+FieldErrors);
    vec3 end=grid.xyz+vec3(words(field).xyz)*grid.w;
    vec3 sampleAt=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float sampleValue=sampleMesh(field,sampleAt),extension=stableLength(p-sampleAt);
    float outside=stableLength(p-clamp(p,grid.xyz,end));
    float guard=Frame.Numeric.y*max(magnitude(p),max(magnitude(grid.xyz),magnitude(end)));
    float signedValue=outside>0.0?max(outside,sampleValue-extension):sampleValue;
    float clearance=max(outside,sampleValue-extension-SourceErrorBands*errors.y-Frame.Numeric.x*grid.w)-guard;
    if (!finiteScalar(signedValue) || !finiteScalar(clearance)) return vec2(0.0);
    return vec2(clamp(signedValue/norm,-band,band),clamp(clearance/norm,0.0,band));
}
layout(local_size_x=ComposeThreads) in;
void main()
{
    uint work=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*DispatchStride; if (work>=Frame.Dispatch.y) return;
    uint index=Dirty.Indices[Frame.Dispatch.x+work];
    uint side=Frame.Layout.y,volume=side*side*side,level=index/volume,voxel=index%volume;
    Clipmap grid=Frame.Grids[level]; uvec3 coordinate=uvec3(voxel%side,(voxel/side)%side,voxel/(side*side));
    vec3 world=grid.OriginCell.xyz+vec3(coordinate)*grid.OriginCell.w; vec4 result=vec4(grid.Metrics.y);
    for (uint instance=0u;instance<Frame.Counts.y;++instance)
    {
        vec2 distance=instanceDistance(instance,world,grid.Metrics.y); result.xy=min(result.xy,distance);
        if ((words(Frame.Offsets.y+instance*InstanceWords+InstanceData).z&ShadowFlag)!=0u) result.zw=min(result.zw,distance);
    }
    uint address=grid.Storage.x+voxel*VectorWords; uvec4 encoded=floatBitsToUint(result);
    Global.Words[address]=encoded.x; Global.Words[address+1u]=encoded.y; Global.Words[address+2u]=encoded.z; Global.Words[address+3u]=encoded.w;
}
