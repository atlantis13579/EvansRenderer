#version 310 es
precision highp float;
precision highp int;
// ES 3.1 uses two storage bindings, below the mandatory compute limit. A native
// buffer copy retains the source prefix; this kernel writes only clipmap voxels.
#define ComposeThreads 64
const uint MaximumCascades=4u,DispatchColumns=1024u;
// Three source error bands cover the sampled point, native hit threshold
// and the uncertainty of the SDF at that possible hit.
struct Grid { vec4 OriginCell,Metrics; uvec4 Storage; };
layout(std140,binding=0) uniform ComposeFrame
{
    uvec4 Counts,Offsets,Layout;
    vec4 Numeric;
    Grid Levels[MaximumCascades];
    uvec4 Dispatch;
} Frame;
layout(std430,binding=0) readonly buffer InputScene { uint Values[]; } Input;
layout(std430,binding=1) readonly buffer DirtyVoxels { uint Indices[]; } Dirty;
layout(std430,binding=2) writeonly buffer OutputScene { uint Values[]; } Output;
const uint FieldStride=20u,InstanceStride=20u,VectorStride=4u;
const uint FieldBricks=4u,FieldStorage=8u,FieldGrid=12u,FieldError=16u,MatrixRows=4u,InstanceData=16u;
const uint BrickInner=6u,BrickBorder=1u,BrickSide=8u,BrickVolume=512u,FirstBrick=2u,CastShadow=1u,RowStride=uint(ComposeThreads)*DispatchColumns;
const float Quantization=32767.0,SourceErrorBands=3.0;
uvec4 readWords(uint i) { return uvec4(Input.Values[i],Input.Values[i+1u],Input.Values[i+2u],Input.Values[i+3u]); }
vec4 readFloats(uint i) { return uintBitsToFloat(readWords(i)); }
float maxAbs(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float norm3(vec3 v) { float scale=maxAbs(v); return scale>0.0?length(v/scale)*scale:0.0; }
bool finiteValue(float v) { return !isnan(v) && !isinf(v); }
float fieldValue(uint field,vec3 point)
{
    vec4 grid=readFloats(field+FieldGrid); uvec3 dimensions=readWords(field).xyz;
    vec3 coordinate=clamp((point-grid.xyz)/grid.w-0.5,vec3(0),vec3(dimensions)-1.0);
    uvec3 lower=uvec3(floor(coordinate)),brick=lower/BrickInner,local=lower%BrickInner+BrickBorder;
    uvec3 bricks=readWords(field+FieldBricks).xyz; uvec4 storage=readWords(field+FieldStorage);
    uint page=Input.Values[storage.x+(brick.z*bricks.y+brick.y)*bricks.x+brick.x];
    float band=readFloats(field+FieldError).x; if (page<FirstBrick) return page==1u?-band:band;
    float value=0.0; vec3 f=coordinate-vec3(lower);
    for (uint corner=0u;corner<8u;++corner)
    {
        uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),cell=local+bit;
        uint sampleIndex=(page-FirstBrick)*BrickVolume+(cell.z*BrickSide+cell.y)*BrickSide+cell.x;
        uint pair=Input.Values[storage.y+sampleIndex/2u]; int sampleValue=(sampleIndex&1u)!=0u?int(pair)>>16:int(pair<<16)>>16;
        vec3 weight=mix(vec3(1)-f,f,notEqual(bit,uvec3(0)));
        value+=float(sampleValue)*weight.x*weight.y*weight.z;
    }
    return value*(band/Quantization);
}
vec2 instanceBounds(uint id,vec3 world,float band)
{
    uint instance=Frame.Offsets.y+id*InstanceStride,field=Frame.Offsets.x+readWords(instance+InstanceData).x*FieldStride;
    vec3 delta=world-readFloats(instance).xyz;
    vec3 a=readFloats(instance+MatrixRows).xyz,b=readFloats(instance+MatrixRows+VectorStride).xyz,c=readFloats(instance+MatrixRows+VectorStride*2u).xyz;
    vec3 local=vec3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(maxAbs(a),max(maxAbs(b),maxAbs(c)));
    if (any(isnan(local)) || any(isinf(local)) || !finiteValue(scale) || scale<=0.0) return vec2(0);
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    float row=max(dot(a,vec3(1)),max(dot(b,vec3(1)),dot(c,vec3(1))));
    float rate=scale*sqrt(row*maxAbs(a+b+c)); if (!finiteValue(rate) || rate<=0.0) return vec2(0);
    vec4 grid=readFloats(field+FieldGrid),errors=readFloats(field+FieldError);
    vec3 upper=grid.xyz+vec3(readWords(field).xyz)*grid.w;
    vec3 sampledAt=clamp(local,grid.xyz+grid.w*0.5,upper-grid.w*0.5);
    float sdf=fieldValue(field,sampledAt),extension=norm3(local-sampledAt),outside=norm3(local-clamp(local,grid.xyz,upper));
    float guard=Frame.Numeric.y*max(maxAbs(local),max(maxAbs(grid.xyz),maxAbs(upper)));
    float signedDistance=outside>0.0?max(outside,sdf-extension):sdf;
    float empty=max(outside,sdf-extension-SourceErrorBands*errors.y-Frame.Numeric.x*grid.w)-guard;
    if (!finiteValue(signedDistance) || !finiteValue(empty)) return vec2(0);
    return vec2(clamp(signedDistance/rate,-band,band),clamp(empty/rate,0.0,band));
}
layout(local_size_x=ComposeThreads) in;
void main()
{
    uint work=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*RowStride; if (work>=Frame.Dispatch.y) return;
    uint index=Dirty.Indices[Frame.Dispatch.x+work];
    uint side=Frame.Layout.y,volume=side*side*side,level=index/volume,voxel=index%volume;
    Grid grid=Frame.Levels[level]; uvec3 cell=uvec3(voxel%side,voxel/side%side,voxel/(side*side));
    vec3 world=grid.OriginCell.xyz+vec3(cell)*grid.OriginCell.w; vec4 value=vec4(grid.Metrics.y);
    for (uint instance=0u;instance<Frame.Counts.y;++instance)
    {
        vec2 bound=instanceBounds(instance,world,grid.Metrics.y); value.xy=min(value.xy,bound);
        if ((readWords(Frame.Offsets.y+instance*InstanceStride+InstanceData).z&CastShadow)!=0u) value.zw=min(value.zw,bound);
    }
    uint word=grid.Storage.x+voxel*VectorStride; uvec4 packed=floatBitsToUint(value);
    Output.Values[word]=packed.x; Output.Values[word+1u]=packed.y; Output.Values[word+2u]=packed.z; Output.Values[word+3u]=packed.w;
}
