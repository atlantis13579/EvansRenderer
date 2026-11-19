// DX12 directional radiance-volume reconstruction. The six outgoing faces store
// premultiplied radiance; parent cells composite depth layers before averaging.
// Crassin et al. (2011): https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
static const uint LevelCapacity=7,TracePhase=0,HorizontalPhase=1,VerticalPhase=2,CompositePhase=3,PhysicalModel=3;
static const float Pi=3.14159265358979323846,SpiralAngle=Pi*(3-sqrt(5.0));
static const float CenterOffset=0.5,EmptyDepth=1,ProjectiveFloor=1e-7,DirectionFloor=1e-12;
static const float WeightFloor=1e-6,TransmissionFloor=0.01,HalfCeiling=65504,HalfGuard=16,DielectricReflectance=0.04;
static const int ReconstructionRadius=2;
cbuffer VoxelParameters:register(b0)
{
    row_major float4x4 WorldFromClip;
    float4 FrameExtent,VolumeOriginSpacing,MarchControls,Reconstruction,EyeForward;
    uint4 VolumeControl,TextureControl;
    uint4 Hierarchy[LevelCapacity];
};
Texture2D<float> CameraDepth:register(t0);
Texture2D<float4> SurfaceNormal:register(t1),SurfaceDiffuse:register(t2),SurfaceAmbient:register(t3);
Texture2D<float4> RadianceVolume:register(t4),FilteredInput:register(t5),OpaqueRadiance:register(t6);
struct RasterPoint { float4 Position:SV_POSITION; };
RasterPoint VSMain(uint index:SV_VertexID)
{
    RasterPoint result; float2 coordinate=float2((index<<1)&2,index&2);
    result.Position=float4(coordinate*2-1,0,1); return result;
}
bool ReadReceiver(int2 pixel,out float3 position,out float3 normal)
{
    position=normal=0;
    if (any(pixel<0) || any(pixel>=int2(FrameExtent.xy))) return false;
    float depth=CameraDepth.Load(int3(pixel,0)); float4 material=SurfaceNormal.Load(int3(pixel,0));
    if (depth>=EmptyDepth || material.w<0 || dot(material.xyz,material.xyz)<=DirectionFloor) return false;
    float2 coordinate=(float2(pixel)+CenterOffset)*FrameExtent.zw;
    float4 p=mul(WorldFromClip,float4(2*coordinate.x-1,1-2*coordinate.y,depth,1));
    if (abs(p.w)<ProjectiveFloor) return false;
    position=p.xyz/p.w; normal=normalize(material.xyz); return all(isfinite(position));
}
float4 ReadLevel(float3 position,float3 outward,uint levelIndex)
{
    uint4 level=Hierarchy[levelIndex];
    float3 local=(position-VolumeOriginSpacing.xyz)/(VolumeOriginSpacing.w*exp2(float(levelIndex)))-CenterOffset;
    int3 first=int3(floor(local)); float3 fraction=frac(local),axisMass=abs(outward);
    axisMass/=max(axisMass.x+axisMass.y+axisMass.z,WeightFloor); float4 total=0;
    [unroll] for (uint z=0;z<2;++z) [unroll] for (uint y=0;y<2;++y) [unroll] for (uint x=0;x<2;++x)
    {
        int3 at=first+int3(x,y,z); if (any(at<0) || any(at>=int(level.x))) continue;
        float spatial=(x?fraction.x:1-fraction.x)*(y?fraction.y:1-fraction.y)*(z?fraction.z:1-fraction.z);
        uint voxel=uint(at.x)+level.x*(uint(at.y)+level.x*uint(at.z));
        [unroll] for (uint axis=0;axis<3;++axis)
        {
            uint face=axis*2+uint(outward[axis]<0),address=level.y+face*level.z+voxel;
            total+=RadianceVolume.Load(int3(address%TextureControl.x,address/TextureControl.x,0))*(spatial*axisMass[axis]);
        }
    }
    return total;
}
float4 ReadHierarchy(float3 position,float3 outward,float footprint)
{
    float level=clamp(log2(footprint/VolumeOriginSpacing.w),0,float(VolumeControl.z-1)); uint first=uint(level);
    return lerp(ReadLevel(position,outward,first),ReadLevel(position,outward,min(first+1,VolumeControl.z-1)),frac(level));
}
float3 IntegrateCone(float3 start,float3 direction)
{
    float spacing=VolumeOriginSpacing.w,extent=spacing*VolumeControl.y,traveled=0,through=1; float3 sum=0;
    [loop] for (uint iteration=0;iteration<TextureControl.z && traveled<MarchControls.x && through>TransmissionFloor;++iteration)
    {
        float footprint=min(max(2*MarchControls.y*(traveled+spacing*CenterOffset),spacing),extent);
        float segmentLength=min(footprint*MarchControls.z,MarchControls.x-traveled);
        float3 samplePoint=start+direction*(traveled+segmentLength*CenterOffset),relative=(samplePoint-VolumeOriginSpacing.xyz)/extent;
        if (any(relative<0) || any(relative>1)) break;
        float4 value=ReadHierarchy(samplePoint,-direction,footprint); float opacity=saturate(value.a);
        float segmentOpacity=1-pow(1-opacity,segmentLength/footprint);
        if (opacity>WeightFloor) sum+=through*value.rgb*(segmentOpacity/opacity);
        through*=1-segmentOpacity; traveled+=segmentLength;
    }
    return sum;
}
float3 DiffuseCones(float3 position,float3 normal)
{
    const float AxisParallelLimit=0.9;
    float3 reference=abs(normal.y)<AxisParallelLimit ? float3(0,1,0):float3(1,0,0);
    float3 u=normalize(cross(reference,normal)),v=cross(normal,u),total=0;
    [loop] for (uint index=0;index<VolumeControl.w;++index)
    {
        float diskRadiusSquared=(index+CenterOffset)/VolumeControl.w,azimuth=index*SpiralAngle;
        float3 axis=u*(sqrt(diskRadiusSquared)*cos(azimuth))+v*(sqrt(diskRadiusSquared)*sin(azimuth))+normal*sqrt(1-diskRadiusSquared);
        total+=IntegrateCone(position+normal*(VolumeOriginSpacing.w*MarchControls.w),axis);
    }
    return total/VolumeControl.w;
}
float3 Reconstruct(int2 pixel,float3 position,float3 normal)
{
    float3 total=0; float mass=0,depthScale=max(VolumeOriginSpacing.w*Reconstruction.x,WeightFloor);
    [unroll] for (int shift=-ReconstructionRadius;shift<=ReconstructionRadius;++shift)
    {
        int2 neighbor=pixel+(VolumeControl.x==HorizontalPhase ? int2(shift,0):int2(0,shift)); float3 p,n;
        if (!ReadReceiver(neighbor,p,n)) continue;
        float4 sampleValue=FilteredInput.Load(int3(neighbor,0)); if (sampleValue.a<=0) continue;
        float depth=dot(p-position,EyeForward.xyz)/depthScale;
        float angle=Reconstruction.y>0 ? pow(saturate(dot(n,normal)),Reconstruction.y):1;
        float weight=exp(-(float(shift*shift)+depth*depth)/2)*angle;
        total+=sampleValue.rgb*weight; mass+=weight;
    }
    return mass>WeightFloor ? total/mass:FilteredInput.Load(int3(pixel,0)).rgb;
}
float4 PSMain(RasterPoint fragment):SV_TARGET
{
    int2 pixel=int2(fragment.Position.xy);
    if (VolumeControl.x==CompositePhase) return float4(FilteredInput.Load(int3(pixel,0)).rgb,0);
    float3 p,n; if (!ReadReceiver(pixel,p,n)) return 0;
    float3 coordinate=(p-VolumeOriginSpacing.xyz)/(VolumeOriginSpacing.w*VolumeControl.y); if (any(coordinate<0) || any(coordinate>1)) return 0;
    float3 incoming=VolumeControl.x==TracePhase ? DiffuseCones(p,n):Reconstruct(pixel,p,n);
    if (VolumeControl.x==VerticalPhase)
    {
        float materialScale=TextureControl.w==PhysicalModel ? (1-DielectricReflectance)*(1-saturate(SurfaceAmbient.Load(int3(pixel,0)).w)):Pi;
        incoming=max(incoming*max(SurfaceDiffuse.Load(int3(pixel,0)).rgb,0)*(materialScale*Reconstruction.z),0);
        incoming=min(incoming,max(HalfCeiling-HalfGuard-OpaqueRadiance.Load(int3(pixel,0)).rgb,0));
    }
    return float4(incoming,1);
}
