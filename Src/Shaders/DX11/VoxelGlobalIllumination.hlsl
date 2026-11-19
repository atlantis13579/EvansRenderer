// Six-face voxel cone tracing of direct diffuse + emission. Mips contain
// premultiplied radiance and front/back-composited directional coverage.
// Crassin et al. (2011): https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
// This dense atlas implementation has finite bounds, black misses and one bounce.
static const uint MipCapacity=7,TracePass=0,CompositePass=3,HorizontalPass=1,VerticalPass=2,PbrModel=3;
static const float Pi=3.141592653589793,GoldenAngle=Pi*(3-sqrt(5.0));
static const float PixelCenter=0.5,ClearDepth=1,MinimumW=1e-7,NormalEpsilon=1e-12;
static const float MinimumOpacity=1e-6,StopTransmission=0.01,HalfMaximum=65504,HalfHeadroom=16;
static const float DielectricF0=0.04,ParallelThreshold=0.9,FilterSigma=1;
static const int FilterRadius=2;
cbuffer VoxelFrame:register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 Viewport,GridOriginCell,Trace,Filter,CameraForward;
    uint4 Options,Atlas;
    uint4 Levels[MipCapacity]; // Dimension, linear offset, face stride, reserved.
};
Texture2D<float> DepthImage:register(t0);
Texture2D<float4> NormalImage:register(t1),DiffuseImage:register(t2),AmbientImage:register(t3);
Texture2D<float4> VolumeAtlas:register(t4),PreviousRadiance:register(t5),SceneColor:register(t6);
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint vertex:SV_VertexID)
{ float2 p=float2((vertex<<1)&2,vertex&2); ScreenVertex result; result.Position=float4(p*2-1,0,1); return result; }
bool Receiver(int2 pixel,out float3 world,out float3 normal)
{
    world=normal=0; if (any(pixel<0) || any(pixel>=int2(Viewport.xy))) return false;
    float depth=DepthImage.Load(int3(pixel,0)); float4 n=NormalImage.Load(int3(pixel,0));
    if (depth>=ClearDepth || n.w<0 || dot(n.xyz,n.xyz)<=NormalEpsilon) return false;
    float2 uv=(float2(pixel)+PixelCenter)*Viewport.zw;
    float4 homogeneous=mul(InverseViewProjection,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    if (abs(homogeneous.w)<MinimumW) return false;
    world=homogeneous.xyz/homogeneous.w; normal=normalize(n.xyz); return all(isfinite(world));
}
float4 Spatial(float3 world,float3 outgoing,uint mip)
{
    uint4 level=Levels[mip]; float spacing=GridOriginCell.w*exp2(float(mip));
    float3 coordinate=(world-GridOriginCell.xyz)/spacing-PixelCenter;
    int3 lower=int3(floor(coordinate)); float3 fraction=frac(coordinate),axisWeights=abs(outgoing);
    axisWeights/=max(axisWeights.x+axisWeights.y+axisWeights.z,MinimumOpacity); float4 value=0;
    [unroll] for (uint axis=0;axis<3;++axis) [unroll] for (uint corner=0;corner<8;++corner)
    {
        int3 offset=int3(corner&1,(corner>>1)&1,(corner>>2)&1),cell=lower+offset;
        if (any(cell<0) || any(cell>=int(level.x))) continue;
        float3 weight=lerp(1-fraction,fraction,float3(offset));
        uint face=axis*2+(outgoing[axis]<0 ? 1u:0u);
        uint index=level.y+face*level.z+uint(cell.x)+level.x*(uint(cell.y)+level.x*uint(cell.z));
        value+=VolumeAtlas.Load(int3(index%Atlas.x,index/Atlas.x,0))*(axisWeights[axis]*weight.x*weight.y*weight.z);
    }
    return value;
}
float4 VoxelSample(float3 world,float3 outgoing,float lod)
{
    lod=clamp(lod,0,float(Options.z-1)); uint low=uint(floor(lod)),high=min(low+1,Options.z-1);
    return lerp(Spatial(world,outgoing,low),Spatial(world,outgoing,high),frac(lod));
}
float3 Cone(float3 origin,float3 direction)
{
    float cell=GridOriginCell.w,side=cell*Options.y,distance=0,transmission=1; float3 radiance=0;
    [loop] for (uint step=0;step<Atlas.z && distance<Trace.x && transmission>StopTransmission;++step)
    {
        float diameter=clamp(2*Trace.y*(distance+cell*PixelCenter),cell,side);
        float segment=min(diameter*Trace.z,Trace.x-distance);
        float3 location=origin+direction*(distance+segment*PixelCenter),unit=(location-GridOriginCell.xyz)/side;
        if (any(unit<0) || any(unit>1)) break;
        float4 sampleValue=VoxelSample(location,-direction,log2(diameter/cell)); float opacity=saturate(sampleValue.a);
        float adjusted=1-pow(1-opacity,segment/diameter);
        if (opacity>MinimumOpacity) radiance+=transmission*sampleValue.rgb*(adjusted/opacity);
        transmission*=1-adjusted; distance+=segment;
    }
    return radiance;
}
float3 Estimate(float3 world,float3 normal)
{
    float3 helper=abs(normal.y)<ParallelThreshold ? float3(0,1,0):float3(1,0,0);
    float3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent),sum=0;
    [loop] for (uint cone=0;cone<Options.w;++cone)
    {
        float radialSquared=(cone+PixelCenter)/Options.w,angle=GoldenAngle*cone;
        float3 direction=tangent*(sqrt(radialSquared)*cos(angle))+bitangent*(sqrt(radialSquared)*sin(angle))+normal*sqrt(1-radialSquared);
        sum+=Cone(world+normal*(GridOriginCell.w*Trace.w),direction);
    }
    return sum/Options.w;
}
float4 Gather(int2 pixel,float3 world,float3 normal)
{
    float3 sum=0; float mass=0; bool horizontal=Options.x==HorizontalPass;
    float sigma=max(GridOriginCell.w*Filter.x,MinimumOpacity);
    [unroll] for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        int2 at=pixel+(horizontal?int2(offset,0):int2(0,offset)); float3 location,n;
        if (!Receiver(at,location,n)) continue;
        float4 signal=PreviousRadiance.Load(int3(at,0)); if (signal.a<=0) continue;
        float optical=dot(location-world,CameraForward.xyz)/sigma;
        float angular=Filter.y>0 ? pow(saturate(dot(n,normal)),Filter.y):1;
        float weight=exp(-(float(offset*offset)/(FilterSigma*FilterSigma)+optical*optical)/2)*angular;
        sum+=signal.rgb*weight; mass+=weight;
    }
    return float4(mass>MinimumOpacity ? sum/mass:PreviousRadiance.Load(int3(pixel,0)).rgb,1);
}
float4 PSMain(ScreenVertex input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy);
    if (Options.x==CompositePass) return float4(PreviousRadiance.Load(int3(pixel,0)).rgb,0);
    float3 world,normal; if (!Receiver(pixel,world,normal)) return 0;
    float3 inside=(world-GridOriginCell.xyz)/(GridOriginCell.w*Options.y); if (any(inside<0) || any(inside>1)) return 0;
    float4 value=Options.x==TracePass ? float4(Estimate(world,normal),1):Gather(pixel,world,normal);
    if (Options.x==VerticalPass)
    {
        float3 diffuse=max(DiffuseImage.Load(int3(pixel,0)).rgb,0);
        float transfer=Atlas.w==PbrModel ? (1-DielectricF0)*(1-saturate(AmbientImage.Load(int3(pixel,0)).w)):Pi;
        value.rgb=min(max(value.rgb*diffuse*(transfer*Filter.z),0),max(HalfMaximum-HalfHeadroom-SceneColor.Load(int3(pixel,0)).rgb,0));
    }
    return value;
}
