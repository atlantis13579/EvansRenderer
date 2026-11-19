#include <metal_stdlib>
using namespace metal;
// Metal's own dense directional voxel cone tracer. Geometry and direct diffuse
// injection are view-independent CPU producers; this shader gathers one bounce.
// Crassin et al. (2011): https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
constant uint MipCapacity=7,TracePass=0,HorizontalPass=1,VerticalPass=2,CompositePass=3,PbrModel=3;
constant float Pi=3.141592653589793,GoldenAngle=Pi*(3-sqrt(5.0f));
constant float PixelCenter=0.5,BackgroundDepth=1,ProjectionTolerance=1e-7,NormalTolerance=1e-12;
constant float MinimumWeight=1e-6,TransmissionCutoff=0.01,ParallelLimit=0.9;
constant float HalfMaximum=65504,HalfHeadroom=16,DielectricF0=0.04;
constant int FilterRadius=2;
struct VoxelFrame
{
    float4 ClipToWorld[4];
    float4 Viewport,GridOriginCell,Trace,Filter,CameraForward;
    uint4 Options,Atlas;
    uint4 Mips[MipCapacity]; // Dimension, linear offset, face stride, reserved.
};
struct Fullscreen { float4 Position [[position]]; };
vertex Fullscreen VSMain(uint vertexId [[vertex_id]])
{
    float2 uv=float2((vertexId<<1)&2,vertexId&2); return {float4(uv*2-1,0,1)};
}
bool Receiver(int2 pixel,constant VoxelFrame& frame,texture2d<float,access::read> depth,
    texture2d<float,access::read> normals,thread float3& position,thread float3& normal)
{
    position=normal=float3(0);
    if (any(pixel<0) || any(pixel>=int2(frame.Viewport.xy))) return false;
    float z=depth.read(uint2(pixel)).r; float4 n=normals.read(uint2(pixel)); float squared=dot(n.xyz,n.xyz);
    if (!(z<BackgroundDepth && n.w>=0 && squared>NormalTolerance && isfinite(squared))) return false;
    float2 uv=(float2(pixel)+PixelCenter)*frame.Viewport.zw;
    float4 clip=float4(uv.x*2-1,1-uv.y*2,z,1);
    float4 h=float4(dot(frame.ClipToWorld[0],clip),dot(frame.ClipToWorld[1],clip),dot(frame.ClipToWorld[2],clip),dot(frame.ClipToWorld[3],clip));
    if (abs(h.w)<ProjectionTolerance) return false;
    position=h.xyz/h.w; normal=n.xyz*rsqrt(squared); return all(isfinite(position));
}
float4 SpatialRadiance(float3 position,float3 outward,uint mip,constant VoxelFrame& frame,
    texture2d<float,access::read> volume)
{
    uint4 level=frame.Mips[mip];
    float3 coordinate=(position-frame.GridOriginCell.xyz)/(frame.GridOriginCell.w*exp2(float(mip)))-PixelCenter;
    int3 low=int3(floor(coordinate)); float3 blend=fract(coordinate),axisWeight=abs(outward);
    axisWeight/=max(axisWeight.x+axisWeight.y+axisWeight.z,MinimumWeight); float4 result=0;
    for (uint axis=0;axis<3;++axis)
    {
        uint face=axis*2+uint(outward[axis]<0);
        for (uint corner=0;corner<8;++corner)
        {
            int3 offset=int3(corner&1,(corner>>1)&1,(corner>>2)&1),cell=low+offset;
            if (any(cell<0) || any(cell>=int(level.x))) continue;
            float3 weight=mix(1-blend,blend,float3(offset));
            uint address=level.y+face*level.z+uint(cell.x)+level.x*(uint(cell.y)+level.x*uint(cell.z));
            result+=volume.read(uint2(address%frame.Atlas.x,address/frame.Atlas.x))*(axisWeight[axis]*weight.x*weight.y*weight.z);
        }
    }
    return result;
}
float4 MipRadiance(float3 position,float3 outward,float diameter,constant VoxelFrame& frame,
    texture2d<float,access::read> volume)
{
    float lod=clamp(log2(diameter/frame.GridOriginCell.w),0.0f,float(frame.Options.z-1)); uint low=uint(lod);
    float4 first=SpatialRadiance(position,outward,low,frame,volume);
    float4 second=SpatialRadiance(position,outward,min(low+1,frame.Options.z-1),frame,volume);
    return mix(first,second,fract(lod));
}
float3 IntegrateCone(float3 origin,float3 direction,constant VoxelFrame& frame,texture2d<float,access::read> volume)
{
    float cell=frame.GridOriginCell.w,side=cell*frame.Options.y,distance=0,through=1; float3 result=0;
    for (uint step=0;step<frame.Atlas.z && distance<frame.Trace.x && through>TransmissionCutoff;++step)
    {
        float diameter=clamp(2*frame.Trace.y*(distance+cell*PixelCenter),cell,side);
        float segment=min(diameter*frame.Trace.z,frame.Trace.x-distance);
        float3 position=origin+direction*(distance+segment*PixelCenter),relative=(position-frame.GridOriginCell.xyz)/side;
        if (any(relative<0) || any(relative>1)) break;
        float4 value=MipRadiance(position,-direction,diameter,frame,volume); float opacity=saturate(value.a);
        float adjusted=1-pow(1-opacity,segment/diameter);
        if (opacity>MinimumWeight) result+=value.rgb*(through*adjusted/opacity);
        through*=1-adjusted; distance+=segment;
    }
    return result;
}
float3 DiffuseHemisphere(float3 position,float3 normal,constant VoxelFrame& frame,texture2d<float,access::read> volume)
{
    float3 helper=abs(normal.y)<ParallelLimit ? float3(0,1,0):float3(1,0,0);
    float3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent),sum=0;
    for (uint cone=0;cone<frame.Options.w;++cone)
    {
        float radialSquared=(cone+PixelCenter)/frame.Options.w,angle=GoldenAngle*cone;
        float3 direction=tangent*(sqrt(radialSquared)*cos(angle))+bitangent*(sqrt(radialSquared)*sin(angle))+normal*sqrt(1-radialSquared);
        sum+=IntegrateCone(position+normal*(frame.GridOriginCell.w*frame.Trace.w),direction,frame,volume);
    }
    return sum/frame.Options.w;
}
float3 Reconstruct(int2 pixel,float3 position,float3 normal,constant VoxelFrame& frame,
    texture2d<float,access::read> depth,texture2d<float,access::read> normals,texture2d<float,access::read> previous)
{
    float3 sum=0; float mass=0,sigma=max(frame.GridOriginCell.w*frame.Filter.x,MinimumWeight);
    for (int shift=-FilterRadius;shift<=FilterRadius;++shift)
    {
        int2 neighbor=pixel+(frame.Options.x==HorizontalPass ? int2(shift,0):int2(0,shift)); float3 location,n;
        if (!Receiver(neighbor,frame,depth,normals,location,n)) continue;
        float4 value=previous.read(uint2(neighbor)); if (value.a<=0) continue;
        float delta=dot(location-position,frame.CameraForward.xyz)/sigma;
        float angular=frame.Filter.y>0 ? pow(saturate(dot(n,normal)),frame.Filter.y):1;
        float weight=exp(-(float(shift*shift)+delta*delta)/2)*angular;
        sum+=value.rgb*weight; mass+=weight;
    }
    return mass>MinimumWeight ? sum/mass:previous.read(uint2(pixel)).rgb;
}
fragment float4 PSMain(Fullscreen input [[stage_in]],constant VoxelFrame& frame [[buffer(0)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
    texture2d<float,access::read> diffuse [[texture(2)]],texture2d<float,access::read> ambient [[texture(3)]],
    texture2d<float,access::read> volume [[texture(4)]],texture2d<float,access::read> previous [[texture(5)]],
    texture2d<float,access::read> scene [[texture(6)]])
{
    int2 pixel=int2(input.Position.xy);
    if (frame.Options.x==CompositePass) return float4(previous.read(uint2(pixel)).rgb,0);
    float3 position,normal; if (!Receiver(pixel,frame,depth,normals,position,normal)) return 0;
    float3 relative=(position-frame.GridOriginCell.xyz)/(frame.GridOriginCell.w*frame.Options.y);
    if (any(relative<0) || any(relative>1)) return 0;
    float3 incoming=frame.Options.x==TracePass ? DiffuseHemisphere(position,normal,frame,volume):Reconstruct(pixel,position,normal,frame,depth,normals,previous);
    if (frame.Options.x==VerticalPass)
    {
        float materialScale=frame.Atlas.w==PbrModel ? (1-DielectricF0)*(1-saturate(ambient.read(uint2(pixel)).w)):Pi;
        incoming=max(incoming*max(diffuse.read(uint2(pixel)).rgb,0.0f)*(materialScale*frame.Filter.z),0.0f);
        incoming=min(incoming,max(float3(HalfMaximum-HalfHeadroom)-scene.read(uint2(pixel)).rgb,0.0f));
    }
    return float4(incoming,1);
}
