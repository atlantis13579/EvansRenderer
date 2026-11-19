#version 450
// Runtime-compiled Vulkan GLSL. Directional voxel opacity is composed along
// each face before transverse averaging; radiance is stored premultiplied.
// Crassin et al. (2011), Interactive Indirect Illumination Using Voxel Cone Tracing:
// https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
const int HierarchyCapacity=7,FilterSupport=2;
const uint TraceStage=0u,HorizontalStage=1u,VerticalStage=2u,CompositeStage=3u,PhysicalModel=3u;
const float Pi=3.141592653589793,SpiralAngle=Pi*(3.0-sqrt(5.0));
const float CenterOffset=0.5,ClearDepth=1.0,MinimumW=1e-7,NormalTolerance=1e-12;
const float MinimumWeight=1e-6,StopTransmission=0.01,ParallelThreshold=0.9;
const float HalfMaximum=65504.0,HalfGuard=16.0,DielectricReflectance=0.04;
layout(set=0,binding=0,std140,row_major) uniform VoxelParameters
{
    mat4 WorldFromClip;
    vec4 Viewport,GridOriginCell,Trace,Reconstruction,CameraForward;
    uvec4 Control,Atlas;
    uvec4 Hierarchy[HierarchyCapacity]; // Dimension, offset, face stride, reserved.
} Frame;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 uv=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(uv*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=1) uniform sampler2D SceneDepth;
layout(set=0,binding=2) uniform sampler2D SceneNormal;
layout(set=0,binding=3) uniform sampler2D SceneDiffuse;
layout(set=0,binding=4) uniform sampler2D SceneAmbient;
layout(set=0,binding=5) uniform sampler2D RadianceAtlas;
layout(set=0,binding=6) uniform sampler2D PreviousRadiance;
layout(set=0,binding=7) uniform sampler2D OpaqueColor;
layout(location=0) out vec4 OutColor;
bool ReadSurface(ivec2 pixel,out vec3 world,out vec3 normal)
{
    world=normal=vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Frame.Viewport.xy)))) return false;
    float depth=texelFetch(SceneDepth,pixel,0).r; vec4 n=texelFetch(SceneNormal,pixel,0); float squared=dot(n.xyz,n.xyz);
    if (!(depth<ClearDepth && n.w>=0.0 && squared>NormalTolerance) || isnan(squared) || isinf(squared)) return false;
    vec2 uv=(vec2(pixel)+CenterOffset)*Frame.Viewport.zw;
    vec4 h=Frame.WorldFromClip*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (abs(h.w)<MinimumW) return false;
    world=h.xyz/h.w; normal=n.xyz*inversesqrt(squared); return !any(isnan(world)) && !any(isinf(world));
}
vec4 SampleLevel(vec3 world,vec3 outgoing,uint index)
{
    uvec4 level=Frame.Hierarchy[index];
    vec3 coordinate=(world-Frame.GridOriginCell.xyz)/(Frame.GridOriginCell.w*exp2(float(index)))-CenterOffset;
    ivec3 first=ivec3(floor(coordinate)); vec3 fraction=fract(coordinate),mass=abs(outgoing);
    mass/=max(mass.x+mass.y+mass.z,MinimumWeight); vec4 result=vec4(0.0);
    for (int z=0;z<2;++z) for (int y=0;y<2;++y) for (int x=0;x<2;++x)
    {
        ivec3 cell=first+ivec3(x,y,z);
        if (any(lessThan(cell,ivec3(0))) || any(greaterThanEqual(cell,ivec3(level.x)))) continue;
        vec3 weight=mix(vec3(1.0)-fraction,fraction,vec3(x,y,z));
        uint voxel=uint(cell.x)+level.x*(uint(cell.y)+level.x*uint(cell.z));
        for (uint axis=0u;axis<3u;++axis)
        {
            uint face=axis*2u+(outgoing[axis]<0.0 ? 1u:0u),address=level.y+face*level.z+voxel;
            result+=texelFetch(RadianceAtlas,ivec2(address%Frame.Atlas.x,address/Frame.Atlas.x),0)*(mass[axis]*weight.x*weight.y*weight.z);
        }
    }
    return result;
}
vec4 SampleHierarchy(vec3 world,vec3 outgoing,float diameter)
{
    float lod=clamp(log2(diameter/Frame.GridOriginCell.w),0.0,float(Frame.Control.z-1u)); uint lower=uint(lod);
    return mix(SampleLevel(world,outgoing,lower),SampleLevel(world,outgoing,min(lower+1u,Frame.Control.z-1u)),fract(lod));
}
vec3 ConeRadiance(vec3 origin,vec3 direction)
{
    float cell=Frame.GridOriginCell.w,side=cell*float(Frame.Control.y),distance=0.0,transmission=1.0; vec3 result=vec3(0.0);
    for (uint step=0u;step<Frame.Atlas.z && distance<Frame.Trace.x && transmission>StopTransmission;++step)
    {
        float diameter=clamp(2.0*Frame.Trace.y*(distance+cell*CenterOffset),cell,side);
        float segment=min(diameter*Frame.Trace.z,Frame.Trace.x-distance);
        vec3 location=origin+direction*(distance+segment*CenterOffset),relative=(location-Frame.GridOriginCell.xyz)/side;
        if (any(lessThan(relative,vec3(0.0))) || any(greaterThan(relative,vec3(1.0)))) break;
        vec4 value=SampleHierarchy(location,-direction,diameter); float opacity=clamp(value.a,0.0,1.0);
        float corrected=1.0-pow(1.0-opacity,segment/diameter);
        if (opacity>MinimumWeight) result+=value.rgb*(transmission*corrected/opacity);
        transmission*=1.0-corrected; distance+=segment;
    }
    return result;
}
vec3 DiffuseGather(vec3 world,vec3 normal)
{
    vec3 reference=abs(normal.y)<ParallelThreshold ? vec3(0,1,0):vec3(1,0,0);
    vec3 tangent=normalize(cross(reference,normal)),bitangent=cross(normal,tangent),result=vec3(0.0);
    for (uint cone=0u;cone<Frame.Control.w;++cone)
    {
        float radialSquared=(float(cone)+CenterOffset)/float(Frame.Control.w),angle=SpiralAngle*float(cone);
        vec3 direction=tangent*(sqrt(radialSquared)*cos(angle))+bitangent*(sqrt(radialSquared)*sin(angle))+normal*sqrt(1.0-radialSquared);
        result+=ConeRadiance(world+normal*(Frame.GridOriginCell.w*Frame.Trace.w),direction);
    }
    return result/float(Frame.Control.w);
}
vec3 GuidedFilter(ivec2 pixel,vec3 world,vec3 normal)
{
    vec3 sum=vec3(0.0); float total=0.0,sigma=max(Frame.GridOriginCell.w*Frame.Reconstruction.x,MinimumWeight);
    for (int offset=-FilterSupport;offset<=FilterSupport;++offset)
    {
        ivec2 neighbor=pixel+(Frame.Control.x==HorizontalStage ? ivec2(offset,0):ivec2(0,offset)); vec3 location,n;
        if (!ReadSurface(neighbor,location,n)) continue;
        vec4 signal=texelFetch(PreviousRadiance,neighbor,0); if (signal.a<=0.0) continue;
        float depthDifference=dot(location-world,Frame.CameraForward.xyz)/sigma;
        float angular=Frame.Reconstruction.y>0.0 ? pow(clamp(dot(normal,n),0.0,1.0),Frame.Reconstruction.y):1.0;
        float weight=exp(-(float(offset*offset)+depthDifference*depthDifference)/2.0)*angular;
        sum+=signal.rgb*weight; total+=weight;
    }
    return total>MinimumWeight ? sum/total:texelFetch(PreviousRadiance,pixel,0).rgb;
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (Frame.Control.x==CompositeStage) { OutColor=vec4(texelFetch(PreviousRadiance,pixel,0).rgb,0.0); return; }
    vec3 world,normal; if (!ReadSurface(pixel,world,normal)) { OutColor=vec4(0.0); return; }
    vec3 relative=(world-Frame.GridOriginCell.xyz)/(Frame.GridOriginCell.w*float(Frame.Control.y));
    if (any(lessThan(relative,vec3(0.0))) || any(greaterThan(relative,vec3(1.0)))) { OutColor=vec4(0.0); return; }
    vec3 incoming=Frame.Control.x==TraceStage ? DiffuseGather(world,normal):GuidedFilter(pixel,world,normal);
    if (Frame.Control.x==VerticalStage)
    {
        float transfer=Frame.Atlas.w==PhysicalModel ? (1.0-DielectricReflectance)*(1.0-clamp(texelFetch(SceneAmbient,pixel,0).w,0.0,1.0)):Pi;
        incoming=max(incoming*max(texelFetch(SceneDiffuse,pixel,0).rgb,vec3(0.0))*(transfer*Frame.Reconstruction.z),vec3(0.0));
        incoming=min(incoming,max(vec3(HalfMaximum-HalfGuard)-texelFetch(OpaqueColor,pixel,0).rgb,vec3(0.0)));
    }
    OutColor=vec4(incoming,1.0);
}
#endif
