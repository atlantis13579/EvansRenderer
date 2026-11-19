#version 310 es
precision highp float;
precision highp int;
// ES cone tracing uses six directional, premultiplied radiance hierarchies in
// an integer-addressed 2D atlas. World voxels include off-screen scene geometry.
// Crassin et al., Interactive Indirect Illumination Using Voxel Cone Tracing (2011):
// https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
const int MipLimit=7,SpatialCorners=8,AxisCount=3,FilterRadius=2;
const uint ConePhase=0u,HorizontalPhase=1u,VerticalPhase=2u,CompositePhase=3u,PbrModel=3u;
const float Pi=3.141592653589793,GoldenAngle=Pi*(3.0-sqrt(5.0));
const float TexelCenter=0.5,EmptyDepth=1.0,ProjectionEpsilon=1e-7,NormalEpsilon=1e-12;
const float WeightEpsilon=1e-6,TransmissionCutoff=0.01,AxisParallelLimit=0.9;
const float HalfMaximum=65504.0,HalfHeadroom=16.0,DielectricF0=0.04;
layout(std140,binding=0) uniform VoxelFrame
{
    vec4 ClipToWorldRows[4];
    vec4 Viewport,OriginCell,March,Rejection,CameraForward;
    uvec4 Options,Atlas;
    uvec4 Mips[MipLimit]; // Cubic dimension, linear offset, face stride, reserved.
};
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
uniform highp sampler2D CameraDepth,NormalMaterial,DiffuseMaterial,AmbientMaterial;
uniform highp sampler2D DirectionalRadiance,PreviousSignal,SceneRadiance;
layout(location=0) out vec4 OutColor;
bool Receiver(ivec2 pixel,out vec3 position,out vec3 normal)
{
    position=normal=vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Viewport.xy)))) return false;
    float depth=texelFetch(CameraDepth,pixel,0).r; vec4 n=texelFetch(NormalMaterial,pixel,0);
    float squared=dot(n.xyz,n.xyz);
    if (!(depth<EmptyDepth && n.w>=0.0 && squared>NormalEpsilon) || isnan(squared) || isinf(squared)) return false;
    // ES images and gl_FragCoord share bottom-left storage. Camera depth is
    // already canonical [0,1]; only raster clip depth uses the GL conversion.
    vec2 uv=(vec2(pixel)+TexelCenter)*Viewport.zw;
    vec4 clip=vec4(uv*2.0-1.0,depth,1.0);
    vec4 h=vec4(dot(ClipToWorldRows[0],clip),dot(ClipToWorldRows[1],clip),dot(ClipToWorldRows[2],clip),dot(ClipToWorldRows[3],clip));
    if (abs(h.w)<ProjectionEpsilon) return false;
    position=h.xyz/h.w; normal=n.xyz*inversesqrt(squared);
    return !any(isnan(position)) && !any(isinf(position));
}
vec4 DirectionalLevel(vec3 position,vec3 outgoing,uint mip)
{
    uvec4 level=Mips[mip];
    vec3 local=(position-OriginCell.xyz)/(OriginCell.w*exp2(float(mip)))-TexelCenter;
    ivec3 low=ivec3(floor(local)); vec3 blend=fract(local),weights=abs(outgoing);
    weights/=max(weights.x+weights.y+weights.z,WeightEpsilon); vec4 sum=vec4(0.0);
    for (int axis=0;axis<AxisCount;++axis)
    {
        uint face=uint(axis*2)+(outgoing[axis]<0.0 ? 1u:0u);
        for (int corner=0;corner<SpatialCorners;++corner)
        {
            ivec3 offset=ivec3(corner&1,(corner>>1)&1,(corner>>2)&1),cell=low+offset;
            if (any(lessThan(cell,ivec3(0))) || any(greaterThanEqual(cell,ivec3(level.x)))) continue;
            vec3 spatial=mix(vec3(1.0)-blend,blend,vec3(offset));
            uint address=level.y+face*level.z+uint(cell.x)+level.x*(uint(cell.y)+level.x*uint(cell.z));
            sum+=texelFetch(DirectionalRadiance,ivec2(address%Atlas.x,address/Atlas.x),0)*(weights[axis]*spatial.x*spatial.y*spatial.z);
        }
    }
    return sum;
}
vec4 ReadRadiance(vec3 position,vec3 outgoing,float diameter)
{
    float lod=clamp(log2(diameter/OriginCell.w),0.0,float(Options.z-1u)); uint low=uint(lod);
    return mix(DirectionalLevel(position,outgoing,low),DirectionalLevel(position,outgoing,min(low+1u,Options.z-1u)),fract(lod));
}
vec3 TraceCone(vec3 origin,vec3 direction)
{
    float cell=OriginCell.w,side=cell*float(Options.y),distance=0.0,through=1.0; vec3 sum=vec3(0.0);
    for (uint step=0u;step<Atlas.z && distance<March.x && through>TransmissionCutoff;++step)
    {
        float diameter=clamp(2.0*March.y*(distance+cell*TexelCenter),cell,side);
        float segment=min(diameter*March.z,March.x-distance);
        vec3 location=origin+direction*(distance+segment*TexelCenter),relative=(location-OriginCell.xyz)/side;
        if (any(lessThan(relative,vec3(0.0))) || any(greaterThan(relative,vec3(1.0)))) break;
        vec4 value=ReadRadiance(location,-direction,diameter); float opacity=clamp(value.a,0.0,1.0);
        float corrected=1.0-pow(1.0-opacity,segment/diameter);
        if (opacity>WeightEpsilon) sum+=value.rgb*(through*corrected/opacity);
        through*=1.0-corrected; distance+=segment;
    }
    return sum;
}
vec3 Hemisphere(vec3 position,vec3 normal)
{
    vec3 axis=abs(normal.y)<AxisParallelLimit ? vec3(0,1,0):vec3(1,0,0);
    vec3 tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent),sum=vec3(0.0);
    for (uint cone=0u;cone<Options.w;++cone)
    {
        float radiusSquared=(float(cone)+TexelCenter)/float(Options.w),azimuth=GoldenAngle*float(cone);
        vec3 direction=tangent*(sqrt(radiusSquared)*cos(azimuth))+bitangent*(sqrt(radiusSquared)*sin(azimuth))+normal*sqrt(1.0-radiusSquared);
        sum+=TraceCone(position+normal*(OriginCell.w*March.w),direction);
    }
    return sum/float(Options.w); // Cosine-weighted estimate of irradiance / pi.
}
vec3 FilterSignal(ivec2 pixel,vec3 position,vec3 normal)
{
    vec3 sum=vec3(0.0); float mass=0.0,sigma=max(OriginCell.w*Rejection.x,WeightEpsilon);
    for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        ivec2 at=pixel+(Options.x==HorizontalPhase ? ivec2(offset,0):ivec2(0,offset)); vec3 location,n;
        if (!Receiver(at,location,n)) continue;
        vec4 signal=texelFetch(PreviousSignal,at,0); if (signal.a<=0.0) continue;
        float optical=dot(location-position,CameraForward.xyz)/sigma;
        float angular=Rejection.y>0.0 ? pow(clamp(dot(n,normal),0.0,1.0),Rejection.y):1.0;
        float weight=exp(-(float(offset*offset)+optical*optical)/2.0)*angular;
        sum+=signal.rgb*weight; mass+=weight;
    }
    return mass>WeightEpsilon ? sum/mass:texelFetch(PreviousSignal,pixel,0).rgb;
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (Options.x==CompositePhase) { OutColor=vec4(texelFetch(PreviousSignal,pixel,0).rgb,0.0); return; }
    vec3 position,normal; if (!Receiver(pixel,position,normal)) { OutColor=vec4(0.0); return; }
    vec3 relative=(position-OriginCell.xyz)/(OriginCell.w*float(Options.y));
    if (any(lessThan(relative,vec3(0.0))) || any(greaterThan(relative,vec3(1.0)))) { OutColor=vec4(0.0); return; }
    vec3 signal=Options.x==ConePhase ? Hemisphere(position,normal):FilterSignal(pixel,position,normal);
    if (Options.x==VerticalPhase)
    {
        float diffuseScale=Atlas.w==PbrModel ? (1.0-DielectricF0)*(1.0-clamp(texelFetch(AmbientMaterial,pixel,0).w,0.0,1.0)):Pi;
        signal=max(signal*max(texelFetch(DiffuseMaterial,pixel,0).rgb,vec3(0.0))*(diffuseScale*Rejection.z),vec3(0.0));
        signal=min(signal,max(vec3(HalfMaximum-HalfHeadroom)-texelFetch(SceneRadiance,pixel,0).rgb,vec3(0.0)));
    }
    OutColor=vec4(signal,1.0);
}
#endif
