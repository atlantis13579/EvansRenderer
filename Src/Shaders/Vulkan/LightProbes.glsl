#version 450
// Vulkan-native probe interpolation. Texture/raster scene coordinates are
// top-left; inverse projection reconstructs the original +Y-up camera frame.
// Data atlas coordinates have no clip-space conversion. Octahedral edge reads
// are folded locally so adjacent probes never bleed through sampler filtering.
// https://jcgt.org/published/0008/02/01/ ; https://graphics.stanford.edu/papers/envmap/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2,gl_VertexIndex & 2);
    gl_Position = vec4(uv*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=0,std140,row_major) uniform ProbeUniforms
{
    mat4 InverseViewProjection;
    vec4 GridMinimum,GridMaximum,Viewport,Appearance,Visibility;
    uvec4 Dimensions,Options;
} Probe;
layout(set=0,binding=2) uniform texture2D IrradianceSH;
layout(set=0,binding=3) uniform texture2D DepthMoments;
layout(set=0,binding=4) uniform texture2D OpaqueRadiance;
layout(set=0,binding=5) uniform texture2D OpaqueDepth;
layout(set=0,binding=6) uniform texture2D NormalRoughness;
layout(set=0,binding=7) uniform texture2D DiffuseOpacity;
layout(set=0,binding=8) uniform texture2D AmbientMetallic;
layout(set=0,binding=9) uniform sampler NearestSampler;
layout(location=0) out vec4 OutColor;
const float Pi = 3.14159265358979323846, SHDecodeScale = 4.0;
const float MinimumW = 1e-7, SquaredLengthFloor = 1e-12, WeightSumFloor = 1e-6;
const float FacingFloor = 0.05, DielectricReflectance = 0.04, Binary16Maximum = 65504.0, TexelCenter = 0.5;
const uint MetallicModel = 3u;
const int SHCount = 9;
bool Finite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
float EvaluateBasis(int index,vec3 n)
{
    switch(index)
    {
    case 0:return sqrt(1.0/(4.0*Pi));
    case 1:return n.y*sqrt(3.0/(4.0*Pi));
    case 2:return n.z*sqrt(3.0/(4.0*Pi));
    case 3:return n.x*sqrt(3.0/(4.0*Pi));
    case 4:return n.x*n.y*sqrt(15.0/(4.0*Pi));
    case 5:return n.y*n.z*sqrt(15.0/(4.0*Pi));
    case 6:return (3.0*n.z*n.z-1.0)*sqrt(5.0/(16.0*Pi));
    case 7:return n.x*n.z*sqrt(15.0/(4.0*Pi));
    default:return (n.x*n.x-n.y*n.y)*sqrt(15.0/(16.0*Pi));
    }
}
ivec2 FoldCell(ivec2 p)
{
    int side = int(Probe.Options.z);
    if(p.x<0){p.x=-1-p.x;p.y=side-1-p.y;}
    if(p.x>=side){p.x=side*2-1-p.x;p.y=side-1-p.y;}
    if(p.y<0){p.y=-1-p.y;p.x=side-1-p.x;}
    if(p.y>=side){p.y=side*2-1-p.y;p.x=side-1-p.x;}
    return p;
}
vec2 MomentLookup(uint index,vec3 omega)
{
    omega /= dot(abs(omega),vec3(1));
    vec2 oct=omega.xy;
    if(omega.z<0) oct=(1.0-abs(oct.yx))*vec2(oct.x>=0?1:-1,oct.y>=0?1:-1);
    vec2 pixel=(oct*0.5+0.5)*float(Probe.Options.z)-TexelCenter;
    ivec2 first=ivec2(floor(pixel));vec2 alpha=fract(pixel);
    uint columns=uint(Probe.Visibility.w);
    ivec2 origin=ivec2(index%columns,index/columns)*int(Probe.Options.z);
    vec2 value=vec2(0);
    for(int corner=0;corner<4;++corner)
    {
        ivec2 offset=ivec2(corner&1,corner>>1);
        float weight=(offset.x==0?1.0-alpha.x:alpha.x)*(offset.y==0?1.0-alpha.y:alpha.y);
        value+=texelFetch(sampler2D(DepthMoments,NearestSampler),origin+FoldCell(first+offset),0).xy*weight;
    }
    return value;
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    vec4 scene=texelFetch(sampler2D(OpaqueRadiance,NearestSampler),pixel,0);OutColor=scene;
    vec4 material=texelFetch(sampler2D(NormalRoughness,NearestSampler),pixel,0);
    float depth=texelFetch(sampler2D(OpaqueDepth,NearestSampler),pixel,0).r;
    if(depth>=1.0 || material.w<0.0 || !Finite(material) || dot(material.xyz,material.xyz)<=SquaredLengthFloor)return;
    vec2 uv=gl_FragCoord.xy*Probe.Viewport.zw;
    vec4 h=Probe.InverseViewProjection*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1);
    if(!Finite(h) || abs(h.w)<=MinimumW)return;
    vec3 world=h.xyz/h.w,normal=normalize(material.xyz);
    vec3 grid=(world-Probe.GridMinimum.xyz)/(Probe.GridMaximum.xyz-Probe.GridMinimum.xyz)*vec3(Probe.Dimensions.xyz-1u);
    if(any(lessThan(grid,vec3(0))) || any(greaterThan(grid,vec3(Probe.Dimensions.xyz-1u))))return;
    uvec3 base=min(uvec3(floor(grid)),Probe.Dimensions.xyz-2u);
    vec3 alpha=grid-vec3(base),receiver=world+normal*Probe.Appearance.y;
    vec3 sum=vec3(0);float weightSum=0;
    for(uint neighbor=0u;neighbor<8u;++neighbor)
    {
        uvec3 offset=uvec3(neighbor&1u,(neighbor>>1u)&1u,neighbor>>2u),cell=base+offset;
        uint index=cell.x+Probe.Dimensions.x*(cell.y+Probe.Dimensions.y*cell.z);
        if(texelFetch(sampler2D(IrradianceSH,NearestSampler),ivec2(0,int(index)),0).w<=0)continue;
        vec3 position=mix(Probe.GridMinimum.xyz,Probe.GridMaximum.xyz,vec3(cell)/vec3(Probe.Dimensions.xyz-1u));
        vec3 fromProbe=receiver-position;float distance=length(fromProbe);
        vec2 moments=MomentLookup(index,distance>MinimumW?fromProbe/distance:normal);
        float delta=max((distance-Probe.Appearance.z)/Probe.Visibility.z-moments.x,0.0);
        float variance=max(moments.y-moments.x*moments.x,Probe.Visibility.x);
        float visible=pow(clamp(variance/max(variance+delta*delta,SquaredLengthFloor),0.0,1.0),Probe.Appearance.w);
        vec3 toProbe=position-world;
        float facing=dot(toProbe,toProbe)>SquaredLengthFloor?dot(normal,normalize(toProbe)):1.0;
        float wrap=max((facing+1.0)*0.5,FacingFloor);
        vec3 blend=vec3(offset.x==0u?1.0-alpha.x:alpha.x,offset.y==0u?1.0-alpha.y:alpha.y,offset.z==0u?1.0-alpha.z:alpha.z);
        float weight=blend.x*blend.y*blend.z*wrap*wrap*visible;
        vec3 value=vec3(0);
        for(int coefficient=0;coefficient<SHCount;++coefficient)
            value+=texelFetch(sampler2D(IrradianceSH,NearestSampler),ivec2(coefficient,int(index)),0).rgb*EvaluateBasis(coefficient,normal);
        sum+=max(value*SHDecodeScale,vec3(0))*weight;weightSum+=weight;
    }
    if(weightSum<=WeightSumFloor)return;
    vec3 diffuse=clamp(texelFetch(sampler2D(DiffuseOpacity,NearestSampler),pixel,0).rgb,0.0,1.0);
    if(Probe.Options.y==MetallicModel) diffuse*=(1.0-DielectricReflectance)*(1.0-clamp(texelFetch(sampler2D(AmbientMetallic,NearestSampler),pixel,0).w,0.0,1.0));
    OutColor=vec4(clamp(scene.rgb+sum/weightSum*diffuse*Probe.Appearance.x,0.0,Binary16Maximum),scene.a);
}
#endif
