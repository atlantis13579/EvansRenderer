#include <metal_stdlib>
using namespace metal;
// Native Metal L2 irradiance gather with octahedral distance-moment visibility.
// Eight grid neighbors carry spatial and normal-facing weights. Camera motion
// changes reconstruction only, and never changes the baked lighting field.
// https://jcgt.org/published/0008/02/01/ ; https://graphics.stanford.edu/papers/envmap/
struct ProbeUniforms
{
    float4 InverseViewProjection[4];
    float4 GridMinimum,GridMaximum,Viewport,Appearance,Visibility;
    uint4 Dimensions,Options;
};
constant float Pi=3.14159265358979323846,CoefficientDecodeScale=4;
constant float HomogeneousEpsilon=1e-7,VectorSquaredEpsilon=1e-12,WeightEpsilon=1e-6;
constant float MinimumFacing=0.05,DielectricF0=0.04,HalfFloatMaximum=65504,PixelCenter=0.5;
constant uint CoefficientCount=9,PBRModel=3;
struct ProbeVertex {float4 Position [[position]];};
vertex ProbeVertex VSMain(uint vertexId [[vertex_id]])
{
    float2 uv=float2((vertexId<<1)&2,vertexId&2);
    return {float4(uv*2-1,0,1)};
}
float SHValue(uint coefficient,float3 n)
{
    if(coefficient==0)return sqrt(1.0f/(4*Pi));
    if(coefficient==1)return n.y*sqrt(3.0f/(4*Pi));
    if(coefficient==2)return n.z*sqrt(3.0f/(4*Pi));
    if(coefficient==3)return n.x*sqrt(3.0f/(4*Pi));
    if(coefficient==4)return n.x*n.y*sqrt(15.0f/(4*Pi));
    if(coefficient==5)return n.y*n.z*sqrt(15.0f/(4*Pi));
    if(coefficient==6)return (3*n.z*n.z-1)*sqrt(5.0f/(16*Pi));
    if(coefficient==7)return n.x*n.z*sqrt(15.0f/(4*Pi));
    return (n.x*n.x-n.y*n.y)*sqrt(15.0f/(16*Pi));
}
int2 FoldOctCell(int2 cell,int size)
{
    if(cell.x<0){cell.x=-cell.x-1;cell.y=size-1-cell.y;}
    if(cell.x>=size){cell.x=2*size-1-cell.x;cell.y=size-1-cell.y;}
    if(cell.y<0){cell.y=-cell.y-1;cell.x=size-1-cell.x;}
    if(cell.y>=size){cell.y=2*size-1-cell.y;cell.x=size-1-cell.x;}
    return cell;
}
float2 MomentSample(texture2d<float,access::read> texture,uint index,float3 n,constant ProbeUniforms& uniforms)
{
    n/=dot(abs(n),float3(1));float2 oct=n.xy;
    if(n.z<0)oct=(1-abs(oct.yx))*float2(oct.x>=0?1:-1,oct.y>=0?1:-1);
    float2 pixel=(oct*0.5+0.5)*float(uniforms.Options.z)-PixelCenter;
    int2 first=int2(floor(pixel));float2 fraction=fract(pixel);
    uint columns=uint(uniforms.Visibility.w);
    int2 origin=int2(index%columns,index/columns)*int(uniforms.Options.z);
    float2 sum(0);
    for(int y=0;y<2;++y)for(int x=0;x<2;++x)
    {
        float weight=(x==0?1-fraction.x:fraction.x)*(y==0?1-fraction.y:fraction.y);
        sum+=texture.read(uint2(origin+FoldOctCell(first+int2(x,y),int(uniforms.Options.z)))).xy*weight;
    }
    return sum;
}
fragment float4 PSMain(ProbeVertex raster [[stage_in]],constant ProbeUniforms& uniforms [[buffer(0)]],
    texture2d<float,access::read> coefficients [[texture(1)]],texture2d<float,access::read> moments [[texture(2)]],
    texture2d<float,access::read> color [[texture(3)]],depth2d<float,access::read> depth [[texture(4)]],
    texture2d<float,access::read> normalRoughness [[texture(5)]],texture2d<float,access::read> diffuseOpacity [[texture(6)]],
    texture2d<float,access::read> ambientMetallic [[texture(7)]])
{
    uint2 pixel(raster.Position.xy);
    float4 scene=color.read(pixel),material=normalRoughness.read(pixel);
    float z=depth.read(pixel);
    if(z>=1 || material.w<0 || !all(isfinite(material)) || dot(material.xyz,material.xyz)<=VectorSquaredEpsilon)return scene;
    float2 uv=raster.Position.xy*uniforms.Viewport.zw;
    float4 clip(uv.x*2-1,1-uv.y*2,z,1);
    float4 h(dot(uniforms.InverseViewProjection[0],clip),dot(uniforms.InverseViewProjection[1],clip),
        dot(uniforms.InverseViewProjection[2],clip),dot(uniforms.InverseViewProjection[3],clip));
    if(!all(isfinite(h)) || abs(h.w)<=HomogeneousEpsilon)return scene;
    float3 world=h.xyz/h.w,normal=normalize(material.xyz);
    float3 grid=(world-uniforms.GridMinimum.xyz)/(uniforms.GridMaximum.xyz-uniforms.GridMinimum.xyz)*float3(uniforms.Dimensions.xyz-1);
    if(any(grid<0) || any(grid>float3(uniforms.Dimensions.xyz-1)))return scene;
    uint3 base=min(uint3(floor(grid)),uniforms.Dimensions.xyz-2);
    float3 fraction=grid-float3(base),biased=world+normal*uniforms.Appearance.y;
    float3 sum(0);float weightSum=0;
    for(uint zIndex=0;zIndex<2;++zIndex)for(uint y=0;y<2;++y)for(uint x=0;x<2;++x)
    {
        uint3 cell=base+uint3(x,y,zIndex);
        uint index=cell.x+uniforms.Dimensions.x*(cell.y+uniforms.Dimensions.y*cell.z);
        if(coefficients.read(uint2(0,index)).a<=0)continue;
        float3 position=mix(uniforms.GridMinimum.xyz,uniforms.GridMaximum.xyz,float3(cell)/float3(uniforms.Dimensions.xyz-1));
        float3 fromProbe=biased-position;float distance=length(fromProbe);
        float2 moment=MomentSample(moments,index,distance>HomogeneousEpsilon?fromProbe/distance:normal,uniforms);
        float delta=max((distance-uniforms.Appearance.z)/uniforms.Visibility.z-moment.x,0.0f);
        float variance=max(moment.y-moment.x*moment.x,uniforms.Visibility.x);
        float visible=pow(saturate(variance/max(variance+delta*delta,VectorSquaredEpsilon)),uniforms.Appearance.w);
        float3 toProbe=position-world;
        float facing=dot(toProbe,toProbe)>VectorSquaredEpsilon?dot(normal,normalize(toProbe)):1;
        float wrap=max((facing+1)*0.5f,MinimumFacing);
        float3 spatial(x==0?1-fraction.x:fraction.x,y==0?1-fraction.y:fraction.y,zIndex==0?1-fraction.z:fraction.z);
        float weight=spatial.x*spatial.y*spatial.z*wrap*wrap*visible;
        float3 value(0);
        for(uint coefficient=0;coefficient<CoefficientCount;++coefficient)
            value+=coefficients.read(uint2(coefficient,index)).rgb*SHValue(coefficient,normal);
        sum+=max(value*CoefficientDecodeScale,float3(0))*weight;weightSum+=weight;
    }
    if(weightSum<=WeightEpsilon)return scene;
    float3 diffuse=saturate(diffuseOpacity.read(pixel).rgb);
    if(uniforms.Options.y==PBRModel)diffuse*=(1-DielectricF0)*(1-saturate(ambientMetallic.read(pixel).w));
    return float4(clamp(scene.rgb+sum/weightSum*diffuse*uniforms.Appearance.x,0.0f,HalfFloatMaximum),scene.a);
}
