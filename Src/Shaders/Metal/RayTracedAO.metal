#include <metal_stdlib>
using namespace metal;
// Metal scene-ray visibility from immutable device buffers. Ray hits evaluate
// original UV/vertex/material coverage; rejected cutouts do not stop traversal.
// Ray-aligned triangles: Woop et al., JCGT 2013,
// https://jcgt.org/published/0002/01/05/ . Inclusive FP32 error bands replace
// double fallback and are conservative rather than exact predicates.
constant uint EstimatePass=0,HorizontalPass=1,VerticalPass=2,CompositePass=3;
constant float Pi=3.141592653589793;
constant float PixelCenter=0.5,EmptyDepth=1,MinimumW=1e-7,MinimumNormalSquared=1e-12;
constant float Binary32Epsilon=1.1920928955078125e-7;
constant float EdgeError=3*Binary32Epsilon/(1-3*Binary32Epsilon);
constant int FilterRadius=2;
constant float SpatialSigma=1.5,DepthSigmaFraction=0.25,NormalExponent=8,MinimumWeight=1e-7;
struct RayAOConstants
{
    float4 ViewProjection[4],InverseViewProjection[4];
    float4 ViewportRadiusBias,ForwardIntensity;
    uint4 Options,SceneCounts,Sampling;
};
struct RayNode { float4 Minimum,Maximum; uint4 Links; };
struct RayTriangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct RayMaterial { uint4 DiffuseAlpha,OpacityRed; float4 Coverage; };
struct RayStorage
{
    device const RayNode* Nodes;
    device const RayTriangle* Triangles;
    device const RayMaterial* Materials;
    device const float* Texels;
};
struct RayAOVertex { float4 Position [[position]]; };
vertex RayAOVertex VSMain(uint index [[vertex_id]])
{
    float2 corner=float2((index<<1)&2,index&2); return {float4(corner*2-1,0,1)};
}
float Coverage(uint4 image,float2 uv,device const float* texels)
{
    if (!image.w) return 1;
    float2 texel=fract(float2(uv.x,-uv.y))*float2(image.yz)-PixelCenter;
    uint x=texel.x<0 ? image.y-1 : uint(floor(texel.x)),y=texel.y<0 ? image.z-1 : uint(floor(texel.y));
    uint nextX=x+1==image.y ? 0 : x+1,nextY=y+1==image.z ? 0 : y+1;
    float upper=mix(texels[image.x+y*image.y+x],texels[image.x+y*image.y+nextX],fract(texel.x));
    float lower=mix(texels[image.x+nextY*image.y+x],texels[image.x+nextY*image.y+nextX],fract(texel.x));
    return mix(upper,lower,fract(texel.y));
}
bool BoundsHit(RayNode node,float3 origin,float3 ray,float radius)
{
    float nearDistance=0,farDistance=radius;
    for (uint axis=0;axis<3;++axis)
    {
        if (ray[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float first=(node.Minimum[axis]-origin[axis])/ray[axis],last=(node.Maximum[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(first,last)); farDistance=min(farDistance,max(first,last));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float Edge(float2 first,float2 second)
{
    float positive=first.x*second.y,negative=first.y*second.x,area=positive-negative;
    return abs(area)<=EdgeError*(abs(positive)+abs(negative)) ? 0 : area;
}
bool TriangleHit(RayTriangle triangle,float3 origin,float3 ray,float radius,thread float3& weights)
{
    weights=0; uint major=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    float2 shear=float2(ray[horizontal],ray[vertical])/ray[major];
    float2 first=float2(a[horizontal],a[vertical])-shear*a[major];
    float2 second=float2(b[horizontal],b[vertical])-shear*b[major];
    float2 third=float2(c[horizontal],c[vertical])-shear*c[major];
    float3 edges=float3(Edge(second,third),Edge(third,first),Edge(first,second));
    if (any(edges<0) && any(edges>0)) return false;
    float area=edges.x+edges.y+edges.z; if (area==0) return false;
    float distance=dot(edges,float3(a[major],b[major],c[major]))/(area*ray[major]);
    if (!(distance>0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool AnyHit(float3 origin,float3 ray,float radius,uint nodeCount,RayStorage storage)
{
    uint address=0;
    while (address<nodeCount)
    {
        RayNode node=storage.Nodes[address];
        if (!BoundsHit(node,origin,ray,radius)) { address=node.Links.w; continue; }
        if (!node.Links.z) { address=node.Links.x; continue; }
        for (uint local=0;local<node.Links.z;++local)
        {
            RayTriangle triangle=storage.Triangles[node.Links.y+local]; float3 weights;
            if (!TriangleHit(triangle,origin,ray,radius,weights)) continue;
            RayMaterial material=storage.Materials[triangle.Metadata.x]; if (material.Coverage.y<=0) return true;
            float2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float alpha=dot(weights,float3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x;
            alpha*=Coverage(material.DiffuseAlpha,uv,storage.Texels)*Coverage(material.OpacityRed,uv,storage.Texels);
            if (saturate(alpha)>=material.Coverage.y) return true;
        }
        address=node.Links.w;
    }
    return false;
}
bool Receiver(uint2 pixel,constant RayAOConstants& uniforms,texture2d<float,access::read> depth,
    texture2d<float,access::read> material,thread float3& position,thread float3& normal)
{
    position=normal=0; float d=depth.read(pixel).r; float4 stored=material.read(pixel); float lengthSquared=dot(stored.xyz,stored.xyz);
    if (!(d<EmptyDepth && stored.w>=0 && lengthSquared>MinimumNormalSquared && isfinite(lengthSquared))) return false;
    float2 uv=(float2(pixel)+PixelCenter)*uniforms.ViewportRadiusBias.xy;
    float4 clip=float4(uv.x*2-1,1-uv.y*2,d,1);
    float4 world=float4(dot(uniforms.InverseViewProjection[0],clip),dot(uniforms.InverseViewProjection[1],clip),
        dot(uniforms.InverseViewProjection[2],clip),dot(uniforms.InverseViewProjection[3],clip));
    if (!(abs(world.w)>MinimumW)) return false;
    position=world.xyz/world.w; normal=stored.xyz*rsqrt(lengthSquared); return all(isfinite(position));
}
float Estimate(uint2 pixel,float3 position,float3 normal,constant RayAOConstants& uniforms,RayStorage storage)
{
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    float3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent);
    float phase=2*Pi*fract(dot(float2(pixel)+PixelCenter,float2(sqrt(2.0f),sqrt(3.0f))));
    const float goldenAngle=Pi*(3-sqrt(5.0f));
    float3 origin=position+normal*uniforms.ViewportRadiusBias.w; float blocked=0;
    for (uint sampleIndex=0;sampleIndex<uniforms.Sampling.x;++sampleIndex)
    {
        float radial=(float(sampleIndex)+PixelCenter)/uniforms.Sampling.x,angle=phase+sampleIndex*goldenAngle;
        float3 ray=tangent*(sqrt(radial)*cos(angle))+bitangent*(sqrt(radial)*sin(angle))+normal*sqrt(1-radial);
        blocked+=AnyHit(origin,normalize(ray),uniforms.ViewportRadiusBias.z,uniforms.SceneCounts.x,storage) ? 1 : 0;
    }
    return saturate(1-blocked/uniforms.Sampling.x*uniforms.ForwardIntensity.w);
}
float Filter(uint2 pixel,uint2 extent,float3 position,float3 normal,constant RayAOConstants& uniforms,
    texture2d<float,access::read> depth,texture2d<float,access::read> material,texture2d<float,access::read> visibility)
{
    int2 axis=uniforms.Options.x==HorizontalPass ? int2(1,0) : int2(0,1);
    float sigma=uniforms.ViewportRadiusBias.z*DepthSigmaFraction,weighted=0,total=0;
    for (int tap=-FilterRadius;tap<=FilterRadius;++tap)
    {
        uint2 neighbor=uint2(clamp(int2(pixel)+tap*axis,int2(0),int2(extent)-1)); float3 point,other;
        if (!Receiver(neighbor,uniforms,depth,material,point,other)) continue;
        float delta=dot(point-position,uniforms.ForwardIntensity.xyz);
        float weight=exp(-float(tap*tap)/(2*SpatialSigma*SpatialSigma)-delta*delta/(2*sigma*sigma))*pow(saturate(dot(normal,other)),NormalExponent);
        weighted+=visibility.read(neighbor).r*weight; total+=weight;
    }
    return total>MinimumWeight ? weighted/total : 1;
}
fragment float4 PSMain(RayAOVertex input [[stage_in]],constant RayAOConstants& uniforms [[buffer(0)]],
    device const RayNode* nodes [[buffer(1)]],device const RayTriangle* triangles [[buffer(2)]],
    device const RayMaterial* materials [[buffer(3)]],device const float* texels [[buffer(4)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normalRoughness [[texture(1)]],
    texture2d<float,access::read> ambientMetallic [[texture(2)]],texture2d<float,access::read> previousVisibility [[texture(3)]])
{
    if (uniforms.ForwardIntensity.w<=0) return float4(uniforms.Options.x==CompositePass ? 0 : 1,0,0,0);
    uint2 pixel=uint2(input.Position.xy),extent=uint2(depth.get_width(),depth.get_height()); float3 position,normal;
    bool valid=Receiver(pixel,uniforms,depth,normalRoughness,position,normal);
    // Background pixels have no surface frame; do not construct a ray from it.
    if (!valid) return float4(uniforms.Options.x==CompositePass ? 0 : 1,0,0,0);
    if (uniforms.Options.x==EstimatePass)
    {
        RayStorage storage{nodes,triangles,materials,texels};
        return float4(Estimate(pixel,position,normal,uniforms,storage),0,0,0);
    }
    if (uniforms.Options.x==HorizontalPass || uniforms.Options.x==VerticalPass)
        return float4(Filter(pixel,extent,position,normal,uniforms,depth,normalRoughness,previousVisibility),0,0,0);
    return float4(ambientMetallic.read(pixel).rgb*(1-saturate(previousVisibility.read(pixel).r)),0);
}
