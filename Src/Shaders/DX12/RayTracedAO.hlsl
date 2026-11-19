// DX12 scene-ray AO. Four immutable structured SRVs hold a stackless BVH,
// primitive attributes, materials and scalar cutout texels. No CPU AO fallback.
// Ray-aligned primitive projection: Woop et al., JCGT 2013,
// https://jcgt.org/published/0002/01/05/ . FP32 edge uncertainty is accepted
// conservatively; this portable variant does not claim exact predicates.
cbuffer RayOcclusion : register(b0)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 ViewportRadiusBias,ForwardIntensity;
    uint4 Options,SceneCounts,Sampling;
};
struct NodeData { float4 Lower,Upper; uint4 Link; };
struct TriangleData { float4 P0,P1,P2,UV01,UV2; uint4 Identity; };
struct MaterialData { uint4 Diffuse,Opacity; float4 Alpha; };
Texture2D<float> Depth : register(t0);
Texture2D<float4> Normal : register(t1),Ambient : register(t2),Visibility : register(t3);
StructuredBuffer<NodeData> Hierarchy : register(t4);
StructuredBuffer<TriangleData> Geometry : register(t5);
StructuredBuffer<MaterialData> Material : register(t6);
StructuredBuffer<float> AlphaAtlas : register(t7);
static const uint RayPass=0,HorizontalPass=1,VerticalPass=2,ResolvePass=3;
static const float Pi=3.141592653589793,GoldenAngle=Pi*(3-sqrt(5.0));
static const float HalfPixel=0.5,EmptyDepth=1,MinimumW=1e-7,MinimumNormalSquared=1e-12;
static const float Binary32Epsilon=1.1920928955078125e-7;
static const float EdgeRoundoff=3*Binary32Epsilon/(1-3*Binary32Epsilon);
static const int KernelRadius=2;
static const float KernelSigma=1.5,DepthSigmaFraction=0.25,NormalExponent=8,MinimumWeight=1e-7;
struct VertexOutput { float4 Position:SV_POSITION; };
VertexOutput VSMain(uint id:SV_VertexID)
{
    float2 uv=float2((id<<1)&2,id&2); VertexOutput result;
    result.Position=float4(uv*2-1,0,1); return result;
}
float ReadCoverage(uint4 map,float2 uv)
{
    if (!map.w) return 1;
    float2 p=frac(float2(uv.x,-uv.y))*float2(map.yz)-HalfPixel;
    uint x=p.x<0 ? map.y-1 : uint(floor(p.x)),y=p.y<0 ? map.z-1 : uint(floor(p.y));
    uint nextX=x+1==map.y ? 0 : x+1,nextY=y+1==map.z ? 0 : y+1;
    float upper=lerp(AlphaAtlas[map.x+y*map.y+x],AlphaAtlas[map.x+y*map.y+nextX],frac(p.x));
    float lower=lerp(AlphaAtlas[map.x+nextY*map.y+x],AlphaAtlas[map.x+nextY*map.y+nextX],frac(p.x));
    return lerp(upper,lower,frac(p.y));
}
bool InBounds(NodeData node,float3 origin,float3 ray,float limit)
{
    float nearDistance=0,farDistance=limit;
    [unroll] for (uint dimension=0;dimension<3;++dimension)
    {
        if (ray[dimension]==0)
        {
            if (origin[dimension]<node.Lower[dimension] || origin[dimension]>node.Upper[dimension]) return false;
            continue;
        }
        float first=(node.Lower[dimension]-origin[dimension])/ray[dimension];
        float last=(node.Upper[dimension]-origin[dimension])/ray[dimension];
        nearDistance=max(nearDistance,min(first,last)); farDistance=min(farDistance,max(first,last));
        if (nearDistance>farDistance) return false;
    }
    return true;
}
float EdgeFunction(float2 first,float2 second)
{
    precise float positive=first.x*second.y,negative=first.y*second.x;
    precise float signedArea=positive-negative;
    return abs(signedArea)<=EdgeRoundoff*(abs(positive)+abs(negative)) ? 0 : signedArea;
}
bool HitTriangle(TriangleData primitive,float3 origin,float3 ray,float limit,out float3 barycentric)
{
    barycentric=0;
    uint major=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 first=primitive.P0.xyz-origin,second=primitive.P1.xyz-origin,third=primitive.P2.xyz-origin;
    float2 slope=float2(ray[horizontal],ray[vertical])/ray[major];
    precise float2 a=float2(first[horizontal],first[vertical])-slope*first[major];
    precise float2 b=float2(second[horizontal],second[vertical])-slope*second[major];
    precise float2 c=float2(third[horizontal],third[vertical])-slope*third[major];
    float3 area=float3(EdgeFunction(b,c),EdgeFunction(c,a),EdgeFunction(a,b));
    if (any(area<0) && any(area>0)) return false;
    float determinant=area.x+area.y+area.z; if (determinant==0) return false;
    float distance=dot(area,float3(first[major],second[major],third[major]))/(determinant*ray[major]);
    if (!(distance>0 && distance<=limit)) return false;
    barycentric=area/determinant; return true;
}
bool AnyAcceptedHit(float3 origin,float3 ray,float limit)
{
    uint address=0;
    [loop] while (address<SceneCounts.x)
    {
        NodeData node=Hierarchy[address];
        if (!InBounds(node,origin,ray,limit)) { address=node.Link.w; continue; }
        if (!node.Link.z) { address=node.Link.x; continue; }
        [loop] for (uint local=0;local<node.Link.z;++local)
        {
            TriangleData primitive=Geometry[node.Link.y+local]; float3 barycentric;
            if (!HitTriangle(primitive,origin,ray,limit,barycentric)) continue;
            MaterialData material=Material[primitive.Identity.x]; if (material.Alpha.y<=0) return true;
            float2 uv=barycentric.x*primitive.UV01.xy+barycentric.y*primitive.UV01.zw+barycentric.z*primitive.UV2.xy;
            float alpha=dot(barycentric,float3(primitive.P0.w,primitive.P1.w,primitive.P2.w))*material.Alpha.x;
            alpha*=ReadCoverage(material.Diffuse,uv)*ReadCoverage(material.Opacity,uv);
            if (saturate(alpha)>=material.Alpha.y) return true;
        }
        address=node.Link.w;
    }
    return false;
}
bool Receiver(int2 pixel,out float3 position,out float3 unitNormal)
{
    position=unitNormal=0; float depth=Depth.Load(int3(pixel,0)); float4 stored=Normal.Load(int3(pixel,0));
    float normalSquared=dot(stored.xyz,stored.xyz);
    if (!(depth<EmptyDepth && stored.w>=0 && normalSquared>MinimumNormalSquared && isfinite(normalSquared))) return false;
    float2 uv=(float2(pixel)+HalfPixel)*ViewportRadiusBias.xy;
    float4 world=mul(InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!(abs(world.w)>MinimumW)) return false;
    position=world.xyz/world.w; unitNormal=stored.xyz*rsqrt(normalSquared); return all(isfinite(position));
}
float TraceHemisphere(int2 pixel,float3 position,float3 normal)
{
    float3 axis=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    float3 u=normalize(cross(axis,normal)),v=cross(normal,u);
    float phase=2*Pi*frac(dot(float2(pixel)+HalfPixel,float2(sqrt(2.0),sqrt(3.0))));
    float blocked=0; float3 origin=position+normal*ViewportRadiusBias.w;
    [loop] for (uint sampleIndex=0;sampleIndex<Sampling.x;++sampleIndex)
    {
        float diskArea=(sampleIndex+HalfPixel)/Sampling.x,angle=phase+sampleIndex*GoldenAngle;
        float3 ray=u*(sqrt(diskArea)*cos(angle))+v*(sqrt(diskArea)*sin(angle))+normal*sqrt(1-diskArea);
        blocked+=AnyAcceptedHit(origin,normalize(ray),ViewportRadiusBias.z) ? 1 : 0;
    }
    return saturate(1-blocked/Sampling.x*ForwardIntensity.w);
}
float FilterVisibility(int2 pixel,int2 size,float3 position,float3 normal)
{
    int2 step=Options.x==HorizontalPass ? int2(1,0) : int2(0,1);
    float depthSigma=ViewportRadiusBias.z*DepthSigmaFraction,weightSum=0,weighted=0;
    [unroll] for (int tap=-KernelRadius;tap<=KernelRadius;++tap)
    {
        int2 neighbor=clamp(pixel+tap*step,0,size-1); float3 neighborPosition,direction;
        if (!Receiver(neighbor,neighborPosition,direction)) continue;
        float delta=dot(neighborPosition-position,ForwardIntensity.xyz);
        float weight=exp(-float(tap*tap)/(2*KernelSigma*KernelSigma)-delta*delta/(2*depthSigma*depthSigma))*pow(saturate(dot(normal,direction)),NormalExponent);
        weighted+=Visibility.Load(int3(neighbor,0)).r*weight; weightSum+=weight;
    }
    return weightSum>MinimumWeight ? weighted/weightSum : 1;
}
float4 PSMain(VertexOutput input):SV_TARGET
{
    if (ForwardIntensity.w<=0) return float4(Options.x==ResolvePass ? 0 : 1,0,0,0);
    int2 pixel=int2(input.Position.xy); uint width,height; Depth.GetDimensions(width,height);
    float3 position,normal; bool valid=Receiver(pixel,position,normal);
    // Reject absent receivers with control flow, not a speculative ?: select.
    // Otherwise zero normals create invalid rays that can walk the entire BVH.
    [branch] if (!valid) return float4(Options.x==ResolvePass ? 0 : 1,0,0,0);
    if (Options.x==RayPass) return float4(TraceHemisphere(pixel,position,normal),0,0,0);
    if (Options.x==HorizontalPass || Options.x==VerticalPass) return float4(FilterVisibility(pixel,int2(width,height),position,normal),0,0,0);
    return float4(Ambient.Load(int3(pixel,0)).rgb*(1-saturate(Visibility.Load(int3(pixel,0)).r)),0);
}
