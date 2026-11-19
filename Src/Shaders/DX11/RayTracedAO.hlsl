// Finite-radius cosine visibility, traversed in the native DX11 pixel shader.
// Ray-aligned edge tests follow Woop, Benthin and Wald (JCGT 2013):
// https://jcgt.org/published/0002/01/05/ . Inclusive FP32 roundoff bands replace
// the paper's rare double fallback; this is conservative, not an exact predicate.
cbuffer RayAOConstants : register(b0)
{
    row_major float4x4 ViewProjection,InverseViewProjection;
    float4 ViewportRadiusBias,ForwardIntensity;
    uint4 Options,SceneCounts,Sampling;
};
struct RayNode { float4 Minimum,Maximum; uint4 Links; };
struct RayTriangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct RayMaterial { uint4 DiffuseAlpha,OpacityRed; float4 Coverage; };
Texture2D<float> OpaqueDepth : register(t0);
Texture2D<float4> NormalRoughness : register(t1),AmbientMetallic : register(t2),PreviousVisibility : register(t3);
StructuredBuffer<RayNode> Nodes : register(t4);
StructuredBuffer<RayTriangle> Triangles : register(t5);
StructuredBuffer<RayMaterial> Materials : register(t6);
StructuredBuffer<float> CoverageTexels : register(t7);
static const uint EstimatePass=0,HorizontalPass=1,VerticalPass=2,CompositePass=3;
static const float Pi=3.141592653589793;
static const float GoldenAngle=Pi*(3-sqrt(5.0));
static const float PixelCenter=0.5,ClearDepth=1;
static const float FloatEpsilon=1.1920928955078125e-7; // IEEE-754 binary32 spacing at one.
static const float EdgeError=(3*FloatEpsilon)/(1-3*FloatEpsilon);
static const float ProjectionEpsilon=1e-7,NormalLengthEpsilon=1e-12;
static const int FilterRadius=2;
static const float SpatialSigma=1.5,DepthSigmaFraction=0.25,NormalWeightPower=8,WeightEpsilon=1e-7;

struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint id:SV_VertexID)
{
    float2 corner=float2((id<<1)&2,id&2);
    ScreenVertex output; output.Position=float4(corner*2-1,0,1); return output;
}
float Coverage(uint4 image,float2 uv)
{
    if (image.w==0) return 1;
    float2 texel=frac(float2(uv.x,-uv.y))*float2(image.yz)-PixelCenter;
    uint2 a=uint2(texel.x<0 ? image.y-1 : uint(floor(texel.x)),texel.y<0 ? image.z-1 : uint(floor(texel.y)));
    uint2 b=uint2(a.x+1==image.y ? 0 : a.x+1,a.y+1==image.z ? 0 : a.y+1);
    float2 weight=frac(texel);
    float top=lerp(CoverageTexels[image.x+a.y*image.y+a.x],CoverageTexels[image.x+a.y*image.y+b.x],weight.x);
    float bottom=lerp(CoverageTexels[image.x+b.y*image.y+a.x],CoverageTexels[image.x+b.y*image.y+b.x],weight.x);
    return lerp(top,bottom,weight.y);
}
bool BoxHit(RayNode node,float3 origin,float3 direction,float radius)
{
    float enter=0,leave=radius;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(float2 a,float2 b)
{
    precise float forward=a.x*b.y,backward=a.y*b.x;
    precise float value=forward-backward;
    return abs(value)<=EdgeError*(abs(forward)+abs(backward)) ? 0 : value;
}
bool TriangleHit(RayTriangle primitive,float3 origin,float3 direction,float radius,out float3 weights)
{
    weights=0;
    uint z=abs(direction.x)>abs(direction.y) ? 0 : 1; if (abs(direction.z)>abs(direction[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3;
    float3 a=primitive.A.xyz-origin,b=primitive.B.xyz-origin,c=primitive.C.xyz-origin;
    float2 shear=float2(direction[x],direction[y])/direction[z];
    precise float2 pa=float2(a[x],a[y])-shear*a[z],pb=float2(b[x],b[y])-shear*b[z],pc=float2(c[x],c[y])-shear*c[z];
    float3 edges=float3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(edges<0) && any(edges>0)) return false;
    float area=edges.x+edges.y+edges.z; if (area==0) return false;
    float distance=dot(edges,float3(a[z],b[z],c[z]))/(area*direction[z]);
    if (!(distance>0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool Occluded(float3 origin,float3 direction,float radius)
{
    uint nodeIndex=0;
    [loop] while (nodeIndex<SceneCounts.x)
    {
        RayNode node=Nodes[nodeIndex];
        if (!BoxHit(node,origin,direction,radius)) { nodeIndex=node.Links.w; continue; }
        if (node.Links.z==0) { nodeIndex=node.Links.x; continue; }
        [loop] for (uint index=0;index<node.Links.z;++index)
        {
            RayTriangle primitive=Triangles[node.Links.y+index]; float3 weights;
            if (!TriangleHit(primitive,origin,direction,radius,weights)) continue;
            RayMaterial material=Materials[primitive.Metadata.x]; if (material.Coverage.y<=0) return true;
            float2 uv=primitive.UVAB.xy*weights.x+primitive.UVAB.zw*weights.y+primitive.UVC.xy*weights.z;
            float alpha=saturate(dot(weights,float3(primitive.A.w,primitive.B.w,primitive.C.w))*material.Coverage.x*
                Coverage(material.DiffuseAlpha,uv)*Coverage(material.OpacityRed,uv));
            if (alpha>=material.Coverage.y) return true;
        }
        nodeIndex=node.Links.w;
    }
    return false;
}
bool Surface(int2 pixel,out float3 position,out float3 normal)
{
    position=normal=0; float depth=OpaqueDepth.Load(int3(pixel,0)); float4 value=NormalRoughness.Load(int3(pixel,0));
    float lengthSquared=dot(value.xyz,value.xyz);
    if (!(depth<ClearDepth && value.w>=0 && lengthSquared>NormalLengthEpsilon && isfinite(lengthSquared))) return false;
    float2 uv=(float2(pixel)+PixelCenter)*ViewportRadiusBias.xy;
    float4 world=mul(InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!(abs(world.w)>ProjectionEpsilon)) return false;
    position=world.xyz/world.w; normal=value.xyz*rsqrt(lengthSquared); return all(isfinite(position));
}
float Estimate(int2 pixel,float3 position,float3 normal)
{
    // Choose the least parallel axis. Irrational pixel phases rotate the
    // deterministic disk without a noise texture or changing temporal pattern.
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    float3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent);
    float rotation=2*Pi*frac(dot(float2(pixel)+PixelCenter,float2(sqrt(2.0),sqrt(3.0))));
    float hits=0; float3 origin=position+normal*ViewportRadiusBias.w;
    [loop] for (uint index=0;index<Sampling.x;++index)
    {
        float radial=(float(index)+PixelCenter)/Sampling.x,angle=rotation+GoldenAngle*index;
        float3 direction=tangent*(cos(angle)*sqrt(radial))+bitangent*(sin(angle)*sqrt(radial))+normal*sqrt(1-radial);
        hits+=Occluded(origin,normalize(direction),ViewportRadiusBias.z) ? 1 : 0;
    }
    return saturate(1-hits/Sampling.x*ForwardIntensity.w);
}
float Filter(int2 pixel,int2 extent,float3 center,float3 normal)
{
    int2 axis=Options.x==HorizontalPass ? int2(1,0) : int2(0,1);
    float sigma=ViewportRadiusBias.z*DepthSigmaFraction,weighted=0,total=0;
    [unroll] for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        int2 neighbor=clamp(pixel+axis*offset,0,extent-1); float3 position,other;
        if (!Surface(neighbor,position,other)) continue;
        float distance=dot(position-center,ForwardIntensity.xyz);
        float weight=exp(-float(offset*offset)/(2*SpatialSigma*SpatialSigma)-distance*distance/(2*sigma*sigma))*pow(saturate(dot(normal,other)),NormalWeightPower);
        weighted+=PreviousVisibility.Load(int3(neighbor,0)).r*weight; total+=weight;
    }
    return total>WeightEpsilon ? weighted/total : 1;
}
float4 PSMain(ScreenVertex input):SV_TARGET
{
    if (ForwardIntensity.w<=0) return float4(Options.x==CompositePass ? 0 : 1,0,0,0);
    int2 pixel=int2(input.Position.xy); uint width,height; OpaqueDepth.GetDimensions(width,height);
    float3 position,normal; bool valid=Surface(pixel,position,normal);
    // HLSL can evaluate both arms of a conditional expression. Branch before
    // traversal: background pixels have no receiver frame or valid ray direction.
    [branch] if (!valid) return float4(Options.x==CompositePass ? 0 : 1,0,0,0);
    if (Options.x==EstimatePass) return float4(Estimate(pixel,position,normal),0,0,0);
    if (Options.x==HorizontalPass || Options.x==VerticalPass) return float4(Filter(pixel,int2(width,height),position,normal),0,0,0);
    return float4(AmbientMetallic.Load(int3(pixel,0)).rgb*(1-saturate(PreviousVisibility.Load(int3(pixel,0)).r)),0);
}
