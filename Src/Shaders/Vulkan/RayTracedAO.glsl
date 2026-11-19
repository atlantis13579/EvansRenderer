#version 450
// Vulkan-native scene BVH traversal, finite-radius cosine visibility and
// bilateral reconstruction. Only lost ambient is subtracted during composition.
// Ray-aligned triangles: Woop et al., JCGT 2013,
// https://jcgt.org/published/0002/01/05/ . FP32 uncertainty bands are inclusive;
// the rare double fallback from the paper is not used by this portable path.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=0,std140,row_major) uniform RayAOParameters
{
    mat4 ViewProjection,InverseViewProjection;
    vec4 ViewportRadiusBias,ForwardIntensity;
    uvec4 Options,SceneCounts,Sampling;
} Parameters;
layout(set=0,binding=1) uniform texture2D OpaqueDepth;
layout(set=0,binding=2) uniform texture2D NormalRoughness;
layout(set=0,binding=3) uniform texture2D AmbientMetallic;
layout(set=0,binding=4) uniform texture2D PreviousVisibility;
layout(set=0,binding=5) uniform sampler PointSampler;
struct SceneNode { vec4 Minimum,Maximum; uvec4 Links; };
struct SceneTriangle { vec4 A,B,C,UVAB,UVC; uvec4 Metadata; };
struct SceneMaterial { uvec4 DiffuseAlpha,OpacityRed; vec4 Coverage; };
layout(set=0,binding=6,std430) readonly buffer NodeStorage { SceneNode Nodes[]; };
layout(set=0,binding=7,std430) readonly buffer TriangleStorage { SceneTriangle Triangles[]; };
layout(set=0,binding=8,std430) readonly buffer MaterialStorage { SceneMaterial Materials[]; };
layout(set=0,binding=9,std430) readonly buffer CoverageStorage { float CoverageTexels[]; };
layout(location=0) out vec4 OutValue;
const uint EstimatePass=0u,HorizontalPass=1u,VerticalPass=2u,CompositePass=3u;
const float Pi=3.141592653589793,GoldenAngle=Pi*(3.0-sqrt(5.0));
const float PixelCenter=0.5,ClearDepth=1.0,MinimumW=1e-7,MinimumNormalSquared=1e-12;
const float Binary32Epsilon=1.1920928955078125e-7;
const float EdgeError=3.0*Binary32Epsilon/(1.0-3.0*Binary32Epsilon);
const int FilterRadius=2;
const float SpatialSigma=1.5,DepthSigmaFraction=0.25,NormalExponent=8.0,MinimumWeight=1e-7;

float SampleCoverage(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0;
    vec2 texel=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uvec2 a=uvec2(texel.x<0.0 ? image.y-1u : uint(floor(texel.x)),texel.y<0.0 ? image.z-1u : uint(floor(texel.y)));
    uvec2 b=uvec2(a.x+1u==image.y ? 0u : a.x+1u,a.y+1u==image.z ? 0u : a.y+1u);
    float top=mix(CoverageTexels[image.x+a.y*image.y+a.x],CoverageTexels[image.x+a.y*image.y+b.x],fract(texel.x));
    float bottom=mix(CoverageTexels[image.x+b.y*image.y+a.x],CoverageTexels[image.x+b.y*image.y+b.x],fract(texel.x));
    return mix(top,bottom,fract(texel.y));
}
bool IntersectsBounds(SceneNode node,vec3 origin,vec3 ray,float radius)
{
    float nearDistance=0.0,farDistance=radius;
    for (int axis=0;axis<3;++axis)
    {
        if (ray[axis]==0.0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/ray[axis],b=(node.Maximum[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(a,b)); farDistance=min(farDistance,max(a,b));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float ProjectedEdge(vec2 first,vec2 second)
{
    precise float positive=first.x*second.y,negative=first.y*second.x;
    precise float area=positive-negative;
    return abs(area)<=EdgeError*(abs(positive)+abs(negative)) ? 0.0 : area;
}
bool IntersectsTriangle(SceneTriangle triangle,vec3 origin,vec3 ray,float radius,out vec3 weights)
{
    weights=vec3(0.0); int major=abs(ray.x)>abs(ray.y) ? 0 : 1;
    if (abs(ray.z)>abs(ray[major])) major=2;
    int horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(ray[horizontal],ray[vertical])/ray[major];
    precise vec2 first=vec2(a[horizontal],a[vertical])-shear*a[major];
    precise vec2 second=vec2(b[horizontal],b[vertical])-shear*b[major];
    precise vec2 third=vec2(c[horizontal],c[vertical])-shear*c[major];
    vec3 edges=vec3(ProjectedEdge(second,third),ProjectedEdge(third,first),ProjectedEdge(first,second));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float area=edges.x+edges.y+edges.z; if (area==0.0) return false;
    float distance=dot(edges,vec3(a[major],b[major],c[major]))/(area*ray[major]);
    if (!(distance>0.0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool TraceOcclusion(vec3 origin,vec3 ray,float radius)
{
    uint current=0u;
    while (current<Parameters.SceneCounts.x)
    {
        SceneNode node=Nodes[current];
        if (!IntersectsBounds(node,origin,ray,radius)) { current=node.Links.w; continue; }
        if (node.Links.z==0u) { current=node.Links.x; continue; }
        for (uint index=0u;index<node.Links.z;++index)
        {
            SceneTriangle triangle=Triangles[node.Links.y+index]; vec3 weights;
            if (!IntersectsTriangle(triangle,origin,ray,radius,weights)) continue;
            SceneMaterial material=Materials[triangle.Metadata.x]; if (material.Coverage.y<=0.0) return true;
            vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float alpha=dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x;
            alpha*=SampleCoverage(material.DiffuseAlpha,uv)*SampleCoverage(material.OpacityRed,uv);
            if (clamp(alpha,0.0,1.0)>=material.Coverage.y) return true;
        }
        current=node.Links.w;
    }
    return false;
}
bool ReadSurface(ivec2 pixel,out vec3 position,out vec3 normal)
{
    position=normal=vec3(0.0); float depth=texelFetch(sampler2D(OpaqueDepth,PointSampler),pixel,0).r;
    vec4 value=texelFetch(sampler2D(NormalRoughness,PointSampler),pixel,0); float lengthSquared=dot(value.xyz,value.xyz);
    if (!(depth<ClearDepth && value.w>=0.0 && lengthSquared>MinimumNormalSquared) || isinf(lengthSquared)) return false;
    vec2 uv=(vec2(pixel)+PixelCenter)*Parameters.ViewportRadiusBias.xy;
    // Undo the geometry's Vulkan clip-Y inversion before CPU-camera unprojection.
    vec4 world=Parameters.InverseViewProjection*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!(abs(world.w)>MinimumW)) return false;
    position=world.xyz/world.w; normal=value.xyz*inversesqrt(lengthSquared);
    return !any(isnan(position)) && !any(isinf(position));
}
float EstimateVisibility(ivec2 pixel,vec3 position,vec3 normal)
{
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    vec3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent);
    float rotation=2.0*Pi*fract(dot(vec2(pixel)+PixelCenter,vec2(sqrt(2.0),sqrt(3.0))));
    vec3 origin=position+normal*Parameters.ViewportRadiusBias.w; float blocked=0.0;
    for (uint sampleIndex=0u;sampleIndex<Parameters.Sampling.x;++sampleIndex)
    {
        float radial=(float(sampleIndex)+PixelCenter)/float(Parameters.Sampling.x),angle=rotation+GoldenAngle*float(sampleIndex);
        vec3 ray=tangent*(sqrt(radial)*cos(angle))+bitangent*(sqrt(radial)*sin(angle))+normal*sqrt(1.0-radial);
        blocked+=TraceOcclusion(origin,normalize(ray),Parameters.ViewportRadiusBias.z) ? 1.0 : 0.0;
    }
    return clamp(1.0-blocked/float(Parameters.Sampling.x)*Parameters.ForwardIntensity.w,0.0,1.0);
}
float FilterVisibility(ivec2 pixel,ivec2 extent,vec3 center,vec3 normal)
{
    ivec2 axis=Parameters.Options.x==HorizontalPass ? ivec2(1,0) : ivec2(0,1);
    float sigma=Parameters.ViewportRadiusBias.z*DepthSigmaFraction,weighted=0.0,total=0.0;
    for (int offset=-FilterRadius;offset<=FilterRadius;++offset)
    {
        ivec2 neighbor=clamp(pixel+axis*offset,ivec2(0),extent-1); vec3 position,other;
        if (!ReadSurface(neighbor,position,other)) continue;
        float delta=dot(position-center,Parameters.ForwardIntensity.xyz);
        float weight=exp(-float(offset*offset)/(2.0*SpatialSigma*SpatialSigma)-delta*delta/(2.0*sigma*sigma))*pow(clamp(dot(normal,other),0.0,1.0),NormalExponent);
        weighted+=texelFetch(sampler2D(PreviousVisibility,PointSampler),neighbor,0).r*weight; total+=weight;
    }
    return total>MinimumWeight ? weighted/total : 1.0;
}
void main()
{
    if (Parameters.ForwardIntensity.w<=0.0) { OutValue=vec4(Parameters.Options.x==CompositePass ? 0.0 : 1.0,0,0,0); return; }
    ivec2 pixel=ivec2(gl_FragCoord.xy),extent=textureSize(sampler2D(OpaqueDepth,PointSampler),0);
    vec3 position,normal; bool valid=ReadSurface(pixel,position,normal);
    // A missing receiver has no hemisphere frame and must never trace rays.
    if (!valid) { OutValue=vec4(Parameters.Options.x==CompositePass ? 0.0 : 1.0,0,0,0); return; }
    if (Parameters.Options.x==EstimatePass) OutValue=vec4(EstimateVisibility(pixel,position,normal),0,0,0);
    else if (Parameters.Options.x==HorizontalPass || Parameters.Options.x==VerticalPass) OutValue=vec4(FilterVisibility(pixel,extent,position,normal),0,0,0);
    else OutValue=vec4(texelFetch(sampler2D(AmbientMetallic,PointSampler),pixel,0).rgb*
        (1.0-clamp(texelFetch(sampler2D(PreviousVisibility,PointSampler),pixel,0).r,0.0,1.0)),0.0);
}
#endif
