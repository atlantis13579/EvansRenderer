#version 310 es
// ES 3.1 fragment-storage traversal of the scene BVH, independent of camera
// depth visibility. Requires four fragment SSBO bindings, no ray-query extension.
// Ray-aligned triangle projection follows Woop et al., JCGT 2013:
// https://jcgt.org/published/0002/01/05/ . Inclusive FP32 edge bands are used
// instead of the paper's double fallback; they are not exact predicates.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
uniform mat4 ViewProjection,InverseViewProjection;
uniform vec4 ViewportRadiusBias,ForwardIntensity;
uniform uvec4 Options,SceneCounts,Sampling;
uniform highp sampler2D SceneDepth,NormalRoughness,AmbientMetallic,PreviousVisibility;
struct RayNode { vec4 Lower,Upper; uvec4 Link; };
struct RayTriangle { vec4 A,B,C,UVAB,UVC; uvec4 Identity; };
struct RayMaterial { uvec4 Diffuse,Opacity; vec4 Alpha; };
layout(std430,binding=0) readonly buffer NodeBlock { RayNode Nodes[]; };
layout(std430,binding=1) readonly buffer TriangleBlock { RayTriangle Triangles[]; };
layout(std430,binding=2) readonly buffer MaterialBlock { RayMaterial Materials[]; };
layout(std430,binding=3) readonly buffer AlphaBlock { float AlphaTexels[]; };
layout(location=0) out vec4 OutColor;
const uint RayPass=0u,HorizontalPass=1u,VerticalPass=2u,CompositePass=3u;
const float Pi=3.141592653589793,GoldenAngle=Pi*(3.0-sqrt(5.0));
const float PixelCenter=0.5,EmptyDepth=1.0,MinimumW=1e-7,MinimumNormalSquared=1e-12;
const float FloatEpsilon=1.1920928955078125e-7;
const float EdgeUncertainty=3.0*FloatEpsilon/(1.0-3.0*FloatEpsilon);
const int BlurRadius=2;
const float SpatialSigma=1.5,DepthSigmaFraction=0.25,NormalPower=8.0,MinimumWeight=1e-7;

float MaterialCoverage(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0;
    vec2 point=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uint x=point.x<0.0 ? image.y-1u : uint(floor(point.x));
    uint y=point.y<0.0 ? image.z-1u : uint(floor(point.y));
    uint right=x+1u==image.y ? 0u : x+1u,bottom=y+1u==image.z ? 0u : y+1u;
    float upper=mix(AlphaTexels[image.x+y*image.y+x],AlphaTexels[image.x+y*image.y+right],fract(point.x));
    float lower=mix(AlphaTexels[image.x+bottom*image.y+x],AlphaTexels[image.x+bottom*image.y+right],fract(point.x));
    return mix(upper,lower,fract(point.y));
}
bool BoundsHit(RayNode node,vec3 origin,vec3 direction,float radius)
{
    float enter=0.0,leave=radius;
    for (int axis=0;axis<3;++axis)
    {
        if (direction[axis]==0.0)
        {
            if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false;
        }
        else
        {
            float first=(node.Lower[axis]-origin[axis])/direction[axis],last=(node.Upper[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(first,last)); leave=min(leave,max(first,last));
            if (enter>leave) return false;
        }
    }
    return true;
}
float EdgeValue(vec2 a,vec2 b)
{
    float positive=a.x*b.y,negative=a.y*b.x,value=positive-negative;
    return abs(value)<=EdgeUncertainty*(abs(positive)+abs(negative)) ? 0.0 : value;
}
bool TriangleHit(RayTriangle triangle,vec3 origin,vec3 direction,float radius,out vec3 weights)
{
    weights=vec3(0.0); int major=abs(direction.x)>abs(direction.y) ? 0 : 1;
    if (abs(direction.z)>abs(direction[major])) major=2;
    int horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    vec3 first=triangle.A.xyz-origin,second=triangle.B.xyz-origin,third=triangle.C.xyz-origin;
    vec2 shear=vec2(direction[horizontal],direction[vertical])/direction[major];
    vec2 a=vec2(first[horizontal],first[vertical])-shear*first[major];
    vec2 b=vec2(second[horizontal],second[vertical])-shear*second[major];
    vec2 c=vec2(third[horizontal],third[vertical])-shear*third[major];
    vec3 edges=vec3(EdgeValue(b,c),EdgeValue(c,a),EdgeValue(a,b));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float area=edges.x+edges.y+edges.z; if (area==0.0) return false;
    float distance=dot(edges,vec3(first[major],second[major],third[major]))/(area*direction[major]);
    if (!(distance>0.0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool AnyHit(vec3 origin,vec3 direction,float radius)
{
    uint index=0u;
    while (index<SceneCounts.x)
    {
        // Member loads avoid ANGLE's aggregate-SSBO-load translation defect.
        // The std430 layout and traversal are unchanged on native ES drivers.
        RayNode node; node.Lower=Nodes[index].Lower; node.Upper=Nodes[index].Upper; node.Link=Nodes[index].Link;
        if (!BoundsHit(node,origin,direction,radius)) { index=node.Link.w; continue; }
        if (node.Link.z==0u) { index=node.Link.x; continue; }
        for (uint primitive=0u;primitive<node.Link.z;++primitive)
        {
            uint triangleIndex=node.Link.y+primitive;
            RayTriangle triangle; triangle.A=Triangles[triangleIndex].A; triangle.B=Triangles[triangleIndex].B;
            triangle.C=Triangles[triangleIndex].C; triangle.UVAB=Triangles[triangleIndex].UVAB;
            triangle.UVC=Triangles[triangleIndex].UVC; triangle.Identity=Triangles[triangleIndex].Identity;
            vec3 weights;
            if (!TriangleHit(triangle,origin,direction,radius,weights)) continue;
            RayMaterial material; material.Diffuse=Materials[triangle.Identity.x].Diffuse;
            material.Opacity=Materials[triangle.Identity.x].Opacity; material.Alpha=Materials[triangle.Identity.x].Alpha;
            if (material.Alpha.y<=0.0) return true;
            vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float coverage=dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Alpha.x;
            coverage*=MaterialCoverage(material.Diffuse,uv)*MaterialCoverage(material.Opacity,uv);
            if (clamp(coverage,0.0,1.0)>=material.Alpha.y) return true;
        }
        index=node.Link.w;
    }
    return false;
}
bool Surface(ivec2 pixel,out vec3 position,out vec3 normal)
{
    position=normal=vec3(0.0); float depth=texelFetch(SceneDepth,pixel,0).r;
    vec4 stored=texelFetch(NormalRoughness,pixel,0); float lengthSquared=dot(stored.xyz,stored.xyz);
    if (!(depth<EmptyDepth && stored.w>=0.0 && lengthSquared>MinimumNormalSquared) || isinf(lengthSquared)) return false;
    // ES framebuffer rows run upward; window depth already uses the CPU [0,1]
    // interval after the geometry vertex stage's clip-Z conversion.
    vec2 uv=(vec2(pixel)+PixelCenter)*ViewportRadiusBias.xy;
    vec4 world=InverseViewProjection*vec4(uv*2.0-1.0,depth,1.0);
    if (!(abs(world.w)>MinimumW)) return false;
    position=world.xyz/world.w; normal=stored.xyz*inversesqrt(lengthSquared);
    return !any(isnan(position)) && !any(isinf(position));
}
float TraceHemisphere(ivec2 pixel,ivec2 size,vec3 position,vec3 normal)
{
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    vec3 tangent=normalize(cross(helper,normal)),bitangent=cross(normal,tangent);
    vec2 canonicalPixel=vec2(pixel.x,size.y-1-pixel.y);
    float phase=2.0*Pi*fract(dot(canonicalPixel+PixelCenter,vec2(sqrt(2.0),sqrt(3.0))));
    vec3 origin=position+normal*ViewportRadiusBias.w; float blocked=0.0;
    for (uint sampleIndex=0u;sampleIndex<Sampling.x;++sampleIndex)
    {
        float radial=(float(sampleIndex)+PixelCenter)/float(Sampling.x),angle=phase+float(sampleIndex)*GoldenAngle;
        vec3 direction=tangent*(sqrt(radial)*cos(angle))+bitangent*(sqrt(radial)*sin(angle))+normal*sqrt(1.0-radial);
        blocked+=AnyHit(origin,normalize(direction),ViewportRadiusBias.z) ? 1.0 : 0.0;
    }
    return clamp(1.0-blocked/float(Sampling.x)*ForwardIntensity.w,0.0,1.0);
}
float Filter(ivec2 pixel,ivec2 size,vec3 center,vec3 normal)
{
    ivec2 axis=Options.x==HorizontalPass ? ivec2(1,0) : ivec2(0,1);
    float sigma=ViewportRadiusBias.z*DepthSigmaFraction,weighted=0.0,total=0.0;
    for (int tap=-BlurRadius;tap<=BlurRadius;++tap)
    {
        ivec2 neighbor=clamp(pixel+tap*axis,ivec2(0),size-1); vec3 position,other;
        if (!Surface(neighbor,position,other)) continue;
        float delta=dot(position-center,ForwardIntensity.xyz);
        float weight=exp(-float(tap*tap)/(2.0*SpatialSigma*SpatialSigma)-delta*delta/(2.0*sigma*sigma))*pow(clamp(dot(normal,other),0.0,1.0),NormalPower);
        weighted+=texelFetch(PreviousVisibility,neighbor,0).r*weight; total+=weight;
    }
    return total>MinimumWeight ? weighted/total : 1.0;
}
void main()
{
    if (ForwardIntensity.w<=0.0) { OutColor=vec4(Options.x==CompositePass ? 0.0 : 1.0,0,0,0); return; }
    ivec2 pixel=ivec2(gl_FragCoord.xy),size=textureSize(SceneDepth,0); vec3 position,normal;
    bool valid=Surface(pixel,position,normal);
    // Return before evaluating any hemisphere frame for empty/background pixels.
    if (!valid) { OutColor=vec4(Options.x==CompositePass ? 0.0 : 1.0,0,0,0); return; }
    if (Options.x==RayPass) OutColor=vec4(TraceHemisphere(pixel,size,position,normal),0,0,0);
    else if (Options.x==HorizontalPass || Options.x==VerticalPass) OutColor=vec4(Filter(pixel,size,position,normal),0,0,0);
    else OutColor=vec4(texelFetch(AmbientMetallic,pixel,0).rgb*(1.0-clamp(texelFetch(PreviousVisibility,pixel,0).r,0.0,1.0)),0.0);
}
#endif
