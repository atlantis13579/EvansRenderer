#version 310 es
// Independent ES camera-path tracing. FP32 means are committed only after a
// successful frame; center-ray depth remains separate from stochastic coverage.
// Diffuse/GGX mixture PDFs follow PBRT; VNDF sampling follows Heitz (2018).
// https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
// https://jcgt.org/published/0007/04/01/
// Conservative edge predicates: https://jcgt.org/published/0002/01/05/
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
uniform mat4 WorldFromClip,ClipFromWorld;
uniform vec4 FrameSize,RayParameters,BackgroundColor,MaterialParameters;
uniform uvec4 GeometryCounts,SurfaceCounts,Samples,Control;
uniform highp sampler2D CommittedMean,PathDepth;
layout(location=0) out vec4 OutColor;
#ifndef EVANS_PATH_COMPOSE
layout(location=1) out float OutDepth;
#endif
const uint Lambertian=1u,FlatFlag=1u,CasterFlag=1u,Directional=0u,Point=1u;
const float Pi=3.141592653589793,PixelCenter=0.5,InteriorDepth=0.5,BackgroundDepth=1.0;
const float ProjectionEpsilon=1e-7,VectorEpsilon=1e-16,TangentEpsilon=1e-8,ParallelAxis=0.999;
const float FloatSpacing=1.1920928955078125e-7,EdgeError=3.0*FloatSpacing/(1.0-3.0*FloatSpacing);
const float AlphaFloor=0.0025,CosineFloor=1e-6,Dielectric=0.04,FresnelPower=5.0;
const float DistanceSquaredFloor=0.01,MinimumLobeProbability=0.05;
const float SurvivalFloor=0.05,SurvivalCeiling=0.95,RadianceMaximum=65504.0,RadianceSpacing=32.0;
const uint RouletteDepth=3u;
// PCG constants specify the permutation, not scene-dependent tuning.
// Jarzynski and Olano (2020): https://jcgt.org/published/0009/03/02/
const uint PcgMultiplier=747796405u,PcgIncrement=2891336453u,PcgOutputMultiplier=277803737u;
const uint PcgDynamicShift=28u,PcgMinimumShift=4u,PcgOutputShift=22u,MantissaShift=8u;
const float UnitInterval=1.0/16777216.0;
struct NodeData { vec4 Lower,Upper; uvec4 Link; };
struct PrimitiveData { vec4 A,B,C,UVAB,UVC; uvec4 Identity; };
struct CoverageData { uvec4 Diffuse,Opacity; vec4 Factor; };
struct SurfaceData { vec4 NA,NB,NC,CA,CB,CC; };
struct MaterialData { vec4 Base,Ambient,Specular,Emission; uvec4 Diffuse,Normal; };
struct LightData { vec4 PositionType,DirectionIntensity,ColorShadow,Area; };
layout(std430,binding=0) readonly buffer NodesBlock { NodeData Nodes[]; };
layout(std430,binding=1) readonly buffer PrimitivesBlock { PrimitiveData Primitives[]; };
layout(std430,binding=2) readonly buffer CoverageBlock { CoverageData CoverageMaterials[]; };
layout(std430,binding=3) readonly buffer AlphaBlock { float AlphaTexels[]; };
layout(std430,binding=4) readonly buffer SurfacesBlock { SurfaceData Surfaces[]; };
layout(std430,binding=5) readonly buffer MaterialsBlock { MaterialData Materials[]; };
// RGBA32F texel addressing avoids the much smaller single-SSBO byte limit.
uniform highp sampler2D ColorAtlas;
layout(std430,binding=7) readonly buffer LightsBlock { LightData Lights[]; };

// Member loads also support ANGLE drivers whose aggregate SSBO load helpers
// cannot represent a returned structure on every generated HLSL control path.
NodeData ReadNode(uint index)
{ NodeData value; value.Lower=Nodes[index].Lower; value.Upper=Nodes[index].Upper; value.Link=Nodes[index].Link; return value; }
PrimitiveData ReadPrimitive(uint index)
{
    PrimitiveData value; value.A=Primitives[index].A; value.B=Primitives[index].B; value.C=Primitives[index].C;
    value.UVAB=Primitives[index].UVAB; value.UVC=Primitives[index].UVC; value.Identity=Primitives[index].Identity; return value;
}
CoverageData ReadCoverage(uint index)
{
    CoverageData value; value.Diffuse=CoverageMaterials[index].Diffuse; value.Opacity=CoverageMaterials[index].Opacity;
    value.Factor=CoverageMaterials[index].Factor; return value;
}
SurfaceData ReadAttributes(uint index)
{
    SurfaceData value; value.NA=Surfaces[index].NA; value.NB=Surfaces[index].NB; value.NC=Surfaces[index].NC;
    value.CA=Surfaces[index].CA; value.CB=Surfaces[index].CB; value.CC=Surfaces[index].CC; return value;
}
MaterialData ReadMaterial(uint index)
{
    MaterialData value; value.Base=Materials[index].Base; value.Ambient=Materials[index].Ambient;
    value.Specular=Materials[index].Specular; value.Emission=Materials[index].Emission;
    value.Diffuse=Materials[index].Diffuse; value.Normal=Materials[index].Normal; return value;
}
LightData ReadLight(uint index)
{
    LightData value; value.PositionType=Lights[index].PositionType; value.DirectionIntensity=Lights[index].DirectionIntensity;
    value.ColorShadow=Lights[index].ColorShadow; value.Area=Lights[index].Area; return value;
}
vec3 UnitVector(vec3 vector,vec3 fallback)
{
    float largest=max(abs(vector.x),max(abs(vector.y),abs(vector.z)));
    if (!(largest>0.0) || isnan(largest) || isinf(largest)) return fallback;
    vector/=largest; return vector*inversesqrt(dot(vector,vector));
}
void TexelFootprint(uvec4 image,vec2 uv,out uvec4 address,out vec2 blend)
{
    vec2 coordinate=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uint x=coordinate.x<0.0 ? image.y-1u : uint(floor(coordinate.x)),y=coordinate.y<0.0 ? image.z-1u : uint(floor(coordinate.y));
    uint right=x+1u==image.y ? 0u : x+1u,bottom=y+1u==image.z ? 0u : y+1u;
    address=image.x+uvec4(y*image.y+x,y*image.y+right,bottom*image.y+x,bottom*image.y+right); blend=fract(coordinate);
}
float SampleAlpha(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0; uvec4 a; vec2 b; TexelFootprint(image,uv,a,b);
    return mix(mix(AlphaTexels[a.x],AlphaTexels[a.y],b.x),mix(AlphaTexels[a.z],AlphaTexels[a.w],b.x),b.y);
}
vec3 ReadColorTexel(uint index)
{
    uint rowWidth=uint(textureSize(ColorAtlas,0).x);
    return texelFetch(ColorAtlas,ivec2(index%rowWidth,index/rowWidth),0).rgb;
}
vec3 SampleColor(uvec4 image,vec2 uv)
{
    if (image.w==0u) return vec3(1.0); uvec4 a; vec2 b; TexelFootprint(image,uv,a,b);
    return mix(mix(ReadColorTexel(a.x),ReadColorTexel(a.y),b.x),mix(ReadColorTexel(a.z),ReadColorTexel(a.w),b.x),b.y);
}
bool IntersectBounds(NodeData node,vec3 origin,vec3 ray,float maximum)
{
    float enter=0.0,leave=maximum;
    for (uint axis=0u;axis<3u;++axis)
    {
        if (ray[axis]==0.0) { if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false; }
        else
        {
            float lower=(node.Lower[axis]-origin[axis])/ray[axis],upper=(node.Upper[axis]-origin[axis])/ray[axis];
            enter=max(enter,min(lower,upper)); leave=min(leave,max(lower,upper)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(vec2 a,vec2 b)
{
    // ES 3.1 has no core precise qualifier. The conservative band handles the
    // two products' FP32 uncertainty, including drivers that contract them.
    float positive=a.x*b.y,negative=a.y*b.x,edge=positive-negative;
    return abs(edge)<=EdgeError*(abs(positive)+abs(negative)) ? 0.0 : edge;
}
bool IntersectTriangle(PrimitiveData triangle,vec3 origin,vec3 ray,float maximum,out float distance,out vec3 weights)
{
    distance=0.0; weights=vec3(0.0); uint z=abs(ray.x)>abs(ray.y) ? 0u : 1u; if (abs(ray.z)>abs(ray[z])) z=2u;
    uint x=(z+1u)%3u,y=(x+1u)%3u; vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(ray[x],ray[y])/ray[z];
    vec2 pa=vec2(a[x],a[y])-shear*a[z],pb=vec2(b[x],b[y])-shear*b[z],pc=vec2(c[x],c[y])-shear*c[z];
    vec3 edges=vec3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float determinant=edges.x+edges.y+edges.z; if (determinant==0.0) return false;
    distance=dot(edges,vec3(a[z],b[z],c[z]))/(determinant*ray[z]);
    if (!(distance>0.0 && distance<=maximum)) return false; weights=edges/determinant; return true;
}
struct RayHit { uint Index; float Distance; vec3 Weights; };
bool TraceGeometry(vec3 origin,vec3 ray,float maximum,bool shadow,out RayHit hit)
{
    hit=RayHit(0u,maximum,vec3(0.0)); uint cursor=0u; bool found=false;
    while (cursor<GeometryCounts.x)
    {
        NodeData node=ReadNode(cursor);
        if (!IntersectBounds(node,origin,ray,hit.Distance)) { cursor=node.Link.w; continue; }
        if (node.Link.z==0u) { cursor=node.Link.x; continue; }
        for (uint offset=0u;offset<node.Link.z;++offset)
        {
            uint index=node.Link.y+offset; PrimitiveData triangle=ReadPrimitive(index);
            if (shadow && (triangle.Identity.w&CasterFlag)==0u) continue;
            float distance; vec3 weights;
            if (!IntersectTriangle(triangle,origin,ray,hit.Distance,distance,weights)) continue;
            CoverageData material=ReadCoverage(triangle.Identity.x);
            if (shadow || material.Factor.y>0.0)
            {
                vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float alpha=clamp(dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Factor.x*
                    SampleAlpha(material.Diffuse,uv)*SampleAlpha(material.Opacity,uv),0.0,1.0);
                if (alpha<material.Factor.y || (shadow && alpha<=0.0)) continue;
            }
            hit=RayHit(index,distance,weights); found=true; if (shadow) return true;
        }
        cursor=node.Link.w;
    }
    return found;
}
struct ShadedPoint { vec3 Position,Normal,Geometric,Diffuse,Specular,Ambient,Emission; float Shininess,Opacity,IOR; };
ShadedPoint HitPoint(RayHit hit,vec3 origin,vec3 ray)
{
    PrimitiveData triangle=ReadPrimitive(hit.Index); SurfaceData surface=ReadAttributes(hit.Index);
    MaterialData material=ReadMaterial(triangle.Identity.x); vec3 weights=hit.Weights;
    vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
    vec3 edgeB=triangle.B.xyz-triangle.A.xyz,edgeC=triangle.C.xyz-triangle.A.xyz;
    ShadedPoint point; point.Position=origin+ray*hit.Distance;
    point.Geometric=UnitVector(cross(UnitVector(edgeB,vec3(1,0,0)),UnitVector(edgeC,vec3(0,1,0))),vec3(0,0,1));
    point.Normal=(Control.y&FlatFlag)!=0u ? point.Geometric : UnitVector(surface.NA.xyz*weights.x+surface.NB.xyz*weights.y+surface.NC.xyz*weights.z,point.Geometric);
    if (dot(point.Geometric,ray)>0.0) point.Normal=-point.Normal;
    if (material.Normal.w!=0u)
    {
        vec2 uvB=vec2(triangle.UVAB.z-triangle.UVAB.x,triangle.UVAB.y-triangle.UVAB.w);
        vec2 uvC=vec2(triangle.UVC.x-triangle.UVAB.x,triangle.UVAB.y-triangle.UVC.y);
        float determinant=uvB.x*uvC.y-uvB.y*uvC.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            vec3 tangent=(edgeB*uvC.y-edgeC*uvB.y)/determinant,sourceBitangent=(edgeC*uvB.x-edgeB*uvC.x)/determinant;
            vec3 helper=abs(point.Normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            tangent=UnitVector(tangent-point.Normal*dot(point.Normal,tangent),UnitVector(cross(helper,point.Normal),vec3(1,0,0)));
            vec3 bitangent=cross(point.Normal,tangent); bitangent*=dot(bitangent,sourceBitangent)<0.0 ? -1.0 : 1.0;
            vec3 mapped=SampleColor(material.Normal,uv)*2.0-1.0;
            point.Normal=UnitVector(tangent*mapped.x+bitangent*mapped.y+point.Normal*mapped.z,point.Normal);
        }
    }
    point.Diffuse=max((surface.CA.rgb*weights.x+surface.CB.rgb*weights.y+surface.CC.rgb*weights.z)*material.Base.rgb*SampleColor(material.Diffuse,uv),vec3(0));
    point.Specular=material.Specular.rgb; point.Shininess=material.Specular.w; point.Ambient=material.Ambient.rgb; point.Emission=material.Emission.rgb;
    point.Opacity=clamp(material.Base.w*(triangle.A.w*weights.x+triangle.B.w*weights.y+triangle.C.w*weights.z),0.0,1.0); point.IOR=material.Emission.w;
    return point;
}

uint Hash(uint value)
{
    uint state=value*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return (word>>PcgOutputShift)^word;
}
float Random(inout uint state)
{
    state=state*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return float(((word>>PcgOutputShift)^word)>>MantissaShift)*UnitInterval;
}
vec2 Random2(inout uint state) { float x=Random(state),y=Random(state); return vec2(x,y); }
vec3 Schlick(vec3 f0,float cosine)
{ return f0+(1.0-f0)*pow(1.0-clamp(cosine,0.0,1.0),FresnelPower); }
float Smith(float cosine,float alpha)
{ return 2.0*cosine/max(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine),CosineFloor); }
float Distribution(float nh,float alpha)
{ float a2=alpha*alpha,d=nh*nh*(a2-1.0)+1.0; return a2/(Pi*d*d); }
float SpecularChance(vec3 albedo)
{
    if (Control.x==Lambertian) return 0.0;
    vec3 f0=mix(vec3(Dielectric),albedo,MaterialParameters.y);
    vec3 diffuse=(1.0-f0)*(1.0-MaterialParameters.y)*albedo;
    float s=max(f0.x,max(f0.y,f0.z)),d=max(diffuse.x,max(diffuse.y,diffuse.z));
    return d<=0.0 ? 1.0 : clamp(s/(s+d),MinimumLobeProbability,1.0-MinimumLobeProbability);
}
vec3 Response(vec3 albedo,vec3 normal,vec3 view,vec3 incoming,out float pdf)
{
    pdf=0.0; float nv=dot(normal,view),nl=dot(normal,incoming);
    if (nv<=0.0 || nl<=0.0) return vec3(0);
    if (Control.x==Lambertian) { pdf=nl/Pi; return albedo*pdf; }
    vec3 h=UnitVector(view+incoming,normal);
    float alpha=max(MaterialParameters.x*MaterialParameters.x,AlphaFloor);
    float distribution=Distribution(clamp(dot(normal,h),0.0,1.0),alpha),mask=Smith(nv,alpha);
    pdf=mix(nl/Pi,distribution*mask/max(4.0*nv,CosineFloor),SpecularChance(albedo));
    vec3 f=Schlick(mix(vec3(Dielectric),albedo,MaterialParameters.y),dot(view,h));
    return (1.0-f)*(1.0-MaterialParameters.y)*albedo*(nl/Pi)+f*distribution*mask*Smith(nl,alpha)/max(4.0*nv,CosineFloor);
}
void Basis(vec3 normal,out vec3 tangent,out vec3 bitangent)
{
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    tangent=UnitVector(cross(helper,normal),vec3(1,0,0)); bitangent=cross(normal,tangent);
}
vec3 VisibleNormal(vec3 view,float alpha,vec2 xi)
{
    vec3 stretched=UnitVector(vec3(alpha*view.xy,view.z),vec3(0,0,1));
    float len=dot(stretched.xy,stretched.xy);
    vec3 tangent=len>0.0 ? vec3(-stretched.y,stretched.x,0)/sqrt(len) : vec3(1,0,0),bitangent=cross(stretched,tangent);
    float radius=sqrt(xi.x),angle=2.0*Pi*xi.y,x=radius*cos(angle),y=radius*sin(angle);
    y=mix(sqrt(max(0.0,1.0-x*x)),y,(1.0+stretched.z)/2.0);
    vec3 projected=tangent*x+bitangent*y+stretched*sqrt(max(0.0,1.0-x*x-y*y));
    return UnitVector(vec3(alpha*projected.xy,max(projected.z,0.0)),vec3(0,0,1));
}
bool Scatter(ShadedPoint point,vec3 view,inout uint state,out vec3 incoming,out vec3 weight)
{
    vec3 albedo=clamp(point.Diffuse,0.0,1.0),tangent,bitangent; Basis(point.Normal,tangent,bitangent);
    float choice=Random(state); vec2 xi=Random2(state);
    if (choice<SpecularChance(albedo))
    {
        vec3 localView=vec3(dot(view,tangent),dot(view,bitangent),dot(view,point.Normal));
        vec3 h=VisibleNormal(localView,max(MaterialParameters.x*MaterialParameters.x,AlphaFloor),xi);
        incoming=UnitVector(reflect(-view,tangent*h.x+bitangent*h.y+point.Normal*h.z),point.Normal);
    }
    else
    {
        float radius=sqrt(xi.x),angle=2.0*Pi*xi.y;
        incoming=tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+point.Normal*sqrt(1.0-xi.x);
    }
    float pdf; vec3 value=Response(albedo,point.Normal,view,incoming,pdf); weight=vec3(0);
    if (!(pdf>0.0) || dot(point.Geometric,view)*dot(point.Geometric,incoming)<=0.0) return false;
    weight=value/pdf; return !any(isnan(weight)) && !any(isinf(weight)) && any(greaterThan(weight,vec3(0)));
}
vec3 OffsetOrigin(ShadedPoint point,vec3 direction)
{ return point.Position+point.Geometric*(dot(point.Geometric,direction)>=0.0 ? RayParameters.y : -RayParameters.y); }
vec3 DirectLighting(ShadedPoint point,vec3 view,inout uint state)
{
    vec3 sum=vec3(0);
    for (uint index=0u;index<SurfaceCounts.w;++index)
    {
        LightData light=ReadLight(index); uint type=uint(light.PositionType.w);
        bool distant=type==Directional,rectangle=type!=Directional && type!=Point;
        vec3 source=light.PositionType.xyz,normal=UnitVector(light.DirectionIntensity.xyz,vec3(0,-1,0)); float measure=1.0;
        if (rectangle)
        {
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=UnitVector(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
            vec2 offset=Random2(state)-PixelCenter;
            source+=tangent*(offset.x*light.Area.x)+bitangent*(offset.y*light.Area.y); measure=light.Area.w>0.0 ? abs(light.Area.x*light.Area.y) : 1.0;
        }
        vec3 delta=distant ? UnitVector(light.DirectionIntensity.xyz,vec3(0,1,0))*RayParameters.z : source-point.Position;
        float squared=dot(delta,delta); if (!(squared>VectorEpsilon)) continue;
        float distance=sqrt(squared); vec3 incoming=delta/distance;
        if (dot(point.Geometric,view)*dot(point.Geometric,incoming)<=0.0) continue;
        float attenuation=1.0;
        if (!distant)
        {
            float window=1.0;
            if (light.Area.z>0.0) { float ratio=distance/(light.Area.z>0.0 ? light.Area.z : 1.0); window=clamp(1.0-ratio*ratio*ratio*ratio,0.0,1.0); window*=window; }
            attenuation=window/max(squared,DistanceSquaredFloor);
        }
        if (rectangle) attenuation*=measure*max(dot(normal,-incoming),0.0);
        if (!(attenuation>0.0)) continue;
        RayHit blocker;
        if (Control.z!=0u && light.ColorShadow.w>0.0 && TraceGeometry(OffsetOrigin(point,incoming),incoming,max(distance-RayParameters.y,0.0),true,blocker)) continue;
        float pdf; sum+=Response(clamp(point.Diffuse,0.0,1.0),point.Normal,view,incoming,pdf)*
            max(light.ColorShadow.rgb,vec3(0))*max(light.DirectionIntensity.w,0.0)*attenuation;
    }
    return sum;
}
bool WorldPoint(vec2 uv,float depth,out vec3 point)
{
    vec4 h=WorldFromClip*vec4(uv*2.0-1.0,depth,1); point=vec3(0);
    if (!(abs(h.w)>ProjectionEpsilon)) return false;
    point=h.xyz/h.w; return !any(isnan(point)) && !any(isinf(point));
}
bool PrimaryRay(vec2 pixel,out vec3 origin,out vec3 direction,out float maximum)
{
    vec2 uv=pixel*FrameSize.zw; vec3 middle,farPoint; direction=vec3(0); maximum=RayParameters.x;
    if (!WorldPoint(uv,0.0,origin) || !WorldPoint(uv,InteriorDepth,middle)) return false;
    direction=UnitVector(middle-origin,vec3(0)); if (!(dot(direction,direction)>0.0)) return false;
    if (WorldPoint(uv,1.0,farPoint)) maximum=min(maximum,length(farPoint-origin));
    return maximum>0.0;
}
vec4 Integrate(vec2 pixel,inout uint state)
{
    vec3 origin,direction; float maximum;
    if (!PrimaryRay(pixel,origin,direction,maximum)) return BackgroundColor;
    vec3 radiance=vec3(0),throughput=vec3(1); float coverage=1.0;
    for (uint depth=0u;depth<Samples.z;++depth)
    {
        RayHit hit;
        if (!TraceGeometry(origin,direction,maximum,false,hit))
        { radiance+=throughput*BackgroundColor.rgb; if (depth==0u) coverage=BackgroundColor.a; break; }
        ShadedPoint point=HitPoint(hit,origin,direction); vec3 view=-direction;
        if (dot(point.Normal,view)<=0.0) point.Normal=dot(point.Geometric,view)>0.0 ? point.Geometric : -point.Geometric;
        // Fresnel branching traces both dielectric reflection and transmission.
        // Thin opacity uses stochastic coverage, preserving the unweighted expectation.
        if (point.IOR>1.0)
        {
            bool entering=dot(direction,point.Geometric)<0.0;
            vec3 face=entering?point.Geometric:-point.Geometric;
            float eta=entering?1.0/point.IOR:point.IOR;
            float f0=(point.IOR-1.0)/(point.IOR+1.0);f0*=f0;
            float cosine=clamp(-dot(direction,face),0.0,1.0);
            float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
            vec3 transmitted=refract(direction,face,eta);
            if (dot(transmitted,transmitted)<0.000001 || Random(state)<fresnel) direction=reflect(direction,face);
            else { direction=transmitted;throughput*=clamp(point.Diffuse,0.0,1.0)*(eta*eta); }
            origin=OffsetOrigin(point,direction);maximum=RayParameters.x;continue;
        }
        if (point.Opacity<1.0 && Random(state)>point.Opacity)
        { origin=OffsetOrigin(point,direction);maximum=RayParameters.x;continue; }
        radiance+=throughput*(point.Emission+DirectLighting(point,view,state));
        if (depth+1u==Samples.z) break;
        vec3 weight; if (!Scatter(point,view,state,direction,weight)) break;
        throughput*=weight;
        if (depth+1u>=RouletteDepth)
        {
            float survival=clamp(max(throughput.x,max(throughput.y,throughput.z)),SurvivalFloor,SurvivalCeiling);
            if (Random(state)>=survival) break;
            throughput/=survival;
        }
        origin=OffsetOrigin(point,direction); maximum=RayParameters.x;
    }
    return vec4(radiance,coverage);
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
#ifdef EVANS_PATH_COMPOSE
    vec4 value=texelFetch(CommittedMean,pixel,0);
    OutColor=vec4(clamp(value.rgb,0.0,RadianceMaximum-RadianceSpacing),clamp(value.a,0.0,1.0));
    gl_FragDepth=texelFetch(PathDepth,pixel,0).r;
#else
    OutColor=Samples.x>0u ? texelFetch(CommittedMean,pixel,0) : vec4(0);
    // Random dimensions use top-left coordinates; native ES ray coordinates
    // remain bottom-left, including the mirrored subpixel offset.
    uvec2 seedPixel=uvec2(pixel.x,int(FrameSize.y)-1-pixel.y);
    for (uint path=0u;path<Samples.y;++path)
    {
        uint sequence=Samples.x+path,state=Hash(seedPixel.x^Hash(seedPixel.y^Hash(sequence^Samples.w)));
        vec2 jitter=Random2(state); jitter.y=1.0-jitter.y;
        vec4 value=Integrate(vec2(pixel)+jitter,state);
        OutColor+=(value-OutColor)/float(sequence+1u);
    }
    vec3 origin,direction; float maximum; RayHit center; OutDepth=BackgroundDepth;
    if (PrimaryRay(vec2(pixel)+PixelCenter,origin,direction,maximum) && TraceGeometry(origin,direction,maximum,false,center))
    {
        vec4 clip=ClipFromWorld*vec4(origin+direction*center.Distance,1);
        if (abs(clip.w)>ProjectionEpsilon) OutDepth=clamp(clip.z/clip.w,0.0,1.0);
    }
#endif
}
#endif
