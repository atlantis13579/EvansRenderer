#version 450
// Independent Vulkan GLSL, compiled in memory. Camera paths accumulate into
// FP32 means; a separate composition entry publishes HDR and center-ray depth.
// Each continuation evaluates both diffuse and GGX terms over the mixture PDF.
// Analytic lights use NEE; mesh emission uses path hits, without double counting.
// PBRT 4e: https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
// Heitz VNDF: https://jcgt.org/published/0007/04/01/
// Conservative FP32 edges: https://jcgt.org/published/0002/01/05/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(corner*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=0,std140,row_major) uniform PathParameters
{
    mat4 InverseViewProjection,ViewProjection;
    vec4 Viewport,Trace,Background,Material;
    uvec4 SceneCounts,ShadingCounts,Sampling,Options;
} Frame;
layout(set=0,binding=1) uniform texture2D PreviousMean;
layout(set=0,binding=2) uniform texture2D CenterDepth;
layout(set=0,binding=3) uniform sampler PointSampler;
layout(location=0) out vec4 OutColor;
#ifndef EVANS_PATH_COMPOSE
layout(location=1) out float OutDepth;
#endif
const uint Lambertian=1u,FlatNormals=1u,ShadowCaster=1u,DirectionalLight=0u,PointLight=1u;
const float Pi=3.141592653589793,PixelCenter=0.5,InteriorDepth=0.5,EmptyDepth=1.0;
const float MinimumW=1e-7,MinimumSquared=1e-16,TangentEpsilon=1e-8,ParallelAxis=0.999;
const float FloatEpsilon=1.1920928955078125e-7,EdgeUncertainty=3.0*FloatEpsilon/(1.0-3.0*FloatEpsilon);
const float MinimumAlpha=0.0025,MinimumCosine=1e-6,DielectricF0=0.04,SchlickExponent=5.0;
const float LightDistanceFloor=0.01,LobeFloor=0.05,SurvivalFloor=0.05,SurvivalCeiling=0.95;
const float HalfMaximum=65504.0,HalfSpacing=32.0;
const uint RouletteStart=3u;
// The integer constants define PCG RXS-M-XS, as analyzed by Jarzynski and
// Olano, Hash Functions for GPU Rendering: https://jcgt.org/published/0009/03/02/
const uint Multiplier=747796405u,Increment=2891336453u,OutputMultiplier=277803737u;
const uint DynamicShift=28u,MinimumShift=4u,OutputShift=22u,FloatShift=8u;
const float RandomScale=1.0/16777216.0;
struct BvhNode { vec4 Minimum,Maximum; uvec4 Links; };
struct Triangle { vec4 A,B,C,UVAB,UVC; uvec4 Metadata; };
struct CutoutMaterial { uvec4 Diffuse,Opacity; vec4 Coverage; };
struct Surface { vec4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct MaterialRecord { vec4 Base,AmbientColor,Specular,Emission; uvec4 Diffuse,Normal; };
struct LightRecord { vec4 PositionType,DirectionIntensity,ColorShadow,Area; };
layout(set=0,binding=4,std430) readonly buffer NodeStorage { BvhNode Nodes[]; };
layout(set=0,binding=5,std430) readonly buffer TriangleStorage { Triangle Triangles[]; };
layout(set=0,binding=6,std430) readonly buffer CutoutStorage { CutoutMaterial Cutouts[]; };
layout(set=0,binding=7,std430) readonly buffer AlphaStorage { float AlphaTexels[]; };
layout(set=0,binding=8,std430) readonly buffer SurfaceStorage { Surface Surfaces[]; };
layout(set=0,binding=9,std430) readonly buffer MaterialStorage { MaterialRecord Materials[]; };
// Preserve the Float4 atlas exactly without depending on maxStorageBufferRange.
layout(set=0,binding=10) uniform texture2D ColorAtlas;
layout(set=0,binding=11,std430) readonly buffer LightStorage { LightRecord Lights[]; };
bool Finite(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 Unit(vec3 value,vec3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (!(scale>0.0) || isinf(scale) || isnan(scale)) return fallback;
    value/=scale; return value*inversesqrt(dot(value,value));
}
void AtlasAddresses(uvec4 image,vec2 uv,out uvec4 addresses,out vec2 weight)
{
    vec2 position=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-PixelCenter;
    uvec2 a=uvec2(position.x<0.0 ? image.y-1u : uint(floor(position.x)),position.y<0.0 ? image.z-1u : uint(floor(position.y)));
    uvec2 b=uvec2(a.x+1u==image.y ? 0u : a.x+1u,a.y+1u==image.z ? 0u : a.y+1u);
    addresses=image.x+uvec4(a.y*image.y+a.x,a.y*image.y+b.x,b.y*image.y+a.x,b.y*image.y+b.x); weight=fract(position);
}
float Alpha(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0; uvec4 p; vec2 w; AtlasAddresses(image,uv,p,w);
    return mix(mix(AlphaTexels[p.x],AlphaTexels[p.y],w.x),mix(AlphaTexels[p.z],AlphaTexels[p.w],w.x),w.y);
}
vec3 ReadColorTexel(uint index)
{
    uint rowWidth=uint(textureSize(sampler2D(ColorAtlas,PointSampler),0).x);
    return texelFetch(sampler2D(ColorAtlas,PointSampler),ivec2(index%rowWidth,index/rowWidth),0).rgb;
}
vec3 Color(uvec4 image,vec2 uv)
{
    if (image.w==0u) return vec3(1.0); uvec4 p; vec2 w; AtlasAddresses(image,uv,p,w);
    return mix(mix(ReadColorTexel(p.x),ReadColorTexel(p.y),w.x),mix(ReadColorTexel(p.z),ReadColorTexel(p.w),w.x),w.y);
}
bool BoundsHit(BvhNode node,vec3 origin,vec3 direction,float limit)
{
    float enter=0.0,leave=limit;
    for (uint axis=0u;axis<3u;++axis)
    {
        if (direction[axis]==0.0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float Edge(vec2 a,vec2 b)
{ precise float positive=a.x*b.y,negative=a.y*b.x; precise float edge=positive-negative; return abs(edge)<=EdgeUncertainty*(abs(positive)+abs(negative)) ? 0.0 : edge; }
bool Intersect(Triangle triangle,vec3 origin,vec3 direction,float limit,out float distance,out vec3 barycentric)
{
    distance=0.0; barycentric=vec3(0.0);
    uint z=abs(direction.x)>abs(direction.y) ? 0u : 1u; if (abs(direction.z)>abs(direction[z])) z=2u;
    uint x=(z+1u)%3u,y=(x+1u)%3u;
    vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(direction[x],direction[y])/direction[z];
    precise vec2 pa=vec2(a[x],a[y])-a[z]*shear,pb=vec2(b[x],b[y])-b[z]*shear,pc=vec2(c[x],c[y])-c[z]*shear;
    vec3 edge=vec3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(lessThan(edge,vec3(0.0))) && any(greaterThan(edge,vec3(0.0)))) return false;
    float area=edge.x+edge.y+edge.z; if (area==0.0) return false;
    distance=dot(edge,vec3(a[z],b[z],c[z]))/(area*direction[z]);
    if (!(distance>0.0 && distance<=limit)) return false; barycentric=edge/area; return true;
}
struct Hit { uint Index; float Distance; vec3 Weights; };
bool TraceScene(vec3 origin,vec3 direction,float limit,bool shadow,out Hit closest)
{
    closest=Hit(0u,limit,vec3(0.0)); uint nodeIndex=0u; bool found=false;
    while (nodeIndex<Frame.SceneCounts.x)
    {
        BvhNode node=Nodes[nodeIndex];
        if (!BoundsHit(node,origin,direction,closest.Distance)) { nodeIndex=node.Links.w; continue; }
        if (node.Links.z==0u) { nodeIndex=node.Links.x; continue; }
        for (uint offset=0u;offset<node.Links.z;++offset)
        {
            uint index=node.Links.y+offset; Triangle triangle=Triangles[index];
            if (shadow && (triangle.Metadata.w&ShadowCaster)==0u) continue;
            float distance; vec3 weights;
            if (!Intersect(triangle,origin,direction,closest.Distance,distance,weights)) continue;
            CutoutMaterial material=Cutouts[triangle.Metadata.x];
            if (shadow || material.Coverage.y>0.0)
            {
                vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
                float coverage=clamp(dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x*
                    Alpha(material.Diffuse,uv)*Alpha(material.Opacity,uv),0.0,1.0);
                if (coverage<material.Coverage.y || (shadow && coverage<=0.0)) continue;
            }
            closest=Hit(index,distance,weights); found=true; if (shadow) return true;
        }
        nodeIndex=node.Links.w;
    }
    return found;
}
struct HitSurface { vec3 Position,Normal,Geometric,Diffuse,Specular,AmbientColor,Emission; float Shininess,Opacity,IOR; };
HitSurface ReadHit(Hit hit,vec3 origin,vec3 direction)
{
    Triangle triangle=Triangles[hit.Index]; Surface surface=Surfaces[hit.Index]; MaterialRecord material=Materials[triangle.Metadata.x];
    vec3 w=hit.Weights,ab=triangle.B.xyz-triangle.A.xyz,ac=triangle.C.xyz-triangle.A.xyz;
    vec2 uv=triangle.UVAB.xy*w.x+triangle.UVAB.zw*w.y+triangle.UVC.xy*w.z;
    HitSurface result; result.Position=origin+direction*hit.Distance;
    result.Geometric=Unit(cross(Unit(ab,vec3(1,0,0)),Unit(ac,vec3(0,1,0))),vec3(0,0,1));
    vec3 normal=(Frame.Options.y&FlatNormals)!=0u ? result.Geometric : Unit(surface.NormalA.xyz*w.x+surface.NormalB.xyz*w.y+surface.NormalC.xyz*w.z,result.Geometric);
    if (dot(result.Geometric,direction)>0.0) normal=-normal;
    if (material.Normal.w!=0u)
    {
        vec2 du=vec2(triangle.UVAB.z-triangle.UVAB.x,triangle.UVAB.y-triangle.UVAB.w);
        vec2 dv=vec2(triangle.UVC.x-triangle.UVAB.x,triangle.UVAB.y-triangle.UVC.y);
        float determinant=du.x*dv.y-du.y*dv.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            vec3 tangent=(ab*dv.y-ac*du.y)/determinant,sourceBitangent=(ac*du.x-ab*dv.x)/determinant;
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            tangent=Unit(tangent-normal*dot(tangent,normal),Unit(cross(helper,normal),vec3(1,0,0)));
            vec3 crossed=cross(normal,tangent),bitangent=crossed*(dot(crossed,sourceBitangent)<0.0 ? -1.0 : 1.0);
            vec3 mapped=Color(material.Normal,uv)*2.0-1.0;
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=max((surface.ColorA.rgb*w.x+surface.ColorB.rgb*w.y+surface.ColorC.rgb*w.z)*material.Base.rgb*Color(material.Diffuse,uv),vec3(0.0));
    result.Specular=material.Specular.rgb; result.Shininess=material.Specular.w; result.AmbientColor=material.AmbientColor.rgb; result.Emission=material.Emission.rgb;
    result.Opacity=clamp(material.Base.w*(triangle.A.w*w.x+triangle.B.w*w.y+triangle.C.w*w.z),0.0,1.0); result.IOR=material.Emission.w;
    return result;
}

uint Hash(uint value)
{
    uint state=value*Multiplier+Increment;
    uint word=((state>>((state>>DynamicShift)+MinimumShift))^state)*OutputMultiplier;
    return (word>>OutputShift)^word;
}
float Random(inout uint state)
{
    state=state*Multiplier+Increment;
    uint word=((state>>((state>>DynamicShift)+MinimumShift))^state)*OutputMultiplier;
    return float(((word>>OutputShift)^word)>>FloatShift)*RandomScale;
}
vec2 Random2(inout uint state) { float x=Random(state),y=Random(state); return vec2(x,y); }
vec3 Fresnel(vec3 f0,float cosine)
{ return f0+(1.0-f0)*pow(1.0-clamp(cosine,0.0,1.0),SchlickExponent); }
float Smith(float cosine,float alpha)
{ return 2.0*cosine/max(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine),MinimumCosine); }
float Distribution(float nh,float alpha)
{ float a2=alpha*alpha,d=nh*nh*(a2-1.0)+1.0; return a2/(Pi*d*d); }
float SpecularProbability(vec3 albedo)
{
    if (Frame.Options.x==Lambertian) return 0.0;
    vec3 f0=mix(vec3(DielectricF0),albedo,Frame.Material.y),kd=(1.0-f0)*(1.0-Frame.Material.y)*albedo;
    float s=max(f0.x,max(f0.y,f0.z)),d=max(kd.x,max(kd.y,kd.z));
    return d<=0.0 ? 1.0 : clamp(s/(s+d),LobeFloor,1.0-LobeFloor);
}
vec3 EvaluateBSDF(vec3 albedo,vec3 normal,vec3 view,vec3 incoming,out float pdf)
{
    pdf=0.0; float nv=dot(normal,view),nl=dot(normal,incoming);
    if (nv<=0.0 || nl<=0.0) return vec3(0);
    if (Frame.Options.x==Lambertian) { pdf=nl/Pi; return albedo*pdf; }
    vec3 h=Unit(view+incoming,normal); float alpha=max(Frame.Material.x*Frame.Material.x,MinimumAlpha);
    float distribution=Distribution(clamp(dot(normal,h),0.0,1.0),alpha),mask=Smith(nv,alpha);
    pdf=mix(nl/Pi,distribution*mask/max(4.0*nv,MinimumCosine),SpecularProbability(albedo));
    vec3 f=Fresnel(mix(vec3(DielectricF0),albedo,Frame.Material.y),dot(view,h));
    return (1.0-f)*(1.0-Frame.Material.y)*albedo*(nl/Pi)+f*distribution*mask*Smith(nl,alpha)/max(4.0*nv,MinimumCosine);
}
void Basis(vec3 normal,out vec3 tangent,out vec3 bitangent)
{
    vec3 helper=abs(normal.x)<abs(normal.y) ? vec3(1,0,0) : vec3(0,1,0);
    tangent=Unit(cross(helper,normal),vec3(1,0,0)); bitangent=cross(normal,tangent);
}
vec3 VisibleNormal(vec3 view,float alpha,vec2 xi)
{
    vec3 stretched=Unit(vec3(view.xy*alpha,view.z),vec3(0,0,1));
    float len=dot(stretched.xy,stretched.xy);
    vec3 t=len>0.0 ? vec3(-stretched.y,stretched.x,0)/sqrt(len) : vec3(1,0,0),b=cross(stretched,t);
    float radius=sqrt(xi.x),angle=2.0*Pi*xi.y,x=radius*cos(angle),y=radius*sin(angle);
    y=mix(sqrt(max(0.0,1.0-x*x)),y,(1.0+stretched.z)/2.0);
    vec3 projected=t*x+b*y+stretched*sqrt(max(0.0,1.0-x*x-y*y));
    return Unit(vec3(projected.xy*alpha,max(projected.z,0.0)),vec3(0,0,1));
}
bool Scatter(HitSurface surface,vec3 view,inout uint state,out vec3 incoming,out vec3 weight)
{
    vec3 albedo=clamp(surface.Diffuse,0.0,1.0),tangent,bitangent; Basis(surface.Normal,tangent,bitangent);
    float choice=Random(state); vec2 xi=Random2(state);
    if (choice<SpecularProbability(albedo))
    {
        vec3 localView=vec3(dot(view,tangent),dot(view,bitangent),dot(view,surface.Normal));
        vec3 h=VisibleNormal(localView,max(Frame.Material.x*Frame.Material.x,MinimumAlpha),xi);
        incoming=Unit(reflect(-view,tangent*h.x+bitangent*h.y+surface.Normal*h.z),surface.Normal);
    }
    else
    {
        float radius=sqrt(xi.x),angle=2.0*Pi*xi.y;
        incoming=tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+surface.Normal*sqrt(1.0-xi.x);
    }
    float pdf; vec3 value=EvaluateBSDF(albedo,surface.Normal,view,incoming,pdf); weight=vec3(0);
    if (!(pdf>0.0) || dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0.0) return false;
    weight=value/pdf; return !any(isnan(weight)) && !any(isinf(weight)) && any(greaterThan(weight,vec3(0)));
}
vec3 OffsetOrigin(HitSurface surface,vec3 direction)
{ return surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0.0 ? Frame.Trace.y : -Frame.Trace.y); }
vec3 DirectLighting(HitSurface surface,vec3 view,inout uint state)
{
    vec3 sum=vec3(0);
    for (uint index=0u;index<Frame.ShadingCounts.w;++index)
    {
        LightRecord light=Lights[index]; uint type=uint(light.PositionType.w);
        bool distant=type==DirectionalLight,rectangle=type!=DirectionalLight && type!=PointLight;
        vec3 emitter=light.PositionType.xyz,normal=Unit(light.DirectionIntensity.xyz,vec3(0,-1,0)); float measure=1.0;
        if (rectangle)
        {
            vec3 helper=abs(normal.y)<ParallelAxis ? vec3(0,1,0) : vec3(1,0,0);
            vec3 tangent=Unit(cross(helper,normal),vec3(1,0,0)),bitangent=cross(normal,tangent);
            vec2 uv=Random2(state)-PixelCenter;
            emitter+=tangent*(uv.x*light.Area.x)+bitangent*(uv.y*light.Area.y); measure=light.Area.w>0.0 ? abs(light.Area.x*light.Area.y) : 1.0;
        }
        vec3 delta=distant ? Unit(light.DirectionIntensity.xyz,vec3(0,1,0))*Frame.Trace.z : emitter-surface.Position;
        float squared=dot(delta,delta); if (!(squared>MinimumSquared)) continue;
        float distance=sqrt(squared); vec3 incoming=delta/distance;
        if (dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0.0) continue;
        float attenuation=1.0;
        if (!distant)
        {
            float window=1.0;
            if (light.Area.z>0.0) { float ratio=distance/(light.Area.z>0.0 ? light.Area.z : 1.0); window=clamp(1.0-ratio*ratio*ratio*ratio,0.0,1.0); window*=window; }
            attenuation=window/max(squared,LightDistanceFloor);
        }
        if (rectangle) attenuation*=measure*max(dot(normal,-incoming),0.0);
        if (!(attenuation>0.0)) continue;
        Hit blocker;
        if (Frame.Options.z!=0u && light.ColorShadow.w>0.0 && TraceScene(OffsetOrigin(surface,incoming),incoming,max(distance-Frame.Trace.y,0.0),true,blocker)) continue;
        float pdf; sum+=EvaluateBSDF(clamp(surface.Diffuse,0.0,1.0),surface.Normal,view,incoming,pdf)*
            max(light.ColorShadow.rgb,vec3(0))*max(light.DirectionIntensity.w,0.0)*attenuation;
    }
    return sum;
}
bool Unproject(vec2 uv,float depth,out vec3 point)
{
    vec4 h=Frame.InverseViewProjection*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0); point=vec3(0);
    if (!(abs(h.w)>MinimumW)) return false;
    point=h.xyz/h.w; return !any(isnan(point)) && !any(isinf(point));
}
bool PrimaryRay(vec2 pixel,out vec3 origin,out vec3 direction,out float maximum)
{
    vec2 uv=pixel*Frame.Viewport.zw; vec3 middle,farPoint; direction=vec3(0); maximum=Frame.Trace.x;
    if (!Unproject(uv,0.0,origin) || !Unproject(uv,InteriorDepth,middle)) return false;
    direction=Unit(middle-origin,vec3(0)); if (!(dot(direction,direction)>0.0)) return false;
    if (Unproject(uv,1.0,farPoint)) maximum=min(maximum,length(farPoint-origin));
    return maximum>0.0;
}
vec4 PathSample(vec2 pixel,inout uint state)
{
    vec3 origin,direction; float maximum;
    if (!PrimaryRay(pixel,origin,direction,maximum)) return Frame.Background;
    vec3 radiance=vec3(0),throughput=vec3(1); float coverage=1.0;
    for (uint depth=0u;depth<Frame.Sampling.z;++depth)
    {
        Hit hit;
        if (!TraceScene(origin,direction,maximum,false,hit))
        { radiance+=throughput*Frame.Background.rgb; if (depth==0u) coverage=Frame.Background.a; break; }
        HitSurface surface=ReadHit(hit,origin,direction); vec3 view=-direction;
        if (dot(surface.Normal,view)<=0.0) surface.Normal=dot(surface.Geometric,view)>0.0 ? surface.Geometric : -surface.Geometric;
        // Fresnel branching traces both dielectric reflection and transmission.
        // Thin opacity uses stochastic coverage, preserving the unweighted expectation.
        if (surface.IOR>1.0)
        {
            bool entering=dot(direction,surface.Geometric)<0.0;
            vec3 face=entering?surface.Geometric:-surface.Geometric;
            float eta=entering?1.0/surface.IOR:surface.IOR;
            float f0=(surface.IOR-1.0)/(surface.IOR+1.0);f0*=f0;
            float cosine=clamp(-dot(direction,face),0.0,1.0);
            float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
            vec3 transmitted=refract(direction,face,eta);
            if (dot(transmitted,transmitted)<0.000001 || Random(state)<fresnel) direction=reflect(direction,face);
            else { direction=transmitted;throughput*=clamp(surface.Diffuse,0.0,1.0)*(eta*eta); }
            origin=OffsetOrigin(surface,direction);maximum=Frame.Trace.x;continue;
        }
        if (surface.Opacity<1.0 && Random(state)>surface.Opacity)
        { origin=OffsetOrigin(surface,direction);maximum=Frame.Trace.x;continue; }
        radiance+=throughput*(surface.Emission+DirectLighting(surface,view,state));
        if (depth+1u==Frame.Sampling.z) break;
        vec3 weight; if (!Scatter(surface,view,state,direction,weight)) break;
        throughput*=weight;
        if (depth+1u>=RouletteStart)
        {
            float survival=clamp(max(throughput.x,max(throughput.y,throughput.z)),SurvivalFloor,SurvivalCeiling);
            if (Random(state)>=survival) break;
            throughput/=survival;
        }
        origin=OffsetOrigin(surface,direction); maximum=Frame.Trace.x;
    }
    return vec4(radiance,coverage);
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
#ifdef EVANS_PATH_COMPOSE
    vec4 value=texelFetch(sampler2D(PreviousMean,PointSampler),pixel,0);
    OutColor=vec4(clamp(value.rgb,0.0,HalfMaximum-HalfSpacing),clamp(value.a,0.0,1.0));
    gl_FragDepth=texelFetch(sampler2D(CenterDepth,PointSampler),pixel,0).r;
#else
    OutColor=Frame.Sampling.x>0u ? texelFetch(sampler2D(PreviousMean,PointSampler),pixel,0) : vec4(0);
    for (uint path=0u;path<Frame.Sampling.y;++path)
    {
        uint sequence=Frame.Sampling.x+path,state=Hash(uint(pixel.x)^Hash(uint(pixel.y)^Hash(sequence^Frame.Sampling.w)));
        vec2 jitter=Random2(state); vec4 value=PathSample(vec2(pixel)+jitter,state);
        OutColor+=(value-OutColor)/float(sequence+1u);
    }
    vec3 origin,direction; float maximum; Hit center; OutDepth=EmptyDepth;
    if (PrimaryRay(vec2(pixel)+PixelCenter,origin,direction,maximum) && TraceScene(origin,direction,maximum,false,center))
    {
        vec4 projected=Frame.ViewProjection*vec4(origin+direction*center.Distance,1);
        if (abs(projected.w)>MinimumW) OutDepth=clamp(projected.z/projected.w,0.0,1.0);
    }
#endif
}
#endif
