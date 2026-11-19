// Independent DX11 camera-path integrator. Trace stores an FP32 running mean
// and center-ray depth; PSCompose publishes HDR color and opaque depth.
// Cosine/GGX VNDF mixture sampling uses the complete mixture PDF. Explicit
// analytic lights and intersectable mesh emission are disjoint emitter sets.
// PBRT 4e: https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
// Heitz 2018: https://jcgt.org/published/0007/04/01/
// Conservative FP32 triangle edges: https://jcgt.org/published/0002/01/05/
cbuffer PathFrame : register(b0)
{
    row_major float4x4 InverseViewProjection,ViewProjection;
    float4 Viewport,Trace,Background,Material;
    uint4 SceneCounts,ShadingCounts,Sampling,Options;
};
Texture2D<float4> PreviousMean : register(t0);
Texture2D<float> CenterDepth : register(t1);
static const uint Lambertian=1,FlatNormals=1,ShadowCaster=1,DirectionalLight=0,PointLight=1;
static const float Pi=3.141592653589793,PixelCenter=0.5,InteriorDepth=0.5,EmptyDepth=1;
static const float MinimumW=1e-7,MinimumSquared=1e-16,TangentEpsilon=1e-8,ParallelAxis=0.999;
static const float Binary32Epsilon=1.1920928955078125e-7,EdgeError=3*Binary32Epsilon/(1-3*Binary32Epsilon);
static const float MinimumAlpha=0.0025,MinimumCosine=1e-6,DielectricF0=0.04,SchlickPower=5;
static const float LightDistanceFloor=0.01,MinimumLobeProbability=0.05;
static const float RouletteFloor=0.05,RouletteCeiling=0.95,HalfMaximum=65504,HalfSpacing=32;
static const uint RouletteStartDepth=3;
// PCG RXS-M-XS permutation constants, not material tuning values.
// Jarzynski and Olano, Hash Functions for GPU Rendering (2020).
// https://jcgt.org/published/0009/03/02/
static const uint PcgMultiplier=747796405u,PcgIncrement=2891336453u,PcgOutputMultiplier=277803737u;
static const uint PcgDynamicShift=28,PcgMinimumShift=4,PcgOutputShift=22,FloatDiscardBits=8;
static const float UnitIntervalScale=1.0/16777216.0;
struct Node { float4 Lower,Upper; uint4 Links; };
struct Triangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct CoverageMaterial { uint4 Diffuse,Opacity; float4 Alpha; };
struct Surface { float4 NormalA,NormalB,NormalC,ColorA,ColorB,ColorC; };
struct SurfaceMaterial { float4 Base,AmbientColor,Specular,Emission; uint4 Diffuse,Normal; };
struct Light { float4 PositionType,DirectionIntensity,ColorShadow,Area; };
StructuredBuffer<Node> Nodes : register(t2);
StructuredBuffer<Triangle> Triangles : register(t3);
StructuredBuffer<CoverageMaterial> CoverageMaterials : register(t4);
StructuredBuffer<float> AlphaTexels : register(t5);
StructuredBuffer<Surface> Surfaces : register(t6);
StructuredBuffer<SurfaceMaterial> SurfaceMaterials : register(t7);
StructuredBuffer<float4> ColorTexels : register(t8);
StructuredBuffer<Light> Lights : register(t9);
struct ScreenVertex { float4 Position:SV_POSITION; };
ScreenVertex VSMain(uint index:SV_VertexID)
{ float2 p=float2((index<<1)&2,index&2); ScreenVertex v; v.Position=float4(p*2-1,0,1); return v; }
float3 Unit(float3 value,float3 fallback)
{
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z)));
    if (!(scale>0) || !isfinite(scale)) return fallback;
    value/=scale; return value*rsqrt(dot(value,value));
}
void TexelCoordinates(uint4 image,float2 uv,out uint4 indices,out float2 blend)
{
    float2 p=frac(float2(uv.x,-uv.y))*image.yz-PixelCenter;
    uint2 a=uint2(p.x<0 ? image.y-1 : uint(floor(p.x)),p.y<0 ? image.z-1 : uint(floor(p.y)));
    uint2 b=uint2(a.x+1==image.y ? 0 : a.x+1,a.y+1==image.z ? 0 : a.y+1);
    indices=image.x+uint4(a.y*image.y+a.x,a.y*image.y+b.x,b.y*image.y+a.x,b.y*image.y+b.x); blend=frac(p);
}
float Alpha(uint4 image,float2 uv)
{
    if (!image.w) return 1; uint4 p; float2 t; TexelCoordinates(image,uv,p,t);
    return lerp(lerp(AlphaTexels[p.x],AlphaTexels[p.y],t.x),lerp(AlphaTexels[p.z],AlphaTexels[p.w],t.x),t.y);
}
float3 TextureRGB(uint4 image,float2 uv)
{
    if (!image.w) return 1; uint4 p; float2 t; TexelCoordinates(image,uv,p,t);
    return lerp(lerp(ColorTexels[p.x].rgb,ColorTexels[p.y].rgb,t.x),lerp(ColorTexels[p.z].rgb,ColorTexels[p.w].rgb,t.x),t.y);
}
bool Box(Node node,float3 origin,float3 ray,float limit)
{
    float nearDistance=0,farDistance=limit;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (ray[axis]==0) { if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false; }
        else
        {
            float a=(node.Lower[axis]-origin[axis])/ray[axis],b=(node.Upper[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(a,b)); farDistance=min(farDistance,max(a,b));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float Edge(float2 a,float2 b)
{ precise float x=a.x*b.y,y=a.y*b.x; precise float e=x-y; return abs(e)<=EdgeError*(abs(x)+abs(y)) ? 0 : e; }
bool Intersect(Triangle tri,float3 origin,float3 ray,float limit,out float distance,out float3 weights)
{
    distance=0; weights=0;
    uint z=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3;
    float3 a=tri.A.xyz-origin,b=tri.B.xyz-origin,c=tri.C.xyz-origin;
    float2 shear=float2(ray[x],ray[y])/ray[z];
    precise float2 pa=float2(a[x],a[y])-shear*a[z],pb=float2(b[x],b[y])-shear*b[z],pc=float2(c[x],c[y])-shear*c[z];
    float3 e=float3(Edge(pb,pc),Edge(pc,pa),Edge(pa,pb));
    if (any(e<0) && any(e>0)) return false;
    float determinant=e.x+e.y+e.z; if (determinant==0) return false;
    distance=dot(e,float3(a[z],b[z],c[z]))/(determinant*ray[z]);
    if (!(distance>0 && distance<=limit)) return false;
    weights=e/determinant; return true;
}
struct Hit { uint Index; float Distance; float3 Weights; };
bool TraceScene(float3 origin,float3 ray,float limit,bool shadow,out Hit hit)
{
    hit.Index=0; hit.Distance=limit; hit.Weights=0; bool found=false; uint nodeIndex=0;
    [loop] while (nodeIndex<SceneCounts.x)
    {
        Node node=Nodes[nodeIndex];
        if (!Box(node,origin,ray,hit.Distance)) { nodeIndex=node.Links.w; continue; }
        if (!node.Links.z) { nodeIndex=node.Links.x; continue; }
        [loop] for (uint offset=0;offset<node.Links.z;++offset)
        {
            uint index=node.Links.y+offset; Triangle tri=Triangles[index]; float distance; float3 weights;
            if (shadow && !(tri.Metadata.w&ShadowCaster)) continue;
            if (!Intersect(tri,origin,ray,hit.Distance,distance,weights)) continue;
            CoverageMaterial material=CoverageMaterials[tri.Metadata.x];
            if (shadow || material.Alpha.y>0.0)
            {
                float2 uv=tri.UVAB.xy*weights.x+tri.UVAB.zw*weights.y+tri.UVC.xy*weights.z;
                float coverage=saturate(dot(weights,float3(tri.A.w,tri.B.w,tri.C.w))*material.Alpha.x*Alpha(material.Diffuse,uv)*Alpha(material.Opacity,uv));
                if (coverage<material.Alpha.y || (shadow && coverage<=0.0)) continue;
            }
            hit.Index=index; hit.Distance=distance; hit.Weights=weights; found=true;
            if (shadow) return true;
        }
        nodeIndex=node.Links.w;
    }
    return found;
}
struct HitSurface { float3 Position,Normal,Geometric,Diffuse,Specular,Emission,AmbientColor; float Shininess,Opacity,IOR; };
HitSurface ReadHit(Hit hit,float3 origin,float3 ray)
{
    Triangle t=Triangles[hit.Index]; Surface s=Surfaces[hit.Index]; SurfaceMaterial m=SurfaceMaterials[t.Metadata.x];
    float3 w=hit.Weights; float2 uv=t.UVAB.xy*w.x+t.UVAB.zw*w.y+t.UVC.xy*w.z;
    float3 edgeB=t.B.xyz-t.A.xyz,edgeC=t.C.xyz-t.A.xyz;
    HitSurface result; result.Position=origin+ray*hit.Distance;
    result.Geometric=Unit(cross(Unit(edgeB,float3(1,0,0)),Unit(edgeC,float3(0,1,0))),float3(0,0,1));
    float3 normal=(Options.y&FlatNormals) ? result.Geometric : Unit(s.NormalA.xyz*w.x+s.NormalB.xyz*w.y+s.NormalC.xyz*w.z,result.Geometric);
    if (dot(result.Geometric,ray)>0) normal=-normal;
    if (m.Normal.w)
    {
        float2 uvB=float2(t.UVAB.z-t.UVAB.x,t.UVAB.y-t.UVAB.w),uvC=float2(t.UVC.x-t.UVAB.x,t.UVAB.y-t.UVC.y);
        float determinant=uvB.x*uvC.y-uvB.y*uvC.x;
        if (abs(determinant)>=TangentEpsilon)
        {
            float3 tangent=(edgeB*uvC.y-edgeC*uvB.y)/determinant,sourceBitangent=(edgeC*uvB.x-edgeB*uvC.x)/determinant;
            float3 helper=abs(normal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            tangent=Unit(tangent-normal*dot(normal,tangent),Unit(cross(helper,normal),float3(1,0,0)));
            float3 crossed=cross(normal,tangent),bitangent=crossed*(dot(crossed,sourceBitangent)<0 ? -1 : 1);
            float3 mapped=TextureRGB(m.Normal,uv)*2-1;
            normal=Unit(tangent*mapped.x+bitangent*mapped.y+normal*mapped.z,normal);
        }
    }
    result.Normal=normal; result.Diffuse=max((s.ColorA.rgb*w.x+s.ColorB.rgb*w.y+s.ColorC.rgb*w.z)*m.Base.rgb*TextureRGB(m.Diffuse,uv),0);
    result.Specular=m.Specular.rgb; result.Shininess=m.Specular.w; result.Emission=m.Emission.rgb; result.AmbientColor=m.AmbientColor.rgb;
    result.Opacity=clamp(m.Base.w*(t.A.w*w.x+t.B.w*w.y+t.C.w*w.z),0.0,1.0); result.IOR=m.Emission.w;
    return result;
}

uint Permute(uint value)
{
    uint state=value*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return (word>>PcgOutputShift)^word;
}
float Random(inout uint state)
{
    state=state*PcgMultiplier+PcgIncrement;
    uint word=((state>>((state>>PcgDynamicShift)+PcgMinimumShift))^state)*PcgOutputMultiplier;
    return float(((word>>PcgOutputShift)^word)>>FloatDiscardBits)*UnitIntervalScale;
}
float2 Random2(inout uint state)
{ float a=Random(state),b=Random(state); return float2(a,b); }
float3 Fresnel(float3 f0,float cosine)
{ return f0+(1-f0)*pow(1-saturate(cosine),SchlickPower); }
float SmithG1(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),MinimumCosine); }
float Distribution(float cosine,float alpha)
{ float a2=alpha*alpha,d=cosine*cosine*(a2-1)+1; return a2/(Pi*d*d); }
float SpecularProbability(float3 albedo)
{
    if (Options.x==Lambertian) return 0;
    float3 f0=lerp(DielectricF0.xxx,albedo,Material.y);
    float specular=max(f0.x,max(f0.y,f0.z));
    float3 kd=(1-f0)*(1-Material.y)*albedo;
    float diffuse=max(kd.x,max(kd.y,kd.z));
    if (diffuse<=0) return 1;
    return clamp(specular/(specular+diffuse),MinimumLobeProbability,1-MinimumLobeProbability);
}
float3 EvaluateBSDF(float3 albedo,float3 normal,float3 view,float3 incoming,out float pdf)
{
    pdf=0; float nv=dot(normal,view),nl=dot(normal,incoming);
    if (nv<=0 || nl<=0) return 0;
    if (Options.x==Lambertian) { pdf=nl/Pi; return albedo*(nl/Pi); }
    float3 halfVector=Unit(view+incoming,normal);
    float nh=saturate(dot(normal,halfVector)),alpha=max(Material.x*Material.x,MinimumAlpha);
    float distribution=Distribution(nh,alpha),viewMask=SmithG1(nv,alpha);
    float probability=SpecularProbability(albedo);
    pdf=lerp(nl/Pi,distribution*viewMask/max(4*nv,MinimumCosine),probability);
    float3 f=Fresnel(lerp(DielectricF0.xxx,albedo,Material.y),dot(view,halfVector));
    return (1-f)*(1-Material.y)*albedo*(nl/Pi)+f*distribution*viewMask*SmithG1(nl,alpha)/max(4*nv,MinimumCosine);
}
void Basis(float3 normal,out float3 tangent,out float3 bitangent)
{
    float3 helper=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    tangent=Unit(cross(helper,normal),float3(1,0,0)); bitangent=cross(normal,tangent);
}
float3 VisibleNormal(float3 view,float alpha,float2 xi)
{
    float3 stretched=Unit(float3(alpha*view.xy,view.z),float3(0,0,1));
    float lens=dot(stretched.xy,stretched.xy);
    float3 t=lens>0 ? float3(-stretched.y,stretched.x,0)/sqrt(lens) : float3(1,0,0);
    float3 b=cross(stretched,t);
    float radius=sqrt(xi.x),angle=2*Pi*xi.y;
    float x=radius*cos(angle),y=radius*sin(angle),blend=(1+stretched.z)/2;
    y=lerp(sqrt(max(0,1-x*x)),y,blend);
    float3 normal=t*x+b*y+stretched*sqrt(max(0,1-x*x-y*y));
    return Unit(float3(alpha*normal.xy,max(normal.z,0)),float3(0,0,1));
}
bool Scatter(HitSurface surface,float3 view,inout uint randomState,out float3 incoming,out float3 weight)
{
    float3 albedo=saturate(surface.Diffuse),tangent,bitangent;
    Basis(surface.Normal,tangent,bitangent); float choice=Random(randomState); float2 xi=Random2(randomState);
    if (choice<SpecularProbability(albedo))
    {
        float3 localView=float3(dot(view,tangent),dot(view,bitangent),dot(view,surface.Normal));
        float3 h=VisibleNormal(localView,max(Material.x*Material.x,MinimumAlpha),xi);
        incoming=Unit(reflect(-view,tangent*h.x+bitangent*h.y+surface.Normal*h.z),surface.Normal);
    }
    else
    {
        float radius=sqrt(xi.x),angle=2*Pi*xi.y;
        incoming=tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+surface.Normal*sqrt(1-xi.x);
    }
    float pdf; float3 value=EvaluateBSDF(albedo,surface.Normal,view,incoming,pdf); weight=0;
    if (!(pdf>0) || dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0) return false;
    weight=value/pdf; return all(isfinite(weight)) && any(weight>0);
}
float3 OffsetOrigin(HitSurface surface,float3 direction)
{ return surface.Position+surface.Geometric*(dot(surface.Geometric,direction)>=0 ? Trace.y : -Trace.y); }
float3 DirectLighting(HitSurface surface,float3 view,inout uint randomState)
{
    float3 sum=0;
    [loop] for (uint index=0;index<ShadingCounts.w;++index)
    {
        Light light=Lights[index]; uint type=uint(light.PositionType.w);
        bool distant=type==DirectionalLight,rectangle=type!=DirectionalLight && type!=PointLight;
        float3 source=light.PositionType.xyz,emitterNormal=Unit(light.DirectionIntensity.xyz,float3(0,-1,0));
        float measure=1;
        if (rectangle)
        {
            float3 helper=abs(emitterNormal.y)<ParallelAxis ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Unit(cross(helper,emitterNormal),float3(1,0,0)),bitangent=cross(emitterNormal,tangent);
            float2 offset=Random2(randomState)-PixelCenter;
            source+=tangent*(offset.x*light.Area.x)+bitangent*(offset.y*light.Area.y);
            // Legacy rectangles store normalized power; physical rectangles store radiance.
            measure=light.Area.w>0 ? abs(light.Area.x*light.Area.y) : 1;
        }
        float3 displacement=distant ? Unit(light.DirectionIntensity.xyz,float3(0,1,0))*Trace.z : source-surface.Position;
        float squared=dot(displacement,displacement); if (!(squared>MinimumSquared)) continue;
        float distance=sqrt(squared); float3 incoming=displacement/distance;
        if (dot(surface.Geometric,incoming)*dot(surface.Geometric,view)<=0) continue;
        float attenuation=1;
        if (!distant)
        {
            float window=1;
            if (light.Area.z>0) { float ratio=distance/(light.Area.z>0 ? light.Area.z : 1.0); window=saturate(1-ratio*ratio*ratio*ratio); window*=window; }
            attenuation=window/max(squared,LightDistanceFloor);
        }
        if (rectangle) attenuation*=max(dot(emitterNormal,-incoming),0)*measure;
        if (!(attenuation>0)) continue;
        Hit blocker;
        if (Options.z!=0 && light.ColorShadow.w>0 && TraceScene(OffsetOrigin(surface,incoming),incoming,max(distance-Trace.y,0),true,blocker)) continue;
        float pdf; sum+=EvaluateBSDF(saturate(surface.Diffuse),surface.Normal,view,incoming,pdf)*
            max(light.ColorShadow.rgb,0)*max(light.DirectionIntensity.w,0)*attenuation;
    }
    return sum;
}
bool Unproject(float2 uv,float depth,out float3 surfacePoint)
{
    float4 homogeneous=mul(InverseViewProjection,float4(uv.x*2-1,1-uv.y*2,depth,1)); surfacePoint=0;
    if (!(abs(homogeneous.w)>MinimumW)) return false;
    surfacePoint=homogeneous.xyz/homogeneous.w; return all(isfinite(surfacePoint));
}
bool PrimaryRay(float2 pixel,out float3 origin,out float3 direction,out float maximum)
{
    float2 uv=pixel*Viewport.zw; float3 interior,farPoint; direction=0; maximum=Trace.x;
    if (!Unproject(uv,0,origin) || !Unproject(uv,InteriorDepth,interior)) return false;
    direction=Unit(interior-origin,0); if (!(dot(direction,direction)>0)) return false;
    if (Unproject(uv,1,farPoint)) maximum=min(maximum,length(farPoint-origin));
    return maximum>0;
}
float4 PathSample(float2 pixel,inout uint randomState)
{
    float3 origin,direction; float maximum;
    if (!PrimaryRay(pixel,origin,direction,maximum)) return Background;
    float3 radiance=0,throughput=1; float coverage=1;
    [loop] for (uint depth=0;depth<Sampling.z;++depth)
    {
        Hit hit;
        if (!TraceScene(origin,direction,maximum,false,hit))
        { radiance+=throughput*Background.rgb; if (!depth) coverage=Background.a; break; }
        HitSurface surface=ReadHit(hit,origin,direction); float3 view=-direction;
        if (dot(surface.Normal,view)<=0) surface.Normal=dot(surface.Geometric,view)>0 ? surface.Geometric : -surface.Geometric;
        // Fresnel branching traces both dielectric reflection and transmission.
        // Thin opacity uses stochastic coverage, preserving the unweighted expectation.
        if (surface.IOR>1.0)
        {
            bool entering=dot(direction,surface.Geometric)<0.0;
            float3 face=entering?surface.Geometric:-surface.Geometric;
            float eta=entering?1.0/surface.IOR:surface.IOR;
            float f0=(surface.IOR-1.0)/(surface.IOR+1.0);f0*=f0;
            float cosine=clamp(-dot(direction,face),0.0,1.0);
            float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
            float3 transmitted=refract(direction,face,eta);
            if (dot(transmitted,transmitted)<0.000001 || Random(randomState)<fresnel) direction=reflect(direction,face);
            else { direction=transmitted;throughput*=clamp(surface.Diffuse,0.0,1.0)*(eta*eta); }
            origin=OffsetOrigin(surface,direction);maximum=Trace.x;continue;
        }
        if (surface.Opacity<1.0 && Random(randomState)>surface.Opacity)
        { origin=OffsetOrigin(surface,direction);maximum=Trace.x;continue; }
        radiance+=throughput*(surface.Emission+DirectLighting(surface,view,randomState));
        if (depth+1==Sampling.z) break;
        float3 weight;
        if (!Scatter(surface,view,randomState,direction,weight)) break;
        throughput*=weight;
        if (depth+1>=RouletteStartDepth)
        {
            float survival=clamp(max(throughput.x,max(throughput.y,throughput.z)),RouletteFloor,RouletteCeiling);
            if (Random(randomState)>=survival) break;
            throughput/=survival;
        }
        origin=OffsetOrigin(surface,direction); maximum=Trace.x;
    }
    return float4(radiance,coverage);
}
struct TraceOutput { float4 Mean:SV_TARGET0; float Depth:SV_TARGET1; };
TraceOutput PSMain(ScreenVertex input)
{
    int2 pixel=int2(input.Position.xy); TraceOutput output;
    output.Mean=Sampling.x ? PreviousMean.Load(int3(pixel,0)) : 0;
    [loop] for (uint sampleIndex=0;sampleIndex<Sampling.y;++sampleIndex)
    {
        uint sequence=Sampling.x+sampleIndex;
        uint randomState=Permute(uint(pixel.x)^Permute(uint(pixel.y)^Permute(sequence^Sampling.w)));
        float2 offset=Random2(randomState); float4 value=PathSample(float2(pixel)+offset,randomState);
        output.Mean+=(value-output.Mean)/float(sequence+1);
    }
    float3 origin,direction; float maximum; Hit center;
    output.Depth=EmptyDepth;
    if (PrimaryRay(float2(pixel)+PixelCenter,origin,direction,maximum) && TraceScene(origin,direction,maximum,false,center))
    {
        float4 clip=mul(ViewProjection,float4(origin+direction*center.Distance,1));
        if (abs(clip.w)>MinimumW) output.Depth=saturate(clip.z/clip.w);
    }
    return output;
}
struct CompositeOutput { float4 Color:SV_TARGET; float Depth:SV_DEPTH; };
CompositeOutput PSCompose(ScreenVertex input)
{
    int2 pixel=int2(input.Position.xy); CompositeOutput output;
    float4 mean=PreviousMean.Load(int3(pixel,0));
    output.Color=float4(clamp(mean.rgb,0,HalfMaximum-HalfSpacing),saturate(mean.a));
    output.Depth=CenterDepth.Load(int3(pixel,0)); return output;
}
