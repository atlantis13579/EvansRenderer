// Independent DX12 camera paths, direct-light sampling and progressive FP32
// radiance. Geometry/material/light snapshots remain fixed throughout a frame.
// A cosine/GGX mixture samples both reflective lobes with its full mixture PDF.
// Analytic light sampling and intersected mesh emission have disjoint support.
// PBRT 4e: https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
// GGX VNDF: https://jcgt.org/published/0007/04/01/
// Conservative triangle predicates: https://jcgt.org/published/0002/01/05/
cbuffer PathFrame : register(b0)
{
    row_major float4x4 WorldFromClip,ClipFromWorld;
    float4 FrameSize,RayParameters,BackgroundColor,MaterialParameters;
    uint4 GeometryCounts,SurfaceCounts,Samples,Control;
};
Texture2D<float4> CommittedMean : register(t0);
Texture2D<float> RayDepth : register(t1);
static const uint DiffuseMaterial=1,FlatFlag=1,CasterFlag=1,Directional=0,Point=1;
static const float Pi=3.141592653589793,Center=0.5,MidDepth=0.5,BackgroundDepth=1;
static const float WThreshold=1e-7,VectorThreshold=1e-16,UvThreshold=1e-8,AxisLimit=0.999;
static const float FloatSpacing=1.1920928955078125e-7,EdgeBound=3*FloatSpacing/(1-3*FloatSpacing);
static const float AlphaMinimum=0.0025,CosineMinimum=1e-6,DielectricReflectance=0.04,FresnelExponent=5;
static const float DistanceMinimumSquared=0.01,LobeProbabilityFloor=0.05;
static const float SurvivalFloor=0.05,SurvivalCeiling=0.95,HalfLimit=65504,HalfUlp=32;
static const uint RouletteDepth=3;
// PCG permutation/LCG constants; Jarzynski and Olano (2020).
// https://jcgt.org/published/0009/03/02/
static const uint RandomMultiplier=747796405u,RandomIncrement=2891336453u,OutputMultiplier=277803737u;
static const uint DynamicShift=28,MinimumShift=4,OutputShift=22,MantissaShift=8;
static const float RandomUnit=1.0/16777216.0;
struct BvhNode { float4 Minimum,Maximum; uint4 Link; };
struct Primitive { float4 P0,P1,P2,UV01,UV2; uint4 Identity; };
struct AlphaMaterial { uint4 Diffuse,Opacity; float4 Factor; };
struct CornerData { float4 N0,N1,N2,C0,C1,C2; };
struct HitMaterial { float4 Base,Ambient,Specular,Emitted; uint4 ColorMap,NormalMap; };
struct SceneLight { float4 PositionType,DirectionPower,ColorShadow,Shape; };
StructuredBuffer<BvhNode> Hierarchy : register(t2);
StructuredBuffer<Primitive> Geometry : register(t3);
StructuredBuffer<AlphaMaterial> AlphaMaterials : register(t4);
StructuredBuffer<float> AlphaAtlas : register(t5);
StructuredBuffer<CornerData> Attributes : register(t6);
StructuredBuffer<HitMaterial> HitMaterials : register(t7);
StructuredBuffer<float4> ColorAtlas : register(t8);
StructuredBuffer<SceneLight> SceneLights : register(t9);
struct VertexOut { float4 Position:SV_POSITION; };
VertexOut VSMain(uint vertex:SV_VertexID)
{ VertexOut output; float2 corner=float2((vertex<<1)&2,vertex&2); output.Position=float4(corner*2-1,0,1); return output; }
float3 Normalized(float3 v,float3 fallback)
{
    float scale=max(abs(v.x),max(abs(v.y),abs(v.z))); if (!(scale>0) || !isfinite(scale)) return fallback;
    v/=scale; return v*rsqrt(dot(v,v));
}
void AtlasFootprint(uint4 map,float2 uv,out uint4 address,out float2 fraction)
{
    float2 position=frac(float2(uv.x,-uv.y))*map.yz-Center;
    uint x=position.x<0 ? map.y-1 : uint(floor(position.x)),y=position.y<0 ? map.z-1 : uint(floor(position.y));
    uint nextX=x+1==map.y ? 0 : x+1,nextY=y+1==map.z ? 0 : y+1;
    address=map.x+uint4(y*map.y+x,y*map.y+nextX,nextY*map.y+x,nextY*map.y+nextX); fraction=frac(position);
}
float ReadAlpha(uint4 map,float2 uv)
{
    if (map.w==0) return 1; uint4 address; float2 f; AtlasFootprint(map,uv,address,f);
    return lerp(lerp(AlphaAtlas[address.x],AlphaAtlas[address.y],f.x),lerp(AlphaAtlas[address.z],AlphaAtlas[address.w],f.x),f.y);
}
float3 ReadColor(uint4 map,float2 uv)
{
    if (map.w==0) return 1; uint4 address; float2 f; AtlasFootprint(map,uv,address,f);
    return lerp(lerp(ColorAtlas[address.x].rgb,ColorAtlas[address.y].rgb,f.x),lerp(ColorAtlas[address.z].rgb,ColorAtlas[address.w].rgb,f.x),f.y);
}
bool Slabs(BvhNode node,float3 start,float3 direction,float maximum)
{
    float enter=0,leave=maximum;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (start[axis]<node.Minimum[axis] || start[axis]>node.Maximum[axis]) return false; }
        else
        {
            float lower=(node.Minimum[axis]-start[axis])/direction[axis],upper=(node.Maximum[axis]-start[axis])/direction[axis];
            enter=max(enter,min(lower,upper)); leave=min(leave,max(lower,upper)); if (enter>leave) return false;
        }
    }
    return true;
}
float EdgeFunction(float2 a,float2 b)
{ precise float positive=a.x*b.y,negative=a.y*b.x; precise float edge=positive-negative; return abs(edge)<=EdgeBound*(abs(positive)+abs(negative)) ? 0 : edge; }
bool HitPrimitive(Primitive p,float3 start,float3 direction,float maximum,out float distance,out float3 barycentric)
{
    distance=0; barycentric=0; uint major=abs(direction.x)>abs(direction.y) ? 0 : 1;
    if (abs(direction.z)>abs(direction[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 a=p.P0.xyz-start,b=p.P1.xyz-start,c=p.P2.xyz-start;
    float2 shear=float2(direction[horizontal],direction[vertical])/direction[major];
    precise float2 q0=float2(a[horizontal],a[vertical])-a[major]*shear;
    precise float2 q1=float2(b[horizontal],b[vertical])-b[major]*shear,q2=float2(c[horizontal],c[vertical])-c[major]*shear;
    float3 edges=float3(EdgeFunction(q1,q2),EdgeFunction(q2,q0),EdgeFunction(q0,q1));
    if (any(edges<0) && any(edges>0)) return false;
    float sum=edges.x+edges.y+edges.z; if (sum==0) return false;
    distance=dot(edges,float3(a[major],b[major],c[major]))/(sum*direction[major]);
    if (!(distance>0 && distance<=maximum)) return false; barycentric=edges/sum; return true;
}
struct Intersection { uint PrimitiveIndex; float Distance; float3 Barycentric; };
bool Traverse(float3 start,float3 direction,float maximum,bool visibilityOnly,out Intersection closest)
{
    closest.PrimitiveIndex=0; closest.Distance=maximum; closest.Barycentric=0;
    bool accepted=false; uint index=0;
    [loop] while (index<GeometryCounts.x)
    {
        BvhNode node=Hierarchy[index];
        if (!Slabs(node,start,direction,closest.Distance)) { index=node.Link.w; continue; }
        if (node.Link.z==0) { index=node.Link.x; continue; }
        [loop] for (uint local=0;local<node.Link.z;++local)
        {
            uint primitiveIndex=node.Link.y+local; Primitive p=Geometry[primitiveIndex];
            if (visibilityOnly && (p.Identity.w&CasterFlag)==0) continue;
            float distance; float3 barycentric;
            if (!HitPrimitive(p,start,direction,closest.Distance,distance,barycentric)) continue;
            AlphaMaterial material=AlphaMaterials[p.Identity.x];
            if (visibilityOnly || material.Factor.y>0.0)
            {
                float2 uv=p.UV01.xy*barycentric.x+p.UV01.zw*barycentric.y+p.UV2.xy*barycentric.z;
                float coverage=saturate(dot(barycentric,float3(p.P0.w,p.P1.w,p.P2.w))*material.Factor.x*
                    ReadAlpha(material.Diffuse,uv)*ReadAlpha(material.Opacity,uv));
                if (coverage<material.Factor.y || (visibilityOnly && coverage<=0.0)) continue;
            }
            closest.PrimitiveIndex=primitiveIndex; closest.Distance=distance; closest.Barycentric=barycentric; accepted=true;
            if (visibilityOnly) return true;
        }
        index=node.Link.w;
    }
    return accepted;
}
struct MaterialPoint { float3 Position,N,Ng,Kd,Ks,Ka,Emission; float Exponent,Opacity,IOR; };
MaterialPoint EvaluateMaterial(Intersection hit,float3 start,float3 direction)
{
    Primitive primitive=Geometry[hit.PrimitiveIndex]; CornerData corners=Attributes[hit.PrimitiveIndex];
    HitMaterial material=HitMaterials[primitive.Identity.x]; float3 bary=hit.Barycentric;
    float2 uv=primitive.UV01.xy*bary.x+primitive.UV01.zw*bary.y+primitive.UV2.xy*bary.z;
    float3 ab=primitive.P1.xyz-primitive.P0.xyz,ac=primitive.P2.xyz-primitive.P0.xyz;
    MaterialPoint surfacePoint; surfacePoint.Position=start+direction*hit.Distance;
    surfacePoint.Ng=Normalized(cross(Normalized(ab,float3(1,0,0)),Normalized(ac,float3(0,1,0))),float3(0,0,1));
    surfacePoint.N=(Control.y&FlatFlag) ? surfacePoint.Ng : Normalized(corners.N0.xyz*bary.x+corners.N1.xyz*bary.y+corners.N2.xyz*bary.z,surfacePoint.Ng);
    if (dot(surfacePoint.Ng,direction)>0) surfacePoint.N=-surfacePoint.N;
    if (material.NormalMap.w)
    {
        float2 st1=float2(primitive.UV01.z-primitive.UV01.x,primitive.UV01.y-primitive.UV01.w);
        float2 st2=float2(primitive.UV2.x-primitive.UV01.x,primitive.UV01.y-primitive.UV2.y);
        float determinant=st1.x*st2.y-st1.y*st2.x;
        if (abs(determinant)>=UvThreshold)
        {
            float3 tangent=(ab*st2.y-ac*st1.y)/determinant,sourceBitangent=(ac*st1.x-ab*st2.x)/determinant;
            float3 axis=abs(surfacePoint.N.y)<AxisLimit ? float3(0,1,0) : float3(1,0,0);
            tangent=Normalized(tangent-surfacePoint.N*dot(tangent,surfacePoint.N),Normalized(cross(axis,surfacePoint.N),float3(1,0,0)));
            float3 bitangent=cross(surfacePoint.N,tangent); bitangent*=dot(bitangent,sourceBitangent)<0 ? -1 : 1;
            float3 mapped=ReadColor(material.NormalMap,uv)*2-1;
            surfacePoint.N=Normalized(tangent*mapped.x+bitangent*mapped.y+surfacePoint.N*mapped.z,surfacePoint.N);
        }
    }
    surfacePoint.Kd=max((corners.C0.rgb*bary.x+corners.C1.rgb*bary.y+corners.C2.rgb*bary.z)*material.Base.rgb*ReadColor(material.ColorMap,uv),0);
    surfacePoint.Ks=material.Specular.rgb; surfacePoint.Exponent=material.Specular.w; surfacePoint.Ka=material.Ambient.rgb; surfacePoint.Emission=material.Emitted.rgb;
    surfacePoint.Opacity=clamp(material.Base.w*(primitive.P0.w*bary.x+primitive.P1.w*bary.y+primitive.P2.w*bary.z),0.0,1.0); surfacePoint.IOR=material.Emitted.w;
    return surfacePoint;
}

uint Hash(uint inputValue)
{
    uint state=inputValue*RandomMultiplier+RandomIncrement;
    uint word=((state>>((state>>DynamicShift)+MinimumShift))^state)*OutputMultiplier;
    return (word>>OutputShift)^word;
}

float Next(inout uint state)
{
    state=state*RandomMultiplier+RandomIncrement;
    uint word=((state>>((state>>DynamicShift)+MinimumShift))^state)*OutputMultiplier;
    return float(((word>>OutputShift)^word)>>MantissaShift)*RandomUnit;
}
float2 Next2(inout uint state) { float x=Next(state),y=Next(state); return float2(x,y); }
float3 Schlick(float3 f0,float cosine)
{ return f0+(1-f0)*pow(1-saturate(cosine),FresnelExponent); }
float Masking(float cosine,float alpha)
{ return 2*cosine/max(cosine+sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine),CosineMinimum); }
float Ndf(float nh,float alpha)
{ float squared=alpha*alpha,denominator=nh*nh*(squared-1)+1; return squared/(Pi*denominator*denominator); }
float LobeProbability(float3 diffuse)
{
    if (Control.x==DiffuseMaterial) return 0;
    float3 specular=lerp(DielectricReflectance.xxx,diffuse,MaterialParameters.y);
    float3 lambert=(1-specular)*(1-MaterialParameters.y)*diffuse;
    float s=max(specular.x,max(specular.y,specular.z)),d=max(lambert.x,max(lambert.y,lambert.z));
    return d<=0 ? 1 : clamp(s/(s+d),LobeProbabilityFloor,1-LobeProbabilityFloor);
}
float3 Response(float3 diffuse,float3 normal,float3 outgoing,float3 incoming,out float density)
{
    density=0; float nv=dot(normal,outgoing),nl=dot(normal,incoming);
    if (nv<=0 || nl<=0) return 0;
    if (Control.x==DiffuseMaterial) { density=nl/Pi; return diffuse*density; }
    float3 halfDirection=Normalized(outgoing+incoming,normal);
    float alpha=max(MaterialParameters.x*MaterialParameters.x,AlphaMinimum);
    float distribution=Ndf(saturate(dot(normal,halfDirection)),alpha),g=Masking(nv,alpha);
    density=lerp(nl/Pi,distribution*g/max(4*nv,CosineMinimum),LobeProbability(diffuse));
    float3 f=Schlick(lerp(DielectricReflectance.xxx,diffuse,MaterialParameters.y),dot(outgoing,halfDirection));
    return (1-f)*(1-MaterialParameters.y)*diffuse*(nl/Pi)+f*distribution*g*Masking(nl,alpha)/max(4*nv,CosineMinimum);
}
void TangentFrame(float3 normal,out float3 tangent,out float3 bitangent)
{
    float3 axis=abs(normal.x)<abs(normal.y) ? float3(1,0,0) : float3(0,1,0);
    tangent=Normalized(cross(axis,normal),float3(1,0,0)); bitangent=cross(normal,tangent);
}
float3 SampleVisible(float3 view,float alpha,float2 xi)
{
    float3 stretched=Normalized(float3(view.xy*alpha,view.z),float3(0,0,1));
    float len=dot(stretched.xy,stretched.xy);
    float3 t=len>0 ? float3(-stretched.y,stretched.x,0)/sqrt(len) : float3(1,0,0),b=cross(stretched,t);
    float radius=sqrt(xi.x),angle=2*Pi*xi.y,x=radius*cos(angle),y=radius*sin(angle);
    y=lerp(sqrt(max(0,1-x*x)),y,(1+stretched.z)/2);
    float3 projected=t*x+b*y+stretched*sqrt(max(0,1-x*x-y*y));
    return Normalized(float3(projected.xy*alpha,max(projected.z,0)),float3(0,0,1));
}
bool ContinuePath(MaterialPoint surfacePoint,float3 outgoing,inout uint state,out float3 incoming,out float3 weight)
{
    float3 diffuse=saturate(surfacePoint.Kd),tangent,bitangent; TangentFrame(surfacePoint.N,tangent,bitangent);
    float choice=Next(state); float2 xi=Next2(state);
    if (choice<LobeProbability(diffuse))
    {
        float3 localView=float3(dot(outgoing,tangent),dot(outgoing,bitangent),dot(outgoing,surfacePoint.N));
        float3 h=SampleVisible(localView,max(MaterialParameters.x*MaterialParameters.x,AlphaMinimum),xi);
        incoming=Normalized(reflect(-outgoing,tangent*h.x+bitangent*h.y+surfacePoint.N*h.z),surfacePoint.N);
    }
    else
    {
        float radius=sqrt(xi.x),angle=2*Pi*xi.y;
        incoming=tangent*(radius*cos(angle))+bitangent*(radius*sin(angle))+surfacePoint.N*sqrt(1-xi.x);
    }
    float density; float3 value=Response(diffuse,surfacePoint.N,outgoing,incoming,density); weight=0;
    if (!(density>0) || dot(surfacePoint.Ng,outgoing)*dot(surfacePoint.Ng,incoming)<=0) return false;
    weight=value/density; return all(isfinite(weight)) && any(weight>0);
}
float3 RayStart(MaterialPoint surfacePoint,float3 direction)
{ return surfacePoint.Position+surfacePoint.Ng*(dot(surfacePoint.Ng,direction)>=0 ? RayParameters.y : -RayParameters.y); }
float3 SampleLights(MaterialPoint surfacePoint,float3 outgoing,inout uint state)
{
    float3 sum=0;
    [loop] for (uint index=0;index<SurfaceCounts.w;++index)
    {
        SceneLight light=SceneLights[index]; uint type=uint(light.PositionType.w);
        bool distant=type==Directional,rectangle=type!=Directional && type!=Point;
        float3 emitter=light.PositionType.xyz,normal=Normalized(light.DirectionPower.xyz,float3(0,-1,0)); float measure=1;
        if (rectangle)
        {
            float3 helper=abs(normal.y)<AxisLimit ? float3(0,1,0) : float3(1,0,0);
            float3 tangent=Normalized(cross(helper,normal),float3(1,0,0)),bitangent=cross(normal,tangent);
            float2 uv=Next2(state)-Center;
            emitter+=tangent*(uv.x*light.Shape.x)+bitangent*(uv.y*light.Shape.y); measure=light.Shape.w>0 ? abs(light.Shape.x*light.Shape.y) : 1;
        }
        float3 delta=distant ? Normalized(light.DirectionPower.xyz,float3(0,1,0))*RayParameters.z : emitter-surfacePoint.Position;
        float squared=dot(delta,delta); if (!(squared>VectorThreshold)) continue;
        float distance=sqrt(squared); float3 incoming=delta/distance;
        if (dot(surfacePoint.Ng,incoming)*dot(surfacePoint.Ng,outgoing)<=0) continue;
        float attenuation=1;
        if (!distant)
        {
            float window=1;
            if (light.Shape.z>0) { float ratio=distance/(light.Shape.z>0 ? light.Shape.z : 1.0); window=saturate(1-ratio*ratio*ratio*ratio); window*=window; }
            attenuation=window/max(squared,DistanceMinimumSquared);
        }
        if (rectangle) attenuation*=measure*max(dot(normal,-incoming),0);
        if (!(attenuation>0)) continue;
        Intersection occluder;
        if (Control.z!=0 && light.ColorShadow.w>0 && Traverse(RayStart(surfacePoint,incoming),incoming,max(distance-RayParameters.y,0),true,occluder)) continue;
        float density; sum+=Response(saturate(surfacePoint.Kd),surfacePoint.N,outgoing,incoming,density)*
            max(light.ColorShadow.rgb,0)*max(light.DirectionPower.w,0)*attenuation;
    }
    return sum;
}
bool CameraPoint(float2 uv,float depth,out float3 surfacePoint)
{
    float4 h=mul(WorldFromClip,float4(uv.x*2-1,1-uv.y*2,depth,1)); surfacePoint=0;
    if (!(abs(h.w)>WThreshold)) return false;
    surfacePoint=h.xyz/h.w; return all(isfinite(surfacePoint));
}
bool CameraRay(float2 pixel,out float3 start,out float3 direction,out float maximum)
{
    float2 uv=pixel*FrameSize.zw; float3 middle,farPoint; direction=0; maximum=RayParameters.x;
    if (!CameraPoint(uv,0,start) || !CameraPoint(uv,MidDepth,middle)) return false;
    direction=Normalized(middle-start,0); if (!(dot(direction,direction)>0)) return false;
    if (CameraPoint(uv,1,farPoint)) maximum=min(maximum,length(farPoint-start));
    return maximum>0;
}
float4 Integrate(float2 pixel,inout uint state)
{
    float3 origin,direction; float maximum;
    if (!CameraRay(pixel,origin,direction,maximum)) return BackgroundColor;
    float3 result=0,throughput=1; float coverage=1;
    [loop] for (uint depth=0;depth<Samples.z;++depth)
    {
        Intersection hit;
        if (!Traverse(origin,direction,maximum,false,hit))
        { result+=throughput*BackgroundColor.rgb; if (!depth) coverage=BackgroundColor.a; break; }
        MaterialPoint surfacePoint=EvaluateMaterial(hit,origin,direction); float3 outgoing=-direction;
        if (dot(surfacePoint.N,outgoing)<=0) surfacePoint.N=dot(surfacePoint.Ng,outgoing)>0 ? surfacePoint.Ng : -surfacePoint.Ng;
        // Fresnel branching traces both dielectric reflection and transmission.
        // Thin opacity uses stochastic coverage, preserving the unweighted expectation.
        if (surfacePoint.IOR>1.0)
        {
            bool entering=dot(direction,surfacePoint.Ng)<0.0;
            float3 face=entering?surfacePoint.Ng:-surfacePoint.Ng;
            float eta=entering?1.0/surfacePoint.IOR:surfacePoint.IOR;
            float f0=(surfacePoint.IOR-1.0)/(surfacePoint.IOR+1.0);f0*=f0;
            float cosine=clamp(-dot(direction,face),0.0,1.0);
            float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
            float3 transmitted=refract(direction,face,eta);
            if (dot(transmitted,transmitted)<0.000001 || Next(state)<fresnel) direction=reflect(direction,face);
            else { direction=transmitted;throughput*=clamp(surfacePoint.Kd,0.0,1.0)*(eta*eta); }
            origin=RayStart(surfacePoint,direction);maximum=RayParameters.x;continue;
        }
        if (surfacePoint.Opacity<1.0 && Next(state)>surfacePoint.Opacity)
        { origin=RayStart(surfacePoint,direction);maximum=RayParameters.x;continue; }
        result+=throughput*(surfacePoint.Emission+SampleLights(surfacePoint,outgoing,state));
        if (depth+1==Samples.z) break;
        float3 weight; if (!ContinuePath(surfacePoint,outgoing,state,direction,weight)) break;
        throughput*=weight;
        if (depth+1>=RouletteDepth)
        {
            float survival=clamp(max(throughput.x,max(throughput.y,throughput.z)),SurvivalFloor,SurvivalCeiling);
            if (Next(state)>=survival) break;
            throughput/=survival;
        }
        origin=RayStart(surfacePoint,direction); maximum=RayParameters.x;
    }
    return float4(result,coverage);
}
struct PathOutput { float4 Mean:SV_TARGET0; float Depth:SV_TARGET1; };
PathOutput PSMain(VertexOut input)
{
    int2 pixel=int2(input.Position.xy); PathOutput output;
    output.Mean=Samples.x ? CommittedMean.Load(int3(pixel,0)) : 0;
    [loop] for (uint path=0;path<Samples.y;++path)
    {
        uint sequence=Samples.x+path,state=Hash(uint(pixel.x)^Hash(uint(pixel.y)^Hash(sequence^Samples.w)));
        float2 jitter=Next2(state); float4 value=Integrate(float2(pixel)+jitter,state);
        output.Mean+=(value-output.Mean)/float(sequence+1);
    }
    float3 origin,direction; float maximum; Intersection center; output.Depth=BackgroundDepth;
    if (CameraRay(float2(pixel)+Center,origin,direction,maximum) && Traverse(origin,direction,maximum,false,center))
    {
        float4 projected=mul(ClipFromWorld,float4(origin+direction*center.Distance,1));
        if (abs(projected.w)>WThreshold) output.Depth=saturate(projected.z/projected.w);
    }
    return output;
}
struct ResolveOutput { float4 Color:SV_TARGET; float Depth:SV_DEPTH; };
ResolveOutput PSCompose(VertexOut input)
{
    int2 pixel=int2(input.Position.xy); ResolveOutput output;
    float4 value=CommittedMean.Load(int3(pixel,0));
    output.Color=float4(clamp(value.rgb,0,HalfLimit-HalfUlp),saturate(value.a));
    output.Depth=RayDepth.Load(int3(pixel,0)); return output;
}
