#include "Editor/BigWorld.hpp"
#include "Maths/ColorSpace.h"
#include "Rasterizer/ParallelExecutor.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/Atmosphere/Atmosphere.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <future>
#include <chrono>
#include <tuple>
#include <map>

namespace Evans::Editor
{
struct BigWorldGeometry
{
    struct GrassPatch { unsigned Lod=0; std::shared_ptr<const MeshAsset> Mesh; };
    struct Tree { SceneNodeId Id; unsigned Lod=0; std::array<std::shared_ptr<const MeshAsset>,3> Meshes; };
    std::map<std::pair<int,int>,GrassPatch> Grass;
    std::vector<Tree> Trees;
    struct WaterVertex { float Depth,Shore; };
    struct WaterTile
    {
        int X=0,Z=0;
        unsigned Lod=0,Segments=0;
        MeshAsset Mesh;
        std::vector<WaterVertex> Surface;
        std::vector<std::pair<unsigned,bool>> Skirts;
    };
    std::vector<std::shared_ptr<const WaterTile>> Water;
    std::vector<std::pair<SceneNodeId,std::shared_ptr<const MeshAsset>>> Lakebed;
};
struct BigWorldSkyJob
{
    std::future<std::shared_ptr<const Renderer::EnvironmentRadiance>> Result;
    float Hour=0;
    Float3 Position;
};
namespace
{
constexpr float Pi=3.14159265358979323846f;
float Saturate(float x) { return std::clamp(x,0.0f,1.0f); }
float Smooth(float x) { x=Saturate(x); return x*x*(3-2*x); }
float Mix(float a,float b,float t) { return a+(b-a)*t; }
Color Mix(Color a,Color b,float t) { return {Mix(a.R,b.R,t),Mix(a.G,b.G,t),Mix(a.B,b.B,t),1}; }
Float3 Unit(Float3 p)
{
    const float inverse=1/std::sqrt(std::max(1e-12f,p.X*p.X+p.Y*p.Y+p.Z*p.Z));
    return {p.X*inverse,p.Y*inverse,p.Z*inverse};
}
float Hash(int x,int z,unsigned seed)
{
    unsigned h=unsigned(x)*374761393u+unsigned(z)*668265263u+seed*1442695041u;
    h=(h^(h>>13))*1274126177u; h^=h>>16;
    return float(h&0xffffffu)/16777215.0f;
}
float Noise(float x,float z,unsigned seed)
{
    const int ix=int(std::floor(x)),iz=int(std::floor(z));
    const float u=Smooth(x-std::floor(x)),v=Smooth(z-std::floor(z));
    return Mix(Mix(Hash(ix,iz,seed),Hash(ix+1,iz,seed),u),
        Mix(Hash(ix,iz+1,seed),Hash(ix+1,iz+1,seed),u),v);
}
float Fbm(float x,float z,unsigned seed)
{
    float value=0,weight=0.5f;
    for (unsigned octave=0;octave<5;++octave)
    { value+=Noise(x,z,seed+octave)*weight; x=x*2.03f+7; z=z*2.03f-3; weight*=0.5f; }
    return value;
}
Float3 TerrainNormal(const BigWorldSettings& settings,float x,float z)
{
    constexpr float e=0.5f;
    return Unit({BigWorldHeight(settings,x-e,z)-BigWorldHeight(settings,x+e,z),2*e,
        BigWorldHeight(settings,x,z-e)-BigWorldHeight(settings,x,z+e)});
}
Color LakebedColor(float x,float z,unsigned seed)
{
    const float sediment=Noise(x*0.045f,z*0.045f,seed);
    const float gravel=Noise(x*0.28f,z*0.28f,seed+13);
    return Mix({0.25f,0.28f,0.23f,1},{0.53f,0.49f,0.38f,1},0.75f*sediment+0.25f*gravel);
}
Color TerrainColor(const BigWorldSettings& s,Float3 position,Float3 normal)
{
    const float variation=Noise(position.X*0.045f,position.Z*0.045f,s.Seed);
    Color color=Mix({0.29f,0.32f,0.20f,1},{0.46f,0.43f,0.29f,1},variation);
    const float rock=Smooth((0.87f-normal.Y)*5+Noise(position.X*0.13f,position.Z*0.13f,s.Seed+21)*0.3f);
    color=Mix(color,{0.39f,0.39f,0.37f,1},rock);
    color=Mix(color,{0.45f,0.42f,0.34f,1},1-Smooth(std::abs(position.Y-LakeElevation)/6));
    if (position.Y<LakeElevation) color=LakebedColor(position.X,position.Z,s.Seed);
    const float snowLine=LakeElevation+s.MountainHeight*(s.Weather==WorldWeather::Snow?0.35f:0.92f);
    const float snow=Smooth((position.Y-snowLine)/24)*Smooth((normal.Y-0.5f)*3);
    color=Mix(color,{0.82f,0.85f,0.87f,1},snow);
    if (s.Weather==WorldWeather::Snow && position.Y>LakeElevation)
        color=Mix(color,{0.82f,0.85f,0.87f,1},s.SnowCover*Smooth((normal.Y-0.4f)*2));
    if (s.Weather==WorldWeather::Rain) { color.R*=0.7f; color.G*=0.7f; color.B*=0.7f; }
    return color;
}
// World-space procedural strata, pebbles and organic litter. A fixed seed and
// shared coordinates make neighboring material tiles continuous without assets.
float SurfaceRelief(float x,float z,unsigned seed)
{
    const float warp=Noise(x*0.17f,z*0.17f,seed+8)*1.8f;
    const float cracks=std::pow(std::abs(std::sin((x+warp)*2.1f+z*0.37f)),0.3f);
    return Noise(x*1.8f,z*1.8f,seed)*0.55f+Noise(x*5.7f,z*5.7f,seed+1)*0.12f+cracks*0.33f;
}
void Pixel(ImageAsset& image,unsigned x,unsigned y,Color color)
{
    const auto i=(std::size_t(y)*image.Width+x)*4;
    image.PixelsRGBA8[i]=std::uint8_t(Saturate(color.R)*255);
    image.PixelsRGBA8[i+1]=std::uint8_t(Saturate(color.G)*255);
    image.PixelsRGBA8[i+2]=std::uint8_t(Saturate(color.B)*255);
    image.PixelsRGBA8[i+3]=std::uint8_t(Saturate(color.A)*255);
}
std::shared_ptr<ImageAsset> Image(unsigned size)
{
    auto image=std::make_shared<ImageAsset>(); image->Width=image->Height=size;
    image->PixelsRGBA8.resize(std::size_t(size)*size*4); return image;
}
// Integer wave vectors and periodic phase modulation keep both slopes continuous
// across the tile. The material sampler filters these short waves at a distance.
constexpr float RipplePatch=12;
std::shared_ptr<const ImageAsset> LakeRippleNormals(float strength)
{
    if (strength<=0) return {};
    auto image=Image(128);
    for (unsigned y=0;y<image->Height;++y) for (unsigned x=0;x<image->Width;++x)
    {
        const float u=(x+0.5f)*2*Pi/image->Width,v=(y+0.5f)*2*Pi/image->Height;
        const float bend=std::sin(u+2*v)+0.6f*std::sin(3*u-v);
        const float a=std::cos(7*u+2*v+1.4f*bend);
        const float b=std::cos(12*u-5*v+2*bend);
        const float c=std::cos(5*u+13*v-1.8f*bend);
        const float envelope=0.7f+0.3f*std::sin(2*u-v+std::cos(v));
        const auto n=Unit({-strength*envelope*(0.085f*a+0.050f*b+0.017f*c),
            -strength*envelope*(0.024f*a-0.021f*b+0.044f*c),1});
        Pixel(*image,x,y,{0.5f+0.5f*n.X,0.5f+0.5f*n.Y,0.5f+0.5f*n.Z,1});
    }
    return image;
}
float RippleStrength(const BigWorldSettings& s)
{ return s.FFT?std::min(1.8f,s.WaveAmplitude*s.WindSpeed/3):0; }
void SetWaterOptics(Material& water,const BigWorldSettings& s)
{
    const float f0=std::pow((s.WaterIOR-1)/(s.WaterIOR+1),2.0f);
    water.Roughness=s.WaterRoughness;
    water.DielectricReflectance=f0;
    water.Specular={f0,f0,f0,1};
    water.Shininess=2/(s.WaterRoughness*s.WaterRoughness)-2;
}
void LandscapeMaterial(SceneSnapshot& scene,const BigWorldSettings& s,MeshAsset& mesh,unsigned segments,const std::string& name)
{
    const auto origin=mesh.Vertices.front().Position;
    const float width=s.Extent/(s.TerrainSegments/16);
    const auto eye=scene.SceneCamera.Position;
    const unsigned size=std::hypot(origin.X+width*0.5f-eye.X,origin.Z+width*0.5f-eye.Z)<160?512:128;
    auto diffuse=Image(size),normal=Image(size);
    for (unsigned y=0;y<size;++y) for (unsigned x=0;x<size;++x)
    {
        const float u=float(x)/(size-1),v=float(y)/(size-1);
        const float wx=origin.X+u*width,wz=origin.Z+v*width;
        const unsigned ix=std::min(unsigned(u*segments),segments-1),iz=std::min(unsigned(v*segments),segments-1);
        const float fx=u*segments-ix,fz=v*segments-iz;
        const auto& a=mesh.Vertices[iz*(segments+1)+ix]; const auto& b=mesh.Vertices[iz*(segments+1)+ix+1];
        const auto& c=mesh.Vertices[(iz+1)*(segments+1)+ix]; const auto& d=mesh.Vertices[(iz+1)*(segments+1)+ix+1];
        const float height=Mix(Mix(a.Position.Y,b.Position.Y,fx),Mix(c.Position.Y,d.Position.Y,fx),fz);
        const float slope=Mix(Mix(a.Normal.Y,b.Normal.Y,fx),Mix(c.Normal.Y,d.Normal.Y,fx),fz);
        Color color=TerrainColor(s,{wx,height,wz},{0,slope,0});
        const float relief=SurfaceRelief(wx,wz,s.Seed);
        const float grain=Hash(int(wx*39),int(wz*39),s.Seed+31);
        const float litter=Noise(wx*0.55f,wz*0.55f,s.Seed+51);
        const float brightness=0.76f+relief*0.33f+grain*0.025f;
        color.R*=brightness; color.G*=brightness; color.B*=brightness;
        color=Mix(color,{0.36f*brightness,0.29f*brightness,0.20f*brightness,1},Smooth((litter-0.60f)*5)*(1-Smooth((0.85f-slope)*6)));
        color.R/=0.9f; color.G/=0.9f; color.B/=0.9f;
        Pixel(*diffuse,x,y,color);
        const float dx=(SurfaceRelief(wx+0.15f,wz,s.Seed)-SurfaceRelief(wx-0.15f,wz,s.Seed))*0.18f;
        const float dz=(SurfaceRelief(wx,wz+0.15f,s.Seed)-SurfaceRelief(wx,wz-0.15f,s.Seed))*0.18f;
        const auto n=Unit({-dx,-dz,1}); Pixel(*normal,x,y,{n.X*0.5f+0.5f,n.Y*0.5f+0.5f,n.Z*0.5f+0.5f,1});
    }
    // Keep the outer world-space samples at texel centers. The shared material
    // sampler wraps, so sampling exactly 0/1 would mix opposite tile boundaries.
    for (auto& vertex:mesh.Vertices)
    {
        vertex.TexCoord.X=(0.5f+vertex.TexCoord.X*(size-1))/size;
        vertex.TexCoord.Y=(0.5f+vertex.TexCoord.Y*(size-1))/size;
    }
    Material material; material.Name=name; material.Diffuse={1,1,1,1}; material.Roughness=s.Weather==WorldWeather::Rain?0.54f:0.92f;
    material.Metallic=0.0f; material.DiffuseImage=diffuse; material.NormalImage=normal;
    scene.Materials.push_back(std::move(material));
}
void VegetationMaterials(SceneSnapshot& scene)
{
    auto bark=Image(256),barkNormal=Image(256),needles=Image(256),mask=Image(256);
    for (unsigned y=0;y<256;++y) for (unsigned x=0;x<256;++x)
    {
        const float u=(x+0.5f)/256,v=(y+0.5f)/256;
        const float ridge=std::pow(std::abs(std::sin(u*Pi*32+Noise(u*8,v*16,43)*2)),0.4f);
        const float shade=0.35f+0.65f*ridge;
        Pixel(*bark,x,y,{0.38f*shade,0.31f*shade,0.24f*shade,1});
        const auto n=Unit({std::cos(u*Pi*32)*0.4f,Noise(u*32,v*64,8)*0.15f,1});
        Pixel(*barkNormal,x,y,{n.X*0.5f+0.5f,n.Y*0.5f+0.5f,n.Z*0.5f+0.5f,1});
        const float axis=std::abs(u-0.5f),branch=std::max(0.0f,0.45f*(1-v));
        const float needlePhase=(v-axis*0.56f)*44;
        const float needle=std::abs(needlePhase-std::round(needlePhase));
        const float coverage=axis<0.015f?1.0f:axis<branch && needle<0.30f?1.0f:0.0f;
        const float light=0.7f+Hash(int(x),int(y),77)*0.3f;
        Pixel(*needles,x,y,{(0.24f+v*0.08f)*light,(0.31f+v*0.05f)*light,(0.19f+v*0.03f)*light,1});
        Pixel(*mask,x,y,{coverage,coverage,coverage,1});
    }
    for (auto& material:scene.Materials)
    {
        if (material.Name=="Bark") { material.DiffuseImage=bark; material.NormalImage=barkNormal; }
        if (material.Name=="Canopy") material.DiffuseImage=needles;
        if (material.Name=="Foliage") { material.DiffuseImage=needles; material.OpacityImage=mask; material.SubsurfaceWeight=0.35f; }
    }
}
void Triangle(MeshAsset& mesh,Vertex a,Vertex b,Vertex c,bool doubleSided=false)
{
    const auto first=unsigned(mesh.Vertices.size());
    mesh.Vertices.insert(mesh.Vertices.end(),{a,b,c});
    mesh.Indices.insert(mesh.Indices.end(),{first,first+1,first+2});
    if (doubleSided) mesh.Indices.insert(mesh.Indices.end(),{first+2,first+1,first});
}
void Finish(MeshAsset& mesh,const char* material)
{ mesh.Sections.push_back({material,0,unsigned(mesh.Indices.size())}); }
void Node(SceneSnapshot& scene,std::string name,MeshAsset mesh)
{
    scene.Nodes.push_back({SceneNodeId(scene.Nodes.size()+1),std::move(name),{},
        std::make_shared<MeshAsset>(std::move(mesh))});
}
void GridIndices(MeshAsset& mesh,unsigned segments)
{
    for (unsigned z=0;z<segments;++z) for (unsigned x=0;x<segments;++x)
    {
        const unsigned a=z*(segments+1)+x,b=a+segments+1;
        mesh.Indices.insert(mesh.Indices.end(),{a,b,b+1,a,b+1,a+1});
    }
}
void BuildTerrain(SceneSnapshot& scene,const BigWorldSettings& settings)
{
    if (!settings.Terrain) return;
    constexpr unsigned ChunkSegments=16;
    const unsigned chunks=settings.TerrainSegments/ChunkSegments;
    const float step=settings.Extent/settings.TerrainSegments;
    for (unsigned cz=0;cz<chunks;++cz) for (unsigned cx=0;cx<chunks;++cx)
    {
        MeshAsset mesh; mesh.Name="Landscape";
        for (unsigned z=0;z<=ChunkSegments;++z) for (unsigned x=0;x<=ChunkSegments;++x)
        {
            Vertex vertex;
            vertex.Position={(cx*ChunkSegments+x)*step-settings.Extent*0.5f,0,
                (cz*ChunkSegments+z)*step-settings.Extent*0.5f};
            vertex.Position.Y=BigWorldHeight(settings,vertex.Position.X,vertex.Position.Z);
            vertex.Normal=TerrainNormal(settings,vertex.Position.X,vertex.Position.Z);
            vertex.VertexColor={0.9f,0.9f,0.9f,1};
            vertex.TexCoord={float(x)/ChunkSegments,1-float(z)/ChunkSegments};
            mesh.Vertices.push_back(vertex);
        }
        GridIndices(mesh,ChunkSegments);
        const auto material="Landscape "+std::to_string(cx)+":"+std::to_string(cz);
        LandscapeMaterial(scene,settings,mesh,ChunkSegments,material);
        // Skirts prevent daylight cracks when neighboring chunks choose different LODs.
        const auto skirt=[&](unsigned a,unsigned b)
        {
            Vertex va=mesh.Vertices[a],vb=mesh.Vertices[b],vc=vb,vd=va;
            vc.Position.Y-=8; vd.Position.Y-=8; Triangle(mesh,va,vb,vc,true); Triangle(mesh,va,vc,vd,true);
        };
        for (unsigned i=0;i<ChunkSegments;++i)
        { skirt(i,i+1); skirt(ChunkSegments*(ChunkSegments+1)+i,ChunkSegments*(ChunkSegments+1)+i+1);
          skirt(i*(ChunkSegments+1),(i+1)*(ChunkSegments+1)); skirt(i*(ChunkSegments+1)+ChunkSegments,(i+1)*(ChunkSegments+1)+ChunkSegments); }
        Finish(mesh,material.c_str());
        const bool lakebed=std::any_of(mesh.Vertices.begin(),mesh.Vertices.end(),[&](const auto& v){return v.Position.Y<LakeElevation;});
        Node(scene,std::string(lakebed?"Lakebed ":"Terrain ")+std::to_string(cx)+":"+std::to_string(cz),std::move(mesh));
    }
}
Float3 Add(Float3 a,Float3 b) { return {a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }
Float3 Scale(Float3 a,float s) { return {a.X*s,a.Y*s,a.Z*s}; }
Float3 Cross(Float3 a,Float3 b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
void Branch(MeshAsset& mesh,Float3 root,Float3 tip,float radius,unsigned sides)
{
    const auto axis=Unit(Add(tip,Scale(root,-1))),right=Unit(Cross(axis,{0.1f,0,1})),up=Cross(right,axis);
    for (unsigned side=0;side<sides;++side)
    {
        Vertex a,b,c,d;
        const auto n0=Add(Scale(right,std::cos(side*2*Pi/sides)),Scale(up,std::sin(side*2*Pi/sides)));
        const auto n1=Add(Scale(right,std::cos((side+1)*2*Pi/sides)),Scale(up,std::sin((side+1)*2*Pi/sides)));
        a.Position=Add(root,Scale(n0,radius)); b.Position=Add(root,Scale(n1,radius));
        c.Position=Add(tip,Scale(n1,radius*0.1f)); d.Position=Add(tip,Scale(n0,radius*0.1f));
        a.Normal=d.Normal=n0; b.Normal=c.Normal=n1;
        a.VertexColor=b.VertexColor=c.VertexColor=d.VertexColor={1,1,1,1};
        a.TexCoord={float(side)/sides,0}; b.TexCoord={float(side+1)/sides,0};
        c.TexCoord={float(side+1)/sides,1}; d.TexCoord={float(side)/sides,1};
        Triangle(mesh,a,d,c); Triangle(mesh,a,c,b);
    }
}
void Frond(MeshAsset& mesh,Float3 root,Float3 tip,float width,float angle,Color color)
{
    const auto direction=Unit(Add(tip,Scale(root,-1)));
    const auto side=Unit(Add(Scale(Cross(direction,{0,1,0}),0.55f),{0,0.75f+0.15f*std::sin(angle),0}));
    const auto normal=Unit(Add(Scale(Cross(side,direction),0.5f),{0,0.65f,0}));
    Vertex a,b,c,d;
    a.Position=Add(root,Scale(side,-width)); b.Position=Add(root,Scale(side,width));
    c.Position=Add(tip,Scale(side,width*0.3f)); d.Position=Add(tip,Scale(side,-width*0.3f));
    a.Normal=b.Normal=c.Normal=d.Normal=normal;
    a.VertexColor=b.VertexColor=c.VertexColor=d.VertexColor=color;
    a.TexCoord={0,1}; b.TexCoord={1,1}; c.TexCoord={1,0}; d.TexCoord={0,0};
    Triangle(mesh,a,b,c); Triangle(mesh,a,c,d);
}
std::shared_ptr<const MeshAsset> PineMesh(unsigned seed,unsigned lod)
{
    auto mesh=std::make_shared<MeshAsset>(); mesh->Name="Procedural needle pine LOD "+std::to_string(lod);
    Branch(*mesh,{0,0,0},{0.12f,14,0.05f},0.24f,lod?5:8);
    if (!lod) for (unsigned i=0;i<24;++i)
    {
        const float y=2.2f+i*0.44f,angle=i*2.39996f+Hash(i,3,seed);
        const float length=(14-y)*0.26f*(0.65f+Hash(i,7,seed)*0.5f);
        Branch(*mesh,{0,y,0},{std::cos(angle)*length,y-0.3f,std::sin(angle)*length},0.065f*(1-y/15),4);
    }
    Finish(*mesh,"Bark"); const unsigned first=unsigned(mesh->Indices.size());
    const unsigned layers=lod==0?22:lod==1?14:8,arms=lod==0?7:lod==1?5:4;
    for (unsigned layer=0;layer<layers;++layer)
    {
        const float y=2.4f+float(layer)/(layers-1)*11.1f;
        const float envelope=(14-y)*0.26f;
        for (unsigned arm=0;arm<arms;++arm)
        {
            const float angle=arm*2*Pi/arms+layer*2.39996f+Hash(layer,arm,seed)*0.5f;
            const float radius=envelope*(0.7f+Hash(layer,arm,seed+1)*0.6f);
            const Float3 root{0,y,0},tip{std::cos(angle)*radius,y+0.12f,std::sin(angle)*radius};
            const float shade=0.65f+Hash(layer,arm,seed+3)*0.35f;
            const Color color{shade,shade,shade,1};
            Frond(*mesh,root,tip,0.48f+radius*0.2f,angle,color);
            if (lod==0) for (unsigned twig=0;twig<2;++twig)
            {
                const float t=0.4f+twig*0.28f,a=angle+(twig?0.7f:-0.7f);
                const auto start=Add(Scale(tip,t),{0,y*(1-t),0});
                Frond(*mesh,start,Add(start,{std::cos(a)*radius*0.65f,0.35f,std::sin(a)*radius*0.65f}),0.45f,angle,color);
            }
        }
    }
    mesh->Sections.push_back({lod==2?"Canopy":"Foliage",first,unsigned(mesh->Indices.size())-first}); return mesh;
}
void BuildTrees(SceneSnapshot& scene,const BigWorldSettings& settings,BigWorldGeometry& cache)
{
    if (!settings.Trees) return;
    std::array<std::array<std::shared_ptr<const MeshAsset>,3>,4> variants;
    for (unsigned variant=0;variant<4;++variant) for (unsigned lod=0;lod<3;++lod)
        variants[variant][lod]=PineMesh(settings.Seed+variant*13,lod);
    std::mt19937 random(settings.Seed+17); std::uniform_real_distribution<float> unit(0,1);
    unsigned placed=0;
    for (unsigned attempt=0;attempt<settings.TreeCount*20 && placed<settings.TreeCount;++attempt)
    {
        const float x=(unit(random)-0.5f)*settings.Extent*0.94f,z=(unit(random)-0.5f)*settings.Extent*0.94f;
        const float y=BigWorldHeight(settings,x,z);
        if (y<LakeElevation+3 || y>settings.MountainHeight*0.72f || TerrainNormal(settings,x,z).Y<0.72f) continue;
        if (Noise(x*0.018f,z*0.018f,settings.Seed+13)<0.35f) continue;
        Transform transform; transform.Translation={x,y-0.25f,z};
        const float scale=settings.TreeHeight/14*(0.65f+unit(random)*0.8f);
        transform.Scale={scale*(0.8f+unit(random)*0.3f),scale,scale}; transform.RotationDegrees.Y=unit(random)*360;
        const auto& meshes=variants[placed%4];
        const float distance=std::hypot(x-scene.SceneCamera.Position.X,z-scene.SceneCamera.Position.Z);
        const unsigned lod=distance<settings.TreeLodDistance?0:distance<settings.TreeLodDistance*2.5f?1:2;
        scene.Nodes.push_back({SceneNodeId(scene.Nodes.size()+1),"Pine "+std::to_string(placed++),transform,meshes[lod]});
        cache.Trees.push_back({scene.Nodes.back().Id,lod,meshes});
    }
}
float RenderedGround(const BigWorldSettings& s,float x,float z)
{
    // Root positions follow the actual piecewise-linear terrain, not the finer
    // continuous height function; otherwise dense grass can float or disappear.
    const float step=s.Extent/s.TerrainSegments;
    const float gx=(x+s.Extent*0.5f)/step,gz=(z+s.Extent*0.5f)/step;
    const float ix=std::floor(gx),iz=std::floor(gz),u=gx-ix,v=gz-iz;
    const float ax=ix*step-s.Extent*0.5f,az=iz*step-s.Extent*0.5f;
    const float a=BigWorldHeight(s,ax,az),b=BigWorldHeight(s,ax+step,az),c=BigWorldHeight(s,ax,az+step),d=BigWorldHeight(s,ax+step,az+step);
    return u>v?a+(b-a)*u+(d-b)*v:a+(d-c)*u+(c-a)*v;
}
constexpr float GrassPatchSize=20;
std::shared_ptr<const MeshAsset> BuildGrassPatch(const BigWorldSettings& s,int px,int pz,unsigned lod)
{
    auto mesh=std::make_shared<MeshAsset>(); mesh->Name="Root-anchored meadow LOD "+std::to_string(lod);
    const unsigned roots=unsigned(s.GrassCount*GrassPatchSize*GrassPatchSize/(Pi*70*70));
    const unsigned stride=lod==0?1:lod==1?4:16,segments=lod==0?3:lod==1?2:1;
    for (unsigned i=0;i<roots;i+=stride)
    {
        const unsigned seed=s.Seed+i*31;
        const float x=(px+Hash(px,pz,seed))*GrassPatchSize,z=(pz+Hash(px,pz,seed+1))*GrassPatchSize;
        if (std::abs(x)>s.Extent*0.49f || std::abs(z)>s.Extent*0.49f) continue;
        const float ground=RenderedGround(s,x,z);
        const auto surface=TerrainNormal(s,x,z);
        if (ground<LakeElevation+0.65f || ground>LakeElevation+s.MountainHeight*0.66f || surface.Y<0.8f) continue;
        const float lush=Noise(x*0.06f,z*0.06f,s.Seed+4);
        if (Hash(px,pz,seed+9)>0.55f+lush*0.55f) continue;
        const bool dry=Noise(x*0.035f,z*0.035f,s.Seed+8)>0.55f;
        for (unsigned blade=0;blade<3;++blade)
        {
            const float angle=Hash(px,pz,seed+blade+17)*2*Pi;
            const float height=s.GrassHeight*(0.45f+Hash(px,pz,seed+blade+3)*0.95f);
            const float width=s.GrassWidth*(lod==0?1:lod==1?1.65f:2.8f);
            const float lean=height*(Hash(px,pz,seed+blade+11)*0.35f+0.12f);
            const float dx=std::cos(angle),dz=std::sin(angle);
            const float ox=x+dx*blade*0.09f,oz=z+dz*blade*0.09f;
            const Color base=dry?Color{0.28f,0.27f,0.16f,1}:Color{0.20f,0.26f,0.13f,1};
            const Color tip=dry?Color{0.63f,0.56f,0.36f,1}:Color{0.45f,0.49f,0.28f,1};
            const unsigned first=unsigned(mesh->Vertices.size());
            for (unsigned segment=0;segment<=segments;++segment)
            {
                const float t=float(segment)/segments;
                for (unsigned edge=0;edge<2;++edge)
                {
                    const float side=(edge?1.0f:-1.0f)*width*(1-t)*0.5f;
                    Vertex v; v.Position={ox+dx*side-dz*lean*t*t,ground+height*t,oz+dz*side+dx*lean*t*t};
                    v.Normal=Unit({dz*0.45f,0.9f,-dx*0.45f}); v.VertexColor=Mix(base,tip,t);
                    v.TexCoord={float(edge),height*t}; mesh->Vertices.push_back(v);
                }
                if (segment)
                {
                    const unsigned a=first+(segment-1)*2,b=a+2;
                    mesh->Indices.insert(mesh->Indices.end(),{a,b,b+1,a,b+1,a+1});
                }
            }
        }
    }
    Finish(*mesh,"Meadow"); return mesh;
}
void UpdateVegetation(SceneSnapshot& scene,BigWorldState& state,bool lodEnabled)
{
    if (!state.Geometry) return;
    const auto& s=state.Settings; const auto eye=scene.SceneCamera.Position;
    std::shared_ptr<BigWorldGeometry> updated;
    const auto writable=[&]() -> BigWorldGeometry&
    { if (!updated) updated=std::make_shared<BigWorldGeometry>(*state.Geometry); return *updated; };
    if (s.Grass)
    {
        const int extent=int(std::ceil(s.GrassDistance/GrassPatchSize));
        const int cx=int(std::floor(eye.X/GrassPatchSize)),cz=int(std::floor(eye.Z/GrassPatchSize));
        std::map<std::pair<int,int>,BigWorldGeometry::GrassPatch> active;
        for (int z=cz-extent;z<=cz+extent;++z) for (int x=cx-extent;x<=cx+extent;++x)
        {
            const float distance=std::hypot((x+0.5f)*GrassPatchSize-eye.X,(z+0.5f)*GrassPatchSize-eye.Z);
            const auto key=std::make_pair(x,z); const auto existing=state.Geometry->Grass.find(key);
            const bool present=existing!=state.Geometry->Grass.end();
            if (distance>s.GrassDistance+(present?8:0)) continue;
            const float boundary=s.GrassLodDistance;
            const float near=boundary*(present && existing->second.Lod==0?1.15f:0.85f);
            const float far=boundary*2.5f*(present && existing->second.Lod<2?1.12f:0.88f);
            const unsigned lod=lodEnabled?(distance<near?0:distance<far?1:2):0;
            active[key]=present && existing->second.Lod==lod?existing->second:BigWorldGeometry::GrassPatch{lod,BuildGrassPatch(s,x,z,lod)};
        }
        bool changed=active.size()!=state.Geometry->Grass.size();
        for (const auto& entry:active)
        { const auto old=state.Geometry->Grass.find(entry.first); changed|=old==state.Geometry->Grass.end() || old->second.Mesh!=entry.second.Mesh; }
        if (changed)
        {
            writable().Grass=std::move(active);
            scene.Nodes.erase(std::remove_if(scene.Nodes.begin(),scene.Nodes.end(),[](const auto& n){return n.Name.find("Grass patch ")==0;}),scene.Nodes.end());
            for (const auto& entry:updated->Grass) if (!entry.second.Mesh->Indices.empty())
            {
                const auto x=entry.first.first,z=entry.first.second;
                scene.Nodes.push_back({SceneNodeId(1000000+unsigned(z+2048)*4096u+unsigned(x+2048)),
                    "Grass patch "+std::to_string(x)+":"+std::to_string(z),{},entry.second.Mesh});
            }
        }
    }
    for (unsigned i=0;i<state.Geometry->Trees.size();++i)
    {
        const auto& tree=state.Geometry->Trees[i];
        const auto node=std::find_if(scene.Nodes.begin(),scene.Nodes.end(),[&](const auto& n){return n.Id==tree.Id;});
        if (node==scene.Nodes.end()) continue;
        const auto pos=node->LocalTransform.Translation;
        const float distance=std::hypot(pos.X-eye.X,pos.Z-eye.Z);
        const float near=s.TreeLodDistance*(tree.Lod==0?1.15f:0.85f),far=s.TreeLodDistance*2.5f*(tree.Lod<2?1.12f:0.88f);
        const unsigned lod=lodEnabled?(distance<near?0:distance<far?1:2):0;
        if (lod!=tree.Lod) { writable().Trees[i].Lod=lod; node->Mesh=tree.Meshes[lod]; }
    }
    if (updated) state.Geometry=std::move(updated);
}
Float3 SunDirection(float hour)
{
    const float a=(hour-6)*Pi/12;
    return Unit({std::cos(a),std::sin(a),0.3f*std::cos(a)});
}
Float3 CausticSun(const SceneSnapshot& scene,const BigWorldSettings& s)
{
    if (!scene.Lights.empty() && scene.Lights.front().EditorOverride)
    {
        const auto& light=scene.Lights.front();
        return light.Enabled && light.Intensity>0 ? Unit(Scale(light.Direction,-1)) : Float3{};
    }
    return s.SunIntensity>0?SunDirection(s.Hour):Float3{};
}
Techniques::UnderwaterLightSettings LakeLight(const SceneSnapshot& scene,const BigWorldState& state)
{
    Techniques::UnderwaterLightSettings water;
    const auto& s=state.Settings;
    if (!s.Water || !s.UnderwaterLighting || !state.Waves) return water;
    const auto sun=CausticSun(scene,s);
    const float eta=1/s.WaterIOR;
    water.SurfaceHeight=LakeElevation; water.BottomHeight=LakeElevation-LakeDepth;
    water.Radius=LakeRadius+35; water.Extinction=s.Absorption;
    water.ToSun={eta*sun.X,std::sqrt(1-eta*eta*(1-sun.Y*sun.Y)),eta*sun.Z};
    if (sun.Y>0 && !scene.Lights.empty())
    {
        const auto& light=scene.Lights.front();
        const float power=light.Enabled && light.Type==LightType::Directional?light.Intensity*sun.Y:0;
        water.Irradiance={power*light.LightColor.R,power*light.LightColor.G,power*light.LightColor.B};
    }
    const auto waves=state.Waves->LightWaves();
    water.WaveA=waves[0]; water.WaveB=waves[1];
    return water;
}
// Estimate incident sunlight along the short refracted path represented by each
// surface vertex. The submerged camera resolves this same field per pixel.
float RefractedShaftLight(const Techniques::UnderwaterLightSettings& water,Float3 start,Float3 ray,float path)
{
    const auto& sun=water.ToSun; const auto& a=water.WaveA; const auto& b=water.WaveB;
    const float cosine=ray.X*sun[0]+ray.Y*sun[1]+ray.Z*sun[2];
    const float phase=0.84f/std::pow(std::max(0.36f,1.16f-0.8f*cosine),1.5f);
    float sum=0;
    for (float fraction:{0.2f,0.5f,0.8f})
    {
        const float distance=path*fraction;
        const float depth=std::max(0.0f,water.SurfaceHeight-start.Y-distance*ray.Y);
        const float sunlightPath=depth/sun[1];
        const float x=start.X+ray.X*distance+sun[0]*sunlightPath,z=start.Z+ray.Z*distance+sun[2]*sunlightPath;
        const float u=a[0]*x+a[1]*z,v=b[0]*x+b[1]*z;
        // Apply fractional frequencies to space only; the animation phases wrap at 2 pi.
        const float warp=std::sin(0.43f*u-0.63f*v+a[2]-b[2]);
        const float bandA=std::pow(0.5f+0.5f*std::cos(u+a[2]+2.1f*std::sin(v+b[2]+0.65f*warp)),16.0f);
        const float bandB=std::pow(0.5f+0.5f*std::cos(v+b[2]-1.2f*std::sin(0.63f*u+a[2])),12.0f);
        const float total=a[3]+b[3];
        const float focus=Mix(1,0.07f+(8.2f*a[3]*bandA+5.8f*b[3]*bandB)/std::max(0.01f,total),
            Saturate(1.35f*total)*(1-std::exp(-0.65f*depth)));
        sum+=focus*std::exp(-water.Extinction*0.85f*sunlightPath);
    }
    return phase*sum/3;
}
std::uint16_t Half(float value)
{
    std::uint32_t bits; std::memcpy(&bits,&value,sizeof(bits));
    const int exponent=int((bits>>23)&255)-127+15;
    if (exponent<=0) return 0;
    if (exponent>=31) return 0x7bff;
    return std::uint16_t((unsigned(exponent)<<10)|((bits&0x7fffff)>>13));
}
std::shared_ptr<const Renderer::EnvironmentRadiance> BuildSky(Float3 eye,const BigWorldSettings& s,const std::shared_ptr<const Renderer::AtmosphereModel>& model,
    const Techniques::AtmosphereSettings& atmosphere)
{
    auto sky=std::make_shared<Renderer::EnvironmentRadiance>(); sky->Resolution=64;
    const auto sun=SunDirection(s.Hour);
    const float daylight=Smooth((sun.Y+0.09f)*3),twilight=std::exp(-std::abs(sun.Y)*12);
    const float storm=s.Weather==WorldWeather::Clear?0.0f:s.Precipitation*0.75f;
    for (unsigned face=0;face<6;++face) for (unsigned y=0;y<sky->Resolution;++y) for (unsigned x=0;x<sky->Resolution;++x)
    {
        Renderer::Float3 direction;
        if (!Renderer::EnvironmentFaceDirection(Renderer::EnvironmentFace(face),
            {(x+0.5f)/sky->Resolution,(y+0.5f)/sky->Resolution},direction)) continue;
        const float height=Saturate(direction.Y),horizon=std::pow(1-height,6.0f);
        Color color=Mix({0.008f,0.012f,0.035f,1},Mix({0.12f,0.30f,0.62f,1},{0.56f,0.66f,0.76f,1},horizon),daylight);
        color=Mix(color,{0.8f,0.22f,0.055f,1},twilight*horizon*0.65f);
        color=Mix(color,{0.18f*daylight+0.006f,0.21f*daylight+0.008f,0.25f*daylight+0.012f,1},storm);
        if (direction.Y<0) color=Mix(color,{0.045f*daylight,0.065f*daylight,0.035f*daylight,1},Smooth(-direction.Y*5));
        const float cosine=direction.X*sun.X+direction.Y*sun.Y+direction.Z*sun.Z;
        const float disk=Smooth((cosine-0.9992f)/0.0005f)*daylight*s.SunIntensity*(1-storm);
        const float moon=Smooth((-cosine-0.9994f)/0.0004f)*(1-daylight)*s.MoonIntensity*12;
        const float stars=direction.Y>0 && Hash(int(direction.X*2500),int(direction.Z*2500),s.Seed)>0.998f?
            (1-daylight)*(1-storm)*0.6f:0;
        if (atmosphere.Enabled && model)
        {
            const auto transport=model->Sky({eye.X,eye.Y,eye.Z},direction,{sun.X,sun.Y,sun.Z},
                {atmosphere.SunRadiance[0],atmosphere.SunRadiance[1],atmosphere.SunRadiance[2]});
            const Color boundary=transport.Ground ? Color{0.045f*daylight*s.SunIntensity,0.065f*daylight*s.SunIntensity,0.035f*daylight*s.SunIntensity,1} :
                Color{disk*8+moon*0.65f+stars,disk*8+moon*0.78f+stars,disk*8+moon+stars,1};
            color={transport.Radiance.X+transport.Transmittance.X*boundary.R,
                transport.Radiance.Y+transport.Transmittance.Y*boundary.G,
                transport.Radiance.Z+transport.Transmittance.Z*boundary.B,1};
        }
        else
        { color.R+=disk+moon*0.65f+stars; color.G+=disk*0.87f+moon*0.78f+stars; color.B+=disk*0.65f+moon+stars; }
        auto& pixels=sky->FacesRGBA16[face]; pixels.insert(pixels.end(),{Half(color.R),Half(color.G),Half(color.B),Half(1)});
    }
    return sky;
}
unsigned WaterLod(const BigWorldSettings& s,Float3 eye,int x,int z,int previous=-1)
{
    const float distance=std::hypot((x+0.5f)*s.TileSize-eye.X,(z+0.5f)*s.TileSize-eye.Z);
    const float near=s.WaterLodDistance*(previous==0?1.1f:previous>0?0.9f:1.0f);
    const float far=s.WaterLodDistance*(previous>=2?1.8f:previous>=0?2.2f:2.0f);
    const float distant=s.WaterLodDistance*(previous==3?3.6f:previous>=0?4.4f:4.0f);
    return distance<near?0:distance<far?1:distance<distant?2:3;
}
std::shared_ptr<const BigWorldGeometry::WaterTile> BuildWaterTile(const BigWorldSettings& s,int x,int z,unsigned lod)
{
    auto result=std::make_shared<BigWorldGeometry::WaterTile>(); auto& tile=*result; tile.X=x; tile.Z=z; tile.Lod=lod;
    const unsigned segments=tile.Segments=std::max(4u,s.WaterSegments>>lod);
    auto& mesh=tile.Mesh; mesh.Name="FFT and Gerstner lake";
    const unsigned width=segments+1;
    std::vector<float> depths(width*width);
    std::vector<unsigned> remap(depths.size(),~0u);
    for (unsigned row=0;row<=segments;++row) for (unsigned column=0;column<=segments;++column)
    {
        const float px=(x+float(column)/segments)*s.TileSize,pz=(z+float(row)/segments)*s.TileSize;
        depths[row*width+column]=LakeElevation-BigWorldHeight(s,px,pz);
    }
    const auto surfaceVertex=[&](unsigned grid)
    {
        auto& index=remap[grid];
        if (index==~0u)
        {
            index=unsigned(mesh.Vertices.size());
            Vertex v; v.Position={(x+float(grid%width)/segments)*s.TileSize,LakeElevation,(z+float(grid/width)/segments)*s.TileSize};
            mesh.Vertices.push_back(v);
            tile.Surface.push_back({depths[grid],Smooth(depths[grid]/3)});
        }
        return index;
    };
    for (unsigned row=0;row<segments;++row) for (unsigned column=0;column<segments;++column)
    {
        const unsigned a=row*(segments+1)+column,b=a+segments+1;
        if (depths[a]>-0.5f || depths[b]>-0.5f || depths[b+1]>-0.5f || depths[a+1]>-0.5f)
        {
            const auto i=surfaceVertex(a),j=surfaceVertex(b),k=surfaceVertex(b+1),l=surfaceVertex(a+1);
            mesh.Indices.insert(mesh.Indices.end(),{i,j,k,i,k,l});
        }
    }
    if (mesh.Indices.empty()) return result;
    const auto skirt=[&](unsigned a,unsigned b)
    {
        if (depths[a]<=0 && depths[b]<=0) return;
        a=remap[a]; b=remap[b];
        for (const auto corner:std::initializer_list<std::pair<unsigned,bool>>{{a,false},{a,true},{b,true},{a,false},{b,true},{b,false}})
        { tile.Skirts.push_back(corner); mesh.Vertices.push_back(mesh.Vertices[corner.first]); }
        const unsigned first=unsigned(mesh.Vertices.size())-6;
        mesh.Indices.insert(mesh.Indices.end(),{first,first+1,first+2,first+2,first+1,first,
            first+3,first+4,first+5,first+5,first+4,first+3});
    };
    for (unsigned i=0;i<segments;++i)
    { skirt(i,i+1); skirt(segments*(segments+1)+i,segments*(segments+1)+i+1);
      skirt(i*(segments+1),(i+1)*(segments+1)); skirt(i*(segments+1)+segments,(i+1)*(segments+1)+segments); }
    Finish(mesh,"Water"); return result;
}
void SortWaterWork(BigWorldGeometry& geometry)
{
    // Start the fine shoreline grids first so their longer jobs overlap.
    std::stable_sort(geometry.Water.begin(),geometry.Water.end(),[](const auto& a,const auto& b)
    { return a->Surface.size()>b->Surface.size(); });
}
std::shared_ptr<const BigWorldGeometry> BuildGeometry(SceneSnapshot& scene,const BigWorldSettings& s)
{
    auto result=std::make_shared<BigWorldGeometry>();
    for (const auto& node:scene.Nodes) if (node.Name.find("Lakebed ")==0)
        result->Lakebed.emplace_back(node.Id,node.Mesh);
    BuildTrees(scene,s,*result);
    if (!s.Water) return result;
    const int extent=int(std::ceil((LakeRadius+40)/s.TileSize));
    // Retain the active topology per tile. Fine grids are allocated near the
    // camera, keeping memory proportional to the visible LOD distribution.
    for (int z=-extent;z<extent;++z) for (int x=-extent;x<extent;++x)
    {
        if (std::hypot((x+0.5f)*s.TileSize,(z+0.5f)*s.TileSize)>LakeRadius+s.TileSize+30) continue;
        result->Water.push_back(BuildWaterTile(s,x,z,WaterLod(s,scene.SceneCamera.Position,x,z)));
    }
    SortWaterWork(*result);
    return result;
}
void UpdateWaterLods(const SceneSnapshot& scene,BigWorldState& state)
{
    if (!state.Geometry || !state.Settings.Water) return;
    std::shared_ptr<BigWorldGeometry> updated;
    for (unsigned i=0;i<state.Geometry->Water.size();++i)
    {
        const auto& tile=*state.Geometry->Water[i];
        const auto lod=WaterLod(state.Settings,scene.SceneCamera.Position,tile.X,tile.Z,int(tile.Lod));
        if (lod==tile.Lod) continue;
        if (!updated) updated=std::make_shared<BigWorldGeometry>(*state.Geometry);
        updated->Water[i]=BuildWaterTile(state.Settings,tile.X,tile.Z,lod);
    }
    if (updated)
    {
        SortWaterWork(*updated);
        state.Geometry=std::move(updated);
    }
}
Color SubmergedInterface(const Renderer::EnvironmentRadiance& sky,Float3 view,Float3 normal,float ior,Color reflected)
{
    // Snell's window transmits the sky inside the critical angle. Outside it
    // the underside reflects light from the lake interior.
    const float cosine=Saturate(-view.X*normal.X-view.Y*normal.Y-view.Z*normal.Z);
    const float sinSquared=ior*ior*(1-cosine*cosine);
    if (sinSquared<1)
    {
        const float airCosine=std::sqrt(1-sinSquared);
        const auto air=Add(Scale(view,-ior),Scale(normal,airCosine-ior*cosine));
        Renderer::EnvironmentLookup lookup;
        if (Renderer::LocateEnvironmentDirection({air.X,air.Y,air.Z},lookup))
        {
            const unsigned side=sky.Resolution;
            const auto radiance=sky.Texel(lookup.Face,std::min(unsigned(lookup.UV.X*side),side-1),
                std::min(unsigned(lookup.UV.Y*side),side-1));
            const float rs=(ior*cosine-airCosine)/(ior*cosine+airCosine);
            const float rp=(cosine-ior*airCosine)/(cosine+ior*airCosine);
            reflected=Mix({radiance.X,radiance.Y,radiance.Z,1},reflected,0.5f*(rs*rs+rp*rp));
        }
    }
    return {Maths::LinearToSRGB(reflected.R),Maths::LinearToSRGB(reflected.G),Maths::LinearToSRGB(reflected.B),1};
}
void ForWorldPatches(unsigned count,const std::function<void(unsigned)>& update)
{
    static Rasterizer::Detail::ParallelExecutor workers;
    const unsigned workerCount=std::clamp(std::thread::hardware_concurrency()/2,1u,8u);
    workers.Run(count,workerCount,update);
}
void BuildWater(SceneSnapshot& scene,const BigWorldState& state,const Techniques::UnderwaterLightSettings& light)
{
    const auto& s=state.Settings;
    if (!s.Water || !state.Waves || !state.Geometry) return;
    const auto sun=CausticSun(scene,s);
    const auto eye=scene.SceneCamera.Position;
    const float eta=1/s.WaterIOR,wind=s.WindDirection*Pi/180;
    const float f0=(1-eta)*(1-eta)/((1+eta)*(1+eta));
    const float wx=std::cos(wind),wz=std::sin(wind);
    const float drift=float(std::fmod(state.WaterTime*std::min(0.85f,s.WindSpeed*0.07f)/RipplePatch,1.0));
    const Color scatter{Maths::SRGBToLinear(s.WaterTint[0]),Maths::SRGBToLinear(s.WaterTint[1]),
        Maths::SRGBToLinear(s.WaterTint[2]),1};
    const auto& tiles=state.Geometry->Water;
    std::vector<std::shared_ptr<const MeshAsset>> surfaces(tiles.size());
    ForWorldPatches(static_cast<unsigned>(tiles.size()),[&](unsigned index)
    {
        const auto& tile=*tiles[index];
        if (tile.Mesh.Indices.empty()) return;
        auto mesh=tile.Mesh;
        // Coarse facets cannot resolve the alternating bright and dark caustic
        // bands. Fade the modulation to its mean as the vertex spacing grows.
        const float causticWeight=tile.Lod==0?1.0f:1-Smooth((s.TileSize/tile.Segments-1)/2);
        for (unsigned i=0;i<tile.Surface.size();++i)
        {
            auto& vertex=mesh.Vertices[i]; const auto& bed=tile.Surface[i];
            const float x=vertex.Position.X,z=vertex.Position.Z;
            const auto wave=state.Waves->Sample(x,z);
            vertex.Position={x+wave.Displacement.X*bed.Shore,LakeElevation+wave.Displacement.Y*bed.Shore,z+wave.Displacement.Z*bed.Shore};
            vertex.Normal=Unit({wave.Normal.X*bed.Shore,Mix(1,wave.Normal.Y,bed.Shore),wave.Normal.Z*bed.Shore});
            const float windX=wx*x+wz*z,windZ=-wz*x+wx*z;
            vertex.TexCoord={windX/RipplePatch-drift+0.16f*std::sin(windZ*0.21f+windX*0.08f),
                windZ/RipplePatch+0.12f*std::sin(windX*0.19f-windZ*0.13f)};

            // Trace the view through the local facet to the lakebed. Snell's law
            // limits the submerged path even when looking across the surface.
            const auto view=Unit({eye.X-vertex.Position.X,eye.Y-vertex.Position.Y,eye.Z-vertex.Position.Z});
            const auto n=vertex.Normal;
            if (state.CameraUnderwater && scene.Environment && scene.Environment->Resolution>0)
            {
                vertex.VertexColor=SubmergedInterface(*scene.Environment,view,n,s.WaterIOR,scatter);
                continue;
            }
            const float cosine=Saturate(view.X*n.X+view.Y*n.Y+view.Z*n.Z);
            const float bend=eta*cosine-std::sqrt(1-eta*eta*(1-cosine*cosine));
            const Float3 ray{bend*n.X-eta*view.X,bend*n.Y-eta*view.Y,bend*n.Z-eta*view.Z};
            const float depth=std::max(0.0f,bed.Depth+wave.Displacement.Y*bed.Shore);
            const float path=depth/std::max(0.2f,-ray.Y);
            const float bedX=vertex.Position.X+path*ray.X,bedZ=vertex.Position.Z+path*ray.Z;
            const float focus=causticWeight>0?causticWeight*state.Waves->Caustic(bedX,bedZ,depth,sun):0;
            const auto ground=LakebedColor(bedX,bedZ,s.Seed);
            const float opticalDepth=path*s.Absorption;
            const float transmittedFraction=(1-f0)*(1-std::pow(1-cosine,5.0f));
            const float shafts=light.Extinction>0 && eye.Y>vertex.Position.Y && sun.Y>0 ?
                RefractedShaftLight(light,vertex.Position,ray,path) : 0;
            const auto channel=[&](float bottom,float water,float extinction,float incident)
            {
                const float transmission=std::exp(-opticalDepth*extinction);
                return Maths::LinearToSRGB(transmittedFraction*Mix(water+incident*shafts,Maths::SRGBToLinear(bottom)*(1+focus),transmission));
            };
            vertex.VertexColor={channel(ground.R,scatter.R,2.5f,0.010f*light.Irradiance[0]),
                channel(ground.G,scatter.G,0.83f,0.050f*light.Irradiance[1]),
                channel(ground.B,scatter.B,0.64f,0.068f*light.Irradiance[2]),1};
        }
        for (unsigned i=0;i<tile.Skirts.size();++i)
        {
            auto vertex=mesh.Vertices[tile.Skirts[i].first];
            if (tile.Skirts[i].second && !state.CameraUnderwater) vertex.Position.Y-=2;
            mesh.Vertices[tile.Surface.size()+i]=vertex;
        }
        surfaces[index]=std::make_shared<MeshAsset>(std::move(mesh));
    });
    for (std::size_t i=0;i<tiles.size();++i) if (surfaces[i])
    {
        const auto& tile=*tiles[i];
        const auto id=100000u+unsigned(tile.Z+128)*256u+unsigned(tile.X+128);
        scene.Nodes.push_back({id,"Water tile "+std::to_string(tile.X)+":"+std::to_string(tile.Z),{},std::move(surfaces[i])});
    }
}
MeshAsset BuildWeather(const SceneSnapshot& scene,const BigWorldState& state)
{
    MeshAsset mesh; mesh.Name="Precipitation"; const auto& s=state.Settings;
    if (s.Weather==WorldWeather::Clear || s.Precipitation<=0) return mesh;
    const auto camera=scene.SceneCamera.Position;
    const bool snow=s.Weather==WorldWeather::Snow;
    const float fall=snow?2.8f:28,angle=s.WindDirection*Pi/180;
    const float time=float(std::fmod(scene.SampleTimeSeconds,10000.0));
    const Float3 forward=Unit({scene.SceneCamera.Target.X-camera.X,scene.SceneCamera.Target.Y-camera.Y,scene.SceneCamera.Target.Z-camera.Z});
    const Float3 right=Unit({-forward.Z,0,forward.X});
    const Float3 up=Unit({forward.Y*right.Z,forward.Z*right.X-forward.X*right.Z,-forward.Y*right.X});
    const auto wrap=[](float value,float size){return value-std::floor(value/size)*size;};
    for (unsigned i=0;i<unsigned(s.Precipitation*1800);++i)
    {
        const float x=camera.X+wrap(Hash(int(i),3,s.Seed)*100+time*s.WindSpeed*std::cos(angle)-camera.X,100)-50;
        const float z=camera.Z+wrap(Hash(int(i),7,s.Seed)*100+time*s.WindSpeed*std::sin(angle)-camera.Z,100)-50;
        const float y=camera.Y+wrap(Hash(int(i),11,s.Seed)*70-time*fall-camera.Y,70)-20;
        if (y<std::max(LakeElevation,BigWorldHeight(s,x,z))) continue;
        const float radius=snow?0.13f:0.035f,height=snow?0.13f:1.7f;
        Vertex a,b,c,d; const Color color=snow?Color{0.9f,0.94f,1,1}:Color{0.42f,0.57f,0.72f,1};
        a.Position={x-right.X*radius-up.X*height,y-up.Y*height,z-right.Z*radius-up.Z*height};
        b.Position={x+right.X*radius-up.X*height,y-up.Y*height,z+right.Z*radius-up.Z*height};
        c.Position={x+right.X*radius+up.X*height,y+up.Y*height,z+right.Z*radius+up.Z*height};
        d.Position={x-right.X*radius+up.X*height,y+up.Y*height,z-right.Z*radius+up.Z*height};
        a.VertexColor=b.VertexColor=c.VertexColor=d.VertexColor=color;
        a.Normal=b.Normal=c.Normal=d.Normal={-forward.X,-forward.Y,-forward.Z};
        Triangle(mesh,a,b,c,true); Triangle(mesh,a,c,d,true);
    }
    Finish(mesh,"Precipitation"); return mesh;
}
Techniques::UnderwaterLightSettings DynamicGeometry(SceneSnapshot& scene,BigWorldState& state)
{
    scene.Nodes.erase(std::remove_if(scene.Nodes.begin(),scene.Nodes.end(),[](const auto& node)
    { return node.Name=="Weather" || node.Name.find("Water tile ")==0; }),scene.Nodes.end());
    auto light=LakeLight(scene,state);
    state.CameraUnderwater=false;
    float surface=LakeElevation;
    if (state.Settings.Water && state.Waves)
    {
        const auto eye=scene.SceneCamera.Position;
        const float bed=BigWorldHeight(state.Settings,eye.X,eye.Z);
        surface+=Smooth((LakeElevation-bed)/3)*state.Waves->Sample(eye.X,eye.Z).Displacement.Y;
        state.CameraUnderwater=eye.Y>=bed && eye.Y<surface;
    }
    BuildWater(scene,state,light);
    if (state.Settings.Weather!=WorldWeather::Clear) { Node(scene,"Weather",BuildWeather(scene,state)); scene.Nodes.back().Id=90001; }
    const auto sun=CausticSun(scene,state.Settings);
    std::vector<SceneNode*> bedNodes;
    for (auto& node:scene.Nodes) if (node.Name.find("Lakebed ")==0 && node.Mesh) bedNodes.push_back(&node);
    ForWorldPatches(unsigned(bedNodes.size()),[&](unsigned i)
    {
        auto& node=*bedNodes[i];
        const auto source=std::find_if(state.Geometry->Lakebed.begin(),state.Geometry->Lakebed.end(),[&](const auto& entry){return entry.first==node.Id;});
        if (source==state.Geometry->Lakebed.end()) return;
        if (!state.Settings.Water || !state.Settings.Caustics || state.Settings.CausticIntensity<=0 || sun.Y<=0.02f)
        {
            node.Mesh=source->second;
            return;
        }
        auto mesh=std::make_shared<MeshAsset>(*source->second);
        for (auto& vertex:mesh->Vertices)
        {
            const float depth=LakeElevation-vertex.Position.Y;
            const float focus=state.Waves->Caustic(vertex.Position.X,vertex.Position.Z,depth,sun);
            const float energy=1+focus*0.35f;
            vertex.VertexColor.R=std::min(1.0f,vertex.VertexColor.R*energy);
            vertex.VertexColor.G=std::min(1.0f,vertex.VertexColor.G*energy);
            vertex.VertexColor.B=std::min(1.0f,vertex.VertexColor.B*energy);
        }
        node.Mesh=mesh;
    });
    if (!state.CameraUnderwater) return {};
    light.SurfaceHeight=surface;
    return light;
}
}

BigWorldSettings SanitizeBigWorldSettings(BigWorldSettings s)
{
    const BigWorldSettings defaults;
    const auto limit=[](float& value,float fallback,float minimum,float maximum)
    { value=std::clamp(std::isfinite(value)?value:fallback,minimum,maximum); };
    limit(s.Extent,defaults.Extent,600,3000); limit(s.MountainHeight,defaults.MountainHeight,30,450);
    s.TerrainSegments=std::clamp((s.TerrainSegments/16)*16,64u,256u);
    s.TreeCount=std::min(s.TreeCount,1500u); s.GrassCount=std::min(s.GrassCount,180000u);
    limit(s.WindSpeed,defaults.WindSpeed,0,25); limit(s.WindDirection,defaults.WindDirection,0,360);
    limit(s.GrassHeight,defaults.GrassHeight,0.2f,4); limit(s.TreeHeight,defaults.TreeHeight,4,35);
    limit(s.GrassWidth,defaults.GrassWidth,0.01f,0.15f); limit(s.GrassDistance,defaults.GrassDistance,40,220);
    limit(s.GrassLodDistance,defaults.GrassLodDistance,15,80); limit(s.TreeLodDistance,defaults.TreeLodDistance,50,300);
    limit(s.GustStrength,defaults.GustStrength,0,2); limit(s.GustWavelength,defaults.GustWavelength,5,120);
    limit(s.LeafFlutter,defaults.LeafFlutter,0,0.7f);
    limit(s.Hour,defaults.Hour,0,24); limit(s.DayDuration,defaults.DayDuration,30,3600);
    limit(s.SunIntensity,defaults.SunIntensity,0,12); limit(s.MoonIntensity,defaults.MoonIntensity,0,1);
    limit(s.CameraSpeed,defaults.CameraSpeed,1,200); limit(s.IBLIntensity,defaults.IBLIntensity,0,4);
    s.Atmosphere=Techniques::SanitizeAtmosphereSettings(s.Atmosphere);
    s.Atmosphere.GroundHeight=LakeElevation;
    limit(s.Precipitation,defaults.Precipitation,0,1); limit(s.SnowCover,defaults.SnowCover,0,1);
    limit(s.FogDensity,defaults.FogDensity,0,0.03f); limit(s.FogHeight,defaults.FogHeight,-20,200);
    limit(s.FogFalloff,defaults.FogFalloff,0,0.1f); limit(s.FogDistance,defaults.FogDistance,100,10000);
    limit(s.CloudAltitude,defaults.CloudAltitude,80,900); limit(s.CloudThickness,defaults.CloudThickness,40,400);
    limit(s.CloudDensity,defaults.CloudDensity,0,0.3f); limit(s.CloudSpeed,defaults.CloudSpeed,0,5);
    s.FFTSize=s.FFTSize<=32?32:s.FFTSize<=64?64:128;
    limit(s.FFTPatch,defaults.FFTPatch,32,512); limit(s.WaveAmplitude,defaults.WaveAmplitude,0,3);
    limit(s.Choppiness,defaults.Choppiness,0,2); limit(s.SwellAmplitude,defaults.SwellAmplitude,0,3);
    limit(s.SwellLength,defaults.SwellLength,5,100); limit(s.SwellSteepness,defaults.SwellSteepness,0,0.9f);
    limit(s.WaterSpeed,defaults.WaterSpeed,0,4); limit(s.WaterRoughness,defaults.WaterRoughness,0.045f,0.6f);
    limit(s.WaterIOR,defaults.WaterIOR,1.01f,2); limit(s.Absorption,defaults.Absorption,0.005f,0.5f);
    for (unsigned i=0;i<3;++i) limit(s.WaterTint[i],defaults.WaterTint[i],0,1);
    limit(s.TileSize,defaults.TileSize,24,128); limit(s.WaterLodDistance,defaults.WaterLodDistance,30,500);
    s.WaterSegments=s.WaterSegments<=16?16:s.WaterSegments<=32?32:64;
    limit(s.CausticIntensity,defaults.CausticIntensity,0,5); limit(s.CausticDepth,defaults.CausticDepth,1,50);
    if (unsigned(s.Weather)>unsigned(WorldWeather::Snow)) s.Weather=WorldWeather::Clear;
    if (unsigned(s.Fog)>unsigned(WorldFog::Linear)) s.Fog=WorldFog::Height;
    return s;
}

float BigWorldHeight(const BigWorldSettings& s,float x,float z)
{
    const float radius=std::hypot(x,z);
    const float shoreline=LakeRadius*(0.93f+0.13f*Noise(x*0.006f,z*0.006f,s.Seed));
    const float land=Smooth((radius-shoreline+30)/75);
    const float hills=Fbm(x*0.006f,z*0.006f,s.Seed);
    const float qx=x+Noise(x*0.004f,z*0.004f,s.Seed+19)*65;
    const float qz=z+Noise(x*0.004f,z*0.004f,s.Seed+23)*65;
    const float ridge=1-std::abs(2*Noise(qx*0.005f+9,qz*0.005f-7,s.Seed+3)-1);
    const float detail=1-std::abs(2*Noise(qx*0.019f,qz*0.019f,s.Seed+7)-1);
    const float mountain=Smooth((radius-shoreline-70)/190)*std::pow(ridge,2.8f)*(0.72f+detail*0.28f)*s.MountainHeight;
    return LakeElevation-LakeDepth+(18+hills*24)*land+mountain;
}

void BuildBigWorld(SceneSnapshot& scene)
{
    auto state=std::make_shared<BigWorldState>(scene.World?*scene.World:BigWorldState{});
    state->Settings=SanitizeBigWorldSettings(state->Settings); const auto& s=state->Settings;
    scene.Nodes.clear(); scene.Materials.clear();
    const auto material=[&](const char* name,float roughness,Color color)
    { Material value; value.Name=name; value.Diffuse=color; value.Roughness=roughness; value.Metallic=0.0f; scene.Materials.push_back(value); };
    material("Landscape",s.Weather==WorldWeather::Rain?0.5f:0.95f,{1,1,1,1});
    material("Meadow",0.83f,{1,1,1,1}); scene.Materials.back().SubsurfaceWeight=0.3f;
    material("Bark",0.95f,{1,1,1,1}); material("Foliage",0.88f,{1,1,1,1}); material("Canopy",0.92f,{1,1,1,1});
    material("Water",s.WaterRoughness,{1,1,1,1});
    auto& water=scene.Materials.back();
    SetWaterOptics(water,s);
    water.NormalImage=LakeRippleNormals(RippleStrength(s));
    material("Precipitation",0.6f,{1,1,1,1});
    scene.Materials.back().Emissive={0.05f,0.06f,0.08f,1};
    auto waves=std::make_shared<WaterSpectrum>(s); waves->Evaluate(state->WaterTime); state->Waves=waves;
    state->Atmosphere=s.Atmosphere.Enabled ? std::make_shared<Renderer::AtmosphereModel>(s.Atmosphere) : nullptr;
    VegetationMaterials(scene); BuildTerrain(scene,s);
    state->Geometry=BuildGeometry(scene,s); state->SkyJob.reset(); UpdateVegetation(scene,*state,true); DynamicGeometry(scene,*state);
    state->SkyHour=-1; scene.World=state; ++scene.Revision;
}

void ConfigureBigWorld(SceneSnapshot& scene,const BigWorldSettings& requested)
{
    if (!scene.World) return;
    auto state=std::make_shared<BigWorldState>(*scene.World);
    const auto previous=state->Settings;
    state->Settings=SanitizeBigWorldSettings(requested);
    const auto geometryKey=[](const BigWorldSettings& s)
    {
        return std::make_tuple(s.Seed,s.Extent,s.MountainHeight,s.TerrainSegments,
            s.TreeCount,s.GrassCount,s.Terrain,s.Trees,s.Grass,s.GrassHeight,s.GrassWidth,s.GrassDistance,s.GrassLodDistance,s.TreeLodDistance,s.TreeHeight,s.Weather,s.SnowCover,
            s.Water,s.TileSize,s.WaterSegments);
    };
    state->SettingsChanged=true;
    scene.World=state;
    if (geometryKey(previous)!=geometryKey(state->Settings)) { BuildBigWorld(scene); return; }
    if (state->Settings.Water && !state->CameraUnderwater && previous.UnderwaterLighting!=state->Settings.UnderwaterLighting)
        ++scene.Revision;
    const auto transportKey=[](const BigWorldSettings& s)
    {
        return std::make_tuple(s.WaterTint,s.WaterRoughness,s.WaterIOR,s.Absorption,s.FFTPatch,s.FFT,s.Gerstner,s.FFTSize,s.WindSpeed,s.WindDirection,
            s.WaveAmplitude,s.Choppiness,s.SwellAmplitude,s.SwellLength,s.SwellSteepness,
            s.Caustics,s.CausticIntensity,s.CausticDepth);
    };
    // Explicit appearance/deformation edits replace the cached transport proxy;
    // ordinary animation retains it without regenerating SDFs every frame.
    if (transportKey(previous)!=transportKey(state->Settings))
    {
        ++scene.Revision;
        auto waves=std::make_shared<WaterSpectrum>(state->Settings);
        waves->Evaluate(state->WaterTime);
        state->Waves=std::move(waves);
        for (auto& material:scene.Materials) if (material.Name=="Water")
        {
            SetWaterOptics(material,state->Settings);
            if (RippleStrength(previous)!=RippleStrength(state->Settings))
                material.NormalImage=LakeRippleNormals(RippleStrength(state->Settings));
        }
    }
    if (previous.Hour!=state->Settings.Hour || previous.SunIntensity!=state->Settings.SunIntensity ||
        previous.MoonIntensity!=state->Settings.MoonIntensity || previous.Precipitation!=state->Settings.Precipitation)
    {
        state->SkyJob.reset(); state->SkyHour=-1;
    }
    if (!(previous.Atmosphere==state->Settings.Atmosphere))
    {
        state->Atmosphere=state->Settings.Atmosphere.Enabled ? std::make_shared<Renderer::AtmosphereModel>(state->Settings.Atmosphere) : nullptr;
        state->SkyJob.reset(); state->SkyHour=-1;
    }
}

void ApplyBigWorldAppearance(const SceneSnapshot& scene,Techniques::Settings& settings)
{
    using namespace Techniques;
    settings={};
    settings.PostProcessing.HDR.ToneMapping=ToneMappingMode::ACESFitted;
    settings.PostProcessing.AntiAliasing=AntiAliasingMode::FXAA;
    settings.FrustumCulling=true; settings.Instancing=true; settings.Lod.Enabled=true; settings.Lod.MaximumErrorPixels=1.5f;
    settings.Shadows=ShadowRepresentation::Cascaded; settings.Cascades.CascadeCount=4;
    settings.Cascades.MaximumDistance=500; settings.Cascades.SplitLambda=0.88f; settings.Cascades.BlendFraction=0.15f;
    settings.ContactShadows.Enabled=true; settings.ContactShadows.MaximumDistance=1.2f;
    settings.ContactShadows.Thickness=0.12f; settings.ContactShadows.MaximumSteps=24;
    settings.Reflections.Skybox=true; settings.Reflections.Mode=ReflectionMode::Cubemap;
    settings.Reflections.Planar.Enabled=true;
    settings.Reflections.EnvironmentIntensity=1; settings.Reflections.MaximumDistance=1500;
    settings.Reflections.MaximumRoughness=0.5f; settings.Reflections.Thickness=0.5f;
    settings.GlobalIllumination.Lumen.CardSide=8;
    settings.GlobalIllumination.Lumen.CaptureCardsPerUpdate=128;
    settings.GlobalIllumination.Lumen.LightingTilesPerUpdate=256;
    settings.GlobalIllumination.Lumen.Radiosity.Enabled=false;
    settings.PathTracing.MaximumDistance=6000;
    auto& fog=settings.VolumetricFog;
    fog.Enabled=true; fog.MediumMode=VolumeMediumMode::Clouds; fog.MaximumDistance=6000;
    fog.SliceCount=64; fog.SamplesPerSlice=2; fog.BoundedTilePixels=8; fog.Radiance={0.30f,0.39f,0.50f};
    fog.Clouds.Medium.Absorption={0,0,0}; fog.Clouds.Medium.Scattering={1,1,1};
    fog.Clouds.Medium.Emission={0,0,0}; fog.Clouds.Medium.LightSamples=6;
    settings.VolumetricLighting.Enabled=true; settings.VolumetricLighting.Anisotropy=0.4f;
    settings.LightShafts.LightIndex=0; settings.LightShafts.SourceRadius=0.035f;
    settings.LightShafts.SampleCount=64;
    settings=SanitizeSettings(settings);
    (void)scene;
}

void UpdateBigWorld(SceneSnapshot& scene,Techniques::Settings& settings,bool force)
{
    if (scene.Demo!=DemoId::BigWorld || !scene.World) return;
    const double elapsed=std::max(0.0,scene.SampleTimeSeconds-scene.World->LastUpdate);
    const bool worldEdited=force || scene.World->SettingsChanged;
    const bool viewChanged=worldEdited || scene.CameraRevision!=scene.World->CameraRevision;
    const bool lodChanged=viewChanged || settings.Lod.Enabled!=scene.World->VegetationLod;
    const bool lightEdited=scene.LightingRevision!=scene.World->LightingRevision;
    if (!lodChanged && !lightEdited && elapsed<BigWorldUpdateInterval) return;
    auto state=std::make_shared<BigWorldState>(*scene.World); auto& s=state->Settings;
    state->SettingsChanged=false;
    state->CameraRevision=scene.CameraRevision;
    state->VegetationLod=settings.Lod.Enabled;
    state->LastUpdate=scene.SampleTimeSeconds;
    if (s.DayNight) s.Hour=std::fmod(s.Hour+float(elapsed*24/s.DayDuration),24.0f);
    if (s.AnimateWater) state->WaterTime+=elapsed*s.WaterSpeed;
    if (s.AnimateVegetation) state->VegetationTime+=elapsed;
    const bool wavesMoving=s.AnimateWater && s.WaterSpeed>0 && s.Water && elapsed>0;
    if (wavesMoving)
    {
        auto waves=std::make_shared<WaterSpectrum>(*state->Waves);
        waves->Evaluate(state->WaterTime); state->Waves=std::move(waves);
    }
    if (viewChanged) UpdateWaterLods(scene,*state);
    if (lodChanged) UpdateVegetation(scene,*state,settings.Lod.Enabled);
    const auto sun=SunDirection(s.Hour); const float day=Smooth((sun.Y+0.05f)*8);
    const float storm=s.Weather==WorldWeather::Clear?0.0f:s.Precipitation*0.75f;
    auto atmosphere=s.Atmosphere;
    atmosphere.SunDirection={sun.X,sun.Y,sun.Z};
    atmosphere.SunRadiance.fill(s.SunIntensity*4*(1-storm));
    Color sunlight{1,Mix(0.47f,0.94f,day),Mix(0.22f,0.84f,day),1};
    if (atmosphere.Enabled && state->Atmosphere)
    {
        const auto eye=scene.SceneCamera.Position;
        const auto transmission=state->Atmosphere->SunTransmittance({eye.X,eye.Y,eye.Z},{sun.X,sun.Y,sun.Z});
        sunlight={transmission.X*(1-storm),transmission.Y*(1-storm),transmission.Z*(1-storm),1};
    }
    const std::vector<Light> celestial={{LightType::Directional,{}, {-sun.X,-sun.Y,-sun.Z},sunlight,
        s.SunIntensity*day,{},true},{LightType::Directional,{}, {sun.X,sun.Y,sun.Z},{0.52f,0.66f,1,1},s.MoonIntensity*(1-day),{},false}};
    if (scene.Lights.size()<celestial.size()) scene.Lights.resize(celestial.size());
    for (std::size_t i=0;i<celestial.size();++i) if (!scene.Lights[i].EditorOverride) scene.Lights[i]=celestial[i];
    const auto eye=scene.SceneCamera.Position;
    const auto previousSky=scene.Environment;
    const bool skyMoved=std::abs(eye.Y-state->SkyPosition.Y)>4 ||
        std::hypot(eye.X-state->SkyPosition.X,eye.Z-state->SkyPosition.Z)>64;
    if (state->SkyJob && state->SkyJob->Result.wait_for(std::chrono::seconds(0))==std::future_status::ready)
    {
        scene.Environment=state->SkyJob->Result.get();
        state->SkyHour=state->SkyJob->Hour; state->SkyPosition=state->SkyJob->Position; state->SkyJob.reset();
    }
    if (state->SkyHour<0 || force)
    {
        scene.Environment=BuildSky(eye,s,state->Atmosphere,atmosphere);
        state->SkyHour=s.Hour; state->SkyPosition=eye; state->SkyJob.reset();
    }
    else if (!state->SkyJob && (std::abs(state->SkyHour-s.Hour)>0.035f || skyMoved))
    {
        auto job=std::make_shared<BigWorldSkyJob>(); job->Hour=s.Hour; job->Position=eye;
        job->Result=std::async(std::launch::async,[eye,s,model=state->Atmosphere,atmosphere]
        { return BuildSky(eye,s,model,atmosphere); });
        state->SkyJob=std::move(job);
    }
    auto& fog=settings.VolumetricFog;
    const bool geometryChanged=viewChanged || wavesMoving || (s.Water && (lightEdited || (s.DayNight && elapsed>0))) ||
        (state->CameraUnderwater && previousSky!=scene.Environment) || s.Weather!=WorldWeather::Clear;
    if (geometryChanged) fog.Underwater=DynamicGeometry(scene,*state);
    fog.Enabled=s.Clouds || s.Fog!=WorldFog::None || fog.Underwater.Extinction>0;
    fog.MediumMode=Techniques::VolumeMediumMode::Clouds;
    fog.AtmosphereMode=unsigned(s.Fog);
    fog.MaximumDistance=s.FogDistance; fog.Extinction=s.Fog==WorldFog::None?0:s.FogDensity;
    fog.BaseHeight=s.FogHeight; fog.HeightFalloff=s.Fog==WorldFog::Height?s.FogFalloff:0;
    fog.Radiance={0.35f*day+0.005f,0.43f*day+0.008f,0.55f*day+0.018f};
    fog.Clouds.Animate=s.CloudSpeed>0; fog.Clouds.AnimationSpeed=s.CloudSpeed;
    fog.Clouds.Medium.Density=s.Clouds?s.CloudDensity:0;
    fog.Clouds.Medium.BoundsMinimum={-s.Extent*0.6f,s.CloudAltitude,-s.Extent*0.6f};
    fog.Clouds.Medium.BoundsMaximum={s.Extent*0.6f,s.CloudAltitude+s.CloudThickness,s.Extent*0.6f};
    scene.Volume.reset();
    settings.Reflections.DiffuseIBL=s.IBLIntensity;
    settings.Reflections.Atmosphere=atmosphere;
    settings.Reflections.Planar.BoundedReceiver=true;
    const float receiverHeight=std::max(8.0f,s.WaveAmplitude*3+s.SwellAmplitude*3);
    settings.Reflections.Planar.ReceiverMinimum={-LakeRadius-35,LakeElevation-receiverHeight,-LakeRadius-35};
    settings.Reflections.Planar.ReceiverMaximum={LakeRadius+35,LakeElevation+receiverHeight,LakeRadius+35};
    settings.Reflections.Planar.MinimumNormalCosine=0.5f;
    settings.Reflections.Planar.Plane={0,1,0,-LakeElevation};
    settings.Reflections.Planar.ReceiverTolerance=std::max(0.1f,s.WaveAmplitude*3+s.SwellAmplitude*3);
    if (geometryChanged || lodChanged) ++scene.DynamicRevision;
    state->LightingRevision=++scene.LightingRevision;
    scene.World=state; ++scene.TransformRevision;
}
}
