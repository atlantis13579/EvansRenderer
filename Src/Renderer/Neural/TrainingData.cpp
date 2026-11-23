#include "Renderer/Neural/TrainingJob.h"
#include "Editor/Resources.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace Evans::Neural
{
namespace
{
struct Vertex { Vec3 Position,Normal;float U,V; };
struct Triangle { Vertex Vertices[3];const Editor::Material* Material=nullptr; };
float Edge(float ax,float ay,float bx,float by,float px,float py) { return (px-ax)*(by-ay)-(py-ay)*(bx-ax); }
Vec3 Position(const Editor::Float3& value) { return {value.X,value.Y,value.Z}; }
std::array<float,3> Shade(const Triangle& triangle,Vec3 normal,float u,float v)
{
    const auto* material=triangle.Material;std::array<float,3> color{0.75f,0.65f,0.45f};
    if(material) color={material->Diffuse.R,material->Diffuse.G,material->Diffuse.B};
    if(material && material->DiffuseImage && material->DiffuseImage->IsValid()) {
        const auto& texture=*material->DiffuseImage;
        const int tx=int((u-std::floor(u))*texture.Width)%texture.Width,ty=int((1-(v-std::floor(v)))*texture.Height)%texture.Height;
        const size_t index=(size_t(ty)*texture.Width+tx)*4;
        for(int c=0;c<3;++c) color[c]*=texture.PixelsRGBA8[index+c]/255.f;
    }
    const float light=0.25f+0.7f*std::max(0.f,Dot(Normalize(normal),Normalize({0.35f,1,0.25f})));
    for(auto& channel:color) channel=std::clamp(channel*light,0.f,1.f);return color;
}
FloatImage Capture(const std::vector<Triangle>& triangles,View view)
{
    // Supersampled mesh images provide independent training targets.
    const int width=view.Width,height=view.Height;view.Width*=2;view.Height*=2;view.Focal*=2;
    FloatImage high{view.Width,view.Height,std::vector<float>(size_t(view.Width)*view.Height*3)};
    std::vector<float> depth(size_t(view.Width)*view.Height,std::numeric_limits<float>::infinity());
    for(size_t i=0;i<depth.size();++i) for(int c=0;c<3;++c) high.RGB[i*3+c]=Background[c];
    for(const auto& triangle:triangles) {
        float sx[3],sy[3],sz[3];bool valid=true;
        for(int i=0;i<3;++i) { const auto p=triangle.Vertices[i].Position-view.Eye;sz[i]=Dot(p,view.Forward);if(sz[i]<=0.05f) valid=false;
            sx[i]=view.Focal*Dot(p,view.Right)/sz[i]+view.Width*0.5f;sy[i]=view.Focal*Dot(p,view.Down)/sz[i]+view.Height*0.5f; }
        if(!valid) continue;const float area=Edge(sx[0],sy[0],sx[1],sy[1],sx[2],sy[2]);if(std::abs(area)<1e-8f) continue;
        const int left=int(std::clamp(std::floor(std::min({sx[0],sx[1],sx[2]})),0.f,float(view.Width))),right=int(std::clamp(std::ceil(std::max({sx[0],sx[1],sx[2]})),0.f,float(view.Width)));
        const int top=int(std::clamp(std::floor(std::min({sy[0],sy[1],sy[2]})),0.f,float(view.Height))),bottom=int(std::clamp(std::ceil(std::max({sy[0],sy[1],sy[2]})),0.f,float(view.Height)));
        for(int y=top;y<bottom;++y) for(int x=left;x<right;++x) {
            float a=Edge(sx[1],sy[1],sx[2],sy[2],x+0.5f,y+0.5f)/area,b=Edge(sx[2],sy[2],sx[0],sy[0],x+0.5f,y+0.5f)/area,c=1-a-b;
            if(a<0 || b<0 || c<0) continue;a/=sz[0];b/=sz[1];c/=sz[2];const float z=1/(a+b+c);const size_t pixel=size_t(y)*view.Width+x;if(z>=depth[pixel]) continue;
            depth[pixel]=z;a*=z;b*=z;c*=z;const auto& va=triangle.Vertices[0];const auto& vb=triangle.Vertices[1];const auto& vc=triangle.Vertices[2];
            const auto color=Shade(triangle,va.Normal*a+vb.Normal*b+vc.Normal*c,va.U*a+vb.U*b+vc.U*c,va.V*a+vb.V*b+vc.V*c);
            for(int ch=0;ch<3;++ch) high.RGB[pixel*3+ch]=color[ch];
        }
    }
    FloatImage image{width,height,std::vector<float>(size_t(width)*height*3,0)};
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) for(int c=0;c<3;++c)
        image.RGB[(size_t(y)*width+x)*3+c]+=high.RGB[(size_t(y*2+dy)*view.Width+x*2+dx)*3+c]*0.25f;return image;
}
void Append(std::vector<float>& data,Vec3 value) { data.insert(data.end(),{value.X,value.Y,value.Z}); }
}
std::filesystem::path GenerateTeapotJob(const std::filesystem::path& contentRoot,const JobOptions& supplied)
{
    JobOptions options=supplied;options.DataDirectory=std::filesystem::absolute(options.DataDirectory);options.OutputDirectory=std::filesystem::absolute(options.OutputDirectory);
    if(options.Resolution<16 || options.Resolution>512 || options.Views<4 || options.Views>512 || options.Points<32 || options.Points>100000 || options.PixelBatch<1 || options.Steps<1)
        throw std::runtime_error("Invalid neural training dimensions");
    std::filesystem::create_directories(options.DataDirectory);std::filesystem::create_directories(options.OutputDirectory);
    auto model=Editor::FileResourceLoader::LoadObj(contentRoot/"Models/UtahTeapot/teapot.obj");
    Vec3 lo{1e20f,1e20f,1e20f},hi{-1e20f,-1e20f,-1e20f};
    for(const auto& mesh:model.Geometry.Meshes) for(const auto& vertex:mesh.Vertices) { const auto p=Position(vertex.Position);
        lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)};hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)}; }
    const auto center=(lo+hi)*0.5f;const auto extent=hi-lo;const float scale=3/std::max({extent.X,extent.Y,extent.Z});
    std::vector<Triangle> triangles;std::vector<double> areas;
    for(const auto& mesh:model.Geometry.Meshes) for(size_t i=0;i+2<mesh.Indices.size();i+=3) {
        Triangle triangle;
        for(const auto& section:mesh.Sections) if(i>=section.FirstIndex && i<section.FirstIndex+section.IndexCount) for(const auto& material:model.Materials) if(material.Name==section.MaterialName) triangle.Material=&material;
        for(int v=0;v<3;++v) { const auto& vertex=mesh.Vertices.at(mesh.Indices[i+v]);triangle.Vertices[v]={(Position(vertex.Position)-center)*scale,Position(vertex.Normal),vertex.TexCoord.X,vertex.TexCoord.Y}; }
        const auto normal=Cross(triangle.Vertices[1].Position-triangle.Vertices[0].Position,triangle.Vertices[2].Position-triangle.Vertices[0].Position);
        const float area=0.5f*std::sqrt(Dot(normal,normal));if(area<1e-10f) continue;
        for(auto& vertex:triangle.Vertices) if(Dot(vertex.Normal,vertex.Normal)<1e-10f) vertex.Normal=Normalize(normal);
        triangles.push_back(triangle);areas.push_back(area);
    }
    if(triangles.empty()) throw std::runtime_error("Teapot has no triangles");
    std::mt19937 random(42);std::discrete_distribution<size_t> select(areas.begin(),areas.end());std::uniform_real_distribution<float> unit(0,1);
    Array means{{options.Points,3},{}},offset{{options.Points,3},std::vector<float>(size_t(options.Points)*3,0)},scales{{options.Points,3},{}},rotation{{options.Points,4},{}},opacity{{options.Points,1},{}},color{{options.Points,3},{}};
    for(int i=0;i<options.Points;++i) {
        const auto& triangle=triangles[select(random)];const float root=std::sqrt(unit(random)),a=1-root,b=root*(1-unit(random)),c=1-a-b;
        const auto position=triangle.Vertices[0].Position*a+triangle.Vertices[1].Position*b+triangle.Vertices[2].Position*c;
        const auto normal=Normalize(triangle.Vertices[0].Normal*a+triangle.Vertices[1].Normal*b+triangle.Vertices[2].Normal*c);
        Append(means.Data,position);const float radius=0.055f*std::sqrt(2048.f/options.Points);
        scales.Data.insert(scales.Data.end(),{std::log(radius),std::log(radius),std::log(radius*0.25f)});
        float qw=1+normal.Z,qx=-normal.Y,qy=normal.X,qz=0;if(qw<1e-5f) { qw=0;qx=1;qy=0; }
        const float inv=1/std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);rotation.Data.insert(rotation.Data.end(),{qw*inv,qx*inv,qy*inv,qz*inv});
        opacity.Data.push_back(0);color.Data.insert(color.Data.end(),{0,0,0});
    }
    for(const auto& pair:std::vector<std::pair<std::string,const Array*>>{{"initial_means",&means},{"offset",&offset},{"log_scales",&scales},{"rotation",&rotation},{"opacity",&opacity},{"color",&color}}) SaveArray(options.DataDirectory/(pair.first+".npy"),*pair.second);
    Array images{{options.Views,options.Resolution*options.Resolution,3},{}},eyes{{options.Views,3},{}},basis{{options.Views,3,3},{}};
    for(int i=0;i<options.Views;++i) {
        const float angle=6.28318530718f*i/options.Views,elevation=0.18f+0.48f*float(i%3)/2;
        const auto view=OrbitView(angle,elevation,options.Resolution,options.Resolution);const auto image=Capture(triangles,view);
        images.Data.insert(images.Data.end(),image.RGB.begin(),image.RGB.end());Append(eyes.Data,view.Eye);
        basis.Data.insert(basis.Data.end(),{view.Right.X,view.Down.X,view.Forward.X,view.Right.Y,view.Down.Y,view.Forward.Y,view.Right.Z,view.Down.Z,view.Forward.Z});
    }
    SaveArray(options.DataDirectory/"images.npy",images);SaveArray(options.DataDirectory/"eyes.npy",eyes);SaveArray(options.DataDirectory/"basis.npy",basis);
    Array coords{{options.Resolution*options.Resolution,2},{}};for(int y=0;y<options.Resolution;++y) for(int x=0;x<options.Resolution;++x) coords.Data.insert(coords.Data.end(),{x+0.5f,y+0.5f});SaveArray(options.DataDirectory/"pixel_coordinates.npy",coords);
    WriteTrainingGraph(options);
    return options.DataDirectory/"training.json";
}
}
