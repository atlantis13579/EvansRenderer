#include "Renderer/Neural/GaussianScene.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Evans::Neural
{
float Dot(Vec3 a,Vec3 b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
Vec3 Cross(Vec3 a,Vec3 b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
Vec3 Normalize(Vec3 a) { return a*(1/std::sqrt(std::max(1e-20f,Dot(a,a)))); }
View MakeView(const Editor::Camera& camera,int width,int height)
{
    if(width<1 || height<1 || width>8192 || height>8192) throw std::runtime_error("Invalid neural viewport size");
    View view;view.Width=width;view.Height=height;
    view.Eye={camera.Position.X,camera.Position.Y,camera.Position.Z};
    view.Forward=Normalize(Vec3{camera.Target.X,camera.Target.Y,camera.Target.Z}-view.Eye);
    view.Right=Normalize(Cross(view.Forward,{camera.Up.X,camera.Up.Y,camera.Up.Z}));
    view.Down=Cross(view.Forward,view.Right);
    view.Focal=0.5f*height/std::tan(camera.VerticalFieldOfViewDegrees*0.00872664626f);
    return view;
}
View ModelView(View view,float yawDegrees)
{
    const float radians=yawDegrees*0.01745329252f,c=std::cos(radians),s=std::sin(radians);
    const auto inverse=[c,s](Vec3 v) { return Vec3{c*v.X-s*v.Z,v.Y,s*v.X+c*v.Z}; };
    view.Eye=inverse(view.Eye);view.Right=inverse(view.Right);
    view.Down=inverse(view.Down);view.Forward=inverse(view.Forward);
    return view;
}
View OrbitView(float angle,float elevation,int width,int height)
{
    Editor::Camera camera;camera.Position={4.5f*std::cos(angle)*std::cos(elevation),4.5f*std::sin(elevation),4.5f*std::sin(angle)*std::cos(elevation)};
    camera.Target={0,0,0};camera.VerticalFieldOfViewDegrees=45;
    return MakeView(camera,width,height);
}
void SaveArray(const std::filesystem::path& path,const Array& array)
{
    size_t count=1;std::ostringstream dims;
    for(int size:array.Shape) { if(size<0) throw std::runtime_error("Negative array dimension");count*=size;dims<<size<<", "; }
    if(count!=array.Data.size()) throw std::runtime_error("Array shape mismatch");
    std::string header="{'descr': '<f4', 'fortran_order': False, 'shape': ("+dims.str()+"), }";
    header.append((64-(10+header.size()+1)%64)%64,' ');header+='\n';
    std::filesystem::create_directories(path.parent_path());std::ofstream out(path,std::ios::binary);
    const char prefix[]={char(0x93),'N','U','M','P','Y',1,0,char(header.size()&255),char(header.size()>>8)};
    out.write(prefix,sizeof(prefix));out<<header;
    out.write(reinterpret_cast<const char*>(array.Data.data()),std::streamsize(array.Data.size()*sizeof(float)));
    if(!out) throw std::runtime_error("Cannot write tensor "+path.u8string());
}
Array LoadArray(const std::filesystem::path& path)
{
    std::ifstream in(path,std::ios::binary);unsigned char prefix[10]{};in.read(reinterpret_cast<char*>(prefix),10);
    if(!in || prefix[0]!=0x93 || std::string(reinterpret_cast<char*>(prefix)+1,5)!="NUMPY" || prefix[6]!=1) throw std::runtime_error("Expected NPY v1 tensor: "+path.u8string());
    std::string header(size_t(prefix[8])+(size_t(prefix[9])<<8),' ');in.read(header.data(),header.size());
    if(header.find("'<f4'")==std::string::npos || header.find("False")==std::string::npos) throw std::runtime_error("Expected row-major Float32 NPY");
    const auto start=header.find('('),end=header.find(')',start);
    if(start==std::string::npos || end==std::string::npos) throw std::runtime_error("Invalid NPY shape");
    std::string dims=header.substr(start+1,end-start-1);std::replace(dims.begin(),dims.end(),',',' ');
    std::istringstream sizes(dims);Array result;int dim;size_t count=1;
    while(sizes>>dim) { if(dim<0 || count>size_t(268435456)/std::max(1,dim)) throw std::runtime_error("NPY size exceeds limit");result.Shape.push_back(dim);count*=dim; }
    if(result.Shape.empty()) throw std::runtime_error("Expected non-scalar array");
    result.Data.resize(count);in.read(reinterpret_cast<char*>(result.Data.data()),std::streamsize(count*sizeof(float)));
    if(!in || in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Invalid NPY payload length");
    for(float value:result.Data) if(!std::isfinite(value)) throw std::runtime_error("Non-finite array value");
    return result;
}
namespace
{
float Sigmoid(float x) { return 1/(1+std::exp(-std::clamp(x,-30.f,30.f))); }
struct Projected { float X,Y,A,B,C,Depth,Distance,Opacity,Radius;std::array<float,3> Color; };
Projected Project(const Gaussian& point,const View& view,float supportCutoff)
{
    const auto& p=point.Values;const Vec3 relative=Vec3{p[0],p[1],p[2]}-view.Eye;
    const float x=Dot(relative,view.Right),y=Dot(relative,view.Down),z=Dot(relative,view.Forward);
    Projected projected{};projected.Depth=z;projected.Distance=Dot(relative,relative);if(z<=0.05f) return projected;
    float norm=0;for(int i=6;i<10;++i) norm+=p[i]*p[i];norm=1/std::sqrt(std::max(norm,1e-12f));
    const float w=p[6]*norm,a=p[7]*norm,b=p[8]*norm,c=p[9]*norm;
    const Vec3 columns[]={{1-2*(b*b+c*c),2*(a*b+w*c),2*(a*c-w*b)},
        {2*(a*b-w*c),1-2*(a*a+c*c),2*(b*c+w*a)}, {2*(a*c+w*b),2*(b*c-w*a),1-2*(a*a+b*b)}};
    float xx=0.3f,xy=0,yy=0.3f;
    for(int i=0;i<3;++i)
    {
        const float scale=std::exp(std::clamp(p[3+i],-9.f,1.f));const auto axis=columns[i];
        const float vx=view.Focal/z*(Dot(axis,view.Right)-x/z*Dot(axis,view.Forward))*scale;
        const float vy=view.Focal/z*(Dot(axis,view.Down)-y/z*Dot(axis,view.Forward))*scale;
        xx+=vx*vx;xy+=vx*vy;yy+=vy*vy;
    }
    const float determinant=std::max(1e-8f,xx*yy-xy*xy);
    projected.X=view.Focal*x/z+0.5f*view.Width;projected.Y=view.Focal*y/z+0.5f*view.Height;
    projected.A=yy/determinant;projected.B=-xy/determinant;projected.C=xx/determinant;
    projected.Radius=std::sqrt(supportCutoff*0.5f*(xx+yy+std::sqrt((xx-yy)*(xx-yy)+4*xy*xy)));
    projected.Opacity=Sigmoid(p[10]);for(int i=0;i<3;++i) projected.Color[i]=Sigmoid(p[11+i]);return projected;
}
}
GaussianScene LoadScene(const std::filesystem::path& path)
{
    if(path.extension()==".glb") return LoadGlb(path);
    if(path.extension()!=".npy") throw std::runtime_error("Expected Gaussian GLB or NPY asset");
    GaussianScene scene;
    const auto array=LoadArray(path);
    if(array.Shape.size()!=2 || array.Shape[1]!=14 || array.Shape[0]<1) throw std::runtime_error("Expected Gaussian [N,14] tensor");
    scene.Points.resize(array.Shape[0]);for(size_t i=0;i<scene.Points.size();++i) std::copy_n(array.Data.begin()+i*14,14,scene.Points[i].Values.begin());
    return scene;
}
FloatImage Render(const GaussianScene& scene,const View& view,const std::function<bool()>& cancelled)
{
    FloatImage image{view.Width,view.Height,std::vector<float>(size_t(view.Width)*view.Height*3,0)};
    std::vector<float> transmittance(size_t(view.Width)*view.Height,1);
    std::vector<Projected> points;points.reserve(scene.Points.size());for(const auto& point:scene.Points) { auto p=Project(point,view,scene.SupportCutoff);if(p.Depth>0.05f) points.push_back(p); }
    std::stable_sort(points.begin(),points.end(),[&scene](const auto& a,const auto& b){return scene.SortByCameraDistance?a.Distance<b.Distance:a.Depth<b.Depth;});
    for(const auto& p:points)
    {
        if(cancelled && cancelled()) throw std::runtime_error("Neural rendering cancelled");
        const int left=int(std::clamp(std::floor(p.X-p.Radius),0.f,float(view.Width))),right=int(std::clamp(std::ceil(p.X+p.Radius),0.f,float(view.Width)));
        const int top=int(std::clamp(std::floor(p.Y-p.Radius),0.f,float(view.Height))),bottom=int(std::clamp(std::ceil(p.Y+p.Radius),0.f,float(view.Height)));
        for(int y=top;y<bottom;++y) for(int x=left;x<right;++x) {
            const size_t pixel=size_t(y)*view.Width+x;if(transmittance[pixel]<1e-5f) continue;
            const float dx=x+0.5f-p.X,dy=y+0.5f-p.Y,q=p.A*dx*dx+2*p.B*dx*dy+p.C*dy*dy;if(q>scene.SupportCutoff) continue;
            const float alpha=std::min(0.99f,p.Opacity*std::exp(-0.5f*std::max(q,0.f))),weight=alpha*transmittance[pixel];
            for(int c=0;c<3;++c) image.RGB[pixel*3+c]+=weight*p.Color[c];transmittance[pixel]*=1-alpha;
        }
    }
    for(size_t i=0;i<transmittance.size();++i) for(int c=0;c<3;++c) image.RGB[i*3+c]+=transmittance[i]*Background[c];return image;
}
Editor::Image DisplayImage(const FloatImage& image)
{
    Editor::Image result;result.Width=image.Width;result.Height=image.Height;result.PixelsRGBA8.resize(size_t(image.Width)*image.Height*4,255);
    for(size_t i=0;i<image.RGB.size()/3;++i) for(int c=0;c<3;++c) result.PixelsRGBA8[i*4+c]=uint8_t(std::clamp(image.RGB[i*3+c]*255+0.5f,0.f,255.f));return result;
}
}
