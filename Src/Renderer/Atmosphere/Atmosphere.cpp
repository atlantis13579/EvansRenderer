#include "Renderer/Atmosphere/Atmosphere.h"
#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
namespace
{
constexpr double Pi=3.14159265358979323846;
constexpr unsigned HeightCount=64,AngleCount=128;
struct Vector { double X,Y,Z; };
Vector Add(Vector a,Vector b,double t) { return {a.X+b.X*t,a.Y+b.Y*t,a.Z+b.Z*t}; }
double Dot(Vector a,Vector b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
double Length(Vector p) { return std::sqrt(Dot(p,p)); }
Vector Unit(Float3 p)
{
    const double inverse=1/std::sqrt(std::max(1e-12,double(p.X)*p.X+double(p.Y)*p.Y+double(p.Z)*p.Z));
    return {p.X*inverse,p.Y*inverse,p.Z*inverse};
}
double Horizon(double radius,double planet)
{ return -std::sqrt(std::max(0.0,1-planet*planet/(radius*radius))); }
double ExitDistance(double radius,double cosine,double outer)
{ const double b=radius*cosine; return std::max(0.0,-b+std::sqrt(std::max(0.0,b*b+outer*outer-radius*radius))); }
Float2 Density(double height,const Techniques::AtmosphereSettings& s)
{
    const double taper=std::clamp(1-height/(s.Height*0.001),0.0,1.0);
    height=std::max(0.0,height);
    return {float(std::exp(-height/(s.RayleighHeight*0.001))*taper),
        float(std::exp(-height/(s.MieHeight*0.001))*taper)};
}
Float3 Transmission(Float2 depth,Float3 beta,float mie)
{
    return {std::exp(-std::min(80.0f,depth.X*beta.X+depth.Y*mie)),
        std::exp(-std::min(80.0f,depth.X*beta.Y+depth.Y*mie)),
        std::exp(-std::min(80.0f,depth.X*beta.Z+depth.Y*mie))};
}
}
AtmosphereModel::AtmosphereModel(const Techniques::AtmosphereSettings& settings)
    : Settings_(Techniques::SanitizeAtmosphereSettings(settings))
{
    const auto& s=Settings_;
    const auto coefficient=[&](unsigned i) { return 0.0135f*s.RayleighStrength*std::pow(550.0f/s.Wavelengths[i],4.0f); };
    Rayleigh_={coefficient(0),coefficient(1),coefficient(2)};
    OpticalDepth_.resize(HeightCount*AngleCount);
    const double planet=s.PlanetRadius*0.001,outer=planet+s.Height*0.001;
    const unsigned samples=std::max(32u,s.LightSamples*4);
    for (unsigned y=0;y<HeightCount;++y) for (unsigned x=0;x<AngleCount;++x)
    {
        const double h=double(y)/(HeightCount-1),radius=planet+h*h*s.Height*0.001;
        const double horizon=Horizon(radius,planet),mu=horizon+(1-horizon)*x/(AngleCount-1);
        const double distance=ExitDistance(radius,mu,outer);
        Float2 depth{};
        for (unsigned i=0;i<samples;++i)
        {
            const double begin=distance*double(i)*i/(double(samples)*samples);
            const double end=distance*double(i+1)*(i+1)/(double(samples)*samples),t=(begin+end)*0.5;
            const auto density=Density(std::sqrt(radius*radius+t*t+2*radius*mu*t)-planet,s);
            depth.X+=density.X*float(end-begin); depth.Y+=density.Y*float(end-begin);
        }
        OpticalDepth_[y*AngleCount+x]=depth;
    }
}
Float2 AtmosphereModel::SolarDepth(double radius,double cosine) const
{
    const double planet=Settings_.PlanetRadius*0.001,horizon=Horizon(std::max(planet,radius),planet);
    if (cosine<horizon) return {1e6f,1e6f};
    const double x=std::clamp((cosine-horizon)/(1-horizon),0.0,1.0)*(AngleCount-1);
    const double y=std::sqrt(std::clamp((radius-planet)/(Settings_.Height*0.001),0.0,1.0))*(HeightCount-1);
    const unsigned ix=std::min(unsigned(x),AngleCount-2),iy=std::min(unsigned(y),HeightCount-2);
    const float u=float(x-ix),v=float(y-iy);
    const auto a=OpticalDepth_[iy*AngleCount+ix],b=OpticalDepth_[iy*AngleCount+ix+1];
    const auto c=OpticalDepth_[(iy+1)*AngleCount+ix],d=OpticalDepth_[(iy+1)*AngleCount+ix+1];
    return {(a.X*(1-u)+b.X*u)*(1-v)+(c.X*(1-u)+d.X*u)*v,
        (a.Y*(1-u)+b.Y*u)*(1-v)+(c.Y*(1-u)+d.Y*u)*v};
}
Float3 AtmosphereModel::SunTransmittance(Float3 point,Float3 sun) const
{
    const double planet=Settings_.PlanetRadius*0.001;
    Vector p{point.X*0.001,(point.Y-Settings_.GroundHeight)*0.001+planet,point.Z*0.001};
    const auto light=Unit(sun);
    double radius=std::max(planet,Length(p));
    const double outer=planet+Settings_.Height*0.001;
    if (radius>outer)
    {
        const double b=Dot(p,light),discriminant=b*b+outer*outer-radius*radius;
        if (discriminant<=0 || b>=0) return {1,1,1};
        p=Add(p,light,-b-std::sqrt(discriminant)); radius=outer;
    }
    const double cosine=Dot(p,light)/radius;
    if (cosine<Horizon(radius,planet)) return {};
    return Transmission(SolarDepth(radius,cosine),Rayleigh_,0.0044f*Settings_.MieStrength);
}
AtmosphereTransport AtmosphereModel::Sky(Float3 eye,Float3 direction,Float3 sun,Float3 irradiance) const
{
    const auto& s=Settings_;
    const double planet=s.PlanetRadius*0.001,outer=planet+s.Height*0.001;
    Vector p{eye.X*0.001,(eye.Y-s.GroundHeight)*0.001+planet,eye.Z*0.001};
    const auto ray=Unit(direction),light=Unit(sun);
    double radius=Length(p);
    if (radius<planet+0.0001) { p=Add({},p,(planet+0.0001)/radius); radius=Length(p); }
    const double b=Dot(p,ray),disc=b*b+outer*outer-radius*radius;
    AtmosphereTransport result;
    if (disc<=0) return result;
    const double entry=std::max(0.0,-b-std::sqrt(disc));
    double end=-b+std::sqrt(disc);
    if (end<=entry) return result;
    const double groundDisc=b*b+planet*planet-radius*radius;
    if (b<0 && groundDisc>=0) { end=std::min(end,-b-std::sqrt(groundDisc)); result.Ground=true; }
    const double cosine=Dot(ray,light),g=s.MieAnisotropy;
    const float phaseR=float(3*(1+cosine*cosine)/(16*Pi));
    const float phaseM=float((1-g*g)/(4*Pi*std::pow(std::max(0.001,1+g*g-2*g*cosine),1.5)));
    const float mie=0.004f*s.MieStrength;
    Float2 viewDepth{};
    const unsigned samples=s.ViewSamples*2;
    for (unsigned i=0;i<samples;++i)
    {
        const double begin=entry+(end-entry)*double(i)*i/(double(samples)*samples);
        const double finish=entry+(end-entry)*double(i+1)*(i+1)/(double(samples)*samples);
        const float step=float(finish-begin);
        const auto point=Add(p,ray,(begin+finish)*0.5);
        const double r=Length(point),mu=Dot(point,light)/r;
        const auto density=Density(r-planet,s),solar=SolarDepth(r,mu);
        const Float2 midpoint{viewDepth.X+density.X*step*0.5f,viewDepth.Y+density.Y*step*0.5f};
        if (mu>=Horizon(r,planet))
        {
            const auto t=Transmission({midpoint.X+solar.X,midpoint.Y+solar.Y},Rayleigh_,mie*1.1f);
            result.Radiance.X+=t.X*(Rayleigh_.X*density.X*phaseR+mie*density.Y*phaseM)*irradiance.X*s.Intensity*step;
            result.Radiance.Y+=t.Y*(Rayleigh_.Y*density.X*phaseR+mie*density.Y*phaseM)*irradiance.Y*s.Intensity*step;
            result.Radiance.Z+=t.Z*(Rayleigh_.Z*density.X*phaseR+mie*density.Y*phaseM)*irradiance.Z*s.Intensity*step;
        }
        viewDepth.X+=density.X*step; viewDepth.Y+=density.Y*step;
    }
    result.Transmittance=Transmission(viewDepth,Rayleigh_,mie*1.1f);
    return result;
}
}
