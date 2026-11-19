#include "Editor/BigWorld.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace Evans::Editor
{
namespace
{
constexpr float Pi=3.14159265358979323846f, Gravity=9.81f;
using Complex=std::complex<float>;

void InverseFFT(std::vector<Complex>& values)
{
    const auto n=values.size();
    for (std::size_t i=1,j=0;i<n;++i)
    {
        auto bit=n>>1;
        for (;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if (i<j) std::swap(values[i],values[j]);
    }
    for (std::size_t width=2;width<=n;width*=2)
    {
        const Complex step=std::polar(1.0f,2*Pi/float(width));
        for (std::size_t start=0;start<n;start+=width)
        {
            Complex phase{1,0};
            for (std::size_t j=0;j<width/2;++j)
            {
                const auto even=values[start+j],odd=values[start+j+width/2]*phase;
                values[start+j]=even+odd; values[start+j+width/2]=even-odd;
                phase*=step;
            }
        }
    }
    for (auto& value:values) value/=float(n);
}
Float3 Unit(Float3 n)
{
    const float scale=1/std::sqrt(std::max(1e-12f,n.X*n.X+n.Y*n.Y+n.Z*n.Z));
    return {n.X*scale,n.Y*scale,n.Z*scale};
}
}

WaterSpectrum::WaterSpectrum(const BigWorldSettings& settings):Settings_(SanitizeBigWorldSettings(settings))
{
    const auto n=Settings_.FFTSize;
    Initial_.resize(n*n);
    for (auto& field:Field_) field.resize(n*n);
    for (auto& axis:Derivatives_) for (auto& field:axis) field.resize(n*n);
    std::mt19937 random(Settings_.Seed);
    std::normal_distribution<float> gaussian(0,1);
    const float angle=Settings_.WindDirection*Pi/180,wx=std::cos(angle),wz=std::sin(angle);
    const float largest=Settings_.WindSpeed*Settings_.WindSpeed/Gravity;
    const float frequency=2*Pi/Settings_.FFTPatch;
    for (unsigned z=0;z<n;++z) for (unsigned x=0;x<n;++x)
    {
        const float kx=float(x<=n/2?int(x):int(x)-int(n))*frequency;
        const float kz=float(z<=n/2?int(z):int(z)-int(n))*frequency;
        const float k2=kx*kx+kz*kz;
        if (k2==0 || largest<0.001f) continue;
        const float alignment=(kx*wx+kz*wz)/std::sqrt(k2);
        const float spectrum=std::exp(-1/(k2*largest*largest))*alignment*alignment*
            std::exp(-k2*0.04f)/(k2*k2);
        // Multiplying by N^2 compensates the normalized inverse transform;
        // delta-k preserves physical amplitude when the resolution changes.
        Initial_[z*n+x]=Complex(gaussian(random),gaussian(random))*
            std::sqrt(spectrum*0.5f)*frequency*0.035f*float(n*n);
    }
}

void WaterSpectrum::Evaluate(double seconds)
{
    Time_=seconds;
    for (unsigned i=0;i<Swells_.size();++i)
    {
        const float angle=(Settings_.WindDirection+float(i)*43-52)*Pi/180;
        auto& wave=Swells_[i]; wave.X=std::cos(angle); wave.Z=std::sin(angle);
        wave.K=2*Pi/(Settings_.SwellLength*(1+0.37f*i));
        wave.Amplitude=Settings_.SwellAmplitude/(1+float(i)*2);
        wave.Horizontal=std::min(wave.Amplitude,Settings_.SwellSteepness/(4*wave.K));
        wave.Phase=-float(std::remainder(std::sqrt(Gravity*wave.K)*seconds,2*Pi))+i*1.73f;
    }
    // Bridge the shortest well-resolved FFT waves and the material ripples.
    // Their curvature also refracts sunlight into narrow bands on the bed.
    for (unsigned i=0;i<WindWaves_.size();++i)
    {
        const float angle=(Settings_.WindDirection+(i?31.0f:-17.0f))*Pi/180;
        auto& wave=WindWaves_[i]; wave.X=std::cos(angle); wave.Z=std::sin(angle);
        wave.K=2*Pi/(i?5.3f:8.7f);
        wave.Amplitude=(i?0.11f:0.16f)*Settings_.WaveAmplitude*std::min(Settings_.WindSpeed/5,1.5f);
        wave.Phase=-float(std::remainder(std::sqrt(Gravity*wave.K)*seconds,2*Pi))+i*2.41f;
    }
    const unsigned n=Settings_.FFTSize;
    std::array<std::vector<Complex>,3> frequency;
    for (auto& field:frequency) field.resize(n*n);
    if (!Settings_.FFT)
    {
        for (auto& field:Field_) std::fill(field.begin(),field.end(),0.0f);
        for (auto& axis:Derivatives_) for (auto& field:axis) std::fill(field.begin(),field.end(),0.0f);
        return;
    }
    for (unsigned z=0;z<n;++z) for (unsigned x=0;x<n;++x)
    {
        const float kx=float(x<=n/2?int(x):int(x)-int(n))*2*Pi/Settings_.FFTPatch;
        const float kz=float(z<=n/2?int(z):int(z)-int(n))*2*Pi/Settings_.FFTPatch;
        const float k=std::sqrt(kx*kx+kz*kz);
        if (k<1e-6f) continue;
        const float omega=std::sqrt(Gravity*k*std::tanh(k*20));
        const Complex phase=std::polar(1.0f,float(std::remainder(omega*seconds,2*Pi)));
        const Complex h=Initial_[z*n+x]*phase+std::conj(Initial_[((n-z)%n)*n+(n-x)%n])*std::conj(phase);
        frequency[1][z*n+x]=h*Settings_.WaveAmplitude;
        // The Nyquist derivative is zero to preserve Hermitian symmetry.
        frequency[0][z*n+x]=x==n/2?Complex{}:Complex(0,-kx/k)*h*Settings_.Choppiness*Settings_.WaveAmplitude;
        frequency[2][z*n+x]=z==n/2?Complex{}:Complex(0,-kz/k)*h*Settings_.Choppiness*Settings_.WaveAmplitude;
    }
    std::vector<Complex> line(n);
    for (unsigned axis=0;axis<3;++axis)
    {
        auto& values=frequency[axis];
        for (unsigned z=0;z<n;++z)
        {
            std::copy_n(values.begin()+z*n,n,line.begin()); InverseFFT(line);
            std::copy(line.begin(),line.end(),values.begin()+z*n);
        }
        for (unsigned x=0;x<n;++x)
        {
            for (unsigned z=0;z<n;++z) line[z]=values[z*n+x];
            InverseFFT(line);
            for (unsigned z=0;z<n;++z) Field_[axis][z*n+x]=line[z].real();
        }
        const float inverseStep=float(n)/(2*Settings_.FFTPatch);
        for (unsigned z=0;z<n;++z) for (unsigned x=0;x<n;++x)
        {
            const auto& field=Field_[axis];
            Derivatives_[0][axis][z*n+x]=(field[z*n+((x+1)&(n-1))]-field[z*n+((x-1)&(n-1))])*inverseStep;
            Derivatives_[1][axis][z*n+x]=(field[((z+1)&(n-1))*n+x]-field[((z-1)&(n-1))*n+x])*inverseStep;
        }
    }
}

WaterSample WaterSpectrum::Sample(float x,float z) const
{
    const unsigned n=Settings_.FFTSize;
    const float u=x/Settings_.FFTPatch*n,v=z/Settings_.FFTPatch*n;
    const int ix=int(std::floor(u)),iz=int(std::floor(v));
    const float fx=u-std::floor(u),fz=v-std::floor(v);
    const unsigned x0=unsigned(ix)&(n-1),x1=unsigned(ix+1)&(n-1);
    const unsigned z0=unsigned(iz)&(n-1),z1=unsigned(iz+1)&(n-1);
    const auto sample=[&](const std::vector<float>& f)
    {
        const float a=f[z0*n+x0]*(1-fx)+f[z0*n+x1]*fx;
        const float b=f[z1*n+x0]*(1-fx)+f[z1*n+x1]*fx;
        return a*(1-fz)+b*fz;
    };
    Float3 displacement{sample(Field_[0]),sample(Field_[1]),sample(Field_[2])};
    Float3 tangent{1+sample(Derivatives_[0][0]),sample(Derivatives_[0][1]),sample(Derivatives_[0][2])};
    Float3 binormal{sample(Derivatives_[1][0]),sample(Derivatives_[1][1]),1+sample(Derivatives_[1][2])};
    // The FFT gradient is sampled once. Gerstner derivatives are analytic;
    // neither normals nor caustic rays resample five displaced wave surfaces.
    if (Settings_.Gerstner) for (const auto& wave:Swells_)
    {
        const float phase=wave.K*(wave.X*x+wave.Z*z)+wave.Phase;
        const float sine=std::sin(phase),cosine=std::cos(phase);
        displacement.X+=wave.X*wave.Horizontal*cosine;
        displacement.Y+=wave.Amplitude*sine;
        displacement.Z+=wave.Z*wave.Horizontal*cosine;
        const float horizontal=-wave.Horizontal*wave.K*sine,vertical=wave.Amplitude*wave.K*cosine;
        tangent.X+=horizontal*wave.X*wave.X; tangent.Y+=vertical*wave.X; tangent.Z+=horizontal*wave.X*wave.Z;
        binormal.X+=horizontal*wave.X*wave.Z; binormal.Y+=vertical*wave.Z; binormal.Z+=horizontal*wave.Z*wave.Z;
    }
    if (Settings_.FFT) for (const auto& wave:WindWaves_)
    {
        const float phase=wave.K*(wave.X*x+wave.Z*z)+wave.Phase;
        displacement.Y+=wave.Amplitude*std::sin(phase);
        const float slope=wave.Amplitude*wave.K*std::cos(phase);
        tangent.Y+=slope*wave.X; binormal.Y+=slope*wave.Z;
    }
    return {displacement,Unit({binormal.Y*tangent.Z-binormal.Z*tangent.Y,
        binormal.Z*tangent.X-binormal.X*tangent.Z,binormal.X*tangent.Y-binormal.Y*tangent.X})};
}

std::array<std::array<float,4>,2> WaterSpectrum::LightWaves() const
{
    const auto pack=[](const Swell& wave,bool active)
    {
        return std::array<float,4>{wave.X*wave.K,wave.Z*wave.K,wave.Phase,
            active?std::min(1.0f,9*wave.K*wave.Amplitude):0.0f};
    };
    const bool wind=Settings_.FFT && WindWaves_[0].Amplitude>0;
    const bool swell=Settings_.Gerstner && Swells_[0].Amplitude>0;
    return {pack(wind?WindWaves_[0]:Swells_[1],wind||swell),
        pack(swell?Swells_[0]:WindWaves_[1],wind||swell)};
}

float WaterSpectrum::Caustic(float x,float z,float depth,Float3 sun) const
{
    if (!Settings_.Caustics || Settings_.CausticIntensity<=0 || depth<=0 || depth>=Settings_.CausticDepth ||
        sun.Y<=0.02f) return 0;
    const float windSlope=Settings_.FFT?std::min(1.3f,Settings_.WindSpeed*Settings_.WaveAmplitude/3):0;
    const auto projected=[&](float px,float pz)
    {
        const auto sample=Sample(px,pz);
        auto normal=sample.Normal;
        if (windSlope>0)
        {
            // Include the short wind-wave slopes that the material resolves
            // between mesh vertices. Crossing trains form folds in the ray map.
            const auto& axis=WindWaves_[0];
            const float along=px*axis.X+pz*axis.Z,across=pz*axis.X-px*axis.Z;
            const float warp=1.1f*std::sin(along*0.24f+across*0.37f)+0.7f*std::sin(across*0.58f-along*0.31f);
            const float a=0.21f*std::cos(along*1.60f+across*0.36f+axis.Phase*1.5f+warp);
            const float b=0.15f*std::cos(across*1.26f-along*0.78f+WindWaves_[1].Phase-0.8f*warp);
            normal=Unit({normal.X-windSlope*(axis.X*a-axis.Z*b),normal.Y,
                normal.Z-windSlope*(axis.Z*a+axis.X*b)});
        }
        const float eta=1/Settings_.WaterIOR,cosine=sun.X*normal.X+sun.Y*normal.Y+sun.Z*normal.Z;
        const float refracted=std::sqrt(std::max(0.0f,1-eta*eta*(1-cosine*cosine)));
        const Float3 ray{-eta*sun.X+(eta*cosine-refracted)*normal.X,
            -eta*sun.Y+(eta*cosine-refracted)*normal.Y,-eta*sun.Z+(eta*cosine-refracted)*normal.Z};
        const float length=(depth+sample.Displacement.Y)/std::max(0.05f,-ray.Y);
        return Float2{px+sample.Displacement.X+ray.X*length,pz+sample.Displacement.Z+ray.Z*length};
    };
    // Start at the refracting surface point for a flat interface. This keeps
    // the ripple pattern anchored to the bed at oblique sun angles.
    const float eta=1/Settings_.WaterIOR;
    const float vertical=std::sqrt(1-eta*eta*(1-sun.Y*sun.Y));
    const float targetX=x,targetZ=z;
    x+=depth*eta*sun.X/vertical; z+=depth*eta*sun.Z/vertical;
    for (unsigned iteration=0;iteration<2;++iteration)
    {
        const auto hit=projected(x,z);
        x+=std::clamp(targetX-hit.X,-depth,depth)*0.6f;
        z+=std::clamp(targetZ-hit.Y,-depth,depth)*0.6f;
    }
    constexpr float e=0.5f;
    const auto a=projected(x-e,z),b=projected(x+e,z),c=projected(x,z-e),d=projected(x,z+e);
    const float jacobian=std::abs((b.X-a.X)*(d.Y-c.Y)-(b.Y-a.Y)*(d.X-c.X))/(4*e*e);
    const float reach=std::clamp((Settings_.CausticDepth-depth)/(0.25f*Settings_.CausticDepth),0.0f,1.0f);
    // Converging rays brighten the bed and diverging rays leave darker gaps.
    // Absorption along the return path is applied when composing the surface.
    // A finite solar footprint softens the peak at a fold.
    const float contrast=(1-jacobian)/(0.30f+jacobian);
    const float incident=std::exp(-Settings_.Absorption*0.5f*depth/vertical)*sun.Y;
    return std::max(-0.85f,contrast*reach*incident*Settings_.CausticIntensity);
}
}
