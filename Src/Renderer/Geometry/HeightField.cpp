#include "Renderer/Geometry/HeightField.h"

#include <cmath>

namespace Evans::Renderer
{
    HeightFieldSample EvaluatePeriodicHeight(Float2 coordinate,Float2 dx,Float2 dy) noexcept
    {
        constexpr double Pi=3.14159265358979323846,TwoPi=2*Pi;
        // cos(u)cos(v) is the sum of the (1,1) and (1,-1) Fourier modes.
        // exp(-2*pi^2*sigma^2) is their Gaussian low-pass response. Keeping the
        // modes separate handles anisotropic footprints instead of a scalar mip.
        const double sum=TwoPi*std::remainder(double(coordinate.X)+coordinate.Y,1.0);
        const double difference=TwoPi*std::remainder(double(coordinate.X)-coordinate.Y,1.0);
        const double sumDx=double(dx.X)+dx.Y,sumDy=double(dy.X)+dy.Y;
        const double differenceDx=double(dx.X)-dx.Y,differenceDy=double(dy.X)-dy.Y;
        const double sumWeight=std::exp(-2*Pi*Pi*(sumDx*sumDx+sumDy*sumDy));
        const double differenceWeight=std::exp(-2*Pi*Pi*(differenceDx*differenceDx+differenceDy*differenceDy));
        constexpr double ModeWeight=0.25;
        const double sumSlope=-ModeWeight*TwoPi*sumWeight*std::sin(sum);
        const double differenceSlope=-ModeWeight*TwoPi*differenceWeight*std::sin(difference);
        return {float(0.5+ModeWeight*(sumWeight*std::cos(sum)+differenceWeight*std::cos(difference))),
            {float(sumSlope+differenceSlope),float(sumSlope-differenceSlope)}};
    }
}
