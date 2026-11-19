#include "Renderer/Hair/HairScattering.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    namespace
    {
        constexpr double Pi=3.14159265358979323846;
    }
    FiberScatteringProfile BuildFiberScatteringProfile(const Techniques::HairFurSettings& request) noexcept
    {
        const auto settings=Techniques::SanitizeHairFurSettings(request); const double m=settings.LongitudinalRoughness,n=settings.AzimuthalRoughness;
        // Published perceptual fits. Keeping them in the profile builder avoids
        // repeating their evaluation per light or confusing them with presets.
        constexpr double LongLinear=0.726,LongQuadratic=0.812,LongHigh=3.7;
        constexpr double AziLinear=0.265,AziQuadratic=1.194,AziHigh=5.372;
        constexpr double TransmissionVariance=0.25,InternalVariance=4;
        const double rootVariance=LongLinear*m+LongQuadratic*m*m+LongHigh*std::pow(m,20),variance=rootVariance*rootVariance;
        const double scale=std::sqrt(Pi/8)*(AziLinear*n+AziQuadratic*n*n+AziHigh*std::pow(n,22));
        constexpr double AbsorptionFit[]{5.969,-0.215,2.532,-10.73,5.574,0.245};
        double absorption=AbsorptionFit[std::size(AbsorptionFit)-1];
        for (std::size_t term=std::size(AbsorptionFit)-1;term>0;--term) absorption=absorption*n+AbsorptionFit[term-1];
        constexpr double DegreesPerHalfCircle=180; const double tilt=settings.CuticleTiltDegrees*Pi/DegreesPerHalfCircle;
        FiberScatteringProfile result;
        result.Variance={float(variance),float(variance*TransmissionVariance),float(variance*InternalVariance),float(variance*InternalVariance)};
        result.SinTilt={float(std::sin(tilt)),float(std::sin(tilt*2)),float(std::sin(tilt*4)),0};
        result.CosTilt={float(std::cos(tilt)),float(std::cos(tilt*2)),float(std::cos(tilt*4)),0};
        result.Optics={settings.IndexOfRefraction,float(scale),float(absorption),settings.Intensity}; return result;
    }
}
