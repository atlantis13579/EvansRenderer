#include "Renderer/Geometry/ParallaxOcclusionMapping.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float3 p) { return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z); }
        constexpr double DirectionEpsilon=1.0e-12;
    }
    bool BuildParallaxConstants(const Techniques::ParallaxOcclusionSettings& requested,
        Float3 minimum,Float3 maximum,ParallaxConstants& output) noexcept
    {
        if (!Finite(minimum) || !Finite(maximum) || minimum.X>maximum.X || minimum.Y>maximum.Y || minimum.Z>maximum.Z) return false;
        const double extent=std::max({double(maximum.X)-minimum.X,double(maximum.Y)-minimum.Y,double(maximum.Z)-minimum.Z});
        if (extent<=DirectionEpsilon || !std::isfinite(extent)) return false;
        const auto settings=Techniques::SanitizeParallaxOcclusionSettings(requested);
        ParallaxConstants constants{{Techniques::UsesParallaxOcclusion(settings)?settings.DepthFraction:0.0f,settings.Tiling,settings.NormalStrength,settings.GrazingCosine},
            {settings.MinimumLayers,settings.MaximumLayers,settings.RefinementSteps,static_cast<unsigned>(settings.Coordinates)},
            {minimum.X,minimum.Y,minimum.Z,float(1/extent)}};
        if (!std::isfinite(constants.ObjectBounds.W)) return false;
        output=constants; return true;
    }
}
