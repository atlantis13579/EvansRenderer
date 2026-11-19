#pragma once

#include "Renderer/Types.h"
#include <vector>

namespace Evans::Renderer
{
struct AtmosphereTransport
{
    Float3 Radiance{}, Transmittance{1,1,1};
    bool Ground=false;
};

// Spherical single scattering, following Coding Adventure: Atmosphere and
// GPU Gems 2, chapter 16. A height/zenith optical-depth table amortizes the
// nested solar integral when generating the sky and its IBL environment.
// https://www.youtube.com/watch?v=DxfEbulyFcY
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-16-accurate-atmospheric-scattering
class AtmosphereModel
{
public:
    explicit AtmosphereModel(const Techniques::AtmosphereSettings& settings);
    [[nodiscard]] AtmosphereTransport Sky(Float3 eye,Float3 direction,Float3 sun,Float3 irradiance) const;
    [[nodiscard]] Float3 SunTransmittance(Float3 point,Float3 sun) const;
private:
    [[nodiscard]] Float2 SolarDepth(double radius,double cosine) const;
    Techniques::AtmosphereSettings Settings_;
    Float3 Rayleigh_;
    std::vector<Float2> OpticalDepth_;
};
}
