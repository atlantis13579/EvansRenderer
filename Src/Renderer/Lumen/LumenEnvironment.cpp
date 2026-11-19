#include "Renderer/Lumen/LumenEnvironment.h"

#include <cmath>

namespace Evans::Renderer
{
    LumenEnvironmentLobe MakeLumenEnvironmentLobe(const Techniques::LumenSettings& settings) noexcept
    {
        const auto& source=settings.Environment;
        if (!source.Enabled || (source.Radiance[0]==0 && source.Radiance[1]==0 && source.Radiance[2]==0)) return {};
        return {{source.Radiance[0],source.Radiance[1],source.Radiance[2]},
            {source.Direction[0],source.Direction[1],source.Direction[2]},source.Exponent};
    }
    bool BuildLumenEnvironmentFrame(const LumenEnvironmentLobe& source,LumenEnvironmentFrame& output,std::string& error)
    {
        constexpr float MaximumRadiance=65504; // Binary16 HDR target range.
        const auto channel=[](float value) { return std::isfinite(value) && value>=0 && value<=MaximumRadiance; };
        const auto& axis=source.Direction;
        const double length=std::sqrt(double(axis.X)*axis.X+double(axis.Y)*axis.Y+double(axis.Z)*axis.Z);
        if (!channel(source.Radiance.X) || !channel(source.Radiance.Y) || !channel(source.Radiance.Z) ||
            !std::isfinite(length) || length<=0 || !std::isfinite(source.Exponent) || source.Exponent<1 ||
            source.Exponent>Techniques::MaximumLumenEnvironmentExponent)
        { error="Lumen environment needs bounded nonnegative radiance, a nonzero finite axis and a bounded cosine exponent"; return false; }
        LumenEnvironmentFrame frame;
        frame.AxisPower={float(axis.X/length),float(axis.Y/length),float(axis.Z/length),source.Exponent};
        frame.Radiance={source.Radiance.X,source.Radiance.Y,source.Radiance.Z,0};
        output=frame; error.clear(); return true;
    }
}
