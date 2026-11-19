#pragma once

#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Incoming radiance along direction w is baseSky + peak * max(dot(axis,w),0)^p.
    // The peak is radiance, not integrated flux; narrowing the lobe reduces its
    // total energy. All transport consumers use the same world-space convention.
    // This procedural environment needs no new texture assets.
    struct LumenEnvironmentLobe
    {
        Float3 Radiance{},Direction{0,1,0};
        float Exponent=1;
        bool operator==(const LumenEnvironmentLobe& b) const noexcept
        { return Radiance.X==b.Radiance.X && Radiance.Y==b.Radiance.Y && Radiance.Z==b.Radiance.Z &&
            Direction.X==b.Direction.X && Direction.Y==b.Direction.Y && Direction.Z==b.Direction.Z && Exponent==b.Exponent; }
    };
    struct alignas(16) LumenEnvironmentFrame
    {
        Float4 AxisPower{0,1,0,1};
        Float4 Radiance{};
    };
    static_assert(sizeof(LumenEnvironmentFrame)==32,"Lumen environment ABI changed");
    LumenEnvironmentLobe MakeLumenEnvironmentLobe(const Techniques::LumenSettings& settings) noexcept;
    bool BuildLumenEnvironmentFrame(const LumenEnvironmentLobe&,LumenEnvironmentFrame&,std::string& error);
}
