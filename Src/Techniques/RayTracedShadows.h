#pragma once

#include <algorithm>
#include <cmath>

namespace Evans::Techniques
{
    inline constexpr float MinimumRayShadowDistance=0.001f,MaximumRayShadowDistance=1000000;
    inline constexpr float MaximumRayShadowBias=1000;
    struct RayTracedShadowSettings
    {
        float MaximumDistance=1000;
        float NormalBias=0.002f;
        float StartBias=0.0001f;
    };
    inline bool operator==(const RayTracedShadowSettings& a,const RayTracedShadowSettings& b) noexcept
    { return a.MaximumDistance==b.MaximumDistance && a.NormalBias==b.NormalBias && a.StartBias==b.StartBias; }
    inline RayTracedShadowSettings SanitizeRayTracedShadowSettings(RayTracedShadowSettings value) noexcept
    {
        const RayTracedShadowSettings defaults;
        const auto finiteClamp=[](float x,float fallback,float minimum,float maximum)
        { return std::clamp(std::isfinite(x)?x:fallback,minimum,maximum); };
        value.MaximumDistance=finiteClamp(value.MaximumDistance,defaults.MaximumDistance,MinimumRayShadowDistance,MaximumRayShadowDistance);
        const float biasLimit=std::min(MaximumRayShadowBias,value.MaximumDistance/4);
        value.NormalBias=finiteClamp(value.NormalBias,defaults.NormalBias,0,biasLimit);
        value.StartBias=finiteClamp(value.StartBias,defaults.StartBias,0,biasLimit);
        return value;
    }
}
