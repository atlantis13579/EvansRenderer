#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    enum class PathMaterial : std::uint32_t { GGX, Lambertian };
    inline constexpr std::uint32_t MaximumPathDepth=12,MaximumPathsPerFrame=16,MaximumAccumulatedPaths=65536;
    inline constexpr float MinimumPathDistance=0.001f,MaximumPathDistance=1000000.0f,MaximumPathBias=1000.0f;
    struct PathTracingSettings
    {
        PathMaterial Material=PathMaterial::GGX;
        std::uint32_t SamplesPerFrame=1,MaximumDepth=6,AccumulationLimit=512,Seed=0;
        float MaximumDistance=1000.0f,NormalBias=0.001f;
        bool Accumulate=true;
    };
    inline PathTracingSettings SanitizePathTracingSettings(PathTracingSettings settings) noexcept
    {
        const PathTracingSettings defaults;
        if (settings.Material!=PathMaterial::GGX && settings.Material!=PathMaterial::Lambertian) settings.Material=defaults.Material;
        settings.SamplesPerFrame=std::clamp(settings.SamplesPerFrame,1u,MaximumPathsPerFrame);
        settings.MaximumDepth=std::clamp(settings.MaximumDepth,1u,MaximumPathDepth);
        settings.AccumulationLimit=std::clamp(settings.AccumulationLimit,1u,MaximumAccumulatedPaths);
        settings.MaximumDistance=std::clamp(std::isfinite(settings.MaximumDistance) ? settings.MaximumDistance : defaults.MaximumDistance,MinimumPathDistance,MaximumPathDistance);
        settings.NormalBias=std::clamp(std::isfinite(settings.NormalBias) ? settings.NormalBias : defaults.NormalBias,0.0f,std::min(MaximumPathBias,settings.MaximumDistance));
        return settings;
    }
    inline bool operator==(const PathTracingSettings& a,const PathTracingSettings& b) noexcept
    {
        return a.Material==b.Material && a.SamplesPerFrame==b.SamplesPerFrame && a.MaximumDepth==b.MaximumDepth &&
            a.AccumulationLimit==b.AccumulationLimit && a.Seed==b.Seed && a.MaximumDistance==b.MaximumDistance &&
            a.NormalBias==b.NormalBias && a.Accumulate==b.Accumulate;
    }
    // A larger target or a different batch size continues the same estimator.
    // Its stochastic sequence is indexed by the total committed sample count.
    inline bool SamePathTransport(const PathTracingSettings& a,const PathTracingSettings& b) noexcept
    {
        return a.Material==b.Material && a.MaximumDepth==b.MaximumDepth && a.Seed==b.Seed &&
            a.MaximumDistance==b.MaximumDistance && a.NormalBias==b.NormalBias && a.Accumulate==b.Accumulate;
    }
}
