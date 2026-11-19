#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MaximumHairStrands=65536,MaximumHairSegments=16;
    struct HairFurSettings
    {
        bool Enabled=false,UseSelectedObject=true,Animate=false;
        float WindStrength=0.3f;
        std::uint32_t StrandCount=8192,Segments=4,Seed=1;
        // Length is normal-axis growth relative to the selected scene extent;
        // RadiusRatio is root radius / growth length. Curves may bend farther.
        float LengthFraction=0.025f,RadiusRatio=0.03f,Bend=0.3f,LengthVariation=0.25f,RootOffset=0.01f;
        float LongitudinalRoughness=0.3f,AzimuthalRoughness=0.3f,CuticleTiltDegrees=2,IndexOfRefraction=1.55f;
        std::array<float,3> RootTint{1,1,1},TipTint{1,1,1};
        float Intensity=1;
    };
    inline HairFurSettings SanitizeHairFurSettings(HairFurSettings value) noexcept
    {
        const HairFurSettings defaults;
        const auto finite=[](float x,float fallback){return std::isfinite(x)?x:fallback;};
        value.StrandCount=std::min(value.StrandCount,MaximumHairStrands); value.Segments=std::clamp(value.Segments,1u,MaximumHairSegments);
        value.LengthFraction=std::clamp(finite(value.LengthFraction,defaults.LengthFraction),0.0f,0.25f);
        value.RadiusRatio=std::clamp(finite(value.RadiusRatio,defaults.RadiusRatio),0.0f,0.25f);
        value.Bend=std::clamp(finite(value.Bend,defaults.Bend),0.0f,2.0f);
        value.LengthVariation=std::clamp(finite(value.LengthVariation,defaults.LengthVariation),0.0f,0.95f);
        value.RootOffset=std::clamp(finite(value.RootOffset,defaults.RootOffset),0.0f,0.25f);
        constexpr float MinimumRoughness=0.05f,MaximumTilt=15,MinimumIor=1.01f,MaximumIor=2.5f;
        value.LongitudinalRoughness=std::clamp(finite(value.LongitudinalRoughness,defaults.LongitudinalRoughness),MinimumRoughness,1.0f);
        value.AzimuthalRoughness=std::clamp(finite(value.AzimuthalRoughness,defaults.AzimuthalRoughness),MinimumRoughness,1.0f);
        value.CuticleTiltDegrees=std::clamp(finite(value.CuticleTiltDegrees,defaults.CuticleTiltDegrees),-MaximumTilt,MaximumTilt);
        value.IndexOfRefraction=std::clamp(finite(value.IndexOfRefraction,defaults.IndexOfRefraction),MinimumIor,MaximumIor);
        value.WindStrength=std::clamp(finite(value.WindStrength,defaults.WindStrength),0.0f,1.0f);
        value.Intensity=std::clamp(finite(value.Intensity,defaults.Intensity),0.0f,4.0f);
        for (std::size_t channel=0;channel<3;++channel)
        {
            value.RootTint[channel]=std::clamp(finite(value.RootTint[channel],defaults.RootTint[channel]),0.0f,1.0f);
            value.TipTint[channel]=std::clamp(finite(value.TipTint[channel],defaults.TipTint[channel]),0.0f,1.0f);
        }
        return value;
    }
    inline bool operator==(const HairFurSettings& a,const HairFurSettings& b) noexcept
    {
        return a.Animate==b.Animate && a.WindStrength==b.WindStrength && a.Enabled==b.Enabled && a.UseSelectedObject==b.UseSelectedObject && a.StrandCount==b.StrandCount &&
            a.Segments==b.Segments && a.Seed==b.Seed && a.LengthFraction==b.LengthFraction && a.RadiusRatio==b.RadiusRatio &&
            a.Bend==b.Bend && a.LengthVariation==b.LengthVariation && a.RootOffset==b.RootOffset &&
            a.LongitudinalRoughness==b.LongitudinalRoughness && a.AzimuthalRoughness==b.AzimuthalRoughness &&
            a.CuticleTiltDegrees==b.CuticleTiltDegrees && a.IndexOfRefraction==b.IndexOfRefraction &&
            a.RootTint==b.RootTint && a.TipTint==b.TipTint && a.Intensity==b.Intensity;
    }
    inline bool UsesHairFur(const HairFurSettings& request) noexcept
    {
        const auto value=SanitizeHairFurSettings(request);
        return value.Enabled && value.StrandCount && value.LengthFraction>0 && value.RadiusRatio>0 && value.Intensity>0;
    }
}
