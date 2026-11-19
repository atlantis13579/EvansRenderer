#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    // Global photographic exposure with temporal eye/camera adaptation. Values
    // are relative scene-linear scales, not calibrated lux or camera EV100.
    struct TemporalEffectsSettings
    {
        bool Enabled=false;
        float KeyValue=0.18f;
        float MinimumExposureEV=-8.0f,MaximumExposureEV=8.0f;
        float BrightenHalfLife=0.5f,DarkenHalfLife=0.25f; // seconds
    };
    inline bool operator==(const TemporalEffectsSettings& a,const TemporalEffectsSettings& b) noexcept
    {
        return a.Enabled==b.Enabled && a.KeyValue==b.KeyValue && a.MinimumExposureEV==b.MinimumExposureEV &&
            a.MaximumExposureEV==b.MaximumExposureEV && a.BrightenHalfLife==b.BrightenHalfLife && a.DarkenHalfLife==b.DarkenHalfLife;
    }
    inline constexpr float MinimumExposureKey=0.01f,MaximumExposureKey=1.0f;
    inline constexpr float MaximumAutoExposureEV=12.0f;
    inline constexpr float MinimumAdaptationHalfLife=0.01f,MaximumAdaptationHalfLife=10.0f;
    inline TemporalEffectsSettings SanitizeTemporalEffectsSettings(TemporalEffectsSettings settings) noexcept
    {
        const TemporalEffectsSettings defaults;
        const auto finite=[](float value,float fallback){return std::isfinite(value)?value:fallback;};
        settings.KeyValue=std::clamp(finite(settings.KeyValue,defaults.KeyValue),MinimumExposureKey,MaximumExposureKey);
        settings.MinimumExposureEV=std::clamp(finite(settings.MinimumExposureEV,defaults.MinimumExposureEV),-MaximumAutoExposureEV,MaximumAutoExposureEV);
        settings.MaximumExposureEV=std::clamp(finite(settings.MaximumExposureEV,defaults.MaximumExposureEV),settings.MinimumExposureEV,MaximumAutoExposureEV);
        settings.BrightenHalfLife=std::clamp(finite(settings.BrightenHalfLife,defaults.BrightenHalfLife),MinimumAdaptationHalfLife,MaximumAdaptationHalfLife);
        settings.DarkenHalfLife=std::clamp(finite(settings.DarkenHalfLife,defaults.DarkenHalfLife),MinimumAdaptationHalfLife,MaximumAdaptationHalfLife);
        return settings;
    }
    enum class TemporalEffectsPassKind : std::uint32_t { Meter=0,Reduce=1,Adapt=2,Composite=3 };
}
