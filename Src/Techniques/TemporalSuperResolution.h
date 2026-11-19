#pragma once

#include <algorithm>
#include <cmath>

namespace Evans::Techniques
{
    // Independent temporal upscaling, not an implementation of a vendor SDK.
    struct TemporalSuperResolutionSettings
    {
        float RenderScale=2.0f/3.0f;
        float MaximumHistoryWeight=0.95f,DepthTolerance=0.01f;
        float VarianceGamma=1.25f,ReactiveScale=1,MaximumSamples=32;
    };
    inline bool operator==(const TemporalSuperResolutionSettings& a,const TemporalSuperResolutionSettings& b) noexcept
    { return a.RenderScale==b.RenderScale && a.MaximumHistoryWeight==b.MaximumHistoryWeight && a.DepthTolerance==b.DepthTolerance && a.VarianceGamma==b.VarianceGamma && a.ReactiveScale==b.ReactiveScale && a.MaximumSamples==b.MaximumSamples; }
    inline TemporalSuperResolutionSettings SanitizeTemporalSuperResolutionSettings(TemporalSuperResolutionSettings value) noexcept
    {
        const TemporalSuperResolutionSettings defaults;
        const auto clamp=[](float x,float fallback,float low,float high){return std::isfinite(x)?std::clamp(x,low,high):fallback;};
        value.RenderScale=clamp(value.RenderScale,defaults.RenderScale,0.5f,1);
        value.MaximumHistoryWeight=clamp(value.MaximumHistoryWeight,defaults.MaximumHistoryWeight,0,0.99f);
        value.DepthTolerance=clamp(value.DepthTolerance,defaults.DepthTolerance,0,0.1f);
        value.VarianceGamma=clamp(value.VarianceGamma,defaults.VarianceGamma,0.5f,3);
        value.ReactiveScale=clamp(value.ReactiveScale,defaults.ReactiveScale,0,4);
        value.MaximumSamples=clamp(value.MaximumSamples,defaults.MaximumSamples,1,64);
        return value;
    }
}
