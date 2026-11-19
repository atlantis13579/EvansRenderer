#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Techniques/TemporalEffects.h"

namespace Evans::Renderer
{
    struct TemporalEffectsDesc
    {
        double SampleTimeSeconds=0;
        std::uint64_t Epoch=0,ResetSerial=0;
    };
    struct TemporalEffectsConstants
    {
        ShaderUInt4 PhaseExtentHistory; // phase, source width/height, valid history
        Float4 KeyLimitsDelta; // target key, minimum/maximum EV, elapsed seconds
        Float4 HalfLivesLimits; // brighten/darken seconds, luminance floor, FP16 ceiling
    };
    static_assert(sizeof(TemporalEffectsConstants)==48 && offsetof(TemporalEffectsConstants,HalfLivesLimits)==32,
        "Temporal exposure constants require three aligned vectors");
    inline constexpr float MinimumMeterLuminance=1.0e-6f;
    inline constexpr float ExposureHalfFloatCeiling=65504.0f;
    inline constexpr double MaximumExposureHistoryGap=2.0;

    struct ExposureExtent { int Width=0,Height=0; };
    inline std::vector<ExposureExtent> BuildExposureReductionPlan(int width,int height)
    {
        std::vector<ExposureExtent> result;
        if (width<=0 || height<=0) return result;
        do
        {
            width=width/2+width%2; height=height/2+height%2;
            result.push_back({width,height});
        } while (width>1 || height>1);
        return result;
    }

    // Goodnight et al., "Interactive Time-Dependent Tone Mapping Using
    // Programmable Graphics Hardware", EGSR 2003. Global log-average metering
    // follows Reinhard et al. (2002); adaptation here is exponential in EV with
    // asymmetric half-lives. A sum/count pyramid retains every odd-edge pixel.
    // https://diglib.eg.org/server/api/core/bitstreams/5e337aa0-b9a6-4e79-afff-8a83bf92034f/content
    // Native history stores (adapted EV, target EV, mean log luminance, 1) in
    // RGBA32F. Two private slots separate successful presentation from pending
    // recording; later HDR operations never overwrite the accepted exposure.
    class TemporalExposureHistory
    {
    public:
        bool Resolved=false;
        void Begin() noexcept { Resolved=false; }
        void Invalidate() noexcept { Valid=false; Resolved=false; }
        unsigned ReadIndex() const noexcept { return Read; }
        bool Prepare(const TemporalEffectsDesc& desc,const Techniques::TemporalEffectsSettings& requested,
            int width,int height,TemporalEffectsConstants& constants) noexcept
        {
            if (width<=0 || height<=0 || !std::isfinite(desc.SampleTimeSeconds)) return false;
            const auto settings=Techniques::SanitizeTemporalEffectsSettings(requested);
            const double interval=desc.SampleTimeSeconds-Last.SampleTimeSeconds;
            const bool current=Valid && Width==width && Height==height && desc.Epoch==Last.Epoch &&
                desc.ResetSerial==Last.ResetSerial && interval>=0 && interval<=MaximumExposureHistoryGap;
            constants={{0,static_cast<std::uint32_t>(width),static_cast<std::uint32_t>(height),current?1u:0u},
                {settings.KeyValue,settings.MinimumExposureEV,settings.MaximumExposureEV,current?static_cast<float>(interval):0.0f},
                {settings.BrightenHalfLife,settings.DarkenHalfLife,MinimumMeterLuminance,ExposureHalfFloatCeiling}};
            Pending=desc; PendingWidth=width; PendingHeight=height; return true;
        }
        void Commit() noexcept
        {
            if (!Resolved) { Valid=false; return; }
            Read=1-Read; Last=Pending; Width=PendingWidth; Height=PendingHeight; Valid=true;
        }
    private:
        TemporalEffectsDesc Last,Pending;
        int Width=0,Height=0,PendingWidth=0,PendingHeight=0;
        unsigned Read=0;
        bool Valid=false;
    };
}
