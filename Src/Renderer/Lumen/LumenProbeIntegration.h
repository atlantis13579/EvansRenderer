#pragma once

#include "Renderer/Lumen/LumenRadianceTrace.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenProbeCoefficients=9,LumenProbeThreads=64,LumenProbeColumns=1024;
    enum class LumenProbeStatus:unsigned { Complete=1,Partial=2,Invalid=3 };
    struct LumenProbeIntegrationUpload { Float3 SkyRadiance; bool Readback=false; LumenEnvironmentLobe Environment; };
    struct alignas(16) LumenProbeIrradiance
    {
        std::array<Float4,LumenProbeCoefficients> Coefficients; // RGB cosine-convolved SH, W=0; signed coefficients.
        Float4 Coverage; // Resolved ray fraction, average material coverage, miss fraction, mean surface distance.
        ShaderUInt4 TraceCounts; // Surface, miss, unresolved, world-cache reconstruction.
        ShaderUInt4 CacheCounts; // Complete, partial, unavailable, invalid surface-cache queries.
        ShaderUInt4 Metadata; // Stable probe ID, status, rays per probe, sequence.
    };
    struct alignas(16) LumenProbeIntegrationFrame
    {
        ShaderUInt4 Counts; // Probes, rays per probe, sequence, dispatch row stride.
        Float4 SkySolidAngle; // Explicit constant sky RGB; each ray's spherical measure.
        LumenEnvironmentFrame Environment;
    };
    static_assert(sizeof(LumenProbeIrradiance)==208 && sizeof(LumenProbeIntegrationFrame)==64,"Lumen probe irradiance ABI changed");
    struct LumenProbeInfo { LumenRadianceInfo Source; Float3 SkyRadiance; LumenEnvironmentLobe Environment; };
    struct LumenProbeHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(LumenProbeHandle other) const noexcept { return Value==other.Value; }
    };
    struct LumenProbeIntegrationResult { LumenProbeHandle Handle; LumenProbeInfo Info; std::vector<LumenProbeIrradiance> Probes; };

    // Nine real SH coefficients approximate low-frequency diffuse irradiance.
    // Missing/unknown transport stays zero without renormalizing resolved rays;
    // only geometric misses receive the explicitly supplied sky. This source
    // is not yet a screen-probe gather or an occlusion-aware spatial interpolator.
    bool BuildLumenProbeIntegrationFrame(const LumenRadianceInfo& source,const LumenProbeIntegrationUpload& upload,
        LumenProbeIntegrationFrame& output,std::string& error);
    bool ValidateLumenProbeIrradiance(const LumenProbeInfo& info,const std::vector<LumenProbeIrradiance>& probes,std::string& error);
    LumenProbeHandle NewLumenProbeHandle() noexcept;
}
