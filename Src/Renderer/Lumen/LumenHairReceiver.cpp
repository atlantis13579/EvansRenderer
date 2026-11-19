#include "Renderer/Lumen/LumenHairReceiver.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    bool BuildLumenHairReceiverFrame(const LumenTranslucencyFrame& volume,
        const Techniques::HairFurSettings& request,const LightingState& lighting,Float4 clipPlane,
        LumenHairReceiverFrame& output,std::string& error)
    {
        const auto finite=[](Float4 v) { return std::isfinite(v.X)&&std::isfinite(v.Y)&&std::isfinite(v.Z)&&std::isfinite(v.W); };
        if (!volume.Grid.X || !volume.Grid.Y || !volume.Grid.Z || volume.Grid.X>LumenTranslucencyMaximumProbes ||
            volume.Grid.Y>LumenTranslucencyMaximumProbes || volume.Grid.Z>LumenTranslucencyMaximumProbes || !(volume.Grid.W&LumenTranslucencyActive) ||
            (volume.Grid.W&~(LumenTranslucencyActive|LumenTranslucencyVisibilitySupport)) ||
            std::uint64_t(volume.Grid.X)*volume.Grid.Y*volume.Grid.Z>LumenTranslucencyMaximumProbes ||
            !finite(volume.Distribution) || !finite(volume.Atlas) || volume.Distribution.X<=0 || volume.Distribution.Y<=0 ||
            volume.Distribution.Z<0 || volume.Distribution.W<=0 || volume.Atlas.X!=float(volume.Grid.X*LumenProbeCoefficients) ||
            volume.Atlas.Y!=float(volume.Grid.Y*volume.Grid.Z) || volume.Atlas.Z<=0 || volume.Atlas.W<=0 ||
            !finite(clipPlane) || !finite({lighting.AmbientColor.X,lighting.AmbientColor.Y,lighting.AmbientColor.Z,lighting.AmbientStrength}))
        { error="Lumen hair receivers require a valid bounded SH volume, clipping plane and ambient source"; return false; }
        for (const auto& matrix:{volume.ViewProjection,volume.Inverse}) for (float v:matrix.Values) if (!std::isfinite(v))
        { error="Lumen hair receiver camera must be finite"; return false; }
        const auto hair=Techniques::SanitizeHairFurSettings(request);
        const float strength=std::max(lighting.AmbientStrength,0.0f);
        LumenHairReceiverFrame result; result.Volume=volume; result.ClipPlane=clipPlane;
        result.RootTint={hair.RootTint[0],hair.RootTint[1],hair.RootTint[2],hair.Intensity};
        result.TipTint={hair.TipTint[0],hair.TipTint[1],hair.TipTint[2],0};
        result.Ambient={std::max(lighting.AmbientColor.X,0.0f)*strength,
            std::max(lighting.AmbientColor.Y,0.0f)*strength,std::max(lighting.AmbientColor.Z,0.0f)*strength,0};
        if (!finite(result.Ambient)) { error="Lumen hair ambient replacement exceeds finite radiance"; return false; }
        output=result; error.clear(); return true;
    }
}
