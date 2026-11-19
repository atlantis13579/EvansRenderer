#pragma once

#include "Renderer/Shadows/CascadedShadowContract.h"
#include "Renderer/VirtualShadowMaps/VirtualShadowMaps.h"

namespace Evans::Renderer
{
    struct VirtualShadowSamplingFrame
    {
        VirtualShadowAtlasHandle Atlas;
        Float2 OutputPixelScale{1,1}; // input-pixel spacing per final-output pixel
        float MipBias=0;
    };
    inline bool ValidateVirtualShadowSamplingFrame(const VirtualShadowSamplingFrame& frame,std::string& error)
    {
        constexpr float MaximumMipBias=4;
        if (!frame.Atlas) return true;
        if (!std::isfinite(frame.MipBias) || frame.MipBias<-MaximumMipBias || frame.MipBias>MaximumMipBias ||
            !std::isfinite(frame.OutputPixelScale.X) || !std::isfinite(frame.OutputPixelScale.Y) ||
            frame.OutputPixelScale.X<=0 || frame.OutputPixelScale.Y<=0)
        { error="Virtual shadow sampling requires bounded mip bias and positive output-pixel footprint"; return false; }
        return true;
    }
    inline CascadedShadowConstants BuildVirtualShadowSamplingConstants(const VirtualShadowSnapshot& snapshot,
        const VirtualShadowSamplingFrame& frame,const Techniques::ShadowFilterSettings& filter)
    {
        // CSM and virtual maps are exclusive representations. Reuse the same
        // receiver block and two existing depth bindings; no extra ES texture
        // units or API-specific packing is required. Filtering.Z tags this ABI.
        CascadedShadowConstants result{}; const auto& layout=snapshot.Layout;
        result.LightViewProjection[0]=snapshot.Identity.LightViewProjection;
        result.FarDistances={float(layout.Resolution),float(layout.PageSide),float(layout.MipCount),float(std::min(VirtualShadowDirectoryWidth,layout.PageCount))};
        result.BlendStarts={float(layout.AtlasWidth),float(layout.AtlasHeight),float(layout.AtlasColumns),float(layout.PhysicalPages+1)};
        constexpr float HomogeneousEpsilon=1.0e-7f;
        result.EyeDepthPlane={frame.MipBias,frame.OutputPixelScale.X,frame.OutputPixelScale.Y,HomogeneousEpsilon};
        result.Parameters={0,1,1,float(layout.Resolution)}; // one active hierarchical representation
        const auto radius=filter.Mode==Techniques::ShadowFilterMode::Hard?0U:filter.Radius,width=radius*2+1;
        result.Filtering={float(radius),float(width*width),1,0};
        return result;
    }
}
