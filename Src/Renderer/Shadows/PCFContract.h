#pragma once
#include "Renderer/Types.h"
#include "Techniques/TechniqueCatalog.h"

namespace Evans::Renderer
{
    // Percentage-closer filtering averages binary visibility, not stored depth.
    // A radius r contributes (2r+1)^2 bilinear comparison taps. Surface strength
    // remains a material policy; the filter itself returns raw [0,1] visibility.
    struct alignas(16) PCFConstants { Float4 Kernel; };
    static_assert(sizeof(PCFConstants)==16,"PCF constant ABI changed");
    inline PCFConstants BuildPCFConstants(Techniques::ShadowFilterSettings settings,bool enabled) noexcept
    {
        settings=Techniques::SanitizeShadowFilterSettings(settings);
        const auto radius=settings.Mode==Techniques::ShadowFilterMode::PCF?settings.Radius:0u;
        const auto width=2u*radius+1u;
        return {{float(radius),1.0f/float(width*width),enabled?1.0f:0.0f,0.0f}};
    }
}
