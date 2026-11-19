#pragma once

#include "Techniques/TechniqueCatalog.h"

#include <cstddef>
#include <string>

namespace Evans::Renderer
{
    inline constexpr std::size_t ShadingTechniqueCount = 5;
    inline constexpr std::size_t AreaLightTechniqueIndex = ShadingTechniqueCount;
    inline constexpr std::size_t StandaloneShadingTechniqueCount = ShadingTechniqueCount + 1;
    inline std::size_t ShadingTechniqueIndex(Techniques::ShadingModel model, bool areaLights) noexcept
    { return model == Techniques::ShadingModel::PBR && areaLights ? AreaLightTechniqueIndex : static_cast<std::size_t>(model); }
    inline constexpr std::size_t PostProcessTechniqueCount =
        static_cast<std::size_t>(Techniques::PostProcessPassKind::Count);
    inline constexpr std::size_t HdrTechniqueCount = 4;
    std::size_t HdrShaderIndex(Techniques::HdrPassKind pass) noexcept;

    const char* ShadingTechniqueFile(Techniques::ShadingModel model, bool areaLights = false) noexcept;
    const char* PostProcessTechniqueFile(Techniques::PostProcessPassKind pass) noexcept;
    const char* HdrShaderFilename(Techniques::HdrPassKind pass) noexcept;
    bool LoadTechniqueShaderSource(
        const char* api,
        const char* techniqueFile,
        std::string& source,
        std::string& error);
}
