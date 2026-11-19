#pragma once

#include "Renderer/Reflections/EnvironmentCapture.h"
#include <filesystem>

namespace Evans::Renderer
{
    // Immutable linear HDR faces in the same native cube orientation as scene
    // captures. Uploads preserve radiance above one; exposure belongs to display.
    struct EnvironmentRadiance
    {
        std::uint32_t Resolution=0;
        std::array<std::vector<std::uint16_t>,EnvironmentFaceCount> FacesRGBA16;
        [[nodiscard]] Float3 Texel(EnvironmentFace face,unsigned x,unsigned y) const noexcept;
    };

    [[nodiscard]] bool ValidEnvironmentSource(const EnvironmentCaptureFrame& frame) noexcept;

    [[nodiscard]] std::shared_ptr<const EnvironmentRadiance> LoadEnvironmentRadiance(
        const std::filesystem::path& path,std::uint32_t faceResolution,std::string& error);
}
