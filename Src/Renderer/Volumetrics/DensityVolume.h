#pragma once

#include "Renderer/Types.h"
#include <filesystem>

namespace Evans::Renderer
{
    // Cell-centered density samples. A tiled 2D atlas keeps the same float data
    // accessible to every native API without requiring a 3D texture extension.
    struct DensityVolume
    {
        std::array<std::uint32_t,3> Dimensions{};
        Float3 BoundsMinimum,BoundsMaximum;
        std::uint32_t TileColumns=0;
        std::vector<std::uint8_t> AtlasRGBA8;
        [[nodiscard]] std::uint32_t AtlasWidth() const noexcept { return Dimensions[0]*TileColumns; }
        [[nodiscard]] std::uint32_t AtlasHeight() const noexcept
        { return TileColumns ? Dimensions[1]*((Dimensions[2]+TileColumns-1)/TileColumns) : 0; }
    };

    struct DensityVolumeBinding
    {
        TextureHandle Texture;
        Float4 Grid; // voxel dimensions XYZ, atlas tile columns W
    };

    [[nodiscard]] bool ValidDensityBinding(const DensityVolumeBinding& binding,const TextureDesc& texture) noexcept;

    [[nodiscard]] std::shared_ptr<const DensityVolume> LoadDensityVolume(
        const std::filesystem::path& path,std::string& error);
}
