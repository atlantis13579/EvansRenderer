#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr float MaximumNaniteErrorPixels=256;
    inline constexpr std::uint32_t MaximumNanitePages=16*1024,MaximumNaniteUploads=1024;
    inline constexpr std::uint32_t DefaultNanitePhysicalPages=256;
    inline constexpr std::uint32_t DefaultNaniteScenePages=2048,MaximumNaniteScenePages=16*1024;
    struct NaniteSettings
    {
        bool Enabled=false,Occlusion=true,MaterialTiles=true,GpuInstances=true,GpuHardware=true;
        float MaximumErrorPixels=1;
        std::uint32_t PhysicalPages=DefaultNanitePhysicalPages,UploadsPerFrame=64; // Per-family ceiling; zero removes this ceiling.
        // Total resident page slots in one scene generation. Immutable in-flight
        // replacements have their own bounded native transaction allocations.
        std::uint32_t ScenePages=DefaultNaniteScenePages;
        unsigned MaximumSoftwareExtent=8,SubpixelBits=8;
    };
    inline bool operator==(const NaniteSettings& a,const NaniteSettings& b) noexcept
    {
        return a.Enabled==b.Enabled && a.Occlusion==b.Occlusion && a.MaterialTiles==b.MaterialTiles && a.GpuInstances==b.GpuInstances && a.GpuHardware==b.GpuHardware && a.MaximumErrorPixels==b.MaximumErrorPixels && a.PhysicalPages==b.PhysicalPages && a.ScenePages==b.ScenePages &&
            a.UploadsPerFrame==b.UploadsPerFrame && a.MaximumSoftwareExtent==b.MaximumSoftwareExtent && a.SubpixelBits==b.SubpixelBits;
    }
    inline NaniteSettings SanitizeNaniteSettings(NaniteSettings value) noexcept
    {
        value.MaximumErrorPixels=std::isfinite(value.MaximumErrorPixels)?std::clamp(value.MaximumErrorPixels,0.0f,MaximumNaniteErrorPixels):1;
        value.PhysicalPages=std::min(value.PhysicalPages,MaximumNanitePages); value.UploadsPerFrame=std::min(value.UploadsPerFrame,MaximumNaniteUploads);
        value.ScenePages=std::clamp(value.ScenePages,1u,MaximumNaniteScenePages);
        value.MaximumSoftwareExtent=std::clamp(value.MaximumSoftwareExtent,1u,32u); value.SubpixelBits=std::clamp(value.SubpixelBits,4u,8u); return value;
    }
}
