#pragma once

#include "Maths/ColorSpace.h"
#include "Renderer/Types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace Evans::Renderer
{
    constexpr unsigned MaterialAnisotropy = 8;

    inline int MaterialMipLevels(const TextureDesc& desc)
    {
        int levels = 1;
        for (int extent = std::max(desc.Width, desc.Height); extent > 1; extent /= 2) ++levels;
        return levels;
    }

    inline bool ValidTextureMipLevels(const TextureDesc& desc, const void* data)
    {
        return desc.MipLevels == 1 || (desc.MipLevels > 1 && data && !desc.RenderTarget &&
            desc.Format == TextureFormat::RGBA8 && desc.Width > 0 && desc.Height > 0 &&
            desc.MipLevels <= MaterialMipLevels(desc));
    }

    // A packed RGBA8 mip upload has no row or subresource alignment padding.
    template<class Visitor>
    std::size_t VisitTextureMips(const TextureDesc& desc, Visitor visit)
    {
        int width = desc.Width, height = desc.Height;
        std::size_t offset = 0;
        for (int level = 0; level < desc.MipLevels; ++level)
        {
            visit(level, width, height, offset);
            offset += std::size_t(width) * height * 4;
            width = std::max(1, width / 2);
            height = std::max(1, height / 2);
        }
        return offset;
    }

    inline std::size_t TextureMipBytes(const TextureDesc& desc)
    {
        return VisitTextureMips(desc, [](int, int, int, std::size_t) {});
    }

    inline std::vector<std::uint8_t> BuildMaterialMipChain(const TextureDesc& desc,
        const std::vector<std::uint8_t>& base)
    {
        static const auto linear = [] {
            std::array<float, 256> values{};
            for (unsigned i = 0; i < values.size(); ++i) values[i] = Maths::SRGBToLinear(i / 255.0f);
            return values;
        }();
        const bool srgb = desc.ColorSpace == TextureColorSpace::SRGB;
        std::vector<std::uint8_t> mips(TextureMipBytes(desc));
        std::copy(base.begin(), base.end(), mips.begin());
        const std::uint8_t* source = mips.data();
        int sourceWidth = desc.Width, sourceHeight = desc.Height;
        VisitTextureMips(desc, [&](int level, int width, int height, std::size_t offset) {
            if (level == 0) return;
            auto* target = mips.data() + offset;
            for (int y = 0; y < height; ++y)
            {
                const int top = y * sourceHeight / height;
                const int bottom = (y + 1) * sourceHeight / height;
                for (int x = 0; x < width; ++x)
                {
                    const int left = x * sourceWidth / width;
                    const int right = (x + 1) * sourceWidth / width;
                    std::array<float, 4> sum{};
                    for (int row = top; row < bottom; ++row)
                        for (int column = left; column < right; ++column)
                        {
                            const auto* texel = source + (std::size_t(row) * sourceWidth + column) * 4;
                            for (unsigned channel = 0; channel < 4; ++channel)
                                sum[channel] += srgb && channel < 3 ? linear[texel[channel]] : texel[channel];
                        }
                    const float inverseCount = 1.0f / ((bottom - top) * (right - left));
                    for (unsigned channel = 0; channel < 4; ++channel)
                    {
                        const float average = sum[channel] * inverseCount;
                        *target++ = srgb && channel < 3 ? Maths::LinearToSRGB8(average)
                            : static_cast<std::uint8_t>(average + 0.5f);
                    }
                }
            }
            source = mips.data() + offset;
            sourceWidth = width;
            sourceHeight = height;
        });
        return mips;
    }
}
