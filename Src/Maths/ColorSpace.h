#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Maths
{

[[nodiscard]] inline float SRGBToLinear(float value) noexcept
{
    if (!std::isfinite(value))
    {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.04045f
        ? value / 12.92f
        : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

[[nodiscard]] inline float LinearToSRGB(float value) noexcept
{
    if (!std::isfinite(value))
    {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.0031308f
        ? value * 12.92f
        : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

[[nodiscard]] inline std::uint8_t LinearToSRGB8(float value) noexcept
{
    return static_cast<std::uint8_t>(LinearToSRGB(value) * 255.0f + 0.5f);
}

[[nodiscard]] inline std::uint8_t LinearToUNorm8(float value) noexcept
{
    if (!std::isfinite(value))
    {
        value = value > 0.0f ? 1.0f : 0.0f;
    }
    return static_cast<std::uint8_t>(
        std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Scale non-negative linear RGB into the display range while preserving hue.
// This is the shared tone-mapping contract used before sRGB or linear-UNORM
// output encoding. Values already in [0, 1] are unchanged.
[[nodiscard]] inline float DisplayLinearScale(
    float red,
    float green,
    float blue) noexcept
{
    const float maximum = std::max({
        1.0f,
        std::max(red, 0.0f),
        std::max(green, 0.0f),
        std::max(blue, 0.0f)});
    return 1.0f / maximum;
}

} // namespace Evans::Maths
