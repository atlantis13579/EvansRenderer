#pragma once

#include "Renderer/Lighting/SceneLightCollection.h"

namespace Evans::Renderer
{
    inline bool SameTemporalLighting(const LightingState& a, const LightingState& b) noexcept
    {
        const auto same3 = [](Float3 x, Float3 y) { return x.X == y.X && x.Y == y.Y && x.Z == y.Z; };
        if (a.AmbientStrength != b.AmbientStrength || !same3(a.AmbientColor, b.AmbientColor) ||
            !SameSceneLights(a, b)) return false;
        return true; // camera translation is handled by reprojection, not a light reset
    }
    // Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014.
    // https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
    // A base-2/base-3 Halton cycle varies the raster sample without moving the
    // physical camera. Commit only after presentation; aborted frames retry the
    // same sample and retain the previous successful jitter for reprojection.
    class TemporalSampling
    {
    public:
        static constexpr std::uint32_t CycleLength = 8;
        static float RadicalInverse(std::uint32_t index, std::uint32_t base) noexcept
        {
            float value = 0.0f, fraction = 1.0f;
            while (index) { fraction /= base; value += fraction * (index % base); index /= base; }
            return value;
        }
        [[nodiscard]] TemporalFrame Describe(bool correspondence, bool rasterJitter = true, std::uint32_t cycleLength = CycleLength) const noexcept
        {
            TemporalFrame frame;
            frame.HistoryValid = Committed && correspondence;
            frame.SampleIndex = frame.HistoryValid ? NextSample % std::max(1u,cycleLength) : 0;
            const auto index = frame.SampleIndex + 1;
            // Diffuse transport advances its noise sequence without requiring TAA jitter.
            if (rasterJitter) frame.JitterPixels = {RadicalInverse(index, 2) - 0.5f, RadicalInverse(index, 3) - 0.5f};
            frame.PreviousJitterPixels = frame.HistoryValid ? PreviousJitter : frame.JitterPixels;
            return frame;
        }
        void Commit(const TemporalFrame& frame, std::uint32_t cycleLength = CycleLength) noexcept
        {
            PreviousJitter = frame.JitterPixels;
            NextSample = (frame.SampleIndex + 1) % std::max(1u,cycleLength);
            Committed = true;
        }
        // Upscaling needs more distinct input samples per presentation pixel.
        // The supported half-size input bounds the cycle to four base periods.
        [[nodiscard]] static std::uint32_t ReconstructionCycleLength(unsigned renderWidth,unsigned renderHeight,unsigned outputWidth,unsigned outputHeight) noexcept
        {
            const std::uint64_t inputPixels=std::uint64_t(renderWidth)*renderHeight,outputPixels=std::uint64_t(outputWidth)*outputHeight;
            if (!inputPixels) return CycleLength;
            constexpr std::uint32_t MaximumAreaRatio=4;
            return static_cast<std::uint32_t>(std::clamp<std::uint64_t>((CycleLength*outputPixels+inputPixels-1)/inputPixels,CycleLength,CycleLength*MaximumAreaRatio));
        }
        void Reset() noexcept { Committed = false; }
        [[nodiscard]] static Matrix4 Jitter(const Matrix4& projection, Float2 pixels, int width, int height) noexcept
        {
            Matrix4 result = projection;
            if (width <= 0 || height <= 0) return result;
            const float x = 2.0f * pixels.X / width, y = -2.0f * pixels.Y / height;
            for (int column = 0; column < 4; ++column)
            {
                result.Values[column] += x * projection.Values[12 + column];
                result.Values[4 + column] += y * projection.Values[12 + column];
            }
            return result;
        }
    private:
        Float2 PreviousJitter;
        std::uint32_t NextSample = 0;
        bool Committed = false;
    };
}
