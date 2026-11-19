#pragma once

#include "Renderer/Types.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    // Transform history is a transaction over successful frames, not draw calls.
    // Sample time belongs to the scene producer, so worker latency does not change
    // the inferred speed. The current/previous clip pair supports object motion
    // as well as camera motion (GPU Gems 3, ch. 27, Gilberto Rosado, 2007).
    // https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
    // Commit only after Present succeeds. ClearScene, camera cuts, pipeline/size
    // changes and backend epochs discard correspondence. Long gaps and duplicate
    // timestamps disable shutter velocity, but preserve geometric reprojection.
    // Otherwise slow GPU frames keep restarting TAA/TSR and temporal GI at sample
    // zero. Explicit epochs and reversed time still discard correspondence.
    class FrameHistory
    {
    public:
        static constexpr double MinimumIntervalSeconds = 1.0e-6;
        static constexpr double MaximumIntervalSeconds = 0.25;

        static bool IsFinite(const Matrix4& matrix) noexcept
        {
            return std::all_of(matrix.Values.begin(), matrix.Values.end(),
                [](float value) { return std::isfinite(value); });
        }

        [[nodiscard]] MotionFrame Describe(const Matrix4& currentViewProjection,
            double sampleTimeSeconds, std::uint64_t epoch) const noexcept
        {
            MotionFrame result;
            result.PreviousViewProjection = currentViewProjection;
            const double interval = sampleTimeSeconds - SampleTimeSeconds;
            if (Valid && Epoch == epoch && std::isfinite(sampleTimeSeconds) &&
                IsFinite(currentViewProjection) && std::isfinite(interval) && interval >= 0.0)
            {
                result.PreviousViewProjection = ViewProjection;
                result.CorrespondenceValid = true;
                result.Valid = interval >= MinimumIntervalSeconds && interval <= MaximumIntervalSeconds;
                result.DeltaSeconds = result.Valid ? static_cast<float>(interval) : 0.0f;
            }
            return result;
        }

        void Commit(const Matrix4& viewProjection, double sampleTimeSeconds,
            std::uint64_t epoch) noexcept
        {
            Valid = std::isfinite(sampleTimeSeconds) && IsFinite(viewProjection);
            ViewProjection = viewProjection;
            SampleTimeSeconds = sampleTimeSeconds;
            Epoch = epoch;
        }

        void Reset() noexcept { Valid = false; }

    private:
        Matrix4 ViewProjection;
        double SampleTimeSeconds = 0.0;
        std::uint64_t Epoch = 0;
        bool Valid = false;
    };
}
