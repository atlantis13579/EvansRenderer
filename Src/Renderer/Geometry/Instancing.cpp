#include "Renderer/Geometry/Instancing.h"
#include "Renderer/Lighting/SceneLightCollection.h"

#include <cstring>

namespace Evans::Renderer
{
    /*
    Constant-buffer instancing
    --------------------------
    One indexed native command replays shared geometry with an array of object
    transforms. See Dudash, "Animated Crowd Rendering", GPU Gems 3, chapter 2:
    https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-2-animated-crowd-rendering
    This rigid variant has no skinning or extra authored data. Inverse-transpose
    normals are computed per instance, including nonuniform scale and shear.

    Only adjacent, compatible submissions are combined. This preserves primitive
    order and coplanar depth ties. Geometry identity, selected indices, material,
    raster state and winding must match; visibility is decided before batching.
    Sorted alpha draws and selection outlines remain individual. Current/previous
    transforms and validity are independent, even in a common native draw.
    Camera jitter is frozen per frame and shared by the fragment stage: instances
    with different jitter deltas cannot share one draw's motion-vector uniforms.
    */
    bool BuildInstanceConstants(const std::vector<DrawInstance>& instances,
        const Matrix4& representativeWorld, InstanceConstantBlock& output)
    {
        if (instances.empty() || instances.size() > MaximumDrawInstances ||
            !FrameHistory::IsFinite(representativeWorld)) return false;
        const bool mirrored = RealtimeWorldIsMirrored(representativeWorld);
        for (std::size_t index = 0; index < instances.size(); ++index)
        {
            const auto& source = instances[index];
            if (!FrameHistory::IsFinite(source.WorldTransform) ||
                RealtimeWorldIsMirrored(source.WorldTransform) != mirrored ||
                !std::isfinite(source.Motion.JitterDeltaPixels.X) || !std::isfinite(source.Motion.JitterDeltaPixels.Y) ||
                source.Motion.JitterDeltaPixels.X != instances.front().Motion.JitterDeltaPixels.X ||
                source.Motion.JitterDeltaPixels.Y != instances.front().Motion.JitterDeltaPixels.Y ||
                (source.Motion.Valid && !FrameHistory::IsFinite(source.Motion.PreviousWorldViewProjection))) return false;
            auto& target = output.Instances[index];
            target.Wind=source.Wind; target.Gust=source.Gust; target.VegetationHistory=source.VegetationHistory;
            target.World = source.WorldTransform;
            target.NormalWorld = BuildRealtimeNormalMatrix(source.WorldTransform);
            target.PreviousWorldViewProjection.Values.fill(0);
            if (source.Motion.Valid) target.PreviousWorldViewProjection = source.Motion.PreviousWorldViewProjection;
        }
        return true;
    }

    bool SameInstancedPipeline(const PipelineState& a, const PipelineState& b)
    {
        if (a.Fill != b.Fill || a.Shading != b.Shading || a.BackFaceCulling != b.BackFaceCulling ||
            a.DepthTest != b.DepthTest || a.DepthWrite != b.DepthWrite || a.ColorWrite != b.ColorWrite ||
            a.BlendEnabled != b.BlendEnabled || a.SamplesShadowMap != b.SamplesShadowMap ||
            a.OutlineExtrusion != b.OutlineExtrusion || a.DiffuseTexture != b.DiffuseTexture ||
            a.NormalTexture != b.NormalTexture || a.OpacityTexture != b.OpacityTexture ||
            !SameSceneLights(a.Lighting, b.Lighting)) return false;
        // Compare the explicitly packed shader ABI, not PipelineState's padding.
        // Fixed identity transforms remove per-object state. The comparison above
        // covers full scene-light storage; constants cover the inline subset,
        // camera, material and surface parameters.
        const auto left = BuildRealtimeDrawConstants(a, {}, {}, {}, PassType::Color, true, true, true, true);
        const auto right = BuildRealtimeDrawConstants(b, {}, {}, {}, PassType::Color, true, true, true, true);
        return std::memcmp(&left, &right, sizeof(left)) == 0;
    }
}
