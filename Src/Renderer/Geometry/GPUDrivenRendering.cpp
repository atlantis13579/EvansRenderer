#include "Renderer/Geometry/GPUDrivenRendering.h"

#include <limits>

namespace Evans::Renderer
{
    bool BuildGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchUpload& output)
    {
        output = {};
        const auto finite3 = [](const Float3& value)
        { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); };
        if (!finite3(desc.BoundsMinimum) || !finite3(desc.BoundsMaximum) ||
            desc.BoundsMinimum.X > desc.BoundsMaximum.X || desc.BoundsMinimum.Y > desc.BoundsMaximum.Y ||
            desc.BoundsMinimum.Z > desc.BoundsMaximum.Z || !FrameHistory::IsFinite(desc.CullingViewProjection) ||
            !FrameHistory::IsFinite(desc.RenderViewProjection) || !std::isfinite(desc.Material.X) ||
            !std::isfinite(desc.Material.Y) || !std::isfinite(desc.Material.Z) || !std::isfinite(desc.Material.W) ||
            desc.IndexCount > std::numeric_limits<std::uint32_t>::max() - desc.FirstIndex ||
            desc.Instances.size() > MaximumDrawInstances) return false;

        InstanceConstantBlock instances;
        if (!desc.Instances.empty())
        {
            if (!BuildInstanceConstants(desc.Instances, desc.Instances.front().WorldTransform, instances)) return false;
            for (std::size_t index = 0; index < desc.Instances.size(); ++index)
                if (!FrameHistory::IsFinite(instances.Instances[index].NormalWorld)) return false;
        }
        auto& constants = output.Constants;
        constants.CullingViewProjection = desc.CullingViewProjection;
        constants.RenderViewProjection = desc.RenderViewProjection;
        constants.BoundsMinimum = {desc.BoundsMinimum.X, desc.BoundsMinimum.Y, desc.BoundsMinimum.Z, 0};
        constants.BoundsMaximum = {desc.BoundsMaximum.X, desc.BoundsMaximum.Y, desc.BoundsMaximum.Z, 0};
        constants.Material = desc.Material;
        constants.Draw = {desc.IndexCount, desc.FirstIndex, static_cast<std::uint32_t>(desc.BaseVertex),
            static_cast<std::uint32_t>(desc.Instances.size())};
        constants.Control = {desc.FrustumCulling ? 1u : 0u, 0, 0, 0};
        output.Candidates = instances.Instances;
        return true;
    }
}
