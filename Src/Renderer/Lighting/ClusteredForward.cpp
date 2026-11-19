#include "Renderer/Lighting/ClusteredForward.h"
#include "Renderer/Shadows/ShadowProjection.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    // Olsson, Billeter and Assarsson, "Clustered Deferred and Forward
    // Shading", HPG 2012. https://www.cse.chalmers.se/~uffe/clustered_shading_preprint.pdf
    // Exponential eye-depth slices keep perspective clusters approximately
    // similar in shape. Orthographic frusta instead use linear slices.
    // This dense grid covers empty space and transparent receivers too; it
    // deliberately omits occupancy and normal-cone rejection. Thus light
    // assignment needs camera/bounds, not a depth prepass or a G-buffer.
    //
    // Pipeline alternatives remain exclusive. Gouraud cannot consume a
    // per-fragment cluster. AO/contact depth is independently required by
    // those effects. Captures use their own camera or a full-light loop;
    // volume integration can later query these full-frustum clusters.
    bool BuildClusteredForwardGrid(std::uint32_t width, std::uint32_t height, std::size_t lightCount,
        const ClusteredForwardParameters& parameters, ClusteredForwardGrid& result, std::string& error)
    {
        result = {}; error.clear();
        if (!width || !height || lightCount > MaximumForwardPlusLights || !parameters.TileSize ||
            parameters.TileSize > MaximumClusteredTileSize || !parameters.DepthSlices || parameters.DepthSlices > MaximumClusteredDepthSlices ||
            !parameters.ListCapacity || parameters.ListCapacity > MaximumClusteredListCapacity)
        { error = "Clustered dimensions, light count and grid parameters must be bounded"; return false; }
        ClusteredForwardGrid candidate;
        candidate.Width = width; candidate.Height = height; candidate.LightCount = std::uint32_t(lightCount);
        candidate.TileSize = parameters.TileSize; candidate.DepthSlices = parameters.DepthSlices; candidate.ListCapacity = parameters.ListCapacity;
        candidate.TilesX = width / parameters.TileSize + (width % parameters.TileSize != 0 ? 1u : 0u);
        candidate.TilesY = height / parameters.TileSize + (height % parameters.TileSize != 0 ? 1u : 0u);
        constexpr auto MaximumBufferBytes = std::numeric_limits<std::uint32_t>::max();
        const auto columns = std::uint64_t(candidate.TilesX) * candidate.TilesY;
        // Check before the slice multiplication and before converting to shader
        // addresses. Native backends also apply their device allocation limits.
        if (columns > MaximumBufferBytes / parameters.DepthSlices / sizeof(ClusteredForwardHeader))
        { error = "Clustered grid exceeds native 32-bit header addressing"; return false; }
        const auto clusters = columns * parameters.DepthSlices;
        if (clusters + ClusteredBoundarySlots > MaximumBufferBytes / sizeof(ClusteredForwardHeader) ||
            clusters > MaximumBufferBytes / (parameters.ListCapacity * sizeof(std::uint32_t)))
        { error = "Clustered list storage exceeds native 32-bit buffer addressing"; return false; }
        candidate.ClusterCount = std::uint32_t(clusters);
        candidate.IndexCount = candidate.ClusterCount * candidate.ListCapacity;
        result = candidate; return true;
    }
    bool BuildClusteredForwardDepthBoundaries(const Matrix4& camera, std::uint32_t slices,
        std::array<float, ClusteredBoundarySlots>& result, std::string& error)
    {
        result = {}; error.clear();
        ShadowProjection metric;
        if (!slices || slices > MaximumClusteredDepthSlices)
        { error = "Clustered depth slice count is out of range"; return false; }
        if (!BuildShadowProjection(camera, metric, error)) return false;
        std::array<float, ClusteredBoundarySlots> candidate {};
        for (std::uint32_t slice = 1; slice < slices; ++slice)
        {
            const double fraction = double(slice) / slices;
            double depth = fraction;
            if (metric.Perspective)
            {
                const double eye = std::exp(std::log(metric.NearDistance) +
                    fraction * (std::log(metric.FarDistance) - std::log(metric.NearDistance)));
                depth = (1 - metric.NearDistance / eye) / (1 - metric.NearDistance / metric.FarDistance);
            }
            candidate[slice] = std::max(candidate[slice - 1], float(std::clamp(depth, 0.0, 1.0)));
        }
        // Inactive records also contain far depth, preventing uninitialized
        // metadata and permitting one fixed native upload size.
        std::fill(candidate.begin() + slices, candidate.end(), 1.0f);
        result = candidate; return true;
    }
}
