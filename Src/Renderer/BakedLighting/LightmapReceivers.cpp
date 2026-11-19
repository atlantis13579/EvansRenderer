#include "Renderer/BakedLighting/LightmapReceivers.h"

#include <algorithm>

namespace Evans::Renderer
{
    bool BuildLightmapReceivers(const std::vector<LightmapSourceInstance>& instances,
        const Techniques::Settings& settings, const LightmapAtlasSettings& atlasSettings,
        LightmapReceivers& output, std::string& error, const std::function<bool()>& cancelled)
    {
        error.clear();
        auto stop = [&] {
            if (cancelled && cancelled()) { error = "Lightmap receiver preparation cancelled"; return true; }
            return false;
        };
        if (stop()) return false;
        LightmapReceivers result;
        std::vector<SceneRayMeshView> baseViews, receiverViews;
        struct ReceiverView { std::uint32_t Instance,Level,FirstRange,RangeCount; };
        std::vector<ReceiverView> views;
        constexpr std::uint32_t CornersPerTriangle = 3;
        const auto indexLimit = std::numeric_limits<std::uint32_t>::max();
        if (instances.size() > indexLimit) { error = "Too many lightmap instances"; return false; }
        std::size_t cornerCount = 0;
        for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex)
        {
            if (stop()) return false;
            const auto& instance = instances[instanceIndex];
            if (!instance.Visible) continue;
            if (!instance.Geometry || instance.Geometry->Levels.empty() || instance.Geometry->Levels.size() > indexLimit)
            { error = "Visible lightmap instances require original geometry and LOD indices"; return false; }
            SceneRayMeshView view;
            view.Vertices = &instance.Geometry->Vertices; view.Indices = &instance.Geometry->Levels.front();
            view.World = instance.World; view.Color = instance.Color; view.Material = instance.Material;
            view.CastShadow = instance.CastShadow;
            baseViews.push_back(view);
            for (std::size_t level = 0; level < instance.Geometry->Levels.size(); ++level)
            {
                const auto& indices = instance.Geometry->Levels[level];
                if (indices.size() > indexLimit-cornerCount || receiverViews.size() == indexLimit)
                { error = "Lightmap receiver streams exceed native 32-bit draw capacity"; return false; }
                cornerCount += indices.size(); view.Indices = &indices;
                const auto firstRange=static_cast<std::uint32_t>(result.Ranges.size());
                if (level<instance.Geometry->Segments.size() && !instance.Geometry->Segments[level].empty())
                {
                    std::uint64_t covered=0;
                    for (const auto& segment:instance.Geometry->Segments[level])
                    {
                        if (!segment.TriangleCount || segment.FirstTriangle!=covered || segment.Tag==LightmapUnsegmented || result.Ranges.size()==indexLimit)
                        { error="Lightmap receiver segments overlap, leave a gap or have an invalid tag"; return false; }
                        covered+=segment.TriangleCount;
                        result.Ranges.push_back({static_cast<std::uint32_t>(instanceIndex),static_cast<std::uint32_t>(level),0,0,segment.Tag});
                    }
                    if (covered*CornersPerTriangle!=indices.size()) { error="Lightmap receiver segments do not cover their level"; return false; }
                }
                else result.Ranges.push_back({static_cast<std::uint32_t>(instanceIndex),static_cast<std::uint32_t>(level),0,0});
                views.push_back({static_cast<std::uint32_t>(instanceIndex),static_cast<std::uint32_t>(level),firstRange,static_cast<std::uint32_t>(result.Ranges.size()-firstRange)});
                receiverViews.push_back(view);
            }
        }
        result.TransportScene = SceneRayQuery::Create(baseViews,settings,error);
        if (!result.TransportScene || stop()) return false;
        const auto receiverScene = SceneRayQuery::Create(receiverViews,settings,error);
        if (!receiverScene || stop()) return false;
        auto atlas = std::make_shared<LightmapAtlas>();
        if (!BuildLightmapAtlas(receiverScene,atlasSettings,*atlas,error) || stop()) return false;
        result.Atlas = atlas;
        result.Vertices.reserve(receiverScene->GetTriangleCount()*CornersPerTriangle);
        std::size_t nextRange = 0;
        for (std::uint32_t triangleIndex = 0; triangleIndex < atlas->Triangles.size(); ++triangleIndex)
        {
            if (stop()) return false;
            const auto& mapping = atlas->Triangles[triangleIndex];
            SceneRayTriangle triangle;
            if (!receiverScene->GetTriangle(triangleIndex,triangle) || triangle.MeshIndex >= views.size())
            { error = "Lightmap query lost its source receiver mapping"; return false; }
            // Query triangles are contiguous in source-view order, including
            // zero-length ranges for filtered blend instances. Degenerate
            // triangles have no UV area and produce no native receiver vertices.
            const auto& view=views[triangle.MeshIndex]; auto rangeIndex=view.FirstRange;
            const auto& segments=instances[view.Instance].Geometry->Segments;
            if (view.Level<segments.size() && !segments[view.Level].empty())
            {
                const auto& partition=segments[view.Level];
                const auto found=std::upper_bound(partition.begin(),partition.end(),triangle.SourceTriangleIndex,
                    [](auto face,const auto& segment){return face<segment.FirstTriangle;});
                if (found==partition.begin()) { error="Lightmap query lost its receiver segment"; return false; }
                rangeIndex+=static_cast<std::uint32_t>((found-partition.begin())-1);
            }
            while (nextRange <= rangeIndex)
                result.Ranges[nextRange++].FirstVertex = static_cast<std::uint32_t>(result.Vertices.size());
            if (mapping.Chart == InvalidLightmapChart) continue;
            auto& range = result.Ranges[rangeIndex];
            const auto& source = *instances[range.InstanceIndex].Geometry;
            std::array<LightmapReceiverVertex,CornersPerTriangle> corners;
            for (std::size_t queryCorner = 0; queryCorner < CornersPerTriangle; ++queryCorner)
            {
                const auto& vertex = source.Vertices[triangle.SourceVertexIndices[queryCorner]];
                corners[triangle.SourceCorners[queryCorner]] = {vertex.Position,vertex.TexCoord,mapping.UV[queryCorner],vertex.Color};
            }
            result.Vertices.insert(result.Vertices.end(),corners.begin(),corners.end());
            range.VertexCount += CornersPerTriangle;
        }
        while (nextRange < result.Ranges.size())
            result.Ranges[nextRange++].FirstVertex = static_cast<std::uint32_t>(result.Vertices.size());
        if (stop()) return false;
        output = std::move(result);
        return true;
    }
}
