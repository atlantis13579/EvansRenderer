#include "Renderer/Transparency/RefractionGeometry.h"

#include <cmath>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        constexpr std::size_t CornersPerTriangle = 3;
        // Byte lengths as well as element counts must fit native 32-bit buffers.
        constexpr auto MaximumBufferBytes = std::numeric_limits<std::uint32_t>::max();
        constexpr auto MaximumVertices = MaximumBufferBytes/sizeof(Vertex);
        constexpr auto MaximumIndices = MaximumBufferBytes/sizeof(std::uint32_t);

        bool Finite(const Vertex& v) noexcept
        {
            return std::isfinite(v.Position.X) && std::isfinite(v.Position.Y) && std::isfinite(v.Position.Z) &&
                std::isfinite(v.Normal.X) && std::isfinite(v.Normal.Y) && std::isfinite(v.Normal.Z) &&
                std::isfinite(v.Color.X) && std::isfinite(v.Color.Y) && std::isfinite(v.Color.Z) && std::isfinite(v.Color.W) &&
                std::isfinite(v.TexCoord.X) && std::isfinite(v.TexCoord.Y);
        }
    }

    bool BuildRefractionGeometry(const std::vector<std::shared_ptr<const RefractionSourceGeometry>>& sources,
        RefractionGeometry& output, std::string& error)
    {
        output = {}; error.clear();
        if (sources.size() > MaximumBufferBytes) { error = "Too many refraction geometry sources"; return false; }
        RefractionGeometry geometry;
        geometry.Sources.reserve(sources.size());
        std::unordered_map<const RefractionSourceGeometry*,std::size_t> unique;
        for (std::size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex)
        {
            const auto& source = sources[sourceIndex];
            const auto fail = [&](const char* reason)
            { error = "Refraction geometry " + std::to_string(sourceIndex) + ": " + reason; return false; };
            if (!source || source->Levels.empty()) return fail("missing immutable geometry or original LOD");
            const auto existing = unique.find(source.get());
            if (existing != unique.end()) { geometry.Sources.push_back(geometry.Sources[existing->second]); continue; }
            if (source->Vertices.size() > MaximumVertices-geometry.Vertices.size()) return fail("vertex buffer exceeds native byte capacity");
            if (source->Levels.size() > MaximumBufferBytes) return fail("too many LOD levels");
            for (const auto& vertex:source->Vertices) if (!Finite(vertex)) return fail("nonfinite vertex attribute");
            std::size_t indexCount = geometry.Indices.size();
            for (const auto& level:source->Levels)
            {
                if (level.size()%CornersPerTriangle) return fail("LOD indices do not form complete triangles");
                if (level.size() > MaximumIndices-indexCount) return fail("index buffer exceeds native byte capacity");
                for (auto index:level) if (index >= source->Vertices.size()) return fail("LOD vertex index is out of bounds");
                indexCount += level.size();
            }
            const auto base = static_cast<std::uint32_t>(geometry.Vertices.size());
            geometry.Vertices.insert(geometry.Vertices.end(),source->Vertices.begin(),source->Vertices.end());
            std::vector<RefractionGeometryRange> ranges;
            ranges.reserve(source->Levels.size());
            for (const auto& level:source->Levels)
            {
                ranges.push_back({static_cast<std::uint32_t>(geometry.Indices.size()),static_cast<std::uint32_t>(level.size())});
                for (auto index:level) geometry.Indices.push_back(base+index);
            }
            unique.emplace(source.get(),geometry.Sources.size());
            geometry.Sources.push_back(std::move(ranges));
        }
        output = std::move(geometry); return true;
    }

    bool PreparedRefractionGeometry::Prepare(const std::vector<std::shared_ptr<const RefractionSourceGeometry>>& sources, std::string& error)
    {
        error.clear();
        if (Geometry && Sources == sources) return true;
        auto geometry = std::make_shared<RefractionGeometry>();
        if (!BuildRefractionGeometry(sources,*geometry,error)) { Reset(); return false; }
        Sources = sources; Geometry = std::move(geometry); return true;
    }

    void PreparedRefractionGeometry::Reset() noexcept
    { Geometry.reset(); Sources.clear(); }
}
