#include "Renderer/Lumen/LumenMeshPartition.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace Evans::Renderer
{
    /* Lumen's material charts and signed-distance volumes have finite spatial
       resolution. One imported mesh can contain hundreds of distant columns,
       rods or leaves, so using its full bounds for every proxy loses detail.
       This independent preprocessing extension turns edge-connected components
       into smaller local proxy domains. It follows the modular-geometry guidance
       in Epic's Lumen Technical Details, not Unreal's implementation code:
       https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine

       Exact-position edge welding ignores attribute seams for connectivity;
       compact render vertices still retain every original normal, UV and color.
       Balanced spatial gap splits group component centroids under a caller budget.
       Components remain indivisible, preserving closed-shell sign information.
       This cannot repair nonmanifold input or thin features in a connected mesh.
    */
    namespace
    {
        constexpr unsigned TriangleCorners=3,Invalid=UINT32_MAX,CancellationInterval=1024;
        struct Position
        {
            std::array<std::uint32_t,3> Bits;
            bool operator==(const Position& other) const { return Bits==other.Bits; }
        };
        struct PositionHash
        {
            std::size_t operator()(const Position& value) const noexcept
            {
                // FNV-1a over canonical float words; hash order never determines output order.
                constexpr std::uint64_t Offset=14695981039346656037ull,Prime=1099511628211ull;
                std::uint64_t hash=Offset; for (const auto word:value.Bits) { hash^=word; hash*=Prime; } return std::size_t(hash);
            }
        };
        Position Key(Float3 p)
        {
            Position result; const float values[]{p.X==0?0:p.X,p.Y==0?0:p.Y,p.Z==0?0:p.Z};
            std::memcpy(result.Bits.data(),values,sizeof(values)); return result;
        }
        bool Finite(const Vertex& vertex)
        {
            for (const auto value:{vertex.Position.X,vertex.Position.Y,vertex.Position.Z,vertex.Normal.X,vertex.Normal.Y,vertex.Normal.Z,
                vertex.Color.X,vertex.Color.Y,vertex.Color.Z,vertex.Color.W,vertex.TexCoord.X,vertex.TexCoord.Y}) if (!std::isfinite(value)) return false;
            return true;
        }
        double Area(Float3 a,Float3 b,Float3 c)
        {
            const double ab[]{double(b.X)-a.X,double(b.Y)-a.Y,double(b.Z)-a.Z},ac[]{double(c.X)-a.X,double(c.Y)-a.Y,double(c.Z)-a.Z};
            return std::hypot(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])*0.5;
        }
        unsigned Root(std::vector<unsigned>& parents,unsigned node)
        {
            while (parents[node]!=node) { parents[node]=parents[parents[node]]; node=parents[node]; } return node;
        }
        bool Interrupted(const LumenPartitionCancellation& cancelled,std::string& error)
        { if (cancelled && cancelled()) { error="Lumen mesh partition cancelled"; return true; } return false; }
        struct Component { Maths::AABB Bounds; unsigned FirstTriangle=Invalid; };
        struct Range { unsigned First=0,Count=0,Budget=0; };
    }
    bool BuildLumenMeshPartition(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const LumenMeshPartitionSettings& settings,LumenMeshPartition& output,std::string& error,const LumenPartitionCancellation& cancelled)
    {
        if (!settings.MaximumParts || settings.MaximumParts>LumenPartitionMaximumParts || indices.size()%TriangleCorners ||
            indices.size()/TriangleCorners>LumenPartitionMaximumTriangles || vertices.size()>LumenPartitionMaximumTriangles*TriangleCorners)
        { error="Lumen mesh partition exceeds triangle, vertex or part limits"; return false; }
        if (Interrupted(cancelled,error)) return false;
        LumenMeshPartition result;
        if (indices.empty()) { output=std::move(result); error.clear(); return true; }
        for (const auto index:indices) if (index>=vertices.size()) { error="Lumen partition index is outside the source vertices"; return false; }
        for (const auto& vertex:vertices) if (!Finite(vertex)) { error="Lumen partition source vertices must be finite"; return false; }
        const auto triangleCount=unsigned(indices.size()/TriangleCorners);
        std::vector<unsigned> parents(triangleCount),welded(vertices.size(),Invalid); std::iota(parents.begin(),parents.end(),0u);
        std::unordered_map<Position,unsigned,PositionHash> positions;
        std::unordered_map<std::uint64_t,unsigned> edges;
        for (unsigned triangle=0;triangle<triangleCount;++triangle)
        {
            if (triangle%CancellationInterval==0 && Interrupted(cancelled,error)) return false;
            unsigned corners[TriangleCorners];
            for (unsigned corner=0;corner<TriangleCorners;++corner)
            {
                const auto index=indices[std::size_t(triangle)*TriangleCorners+corner]; auto& id=welded[index];
                if (id==Invalid) id=positions.emplace(Key(vertices[index].Position),unsigned(positions.size())).first->second;
                corners[corner]=id;
            }
            const auto first=std::size_t(triangle)*TriangleCorners;
            // A zero-area connector must not join separate shells. It remains
            // an explicit component and retains its source triangle mapping.
            if (Area(vertices[indices[first]].Position,vertices[indices[first+1]].Position,vertices[indices[first+2]].Position)==0) continue;
            for (unsigned corner=0;corner<TriangleCorners;++corner)
            {
                const auto a=corners[corner],b=corners[(corner+1)%TriangleCorners];
                const auto key=(std::uint64_t(std::min(a,b))<<32)|std::max(a,b);
                const auto inserted=edges.emplace(key,triangle);
                if (!inserted.second)
                {
                    const auto left=Root(parents,triangle),right=Root(parents,inserted.first->second);
                    parents[std::max(left,right)]=std::min(left,right);
                }
            }
        }
        std::vector<Component> components; std::vector<unsigned> roots(triangleCount,Invalid),triangleComponents(triangleCount);
        for (unsigned triangle=0;triangle<triangleCount;++triangle)
        {
            if (triangle%CancellationInterval==0 && Interrupted(cancelled,error)) return false;
            auto& component=roots[Root(parents,triangle)];
            if (component==Invalid) { component=unsigned(components.size()); components.push_back({{},triangle}); }
            triangleComponents[triangle]=component;
            for (unsigned corner=0;corner<TriangleCorners;++corner)
            { const auto p=vertices[indices[std::size_t(triangle)*TriangleCorners+corner]].Position; components[component].Bounds.Expand({p.X,p.Y,p.Z}); }
        }
        result.ConnectedComponents=unsigned(components.size());
        std::vector<unsigned> order(components.size()); std::iota(order.begin(),order.end(),0u);
        std::vector<Range> stack{{0,unsigned(order.size()),std::min(settings.MaximumParts,unsigned(order.size()))}},groups;
        while (!stack.empty())
        {
            if (Interrupted(cancelled,error)) return false;
            const auto range=stack.back(); stack.pop_back();
            if (range.Budget==1) { groups.push_back(range); continue; }
            std::array<double,3> minimum{INFINITY,INFINITY,INFINITY},maximum{-INFINITY,-INFINITY,-INFINITY};
            const auto center=[&](unsigned component,unsigned axis)
            { return (double(components[component].Bounds.Min[axis])+components[component].Bounds.Max[axis])*0.5; };
            for (unsigned index=range.First;index<range.First+range.Count;++index) for (unsigned axis=0;axis<3;++axis)
            { const auto value=center(order[index],axis); minimum[axis]=std::min(minimum[axis],value); maximum[axis]=std::max(maximum[axis],value); }
            unsigned axis=0; for (unsigned other=1;other<3;++other) if (maximum[other]-minimum[other]>maximum[axis]-minimum[axis]) axis=other;
            std::sort(order.begin()+range.First,order.begin()+range.First+range.Count,[&](unsigned a,unsigned b)
            { const auto ca=center(a,axis),cb=center(b,axis); return ca!=cb?ca<cb:components[a].FirstTriangle<components[b].FirstTriangle; });
            // Prefer actual spatial gaps near the median, while bounding tree
            // imbalance so a sequence of outliers cannot cause quadratic work.
            constexpr unsigned BalanceDivisor=4;
            const unsigned margin=std::max(1u,range.Count/BalanceDivisor),target=range.Count/2;
            unsigned leftCount=target; double bestGap=-1;
            for (unsigned split=margin;split<=range.Count-margin;++split)
            {
                const auto gap=center(order[range.First+split],axis)-center(order[range.First+split-1],axis);
                if (gap>bestGap || (gap==bestGap && std::abs(int(split)-int(target))<std::abs(int(leftCount)-int(target))))
                { bestGap=gap; leftCount=split; }
            }
            const unsigned proposed=unsigned((std::uint64_t(range.Budget)*leftCount+range.Count/2)/range.Count);
            const unsigned minimumBudget=std::max(1,int(range.Budget)-int(range.Count-leftCount));
            const unsigned leftBudget=std::clamp(proposed,minimumBudget,std::min(leftCount,range.Budget-1)),middle=range.First+leftCount;
            stack.push_back({middle,range.Count-leftCount,range.Budget-leftBudget}); stack.push_back({range.First,leftCount,leftBudget});
        }
        // Source order is the stable public identity, independent of spatial
        // traversal order or the standard library's partition implementation.
        const auto firstTriangle=[&](const Range& range)
        { unsigned first=Invalid; for (unsigned i=range.First;i<range.First+range.Count;++i) first=std::min(first,components[order[i]].FirstTriangle); return first; };
        std::sort(groups.begin(),groups.end(),[&](const auto& a,const auto& b){return firstTriangle(a)<firstTriangle(b);});
        result.Parts.resize(groups.size()); std::vector<unsigned> componentParts(components.size());
        for (unsigned part=0;part<groups.size();++part)
        {
            const auto range=groups[part]; result.Parts[part].Components=range.Count;
            for (unsigned i=range.First;i<range.First+range.Count;++i) componentParts[order[i]]=part;
        }
        result.TriangleParts.resize(triangleCount);
        for (unsigned triangle=0;triangle<triangleCount;++triangle)
        {
            const auto part=componentParts[triangleComponents[triangle]]; result.TriangleParts[triangle]=part; result.Parts[part].SourceTriangles.push_back(triangle);
        }
        std::vector<unsigned> compact(vertices.size(),Invalid); unsigned processed=0;
        for (auto& part:result.Parts)
        {
            for (const auto triangle:part.SourceTriangles)
            {
                if (processed++%CancellationInterval==0 && Interrupted(cancelled,error)) return false;
                const auto first=std::size_t(triangle)*TriangleCorners;
                part.SurfaceArea+=Area(vertices[indices[first]].Position,vertices[indices[first+1]].Position,vertices[indices[first+2]].Position);
                for (unsigned corner=0;corner<TriangleCorners;++corner)
                {
                    const auto source=indices[first+corner]; auto& local=compact[source];
                    if (local==Invalid)
                    {
                        local=unsigned(part.Vertices.size()); part.SourceVertices.push_back(source); part.Vertices.push_back(vertices[source]);
                        const auto p=vertices[source].Position; part.Bounds.Expand({p.X,p.Y,p.Z});
                    }
                    part.Indices.push_back(local);
                }
            }
            for (const auto source:part.SourceVertices) compact[source]=Invalid;
        }
        output=std::move(result); error.clear(); return true;
    }
}
