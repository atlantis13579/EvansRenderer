#include "Renderer/VirtualGeometry/VirtualGeometryHierarchy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>

namespace Evans::Renderer
{
    // Region boundaries stay fixed at every reduction. Therefore two unrelated
    // nodes in a cut retain the same source edges along their shared boundary.
    // A parent bounds the source-to-child and child-to-parent correspondences by
    // the triangle inequality. Endpoint QEM ranks contractions; its fitting cost
    // is not the displacement bound (Garland/Heckbert, SIGGRAPH 1997, see
    // LevelOfDetail.cpp). This forest and page protocol are independently built.
    namespace
    {
        constexpr std::uint32_t MaximumChildren=8,MaximumDepth=32,TriangleCorners=3;
        constexpr double HalfTurnDegrees=180;
        constexpr float ResidentDemandPriority=1;
        constexpr float TargetDemandPriority=2;
        constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        unsigned Word(VirtualClusterWord value) { return static_cast<unsigned>(value); }
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        void Include(Float3 lo,Float3 hi,Float3& minimum,Float3& maximum)
        {
            minimum={std::min(minimum.X,lo.X),std::min(minimum.Y,lo.Y),std::min(minimum.Z,lo.Z)};
            maximum={std::max(maximum.X,hi.X),std::max(maximum.Y,hi.Y),std::max(maximum.Z,hi.Z)};
        }
        bool Contains(const VirtualGeometryNode& node,Float3 lo,Float3 hi)
        {
            return lo.X>=node.BoundsMinimum.X && lo.Y>=node.BoundsMinimum.Y && lo.Z>=node.BoundsMinimum.Z &&
                hi.X<=node.BoundsMaximum.X && hi.Y<=node.BoundsMaximum.Y && hi.Z<=node.BoundsMaximum.Z;
        }
        std::uint64_t Signature(const VirtualGeometryHierarchy& h)
        {
            std::uint64_t hash=HashSeed;
            auto mix=[&](std::uint64_t value){hash^=value; hash*=HashPrime;};
            auto scalar=[&](double value){std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); mix(bits);};
            mix(h.Data.Signature); mix(h.Nodes.size());
            for (const auto& n:h.Nodes)
            {
                for (auto value:{n.Parent,n.Material,n.FirstChild,n.ChildCount,n.FirstCluster,n.ClusterCount,n.FirstPage,n.PageCount,
                    n.FirstTriangle,n.TriangleCount,n.RepresentationTriangles}) mix(value);
                for (double value:std::initializer_list<double>{n.BoundsMinimum.X,n.BoundsMinimum.Y,n.BoundsMinimum.Z,n.BoundsMaximum.X,n.BoundsMaximum.Y,n.BoundsMaximum.Z,n.MaximumDisplacement}) scalar(value);
            }
            for (const auto* values:{&h.Children,&h.Clusters,&h.Pages,&h.Roots,&h.SourceTriangles})
            { mix(values->size()); for (auto value:*values) mix(value); }
            return hash;
        }
        bool Metadata(const VirtualGeometryHierarchy& h,std::string& error)
        {
            auto invalid=[&](const char* message){error=message; return false;};
            if (h.Nodes.empty() || h.Nodes.size()>VirtualGeometryInvalid || h.Roots.empty() || h.SourceTriangles.size()!=h.Data.SourceTriangleCount ||
                h.Data.Pages.empty() || h.Data.Clusters.empty()) return invalid("Virtual hierarchy has empty or unaddressable dimensions");
            std::vector<bool> sourceSeen(h.SourceTriangles.size()),rootClusters(h.Data.Clusters.size());
            std::vector<std::uint32_t> owned(h.Data.Clusters.size()),parents(h.Nodes.size()),roots;
            for (auto key:h.SourceTriangles)
            { if (key>=sourceSeen.size() || sourceSeen[key]) return invalid("Virtual hierarchy source order is not a permutation"); sourceSeen[key]=true; }
            std::size_t childCursor=0,clusterCursor=0,pageCursor=0;
            for (std::uint32_t id=0;id<h.Nodes.size();++id)
            {
                const auto& n=h.Nodes[id];
                if ((n.Parent!=VirtualGeometryInvalid && (n.Parent<=id || n.Parent>=h.Nodes.size())) || !n.TriangleCount || !n.ClusterCount || !n.PageCount ||
                    std::uint64_t(n.FirstTriangle)+n.TriangleCount>h.SourceTriangles.size() || !std::isfinite(n.MaximumDisplacement) || n.MaximumDisplacement<0 ||
                    n.FirstChild!=childCursor || std::uint64_t(n.FirstChild)+n.ChildCount>h.Children.size() || n.ChildCount>MaximumChildren || n.ChildCount==1 ||
                    n.FirstCluster!=clusterCursor || std::uint64_t(n.FirstCluster)+n.ClusterCount>h.Clusters.size() ||
                    n.FirstPage!=pageCursor || std::uint64_t(n.FirstPage)+n.PageCount>h.Pages.size()) return invalid("Virtual hierarchy node spans, parent or error are invalid");
                for (float value:{n.BoundsMinimum.X,n.BoundsMinimum.Y,n.BoundsMinimum.Z,n.BoundsMaximum.X,n.BoundsMaximum.Y,n.BoundsMaximum.Z})
                    if (!std::isfinite(value)) return invalid("Virtual hierarchy bounds are not finite");
                if (n.BoundsMinimum.X>n.BoundsMaximum.X || n.BoundsMinimum.Y>n.BoundsMaximum.Y || n.BoundsMinimum.Z>n.BoundsMaximum.Z)
                    return invalid("Virtual hierarchy bounds are reversed");
                std::uint64_t childTriangles=0; std::uint32_t triangle=n.FirstTriangle; double childError=0;
                Float3 lo=n.BoundsMinimum,hi=n.BoundsMaximum; std::vector<std::uint32_t> childClusters;
                for (unsigned offset=0;offset<n.ChildCount;++offset)
                {
                    const auto child=h.Children[n.FirstChild+offset];
                    if (child>=id || h.Nodes[child].Parent!=id || ++parents[child]!=1) return invalid("Virtual hierarchy child ownership is invalid");
                    const auto& c=h.Nodes[child];
                    if (c.Material!=n.Material || c.FirstTriangle!=triangle) return invalid("Virtual hierarchy crosses materials or leaves a source-region gap");
                    triangle+=c.TriangleCount; childTriangles+=c.RepresentationTriangles; childError=std::max(childError,c.MaximumDisplacement);
                    if (!offset) { lo=c.BoundsMinimum; hi=c.BoundsMaximum; } else Include(c.BoundsMinimum,c.BoundsMaximum,lo,hi);
                    childClusters.insert(childClusters.end(),h.Clusters.begin()+c.FirstCluster,h.Clusters.begin()+c.FirstCluster+c.ClusterCount);
                }
                if (n.ChildCount && (triangle!=n.FirstTriangle+n.TriangleCount || !Same(lo,n.BoundsMinimum) || !Same(hi,n.BoundsMaximum) || n.MaximumDisplacement<childError))
                    return invalid("Virtual parent does not conservatively cover its children");
                std::vector<std::uint32_t> pages,refs,leafKeys; bool owns=true; std::uint64_t triangles=0;
                for (unsigned offset=0;offset<n.ClusterCount;++offset)
                {
                    const auto clusterId=h.Clusters[n.FirstCluster+offset];
                    if (clusterId>=h.Data.Clusters.size()) return invalid("Virtual node references a missing cluster");
                    const auto& c=h.Data.Clusters[clusterId];
                    if (c.Node>=h.Nodes.size() || c.Node>id || c.Material!=n.Material || c.Page>=h.Data.Pages.size() ||
                        std::uint64_t(c.Word)+VirtualGeometryClusterHeaderWords>VirtualGeometryPageWords || !Contains(n,c.BoundsMinimum,c.BoundsMaximum))
                        return invalid("Virtual cluster owner, material, page or region is inconsistent");
                    const auto& owner=h.Nodes[c.Node];
                    if (owner.FirstTriangle<n.FirstTriangle || std::uint64_t(owner.FirstTriangle)+owner.TriangleCount>std::uint64_t(n.FirstTriangle)+n.TriangleCount)
                        return invalid("Virtual node references another source region's representation");
                    owns=owns && c.Node==id; if (c.Node==id) ++owned[clusterId];
                    const auto* header=h.Data.Pages[c.Page].Words.data()+c.Word;
                    const auto indices=header[Word(VirtualClusterWord::IndexCount)];
                    if (!indices || indices%TriangleCorners || indices>VirtualGeometryClusterTriangles*TriangleCorners) return invalid("Virtual cluster triangle count is invalid");
                    triangles+=indices/TriangleCorners; pages.push_back(c.Page); refs.push_back(clusterId);
                    if (!n.ChildCount)
                    {
                        const auto first=header[Word(VirtualClusterWord::SourceTriangles)];
                        if (!(header[Word(VirtualClusterWord::Flags)]&VirtualGeometrySourceTriangleKeys) || std::uint64_t(first)+indices/TriangleCorners>h.Data.SourceTriangles.size())
                            return invalid("Virtual leaf has no complete source correspondence");
                        leafKeys.insert(leafKeys.end(),h.Data.SourceTriangles.begin()+first,h.Data.SourceTriangles.begin()+first+indices/TriangleCorners);
                    }
                    if (n.Parent==VirtualGeometryInvalid) rootClusters[clusterId]=true;
                }
                auto sorted=refs; std::sort(sorted.begin(),sorted.end());
                if (std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end() || triangles!=n.RepresentationTriangles || triangles>n.TriangleCount)
                    return invalid("Virtual node repeats a cluster or misstates its triangle budget");
                if (!n.ChildCount)
                {
                    if (!owns || n.MaximumDisplacement!=0 || leafKeys.size()!=n.TriangleCount ||
                        !std::equal(leafKeys.begin(),leafKeys.end(),h.SourceTriangles.begin()+n.FirstTriangle)) return invalid("Virtual leaf is not lossless in its owned region");
                }
                else if (owns)
                { if (triangles>=childTriangles) return invalid("A new parent representation did not reduce its children"); }
                else if (refs!=childClusters || triangles!=childTriangles || n.MaximumDisplacement!=childError)
                    return invalid("An unchanged parent does not exactly reuse its child representation");
                std::sort(pages.begin(),pages.end()); pages.erase(std::unique(pages.begin(),pages.end()),pages.end());
                if (pages.size()!=n.PageCount || !std::equal(pages.begin(),pages.end(),h.Pages.begin()+n.FirstPage)) return invalid("Virtual node page bundle differs from its clusters");
                childCursor+=n.ChildCount; clusterCursor+=n.ClusterCount; pageCursor+=n.PageCount;
                if (n.Parent==VirtualGeometryInvalid) roots.push_back(id);
            }
            if (childCursor!=h.Children.size() || clusterCursor!=h.Clusters.size() || pageCursor!=h.Pages.size() ||
                std::find_if(owned.begin(),owned.end(),[](auto count){return count!=1;})!=owned.end()) return invalid("Virtual hierarchy has unowned metadata or payloads");
            std::sort(roots.begin(),roots.end(),[&](auto a,auto b){return h.Nodes[a].FirstTriangle<h.Nodes[b].FirstTriangle;});
            if (roots!=h.Roots) return invalid("Virtual root order or ownership is invalid");
            std::uint32_t triangle=0;
            for (auto root:roots) { if (h.Nodes[root].FirstTriangle!=triangle) return invalid("Virtual roots overlap or leave a coverage gap"); triangle+=h.Nodes[root].TriangleCount; }
            if (triangle!=h.SourceTriangles.size()) return invalid("Virtual roots do not cover the original source");
            for (std::size_t id=0;id<h.Nodes.size();++id) if (parents[id]!=(h.Nodes[id].Parent==VirtualGeometryInvalid?0u:1u)) return invalid("Virtual parent lacks a reciprocal child link");
            for (std::size_t id=0;id<h.Data.Clusters.size();++id)
            {
                const auto& c=h.Data.Clusters[id];
                if (((h.Data.Pages[c.Page].Words[4]&VirtualGeometryPinnedPage)!=0)!=rootClusters[id]) return invalid("Root and evictable virtual payloads share the wrong page class");
            }
            if (h.Signature!=Signature(h)) return invalid("Virtual hierarchy signature is stale");
            error.clear(); return true;
        }
        bool SettingsValid(const VirtualGeometryHierarchySettings& s)
        {
            const auto& q=s.Simplification;
            return s.ChildrenPerNode>=2 && s.ChildrenPerNode<=MaximumChildren && s.MaximumDepth && s.MaximumDepth<=MaximumDepth && s.MaximumNodes &&
                s.MaximumNodes<=VirtualGeometryInvalid && q.MinimumTriangles && std::isfinite(q.TriangleRatio) && q.TriangleRatio>0 && q.TriangleRatio<1 &&
                std::isfinite(q.MaximumNormalAngleDegrees) && q.MaximumNormalAngleDegrees>=0 && q.MaximumNormalAngleDegrees<=HalfTurnDegrees &&
                std::isfinite(q.MaximumTexCoordDisplacement) && q.MaximumTexCoordDisplacement>=0 && std::isfinite(q.MaximumColorDisplacement) && q.MaximumColorDisplacement>=0;
        }
        bool Simplify(const std::vector<Vertex>& source,const std::vector<std::uint32_t>& indices,const std::vector<bool>& fixed,
            const LodBuildSettings& settings,std::vector<std::uint32_t>& result,double& displacement,std::string& error)
        {
            auto keys=indices; std::sort(keys.begin(),keys.end()); keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
            std::vector<Vertex> vertices; std::vector<std::uint32_t> local,locked;
            for (std::uint32_t id=0;id<keys.size();++id) { vertices.push_back(source[keys[id]]); if (fixed[keys[id]]) locked.push_back(id); }
            for (auto key:indices) local.push_back(static_cast<std::uint32_t>(std::lower_bound(keys.begin(),keys.end(),key)-keys.begin()));
            auto step=settings; step.MaximumLevels=2; MeshLodChain chain;
            if (!BuildMeshLods(vertices,local,step,chain,error,locked)) return false;
            const auto& level=chain.Levels.back(); displacement=level.MaximumDisplacement; result.reserve(level.Indices.size());
            for (auto index:level.Indices) result.push_back(keys[index]);
            return true;
        }
    }

    bool BuildVirtualGeometryHierarchy(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const std::vector<std::uint32_t>& materials,const VirtualGeometryHierarchySettings& settings,VirtualGeometryHierarchy& output,std::string& error)
    {
        if (!SettingsValid(settings)) { error="Virtual hierarchy grouping or simplification settings are invalid"; return false; }
        VirtualGeometryPages leaves;
        if (!BuildVirtualGeometryPages(vertices,indices,materials,settings.Pages,leaves,error)) return false;
        if (leaves.Clusters.size()>settings.MaximumNodes) { error="Virtual leaf count exceeds the node budget"; return false; }
        VirtualGeometryHierarchy h; std::vector<VirtualGeometryClusterInput> inputs;
        std::vector<std::vector<std::uint32_t>> geometry,references; std::vector<std::uint32_t> level;
        for (std::uint32_t cluster=0;cluster<leaves.Clusters.size();++cluster)
        {
            VirtualGeometryDecodedCluster decoded; if (!DecodeVirtualGeometryCluster(leaves,cluster,decoded,error)) return false;
            const auto id=static_cast<std::uint32_t>(h.Nodes.size()); VirtualGeometryClusterInput input; input.Material=decoded.Material; input.Node=id; input.SourceTriangles=decoded.SourceTriangles;
            for (auto index:decoded.Indices) input.Indices.push_back(decoded.SourceVertices[index]);
            VirtualGeometryNode n; n.Material=input.Material; n.FirstTriangle=static_cast<std::uint32_t>(h.SourceTriangles.size());
            n.TriangleCount=n.RepresentationTriangles=static_cast<std::uint32_t>(input.SourceTriangles.size()); n.BoundsMinimum=leaves.Clusters[cluster].BoundsMinimum; n.BoundsMaximum=leaves.Clusters[cluster].BoundsMaximum;
            h.SourceTriangles.insert(h.SourceTriangles.end(),input.SourceTriangles.begin(),input.SourceTriangles.end()); h.Nodes.push_back(n);
            geometry.push_back(input.Indices); references.push_back({static_cast<std::uint32_t>(inputs.size())}); inputs.push_back(std::move(input)); level.push_back(id);
        }
        // Local compaction must retain seam locks known only to other regions.
        std::vector<std::uint32_t> positions(vertices.size()); std::iota(positions.begin(),positions.end(),0); std::vector<bool> fixed(vertices.size());
        auto key=[&](auto id){const auto& p=vertices[id].Position; return std::make_tuple(p.X,p.Y,p.Z);};
        std::sort(positions.begin(),positions.end(),[&](auto a,auto b){return key(a)!=key(b)?key(a)<key(b):a<b;});
        for (std::size_t begin=0;begin<positions.size();)
        {
            std::size_t end=begin+1; while (end<positions.size() && key(positions[begin])==key(positions[end])) ++end;
            if (end-begin>1) for (auto at=begin;at<end;++at) fixed[positions[at]]=true;
            begin=end;
        }
        for (unsigned depth=1;depth<settings.MaximumDepth && level.size()>1;++depth)
        {
            std::vector<std::uint32_t> next; bool combined=false;
            for (std::size_t begin=0;begin<level.size();)
            {
                std::size_t end=begin+1;
                while (end<level.size() && end-begin<settings.ChildrenPerNode && h.Nodes[level[end]].Material==h.Nodes[level[begin]].Material) ++end;
                if (end-begin==1) { next.push_back(level[begin]); begin=end; continue; }
                if (h.Nodes.size()>=settings.MaximumNodes) { error="Virtual hierarchy exceeds the node budget"; return false; }
                const auto id=static_cast<std::uint32_t>(h.Nodes.size()); const auto& first=h.Nodes[level[begin]];
                VirtualGeometryNode n; n.Material=first.Material; n.FirstTriangle=first.FirstTriangle; n.BoundsMinimum=first.BoundsMinimum; n.BoundsMaximum=first.BoundsMaximum;
                n.FirstChild=static_cast<std::uint32_t>(h.Children.size()); n.ChildCount=static_cast<std::uint32_t>(end-begin);
                std::vector<std::uint32_t> joined,refs;
                for (auto at=begin;at<end;++at)
                {
                    const auto child=level[at]; auto& c=h.Nodes[child]; c.Parent=id; h.Children.push_back(child); n.TriangleCount+=c.TriangleCount;
                    n.MaximumDisplacement=std::max(n.MaximumDisplacement,c.MaximumDisplacement); Include(c.BoundsMinimum,c.BoundsMaximum,n.BoundsMinimum,n.BoundsMaximum);
                    joined.insert(joined.end(),geometry[child].begin(),geometry[child].end()); refs.insert(refs.end(),references[child].begin(),references[child].end());
                    std::vector<std::uint32_t>().swap(geometry[child]);
                }
                std::vector<std::uint32_t> reduced; double displacement=0;
                if (!Simplify(vertices,joined,fixed,settings.Simplification,reduced,displacement,error)) return false;
                if (reduced.size()<joined.size())
                {
                    refs.clear(); n.MaximumDisplacement+=displacement;
                    for (std::size_t start=0;start<reduced.size();start+=VirtualGeometryClusterTriangles*TriangleCorners)
                    {
                        if (inputs.size()>=settings.Pages.MaximumClusters) { error="Virtual parent payloads exceed the cluster budget"; return false; }
                        VirtualGeometryClusterInput input; input.Material=n.Material; input.Node=id;
                        const auto stop=std::min(reduced.size(),start+VirtualGeometryClusterTriangles*TriangleCorners);
                        input.Indices.assign(reduced.begin()+start,reduced.begin()+stop); refs.push_back(static_cast<std::uint32_t>(inputs.size())); inputs.push_back(std::move(input));
                    }
                }
                n.RepresentationTriangles=static_cast<std::uint32_t>(reduced.size()/TriangleCorners); h.Nodes.push_back(n);
                geometry.push_back(std::move(reduced)); references.push_back(std::move(refs)); next.push_back(id); combined=true; begin=end;
            }
            level=std::move(next); if (!combined) break;
        }
        h.Roots=std::move(level);
        for (auto root:h.Roots) for (auto cluster:references[root]) inputs[cluster].Pinned=true;
        std::vector<VirtualGeometryClusterInput> ordered; std::vector<std::uint32_t> remap(inputs.size()); ordered.reserve(inputs.size());
        for (bool pinned:{true,false}) for (std::size_t id=0;id<inputs.size();++id) if (inputs[id].Pinned==pinned)
        { remap[id]=static_cast<std::uint32_t>(ordered.size()); ordered.push_back(std::move(inputs[id])); }
        if (!EncodeVirtualGeometryPages(vertices,static_cast<std::uint32_t>(indices.size()/TriangleCorners),ordered,settings.Pages,h.Data,error)) return false;
        for (std::size_t id=0;id<h.Nodes.size();++id)
        {
            auto& n=h.Nodes[id]; n.FirstCluster=static_cast<std::uint32_t>(h.Clusters.size()); n.ClusterCount=static_cast<std::uint32_t>(references[id].size());
            std::vector<std::uint32_t> pages;
            for (auto cluster:references[id]) { const auto mapped=remap[cluster]; h.Clusters.push_back(mapped); pages.push_back(h.Data.Clusters[mapped].Page); }
            std::sort(pages.begin(),pages.end()); pages.erase(std::unique(pages.begin(),pages.end()),pages.end());
            n.FirstPage=static_cast<std::uint32_t>(h.Pages.size()); n.PageCount=static_cast<std::uint32_t>(pages.size()); h.Pages.insert(h.Pages.end(),pages.begin(),pages.end());
        }
        h.Signature=Signature(h); if (!ValidateVirtualGeometryHierarchy(h,error)) return false;
        output=std::move(h); error.clear(); return true;
    }

    bool ValidateVirtualGeometryHierarchy(const VirtualGeometryHierarchy& hierarchy,std::string& error)
    { return ValidateVirtualGeometryPages(hierarchy.Data,error) && Metadata(hierarchy,error); }

    std::vector<std::uint32_t> VirtualGeometryPinnedPages(const VirtualGeometryHierarchy& hierarchy)
    {
        std::vector<std::uint32_t> pages;
        for (std::uint32_t page=0;page<hierarchy.Data.Pages.size();++page) if (hierarchy.Data.Pages[page].Words[4]&VirtualGeometryPinnedPage) pages.push_back(page);
        return pages;
    }

    bool AdmitVirtualGeometryTransitions(const VirtualGeometryHierarchy& h,const VirtualGeometryResidency& residency,
        VirtualGeometryCut cut,const std::vector<VirtualGeometryTransition>& requested,VirtualGeometryCut& output,std::string& error)
    {
        if (!Metadata(h,error) || !ValidateVirtualGeometryResidency(residency,error)) return false;
        if (cut.Source!=h.Signature || residency.Source()!=h.Signature || cut.CacheEpoch!=residency.Epoch() || cut.CacheRevision!=residency.Revision() ||
            residency.PageSlots().size()!=h.Data.Pages.size() || residency.PinnedPages()!=VirtualGeometryPinnedPages(h))
        { error="Virtual transition feedback belongs to another source/cache revision"; return false; }
        std::vector<std::uint32_t> clusters; std::uint32_t covered=0;
        std::vector<float> priorities(h.Data.Pages.size());
        for (auto id:cut.Nodes)
        {
            if (id>=h.Nodes.size() || h.Nodes[id].FirstTriangle!=covered) { error="Virtual feedback cut overlaps or leaves a gap"; return false; }
            const auto& n=h.Nodes[id]; covered+=n.TriangleCount;
            clusters.insert(clusters.end(),h.Clusters.begin()+n.FirstCluster,h.Clusters.begin()+n.FirstCluster+n.ClusterCount);
            for (unsigned at=0;at<n.PageCount;++at)
            {
                const auto page=h.Pages[n.FirstPage+at]; if (residency.PageSlots()[page]==VirtualGeometryInvalid) { error="Virtual feedback selected a missing page"; return false; }
                priorities[page]=ResidentDemandPriority;
            }
        }
        if (covered!=h.SourceTriangles.size() || clusters!=cut.Clusters) { error="Virtual feedback lacks complete cluster/source coverage"; return false; }
        auto transitions=requested;
        for (const auto& t:transitions)
            if (t.Node>=h.Nodes.size() || !h.Nodes[t.Node].ChildCount || !std::isfinite(t.Priority) || t.Priority<0)
            { error="Virtual feedback transition has an invalid node or priority"; return false; }
        std::sort(transitions.begin(),transitions.end(),[&](const auto& a,const auto& b){return h.Nodes[a.Node].FirstTriangle<h.Nodes[b.Node].FirstTriangle;});
        std::uint32_t transitionEnd=0;
        for (const auto& t:transitions)
        {
            const auto& n=h.Nodes[t.Node]; if (n.FirstTriangle<transitionEnd) { error="Virtual feedback transitions overlap"; return false; } transitionEnd=n.FirstTriangle+n.TriangleCount;
            const auto first=std::lower_bound(cut.Nodes.begin(),cut.Nodes.end(),n.FirstTriangle,[&](auto node,auto triangle){return h.Nodes[node].FirstTriangle<triangle;});
            const auto last=std::lower_bound(first,cut.Nodes.end(),transitionEnd,[&](auto node,auto triangle){return h.Nodes[node].FirstTriangle<triangle;});
            if (first==last || h.Nodes[*first].FirstTriangle!=n.FirstTriangle || h.Nodes[*(last-1)].FirstTriangle+h.Nodes[*(last-1)].TriangleCount!=transitionEnd ||
                (t.Refine?last-first!=1 || *first!=t.Node:last-first==1 && *first==t.Node))
            { error="Virtual feedback transition does not replace its complete current region"; return false; }
        }
        cut.TargetNodes.clear(); cut.TargetPages.clear(); cut.Requests.clear(); cut.BudgetBlockedNodes=0;
        // Admit replacements against a coherent complete cut, not independent
        // page priorities. Otherwise one branch's refinement could evict another
        // branch's only fine coverage and repeatedly bounce back to the root.
        std::vector<std::uint32_t> references(h.Data.Pages.size());
        std::size_t working=0;
        auto addNode=[&](std::uint32_t id)
        { const auto& n=h.Nodes[id]; for (unsigned at=0;at<n.PageCount;++at) working+=references[h.Pages[n.FirstPage+at]]++==0; };
        for (auto page:residency.PinnedPages()) { ++references[page]; ++working; }
        for (auto id:cut.Nodes) addNode(id);
        std::sort(transitions.begin(),transitions.end(),[&](const auto& a,const auto& b)
        { return a.Priority!=b.Priority?a.Priority>b.Priority:h.Nodes[a.Node].FirstTriangle<h.Nodes[b.Node].FirstTriangle; });
        struct Replacement { std::size_t Begin,End; std::vector<std::uint32_t> Nodes; };
        std::vector<Replacement> replacements;
        for (const auto& transition:transitions)
        {
            const auto& region=h.Nodes[transition.Node];
            auto first=std::lower_bound(cut.Nodes.begin(),cut.Nodes.end(),region.FirstTriangle,[&](auto node,auto triangle){return h.Nodes[node].FirstTriangle<triangle;});
            auto last=std::lower_bound(first,cut.Nodes.end(),region.FirstTriangle+region.TriangleCount,[&](auto node,auto triangle){return h.Nodes[node].FirstTriangle<triangle;});
            Replacement replacement{static_cast<std::size_t>(first-cut.Nodes.begin()),static_cast<std::size_t>(last-cut.Nodes.begin()),{}};
            if (transition.Refine) replacement.Nodes.assign(h.Children.begin()+region.FirstChild,h.Children.begin()+region.FirstChild+region.ChildCount);
            else replacement.Nodes.push_back(transition.Node);
            std::vector<std::pair<std::uint32_t,int>> changes;
            auto change=[&](auto node,int delta)
            { const auto& n=h.Nodes[node]; for (unsigned at=0;at<n.PageCount;++at) changes.emplace_back(h.Pages[n.FirstPage+at],delta); };
            for (auto it=first;it!=last;++it) change(*it,-1);
            for (auto node:replacement.Nodes) change(node,1);
            std::sort(changes.begin(),changes.end()); std::vector<std::pair<std::uint32_t,int>> deltas;
            std::int64_t nextWorking=static_cast<std::int64_t>(working);
            for (std::size_t begin=0;begin<changes.size();)
            {
                std::size_t end=begin; int delta=0; const auto page=changes[begin].first;
                while (end<changes.size() && changes[end].first==page) delta+=changes[end++].second;
                const auto before=references[page]; const auto after=std::int64_t(before)+delta;
                if (after<0) { error="Virtual target transitions overlap their source coverage"; return false; }
                nextWorking+=(after>0)-(before>0); deltas.emplace_back(page,delta); begin=end;
            }
            if (nextWorking>static_cast<std::int64_t>(residency.Slots().size())) { ++cut.BudgetBlockedNodes; continue; }
            for (const auto& delta:deltas) references[delta.first]=static_cast<std::uint32_t>(std::int64_t(references[delta.first])+delta.second);
            working=static_cast<std::size_t>(nextWorking); replacements.push_back(std::move(replacement));
        }
        std::sort(replacements.begin(),replacements.end(),[](const auto& a,const auto& b){return a.Begin<b.Begin;});
        std::size_t read=0;
        for (const auto& replacement:replacements)
        {
            if (replacement.Begin<read) { error="Virtual target replacements overlap"; return false; }
            cut.TargetNodes.insert(cut.TargetNodes.end(),cut.Nodes.begin()+read,cut.Nodes.begin()+replacement.Begin);
            cut.TargetNodes.insert(cut.TargetNodes.end(),replacement.Nodes.begin(),replacement.Nodes.end()); read=replacement.End;
        }
        cut.TargetNodes.insert(cut.TargetNodes.end(),cut.Nodes.begin()+read,cut.Nodes.end());
        for (std::uint32_t page=0;page<references.size();++page) if (references[page]) { cut.TargetPages.push_back(page); priorities[page]=TargetDemandPriority; }
        for (std::uint32_t page=0;page<priorities.size();++page) if (priorities[page]>0) cut.Requests.push_back({page,priorities[page]});
        output=std::move(cut); error.clear(); return true;
    }

    bool PlanVirtualGeometryCutResidency(const VirtualGeometryHierarchy& h,const VirtualGeometryResidency& residency,
        const VirtualGeometryCut& cut,std::uint64_t frame,VirtualGeometryResidencyPlan& output,std::string& error)
    {
        if (!Metadata(h,error) || !ValidateVirtualGeometryResidency(residency,error)) return false;
        if (cut.Source!=h.Signature || residency.Source()!=h.Signature || cut.CacheEpoch!=residency.Epoch() || cut.CacheRevision!=residency.Revision() ||
            residency.PageSlots().size()!=h.Data.Pages.size() || residency.PinnedPages()!=VirtualGeometryPinnedPages(h))
        { error="Virtual target cut is stale or belongs to another hierarchy/cache"; return false; }
        std::vector<std::uint32_t> pages=residency.PinnedPages(); std::uint32_t first=0;
        for (auto id:cut.TargetNodes)
        {
            if (id>=h.Nodes.size() || h.Nodes[id].FirstTriangle!=first) { error="Virtual target cut overlaps or leaves a source gap"; return false; }
            const auto& n=h.Nodes[id]; first+=n.TriangleCount; pages.insert(pages.end(),h.Pages.begin()+n.FirstPage,h.Pages.begin()+n.FirstPage+n.PageCount);
        }
        std::sort(pages.begin(),pages.end()); pages.erase(std::unique(pages.begin(),pages.end()),pages.end());
        if (first!=h.SourceTriangles.size() || pages!=cut.TargetPages || pages.size()>residency.Slots().size())
        { error="Virtual target pages do not describe a complete admitted cut"; return false; }
        std::vector<VirtualGeometryPageRequest> requests;
        for (auto page:pages) requests.push_back({page,TargetDemandPriority});
        // Current resident demand is lower priority than the complete target.
        // Native execution retains the old pool until this whole plan completes.
        first=0;
        for (auto id:cut.Nodes)
        {
            if (id>=h.Nodes.size() || h.Nodes[id].FirstTriangle!=first) { error="Virtual current cut overlaps or leaves a source gap"; return false; }
            const auto& n=h.Nodes[id]; first+=n.TriangleCount;
            for (unsigned at=0;at<n.PageCount;++at)
            {
                const auto page=h.Pages[n.FirstPage+at];
                if (residency.PageSlots()[page]==VirtualGeometryInvalid) { error="Virtual current cut uses a missing page"; return false; }
                requests.push_back({page,ResidentDemandPriority});
            }
        }
        if (first!=h.SourceTriangles.size()) { error="Virtual current cut lacks complete source coverage"; return false; }
        VirtualGeometryResidencyPlan candidate;
        if (!PlanVirtualGeometryResidency(residency,requests,frame,std::numeric_limits<std::uint32_t>::max(),candidate,error)) return false;
        for (auto page:pages) if (candidate.Next().PageSlots()[page]==VirtualGeometryInvalid)
        { error="Virtual residency planner omitted an admitted target page"; return false; }
        output=std::move(candidate); error.clear(); return true;
    }
}
