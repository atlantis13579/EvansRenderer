#include "Renderer/Nanite/NaniteHierarchy.h"
#include "Renderer/Geometry/GraphPartition.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace Evans::Renderer
{
    /* The Nanite build replaces a group of fine clusters with several newly
       partitioned coarse clusters. Their common generating group owns the LOD
       error. A later group may consume only some of those outputs, producing a
       DAG with shared dependencies. Keeping the old partition at every level
       would progressively lock the simplifier against its original boundaries.

       Weighted cuts use the project-local Riemann/METIS multilevel partitioner.
       Material regions are independent graphs. Oversized partitions are split
       again to enforce triangle, vertex and group capacities. Spatial ordering
       gives disconnected islands a deterministic input order. See
       GraphPartition.cpp for the algorithm, provenance and license references.
       Endpoint QEM reuses the project's topology/attribute safeguards; its
       displacement radius, not the quadric cost, bounds the approximation.
       Group error is max(child error) + local displacement. Inherited bounds
       include entire generating groups, so every output makes the same decision.
       See Karis et al., SIGGRAPH 2021, slides 33-66:
       https://www.wihlidal.com/projects/nanite-deepdive/
       QEM and correspondence bounds are documented in LevelOfDetail.cpp.
    */
    namespace
    {
        constexpr unsigned Corners=3,MaximumClusterTriangles=256,MaximumGroupClusters=32,MaximumLevels=32;
        constexpr double HalfTurnDegrees=180,Pi=3.14159265358979323846;
        constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        using Point=std::array<double,3>;
        using Edge=std::pair<std::uint32_t,std::uint32_t>;
        using EdgeCounts=std::map<Edge,std::pair<unsigned,std::int64_t>>;
        using Graph=std::vector<std::map<std::uint32_t,unsigned>>;
        struct Element
        {
            const std::vector<std::uint32_t>* Indices=nullptr;
            std::size_t First=0,Count=0;
            std::uint32_t Material=0;
            Point Center{};
        };
        Element MakeElement(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
            std::size_t first,std::size_t count,std::uint32_t material)
        {
            Element result{&indices,first,count,material,{}};
            for (auto at=first;at<first+count;++at)
            {
                const auto p=vertices[indices[at]].Position;
                result.Center[0]+=p.X; result.Center[1]+=p.Y; result.Center[2]+=p.Z;
            }
            for (auto& value:result.Center) value/=double(count);
            return result;
        }
        void Include(Float3 lo,Float3 hi,Float3& minimum,Float3& maximum)
        {
            minimum={std::min(minimum.X,lo.X),std::min(minimum.Y,lo.Y),std::min(minimum.Z,lo.Z)};
            maximum={std::max(maximum.X,hi.X),std::max(maximum.Y,hi.Y),std::max(maximum.Z,hi.Z)};
        }
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        void Bound(const std::vector<Vertex>& vertices,NaniteCluster& cluster)
        {
            cluster.BoundsMinimum=cluster.BoundsMaximum=vertices[cluster.Indices.front()].Position;
            for (auto index:cluster.Indices) Include(vertices[index].Position,vertices[index].Position,cluster.BoundsMinimum,cluster.BoundsMaximum);
        }
        std::array<float,9> BoundaryKey(const Vertex& v)
        { return {v.Position.X,v.Position.Y,v.Position.Z,v.TexCoord.X,v.TexCoord.Y,v.Color.X,v.Color.Y,v.Color.Z,v.Color.W}; }
        std::vector<std::uint32_t> VertexIds(const std::vector<Vertex>& vertices,bool attributes=false)
        {
            std::vector<std::uint32_t> order(vertices.size()),ids(vertices.size()); std::iota(order.begin(),order.end(),0);
            const auto key=[&](auto id) { const auto& v=vertices[id]; return attributes?BoundaryKey(v):std::array<float,9>{v.Position.X,v.Position.Y,v.Position.Z}; };
            std::sort(order.begin(),order.end(),[&](auto a,auto b) { return key(a)!=key(b)?key(a)<key(b):a<b; });
            for (std::size_t begin=0;begin<order.size();)
            {
                auto end=begin+1; while (end<order.size() && key(order[begin])==key(order[end])) ++end;
                for (auto at=begin;at<end;++at) ids[order[at]]=order[begin];
                begin=end;
            }
            return ids;
        }
        Point UnitNormal(const Vertex& vertex)
        {
            Point n{vertex.Normal.X,vertex.Normal.Y,vertex.Normal.Z}; const auto length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (length>0) for (auto& value:n) value/=length;
            return n;
        }
        double Distance(Point a,Point b)
        { double squared=0; for (unsigned axis=0;axis<Corners;++axis) { const auto delta=a[axis]-b[axis]; squared+=delta*delta; } return std::sqrt(squared); }
        void WeldNormalFans(const std::vector<Vertex>& vertices,const std::vector<unsigned>& positions,double maximumAngle,
            std::vector<unsigned>& representatives,std::vector<double>& radii,std::vector<bool>& fixed)
        {
            std::vector<unsigned> order(vertices.size()); std::iota(order.begin(),order.end(),0); representatives=order; radii.resize(vertices.size());
            std::sort(order.begin(),order.end(),[&](auto a,auto b) { return std::tie(positions[a],a)<std::tie(positions[b],b); });
            const double limit=2*std::sin(maximumAngle*Pi/HalfTurnDegrees/2);
            for (std::size_t begin=0;begin<order.size();)
            {
                auto end=begin+1; while (end<order.size() && positions[order[end]]==positions[order[begin]]) ++end;
                Point mean{}; bool compatible=true;
                for (auto at=begin;at<end;++at)
                {
                    const auto& vertex=vertices[order[at]]; const auto normal=UnitNormal(vertex);
                    compatible=compatible && BoundaryKey(vertex)==BoundaryKey(vertices[order[begin]]) && normal!=Point{};
                    for (unsigned axis=0;axis<Corners;++axis) mean[axis]+=normal[axis]/double(end-begin);
                }
                auto representative=order[begin]; double closest=std::numeric_limits<double>::infinity();
                for (auto at=begin;at<end;++at)
                {
                    const auto candidate=order[at]; const double distance=Distance(UnitNormal(vertices[candidate]),mean);
                    if (distance<closest) { closest=distance; representative=candidate; }
                }
                double radius=0;
                for (auto at=begin;at<end;++at) radius=std::max(radius,Distance(UnitNormal(vertices[representative]),UnitNormal(vertices[order[at]])));
                if (radius>0) radius=std::nextafter(radius,std::numeric_limits<double>::infinity());
                compatible=compatible && radius<=limit;
                // Leaves retain exact corner normals. Coarse construction can
                // weld a smooth fan to an existing normal, carrying its error
                // into QEM. UV/color seams and incompatible normals stay fixed.
                for (auto at=begin;at<end;++at)
                    if (compatible) representatives[order[at]]=representative; else fixed[order[at]]=true;
                if (compatible) radii[representative]=radius;
                begin=end;
            }
        }
        struct EdgeUse { std::uint32_t Material,A,B,Element; };
        Graph Adjacency(const std::vector<Element>& elements,const std::vector<std::uint32_t>& positions)
        {
            std::vector<EdgeUse> edges; std::size_t corners=0;
            for (const auto& e:elements) corners+=e.Count;
            edges.reserve(corners);
            for (std::uint32_t id=0;id<elements.size();++id)
            {
                const auto& e=elements[id]; const auto& indices=*e.Indices;
                for (auto face=e.First;face<e.First+e.Count;face+=Corners) for (unsigned corner=0;corner<Corners;++corner)
                {
                    const auto a=positions[indices[face+corner]],b=positions[indices[face+(corner+1)%Corners]];
                    if (a!=b) edges.push_back({e.Material,std::min(a,b),std::max(a,b),id});
                }
            }
            const auto key=[](const EdgeUse& e) { return std::tie(e.Material,e.A,e.B,e.Element); };
            std::sort(edges.begin(),edges.end(),[&](const auto& a,const auto& b) { return key(a)<key(b); });
            Graph result(elements.size());
            for (std::size_t begin=0;begin<edges.size();)
            {
                auto end=begin+1;
                while (end<edges.size() && std::tie(edges[begin].Material,edges[begin].A,edges[begin].B)==std::tie(edges[end].Material,edges[end].A,edges[end].B)) ++end;
                // Non-manifold fans are not expanded into quadratic cliques.
                if (end-begin==2 && edges[begin].Element!=edges[begin+1].Element)
                {
                    const auto a=edges[begin].Element,b=edges[begin+1].Element;
                    ++result[a][b]; ++result[b][a];
                }
                begin=end;
            }
            return result;
        }
        bool Partition(const std::vector<Element>& elements,const std::vector<std::uint32_t>& positions,
            unsigned maximumItems,unsigned maximumVertices,std::vector<std::vector<std::uint32_t>>& result,std::string& error)
        {
            const auto graph=Adjacency(elements,positions);
            std::vector<std::uint32_t> order(elements.size()),local(elements.size(),NaniteInvalid); std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](auto a,auto b) {
                return std::tie(elements[a].Material,elements[a].Center,a)<std::tie(elements[b].Material,elements[b].Center,b); });
            std::vector<std::vector<std::uint32_t>> pending;
            for (std::size_t begin=0;begin<order.size();)
            {
                auto end=begin+1; while (end<order.size() && elements[order[end]].Material==elements[order[begin]].Material) ++end;
                pending.emplace_back(order.begin()+begin,order.begin()+end); begin=end;
            }
            while (!pending.empty())
            {
                auto nodes=std::move(pending.back()); pending.pop_back(); bool fits=nodes.size()<=maximumItems;
                if (fits && maximumVertices)
                {
                    std::set<std::uint32_t> vertices;
                    for (auto id:nodes)
                    {
                        const auto& e=elements[id]; vertices.insert(e.Indices->begin()+e.First,e.Indices->begin()+e.First+e.Count);
                        if (vertices.size()>maximumVertices) { fits=false; break; }
                    }
                }
                if (fits) { std::sort(nodes.begin(),nodes.end()); result.push_back(std::move(nodes)); continue; }
                if (nodes.size()<2) { error="A Nanite graph element exceeds its payload capacity"; return false; }
                for (unsigned index=0;index<nodes.size();++index) local[nodes[index]]=index;
                PartitionGraph csr; csr.Offsets.push_back(0);
                for (auto id:nodes)
                {
                    std::vector<std::pair<int,int>> neighbors;
                    for (const auto& edge:graph[id]) if (local[edge.first]!=NaniteInvalid)
                    {
                        if (edge.second>unsigned(std::numeric_limits<int>::max())) { error="Nanite adjacency weight overflow"; return false; }
                        neighbors.emplace_back(int(local[edge.first]),int(edge.second));
                    }
                    std::sort(neighbors.begin(),neighbors.end());
                    for (const auto& edge:neighbors) { csr.Neighbors.push_back(edge.first); csr.Weights.push_back(edge.second); }
                    csr.Offsets.push_back(int(csr.Neighbors.size()));
                }
                for (auto id:nodes) local[id]=NaniteInvalid;
                const auto count=unsigned(std::max(std::size_t(2),(nodes.size()+maximumItems-1)/maximumItems));
                GraphPartitionResult split;
                if (!PartitionWeightedGraph(csr,count,split,error)) return false;
                // Every nonempty child is strictly smaller, so even vertex
                // seams or a disconnected graph cannot prevent termination.
                for (const auto& part:split.Parts)
                {
                    std::vector<unsigned> child; child.reserve(part.size());
                    for (auto index:part) child.push_back(nodes[index]);
                    pending.push_back(std::move(child));
                }
            }
            std::sort(result.begin(),result.end()); return true;
        }
        std::vector<Element> Triangles(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
            const std::vector<std::uint32_t>& materials,unsigned material=0)
        {
            std::vector<Element> result; result.reserve(indices.size()/Corners);
            for (std::size_t face=0;face<indices.size()/Corners;++face)
                result.push_back(MakeElement(vertices,indices,face*Corners,Corners,materials.empty()?material:materials[face]));
            return result;
        }
        void AddBoundary(EdgeCounts& edges,const std::vector<std::uint32_t>& indices,const std::vector<unsigned>& identities)
        {
            for (std::size_t face=0;face<indices.size();face+=Corners) for (unsigned corner=0;corner<Corners;++corner)
            {
                const auto a=identities[indices[face+corner]],b=identities[indices[face+(corner+1)%Corners]]; auto& counts=edges[std::minmax(a,b)];
                ++counts.first; counts.second+=a<b?1:-1;
            }
        }
        void RemoveInterior(EdgeCounts& edges)
        {
            for (auto it=edges.begin();it!=edges.end();) if (it->second==std::make_pair(2u,std::int64_t(0))) it=edges.erase(it); else ++it;
        }
        void LockSourceTopology(const std::vector<std::uint32_t>& indices,std::vector<bool>& locked)
        {
            struct Use { unsigned Face,From,To; };
            std::map<Edge,std::vector<Use>> edges;
            std::vector<unsigned> parent(indices.size()/Corners); std::iota(parent.begin(),parent.end(),0);
            const auto root=[&](unsigned value) { while (parent[value]!=value) { parent[value]=parent[parent[value]]; value=parent[value]; } return value; };
            for (unsigned face=0;face<parent.size();++face) for (unsigned corner=0;corner<Corners;++corner)
            {
                const auto a=indices[face*Corners+corner],b=indices[face*Corners+(corner+1)%Corners]; edges[std::minmax(a,b)].push_back({face,a,b});
            }
            for (const auto& edge:edges)
            {
                const auto& uses=edge.second; std::int64_t orientation=0;
                for (const auto& use:uses) { orientation+=use.From<use.To?1:-1; parent[root(use.Face)]=root(uses.front().Face); }
                if (uses.size()!=2 || orientation) locked[edge.first.first]=locked[edge.first.second]=true;
            }
            // Two closed components touching at one vertex are also a seam,
            // even when a local group sees only one manifold fan.
            std::vector<unsigned> component(locked.size(),NaniteInvalid);
            for (unsigned face=0;face<parent.size();++face) for (unsigned corner=0;corner<Corners;++corner)
            {
                const auto vertex=indices[face*Corners+corner],id=root(face);
                if (component[vertex]!=NaniteInvalid && component[vertex]!=id) locked[vertex]=true;
                component[vertex]=id;
            }
        }
        bool Reduce(const std::vector<Vertex>& source,const std::vector<unsigned>& indices,const std::vector<bool>& fixed,
            const std::vector<unsigned>& representatives,const std::vector<double>& radii,const LodBuildSettings& settings,
            std::vector<unsigned>& reduced,double& displacement,std::string& error)
        {
            std::vector<unsigned> welded; welded.reserve(indices.size()); for (auto index:indices) welded.push_back(representatives[index]);
            auto keys=welded; std::sort(keys.begin(),keys.end()); keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
            std::vector<Vertex> vertices; std::vector<unsigned> local,locked; std::vector<double> normalRadii;
            for (unsigned id=0;id<keys.size();++id) { vertices.push_back(source[keys[id]]); normalRadii.push_back(radii[keys[id]]); if (fixed[keys[id]]) locked.push_back(id); }
            for (auto index:welded) local.push_back(unsigned(std::lower_bound(keys.begin(),keys.end(),index)-keys.begin()));
            auto step=settings; step.MaximumLevels=2; MeshLodChain chain;
            // Additional locks include global seams invisible to this group.
            if (!BuildMeshLods(vertices,local,step,chain,error,locked,normalRadii)) return false;
            displacement=chain.Levels.back().MaximumDisplacement; reduced.reserve(chain.Levels.back().Indices.size());
            for (auto index:chain.Levels.back().Indices) reduced.push_back(keys[index]);
            return true;
        }
        bool SettingsValid(const NaniteHierarchySettings& s)
        {
            const auto& q=s.Simplification;
            return s.ClusterTriangles && s.ClusterTriangles<=MaximumClusterTriangles && s.ClusterVertices>=Corners && s.ClusterVertices<=s.ClusterTriangles*Corners &&
                s.ClustersPerGroup>=2 && s.ClustersPerGroup<=MaximumGroupClusters && s.MaximumLevels && s.MaximumLevels<=MaximumLevels &&
                s.MaximumClusters && s.MaximumClusters<NaniteInvalid && s.MaximumGroups && s.MaximumGroups<NaniteInvalid && s.MaximumStoredIndices &&
                q.MinimumTriangles && std::isfinite(q.TriangleRatio) && q.TriangleRatio>0 && q.TriangleRatio<1 &&
                std::isfinite(q.MaximumNormalAngleDegrees) && q.MaximumNormalAngleDegrees>=0 && q.MaximumNormalAngleDegrees<=HalfTurnDegrees &&
                std::isfinite(q.MaximumTexCoordDisplacement) && q.MaximumTexCoordDisplacement>=0 && std::isfinite(q.MaximumColorDisplacement) && q.MaximumColorDisplacement>=0;
        }
        std::uint64_t Signature(const NaniteHierarchy& h)
        {
            std::uint64_t hash=HashSeed;
            const auto mix=[&](std::uint64_t value) { hash^=value; hash*=HashPrime; };
            const auto scalar=[&](double value) { std::uint64_t word; std::memcpy(&word,&value,sizeof(word)); mix(word); };
            const auto list=[&](const auto& values) { mix(values.size()); for (auto value:values) mix(value); };
            mix(h.ClusterTriangleLimit); mix(h.ClusterVertexLimit); mix(h.GroupClusterLimit); mix(h.Vertices.size());
            for (const auto& v:h.Vertices) for (double value:{v.Position.X,v.Position.Y,v.Position.Z,v.Normal.X,v.Normal.Y,v.Normal.Z,
                v.Color.X,v.Color.Y,v.Color.Z,v.Color.W,v.TexCoord.X,v.TexCoord.Y}) scalar(value);
            list(h.SourceIndices); list(h.SourceMaterials); mix(h.Clusters.size());
            for (const auto& c:h.Clusters)
            {
                list(c.Indices); list(c.SourceTriangles); mix(c.Material); mix(c.Level); mix(c.GeneratingGroup); mix(c.ParentGroup);
                for (double value:{c.BoundsMinimum.X,c.BoundsMinimum.Y,c.BoundsMinimum.Z,c.BoundsMaximum.X,c.BoundsMaximum.Y,c.BoundsMaximum.Z}) scalar(value);
            }
            mix(h.Groups.size());
            for (const auto& g:h.Groups)
            {
                list(g.Children); list(g.Parents); mix(g.Level); mix(g.Material); scalar(g.LocalDisplacement); scalar(g.MaximumDisplacement);
                for (double value:{g.BoundsMinimum.X,g.BoundsMinimum.Y,g.BoundsMinimum.Z,g.BoundsMaximum.X,g.BoundsMaximum.Y,g.BoundsMaximum.Z}) scalar(value);
            }
            list(h.Roots); return hash;
        }
    }

    bool BuildNaniteHierarchy(const std::vector<Vertex>& vertices,const std::vector<unsigned>& indices,const std::vector<unsigned>& materials,
        const NaniteHierarchySettings& settings,NaniteHierarchy& output,std::string& error)
    {
        if (!SettingsValid(settings) || vertices.size()>settings.Source.MaximumVertices || indices.size()>settings.Source.MaximumIndices ||
            indices.size()>settings.MaximumStoredIndices || indices.size()>=NaniteInvalid || (!materials.empty() && materials.size()!=indices.size()/Corners))
        { error="Nanite hierarchy source, settings or storage budgets are invalid"; return false; }
        if (!ValidateTriangleGeometry(vertices,indices,error)) return false;
        NaniteHierarchy h; h.Vertices=vertices; h.SourceIndices=indices; h.SourceMaterials=materials;
        if (h.SourceMaterials.empty()) h.SourceMaterials.resize(indices.size()/Corners);
        h.ClusterTriangleLimit=settings.ClusterTriangles; h.ClusterVertexLimit=settings.ClusterVertices; h.GroupClusterLimit=settings.ClustersPerGroup;
        std::vector<bool> fixed(vertices.size()),topology(vertices.size()); const auto positions=VertexIds(vertices);
        std::vector<unsigned> geometry,representatives; std::vector<double> normalRadii;
        geometry.reserve(indices.size()); for (auto index:indices) geometry.push_back(positions[index]); LockSourceTopology(geometry,topology);
        WeldNormalFans(vertices,positions,settings.Simplification.MaximumNormalAngleDegrees,representatives,normalRadii,fixed);
        for (unsigned id=0;id<vertices.size();++id) fixed[id]=fixed[id] || topology[positions[id]];
        std::vector<std::vector<unsigned>> leaves;
        if (!Partition(Triangles(vertices,indices,h.SourceMaterials),positions,settings.ClusterTriangles,settings.ClusterVertices,leaves,error)) return false;
        if (leaves.size()>settings.MaximumClusters) { error="Nanite leaf clusters exceed the storage budget"; return false; }
        std::vector<unsigned> level; std::size_t stored=indices.size();
        for (const auto& faces:leaves)
        {
            NaniteCluster cluster; cluster.SourceTriangles=faces; cluster.Material=h.SourceMaterials[faces.front()];
            for (auto face:faces) cluster.Indices.insert(cluster.Indices.end(),indices.begin()+face*Corners,indices.begin()+(face+1)*Corners);
            Bound(vertices,cluster); level.push_back(unsigned(h.Clusters.size())); h.Clusters.push_back(std::move(cluster));
        }
        for (unsigned depth=1;depth<settings.MaximumLevels && level.size()>1;++depth)
        {
            std::vector<Element> elements;
            for (auto id:level) { const auto& c=h.Clusters[id]; elements.push_back(MakeElement(vertices,c.Indices,0,c.Indices.size(),c.Material)); }
            std::vector<std::vector<unsigned>> partitions;
            if (!Partition(elements,positions,settings.ClustersPerGroup,0,partitions,error)) return false;
            std::vector<unsigned> next; bool changed=false;
            for (const auto& part:partitions)
            {
                std::vector<unsigned> children; for (auto index:part) children.push_back(level[index]); std::sort(children.begin(),children.end());
                if (children.size()==1) { next.push_back(children.front()); continue; }
                NaniteClusterGroup group; group.Children=children; group.Material=h.Clusters[children.front()].Material;
                std::vector<unsigned> joined;
                for (auto child:children)
                {
                    const auto& c=h.Clusters[child]; joined.insert(joined.end(),c.Indices.begin(),c.Indices.end()); group.Level=std::max(group.Level,c.Level+1);
                    const auto inherited=c.GeneratingGroup; const auto lo=inherited==NaniteInvalid?c.BoundsMinimum:h.Groups[inherited].BoundsMinimum;
                    const auto hi=inherited==NaniteInvalid?c.BoundsMaximum:h.Groups[inherited].BoundsMaximum;
                    if (child==children.front()) { group.BoundsMinimum=lo; group.BoundsMaximum=hi; } else Include(lo,hi,group.BoundsMinimum,group.BoundsMaximum);
                    if (inherited!=NaniteInvalid) group.MaximumDisplacement=std::max(group.MaximumDisplacement,h.Groups[inherited].MaximumDisplacement);
                }
                std::vector<unsigned> reduced;
                if (!Reduce(vertices,joined,fixed,representatives,normalRadii,settings.Simplification,reduced,group.LocalDisplacement,error)) return false;
                if (reduced.size()==joined.size()) { next.insert(next.end(),children.begin(),children.end()); continue; }
                std::vector<std::vector<unsigned>> parents;
                if (!Partition(Triangles(vertices,reduced,{},group.Material),positions,settings.ClusterTriangles,settings.ClusterVertices,parents,error)) return false;
                if (parents.size()>=children.size()) { next.insert(next.end(),children.begin(),children.end()); continue; }
                if (h.Groups.size()>=settings.MaximumGroups || parents.size()>settings.MaximumClusters-h.Clusters.size() || reduced.size()>settings.MaximumStoredIndices-stored)
                { error="Nanite group or parent payload exceeds the storage budget"; return false; }
                const auto sum=group.MaximumDisplacement+group.LocalDisplacement;
                group.MaximumDisplacement=sum>0?std::nextafter(sum,std::numeric_limits<double>::infinity()):0;
                if (!std::isfinite(group.MaximumDisplacement)) { error="Nanite displacement bound overflowed"; return false; }
                const auto id=unsigned(h.Groups.size());
                for (auto child:children) h.Clusters[child].ParentGroup=id;
                for (const auto& faces:parents)
                {
                    NaniteCluster parent; parent.Material=group.Material; parent.GeneratingGroup=id; parent.Level=group.Level;
                    for (auto face:faces) parent.Indices.insert(parent.Indices.end(),reduced.begin()+face*Corners,reduced.begin()+(face+1)*Corners);
                    Bound(vertices,parent); const auto parentId=unsigned(h.Clusters.size()); group.Parents.push_back(parentId); next.push_back(parentId); h.Clusters.push_back(std::move(parent));
                }
                stored+=reduced.size(); h.Groups.push_back(std::move(group)); changed=true;
            }
            level=std::move(next); if (!changed) break;
        }
        h.Roots=std::move(level); std::sort(h.Roots.begin(),h.Roots.end()); h.Signature=Signature(h);
        if (!ValidateNaniteHierarchy(h,error)) return false;
        output=std::move(h); error.clear(); return true;
    }

    bool ValidateNaniteHierarchy(const NaniteHierarchy& h,std::string& error)
    {
        const auto fail=[&](const char* message) { error=message; return false; };
        if (!ValidateTriangleGeometry(h.Vertices,h.SourceIndices,error)) return false;
        if (!h.ClusterTriangleLimit || h.ClusterTriangleLimit>MaximumClusterTriangles || h.ClusterVertexLimit<Corners || h.ClusterVertexLimit>h.ClusterTriangleLimit*Corners ||
            h.GroupClusterLimit<2 || h.GroupClusterLimit>MaximumGroupClusters || h.Clusters.empty() || h.Clusters.size()>=NaniteInvalid || h.Groups.size()>=NaniteInvalid ||
            h.SourceIndices.size()>=NaniteInvalid || h.SourceMaterials.size()!=h.SourceIndices.size()/Corners) return fail("Nanite hierarchy dimensions are invalid");
        const auto boundaryIds=VertexIds(h.Vertices,true);
        std::vector<bool> sourceSeen(h.SourceMaterials.size()); std::vector<unsigned> generated(h.Clusters.size()),consumed(h.Clusters.size()),roots;
        for (unsigned id=0;id<h.Clusters.size();++id)
        {
            const auto& c=h.Clusters[id];
            if (c.Indices.empty() || c.Indices.size()%Corners || c.Indices.size()/Corners>h.ClusterTriangleLimit || c.Level>=MaximumLevels ||
                (c.GeneratingGroup!=NaniteInvalid && c.GeneratingGroup>=h.Groups.size()) || (c.ParentGroup!=NaniteInvalid && c.ParentGroup>=h.Groups.size()))
                return fail("Nanite cluster payload or ownership is invalid");
            for (auto index:c.Indices) if (index>=h.Vertices.size()) return fail("Nanite cluster vertex is outside the source");
            std::set<unsigned> vertices(c.Indices.begin(),c.Indices.end()); if (vertices.size()>h.ClusterVertexLimit) return fail("Nanite cluster exceeds its vertex budget");
            auto bounds=c; Bound(h.Vertices,bounds);
            if (!Same(bounds.BoundsMinimum,c.BoundsMinimum) || !Same(bounds.BoundsMaximum,c.BoundsMaximum)) return fail("Nanite cluster bounds changed");
            if (c.GeneratingGroup==NaniteInvalid)
            {
                if (c.Level || c.SourceTriangles.size()!=c.Indices.size()/Corners) return fail("Nanite leaf lost source correspondence");
                for (unsigned face=0;face<c.SourceTriangles.size();++face)
                {
                    const auto source=c.SourceTriangles[face];
                    if (source>=sourceSeen.size() || sourceSeen[source] || h.SourceMaterials[source]!=c.Material ||
                        !std::equal(c.Indices.begin()+face*Corners,c.Indices.begin()+(face+1)*Corners,h.SourceIndices.begin()+source*Corners)) return fail("Nanite leaves overlap or change source triangles");
                    sourceSeen[source]=true;
                }
            }
            else if (!c.SourceTriangles.empty() || !c.Level) return fail("Nanite parent impersonates a lossless leaf");
            if (c.ParentGroup==NaniteInvalid) roots.push_back(id);
        }
        if (roots!=h.Roots || roots.empty() || std::find(sourceSeen.begin(),sourceSeen.end(),false)!=sourceSeen.end()) return fail("Nanite roots or source coverage are incomplete");
        for (unsigned id=0;id<h.Groups.size();++id)
        {
            const auto& g=h.Groups[id];
            if (g.Children.size()<2 || g.Children.size()>h.GroupClusterLimit || g.Parents.empty() || g.Parents.size()>=g.Children.size() ||
                !std::isfinite(g.LocalDisplacement) || g.LocalDisplacement<0 || !std::isfinite(g.MaximumDisplacement) || g.MaximumDisplacement<0)
                return fail("Nanite group replacement or error is invalid");
            EdgeCounts fine,coarse; std::size_t fineTriangles=0,coarseTriangles=0; unsigned level=0; double inheritedError=0; Float3 lo{},hi{};
            unsigned last=0;
            for (unsigned at=0;at<g.Children.size();++at)
            {
                const auto child=g.Children[at]; if (child>=h.Clusters.size() || (at && child<=last) || ++consumed[child]!=1) return fail("Nanite child ownership is duplicated or missing"); last=child;
                const auto& c=h.Clusters[child]; const auto producer=c.GeneratingGroup;
                if (c.ParentGroup!=id || c.Material!=g.Material || (producer!=NaniteInvalid && producer>=id)) return fail("Nanite group contains a cycle or crosses materials");
                const auto minimum=producer==NaniteInvalid?c.BoundsMinimum:h.Groups[producer].BoundsMinimum,maximum=producer==NaniteInvalid?c.BoundsMaximum:h.Groups[producer].BoundsMaximum;
                if (!at) { lo=minimum; hi=maximum; } else Include(minimum,maximum,lo,hi);
                if (producer!=NaniteInvalid) inheritedError=std::max(inheritedError,h.Groups[producer].MaximumDisplacement);
                level=std::max(level,c.Level+1); fineTriangles+=c.Indices.size()/Corners; AddBoundary(fine,c.Indices,boundaryIds);
            }
            const auto lastChild=last;
            for (auto parent:g.Parents)
            {
                if (parent>=h.Clusters.size() || parent<=last || ++generated[parent]!=1) return fail("Nanite parent ownership is duplicated or not topological"); last=parent;
                const auto& c=h.Clusters[parent];
                if (parent<=lastChild || c.GeneratingGroup!=id || c.Material!=g.Material || c.Level!=g.Level) return fail("Nanite parent has inconsistent shared LOD data");
                coarseTriangles+=c.Indices.size()/Corners; AddBoundary(coarse,c.Indices,boundaryIds);
            }
            RemoveInterior(fine); RemoveInterior(coarse);
            if (fine!=coarse || coarseTriangles>=fineTriangles) return fail("Nanite replacement changed its exterior boundary or failed to simplify");
            if (g.Level!=level || !Same(lo,g.BoundsMinimum) || !Same(hi,g.BoundsMaximum) || g.MaximumDisplacement<inheritedError+g.LocalDisplacement)
                return fail("Nanite parent bounds or errors are not monotonic");
        }
        for (unsigned id=0;id<h.Clusters.size();++id)
            if (generated[id]!=(h.Clusters[id].GeneratingGroup==NaniteInvalid?0u:1u) || consumed[id]!=(h.Clusters[id].ParentGroup==NaniteInvalid?0u:1u))
                return fail("Nanite reciprocal group ownership is incomplete");
        if (h.Signature!=Signature(h)) return fail("Nanite content signature is stale");
        error.clear(); return true;
    }
}
