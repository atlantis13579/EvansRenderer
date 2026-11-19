#include "Renderer/VirtualGeometry/VirtualGeometryGpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace Evans::Renderer
{
    namespace
    {
        constexpr std::uint32_t NodeCoverage=8,NodeMinimum=12,NodeMaximum=16,ViewErrors=16,ViewViewport=32;
        constexpr float Roundoff=16*std::numeric_limits<float>::epsilon(),Underflow=16*std::numeric_limits<float>::min();
        constexpr float Infinity=std::numeric_limits<float>::infinity();
        constexpr std::uint32_t MaximumDispatchAxis=65535;
        std::uint32_t Bits(float value) { std::uint32_t bits; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
        float Scalar(std::uint32_t value) { float scalar; std::memcpy(&scalar,&value,sizeof(scalar)); return scalar; }
        float Upper(double value)
        { float f=static_cast<float>(value); return double(f)<value?std::nextafter(f,Infinity):f; }
        float Lower(double value)
        { float f=static_cast<float>(value); if (!std::isfinite(f)) return std::numeric_limits<float>::max(); return double(f)>value?std::nextafter(f,0.0f):f; }
        float Expand(float value) { return std::isfinite(value) && value>=0?value*(1+Roundoff)+Underflow:Infinity; }
        float Contract(float value) { return value>0?value*(1-Roundoff)-Underflow:value; }
        std::uint32_t Blocks(std::uint32_t count) { return (count+VirtualGeometryGroupWidth-1)/VirtualGeometryGroupWidth; }
        std::uint32_t ClusterBase(const VirtualGeometryGpuConstants& c) { return c.Counts.X*VirtualGeometryNodeStateWords; }
        std::uint32_t BlockBase(const VirtualGeometryGpuConstants& c) { return ClusterBase(c)+c.Limits.X*VirtualGeometryClusterStateWords; }
        std::uint32_t StatusBase(const VirtualGeometryGpuConstants& c) { return BlockBase(c)+Blocks(c.Limits.X)*VirtualGeometryBlockWords; }
        void Append(std::vector<std::uint32_t>& words,std::initializer_list<std::uint32_t> values) { words.insert(words.end(),values); }
        void Append(std::vector<std::uint32_t>& words,const Float4& v) { Append(words,{Bits(v.X),Bits(v.Y),Bits(v.Z),Bits(v.W)}); }
        unsigned Word(VirtualClusterWord value) { return static_cast<unsigned>(value); }
        std::uint32_t Node(const VirtualGeometryGpuUpload& u,unsigned id) { return u.Constants.Offsets.X+id*VirtualGeometryNodeWords; }
        std::uint32_t Cluster(const VirtualGeometryGpuUpload& u,unsigned id) { return u.Constants.Tables.X+id*VirtualGeometryClusterWords; }
    }

    bool PrepareVirtualGeometryGpu(const VirtualGeometryHierarchy& h,const VirtualGeometryResidency& r,const std::vector<LodView>& views,
        const VirtualGeometryCutSettings& settings,const std::vector<std::uint32_t>& previous,const GeometryPreparationLimits& limits,VirtualGeometryGpuUpload& output,std::string& error)
    {
        if (!ValidateVirtualGeometryHierarchy(h,error) || !ValidateVirtualGeometryResidency(r,error)) return false;
        if (r.Source()!=h.Signature || r.PageSlots().size()!=h.Data.Pages.size() || r.PinnedPages()!=VirtualGeometryPinnedPages(h) || views.empty() || views.size()>VirtualGeometryInvalid ||
            !std::isfinite(settings.MaximumErrorPixels) || settings.MaximumErrorPixels<=0 || !std::isfinite(settings.HysteresisFraction) || settings.HysteresisFraction<0 || settings.HysteresisFraction>=1 || !limits.MaximumVertices || !limits.MaximumIndices)
        { error="Virtual GPU source, cache, views or quality settings are invalid"; return false; }
        std::vector<bool> selected(h.Nodes.size()); auto ordered=previous;
        for (auto id:ordered) if (id>=h.Nodes.size()) { error="Virtual GPU previous cut references a missing node"; return false; }
        std::sort(ordered.begin(),ordered.end(),[&](auto a,auto b){return h.Nodes[a].FirstTriangle<h.Nodes[b].FirstTriangle;});
        unsigned covered=0;
        for (auto id:ordered) { if (h.Nodes[id].FirstTriangle!=covered) { error="Virtual GPU previous cut overlaps or has gaps"; return false; } covered+=h.Nodes[id].TriangleCount; selected[id]=true; }
        if (!ordered.empty() && covered!=h.SourceTriangles.size()) { error="Virtual GPU previous cut is incomplete"; return false; }
        VirtualGeometryGpuUpload u; u.Source=h.Signature; auto& c=u.Constants;
        c.Counts={static_cast<unsigned>(h.Nodes.size()),static_cast<unsigned>(views.size()),static_cast<unsigned>(h.Data.Pages.size()),static_cast<unsigned>(r.Slots().size())};
        c.Identity={static_cast<unsigned>(r.Epoch()),static_cast<unsigned>(r.Epoch()>>32),static_cast<unsigned>(r.Revision()),static_cast<unsigned>(r.Revision()>>32)};
        c.Limits.X=static_cast<unsigned>(h.Data.Clusters.size());
        const GeometryPreparationLimits native; c.Capacity={static_cast<unsigned>(std::min(limits.MaximumVertices,native.MaximumVertices)),static_cast<unsigned>(std::min(limits.MaximumIndices,native.MaximumIndices)),h.Data.SourceTriangleCount,0};
        c.Quality={Lower(settings.MaximumErrorPixels),Lower(settings.MaximumErrorPixels*(1-settings.HysteresisFraction)),0,0};
        std::vector<unsigned> heights(h.Nodes.size()),inverse(h.Data.Clusters.size());
        u.ClusterIds.resize(h.Data.Clusters.size()); std::iota(u.ClusterIds.begin(),u.ClusterIds.end(),0);
        std::sort(u.ClusterIds.begin(),u.ClusterIds.end(),[&](auto a,auto b){const auto x=h.Data.Clusters[a].Material,y=h.Data.Clusters[b].Material; return x!=y?x<y:a<b;});
        for (unsigned id=0;id<u.ClusterIds.size();++id) inverse[u.ClusterIds[id]]=id;
        c.Offsets.X=static_cast<unsigned>(u.Metadata.size());
        for (unsigned id=0;id<h.Nodes.size();++id)
        {
            const auto& n=h.Nodes[id]; for (unsigned at=0;at<n.ChildCount;++at) heights[id]=std::max(heights[id],heights[h.Children[n.FirstChild+at]]+1);
            Append(u.Metadata,{n.Parent,n.FirstChild,n.ChildCount,heights[id],n.FirstCluster,n.ClusterCount,n.FirstPage,n.PageCount,n.FirstTriangle,n.TriangleCount,selected[id]?VirtualGeometryPreviousSelection:0,n.Material});
            Append(u.Metadata,Float4{n.BoundsMinimum.X,n.BoundsMinimum.Y,n.BoundsMinimum.Z,Upper(n.MaximumDisplacement)});
            Append(u.Metadata,Float4{n.BoundsMaximum.X,n.BoundsMaximum.Y,n.BoundsMaximum.Z,0});
        }
        c.Offsets.Y=static_cast<unsigned>(u.Metadata.size()); u.Metadata.insert(u.Metadata.end(),h.Children.begin(),h.Children.end());
        c.Offsets.Z=static_cast<unsigned>(u.Metadata.size()); for (auto cluster:h.Clusters) u.Metadata.push_back(inverse[cluster]);
        c.Offsets.W=static_cast<unsigned>(u.Metadata.size()); u.Metadata.insert(u.Metadata.end(),h.Pages.begin(),h.Pages.end());
        c.Tables.X=static_cast<unsigned>(u.Metadata.size());
        for (auto id:u.ClusterIds)
        {
            const auto& cluster=h.Data.Clusters[id]; const auto* header=h.Data.Pages[cluster.Page].Words.data()+cluster.Word;
            Append(u.Metadata,{cluster.Page,cluster.Word,header[Word(VirtualClusterWord::VertexCount)],header[Word(VirtualClusterWord::IndexCount)],cluster.Material,cluster.Node,id,0});
        }
        c.Tables.Y=static_cast<unsigned>(u.Metadata.size());
        for (const auto& view:views)
        {
            if (!view.Width || !view.Height) { error="Virtual GPU view has zero resolution"; return false; }
            for (const auto* matrix:{&view.World,&view.ViewProjection}) for (float value:matrix->Values)
                if (!std::isfinite(value)) { error="Virtual GPU view matrix is not finite"; return false; }
            std::array<Float4,8> rows{};
            for (unsigned row=0;row<4;++row) for (unsigned column=0;column<4;++column)
            {
                double value=0,magnitude=0;
                for (unsigned k=0;k<4;++k) { const double term=double(view.ViewProjection.Values[row*4+k])*view.World.Values[k*4+column]; value+=term; magnitude+=std::abs(term); }
                const float coefficient=static_cast<float>(value),uncertainty=Upper(std::abs(double(coefficient)-value)+magnitude*8*std::numeric_limits<double>::epsilon());
                float* coefficients[]{&rows[row].X,&rows[row].Y,&rows[row].Z,&rows[row].W}; float* errors[]{&rows[row+4].X,&rows[row+4].Y,&rows[row+4].Z,&rows[row+4].W};
                *coefficients[column]=coefficient; *errors[column]=uncertainty;
            }
            for (const auto& row:rows) Append(u.Metadata,row);
            Append(u.Metadata,Float4{Upper(double(view.Width)/2),Upper(double(view.Height)/2),0,0});
        }
        c.Tables.Z=static_cast<unsigned>(u.Metadata.size()); u.Levels.resize(*std::max_element(heights.begin(),heights.end())+1);
        std::vector<unsigned> order(h.Nodes.size()); std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](auto a,auto b){return heights[a]!=heights[b]?heights[a]<heights[b]:a<b;});
        for (std::size_t begin=0;begin<order.size();)
        {
            std::size_t end=begin+1; while (end<order.size() && heights[order[end]]==heights[order[begin]]) ++end;
            u.Levels[heights[order[begin]]]={static_cast<unsigned>(begin),static_cast<unsigned>(end-begin)}; begin=end;
        }
        u.Metadata.insert(u.Metadata.end(),order.begin(),order.end());
        if (u.Metadata.size()>std::numeric_limits<std::uint32_t>::max()/sizeof(std::uint32_t)) { error="Virtual GPU metadata exceeds byte addressing"; return false; }
        c.Tables.W=static_cast<unsigned>(u.Metadata.size()); c.Limits.Y=StatusBase(c)+VirtualGeometryStatusWords;
        c.Limits.Z=c.Counts.W*VirtualGeometryPageWords; u.InitialState.resize(c.Limits.Y); u.PageTable.resize(c.Counts.Z);
        u.PhysicalPages.resize(c.Counts.W);
        for (unsigned page=0;page<c.Counts.Z;++page)
        {
            const auto slot=r.PageSlots()[page]; auto& entry=u.PageTable[page]; entry={slot,0,c.Identity.X,c.Identity.Y}; if (slot==VirtualGeometryInvalid) continue;
            entry.Y=r.Slots()[slot].Generation; u.PhysicalPages[slot]=h.Data.Pages[page]; auto& words=u.PhysicalPages[slot].Words;
            words[5]=page; words[6]=entry.Y; words[7]=entry.Z;
        }
        if (!ValidateVirtualGeometryGpuUpload(u,error)) return false;
        output=std::move(u); error.clear(); return true;
    }

    bool ValidateVirtualGeometryGpuUpload(const VirtualGeometryGpuUpload& u,std::string& error)
    {
        const auto& c=u.Constants; auto invalid=[&](const char* message){error=message; return false;};
        const GeometryPreparationLimits native;
        auto span=[&](std::uint32_t first,std::uint64_t count){return std::uint64_t(first)+count<=u.Metadata.size();};
        if (!c.Counts.X || !c.Counts.Y || !c.Counts.Z || !c.Counts.W || !c.Limits.X || c.Tables.W!=u.Metadata.size() ||
            u.Metadata.size()>UINT32_MAX/sizeof(std::uint32_t) || u.InitialState.size()!=StatusBase(c)+VirtualGeometryStatusWords || c.Limits.Y!=u.InitialState.size() ||
            u.PhysicalPages.size()!=c.Counts.W || u.PageTable.size()!=c.Counts.Z || u.ClusterIds.size()!=c.Limits.X || std::uint64_t(c.Counts.W)*VirtualGeometryPageWords!=c.Limits.Z ||
            !span(c.Offsets.X,std::uint64_t(c.Counts.X)*VirtualGeometryNodeWords) || !span(c.Tables.X,std::uint64_t(c.Limits.X)*VirtualGeometryClusterWords) ||
            !span(c.Tables.Y,std::uint64_t(c.Counts.Y)*VirtualGeometryViewWords) || !span(c.Tables.Z,c.Counts.X) || !c.Capacity.X || !c.Capacity.Y ||
            !std::isfinite(c.Quality.X) || !std::isfinite(c.Quality.Y) || c.Quality.X<0 || c.Quality.Y<0 || c.Quality.Y>c.Quality.X || u.Levels.empty() ||
            c.Capacity.X>native.MaximumVertices || c.Capacity.Y>native.MaximumIndices || !c.Capacity.Z || std::uint64_t(c.Capacity.Z)*3>native.MaximumIndices ||
            c.Limits.X>UINT32_MAX/VirtualGeometryMaximumClusterVertices || std::uint64_t(c.Counts.W)*VirtualGeometryPageBytes>UINT32_MAX ||
            !(c.Identity.X || c.Identity.Y) || !(c.Identity.Z || c.Identity.W)) return invalid("Virtual GPU buffer dimensions or frame ABI are invalid");
        for (auto value:u.InitialState) if (value) return invalid("Virtual GPU initial scratch must be cleared");
        std::vector<bool> ordered(c.Counts.X),clusterIds(c.Limits.X),slots(c.Counts.W); unsigned levelCount=0;
        for (unsigned page=0;page<u.PageTable.size();++page)
        {
            const auto& entry=u.PageTable[page]; if (entry.X==VirtualGeometryInvalid) continue;
            if (entry.X>=slots.size() || slots[entry.X] || !entry.Y || entry.Z!=c.Identity.X || entry.W!=c.Identity.Y) return invalid("Virtual GPU page mapping has duplicate slots or stale identity");
            slots[entry.X]=true;
        }
        for (unsigned height=0;height<u.Levels.size();++height)
        {
            const auto& level=u.Levels[height]; if (level.First!=levelCount || !level.Count || std::uint64_t(level.First)+level.Count>c.Counts.X) return invalid("Virtual GPU level order has gaps");
            for (unsigned at=0;at<level.Count;++at)
            { const auto id=u.Metadata[c.Tables.Z+level.First+at]; if (id>=c.Counts.X || ordered[id] || u.Metadata[Node(u,id)+3]!=height) return invalid("Virtual GPU height order is not a permutation"); ordered[id]=true; }
            levelCount+=level.Count;
        }
        if (levelCount!=c.Counts.X) return invalid("Virtual GPU height order is incomplete");
        std::uint64_t childCursor=0,clusterCursor=0,pageCursor=0;
        std::vector<unsigned> parents(c.Counts.X); std::vector<std::pair<unsigned,unsigned>> roots;
        for (unsigned id=0;id<c.Counts.X;++id)
        {
            const auto n=Node(u,id); const auto* words=u.Metadata.data()+n;
            if ((words[0]!=VirtualGeometryInvalid && (words[0]<=id || words[0]>=c.Counts.X)) || words[2]>8 ||
                !span(c.Offsets.Y,std::uint64_t(words[1])+words[2]) || !span(c.Offsets.Z,std::uint64_t(words[4])+words[5]) ||
                !span(c.Offsets.W,std::uint64_t(words[6])+words[7]) || !words[5] || !words[7] || words[1]!=childCursor || words[4]!=clusterCursor || words[6]!=pageCursor ||
                !words[NodeCoverage+1] || std::uint64_t(words[NodeCoverage])+words[NodeCoverage+1]>c.Capacity.Z || (words[NodeCoverage+2]&~VirtualGeometryPreviousSelection)) return invalid("Virtual GPU node table span is invalid");
            childCursor+=words[2]; clusterCursor+=words[5]; pageCursor+=words[7];
            if (words[0]==VirtualGeometryInvalid) roots.emplace_back(words[NodeCoverage],words[NodeCoverage+1]);
            for (unsigned axis=0;axis<3;++axis)
                if (!std::isfinite(Scalar(words[NodeMinimum+axis])) || !std::isfinite(Scalar(words[NodeMaximum+axis])) || Scalar(words[NodeMinimum+axis])>Scalar(words[NodeMaximum+axis])) return invalid("Virtual GPU node bounds are invalid");
            const auto radius=Scalar(words[NodeMinimum+3]); if (std::isnan(radius) || radius<0 || (!words[2] && radius!=0)) return invalid("Virtual GPU node error is invalid");
            unsigned height=0;
            unsigned coverage=words[NodeCoverage];
            for (unsigned at=0;at<words[2];++at)
            {
                const auto child=u.Metadata[c.Offsets.Y+words[1]+at]; if (child>=id || u.Metadata[Node(u,child)]!=id || ++parents[child]!=1) return invalid("Virtual GPU child graph is invalid");
                const auto other=Node(u,child); if (u.Metadata[other+NodeCoverage]!=coverage || u.Metadata[other+NodeCoverage+3]!=words[NodeCoverage+3] || radius<Scalar(u.Metadata[other+NodeMinimum+3])) return invalid("Virtual GPU child coverage, material or error is inconsistent");
                coverage+=u.Metadata[other+NodeCoverage+1]; height=std::max(height,u.Metadata[other+3]+1);
            }
            if (words[2] && coverage!=words[NodeCoverage]+words[NodeCoverage+1]) return invalid("Virtual GPU parent coverage has a gap");
            if (height!=words[3]) return invalid("Virtual GPU node height differs from its children");
            std::vector<unsigned> pages; std::uint64_t triangles=0;
            for (unsigned at=0;at<words[5];++at)
            {
                const auto cluster=u.Metadata[c.Offsets.Z+words[4]+at]; if (cluster>=c.Limits.X) return invalid("Virtual GPU cluster reference is outside its directory");
                const auto d=Cluster(u,cluster); if (u.Metadata[d+4]!=words[NodeCoverage+3]) return invalid("Virtual GPU node crosses materials");
                pages.push_back(u.Metadata[d]); triangles+=u.Metadata[d+3]/3;
            }
            if (!triangles || triangles>words[NodeCoverage+1] || (!words[2] && triangles!=words[NodeCoverage+1])) return invalid("Virtual GPU representation exceeds its source region");
            std::sort(pages.begin(),pages.end()); pages.erase(std::unique(pages.begin(),pages.end()),pages.end());
            if (pages.size()!=words[7] || !std::equal(pages.begin(),pages.end(),u.Metadata.begin()+c.Offsets.W+words[6])) return invalid("Virtual GPU node page bundle differs from its representation");
            for (auto page:pages)
                if (page>=c.Counts.Z || (words[0]==VirtualGeometryInvalid && u.PageTable[page].X==VirtualGeometryInvalid)) return invalid("Virtual GPU root page is missing");
        }
        if (c.Offsets.X!=0 || c.Offsets.Y!=std::uint64_t(c.Counts.X)*VirtualGeometryNodeWords || c.Offsets.Z!=std::uint64_t(c.Offsets.Y)+childCursor ||
            c.Offsets.W!=std::uint64_t(c.Offsets.Z)+clusterCursor || c.Tables.X!=std::uint64_t(c.Offsets.W)+pageCursor || c.Tables.Y!=std::uint64_t(c.Tables.X)+std::uint64_t(c.Limits.X)*VirtualGeometryClusterWords ||
            c.Tables.Z!=std::uint64_t(c.Tables.Y)+std::uint64_t(c.Counts.Y)*VirtualGeometryViewWords || c.Tables.W!=std::uint64_t(c.Tables.Z)+c.Counts.X) return invalid("Virtual GPU metadata tables overlap or contain unowned words");
        std::sort(roots.begin(),roots.end()); unsigned coverage=0;
        for (const auto& root:roots) { if (root.first!=coverage) return invalid("Virtual GPU roots overlap or leave a gap"); coverage+=root.second; }
        if (coverage!=c.Capacity.Z) return invalid("Virtual GPU roots omit source triangles");
        for (unsigned id=0;id<c.Counts.X;++id) if (parents[id]!=(u.Metadata[Node(u,id)]==VirtualGeometryInvalid?0u:1u)) return invalid("Virtual GPU parent graph lacks reciprocity");
        for (unsigned view=0;view<c.Counts.Y;++view)
        {
            const auto v=c.Tables.Y+view*VirtualGeometryViewWords;
            for (unsigned word=0;word<ViewViewport;++word)
            { const auto value=Scalar(u.Metadata[v+word]); if (std::isnan(value) || (word>=ViewErrors && value<0)) return invalid("Virtual GPU projection interval is invalid"); }
            for (unsigned axis=0;axis<2;++axis) if (!(Scalar(u.Metadata[v+ViewViewport+axis])>0) || !std::isfinite(Scalar(u.Metadata[v+ViewViewport+axis]))) return invalid("Virtual GPU viewport is invalid");
        }
        for (unsigned id=0;id<c.Limits.X;++id)
        {
            const auto* cluster=u.Metadata.data()+Cluster(u,id);
            if (cluster[0]>=c.Counts.Z || cluster[1]<VirtualGeometryPageHeaderWords || std::uint64_t(cluster[1])+VirtualGeometryClusterHeaderWords>VirtualGeometryPageWords ||
                !cluster[2] || cluster[2]>VirtualGeometryMaximumClusterVertices || !cluster[3] || cluster[3]%3 || cluster[3]>VirtualGeometryClusterTriangles*3 || cluster[5]>=c.Counts.X || cluster[6]!=u.ClusterIds[id] ||
                cluster[6]>=clusterIds.size() || clusterIds[cluster[6]] || (id && cluster[4]<u.Metadata[Cluster(u,id-1)+4])) return invalid("Virtual GPU material directory is invalid");
            clusterIds[cluster[6]]=true;
        }
        error.clear(); return true;
    }

    bool BuildVirtualGeometryGpuDispatch(const VirtualGeometryGpuUpload& u,VirtualGeometryGpuPass pass,unsigned height,
        VirtualGeometryGpuConstants& constants,std::uint32_t& groupsX,std::uint32_t& groupsY) noexcept
    {
        if (pass>=VirtualGeometryGpuPass::Count) return false;
        auto c=u.Constants; unsigned count=0,groups=0;
        if (pass==VirtualGeometryGpuPass::Propagate)
        { if (!height || height>=u.Levels.size()) return false; c.Dispatch.X=u.Levels[height].First; count=u.Levels[height].Count; }
        else if (pass==VirtualGeometryGpuPass::Evaluate || pass==VirtualGeometryGpuPass::Select) count=c.Counts.X;
        else count=c.Limits.X;
        groups=pass==VirtualGeometryGpuPass::Decode?count:pass==VirtualGeometryGpuPass::BlockScan?1:Blocks(count);
        if (!groups) return false;
        groupsY=static_cast<unsigned>((std::uint64_t(groups)+MaximumDispatchAxis-1)/MaximumDispatchAxis);
        groupsX=(groups+groupsY-1)/groupsY; c.Dispatch.Y=count; c.Dispatch.Z=groupsX;
        constants=c; return groupsY<=MaximumDispatchAxis;
    }

    bool BuildVirtualGeometryGpuSchedule(const VirtualGeometryGpuUpload& u,std::vector<VirtualGeometryGpuDispatch>& output,std::string& error)
    {
        if (!ValidateVirtualGeometryGpuUpload(u,error)) return false;
        std::vector<VirtualGeometryGpuDispatch> jobs;
        auto add=[&](VirtualGeometryGpuPass pass,unsigned height=0)
        { VirtualGeometryGpuDispatch job; job.Pass=pass; if (!BuildVirtualGeometryGpuDispatch(u,pass,height,job.Constants,job.GroupsX,job.GroupsY)) return false; jobs.push_back(job); return true; };
        if (!add(VirtualGeometryGpuPass::Evaluate)) return false;
        for (unsigned height=1;height<u.Levels.size();++height) if (!add(VirtualGeometryGpuPass::Propagate,height)) return false;
        for (auto pass:{VirtualGeometryGpuPass::Select,VirtualGeometryGpuPass::ClusterScan,VirtualGeometryGpuPass::BlockScan,VirtualGeometryGpuPass::Decode}) if (!add(pass)) return false;
        output=std::move(jobs); error.clear(); return true;
    }

    bool ValidateVirtualGeometryGpuState(const VirtualGeometryGpuUpload& u,const std::vector<std::uint32_t>& state,VirtualGeometryGpuSelection& output,std::string& error)
    {
        if (!ValidateVirtualGeometryGpuUpload(u,error)) return false;
        const auto& c=u.Constants;
        if (state.size()!=c.Limits.Y) { error="Virtual GPU readback has an invalid state size"; return false; }
        if (state[StatusBase(c)]) { error="Virtual GPU rejected its job, status bits="+std::to_string(state[StatusBase(c)]); return false; }
        VirtualGeometryGpuSelection selected; std::vector<bool> active(c.Limits.X);
        for (unsigned id=0;id<c.Counts.X;++id)
        {
            const auto s=id*VirtualGeometryNodeStateWords,n=Node(u,id); const auto pixels=Scalar(state[s]);
            if (std::isnan(pixels) || pixels<0 || state[s+1]>1 || state[s+2]>1 || state[s+3]>1 || state[s+4]>1 || state[s+5]>2 ||
                (state[s+5] && (!std::isfinite(Scalar(state[s+6])) || Scalar(state[s+6])<0)))
            { error="Virtual GPU node feedback is malformed"; return false; }
            if (!state[s+4]) continue;
            if (!state[s+1] || !state[s+2]) { error="Virtual GPU selected a nonresident node"; return false; }
            selected.Nodes.push_back(id);
            for (unsigned at=0;at<u.Metadata[n+5];++at)
            {
                const auto cluster=u.Metadata[c.Offsets.Z+u.Metadata[n+4]+at];
                if (active[cluster]) { error="Virtual GPU cut references one cluster twice"; return false; } active[cluster]=true;
            }
        }
        std::sort(selected.Nodes.begin(),selected.Nodes.end(),[&](auto a,auto b){return u.Metadata[Node(u,a)+NodeCoverage]<u.Metadata[Node(u,b)+NodeCoverage];});
        unsigned covered=0;
        for (auto id:selected.Nodes)
        { const auto n=Node(u,id); if (u.Metadata[n+NodeCoverage]!=covered) { error="Virtual GPU cut has overlap or missing source coverage"; return false; } covered+=u.Metadata[n+NodeCoverage+1]; }
        if (covered!=c.Capacity.Z) { error="Virtual GPU cut omits source triangles"; return false; }
        unsigned localV=0,localI=0,localA=0;
        for (unsigned id=0;id<c.Limits.X;++id)
        {
            if (id%VirtualGeometryGroupWidth==0)
            {
                const auto b=BlockBase(c)+(id/VirtualGeometryGroupWidth)*VirtualGeometryBlockWords;
                if (state[b]!=selected.VertexCount || state[b+1]!=selected.IndexCount || state[b+2]!=selected.ActiveClusters) { error="Virtual GPU block prefix is inconsistent"; return false; }
                localV=localI=localA=0;
            }
            const auto s=ClusterBase(c)+id*VirtualGeometryClusterStateWords,d=Cluster(u,id);
            if (state[s]!=(active[id]?1u:0u) || state[s+1]!=localV || state[s+2]!=localI || state[s+3]!=localA) { error="Virtual GPU cluster prefix or selection mask is inconsistent"; return false; }
            if (!active[id]) continue;
            const auto vertices=u.Metadata[d+2],indices=u.Metadata[d+3],material=u.Metadata[d+4];
            if (selected.Materials.empty() || selected.Materials.back().Material!=material) selected.Materials.push_back({material,selected.IndexCount,0});
            selected.Materials.back().IndexCount+=indices; selected.VertexCount+=vertices; selected.IndexCount+=indices; ++selected.ActiveClusters;
            localV+=vertices; localI+=indices; ++localA;
        }
        const auto status=StatusBase(c);
        if (!selected.VertexCount || !selected.IndexCount || selected.VertexCount>c.Capacity.X || selected.IndexCount>c.Capacity.Y ||
            state[status+1]!=selected.VertexCount || state[status+2]!=selected.IndexCount || state[status+3]!=selected.ActiveClusters)
        { error="Virtual GPU summary exceeds its capacity or differs from its prefix"; return false; }
        output=std::move(selected); error.clear(); return true;
    }

    bool ValidateVirtualGeometryGpuReadback(const VirtualGeometryGpuUpload& u,const VirtualGeometryGpuReadback& readback,VirtualGeometryGpuSelection& output,std::string& error)
    {
        VirtualGeometryGpuSelection candidate;
        if (!ValidateVirtualGeometryGpuState(u,readback.State,candidate,error)) return false;
        if (readback.Vertices.size()!=candidate.VertexCount || readback.Indices.size()!=candidate.IndexCount) { error="Virtual GPU drawable stream sizes differ from feedback"; return false; }
        if (!ValidateTriangleGeometry(readback.Vertices,readback.Indices,error)) return false;
        output=std::move(candidate); error.clear(); return true;
    }

    bool AdmitVirtualGeometryGpuFeedback(const VirtualGeometryHierarchy& h,const VirtualGeometryResidency& residency,
        const VirtualGeometryGpuUpload& u,const std::vector<std::uint32_t>& state,VirtualGeometryCut& output,std::string& error)
    {
        VirtualGeometryGpuSelection selected;
        if (!ValidateVirtualGeometryGpuState(u,state,selected,error)) return false;
        const auto& c=u.Constants;
        if (u.Source!=h.Signature || residency.Source()!=h.Signature || c.Counts.X!=h.Nodes.size() || c.Counts.Z!=residency.PageSlots().size() || c.Counts.W!=residency.Slots().size() ||
            c.Identity.X!=std::uint32_t(residency.Epoch()) || c.Identity.Y!=std::uint32_t(residency.Epoch()>>32) || c.Identity.Z!=std::uint32_t(residency.Revision()) || c.Identity.W!=std::uint32_t(residency.Revision()>>32))
        { error="Native virtual feedback is stale or belongs to another hierarchy/cache"; return false; }
        for (unsigned page=0;page<c.Counts.Z;++page)
        {
            const auto& entry=u.PageTable[page]; const auto slot=residency.PageSlots()[page];
            if (entry.X!=slot || (slot!=VirtualGeometryInvalid && entry.Y!=residency.Slots()[slot].Generation)) { error="Native virtual feedback uses another page mapping"; return false; }
        }
        std::vector<VirtualGeometryTransition> transitions; VirtualGeometryCut cut;
        cut.Source=h.Signature; cut.CacheEpoch=residency.Epoch(); cut.CacheRevision=residency.Revision(); cut.Nodes=selected.Nodes;
        for (unsigned id=0;id<c.Counts.X;++id)
        {
            const auto& n=h.Nodes[id]; const auto* words=u.Metadata.data()+Node(u,id); const auto s=id*VirtualGeometryNodeStateWords;
            if (words[0]!=n.Parent || words[1]!=n.FirstChild || words[2]!=n.ChildCount || words[4]!=n.FirstCluster || words[5]!=n.ClusterCount || words[6]!=n.FirstPage || words[7]!=n.PageCount ||
                words[NodeCoverage]!=n.FirstTriangle || words[NodeCoverage+1]!=n.TriangleCount || words[NodeCoverage+3]!=n.Material)
            { error="Native virtual feedback uses another node graph"; return false; }
            const auto pixels=Scalar(state[s]);
            if (state[s+4])
            { cut.MaximumSelectedErrorPixels=std::max(cut.MaximumSelectedErrorPixels,double(pixels)); cut.OverBudgetNodes+=pixels>c.Quality.X; cut.MissingDetailNodes+=n.ChildCount && pixels>(state[s+3]?c.Quality.Y:c.Quality.X); }
            if (state[s+5]) transitions.push_back({id,Scalar(state[s+6]),state[s+5]==1});
        }
        for (auto id:cut.Nodes) { const auto& n=h.Nodes[id]; cut.Clusters.insert(cut.Clusters.end(),h.Clusters.begin()+n.FirstCluster,h.Clusters.begin()+n.FirstCluster+n.ClusterCount); }
        return AdmitVirtualGeometryTransitions(h,residency,std::move(cut),transitions,output,error);
    }

}
