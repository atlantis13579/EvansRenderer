#include "Renderer/Nanite/NaniteGpu.h"
#include "Renderer/Nanite/NaniteAsset.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <numeric>

namespace Evans::Renderer
{
    namespace
    {
        constexpr unsigned GroupMinimum=8,GroupMaximum=12,ViewErrors=16,ViewViewport=32,MaximumDispatchAxis=65535;
        constexpr float Infinity=std::numeric_limits<float>::infinity();
        constexpr float Roundoff=16*std::numeric_limits<float>::epsilon(),Underflow=16*std::numeric_limits<float>::min();
        unsigned Bits(float value) { unsigned bits; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
        float Scalar(unsigned bits) { float value; std::memcpy(&value,&bits,sizeof(value)); return value; }
        int Signed(unsigned bits) { int value; std::memcpy(&value,&bits,sizeof(value)); return value; }
        float Upper(double value) { const float f=float(value); return double(f)<value?std::nextafter(f,Infinity):f; }
        float Lower(double value) { const float f=float(value); return !std::isfinite(f)?std::numeric_limits<float>::max():double(f)>value?std::nextafter(f,0.0f):f; }
        unsigned Blocks(unsigned count) { return unsigned((std::uint64_t(count)+NaniteGpuWidth-1)/NaniteGpuWidth); }
        unsigned ClusterState(const NaniteGpuConstants& c) { return c.Counts.X*NaniteGpuGroupStateWords; }
        unsigned BlockState(const NaniteGpuConstants& c) { return ClusterState(c)+c.Tables.W*NaniteGpuClusterStateWords; }
        unsigned Status(const NaniteGpuConstants& c) { return BlockState(c)+c.Limits.Z*NaniteGpuBlockWords; }
        unsigned Group(const NaniteGpuUpload& u,unsigned id) { return u.Constants.Offsets.X+id*NaniteGpuGroupWords; }
        unsigned Cluster(const NaniteGpuUpload& u,unsigned id) { return u.Constants.Offsets.W+id*NaniteGpuClusterWords; }
        void Append(std::vector<unsigned>& words,std::initializer_list<unsigned> values) { words.insert(words.end(),values); }
        void Append(std::vector<unsigned>& words,const Float4& v) { Append(words,{Bits(v.X),Bits(v.Y),Bits(v.Z),Bits(v.W)}); }
        std::uint64_t Signature(const NaniteGpuUpload& u)
        {
            std::uint64_t hash=14695981039346656037ull; const auto mix=[&](std::uint64_t value) { hash^=value; hash*=1099511628211ull; };
            mix(u.Source); std::array<unsigned,sizeof(NaniteGpuConstants)/sizeof(unsigned)> constants{}; std::memcpy(constants.data(),&u.Constants,sizeof(u.Constants));
            for (auto value:constants) mix(value);
            for (const auto* values:{&u.Metadata,&u.InitialState,&u.ClusterIds}) { mix(values->size()); for (auto value:*values) mix(value); }
            const auto identity=[&](const ShaderUInt4& value) { mix(value.X); mix(value.Y); mix(value.Z); mix(value.W); };
            mix(u.PageTable.size()); for (const auto& entry:u.PageTable) identity(entry);
            mix(u.PhysicalPages.size()); for (const auto& page:u.PhysicalPages) { identity(page.Identity); for (auto value:page.Payload.Words) mix(value); }
            mix(u.Levels.size()); for (const auto& level:u.Levels) { mix(level.First); mix(level.Count); } return hash;
        }
        bool Resident(const NaniteGpuUpload& u,unsigned page,unsigned& errors)
        {
            const auto& entry=u.PageTable[page]; if (entry.X==NaniteInvalid) return false;
            if (entry.X>=u.PhysicalPages.size() || !entry.Y || entry.Z!=u.Constants.Identity.X || entry.W!=u.Constants.Identity.Y)
            { errors|=NaniteGpuStalePage; return false; }
            const auto& identity=u.PhysicalPages[entry.X].Identity;
            if (identity.X!=page || identity.Y!=entry.Y || identity.Z!=entry.Z || identity.W!=entry.W)
            { errors|=NaniteGpuStalePage; return false; }
            return true;
        }
        bool Ancestors(const NaniteGpuUpload& u,unsigned id,const std::vector<unsigned>& state)
        {
            const auto g=Group(u,id);
            for (unsigned at=0;at<u.Metadata[g+3];++at)
            {
                const auto parent=u.Metadata[u.Constants.Offsets.Z+u.Metadata[g+2]+at],consumer=u.Metadata[Cluster(u,parent)+7];
                if (consumer!=NaniteInvalid && !state[consumer*NaniteGpuGroupStateWords+2]) return false;
            }
            return true;
        }
    }

    bool PrepareNaniteGpu(const NaniteHierarchy& h,const NanitePages& pages,const VirtualGeometryResidency& r,const std::vector<LodView>& views,
        double maximumErrorPixels,const GeometryPreparationLimits& limits,NaniteGpuUpload& output,std::string& error)
    {
        if (!ValidateNanitePages(h,pages,error) || !ValidateVirtualGeometryResidency(r,error)) return false;
        if (r.Source()!=pages.Signature || r.PageSlots().size()!=pages.PageCount() || r.PinnedPages()!=pages.PinnedPages || views.empty() || views.size()>NaniteInvalid ||
            !std::isfinite(maximumErrorPixels) || maximumErrorPixels<0 || !limits.MaximumVertices || !limits.MaximumIndices ||
            r.Slots().size()>std::numeric_limits<unsigned>::max()/sizeof(NaniteGpuPhysicalPage))
        { error="Nanite GPU source, cache, view count or budgets are invalid"; return false; }
        NaniteGpuUpload u; u.Source=pages.Signature; auto& c=u.Constants; const GeometryPreparationLimits native;
        c.Counts={unsigned(h.Groups.size()),unsigned(views.size()),pages.PageCount(),unsigned(r.Slots().size())};
        c.Identity={unsigned(r.Epoch()),unsigned(r.Epoch()>>32),unsigned(r.Revision()),unsigned(r.Revision()>>32)};
        c.Quality={Lower(maximumErrorPixels),Upper(pages.MaximumQuantizationDisplacement),0,0};
        c.Capacity={unsigned(std::min(limits.MaximumVertices,native.MaximumVertices)),unsigned(std::min(limits.MaximumIndices,native.MaximumIndices)),unsigned(h.SourceIndices.size()/3),0};
        u.ClusterIds.resize(h.Clusters.size()); std::iota(u.ClusterIds.begin(),u.ClusterIds.end(),0);
        std::sort(u.ClusterIds.begin(),u.ClusterIds.end(),[&](auto a,auto b) { return h.Clusters[a].Material!=h.Clusters[b].Material?h.Clusters[a].Material<h.Clusters[b].Material:a<b; });
        std::vector<unsigned> inverse(h.Clusters.size()),children,parents;
        for (unsigned id=0;id<u.ClusterIds.size();++id) inverse[u.ClusterIds[id]]=id;
        for (unsigned id=0;id<h.Groups.size();++id)
        {
            const auto& g=h.Groups[id]; const auto& range=pages.Groups[id];
            Append(u.Metadata,{unsigned(children.size()),unsigned(g.Children.size()),unsigned(parents.size()),unsigned(g.Parents.size()),range.First,range.Count,g.Level,0});
            for (auto child:g.Children) children.push_back(inverse[child]); for (auto parent:g.Parents) parents.push_back(inverse[parent]);
            const double sum=g.MaximumDisplacement+pages.MaximumQuantizationDisplacement;
            Append(u.Metadata,Float4{g.BoundsMinimum.X,g.BoundsMinimum.Y,g.BoundsMinimum.Z,Upper(sum>0?std::nextafter(sum,std::numeric_limits<double>::infinity()):0)});
            Append(u.Metadata,Float4{g.BoundsMaximum.X,g.BoundsMaximum.Y,g.BoundsMaximum.Z,0});
        }
        c.Offsets.Y=unsigned(u.Metadata.size()); u.Metadata.insert(u.Metadata.end(),children.begin(),children.end());
        c.Offsets.Z=unsigned(u.Metadata.size()); u.Metadata.insert(u.Metadata.end(),parents.begin(),parents.end()); c.Offsets.W=unsigned(u.Metadata.size());
        for (auto id:u.ClusterIds)
        {
            const auto& cluster=h.Clusters[id]; const auto& location=pages.Clusters[id];
            Append(u.Metadata,{location.Page,location.Word,location.VertexCount,unsigned(cluster.Indices.size()),cluster.Material,id,cluster.GeneratingGroup,cluster.ParentGroup});
            Append(u.Metadata,Float4{cluster.BoundsMinimum.X,cluster.BoundsMinimum.Y,cluster.BoundsMinimum.Z,c.Quality.Y});
            Append(u.Metadata,Float4{cluster.BoundsMaximum.X,cluster.BoundsMaximum.Y,cluster.BoundsMaximum.Z,0});
        }
        c.Tables.X=unsigned(u.Metadata.size());
        for (const auto& view:views)
        {
            if (!view.Width || !view.Height) { error="Nanite GPU view has no pixel extent"; return false; }
            for (const auto* matrix:{&view.World,&view.ViewProjection}) for (auto value:matrix->Values)
                if (!std::isfinite(value)) { error="Nanite GPU transform is not finite"; return false; }
            std::array<Float4,8> rows{};
            for (unsigned row=0;row<4;++row) for (unsigned column=0;column<4;++column)
            {
                double value=0,magnitude=0;
                for (unsigned k=0;k<4;++k) { const double term=double(view.ViewProjection.Values[row*4+k])*view.World.Values[k*4+column]; value+=term; magnitude+=std::abs(term); }
                const float coefficient=float(value),uncertainty=Upper(std::abs(double(coefficient)-value)+magnitude*8*std::numeric_limits<double>::epsilon());
                float* coefficients[]{&rows[row].X,&rows[row].Y,&rows[row].Z,&rows[row].W}; float* errors[]{&rows[row+4].X,&rows[row+4].Y,&rows[row+4].Z,&rows[row+4].W};
                *coefficients[column]=coefficient; *errors[column]=uncertainty;
            }
            for (const auto& row:rows) Append(u.Metadata,row);
            Append(u.Metadata,Float4{Upper(double(view.Width)/2),Upper(double(view.Height)/2),0,0});
        }
        c.Tables.Y=unsigned(u.Metadata.size()); c.Tables.W=unsigned(h.Clusters.size());
        std::vector<unsigned> order(h.Groups.size()); std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](auto a,auto b) { return h.Groups[a].Level!=h.Groups[b].Level?h.Groups[a].Level<h.Groups[b].Level:a<b; });
        if (!order.empty()) u.Levels.resize(h.Groups[order.back()].Level+1);
        for (std::size_t begin=0;begin<order.size();)
        {
            auto end=begin+1; while (end<order.size() && h.Groups[order[end]].Level==h.Groups[order[begin]].Level) ++end;
            u.Levels[h.Groups[order[begin]].Level]={unsigned(begin),unsigned(end-begin)}; begin=end;
        }
        u.Metadata.insert(u.Metadata.end(),order.begin(),order.end());
        if (u.Metadata.size()>std::numeric_limits<unsigned>::max()/sizeof(unsigned)) { error="Nanite GPU metadata exceeds byte addressing"; return false; }
        c.Tables.Z=unsigned(u.Metadata.size()); c.Limits.Z=Blocks(c.Tables.W); c.Limits.X=Status(c)+NaniteGpuStatusWords;
        c.Limits.Y=c.Counts.W*NaniteGpuPageWords; c.Limits.W=unsigned(pages.PositionExponent);
        u.InitialState.resize(c.Limits.X); u.PageTable.resize(c.Counts.Z); u.PhysicalPages.resize(c.Counts.W);
        for (auto& physical:u.PhysicalPages) physical.Identity={NaniteInvalid,0,0,0};
        for (unsigned page=0;page<c.Counts.Z;++page)
        {
            const auto slot=r.PageSlots()[page]; auto& entry=u.PageTable[page]; entry={slot,0,c.Identity.X,c.Identity.Y};
            if (slot==NaniteInvalid) continue;
            entry.Y=r.Slots()[slot].Generation; u.PhysicalPages[slot].Identity={page,entry.Y,entry.Z,entry.W};
            if (!ReadNanitePage(pages,page,u.PhysicalPages[slot].Payload,error)) return false;
        }
        u.Signature=Signature(u);
        if (!ValidateNaniteGpuUpload(u,error)) return false;
        output=std::move(u); error.clear(); return true;
    }

    bool ValidateNaniteGpuUpload(const NaniteGpuUpload& u,std::string& error)
    {
        const auto fail=[&](const char* message) { error=message; return false; }; const auto& c=u.Constants;
        const GeometryPreparationLimits native;
        const std::uint64_t stateWords=std::uint64_t(c.Counts.X)*NaniteGpuGroupStateWords+std::uint64_t(c.Tables.W)*NaniteGpuClusterStateWords+
            std::uint64_t(Blocks(c.Tables.W))*NaniteGpuBlockWords+NaniteGpuStatusWords;
        if (!c.Tables.W || !c.Counts.Y || !c.Counts.Z || !c.Counts.W || c.Offsets.X || c.Offsets.Y!=std::uint64_t(c.Counts.X)*NaniteGpuGroupWords ||
            c.Offsets.Z<c.Offsets.Y || c.Offsets.W<c.Offsets.Z || c.Tables.X!=std::uint64_t(c.Offsets.W)+std::uint64_t(c.Tables.W)*NaniteGpuClusterWords ||
            c.Tables.Y!=std::uint64_t(c.Tables.X)+std::uint64_t(c.Counts.Y)*NaniteGpuViewWords || c.Tables.Z!=std::uint64_t(c.Tables.Y)+c.Counts.X ||
            c.Tables.Z!=u.Metadata.size() || c.Tables.W!=u.ClusterIds.size() || c.Counts.Z!=u.PageTable.size() || c.Counts.W!=u.PhysicalPages.size() ||
            c.Limits.X!=stateWords || stateWords!=u.InitialState.size() || stateWords>std::numeric_limits<unsigned>::max()/sizeof(unsigned) ||
            std::uint64_t(c.Counts.W)*NaniteGpuPageWords!=c.Limits.Y || c.Limits.Y>std::numeric_limits<unsigned>::max()/sizeof(unsigned) || c.Limits.Z!=Blocks(c.Tables.W) ||
            !c.Capacity.X || c.Capacity.X>native.MaximumVertices || !c.Capacity.Y || c.Capacity.Y>native.MaximumIndices || !c.Capacity.Z ||
            !std::isfinite(c.Quality.X) || c.Quality.X<0 || !std::isfinite(c.Quality.Y) || c.Quality.Y<0 || c.Quality.Z || c.Quality.W ||
            Signed(c.Limits.W)<-126 || Signed(c.Limits.W)>127 || c.Dispatch.X || c.Dispatch.Y || c.Dispatch.Z || c.Dispatch.W || c.Capacity.W ||
            (!c.Identity.X && !c.Identity.Y) || (!c.Identity.Z && !c.Identity.W)) return fail("Nanite GPU protocol dimensions or constants are invalid");
        // The seal detects accidental changes after preparation, including
        // payload/table changes. It is a content identity, not authentication.
        if (!u.Signature || u.Signature!=Signature(u)) return fail("Nanite GPU upload content changed after preparation");
        if (std::any_of(u.InitialState.begin(),u.InitialState.end(),[](unsigned value) { return value!=0; })) return fail("Nanite GPU initial state is not clear");
        std::vector<bool> ordered(c.Counts.X),clusters(c.Tables.W); unsigned next=0;
        for (unsigned height=0;height<u.Levels.size();++height)
        {
            const auto& level=u.Levels[height]; if (!level.Count) continue;
            if (level.First!=next || std::uint64_t(next)+level.Count>c.Counts.X) return fail("Nanite GPU level range is invalid");
            for (unsigned at=0;at<level.Count;++at)
            {
                const auto id=u.Metadata[c.Tables.Y+next+at];
                if (id>=c.Counts.X || ordered[id] || u.Metadata[Group(u,id)+6]!=height) return fail("Nanite GPU level order is invalid");
                ordered[id]=true;
            }
            next+=level.Count;
        }
        if (next!=c.Counts.X) return fail("Nanite GPU level order omits a group");
        for (unsigned id=0;id<c.Counts.X;++id)
        {
            const auto g=Group(u,id); const auto& m=u.Metadata;
            if (!m[g+1] || !m[g+3] || std::uint64_t(m[g])+m[g+1]>c.Offsets.Z-c.Offsets.Y || std::uint64_t(m[g+2])+m[g+3]>c.Offsets.W-c.Offsets.Z ||
                !m[g+5] || std::uint64_t(m[g+4])+m[g+5]>c.Counts.Z || !m[g+6]) return fail("Nanite GPU group ranges are invalid");
            for (unsigned at=0;at<m[g+1];++at)
            {
                const auto child=m[c.Offsets.Y+m[g]+at]; if (child>=c.Tables.W || m[Cluster(u,child)+7]!=id) return fail("Nanite GPU child ownership is invalid");
                const auto producer=m[Cluster(u,child)+6];
                if (producer!=NaniteInvalid && (producer>=id || m[Group(u,producer)+6]>=m[g+6])) return fail("Nanite GPU group dependency is not topological");
            }
            for (unsigned at=0;at<m[g+3];++at)
            { const auto parent=m[c.Offsets.Z+m[g+2]+at]; if (parent>=c.Tables.W || m[Cluster(u,parent)+6]!=id) return fail("Nanite GPU parent ownership is invalid"); }
        }
        for (unsigned id=0;id<c.Tables.W;++id)
        {
            const auto d=Cluster(u,id); const auto& m=u.Metadata;
            if (u.ClusterIds[id]>=c.Tables.W || clusters[u.ClusterIds[id]] || m[d+5]!=u.ClusterIds[id] || m[d]>=c.Counts.Z ||
                m[d+1]<NanitePageHeaderWords || m[d+1]>NanitePageWords-NaniteClusterHeaderWords || !m[d+2] || m[d+2]>768 || !m[d+3] || m[d+3]%3 || m[d+3]>768 ||
                (m[d+6]!=NaniteInvalid && m[d+6]>=c.Counts.X) || (m[d+7]!=NaniteInvalid && m[d+7]>=c.Counts.X) ||
                (id && m[d+4]<m[d-NaniteGpuClusterWords+4])) return fail("Nanite GPU cluster metadata is invalid");
            clusters[u.ClusterIds[id]]=true; unsigned errors=0;
            const bool resident=Resident(u,m[d],errors);
            if (errors || (m[d+7]==NaniteInvalid && !resident)) return fail("Nanite GPU root payload or physical identity is missing");
        }
        error.clear(); return true;
    }

    NaniteGeometryTask NewNaniteGeometryTask() noexcept
    { static std::atomic<std::uint64_t> next{1}; return {next.fetch_add(1,std::memory_order_relaxed)}; }
    bool BuildNaniteAsyncPlan(const NaniteGpuUpload& upload,NaniteAsyncPlan& output,std::string& error)
    {
        NaniteAsyncPlan plan;
        if (!BuildNaniteGpuSchedule(upload,plan.Dispatches,error)) return false;
        std::uint64_t vertices=0,indices=0;
        constexpr unsigned PageField=0,VertexField=2,IndexField=3;
        for (unsigned id=0;id<upload.Constants.Tables.W;++id)
        {
            const auto offset=Cluster(upload,id); unsigned errors=0;
            if (Resident(upload,upload.Metadata[offset+PageField],errors))
            { vertices+=upload.Metadata[offset+VertexField]; indices+=upload.Metadata[offset+IndexField]; }
        }
        vertices=std::min(vertices,std::uint64_t(upload.Constants.Capacity.X));
        indices=std::min(indices,std::uint64_t(upload.Constants.Capacity.Y));
        if (!vertices || !indices) { error="Nanite async production has no resident output capacity"; return false; }
        for (auto& job:plan.Dispatches) { job.Constants.Capacity.X=unsigned(vertices); job.Constants.Capacity.Y=unsigned(indices); }
        plan.VertexBytes=std::size_t(vertices)*sizeof(Vertex); plan.IndexBytes=std::size_t(indices)*sizeof(unsigned);
        // Charge host snapshots, upload/state resources, staging and publication mirrors.
        // Native backends may use fewer copies, but all share the same hard bound.
        plan.ReservedBytes=(upload.ReadbackGeometry?4:1)*(plan.VertexBytes+plan.IndexBytes)+2*(upload.Metadata.size()*sizeof(unsigned)+
            upload.PageTable.size()*sizeof(ShaderUInt4)+upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage))+
            4*upload.InitialState.size()*sizeof(unsigned)+plan.Dispatches.size()*sizeof(NaniteGpuConstants)*2;
        output=std::move(plan); error.clear(); return true;
    }
    bool BuildNaniteGpuSchedule(const NaniteGpuUpload& u,std::vector<NaniteGpuDispatch>& output,std::string& error)
    {
        if (!ValidateNaniteGpuUpload(u,error)) return false;
        std::vector<NaniteGpuDispatch> jobs;
        const auto add=[&](NaniteGpuPass pass,unsigned first,unsigned count)
        {
            NaniteGpuDispatch job; job.Pass=pass; job.Constants=u.Constants;
            const auto groups=pass==NaniteGpuPass::Decode?count:pass==NaniteGpuPass::Blocks?1:Blocks(count);
            job.GroupsY=unsigned((std::uint64_t(groups)+MaximumDispatchAxis-1)/MaximumDispatchAxis); job.GroupsX=(groups+job.GroupsY-1)/job.GroupsY;
            job.Constants.Dispatch={first,count,job.GroupsX,0}; jobs.push_back(job);
        };
        for (const auto& level:u.Levels) if (level.Count) add(NaniteGpuPass::Evaluate,level.First,level.Count);
        for (auto level=u.Levels.rbegin();level!=u.Levels.rend();++level) if (level->Count) add(NaniteGpuPass::Refine,level->First,level->Count);
        add(NaniteGpuPass::Clusters,0,u.Constants.Tables.W); add(NaniteGpuPass::Blocks,0,u.Constants.Limits.Z); add(NaniteGpuPass::Decode,0,u.Constants.Tables.W);
        output=std::move(jobs); error.clear(); return true;
    }

    bool ValidateNaniteGpuState(const NaniteGpuUpload& u,const std::vector<unsigned>& state,NaniteGpuSelection& output,std::string& error)
    {
        if (!ValidateNaniteGpuUpload(u,error)) return false;
        const auto fail=[&](const char* message) { error=message; return false; }; const auto& c=u.Constants;
        if (state.size()!=c.Limits.X || state[Status(c)]) return fail("Nanite GPU reported invalid feedback or a device-side error");
        NaniteGpuSelection selected;
        for (unsigned id=0;id<c.Counts.X;++id)
        {
            const auto g=Group(u,id),s=id*NaniteGpuGroupStateWords; const auto pixels=Scalar(state[s]); unsigned errors=0; bool complete=true;
            if (std::isnan(pixels) || pixels<0 || state[s+1]>1 || state[s+2]>1 || state[s+3]>1) return fail("Nanite GPU group state is malformed");
            for (unsigned at=0;at<u.Metadata[g+5];++at) complete=Resident(u,u.Metadata[g+4]+at,errors) && complete;
            const bool requested=pixels>c.Quality.X,refined=requested && complete && Ancestors(u,id,state);
            if (errors || state[s+1]!=unsigned(complete) || state[s+2]!=unsigned(refined) || state[s+3]!=unsigned(requested)) return fail("Nanite GPU group transition violates residency or dependency closure");
            for (unsigned at=0;at<u.Metadata[g+1];++at)
            {
                const auto child=u.Metadata[c.Offsets.Y+u.Metadata[g]+at],producer=u.Metadata[Cluster(u,child)+6];
                if (producer!=NaniteInvalid && pixels<Scalar(state[producer*NaniteGpuGroupStateWords])) return fail("Nanite GPU errors are not monotonic");
            }
            if (refined) selected.RefinedGroups.push_back(id);
        }
        unsigned localV=0,localI=0,localA=0;
        for (unsigned id=0;id<c.Tables.W;++id)
        {
            if (id%NaniteGpuWidth==0)
            {
                const auto b=BlockState(c)+(id/NaniteGpuWidth)*NaniteGpuBlockWords;
                if (state[b]!=selected.VertexCount || state[b+1]!=selected.IndexCount || state[b+2]!=selected.ActiveClusters) return fail("Nanite GPU block prefix is inconsistent");
                localV=localI=localA=0;
            }
            const auto d=Cluster(u,id),s=ClusterState(c)+id*NaniteGpuClusterStateWords,producer=u.Metadata[d+6],consumer=u.Metadata[d+7];
            const bool active=(consumer==NaniteInvalid || state[consumer*NaniteGpuGroupStateWords+2]) && (producer==NaniteInvalid || !state[producer*NaniteGpuGroupStateWords+2]);
            if (state[s]!=unsigned(active) || state[s+1]!=localV || state[s+2]!=localI || state[s+3]!=localA) return fail("Nanite GPU cluster prefix is inconsistent");
            if (!active) continue;
            unsigned errors=0; if (!Resident(u,u.Metadata[d],errors) || errors) return fail("Nanite GPU selected a missing page");
            const auto vertices=u.Metadata[d+2],indices=u.Metadata[d+3],material=u.Metadata[d+4];
            if (std::uint64_t(selected.VertexCount)+vertices>c.Capacity.X || std::uint64_t(selected.IndexCount)+indices>c.Capacity.Y) return fail("Nanite GPU drawable counts exceed capacity");
            if (selected.Materials.empty() || selected.Materials.back().Material!=material) selected.Materials.push_back({material,selected.IndexCount,0});
            selected.Materials.back().IndexCount+=indices; selected.VertexCount+=vertices; selected.IndexCount+=indices; ++selected.ActiveClusters;
            selected.Clusters.push_back(u.ClusterIds[id]); localV+=vertices; localI+=indices; ++localA;
        }
        const auto status=Status(c);
        if (!selected.VertexCount || !selected.IndexCount || state[status+1]!=selected.VertexCount || state[status+2]!=selected.IndexCount || state[status+3]!=selected.ActiveClusters)
            return fail("Nanite GPU summary differs from its selected prefix");
        output=std::move(selected); error.clear(); return true;
    }

    bool ValidateNaniteGpuReadback(const NaniteGpuUpload& u,const NaniteGpuReadback& readback,NaniteGpuSelection& output,std::string& error)
    {
        NaniteGpuSelection candidate; if (!ValidateNaniteGpuState(u,readback.State,candidate,error)) return false;
        if (readback.Vertices.size()!=candidate.VertexCount || readback.Indices.size()!=candidate.IndexCount)
        { error="Nanite GPU drawable stream sizes differ from prefix feedback"; return false; }
        if (!ValidateTriangleGeometry(readback.Vertices,readback.Indices,error)) return false;
        output=std::move(candidate); error.clear(); return true;
    }

    bool AdmitNaniteGpuFeedback(const NaniteHierarchy& h,const NanitePages& pages,const VirtualGeometryResidency& r,
        const std::vector<LodView>& views,const NaniteGpuUpload& u,const std::vector<unsigned>& state,NaniteResidentCut& output,std::string& error)
    {
        NaniteGpuSelection selected; if (!ValidateNaniteGpuState(u,state,selected,error)) return false;
        const auto& c=u.Constants;
        if (u.Source!=pages.Signature || u.Source!=r.Source() || c.Counts.X!=h.Groups.size() || c.Tables.W!=h.Clusters.size() ||
            c.Identity.X!=unsigned(r.Epoch()) || c.Identity.Y!=unsigned(r.Epoch()>>32) ||
            c.Identity.Z!=unsigned(r.Revision()) || c.Identity.W!=unsigned(r.Revision()>>32))
        { error="Nanite GPU feedback belongs to another source or residency revision"; return false; }
        std::vector<double> errors(c.Counts.X);
        for (unsigned id=0;id<errors.size();++id) errors[id]=Scalar(state[id*NaniteGpuGroupStateWords]);
        NaniteResidentCut cut;
        if (!SelectNaniteResidentCutFromErrors(h,pages,r,views,errors,c.Quality.X,cut,error)) return false;
        std::sort(selected.Clusters.begin(),selected.Clusters.end());
        if (selected.Clusters!=cut.Clusters || selected.RefinedGroups!=cut.RefinedGroups)
        { error="Nanite GPU resident cut differs from admitted dependency closure"; return false; }
        output=std::move(cut); error.clear(); return true;
    }

}
