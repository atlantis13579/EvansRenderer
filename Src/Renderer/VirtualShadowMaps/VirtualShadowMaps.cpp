#include "Renderer/VirtualShadowMaps/VirtualShadowMaps.h"
#include "Maths/Matrix4.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <numeric>

namespace Evans::Renderer
{
    // Paged shadow maps retain dense virtual addresses while only allocating
    // requested depth tiles. Every admitted fine page retains its ancestors;
    // missing detail therefore resolves through rendered coarse depth, never an
    // uninitialized tile or a fabricated fully-lit result.
    // Native integration order: receiver depth -> demand/ancestor closure ->
    // residency -> copy valid tiles -> clear/draw dirty pages -> shadow lookup.
    // All view captures and transparent/volumetric receivers can use the pinned
    // root when the primary depth view did not request their fine pages.
    // Camera/light projection changes invalidate this projection's page space.
    // Caster changes dirty intersecting old AND new projected bounds, including
    // coarse ancestors. A revision without bounds conservatively dirties all.
    // The depth representation is exclusive with dense/CSM shadow maps; contact
    // shadows remain independent. Cross-page PCF/PCSS lookup requires per-tap
    // indirection. Variance prefiltering needs a separate cross-page moment path.
    // Principles: Karis, "A Deep Dive into Nanite Virtualized Geometry" (2021),
    // Virtual Shadow Maps section; Epic, "Virtual Shadow Maps in Fortnite
    // Battle Royale Chapter 4" (2023), cache invalidation discussion.
    // https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf
    // https://www.unrealengine.com/tech-blog/virtual-shadow-maps-in-fortnite-battle-royale-chapter-4
    // This implementation is independent of Unreal Engine and Nanite.
    namespace
    {
        std::atomic<std::uint64_t> NextAtlasIdentity{1};
        constexpr double HomogeneousEpsilon=1e-7,PageBoundaryRoundoff=1e-6,ClearDepth=1,PixelCenter=0.5,MipBoundaryRoundoff=1e-5;
        bool Finite(const Matrix4& matrix)
        { return std::all_of(matrix.Values.begin(),matrix.Values.end(),[](float x){return std::isfinite(x);}); }
        bool SameLayout(const VirtualShadowLayout& a,const VirtualShadowLayout& b)
        { return a.Resolution==b.Resolution && a.PageSide==b.PageSide; }
        bool Intersects(const VirtualShadowLayout& layout,const VirtualShadowPage& page,const VirtualShadowRectangle& rectangle)
        {
            const double side=layout.PagesPerSide[page.Mip],guard=double(VirtualShadowPageBorder)/(side*layout.PageSide);
            return rectangle.Maximum.X>=double(page.X)/side-guard && rectangle.Minimum.X<=double(page.X+1)/side+guard &&
                rectangle.Maximum.Y>=double(page.Y)/side-guard && rectangle.Minimum.Y<=double(page.Y+1)/side+guard;
        }
    }
    bool BuildVirtualShadowLayout(const Techniques::VirtualShadowSettings& requested,VirtualShadowLayout& output,std::string& error)
    {
        const auto settings=Techniques::SanitizeVirtualShadowSettings(requested);
        if (!(settings==requested)) { error="Virtual shadow layout requires canonical bounded settings"; return false; }
        VirtualShadowLayout result; result.Resolution=settings.Resolution; result.PageSide=settings.PageSide; result.PhysicalPages=settings.PhysicalPages;
        for (unsigned side=result.Resolution/result.PageSide;side;side/=2)
        { result.MipOffsets[result.MipCount]=result.PageCount; result.PagesPerSide[result.MipCount++]=side; result.PageCount+=side*side; }
        result.AtlasColumns=1;
        while (result.AtlasColumns*result.AtlasColumns<result.PhysicalPages) ++result.AtlasColumns;
        result.AtlasRows=(result.PhysicalPages+result.AtlasColumns-1)/result.AtlasColumns;
        const auto stride=result.PageSide+VirtualShadowPageBorder*2;
        result.AtlasWidth=result.AtlasColumns*stride; result.AtlasHeight=result.AtlasRows*stride;
        output=result; error.clear(); return true;
    }
    bool DecodeVirtualShadowPage(const VirtualShadowLayout& layout,std::uint32_t id,VirtualShadowPage& output) noexcept
    {
        if (id>=layout.PageCount || layout.MipCount>VirtualShadowMaximumMipCount) return false;
        for (unsigned mip=0;mip<layout.MipCount;++mip)
        {
            const auto side=layout.PagesPerSide[mip],first=layout.MipOffsets[mip];
            if (side && id>=first && id-first<side*side) { output={id,mip,(id-first)%side,(id-first)/side}; return true; }
        }
        return false;
    }
    std::uint32_t VirtualShadowParent(const VirtualShadowLayout& layout,const VirtualShadowPage& page) noexcept
    { return page.Mip+1>=layout.MipCount?layout.Root():layout.MipOffsets[page.Mip+1]+(page.Y/2)*layout.PagesPerSide[page.Mip+1]+page.X/2; }
    bool BuildVirtualShadowRequestConstants(const VirtualShadowRequestFrame& frame,VirtualShadowRequestConstants& output,std::string& error)
    {
        VirtualShadowLayout layout;
        if (!BuildVirtualShadowLayout(frame.Settings,layout,error)) return false;
        if (!frame.Width || !frame.Height || frame.Width>Techniques::VirtualShadowMaximumResolution || frame.Height>Techniques::VirtualShadowMaximumResolution ||
            std::uint64_t(frame.Width)*frame.Height>VirtualShadowMaximumReceiverPixels || frame.FilterRadius>VirtualShadowMaximumRequestRadius ||
            !Finite(frame.CameraViewProjection) || !Finite(frame.LightViewProjection))
        { error="Virtual shadow demand requires bounded receiver dimensions, filter footprint and finite cameras"; return false; }
        const unsigned outputWidth=frame.OutputWidth?frame.OutputWidth:frame.Width,outputHeight=frame.OutputHeight?frame.OutputHeight:frame.Height;
        if (bool(frame.OutputWidth)!=bool(frame.OutputHeight) || outputWidth>Techniques::VirtualShadowMaximumResolution || outputHeight>Techniques::VirtualShadowMaximumResolution ||
            std::uint64_t(outputWidth)*outputHeight>VirtualShadowMaximumReceiverPixels)
        { error="Virtual shadow quality viewport must be bounded and specify both output dimensions"; return false; }
        Maths::Matrix4 inverse,lightInverse;
        if (!Maths::Matrix4(frame.CameraViewProjection.Values).Inverse(inverse) || !Maths::Matrix4(frame.LightViewProjection.Values).Inverse(lightInverse))
        { error="Virtual shadow demand requires invertible camera and light projections"; return false; }
        VirtualShadowRequestConstants result; std::copy(inverse.Data(),inverse.Data()+result.CameraInverse.Values.size(),result.CameraInverse.Values.begin()); result.LightViewProjection=frame.LightViewProjection;
        if (!Finite(result.CameraInverse)) { error="Virtual shadow inverse cannot be represented by the native float interface"; return false; }
        result.Extent={frame.Width,frame.Height,layout.Resolution,layout.PageSide}; result.Limits={layout.MipCount,layout.PageCount,frame.FilterRadius,VirtualShadowGroupColumns*VirtualShadowGroupWidth};
        result.Quality={frame.Settings.MipBias,float(HomogeneousEpsilon),float(PageBoundaryRoundoff),float(MipBoundaryRoundoff)};
        result.Sampling={float(frame.Width)/outputWidth,float(frame.Height)/outputHeight,0,0}; output=result; error.clear(); return true;
    }
    bool ValidateVirtualShadowRequestUpload(const VirtualShadowRequestUpload& upload,std::string& error)
    {
        VirtualShadowRequestConstants constants;
        if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,error)) return false;
        if (upload.Depth.size()!=std::size_t(upload.Frame.Width)*upload.Frame.Height ||
            std::any_of(upload.Depth.begin(),upload.Depth.end(),[](float depth){return !std::isfinite(depth) || depth<0 || depth>ClearDepth;}))
        { error="Virtual shadow receiver upload requires one finite [0,1] hardware-depth value per pixel"; return false; }
        error.clear(); return true;
    }
    bool ValidateVirtualShadowRequests(const VirtualShadowLayout& layout,const std::vector<std::uint32_t>& requests,std::string& error)
    {
        if (requests.size()!=layout.PageCount || !layout.PageCount || std::any_of(requests.begin(),requests.end(),[](auto x){return x>1;}))
        { error="Virtual shadow demand must be a complete binary virtual-page directory"; return false; }
        for (unsigned id=0;id<requests.size();++id) if (requests[id])
        {
            VirtualShadowPage page;
            if (!DecodeVirtualShadowPage(layout,id,page) || !requests[VirtualShadowParent(layout,page)])
            { error="Virtual shadow demand omitted a required coarse ancestor"; return false; }
        }
        error.clear(); return true;
    }
    bool ValidateVirtualShadowSnapshot(const VirtualShadowSnapshot& snapshot,std::string& error)
    {
        const auto& layout=snapshot.Layout;
        Techniques::VirtualShadowSettings settings; settings.Resolution=layout.Resolution; settings.PageSide=layout.PageSide; settings.PhysicalPages=layout.PhysicalPages;
        VirtualShadowLayout canonical;
        if (!BuildVirtualShadowLayout(settings,canonical,error)) return false;
        if (layout.PageCount!=canonical.PageCount || layout.MipCount!=canonical.MipCount || layout.MipOffsets!=canonical.MipOffsets || layout.PagesPerSide!=canonical.PagesPerSide ||
            layout.AtlasColumns!=canonical.AtlasColumns || layout.AtlasRows!=canonical.AtlasRows || layout.AtlasWidth!=canonical.AtlasWidth || layout.AtlasHeight!=canonical.AtlasHeight ||
            !snapshot.Identity.Epoch || !snapshot.Identity.Serial || snapshot.Identity.PreviousSerial>=snapshot.Identity.Serial || !snapshot.Identity.CasterRevision || !Finite(snapshot.Identity.LightViewProjection) ||
            snapshot.PageTable.size()!=layout.PageCount || snapshot.Residents.empty() || snapshot.Residents.size()>layout.PhysicalPages)
        { error="Virtual shadow snapshot has an invalid layout, identity or residency directory"; return false; }
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(snapshot.Identity.LightViewProjection.Values).Inverse(inverse))
        { error="Virtual shadow snapshot has a singular light projection"; return false; }
        std::vector<bool> slots(layout.PhysicalPages),pages(layout.PageCount);
        for (const auto& resident:snapshot.Residents)
        {
            if (resident.Page>=layout.PageCount || resident.Slot>=layout.PhysicalPages || pages[resident.Page] || slots[resident.Slot] ||
                !resident.LastRequested || resident.LastRequested>snapshot.Identity.Serial || snapshot.PageTable[resident.Page]!=resident.Slot+1)
            { error="Virtual shadow snapshot contains aliased/stale physical slots or page addresses"; return false; }
            pages[resident.Page]=slots[resident.Slot]=true;
            VirtualShadowPage page; if (!DecodeVirtualShadowPage(layout,resident.Page,page) || !snapshot.PageTable[VirtualShadowParent(layout,page)])
            { error="A resident virtual shadow page lost its fallback ancestor"; return false; }
        }
        if (!pages[layout.Root()]) { error="Virtual shadow root must remain pinned"; return false; }
        for (unsigned id=0;id<layout.PageCount;++id) if (bool(snapshot.PageTable[id])!=bool(pages[id]))
        { error="Virtual shadow page table does not match the resident tile set"; return false; }
        error.clear(); return true;
    }
    bool BuildVirtualShadowPlan(const VirtualShadowIdentity& identity,const Techniques::VirtualShadowSettings& settings,
        const std::vector<std::uint32_t>& requests,const std::vector<VirtualShadowRectangle>& invalidations,const VirtualShadowSnapshot* previous,
        VirtualShadowPlan& output,std::string& error)
    {
        VirtualShadowLayout layout;
        if (!BuildVirtualShadowLayout(settings,layout,error) || !ValidateVirtualShadowRequests(layout,requests,error)) return false;
        if (!identity.Epoch || !identity.Serial || !identity.CasterRevision || !Finite(identity.LightViewProjection) ||
            std::any_of(invalidations.begin(),invalidations.end(),[](const auto& r){return !std::isfinite(r.Minimum.X) || !std::isfinite(r.Minimum.Y) || !std::isfinite(r.Maximum.X) || !std::isfinite(r.Maximum.Y) || r.Minimum.X>r.Maximum.X || r.Minimum.Y>r.Maximum.Y;}))
        { error="Virtual shadow plan requires a versioned projection and finite ordered invalidation bounds"; return false; }
        if (previous && !ValidateVirtualShadowSnapshot(*previous,error)) return false;
        VirtualShadowPlan result; result.Candidate.Layout=layout; result.Candidate.Identity=identity; result.Candidate.PageTable.resize(layout.PageCount);
        result.ParentCompatible=previous && previous->Identity.Epoch==identity.Epoch && SameLayout(layout,previous->Layout) && previous->Identity.LightViewProjection.Values==identity.LightViewProjection.Values;
        if (previous && previous->Identity.Epoch==identity.Epoch && (identity.PreviousSerial!=previous->Identity.Serial || identity.Serial<=previous->Identity.Serial || identity.CasterRevision<previous->Identity.CasterRevision))
        { error="Virtual shadow candidate does not continue the previous presented version"; return false; }
        if (result.ParentCompatible && identity.CasterRevision==previous->Identity.CasterRevision && !invalidations.empty())
        { error="Caster invalidation bounds require a new caster revision"; return false; }
        result.SourcePhysicalPages=result.ParentCompatible?previous->Layout.PhysicalPages:0;
        std::vector<const VirtualShadowSlot*> old(layout.PageCount);
        if (result.ParentCompatible) for (const auto& resident:previous->Residents) old[resident.Page]=&resident;
        std::vector<unsigned> order;
        for (unsigned id=0;id<layout.PageCount;++id)
            if (id==layout.Root() || requests[id] || (old[id] && identity.Serial-old[id]->LastRequested<=settings.MaximumUnusedFrames)) order.push_back(id);
        std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b)
        {
            const bool requestA=a==layout.Root() || requests[a],requestB=b==layout.Root() || requests[b];
            if (requestA!=requestB) return requestA;
            VirtualShadowPage pa,pb; (void)DecodeVirtualShadowPage(layout,a,pa); (void)DecodeVirtualShadowPage(layout,b,pb);
            if (pa.Mip!=pb.Mip) return pa.Mip>pb.Mip;
            if (bool(old[a])!=bool(old[b])) return bool(old[a]);
            if (old[a] && old[a]->LastRequested!=old[b]->LastRequested) return old[a]->LastRequested>old[b]->LastRequested;
            return a<b;
        });
        result.RequestedPages=static_cast<std::uint32_t>(std::count(requests.begin(),requests.end(),1u));
        for (auto id:order)
        {
            if (result.Candidate.Residents.size()==layout.PhysicalPages) break;
            VirtualShadowPage page; (void)DecodeVirtualShadowPage(layout,id,page);
            if (id!=layout.Root() && !result.Candidate.PageTable[VirtualShadowParent(layout,page)]) continue;
            const unsigned slot=static_cast<unsigned>(result.Candidate.Residents.size());
            const bool requested=id==layout.Root() || requests[id];
            VirtualShadowSlot resident{id,slot,requested?identity.Serial:old[id]->LastRequested};
            result.Candidate.PageTable[id]=slot+1; result.Candidate.Residents.push_back(resident);
            bool dirty=!old[id];
            if (!dirty && identity.CasterRevision!=previous->Identity.CasterRevision)
                dirty=invalidations.empty() || std::any_of(invalidations.begin(),invalidations.end(),[&](const auto& rectangle){return Intersects(layout,page,rectangle);});
            if (dirty) result.Dirty.push_back(resident); else result.Copies.push_back({old[id]->Slot,slot});
        }
        for (unsigned id=0;id<layout.PageCount;++id) result.UnservedPages+=requests[id] && !result.Candidate.PageTable[id];
        if (!ValidateVirtualShadowSnapshot(result.Candidate,error)) return false;
        output=std::move(result); error.clear(); return true;
    }
    bool CompleteVirtualShadowPlan(const VirtualShadowPlan& plan,const std::vector<std::uint32_t>& renderedPages,VirtualShadowSnapshot& output,std::string& error)
    {
        if (!ValidateVirtualShadowSnapshot(plan.Candidate,error)) return false;
        std::vector<unsigned> expected,actual=renderedPages;
        for (const auto& dirty:plan.Dirty) expected.push_back(dirty.Page);
        std::sort(expected.begin(),expected.end()); std::sort(actual.begin(),actual.end());
        if (actual!=expected || plan.Dirty.size()+plan.Copies.size()!=plan.Candidate.Residents.size())
        { error="Virtual shadow candidate cannot publish before every dirty tile completes exactly once"; return false; }
        std::vector<bool> written(plan.Candidate.Layout.PhysicalPages);
        for (const auto& dirty:plan.Dirty)
        {
            if (dirty.Slot>=written.size() || dirty.Page>=plan.Candidate.PageTable.size() || written[dirty.Slot] || plan.Candidate.PageTable[dirty.Page]!=dirty.Slot+1)
            { error="Virtual shadow render completion aliases a physical slot or names an unmapped page"; return false; }
            written[dirty.Slot]=true;
        }
        for (const auto& copy:plan.Copies)
        {
            if (!plan.ParentCompatible || copy.Source>=plan.SourcePhysicalPages || copy.Destination>=written.size() || written[copy.Destination])
            { error="Virtual shadow retained-tile copy is not a bounded disjoint parent-to-candidate operation"; return false; }
            written[copy.Destination]=true;
        }
        for (const auto& resident:plan.Candidate.Residents) if (!written[resident.Slot])
        { error="Virtual shadow candidate contains a resident tile with neither copied nor rendered depth"; return false; }
        output=plan.Candidate; error.clear(); return true;
    }
    bool FindVirtualShadowPage(const VirtualShadowSnapshot& snapshot,Float2 uv,std::uint32_t requestedMip,VirtualShadowPage& output,std::uint32_t& physicalSlot) noexcept
    {
        const auto& layout=snapshot.Layout;
        if (!std::isfinite(uv.X) || !std::isfinite(uv.Y) || uv.X<0 || uv.X>1 || uv.Y<0 || uv.Y>1 || requestedMip>=layout.MipCount || snapshot.PageTable.size()!=layout.PageCount) return false;
        for (unsigned mip=requestedMip;mip<layout.MipCount;++mip)
        {
            const auto side=layout.PagesPerSide[mip]; if (!side) return false;
            const auto x=std::min(unsigned(double(uv.X)*side),side-1),y=std::min(unsigned(double(uv.Y)*side),side-1);
            const auto id=layout.MipOffsets[mip]+y*side+x;
            if (id>=snapshot.PageTable.size()) return false;
            if (snapshot.PageTable[id]) { if (snapshot.PageTable[id]>layout.PhysicalPages) return false; output={id,mip,x,y}; physicalSlot=snapshot.PageTable[id]-1; return true; }
        }
        return false;
    }
    Float2 VirtualShadowAtlasUV(const VirtualShadowLayout& layout,const VirtualShadowPage& page,std::uint32_t slot,Float2 uv) noexcept
    {
        if (page.Mip>=layout.MipCount || slot>=layout.PhysicalPages || !layout.AtlasColumns || !layout.AtlasWidth || !layout.AtlasHeight) return {};
        const double side=layout.PagesPerSide[page.Mip],stride=layout.PageSide+VirtualShadowPageBorder*2;
        const double mipResolution=side*layout.PageSide;
        const double virtualX=std::clamp(double(uv.X)*mipResolution,PixelCenter,mipResolution-PixelCenter);
        const double virtualY=std::clamp(double(uv.Y)*mipResolution,PixelCenter,mipResolution-PixelCenter);
        const double x=(slot%layout.AtlasColumns)*stride+VirtualShadowPageBorder+virtualX-page.X*layout.PageSide;
        const double y=(slot/layout.AtlasColumns)*stride+VirtualShadowPageBorder+virtualY-page.Y*layout.PageSide;
        return {float(x/layout.AtlasWidth),float(y/layout.AtlasHeight)};
    }
    Matrix4 VirtualShadowPageProjection(const VirtualShadowLayout& layout,const VirtualShadowPage& page,const Matrix4& light) noexcept
    {
        Matrix4 result=light;
        if (page.Mip>=layout.MipCount || !layout.PageSide) return result;
        const double fraction=double(layout.PageSide)/(layout.PageSide+VirtualShadowPageBorder*2),side=layout.PagesPerSide[page.Mip];
        const float scale=float(side*fraction),offsetX=float((side-2*page.X-1)*fraction),offsetY=float((1-side+2*page.Y)*fraction);
        for (unsigned column=0;column<4;++column)
        { result.Values[column]=scale*light.Values[column]+offsetX*light.Values[12+column]; result.Values[4+column]=scale*light.Values[4+column]+offsetY*light.Values[12+column]; }
        return result;
    }
    bool ValidateVirtualShadowCaster(const VirtualShadowCaster& caster,const BufferDesc& vertices,const BufferDesc& indices,
        const std::vector<std::uint8_t>& mirror,bool naniteGeometry,std::string& error)
    {
        if (!caster.Vertices || !caster.Indices || !caster.IndexCount || caster.IndexCount%3 ||
            vertices.Usage!=BufferUsage::Vertex || vertices.SizeInBytes%sizeof(Vertex) || !vertices.SizeInBytes ||
            indices.Usage!=BufferUsage::Index || indices.SizeInBytes%sizeof(std::uint32_t) ||
            std::uint64_t(caster.FirstIndex)+caster.IndexCount>indices.SizeInBytes/sizeof(std::uint32_t) || !Finite(caster.World) ||
            !std::isfinite(caster.Coverage) || caster.Coverage<0 || caster.Coverage>1 || !std::isfinite(caster.Cutoff) || caster.Cutoff<0 || caster.Cutoff>1)
        { error="Virtual shadow caster needs bounded indexed triangles, finite transforms and canonical material coverage"; return false; }
        // The bounded Nanite GPU decoder publishes native vertex/index buffers
        // without CPU mirrors. Their backend-owned provenance permits direct
        // shadow rasterization; ordinary uploaded indices still need validation.
        if (mirror.empty() && naniteGeometry) { error.clear(); return true; }
        if (mirror.size()!=indices.SizeInBytes)
        { error="Virtual shadow uploaded indices require a complete CPU mirror"; return false; }
        for (std::size_t offset=std::size_t(caster.FirstIndex)*sizeof(std::uint32_t);offset<std::size_t(caster.FirstIndex+std::uint64_t(caster.IndexCount))*sizeof(std::uint32_t);offset+=sizeof(std::uint32_t))
        {
            std::uint32_t index=0; std::memcpy(&index,mirror.data()+offset,sizeof(index));
            if (index>=vertices.SizeInBytes/sizeof(Vertex)) { error="Virtual shadow caster index exceeds its native vertex buffer"; return false; }
        }
        error.clear(); return true;
    }
    bool BuildVirtualShadowRasterConstants(const VirtualShadowSnapshot& snapshot,const VirtualShadowSlot& resident,const VirtualShadowCaster& caster,
        VirtualShadowRasterConstants& output,std::string& error)
    {
        const auto& layout=snapshot.Layout; VirtualShadowPage page;
        if (!DecodeVirtualShadowPage(layout,resident.Page,page) || resident.Slot>=layout.PhysicalPages || !layout.AtlasColumns ||
            snapshot.PageTable.size()!=layout.PageCount || snapshot.PageTable[resident.Page]!=resident.Slot+1 || !Finite(caster.World))
        { error="Virtual shadow draw does not address a resident candidate tile"; return false; }
        VirtualShadowRasterConstants result;
        const auto projection=VirtualShadowPageProjection(layout,page,snapshot.Identity.LightViewProjection);
        const auto matrix=Maths::Matrix4(projection.Values)*Maths::Matrix4(caster.World.Values);
        std::copy(matrix.Data(),matrix.Data()+result.WorldViewProjection.Values.size(),result.WorldViewProjection.Values.begin());
        if (!Finite(result.WorldViewProjection)) { error="Virtual shadow caster projection exceeds native float precision"; return false; }
        result.VegetationWorld=caster.World; result.VegetationNormalWorld=BuildRealtimeNormalMatrix(caster.World);
        result.Wind=caster.Wind; result.Gust=caster.Gust; result.VegetationHistory=caster.VegetationHistory;
        result.Material={caster.Coverage,caster.Cutoff,caster.Dithered?1.0f:0.0f,0}; result.Textures={caster.Diffuse?1u:0u,caster.Opacity?1u:0u,0,0};
        const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
        result.Origins={float((resident.Slot%layout.AtlasColumns)*stride),float((resident.Slot/layout.AtlasColumns)*stride),
            float(int(page.X*layout.PageSide)-int(VirtualShadowPageBorder)),float(int(page.Y*layout.PageSide)-int(VirtualShadowPageBorder))};
        result.Extent={layout.AtlasWidth,layout.AtlasHeight,0,0}; output=result; error.clear(); return true;
    }
    bool BuildVirtualShadowReceiverConstants(const VirtualShadowRequestFrame& frame,const VirtualShadowCaster& caster,
        VirtualShadowRasterConstants& output,std::string& error)
    {
        VirtualShadowRasterConstants result;
        const auto matrix=Maths::Matrix4(frame.CameraViewProjection.Values)*Maths::Matrix4(caster.World.Values);
        std::copy(matrix.Data(),matrix.Data()+result.WorldViewProjection.Values.size(),result.WorldViewProjection.Values.begin());
        if (!Finite(result.WorldViewProjection)) { error="Virtual shadow receiver projection exceeds native float precision"; return false; }
        result.VegetationWorld=caster.World; result.VegetationNormalWorld=BuildRealtimeNormalMatrix(caster.World);
        result.Wind=caster.Wind; result.Gust=caster.Gust; result.VegetationHistory=caster.VegetationHistory;
        result.Material={caster.Coverage,caster.Cutoff,caster.Dithered?1.0f:0.0f,0};
        result.Textures={caster.Diffuse?1u:0u,caster.Opacity?1u:0u,0,0};
        result.Origins={0,0,0,0}; result.Extent={frame.Width,frame.Height,0,0};
        output=result; error.clear(); return true;
    }
    VirtualShadowRasterConstants BuildVirtualShadowCopyConstants(const VirtualShadowLayout& source,const VirtualShadowLayout& destination,const VirtualShadowCopy& copy) noexcept
    {
        VirtualShadowRasterConstants result; const auto stride=destination.PageSide+VirtualShadowPageBorder*2;
        if (!source.AtlasColumns || !destination.AtlasColumns) return result;
        result.Origins={float((copy.Destination%destination.AtlasColumns)*stride),float((copy.Destination/destination.AtlasColumns)*stride),
            float((copy.Source%source.AtlasColumns)*stride),float((copy.Source/source.AtlasColumns)*stride)};
        result.Extent={destination.AtlasWidth,destination.AtlasHeight,source.AtlasWidth,source.AtlasHeight}; return result;
    }
    bool ValidateVirtualShadowAtlasDepth(const VirtualShadowSnapshot& snapshot,const std::vector<float>& depth,std::string& error)
    {
        if (!ValidateVirtualShadowSnapshot(snapshot,error)) return false;
        if (depth.size()!=std::size_t(snapshot.Layout.AtlasWidth)*snapshot.Layout.AtlasHeight ||
            std::any_of(depth.begin(),depth.end(),[](float value){return !std::isfinite(value) || value<0 || value>1;}))
        { error="Virtual shadow atlas readback must contain finite hardware depth over its complete physical extent"; return false; }
        error.clear(); return true;
    }
    bool BuildVirtualShadowDirectory(const VirtualShadowSnapshot& snapshot,VirtualShadowDirectory& output,std::string& error)
    {
        if (!ValidateVirtualShadowSnapshot(snapshot,error)) return false;
        VirtualShadowDirectory result; result.Width=std::min(VirtualShadowDirectoryWidth,snapshot.Layout.PageCount);
        result.Height=(snapshot.Layout.PageCount+result.Width-1)/result.Width; result.Denominator=snapshot.Layout.PhysicalPages+1;
        result.Depth.resize(std::size_t(result.Width)*result.Height,0);
        for (std::size_t page=0;page<snapshot.PageTable.size();++page)
            result.Depth[page]=float(snapshot.PageTable[page])/result.Denominator;
        output=std::move(result); return true;
    }
    bool BuildVirtualShadowLookupConstants(const VirtualShadowSnapshot& snapshot,const VirtualShadowLookupUpload& upload,
        VirtualShadowLookupConstants& output,std::string& error)
    {
        if (!ValidateVirtualShadowSnapshot(snapshot,error)) return false;
        if (upload.Samples.empty() || upload.Samples.size()>VirtualShadowMaximumLookupQueries || upload.Radius>VirtualShadowMaximumLookupRadius)
        { error="Virtual shadow lookup requires bounded nonempty queries and filter radius"; return false; }
        for (const auto& sample:upload.Samples)
            if (!std::isfinite(sample.X) || !std::isfinite(sample.Y) || !std::isfinite(sample.Z) || !std::isfinite(sample.W) ||
                sample.W<0 || sample.W>=snapshot.Layout.MipCount || std::floor(sample.W)!=sample.W)
            { error="Virtual shadow queries require finite UV/depth and an integer virtual mip"; return false; }
        const auto& layout=snapshot.Layout; const auto width=upload.Radius*2+1; VirtualShadowLookupConstants result;
        result.LightViewProjection=snapshot.Identity.LightViewProjection;
        result.VirtualExtent={layout.Resolution,layout.PageSide,layout.MipCount,std::min(VirtualShadowDirectoryWidth,layout.PageCount)};
        result.PhysicalExtent={layout.AtlasWidth,layout.AtlasHeight,layout.AtlasColumns,layout.PhysicalPages+1};
        result.Filtering={upload.Radius,width*width,static_cast<unsigned>(upload.Samples.size()),VirtualShadowGroupColumns*VirtualShadowGroupWidth};
        output=result; error.clear(); return true;
    }
    bool ValidateVirtualShadowLookupResult(const VirtualShadowSnapshot& snapshot,const VirtualShadowLookupUpload& upload,const VirtualShadowLookupResult& output,std::string& error)
    {
        if (output.Samples.size()!=upload.Samples.size()) { error="Virtual shadow lookup feedback has an incomplete query range"; return false; }
        for (std::size_t index=0;index<output.Samples.size();++index)
        {
            const auto& sample=output.Samples[index]; const auto& query=upload.Samples[index];
            const bool outside=query.X<0 || query.Y<0 || query.X>1 || query.Y>1 || query.Z<0 || query.Z>1;
            if (!std::isfinite(sample.X) || !std::isfinite(sample.Y) || !std::isfinite(sample.Z) || !std::isfinite(sample.W) ||
                sample.X<0 || sample.X>1 || sample.Y<0 || sample.Y>1 ||
                (outside?(sample.X!=1 || sample.Y!=1 || sample.Z!=-1 || sample.W!=0):
                (sample.Z<query.W || sample.Z>=snapshot.Layout.MipCount || std::floor(sample.Z)!=sample.Z || sample.W<1 ||
                 sample.W>snapshot.Layout.PhysicalPages || std::floor(sample.W)!=sample.W)))
            { error="Virtual shadow lookup returned noncanonical visibility, depth or residency"; return false; }
        }
        error.clear(); return true;
    }
    VirtualShadowAtlasHandle NewVirtualShadowAtlasHandle() noexcept
    {
        auto value=NextAtlasIdentity.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX)
            if (NextAtlasIdentity.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
