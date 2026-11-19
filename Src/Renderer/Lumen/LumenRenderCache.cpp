#include "Renderer/Lumen/LumenRenderCache.h"
#include "Renderer/IRHI.h"
#include "Renderer/Lighting/SceneLightCollection.h"

#include <atomic>
#include <chrono>
#include <future>
#include <optional>

namespace Evans::Renderer
{
    namespace
    {
        std::atomic<std::uint64_t> NextLumenRuntimeEpoch{1};
        bool SameCardPreparation(const Techniques::LumenSettings& a,const Techniques::LumenSettings& b)
        { return a.CardSide==b.CardSide && a.RefineCards==b.RefineCards && a.MaximumMeshCards==b.MaximumMeshCards && a.CardCoverage==b.CardCoverage; }
        bool SameMaterials(const Techniques::Settings& a,const Techniques::Settings& b)
        {
            return a.TextureMapping==b.TextureMapping && a.NormalMapping==b.NormalMapping && a.AlphaTesting==b.AlphaTesting &&
                a.GlobalIllumination.Lumen.HardwareRayTracing==b.GlobalIllumination.Lumen.HardwareRayTracing &&
                a.AlphaBlending==b.AlphaBlending && a.Shading==b.Shading && a.Metallic==b.Metallic && a.Roughness==b.Roughness &&
                a.BackFaceCulling==b.BackFaceCulling && a.GlobalIllumination.Lumen.HairTransport==b.GlobalIllumination.Lumen.HairTransport &&
                a.GlobalIllumination.Lumen.HairGridSide==b.GlobalIllumination.Lumen.HairGridSide && a.Hair.RootTint==b.Hair.RootTint && a.Hair.TipTint==b.Hair.TipTint &&
                a.GlobalIllumination.Lumen.SourceResolution==b.GlobalIllumination.Lumen.SourceResolution &&
                SameCardPreparation(a.GlobalIllumination.Lumen,b.GlobalIllumination.Lumen);
        }
        bool SameProxy(const LumenSourceInstance& a,const LumenSourceInstance& b)
        {
            return a.Geometry==b.Geometry && a.Material.DiffuseImage==b.Material.DiffuseImage && a.Material.NormalImage==b.Material.NormalImage &&
                a.Material.OpacityImage==b.Material.OpacityImage && a.Material.AlphaCutoff==b.Material.AlphaCutoff &&
                a.Material.BaseColor.W==b.Material.BaseColor.W && a.Color.W==b.Color.W;
        }
        struct Prepared
        {
            std::vector<LumenSourceInstance> Sources;
            std::vector<std::shared_ptr<const LumenMeshRepresentation>> Meshes;
            std::vector<unsigned> SourceGroups,MaximumParts,MaximumCards;
            Techniques::Settings Settings;
            std::uint64_t Generation=0,BuildCount=0;
            std::shared_ptr<const HairStrandGeometry> Hair,HairBase;
            std::shared_ptr<const LumenHairVolume> HairVolume;
            std::string Error;
        };
        struct LumenBuildRequest
        {
            std::vector<LumenSourceInstance> Sources;
            Techniques::Settings Settings;
            std::shared_ptr<const HairStrandGeometry> Hair,HairBase;
            std::uint64_t Generation=0,Revision=0;
            std::shared_ptr<std::atomic<bool>> Cancelled=std::make_shared<std::atomic<bool>>(false);
        };
        bool SamePreparation(const std::vector<LumenSourceInstance>& sources,const Techniques::Settings& settings,const std::shared_ptr<const HairStrandGeometry>& base,const LumenBuildRequest& request)
        {
            if (base!=request.HairBase || !SameMaterials(settings,request.Settings) || sources.size()!=request.Sources.size()) return false;
            for (unsigned i=0;i<sources.size();++i)
                if (sources[i].Key!=request.Sources[i].Key || !SameProxy(sources[i],request.Sources[i])) return false;
            return true;
        }
        bool SamePoseFamily(const std::vector<LumenSourceInstance>& sources,const Techniques::Settings& settings,const std::shared_ptr<const HairStrandGeometry>& base,const Prepared& previous)
        {
            if (base!=previous.HairBase || !SameMaterials(settings,previous.Settings) || sources.size()!=previous.Sources.size()) return false;
            const auto same3=[](Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; };
            const auto same4=[](Float4 a,Float4 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; };
            for (unsigned i=0;i<sources.size();++i)
            {
                const auto& a=sources[i]; const auto& b=previous.Sources[i]; const auto& x=a.Capture; const auto& y=b.Capture;
                if (a.Key!=b.Key || a.CastShadow!=b.CastShadow || !SameProxy(a,b) || !same4(a.Color,b.Color) ||
                    !same4(x.BaseColor,y.BaseColor) || !same3(x.Emission,y.Emission) || x.Roughness!=y.Roughness || x.Metallic!=y.Metallic ||
                    x.Cutoff!=y.Cutoff || x.BackFaceCulling!=y.BackFaceCulling || !(x.Diffuse==y.Diffuse) || !(x.Normal==y.Normal) || !(x.Opacity==y.Opacity)) return false;
            }
            return true;
        }
        Prepared Prepare(const LumenBuildRequest& request,const std::shared_ptr<const Prepared>& previous)
        {
            Prepared output; output.Generation=request.Generation; output.Sources=request.Sources; output.Settings=request.Settings;
            output.Hair=request.Hair; output.HairBase=request.HairBase;
            const auto cancelled=[&] { return request.Cancelled->load(std::memory_order_relaxed); };
            std::vector<unsigned> representatives;
            std::vector<LumenSceneMeshDemand> demand;
            for (unsigned i=0;i<output.Sources.size();++i)
            {
                if (cancelled()) return output;
                if (!output.Sources[i].Geometry) { output.Error="Lumen source geometry is missing"; return output; }
                unsigned group=0; for (;group<representatives.size();++group) if (SameProxy(output.Sources[i],output.Sources[representatives[group]])) break;
                if (group==representatives.size())
                { representatives.push_back(i); demand.push_back({output.Sources[i].Key,0,64,output.Sources[i].Geometry->Indices.size()/3}); }
                ++demand[group].Instances; output.SourceGroups.push_back(group);
            }
            if (demand.empty()) { output.Error="No opaque Lumen transport geometry"; return output; }
            const auto& lumen=request.Settings.GlobalIllumination.Lumen;
            LumenMeshBuildSettings settings; settings.Distance.Resolution=lumen.SourceResolution;
            settings.HardwareRayTracing=lumen.HardwareRayTracing;
            settings.Cards.CaptureResolution=lumen.CardSide; settings.Cards.RefineCoverage=lumen.RefineCards;
            settings.Cards.MaximumCards=lumen.MaximumMeshCards; settings.Cards.TargetCoverage=lumen.CardCoverage;
            std::vector<LumenMeshBuildSettings> allocation;
            if (!AllocateLumenSceneMeshParts(demand,settings,request.Settings.GlobalIllumination.Lumen.CardSide,{},allocation,output.Error)) return output;
            for (unsigned group=0;group<representatives.size();++group)
            {
                if (cancelled()) return output;
                const auto& source=output.Sources[representatives[group]]; std::shared_ptr<const LumenMeshRepresentation> mesh;
                // World transforms and capture-only color edits reuse local SDFs.
                if (previous && previous->Settings.TextureMapping==request.Settings.TextureMapping && previous->Settings.NormalMapping==request.Settings.NormalMapping &&
                    previous->Settings.AlphaTesting==request.Settings.AlphaTesting && previous->Settings.AlphaBlending==request.Settings.AlphaBlending &&
                    previous->Settings.GlobalIllumination.Lumen.HardwareRayTracing==lumen.HardwareRayTracing &&
                    previous->Settings.GlobalIllumination.Lumen.SourceResolution==settings.Distance.Resolution &&
                    SameCardPreparation(previous->Settings.GlobalIllumination.Lumen,lumen))
                    for (unsigned old=0;old<previous->Sources.size();++old)
                    {
                        const auto oldGroup=previous->SourceGroups[old];
                        if (SameProxy(source,previous->Sources[old]) && previous->MaximumParts[oldGroup]==allocation[group].Partition.MaximumParts &&
                            previous->MaximumCards[oldGroup]==allocation[group].Cards.MaximumCards)
                        { mesh=previous->Meshes[oldGroup]; break; }
                    }
                if (!mesh)
                {
                    SceneRayMeshView view; view.Vertices=&source.Geometry->Vertices; view.Indices=&source.Geometry->Indices; view.Material=source.Material; view.Color=source.Color;
                    auto candidate=std::make_shared<LumenMeshRepresentation>();
                    if (!BuildLumenMeshRepresentation(view,request.Settings,allocation[group],*candidate,output.Error,cancelled)) return output;
                    mesh=std::move(candidate); ++output.BuildCount;
                }
                output.Meshes.push_back(std::move(mesh)); output.MaximumParts.push_back(allocation[group].Partition.MaximumParts);
                output.MaximumCards.push_back(allocation[group].Cards.MaximumCards);
            }
            if (request.Hair)
            {
                if (previous && previous->Hair==request.Hair && SameMaterials(previous->Settings,request.Settings)) output.HairVolume=previous->HairVolume;
                else
                {
                    auto volume=std::make_shared<LumenHairVolume>(); LumenHairBuildSettings hairSettings; hairSettings.GridSide=lumen.HairGridSide;
                    if (!BuildLumenHairVolume(*request.Hair,request.Settings.Hair,hairSettings,*volume,output.Error,cancelled)) return output;
                    if (!volume->Voxels.empty()) output.HairVolume=std::move(volume);
                }
            }
            return output;
        }
    }
    struct LumenRenderCache::Impl
    {
        IRHI& Rhi;
        LumenSceneCache Cache;
        std::shared_ptr<LumenBuildRequest> Desired,Running;
        std::future<Prepared> Worker;
        std::shared_ptr<const Prepared> PreparedScene,PublishedScene;
        std::uint64_t ObservedPublication=0;
        std::shared_ptr<const LumenScenePlan> Plan;
        std::vector<BufferHandle> Buffers;
        LightingState Lighting;
        LumenSurfaceLightingUpload LightUpload;
        bool Shadows=true,HaveLighting=false,LightDirty=true;
        unsigned LightingTileBudget=128,CaptureCardBudget=8,IndirectBounces=0;
        bool PreservePoseLighting=false;
        std::uint64_t Epoch=NextLumenRuntimeEpoch.fetch_add(1),NextGeneration=0,LightRevision=0;
        LumenRenderStatus Progress;
        explicit Impl(IRHI& rhi):Rhi(rhi),Cache(rhi) {}
        ~Impl()
        {
            if (Running) Running->Cancelled->store(true,std::memory_order_relaxed);
            if (Worker.valid()) Worker.wait();
            Cache.Reset(); ReleaseBuffers();
        }
        void ReleaseBuffers() { for (auto buffer:Buffers) if (buffer) Rhi.DestroyBuffer(buffer); Buffers.clear(); }
        void Fail(std::string error) { Cache.Cancel(); Progress.Lighting={}; Progress.State=LumenRenderState::Failed; Progress.Error=std::move(error); }
        bool Install(std::shared_ptr<const Prepared> prepared)
        {
            LumenSceneUpload upload; upload.CardSide=prepared->Settings.GlobalIllumination.Lumen.CardSide; upload.Hair=prepared->HairVolume;
            upload.HardwareRayTracing=prepared->Settings.GlobalIllumination.Lumen.HardwareRayTracing;
            upload.Identity={Epoch,prepared->Generation,prepared->Generation,prepared->Generation};
            std::vector<BufferHandle> buffers; std::vector<std::vector<LumenSurfaceDraw>> draws(prepared->Meshes.size());
            const auto discard=[&] { for (auto buffer:buffers) if (buffer) Rhi.DestroyBuffer(buffer); return false; };
            for (unsigned group=0;group<draws.size();++group) for (const auto& part:prepared->Meshes[group]->Geometry.Parts)
            {
                LumenSurfaceDraw draw;
                draw.Vertices=Rhi.CreateBuffer({BufferUsage::Vertex,part.Vertices.size()*sizeof(Vertex),false,"Lumen compact proxy vertices"},part.Vertices.data()); buffers.push_back(draw.Vertices);
                draw.Indices=Rhi.CreateBuffer({BufferUsage::Index,part.Indices.size()*sizeof(unsigned),false,"Lumen compact proxy indices"},part.Indices.data()); buffers.push_back(draw.Indices);
                if (!draw.Vertices || !draw.Indices) { Fail(Rhi.GetLastError()); return discard(); }
                draw.IndexCount=unsigned(part.Indices.size()); draws[group].push_back(draw);
            }
            for (unsigned i=0;i<prepared->Sources.size();++i)
            {
                const auto& source=prepared->Sources[i]; const auto group=prepared->SourceGroups[i];
                LumenSceneInstance instance; instance.Key=source.Key; instance.Mesh=prepared->Meshes[group]; instance.World=source.World; instance.CastShadow=source.CastShadow;
                for (auto geometry:draws[group])
                { auto draw=source.Capture; draw.Vertices=geometry.Vertices; draw.Indices=geometry.Indices; draw.IndexCount=geometry.IndexCount; instance.Draws.push_back(draw); }
                upload.Instances.push_back(std::move(instance));
            }
            auto plan=std::make_shared<LumenScenePlan>(); std::string error;
            if (!BuildLumenScenePlan(upload,*plan,error)) { Fail(error); return discard(); }
            Cache.Cancel(); ReleaseBuffers(); Buffers=std::move(buffers); Plan=std::move(plan); PreparedScene=std::move(prepared); LightDirty=true;
            Progress.PreparedMeshes+=PreparedScene->BuildCount; Progress.Cards=Plan->Layout.CardCount; Progress.Instances=unsigned(Plan->Parts.size());
            double total=0,covered=0,resolved=0,unchecked=0; Progress.CardSpatialSplits=0; Progress.CardCoverageBudgetExhausted=false;
            for (const auto& mesh:PreparedScene->Meshes) for (const auto& cards:mesh->Cards)
            {
                total+=cards.TotalWeight; covered+=cards.CoveredWeight; resolved+=cards.ResolvedWeight; unchecked+=cards.UncheckedWeight;
                Progress.CardSpatialSplits+=cards.SpatialSplits; Progress.CardCoverageBudgetExhausted|=cards.CoverageBudgetExhausted;
            }
            Progress.SampledCardCoverage=total?covered/total:0; Progress.SampledTexelCoverage=total?resolved/total:0;
            Progress.UncheckedTexelCoverage=total?unchecked/total:0; return true;
        }
    };
    LumenRenderCache::LumenRenderCache(IRHI& rhi):State(std::make_unique<Impl>(rhi)) {}
    LumenRenderCache::~LumenRenderCache()=default;
    void LumenRenderCache::Request(std::vector<LumenSourceInstance> sources,std::uint64_t revision,const Techniques::Settings& settings,const LightingState& lighting,bool shadows,
        std::shared_ptr<const HairStrandGeometry> hair,std::shared_ptr<const HairStrandGeometry> hairBase)
    {
        auto& state=*State; shadows=shadows && settings.GlobalIllumination.Lumen.Shadows;
        if (!settings.GlobalIllumination.Lumen.HairTransport) { hair.reset(); hairBase.reset(); }
        if (hair && !hairBase) hairBase=hair;
        state.LightingTileBudget=settings.GlobalIllumination.Lumen.LightingTilesPerUpdate;
        state.CaptureCardBudget=settings.GlobalIllumination.Lumen.CaptureCardsPerUpdate;
        const auto& lumen=settings.GlobalIllumination.Lumen;
        const unsigned bounces=lumen.Radiosity.Enabled?lumen.Radiosity.Bounces:0,rays=bounces?lumen.Radiosity.RaysPerTexel:0;
        const auto& sky=lumen.SkyRadiance; const auto& old=state.LightUpload; const auto environment=MakeLumenEnvironmentLobe(lumen);
        const bool transportChanged=state.IndirectBounces!=bounces || old.IndirectRays!=rays ||
            old.IndirectMaximumDistance!=settings.GlobalIllumination.MaximumDistance ||
            old.IndirectSky.X!=sky[0] || old.IndirectSky.Y!=sky[1] || old.IndirectSky.Z!=sky[2] || !(old.Environment==environment);
        if (!state.HaveLighting || !SameSceneLights(lighting,state.Lighting) || state.Shadows!=shadows || transportChanged)
        {
            state.HaveLighting=true; state.Lighting=lighting; state.Shadows=shadows; state.LightDirty=true;
            auto& upload=state.LightUpload; upload={}; upload.LightingRevision=++state.LightRevision;
            state.IndirectBounces=bounces; upload.IndirectRays=rays; upload.IndirectSky={sky[0],sky[1],sky[2]}; upload.Environment=environment;
            upload.IndirectMaximumDistance=settings.GlobalIllumination.MaximumDistance;
            upload.Lights=lighting.SceneLights?lighting.SceneLights->GetLights():std::vector<SceneLight>(lighting.Lights.begin(),lighting.Lights.begin()+lighting.LightCount);
            if (!shadows) for (auto& light:upload.Lights) light.CastShadows=false;
            if (state.PreparedScene && state.Desired && state.PreparedScene->Generation==state.Desired->Generation && state.Progress.State==LumenRenderState::Failed)
            { state.Progress.State=LumenRenderState::Lighting; state.Progress.Error.clear(); }
        }
        if (state.Desired && state.Desired->Revision==revision && state.Desired->Hair==hair && state.Desired->HairBase==hairBase && SameMaterials(state.Desired->Settings,settings)) return;
        auto request=std::make_shared<LumenBuildRequest>(); request->Sources=std::move(sources); request->Hair=std::move(hair); request->HairBase=std::move(hairBase); request->Settings=settings; request->Revision=revision; request->Generation=++state.NextGeneration;
        // Pose updates must not repeatedly cancel an otherwise reusable mesh
        // build. Rebase its completed local proxies onto the latest instances.
        if (state.Running && !SamePreparation(state.Running->Sources,state.Running->Settings,state.Running->HairBase,*request))
            state.Running->Cancelled->store(true,std::memory_order_relaxed);
        const bool poseOnly=state.PreparedScene && SamePoseFamily(request->Sources,settings,request->HairBase,*state.PreparedScene);
        state.PreservePoseLighting=state.PublishedScene && SamePoseFamily(request->Sources,settings,request->HairBase,*state.PublishedScene);
        state.Desired=std::move(request);
        if (!poseOnly) { state.Cache.Cancel(); state.Progress.Lighting={}; state.Progress.Distance={}; }
        state.Progress.State=LumenRenderState::Preparing; state.Progress.Generation=state.NextGeneration; state.Progress.Error.clear();
    }
    void LumenRenderCache::Advance()
    {
        auto& state=*State;
        if (state.Worker.valid())
        {
            if (state.Worker.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return;
            Prepared result;
            try { result=state.Worker.get(); } catch (const std::exception& error) { result.Generation=state.Running->Generation; result.Error=error.what(); }
            const auto running=std::move(state.Running);
            if (state.Desired && !running->Cancelled->load(std::memory_order_relaxed) &&
                SamePreparation(running->Sources,running->Settings,running->HairBase,*state.Desired) && running->Hair==state.Desired->Hair)
            {
                result.Generation=state.Desired->Generation; result.Sources=state.Desired->Sources; result.Settings=state.Desired->Settings;
            }
            if (state.Desired && !running->Cancelled->load(std::memory_order_relaxed) &&
                (result.Generation==state.Desired->Generation || SamePoseFamily(state.Desired->Sources,state.Desired->Settings,state.Desired->HairBase,result)))
            {
                if (!result.Error.empty()) { state.Fail(result.Error); return; }
                if (!state.Install(std::make_shared<Prepared>(std::move(result)))) return;
            }
        }
        if (!state.Desired || state.Progress.State==LumenRenderState::Failed) return;
        // A moving instance may update Desired every frame. Finish the frozen
        // native transaction first, then adopt the newest compatible pose once.
        const bool finishPose=state.PreparedScene && (state.LightDirty || state.Cache.Status().Building) &&
            SamePoseFamily(state.Desired->Sources,state.Desired->Settings,state.Desired->HairBase,*state.PreparedScene);
        if ((!state.PreparedScene || state.PreparedScene->Generation!=state.Desired->Generation) && !finishPose)
        {
            if (state.PreparedScene && SamePreparation(state.PreparedScene->Sources,state.PreparedScene->Settings,state.PreparedScene->HairBase,*state.Desired) &&
                state.PreparedScene->Hair==state.Desired->Hair)
            {
                auto prepared=std::make_shared<Prepared>(*state.PreparedScene); prepared->Generation=state.Desired->Generation;
                prepared->Sources=state.Desired->Sources; prepared->Settings=state.Desired->Settings; prepared->BuildCount=0;
                if (!state.Install(std::move(prepared))) return;
            }
            else
            {
                state.Running=state.Desired; const auto request=state.Running; const auto previous=state.PreparedScene;
                try { state.Worker=std::async(std::launch::async,[request,previous] { return Prepare(*request,previous); }); }
                catch (const std::exception& error) { state.Running.reset(); state.Fail(error.what()); }
                return;
            }
        }
        if (state.LightDirty)
        {
            std::string error; if (!state.Cache.Request(state.Plan,state.LightUpload,error,state.IndirectBounces)) { state.Fail(error); return; }
            state.LightDirty=false;
        }
        if (!state.Cache.Advance(1,state.LightingTileBudget,state.CaptureCardBudget)) { state.Fail(state.Cache.Status().Error); return; }
        const auto native=state.Cache.Status();
        if (native.PublishedGeneration!=state.ObservedPublication)
        { state.ObservedPublication=native.PublishedGeneration; state.PublishedScene=state.PreparedScene; }
        switch (native.State)
        {
            case LumenSceneUpdateState::Distance:state.Progress.State=LumenRenderState::Distance;break;
            case LumenSceneUpdateState::Capture:state.Progress.State=LumenRenderState::Capture;break;
            case LumenSceneUpdateState::Lighting:state.Progress.State=LumenRenderState::Lighting;break;
            case LumenSceneUpdateState::Ready:state.Progress.State=LumenRenderState::Ready;break;
            default:state.Progress.State=LumenRenderState::Failed;break;
        }
        // Compare against the published appearance, not merely the in-flight
        // pose: a material edit followed by motion must not resurrect old light.
        state.PreservePoseLighting=state.PublishedScene && SamePoseFamily(state.Desired->Sources,state.Desired->Settings,state.Desired->HairBase,*state.PublishedScene);
        const bool usable=native.Usable || state.PreservePoseLighting;
        state.Progress.Lighting=usable?native.Lighting:LumenLightingHandle{};
        state.Progress.CompletedBounces=native.CompletedBounces; state.Progress.TargetBounces=native.TargetBounces;
        state.Progress.LightingTiles=native.LightingTiles; state.Progress.CaptureCards=native.CaptureCards;
        state.Progress.Distance=usable?native.Distance:LumenDistanceSceneHandle{};
        if (!native.Building && state.PreparedScene->Generation!=state.Desired->Generation) state.Progress.State=LumenRenderState::Preparing;
    }
    void LumenRenderCache::Cancel()
    {
        auto& state=*State; if (state.Running) state.Running->Cancelled->store(true,std::memory_order_relaxed);
        state.Desired.reset(); state.Cache.Cancel(); state.PreservePoseLighting=false; state.Progress.State=LumenRenderState::Idle; state.Progress.Lighting={}; state.Progress.Distance={};
    }
    LumenRenderStatus LumenRenderCache::Status() const { return State->Progress; }
}
