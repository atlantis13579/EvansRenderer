#include "Editor/TechniqueControls.h"

#include "Editor/Editor.hpp"
#include "Editor/Imgui.h"
#include "Techniques/TechniqueCatalog.h"
#include "Renderer/Types.h"
#include "Renderer/Lighting/ForwardPlus.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

namespace Evans::Editor
{
namespace
{
constexpr bool ShowControlHelp=false;
void BeginControlGroup(const char* name,bool available=true,bool initiallyOpen=false)
{
    available=available && imguiIsEnabled();
    static std::unordered_map<std::string,bool> groups;
    auto& open=groups.emplace(name,initiallyOpen).first->second;
    imguiPushVisibility(true);
    if (imguiCollapse(name,"",open && available,available)) open=!open;
    imguiPushVisibility(open && available);
    imguiIndent();
}
void EndControlGroup()
{
    imguiUnindent();
    imguiPopVisibility(); imguiPopVisibility();
}
class ControlSections
{
    bool Active=false,Previous=imguiSetHideDisabled(false);
public:
    void Next(const char* name,bool available=true,const char* reason=nullptr)
    {
        End();
        // Categories keep independent expansion state, including child groups.
        static std::unordered_map<std::string,bool> expanded;
        imguiPushVisibility(true);
        imguiSeparatorLine();
        auto& open=expanded.emplace(name,std::string(name)=="Rendering").first->second;
        if (imguiCollapse(name,"",open)) open=!open;
        imguiPushVisibility(open); imguiIndent();
        imguiPushEnabled(available,reason?reason:"This category's techniques are unavailable in the current scene, pipeline or backend.");
        Active=true;
    }
    void End() { if (Active) { imguiPopEnabled(); EndControlGroup(); Active=false; } }
    ~ControlSections() { End(); imguiSetHideDisabled(Previous); }
};
void HelpLabel(const char* text)
{ if (ShowControlHelp) imguiLabel(text); }
bool DrawControlChoices(int& selected,const std::vector<const char*>& names,bool enabled=true,const std::vector<std::string>& reasons={})
{
    bool changed=false;
    for (std::size_t index=0;index<names.size();++index)
    {
        imguiSetDisabledReason(index<reasons.size()?reasons[index].c_str():nullptr);
        if (imguiCheck(names[index],selected==static_cast<int>(index),enabled && (index>=reasons.size() || reasons[index].empty())) && selected!=static_cast<int>(index))
        { selected=static_cast<int>(index); changed=true; }
    }
    imguiSetDisabledReason(nullptr);
    return changed;
}
bool DrawLumenEnvironment(Techniques::LumenEnvironmentSettings& environment,const char* context)
{
    const std::string prefix=std::string(context)+" env "; bool changed=false;
    if (imguiCheck((prefix+"directional").c_str(),environment.Enabled))
    { environment.Enabled=!environment.Enabled; changed=true; }
    if (environment.Enabled)
    {
        BeginControlGroup((std::string(context)+" sky settings").c_str());
        const char* colors[]{"R","G","B"}; const char* axes[]{"X","Y","Z"};
        for (unsigned channel=0;channel<3;++channel)
            changed|=imguiSlider((prefix+colors[channel]).c_str(),&environment.Radiance[channel],0,Techniques::MaximumLumenEnvironmentRadiance,0.01f);
        for (unsigned axis=0;axis<3;++axis)
            changed|=imguiSlider((prefix+"axis "+axes[axis]).c_str(),&environment.Direction[axis],-1,1,0.01f);
        changed|=imguiSlider((prefix+"power").c_str(),&environment.Exponent,1,Techniques::MaximumLumenEnvironmentExponent,0.25f);
        HelpLabel("The axis points toward incoming sky. Power narrows a cosine lobe; RGB is its peak radiance.");
        EndControlGroup();
    }
    return changed;
}
} // namespace

void DrawViewportStatus(const EditorCore& editor,const Rect& viewport,int clientHeight)
{
    if (editor.Settings().ShowLights) for (std::size_t i=0;i<editor.Scene().Lights.size();++i) if (const auto point=editor.LightScreenPosition(i,viewport))
    {
        const float x=point->X,y=clientHeight-point->Y;
        const bool selected=int(i)==editor.SelectedLight();const auto color=selected?imguiRGBA(255,190,40):imguiRGBA(255,240,140,190);
        imguiDrawRoundedRect(x-6,y-6,12,12,6,color);
        imguiDrawLine(x-11,y,x+11,y,1,color);imguiDrawLine(x,y-11,x,y+11,1,color);
        if (selected) imguiDrawText(int(x+15),int(y),IMGUI_ALIGN_LEFT,("Light "+std::to_string(i+1)+" - drag to move").c_str(),color);
    }
    if (viewport.Width>=120 && viewport.Height>=36)
    {
        char fps[48];std::snprintf(fps,sizeof(fps),"%.1f FPS",editor.ViewportFramesPerSecond());
        const float right=viewport.X+viewport.Width-12,top=clientHeight-viewport.Y-12;
        imguiDrawRoundedRect(right-100,top-22,108,26,4,imguiRGBA(18,20,26,190));
        imguiDrawText(int(right),int(top-16),IMGUI_ALIGN_RIGHT,fps,imguiRGBA(240,240,240));
    }
    const auto preparation = editor.PendingPreparation();
    std::string lighting;
    const auto pending=[&](const std::string& status)
    {
        if (lighting.empty() && !status.empty() && status!="Idle" && status.find("Ready")!=0 && status!="Loaded baked resource") lighting=status;
    };
    if (editor.ViewportBackend()==editor.SelectedBackend())
    {
        const auto& settings=editor.Settings().Techniques;
        if (Techniques::UsesLightmaps(settings.GlobalIllumination) || Techniques::UsesBakedProbes(settings.GlobalIllumination))
            pending(editor.ViewportBakedLightingStatus());
        if (Techniques::UsesVoxelGI(settings.GlobalIllumination)) pending(editor.ViewportVoxelGIStatus());
        if (Techniques::UsesLumenScene(settings)) pending(editor.ViewportLumenGIStatus());
    }
    constexpr double NoticeDelaySeconds=1;
    if (lighting.empty() && editor.ViewportImage().IsValid() && !editor.IsDemoLoading() &&
        (!preparation.Active || preparation.Seconds<NoticeDelaySeconds)) return;
    constexpr float Padding=16,NoticeHeight=112,MaximumWidth=620;
    const float width=std::min(MaximumWidth,std::max(0.0f,viewport.Width-Padding*2));
    if (width<=Padding*2 || viewport.Height<NoticeHeight) return;
    const float x=viewport.X+(viewport.Width-width)/2;
    const float y=clientHeight-viewport.Y-(editor.ViewportImage().IsValid()?
        Padding+NoticeHeight:(viewport.Height+NoticeHeight)/2);
    constexpr float CharacterWidth=8;
    const auto characters=std::size_t(std::max(0.0f,width-Padding*2)/CharacterWidth);
    const auto drawText=[&](float baseline,std::string text,unsigned color)
    {
        if (text.size()>characters) text=text.substr(0,characters>3?characters-3:0)+"...";
        imguiDrawText(int(x+Padding),int(y+baseline),IMGUI_ALIGN_LEFT,text.c_str(),color);
    };
    imguiDrawRoundedRect(x,y,width,NoticeHeight,5,imguiRGBA(24,27,32,235));
    const bool lightingFailed=lighting.find("Failed")==0;
    const std::string title=editor.IsDemoLoading()?"Loading scene...":preparation.WaitingForPreviousFrame?
        "Waiting for previous frame...":!lighting.empty()?
        lightingFailed?"Lighting unavailable":"Preparing lighting...":preparation.Active?
        "Preparing frame...":"Viewport unavailable";
    drawText(84,title,imguiRGBA(235,235,240));
    const auto detail=lightingFailed?lighting:preparation.Shaders.Source.empty()?(lighting.empty()?editor.StatusText():lighting):
        "Shader preparation: "+preparation.Shaders.Source;
    drawText(60,detail,imguiRGBA(175,190,210));
    if (!lightingFailed && (preparation.Active || !lighting.empty()))
    {
        std::string activity="Waiting for lighting updates";
        if (preparation.Active)
        {
            activity="Elapsed: "+std::to_string(int(preparation.Seconds))+" s";
            if (preparation.Shaders.SourcesLoaded)
                activity+="  |  Shader source loads: "+std::to_string(preparation.Shaders.SourcesLoaded);
        }
        drawText(36,activity,imguiRGBA(175,190,210));
        // Compilers expose no work total. A moving segment signals activity
        // without inventing a percentage or freezing during a single compile.
        constexpr float BarHeight=4,SegmentFraction=0.2f;
        constexpr double CycleSeconds=1.6;
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const float phase=float(std::fmod(seconds,CycleSeconds)/CycleSeconds);
        const float trackWidth=width-Padding*2,segmentWidth=trackWidth*SegmentFraction;
        const float position=(1-std::abs(phase*2-1))*(trackWidth-segmentWidth);
        imguiDrawRoundedRect(x+Padding,y+Padding,trackWidth,BarHeight,BarHeight/2,imguiRGBA(55,62,73));
        imguiDrawRoundedRect(x+Padding+position,y+Padding,segmentWidth,BarHeight,BarHeight/2,imguiRGBA(94,177,235));
    }
}

void DrawRenderModeControls(EditorCore& editor)
{
    imguiSeparatorLine();
    imguiLabel("Render Mode");
    const bool path=Techniques::UsesPathTracing(editor.Settings().Techniques.Pipeline) &&
        editor.QueryTechniqueSupport(Techniques::Technique::PathTracing).Available();
    imguiSetDisabledReason("Ray Tracing Full owns primary visibility; wire modes require rasterization.");
    const char* labels[]{"Solid","Wireframe","Solid + Wireframe"};
    for (int i=0;i<3;++i)
    {
        const auto mode=FillMode(i);
        if (imguiCheck(labels[i],path?mode==FillMode::Solid:editor.Settings().Mode==mode,!path) && editor.Settings().Mode!=mode)
            editor.SetFillMode(mode);
    }
    imguiSetDisabledReason(nullptr);
}

void DrawRenderBufferControls(EditorCore& editor)
{
    using namespace Techniques;using B=RenderBuffer;
    const auto settings=ResolveSettings(editor.Settings().Techniques,BackendFromId(editor.SelectedBackend())).Effective;
    const bool paths=UsesPathTracing(settings.Pipeline),native=IsNativeRaster(BackendFromId(editor.SelectedBackend()));
    const char* labels[]{"Color","Depth","Shadow map","Shadow cascade 1","Shadow cascade 2","Shadow cascade 3","Shadow cascade 4","Albedo","World normals","Roughness","Metallic","Emission","AO / SSAO","Motion vectors"};
    std::vector<std::string> reasons(14);std::vector<const char*> pointers;
    for (int i=1;i<14;++i)
    {
        if (!native) reasons[i]="Buffer inspection requires a native renderer backend.";
        else if (i>=2 && i<=6 && (paths || !editor.Settings().ShadowsEnabled || settings.Shadows==ShadowRepresentation::RayTraced)) reasons[i]="This pipeline does not produce a raster shadow map.";
        else if (i>=3 && i<=6 && (settings.Shadows!=ShadowRepresentation::Cascaded || unsigned(i-3)>=settings.Cascades.CascadeCount)) reasons[i]="This shadow cascade is inactive.";
        else if (i>=7 && i<=11 && (paths || (!UsesDeferredLighting(settings.Pipeline) && !UsesAmbientOcclusion(settings.AmbientOcclusion) && settings.Reflections.Mode==ReflectionMode::None && settings.GlobalIllumination.Mode==GlobalIlluminationMode::None))) reasons[i]="The current pipeline does not produce material buffers.";
        else if (i==12 && !UsesAmbientOcclusion(settings.AmbientOcclusion)) reasons[i]="Enable an ambient occlusion technique to inspect its output.";
        else if (i==13 && !RequiresMotionVectors(settings.PostProcessing) && !UsesTemporalReflections(settings.Reflections) && !UsesTemporalGI(settings.GlobalIllumination)) reasons[i]="Enable a temporal technique or motion blur to generate motion vectors.";
    }
    for (const auto& reason:reasons) pointers.push_back(reason.c_str());
    int selected=int(editor.Settings().Buffer);
    if (selected<0 || selected>=14 || !reasons[selected].empty()) {selected=0;editor.SetRenderBuffer(B::Color);}
    imguiSeparatorLine();
    if (imguiCombo("Render Buffer",&selected,labels,14,true,pointers.data(),true)) editor.SetRenderBuffer(B(selected));
}

void DrawTechniqueControls(EditorCore& editor)
{
    using namespace Evans::Techniques;
    const Backend backend = BackendFromId(editor.SelectedBackend());
    const bool native = IsNativeRaster(backend);
    const auto hasImage = [&](const auto& imageMember)
    {
        return std::any_of(editor.Scene().Materials.begin(), editor.Scene().Materials.end(),
            [&](const Material& material)
            {
                const auto& image = material.*imageMember;
                return image && image->IsValid();
            });
    };
    const bool hasDiffuseMap = hasImage(&Material::DiffuseImage);
    const bool hasNormalMap = hasImage(&Material::NormalImage);
    const bool hasOpacityMap = hasImage(&Material::OpacityImage);
    Settings requested = editor.Settings().Techniques;
    const Support deferredSupport = editor.QueryTechniqueSupport(Technique::Deferred);
    const bool deferredAvailable = deferredSupport.Available();
    const Support forwardPlusSupport = editor.QueryTechniqueSupport(Technique::ForwardPlus);
    const bool forwardPlusAvailable = native && forwardPlusSupport.Available();
    const Support clusteredSupport = editor.QueryTechniqueSupport(Technique::ClusteredForward);
    const bool clusteredAvailable = native && clusteredSupport.Available();
    const Support tiledSupport = editor.QueryTechniqueSupport(Technique::TiledDeferred);
    const bool tiledAvailable = native && tiledSupport.Available();
    const auto visibilitySupport = editor.QueryTechniqueSupport(Technique::VisibilityBuffer);
    const bool visibilityAvailable = native && visibilitySupport.Available();
    const bool hybridAvailable = native && editor.QueryTechniqueSupport(Technique::Hybrid).Available();
    const Support saoSupport = editor.QueryTechniqueSupport(Technique::SAO);
    const bool saoAvailable = native && saoSupport.Available();
    const Support pathSupport = editor.QueryTechniqueSupport(Technique::PathTracing);
    const bool pathAvailable = native && pathSupport.Available();
    const auto lumenSupport=editor.QueryTechniqueSupport(Technique::Lumen);
    const bool lumenAvailable=native && lumenSupport.Available();
    const auto voxelSupport=editor.QueryTechniqueSupport(Technique::VoxelGI);
    const bool voxelAvailable=native && voxelSupport.Available();
    const Support rayGISupport = editor.QueryTechniqueSupport(Technique::RayTracedGI);
    const bool rayGIAvailable = native && rayGISupport.Available();
    const Support rayReflectionSupport = editor.QueryTechniqueSupport(Technique::RayTracedReflections);
    const bool rayReflectionAvailable = native && rayReflectionSupport.Available();
    const Support rayAOSupport = editor.QueryTechniqueSupport(Technique::RayTracedAO);
    const bool rayAOAvailable = native && rayAOSupport.Available();
    const Support giSupport = editor.QueryTechniqueSupport(Technique::SSGI);
    const bool giAvailable = native && giSupport.Available();
    const Support probeSupport = editor.QueryTechniqueSupport(Technique::LightProbes);
    const bool probesAvailable = native && probeSupport.Available();
    const Support lightmapSupport = editor.QueryTechniqueSupport(Technique::Lightmaps);
    const bool lightmapsAvailable = native && lightmapSupport.Available();
    const auto peelingSupport = editor.QueryTechniqueSupport(Technique::OIT);
    const bool peelingAvailable = native && peelingSupport.Available();
    const auto refractionSupport = editor.QueryTechniqueSupport(Technique::Refraction);
    const bool refractionAvailable = native && refractionSupport.Available();
    const Support taaSupport = editor.QueryTechniqueSupport(Technique::TAA);
    const bool taaAvailable = native && taaSupport.Available();
    const Support tsrSupport=editor.QueryTechniqueSupport(Technique::TSR);
    const bool tsrAvailable=native && tsrSupport.Available();
    const auto fogSupport = editor.QueryTechniqueSupport(Technique::VolumetricFog);
    const auto volumeLightSupport = editor.QueryTechniqueSupport(Technique::VolumetricLighting);
    const auto shaftsSupport = editor.QueryTechniqueSupport(Technique::LightShafts);
    const auto resolve=[&](const Settings& candidate) { return ResolveSettingsForDevice(candidate, backend, deferredAvailable, saoAvailable,
        editor.QueryMultisampleCounts(), editor.QueryTechniqueSupport(Technique::Instancing).Available(), taaAvailable,
        fogSupport.Available(), editor.QueryWeightedTransparencySampleCounts(), volumeLightSupport.Available(), shaftsSupport.Available(),
        editor.QueryTechniqueSupport(Technique::VSM).Available(), editor.QueryTechniqueSupport(Technique::ContactShadows).Available(),
        forwardPlusAvailable, clusteredAvailable, tiledAvailable, giAvailable, probesAvailable, lightmapsAvailable, refractionAvailable, peelingAvailable, rayAOAvailable, rayReflectionAvailable, rayGIAvailable, pathAvailable,
        editor.QueryTechniqueSupport(Technique::GPUDriven).Available(), visibilityAvailable, hybridAvailable, voxelAvailable,
        editor.QueryTechniqueSupport(Technique::TemporalEffects).Available(), editor.QueryTechniqueSupport(Technique::ParallaxOcclusionMapping).Available(),
        editor.QueryTechniqueSupport(Technique::DisplacementMapping).Available(), editor.QueryTechniqueSupport(Technique::Tessellation).Available(),
        editor.QueryTechniqueSupport(Technique::VirtualGeometry).Available(),tsrAvailable,
        editor.QueryTechniqueSupport(Technique::VirtualShadowMaps).Available(),lumenAvailable,editor.QueryTechniqueSupport(Technique::Nanite).Available(),
        editor.QueryTechniqueSupport(Technique::MeshShaders).Available(),editor.QueryNaniteHardwareScheduling(),editor.QueryLumenHardwareRayTracing()); };
    const Resolution resolution=resolve(requested);
    bool changed = false;

    const auto toggle = [&](const char* name, bool& value, bool enabled)
    {
        const auto address=reinterpret_cast<std::uintptr_t>(&value),base=reinterpret_cast<std::uintptr_t>(&requested);
        if (address<base || address>=base+sizeof(requested)) {imguiCheck(name,value,enabled);return;}
        const auto offset=address-base;
        auto candidate=requested;*reinterpret_cast<bool*>(reinterpret_cast<char*>(&candidate)+offset)=true;
        const auto result=resolve(candidate);
        const bool supported=*reinterpret_cast<const bool*>(reinterpret_cast<const char*>(&result.Effective)+offset);
        const std::string reason=!supported && !result.Diagnostics.empty()?result.Diagnostics.back():"Requires a compatible pipeline, scene material and backend.";
        imguiSetDisabledReason(reason.c_str());
        if (imguiCheck(name, value, enabled && supported))
        {
            value = !value;
            changed = true;
        }
        imguiSetDisabledReason(nullptr);
    };
    const bool path = pathAvailable && UsesPathTracing(resolution.Effective.Pipeline);
    const bool tiled = tiledAvailable && resolution.Effective.Pipeline == RenderingPipeline::TiledDeferred;
    const bool visibility = visibilityAvailable && UsesVisibilityBuffer(resolution.Effective.Pipeline);
    const bool hybrid = hybridAvailable && UsesHybridRendering(resolution.Effective.Pipeline);
    const bool deferred = deferredAvailable && UsesDeferredLighting(resolution.Effective.Pipeline);
    const bool forwardPlus = forwardPlusAvailable && resolution.Effective.Pipeline == RenderingPipeline::ForwardPlus;
    const bool clustered = clusteredAvailable && resolution.Effective.Pipeline == RenderingPipeline::ClusteredForward;
    PostProcessSettings& post = requested.PostProcessing;
    const bool fxaaAvailable = native && !path && QuerySupport(Technique::FXAA, backend).Available();
    const auto& scene=editor.Scene();
    ControlSections sections;
    sections.Next("Rendering",native);
    if (native)
    {
        std::vector<const char*> pipelines {"Forward","Deferred","Forward+","Clustered Forward","Tiled Deferred","GPU Driven Visibility Buffer","Ray Tracing Hybrid","Ray Tracing Full"};
        std::vector<RenderingPipeline> pipelineModes {RenderingPipeline::Forward,RenderingPipeline::Deferred,RenderingPipeline::ForwardPlus,RenderingPipeline::ClusteredForward,RenderingPipeline::TiledDeferred,RenderingPipeline::VisibilityBuffer,RenderingPipeline::Hybrid,RenderingPipeline::PathTracing};
        std::vector<std::string> reasons(pipelines.size());std::vector<const char*> reasonPointers;
        int pipelineIndex=0;
        for (std::size_t i=0;i<pipelineModes.size();++i)
        {
            auto candidate=requested; candidate.Pipeline=pipelineModes[i];
            // Selecting a pipeline releases constraints owned by the previous one.
            candidate.Nanite.Enabled=UsesVisibilityBuffer(candidate.Pipeline);candidate.MeshShaders=false;
            auto resolved=resolve(candidate);
            if (requested.Pipeline==pipelineModes[i]) pipelineIndex=int(i);
            if (resolved.Effective.Pipeline!=candidate.Pipeline) reasons[i]=resolved.Diagnostics.empty()?"Unavailable on this device.":resolved.Diagnostics.front();
            if (UsesVisibilityBuffer(candidate.Pipeline) && !editor.QueryTechniqueSupport(Technique::Nanite).Available()) reasons[i]="Requires native GPU cluster visibility and indirect geometry assembly.";
            if (scene.Demo==DemoId::FurCube && requested.Hair.Enabled && !resolved.Effective.Hair.Enabled) reasons[i]="Disable native fur before selecting this pipeline; ray geometry does not yet include generated fibers.";
        }
        for (const auto& reason:reasons) reasonPointers.push_back(reason.c_str());
        if (imguiCombo("Pipeline",&pipelineIndex,pipelines.data(),int(pipelines.size()),true,reasonPointers.data()))
        {
            requested.Pipeline=pipelineModes[pipelineIndex];requested.MeshShaders=false;
            requested.Nanite.Enabled=UsesVisibilityBuffer(requested.Pipeline);
            if (requested.Pipeline!=RenderingPipeline::Forward && requested.Shading==ShadingModel::Gouraud) requested.Shading=ShadingModel::Phong;
            changed=true;
        }
    }
    else imguiLabel(backend == Backend::DXR ? "Pipeline: DXR direct lighting" : "Pipeline: Software raster");
    if (native && QuerySupport(Technique::Deferred, backend).Available() && !deferredAvailable)
        HelpLabel(deferredSupport.Reason);
    if (tiled) HelpLabel("G-buffer depth feeds tiled opaque lighting. Transparency uses full-frustum tile lists; HDR and per-pixel shading are required.");
    if (native && !tiledAvailable) HelpLabel(tiledSupport.Reason);
    if (clustered) HelpLabel("3D light lists cover opaque and transparent surfaces without a depth prepass. Gouraud requires Forward.");
    if (native && !clusteredAvailable) HelpLabel(clusteredSupport.Reason);
    if (deferred) HelpLabel("Deferred opaque / Forward transparency. Gouraud requires Forward.");
    if (visibility) HelpLabel("Triangle visibility precedes material reconstruction. GPU instance/cluster culling, hierarchical depth, indirect raster and tiled material resolve are active. Instancing still applies to shadow, capture and velocity passes.");
    if (forwardPlus) HelpLabel("Depth prepass and tile light lists. Transparent and volume lighting retain lights outside opaque depth. Gouraud requires Forward.");
    if (native && QuerySupport(Technique::ForwardPlus, backend).Available() && !forwardPlusAvailable)
        HelpLabel(forwardPlusSupport.Reason);

    if (path)
    {
        auto& settings = requested.PathTracing;
        HelpLabel("Camera paths own lighting, visibility and antialiasing. Raster options are retained for switching back. Primary/shadow/multi-bounce rays, opaque/cutout surfaces, thin opacity and dielectric reflection/refraction use the scene BVH. Participating-media passes are unavailable.");
        const char* materials[] {"GGX + diffuse", "Lambertian"};
        int material = static_cast<int>(settings.Material);
        if (imguiCombo("Path material", &material, materials, 2))
        { settings.Material = static_cast<PathMaterial>(material); changed = true; }
        if (imguiCheck("Accumulate samples", settings.Accumulate)) { settings.Accumulate = !settings.Accumulate; changed = true; }
        float batch = float(settings.SamplesPerFrame), depth = float(settings.MaximumDepth), budget = float(settings.AccumulationLimit);
        if (imguiSlider("Paths per frame", &batch, 1, MaximumPathsPerFrame, 1)) { settings.SamplesPerFrame = unsigned(batch); changed = true; }
        if (imguiSlider("Maximum path depth", &depth, 1, MaximumPathDepth, 1)) { settings.MaximumDepth = unsigned(depth); changed = true; }
        if (settings.Accumulate && imguiSlider("Sample budget", &budget, 1, MaximumAccumulatedPaths, 1))
        { settings.AccumulationLimit = unsigned(budget); changed = true; }
        changed |= imguiSlider("Ray distance (world)", &settings.MaximumDistance, MinimumPathDistance, MaximumPathDistance, MinimumPathDistance);
        changed |= imguiSlider("Ray bias (world)", &settings.NormalBias, 0, std::min(MaximumPathBias,settings.MaximumDistance), MinimumPathDistance);
        imguiLabel(("Committed samples: " + std::to_string(editor.ViewportPathTracingSamples())).c_str());
        imguiLabel(("Sample sequence: " + std::to_string(settings.Seed)).c_str());
        if (imguiButton("Previous path sequence")) { --settings.Seed; changed = true; }
        if (imguiButton("New path sequence")) { ++settings.Seed; changed = true; }
        HelpLabel("Camera, geometry, material and light changes restart accumulation. Display color controls preserve the accumulated radiance.");
    }

    constexpr std::array<ShadingModel, 5> models = {
        ShadingModel::Phong, ShadingModel::Flat, ShadingModel::Gouraud,
        ShadingModel::PBR, ShadingModel::Toon};
    constexpr std::array<Technique, 5> shadingTechniques = {
        Technique::Phong, Technique::Flat, Technique::Gouraud, Technique::PBR, Technique::Toon};
    constexpr std::array<const char*, 5> shadingLabels = {"Phong", "Flat", "Gouraud", "PBR", "Toon / cel (cartoon)"};
    std::vector<const char*> shadingNames;
    std::vector<ShadingModel> supportedModels;
    std::vector<std::string> shadingReasons;
    for (std::size_t i=0;i<models.size();++i)
    {
        shadingNames.push_back(shadingLabels[i]);supportedModels.push_back(models[i]);
        auto candidate=requested;candidate.Shading=models[i];const auto result=resolve(candidate);
        const auto support=editor.QueryTechniqueSupport(shadingTechniques[i]);
        shadingReasons.push_back(!support.Available()?support.Reason:result.Effective.Shading!=candidate.Shading?"This pipeline does not evaluate this shading model.":"");
    }
    std::vector<const char*> shadingReasonPointers;for (const auto& reason:shadingReasons) shadingReasonPointers.push_back(reason.c_str());
    int shadingIndex = 0;
    for (std::size_t index = 0; index < supportedModels.size(); ++index)
        if (supportedModels[index] == resolution.Effective.Shading) shadingIndex = static_cast<int>(index);
    if (!shadingNames.empty() && imguiCombo("Shading", &shadingIndex, shadingNames.data(),
        static_cast<int>(shadingNames.size()), native && !path,shadingReasonPointers.data()))
    {
        requested.Shading = supportedModels[static_cast<std::size_t>(shadingIndex)];
        changed = true;
    }
    if (!native) HelpLabel("Native shading controls are unavailable on this backend.");
    if (native && (path ? requested.PathTracing.Material == PathMaterial::GGX : requested.Shading == ShadingModel::PBR && !scene.UsingPbrMaterials))
    {
        changed |= imguiSlider("Roughness", &requested.Roughness, 0.045f, 1.0f, 0.01f);
        changed |= imguiSlider("Metallic", &requested.Metallic, 0.0f, 1.0f, 0.01f);
    }
    if (native && !path && requested.Shading == ShadingModel::Toon)
        changed |= imguiSlider("Light bands", &requested.ToonBands, 2.0f, 12.0f, 1.0f);

    const bool softwareMaterials = IsSoftwareRaster(backend);
    const bool rasterMaterials = native || softwareMaterials;
    sections.Next("Materials",rasterMaterials);
    toggle("Texture mapping", requested.TextureMapping, rasterMaterials && hasDiffuseMap);
    if (rasterMaterials && !hasDiffuseMap) HelpLabel("Scene has no diffuse map");
    const bool hasMaterialTextures = (requested.TextureMapping && hasDiffuseMap) ||
        (requested.NormalMapping && hasNormalMap) || (requested.AlphaTesting && hasOpacityMap) || scene.UsingPbrMaterials;
    const bool supportsAnisotropy = editor.QueryMaterialAnisotropicFiltering();
    toggle("Anisotropic filtering", requested.AnisotropicFiltering,
        rasterMaterials && !path && hasMaterialTextures && supportsAnisotropy);
    if (rasterMaterials && !path && hasMaterialTextures && !supportsAnisotropy)
        HelpLabel("Anisotropic filtering requires device support.");
    const bool normalMappingCompatible = softwareMaterials || path || (resolution.Effective.Shading != ShadingModel::Gouraud &&
        resolution.Effective.Shading != ShadingModel::Flat);
    if (normalMappingCompatible)
    {
        toggle("Normal mapping", requested.NormalMapping, rasterMaterials && hasNormalMap);
        if (rasterMaterials && !hasNormalMap) HelpLabel("Scene has no normal map");
    }
    else
    {
        imguiCheck("Normal mapping", false, false);
        imguiLabel(requested.Shading == ShadingModel::Flat ?
            "Flat: geometric face normals" : "Gouraud: vertex lighting only");
    }
    toggle("Alpha testing", requested.AlphaTesting, rasterMaterials && hasOpacityMap);
    if (rasterMaterials && !hasOpacityMap) HelpLabel("Scene has no cutout material");
    const bool lightmapAvailable=(lightmapsAvailable || (softwareMaterials && lightmapSupport.Available())) && !path && scene.Demo==DemoId::SponzaAtrium &&
        DemoCatalog::SupportsSponzaLightmaps(requested);
    const bool lightmapOn=UsesLightmaps(requested.GlobalIllumination);
    imguiPushVisibility(scene.Demo==DemoId::SponzaAtrium);
    const bool hideDisabledLightmap=imguiSetHideDisabled(false);
    if (imguiCheck("Lightmap",lightmapOn,lightmapAvailable))
    {
        requested.GlobalIllumination.Mode=lightmapOn ? GlobalIlluminationMode::None : GlobalIlluminationMode::Lightmaps;
        changed=true;
    }
    imguiSetHideDisabled(hideDisabledLightmap);
    if (!lightmapAvailable) imguiLabel(path ? "Lightmap requires raster rendering." :
        !DemoCatalog::SupportsSponzaLightmaps(requested) ? "Lightmap requires the original mesh topology." : lightmapSupport.Reason);
    if (lightmapOn && lightmapAvailable)
    {
        changed|=imguiSlider("Lightmap intensity",&requested.GlobalIllumination.Intensity,0,4,0.01f);
        if (editor.ViewportBackend()==editor.SelectedBackend() && !editor.ViewportBakedLightingStatus().empty())
            imguiLabel(editor.ViewportBakedLightingStatus().c_str());
    }
    imguiPopVisibility();
    const auto parallaxSupport=editor.QueryTechniqueSupport(Technique::ParallaxOcclusionMapping);
    toggle("Parallax occlusion mapping",requested.Parallax.Enabled,native && parallaxSupport.Available() && !path && !UsesHybridRendering(requested.Pipeline));
    if (requested.Parallax.Enabled)
    {
        BeginControlGroup("Parallax settings",resolution.Effective.Parallax.Enabled);
        if (path || UsesHybridRendering(requested.Pipeline)) HelpLabel("POM raster materials are inactive in scene-ray primary pipelines; the height request is retained.");
        else if (!parallaxSupport.Available()) HelpLabel(parallaxSupport.Reason);
        else
        {
            auto& parallax=requested.Parallax;
            const char* coordinates[]{"Automatic mesh UV / object projection","Mesh UV","Object projection"}; int coordinate=int(parallax.Coordinates);
            if (imguiCombo("Height coordinates",&coordinate,coordinates,3,true)) { parallax.Coordinates=HeightCoordinates(coordinate); changed=true; }
            changed|=imguiSlider("Height / local tile extent",&parallax.DepthFraction,0,MaximumParallaxDepth,0.001f);
            changed|=imguiSlider("Height field tiling",&parallax.Tiling,MinimumHeightTiling,MaximumHeightTiling,0.25f);
            changed|=imguiSlider("Height normal strength",&parallax.NormalStrength,0,2,0.01f);
            changed|=imguiSlider("Grazing fade cosine",&parallax.GrazingCosine,MinimumParallaxCosine,MaximumParallaxCosine,0.005f);
            float minimum=float(parallax.MinimumLayers),maximum=float(parallax.MaximumLayers),refinements=float(parallax.RefinementSteps);
            if (imguiSlider("Minimum height layers",&minimum,float(MinimumParallaxLayers),float(parallax.MaximumLayers),1)) { parallax.MinimumLayers=std::uint32_t(minimum); changed=true; }
            if (imguiSlider("Maximum height layers",&maximum,float(parallax.MinimumLayers),float(MaximumParallaxLayers),1)) { parallax.MaximumLayers=std::uint32_t(maximum); changed=true; }
            if (imguiSlider("Height intersection refinements",&refinements,0,MaximumParallaxRefinements,1)) { parallax.RefinementSteps=std::uint32_t(refinements); changed=true; }
            HelpLabel("Procedural height reuses existing meshes. Opaque Deferred/Tiled materials support MSAA, TAA and screen effects; base silhouette, depth and shadow geometry stay unchanged.");
            HelpLabel("Cutout/blended materials retain their ordinary evaluation. Capture/scene-ray reflections pause while POM is active; their material variants remain pending.");
            HelpLabel("POM material capture uses individual meshes; other geometry passes retain requested instancing and GPU culling. Wireframe and zero height bypass POM.");
        }
        EndControlGroup();
    }
    changed|=DrawDragonMaterialControls(editor,requested);
    sections.Next("Scene lights");
    if (imguiCheck("Show Light",editor.Settings().ShowLights)) editor.SetShowLights(!editor.Settings().ShowLights);

    const bool completeLighting = UsesLightLists(resolution.Effective.Pipeline) || UsesPathTracing(resolution.Effective.Pipeline);
    const auto lightCapacity = completeLighting ? Renderer::MaximumForwardPlusLights : Renderer::MaximumSceneLights;
    const bool canAddLight = scene.Lights.size() < lightCapacity && !editor.IsDemoLoading();
    imguiLabel(("Scene lights: " + std::to_string(scene.Lights.size())).c_str());
    if (scene.Lights.size() > Renderer::MaximumSceneLights && !completeLighting)
        HelpLabel("Forward+, Clustered Forward, Tiled Deferred and Ray Tracing Full use all scene lights. This pipeline shades at most four, retaining the shadow caster.");
    if (imguiButton("Add point light", canAddLight)) editor.AddLocalLight(LightType::Point);
    bool hasLocalLights = false;
    const auto& lightViewCamera = scene.SceneCamera;
    const float lightViewDeltaX = lightViewCamera.Position.X - lightViewCamera.Target.X, lightViewDeltaY = lightViewCamera.Position.Y - lightViewCamera.Target.Y;
    const float lightViewDeltaZ = lightViewCamera.Position.Z - lightViewCamera.Target.Z;
    constexpr float RangeSliderViewRadii = 8.0f, RangeSliderSteps = 100.0f;
    const float rangeLimit = std::max(1.0f, std::sqrt(lightViewDeltaX * lightViewDeltaX + lightViewDeltaY * lightViewDeltaY + lightViewDeltaZ * lightViewDeltaZ) * RangeSliderViewRadii);
    for (std::size_t index = 0; index < scene.Lights.size(); ++index)
    {
        const auto& light = scene.Lights[index];
        if (light.Type != LightType::Point) continue;
        hasLocalLights = true;
        float range = light.Range;
        const auto label = std::string(light.Type == LightType::Point ? "Point " : "Area ") +
            std::to_string(index + 1) + " range";
        if (imguiSlider(label.c_str(), &range, 0.0f, std::max(rangeLimit, range), rangeLimit / RangeSliderSteps))
            editor.SetLightRange(index, range);
        const auto removeLabel = "Remove light " + std::to_string(index + 1);
        if (imguiButton(removeLabel.c_str())) { editor.RemoveLocalLight(index); break; }
    }
    if (hasLocalLights) HelpLabel("Light range: world units; 0 is unlimited.");
    std::vector<const char*> lightNames;std::vector<std::string> lightLabels;
    for (std::size_t i=0;i<scene.Lights.size();++i) lightLabels.push_back("Light "+std::to_string(i+1)+(scene.Lights[i].Type==LightType::Directional?" (directional)":scene.Lights[i].Type==LightType::Area?" (area)":" (point)"));
    for (const auto& label:lightLabels) lightNames.push_back(label.c_str());
    int selectedLight=editor.SelectedLight();
    if (!lightNames.empty() && imguiCombo("Selected light",&selectedLight,lightNames.data(),int(lightNames.size()))) editor.SelectLight(selectedLight);
    if (editor.SelectedLight()>=0 && std::size_t(editor.SelectedLight())<scene.Lights.size())
    {
        auto light=scene.Lights[editor.SelectedLight()];bool lightChanged=false;
        if (imguiCheck("Light enabled",light.Enabled)) {light.Enabled=!light.Enabled;lightChanged=true;}
        lightChanged|=imguiSlider("Intensity",&light.Intensity,0,std::max(100.0f,light.Intensity*2),0.1f);
        lightChanged|=imguiSlider("Light red",&light.LightColor.R,0,1,0.01f);
        lightChanged|=imguiSlider("Light green",&light.LightColor.G,0,1,0.01f);
        lightChanged|=imguiSlider("Light blue",&light.LightColor.B,0,1,0.01f);
        auto& vector=light.Type==LightType::Directional?light.Direction:light.Position;
        const float bound=light.Type==LightType::Directional?1.0f:5000.0f;
        lightChanged|=imguiSlider("Light X",&vector.X,-bound,bound,bound/1000);
        lightChanged|=imguiSlider("Light Y",&vector.Y,-bound,bound,bound/1000);
        lightChanged|=imguiSlider("Light Z",&vector.Z,-bound,bound,bound/1000);
        if (light.Type==LightType::Area)
        { lightChanged|=imguiSlider("Emitter width",&light.AreaSize.X,0.01f,30,0.01f);lightChanged|=imguiSlider("Emitter height",&light.AreaSize.Y,0.01f,30,0.01f); }
        if (lightChanged) editor.SetLight(std::size_t(editor.SelectedLight()),light);
        if (UsesLightmaps(requested.GlobalIllumination) || UsesBakedProbes(requested.GlobalIllumination)) imguiLabel("Baked indirect light is fixed; select Lumen, voxel or ray GI for moving-light bounce updates.");
    }

    if (native && !editor.IsDemoLoading())
    {
        const bool areaSupported=editor.QueryTechniqueSupport(Technique::AreaLights).Available();
        imguiPushVisibility(areaSupported);
        if (imguiCheck("Rectangle emitters",requested.AreaLights.Enabled))
        {
            requested.AreaLights.Enabled=!requested.AreaLights.Enabled; changed=true;
            if (requested.AreaLights.Enabled) requested.Shading=ShadingModel::PBR;
        }
        if (requested.AreaLights.Enabled)
        {
            static bool expanded=false;
            if (imguiCollapse("Emitter settings","",expanded)) expanded=!expanded;
            if (expanded)
            {
                imguiIndent();
                const bool pathEmitters=UsesPathTracing(requested.Pipeline);
                float samples=float(requested.AreaLights.SamplesPerAxis);
                if (!pathEmitters && imguiSlider("Emitter samples per axis",&samples,MinimumAreaLightSamplesPerAxis,MaximumAreaLightSamplesPerAxis,1))
                { requested.AreaLights.SamplesPerAxis=unsigned(samples); changed=true; }
                const char* units[]{"Radiance","Total power"};
                int unit=int(requested.AreaLights.EnergyUnit);
                if (imguiCombo("Emitter intensity unit",&unit,units,2))
                { requested.AreaLights.EnergyUnit=AreaLightEnergyUnit(unit); changed=true; }
                const auto capacity=pathEmitters || UsesLightLists(requested.Pipeline) ? Renderer::MaximumForwardPlusLights : Renderer::MaximumSceneLights;
                if (imguiButton("Add rectangle emitter",editor.Scene().Lights.size()<capacity)) editor.AddLocalLight(LightType::Area);
                for (std::size_t index=0;index<editor.Scene().Lights.size();++index)
                {
                    const auto& light=editor.Scene().Lights[index];
                    if (light.Type!=LightType::Area) continue;
                    float range=light.Range;
                    const auto label="Emitter "+std::to_string(index+1)+" range";
                    constexpr float MaximumRange=40,RangeStep=0.05f;
                    if (imguiSlider(label.c_str(),&range,0,std::max(range,MaximumRange),RangeStep)) editor.SetLightRange(index,range);
                    if (imguiButton(("Remove emitter "+std::to_string(index+1)).c_str())) { editor.RemoveLocalLight(index); break; }
                }
                imguiUnindent();
            }
        }
        imguiPopVisibility();
    }
    const auto shadowLight=std::find_if(scene.Lights.begin(),scene.Lights.end(),[](const Light& light) { return light.CastShadows; });
    sections.Next("Shadows",editor.QueryTechniqueSupport(Technique::ShadowMapping).Available() ||
        editor.QueryTechniqueSupport(Technique::RayTracedShadows).Available());
    if (imguiCheck(path || backend==Backend::DXR?"Ray shadows":"Primary shadows",editor.Settings().ShadowsEnabled))
        editor.SetShadowsEnabled(!editor.Settings().ShadowsEnabled);
    if (path)
    {
        imguiSetDisabledReason("Ray Tracing Full integrates light visibility; independent raster and screen-space shadow techniques are inactive.");
        const char* inactive[]{"Shadow map","Cascaded maps","Virtual shadow maps","Independent ray shadows","PCF","PCSS","VSM","Contact shadows"};
        for (const auto* name:inactive) imguiCheck(name,false,false);
        imguiSetDisabledReason(nullptr);
    }
    imguiPushVisibility(!path);
    if (native && editor.Settings().ShadowsEnabled)
    {
        const bool cascadesAvailable = QuerySupport(Technique::CascadedShadowMaps, backend).Available() && shadowLight!=scene.Lights.end() && shadowLight->Type==LightType::Directional;
        const bool virtualAvailable=editor.QueryTechniqueSupport(Technique::VirtualShadowMaps).Available();
        const bool rayAvailable=editor.QueryTechniqueSupport(Technique::RayTracedShadows).Available();
        std::vector<const char*> representations {"Shadow map","Cascaded maps","Virtual shadow maps","Ray-traced BVH"};
        std::vector<ShadowRepresentation> modes {ShadowRepresentation::SingleMap,ShadowRepresentation::Cascaded,ShadowRepresentation::Virtual,ShadowRepresentation::RayTraced};
        std::vector<std::string> shadowReasons(4);std::vector<const char*> shadowReasonPointers;
        if (!cascadesAvailable) shadowReasons[1]="Cascaded shadows require a shadow-casting directional light and native cascade support.";
        if (!virtualAvailable) shadowReasons[2]=editor.QueryTechniqueSupport(Technique::VirtualShadowMaps).Reason;
        if (!rayAvailable) shadowReasons[3]=editor.QueryTechniqueSupport(Technique::RayTracedShadows).Reason;
        for (unsigned i=0;i<4;++i) {if (path) shadowReasons[i]="Ray Tracing Full traces visibility toward every sampled emitter.";shadowReasonPointers.push_back(shadowReasons[i].c_str());}
        int shadowIndex=0;
        for (unsigned i=0;i<modes.size();++i) if (modes[i]==resolution.Effective.Shadows) shadowIndex=int(i);
        if (imguiCombo("Shadow representation",&shadowIndex,representations.data(),int(modes.size()),true,shadowReasonPointers.data()))
        { requested.Shadows=modes[shadowIndex]; changed=true; }
        if (requested.Shadows==ShadowRepresentation::RayTraced)
        {
            auto& ray=requested.RayShadows;
            changed|=imguiSlider("Ray shadow horizon",&ray.MaximumDistance,0.1f,10000.0f,0.1f);
            changed|=imguiSlider("Ray normal bias",&ray.NormalBias,0.0f,0.1f,0.0001f);
            changed|=imguiSlider("Ray start bias",&ray.StartBias,0.0f,0.01f,0.00001f);
            HelpLabel("Triangle BVH queries include off-screen casters and alpha cutouts. Point and area-center rays stop at the emitter. Map filters and contact shadows are inactive; their requests are retained.");
        }
        if (cascadesAvailable && resolution.Effective.Shadows == ShadowRepresentation::Cascaded)
        {
            const char* counts[] = {"1", "2", "3", "4"};
            int count = static_cast<int>(requested.Cascades.CascadeCount) - 1;
            if (imguiCombo("Cascades", &count, counts, 4, true))
            {
                requested.Cascades.CascadeCount = static_cast<std::uint32_t>(count + 1);
                changed = true;
            }
            changed |= imguiSlider("Split logarithmic weight", &requested.Cascades.SplitLambda, 0.0f, 1.0f, 0.01f);
            changed |= imguiSlider("Shadow distance", &requested.Cascades.MaximumDistance, 0.0f, 1000.0f, 0.1f);
            changed |= imguiSlider("Cascade overlap", &requested.Cascades.BlendFraction, 0.0f, 0.25f, 0.01f);
            HelpLabel("Directional light only. Point and area lights use a single map. Distance 0 uses camera far.");
        }
        if (virtualAvailable && resolution.Effective.Shadows==ShadowRepresentation::Virtual)
        {
            auto& pages=requested.VirtualShadows;
            const char* resolutions[]{"1024","2048","4096","8192","16384"};
            const std::uint32_t resolutionValues[]{1024,2048,4096,8192,16384};
            int selected=0; for (int index=0;index<5;++index) if (pages.Resolution==resolutionValues[index]) selected=index;
            if (imguiCombo("Virtual resolution",&selected,resolutions,5,true)) { pages.Resolution=resolutionValues[selected]; changed=true; }
            const char* sizes[]{"64","128","256"}; int size=pages.PageSide==256?2:pages.PageSide==128?1:0;
            if (imguiCombo("Page side",&size,sizes,3,true)) { pages.PageSide=VirtualShadowMinimumPageSide<<size; changed=true; }
            float budget=float(pages.PhysicalPages),retention=float(pages.MaximumUnusedFrames);
            if (imguiSlider("Resident page budget",&budget,1,float(VirtualShadowMaximumPhysicalPages),1)) { pages.PhysicalPages=unsigned(budget); changed=true; }
            if (imguiSlider("Retain unused frames",&retention,0,240,1)) { pages.MaximumUnusedFrames=unsigned(retention); changed=true; }
            changed|=imguiSlider("Shadow mip bias",&pages.MipBias,-4,4,0.25f);
            HelpLabel("Lower mip bias requests finer depth. Unmapped pages use a resident ancestor; old and new caster bounds invalidate cached depth.");
            if (resolution.Effective.Shadows!=ShadowRepresentation::Virtual)
                HelpLabel("Virtual request retained. Fiber shadow casting/reception currently uses a single map.");
            else HelpLabel("Hard/PCF page lookup. Virtual PCSS and moment filtering remain pending.");
            const auto& stats=editor.ViewportGeometry();
            if (stats.VirtualShadowSerial)
            {
                imguiLabel(("Shadow pages: "+std::to_string(stats.VirtualShadowRequestedPages)+" requested, "+std::to_string(stats.VirtualShadowResidentPages)+" resident").c_str());
                imguiLabel((std::to_string(stats.VirtualShadowRenderedPages)+" rendered, "+std::to_string(stats.VirtualShadowCopiedPages)+" copied; "+std::to_string(stats.VirtualShadowDrawCalls)+" caster draws").c_str());
            }
        }
        if (requested.Shadows!=ShadowRepresentation::RayTraced)
        {
        const bool virtualEffective=resolution.Effective.Shadows==ShadowRepresentation::Virtual;
        const bool pcssAvailable = !virtualEffective && editor.QueryTechniqueSupport(Technique::PCSS).Available();
        const bool vsmAvailable = !virtualEffective && editor.QueryTechniqueSupport(Technique::VSM).Available();
        std::vector<const char*> filters {"Hard (nearest)","PCF","PCSS","VSM"};
        std::vector<ShadowFilterMode> filterModes {ShadowFilterMode::Hard,ShadowFilterMode::PCF,ShadowFilterMode::PCSS,ShadowFilterMode::VSM};
        std::vector<std::string> filterReasons(4);std::vector<const char*> filterReasonPointers;
        for (unsigned i=0;i<4;++i)
        {
            if (path || resolution.Effective.Shadows==ShadowRepresentation::RayTraced) filterReasons[i]="The selected pipeline traces shadow visibility and produces no filterable shadow map.";
            else if (virtualEffective) filterReasons[i]="Virtual shadow pages use their own page filtering.";
            else if (i==2 && !pcssAvailable) filterReasons[i]=editor.QueryTechniqueSupport(Technique::PCSS).Reason;
            else if (i==3 && !vsmAvailable) filterReasons[i]=editor.QueryTechniqueSupport(Technique::VSM).Reason;
            filterReasonPointers.push_back(filterReasons[i].c_str());
        }
        int filter = 1;
        for (std::size_t index = 0; index < filterModes.size(); ++index)
            if (filterModes[index] == resolution.Effective.ShadowFilter.Mode) filter = int(index);
        if (imguiCombo("Shadow filter", &filter, filters.data(), int(filters.size()), true,filterReasonPointers.data()))
        { requested.ShadowFilter.Mode = filterModes[filter]; changed = true; }
        if (resolution.Effective.ShadowFilter.Mode == ShadowFilterMode::VSM)
        {
            if (!vsmAvailable) HelpLabel("VSM request retained. This backend or device uses PCF.");
            else
            {
                auto& vsm = requested.ShadowFilter.VSM;
                float radius = float(vsm.BlurRadius);
                if (imguiSlider("Moment blur radius (texels)", &radius, 0.0f, float(MaximumVSMBlurRadius), 1.0f))
                { vsm.BlurRadius = std::uint32_t(radius); changed = true; }
                changed |= imguiSlider("Moment blur sigma (texels)", &vsm.BlurSigma, 0.25f, 32.0f, 0.25f);
                float varianceExponent = std::log10(vsm.MinimumVariance);
                if (imguiSlider("Minimum variance (log10)", &varianceExponent, -8.0f, -1.0f, 0.1f))
                { vsm.MinimumVariance = std::pow(10.0f, varianceExponent); changed = true; }
                changed |= imguiSlider("Raster variance limit", &vsm.MaximumDerivativeVariance, 0.0f, 0.1f, 0.001f);
                changed |= imguiSlider("Light bleeding reduction", &vsm.LightBleedingReduction, 0.0f, 0.95f, 0.01f);
                HelpLabel("Filtered linear-depth moments. Larger blur softens edges; bleeding reduction can darken penumbrae. Compatible with CSM.");
            }
        }
        if (resolution.Effective.ShadowFilter.Mode == ShadowFilterMode::PCF)
        {
            const char* kernels[] = {"3 x 3", "5 x 5", "7 x 7", "9 x 9"};
            int radius = static_cast<int>(requested.ShadowFilter.Radius) - 1;
            if (imguiCombo("PCF comparison kernel", &radius, kernels, 4, true))
            {
                requested.ShadowFilter.Radius = static_cast<std::uint32_t>(radius + 1);
                changed = true;
            }
            HelpLabel("Fixed texel footprint; wider kernels cost more.");
        }
        if (resolution.Effective.ShadowFilter.Mode == ShadowFilterMode::PCSS)
        {
            if (!pcssAvailable) HelpLabel("PCSS request retained. This backend uses PCF until its PCSS implementation is available.");
            else
            {
                auto& pcss = requested.ShadowFilter.PCSS;
                if (shadowLight==scene.Lights.end() || shadowLight->Type!=LightType::Directional) changed |= imguiSlider("Finite light radius", &pcss.EmitterRadius, 0.0f, 10.0f, 0.01f);
                if (shadowLight!=scene.Lights.end() && shadowLight->Type==LightType::Directional) changed |= imguiSlider("Directional radius (degrees)", &pcss.AngularRadiusDegrees, 0.0f, 15.0f, 0.01f);
                float search = float(pcss.MaximumSearchRadius), radius = float(pcss.MaximumFilterRadius);
                if (imguiSlider("Blocker search limit (texels)", &search, 1.0f, float(MaximumPCSSRadius), 1.0f))
                { pcss.MaximumSearchRadius = std::uint32_t(search); changed = true; }
                if (imguiSlider("Penumbra limit (texels)", &radius, 1.0f, search, 1.0f))
                { pcss.MaximumFilterRadius = std::uint32_t(radius); changed = true; }
                const char* counts[] = {"16", "32", "64", "128"};
                int blockers = pcss.BlockerSamples == 16 ? 0 : pcss.BlockerSamples == 32 ? 1 : pcss.BlockerSamples == 64 ? 2 : 3;
                int samples = pcss.FilterSamples == 16 ? 0 : pcss.FilterSamples == 32 ? 1 : pcss.FilterSamples == 64 ? 2 : 3;
                if (imguiCombo("Blocker samples", &blockers, counts, 4, true))
                { pcss.BlockerSamples = 16u << blockers; changed = true; }
                if (imguiCombo("Penumbra samples", &samples, counts, 4, true))
                { pcss.FilterSamples = 16u << samples; changed = true; }
                HelpLabel("Larger emitters soften separated shadows. Zero radius gives hard comparisons. Search and filter limits also respect the shadow-map resolution.");
            }
        }
        }
    }

    imguiSeparatorLine();
    const bool fiberVisibilityContacts = UsesHairFur(resolution.Effective.Hair) &&
        (resolution.Effective.MeshShaders || resolution.Effective.Nanite.Enabled);
    const bool contactAvailable = editor.QueryTechniqueSupport(Technique::ContactShadows).Available() &&
        requested.Shadows!=ShadowRepresentation::RayTraced && !fiberVisibilityContacts;
    toggle("Contact shadows", requested.ContactShadows.Enabled, contactAvailable);
    if (fiberVisibilityContacts)
        HelpLabel("Contact shadows are inactive while fibers use mesh/Nanite visibility. The request is retained for switching back.");
    if (requested.ContactShadows.Enabled)
    {
        if (!contactAvailable && !fiberVisibilityContacts) HelpLabel("Contact shadow request retained. This backend or device has no active contact visibility support.");
        else if (contactAvailable)
        {
            auto& contact = requested.ContactShadows;
            changed |= imguiSlider("Contact distance", &contact.MaximumDistance, 0.001f, 10.0f, 0.01f);
            changed |= imguiSlider("Contact start bias", &contact.StartBias, 0.0f, 1.0f, 0.001f);
            changed |= imguiSlider("Surface thickness", &contact.Thickness, 0.0f, 2.0f, 0.001f);
            float steps = float(contact.MaximumSteps);
            if (imguiSlider("Contact trace steps", &steps, 1.0f, 256.0f, 1.0f))
            { contact.MaximumSteps = std::uint32_t(steps); changed = true; }
            HelpLabel("Supplement the primary shadow light on opaque surfaces. Combines with any shadow filter, including when shadow maps are off.");
        }
    }
    imguiSeparatorLine();
    imguiPopVisibility();
    sections.Next("Geometry & visibility",native && editor.Scene().Demo!=DemoId::StanfordDragon);
    const auto tessellationSupport=editor.QueryTechniqueSupport(Technique::Tessellation);
    toggle("PN triangle tessellation",requested.Tessellation.Enabled,native && tessellationSupport.Available());
    const bool pnDensity=native && tessellationSupport.Available() && UsesTessellation(requested.Tessellation);
    if (requested.Tessellation.Enabled)
    {
        BeginControlGroup("Tessellation settings",true);
        if (!tessellationSupport.Available()) HelpLabel(tessellationSupport.Reason);
        else
        {
            auto& tessellation=requested.Tessellation; float segments=float(tessellation.Segments);
            if (imguiSlider("PN segments",&segments,1,float(MaximumTessellationSegments),1)) { tessellation.Segments=std::uint32_t(segments); changed=true; }
            changed|=imguiSlider("PN curve strength",&tessellation.Curvature,0,1,0.01f);
            HelpLabel("Changes geometry before raster, shadow, capture and scene-ray passes. Curve strength blends linear triangles into curved PN patches; shading normals use a separate quadratic patch.");
            HelpLabel("Segments grow triangle count quadratically. One segment bypasses PN refinement. Each shared object is limited to 2 million vertices and 6 million indices; rejected settings preserve the previous geometry.");
            HelpLabel("PN runs before height displacement and supplies its topology. Geometry changes reset temporal correspondence; rigid motion reuses prepared buffers. Disabling both techniques restores the source exactly.");
        }
        EndControlGroup();
    }
    const auto displacementSupport=editor.QueryTechniqueSupport(Technique::DisplacementMapping);
    toggle("Geometric displacement",requested.Displacement.Enabled,native && displacementSupport.Available());
    if (requested.Displacement.Enabled)
    {
        BeginControlGroup("Displacement settings",true);
        if (!displacementSupport.Available()) HelpLabel(displacementSupport.Reason);
        else
        {
            auto& displacement=requested.Displacement;
            const char* projections[]{"Automatic longest axes","Object XY","Object XZ","Object YZ"}; int projection=int(displacement.Projection);
            if (imguiCombo("Displacement coordinates",&projection,projections,4,true)) { displacement.Projection=DisplacementProjection(projection); changed=true; }
            changed|=imguiSlider("Height / object extent",&displacement.HeightFraction,0,MaximumDisplacementFraction,0.001f);
            changed|=imguiSlider("Neutral scalar height",&displacement.Midlevel,0,1,0.01f);
            changed|=imguiSlider("Displacement tiling",&displacement.Tiling,MinimumDisplacementTiling,MaximumDisplacementTiling,0.1f);
            if (pnDensity) HelpLabel("PN segments supply the displacement topology. The separate displacement sampling request is retained for use when PN is inactive.");
            else
            {
                float segments=float(displacement.Segments);
                if (imguiSlider("Geometry sampling segments",&segments,1,float(MaximumDisplacementSegments),1)) { displacement.Segments=std::uint32_t(segments); changed=true; }
            }
            HelpLabel("Changes actual vertices and normals before all raster, shadow and scene-ray passes. Material sections of one object share their displacement context.");
            HelpLabel("Without active PN tessellation, displacement sampling refines the source on the CPU. Triangle count grows with the square of the segment count; budget failures preserve the previous result.");
            HelpLabel("Zero height bypasses displacement. Disabling it retains active PN geometry; disabling both restores the source. Geometry edits reset motion and temporal correspondence.");
        }
        EndControlGroup();
    }
    imguiPushVisibility(true);
    toggle(visibility?"Coarse frustum culling":resolution.Effective.GPUDriven ? "GPU frustum culling" : "Frustum culling", requested.FrustumCulling,
        native && QuerySupport(Technique::FrustumCulling, backend).Available());
    toggle("Back-face culling", requested.BackFaceCulling,
        native && QuerySupport(Technique::BackFaceCulling, backend).Available());
    const bool lodAvailable = native && editor.QueryTechniqueSupport(Technique::LOD).Available();
    if (visibility)
    { imguiSetDisabledReason("GPU Driven Visibility Buffer requires GPU cluster visibility and indirect drawing.");imguiCheck("GPU-driven rendering (required)",true,false);imguiSetDisabledReason(nullptr); }
    else toggle("GPU-driven rendering", requested.GPUDriven,
        native && !path && editor.QueryTechniqueSupport(Technique::GPUDriven).Available());
    if (resolution.Effective.GPUDriven)
    {
        bool required = true;
        imguiSetDisabledReason("GPU visibility compacts instance records and requires native instancing.");
        toggle("Instanced rendering (required)", required, false);
        imguiSetDisabledReason(nullptr);
        HelpLabel("GPU visibility, stable instance compaction and indirect drawing. Frustum culling controls rejection; LOD and material batching remain on the CPU.");
    }
    else toggle("Instanced rendering", requested.Instancing,
        native && !path && editor.QueryTechniqueSupport(Technique::Instancing).Available());
    const auto meshSupport=editor.QueryTechniqueSupport(Technique::MeshShaders);
    toggle("Native mesh shaders",requested.MeshShaders,meshSupport.Available() && (!requested.Nanite.Enabled || editor.QueryNaniteHardwareScheduling()));
    if (requested.MeshShaders)
    {
        if (!meshSupport.Available()) HelpLabel(meshSupport.Reason);
        else HelpLabel("Mesh stages produce single-sample visibility; material resolve remains deferred. Nanite uses GPU hardware bins. Transparent and shadow passes retain their selected geometry.");
        if (meshSupport.Available() && !requested.Nanite.Enabled)
        {
            const char* modes[]{"Automatic","Direct mesh","Task / amplification"};
            int mode=int(requested.MeshDispatch);
            if (imguiCombo("Mesh dispatch",&mode,modes,3))
            { requested.MeshDispatch=MeshShaderDispatchMode(mode); changed=true; }
            HelpLabel("Automatic uses direct mesh for small draws. Task mode requires device support; unavailable task stages fall back to direct mesh.");
        }
    }
    const auto naniteSupport=editor.QueryTechniqueSupport(Technique::Nanite);
    if (visibility)
    { imguiSetDisabledReason("The selected visibility pipeline owns resident mesh-cluster streaming and reconstruction.");imguiCheck("Mesh-cluster algorithms (required)",true,false);imguiSetDisabledReason(nullptr); }
    else toggle("Nanite algorithms",requested.Nanite.Enabled,native && naniteSupport.Available());
    if (resolution.Effective.Nanite.Enabled)
    {
        BeginControlGroup("Nanite streaming settings",resolution.Effective.Nanite.Enabled);
        if (!naniteSupport.Available()) HelpLabel(naniteSupport.Reason);
        else
        {
            auto& geometry=requested.Nanite;
            imguiPushEnabled(!visibility,"The selected GPU visibility pipeline requires this stage.");
            bool clusterOcclusion=resolution.Effective.Nanite.Occlusion,clusterInstances=resolution.Effective.Nanite.GpuInstances,clusterHardware=resolution.Effective.Nanite.GpuHardware,clusterTiles=resolution.Effective.Nanite.MaterialTiles;
            if (visibility)
            {
                imguiCheck("GPU cluster occlusion",clusterOcclusion,false);imguiCheck("GPU instance assembly",clusterInstances,false);
                imguiCheck("GPU hardware bins",clusterHardware,false);imguiCheck("GPU material tiles",clusterTiles,false);
            }
            else {
            toggle("Nanite two-phase cluster occlusion",geometry.Occlusion,true);
            toggle("Nanite GPU instance assembly",geometry.GpuInstances,true);
            toggle("Nanite GPU hardware bins",geometry.GpuHardware,editor.QueryNaniteHardwareScheduling() && !resolution.Effective.MeshShaders);
            if (resolution.Effective.MeshShaders) HelpLabel("Native mesh stages require GPU hardware bins; the effective setting is enabled.");
            if (!editor.QueryNaniteHardwareScheduling()) HelpLabel("GPU hardware-bin scheduling is unavailable on this device; per-range coverage remains available.");
            toggle("Nanite GPU material tiles",geometry.MaterialTiles,true);
            } imguiPopEnabled();
            changed|=imguiSlider("Nanite error (pixels)",&geometry.MaximumErrorPixels,0,MaximumNaniteErrorPixels,0.25f);
            float pages=float(geometry.PhysicalPages),scenePages=float(geometry.ScenePages),uploads=float(geometry.UploadsPerFrame),extent=float(geometry.MaximumSoftwareExtent);
            if (imguiSlider("Nanite pages per family (0 = automatic)",&pages,0,float(MaximumNanitePages),1)) { geometry.PhysicalPages=std::uint32_t(pages); changed=true; }
            if (imguiSlider("Nanite resident pages per scene generation",&scenePages,1,float(MaximumNaniteScenePages),1)) { geometry.ScenePages=std::uint32_t(scenePages); changed=true; }
            if (imguiSlider("Nanite detail pages per frame",&uploads,0,float(MaximumNaniteUploads),1)) { geometry.UploadsPerFrame=std::uint32_t(uploads); changed=true; }
            if (imguiSlider("Compute triangle extent (pixels)",&extent,1,32,1)) { geometry.MaximumSoftwareExtent=unsigned(extent); changed=true; }
            HelpLabel("Requires Visibility Buffer and single-sample rendering. Resident cuts are shared by materials, instances, shadows and reflections; transparent surfaces use their existing passes. Wire inspection uses hardware drawing of the same cut.");
            HelpLabel("Root uploads, page streaming and selection/decode use asynchronous jobs. Current geometry stays visible until a complete replacement is ready; GPU instance assembly batches shared streams; hardware bins compact coverage tasks; material tiles skip empty screen regions.");
        }
        EndControlGroup();
    }
    const auto virtualSupport=editor.QueryTechniqueSupport(Technique::VirtualGeometry);
    toggle("Virtual geometry",requested.VirtualGeometry.Enabled,native && virtualSupport.Available() && !resolution.Effective.Nanite.Enabled);
    if (requested.VirtualGeometry.Enabled)
    {
        BeginControlGroup("Virtual geometry settings",resolution.Effective.VirtualGeometry.Enabled);
        if (!virtualSupport.Available()) HelpLabel(virtualSupport.Reason);
        else
        {
            auto& geometry=requested.VirtualGeometry;
            changed|=imguiSlider("Cluster error (pixels)",&geometry.MaximumErrorPixels,MinimumVirtualGeometryError,MaximumVirtualGeometryError,MinimumVirtualGeometryError);
            changed|=imguiSlider("Cluster hysteresis",&geometry.HysteresisFraction,0,MaximumVirtualGeometryHysteresis,0.01f);
            float pages=float(geometry.PhysicalPages),uploads=float(geometry.UploadsPerFrame);
            if (imguiSlider("Pages per family (0 = automatic)",&pages,0,float(MaximumVirtualGeometryPages),1)) { geometry.PhysicalPages=std::uint32_t(pages); changed=true; }
            if (imguiSlider("Detail pages per frame",&uploads,0,float(MaximumVirtualGeometryUploads),1)) { geometry.UploadsPerFrame=std::uint32_t(uploads); changed=true; }
            HelpLabel("Complete cluster cuts are shared by material sections, instances, shadows and reflections. Root pages load before the first draw; zero detail uploads pauses refinement.");
        }
        EndControlGroup();
    }
    const bool virtualActive=resolution.Effective.VirtualGeometry.Enabled || resolution.Effective.Nanite.Enabled;
    toggle(virtualActive?"Whole-mesh LOD (inactive)":"Level of detail", requested.Lod.Enabled, lodAvailable && !virtualActive);
    if (virtualActive) HelpLabel("The active virtualized geometry mode owns detail selection. The whole-mesh LOD request is retained for restoration.");
    if (lodAvailable && requested.Lod.Enabled && !virtualActive)
    {
        changed |= imguiSlider("LOD error (pixels)", &requested.Lod.MaximumErrorPixels,
            MinimumLodErrorPixels, MaximumLodErrorPixels, MinimumLodErrorPixels);
        changed |= imguiSlider("LOD hysteresis", &requested.Lod.HysteresisFraction, 0, MaximumLodHysteresis, 0.01f);
        HelpLabel("Smaller error keeps finer detail. Shadows and reflections may require finer levels.");
    }
    imguiPopVisibility();
    {
    sections.Next("Ambient occlusion",native && editor.QueryTechniqueSupport(Technique::SSAO).Available());
    const Support aoSupport = editor.QueryTechniqueSupport(Technique::SSAO);
    const bool aoAvailable = native && aoSupport.Available();
    std::vector<const char*> aoNames{"None"};
    std::vector<AmbientOcclusionMode> aoModes{AmbientOcclusionMode::None};
    std::vector<std::string> aoReasons(1);
    const auto addAo=[&](const char* name,AmbientOcclusionMode mode,Technique technique)
    {
        auto candidate=requested;candidate.AmbientOcclusion.Mode=mode;const auto result=resolve(candidate);
        const auto support=editor.QueryTechniqueSupport(technique);
        aoNames.push_back(name);aoModes.push_back(mode);
        aoReasons.push_back(!support.Available()?support.Reason:result.Effective.AmbientOcclusion.Mode!=mode?"AO is owned by the selected ray transport pipeline.":"");
    };
    addAo("SSAO", AmbientOcclusionMode::SSAO, Technique::SSAO);
    addAo("HBAO", AmbientOcclusionMode::HBAO, Technique::HBAO);
    addAo("SAO", AmbientOcclusionMode::SAO, Technique::SAO);
    addAo("GTAO", AmbientOcclusionMode::GTAO, Technique::GTAO);
    addAo("Ray-traced AO", AmbientOcclusionMode::RayTraced, Technique::RayTracedAO);
    int aoIndex = 0;
    for (std::size_t i = 0; i < aoModes.size(); ++i)
        if (aoModes[i] == resolution.Effective.AmbientOcclusion.Mode) aoIndex = static_cast<int>(i);
    if (DrawControlChoices(aoIndex,aoNames,true,aoReasons))
    {
        requested.AmbientOcclusion.Mode = aoModes[static_cast<std::size_t>(aoIndex)];
        changed = true;
    }
    const bool ambientOcclusion = aoAvailable && requested.AmbientOcclusion.Mode!=AmbientOcclusionMode::None &&
        resolution.Effective.AmbientOcclusion.Mode==requested.AmbientOcclusion.Mode &&
        (requested.AmbientOcclusion.Mode != AmbientOcclusionMode::SAO || saoAvailable) &&
        (requested.AmbientOcclusion.Mode != AmbientOcclusionMode::RayTraced || rayAOAvailable);
    if (ambientOcclusion)
    {
        BeginControlGroup("AO settings",true,true);
        auto& ao = requested.AmbientOcclusion;
        changed |= imguiSlider("AO radius", &ao.Radius, 0.01f, 5.0f, 0.01f);
        if (ao.Mode == AmbientOcclusionMode::HBAO)
            changed |= imguiSlider("Horizon bias (degrees)", &ao.HorizonAngleBiasDegrees, 0.0f, 45.0f, 0.1f);
        else
            changed |= imguiSlider("AO bias", &ao.Bias, 0.0f, ao.Radius * 0.5f, 0.001f);
        changed |= imguiSlider("AO intensity", &ao.Intensity, 0.0f, 4.0f, 0.01f);
        if (ao.Mode == AmbientOcclusionMode::RayTraced)
        {
            float rays = static_cast<float>(ao.RayTracing.RayCount);
            if (imguiSlider("AO rays per pixel", &rays, MinimumAORays, MaximumAORays, 1))
            { ao.RayTracing.RayCount = static_cast<std::uint32_t>(rays); changed = true; }
        }
        imguiLabel(ao.Mode == AmbientOcclusionMode::RayTraced ?
            "Opaque ambient only. Scene visibility includes offscreen occluders." :
            "Opaque ambient only. Depth and normals limit visibility to this view.");
        imguiLabel(ao.Mode == AmbientOcclusionMode::RayTraced ?
            "Finite-radius cosine-weighted rays traverse scene geometry, with alpha cutouts and bilateral filtering." :
            ao.Mode == AmbientOcclusionMode::HBAO ?
            "HBAO searches rising horizons above the surface tangent." :
            ao.Mode == AmbientOcclusionMode::SAO ?
            "SAO reconstructs geometry from a depth hierarchy." :
            ao.Mode == AmbientOcclusionMode::GTAO ?
            "GTAO integrates cosine-weighted horizon visibility." :
            "SSAO projects a rotated hemisphere into scene depth.");
        if (!deferred) HelpLabel("Forward adds a material prepass for AO.");
        EndControlGroup();
    }
    else if (native && !aoAvailable) HelpLabel(aoSupport.Reason);
    else if (native && requested.AmbientOcclusion.Mode == AmbientOcclusionMode::SAO && !saoAvailable)
        HelpLabel(saoSupport.Reason);
    else if (native && requested.AmbientOcclusion.Mode == AmbientOcclusionMode::RayTraced && !rayAOAvailable)
        HelpLabel(rayAOSupport.Reason);

    imguiSeparatorLine();
    sections.Next("Global illumination",giAvailable || rayGIAvailable || voxelAvailable || lumenAvailable || probesAvailable);
    std::vector<const char*> giNames {"None"};
    std::vector<GlobalIlluminationMode> giModes {GlobalIlluminationMode::None};
    giNames={"None","Light probes","Lightmaps","Screen-space GI","Ray-traced scene","World-space voxel cones","Lumen"};
    giModes={GlobalIlluminationMode::None,GlobalIlluminationMode::BakedProbes,GlobalIlluminationMode::Lightmaps,GlobalIlluminationMode::ScreenSpace,GlobalIlluminationMode::RayTraced,GlobalIlluminationMode::Voxel,GlobalIlluminationMode::Lumen};
    std::vector<std::string> giReasons(giModes.size());
    for (std::size_t i=1;i<giModes.size();++i)
    {
        auto candidate=requested;candidate.GlobalIllumination.Mode=giModes[i];const auto result=resolve(candidate);
        if (result.Effective.GlobalIllumination.Mode!=giModes[i]) giReasons[i]=path?"Ray Tracing Full integrates indirect light along camera paths.":"This GI estimator is unavailable on the current backend or pipeline.";
        if ((giModes[i]==GlobalIlluminationMode::BakedProbes || giModes[i]==GlobalIlluminationMode::Lightmaps) && scene.Demo!=DemoId::SponzaAtrium) giReasons[i]="This scene has no baked lighting asset.";
    }
    int giIndex = -1;
    for (std::size_t i = 0; i < giModes.size(); ++i)
        if (giModes[i] == resolution.Effective.GlobalIllumination.Mode) giIndex = static_cast<int>(i);
    if (DrawControlChoices(giIndex,giNames,true,giReasons))
    {
        requested.GlobalIllumination.Mode = giModes[static_cast<std::size_t>(giIndex)];
        changed = true;
    }
    if (UsesBakedProbes(resolution.Effective.GlobalIllumination))
    {
        changed|=imguiSlider("Indirect intensity",&requested.GlobalIllumination.Intensity,0,4,0.01f);
        if (editor.ViewportBackend()==editor.SelectedBackend() && !editor.ViewportBakedLightingStatus().empty())
            imguiLabel(editor.ViewportBakedLightingStatus().c_str());
    }
    if (lumenAvailable && (resolution.Effective.GlobalIllumination.Mode==GlobalIlluminationMode::Lumen || resolution.Effective.Reflections.Mode==ReflectionMode::Lumen))
    {
        toggle("Hardware triangle tracing",requested.GlobalIllumination.Lumen.HardwareRayTracing,editor.QueryLumenHardwareRayTracing());
        if (requested.GlobalIllumination.Lumen.HardwareRayTracing && !editor.QueryLumenHardwareRayTracing())
            HelpLabel("Hardware queries are unavailable on this device. Lumen uses distance-field tracing.");
    }
    if (lumenAvailable && resolution.Effective.GlobalIllumination.Mode==GlobalIlluminationMode::Lumen)
    {
        auto& gi=requested.GlobalIllumination; auto& lumen=gi.Lumen;
        changed |= imguiSlider("GI intensity",&gi.Intensity,0,4,0.01f);
        const auto quality=[&](const char* label,unsigned& value,unsigned minimum,const char** names,int count) {
            int selected=0; for (auto n=minimum;n<value && selected<count-1;n*=2) ++selected;
            if (imguiCombo(label,&selected,names,count,true)) { value=minimum<<selected; changed=true; }
        };
        BeginControlGroup("Probe and surface quality");
        const char* tiles[]{"16 pixels","32 pixels","64 pixels","128 pixels"}; quality("Screen probe spacing",lumen.TileSide,16,tiles,4);
        const char* rays[]{"16 rays","32 rays","64 rays","128 rays","256 rays"}; quality("Rays per probe",lumen.RaysPerProbe,16,rays,5);
        const char* cards[]{"8 pixels","16 pixels","32 pixels"}; quality("Surface card resolution",lumen.CardSide,8,cards,3);
        toggle("Refine surface card coverage",lumen.RefineCards,true);
        float maximumCards=float(lumen.MaximumMeshCards);
        if (imguiSlider("Maximum cards per mesh part",&maximumCards,1,64,1)) { lumen.MaximumMeshCards=unsigned(maximumCards); changed=true; }
        if (lumen.RefineCards) changed |= imguiSlider("Target sampled texel coverage",&lumen.CardCoverage,0.5f,1,0.01f);
        float proxyResolution=float(lumen.SourceResolution); if (imguiSlider("Mesh distance-field resolution",&proxyResolution,16,64,1)) { lumen.SourceResolution=unsigned(proxyResolution); changed=true; }
        EndControlGroup();
        changed |= imguiSlider("GI trace distance",&gi.MaximumDistance,0.01f,1000,0.01f);
        changed |= imguiSlider("Probe normal bias",&gi.NormalBias,0,1,0.001f);
        changed |= imguiSlider("Receiver plane tolerance",&gi.DepthTolerance,0,0.25f,0.001f);
        changed |= imguiSlider("Receiver normal agreement",&gi.NormalThreshold,0,0.99f,0.01f);
        imguiPushVisibility(UsesHairFur(resolution.Effective.Hair));
        toggle("Hair indirect lighting",lumen.HairIndirect,true);
        toggle("Hair volume transport",lumen.HairTransport,true);
        if (lumen.HairTransport)
        {
            const char* hairGrids[]{"4 cells","8 cells","16 cells","32 cells","64 cells"};
            quality("Hair volume longest axis",lumen.HairGridSide,4,hairGrids,5);
            HelpLabel("Directional fiber extinction and low-frequency sky scattering use the existing animated groom. Increasing the grid retains finer density variations.");
        }
        imguiPopVisibility();
        toggle("Translucent indirect lighting",lumen.Translucency.Enabled,true);
        if (lumen.Translucency.Enabled || (requested.Hair.Enabled && lumen.HairIndirect))
        {
        // Refractive glass consumes the same irradiance volume as alpha blending.
        BeginControlGroup("Receiver volume settings",requested.AlphaBlending || UsesRefraction(resolution.Effective) ||
            (UsesHairFur(resolution.Effective.Hair) && lumen.HairIndirect));
            toggle("Volume visibility support",lumen.Translucency.VisibilitySupport,true);
            if (lumen.Translucency.VisibilitySupport) HelpLabel("Reject volume probes separated from the receiver by opaque distance-field surfaces. Coarse grids can lose indirect light near geometry.");
            const char* volumeTiles[]{"32 pixels","64 pixels","128 pixels","256 pixels"};
            quality("Volume probe spacing",lumen.Translucency.TileSide,32,volumeTiles,4);
            quality("Volume rays per probe",lumen.Translucency.RaysPerProbe,16,rays,3);
            float slices=float(lumen.Translucency.SliceCount);
            if (imguiSlider("Volume depth slices",&slices,4,32,1)) { lumen.Translucency.SliceCount=unsigned(slices); changed=true; }
            changed |= imguiSlider("Volume depth distribution",&lumen.Translucency.DepthExponent,1,4,0.1f);
            EndControlGroup();
        }
        toggle("Surface-cache direct shadows",lumen.Shadows,true);
        toggle("Surface-cache indirect bounces",lumen.Radiosity.Enabled,true);
        if (lumen.Radiosity.Enabled)
        {
        BeginControlGroup("Surface radiosity settings",true);
            float bounces=float(lumen.Radiosity.Bounces);
            if (imguiSlider("Indirect bounce count",&bounces,1,8,1)) { lumen.Radiosity.Bounces=unsigned(bounces); changed=true; }
            const char* samples[]{"4 rays","8 rays","16 rays","32 rays","64 rays"};
            quality("Indirect rays per card texel",lumen.Radiosity.RaysPerTexel,4,samples,5);
            EndControlGroup();
        }
        toggle("Temporal diffuse GI",lumen.Temporal.Enabled,true);
        if (lumen.Temporal.Enabled)
        {
            BeginControlGroup("GI temporal settings");
            auto& temporal=lumen.Temporal;
            changed |= imguiSlider("GI temporal weight",&temporal.HistoryWeight,0,0.98f,0.01f);
            float frames=float(temporal.MaximumFrames);
            if (imguiSlider("GI history frame limit",&frames,1,64,1)) { temporal.MaximumFrames=unsigned(frames); changed=true; }
            changed |= imguiSlider("History relative depth tolerance",&temporal.DepthTolerance,0,0.1f,0.001f);
            changed |= imguiSlider("History normal agreement",&temporal.NormalThreshold,0,1,0.01f);
            changed |= imguiSlider("History plane tolerance",&temporal.PlaneTolerance,0,0.25f,0.001f);
            changed |= imguiSlider("History color allowance",&temporal.ClipTolerance,0,1,0.01f);
                EndControlGroup();
        }
        toggle("Adaptive screen probes",lumen.AdaptiveProbes,true);
        if (lumen.AdaptiveProbes)
        {
            float levels=float(lumen.AdaptiveProbeLevels);
            if (imguiSlider("Screen probe refinement levels",&levels,1,float(MaximumLumenAdaptiveProbeLevels),1))
            { lumen.AdaptiveProbeLevels=unsigned(levels); changed=true; }
        }
        toggle("Probe interpolation visibility",lumen.ProbeVisibility.Enabled,true);
        if (lumen.ProbeVisibility.Enabled)
        {
        BeginControlGroup("Probe visibility settings",true);
            float steps=float(lumen.ProbeVisibility.MaximumSteps);
            if (!resolution.Effective.GlobalIllumination.Lumen.HardwareRayTracing && imguiSlider("Probe visibility step budget",&steps,1,512,1)) { lumen.ProbeVisibility.MaximumSteps=unsigned(steps); changed=true; }
            changed |= imguiSlider("Probe visibility normal bias",&lumen.ProbeVisibility.NormalBias,0,1,0.001f);
            EndControlGroup();
        }
        float captureBudget=float(lumen.CaptureCardsPerUpdate);
        if (imguiSlider("Material cards per update",&captureBudget,1,4096,1)) { lumen.CaptureCardsPerUpdate=unsigned(captureBudget); changed=true; }
        float lightingBudget=float(lumen.LightingTilesPerUpdate);
        if (imguiSlider("Surface lighting tiles per update",&lightingBudget,1,4096,1)) { lumen.LightingTilesPerUpdate=unsigned(lightingBudget); changed=true; }
        const bool softwareTracing=!resolution.Effective.GlobalIllumination.Lumen.HardwareRayTracing;
        toggle("Global distance clipmaps",lumen.GlobalDistance.Enabled,softwareTracing);
        if (!softwareTracing) HelpLabel("Software distance clipmaps are inactive during hardware triangle tracing.");
        if (lumen.GlobalDistance.Enabled && softwareTracing)
        {
        BeginControlGroup("Distance clipmap settings",true);
            auto& global=lumen.GlobalDistance;
            float side=float(global.GridSide),levels=float(global.CascadeCount),steps=float(global.MaximumSteps);
            if (imguiSlider("Global field cells per axis",&side,8,64,1)) { global.GridSide=unsigned(side); changed=true; }
            if (imguiSlider("Global field cascades",&levels,1,4,1)) { global.CascadeCount=unsigned(levels); changed=true; }
            changed|=imguiSlider("Global field cell / horizon",&global.CellScale,0.001f,0.1f,0.001f);
            changed|=imguiSlider("Mesh detail distance / horizon",&global.NearScale,0,1,0.01f);
            if (imguiSlider("Global field trace steps",&steps,1,512,1)) { global.MaximumSteps=unsigned(steps); changed=true; }
            float budget=float(global.VoxelsPerUpdate);
            if (imguiSlider("Global field voxels per update",&budget,1,65536,1)) { global.VoxelsPerUpdate=unsigned(budget); changed=true; }
            HelpLabel("Complete GPU generations replace the current field asynchronously. Empty regions use clipmaps; surfaces and uncertain regions use mesh SDFs.");
            EndControlGroup();
        }
        toggle("World radiance cache",lumen.WorldCache.Enabled,true);
        if (lumen.WorldCache.Enabled)
        {
        BeginControlGroup("World cache settings",true);
            auto& world=lumen.WorldCache; toggle("World cache sphere parallax",world.Parallax,true);
            float levels=float(world.CascadeCount);
            if (imguiSlider("World cache cascades",&levels,1,4,1)) { world.CascadeCount=unsigned(levels); changed=true; }
            float depthLayers=float(world.DepthLayers);
            if (imguiSlider("World cache depth layers",&depthLayers,1,float(MaximumLumenWorldDepthLayers),1)) { world.DepthLayers=unsigned(depthLayers); changed=true; }
            if (world.DepthLayers>1) HelpLabel("Additional depth layers improve parallax across occluders and cost memory and tracing work. Layered visibility checks complete SDF rays.");
            float side=float(world.GridSide);
            if (imguiSlider("World probes per axis",&side,2,8,1)) { world.GridSide=unsigned(side); changed=true; }
            const char* directions[]{"4 x 4 directions","8 x 8 directions","16 x 16 directions","32 x 32 directions"};
            float updateBudget=float(world.ProbesPerUpdate);
            if (imguiSlider("World probes traced per frame",&updateBudget,1,1024,1)) { world.ProbesPerUpdate=unsigned(updateBudget); changed=true; }
            const auto probes=world.GridSide*world.GridSide*world.GridSide*world.CascadeCount;
            int choices=int(std::size(directions)); unsigned maximumSide=4u<<(choices-1);
            while (probes*world.DepthLayers*maximumSide*maximumSide>LumenWorldRadianceMaximumRays) { --choices; maximumSide/=2; }
            quality("World probe angular resolution",world.DirectionalSide,4,directions,choices);
            float angularRadius=float(world.AngularFilterRadius);
            if (imguiSlider("World angular filter radius",&angularRadius,0,float(LumenWorldMaximumFilterRadius),1)) { world.AngularFilterRadius=unsigned(angularRadius); changed=true; }
            if (world.AngularFilterRadius) HelpLabel("Wider angular filtering requires a complete footprint with consistent depth and visibility. Surface boundaries and unknown data use full SDF tracing.");
            changed |= imguiSlider("Finest spacing / GI distance",&world.CellScale,0.005f,0.2f/float(1u<<(world.CascadeCount-1)),0.005f);
            if (world.CascadeCount>1) changed |= imguiSlider("Cascade blend width (cells)",&world.BlendCells,0,2,0.05f);
            HelpLabel("Each level scrolls on its own cell grid. Unchanged probes are copied on the GPU; new probes follow the per-frame budget. Complete generations replace the old cache atomically. Unknown footprints still use full SDF rays, and near visibility covers every contributing level.");
            EndControlGroup();
        }
        toggle("Screen-first hybrid tracing",lumen.ScreenTrace.Enabled,true);
        if (lumen.ScreenTrace.Enabled)
        {
        BeginControlGroup("Screen trace settings",true);
            toggle("Hierarchical screen depth",lumen.ScreenTrace.Hierarchical,true);
            float steps=float(lumen.ScreenTrace.MaximumSteps);
            if (imguiSlider("Screen trace node budget",&steps,1,1024,1)) { lumen.ScreenTrace.MaximumSteps=unsigned(steps); changed=true; }
            changed |= imguiSlider("Screen surface thickness",&lumen.ScreenTrace.Thickness,0,1,0.001f);
            changed |= imguiSlider("Screen pixel stride",&lumen.ScreenTrace.PixelStride,1,4,0.25f);
            EndControlGroup();
        }
        imguiPushVisibility(scene.Demo!=DemoId::SponzaAtrium);
        const char* channels[]{"Sky radiance R","Sky radiance G","Sky radiance B"};
        for (unsigned c=0;c<3;++c) changed |= imguiSlider(channels[c],&lumen.SkyRadiance[c],0,4,0.01f);
        changed|=DrawLumenEnvironment(lumen.Environment,"GI");
        imguiPopVisibility();
        HelpLabel("Screen hits use current opaque HDR; unknown rays fall back to mesh distance fields and surface cards. Single-layer screen visibility and coarse proxies can miss detail. Diffuse temporal reuse and surface-cache reflections are available.");
        if (editor.ViewportBackend()==editor.SelectedBackend() && !editor.ViewportLumenGIStatus().empty())
            imguiLabel((std::string("Lumen: ")+editor.ViewportLumenGIStatus()).c_str());
    }
    else if (voxelAvailable && resolution.Effective.GlobalIllumination.Mode==GlobalIlluminationMode::Voxel)
    {
        auto& gi=requested.GlobalIllumination; auto& voxels=gi.Voxels;
        changed |= imguiSlider("GI intensity",&gi.Intensity,0,4,0.01f);
        const char* resolutions[]{"8 x 8 x 8","16 x 16 x 16","32 x 32 x 32","64 x 64 x 64"};
        int resolutionIndex=0; for (auto value=MinimumVoxelResolution;value<voxels.Resolution;value*=2) ++resolutionIndex;
        if (imguiCombo("Voxel resolution",&resolutionIndex,resolutions,4,true))
        { voxels.Resolution=MinimumVoxelResolution<<resolutionIndex; changed=true; }
        toggle("Fit voxel volume to scene",voxels.AutomaticBounds,true);
        if (!voxels.AutomaticBounds)
        {
            const char* axes[]{"X","Y","Z"};
            for (unsigned axis=0;axis<3;++axis)
            {
                const float range=std::max({10.0f,std::abs(voxels.BoundsMinimum[axis])*2,std::abs(voxels.BoundsMaximum[axis])*2});
                changed |= imguiSlider((std::string("Voxel minimum ")+axes[axis]).c_str(),&voxels.BoundsMinimum[axis],-range,range,0.01f);
                changed |= imguiSlider((std::string("Voxel maximum ")+axes[axis]).c_str(),&voxels.BoundsMaximum[axis],-range,range,0.01f);
            }
        }
        toggle("Fit cone distance to volume",voxels.AutomaticDistance,true);
        if (!voxels.AutomaticDistance) changed |= imguiSlider("GI cone distance",&gi.MaximumDistance,0.01f,1000,0.01f);
        toggle("Voxel injection shadows",voxels.Shadows,true);
        float cones=static_cast<float>(voxels.ConeCount),steps=static_cast<float>(voxels.MaximumSteps);
        if (imguiSlider("GI hemisphere cones",&cones,1,16,1)) { voxels.ConeCount=static_cast<unsigned>(cones); changed=true; }
        if (imguiSlider("GI cone step budget",&steps,1,256,1)) { voxels.MaximumSteps=static_cast<unsigned>(steps); changed=true; }
        changed |= imguiSlider("Cone aperture (degrees)",&voxels.ConeApertureDegrees,1,120,1);
        changed |= imguiSlider("Cone step / diameter",&voxels.StepScale,0.125f,1,0.025f);
        changed |= imguiSlider("GI normal bias (cells)",&voxels.NormalBiasCells,0,8,0.01f);
        changed |= imguiSlider("GI depth tolerance (cells)",&voxels.DepthSigmaCells,0.01f,16,0.01f);
        changed |= imguiSlider("GI normal rejection",&voxels.NormalPower,0,64,1);
        HelpLabel("One diffuse bounce from world-space geometry, lights and emission. Coarse cells can leak light; finite-volume misses are black.");
        HelpLabel("Scene and material edits rebuild voxels; light edits only reinject radiance. Camera and cone controls reuse the volume.");
        if (editor.ViewportBackend()==editor.SelectedBackend() && !editor.ViewportVoxelGIStatus().empty())
            imguiLabel((std::string("Voxel GI: ")+editor.ViewportVoxelGIStatus()).c_str());
        if (!deferred) HelpLabel("Forward adds GI material inputs.");
    }
    else if (rayGIAvailable && resolution.Effective.GlobalIllumination.Mode==GlobalIlluminationMode::RayTraced)
    {
        auto& gi=requested.GlobalIllumination;
        changed |= imguiSlider("GI intensity",&gi.Intensity,0.0f,4.0f,0.01f);
        changed |= imguiSlider("GI segment distance",&gi.MaximumDistance,0.01f,1000.0f,0.01f);
        changed |= imguiSlider("GI surface bias",&gi.NormalBias,0.0f,std::min(10.0f,gi.MaximumDistance),0.001f);
        float paths=static_cast<float>(gi.RayTracing.RayCount),bounces=static_cast<float>(gi.RayTracing.BounceCount);
        if (imguiSlider("GI paths per pixel",&paths,float(MinimumGIRays),float(MaximumGIRays),1.0f))
        { gi.RayTracing.RayCount=static_cast<std::uint32_t>(paths); changed=true; }
        if (imguiSlider("Diffuse bounces",&bounces,float(MinimumDiffuseBounces),float(MaximumDiffuseBounces),1.0f))
        { gi.RayTracing.BounceCount=static_cast<std::uint32_t>(bounces); changed=true; }
        toggle("GI hit shadows",gi.RayTracing.Shadows,true);
        HelpLabel("Diffuse scene paths reach off-screen geometry and material textures. Misses are black.");
        HelpLabel("Spatial filtering preserves depth, normal and albedo edges. No private temporal history.");
        if (!deferred) HelpLabel("Forward adds GI material inputs.");
    }
    else if (giAvailable && resolution.Effective.GlobalIllumination.Mode==GlobalIlluminationMode::ScreenSpace)
    {
        auto& gi = requested.GlobalIllumination;
        changed |= imguiSlider("GI intensity", &gi.Intensity, 0.0f, 4.0f, 0.01f);
        changed |= imguiSlider("GI distance (world)", &gi.MaximumDistance, 0.01f, 1000.0f, 0.01f);
        changed |= imguiSlider("GI thickness (world)", &gi.Thickness, 0.001f, 10.0f, 0.001f);
        changed |= imguiSlider("GI bias (world)", &gi.NormalBias, 0.0f, gi.Thickness, 0.001f);
        changed |= imguiSlider("GI edge fade (pixels)", &gi.EdgeFadePixels, 0.0f, 64.0f, 1.0f);
        changed |= imguiSlider("GI history weight", &gi.HistoryWeight, 0.0f, 0.98f, 0.01f);
        changed |= imguiSlider("GI depth tolerance", &gi.DepthTolerance, 0.0f, 0.1f, 0.001f);
        changed |= imguiSlider("GI normal rejection", &gi.NormalThreshold, 0.0f, 1.0f, 0.01f);
        float rays = static_cast<float>(gi.RayCount), steps = static_cast<float>(gi.MaximumSteps);
        if (imguiSlider("GI rays per pixel", &rays, 1.0f, 32.0f, 1.0f)) { gi.RayCount = static_cast<std::uint32_t>(rays); changed = true; }
        if (imguiSlider("GI ray steps", &steps, 4.0f, 256.0f, 4.0f)) { gi.MaximumSteps = static_cast<std::uint32_t>(steps); changed = true; }
        HelpLabel("Adds one diffuse bounce from visible opaque color. Missing surfaces retain existing ambient light.");
        HelpLabel("Uses HDR, material capture and motion history. Works with AO and reflections; TAA jitter is optional.");
    }
    else if (native && requested.GlobalIllumination.Mode==GlobalIlluminationMode::Lumen && !lumenAvailable) HelpLabel(lumenSupport.Reason);
    else if (native && requested.GlobalIllumination.Mode==GlobalIlluminationMode::Voxel && !voxelAvailable) HelpLabel(voxelSupport.Reason);
    else if (native && requested.GlobalIllumination.Mode==GlobalIlluminationMode::RayTraced && !rayGIAvailable) HelpLabel(rayGISupport.Reason);
    else if (native && !giAvailable) HelpLabel(giSupport.Reason);
    imguiSeparator();

    const Support reflectionSupport = editor.QueryTechniqueSupport(Technique::SSR);
    const bool reflectionAvailable = native && reflectionSupport.Available();
    const bool sharedSky=scene.Environment && (scene.Demo==DemoId::UtahTeapot || scene.Demo==DemoId::SponzaAtrium);
    sections.Next("Environment",native && (scene.World || sharedSky));
    if (sharedSky || (scene.Environment && scene.Demo==DemoId::BigWorld))
    {
        toggle("Skybox",requested.Reflections.Skybox,native && !path);
        if (resolution.Effective.Reflections.Skybox) changed|=imguiSlider("Sky intensity",&requested.Reflections.SkyboxIntensity,0,4,0.01f);
        if (scene.Demo==DemoId::SponzaAtrium)
            changed|=imguiSlider("Diffuse sky intensity",&requested.Reflections.DiffuseIBL,0,4,0.01f,native && !path);
        imguiLabel(scene.Demo==DemoId::BigWorld ? "Procedural day-night sky" : "Kloppenheim 06 Pure Sky / Poly Haven");
    }
    changed|=DrawWorldEnvironmentControls(editor,requested);
    sections.Next("Scene reflections",native);
    imguiSetDisabledReason(hybrid ? "Ray Tracing Hybrid owns screen/scene reflection integration." :
        path ? "Ray Tracing Full integrates reflection along camera paths." : nullptr);
    if (imguiCheck("None",requested.Reflections.Mode==ReflectionMode::None,!hybrid && !path) && requested.Reflections.Mode!=ReflectionMode::None)
    {
        requested.Reflections.Mode=ReflectionMode::None;
        changed=true;
    }
    imguiSetDisabledReason(nullptr);
    {
        const Support cubeSupport = editor.QueryTechniqueSupport(Technique::CubemapReflections);
        const Support mapSupport = editor.QueryTechniqueSupport(Technique::EnvironmentReflections);
        const bool cubeAvailable = native && cubeSupport.Available();
        const bool mapAvailable = native && mapSupport.Available();
        std::vector<const char*> environmentNames;
        std::vector<ReflectionMode> environmentModes;
        int environmentIndex=0;
        for (auto mode:{ReflectionMode::Cubemap,ReflectionMode::LatitudeLongitude})
        {
            if (mode==ReflectionMode::Cubemap ? !cubeAvailable : !mapAvailable) continue;
            auto candidate=requested; candidate.Reflections.Mode=mode;
            if (resolve(candidate).Effective.Reflections.Mode!=mode) continue;
            if (mode==requested.Reflections.Mode) environmentIndex=static_cast<int>(environmentModes.size());
            environmentNames.push_back(mode==ReflectionMode::Cubemap ? "Cubemap" : "Latitude-longitude");
            environmentModes.push_back(mode);
        }
        const bool environmentSelected=UsesEnvironmentCapture(requested.Reflections);
        imguiSetDisabledReason(path || hybrid ? "The selected ray pipeline integrates reflections; independent environment mapping is inactive." :
            !cubeAvailable ? cubeSupport.Reason : mapSupport.Reason);
        if (imguiCheck("Environment reflections",environmentSelected,!environmentModes.empty()))
        {
            requested.Reflections.Mode=environmentSelected ? ReflectionMode::None : environmentModes.front();
            if (!environmentSelected && requested.Reflections.EnvironmentIntensity<=0)
                requested.Reflections.EnvironmentIntensity=ReflectionSettings{}.EnvironmentIntensity;
            changed=true;
        }
        imguiSetDisabledReason(nullptr);
        if (UsesEnvironmentCapture(requested.Reflections) && !environmentModes.empty())
        {
            if (imguiCombo("Environment mapping",&environmentIndex,environmentNames.data(),static_cast<int>(environmentNames.size())))
            { requested.Reflections.Mode=environmentModes[environmentIndex]; changed=true; }
            changed|=imguiSlider("Environment intensity",&requested.Reflections.EnvironmentIntensity,0,4,0.01f);
            imguiLabel(scene.Demo==DemoId::BigWorld ? "Procedural day-night sky" :
                "Kloppenheim 06 Pure Sky / Poly Haven");
            HelpLabel(scene.World ? "Both mappings use the procedural sky." : "Both mappings use the same HDR panorama. Roughness selects averaged mips.");
        }
    }
    const std::array<const char*,3> reflectionLabels{"Screen-space (SSR)","Ray-traced scene","Lumen surface cache"};
    const std::array<ReflectionMode,3> reflectionChoices{ReflectionMode::ScreenSpace,ReflectionMode::RayTraced,ReflectionMode::Lumen};
    const std::array<bool,3> reflectionSupported{reflectionAvailable,rayReflectionAvailable,lumenAvailable};
    std::vector<const char*> reflectionNames;
    std::vector<ReflectionMode> reflectionModes;
    int reflectionIndex=-1;
    std::vector<std::string> reflectionReasons;
    for (std::size_t mode=0;mode<reflectionLabels.size();++mode)
    {
        auto candidate=requested;candidate.Reflections.Mode=reflectionChoices[mode];const auto result=resolve(candidate);
        if (reflectionChoices[mode]==requested.Reflections.Mode) reflectionIndex=int(mode);
        reflectionNames.push_back(reflectionLabels[mode]);reflectionModes.push_back(reflectionChoices[mode]);
        reflectionReasons.push_back(hybrid?"Ray Tracing Hybrid owns screen/scene reflection integration.":path?"Ray Tracing Full integrates reflection along camera paths.":!reflectionSupported[mode]?"Unavailable on this backend.":result.Effective.Reflections.Mode!=candidate.Reflections.Mode?"Unavailable with the current pipeline.":"");
    }
    if (DrawControlChoices(reflectionIndex,reflectionNames,true,reflectionReasons))
    {
        requested.Reflections.Mode=reflectionModes[reflectionIndex];
        if ((scene.Demo==DemoId::StanfordDragon || scene.World) && requested.Reflections.Mode!=ReflectionMode::None)
            requested.Reflections.Planar.Enabled=false;
        changed=true;
    }
    changed|=DrawDragonReflectionControls(editor,requested);
    const bool lumenReflectionRequested=!hybrid && UsesLumenReflections(requested.Reflections);
    const bool rayRequested = hybrid || requested.Reflections.Mode == ReflectionMode::RayTraced;
    const bool sceneReflectionRequested=requested.Reflections.Mode==ReflectionMode::ScreenSpace || rayRequested || lumenReflectionRequested;
    if ((hybrid || (sceneReflectionRequested && resolution.Effective.Reflections.Mode==requested.Reflections.Mode)) && (lumenReflectionRequested ? lumenAvailable : rayRequested ? rayReflectionAvailable : reflectionAvailable))
    {
        auto& reflection = requested.Reflections;
        changed |= imguiSlider("Reflection intensity", &reflection.Intensity, 0.0f, 4.0f, 0.01f);
        if (lumenReflectionRequested)
            toggle("Shared hardware tracing",requested.GlobalIllumination.Lumen.HardwareRayTracing,editor.QueryLumenHardwareRayTracing());
        BeginControlGroup("Reflection settings");
        if (lumenReflectionRequested)
        {
            auto& lumen=reflection.Lumen;
            changed |= imguiSlider("Reflection trace distance",&reflection.MaximumDistance,0.01f,scene.World?6000.0f:1000.0f,0.01f);
            changed |= imguiSlider("Reflection normal bias",&reflection.NormalBias,0,std::min(10.0f,reflection.MaximumDistance),0.001f);
            changed |= imguiSlider("Reflection roughness limit",&reflection.MaximumRoughness,0.045f,1,0.01f);
            const char* resolutions[]{"Full","Half","Quarter"}; int divisor=lumen.ResolutionDivisor==4?2:lumen.ResolutionDivisor==2?1:0;
            if (imguiCombo("Reflection trace resolution",&divisor,resolutions,3)) { lumen.ResolutionDivisor=1u<<divisor; changed=true; }
            float steps=float(reflection.MaximumSteps),radius=float(lumen.FilterRadius);
            if (!resolution.Effective.GlobalIllumination.Lumen.HardwareRayTracing && imguiSlider("Reflection trace step budget",&steps,1,512,1)) { reflection.MaximumSteps=unsigned(steps); changed=true; }
            toggle("Reflection screen tracing",lumen.ScreenTracing,true);
            if (lumen.ScreenTracing)
            {
                changed |= imguiSlider("Reflection screen thickness",&reflection.Thickness,0.001f,10,0.001f);
                changed |= imguiSlider("Reflection screen stride",&reflection.StridePixels,1,4,1);
            }
            if (imguiSlider("Reflection reconstruction radius",&radius,0,2,1)) { lumen.FilterRadius=unsigned(radius); changed=true; }
            changed |= imguiSlider("Reflection normal agreement",&lumen.NormalCosine,0,1,0.01f);
            changed |= imguiSlider("Reflection plane tolerance",&lumen.PlaneTolerance,0,0.25f,0.001f);
            changed |= imguiSlider("Reflection relative plane tolerance",&lumen.RelativePlaneTolerance,0,0.1f,0.001f);
            changed |= imguiSlider("Reflection roughness tolerance",&lumen.RoughnessTolerance,0,1,0.01f);
            toggle("Temporal Lumen reflections",lumen.Temporal.Enabled,true);
            if (lumen.Temporal.Enabled)
            {
                auto& history=lumen.Temporal; float length=float(history.MaximumFrames);
                changed |= imguiSlider("Reflection history weight",&history.HistoryWeight,0,0.98f,0.01f);
                if (imguiSlider("Reflection history frames",&length,1,64,1)) { history.MaximumFrames=unsigned(length); changed=true; }
                changed |= imguiSlider("Reflection history depth tolerance",&history.DepthTolerance,0,0.1f,0.001f);
                changed |= imguiSlider("Reflection history normal agreement",&history.NormalThreshold,0,1,0.01f);
                changed |= imguiSlider("Reflection history plane tolerance",&history.PlaneTolerance,0,0.25f,0.001f);
                changed |= imguiSlider("Reflection history color allowance",&history.ClipTolerance,0,1,0.01f);
            }
            HelpLabel("Uses screen hits followed by mesh-distance tracing and the shared Lumen surface cache. Diffuse GI is optional.");
            if (!UsesLumen(requested.GlobalIllumination))
            {
                auto& cache=requested.GlobalIllumination.Lumen; imguiLabel("Shared Lumen scene cache");
                float fieldResolution=float(cache.SourceResolution),cards=float(cache.MaximumMeshCards),captureBudget=float(cache.CaptureCardsPerUpdate),lightingBudget=float(cache.LightingTilesPerUpdate);
                if (imguiSlider("Reflection mesh field resolution",&fieldResolution,16,64,1)) { cache.SourceResolution=unsigned(fieldResolution); changed=true; }
                const char* sizes[]{"8 pixels","16 pixels","32 pixels"}; int side=cache.CardSide==32?2:cache.CardSide==16?1:0;
                if (imguiCombo("Reflection surface card size",&side,sizes,3)) { cache.CardSide=8u<<side; changed=true; }
                toggle("Refine reflection surface cards",cache.RefineCards,true);
                if (imguiSlider("Reflection cards per mesh part",&cards,1,64,1)) { cache.MaximumMeshCards=unsigned(cards); changed=true; }
                changed |= imguiSlider("Reflection card sampled coverage",&cache.CardCoverage,0.5f,1,0.01f);
                if (imguiSlider("Reflection cards per update",&captureBudget,1,4096,1)) { cache.CaptureCardsPerUpdate=unsigned(captureBudget); changed=true; }
                if (imguiSlider("Reflection lighting tiles per update",&lightingBudget,1,4096,1)) { cache.LightingTilesPerUpdate=unsigned(lightingBudget); changed=true; }
                toggle("Reflection cache direct shadows",cache.Shadows,true); toggle("Reflection cache indirect bounces",cache.Radiosity.Enabled,true);
                if (cache.Radiosity.Enabled)
                {
                    float bounces=float(cache.Radiosity.Bounces); if (imguiSlider("Reflection cache bounce count",&bounces,1,8,1)) { cache.Radiosity.Bounces=unsigned(bounces); changed=true; }
                    const char* samples[]{"4 rays","8 rays","16 rays","32 rays","64 rays"}; int sampleIndex=0;
                    for (auto count=cache.Radiosity.RaysPerTexel;count>4;count/=2) ++sampleIndex;
                    if (imguiCombo("Reflection cache indirect rays",&sampleIndex,samples,5)) { cache.Radiosity.RaysPerTexel=4u<<sampleIndex; changed=true; }
                }
                imguiPushVisibility(scene.Demo!=DemoId::SponzaAtrium);
                changed |= imguiSlider("Shared cache sky red",&cache.SkyRadiance[0],0,16,0.01f);
                changed |= imguiSlider("Shared cache sky green",&cache.SkyRadiance[1],0,16,0.01f);
                changed |= imguiSlider("Shared cache sky blue",&cache.SkyRadiance[2],0,16,0.01f);
                changed|=DrawLumenEnvironment(cache.Environment,"Refl");
                imguiPopVisibility();
            }
        }
        else if (rayRequested)
        {
            changed |= imguiSlider("Ray distance (world)", &reflection.MaximumDistance, 0.01f, scene.World?6000.0f:1000.0f, 0.01f);
            changed |= imguiSlider("Normal bias (world)", &reflection.NormalBias, 0.0f, std::min(reflection.MaximumDistance,10.0f), 0.001f);
            changed |= imguiSlider("Reflection roughness limit", &reflection.MaximumRoughness, 0.045f, 1.0f, 0.01f);
            float rays=static_cast<float>(reflection.RayTracing.RayCount);
            if (imguiSlider("Reflection rays", &rays, static_cast<float>(MinimumReflectionRays), static_cast<float>(MaximumReflectionRays), 1.0f))
            { reflection.RayTracing.RayCount=static_cast<std::uint32_t>(rays); changed=true; }
            toggle("Secondary-hit shadows",reflection.RayTracing.Shadows,true);
            if (hybrid)
            {
                toggle("Screen-space reflection hits", requested.Hybrid.ScreenSpace, true);
                toggle("Scene-ray reflection fallback", requested.Hybrid.RayFallback, true);
                if (requested.Hybrid.ScreenSpace)
                {
                    changed |= imguiSlider("Screen hit thickness", &reflection.Thickness, 0.001f, 10.0f, 0.001f);
                    changed |= imguiSlider("Screen edge fade (pixels)", &reflection.EdgeFadePixels, 0.0f, 64.0f, 1.0f);
                    changed |= imguiSlider("Screen ray stride (pixels)", &reflection.StridePixels, 1.0f, 8.0f, 1.0f);
                    float steps = static_cast<float>(reflection.MaximumSteps);
                    if (imguiSlider("Screen ray steps", &steps, 16, 512, 16)) { reflection.MaximumSteps = static_cast<std::uint32_t>(steps); changed = true; }
                }
                HelpLabel("Screen hits reuse visible radiance. Missing or faded hits use scene rays; both share the same GGX sample.");
                HelpLabel("The independent reflection selection is retained for other pipelines.");
            }
            HelpLabel("One scene-BVH bounce includes hidden and off-screen opaque/cutout surfaces.");
            HelpLabel("GGX sampling and bilateral filtering. Misses are black; no recursive or temporal accumulation.");
        }
        else
        {
            changed |= imguiSlider("Ray distance (world)", &reflection.MaximumDistance, 0.01f, scene.World?6000.0f:1000.0f, 0.01f);
            changed |= imguiSlider("Hit thickness (world)", &reflection.Thickness, 0.001f, 10.0f, 0.001f);
            changed |= imguiSlider("Normal bias (world)", &reflection.NormalBias, 0.0f, reflection.Thickness, 0.001f);
            changed |= imguiSlider("Reflection roughness limit", &reflection.MaximumRoughness, 0.045f, 1.0f, 0.01f);
            changed |= imguiSlider("Edge fade (pixels)", &reflection.EdgeFadePixels, 0.0f, 64.0f, 1.0f);
            changed |= imguiSlider("Ray stride (pixels)", &reflection.StridePixels, 1.0f, 8.0f, 1.0f);
            float steps = static_cast<float>(reflection.MaximumSteps);
            if (imguiSlider("Ray step budget", &steps, 16.0f, 512.0f, 16.0f))
            {
                reflection.MaximumSteps = static_cast<std::uint32_t>(steps);
                changed = true;
            }
            HelpLabel("Reflects visible opaque color. Hidden and off-screen surfaces contribute nothing.");
            HelpLabel("Rough surfaces fade this mirror estimate; temporal and roughness filtering are pending.");
        }
        if (!deferred) HelpLabel("Forward adds a material prepass for reflections.");
        EndControlGroup();
    }
    else if (rayRequested && !rayReflectionAvailable) HelpLabel(rayReflectionSupport.Reason);
    else if (!reflectionAvailable) HelpLabel(reflectionSupport.Reason);
    imguiSeparatorLine();
    }
    sections.Next("Transparency",native && scene.Demo==DemoId::StanfordDragon,"Transparency controls require a native renderer and configured translucent materials.");
    changed|=DrawTransparencyControls(editor,requested);
    const bool atmosphereAvailable=native && !path && DemoCatalog::IsVolumeDemo(scene.Demo) &&
        !editor.IsDemoLoading() && fogSupport.Available();
    sections.Next("Atmosphere",atmosphereAvailable,"Atmosphere techniques require Big World and a compatible raster pipeline.");
    changed|=DrawVolumeControls(editor,requested);
    sections.Next("Water Rendering",native && bool(scene.World) && !editor.IsDemoLoading(),"Water controls require the Big World scene and a native renderer backend.");
    changed|=DrawWaterControls(editor,requested);
    sections.Next("Fur Rendering",native && scene.Demo==DemoId::FurCube);
    changed|=DrawHairControls(editor,requested);
    sections.Next("Camera & display");
    if (imguiCheck("Selection outline",editor.Settings().OutlineEnabled,!path))
        editor.SetOutlineEnabled(!editor.Settings().OutlineEnabled);
    const bool dofAvailable = native && editor.QueryTechniqueSupport(Technique::DepthOfField).Available();
    auto& lens = requested.PostProcessing.DepthOfField;
    toggle("Depth of field", lens.Enabled, dofAvailable);
    const bool depthOfField = dofAvailable && lens.Enabled;
    if (depthOfField)
    {
        BeginControlGroup("Lens settings",true);
        const auto& camera = editor.Scene().SceneCamera;
        const float dx = camera.Target.X - camera.Position.X;
        const float dy = camera.Target.Y - camera.Position.Y;
        const float dz = camera.Target.Z - camera.Position.Z;
        const float targetDistance = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 0.01f);
        if (imguiButton("Focus camera target")) { lens.FocusDistance = targetDistance; changed = true; }
        changed |= imguiSlider("Focus distance (world)", &lens.FocusDistance, 0.01f, targetDistance * 4.0f, 0.01f);
        changed |= imguiSlider("Aperture f-number", &lens.FNumber, 0.7f, 32.0f, 0.1f);
        changed |= imguiSlider("Sensor width (mm)", &lens.SensorWidthMillimeters, 1.0f, 100.0f, 1.0f);
        changed |= imguiSlider("Meters per world unit", &lens.MetersPerWorldUnit, 0.01f, 1.0f, 0.01f);
        changed |= imguiSlider("Maximum blur (pixels)", &lens.MaxRadiusPixels, 0.0f, 16.0f, 0.5f);
        HelpLabel("Focal length follows the camera field of view. Smaller f-numbers blur more.");
        HelpLabel("Opaque scene only; transparency stays sharp. Hidden surfaces cannot be recovered.");
        EndControlGroup();
    }

    imguiSeparatorLine();

    const bool motionAvailable = native && !path && QuerySupport(Technique::MotionBlur, backend).Available();
    auto& motion = requested.PostProcessing.MotionBlur;
    toggle("Motion blur", motion.Enabled, motionAvailable);
    const bool motionBlur = motionAvailable && motion.Enabled;
    if (motionBlur)
    {
        BeginControlGroup("Motion blur settings",true);
        float shutterMilliseconds = motion.ShutterSeconds * 1000.0f;
        if (imguiSlider("Shutter (ms)", &shutterMilliseconds, 0.0f, 100.0f, 0.5f))
        {
            motion.ShutterSeconds = shutterMilliseconds / 1000.0f;
            changed = true;
        }
        changed |= imguiSlider("Motion radius (pixels)", &motion.MaxRadiusPixels, 0.0f, 16.0f, 0.5f);
        HelpLabel("Camera and rigid-object motion. First frames and camera cuts stay sharp.");
        HelpLabel("Opaque scene before depth of field; transparency stays sharp.");
        EndControlGroup();
    }
    sections.Next("Color & exposure",native);
    HdrSettings& hdr = requested.PostProcessing.HDR;
    const bool toneAvailable = native && QuerySupport(Technique::ToneMapping, backend).Available();
    const bool bloomAvailable = native && QuerySupport(Technique::Bloom, backend).Available();
    const bool gradeAvailable = native && QuerySupport(Technique::ColorGrading, backend).Available();
    std::vector<const char*> toneNames;
    std::vector<ToneMappingMode> toneModes;
    if (!toneAvailable || (!path && !hdr.Bloom && !hdr.ColorGrading && !deferred && !UsesAmbientOcclusion(resolution.Effective.AmbientOcclusion) && !depthOfField && !motionBlur &&
        !UsesTemporalAA(resolution.Effective.PostProcessing) && !UsesVolumetricFog(resolution.Effective.VolumetricFog) &&
        !UsesLightShafts(resolution.Effective.LightShafts) && !resolution.Effective.PostProcessing.TemporalEffects.Enabled))
    {
        toneNames.push_back("None");
        toneModes.push_back(ToneMappingMode::None);
    }
    toneNames.push_back("Reinhard");
    toneNames.push_back("ACES filmic fit");
    toneModes.push_back(ToneMappingMode::Reinhard);
    toneModes.push_back(ToneMappingMode::ACESFitted);
    const auto effectiveTone = toneAvailable ? EffectiveToneMapping(resolution.Effective.PostProcessing.HDR) : ToneMappingMode::None;
    int toneIndex = 0;
    for (std::size_t index = 0; index < toneModes.size(); ++index)
        if (toneModes[index] == effectiveTone) toneIndex = static_cast<int>(index);
    imguiLabel("Tone mapping");
    if (imguiCombo("", &toneIndex, toneNames.data(), static_cast<int>(toneNames.size()), toneAvailable))
    {
        hdr.ToneMapping = toneModes[static_cast<std::size_t>(toneIndex)];
        changed = true;
    }
    if (toneAvailable && RequiresHDR(resolution.Effective.PostProcessing.HDR))
        changed |= imguiSlider("Exposure (EV)", &hdr.ExposureEV, -10.0f, 10.0f, 0.1f);
    const auto exposureSupport=editor.QueryTechniqueSupport(Technique::TemporalEffects);
    toggle("Temporal exposure adaptation",requested.PostProcessing.TemporalEffects.Enabled,native && exposureSupport.Available());
    if (requested.PostProcessing.TemporalEffects.Enabled && native && exposureSupport.Available())
    {
        BeginControlGroup("Exposure adaptation settings",true);
        auto& exposure=requested.PostProcessing.TemporalEffects;
        changed|=imguiSlider("Meter key value",&exposure.KeyValue,MinimumExposureKey,MaximumExposureKey,0.01f);
        changed|=imguiSlider("Minimum automatic EV",&exposure.MinimumExposureEV,-MaximumAutoExposureEV,exposure.MaximumExposureEV,0.1f);
        changed|=imguiSlider("Maximum automatic EV",&exposure.MaximumExposureEV,exposure.MinimumExposureEV,MaximumAutoExposureEV,0.1f);
        changed|=imguiSlider("Brighten half-life (seconds)",&exposure.BrightenHalfLife,MinimumAdaptationHalfLife,MaximumAdaptationHalfLife,0.01f);
        changed|=imguiSlider("Darken half-life (seconds)",&exposure.DarkenHalfLife,MinimumAdaptationHalfLife,MaximumAdaptationHalfLife,0.01f);
        HelpLabel("Meters final HDR after transparency, before bloom and tone mapping. Manual EV remains an additional compensation. Independent of TAA and velocity.");
        EndControlGroup();
    }
    const bool previousBloom = hdr.Bloom;
    const bool flareAvailable = native && QuerySupport(Technique::LensFlare, backend).Available();
    toggle("Lens flare", hdr.LensFlare, flareAvailable);
    if (flareAvailable && hdr.LensFlare)
    {
        BeginControlGroup("Flare settings",true);
        changed |= imguiSlider("Flare threshold", &hdr.FlareThreshold, 0.0f, MaximumFlareThreshold, 0.05f);
        changed |= imguiSlider("Flare intensity", &hdr.FlareIntensity, 0.0f, MaximumFlareIntensity, 0.01f);
        changed |= imguiSlider("Ghost spacing", &hdr.FlareGhostSpacing, MinimumFlareGhostSpacing, MaximumFlareGhostSpacing, 0.01f);
        changed |= imguiSlider("Halo radius / height", &hdr.FlareHaloRadius, 0.0f, MaximumFlareHaloRadius, 0.01f);
        HelpLabel("Visible HDR highlights; flare precedes bloom. Zero intensity bypasses it.");
        EndControlGroup();
    }
    toggle("Bloom", hdr.Bloom, bloomAvailable);
    if (bloomAvailable && hdr.Bloom)
    {
        BeginControlGroup("Bloom settings",true);
        if (!previousBloom && hdr.ToneMapping == ToneMappingMode::None)
            hdr.ToneMapping = ToneMappingMode::Reinhard;
        changed |= imguiSlider("Bloom threshold", &hdr.BloomThreshold, 0.0f, 64.0f, 0.05f);
        changed |= imguiSlider("Bloom intensity", &hdr.BloomIntensity, 0.0f, 4.0f, 0.01f);
        EndControlGroup();
    }
    const bool previousGrade = hdr.ColorGrading;
    toggle("Color grading", hdr.ColorGrading, gradeAvailable);
    if (gradeAvailable && hdr.ColorGrading)
    {
        BeginControlGroup("Color grading settings",true);
        if (!previousGrade && hdr.ToneMapping == ToneMappingMode::None)
            hdr.ToneMapping = ToneMappingMode::Reinhard;
        changed |= imguiSlider("Contrast", &hdr.Contrast, 0.0f, 2.0f, 0.01f);
        changed |= imguiSlider("Saturation", &hdr.Saturation, 0.0f, 2.0f, 0.01f);
        EndControlGroup();
    }
    if (toneAvailable && (hdr.Bloom || hdr.ColorGrading || UsesLensFlare(hdr)))
        HelpLabel("Flare, bloom and grading require tone mapping.");
    if (depthOfField) HelpLabel("Depth of field requires tone mapping.");
    if (motionBlur) HelpLabel("Motion blur requires tone mapping.");
    if (!native) HelpLabel("HDR effects unavailable on this backend.");
    else if (!toneAvailable) HelpLabel("HDR integration pending.");

    imguiSeparatorLine();
    sections.Next("Anti-aliasing",native);
    struct AntiAliasingChoice
    {
        std::uint32_t Samples;
        AntiAliasingMode Mode;
    };
    std::vector<const char*> antiAliasingNames;
    std::vector<AntiAliasingChoice> antiAliasingChoices;
    std::vector<std::string> aaReasons;
    int antiAliasingIndex=-1;
    const auto addAntiAliasing=[&](const char* name,std::uint32_t samples,AntiAliasingMode mode,bool available)
    {

        auto candidate=requested;
        candidate.SampleCount=samples;
        candidate.PostProcessing.AntiAliasing=mode;
        const auto effective=resolve(candidate).Effective;
        aaReasons.push_back(!available?"Unavailable on the current backend or sample layout.":effective.SampleCount!=samples || effective.PostProcessing.AntiAliasing!=mode?"This pipeline owns antialiasing or requires single-sample visibility.":"");
        if (resolution.Effective.SampleCount==samples && resolution.Effective.PostProcessing.AntiAliasing==mode)
            antiAliasingIndex=static_cast<int>(antiAliasingChoices.size());
        antiAliasingNames.push_back(name);
        antiAliasingChoices.push_back({samples,mode});
    };
    addAntiAliasing("Off",1,AntiAliasingMode::None,true);
    const bool msaaAvailable=!path && !visibility && editor.QueryTechniqueSupport(Technique::MSAA).Available();
    const auto sampleCounts=editor.QueryMultisampleCounts() & (UsesWeightedTransparency(resolution.Effective) ?
        editor.QueryWeightedTransparencySampleCounts() : AllMultisampleCounts);
    addAntiAliasing("2x MSAA",2,AntiAliasingMode::None,msaaAvailable && (sampleCounts & 2));
    addAntiAliasing("4x MSAA",4,AntiAliasingMode::None,msaaAvailable && (sampleCounts & 4));
    addAntiAliasing("8x MSAA",8,AntiAliasingMode::None,msaaAvailable && (sampleCounts & 8));
    addAntiAliasing("FXAA",1,AntiAliasingMode::FXAA,fxaaAvailable);
    addAntiAliasing("TAA",1,AntiAliasingMode::TAA,taaAvailable);
    addAntiAliasing("TSR",1,AntiAliasingMode::TSR,tsrAvailable);
    if (DrawControlChoices(antiAliasingIndex,antiAliasingNames,true,aaReasons))
    {
        const auto& choice=antiAliasingChoices[static_cast<std::size_t>(antiAliasingIndex)];
        requested.SampleCount=choice.Samples;
        post.AntiAliasing=choice.Mode;
        changed=true;
    }
    if (!path && taaAvailable && UsesTemporalAA(post))
    {
        changed |= imguiSlider("TAA history weight", &post.TemporalAA.HistoryWeight, 0.0f, 0.98f, 0.01f);
        changed |= imguiSlider("TAA depth tolerance", &post.TemporalAA.DepthTolerance, 0.0001f, 0.1f, 0.001f);
        HelpLabel("Opaque HDR temporal reconstruction.");
        HelpLabel("Jitter and history reset on scene changes and camera cuts. Transparency is composed afterward.");
    }
    else if (!path && UsesTemporalAA(post)) HelpLabel(taaSupport.Reason);
    if (!path && tsrAvailable && UsesTemporalSuperResolution(post))
    {
        auto& reconstruction=post.TemporalSuperResolution;
        const auto& geometry=editor.ViewportGeometry();
        if (geometry.TSRPasses)
        {
            imguiLabel(("Scene input: "+std::to_string(geometry.RenderWidth)+" x "+std::to_string(geometry.RenderHeight)).c_str());
            if (geometry.TSRForegroundPixels)
            {
                const auto percentage=100ull*geometry.TSRAccumulatedPixels/geometry.TSRForegroundPixels;
                imguiLabel(("Surface history: "+std::to_string(percentage)+"%").c_str());
                HelpLabel("Visible surface pixels accumulating more than one temporal sample.");
            }
        }
        changed |= imguiSlider("TSR render scale",&reconstruction.RenderScale,0.5f,1.0f,0.01f);
        changed |= imguiSlider("TSR history weight",&reconstruction.MaximumHistoryWeight,0.0f,0.99f,0.01f);
        changed |= imguiSlider("TSR depth tolerance",&reconstruction.DepthTolerance,0.0f,0.1f,0.001f);
        changed |= imguiSlider("TSR variance gamma",&reconstruction.VarianceGamma,0.5f,3.0f,0.05f);
        changed |= imguiSlider("TSR reactive scale",&reconstruction.ReactiveScale,0.0f,4.0f,0.05f);
        changed |= imguiSlider("TSR sample limit",&reconstruction.MaximumSamples,1.0f,64.0f,1.0f);
        HelpLabel("Reconstructs output resolution after transparency. Lower render scale reduces scene detail and rendering work. TAA, FXAA and MSAA are alternative choices in this menu.");
        HelpLabel("Render scale changes preserve compatible output history. Scene changes, output resizing and camera cuts reset it. Selection outlines stay current.");
    }
    else if (!path && UsesTemporalSuperResolution(post)) HelpLabel(tsrSupport.Reason);
    sections.Next("Image effects",native);
    const bool nprAvailable = native && QuerySupport(Technique::NPR, backend).Available();
    toggle("NPR ink and hatching", post.NPR.Enabled, nprAvailable);
    if (nprAvailable && post.NPR.Enabled)
    {
        BeginControlGroup("Ink and hatching settings",true);
        auto& ink = post.NPR;
        changed |= imguiSlider("NPR blend", &ink.Blend, 0.0f, 1.0f, 0.01f);
        changed |= imguiSlider("Contour strength", &ink.EdgeStrength, 0.0f, 1.0f, 0.01f);
        changed |= imguiSlider("Contour sigma (pixels)", &ink.EdgeSigma, 0.5f, 1.5f, 0.05f);
        changed |= imguiSlider("Wide Gaussian gain", &ink.EdgeWeight, 0.8f, 1.0f, 0.005f);
        changed |= imguiSlider("Contour threshold", &ink.EdgeThreshold, -0.2f, 0.2f, 0.005f);
        changed |= imguiSlider("Contour sharpness", &ink.EdgeSharpness, 1.0f, 100.0f, 1.0f);
        changed |= imguiSlider("Hatch strength", &ink.HatchStrength, 0.0f, 1.0f, 0.01f);
        changed |= imguiSlider("Hatch spacing (pixels)", &ink.HatchSpacing, 4.0f, 32.0f, 0.5f);
        changed |= imguiSlider("Hatch width (pixels)", &ink.HatchWidth, 0.25f, std::min(4.0f, ink.HatchSpacing * 0.5f), 0.05f);
        changed |= imguiSlider("Hatch angle (degrees)", &ink.HatchAngleDegrees, -180.0f, 180.0f, 1.0f);
        changed |= imguiSlider("Retain source color", &ink.ColorStrength, 0.0f, 1.0f, 0.01f);
        float levels = static_cast<float>(ink.ToneLevels);
        if (imguiSlider("Posterization levels", &levels, 2.0f, 16.0f, 1.0f))
        { ink.ToneLevels = static_cast<std::uint32_t>(levels); changed = true; }
        HelpLabel("Image-space ink after tone mapping, before distortion and FXAA. Composes with Toon/PBR; zero blend bypasses. Hatches stay in screen coordinates.");
        EndControlGroup();
    }
    const bool vignetteAvailable = native && QuerySupport(Technique::Vignette, backend).Available();
    const bool grainAvailable = native && QuerySupport(Technique::FilmGrain, backend).Available();
    const bool distortionAvailable = native && QuerySupport(Technique::ScreenSpaceDistortion, backend).Available();
    toggle("Screen-space distortion", post.ScreenSpaceDistortion, distortionAvailable);
    if (distortionAvailable && post.ScreenSpaceDistortion)
    {
        changed |= imguiSlider("Radial strength", &post.DistortionStrength,
            MinimumDistortionStrength, MaximumDistortionStrength, 0.01f);
        HelpLabel("Radial color warp; edges clamp. Zero bypasses the pass.");
    }
    toggle("Vignette", post.Vignette, vignetteAvailable);
    if (vignetteAvailable && post.Vignette)
        changed |= imguiSlider("Vignette strength", &post.VignetteStrength, 0.0f, 1.0f, 0.01f);
    toggle("Film grain", post.FilmGrain, grainAvailable);
    if (grainAvailable && post.FilmGrain)
    {
        BeginControlGroup("Grain settings",true);
        constexpr float MaximumGrainStrength = 0.25f;
        changed |= imguiSlider("Grain strength", &post.FilmGrainStrength,
            0.0f, MaximumGrainStrength, 0.001f);
        imguiLabel(("Static grain / seed " + std::to_string(post.FilmGrainSeed)).c_str());
        // Integer buttons preserve every bit of a seed supplied by code; a float
        // slider would lose precision for large uint32 values. Wrap is defined.
        if (imguiButton("Previous grain seed"))
        {
            --post.FilmGrainSeed;
            changed = true;
        }
        if (imguiButton("Next grain seed"))
        {
            ++post.FilmGrainSeed;
            changed = true;
        }
        EndControlGroup();
    }
    if (!native) HelpLabel("Post effects unavailable on this backend.");
    else if (!fxaaAvailable && !vignetteAvailable && !grainAvailable && !distortionAvailable)
        HelpLabel("Post effect integration pending.");
    else HelpLabel("Order: distortion, FXAA, vignette, grain");
    sections.End();
    if (changed) editor.SetTechniqueSettings(requested);
}

} // namespace Evans::Editor
