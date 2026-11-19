#include "Editor/TechniqueControls.h"
#include "Editor/Editor.hpp"
#include "Editor/Imgui.h"
#include <vector>

namespace Evans::Editor
{
bool DrawDragonMaterialControls(EditorCore& editor,Techniques::Settings& settings)
{
    using namespace Techniques;
    if (editor.Scene().Demo!=DemoId::StanfordDragon || editor.IsDemoLoading() || !imguiIsEnabled() ||
        !editor.QueryTechniqueSupport(Technique::SubsurfaceScattering).Available() || UsesPathTracing(settings.Pipeline))
    {
        imguiSetDisabledReason("Jade diffusion requires the Dragon demo and a compatible raster pipeline.");
        imguiCheck("1. Jade diffusion",settings.Subsurface.Enabled,false);
        imguiSetDisabledReason(nullptr);
        return false;
    }
    bool changed=false;
    const auto toggle=[&](const char* label,bool& enabled)
    { if (imguiCheck(label,enabled)) { enabled=!enabled; changed=true; } };
    if (editor.QueryTechniqueSupport(Technique::SubsurfaceScattering).Available())
    {
        auto& diffusion=settings.Subsurface;
        toggle("1. Jade diffusion",diffusion.Enabled);
        static bool profile=false;
        if (diffusion.Enabled && imguiCollapse("Diffusion profile","",profile)) profile=!profile;
        if (diffusion.Enabled && profile)
        {
            imguiIndent();
            changed|=imguiSlider("SSS strength",&diffusion.Strength,0,1,0.01f);
            changed|=imguiSlider("SSS distance",&diffusion.Distance,0,1,0.005f);
            changed|=imguiSlider("SSS red radius",&diffusion.ChannelRadius[0],0.001f,1,0.01f);
            changed|=imguiSlider("SSS green radius",&diffusion.ChannelRadius[1],0.001f,1,0.01f);
            changed|=imguiSlider("SSS blue radius",&diffusion.ChannelRadius[2],0.001f,1,0.01f);
            imguiUnindent();
        }
    }
    return changed;
}
bool DrawTransparencyControls(EditorCore& editor,Techniques::Settings& settings)
{
    using namespace Techniques;
    const bool sceneAvailable=editor.Scene().Demo==DemoId::StanfordDragon && !editor.IsDemoLoading();
    const bool raster=!UsesPathTracing(settings.Pipeline);
    const char* reason=!sceneAvailable ? "The current scene has no configured refractive or translucent materials." :
        !raster ? "Ray Tracing Full integrates transmission; independent raster transparency is inactive." : nullptr;
    const auto beginGroup=[&](const char* name,bool& open,bool available,const char* unavailable)
    {
        available=available && imguiIsEnabled();
        imguiSetDisabledReason(unavailable);
        if (imguiCollapse(name,"",open && available,available)) open=!open;
        imguiPushVisibility(open && available);imguiIndent();
        imguiPushEnabled(available,unavailable);
        imguiSetDisabledReason(nullptr);
    };
    const auto endGroup=[](){imguiPopEnabled();imguiUnindent();imguiPopVisibility();};
    bool changed=false;
    const auto toggle=[&](const char* label,bool& enabled)
    { if (imguiCheck(label,enabled)) { enabled=!enabled; changed=true; } };
    const auto refractionSupport=editor.QueryTechniqueSupport(Technique::Refraction);
    static bool refractionOpen=false,transparencyOpen=false;
    beginGroup("Refraction",refractionOpen,sceneAvailable && raster && refractionSupport.Available(),
        reason?reason:refractionSupport.Reason);
    auto& glass=settings.Refraction;
    toggle("Enable refraction",glass.Enabled);
    if (glass.Enabled)
    {
        changed|=imguiSlider("Index of refraction",&glass.MaterialIOR,1,2.5f,0.01f);
        changed|=imguiSlider("Thickness",&glass.Thickness,0,1,0.01f);
        toggle("Material tint",glass.MaterialTint);
        changed|=imguiSlider("Absorption red",&glass.Absorption[0],0,4,0.01f);
        changed|=imguiSlider("Absorption green",&glass.Absorption[1],0,4,0.01f);
        changed|=imguiSlider("Absorption blue",&glass.Absorption[2],0,4,0.01f);
    }
    endGroup();
    const auto blendingSupport = editor.QueryTechniqueSupport(Technique::AlphaBlending);
    beginGroup("Alpha blending and OIT",transparencyOpen,sceneAvailable && raster && blendingSupport.Available(),
        reason ? reason : blendingSupport.Reason);
    toggle("Enable transparency",settings.AlphaBlending);
    if (settings.AlphaBlending)
    {
        const char* names[]{"Sorted alpha blending","Depth peeling OIT","Weighted blended OIT"};
        const TransparencyMode modes[]{TransparencyMode::Sorted,TransparencyMode::DepthPeeling,TransparencyMode::WeightedBlended};
        const auto peeling=editor.QueryTechniqueSupport(Technique::OIT);
        const auto weighted=editor.QueryTechniqueSupport(Technique::WeightedBlendedOIT);
        const char* reasons[]{nullptr,peeling.Available()?nullptr:peeling.Reason,
            !weighted.Available()?weighted.Reason:!(editor.QueryWeightedTransparencySampleCounts()&1)?"Weighted blended OIT requires single-sample rendering support.":nullptr};
        int selected=0;
        for (int i=0;i<3;++i) if (modes[i]==settings.Transparency.Mode) selected=i;
        if (imguiCombo("Algorithm",&selected,names,3,true,reasons))
        { settings.Transparency.Mode=modes[selected];changed=true; }
        if (UsesDepthPeeling(settings))
        {
            float layers=float(settings.OIT.LayerCount);
            if (imguiSlider("Layer count",&layers,MinimumPeelingLayers,MaximumPeelingLayers,1))
            { settings.OIT.LayerCount=unsigned(layers);changed=true; }
            toggle("Show layer overflow",settings.OIT.ShowOverflow);
        }
        if (UsesWeightedTransparency(settings))
        {
            changed|=imguiSlider("Opacity weight",&settings.Transparency.OpacityExponent,0,8,0.1f);
            changed|=imguiSlider("Depth weight",&settings.Transparency.DepthExponent,0,8,0.1f);
        }
    }
    endGroup();
    return changed;
}
bool DrawDragonReflectionControls(EditorCore& editor,Techniques::Settings& settings)
{
    using namespace Techniques;
    if (editor.Scene().Demo!=DemoId::StanfordDragon || editor.IsDemoLoading()) return false;
    if (!imguiIsEnabled() || UsesPathTracing(settings.Pipeline) || UsesHybridRendering(settings.Pipeline))
    {
        imguiSetDisabledReason("The selected ray pipeline integrates reflections; the independent planar mirror pass is inactive.");
        imguiCheck("4. Planar mirror",settings.Reflections.Planar.Enabled,false);
        imguiSetDisabledReason(nullptr);
        return false;
    }
    bool changed=false;
    const auto toggle=[&](const char* label,bool& enabled)
    { if (imguiCheck(label,enabled)) { enabled=!enabled; changed=true; } };
    if (editor.QueryTechniqueSupport(Technique::PlanarReflections).Available())
    {
        toggle("4. Planar mirror",settings.Reflections.Planar.Enabled);
        if (settings.Reflections.Planar.Enabled && !UsesEnvironmentCapture(settings.Reflections))
        {
            changed|=settings.Reflections.Mode!=ReflectionMode::None;
            settings.Reflections.Mode=ReflectionMode::None;
        }
    }
    return changed;
}
}
