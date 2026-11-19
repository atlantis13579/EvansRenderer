#include "Editor/TechniqueControls.h"
#include "Editor/Editor.hpp"
#include "Editor/BigWorld.hpp"
#include "Editor/Imgui.h"
#include <unordered_map>

namespace Evans::Editor
{
namespace
{
enum class WorldControlSection { Environment,Atmosphere,Water };
bool WorldRayFull=false,WorldSceneAvailable=false;
void BeginWorldGroup(const char* name,bool available=true,const char* reason=nullptr)
{
    const std::string group=name;
    const bool raster=group=="Planetary atmosphere" || group=="Fog" || group=="Volumetric clouds" || group=="Light scattering and shafts" || group=="Landscape LOD and shadows";
    available=available && WorldSceneAvailable && imguiIsEnabled() && !(WorldRayFull && raster);
    static std::unordered_map<std::string,bool> expanded;
    auto& open=expanded[name];
    imguiSetDisabledReason(reason?reason:!WorldSceneAvailable?"This technique requires the Big World scene.":"This technique is unavailable in the current rendering pipeline.");
    if (imguiCollapse(name,"",open && available,available)) open=!open;
    imguiPushVisibility(open && available); imguiIndent();
    imguiPushEnabled(available,reason?reason:"This technique is unavailable in the current rendering pipeline.");
    imguiSetDisabledReason(nullptr);
}
void EndWorldGroup() { imguiPopEnabled(); imguiUnindent(); imguiPopVisibility(); }
bool DrawBigWorldControls(EditorCore& editor,Techniques::Settings& settings,WorldControlSection section)
{
    WorldSceneAvailable=bool(editor.Scene().World) && !editor.IsDemoLoading();
    WorldRayFull=Techniques::UsesPathTracing(settings.Pipeline);
    auto world=editor.Scene().World?editor.Scene().World->Settings:BigWorldSettings{};
    bool changed=false,techniqueChanged=false;
    const auto toggle=[&](const char* name,bool& value)
    { if (imguiCheck(name,value)) { value=!value; changed=true; } };
    const auto slider=[&](const char* name,float& value,float minimum,float maximum,float step)
    { changed|=imguiSlider(name,&value,minimum,maximum,step); };
    const auto integer=[&](const char* name,unsigned& value,float minimum,float maximum,float step)
    { float number=float(value); if (imguiSlider(name,&number,minimum,maximum,step)) { value=unsigned(number); changed=true; } };
    if (section==WorldControlSection::Environment)
    {
        BeginWorldGroup("Landscape generation");
        integer("World seed",world.Seed,1,10000,1);
        slider("World extent",world.Extent,600,3000,50);
        slider("Mountain height",world.MountainHeight,30,450,5);
        integer("Terrain resolution",world.TerrainSegments,64,256,16);
        toggle("Landscape",world.Terrain);
        slider("Flight speed",world.CameraSpeed,1,200,1);
        EndWorldGroup();
        BeginWorldGroup("Day and night");
        toggle("Day-night cycle",world.DayNight);
        slider("Time of day",world.Hour,0,24,0.05f);
        slider("Day duration (seconds)",world.DayDuration,30,3600,10);
        slider("Sun radiance",world.SunIntensity,0,12,0.1f);
        slider("Moon radiance",world.MoonIntensity,0,1,0.01f);
        slider("Diffuse IBL",world.IBLIntensity,0,4,0.05f);
        EndWorldGroup();
        BeginWorldGroup("Grass, foliage and wind");
        toggle("Trees",world.Trees); integer("Tree count",world.TreeCount,0,1500,25);
        slider("Tree height",world.TreeHeight,4,35,0.5f);
        toggle("Grass",world.Grass); integer("Grass density budget",world.GrassCount,0,180000,1000);
        slider("Grass height",world.GrassHeight,0.2f,4,0.1f);
        slider("Blade width",world.GrassWidth,0.01f,0.15f,0.005f);
        slider("Grass draw distance",world.GrassDistance,40,220,5);
        slider("Grass LOD distance",world.GrassLodDistance,15,80,5);
        slider("Tree LOD distance",world.TreeLodDistance,50,300,10);
        toggle("Wind animation",world.AnimateVegetation);
        slider("Wind speed",world.WindSpeed,0,25,0.25f);
        slider("Wind direction",world.WindDirection,0,360,1);
        slider("Gust strength",world.GustStrength,0,2,0.05f);
        slider("Gust wavelength",world.GustWavelength,5,120,1);
        slider("Leaf flutter",world.LeafFlutter,0,0.7f,0.01f);
        EndWorldGroup();
        BeginWorldGroup("Landscape LOD and shadows");
        if (imguiCheck("Distance LOD",settings.Lod.Enabled)) { settings.Lod.Enabled=!settings.Lod.Enabled; techniqueChanged=true; }
        techniqueChanged|=imguiSlider("Terrain pixel error",&settings.Lod.MaximumErrorPixels,0.25f,8,0.25f);
        const char* shadowLabels[]{"Single map","Cascaded (large world)","Virtual shadow maps","Ray traced"};
        int shadow=int(settings.Shadows);
        if (imguiCombo("World shadows",&shadow,shadowLabels,4)) { settings.Shadows=Techniques::ShadowRepresentation(shadow); techniqueChanged=true; }
        techniqueChanged|=imguiSlider("Cascade distance",&settings.Cascades.MaximumDistance,100,1500,25);
        techniqueChanged|=imguiSlider("Cascade near detail",&settings.Cascades.SplitLambda,0,1,0.01f);
        techniqueChanged|=imguiSlider("Cascade blend",&settings.Cascades.BlendFraction,0,0.3f,0.01f);
        if (imguiCheck("Grass contact shadows",settings.ContactShadows.Enabled)) { settings.ContactShadows.Enabled=!settings.ContactShadows.Enabled; techniqueChanged=true; }
        techniqueChanged|=imguiSlider("Contact reach",&settings.ContactShadows.MaximumDistance,0.1f,3,0.1f);
        EndWorldGroup();
    }
    else if (section==WorldControlSection::Water)
    {
        BeginWorldGroup("Surface and animation");
        toggle("Water surface",world.Water); toggle("Animate water",world.AnimateWater);
        slider("Water time scale",world.WaterSpeed,0,4,0.05f);
        EndWorldGroup();
        const auto waterVolume=editor.QueryTechniqueSupport(Techniques::Technique::VolumetricClouds);
        imguiPushEnabled(world.Water && !WorldRayFull && waterVolume.Available(),!world.Water?
            "Enable the water surface first.":WorldRayFull?"Underwater volume integration requires a raster camera depth buffer.":waterVolume.Reason);
        toggle("Underwater volumetric lighting",world.UnderwaterLighting);
        imguiPopEnabled();
        BeginWorldGroup("FFT waves");
        toggle("FFT wind waves",world.FFT);
        slider("Wind speed (shared)",world.WindSpeed,0,25,0.25f);
        slider("Wind direction (shared)",world.WindDirection,0,360,1);
        const char* fftLabels[]{"32 x 32","64 x 64","128 x 128"}; int fft=world.FFTSize==32?0:world.FFTSize==64?1:2;
        if (imguiCombo("FFT resolution",&fft,fftLabels,3)) { world.FFTSize=32u<<fft; changed=true; }
        slider("FFT patch length",world.FFTPatch,32,512,8);
        slider("Wave amplitude",world.WaveAmplitude,0,3,0.05f);
        slider("Wave choppiness",world.Choppiness,0,2,0.05f);
        EndWorldGroup();
        BeginWorldGroup("Gerstner swell");
        toggle("Gerstner waves",world.Gerstner);
        slider("Swell amplitude",world.SwellAmplitude,0,3,0.05f);
        slider("Swell wavelength",world.SwellLength,5,100,1);
        slider("Swell steepness",world.SwellSteepness,0,0.9f,0.01f);
        EndWorldGroup();
        BeginWorldGroup("Water material");
        slider("Roughness",world.WaterRoughness,0.045f,0.6f,0.005f);
        slider("Index of refraction",world.WaterIOR,1.01f,2,0.005f);
        slider("Absorption",world.Absorption,0.005f,0.5f,0.005f);
        slider("Tint red",world.WaterTint[0],0,1,0.01f);
        slider("Tint green",world.WaterTint[1],0,1,0.01f);
        slider("Tint blue",world.WaterTint[2],0,1,0.01f);
        EndWorldGroup();
        BeginWorldGroup("Mesh and LOD");
        slider("Tile size",world.TileSize,24,128,4);
        slider("LOD distance",world.WaterLodDistance,30,500,10);
        const char* meshLabels[]{"16 x 16","32 x 32","64 x 64"}; int mesh=world.WaterSegments==16?0:world.WaterSegments==32?1:2;
        if (imguiCombo("Near water grid",&mesh,meshLabels,3)) { world.WaterSegments=16u<<mesh; changed=true; }
        EndWorldGroup();
        using namespace Techniques;
        const bool integrated=UsesPathTracing(settings.Pipeline) || UsesHybridRendering(settings.Pipeline);
        BeginWorldGroup("Water reflections",!integrated,"The selected ray pipeline integrates water reflections; independent reflection algorithms are inactive.");
        const char* reflectionLabels[]{"None","Sky cubemap","Sky latitude-longitude","Screen-space (SSR)","Ray-traced scene","Lumen surface cache","Planar scene"};
        const ReflectionMode reflectionModes[]{ReflectionMode::None,ReflectionMode::Cubemap,ReflectionMode::LatitudeLongitude,
            ReflectionMode::ScreenSpace,ReflectionMode::RayTraced,ReflectionMode::Lumen,ReflectionMode::None};
        const Technique reflectionTechniques[]{Technique::CubemapReflections,Technique::CubemapReflections,Technique::EnvironmentReflections,
            Technique::SSR,Technique::RayTracedReflections,Technique::Lumen,Technique::PlanarReflections};
        std::vector<std::string> reasons(7);std::vector<const char*> reasonPointers;
        int selected=0;
        for (unsigned i=0;i<7;++i)
        {
            const auto support=editor.QueryTechniqueSupport(reflectionTechniques[i]);
            if (i && !support.Available()) reasons[i]=support.Reason;
            if ((settings.Reflections.Planar.Enabled && i==6) ||
                (!settings.Reflections.Planar.Enabled && i<6 && settings.Reflections.Mode==reflectionModes[i])) selected=int(i);
        }
        for (const auto& reason:reasons) reasonPointers.push_back(reason.c_str());
        if (imguiCombo("Algorithm",&selected,reflectionLabels,7,true,reasonPointers.data()))
        {
            settings.Reflections.Planar.Enabled=selected==6;
            settings.Reflections.Mode=reflectionModes[selected];
            techniqueChanged=true;
        }
        if (settings.Reflections.Planar.Enabled)
            techniqueChanged|=imguiSlider("Reflection intensity",&settings.Reflections.Planar.Intensity,0,3,0.05f);
        else if (settings.Reflections.Mode==ReflectionMode::ScreenSpace || settings.Reflections.Mode==ReflectionMode::RayTraced || settings.Reflections.Mode==ReflectionMode::Lumen)
            techniqueChanged|=imguiSlider("Reflection intensity",&settings.Reflections.Intensity,0,4,0.05f);
        else if (UsesEnvironmentCapture(settings.Reflections))
            techniqueChanged|=imguiSlider("Reflection intensity",&settings.Reflections.EnvironmentIntensity,0,4,0.05f);
        imguiLabel("Additional trace settings: Scene reflections");
        EndWorldGroup();
        BeginWorldGroup("Caustics");
        toggle("Caustics enabled",world.Caustics);
        slider("Caustic intensity",world.CausticIntensity,0,5,0.05f);
        slider("Caustic depth",world.CausticDepth,1,50,1);
        EndWorldGroup();
    }
    else
    {
        BeginWorldGroup("Planetary atmosphere");
        auto& atmosphere=world.Atmosphere;
        toggle("Atmospheric scattering",atmosphere.Enabled);
        toggle("Aerial perspective",atmosphere.AerialPerspective);
        const auto kilometres=[&](const char* name,float& metres,float low,float high,float step)
        {
            float value=metres*0.001f;
            if (imguiSlider(name,&value,low,high,step)) { metres=value*1000; changed=true; }
        };
        kilometres("Planet radius (km)",atmosphere.PlanetRadius,100,10000,10);
        kilometres("Atmosphere thickness (km)",atmosphere.Height,5,150,1);
        kilometres("Rayleigh scale height (km)",atmosphere.RayleighHeight,1,atmosphere.Height*0.0005f,0.1f);
        kilometres("Mie scale height (km)",atmosphere.MieHeight,0.1f,atmosphere.Height*0.00025f,0.1f);
        slider("Rayleigh density",atmosphere.RayleighStrength,0,5,0.05f);
        slider("Mie aerosol density",atmosphere.MieStrength,0,10,0.05f);
        slider("Mie forward scattering",atmosphere.MieAnisotropy,0,0.95f,0.01f);
        slider("Red wavelength (nm)",atmosphere.Wavelengths[0],380,780,5);
        slider("Green wavelength (nm)",atmosphere.Wavelengths[1],380,780,5);
        slider("Blue wavelength (nm)",atmosphere.Wavelengths[2],380,780,5);
        slider("Atmosphere intensity",atmosphere.Intensity,0,4,0.05f);
        integer("View ray samples",atmosphere.ViewSamples,4,32,4);
        integer("Sun ray samples",atmosphere.LightSamples,4,32,4);
        EndWorldGroup();
        BeginWorldGroup("Fog");
        const char* algorithms[]{"None","Exponential","Height fog","Linear distance"}; int fog=int(world.Fog);
        if (imguiCombo("Fog algorithm",&fog,algorithms,4)) { world.Fog=WorldFog(fog); changed=true; }
        slider("Fog extinction",world.FogDensity,0,0.03f,0.0001f);
        slider("Fog base height",world.FogHeight,-20,200,1);
        slider("Fog height falloff",world.FogFalloff,0,0.1f,0.001f);
        slider("Fog distance",world.FogDistance,100,10000,50);
        EndWorldGroup();
        BeginWorldGroup("Volumetric clouds");
        toggle("Procedural clouds",world.Clouds);
        slider("Cloud altitude",world.CloudAltitude,80,900,5);
        slider("Cloud thickness",world.CloudThickness,40,400,5);
        slider("Cloud density",world.CloudDensity,0,0.3f,0.002f);
        slider("Cloud advection",world.CloudSpeed,0,5,0.05f);
        EndWorldGroup();
        BeginWorldGroup("Light scattering and shafts");
        if (imguiCheck("Volumetric light scattering",settings.VolumetricLighting.Enabled))
        { settings.VolumetricLighting.Enabled=!settings.VolumetricLighting.Enabled; if (settings.VolumetricLighting.Enabled) settings.LightShafts.Enabled=false; techniqueChanged=true; }
        techniqueChanged|=imguiSlider("Scattering anisotropy",&settings.VolumetricLighting.Anisotropy,-0.9f,0.9f,0.01f);
        techniqueChanged|=imguiSlider("Scattering intensity",&settings.VolumetricLighting.Intensity,0,4,0.05f);
        if (imguiCheck("Sun light shafts",settings.LightShafts.Enabled))
        { settings.LightShafts.Enabled=!settings.LightShafts.Enabled; if (settings.LightShafts.Enabled) settings.VolumetricLighting.Enabled=false; techniqueChanged=true; }
        techniqueChanged|=imguiSlider("Shaft intensity",&settings.LightShafts.Intensity,0,4,0.05f);
        techniqueChanged|=imguiSlider("Shaft decay",&settings.LightShafts.Decay,0,8,0.05f);
        techniqueChanged|=imguiSlider("Sun source radius",&settings.LightShafts.SourceRadius,0.005f,0.15f,0.005f);
        EndWorldGroup();
        BeginWorldGroup("Weather");
        const char* weather[]{"Clear","Rain","Snow"}; int weatherIndex=int(world.Weather);
        if (imguiCombo("Precipitation",&weatherIndex,weather,3)) { world.Weather=WorldWeather(weatherIndex); changed=true; }
        slider("Precipitation intensity",world.Precipitation,0,1,0.05f);
        slider("Snow coverage",world.SnowCover,0,1,0.05f);
        EndWorldGroup();
    }
    if (changed)
    {
        editor.SetBigWorldSettings(world);
        settings=editor.Settings().Techniques;
    }
    return techniqueChanged;
}
}
bool DrawWorldEnvironmentControls(EditorCore& editor,Techniques::Settings& settings)
{ return DrawBigWorldControls(editor,settings,WorldControlSection::Environment); }
bool DrawWaterControls(EditorCore& editor,Techniques::Settings& settings)
{ return DrawBigWorldControls(editor,settings,WorldControlSection::Water); }
bool DrawVolumeControls(EditorCore& editor,Techniques::Settings& settings)
{ return DrawBigWorldControls(editor,settings,WorldControlSection::Atmosphere); }
}
