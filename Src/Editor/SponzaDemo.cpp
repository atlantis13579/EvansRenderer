#include "Editor/Demos.hpp"

namespace Evans::Editor
{
bool DemoCatalog::SupportsSponzaLightmaps(const Techniques::Settings& settings) noexcept
{
    // The saved atlas includes the original meshes and their fixed LODs.
    // TODO: provide authored UV2/bakes for deformed and virtualized Sponza
    // topology; automatic chart expansion can exceed the atlas capacity.
    return !Techniques::UsesTessellation(settings.Tessellation) &&
        !Techniques::UsesDisplacementMapping(settings.Displacement) &&
        !settings.VirtualGeometry.Enabled && !settings.Nanite.Enabled;
}

void DemoCatalog::ConfigureSponzaLighting(SceneSnapshot& scene)
{
    // Coordinates are in the demo's normalized five-unit scene. Keep the
    // rectangles below the roof so their finite extent lights the colonnade.
    // They emit one-sided Lambertian radiance: flux = pi * area * radiance.
    // Native APIs independently integrate the same midpoint quadrature.
    // https://pbr-book.org/4ed/Light_Sources/Area_Lights
    scene.SceneCamera.Position = {-1.15f, -0.55f, 0};
    scene.SceneCamera.Target = {1.6f, -0.45f, 0};
    scene.SceneCamera.VerticalFieldOfViewDegrees = 72;
    // The Crytek sample lights one arcade from across the roof opening.
    // https://www.cryengine.com/asset-db/product/crytek/sponza-sample-scene
    constexpr float SunIntensity = 2, WarmRadiance = 3, CoolRadiance = 2;
    constexpr Float2 EmitterSize{0.8f, 0.45f};
    scene.Lights = {
        {LightType::Directional, {}, {0.55f, -1, 1.25f}, {1, 0.80f, 0.63f, 1}, SunIntensity, {}, true},
        {LightType::Area, {-0.65f, -0.1f, 0.3f}, {0.2f, -1, -0.3f},
            {1, 0.73f, 0.48f, 1}, WarmRadiance, EmitterSize, false, 0, false},
        {LightType::Area, {0.65f, -0.1f, -0.3f}, {-0.2f, -1, 0.3f},
            {0.5f, 0.73f, 1, 1}, CoolRadiance, EmitterSize, false, 0, false}};
    ++scene.LightingRevision;
}

void DemoCatalog::ApplySponzaAppearance(Techniques::Settings& settings)
{
    settings={};
    settings.Pipeline=Techniques::RenderingPipeline::Forward;
    settings.Reflections.Mode=Techniques::ReflectionMode::None;
    settings.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::Lightmaps;
    // The atrium is normalized to five world units. Resolve its narrow contact
    // gaps with a smaller bias and give local occlusion a readable contrast.
    settings.AmbientOcclusion.Radius=0.35f;
    settings.AmbientOcclusion.Bias=0.002f;
    settings.AmbientOcclusion.Intensity=2.5f;
    settings.GlobalIllumination.Lightmaps.Resolution=2048;
    settings.GlobalIllumination.Lightmaps.RayCount=32;
    settings.GlobalIllumination.Probes.GridAxisCount=6;
    settings.GlobalIllumination.Probes.RayCount=256;
    settings.AreaLights.EnergyUnit = Techniques::AreaLightEnergyUnit::Radiance;
    settings.AreaLights.SamplesPerAxis = Techniques::MinimumAreaLightSamplesPerAxis;
    settings.PostProcessing.HDR.ToneMapping = Techniques::ToneMappingMode::ACESFitted;
    settings.VolumetricFog.Enabled=false;
    settings.VolumetricLighting.Enabled=false;
    settings.LightShafts.Enabled=false;
    settings.Reflections.Skybox=true;
    settings.Reflections.SkyboxIntensity=0.105f;
    settings.Reflections.DiffuseIBL=0.60f;
    settings.Reflections.EnvironmentIntensity=0.55f;
    auto& hdr=settings.PostProcessing.HDR;
    hdr.ExposureEV=-0.3f;
    hdr.ColorGrading=true;
    hdr.Contrast=1.0f;
    hdr.Saturation=0.72f;
    settings = Techniques::SanitizeSettings(settings);
}
}
