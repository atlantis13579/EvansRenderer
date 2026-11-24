#pragma once
#include "Techniques/AreaLights.h"
#include "Techniques/Refraction.h"
#include "Techniques/RayTracedAO.h"
#include "Techniques/RayTracedShadows.h"
#include "Techniques/RayTracedReflections.h"
#include "Techniques/RayTracedGI.h"
#include "Techniques/VoxelGlobalIllumination.h"
#include "Techniques/PathTracing.h"
#include "Techniques/HybridRendering.h"
#include "Techniques/SubsurfaceScattering.h"
#include "Techniques/HairFur.h"
#include "Techniques/ParallaxOcclusionMapping.h"
#include "Techniques/DisplacementMapping.h"
#include "Techniques/Tessellation.h"
#include "Techniques/VirtualGeometry.h"
#include "Techniques/Nanite.h"
#include "Techniques/VirtualShadowMaps.h"
#include "Techniques/TemporalEffects.h"
#include "Techniques/TemporalSuperResolution.h"
#include "Techniques/OrderIndependentTransparency.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Evans::Techniques
{

// Values are part of the independently authored native-shader interface.
enum class ShadingModel : std::uint32_t { Phong = 0, Flat = 1, Gouraud = 2, PBR = 3, Toon = 4 };
enum class RenderingPipeline : std::uint32_t { Forward = 0, Deferred = 1, ForwardPlus = 2, ClusteredForward = 3, TiledDeferred = 4, PathTracing = 5, VisibilityBuffer = 6, Hybrid = 7, Neural3DGS = 8 };
[[nodiscard]] inline bool UsesHybridRendering(RenderingPipeline pipeline) noexcept
{ return pipeline == RenderingPipeline::Hybrid; }
[[nodiscard]] inline bool UsesVisibilityBuffer(RenderingPipeline pipeline) noexcept
{ return pipeline == RenderingPipeline::VisibilityBuffer; }
[[nodiscard]] inline bool UsesPathTracing(RenderingPipeline pipeline) noexcept
{ return pipeline == RenderingPipeline::PathTracing; }
[[nodiscard]] inline bool UsesLightLists(RenderingPipeline pipeline) noexcept
{ return pipeline == RenderingPipeline::ForwardPlus || pipeline == RenderingPipeline::ClusteredForward || pipeline == RenderingPipeline::TiledDeferred; }

[[nodiscard]] inline bool UsesDeferredLighting(RenderingPipeline pipeline) noexcept
{ return pipeline == RenderingPipeline::Deferred || pipeline == RenderingPipeline::TiledDeferred || UsesVisibilityBuffer(pipeline) || UsesHybridRendering(pipeline); }

enum class ShadowRepresentation : std::uint32_t { SingleMap = 0, Cascaded = 1, Virtual = 2, RayTraced = 3 };
// Representation chooses the depth maps; filtering chooses how receivers compare
// against those maps. VSM replaces comparisons with filtered linear moments;
// Hard, PCF, PCSS and VSM occupy one filter choice.
enum class ShadowFilterMode : std::uint32_t { Hard = 0, PCF = 1, PCSS = 2, VSM = 3 };
inline constexpr std::uint32_t MaximumShadowFilterRadius = 4;
inline constexpr std::uint32_t MaximumPCSSRadius = 128;
struct PCSSSettings
{
    float EmitterRadius = 0.05f; // world units for finite lights
    float AngularRadiusDegrees = 0.25f; // directional emitter half-angle
    std::uint32_t MaximumSearchRadius = 64, MaximumFilterRadius = 32; // texels
    std::uint32_t BlockerSamples = 32, FilterSamples = 64;
};
[[nodiscard]] bool operator==(const PCSSSettings& left, const PCSSSettings& right) noexcept;
[[nodiscard]] PCSSSettings SanitizePCSSSettings(PCSSSettings settings) noexcept;
inline constexpr std::uint32_t MaximumVSMBlurRadius = 32;
struct VSMSettings
{
    std::uint32_t BlurRadius = 3; // shadow texels
    float BlurSigma = 1.5f; // shadow texels
    float MinimumVariance = 0.00002f; // normalized linear depth squared
    float MaximumDerivativeVariance = 0.01f;
    float LightBleedingReduction = 0.2f;
};
[[nodiscard]] bool operator==(const VSMSettings& left, const VSMSettings& right) noexcept;
[[nodiscard]] VSMSettings SanitizeVSMSettings(VSMSettings settings) noexcept;
struct ShadowFilterSettings
{
    ShadowFilterMode Mode = ShadowFilterMode::PCF;
    std::uint32_t Radius = 1; // texel offsets; PCF averages (2 * radius + 1)^2 bilinear comparisons
    PCSSSettings PCSS;
    VSMSettings VSM;
};
[[nodiscard]] bool operator==(const ShadowFilterSettings& left, const ShadowFilterSettings& right) noexcept;
[[nodiscard]] ShadowFilterSettings SanitizeShadowFilterSettings(ShadowFilterSettings settings) noexcept;
inline constexpr std::uint32_t MaximumShadowCascades = 4;
struct CascadedShadowSettings
{
    std::uint32_t CascadeCount = MaximumShadowCascades;
    float SplitLambda = 0.65f;
    float MaximumDistance = 0.0f; // zero uses the finite camera far plane
    float BlendFraction = 0.1f;
};
[[nodiscard]] bool operator==(const CascadedShadowSettings& left, const CascadedShadowSettings& right) noexcept;
[[nodiscard]] CascadedShadowSettings SanitizeCascadedShadowSettings(CascadedShadowSettings settings) noexcept;

enum class AmbientOcclusionMode : std::uint32_t { None = 0, SSAO = 1, HBAO = 2, SAO = 3, GTAO = 4, RayTraced = 5 };
struct AmbientOcclusionSettings
{
    AmbientOcclusionMode Mode = AmbientOcclusionMode::None;
    float Radius = 0.25f;
    float Bias = 0.01f;
    float Intensity = 1.0f;
    float HorizonAngleBiasDegrees = 10.0f;
    RayTracedAOSettings RayTracing;
};
[[nodiscard]] bool operator==(const AmbientOcclusionSettings& left, const AmbientOcclusionSettings& right) noexcept;
[[nodiscard]] AmbientOcclusionSettings SanitizeAmbientOcclusionSettings(AmbientOcclusionSettings settings) noexcept;
enum class AmbientOcclusionPassKind : std::uint32_t
{
    Estimate = 0, BilateralHorizontal = 1, BilateralVertical = 2, Composite = 3,
    DepthPrepare = 4, DepthReduce = 5
};
[[nodiscard]] inline bool UsesAmbientOcclusion(const AmbientOcclusionSettings& settings) noexcept
{
    return settings.Mode == AmbientOcclusionMode::SSAO || settings.Mode == AmbientOcclusionMode::HBAO ||
        settings.Mode == AmbientOcclusionMode::SAO || settings.Mode == AmbientOcclusionMode::GTAO ||
        settings.Mode == AmbientOcclusionMode::RayTraced;
}
// Native mip dimensions floor-half each axis independently, retaining at least
// one texel. The full chain has a fixed cost for a given viewport, not AO radius.
[[nodiscard]] inline std::uint32_t AmbientOcclusionDepthMipCount(int width, int height) noexcept
{
    if (width <= 0 || height <= 0) return 0;
    std::uint32_t count = 1;
    while (width > 1 || height > 1)
    {
        width = width > 1 ? width / 2 : 1;
        height = height > 1 ? height / 2 : 1;
        ++count;
    }
    return count;
}
[[nodiscard]] std::vector<AmbientOcclusionPassKind> BuildAmbientOcclusionPlan(
    const AmbientOcclusionSettings& settings, int width, int height);

// One primary estimator, plus an optional finite planar receiver.
enum class ReflectionMode : std::uint32_t { None = 0, ScreenSpace = 1, Cubemap = 2, LatitudeLongitude = 3, RayTraced = 5, Lumen = 6 };
struct PlanarReflectionSettings
{
    bool Enabled = false;
    float Intensity = 1;
    std::array<float, 4> Plane {0, 1, 0, 0}; // World plane n dot x + d = 0.
    float ClipBias = 0.01f; // Move capture clipping into the retained half-space.
    float ReceiverTolerance = 0.02f; // World distance from the receiver plane.
    float MinimumNormalCosine = 0.95f;
    std::uint32_t ResolutionDivisor = 2; // Preserve view aspect at full, half or quarter extent.
    bool BoundedReceiver = false;
    std::array<float, 3> ReceiverMinimum {}, ReceiverMaximum {}; // Optional world-space aperture bounds.
};
[[nodiscard]] bool operator==(const PlanarReflectionSettings& left, const PlanarReflectionSettings& right) noexcept;
[[nodiscard]] PlanarReflectionSettings SanitizePlanarReflectionSettings(PlanarReflectionSettings settings) noexcept;
struct LumenTemporalSettings
{
    bool Enabled=true;
    float HistoryWeight=0.9f,DepthTolerance=0.01f,NormalThreshold=0.9f,PlaneTolerance=0.01f,ClipTolerance=0.25f;
    std::uint32_t MaximumFrames=32;
};
[[nodiscard]] LumenTemporalSettings SanitizeLumenTemporalSettings(LumenTemporalSettings settings) noexcept;
[[nodiscard]] bool operator==(const LumenTemporalSettings& a,const LumenTemporalSettings& b) noexcept;
struct LumenReflectionSettings
{
    std::uint32_t ResolutionDivisor=2,FilterRadius=1;
    bool ScreenTracing=true;
    float NormalCosine=0.9f,PlaneTolerance=0.02f,RelativePlaneTolerance=0.01f,RoughnessTolerance=0.1f;
    LumenTemporalSettings Temporal;
};
[[nodiscard]] LumenReflectionSettings SanitizeLumenReflectionSettings(LumenReflectionSettings settings) noexcept;
[[nodiscard]] bool operator==(const LumenReflectionSettings& a,const LumenReflectionSettings& b) noexcept;
// Single-scattering spherical atmosphere. Distances use world-space metres;
// spectral extinction coefficients are evaluated in kilometres.
struct AtmosphereSettings
{
    bool Enabled=false, AerialPerspective=true;
    float PlanetRadius=6360000.0f, Height=80000.0f, GroundHeight=0;
    float RayleighHeight=8000.0f, MieHeight=1200.0f;
    float RayleighStrength=1.0f, MieStrength=1.0f, MieAnisotropy=0.8f;
    std::array<float,3> Wavelengths{680.0f,550.0f,440.0f};
    float Intensity=1.0f;
    std::uint32_t ViewSamples=16, LightSamples=8;
    std::array<float,3> SunDirection{0.5f,0.8660254f,0};
    std::array<float,3> SunRadiance{16,16,16};
};
[[nodiscard]] AtmosphereSettings SanitizeAtmosphereSettings(AtmosphereSettings settings) noexcept;
[[nodiscard]] bool operator==(const AtmosphereSettings& a,const AtmosphereSettings& b) noexcept;
struct ReflectionSettings
{
    ReflectionMode Mode = ReflectionMode::None;
    bool Skybox=false;
    float SkyboxIntensity=1.0f;
    float DiffuseIBL=0.0f; // Cosine-weighted HDR environment irradiance.
    AtmosphereSettings Atmosphere;
    float Intensity = 1.0f;
    float EnvironmentIntensity = 0.8f; // HDR studio radiance scale, independent of scene estimators.
    float MaximumDistance = 25.0f; // world units
    float Thickness = 0.1f; // world-space depth slab behind a visible surface
    float NormalBias = 0.01f;
    float MaximumRoughness = 0.8f;
    float EdgeFadePixels = 8.0f;
    float StridePixels = 1.0f;
    std::uint32_t MaximumSteps = 128;
    std::uint32_t CaptureResolution = 128; // square faces, camera-centered scene capture
    PlanarReflectionSettings Planar;
    RayTracedReflectionSettings RayTracing;
    LumenReflectionSettings Lumen;
};
[[nodiscard]] bool operator==(const ReflectionSettings& left, const ReflectionSettings& right) noexcept;
[[nodiscard]] ReflectionSettings SanitizeReflectionSettings(ReflectionSettings settings) noexcept;
enum class ReflectionPassKind : std::uint32_t { Trace = 0, Composite = 1, EnvironmentLookup = 2, PlanarLookup = 3, BilateralHorizontal = 4, BilateralVertical = 5, LumenReconstruction = 6, LumenTemporal = 7 };
[[nodiscard]] inline bool UsesPlanarCapture(const ReflectionSettings& settings) noexcept
{
    return settings.Planar.Enabled;
}
[[nodiscard]] inline bool UsesEnvironmentCapture(const ReflectionSettings& settings) noexcept
{
    return settings.Mode == ReflectionMode::Cubemap || settings.Mode == ReflectionMode::LatitudeLongitude;
}
[[nodiscard]] inline bool UsesEnvironmentTexture(const ReflectionSettings& settings) noexcept
{ return settings.Skybox || settings.DiffuseIBL>0 || settings.Atmosphere.Enabled || UsesEnvironmentCapture(settings); }
[[nodiscard]] inline bool UsesReflections(const ReflectionSettings& settings) noexcept
{
    return settings.Skybox || settings.DiffuseIBL>0 || settings.Atmosphere.Enabled || settings.Mode == ReflectionMode::ScreenSpace || settings.Mode == ReflectionMode::RayTraced || settings.Mode == ReflectionMode::Lumen || UsesEnvironmentCapture(settings) || UsesPlanarCapture(settings);
}

[[nodiscard]] inline bool UsesLumenReflections(const ReflectionSettings& settings) noexcept
{ return settings.Mode==ReflectionMode::Lumen; }
[[nodiscard]] inline bool UsesTemporalReflections(const ReflectionSettings& settings) noexcept
{ return UsesLumenReflections(settings) && settings.Lumen.Temporal.Enabled; }

enum class AntiAliasingMode : std::uint32_t { None = 0, FXAA = 1, TAA = 2, TSR = 3 };
struct TemporalAASettings
{
    float HistoryWeight = 0.9f;
    float DepthTolerance = 0.01f; // relative optical depth, in the previous camera
};
[[nodiscard]] bool operator==(const TemporalAASettings& left, const TemporalAASettings& right) noexcept;
[[nodiscard]] TemporalAASettings SanitizeTemporalAASettings(TemporalAASettings settings) noexcept;
// Counts are also mask bits because every supported count is a power of two.
inline constexpr std::uint32_t AllMultisampleCounts = 1u | 2u | 4u | 8u;
[[nodiscard]] std::uint32_t SanitizeSampleCount(std::uint32_t requested) noexcept;
[[nodiscard]] std::uint32_t SelectSampleCount(std::uint32_t requested, std::uint32_t supported) noexcept;
enum class MultisamplePassKind : std::uint32_t { OpaqueResolve, VelocityResolve, ColorRestore, ColorResolve };

enum class ToneMappingMode : std::uint32_t { None = 0, Reinhard = 1, ACESFitted = 2 };

struct HdrSettings
{
    ToneMappingMode ToneMapping = ToneMappingMode::None;
    bool Bloom = false;
    bool ColorGrading = false;
    bool LensFlare = false;
    float FlareThreshold = 0.5f;
    float FlareIntensity = 0.25f;
    float FlareGhostSpacing = 1.0f;
    float FlareHaloRadius = 0.3f;
    float ExposureEV = 0.0f;
    float BloomThreshold = 0.3f;
    float BloomIntensity = 0.25f;
    // A mild visible grade when enabled; both controls at one remain neutral.
    float Contrast = 1.1f;
    float Saturation = 1.05f;
};

[[nodiscard]] bool operator==(const HdrSettings& left, const HdrSettings& right) noexcept;
[[nodiscard]] bool operator!=(const HdrSettings& left, const HdrSettings& right) noexcept;
[[nodiscard]] HdrSettings SanitizeHdrSettings(HdrSettings settings) noexcept;
[[nodiscard]] bool RequiresHDR(const HdrSettings& settings) noexcept;
// Call with sanitized settings. An inactive flare has no HDR dependency.
[[nodiscard]] inline bool UsesLensFlare(const HdrSettings& settings) noexcept
{
    return settings.LensFlare && settings.FlareIntensity > 0.0f;
}
inline constexpr float MaximumFlareThreshold = 64.0f;
inline constexpr float MaximumFlareIntensity = 4.0f;
inline constexpr float MinimumFlareGhostSpacing = 0.25f;
inline constexpr float MaximumFlareGhostSpacing = 2.0f;
inline constexpr float MaximumFlareHaloRadius = 1.0f;
[[nodiscard]] ToneMappingMode EffectiveToneMapping(const HdrSettings& settings) noexcept;

enum class HdrPassKind : std::uint32_t
{
    BloomExtract = 0, BloomBlurHorizontal = 1, BloomBlurVertical = 2,
    BloomComposite = 3, ColorGrading = 4, ToneMapping = 5,
    FlareExtract = 6, FlareFeatures = 7, FlareBlurHorizontal = 8,
    FlareBlurVertical = 9, FlareComposite = 10
};
// RHI-owned HDR passes execute after scene drawing and before display post
// processing. An enabled chain always ends with one float-to-display resolve.
[[nodiscard]] std::vector<HdrPassKind> BuildHdrPlan(const HdrSettings& settings);

struct DepthOfFieldSettings
{
    bool Enabled = false;
    float FocusDistance = 3.0f; // optical-axis distance in world units
    float FNumber = 2.8f;
    float SensorWidthMillimeters = 36.0f;
    float MetersPerWorldUnit = 1.0f;
    float MaxRadiusPixels = 12.0f;
};
[[nodiscard]] bool operator==(const DepthOfFieldSettings& left, const DepthOfFieldSettings& right) noexcept;
[[nodiscard]] DepthOfFieldSettings SanitizeDepthOfFieldSettings(DepthOfFieldSettings settings) noexcept;
enum class DepthOfFieldPassKind : std::uint32_t { Gather = 0, Composite = 1 };
[[nodiscard]] std::vector<DepthOfFieldPassKind> BuildDepthOfFieldPlan(const DepthOfFieldSettings& settings);

struct MotionBlurSettings
{
    bool Enabled = false;
    float ShutterSeconds = 1.0f / 120.0f;
    float MaxRadiusPixels = 12.0f;
};
[[nodiscard]] bool operator==(const MotionBlurSettings& left, const MotionBlurSettings& right) noexcept;
[[nodiscard]] MotionBlurSettings SanitizeMotionBlurSettings(MotionBlurSettings settings) noexcept;
enum class MotionBlurPassKind : std::uint32_t { TileMax = 0, NeighborMax = 1, Gather = 2, Composite = 3 };
inline constexpr int MotionBlurTileSize = 16;
[[nodiscard]] std::vector<MotionBlurPassKind> BuildMotionBlurPlan(const MotionBlurSettings& settings);

// Image-space ink rendering is composable with every surface shading model.
// Sigma/spacing/width are pixels; hatch angle is clockwise in top-left image space.
struct NPRSettings
{
    bool Enabled = false;
    float Blend = 1.0f;
    float EdgeSigma = 1.0f;
    float EdgeWeight = 0.98f; // wide Gaussian gain in G(sigma) - tau * G(1.6 sigma)
    float EdgeThreshold = 0.0f;
    float EdgeSharpness = 40.0f;
    float EdgeStrength = 1.0f;
    float HatchStrength = 0.6f;
    float HatchSpacing = 8.0f;
    float HatchWidth = 1.0f;
    float HatchAngleDegrees = 45.0f;
    float ColorStrength = 0.25f; // white paper to posterized source color
    std::uint32_t ToneLevels = 4;
};
[[nodiscard]] bool operator==(const NPRSettings& left, const NPRSettings& right) noexcept;
[[nodiscard]] NPRSettings SanitizeNPRSettings(NPRSettings settings) noexcept;

struct PostProcessSettings
{
    AntiAliasingMode AntiAliasing = AntiAliasingMode::None;
    bool ScreenSpaceDistortion = false;
    float DistortionStrength = 0.2f;
    bool Vignette = false;
    bool FilmGrain = false;
    bool Pixelation = false;
    float PixelSize = 8.0f;
    bool ColorEdgeOutline = false;
    float ColorEdgeThreshold = 0.15f;
    float ColorEdgeStrength = 1.0f;
    bool DynamicDistortion = false;
    float DynamicDistortionAmplitude = 4.0f; // pixels
    float DynamicDistortionFrequency = 12.0f; // cycles across the viewport
    float DynamicDistortionSpeed = 1.0f; // cycles per second
    bool OilPaint = false;
    float OilPaintRadius = 2.0f;
    float OilPaintStrength = 1.0f;
    bool Watercolor = false;
    float WatercolorRadius = 2.0f;
    float WatercolorStrength = 1.0f;
    bool DepthNormalOutline = false;
    float DepthEdgeThreshold = 0.01f;
    float NormalEdgeThreshold = 0.2f;
    float DepthNormalEdgeStrength = 1.0f;
    float VignetteStrength = 0.35f;
    float FilmGrainStrength = 0.08f;
    // Static screen-space noise: the full seed is preserved, without animation.
    std::uint32_t FilmGrainSeed = 1;
    HdrSettings HDR;
    DepthOfFieldSettings DepthOfField;
    MotionBlurSettings MotionBlur;
    TemporalAASettings TemporalAA;
    TemporalSuperResolutionSettings TemporalSuperResolution;
    TemporalEffectsSettings TemporalEffects;
    NPRSettings NPR;
};
[[nodiscard]] inline bool UsesTemporalAA(const PostProcessSettings& settings) noexcept
{
    return settings.AntiAliasing == AntiAliasingMode::TAA;
}
[[nodiscard]] inline bool RequiresMotionVectors(const PostProcessSettings& settings) noexcept
{
    return settings.MotionBlur.Enabled || UsesTemporalAA(settings) || settings.AntiAliasing==AntiAliasingMode::TSR;
}
[[nodiscard]] inline bool UsesTemporalSuperResolution(const PostProcessSettings& settings) noexcept { return settings.AntiAliasing==AntiAliasingMode::TSR; }
[[nodiscard]] inline bool UsesTemporalReconstruction(const PostProcessSettings& settings) noexcept { return UsesTemporalAA(settings) || UsesTemporalSuperResolution(settings); }

[[nodiscard]] bool operator==(const PostProcessSettings& left, const PostProcessSettings& right) noexcept;
[[nodiscard]] bool operator!=(const PostProcessSettings& left, const PostProcessSettings& right) noexcept;
[[nodiscard]] PostProcessSettings SanitizePostProcessSettings(PostProcessSettings settings) noexcept;

// Radius is measured in half-diagonal units. The radial derivative is
// 1 + 3 * strength * radiusSquared; this lower bound prevents foldovers.
inline constexpr float MinimumDistortionStrength = -0.3f;
inline constexpr float MaximumDistortionStrength = 1.0f;
enum class PostProcessPassKind : std::uint32_t
{
    FXAA = 0, Vignette = 1, FilmGrain = 2, ScreenSpaceDistortion = 3, NPR = 4,
    Pixelation = 5, ColorEdgeOutline = 6, DynamicDistortion = 7,
    OilPaint = 8, Watercolor = 9, DepthNormalOutline = 10, Count
};
// Ordered native fullscreen passes. Each reads the previous color and writes a
// different attachment. DepthNormalOutline also reads scene depth and normals.
[[nodiscard]] std::vector<PostProcessPassKind> BuildPostProcessPlan(const PostProcessSettings& settings);

struct LodSettings
{
    bool Enabled = false;
    float MaximumErrorPixels = 2.0f;
    float HysteresisFraction = 0.15f;
};
inline constexpr float MinimumLodErrorPixels = 0.25f;
inline constexpr float MaximumLodErrorPixels = 256.0f;
inline constexpr float MaximumLodHysteresis = 0.5f;
[[nodiscard]] bool operator==(const LodSettings& left, const LodSettings& right) noexcept;
[[nodiscard]] LodSettings SanitizeLodSettings(LodSettings settings) noexcept;

enum class VolumeMediumMode : std::uint32_t { HeightFog = 0, Participating = 1, Clouds = 2 };
// Coefficients and emission are per world unit at unit density. A finite box
// bounds both camera and light paths. An authored density grid takes precedence
// over procedural modulation. Multiple scattering is not approximated.
struct ParticipatingMediaSettings
{
    std::array<float, 3> Absorption {0.01f, 0.025f, 0.05f};
    std::array<float, 3> Scattering {0.02f, 0.02f, 0.02f};
    std::array<float, 3> Emission {0.0f, 0.0f, 0.0f};
    std::array<float, 3> BoundsMinimum {-100.0f, -100.0f, -100.0f};
    std::array<float, 3> BoundsMaximum {100.0f, 100.0f, 100.0f};
    float Density = 1.0f;
    float ModulationStrength = 0.0f;
    float ModulationFrequency = 0.2f; // radians per world unit
    std::uint32_t LightSamples = 8;
};
[[nodiscard]] bool operator==(const ParticipatingMediaSettings& a, const ParticipatingMediaSettings& b) noexcept;
[[nodiscard]] ParticipatingMediaSettings SanitizeParticipatingMediaSettings(ParticipatingMediaSettings settings) noexcept;
// A procedural weather field controls coverage; a three-dimensional shape field
// and cellular erosion produce a finite cloud layer. Offset advects both fields
// in world units without a frame-time dependency. An authored grid bypasses
// the procedural density parameters while retaining the same transport model.
struct VolumetricCloudSettings
{
    bool Animate=false;
    float AnimationSpeed=1.0f;
    ParticipatingMediaSettings Medium;
    float Coverage = 0.65f;
    float WeatherVariation = 0.35f;
    float WeatherFrequency = 0.005f;
    float BaseFrequency = 0.04f;
    float DetailFrequency = 0.12f;
    float Erosion = 0.3f;
    float BaseFade = 0.1f;
    float TopFade = 0.7f;
    float EdgeFade = 0.1f;
    float DensityContrast = 1.0f;
    std::array<float, 3> Offset {};
    std::uint32_t Seed = 1;
    VolumetricCloudSettings()
    {
        Medium.Absorption = {0.002f, 0.002f, 0.002f};
        Medium.Scattering = {0.098f, 0.098f, 0.098f};
        Medium.BoundsMinimum = {-100.0f, 10.0f, -100.0f};
        Medium.BoundsMaximum = {100.0f, 80.0f, 100.0f};
    }
};
[[nodiscard]] bool operator==(const VolumetricCloudSettings& a, const VolumetricCloudSettings& b) noexcept;
[[nodiscard]] VolumetricCloudSettings SanitizeVolumetricCloudSettings(VolumetricCloudSettings settings) noexcept;
struct UnderwaterLightSettings
{
    float SurfaceHeight=0, BottomHeight=0, Radius=0, Extinction=0;
    std::array<float,3> ToSun{0,1,0}, Irradiance{};
    // Angular wave vector XZ, phase in radians, focusing strength.
    std::array<float,4> WaveA{}, WaveB{};
};
// One selected camera medium with prescribed ambient radiance. Height-mode
// coefficients use inverse world distance; bounded models own their materials.
struct VolumetricFogSettings
{
    bool Enabled = false;
    VolumeMediumMode MediumMode = VolumeMediumMode::HeightFog;
    ParticipatingMediaSettings Participating;
    VolumetricCloudSettings Clouds;
    UnderwaterLightSettings Underwater;
    // Optional atmosphere composed with cloud transport: none, exponential, height, linear.
    std::uint32_t AtmosphereMode=0;
    float Extinction = 0.01f;
    float BaseHeight = 0.0f;
    float HeightFalloff = 0.02f;
    float MaximumDistance = 100.0f;
    std::array<float, 3> Radiance {0.5f, 0.6f, 0.7f};
    std::uint32_t SliceCount = 32;
    std::uint32_t SamplesPerSlice = 2;
    std::uint32_t BoundedTilePixels = 4;
};
inline constexpr int FogTilePixels = 16;
[[nodiscard]] bool operator==(const VolumetricFogSettings& a, const VolumetricFogSettings& b) noexcept;
[[nodiscard]] VolumetricFogSettings SanitizeVolumetricFogSettings(VolumetricFogSettings settings) noexcept;
[[nodiscard]] inline bool UsesVolumetricFog(const VolumetricFogSettings& settings) noexcept
{
    return settings.Enabled && settings.MaximumDistance > 0.0f &&
        (settings.MediumMode == VolumeMediumMode::Clouds ? ((settings.Clouds.Medium.Density > 0.0f && settings.Clouds.Coverage > 0.0f) ||
            (settings.AtmosphereMode>0 && settings.Extinction>0) || settings.Underwater.Extinction>0) :
        settings.MediumMode == VolumeMediumMode::Participating ? settings.Participating.Density > 0.0f : settings.Extinction > 0.0f);
}

[[nodiscard]] inline bool UsesParticipatingMedia(const VolumetricFogSettings& settings) noexcept
{
    return UsesVolumetricFog(settings) && settings.MediumMode == VolumeMediumMode::Participating;
}
[[nodiscard]] inline bool UsesVolumetricClouds(const VolumetricFogSettings& settings) noexcept
{
    return UsesVolumetricFog(settings) && settings.MediumMode == VolumeMediumMode::Clouds;
}
[[nodiscard]] inline bool UsesBoundedMedium(const VolumetricFogSettings& settings) noexcept
{
    return UsesParticipatingMedia(settings) || UsesVolumetricClouds(settings);
}

enum class TransparencyMode : std::uint32_t { Sorted, WeightedBlended, DepthPeeling };
// Direct single scattering augments the height medium's prescribed ambient
// radiance. Positive anisotropy favors continuing along the incident light ray.
struct VolumetricLightingSettings
{
    bool Enabled = false;
    std::array<float, 3> ScatteringAlbedo {0.9f, 0.9f, 0.9f};
    float Anisotropy = 0.0f;
    float Intensity = 1.0f;
    float ShadowDepthBias = 0.001f;
};
[[nodiscard]] bool operator==(const VolumetricLightingSettings& a, const VolumetricLightingSettings& b) noexcept;
[[nodiscard]] VolumetricLightingSettings SanitizeVolumetricLightingSettings(VolumetricLightingSettings settings) noexcept;
[[nodiscard]] inline bool UsesVolumetricLighting(const VolumetricLightingSettings& settings) noexcept
{
    return settings.Enabled && settings.Intensity > 0.0f;
}

enum class WeightedTransparencyPassKind : std::uint32_t { Prepare, Resolve };
// Screen-space radial scattering from one projected scene light. Radius uses
// image-height units; decay is optical depth along the normalized screen ray.
struct LightShaftSettings
{
    bool Enabled = false;
    std::uint32_t LightIndex = 0;
    float Intensity = 0.5f;
    float SourceRadius = 0.25f;
    float Decay = 2.0f;
    float DepthBias = 0.0001f;
    std::uint32_t SampleCount = 64;
};
[[nodiscard]] bool operator==(const LightShaftSettings& a, const LightShaftSettings& b) noexcept;
[[nodiscard]] LightShaftSettings SanitizeLightShaftSettings(LightShaftSettings settings) noexcept;
[[nodiscard]] inline bool UsesLightShafts(const LightShaftSettings& settings) noexcept
{
    return settings.Enabled && settings.Intensity > 0.0f && settings.SourceRadius > 0.0f;
}
struct TransparencySettings
{
    TransparencyMode Mode = TransparencyMode::Sorted;
    float OpacityExponent = 1.0f;
    float DepthExponent = 3.0f;
};
[[nodiscard]] bool operator==(const TransparencySettings& a, const TransparencySettings& b) noexcept;
[[nodiscard]] TransparencySettings SanitizeTransparencySettings(TransparencySettings settings) noexcept;

struct ContactShadowSettings
{
    bool Enabled = false;
    float MaximumDistance = 0.25f;
    float StartBias = 0.01f;
    float Thickness = 0.05f;
    std::uint32_t MaximumSteps = 64;
};
[[nodiscard]] bool operator==(const ContactShadowSettings& a, const ContactShadowSettings& b) noexcept;
[[nodiscard]] ContactShadowSettings SanitizeContactShadowSettings(ContactShadowSettings settings) noexcept;

enum class GlobalIlluminationMode : std::uint32_t { None = 0, ScreenSpace = 1, BakedProbes = 2, Lightmaps = 3, RayTraced = 4, Voxel = 5, Lumen = 6 };
struct LumenScreenTraceSettings
{
    bool Enabled=false;
    std::uint32_t MaximumSteps=128;
    float Thickness=0.1f,PixelStride=1;
    bool Hierarchical=true;
};
struct LumenProbeVisibilitySettings
{
    bool Enabled=true;
    std::uint32_t MaximumSteps=64;
    float NormalBias=0.05f;
};
inline constexpr std::uint32_t LumenWorldRadianceMaximumRays=1024*1024;
inline constexpr std::uint32_t LumenWorldMaximumFilterRadius=2;
inline constexpr std::uint32_t MaximumLumenWorldDepthLayers=4;
struct LumenWorldRadianceSettings
{
    bool Enabled=false,Parallax=true;
    std::uint32_t GridSide=4,DirectionalSide=8,CascadeCount=1,ProbesPerUpdate=256,AngularFilterRadius=0,DepthLayers=1;
    float CellScale=0.08f,BlendCells=1; // Finest spacing / GI horizon; transition width in finer cells.
};
struct LumenGlobalDistanceSettings
{
    bool Enabled=false;
    std::uint32_t GridSide=16,CascadeCount=3,MaximumSteps=128,VoxelsPerUpdate=4096;
    float CellScale=0.01f,NearScale=0.05f; // Spacing and mesh-detail distance / GI horizon.
};
struct LumenRadiositySettings
{
    bool Enabled=true;
    std::uint32_t Bounces=2,RaysPerTexel=8;
};
struct LumenTranslucencySettings
{
    bool Enabled=true,VisibilitySupport=true;
    std::uint32_t TileSide=64,SliceCount=16,RaysPerProbe=32;
    float DepthExponent=2;
};
inline constexpr std::uint32_t MaximumLumenAdaptiveProbeLevels=2;
inline constexpr float MaximumLumenEnvironmentExponent=64,MaximumLumenEnvironmentRadiance=16;
struct LumenEnvironmentSettings
{
    bool Enabled=false;
    std::array<float,3> Radiance{1,1,1},Direction{0,1,0};
    float Exponent=1;
};
inline bool operator==(const LumenEnvironmentSettings& a,const LumenEnvironmentSettings& b) noexcept
{ return a.Enabled==b.Enabled && a.Radiance==b.Radiance && a.Direction==b.Direction && a.Exponent==b.Exponent; }
struct LumenSettings
{
    bool HardwareRayTracing=false; // Native triangle queries; unsupported devices retain distance-field tracing.
    bool HairTransport=true,HairIndirect=true;
    std::uint32_t HairGridSide=32;
    std::uint32_t TileSide=32,RaysPerProbe=64,SourceResolution=24,CardSide=16,LightingTilesPerUpdate=128,CaptureCardsPerUpdate=8;
    std::array<float,3> SkyRadiance{0,0,0};
    bool Shadows=true,RefineCards=true,AdaptiveProbes=true;
    std::uint32_t AdaptiveProbeLevels=1;
    LumenEnvironmentSettings Environment;
    std::uint32_t MaximumMeshCards=12;
    float CardCoverage=0.95f;
    LumenScreenTraceSettings ScreenTrace{true};
    LumenProbeVisibilitySettings ProbeVisibility;
    LumenTemporalSettings Temporal;
    LumenWorldRadianceSettings WorldCache;
    LumenGlobalDistanceSettings GlobalDistance;
    LumenRadiositySettings Radiosity;
    LumenTranslucencySettings Translucency;
};
[[nodiscard]] bool operator==(const LumenSettings& a,const LumenSettings& b) noexcept;
[[nodiscard]] LumenSettings SanitizeLumenSettings(LumenSettings settings) noexcept;
struct LightmapSettings
{
    std::uint32_t Resolution = 1024, Padding = 2, RayCount = 8;
    float MaximumNormalAngleDegrees = 45;
};
[[nodiscard]] bool operator==(const LightmapSettings& a, const LightmapSettings& b) noexcept;
[[nodiscard]] LightmapSettings SanitizeLightmapSettings(LightmapSettings settings) noexcept;
struct LightProbeSettings
{
    bool AutomaticBounds = true;
    std::array<float, 3> BoundsMinimum {-1, -1, -1};
    std::array<float, 3> BoundsMaximum {1, 1, 1};
    std::uint32_t GridAxisCount = 4;
    std::uint32_t RayCount = 128;
    float MaximumBackfaceFraction = 0.25f;
    // Biases use the smallest cell spacing; variance uses its square.
    float NormalBias = 0.05f;
    float VisibilityBias = 0.02f;
    float MinimumVariance = 0.0001f;
    float VisibilitySharpness = 3.0f;
    float MomentSharpness = 16.0f;
};
[[nodiscard]] bool operator==(const LightProbeSettings& a, const LightProbeSettings& b) noexcept;
[[nodiscard]] LightProbeSettings SanitizeLightProbeSettings(LightProbeSettings settings) noexcept;
// Diffuse GI estimators share intensity and world-space bounds. Scene-ray paths,
// screen-space history and baked inputs have independent method controls.
struct GlobalIlluminationSettings
{
    GlobalIlluminationMode Mode = GlobalIlluminationMode::None;
    float Intensity = 1.0f;
    float MaximumDistance = 25.0f;
    float Thickness = 0.1f;
    float NormalBias = 0.01f;
    float EdgeFadePixels = 8.0f;
    float HistoryWeight = 0.9f;
    float DepthTolerance = 0.01f;
    float NormalThreshold = 0.85f;
    std::uint32_t RayCount = 4;
    std::uint32_t MaximumSteps = 64;
    LightProbeSettings Probes;
    LightmapSettings Lightmaps;
    RayTracedGISettings RayTracing;
    VoxelGISettings Voxels;
    LumenSettings Lumen;
};
[[nodiscard]] bool operator==(const GlobalIlluminationSettings& a, const GlobalIlluminationSettings& b) noexcept;
[[nodiscard]] GlobalIlluminationSettings SanitizeGlobalIlluminationSettings(GlobalIlluminationSettings settings) noexcept;
[[nodiscard]] inline bool UsesGlobalIllumination(const GlobalIlluminationSettings& settings) noexcept
{ return (settings.Mode == GlobalIlluminationMode::ScreenSpace || settings.Mode == GlobalIlluminationMode::BakedProbes || settings.Mode == GlobalIlluminationMode::Lightmaps || settings.Mode == GlobalIlluminationMode::RayTraced || settings.Mode == GlobalIlluminationMode::Voxel || settings.Mode == GlobalIlluminationMode::Lumen) && settings.Intensity > 0.0f; }
[[nodiscard]] inline bool UsesLumen(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode==GlobalIlluminationMode::Lumen && settings.Intensity>0; }
[[nodiscard]] inline bool UsesVoxelGI(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode == GlobalIlluminationMode::Voxel && settings.Intensity > 0.0f; }
[[nodiscard]] inline bool UsesRayTracedGI(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode == GlobalIlluminationMode::RayTraced && settings.Intensity > 0.0f; }
[[nodiscard]] inline bool UsesScreenSpaceGI(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode == GlobalIlluminationMode::ScreenSpace && settings.Intensity > 0.0f; }
[[nodiscard]] inline bool UsesTemporalGI(const GlobalIlluminationSettings& settings) noexcept
{ return UsesScreenSpaceGI(settings) || (UsesLumen(settings) && settings.Lumen.Temporal.Enabled); }
[[nodiscard]] inline bool UsesBakedProbes(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode == GlobalIlluminationMode::BakedProbes && settings.Intensity > 0.0f; }
[[nodiscard]] inline bool UsesLightmaps(const GlobalIlluminationSettings& settings) noexcept
{ return settings.Mode == GlobalIlluminationMode::Lightmaps && settings.Intensity > 0.0f; }
enum class GlobalIlluminationPassKind : std::uint32_t { Trace, Spatial, Temporal, Composite };
enum class MeshShaderDispatchMode : std::uint32_t { Automatic, Direct, Task };

struct Settings
{
    RenderingPipeline Pipeline = RenderingPipeline::Forward;
    PathTracingSettings PathTracing;
    HybridRenderingSettings Hybrid;
    SubsurfaceScatteringSettings Subsurface;
    HairFurSettings Hair;
    ParallaxOcclusionSettings Parallax;
    DisplacementMappingSettings Displacement;
    TessellationSettings Tessellation;
    VirtualGeometrySettings VirtualGeometry;
    NaniteSettings Nanite;
    bool MeshShaders=false;
    MeshShaderDispatchMode MeshDispatch=MeshShaderDispatchMode::Automatic;
    ShadingModel Shading = ShadingModel::Phong;
    ShadowRepresentation Shadows = ShadowRepresentation::SingleMap;
    ShadowFilterSettings ShadowFilter;
    CascadedShadowSettings Cascades;
    RayTracedShadowSettings RayShadows;
    VirtualShadowSettings VirtualShadows;
    ContactShadowSettings ContactShadows;
    std::uint32_t SampleCount = 1;
    bool TextureMapping = true;
    bool AnisotropicFiltering = true;
    bool NormalMapping = true;
    bool AlphaTesting = true;
    bool AlphaBlending = true;
    TransparencySettings Transparency;
    RefractionSettings Refraction;
    OrderIndependentTransparencySettings OIT;
    bool FrustumCulling = false;
    bool BackFaceCulling = false;
    LodSettings Lod;
    bool Instancing = false;
    bool GPUDriven = false;
    float Roughness = 0.5f;
    float Metallic = 0.0f;
    AreaLightSettings AreaLights;
    float ToonBands = 4.0f;
    AmbientOcclusionSettings AmbientOcclusion;
    ReflectionSettings Reflections;
    GlobalIlluminationSettings GlobalIllumination;
    VolumetricFogSettings VolumetricFog;
    VolumetricLightingSettings VolumetricLighting;
    LightShaftSettings LightShafts;
    PostProcessSettings PostProcessing;
};

[[nodiscard]] inline bool RequiresMotionVectors(const Settings& settings) noexcept
{ return RequiresMotionVectors(settings.PostProcessing) || UsesTemporalGI(settings.GlobalIllumination) || UsesTemporalReflections(settings.Reflections); }
[[nodiscard]] inline bool UsesLumenScene(const Settings& settings) noexcept
{ return UsesLumen(settings.GlobalIllumination) || UsesLumenReflections(settings.Reflections); }

[[nodiscard]] inline bool UsesDepthPeeling(const Settings& settings) noexcept
{ return settings.AlphaBlending && settings.Transparency.Mode == TransparencyMode::DepthPeeling; }

[[nodiscard]] inline bool UsesRefraction(const Settings& settings) noexcept
{ return settings.Refraction.Enabled; }

[[nodiscard]] inline bool UsesWeightedTransparency(const Settings& settings) noexcept
{
    return settings.AlphaBlending && settings.Transparency.Mode == TransparencyMode::WeightedBlended;
}

[[nodiscard]] bool operator==(const Settings& left, const Settings& right) noexcept;
[[nodiscard]] bool operator!=(const Settings& left, const Settings& right) noexcept;
[[nodiscard]] Settings SanitizeSettings(Settings settings) noexcept;

enum class Backend : std::uint8_t
{
    DX11, DX12, Metal, OpenGLES, Vulkan, DXR, CPU, CUDA, Compute, MetalCompute, Unknown
};
[[nodiscard]] Backend BackendFromId(std::string_view id) noexcept;
[[nodiscard]] bool IsNativeRaster(Backend backend) noexcept;
[[nodiscard]] bool IsSoftwareRaster(Backend backend) noexcept;

enum class Group : std::uint8_t
{
    Pipeline, Shading, Materials, Shadows, ShadowFilter, AmbientOcclusion,
    Reflections, Transparency, Geometry, AntiAliasing, GlobalIllumination,
    Volumetrics, PostProcessing, SpecializedShading
};

enum class Technique : std::uint8_t
{
    Forward, Deferred, Flat, Gouraud, Phong, PBR, Toon,
    TextureMapping, NormalMapping, AlphaTesting, AlphaBlending, BasicMaterials,
    ShadowMapping, CascadedShadowMaps, BasicSoftShadows, SSAO,
    EnvironmentReflections, CubemapReflections, PlanarReflections, MSAA, FXAA,
    ToneMapping, Bloom, ColorGrading, DepthOfField, MotionBlur,
    FrustumCulling, BackFaceCulling, LOD, Instancing,
    ForwardPlus, ClusteredForward, TiledDeferred, VisibilityBuffer, Hybrid,
    GPUDriven, PathTracing, SubsurfaceScattering, HairFur, NPR, AreaLights,
    BakedGI, Lightmaps, LightProbes, SSGI, VoxelGI, RayTracedGI, Lumen,
    PCF, PCSS, VSM, VirtualShadowMaps, ContactShadows, RayTracedShadows,
    HBAO, SAO, GTAO, RayTracedAO, SSR, RayTracedReflections, Refraction,
    OIT, WeightedBlendedOIT, Tessellation, DisplacementMapping,
    ParallaxOcclusionMapping, MeshShaders, VirtualGeometry, Nanite,
    TAA, TSR, DLSS, FSR, XeSS, VolumetricFog, VolumetricLighting,
    VolumetricClouds, ParticipatingMedia, LightShafts, ScreenSpaceDistortion,
    LensFlare, FilmGrain, Vignette, TemporalEffects, Count
};

enum class Availability : std::uint8_t
{
    Implemented, Planned, MissingResource, OutOfScope, UnsupportedBackend, Experimental, SceneLimit
};

// Resources describe data contracts, not API-specific allocation flags.
enum class Resource : std::uint64_t
{
    Geometry = 1ull << 0, Camera = 1ull << 1, Materials = 1ull << 2,
    Lights = 1ull << 3, SceneColor = 1ull << 4, SceneDepth = 1ull << 5,
    WorldNormals = 1ull << 6, ShadowDepth = 1ull << 7, GBuffer = 1ull << 8,
    HDRColor = 1ull << 9, Velocity = 1ull << 10, History = 1ull << 11,
    AccelerationStructure = 1ull << 12, LightLists = 1ull << 13,
    Environment = 1ull << 14, HeightMap = 1ull << 15, LightmapUV = 1ull << 16,
    BakedLighting = 1ull << 17, Probes = 1ull << 18, Voxels = 1ull << 19,
    HairStrands = 1ull << 20, Thickness = 1ull << 21, DensityVolume = 1ull << 22,
    MeshClusters = 1ull << 23, IndirectCommands = 1ull << 24,
    MultisampleColor = 1ull << 25, ShadowMoments = 1ull << 26,
    TransparencyStorage = 1ull << 27, NormalTexture = 1ull << 28,
    DiffuseTexture = 1ull << 29, OpacityTexture = 1ull << 30,
    AmbientVisibility = 1ull << 31, LinearDepthHierarchy = 1ull << 32,
    PreviousTransforms = 1ull << 33, MultisampleDepth = 1ull << 34,
    MultisampleGBuffer = 1ull << 35, MultisampleVelocity = 1ull << 36, ReflectionRadiance = 1ull << 37,
    PlanarRadiance = 1ull << 38, FogVolume = 1ull << 39,
    DisplacedGeometry = 1ull << 46,
    TessellatedGeometry = 1ull << 47,
    VirtualizedGeometry = 1ull << 48,
    OpaqueHDRSnapshot = 1ull << 49, ReconstructionHistory = 1ull << 50,
    PresentationHDR = 1ull << 51, PresentationDepth = 1ull << 52, PresentationVelocity = 1ull << 53,
    ContactVisibility = 1ull << 40, IndirectRadiance = 1ull << 41, GlobalIlluminationHistory = 1ull << 42,
    VisibilitySamples = 1ull << 43, DiffuseRadiance = 1ull << 44, ExposureHistory = 1ull << 45,
    SurfaceRadianceCache = 1ull << 54, ReflectionHistory = 1ull << 55, TranslucentIrradiance = 1ull << 56
};
using ResourceMask = std::uint64_t;
[[nodiscard]] constexpr ResourceMask Mask(Resource value) noexcept
{
    return static_cast<ResourceMask>(value);
}

struct Definition
{
    Technique Id;
    const char* Name;
    Group Category;
    Availability State;
    ResourceMask Requires;
    ResourceMask Produces;
    // Zero is composable. Equal nonzero slots are mutually exclusive choices.
    std::uint8_t ExclusiveSlot;
    const char* Explanation;
};

[[nodiscard]] const std::vector<Definition>& Catalog();
[[nodiscard]] const Definition& Describe(Technique technique);
[[nodiscard]] const char* GroupName(Group group) noexcept;
[[nodiscard]] const char* AvailabilityName(Availability availability) noexcept;

struct Support
{
    Availability State = Availability::Planned;
    const char* Reason = "Implementation pending";
    [[nodiscard]] bool Available() const noexcept { return State == Availability::Implemented || State == Availability::Experimental; }
};
[[nodiscard]] Support QuerySupport(Technique technique, Backend backend);

struct Resolution
{
    Settings Effective;
    std::vector<std::string> Diagnostics;
};
// Requested settings remain intact in the editor across backend switches.
[[nodiscard]] Resolution ResolveSettings(const Settings& requested, Backend backend);
// API implementation and device capability are distinct. Apply the device limit
// before resolving dependencies, preserving the user's requested shading/tone state.
[[nodiscard]] Resolution ResolveSettingsForDevice(const Settings& requested, Backend backend,
    bool deferredShadingSupported, bool scalableAmbientObscuranceSupported = true,
    std::uint32_t supportedSampleCounts = AllMultisampleCounts, bool instancingSupported = true,
    bool temporalAntiAliasingSupported = true, bool volumetricFogSupported = true,
    std::uint32_t weightedTransparencySampleCounts = AllMultisampleCounts, bool volumetricLightingSupported = true,
    bool lightShaftsSupported = true, bool varianceShadowMapsSupported = true, bool contactShadowsSupported = true,
    bool forwardPlusSupported = true, bool clusteredForwardSupported = true, bool tiledDeferredSupported = true,
    bool globalIlluminationSupported = true, bool bakedProbesSupported = true, bool lightmapsSupported = true, bool refractionSupported = true, bool depthPeelingSupported = true,
    bool rayTracedAOSupported = true, bool rayTracedReflectionsSupported = true, bool rayTracedGISupported = true, bool pathTracingSupported = true,
    bool gpuDrivenSupported = true, bool visibilityBufferSupported = true, bool hybridRenderingSupported = true, bool voxelGISupported = true, bool temporalEffectsSupported = true, bool parallaxSupported = true, bool displacementSupported = true, bool tessellationSupported = true, bool virtualGeometrySupported = true, bool temporalSuperResolutionSupported = true, bool virtualShadowsSupported = true, bool lumenSupported = true, bool naniteSupported = true, bool meshShadersSupported = true, bool naniteHardwareSupported = true, bool lumenHardwareSupported = false);

enum class PassKind : std::uint8_t
{
    ShadowDepth, OpaqueForward, TransparentForward, Outline, DeferredGeometry, DeferredLighting, VisibilityGeometry, HybridReflections,
    ContactDepth, ContactVisibility, ForwardPlusDepth, ForwardPlusAssignment, ClusteredAssignment, TiledDeferredAssignment,
    MaterialInputs, AmbientOcclusion, GlobalIllumination, DepthOfField, Velocity, TemporalAA, MotionBlur, VolumetricFog, VolumetricLighting, ParticipatingMedia, VolumetricClouds, LightShafts,
    PrepareWeightedTransparency, ResolveWeightedTransparency, Refraction, DepthPeeling,
    ResolveOpaqueSamples, ResolveVelocitySamples, RestoreColorSamples, ResolveColorSamples, Reflections, EnvironmentCapture, PlanarCapture, PathTracing, SubsurfaceScattering, HairFur, TemporalEffects, SnapshotTSROpaque, Skybox
};
struct PlannedPass
{
    PassKind Kind;
    ResourceMask Reads = 0;
    ResourceMask Writes = 0;
};
struct PassPlan
{
    std::vector<PlannedPass> Passes;
    // Present executes these native stages after color draws and before CPU
    // outline composition. The renderer does not submit duplicate draw phases.
    std::vector<HdrPassKind> HDRPasses;
    ResourceMask RhiResolveReads = 0;
    ResourceMask RhiResolveWrites = 0;
    ResourceMask ExternalInputs = 0;
    ResourceMask AllocatedResources = 0;
    std::vector<std::string> Diagnostics;
    [[nodiscard]] bool Valid() const noexcept { return Diagnostics.empty(); }
};
// Native drawing phases; transparency loads the existing color/depth attachments.
// This is not an API render-pass allocation abstraction.
[[nodiscard]] PassPlan BuildPassPlan(const Settings& settings, Backend backend,
    bool shadowsEnabled, bool outlineEnabled, bool wireframeOnly = false);

} // namespace Evans::Techniques
