#include "Techniques/TechniqueCatalog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Evans::Techniques
{
/*
Frame dependencies and selection rules
-------------------------------------
Captures and primary shadows precede opaque geometry and lighting. Screen-space
lighting consumes resolved depth/materials; velocity and private histories feed
temporal stages. Opaque HDR effects precede forward transparency, tone mapping,
display effects and selection outlines. BuildPassPlan records these handoffs;
each native RHI owns the resources and executes the constituent passes.

Pipelines, shading models, AO estimators, diffuse GI and primary reflection
estimators each occupy an exclusive slot. Shadow coverage (single, cascaded,
virtual or ray traced) is separate from map filtering (hard, PCF, PCSS or VSM).
Contact shadows supplement primary visibility. Transparency chooses sorted,
weighted blending or depth peeling. Independent effects compose where their
resource contracts and device capabilities permit.

ResolveSettings preserves requested alternatives and derives an effective
configuration. GPU-driven drawing requires instancing. Gouraud requires Forward;
visibility rendering is single-sampled. Nanite replaces ordinary LOD and virtual
geometry; native mesh stages can consume visibility geometry with or without
Nanite. Path tracing owns visibility, lighting and accumulation instead of the
raster effect chain. HDR and temporal producers are enabled by their consumers.
ResolveSettingsForDevice applies allocation/format/stage limits after API support.

All five raster APIs own independent technique shaders. CPU geometry preparation
and raster-state features need no dedicated shader. Algorithm details, named
parameters and references belong beside the relevant C++ or shader implementation.
Existing scenes provide geometry, captures, charts, probes and procedural inputs;
features needing additional authored assets retain explicit TODOs. Implemented
denotes this project's documented algorithm scope. Native Metal paths remain
unverified on Apple hardware.
*/
namespace
{
constexpr ResourceMask G = Mask(Resource::Geometry);
constexpr ResourceMask C = Mask(Resource::Camera);
constexpr ResourceMask M = Mask(Resource::Materials);
constexpr ResourceMask L = Mask(Resource::Lights);
constexpr ResourceMask Color = Mask(Resource::SceneColor);
constexpr ResourceMask Depth = Mask(Resource::SceneDepth);
constexpr ResourceMask N = Mask(Resource::WorldNormals);
constexpr ResourceMask Shadow = Mask(Resource::ShadowDepth);
constexpr ResourceMask HDR = Mask(Resource::HDRColor);
constexpr ResourceMask Velocity = Mask(Resource::Velocity);
constexpr ResourceMask History = Mask(Resource::History);
constexpr ResourceMask AS = Mask(Resource::AccelerationStructure);
constexpr ResourceMask Screen = Depth | N;
constexpr ResourceMask Temporal = Depth | Velocity | History;
constexpr ResourceMask Surface = G | C | M | L;
constexpr std::uint8_t Composable = 0;
constexpr std::uint8_t PipelineSlot = 1;
constexpr std::uint8_t ShadingSlot = 2;
constexpr std::uint8_t ShadowSlot = 3;
constexpr std::uint8_t ShadowFilterSlot = 4;
constexpr std::uint8_t AOSlot = 5;
constexpr std::uint8_t ReflectionSlot = 6;
constexpr std::uint8_t TransparencySlot = 7;
constexpr std::uint8_t AASlot = 8;
constexpr std::uint8_t GISlot = 9;
constexpr std::uint8_t GeometrySlot = 10;
constexpr std::uint8_t ScatteringSlot = 11;
constexpr std::uint8_t MediumSlot = 12;
constexpr float MinimumRoughness = 0.045f;
constexpr float MinimumToonBands = 2.0f;
constexpr float MaximumToonBands = 12.0f;
constexpr float MaximumFilmGrainStrength = 0.25f;
constexpr float MinimumExposureEV = -10.0f;
constexpr float MaximumExposureEV = 10.0f;
constexpr float MaximumBloomThreshold = 64.0f;
constexpr float MaximumBloomIntensity = 4.0f;
constexpr float MaximumGradeAdjustment = 2.0f;

float FiniteOr(float value, float fallback) noexcept
{
    return std::isfinite(value) ? value : fallback;
}

} // namespace

bool operator==(const HdrSettings& a, const HdrSettings& b) noexcept
{
    return a.ToneMapping == b.ToneMapping && a.Bloom == b.Bloom &&
        a.LensFlare == b.LensFlare && a.FlareThreshold == b.FlareThreshold &&
        a.FlareIntensity == b.FlareIntensity && a.FlareGhostSpacing == b.FlareGhostSpacing &&
        a.FlareHaloRadius == b.FlareHaloRadius &&
        a.ColorGrading == b.ColorGrading && a.ExposureEV == b.ExposureEV &&
        a.BloomThreshold == b.BloomThreshold && a.BloomIntensity == b.BloomIntensity &&
        a.Contrast == b.Contrast && a.Saturation == b.Saturation;
}

bool operator!=(const HdrSettings& a, const HdrSettings& b) noexcept { return !(a == b); }

HdrSettings SanitizeHdrSettings(HdrSettings settings) noexcept
{
    const HdrSettings defaults;
    if (static_cast<std::uint32_t>(settings.ToneMapping) >
        static_cast<std::uint32_t>(ToneMappingMode::ACESFitted)) settings.ToneMapping = ToneMappingMode::None;
    settings.ExposureEV = std::clamp(FiniteOr(settings.ExposureEV, defaults.ExposureEV),
        MinimumExposureEV, MaximumExposureEV);
    settings.BloomThreshold = std::clamp(FiniteOr(settings.BloomThreshold, defaults.BloomThreshold),
        0.0f, MaximumBloomThreshold);
    settings.BloomIntensity = std::clamp(FiniteOr(settings.BloomIntensity, defaults.BloomIntensity),
        0.0f, MaximumBloomIntensity);
    settings.FlareThreshold = std::clamp(FiniteOr(settings.FlareThreshold, defaults.FlareThreshold), 0.0f, MaximumFlareThreshold);
    settings.FlareIntensity = std::clamp(FiniteOr(settings.FlareIntensity, defaults.FlareIntensity), 0.0f, MaximumFlareIntensity);
    settings.FlareGhostSpacing = std::clamp(FiniteOr(settings.FlareGhostSpacing, defaults.FlareGhostSpacing),
        MinimumFlareGhostSpacing, MaximumFlareGhostSpacing);
    settings.FlareHaloRadius = std::clamp(FiniteOr(settings.FlareHaloRadius, defaults.FlareHaloRadius), 0.0f, MaximumFlareHaloRadius);
    settings.Contrast = std::clamp(FiniteOr(settings.Contrast, defaults.Contrast),
        0.0f, MaximumGradeAdjustment);
    settings.Saturation = std::clamp(FiniteOr(settings.Saturation, defaults.Saturation),
        0.0f, MaximumGradeAdjustment);
    return settings;
}

bool RequiresHDR(const HdrSettings& requested) noexcept
{
    const HdrSettings settings = SanitizeHdrSettings(requested);
    return settings.ToneMapping != ToneMappingMode::None || settings.Bloom || settings.ColorGrading || UsesLensFlare(settings);
}

ToneMappingMode EffectiveToneMapping(const HdrSettings& requested) noexcept
{
    const HdrSettings settings = SanitizeHdrSettings(requested);
    if (settings.ToneMapping != ToneMappingMode::None) return settings.ToneMapping;
    return settings.Bloom || settings.ColorGrading || UsesLensFlare(settings) ? ToneMappingMode::Reinhard : ToneMappingMode::None;
}

std::vector<HdrPassKind> BuildHdrPlan(const HdrSettings& requested)
{
    const HdrSettings settings = SanitizeHdrSettings(requested);
    std::vector<HdrPassKind> passes;
    if (!RequiresHDR(settings)) return passes;
    // Screen-space flare derives visibility from the resolved scene, before
    // bloom spreads highlights. Its temporary images can then be reused by bloom.
    if (UsesLensFlare(settings))
    {
        passes.push_back(HdrPassKind::FlareExtract);
        passes.push_back(HdrPassKind::FlareFeatures);
        passes.push_back(HdrPassKind::FlareBlurHorizontal);
        passes.push_back(HdrPassKind::FlareBlurVertical);
        passes.push_back(HdrPassKind::FlareComposite);
    }
    if (settings.Bloom)
    {
        passes.push_back(HdrPassKind::BloomExtract);
        passes.push_back(HdrPassKind::BloomBlurHorizontal);
        passes.push_back(HdrPassKind::BloomBlurVertical);
        passes.push_back(HdrPassKind::BloomComposite);
    }
    if (settings.ColorGrading) passes.push_back(HdrPassKind::ColorGrading);
    passes.push_back(HdrPassKind::ToneMapping);
    return passes;
}

bool operator==(const NPRSettings& a, const NPRSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.Blend == b.Blend && a.EdgeSigma == b.EdgeSigma &&
        a.EdgeWeight == b.EdgeWeight && a.EdgeThreshold == b.EdgeThreshold && a.EdgeSharpness == b.EdgeSharpness &&
        a.EdgeStrength == b.EdgeStrength && a.HatchStrength == b.HatchStrength && a.HatchSpacing == b.HatchSpacing &&
        a.HatchWidth == b.HatchWidth && a.HatchAngleDegrees == b.HatchAngleDegrees &&
        a.ColorStrength == b.ColorStrength && a.ToneLevels == b.ToneLevels;
}
NPRSettings SanitizeNPRSettings(NPRSettings settings) noexcept
{
    const NPRSettings defaults;
    const auto bounded = [](float value, float fallback, float low, float high)
    { return std::clamp(FiniteOr(value, fallback), low, high); };
    settings.Blend = bounded(settings.Blend, defaults.Blend, 0.0f, 1.0f);
    settings.EdgeSigma = bounded(settings.EdgeSigma, defaults.EdgeSigma, 0.5f, 1.5f);
    settings.EdgeWeight = bounded(settings.EdgeWeight, defaults.EdgeWeight, 0.8f, 1.0f);
    settings.EdgeThreshold = bounded(settings.EdgeThreshold, defaults.EdgeThreshold, -0.2f, 0.2f);
    settings.EdgeSharpness = bounded(settings.EdgeSharpness, defaults.EdgeSharpness, 1.0f, 100.0f);
    settings.EdgeStrength = bounded(settings.EdgeStrength, defaults.EdgeStrength, 0.0f, 1.0f);
    settings.HatchStrength = bounded(settings.HatchStrength, defaults.HatchStrength, 0.0f, 1.0f);
    settings.HatchSpacing = bounded(settings.HatchSpacing, defaults.HatchSpacing, 4.0f, 32.0f);
    settings.HatchWidth = bounded(settings.HatchWidth, defaults.HatchWidth, 0.25f, std::min(4.0f, settings.HatchSpacing * 0.5f));
    settings.HatchAngleDegrees = bounded(settings.HatchAngleDegrees, defaults.HatchAngleDegrees, -180.0f, 180.0f);
    settings.ColorStrength = bounded(settings.ColorStrength, defaults.ColorStrength, 0.0f, 1.0f);
    settings.ToneLevels = std::clamp(settings.ToneLevels, 2u, 16u);
    return settings;
}

bool operator==(const PostProcessSettings& a, const PostProcessSettings& b) noexcept
{
    return a.AntiAliasing == b.AntiAliasing && a.Vignette == b.Vignette &&
        a.ScreenSpaceDistortion == b.ScreenSpaceDistortion && a.DistortionStrength == b.DistortionStrength &&
        a.FilmGrain == b.FilmGrain && a.VignetteStrength == b.VignetteStrength &&
        a.FilmGrainStrength == b.FilmGrainStrength && a.FilmGrainSeed == b.FilmGrainSeed && a.HDR == b.HDR &&
        a.DepthOfField == b.DepthOfField && a.MotionBlur == b.MotionBlur && a.TemporalAA == b.TemporalAA && a.TemporalSuperResolution == b.TemporalSuperResolution && a.TemporalEffects == b.TemporalEffects && a.NPR == b.NPR;
}

bool operator!=(const PostProcessSettings& a, const PostProcessSettings& b) noexcept
{
    return !(a == b);
}

PostProcessSettings SanitizePostProcessSettings(PostProcessSettings settings) noexcept
{
    const PostProcessSettings defaults;
    if (static_cast<std::uint32_t>(settings.AntiAliasing) >
        static_cast<std::uint32_t>(AntiAliasingMode::TSR)) settings.AntiAliasing = AntiAliasingMode::None;
    settings.VignetteStrength = std::clamp(
        FiniteOr(settings.VignetteStrength, defaults.VignetteStrength), 0.0f, 1.0f);
    settings.FilmGrainStrength = std::clamp(
        FiniteOr(settings.FilmGrainStrength, defaults.FilmGrainStrength), 0.0f, MaximumFilmGrainStrength);
    settings.DistortionStrength = std::clamp(
        FiniteOr(settings.DistortionStrength, defaults.DistortionStrength),
        MinimumDistortionStrength, MaximumDistortionStrength);
    settings.HDR = SanitizeHdrSettings(settings.HDR);
    settings.DepthOfField = SanitizeDepthOfFieldSettings(settings.DepthOfField);
    settings.MotionBlur = SanitizeMotionBlurSettings(settings.MotionBlur);
    settings.TemporalAA = SanitizeTemporalAASettings(settings.TemporalAA);
    settings.TemporalSuperResolution=SanitizeTemporalSuperResolutionSettings(settings.TemporalSuperResolution);
    settings.TemporalEffects = SanitizeTemporalEffectsSettings(settings.TemporalEffects);
    settings.NPR = SanitizeNPRSettings(settings.NPR);
    return settings;
}

bool operator==(const TemporalAASettings& a, const TemporalAASettings& b) noexcept
{
    return a.HistoryWeight == b.HistoryWeight && a.DepthTolerance == b.DepthTolerance;
}
TemporalAASettings SanitizeTemporalAASettings(TemporalAASettings settings) noexcept
{
    if (!std::isfinite(settings.HistoryWeight)) settings.HistoryWeight = TemporalAASettings{}.HistoryWeight;
    if (!std::isfinite(settings.DepthTolerance)) settings.DepthTolerance = TemporalAASettings{}.DepthTolerance;
    settings.HistoryWeight = std::clamp(settings.HistoryWeight, 0.0f, 0.98f);
    settings.DepthTolerance = std::clamp(settings.DepthTolerance, 0.0001f, 0.1f);
    return settings;
}

bool operator==(const MotionBlurSettings& a, const MotionBlurSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.ShutterSeconds == b.ShutterSeconds && a.MaxRadiusPixels == b.MaxRadiusPixels;
}

MotionBlurSettings SanitizeMotionBlurSettings(MotionBlurSettings settings) noexcept
{
    const MotionBlurSettings defaults;
    constexpr float MaximumShutterSeconds = 0.1f;
    settings.ShutterSeconds = std::clamp(FiniteOr(settings.ShutterSeconds, defaults.ShutterSeconds), 0.0f, MaximumShutterSeconds);
    settings.MaxRadiusPixels = std::clamp(FiniteOr(settings.MaxRadiusPixels, defaults.MaxRadiusPixels),
        0.0f, static_cast<float>(MotionBlurTileSize));
    return settings;
}

std::vector<MotionBlurPassKind> BuildMotionBlurPlan(const MotionBlurSettings& settings)
{
    if (!settings.Enabled) return {};
    return {MotionBlurPassKind::TileMax, MotionBlurPassKind::NeighborMax,
        MotionBlurPassKind::Gather, MotionBlurPassKind::Composite};
}

bool operator==(const DepthOfFieldSettings& a, const DepthOfFieldSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.FocusDistance == b.FocusDistance && a.FNumber == b.FNumber &&
        a.SensorWidthMillimeters == b.SensorWidthMillimeters && a.MetersPerWorldUnit == b.MetersPerWorldUnit &&
        a.MaxRadiusPixels == b.MaxRadiusPixels;
}

DepthOfFieldSettings SanitizeDepthOfFieldSettings(DepthOfFieldSettings settings) noexcept
{
    const DepthOfFieldSettings defaults;
    constexpr float MinimumFocusDistance = 1.0e-4f, MaximumFocusDistance = 1.0e6f;
    constexpr float MinimumFNumber = 0.7f, MaximumFNumber = 64.0f;
    constexpr float MinimumSensorWidth = 1.0f, MaximumSensorWidth = 100.0f;
    constexpr float MinimumWorldScale = 1.0e-6f, MaximumWorldScale = 1.0e6f;
    constexpr float MaximumRadius = 16.0f;
    settings.FocusDistance = std::clamp(FiniteOr(settings.FocusDistance, defaults.FocusDistance), MinimumFocusDistance, MaximumFocusDistance);
    settings.FNumber = std::clamp(FiniteOr(settings.FNumber, defaults.FNumber), MinimumFNumber, MaximumFNumber);
    settings.SensorWidthMillimeters = std::clamp(FiniteOr(settings.SensorWidthMillimeters, defaults.SensorWidthMillimeters), MinimumSensorWidth, MaximumSensorWidth);
    settings.MetersPerWorldUnit = std::clamp(FiniteOr(settings.MetersPerWorldUnit, defaults.MetersPerWorldUnit), MinimumWorldScale, MaximumWorldScale);
    settings.MaxRadiusPixels = std::clamp(FiniteOr(settings.MaxRadiusPixels, defaults.MaxRadiusPixels), 0.0f, MaximumRadius);
    return settings;
}

std::vector<DepthOfFieldPassKind> BuildDepthOfFieldPlan(const DepthOfFieldSettings& settings)
{
    if (!settings.Enabled) return {};
    return {DepthOfFieldPassKind::Gather, DepthOfFieldPassKind::Composite};
}

std::vector<PostProcessPassKind> BuildPostProcessPlan(const PostProcessSettings& requested)
{
    const PostProcessSettings settings = SanitizePostProcessSettings(requested);
    std::vector<PostProcessPassKind> passes;
    // Ink the complete tone-mapped scene before geometric color warping and FXAA.
    if (settings.NPR.Enabled && settings.NPR.Blend > 0.0f) passes.push_back(PostProcessPassKind::NPR);
    // Pull-map the resolved scene before edge filtering. Vignette and grain
    // remain in display coordinates. A neutral warp needs no allocation/draw.
    if (settings.ScreenSpaceDistortion && settings.DistortionStrength != 0.0f)
        passes.push_back(PostProcessPassKind::ScreenSpaceDistortion);
    if (settings.AntiAliasing == AntiAliasingMode::FXAA) passes.push_back(PostProcessPassKind::FXAA);
    if (settings.Vignette) passes.push_back(PostProcessPassKind::Vignette);
    if (settings.FilmGrain) passes.push_back(PostProcessPassKind::FilmGrain);
    return passes;
}

bool operator==(const AmbientOcclusionSettings& a, const AmbientOcclusionSettings& b) noexcept
{
    return a.Mode == b.Mode && a.Radius == b.Radius && a.Bias == b.Bias && a.Intensity == b.Intensity &&
        a.HorizonAngleBiasDegrees == b.HorizonAngleBiasDegrees && a.RayTracing == b.RayTracing;
}

AmbientOcclusionSettings SanitizeAmbientOcclusionSettings(AmbientOcclusionSettings settings) noexcept
{
    const AmbientOcclusionSettings defaults;
    constexpr float MinimumRadius = 0.01f;
    constexpr float MaximumRadius = 5.0f;
    constexpr float MaximumBiasFraction = 0.5f;
    constexpr float MaximumIntensity = 4.0f;
    constexpr float MaximumHorizonAngleBiasDegrees = 45.0f;
    if (static_cast<std::uint32_t>(settings.Mode) > static_cast<std::uint32_t>(AmbientOcclusionMode::RayTraced))
        settings.Mode = AmbientOcclusionMode::None;
    settings.Radius = std::clamp(FiniteOr(settings.Radius, defaults.Radius), MinimumRadius, MaximumRadius);
    settings.Bias = std::clamp(FiniteOr(settings.Bias, defaults.Bias), 0.0f, settings.Radius * MaximumBiasFraction);
    settings.Intensity = std::clamp(FiniteOr(settings.Intensity, defaults.Intensity), 0.0f, MaximumIntensity);
    settings.HorizonAngleBiasDegrees = std::clamp(FiniteOr(settings.HorizonAngleBiasDegrees,
        defaults.HorizonAngleBiasDegrees), 0.0f, MaximumHorizonAngleBiasDegrees);
    settings.RayTracing = SanitizeRayTracedAOSettings(settings.RayTracing);
    return settings;
}

std::vector<AmbientOcclusionPassKind> BuildAmbientOcclusionPlan(
    const AmbientOcclusionSettings& settings, int width, int height)
{
    using Pass = AmbientOcclusionPassKind;
    std::vector<Pass> plan;
    if (!UsesAmbientOcclusion(settings) || width <= 0 || height <= 0) return plan;
    if (settings.Mode == AmbientOcclusionMode::SAO)
    {
        plan.push_back(Pass::DepthPrepare);
        const auto levels = AmbientOcclusionDepthMipCount(width, height);
        for (std::uint32_t level = 1; level < levels; ++level) plan.push_back(Pass::DepthReduce);
    }
    plan.insert(plan.end(), {Pass::Estimate, Pass::BilateralHorizontal, Pass::BilateralVertical, Pass::Composite});
    return plan;
}

CascadedShadowSettings SanitizeCascadedShadowSettings(CascadedShadowSettings value) noexcept
{
    const CascadedShadowSettings defaults;
    value.CascadeCount = std::clamp(value.CascadeCount, 1u, MaximumShadowCascades);
    value.SplitLambda = std::isfinite(value.SplitLambda) ? std::clamp(value.SplitLambda, 0.0f, 1.0f) : defaults.SplitLambda;
    value.MaximumDistance = std::isfinite(value.MaximumDistance) ? std::max(value.MaximumDistance, 0.0f) : defaults.MaximumDistance;
    constexpr float MaximumBlendFraction = 0.25f;
    value.BlendFraction = std::isfinite(value.BlendFraction) ? std::clamp(value.BlendFraction, 0.0f, MaximumBlendFraction) : defaults.BlendFraction;
    return value;
}

bool operator==(const CascadedShadowSettings& a, const CascadedShadowSettings& b) noexcept
{
    return a.CascadeCount == b.CascadeCount && a.SplitLambda == b.SplitLambda &&
        a.MaximumDistance == b.MaximumDistance && a.BlendFraction == b.BlendFraction;
}

std::uint32_t SanitizeSampleCount(std::uint32_t requested) noexcept
{
    for (const auto count : {8u, 4u, 2u}) if (requested >= count) return count;
    return 1;
}

std::uint32_t SelectSampleCount(std::uint32_t requested, std::uint32_t supported) noexcept
{
    for (auto count = SanitizeSampleCount(requested); count > 1; count /= 2)
        if ((supported & count) != 0) return count;
    return 1;
}

bool operator==(const PlanarReflectionSettings& a, const PlanarReflectionSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.Intensity == b.Intensity && a.Plane == b.Plane && a.ClipBias == b.ClipBias && a.ReceiverTolerance == b.ReceiverTolerance &&
        a.MinimumNormalCosine == b.MinimumNormalCosine && a.ResolutionDivisor == b.ResolutionDivisor &&
        a.BoundedReceiver == b.BoundedReceiver && a.ReceiverMinimum == b.ReceiverMinimum && a.ReceiverMaximum == b.ReceiverMaximum;
}

PlanarReflectionSettings SanitizePlanarReflectionSettings(PlanarReflectionSettings s) noexcept
{
    const PlanarReflectionSettings defaults;
    s.Intensity=std::clamp(FiniteOr(s.Intensity,defaults.Intensity),0.0f,4.0f);
    const double length = std::hypot(double(s.Plane[0]), double(s.Plane[1]), double(s.Plane[2]));
    bool valid = std::isfinite(length) && length > 0;
    for (float value : s.Plane) valid &= std::isfinite(value);
    std::array<float, 4> normalized {};
    if (valid)
        for (std::size_t index = 0; index < normalized.size(); ++index)
        {
            normalized[index] = static_cast<float>(s.Plane[index] / length);
            valid &= std::isfinite(normalized[index]);
        }
    // Preserve an already unit-length float plane. Renormalizing every settings
    // resolve can drift its distance and repeatedly invalidate render history.
    constexpr double UnitNormalTolerance = 2 * std::numeric_limits<float>::epsilon();
    if (!valid) s.Plane = defaults.Plane;
    else if (std::abs(length - 1) > UnitNormalTolerance) s.Plane = normalized;
    constexpr float MaximumWorldTolerance = 100.0f, MinimumReceiverTolerance = 1e-5f;
    s.ClipBias = std::clamp(FiniteOr(s.ClipBias, defaults.ClipBias), 0.0f, MaximumWorldTolerance);
    s.ReceiverTolerance = std::clamp(FiniteOr(s.ReceiverTolerance, defaults.ReceiverTolerance), MinimumReceiverTolerance, MaximumWorldTolerance);
    s.MinimumNormalCosine = std::clamp(FiniteOr(s.MinimumNormalCosine, defaults.MinimumNormalCosine), 0.0f, 1.0f);
    for (std::size_t axis = 0; axis < s.ReceiverMinimum.size(); ++axis)
    {
        s.ReceiverMinimum[axis] = FiniteOr(s.ReceiverMinimum[axis], 0.0f);
        s.ReceiverMaximum[axis] = FiniteOr(s.ReceiverMaximum[axis], 0.0f);
        if (s.ReceiverMinimum[axis] > s.ReceiverMaximum[axis]) std::swap(s.ReceiverMinimum[axis], s.ReceiverMaximum[axis]);
    }
    constexpr std::uint32_t FullResolution = 1, HalfResolution = 2, QuarterResolution = 4;
    s.ResolutionDivisor = s.ResolutionDivisor >= QuarterResolution ? QuarterResolution :
        s.ResolutionDivisor >= HalfResolution ? HalfResolution : FullResolution;
    return s;
}

bool operator==(const LumenTemporalSettings& a,const LumenTemporalSettings& b) noexcept
{
    return a.Enabled==b.Enabled && a.HistoryWeight==b.HistoryWeight && a.DepthTolerance==b.DepthTolerance &&
        a.NormalThreshold==b.NormalThreshold && a.PlaneTolerance==b.PlaneTolerance && a.ClipTolerance==b.ClipTolerance && a.MaximumFrames==b.MaximumFrames;
}
LumenTemporalSettings SanitizeLumenTemporalSettings(LumenTemporalSettings temporal) noexcept
{
    const LumenTemporalSettings defaults;
    temporal.HistoryWeight=std::clamp(FiniteOr(temporal.HistoryWeight,defaults.HistoryWeight),0.0f,0.98f);
    temporal.DepthTolerance=std::clamp(FiniteOr(temporal.DepthTolerance,defaults.DepthTolerance),0.0f,1.0f);
    temporal.NormalThreshold=std::clamp(FiniteOr(temporal.NormalThreshold,defaults.NormalThreshold),0.0f,1.0f);
    temporal.PlaneTolerance=std::clamp(FiniteOr(temporal.PlaneTolerance,defaults.PlaneTolerance),0.0f,1000.0f);
    temporal.ClipTolerance=std::clamp(FiniteOr(temporal.ClipTolerance,defaults.ClipTolerance),0.0f,1.0f);
    temporal.MaximumFrames=std::clamp(temporal.MaximumFrames,1u,64u);
    return temporal;
}
bool operator==(const LumenReflectionSettings& a,const LumenReflectionSettings& b) noexcept
{
    return a.ResolutionDivisor==b.ResolutionDivisor && a.FilterRadius==b.FilterRadius && a.ScreenTracing==b.ScreenTracing &&
        a.NormalCosine==b.NormalCosine && a.PlaneTolerance==b.PlaneTolerance && a.RelativePlaneTolerance==b.RelativePlaneTolerance &&
        a.RoughnessTolerance==b.RoughnessTolerance && a.Temporal==b.Temporal;
}
LumenReflectionSettings SanitizeLumenReflectionSettings(LumenReflectionSettings settings) noexcept
{
    const LumenReflectionSettings defaults;
    constexpr unsigned FullResolution=1,HalfResolution=2,QuarterResolution=4,MaximumFilterRadius=2;
    constexpr float MaximumPlaneTolerance=1000;
    settings.ResolutionDivisor=settings.ResolutionDivisor>=QuarterResolution?QuarterResolution:settings.ResolutionDivisor>=HalfResolution?HalfResolution:FullResolution;
    settings.FilterRadius=std::min(settings.FilterRadius,MaximumFilterRadius);
    settings.NormalCosine=std::clamp(FiniteOr(settings.NormalCosine,defaults.NormalCosine),0.0f,1.0f);
    settings.PlaneTolerance=std::clamp(FiniteOr(settings.PlaneTolerance,defaults.PlaneTolerance),0.0f,MaximumPlaneTolerance);
    settings.RelativePlaneTolerance=std::clamp(FiniteOr(settings.RelativePlaneTolerance,defaults.RelativePlaneTolerance),0.0f,1.0f);
    settings.RoughnessTolerance=std::clamp(FiniteOr(settings.RoughnessTolerance,defaults.RoughnessTolerance),0.0f,1.0f);
    settings.Temporal=SanitizeLumenTemporalSettings(settings.Temporal); return settings;
}

AtmosphereSettings SanitizeAtmosphereSettings(AtmosphereSettings s) noexcept
{
    const AtmosphereSettings defaults;
    const auto limit=[](float value,float fallback,float low,float high)
    { return std::clamp(FiniteOr(value,fallback),low,high); };
    s.PlanetRadius=limit(s.PlanetRadius,defaults.PlanetRadius,100000,10000000);
    s.Height=limit(s.Height,defaults.Height,5000,150000);
    s.GroundHeight=limit(s.GroundHeight,0,-10000,10000);
    s.RayleighHeight=limit(s.RayleighHeight,defaults.RayleighHeight,1000,s.Height*0.5f);
    s.MieHeight=limit(s.MieHeight,defaults.MieHeight,100,s.Height*0.25f);
    s.RayleighStrength=limit(s.RayleighStrength,1,0,5);
    s.MieStrength=limit(s.MieStrength,1,0,10);
    s.MieAnisotropy=limit(s.MieAnisotropy,defaults.MieAnisotropy,0,0.95f);
    s.Intensity=limit(s.Intensity,1,0,4);
    s.ViewSamples=std::clamp(s.ViewSamples,4u,32u);
    s.LightSamples=std::clamp(s.LightSamples,4u,32u);
    float length=0;
    for (unsigned i=0;i<3;++i)
    {
        s.Wavelengths[i]=limit(s.Wavelengths[i],defaults.Wavelengths[i],380,780);
        s.SunRadiance[i]=limit(s.SunRadiance[i],defaults.SunRadiance[i],0,200);
        s.SunDirection[i]=limit(s.SunDirection[i],defaults.SunDirection[i],-1,1);
        length+=s.SunDirection[i]*s.SunDirection[i];
    }
    if (length<1e-8f) s.SunDirection=defaults.SunDirection;
    else for (auto& component:s.SunDirection) component/=std::sqrt(length);
    return s;
}
bool operator==(const AtmosphereSettings& a,const AtmosphereSettings& b) noexcept
{
    return a.Enabled==b.Enabled && a.AerialPerspective==b.AerialPerspective && a.PlanetRadius==b.PlanetRadius &&
        a.Height==b.Height && a.GroundHeight==b.GroundHeight && a.RayleighHeight==b.RayleighHeight &&
        a.MieHeight==b.MieHeight && a.RayleighStrength==b.RayleighStrength && a.MieStrength==b.MieStrength &&
        a.MieAnisotropy==b.MieAnisotropy && a.Wavelengths==b.Wavelengths && a.Intensity==b.Intensity &&
        a.ViewSamples==b.ViewSamples && a.LightSamples==b.LightSamples &&
        a.SunDirection==b.SunDirection && a.SunRadiance==b.SunRadiance;
}

bool operator==(const ReflectionSettings& a, const ReflectionSettings& b) noexcept
{
    return a.Mode == b.Mode && a.Skybox==b.Skybox && a.SkyboxIntensity==b.SkyboxIntensity && a.DiffuseIBL==b.DiffuseIBL && a.Intensity == b.Intensity && a.EnvironmentIntensity == b.EnvironmentIntensity && a.MaximumDistance == b.MaximumDistance &&
        a.Thickness == b.Thickness && a.NormalBias == b.NormalBias && a.MaximumRoughness == b.MaximumRoughness &&
        a.EdgeFadePixels == b.EdgeFadePixels && a.StridePixels == b.StridePixels && a.MaximumSteps == b.MaximumSteps &&
        a.CaptureResolution == b.CaptureResolution && a.Planar == b.Planar && a.RayTracing == b.RayTracing && a.Lumen == b.Lumen && a.Atmosphere==b.Atmosphere;
}

ReflectionSettings SanitizeReflectionSettings(ReflectionSettings s) noexcept
{
    const ReflectionSettings defaults;
    s.Atmosphere=SanitizeAtmosphereSettings(s.Atmosphere);
    constexpr float MinimumDistance = 0.001f, MaximumDistance = 10000.0f;
    constexpr float MaximumThickness = 100.0f, MaximumEdgeFade = 256.0f, MaximumStride = 8.0f;
    constexpr std::uint32_t MaximumSteps = 512;
    if (s.Mode!=ReflectionMode::ScreenSpace && s.Mode!=ReflectionMode::Cubemap && s.Mode!=ReflectionMode::LatitudeLongitude &&
        s.Mode!=ReflectionMode::RayTraced && s.Mode!=ReflectionMode::Lumen) s.Mode=ReflectionMode::None;
    s.Intensity = std::clamp(FiniteOr(s.Intensity, defaults.Intensity), 0.0f, 4.0f);
    s.EnvironmentIntensity=std::clamp(FiniteOr(s.EnvironmentIntensity,defaults.EnvironmentIntensity),0.0f,4.0f);
    s.DiffuseIBL=std::clamp(FiniteOr(s.DiffuseIBL,defaults.DiffuseIBL),0.0f,8.0f);
    s.SkyboxIntensity=std::clamp(FiniteOr(s.SkyboxIntensity,defaults.SkyboxIntensity),0.0f,8.0f);
    s.MaximumDistance = std::clamp(FiniteOr(s.MaximumDistance, defaults.MaximumDistance), MinimumDistance, MaximumDistance);
    s.Thickness = std::clamp(FiniteOr(s.Thickness, defaults.Thickness), MinimumDistance, MaximumThickness);
    s.NormalBias = std::clamp(FiniteOr(s.NormalBias, defaults.NormalBias), 0.0f,
        (s.Mode == ReflectionMode::RayTraced || UsesLumenReflections(s)) ? s.MaximumDistance : s.Thickness);
    s.MaximumRoughness = std::clamp(FiniteOr(s.MaximumRoughness, defaults.MaximumRoughness), MinimumRoughness, 1.0f);
    s.EdgeFadePixels = std::clamp(FiniteOr(s.EdgeFadePixels, defaults.EdgeFadePixels), 0.0f, MaximumEdgeFade);
    s.StridePixels = std::clamp(FiniteOr(s.StridePixels, defaults.StridePixels), 1.0f, MaximumStride);
    s.MaximumSteps = std::clamp(s.MaximumSteps, 1u, MaximumSteps);
    constexpr std::uint32_t MinimumCaptureResolution = 16, MaximumCaptureResolution = 1024;
    const auto requestedResolution = std::clamp(s.CaptureResolution, MinimumCaptureResolution, MaximumCaptureResolution);
    s.CaptureResolution = MinimumCaptureResolution;
    while (s.CaptureResolution * 2 <= requestedResolution) s.CaptureResolution *= 2;
    s.Planar = SanitizePlanarReflectionSettings(s.Planar);
    s.RayTracing = SanitizeRayTracedReflectionSettings(s.RayTracing);
    s.Lumen = SanitizeLumenReflectionSettings(s.Lumen);
    if (UsesLumenReflections(s)) { constexpr float MaximumLumenStride=4; s.StridePixels=std::min(s.StridePixels,MaximumLumenStride); }
    return s;
}

bool operator==(const LodSettings& a, const LodSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.MaximumErrorPixels == b.MaximumErrorPixels && a.HysteresisFraction == b.HysteresisFraction;
}

LodSettings SanitizeLodSettings(LodSettings settings) noexcept
{
    const LodSettings defaults;
    settings.MaximumErrorPixels = std::clamp(FiniteOr(settings.MaximumErrorPixels, defaults.MaximumErrorPixels),
        MinimumLodErrorPixels, MaximumLodErrorPixels);
    settings.HysteresisFraction = std::clamp(FiniteOr(settings.HysteresisFraction, defaults.HysteresisFraction), 0.0f, MaximumLodHysteresis);
    return settings;
}

bool operator==(const PCSSSettings& a, const PCSSSettings& b) noexcept
{
    return a.EmitterRadius == b.EmitterRadius && a.AngularRadiusDegrees == b.AngularRadiusDegrees &&
        a.MaximumSearchRadius == b.MaximumSearchRadius && a.MaximumFilterRadius == b.MaximumFilterRadius &&
        a.BlockerSamples == b.BlockerSamples && a.FilterSamples == b.FilterSamples;
}

PCSSSettings SanitizePCSSSettings(PCSSSettings settings) noexcept
{
    const PCSSSettings defaults;
    constexpr float MaximumEmitterRadius = 10000.0f, MaximumAngularRadiusDegrees = 15.0f;
    settings.EmitterRadius = std::clamp(FiniteOr(settings.EmitterRadius, defaults.EmitterRadius), 0.0f, MaximumEmitterRadius);
    settings.AngularRadiusDegrees = std::clamp(FiniteOr(settings.AngularRadiusDegrees, defaults.AngularRadiusDegrees), 0.0f, MaximumAngularRadiusDegrees);
    settings.MaximumSearchRadius = std::clamp(settings.MaximumSearchRadius, 1u, MaximumPCSSRadius);
    settings.MaximumFilterRadius = std::clamp(settings.MaximumFilterRadius, 1u, settings.MaximumSearchRadius);
    const auto samples = [](std::uint32_t count) { return count <= 16 ? 16u : count <= 32 ? 32u : count <= 64 ? 64u : 128u; };
    settings.BlockerSamples = samples(settings.BlockerSamples); settings.FilterSamples = samples(settings.FilterSamples);
    return settings;
}

bool operator==(const VSMSettings& a, const VSMSettings& b) noexcept
{
    return a.BlurRadius == b.BlurRadius && a.BlurSigma == b.BlurSigma && a.MinimumVariance == b.MinimumVariance &&
        a.MaximumDerivativeVariance == b.MaximumDerivativeVariance && a.LightBleedingReduction == b.LightBleedingReduction;
}

VSMSettings SanitizeVSMSettings(VSMSettings settings) noexcept
{
    const VSMSettings defaults;
    constexpr float MinimumSigma = 0.25f, MaximumSigma = 32.0f;
    constexpr float MinimumVarianceFloor = 1.0e-8f, MaximumVarianceFloor = 0.1f, MaximumBleedingReduction = 0.95f;
    settings.BlurRadius = std::min(settings.BlurRadius, MaximumVSMBlurRadius);
    settings.BlurSigma = std::clamp(FiniteOr(settings.BlurSigma, defaults.BlurSigma), MinimumSigma, MaximumSigma);
    settings.MinimumVariance = std::clamp(FiniteOr(settings.MinimumVariance, defaults.MinimumVariance), MinimumVarianceFloor, MaximumVarianceFloor);
    settings.MaximumDerivativeVariance = std::clamp(FiniteOr(settings.MaximumDerivativeVariance, defaults.MaximumDerivativeVariance), 0.0f, 1.0f);
    settings.LightBleedingReduction = std::clamp(FiniteOr(settings.LightBleedingReduction, defaults.LightBleedingReduction), 0.0f, MaximumBleedingReduction);
    return settings;
}

bool operator==(const ShadowFilterSettings& a, const ShadowFilterSettings& b) noexcept
{
    return a.Mode == b.Mode && a.Radius == b.Radius && a.PCSS == b.PCSS && a.VSM == b.VSM;
}

ShadowFilterSettings SanitizeShadowFilterSettings(ShadowFilterSettings settings) noexcept
{
    if (settings.Mode != ShadowFilterMode::Hard && settings.Mode != ShadowFilterMode::PCF && settings.Mode != ShadowFilterMode::PCSS && settings.Mode != ShadowFilterMode::VSM)
        settings.Mode = ShadowFilterMode::PCF;
    settings.Radius = std::clamp(settings.Radius, 1u, MaximumShadowFilterRadius);
    settings.PCSS = SanitizePCSSSettings(settings.PCSS);
    settings.VSM = SanitizeVSMSettings(settings.VSM);
    return settings;
}

bool operator==(const ParticipatingMediaSettings& a, const ParticipatingMediaSettings& b) noexcept
{
    return a.Absorption == b.Absorption && a.Scattering == b.Scattering && a.Emission == b.Emission &&
        a.BoundsMinimum == b.BoundsMinimum && a.BoundsMaximum == b.BoundsMaximum && a.Density == b.Density &&
        a.ModulationStrength == b.ModulationStrength && a.ModulationFrequency == b.ModulationFrequency &&
        a.LightSamples == b.LightSamples;
}
ParticipatingMediaSettings SanitizeParticipatingMediaSettings(ParticipatingMediaSettings settings) noexcept
{
    const ParticipatingMediaSettings defaults;
    constexpr float MaximumCoefficient = 100.0f, MaximumEmission = 1000.0f;
    constexpr float WorldLimit = 100000.0f, MinimumBoxExtent = 0.001f;
    for (std::size_t channel = 0; channel < settings.Absorption.size(); ++channel)
    {
        settings.Absorption[channel] = std::clamp(FiniteOr(settings.Absorption[channel], defaults.Absorption[channel]), 0.0f, MaximumCoefficient);
        settings.Scattering[channel] = std::clamp(FiniteOr(settings.Scattering[channel], defaults.Scattering[channel]), 0.0f, MaximumCoefficient);
        settings.Emission[channel] = std::clamp(FiniteOr(settings.Emission[channel], defaults.Emission[channel]), 0.0f, MaximumEmission);
        auto& low = settings.BoundsMinimum[channel]; auto& high = settings.BoundsMaximum[channel];
        low = std::clamp(FiniteOr(low, defaults.BoundsMinimum[channel]), -WorldLimit, WorldLimit);
        high = std::clamp(FiniteOr(high, defaults.BoundsMaximum[channel]), -WorldLimit, WorldLimit);
        if (low > high) std::swap(low, high);
        if (high - low < MinimumBoxExtent) { low = defaults.BoundsMinimum[channel]; high = defaults.BoundsMaximum[channel]; }
    }
    settings.Density = std::clamp(FiniteOr(settings.Density, defaults.Density), 0.0f, MaximumCoefficient);
    settings.ModulationStrength = std::clamp(FiniteOr(settings.ModulationStrength, defaults.ModulationStrength), 0.0f, 1.0f);
    settings.ModulationFrequency = std::clamp(FiniteOr(settings.ModulationFrequency, defaults.ModulationFrequency), 0.0f, 100.0f);
    settings.LightSamples = std::clamp(settings.LightSamples, 1u, 64u);
    return settings;
}

bool operator==(const VolumetricCloudSettings& a, const VolumetricCloudSettings& b) noexcept
{
    return a.Animate==b.Animate && a.AnimationSpeed==b.AnimationSpeed && a.Medium == b.Medium && a.Coverage == b.Coverage && a.WeatherVariation == b.WeatherVariation &&
        a.WeatherFrequency == b.WeatherFrequency && a.BaseFrequency == b.BaseFrequency &&
        a.DetailFrequency == b.DetailFrequency && a.Erosion == b.Erosion && a.BaseFade == b.BaseFade &&
        a.TopFade == b.TopFade && a.EdgeFade == b.EdgeFade && a.DensityContrast == b.DensityContrast &&
        a.Offset == b.Offset && a.Seed == b.Seed;
}
VolumetricCloudSettings SanitizeVolumetricCloudSettings(VolumetricCloudSettings settings) noexcept
{
    const VolumetricCloudSettings defaults;
    settings.Medium = SanitizeParticipatingMediaSettings(settings.Medium);
    // Sinusoidal modulation belongs to the generic medium, not cloud density.
    settings.Medium.ModulationStrength = 0;
    auto clamp = [](float value, float fallback, float low, float high)
        { return std::clamp(FiniteOr(value, fallback), low, high); };
    settings.AnimationSpeed=clamp(settings.AnimationSpeed,defaults.AnimationSpeed,0.0f,5.0f);
    settings.Coverage = clamp(settings.Coverage, defaults.Coverage, 0.0f, 1.0f);
    settings.WeatherVariation = clamp(settings.WeatherVariation, defaults.WeatherVariation, 0.0f, 1.0f);
    settings.WeatherFrequency = clamp(settings.WeatherFrequency, defaults.WeatherFrequency, 0.0f, 10.0f);
    settings.BaseFrequency = clamp(settings.BaseFrequency, defaults.BaseFrequency, 0.0f, 10.0f);
    settings.DetailFrequency = clamp(settings.DetailFrequency, defaults.DetailFrequency, 0.0f, 10.0f);
    settings.Erosion = clamp(settings.Erosion, defaults.Erosion, 0.0f, 1.0f);
    settings.BaseFade = clamp(settings.BaseFade, defaults.BaseFade, 0.001f, 0.49f);
    settings.TopFade = clamp(settings.TopFade, defaults.TopFade, 0.51f, 0.999f);
    settings.EdgeFade = clamp(settings.EdgeFade, defaults.EdgeFade, 0.001f, 0.5f);
    settings.DensityContrast = clamp(settings.DensityContrast, defaults.DensityContrast, 0.25f, 4.0f);
    for (auto& offset : settings.Offset) offset = clamp(offset, 0.0f, -100000.0f, 100000.0f);
    settings.Seed = std::min(settings.Seed, 65535u);
    return settings;
}

bool operator==(const VolumetricFogSettings& a, const VolumetricFogSettings& b) noexcept
{
    const auto& u=a.Underwater; const auto& v=b.Underwater;
    return u.SurfaceHeight==v.SurfaceHeight && u.BottomHeight==v.BottomHeight && u.Radius==v.Radius &&
        u.Extinction==v.Extinction && u.ToSun==v.ToSun && u.Irradiance==v.Irradiance && u.WaveA==v.WaveA && u.WaveB==v.WaveB &&
        a.Enabled == b.Enabled && a.MediumMode == b.MediumMode && a.Participating == b.Participating && a.Clouds == b.Clouds &&
        a.AtmosphereMode == b.AtmosphereMode && a.Extinction == b.Extinction && a.BaseHeight == b.BaseHeight &&
        a.HeightFalloff == b.HeightFalloff && a.MaximumDistance == b.MaximumDistance &&
        a.Radiance == b.Radiance && a.SliceCount == b.SliceCount && a.SamplesPerSlice == b.SamplesPerSlice &&
        a.BoundedTilePixels == b.BoundedTilePixels;
}

VolumetricFogSettings SanitizeVolumetricFogSettings(VolumetricFogSettings settings) noexcept
{
    const VolumetricFogSettings defaults;
    settings.AtmosphereMode=std::min(settings.AtmosphereMode,3u);
    if (settings.MediumMode != VolumeMediumMode::Participating && settings.MediumMode != VolumeMediumMode::Clouds)
        settings.MediumMode = VolumeMediumMode::HeightFog;
    settings.Participating = SanitizeParticipatingMediaSettings(settings.Participating);
    settings.Clouds = SanitizeVolumetricCloudSettings(settings.Clouds);
    constexpr float MaximumCoefficient = 100.0f;
    constexpr float MaximumWorldDistance = 100000.0f;
    constexpr float MaximumRadiance = 1000.0f;
    auto& water=settings.Underwater;
    water.SurfaceHeight=std::clamp(FiniteOr(water.SurfaceHeight,0),-MaximumWorldDistance,MaximumWorldDistance);
    water.BottomHeight=std::clamp(FiniteOr(water.BottomHeight,0),-MaximumWorldDistance,water.SurfaceHeight);
    water.Radius=std::clamp(FiniteOr(water.Radius,0),0.0f,MaximumWorldDistance);
    water.Extinction=water.Radius>0 && water.SurfaceHeight>water.BottomHeight ?
        std::clamp(FiniteOr(water.Extinction,0),0.0f,MaximumCoefficient) : 0;
    for (auto& value:water.ToSun) value=std::clamp(FiniteOr(value,0),-1.0f,1.0f);
    for (auto& value:water.Irradiance) value=std::clamp(FiniteOr(value,0),0.0f,MaximumRadiance);
    for (auto* wave:{&water.WaveA,&water.WaveB})
    {
        for (auto& value:*wave) value=FiniteOr(value,0);
        (*wave)[3]=std::clamp((*wave)[3],0.0f,1.0f);
    }
    settings.Extinction = std::clamp(FiniteOr(settings.Extinction, defaults.Extinction), 0.0f, MaximumCoefficient);
    settings.BaseHeight = std::clamp(FiniteOr(settings.BaseHeight, defaults.BaseHeight), -MaximumWorldDistance, MaximumWorldDistance);
    settings.HeightFalloff = std::clamp(FiniteOr(settings.HeightFalloff, defaults.HeightFalloff), 0.0f, MaximumCoefficient);
    settings.MaximumDistance = std::clamp(FiniteOr(settings.MaximumDistance, defaults.MaximumDistance), 0.0f, MaximumWorldDistance);
    for (std::size_t channel = 0; channel < settings.Radiance.size(); ++channel)
        settings.Radiance[channel] = std::clamp(FiniteOr(settings.Radiance[channel], defaults.Radiance[channel]), 0.0f, MaximumRadiance);
    settings.SliceCount = settings.SliceCount <= 16 ? 16 : settings.SliceCount <= 32 ? 32 : 64;
    settings.SamplesPerSlice = std::clamp(settings.SamplesPerSlice, 1u, 4u);
    settings.BoundedTilePixels = std::clamp(settings.BoundedTilePixels, 1u, 16u);
    return settings;
}

bool operator==(const TransparencySettings& a, const TransparencySettings& b) noexcept
{
    return a.Mode == b.Mode && a.OpacityExponent == b.OpacityExponent && a.DepthExponent == b.DepthExponent;
}

bool operator==(const VolumetricLightingSettings& a, const VolumetricLightingSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.ScatteringAlbedo == b.ScatteringAlbedo && a.Anisotropy == b.Anisotropy &&
        a.Intensity == b.Intensity && a.ShadowDepthBias == b.ShadowDepthBias;
}

VolumetricLightingSettings SanitizeVolumetricLightingSettings(VolumetricLightingSettings settings) noexcept
{
    const VolumetricLightingSettings defaults;
    constexpr float MaximumAnisotropy = 0.9f, MaximumIntensity = 100.0f, MaximumDepthBias = 0.05f;
    for (std::size_t channel = 0; channel < settings.ScatteringAlbedo.size(); ++channel)
        settings.ScatteringAlbedo[channel] = std::clamp(FiniteOr(settings.ScatteringAlbedo[channel], defaults.ScatteringAlbedo[channel]), 0.0f, 1.0f);
    settings.Anisotropy = std::clamp(FiniteOr(settings.Anisotropy, defaults.Anisotropy), -MaximumAnisotropy, MaximumAnisotropy);
    settings.Intensity = std::clamp(FiniteOr(settings.Intensity, defaults.Intensity), 0.0f, MaximumIntensity);
    settings.ShadowDepthBias = std::clamp(FiniteOr(settings.ShadowDepthBias, defaults.ShadowDepthBias), 0.0f, MaximumDepthBias);
    return settings;
}

TransparencySettings SanitizeTransparencySettings(TransparencySettings settings) noexcept
{
    if (settings.Mode != TransparencyMode::WeightedBlended && settings.Mode != TransparencyMode::DepthPeeling) settings.Mode = TransparencyMode::Sorted;
    settings.OpacityExponent = std::clamp(FiniteOr(settings.OpacityExponent, 1.0f), 0.0f, 8.0f);
    settings.DepthExponent = std::clamp(FiniteOr(settings.DepthExponent, 3.0f), 0.0f, 8.0f);
    return settings;
}

bool operator==(const LightShaftSettings& a, const LightShaftSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.LightIndex == b.LightIndex && a.Intensity == b.Intensity &&
        a.SourceRadius == b.SourceRadius && a.Decay == b.Decay && a.DepthBias == b.DepthBias && a.SampleCount == b.SampleCount;
}

LightShaftSettings SanitizeLightShaftSettings(LightShaftSettings settings) noexcept
{
    const LightShaftSettings defaults;
    constexpr float MaximumIntensity = 10.0f, MaximumRadius = 2.0f, MaximumDecay = 12.0f, MaximumBias = 0.01f;
    settings.LightIndex = std::min(settings.LightIndex, 3u);
    settings.Intensity = std::clamp(FiniteOr(settings.Intensity, defaults.Intensity), 0.0f, MaximumIntensity);
    settings.SourceRadius = std::clamp(FiniteOr(settings.SourceRadius, defaults.SourceRadius), 0.0f, MaximumRadius);
    settings.Decay = std::clamp(FiniteOr(settings.Decay, defaults.Decay), 0.0f, MaximumDecay);
    settings.DepthBias = std::clamp(FiniteOr(settings.DepthBias, defaults.DepthBias), 0.0f, MaximumBias);
    settings.SampleCount = settings.SampleCount <= 16 ? 16 : settings.SampleCount <= 32 ? 32 : settings.SampleCount <= 64 ? 64 : 128;
    return settings;
}

bool operator==(const ContactShadowSettings& a, const ContactShadowSettings& b) noexcept
{
    return a.Enabled == b.Enabled && a.MaximumDistance == b.MaximumDistance && a.StartBias == b.StartBias &&
        a.Thickness == b.Thickness && a.MaximumSteps == b.MaximumSteps;
}
ContactShadowSettings SanitizeContactShadowSettings(ContactShadowSettings settings) noexcept
{
    constexpr float MinimumDistance = 0.001f, MaximumDistance = 100.0f, MaximumBiasOrThickness = 10.0f;
    constexpr std::uint32_t MaximumSteps = 256;
    const ContactShadowSettings defaults;
    settings.MaximumDistance = std::clamp(FiniteOr(settings.MaximumDistance, defaults.MaximumDistance), MinimumDistance, MaximumDistance);
    settings.StartBias = std::clamp(FiniteOr(settings.StartBias, defaults.StartBias), 0.0f, MaximumBiasOrThickness);
    settings.Thickness = std::clamp(FiniteOr(settings.Thickness, defaults.Thickness), 0.0f, MaximumBiasOrThickness);
    settings.MaximumSteps = std::clamp(settings.MaximumSteps, 1u, MaximumSteps);
    return settings;
}

bool operator==(const LightProbeSettings& a, const LightProbeSettings& b) noexcept
{
    return a.AutomaticBounds == b.AutomaticBounds && a.BoundsMinimum == b.BoundsMinimum && a.BoundsMaximum == b.BoundsMaximum &&
        a.GridAxisCount == b.GridAxisCount && a.RayCount == b.RayCount && a.MaximumBackfaceFraction == b.MaximumBackfaceFraction &&
        a.NormalBias == b.NormalBias && a.VisibilityBias == b.VisibilityBias && a.MinimumVariance == b.MinimumVariance &&
        a.VisibilitySharpness == b.VisibilitySharpness && a.MomentSharpness == b.MomentSharpness;
}
LightProbeSettings SanitizeLightProbeSettings(LightProbeSettings settings) noexcept
{
    const LightProbeSettings defaults;
    constexpr float MaximumCoordinate = 1000000.0f, MinimumExtent = 0.001f;
    constexpr std::uint32_t MinimumAxis = 2, MaximumAxis = 8, MinimumRays = 16, MaximumRays = 1024;
    settings.GridAxisCount = std::clamp(settings.GridAxisCount, MinimumAxis, MaximumAxis);
    settings.RayCount = std::clamp(settings.RayCount, MinimumRays, MaximumRays);
    for (std::size_t axis = 0; axis < settings.BoundsMinimum.size(); ++axis)
    {
        settings.BoundsMinimum[axis] = std::clamp(FiniteOr(settings.BoundsMinimum[axis], defaults.BoundsMinimum[axis]), -MaximumCoordinate, MaximumCoordinate);
        settings.BoundsMaximum[axis] = std::clamp(FiniteOr(settings.BoundsMaximum[axis], defaults.BoundsMaximum[axis]), -MaximumCoordinate, MaximumCoordinate);
        if (settings.BoundsMaximum[axis] - settings.BoundsMinimum[axis] < MinimumExtent)
        { settings.BoundsMinimum[axis] = defaults.BoundsMinimum[axis]; settings.BoundsMaximum[axis] = defaults.BoundsMaximum[axis]; }
    }
    settings.MaximumBackfaceFraction = std::clamp(FiniteOr(settings.MaximumBackfaceFraction, defaults.MaximumBackfaceFraction), 0.0f, 1.0f);
    settings.NormalBias = std::clamp(FiniteOr(settings.NormalBias, defaults.NormalBias), 0.0f, 0.5f);
    settings.VisibilityBias = std::clamp(FiniteOr(settings.VisibilityBias, defaults.VisibilityBias), 0.0f, 0.5f);
    settings.MinimumVariance = std::clamp(FiniteOr(settings.MinimumVariance, defaults.MinimumVariance), 0.000001f, 0.25f);
    settings.VisibilitySharpness = std::clamp(FiniteOr(settings.VisibilitySharpness, defaults.VisibilitySharpness), 1.0f, 8.0f);
    settings.MomentSharpness = std::clamp(FiniteOr(settings.MomentSharpness, defaults.MomentSharpness), 1.0f, 64.0f);
    return settings;
}
bool operator==(const LightmapSettings& a, const LightmapSettings& b) noexcept
{
    return a.Resolution == b.Resolution && a.Padding == b.Padding && a.RayCount == b.RayCount &&
        a.MaximumNormalAngleDegrees == b.MaximumNormalAngleDegrees;
}
LightmapSettings SanitizeLightmapSettings(LightmapSettings settings) noexcept
{
    const LightmapSettings defaults;
    constexpr std::uint32_t MinimumResolution = 64, MaximumResolution = 4096, MinimumPadding = 1, MaximumPadding = 16;
    constexpr std::uint32_t MinimumRays = 4, MaximumRays = 1024;
    constexpr float MaximumNormalAngle = 80;
    settings.Resolution = std::clamp(settings.Resolution,MinimumResolution,MaximumResolution);
    settings.Padding = std::clamp(settings.Padding,MinimumPadding,MaximumPadding);
    settings.RayCount = std::clamp(settings.RayCount,MinimumRays,MaximumRays);
    settings.MaximumNormalAngleDegrees = std::clamp(FiniteOr(settings.MaximumNormalAngleDegrees,defaults.MaximumNormalAngleDegrees),0.0f,MaximumNormalAngle);
    return settings;
}
bool operator==(const LumenSettings& a,const LumenSettings& b) noexcept
{ return a.HardwareRayTracing==b.HardwareRayTracing && a.HairTransport==b.HairTransport && a.HairIndirect==b.HairIndirect && a.HairGridSide==b.HairGridSide && a.TileSide==b.TileSide && a.RaysPerProbe==b.RaysPerProbe && a.SourceResolution==b.SourceResolution && a.CardSide==b.CardSide && a.LightingTilesPerUpdate==b.LightingTilesPerUpdate && a.CaptureCardsPerUpdate==b.CaptureCardsPerUpdate && a.SkyRadiance==b.SkyRadiance && a.Environment==b.Environment && a.Shadows==b.Shadows &&
    a.Translucency.Enabled==b.Translucency.Enabled && a.Translucency.VisibilitySupport==b.Translucency.VisibilitySupport &&
    a.Translucency.TileSide==b.Translucency.TileSide && a.Translucency.SliceCount==b.Translucency.SliceCount &&
    a.Translucency.RaysPerProbe==b.Translucency.RaysPerProbe && a.Translucency.DepthExponent==b.Translucency.DepthExponent &&
    a.Radiosity.Enabled==b.Radiosity.Enabled && a.Radiosity.Bounces==b.Radiosity.Bounces && a.Radiosity.RaysPerTexel==b.Radiosity.RaysPerTexel &&
    a.AdaptiveProbes==b.AdaptiveProbes && a.AdaptiveProbeLevels==b.AdaptiveProbeLevels && a.RefineCards==b.RefineCards && a.MaximumMeshCards==b.MaximumMeshCards && a.CardCoverage==b.CardCoverage &&
    a.GlobalDistance.Enabled==b.GlobalDistance.Enabled && a.GlobalDistance.GridSide==b.GlobalDistance.GridSide && a.GlobalDistance.CascadeCount==b.GlobalDistance.CascadeCount &&
    a.GlobalDistance.VoxelsPerUpdate==b.GlobalDistance.VoxelsPerUpdate && a.GlobalDistance.MaximumSteps==b.GlobalDistance.MaximumSteps && a.GlobalDistance.CellScale==b.GlobalDistance.CellScale && a.GlobalDistance.NearScale==b.GlobalDistance.NearScale &&
    a.WorldCache.Enabled==b.WorldCache.Enabled && a.WorldCache.Parallax==b.WorldCache.Parallax && a.WorldCache.GridSide==b.WorldCache.GridSide &&
    a.WorldCache.DirectionalSide==b.WorldCache.DirectionalSide && a.WorldCache.CellScale==b.WorldCache.CellScale &&
    a.WorldCache.CascadeCount==b.WorldCache.CascadeCount && a.WorldCache.BlendCells==b.WorldCache.BlendCells && a.WorldCache.ProbesPerUpdate==b.WorldCache.ProbesPerUpdate &&
    a.WorldCache.AngularFilterRadius==b.WorldCache.AngularFilterRadius && a.WorldCache.DepthLayers==b.WorldCache.DepthLayers &&
    a.Temporal.Enabled==b.Temporal.Enabled && a.Temporal.HistoryWeight==b.Temporal.HistoryWeight && a.Temporal.DepthTolerance==b.Temporal.DepthTolerance &&
    a.Temporal.NormalThreshold==b.Temporal.NormalThreshold && a.Temporal.PlaneTolerance==b.Temporal.PlaneTolerance && a.Temporal.ClipTolerance==b.Temporal.ClipTolerance && a.Temporal.MaximumFrames==b.Temporal.MaximumFrames &&
    a.ProbeVisibility.Enabled==b.ProbeVisibility.Enabled && a.ProbeVisibility.MaximumSteps==b.ProbeVisibility.MaximumSteps && a.ProbeVisibility.NormalBias==b.ProbeVisibility.NormalBias &&
    a.ScreenTrace.Enabled==b.ScreenTrace.Enabled && a.ScreenTrace.Hierarchical==b.ScreenTrace.Hierarchical && a.ScreenTrace.MaximumSteps==b.ScreenTrace.MaximumSteps && a.ScreenTrace.Thickness==b.ScreenTrace.Thickness && a.ScreenTrace.PixelStride==b.ScreenTrace.PixelStride; }
LumenSettings SanitizeLumenSettings(LumenSettings settings) noexcept
{
    const auto powerOfTwo=[](unsigned value,unsigned minimum,unsigned maximum) { unsigned result=minimum; while (result<maximum && result<value) result*=2; return result; };
    settings.HairGridSide=powerOfTwo(settings.HairGridSide,4,64);
    settings.AdaptiveProbeLevels=std::clamp(settings.AdaptiveProbeLevels,1u,MaximumLumenAdaptiveProbeLevels);
    auto& volume=settings.Translucency; volume.TileSide=powerOfTwo(volume.TileSide,32,256);
    volume.SliceCount=std::clamp(volume.SliceCount,4u,32u); volume.RaysPerProbe=powerOfTwo(volume.RaysPerProbe,16,64);
    volume.DepthExponent=std::clamp(FiniteOr(volume.DepthExponent,2.0f),1.0f,4.0f);
    settings.Radiosity.Bounces=std::clamp(settings.Radiosity.Bounces,1u,8u);
    settings.Radiosity.RaysPerTexel=powerOfTwo(settings.Radiosity.RaysPerTexel,4,64);
    settings.LightingTilesPerUpdate=std::clamp(settings.LightingTilesPerUpdate,1u,4096u);
    settings.CaptureCardsPerUpdate=std::clamp(settings.CaptureCardsPerUpdate,1u,4096u);
    settings.MaximumMeshCards=std::clamp(settings.MaximumMeshCards,1u,64u);
    settings.CardCoverage=std::clamp(FiniteOr(settings.CardCoverage,LumenSettings{}.CardCoverage),0.5f,1.0f);
    auto& global=settings.GlobalDistance; global.GridSide=powerOfTwo(global.GridSide,8,64); global.CascadeCount=std::clamp(global.CascadeCount,1u,4u);
    global.VoxelsPerUpdate=std::clamp(global.VoxelsPerUpdate,1u,65536u);
    global.MaximumSteps=std::clamp(global.MaximumSteps,1u,512u); global.CellScale=std::clamp(FiniteOr(global.CellScale,0.01f),0.001f,0.1f);
    global.NearScale=std::clamp(FiniteOr(global.NearScale,0.05f),0.0f,1.0f);
    settings.WorldCache.GridSide=std::clamp(settings.WorldCache.GridSide,2u,8u);
    settings.WorldCache.DirectionalSide=powerOfTwo(settings.WorldCache.DirectionalSide,4,32);
    auto& world=settings.WorldCache; world.ProbesPerUpdate=std::clamp(world.ProbesPerUpdate,1u,4096u); world.CascadeCount=std::clamp(world.CascadeCount,1u,4u);
    world.AngularFilterRadius=std::min(world.AngularFilterRadius,LumenWorldMaximumFilterRadius);
    world.DepthLayers=std::clamp(world.DepthLayers,1u,MaximumLumenWorldDepthLayers);
    const auto probes=world.GridSide*world.GridSide*world.GridSide*world.CascadeCount;
    while (probes*world.DepthLayers*world.DirectionalSide*world.DirectionalSide>LumenWorldRadianceMaximumRays) world.DirectionalSide/=2;
    const float largestScale=float(1u<<(world.CascadeCount-1));
    world.CellScale=std::clamp(FiniteOr(world.CellScale,0.08f),0.005f,0.2f/largestScale);
    world.BlendCells=std::clamp(FiniteOr(world.BlendCells,1.0f),0.0f,2.0f);
    settings.TileSide=powerOfTwo(settings.TileSide,16,128); settings.RaysPerProbe=powerOfTwo(settings.RaysPerProbe,16,256);
    settings.SourceResolution=std::clamp(settings.SourceResolution,16u,64u); settings.CardSide=powerOfTwo(settings.CardSide,8,32);
    for (auto& channel:settings.SkyRadiance) channel=std::clamp(FiniteOr(channel,0.0f),0.0f,MaximumLumenEnvironmentRadiance);
    auto& environment=settings.Environment;
    for (auto& channel:environment.Radiance) channel=std::clamp(FiniteOr(channel,0.0f),0.0f,MaximumLumenEnvironmentRadiance);
    for (auto& component:environment.Direction) component=FiniteOr(component,0.0f);
    const auto& axis=environment.Direction;
    const double axisLength=std::sqrt(double(axis[0])*axis[0]+double(axis[1])*axis[1]+double(axis[2])*axis[2]);
    constexpr double UnitAxisTolerance=4*std::numeric_limits<float>::epsilon();
    if (axisLength>0)
    {
        // Re-sanitizing a unit axis must not drift by an ULP and relight caches.
        if (std::abs(axisLength-1)>UnitAxisTolerance)
            for (auto& component:environment.Direction) component=float(component/axisLength);
    }
    else environment.Direction={0,1,0};
    environment.Exponent=std::clamp(FiniteOr(environment.Exponent,1.0f),1.0f,MaximumLumenEnvironmentExponent);
    settings.Temporal=SanitizeLumenTemporalSettings(settings.Temporal);
    settings.ProbeVisibility.MaximumSteps=std::clamp(settings.ProbeVisibility.MaximumSteps,1u,512u);
    settings.ProbeVisibility.NormalBias=std::clamp(FiniteOr(settings.ProbeVisibility.NormalBias,0.05f),0.0f,1000.0f);
    settings.ScreenTrace.MaximumSteps=std::clamp(settings.ScreenTrace.MaximumSteps,1u,1024u);
    settings.ScreenTrace.Thickness=std::clamp(FiniteOr(settings.ScreenTrace.Thickness,0.1f),0.0f,1000.0f);
    settings.ScreenTrace.PixelStride=std::clamp(FiniteOr(settings.ScreenTrace.PixelStride,1.0f),1.0f,4.0f);
    return settings;
}
bool operator==(const GlobalIlluminationSettings& a, const GlobalIlluminationSettings& b) noexcept
{
    return a.Mode == b.Mode && a.Intensity == b.Intensity && a.MaximumDistance == b.MaximumDistance &&
        a.Thickness == b.Thickness && a.NormalBias == b.NormalBias && a.EdgeFadePixels == b.EdgeFadePixels &&
        a.HistoryWeight == b.HistoryWeight && a.DepthTolerance == b.DepthTolerance && a.NormalThreshold == b.NormalThreshold &&
        a.RayCount == b.RayCount && a.MaximumSteps == b.MaximumSteps && a.Probes == b.Probes && a.Lightmaps == b.Lightmaps && a.RayTracing == b.RayTracing && a.Voxels == b.Voxels && a.Lumen == b.Lumen;
}
GlobalIlluminationSettings SanitizeGlobalIlluminationSettings(GlobalIlluminationSettings settings) noexcept
{
    const GlobalIlluminationSettings defaults;
    if (settings.Mode != GlobalIlluminationMode::ScreenSpace && settings.Mode != GlobalIlluminationMode::BakedProbes && settings.Mode != GlobalIlluminationMode::Lightmaps && settings.Mode != GlobalIlluminationMode::RayTraced && settings.Mode != GlobalIlluminationMode::Voxel && settings.Mode != GlobalIlluminationMode::Lumen) settings.Mode = GlobalIlluminationMode::None;
    settings.Probes = SanitizeLightProbeSettings(settings.Probes);
    settings.Lightmaps = SanitizeLightmapSettings(settings.Lightmaps);
    settings.RayTracing = SanitizeRayTracedGISettings(settings.RayTracing);
    settings.Voxels = SanitizeVoxelGISettings(settings.Voxels);
    settings.Lumen=SanitizeLumenSettings(settings.Lumen);
    constexpr float MaximumIntensity = 16.0f, MinimumDistance = 0.001f, MaximumDistance = 1000000.0f;
    constexpr float MinimumThickness = 0.0001f, MaximumThickness = 10000.0f, MaximumBias = 1000.0f;
    constexpr float MaximumEdgeFade = 256.0f, MaximumHistoryWeight = 0.98f, MaximumDepthTolerance = 0.25f;
    constexpr std::uint32_t MaximumRays = 32, MinimumSteps = 4, MaximumSteps = 256;
    settings.Intensity = std::clamp(FiniteOr(settings.Intensity, defaults.Intensity), 0.0f, MaximumIntensity);
    settings.MaximumDistance = std::clamp(FiniteOr(settings.MaximumDistance, defaults.MaximumDistance), MinimumDistance, MaximumDistance);
    settings.Thickness = std::clamp(FiniteOr(settings.Thickness, defaults.Thickness), MinimumThickness, MaximumThickness);
    settings.NormalBias = std::clamp(FiniteOr(settings.NormalBias, defaults.NormalBias), 0.0f, MaximumBias);
    settings.EdgeFadePixels = std::clamp(FiniteOr(settings.EdgeFadePixels, defaults.EdgeFadePixels), 0.0f, MaximumEdgeFade);
    settings.HistoryWeight = std::clamp(FiniteOr(settings.HistoryWeight, defaults.HistoryWeight), 0.0f, MaximumHistoryWeight);
    settings.DepthTolerance = std::clamp(FiniteOr(settings.DepthTolerance, defaults.DepthTolerance), 0.0f, MaximumDepthTolerance);
    settings.NormalThreshold = std::clamp(FiniteOr(settings.NormalThreshold, defaults.NormalThreshold), 0.0f, settings.Mode==GlobalIlluminationMode::Lumen?0.99f:1.0f);
    settings.RayCount = std::clamp(settings.RayCount, 1u, MaximumRays);
    settings.MaximumSteps = std::clamp(settings.MaximumSteps, MinimumSteps, MaximumSteps);
    return settings;
}

bool operator==(const Settings& a, const Settings& b) noexcept
{
    return a.Pipeline == b.Pipeline && a.PathTracing == b.PathTracing && a.Hybrid == b.Hybrid && a.Subsurface == b.Subsurface && a.Hair == b.Hair && a.Parallax == b.Parallax && a.Displacement == b.Displacement && a.Tessellation == b.Tessellation && a.VirtualGeometry == b.VirtualGeometry && a.Nanite == b.Nanite && a.MeshShaders == b.MeshShaders && a.MeshDispatch == b.MeshDispatch && a.Shading == b.Shading && a.TextureMapping == b.TextureMapping &&
        a.AnisotropicFiltering == b.AnisotropicFiltering && a.NormalMapping == b.NormalMapping && a.AlphaTesting == b.AlphaTesting &&
        a.AlphaBlending == b.AlphaBlending && a.Transparency == b.Transparency && a.Refraction == b.Refraction && a.OIT == b.OIT && a.FrustumCulling == b.FrustumCulling &&
        a.BackFaceCulling == b.BackFaceCulling && a.Roughness == b.Roughness &&
        a.Metallic == b.Metallic && a.AreaLights == b.AreaLights && a.ToonBands == b.ToonBands && a.PostProcessing == b.PostProcessing &&
        a.AmbientOcclusion == b.AmbientOcclusion && a.Shadows == b.Shadows && a.RayShadows == b.RayShadows && a.Cascades == b.Cascades && a.VirtualShadows == b.VirtualShadows &&
        a.SampleCount == b.SampleCount && a.Reflections == b.Reflections && a.Lod == b.Lod && a.Instancing == b.Instancing && a.GPUDriven == b.GPUDriven &&
        a.ShadowFilter == b.ShadowFilter && a.VolumetricFog == b.VolumetricFog && a.VolumetricLighting == b.VolumetricLighting &&
        a.LightShafts == b.LightShafts && a.ContactShadows == b.ContactShadows && a.GlobalIllumination == b.GlobalIllumination;
}

bool operator!=(const Settings& a, const Settings& b) noexcept { return !(a == b); }

Settings SanitizeSettings(Settings settings) noexcept
{
    if (unsigned(settings.MeshDispatch)>unsigned(MeshShaderDispatchMode::Task)) settings.MeshDispatch=MeshShaderDispatchMode::Automatic;
    if (static_cast<std::uint32_t>(settings.Pipeline) >
        static_cast<std::uint32_t>(RenderingPipeline::Hybrid)) settings.Pipeline = RenderingPipeline::Forward;
    if (static_cast<std::uint32_t>(settings.Shading) >
        static_cast<std::uint32_t>(ShadingModel::Toon)) settings.Shading = ShadingModel::Phong;
    if (settings.Shadows != ShadowRepresentation::Cascaded && settings.Shadows != ShadowRepresentation::Virtual && settings.Shadows != ShadowRepresentation::RayTraced) settings.Shadows = ShadowRepresentation::SingleMap;
    settings.RayShadows=SanitizeRayTracedShadowSettings(settings.RayShadows);
    settings.Cascades = SanitizeCascadedShadowSettings(settings.Cascades);
    settings.VirtualShadows = SanitizeVirtualShadowSettings(settings.VirtualShadows);
    settings.ContactShadows = SanitizeContactShadowSettings(settings.ContactShadows);
    settings.ShadowFilter = SanitizeShadowFilterSettings(settings.ShadowFilter);
    settings.SampleCount = SanitizeSampleCount(settings.SampleCount);
    settings.Roughness = std::clamp(FiniteOr(settings.Roughness, 0.5f), MinimumRoughness, 1.0f);
    settings.Metallic = std::clamp(FiniteOr(settings.Metallic, 0.0f), 0.0f, 1.0f);
    settings.AreaLights = SanitizeAreaLightSettings(settings.AreaLights);
    settings.ToonBands = std::round(std::clamp(FiniteOr(settings.ToonBands, 4.0f),
        MinimumToonBands, MaximumToonBands));
    settings.PostProcessing = SanitizePostProcessSettings(settings.PostProcessing);
    settings.AmbientOcclusion = SanitizeAmbientOcclusionSettings(settings.AmbientOcclusion);
    settings.Reflections = SanitizeReflectionSettings(settings.Reflections);
    settings.GlobalIllumination = SanitizeGlobalIlluminationSettings(settings.GlobalIllumination);
    settings.PathTracing = SanitizePathTracingSettings(settings.PathTracing);
    settings.Subsurface = SanitizeSubsurfaceScatteringSettings(settings.Subsurface);
    settings.Hair = SanitizeHairFurSettings(settings.Hair);
    settings.Parallax = SanitizeParallaxOcclusionSettings(settings.Parallax);
    settings.Displacement=SanitizeDisplacementMappingSettings(settings.Displacement);
    settings.Tessellation=SanitizeTessellationSettings(settings.Tessellation);
    settings.VirtualGeometry=SanitizeVirtualGeometrySettings(settings.VirtualGeometry);
    settings.Nanite=SanitizeNaniteSettings(settings.Nanite);
    settings.Lod = SanitizeLodSettings(settings.Lod);
    settings.VolumetricFog = SanitizeVolumetricFogSettings(settings.VolumetricFog);
    settings.VolumetricLighting = SanitizeVolumetricLightingSettings(settings.VolumetricLighting);
    settings.LightShafts = SanitizeLightShaftSettings(settings.LightShafts);
    settings.Transparency = SanitizeTransparencySettings(settings.Transparency);
    settings.Refraction = SanitizeRefractionSettings(settings.Refraction);
    settings.OIT = SanitizeOrderIndependentTransparencySettings(settings.OIT);
    return settings;
}

Backend BackendFromId(std::string_view id) noexcept
{
    if (id == "renderer.dx11") return Backend::DX11;
    if (id == "renderer.dx12") return Backend::DX12;
    if (id == "renderer.metal") return Backend::Metal;
    if (id == "renderer.opengles") return Backend::OpenGLES;
    if (id == "renderer.vulkan") return Backend::Vulkan;
    if (id == "renderer.dxr") return Backend::DXR;
    if (id == "rasterizer.cpu") return Backend::CPU;
    if (id == "rasterizer.cuda") return Backend::CUDA;
    if (id == "rasterizer.compute") return Backend::Compute;
    if (id == "rasterizer.metal") return Backend::MetalCompute;
    return Backend::Unknown;
}

bool IsSoftwareRaster(Backend backend) noexcept
{
    return backend == Backend::CPU || backend == Backend::CUDA || backend == Backend::Compute || backend == Backend::MetalCompute;
}

bool IsNativeRaster(Backend backend) noexcept
{
    return backend == Backend::DX11 || backend == Backend::DX12 ||
        backend == Backend::Metal || backend == Backend::OpenGLES || backend == Backend::Vulkan;
}

const std::vector<Definition>& Catalog()
{
    using T = Technique;
    using A = Availability;
    using R = Resource;
    static const std::vector<Definition> definitions = {
        {T::Forward, "Forward rendering", Group::Pipeline, A::Implemented, Surface, Color | Depth, PipelineSlot, "Opaque lighting, then sorted transparent objects."},
        {T::Deferred, "Deferred rendering", Group::Pipeline, A::Implemented, Surface, Mask(R::GBuffer) | HDR | Depth | N, PipelineSlot, "Five-target material geometry and lighting resolve; device limits gate availability."},
        {T::Flat, "Flat shading", Group::Shading, A::Implemented, Surface, 0, ShadingSlot, "Lighting with each triangle's geometric normal; normal mapping is inactive."},
        {T::Gouraud, "Gouraud shading", Group::Shading, A::Implemented, Surface, 0, ShadingSlot, "Vertex lighting interpolated across the triangle; no fragment normal mapping."},
        {T::Phong, "Phong shading", Group::Shading, A::Implemented, Surface, 0, ShadingSlot, "Interpolated normals with per-fragment Phong lighting."},
        {T::PBR, "PBR", Group::Shading, A::Implemented, Surface, 0, ShadingSlot, "GGX microfacet shading with scalar roughness and metallic controls."},
        {T::Toon, "Toon / cel shading", Group::Shading, A::Implemented, Surface, 0, ShadingSlot, "Quantized direct diffuse lighting with configurable bands."},
        {T::TextureMapping, "Texture mapping", Group::Materials, A::Implemented, M | Mask(R::DiffuseTexture), 0, Composable, "Existing demo UVs and diffuse images."},
        {T::NormalMapping, "Normal mapping", Group::Materials, A::Implemented, G | M | Mask(R::NormalTexture), 0, Composable, "Tangent-space normals; inactive for geometric Flat and vertex-lit Gouraud shading."},
        {T::AlphaTesting, "Alpha testing", Group::Materials, A::Implemented, M, 0, Composable, "Material alpha and optional opacity map; depth/shadow passes use the same cutoff."},
        {T::AlphaBlending, "Alpha blending", Group::Transparency, A::Implemented, M | Color | Depth, 0, TransparencySlot, "Back-to-front straight-alpha blending; opaque depth remains intact."},
        {T::BasicMaterials, "Basic material system", Group::Materials, A::Implemented, G, M, Composable, "Existing OBJ/MTL colors, opacity and image bindings."},
        {T::ShadowMapping, "Shadow mapping", Group::Shadows, A::Implemented, G | M | L, Shadow, ShadowSlot, "Single selected shadow-casting light and depth map."},
        {T::CascadedShadowMaps, "Cascaded shadow maps", Group::Shadows, A::Implemented, G | M | C | L, Shadow, ShadowSlot, "Directional cascades with stable fitting and overlap blending on all five native APIs; Metal awaits Apple compilation and GPU validation."},
        {T::BasicSoftShadows, "Basic soft shadows", Group::ShadowFilter, A::Implemented, Shadow, 0, ShadowFilterSlot, "Existing fixed-radius PCF; this is not contact-hardening PCSS."},
        {T::SSAO, "SSAO", Group::AmbientOcclusion, A::Implemented, Screen | Mask(R::GBuffer) | HDR, Mask(R::AmbientVisibility), AOSlot, "Hemisphere visibility, depth/normal bilateral filtering and ambient-only HDR composition."},
        {T::EnvironmentReflections, "Environment map reflections", Group::Reflections, A::Implemented, Surface | Screen | HDR | Mask(R::GBuffer), Mask(R::Environment) | Mask(R::ReflectionRadiance), ReflectionSlot, "All five native APIs: six-face scene capture, latitude-longitude conversion, mip-filtered 2D lookup and Fresnel composition; Metal awaits Apple compilation and GPU validation."},
        {T::CubemapReflections, "Cubemap reflections", Group::Reflections, A::Implemented, Surface | Screen | HDR | Mask(R::GBuffer), Mask(R::Environment) | Mask(R::ReflectionRadiance), ReflectionSlot, "All five native APIs: six-face HDR scene capture, mip-filtered cube lookup and Fresnel composition; Metal awaits Apple compilation and GPU validation."},
        {T::PlanarReflections, "Planar reflections", Group::Reflections, A::Implemented, Surface | Screen | HDR | Mask(R::GBuffer), Mask(R::PlanarRadiance) | Mask(R::ReflectionRadiance), ReflectionSlot, "All five native APIs: mirrored HDR capture with oblique clipping, area-filtered mips and Fresnel composition; Metal awaits Apple compilation and GPU validation."},
        {T::MSAA, "MSAA", Group::AntiAliasing, A::Implemented, Color | Depth, 0, 0, "Geometry coverage and per-sample depth, sample-frequency deferred lighting and explicit resolves; independent of display FXAA. All five native APIs; Metal awaits Apple compilation and GPU validation."},
        {T::FXAA, "FXAA", Group::AntiAliasing, A::Implemented, Color, 0, AASlot, "Display-domain edge detection and directional filtering; preserves source alpha."},
        {T::ToneMapping, "Tone mapping", Group::PostProcessing, A::Implemented, HDR, Color, Composable, "Exclusive Reinhard or ACES filmic fit; exposure and RGBA16F-to-display resolve."},
        {T::Bloom, "Bloom", Group::PostProcessing, A::Implemented, HDR, 0, Composable, "Half-resolution HDR extraction, separable nine-tap blur, upsample and composite before tone resolve."},
        {T::ColorGrading, "Color grading", Group::PostProcessing, A::Implemented, HDR, 0, Composable, "Scene-linear Rec.709 saturation and contrast around 18% gray; no external LUT."},
        {T::DepthOfField, "Depth of field", Group::PostProcessing, A::Implemented, Depth | HDR | C, HDR, Composable, "Thin-lens foreground/background disk gathering before sharp transparency, bloom and tone mapping."},
        {T::MotionBlur, "Motion blur", Group::PostProcessing, A::Implemented, HDR | Depth | Velocity, HDR, Composable, "Rigid object/camera velocity, tile maxima and depth-aware shutter reconstruction before sharp transparency."},
        {T::FrustumCulling, "Frustum culling", Group::Geometry, A::Implemented, G | C, 0, Composable, "Conservative object bounds against clip planes."},
        {T::BackFaceCulling, "Back-face culling", Group::Geometry, A::Implemented, G, 0, Composable, "Raster winding state; two-sided foliage may require disabling it."},
        {T::LOD, "Level of detail", Group::Geometry, A::Implemented, G | C, 0, GeometrySlot, "Runtime quadric index levels, conservative multi-view pixel error and hysteresis; all passes share geometry and level changes invalidate per-mesh velocity history."},
        {T::Instancing, "Instanced rendering", Group::Geometry, A::Implemented, G, 0, Composable, "Native indexed instance batches with independent transforms/history on all five raster APIs; Metal awaits Apple validation."},
        {T::ForwardPlus, "Forward+", Group::Pipeline, A::Implemented, Surface, Color | Depth | Mask(R::LightLists), PipelineSlot, "Opaque depth, conservative tile assignment and complete scene-light storage; Metal awaits Apple compilation and execution."},
        {T::ClusteredForward, "Clustered forward", Group::Pipeline, A::Implemented, Surface, Color | Depth | Mask(R::LightLists), PipelineSlot, "Dense 3D light lists cover opaque and transparent fragments without an opaque depth prepass; Metal awaits Apple compilation and execution."},
        {T::TiledDeferred, "Tiled deferred", Group::Pipeline, A::Implemented, Surface, HDR | Depth | N | Mask(R::GBuffer) | Mask(R::LightLists), PipelineSlot, "G-buffer depth drives opaque tile assignment; per-sample lighting and separate transparent lists retain scene-order lights. Metal awaits Apple compilation and execution."},
        {T::VisibilityBuffer, "GPU Driven Visibility Buffer", Group::Pipeline, A::Implemented, Surface, HDR | Depth | N | Mask(R::GBuffer) | Mask(R::VisibilitySamples), PipelineSlot, "Resident mesh clusters, GPU instance/cluster visibility, hierarchical depth, indirect hardware bins and tiled visibility/material reconstruction. Metal awaits Apple validation."},
        {T::Hybrid, "Ray Tracing Hybrid", Group::Pipeline, A::Implemented, Surface | AS, HDR | Depth | N | Mask(R::GBuffer) | Mask(R::ReflectionRadiance), PipelineSlot, "Raster primary visibility, stochastic screen intersections and confidence-weighted scene-ray fallback. Shared GGX sampling and spatial reconstruction; Metal awaits Apple validation."},
        {T::GPUDriven, "GPU-driven rendering", Group::Geometry, A::Implemented, G | C, Mask(R::IndirectCommands), Composable, "GPU bounds culling, stable instance compaction and native indexed indirect drawing. CPU material batching and LOD selection; Metal awaits Apple validation."},
        {T::PathTracing, "Ray Tracing Full", Group::Pipeline, A::Implemented, Surface | AS | History, HDR | Depth, PipelineSlot, "Camera paths, diffuse/GGX mixture sampling, explicit light sampling and transactional FP32 accumulation. Opaque/cutout surfaces, stochastic thin opacity, dielectric reflection/refraction and constant-environment transport; Metal device validation pending."},
        {T::SubsurfaceScattering, "Subsurface scattering", Group::SpecializedShading, A::Implemented, HDR | Screen | C | Mask(R::GBuffer) | L, HDR | Mask(R::DiffuseRadiance), Composable, "Normalized RGB diffusion of separated direct diffuse, guided by depth/normals, with coverage-aware MSAA and HDR replacement. Transmission needs generated thickness; Metal awaits Apple validation."},
        {T::HairFur, "Hair and fur", Group::SpecializedShading, A::Implemented, G | M | L | Mask(R::HairStrands), HDR | Depth, Composable, "Area-seeded curved ribbons, normalized cylindrical scattering, shadow/contact reception, capture and motion replay. The dedicated Fur Cube demo exposes procedural growth and fiber material controls. Lumen supports oriented voxels and low-frequency indirect reception; mesh-only scene rays omit fibers. Metal awaits Apple validation."},
        {T::NPR, "Non-photorealistic rendering", Group::SpecializedShading, A::Implemented, Color, 0, Composable, "XDoG ink contours, analytic tonal crosshatching and posterized color after tone mapping; independent of surface shading."},
        {T::AreaLights, "Physically based area lights", Group::SpecializedShading, A::Implemented, Surface | HDR, 0, Composable, "Independent native GGX rectangle quadrature with radiance/power units; PBR/HDR required. Raster visibility follows the selected shadow method; CPU baking traces each sample. Metal device validation pending."},
        {T::BakedGI, "Baked global illumination", Group::GlobalIllumination, A::Implemented, G | M | L, Mask(R::BakedLighting), Composable, "Material-aware scene rays bake one diffuse bounce; native SH and distance-moment projection is cached until scene lighting changes. Metal device validation pending."},
        {T::Lightmaps, "Lightmaps", Group::GlobalIllumination, A::Implemented, Mask(R::LightmapUV) | Mask(R::BakedLighting) | Screen | C | HDR | Mask(R::GBuffer), Mask(R::IndirectRadiance) | HDR, GISlot, "Background one-bounce surface baking, per-instance/LOD UV2 charts and chart-isolated gutters; depth-equal native receivers preserve material masking and MSAA coverage. Metal device validation pending."},
        {T::LightProbes, "Light probes", Group::GlobalIllumination, A::Implemented, Mask(R::BakedLighting) | Screen | C | HDR | Mask(R::GBuffer), Mask(R::Probes) | Mask(R::IndirectRadiance) | HDR, GISlot, "L2 SH irradiance with eight-neighbor interpolation and octahedral distance-moment visibility; independent of camera history. Metal device validation pending."},
        {T::SSGI, "Screen-space GI", Group::GlobalIllumination, A::Implemented, Screen | HDR | C | Velocity | Mask(Resource::GBuffer) | Mask(Resource::GlobalIlluminationHistory), Mask(Resource::IndirectRadiance) | HDR, GISlot, "Cosine hemisphere diffuse transport, bilateral filtering and depth/normal-validated private history; missing rays retain ambient fallback."},
        {T::VoxelGI, "Voxel GI", Group::GlobalIllumination, A::Implemented, Surface | Screen | Mask(R::GBuffer) | Mask(R::Voxels) | HDR, HDR, GISlot, "Asynchronous world-space voxel coverage, directional direct-diffuse radiance mips and native hemisphere cone tracing. Metal device validation pending."},
        {T::RayTracedGI, "Ray-traced GI", Group::GlobalIllumination, A::Implemented, AS | Surface | Screen | Mask(R::GBuffer) | HDR, HDR, GISlot, "Bounded cosine-sampled diffuse paths, material-aware direct visibility, and spatial bilateral reconstruction. Metal compilation and execution pending."},
        {T::Lumen, "Lumen algorithms", Group::GlobalIllumination, A::Implemented, Mask(R::SurfaceRadianceCache) | Screen | Mask(R::GBuffer) | HDR, HDR, GISlot, "Independent multilevel adaptive screen probes with directional environment lighting, hierarchical screen tracing, global distance clipmaps with mesh refinement or optional native DX12/Vulkan/Metal triangle queries, surface-cache transport, scrolling world-radiance reuse, visibility-filtered gathering and diffuse temporal reconstruction. Budgeted asynchronous global-field slices publish complete generations. Capture-resolution card refinement, Jacobi surface radiosity and independent GGX surface-cache reflections are integrated; a bounded frustum SH volume supplies translucent diffuse lighting, cylindrical hair indirect lighting and low-frequency unresolved glass reflections. Independent oriented hair voxels attenuate transport and add an isotropic single-scattering source. No Unreal dependency."},
        {T::PCF, "PCF", Group::ShadowFilter, A::Implemented, Shadow, 0, ShadowFilterSlot, "Independent native material/deferred PCF receivers with selectable 3x3 through 9x9 bilinear comparison kernels. Advanced consumers keep private filtering; software retains its fixed kernel."},
        {T::PCSS, "PCSS", Group::ShadowFilter, A::Implemented, Shadow | L, 0, ShadowFilterSlot, "Independent native surface/volume receivers use linear blocker search and emitter-sized contact-hardening PCF. Metal compilation and device validation pending."},
        {T::VSM, "Variance shadow maps", Group::ShadowFilter, A::Implemented, G | M | L | Shadow, Mask(R::ShadowMoments), ShadowFilterSlot, "Native linear-depth moment generation, Gaussian prefilter and variance visibility for surface/volume receivers. Composes with CSM; replaces PCF/PCSS. Metal compilation and device validation pending."},
        {T::VirtualShadowMaps, "Virtual shadow maps", Group::Shadows, A::Implemented, G | M | L | C, Shadow, ShadowSlot, "Current-camera page demand, immutable physical depth atlases, coarse fallback, caster invalidation and hard/PCF receivers. Fiber casting and virtual PCSS/moments remain pending; Metal awaits Apple validation."},
        {T::ContactShadows, "Contact shadows", Group::Shadows, A::Implemented, Depth | C | L, Mask(R::ContactVisibility), Composable, "Independent native short screen-space rays supplement the primary shadow light, with per-sample visibility and dedicated depth production. Metal compilation and execution still await Apple validation."},
        {T::RayTracedShadows, "Ray-traced shadows", Group::Shadows, A::Implemented, AS | G | L, 0, ShadowSlot, "World-space any-hit BVH visibility with bilinear alpha cutouts; finite emitter rays stop at the light."},
        {T::HBAO, "HBAO", Group::AmbientOcclusion, A::Implemented, Screen | Mask(R::GBuffer) | HDR, Mask(R::AmbientVisibility), AOSlot, "Distance-weighted horizon search above the tangent plane, bilateral filters and ambient-only composition."},
        {T::SAO, "Scalable ambient obscurance", Group::AmbientOcclusion, A::Implemented, Depth | C | Mask(R::GBuffer) | HDR, Mask(R::AmbientVisibility) | Mask(R::LinearDepthHierarchy), AOSlot, "R32F depth mip hierarchy, depth-derived normals, fixed spiral samples and ambient-only composition."},
        {T::GTAO, "GTAO", Group::AmbientOcclusion, A::Implemented, Screen | Mask(R::GBuffer) | HDR, Mask(R::AmbientVisibility), AOSlot, "Cosine-weighted signed horizon integration with projected normals and spatial reconstruction; temporal denoising is optional."},
        {T::RayTracedAO, "Ray-traced AO", Group::AmbientOcclusion, A::Implemented, AS | Screen | Mask(R::GBuffer) | HDR, Mask(R::AmbientVisibility), AOSlot, "Native finite-radius cosine rays traverse an immutable scene BVH, including offscreen cutout geometry; bilateral reconstruction attenuates ambient only. Metal compilation and device validation pending."},
        {T::SSR, "Screen-space reflections", Group::Reflections, A::Implemented, Screen | HDR | Mask(R::GBuffer), Mask(R::ReflectionRadiance), ReflectionSlot, "Perspective-correct spatial depth traversal and Fresnel-weighted HDR reflection; Native raster implementations; Apple compilation and execution pending."},
        {T::RayTracedReflections, "Ray-traced reflections", Group::Reflections, A::Implemented, AS | Screen | L | Mask(R::GBuffer) | HDR, Mask(R::ReflectionRadiance), ReflectionSlot, "Scene BVH closest hits, material-aware direct lighting, GGX visible-normal sampling and spatial reconstruction. Metal compilation and execution pending."},
        {T::Refraction, "Refraction", Group::Transparency, A::Implemented, G | C | M | Depth | HDR, HDR, Composable, "Material-scoped nearest-layer dielectric slab with Snell/Fresnel, Beer absorption and screen-space background tracing. Independent of alpha/OIT objects; single-sample capture."},
        {T::OIT, "Order-independent transparency", Group::Transparency, A::Implemented, Surface | Depth | HDR, HDR | Mask(R::TransparencyStorage), TransparencySlot, "Front depth peeling with 1--32 distinct layers, canonical coplanar order and an overflow probe. Single-sample HDR; Metal native execution pending."},
        {T::WeightedBlendedOIT, "Weighted blended OIT", Group::Transparency, A::Implemented, M | Depth | HDR, HDR | Mask(R::TransparencyStorage), TransparencySlot, "Native raster APIs: float accumulation/revealage and per-sample resolve. Approximate inter-layer color with exact background coverage; Metal native validation pending."},
        {T::Tessellation, "PN triangle tessellation", Group::Geometry, A::Implemented, G, Mask(R::TessellatedGeometry), Composable, "Three-stage native compute tessellation with cubic PN positions, quadratic normals, shared material boundaries and coherent scene publication. PN supplies topology before optional height displacement. Uniform integer density; fixed-function patch stages and adaptive factors remain TODO. Metal compilation and execution are unverified."},
        {T::DisplacementMapping, "Displacement mapping", Group::Materials, A::Implemented, G | Mask(R::HeightMap), Mask(R::DisplacedGeometry), Composable, "Native two-stage displacement with shared material-section context, coherent readback, atomic scene publication and original-geometry restoration. Shadows, captures and scene queries share the resulting geometry. Authored height textures remain TODO; existing demos use an analytic height field. Metal compilation and execution are unverified."},
        {T::ParallaxOcclusionMapping, "Parallax occlusion mapping", Group::Materials, A::Implemented, G | M | C | Mask(R::HeightMap), Mask(R::GBuffer), Composable, "Adaptive height rays, refined first crossings and explicit material gradients on opaque Deferred/Tiled material capture. Base silhouette/depth retained; authored height inputs and view-capture receivers remain TODO. Metal awaits Apple validation."},
        {T::MeshShaders, "Mesh shaders", Group::Geometry, A::Implemented, Surface, Mask(R::VisibilitySamples), Composable, "Native DX12/Vulkan/Metal mesh workgroups reuse indexed vertices, cull clusters and compact primitives. Direct mesh or task/amplification dispatch uses draw size and device support. Nanite consumes GPU hardware bins; unsupported APIs/devices retain the request. Metal compilation and execution are unverified."},
        {T::VirtualGeometry, "Virtual geometry", Group::Geometry, A::Implemented, G | C | Mask(R::MeshClusters), Mask(R::VirtualizedGeometry), GeometrySlot, "Native hierarchical selection, complete resident cuts and bounded page streaming; material sections, instances and all raster/query views share a lossless decoded cut. Whole-mesh LOD is exclusive; PN/displacement remain source preparation. Apple validation remains pending."},
        {T::Nanite, "Nanite algorithms", Group::Geometry, A::Implemented, G | C | Mask(R::MeshClusters), Mask(R::VirtualizedGeometry), GeometrySlot, "Independent cluster DAG, compressed streaming, GPU selection/decoding, two-phase HZB culling, indirect material bins and small-triangle compute coverage with a native frame bridge into visibility/material resolve. GPU world-table assembly connects decoded streams to the frame producer. Scene materials/instances share resident cuts across camera, shadow and capture views. Main-frame two-phase HZB decisions gate both hardware and compute coverage. Bounded asynchronous page-copy and selection/decode jobs retain published cuts until completion. Asynchronous bootstrap preserves visible geometry until a complete generation is ready. GPU counts drive material, instance and hardware-bin scheduling. Scene residency budgets and bounded CPU family preparation coordinate demand paging; auxiliary feedback remains CPU-visible. No external engine dependency; Metal compilation and execution are unverified."},
        {T::TAA, "Temporal anti-aliasing", Group::AntiAliasing, A::Implemented, HDR | Temporal, HDR, AASlot, "Opaque Halton sampling, physical velocity reprojection, neighborhood clipping and depth-history rejection on five native APIs; Metal native validation pending."},
        {T::TSR, "Temporal super resolution", Group::AntiAliasing, A::Implemented, HDR | Depth | Velocity | C | Mask(Resource::OpaqueHDRSnapshot) | Mask(Resource::ReconstructionHistory), Mask(Resource::PresentationHDR) | Mask(Resource::PresentationDepth) | Mask(Resource::PresentationVelocity), AASlot, "Independent HDR reconstruction after transparency, with dual grids, reactive rejection and success-only history on five native APIs; Metal native validation pending."},
        {T::DLSS, "DLSS", Group::AntiAliasing, A::OutOfScope, HDR | Temporal, 0, AASlot, "Outside the project scope; NVIDIA SDK integration is not planned."},
        {T::FSR, "FSR", Group::AntiAliasing, A::OutOfScope, HDR | Temporal, 0, AASlot, "Outside the project scope; AMD FSR SDK integration is not planned."},
        {T::XeSS, "XeSS", Group::AntiAliasing, A::OutOfScope, HDR | Temporal, 0, AASlot, "Outside the project scope; Intel SDK integration is not planned."},
        {T::VolumetricFog, "Volumetric fog", Group::Volumetrics, A::Implemented, Depth | C | HDR, HDR | Mask(R::FogVolume), MediumSlot, "Integrated height medium with ambient radiance, opaque composition and per-fragment transparent lookup on all five native raster APIs. Metal execution pending on Apple hardware."},
        {T::VolumetricLighting, "Volumetric lighting", Group::Volumetrics, A::Implemented, Depth | C | L | HDR, HDR | Mask(R::FogVolume), ScatteringSlot, "Native selected-medium single scattering with scene lights, primary shadow visibility, Henyey-Greenstein phase and prefix transport. Metal compilation and execution require Apple validation."},
        {T::VolumetricClouds, "Volumetric clouds", Group::Volumetrics, A::Implemented, Depth | C | HDR, HDR | Mask(R::FogVolume), MediumSlot, "Procedural weather, height-shaped value-noise octaves and cellular erosion with camera/light-path extinction on five native APIs. Metal shader compilation and execution require Apple validation."},
        {T::ParticipatingMedia, "Participating media", Group::Volumetrics, A::Implemented, Depth | C | HDR, HDR | Mask(R::FogVolume), MediumSlot, "Bounded RGB absorption, scattering and emission with camera/light-path extinction and associative transport on five native APIs. Metal execution requires Apple validation."},
        {T::LightShafts, "Light shafts", Group::Volumetrics, A::Implemented, Depth | C | L | HDR, HDR, ScatteringSlot, "Depth-occluded projected light disk and normalized radial scattering on all five native APIs. Alternative to volume lighting; Metal native validation pending."},
        {T::ScreenSpaceDistortion, "Screen-space distortion", Group::PostProcessing, A::Implemented, Color, 0, Composable, "Aspect-correct radial color pull-map before FXAA; bilinear linear-light sampling with edge clamp. No depth or history."},
        {T::LensFlare, "Lens flare", Group::PostProcessing, A::Implemented, HDR, 0, Composable, "Visible HDR highlights generate mirrored ghosts and an aspect-correct halo, blurred and composed before bloom. No external texture."},
        {T::FilmGrain, "Film grain", Group::PostProcessing, A::Implemented, Color, 0, Composable, "Static screen-space grain with explicit seed and display-RGB amplitude; preserves alpha."},
        {T::Vignette, "Vignette", Group::PostProcessing, A::Implemented, Color, 0, Composable, "Analytic radial attenuation in linear color; preserves alpha."},
        {T::TemporalEffects, "Temporal effects", Group::PostProcessing, A::Implemented, HDR, HDR | Mask(R::ExposureHistory), Composable, "Global log-average metering and asymmetric exponential EV adaptation, with transactional FP32 history before HDR post effects. Metal awaits Apple validation."}
    };
    return definitions;
}

const Definition& Describe(Technique technique)
{
    const auto& catalog = Catalog();
    const auto found = std::find_if(catalog.begin(), catalog.end(),
        [technique](const Definition& definition) { return definition.Id == technique; });
    if (found == catalog.end()) throw std::out_of_range("Unknown rendering technique");
    return *found;
}

const char* GroupName(Group group) noexcept
{
    switch (group)
    {
    case Group::Pipeline: return "Rendering pipelines";
    case Group::Shading: return "Surface shading";
    case Group::Materials: return "Textures and materials";
    case Group::Shadows: return "Shadow representation";
    case Group::ShadowFilter: return "Shadow filtering";
    case Group::AmbientOcclusion: return "Ambient occlusion";
    case Group::Reflections: return "Reflections";
    case Group::Transparency: return "Transparency";
    case Group::Geometry: return "Geometry and visibility";
    case Group::AntiAliasing: return "AA and reconstruction";
    case Group::GlobalIllumination: return "Global illumination";
    case Group::Volumetrics: return "Volumetric rendering";
    case Group::PostProcessing: return "Post-processing";
    case Group::SpecializedShading: return "Specialized shading";
    }
    return "Unknown";
}

const char* AvailabilityName(Availability availability) noexcept
{
    switch (availability)
    {
    case Availability::Implemented: return "Implemented";
    case Availability::Experimental: return "In development";
    case Availability::Planned: return "Planned";
    case Availability::MissingResource: return "TODO: asset";
    case Availability::OutOfScope: return "Out of scope";
    case Availability::UnsupportedBackend: return "Unavailable on backend";
    case Availability::SceneLimit: return "Scene exceeds device budget";
    }
    return "Unknown";
}

Support QuerySupport(Technique technique, Backend backend)
{
    const Definition& definition = Describe(technique);
    if (definition.State == Availability::OutOfScope)
        return {definition.State, definition.Explanation};
    if (backend == Backend::Unknown)
        return {Availability::UnsupportedBackend, "Unknown backend"};
    if (technique == Technique::MeshShaders && backend != Backend::DX12 &&
        backend != Backend::Vulkan && backend != Backend::Metal)
        return {Availability::UnsupportedBackend, "This API does not expose native mesh shader stages."};
    if (IsNativeRaster(backend))
        return {definition.State, definition.Explanation};

    // Software rasterizers share the basic scene contract. Native render-target,
    // compute and history effects are enabled only by the raster APIs above.
    // Keep this list explicit so adding a catalog entry never advertises an
    // unimplemented software or DXR path by default.
    switch (technique)
    {
    case Technique::Phong:
    case Technique::TextureMapping:
    case Technique::BasicMaterials:
        return {definition.State, definition.Explanation};
    case Technique::SubsurfaceScattering:
        if (IsSoftwareRaster(backend)) return {definition.State, "Per-channel diffuse scattering on the Jade material."};
        return {Availability::UnsupportedBackend, "This effect requires raster materials."};
    case Technique::NormalMapping:
    case Technique::Lightmaps:
        if (IsSoftwareRaster(backend)) return {definition.State, definition.Explanation};
        return {Availability::UnsupportedBackend, "This effect requires raster materials."};
    case Technique::AlphaTesting:
        if (backend != Backend::DXR) return {definition.State, definition.Explanation};
        return {Availability::UnsupportedBackend, "DXR scene materials have no opacity or alpha-cutout input."};
    case Technique::Forward:
    case Technique::ShadowMapping:
    case Technique::BasicSoftShadows:
    case Technique::PCF:
        if (backend != Backend::DXR) return {definition.State, definition.Explanation};
        return {Availability::UnsupportedBackend, "DXR uses ray visibility and direct lighting."};
    case Technique::RayTracedShadows:
        if (backend == Backend::DXR) return {definition.State, definition.Explanation};
        return {Availability::UnsupportedBackend, "Ray shadows require native scene acceleration and shader traversal."};
    default:
        return {Availability::UnsupportedBackend, "This technique requires a native raster backend."};
    }
}

Resolution ResolveSettings(const Settings& requested, Backend backend)
{
    Resolution result;
    result.Effective = SanitizeSettings(requested);
    if (result.Effective.GlobalIllumination.Lumen.HardwareRayTracing && backend!=Backend::DX12 && backend!=Backend::Vulkan && backend!=Backend::Metal)
    { result.Effective.GlobalIllumination.Lumen.HardwareRayTracing=false; result.Diagnostics.emplace_back("This API uses Lumen distance-field tracing; the hardware request is retained."); }
    if (UsesLumenScene(result.Effective) && result.Effective.GlobalIllumination.Lumen.HardwareRayTracing && result.Effective.GlobalIllumination.Lumen.GlobalDistance.Enabled)
    { result.Effective.GlobalIllumination.Lumen.GlobalDistance.Enabled=false; result.Diagnostics.emplace_back("Hardware Lumen uses triangle queries; software tracing clipmaps remain requested but inactive."); }
    if (result.Effective.MeshShaders && !QuerySupport(Technique::MeshShaders,backend).Available())
    { result.Effective.MeshShaders=false; result.Diagnostics.emplace_back("Native mesh stages are unavailable on this API; the request is retained."); }
    if (result.Effective.MeshShaders)
    {
        if (!UsesVisibilityBuffer(result.Effective.Pipeline)) result.Diagnostics.emplace_back("Mesh stages require visibility-buffer material reconstruction; the pipeline request is retained.");
        result.Effective.Pipeline=RenderingPipeline::VisibilityBuffer;
        if (result.Effective.Parallax.Enabled)
        { result.Effective.Parallax.Enabled=false; result.Diagnostics.emplace_back("Mesh visibility uses geometric depth; the separate POM material path is inactive and its request is retained."); }
        if (result.Effective.Nanite.Enabled) result.Effective.Nanite.GpuHardware=true;
    }
    // Haar/Aaltonen, SIGGRAPH 2015: GPU instance/cluster visibility,
    // compacted geometry, indirect raster and deferred material reconstruction.
    if (UsesVisibilityBuffer(result.Effective.Pipeline) && QuerySupport(Technique::Nanite,backend).Available())
    {
        result.Effective.Nanite.Enabled=true; result.Effective.Nanite.Occlusion=true;
        result.Effective.Nanite.GpuInstances=true; result.Effective.Nanite.GpuHardware=true;
        result.Effective.Nanite.MaterialTiles=true;
    }
    if (result.Effective.Nanite.Enabled && !QuerySupport(Technique::Nanite,backend).Available())
    { result.Effective.Nanite.Enabled=false; result.Diagnostics.emplace_back("Nanite is unavailable on this backend; the request is retained."); }
    if (result.Effective.Nanite.Enabled)
    {
        auto& effective=result.Effective;
        if (effective.Pipeline!=RenderingPipeline::VisibilityBuffer)
            result.Diagnostics.emplace_back("Nanite requires the visibility-buffer pipeline; the requested pipeline is retained.");
        effective.Pipeline=RenderingPipeline::VisibilityBuffer;
        if (effective.VirtualGeometry.Enabled || effective.Lod.Enabled)
            result.Diagnostics.emplace_back("Nanite owns resident cut selection; virtual geometry and whole-mesh LOD requests are retained but inactive.");
        effective.VirtualGeometry.Enabled=false; effective.Lod.Enabled=false;
        if (effective.Parallax.Enabled) result.Diagnostics.emplace_back("Nanite coverage uses geometric depth; parallax mapping is inactive and its request is retained.");
        effective.Parallax.Enabled=false;
        if (UsesHairFur(effective.Hair) && effective.ContactShadows.Enabled)
        { effective.ContactShadows.Enabled=false; result.Diagnostics.emplace_back("Nanite visibility cannot retain the separate fiber contact prepass; contact shadows are inactive while fibers are enabled."); }
    }
    if (result.Effective.VirtualGeometry.Enabled && !QuerySupport(Technique::VirtualGeometry,backend).Available())
    {
        result.Effective.VirtualGeometry.Enabled=false;
        result.Diagnostics.emplace_back("Virtual geometry is unavailable on this backend; the request is retained.");
    }
    if (result.Effective.VirtualGeometry.Enabled && result.Effective.Lod.Enabled)
    {
        result.Effective.Lod.Enabled=false;
        result.Diagnostics.emplace_back("Virtual geometry selects hierarchical cuts; whole-mesh LOD is inactive and its request is retained.");
    }
    if (result.Effective.Tessellation.Enabled && !QuerySupport(Technique::Tessellation,backend).Available())
    {
        result.Effective.Tessellation.Enabled=false;
        result.Diagnostics.emplace_back("Native PN tessellation is unavailable on this backend; the density/curvature request is retained.");
    }
    if (result.Effective.Displacement.Enabled && !QuerySupport(Technique::DisplacementMapping,backend).Available())
    {
        result.Effective.Displacement.Enabled=false;
        result.Diagnostics.emplace_back("Native geometric displacement is unavailable on this backend; the requested height settings are retained.");
    }
    if (UsesPathTracing(result.Effective.Pipeline))
    {
        if (!QuerySupport(Technique::PathTracing,backend).Available())
        {
            result.Effective.Pipeline=RenderingPipeline::Forward;
            result.Diagnostics.emplace_back("Path tracing is unavailable on this backend; the request is retained.");
        }
        else
        {
            auto& path=result.Effective;
            path.Shading=ShadingModel::PBR; path.SampleCount=1; path.AlphaBlending=false; path.Refraction.Enabled=false;
            path.BackFaceCulling=false; path.ContactShadows.Enabled=false; path.Subsurface.Enabled=false; path.Hair.Enabled=false; path.Parallax.Enabled=false;
            path.GPUDriven=false;
            path.AmbientOcclusion.Mode=AmbientOcclusionMode::None;
            path.Reflections.Mode=ReflectionMode::None; path.Reflections.Planar.Enabled=false; path.Reflections.Skybox=false;
            path.Reflections.Atmosphere.Enabled=false;
            path.GlobalIllumination.Mode=GlobalIlluminationMode::None;
            path.VolumetricFog.Enabled=false; path.VolumetricLighting.Enabled=false; path.LightShafts.Enabled=false;
            path.PostProcessing.AntiAliasing=AntiAliasingMode::None; path.PostProcessing.MotionBlur.Enabled=false;
            if (path.PostProcessing.HDR.ToneMapping==ToneMappingMode::None) path.PostProcessing.HDR.ToneMapping=ToneMappingMode::Reinhard;
            result.Diagnostics.emplace_back("Path tracing owns primary visibility, surface transport and stochastic antialiasing. Raster lighting, blended surfaces, volume effects and temporal filters are inactive; requested settings are retained.");
            return result;
        }
    }
    if (UsesLightLists(result.Effective.Pipeline) && !QuerySupport(result.Effective.Pipeline == RenderingPipeline::ClusteredForward ? Technique::ClusteredForward :
        result.Effective.Pipeline == RenderingPipeline::TiledDeferred ? Technique::TiledDeferred : Technique::ForwardPlus, backend).Available())
    {
        result.Effective.Pipeline = RenderingPipeline::Forward;
        result.Diagnostics.emplace_back("The requested light-list pipeline is unavailable on this backend; Forward is active and the request is retained.");
    }
    if (UsesHybridRendering(result.Effective.Pipeline))
    {
        if (!QuerySupport(Technique::Hybrid, backend).Available())
        {
            result.Effective.Pipeline = RenderingPipeline::Forward;
            result.Diagnostics.emplace_back("Hybrid rendering is unavailable on this backend; the pipeline request is retained.");
        }
        else
        {
            if (result.Effective.Reflections.Mode != ReflectionMode::RayTraced)
                result.Diagnostics.emplace_back("Hybrid rendering selects its screen/scene reflection estimator; the independent reflection request is retained.");
            result.Effective.Reflections.Mode = ReflectionMode::RayTraced;
        }
    }
    if (result.Effective.Parallax.Enabled && (!QuerySupport(Technique::ParallaxOcclusionMapping,backend).Available() || UsesHybridRendering(result.Effective.Pipeline)))
    {
        result.Effective.Parallax.Enabled=false;
        result.Diagnostics.emplace_back("POM uses native raster material capture; Hybrid scene rays retain the requested height settings as inactive.");
    }
    if (UsesParallaxOcclusion(result.Effective.Parallax))
    {
        if (result.Effective.Pipeline!=RenderingPipeline::Deferred && result.Effective.Pipeline!=RenderingPipeline::TiledDeferred)
        {
            result.Effective.Pipeline=RenderingPipeline::Deferred;
            result.Diagnostics.emplace_back("POM material capture uses Deferred lighting; the requested primary pipeline is retained.");
        }
        if (result.Effective.Shading==ShadingModel::Flat || result.Effective.Shading==ShadingModel::Gouraud)
        {
            result.Effective.Shading=ShadingModel::Phong;
            result.Diagnostics.emplace_back("POM requires per-fragment material normals; Phong is active while the requested shading model is retained.");
        }
        if (result.Effective.Reflections.Planar.Enabled)
        {
            result.Effective.Reflections.Planar.Enabled=false;
            result.Diagnostics.emplace_back("Planar receivers require undisplaced capture geometry; the planar request is retained while POM is active.");
        }
        if (result.Effective.Reflections.Mode!=ReflectionMode::None && result.Effective.Reflections.Mode!=ReflectionMode::ScreenSpace)
        {
            result.Effective.Reflections.Mode=ReflectionMode::None;
            result.Diagnostics.emplace_back("POM capture and scene-ray reflection materials are pending; screen-space reflections remain composable and the reflection request is retained.");
        }
    }
    if (UsesVisibilityBuffer(result.Effective.Pipeline))
    {
        if (!QuerySupport(Technique::VisibilityBuffer, backend).Available())
        {
            result.Effective.Pipeline = RenderingPipeline::Forward;
            result.Diagnostics.emplace_back("Visibility buffers are unavailable on this backend; the pipeline request is retained.");
        }
        else
        {
            if (result.Effective.SampleCount != 1)
                result.Diagnostics.emplace_back("The current visibility payload is single-sample; the requested MSAA count is retained.");
            result.Effective.SampleCount = 1;
            result.Effective.GPUDriven=QuerySupport(Technique::GPUDriven,backend).Available();
        }
    }
    if (result.Effective.Hair.Enabled && (!QuerySupport(Technique::HairFur,backend).Available() || UsesHybridRendering(result.Effective.Pipeline)))
    {
        result.Effective.Hair.Enabled=false;
        result.Diagnostics.emplace_back("Fiber ribbons require native raster transport; hybrid scene rays contain only meshes. The requested groom is retained.");
    }
    if (UsesHairFur(result.Effective.Hair))
    {
        // Screen-space consumers see fiber depth/color; the scene BVH has no
        // ribbon primitives. Keep the requested ray estimators for switching back.
        if (result.Effective.AmbientOcclusion.Mode==AmbientOcclusionMode::RayTraced)
        { result.Effective.AmbientOcclusion.Mode=AmbientOcclusionMode::None; result.Diagnostics.emplace_back("Ray AO is inactive while fibers are visible; the mesh BVH does not contain their primitives."); }
        if (result.Effective.Reflections.Mode==ReflectionMode::RayTraced)
        { result.Effective.Reflections.Mode=ReflectionMode::None; result.Diagnostics.emplace_back("Scene-ray reflections are inactive while fibers are visible; cube, planar and screen reflections remain available."); }
        // Lumen represents fibers as an oriented medium in its scene word buffer.
        if (UsesRayTracedGI(result.Effective.GlobalIllumination))
        { result.Effective.GlobalIllumination.Mode=GlobalIlluminationMode::None; result.Diagnostics.emplace_back("Scene-ray GI is inactive while fibers are visible; their ray primitives remain a TODO."); }
        if (UsesVisibilityBuffer(result.Effective.Pipeline) && result.Effective.ContactShadows.Enabled)
        {
            if (result.Effective.MeshShaders)
            {
                result.Effective.ContactShadows.Enabled=false;
                result.Diagnostics.emplace_back("Mesh visibility cannot retain the separate fiber contact prepass; contact shadows are inactive while fibers are enabled and their request is retained.");
            }
            else
            {
                result.Effective.Pipeline=RenderingPipeline::Deferred;
                result.Diagnostics.emplace_back("Fiber contacts require retained camera depth; Deferred is active because visibility reconstruction clears its primary depth.");
            }
        }
        if (result.Effective.PostProcessing.HDR.ToneMapping==ToneMappingMode::None)
            result.Effective.PostProcessing.HDR.ToneMapping=ToneMappingMode::Reinhard;
    }
    if (result.Effective.PostProcessing.TemporalEffects.Enabled)
    {
        if (!QuerySupport(Technique::TemporalEffects,backend).Available()) result.Effective.PostProcessing.TemporalEffects.Enabled=false;
        else if (result.Effective.PostProcessing.HDR.ToneMapping==ToneMappingMode::None) result.Effective.PostProcessing.HDR.ToneMapping=ToneMappingMode::Reinhard;
    }
    if (result.Effective.Subsurface.Enabled && !QuerySupport(Technique::SubsurfaceScattering,backend).Available())
    {
        result.Effective.Subsurface.Enabled=false;
        result.Diagnostics.emplace_back("Subsurface diffusion is unavailable on this backend; the requested profile is retained.");
    }
    if (IsNativeRaster(backend) && UsesSubsurfaceScattering(result.Effective.Subsurface) &&
        (!UsesDeferredLighting(result.Effective.Pipeline) || result.Effective.Pipeline==RenderingPipeline::TiledDeferred))
    {
        result.Effective.Pipeline=RenderingPipeline::Deferred;
        result.Diagnostics.emplace_back("Subsurface diffusion requires separated per-pixel diffuse light; Deferred is active and the requested pipeline is retained.");
    }
    if (UsesLightLists(result.Effective.Pipeline) && result.Effective.Shading == ShadingModel::Gouraud)
    {
        result.Effective.Shading = ShadingModel::Phong;
        result.Diagnostics.emplace_back("Tile and cluster light lists are evaluated per fragment; Gouraud requires Forward. Phong is active.");
    }
    if (result.Effective.ContactShadows.Enabled && !QuerySupport(Technique::ContactShadows, backend).Available())
    {
        result.Effective.ContactShadows.Enabled = false;
        result.Diagnostics.emplace_back("Contact shadows are unavailable on this backend; the request is retained.");
    }
    if (UsesDepthPeeling(result.Effective) && !QuerySupport(Technique::OIT, backend).Available())
    {
        result.Effective.Transparency.Mode = TransparencyMode::Sorted;
        result.Diagnostics.emplace_back("Depth peeling is unavailable on this backend; sorted blending is active and the request is retained.");
    }
    if (UsesDepthPeeling(result.Effective))
    {
        if (result.Effective.SampleCount != 1)
        {
            result.Effective.SampleCount = 1;
            result.Diagnostics.emplace_back("Depth peeling requires single-sample depth-EQUAL replay; the requested MSAA count is retained.");
        }
        if (result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
        {
            result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
            result.Diagnostics.emplace_back("Depth peeling composes scene-linear radiance; Reinhard tone mapping is active.");
        }
    }
    if (UsesRefraction(result.Effective) && !QuerySupport(Technique::Refraction, backend).Available())
    {
        result.Effective.Refraction.Enabled = false;
        result.Diagnostics.emplace_back("Refraction is unavailable on this backend; glass uses its base material and the request is retained.");
    }
    if (UsesRefraction(result.Effective))
    {
        if (result.Effective.SampleCount != 1)
        {
            result.Effective.SampleCount = 1;
            result.Diagnostics.emplace_back("Nearest-layer refraction currently requires single-sample capture; the requested MSAA count is retained.");
        }
        if (result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
        {
            result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
            result.Diagnostics.emplace_back("Refraction composes scene-linear radiance; Reinhard tone mapping is active.");
        }
    }
    if (UsesWeightedTransparency(result.Effective) && !QuerySupport(Technique::WeightedBlendedOIT, backend).Available())
    {
        result.Effective.Transparency.Mode = TransparencyMode::Sorted;
        result.Diagnostics.emplace_back("Weighted transparency is unavailable on this backend; sorted blending is active and the request is retained.");
    }
    if (UsesWeightedTransparency(result.Effective) && result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Weighted transparency accumulates scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (result.Effective.VolumetricLighting.Enabled && !QuerySupport(Technique::VolumetricLighting, backend).Available())
    {
        result.Effective.VolumetricLighting.Enabled = false;
        result.Diagnostics.emplace_back("Volumetric lighting is unavailable on this backend; the requested settings are retained.");
    }
    if (UsesVolumetricLighting(result.Effective.VolumetricLighting) && !result.Effective.VolumetricFog.Enabled)
    {
        result.Effective.VolumetricFog.Enabled = true;
        result.Diagnostics.emplace_back("Volumetric lighting uses the selected medium; its camera transport is active.");
    }
    if (UsesBoundedMedium(result.Effective.VolumetricFog) && !QuerySupport(
        UsesVolumetricClouds(result.Effective.VolumetricFog) ? Technique::VolumetricClouds : Technique::ParticipatingMedia, backend).Available())
    {
        result.Effective.VolumetricFog.Enabled = false;
        result.Effective.VolumetricLighting.Enabled = false;
        result.Diagnostics.emplace_back("The selected bounded medium is unavailable on this backend; the requested settings are retained.");
    }
    if (result.Effective.VolumetricFog.Enabled && !QuerySupport(Technique::VolumetricFog, backend).Available())
    {
        result.Effective.VolumetricFog.Enabled = false;
        result.Diagnostics.emplace_back("Volumetric fog is unavailable on this backend; the requested settings are retained.");
    }
    if (UsesVolumetricFog(result.Effective.VolumetricFog) && result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Volumetric fog transports scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (result.Effective.LightShafts.Enabled && !QuerySupport(Technique::LightShafts, backend).Available())
    {
        result.Effective.LightShafts.Enabled = false;
        result.Diagnostics.emplace_back("Screen-space light shafts are unavailable on this backend; the request is retained.");
    }
    if (UsesLightShafts(result.Effective.LightShafts) && UsesVolumetricFog(result.Effective.VolumetricFog) &&
        UsesVolumetricLighting(result.Effective.VolumetricLighting))
    {
        result.Effective.LightShafts.Enabled = false;
        result.Diagnostics.emplace_back("Volume lighting and screen-space shafts are alternative scattering methods; volume lighting takes precedence.");
    }
    if (UsesLightShafts(result.Effective.LightShafts) && result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Light shafts add scene-linear scattering; Reinhard tone mapping is active.");
    }
    if (result.Effective.GPUDriven)
    {
        if (!QuerySupport(Technique::GPUDriven, backend).Available())
        {
            result.Effective.GPUDriven = false;
            result.Diagnostics.emplace_back("GPU-driven rendering is unavailable on this backend; the requested switch is retained.");
        }
        else
        {
            result.Effective.Instancing = true;
            result.Diagnostics.emplace_back("GPU-driven rendering uses instanced vertex input and indirect commands. Frustum culling controls GPU rejection; CPU material batching and LOD remain active.");
        }
    }
    if (result.Effective.Instancing && !QuerySupport(Technique::Instancing, backend).Available())
    {
        result.Effective.Instancing = false;
        result.Diagnostics.emplace_back("Instanced rendering is unavailable on this backend; the requested switch is retained.");
    }
    const auto reflectionTechnique = UsesLumenReflections(result.Effective.Reflections) ? Technique::Lumen : result.Effective.Reflections.Mode == ReflectionMode::RayTraced ? Technique::RayTracedReflections :
        result.Effective.Reflections.Mode == ReflectionMode::LatitudeLongitude ? Technique::EnvironmentReflections :
        result.Effective.Reflections.Mode == ReflectionMode::Cubemap ? Technique::CubemapReflections : Technique::SSR;
    if (result.Effective.Reflections.Mode!=ReflectionMode::None && !QuerySupport(reflectionTechnique, backend).Available())
    {
        result.Effective.Reflections.Mode = ReflectionMode::None;
        result.Diagnostics.emplace_back("The requested reflection method is unavailable on this backend; the request is retained.");
    }
    if (UsesPlanarCapture(result.Effective.Reflections) && (!QuerySupport(Technique::PlanarReflections,backend).Available() ||
        (result.Effective.Reflections.Mode!=ReflectionMode::None && !UsesEnvironmentCapture(result.Effective.Reflections))))
    {
        result.Effective.Reflections.Planar.Enabled=false;
        result.Diagnostics.emplace_back("Planar receivers compose with environment reflections; the planar request is retained for incompatible estimators.");
    }
    if (result.Effective.SampleCount > 1 && !QuerySupport(Technique::MSAA, backend).Available())
    {
        result.Effective.SampleCount = 1;
        result.Diagnostics.emplace_back("MSAA is unavailable on this backend; the requested sample count is retained.");
    }
    if (result.Effective.Shadows==ShadowRepresentation::RayTraced)
    {
        if (!QuerySupport(Technique::RayTracedShadows,backend).Available() || UsesHairFur(result.Effective.Hair))
        {
            result.Effective.Shadows=ShadowRepresentation::SingleMap;
            result.Diagnostics.emplace_back("Ray-shadow triangle queries cannot represent procedural hair strands on this backend; shadow mapping is effective and the request is retained.");
        }
        else
        {
            result.Effective.ShadowFilter.Mode=ShadowFilterMode::Hard;
            result.Effective.ContactShadows.Enabled=false;
        }
    }
    if (result.Effective.Shadows == ShadowRepresentation::Cascaded && !QuerySupport(Technique::CascadedShadowMaps, backend).Available())
    {
        result.Effective.Shadows = ShadowRepresentation::SingleMap;
        result.Diagnostics.emplace_back("CSM is unavailable on this backend; the requested cascades are retained.");
    }
    if (result.Effective.Shadows == ShadowRepresentation::Virtual)
    {
        if (!QuerySupport(Technique::VirtualShadowMaps, backend).Available() || UsesHairFur(result.Effective.Hair))
        {
            result.Effective.Shadows=ShadowRepresentation::SingleMap;
            result.Diagnostics.emplace_back(UsesHairFur(result.Effective.Hair)?
                "Virtual fiber casting and reception are pending; fibers use a single shadow map and the virtual-map request is retained.":
                "Virtual shadows are unavailable on this backend; a single map is effective and the request is retained.");
        }
        else if (result.Effective.ShadowFilter.Mode!=ShadowFilterMode::Hard && result.Effective.ShadowFilter.Mode!=ShadowFilterMode::PCF)
        {
            result.Effective.ShadowFilter.Mode=ShadowFilterMode::PCF;
            result.Diagnostics.emplace_back("Virtual page lookup currently supports hard/PCF filtering; the requested PCSS/moment settings are retained.");
        }
    }
    if (UsesDeferredLighting(result.Effective.Pipeline) && !QuerySupport(Technique::Deferred, backend).Available())
    {
        result.Effective.Pipeline = RenderingPipeline::Forward;
        result.Diagnostics.emplace_back("Deferred is not implemented on this backend; Forward remains active.");
    }
    if (result.Effective.ShadowFilter.Mode == ShadowFilterMode::PCSS && !QuerySupport(Technique::PCSS, backend).Available())
    {
        result.Effective.ShadowFilter.Mode = ShadowFilterMode::PCF;
        result.Diagnostics.emplace_back("PCSS is unavailable on this backend; PCF is effective and the requested PCSS settings are retained.");
    }
    if (result.Effective.ShadowFilter.Mode == ShadowFilterMode::VSM && !QuerySupport(Technique::VSM, backend).Available())
    {
        result.Effective.ShadowFilter.Mode = ShadowFilterMode::PCF;
        result.Diagnostics.emplace_back("VSM is unavailable on this backend; PCF is effective and the requested VSM settings are retained.");
    }
    if (result.Effective.AreaLights.Enabled && (!IsNativeRaster(backend) || result.Effective.Shading != ShadingModel::PBR))
    {
        result.Effective.AreaLights.Enabled = false;
        result.Diagnostics.emplace_back("Physical area integration requires native PBR shading; the request is retained.");
    }
    if (result.Effective.AreaLights.Enabled && result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Physical area lights integrate scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (!IsNativeRaster(backend))
    {
        if (!(requested.ShadowFilter == ShadowFilterSettings{}))
            result.Diagnostics.emplace_back("Selectable hard/PCF filtering is native-only; the requested filter is retained.");
        if (requested.Lod.Enabled)
            result.Diagnostics.emplace_back("Native mesh LOD is unavailable on this backend; the requested settings are retained.");
        if (requested.Shading != ShadingModel::Phong)
            result.Diagnostics.emplace_back("This backend uses Phong; the requested native shading model is retained.");
        if (requested.FrustumCulling || requested.BackFaceCulling)
            result.Diagnostics.emplace_back("Native visibility controls are unavailable on this backend.");
        if (!BuildPostProcessPlan(requested.PostProcessing).empty())
            result.Diagnostics.emplace_back("Native post-processing is unavailable; requested effects are retained.");
        if (RequiresHDR(requested.PostProcessing.HDR))
            result.Diagnostics.emplace_back("Native HDR effects are unavailable; requested settings are retained.");
        if (UsesAmbientOcclusion(requested.AmbientOcclusion))
            result.Diagnostics.emplace_back("Native ambient occlusion is unavailable; the request is retained.");
        if (requested.PostProcessing.DepthOfField.Enabled)
            result.Diagnostics.emplace_back("Native depth of field is unavailable; lens settings are retained.");
        result.Effective = Settings{};
        result.Effective.PostProcessing = SanitizePostProcessSettings(requested.PostProcessing);
        result.Effective.PostProcessing.AntiAliasing = AntiAliasingMode::None;
        result.Effective.PostProcessing.Vignette = false;
        result.Effective.PostProcessing.FilmGrain = false;
        result.Effective.PostProcessing.ScreenSpaceDistortion = false;
        result.Effective.PostProcessing.NPR.Enabled = false;
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::None;
        result.Effective.PostProcessing.HDR.Bloom = false;
        result.Effective.PostProcessing.HDR.LensFlare = false;
        result.Effective.PostProcessing.HDR.ColorGrading = false;
        result.Effective.PostProcessing.DepthOfField.Enabled = false;
        result.Effective.PostProcessing.MotionBlur.Enabled = false;
        result.Effective.NormalMapping = false;
        result.Effective.AlphaBlending = false;
        if (backend == Backend::DXR) result.Effective.AlphaTesting = false;
        if (requested.TextureMapping != result.Effective.TextureMapping ||
            requested.AlphaTesting != result.Effective.AlphaTesting ||
            requested.AlphaBlending != result.Effective.AlphaBlending)
            result.Diagnostics.emplace_back("Material controls use this backend's implemented defaults.");
        return result;
    }
    if (UsesGlobalIllumination(result.Effective.GlobalIllumination) && !QuerySupport(
        UsesLightmaps(result.Effective.GlobalIllumination) ? Technique::Lightmaps :
        UsesBakedProbes(result.Effective.GlobalIllumination) ? Technique::LightProbes :
        UsesLumen(result.Effective.GlobalIllumination) ? Technique::Lumen :
        UsesVoxelGI(result.Effective.GlobalIllumination) ? Technique::VoxelGI :
        UsesRayTracedGI(result.Effective.GlobalIllumination) ? Technique::RayTracedGI : Technique::SSGI, backend).Available())
    {
        result.Effective.GlobalIllumination.Mode = GlobalIlluminationMode::None;
        result.Diagnostics.emplace_back("The requested diffuse transport is unavailable on this backend; the request is retained.");
    }
    if (UsesGlobalIllumination(result.Effective.GlobalIllumination) &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Global illumination transports scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (UsesTemporalAA(result.Effective.PostProcessing) && !QuerySupport(Technique::TAA, backend).Available())
    {
        result.Effective.PostProcessing.AntiAliasing = AntiAliasingMode::None;
        result.Diagnostics.emplace_back("TAA is not implemented on this backend; the request is retained.");
    }
    if (UsesTemporalSuperResolution(result.Effective.PostProcessing) && !QuerySupport(Technique::TSR,backend).Available())
    { result.Effective.PostProcessing.AntiAliasing=AntiAliasingMode::None; result.Diagnostics.emplace_back("TSR requires a native image reconstruction backend; the request is retained."); }
    if (UsesTemporalSuperResolution(result.Effective.PostProcessing) && result.Effective.PostProcessing.HDR.ToneMapping==ToneMappingMode::None)
    { result.Effective.PostProcessing.HDR.ToneMapping=ToneMappingMode::Reinhard; result.Diagnostics.emplace_back("TSR reconstructs scene-linear HDR; Reinhard tone mapping is active."); }
    if (UsesTemporalAA(result.Effective.PostProcessing) && result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("TAA accumulates scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (result.Effective.PostProcessing.MotionBlur.Enabled && !QuerySupport(Technique::MotionBlur, backend).Available())
    {
        result.Effective.PostProcessing.MotionBlur.Enabled = false;
        result.Diagnostics.emplace_back("Motion blur is not implemented on this backend; the request is retained.");
    }
    if (UsesDeferredLighting(result.Effective.Pipeline))
    {
        if (result.Effective.Shading == ShadingModel::Gouraud)
        {
            result.Effective.Shading = ShadingModel::Phong;
            result.Diagnostics.emplace_back("Deferred lighting is per-pixel; Gouraud is a Forward-only choice. Phong is active.");
        }
        if (result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
        {
            result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
            result.Diagnostics.emplace_back("Deferred lighting produces HDR radiance; Reinhard tone mapping is active.");
        }
    }
    if (UsesReflections(result.Effective.Reflections) &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Reflections compose scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (UsesAmbientOcclusion(result.Effective.AmbientOcclusion) &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Ambient occlusion changes scene-linear ambient radiance; Reinhard tone mapping is active.");
    }
    if (RequiresHDR(result.Effective.PostProcessing.HDR) &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("HDR color effects require a tone resolve; Reinhard is active.");
    }
    if (result.Effective.PostProcessing.DepthOfField.Enabled &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Depth of field gathers scene-linear radiance; Reinhard tone mapping is active.");
    }
    if (result.Effective.PostProcessing.MotionBlur.Enabled &&
        result.Effective.PostProcessing.HDR.ToneMapping == ToneMappingMode::None)
    {
        result.Effective.PostProcessing.HDR.ToneMapping = ToneMappingMode::Reinhard;
        result.Diagnostics.emplace_back("Motion blur integrates scene-linear radiance; Reinhard tone mapping is active.");
    }
    if ((result.Effective.Shading == ShadingModel::Gouraud ||
         result.Effective.Shading == ShadingModel::Flat) && result.Effective.NormalMapping)
    {
        result.Effective.NormalMapping = false;
        result.Diagnostics.emplace_back(result.Effective.Shading == ShadingModel::Flat ?
            "Flat shading uses geometric face normals; normal mapping is inactive." :
            "Gouraud uses vertex lighting; fragment normal mapping is inactive.");
    }
    return result;
}

Resolution ResolveSettingsForDevice(const Settings& requested, Backend backend, bool deferredShadingSupported,
    bool scalableAmbientObscuranceSupported, std::uint32_t supportedSampleCounts, bool instancingSupported,
    bool temporalAntiAliasingSupported, bool volumetricFogSupported, std::uint32_t weightedTransparencySampleCounts,
    bool volumetricLightingSupported, bool lightShaftsSupported, bool varianceShadowMapsSupported, bool contactShadowsSupported,
    bool forwardPlusSupported, bool clusteredForwardSupported, bool tiledDeferredSupported, bool globalIlluminationSupported,
    bool bakedProbesSupported, bool lightmapsSupported, bool refractionSupported, bool depthPeelingSupported, bool rayTracedAOSupported,
    bool rayTracedReflectionsSupported, bool rayTracedGISupported, bool pathTracingSupported, bool gpuDrivenSupported, bool visibilityBufferSupported, bool hybridRenderingSupported, bool voxelGISupported, bool temporalEffectsSupported, bool parallaxSupported, bool displacementSupported, bool tessellationSupported, bool virtualGeometrySupported, bool temporalSuperResolutionSupported, bool virtualShadowsSupported, bool lumenSupported, bool naniteSupported, bool meshShadersSupported, bool naniteHardwareSupported, bool lumenHardwareSupported)
{
    auto fallback = requested;
    std::vector<std::string> diagnostics;
    if (UsesVisibilityBuffer(requested.Pipeline) && (!naniteSupported || !gpuDrivenSupported || !virtualGeometrySupported || !naniteHardwareSupported))
    {
        fallback.Pipeline=RenderingPipeline::Forward;fallback.Nanite.Enabled=false;fallback.MeshShaders=false;
        diagnostics.emplace_back("GPU Driven Visibility Buffer requires native cluster decoding, compute visibility and indirect hardware drawing.");
    }
    if (requested.GlobalIllumination.Lumen.HardwareRayTracing && (!lumenHardwareSupported || !lumenSupported))
    { fallback.GlobalIllumination.Lumen.HardwareRayTracing=false; diagnostics.emplace_back("Native Lumen triangle queries are unavailable; distance-field tracing remains effective and the request is retained."); }
    if (requested.MeshShaders && (!meshShadersSupported || !visibilityBufferSupported || !deferredShadingSupported || (requested.Nanite.Enabled && !naniteHardwareSupported)))
    { fallback.MeshShaders=false; diagnostics.emplace_back("This device lacks native mesh stages or their required visibility/hardware-bin inputs; the request is retained."); }
    if (requested.Nanite.Enabled && (!naniteSupported || !visibilityBufferSupported || !deferredShadingSupported || !virtualGeometrySupported))
    { fallback.Nanite.Enabled=false; diagnostics.emplace_back("This device lacks native Nanite decode, hybrid coverage or visibility material support; the request is retained."); }
    if (UsesLumenReflections(requested.Reflections) && !lumenSupported)
    { fallback.Reflections.Mode=ReflectionMode::None; diagnostics.emplace_back("Lumen reflections need native compute tracing, material targets and storage buffers; the request is retained."); }
    if (UsesLumen(requested.GlobalIllumination) && !lumenSupported)
    { fallback.GlobalIllumination.Mode=GlobalIlluminationMode::None; diagnostics.emplace_back("Lumen needs native compute tracing, material targets and fragment storage buffers."); }
    if (requested.Shadows==ShadowRepresentation::Virtual && !virtualShadowsSupported)
    { fallback.Shadows=ShadowRepresentation::SingleMap; diagnostics.emplace_back("This device lacks virtual shadow demand compute/storage support; a single map is effective and the request is retained."); }
    if (UsesTemporalSuperResolution(requested.PostProcessing) && !temporalSuperResolutionSupported)
    { fallback.PostProcessing.AntiAliasing=AntiAliasingMode::None; diagnostics.emplace_back("This device lacks native TSR compute, fragment storage or HDR image support; the request is retained."); }
    if (requested.VirtualGeometry.Enabled && !virtualGeometrySupported)
    {
        fallback.VirtualGeometry.Enabled=false;
        diagnostics.emplace_back("This device lacks native virtual geometry selection, page storage or drawable decoding; the request is retained.");
    }
    if (requested.Tessellation.Enabled && !tessellationSupported)
    {
        fallback.Tessellation.Enabled=false;
        diagnostics.emplace_back("This device lacks the native tessellation vertex/index compute, storage and readback contract; the request is retained.");
    }
    if (requested.Displacement.Enabled && !displacementSupported)
    {
        fallback.Displacement.Enabled=false;
        diagnostics.emplace_back("This device lacks the native displacement compute/storage/readback contract; the request is retained.");
    }
    if (requested.Parallax.Enabled && !parallaxSupported)
    {
        fallback.Parallax.Enabled=false;
        diagnostics.emplace_back("This device lacks the native POM material interface or floating attachments; the requested height settings are retained.");
    }
    if (requested.PostProcessing.TemporalEffects.Enabled && !temporalEffectsSupported)
    {
        fallback.PostProcessing.TemporalEffects.Enabled=false;
        diagnostics.emplace_back("This device lacks HDR or renderable FP32 exposure targets; the requested effect is retained.");
    }
    if (UsesVoxelGI(requested.GlobalIllumination) && !voxelGISupported)
    {
        fallback.GlobalIllumination.Mode=GlobalIlluminationMode::None;
        diagnostics.emplace_back("This device lacks sampled FP32 voxel storage or the required native shader/texture interface; the request is retained.");
    }
    if (UsesHybridRendering(requested.Pipeline) && (!hybridRenderingSupported || !deferredShadingSupported || !rayTracedReflectionsSupported))
    {
        fallback.Pipeline = deferredShadingSupported ? RenderingPipeline::Deferred : RenderingPipeline::Forward;
        diagnostics.emplace_back("This device lacks Hybrid rendering's deferred or scene-ray inputs; the pipeline request is retained.");
    }
    if (UsesVisibilityBuffer(requested.Pipeline) && (!visibilityBufferSupported || !deferredShadingSupported))
    {
        fallback.Pipeline = deferredShadingSupported ? RenderingPipeline::Deferred : RenderingPipeline::Forward;
        diagnostics.emplace_back("This device lacks visibility storage or integer/float attachments; the requested pipeline is retained.");
    }
    if ((requested.Pipeline == RenderingPipeline::ForwardPlus && !forwardPlusSupported) ||
        (requested.Pipeline == RenderingPipeline::ClusteredForward && !clusteredForwardSupported) ||
        (requested.Pipeline == RenderingPipeline::TiledDeferred && !tiledDeferredSupported))
    {
        fallback.Pipeline = RenderingPipeline::Forward;
        diagnostics.emplace_back("This device lacks the requested light-list compute and material storage support; the request is retained.");
    }
    if (requested.ContactShadows.Enabled && !contactShadowsSupported)
    {
        fallback.ContactShadows.Enabled = false;
        diagnostics.emplace_back("This device lacks contact-shadow compute/storage support; the request is retained.");
    }
    if (requested.ShadowFilter.Mode == ShadowFilterMode::VSM && !varianceShadowMapsSupported)
    {
        fallback.ShadowFilter.Mode = ShadowFilterMode::PCF;
        diagnostics.emplace_back("This device lacks floating moment targets for VSM; PCF is effective and the requested settings are retained.");
    }
    if (UsesLightShafts(requested.LightShafts) && !lightShaftsSupported)
    {
        fallback.LightShafts.Enabled = false;
        diagnostics.emplace_back("This device lacks floating color render/blit targets for light shafts; the requested settings are retained.");
    }
    if ((requested.VolumetricFog.Enabled || UsesVolumetricLighting(requested.VolumetricLighting)) &&
        requested.VolumetricFog.MediumMode != VolumeMediumMode::HeightFog && !volumetricLightingSupported)
    {
        fallback.VolumetricFog.Enabled = false;
        fallback.VolumetricLighting.Enabled = false;
        diagnostics.emplace_back("This device lacks FP32 RGB medium transport targets; the bounded-medium request is retained.");
    }
    if (UsesVolumetricLighting(requested.VolumetricLighting) && !volumetricLightingSupported)
    {
        fallback.VolumetricLighting.Enabled = false;
        diagnostics.emplace_back("This device lacks FP32 volume transport targets; ambient fog remains independently selectable and the lighting request is retained.");
    }
    if (UsesDepthPeeling(requested) && !depthPeelingSupported)
    {
        fallback.Transparency.Mode = TransparencyMode::Sorted;
        diagnostics.emplace_back("The device lacks floating OIT targets or invariant vertex positions; sorted blending is active and the request is retained.");
    }
    if (UsesRefraction(requested) && !refractionSupported)
    {
        fallback.Refraction.Enabled = false;
        diagnostics.emplace_back("The device lacks HDR/sampled-depth targets for refraction; glass uses its base material and the request is retained.");
    }
    if (UsesWeightedTransparency(requested) && QuerySupport(Technique::WeightedBlendedOIT, backend).Available())
    {
        if ((weightedTransparencySampleCounts & supportedSampleCounts & 1u) == 0)
        {
            fallback.Transparency.Mode = TransparencyMode::Sorted;
            diagnostics.emplace_back("The device lacks blendable float transparency targets; sorted blending is active and the request is retained.");
        }
        else supportedSampleCounts &= weightedTransparencySampleCounts;
    }
    if ((requested.VolumetricFog.Enabled || UsesVolumetricLighting(requested.VolumetricLighting)) && !volumetricFogSupported)
    {
        fallback.VolumetricFog.Enabled = false;
        fallback.VolumetricLighting.Enabled = false;
        diagnostics.emplace_back("This device lacks the floating fog target or required texture bindings; the request is retained.");
    }
    if (UsesScreenSpaceGI(requested.GlobalIllumination) && !globalIlluminationSupported)
    {
        fallback.GlobalIllumination.Mode = GlobalIlluminationMode::None;
        diagnostics.emplace_back("This device lacks screen-space GI material and private history targets; the request is retained.");
    }
    if (UsesLightmaps(requested.GlobalIllumination) && !lightmapsSupported)
    {
        fallback.GlobalIllumination.Mode = GlobalIlluminationMode::None;
        diagnostics.emplace_back("Surface lightmaps require native float atlases and depth-equal receiver composition.");
    }
    if (UsesBakedProbes(requested.GlobalIllumination) && !bakedProbesSupported)
    {
        fallback.GlobalIllumination.Mode = GlobalIlluminationMode::None;
        diagnostics.emplace_back("This device lacks baked probe projection and material targets; the request is retained.");
    }
    if (UsesTemporalAA(requested.PostProcessing) && !temporalAntiAliasingSupported)
    {
        fallback.PostProcessing.AntiAliasing = AntiAliasingMode::None;
        diagnostics.emplace_back("This device lacks renderable color/depth history targets; TAA is inactive and its request is retained.");
    }
    if (requested.Instancing && !instancingSupported)
    {
        fallback.Instancing = false;
        diagnostics.emplace_back("This device does not expose native instance submission; the requested switch is retained.");
    }
    if (requested.GPUDriven && (!gpuDrivenSupported || !instancingSupported))
    {
        fallback.GPUDriven = false;
        diagnostics.emplace_back("This device lacks GPU visibility, instance input or indirect drawing; the requested switch is retained.");
    }
    if ((UsesDeferredLighting(requested.Pipeline) || UsesAmbientOcclusion(requested.AmbientOcclusion) || UsesReflections(requested.Reflections) ||
        UsesGlobalIllumination(requested.GlobalIllumination) || requested.Subsurface.Enabled || requested.Hair.Enabled || requested.Parallax.Enabled) &&
        IsNativeRaster(backend) && !deferredShadingSupported)
    {
        fallback.Pipeline = RenderingPipeline::Forward;
        fallback.Reflections.Mode = ReflectionMode::None;
        fallback.GlobalIllumination.Mode = GlobalIlluminationMode::None;
        fallback.Reflections.Planar.Enabled=false;
        fallback.AmbientOcclusion.Mode = AmbientOcclusionMode::None;
        fallback.Subsurface.Enabled=false; fallback.Refraction.Enabled=false;
        fallback.Hair.Enabled=false; fallback.Parallax.Enabled=false;
        diagnostics.emplace_back("This device lacks the floating material G-buffer; Forward remains active without screen-space material effects.");
    }
    else if (requested.AmbientOcclusion.Mode == AmbientOcclusionMode::SAO &&
        IsNativeRaster(backend) && !scalableAmbientObscuranceSupported)
    {
        fallback.AmbientOcclusion.Mode = AmbientOcclusionMode::None;
        diagnostics.emplace_back("This device lacks the R32F depth hierarchy; SAO is inactive and its request is retained.");
    }
    if (requested.AmbientOcclusion.Mode == AmbientOcclusionMode::RayTraced && !rayTracedAOSupported)
    {
        fallback.AmbientOcclusion.Mode = AmbientOcclusionMode::None;
        diagnostics.emplace_back("This device lacks fragment scene-buffer traversal; ray-traced AO is inactive and its request is retained.");
    }
    if (requested.Reflections.Mode == ReflectionMode::RayTraced && !rayTracedReflectionsSupported)
    {
        fallback.Reflections.Mode = ReflectionMode::None;
        diagnostics.emplace_back("This device lacks fragment scene/material traversal; ray-traced reflections are inactive and the request is retained.");
    }
    if (UsesRayTracedGI(requested.GlobalIllumination) && !rayTracedGISupported)
    {
        fallback.GlobalIllumination.Mode=GlobalIlluminationMode::None;
        diagnostics.emplace_back("This device lacks scene/material traversal for diffuse GI; the request is retained.");
    }
    if (UsesPathTracing(requested.Pipeline) && !pathTracingSupported)
    {
        fallback.Pipeline=RenderingPipeline::Forward;
        diagnostics.emplace_back("This device lacks scene traversal or FP32 accumulation targets; path tracing is inactive and its request is retained.");
    }
    auto result = ResolveSettings(fallback, backend);
    const auto count = SelectSampleCount(result.Effective.SampleCount, supportedSampleCounts);
    if (count != result.Effective.SampleCount)
        diagnostics.emplace_back("The device uses a lower supported MSAA count; the requested count is retained.");
    result.Effective.SampleCount = count;
    result.Diagnostics.insert(result.Diagnostics.end(), diagnostics.begin(), diagnostics.end());
    return result;
}

PassPlan BuildPassPlan(const Settings& settings, Backend backend,
    bool shadowsEnabled, bool outlineEnabled, bool wireframeOnly)
{
    PassPlan result;
    if (!IsNativeRaster(backend))
    {
        result.Diagnostics.emplace_back("The native pass graph does not describe DXR or software execution.");
        return result;
    }
    Settings effective = ResolveSettings(settings, backend).Effective;
    if (wireframeOnly) effective.Hair.Enabled=false;
    if (wireframeOnly) { effective.Refraction.Enabled=false; effective.Transparency.Mode=TransparencyMode::Sorted; }
    result.ExternalInputs = Surface;
    const bool rayAO = effective.AmbientOcclusion.Mode == AmbientOcclusionMode::RayTraced;
    // The immutable scene BVH is prepared before native frame recording.
    const bool rayReflections = effective.Reflections.Mode == ReflectionMode::RayTraced;
    const bool rayGI=UsesRayTracedGI(effective.GlobalIllumination);
    const bool voxelGI=UsesVoxelGI(effective.GlobalIllumination);
    const bool lumen=UsesLumen(effective.GlobalIllumination);
    const bool lumenReflections=UsesLumenReflections(effective.Reflections);
    if (lumen || lumenReflections) result.ExternalInputs|=Mask(Resource::SurfaceRadianceCache);
    if (voxelGI) result.ExternalInputs |= Mask(Resource::Voxels);
    const bool lumenHardware=(lumen || lumenReflections) && effective.GlobalIllumination.Lumen.HardwareRayTracing;
    if (rayAO || rayReflections || rayGI || lumenHardware) result.ExternalInputs |= AS;
    const bool temporalAA = UsesTemporalAA(effective.PostProcessing);
    const bool temporalSuperResolution=UsesTemporalSuperResolution(effective.PostProcessing);
    if (temporalSuperResolution) result.ExternalInputs|=Mask(Resource::ReconstructionHistory);
    const bool globalIllumination = UsesGlobalIllumination(effective.GlobalIllumination);
    const bool screenGI = UsesScreenSpaceGI(effective.GlobalIllumination);
    const bool temporalGI = UsesTemporalGI(effective.GlobalIllumination);
    const bool temporalReflections=UsesTemporalReflections(effective.Reflections),earlyVelocity=temporalGI || temporalReflections;
    if (temporalReflections) result.ExternalInputs|=Mask(Resource::ReflectionHistory);
    const bool bakedProbes = UsesBakedProbes(effective.GlobalIllumination);
    if (temporalGI) result.ExternalInputs |= Mask(Resource::GlobalIlluminationHistory);
    if (bakedProbes) result.ExternalInputs |= Mask(Resource::BakedLighting);
    const bool lightmaps = UsesLightmaps(effective.GlobalIllumination);
    if (lightmaps) result.ExternalInputs |= Mask(Resource::BakedLighting) | Mask(Resource::LightmapUV);
    if (RequiresMotionVectors(effective)) result.ExternalInputs |= Mask(Resource::PreviousTransforms);
    if (temporalAA) result.ExternalInputs |= History;
    if (UsesDisplacementMapping(effective.Displacement)) result.ExternalInputs|=Mask(Resource::HeightMap)|Mask(Resource::DisplacedGeometry);
    if (UsesTessellation(effective.Tessellation)) result.ExternalInputs|=Mask(Resource::TessellatedGeometry);
    if (effective.VirtualGeometry.Enabled || effective.Nanite.Enabled) result.ExternalInputs|=Mask(Resource::MeshClusters)|Mask(Resource::VirtualizedGeometry);
    const ResourceMask preparedGeometry=(effective.VirtualGeometry.Enabled || effective.Nanite.Enabled)?Mask(Resource::VirtualizedGeometry):UsesDisplacementMapping(effective.Displacement)?Mask(Resource::DisplacedGeometry):
        UsesTessellation(effective.Tessellation)?Mask(Resource::TessellatedGeometry):G;
    ResourceMask visibleResources = result.ExternalInputs;
    const auto append = [&](PassKind kind, ResourceMask reads, ResourceMask writes)
    {
        if (reads&G) reads=(reads&~G)|preparedGeometry;
        if ((reads & ~visibleResources) != 0)
            result.Diagnostics.emplace_back("A pass reads resources before their producer.");
        result.Passes.push_back({kind, reads, writes});
        result.AllocatedResources |= writes;
        visibleResources |= writes;
    };
    if (UsesPathTracing(effective.Pipeline))
    {
        result.ExternalInputs |= AS | History; visibleResources |= AS | History;
        append(PassKind::PathTracing,Surface | AS | History,HDR | Depth);
        if (effective.PostProcessing.DepthOfField.Enabled) append(PassKind::DepthOfField,HDR | Depth | C,HDR);
        if (effective.PostProcessing.TemporalEffects.Enabled) append(PassKind::TemporalEffects,HDR,HDR | Mask(Resource::ExposureHistory));
        result.HDRPasses=BuildHdrPlan(effective.PostProcessing.HDR);
        result.RhiResolveReads=HDR | (effective.PostProcessing.TemporalEffects.Enabled?Mask(Resource::ExposureHistory):0); result.RhiResolveWrites=Color; result.AllocatedResources|=Color;
        return result;
    }
    if (UsesParallaxOcclusion(effective.Parallax)) { result.ExternalInputs|=Mask(Resource::HeightMap); visibleResources|=Mask(Resource::HeightMap); }
    const bool hair=UsesHairFur(effective.Hair);
    const ResourceMask hairInput=hair?Mask(Resource::HairStrands):0;
    result.ExternalInputs|=hairInput; visibleResources|=hairInput;
    const bool environmentCapture = UsesEnvironmentTexture(effective.Reflections);
    if (environmentCapture) append(PassKind::EnvironmentCapture, Surface | hairInput, Mask(Resource::Environment));
    const bool planarCapture = UsesPlanarCapture(effective.Reflections);
    if (planarCapture) append(PassKind::PlanarCapture, Surface | hairInput, Mask(Resource::PlanarRadiance));
    const bool shadowPass = shadowsEnabled && !wireframeOnly;
    const bool rayShadow=shadowPass && effective.Shadows==ShadowRepresentation::RayTraced;
    const ResourceMask shadowVisibility = rayShadow ? AS : effective.ShadowFilter.Mode == ShadowFilterMode::VSM ? Mask(Resource::ShadowMoments) : Shadow;
    if (rayShadow) { result.ExternalInputs|=AS; visibleResources|=AS; }
    else if (shadowPass && effective.Shadows==ShadowRepresentation::Virtual)
    {
        // The immutable atlas is prepared from this camera and geometry before
        // BeginFrame, like other native geometry publications. No single-map
        // pass may overwrite its exclusive receiver bindings inside the frame.
        result.ExternalInputs|=Shadow; visibleResources|=Shadow;
    }
    else if (shadowPass) append(PassKind::ShadowDepth, G | M | L | hairInput |
        (effective.Shadows == ShadowRepresentation::Cascaded ? C : 0), Shadow | shadowVisibility);
    const bool hdrEnabled = RequiresHDR(effective.PostProcessing.HDR);
    const ResourceMask sceneColor = hdrEnabled ? HDR : Color;
    const bool multisample = effective.SampleCount > 1;
    const ResourceMask rasterColor = multisample ? Mask(Resource::MultisampleColor) : sceneColor;
    const ResourceMask rasterDepth = multisample ? Mask(Resource::MultisampleDepth) : Depth;
    const ResourceMask materialInputs = multisample ? Mask(Resource::MultisampleGBuffer) : Mask(Resource::GBuffer) | N;
    const bool ambientOcclusion = UsesAmbientOcclusion(effective.AmbientOcclusion);
    const bool reflections = UsesReflections(effective.Reflections);
    const bool fog = UsesVolumetricFog(effective.VolumetricFog);
    const bool contact = effective.ContactShadows.Enabled;
    const bool subsurface = UsesSubsurfaceScattering(effective.Subsurface);
    const bool forwardPlus = effective.Pipeline == RenderingPipeline::ForwardPlus;
    const bool tiledDeferred = effective.Pipeline == RenderingPipeline::TiledDeferred;
    const bool clustered = effective.Pipeline == RenderingPipeline::ClusteredForward;
    const ResourceMask lightLists = UsesLightLists(effective.Pipeline) ? Mask(Resource::LightLists) : 0;
    // Dense clusters depend only on the camera and frozen light collection.
    if (clustered) append(PassKind::ClusteredAssignment, C | L, lightLists);
    if (hair && (contact || forwardPlus)) append(forwardPlus?PassKind::ForwardPlusDepth:PassKind::ContactDepth,Surface | Mask(Resource::HairStrands),rasterDepth);
    if ((ambientOcclusion || reflections || globalIllumination || hair) && !UsesDeferredLighting(effective.Pipeline))
        append(PassKind::MaterialInputs, Surface, materialInputs | rasterDepth);
    else if (forwardPlus && !hair)
        append(PassKind::ForwardPlusDepth, Surface, rasterDepth);
    else if (contact && !hair && (effective.Pipeline == RenderingPipeline::Forward || clustered))
        append(PassKind::ContactDepth, Surface, rasterDepth);
    if (UsesDeferredLighting(effective.Pipeline))
    {
        append(UsesVisibilityBuffer(effective.Pipeline) ? PassKind::VisibilityGeometry : PassKind::DeferredGeometry,
            Surface | (UsesParallaxOcclusion(effective.Parallax)?Mask(Resource::HeightMap):0), materialInputs | rasterDepth | (UsesVisibilityBuffer(effective.Pipeline) ? Mask(Resource::VisibilitySamples) : 0));
    }
    if (forwardPlus || tiledDeferred) append(tiledDeferred ? PassKind::TiledDeferredAssignment : PassKind::ForwardPlusAssignment, rasterDepth | C | L, lightLists);
    if (contact) append(PassKind::ContactVisibility, rasterDepth | C | L, Mask(Resource::ContactVisibility));
    const ResourceMask contactInput = contact ? Mask(Resource::ContactVisibility) : 0;
    if (UsesDeferredLighting(effective.Pipeline))
        append(PassKind::DeferredLighting, materialInputs | rasterDepth | C | L | lightLists | (shadowPass ? shadowVisibility : 0) | contactInput, rasterColor);
    else append(PassKind::OpaqueForward, Surface | lightLists | (shadowPass ? shadowVisibility : 0) | contactInput | ((contact || forwardPlus) ? rasterDepth : 0), rasterColor | rasterDepth);
    if (hair) append(PassKind::HairFur,Mask(Resource::HairStrands) | C | L | materialInputs | rasterDepth | rasterColor |
        (shadowPass?shadowVisibility:0) | contactInput,rasterColor | rasterDepth | materialInputs);
    if (multisample)
    {
        const bool materialCapture = ambientOcclusion || reflections || globalIllumination || hair || UsesDeferredLighting(effective.Pipeline);
        append(PassKind::ResolveOpaqueSamples, rasterColor | rasterDepth | (materialCapture ? materialInputs : 0),
            sceneColor | Depth | (materialCapture ? Mask(Resource::GBuffer) | N : 0));
    }
    if (subsurface)
        append(PassKind::SubsurfaceScattering, Mask(Resource::GBuffer) | N | Depth | C | L | HDR |
            (multisample ? materialInputs | rasterDepth : 0) | (shadowPass ? shadowVisibility : 0) | contactInput,
            HDR | Mask(Resource::DiffuseRadiance));
    if (ambientOcclusion)
        append(PassKind::AmbientOcclusion, Mask(Resource::GBuffer) | Depth | C | HDR | (rayAO ? AS : 0) |
                (effective.AmbientOcclusion.Mode == AmbientOcclusionMode::SAO ? 0 : N),
            Mask(Resource::AmbientVisibility) | HDR |
                (effective.AmbientOcclusion.Mode == AmbientOcclusionMode::SAO ? Mask(Resource::LinearDepthHierarchy) : 0));
    const auto appendVelocity = [&]()
    {
        append(PassKind::Velocity, G | M | C | hairInput | rasterDepth | Mask(Resource::PreviousTransforms),
            multisample ? Mask(Resource::MultisampleVelocity) : Velocity);
        if (multisample) append(PassKind::ResolveVelocitySamples, Mask(Resource::MultisampleVelocity) | rasterDepth, Velocity);
    };
    if (earlyVelocity) appendVelocity();
    const auto translucentIrradiance=lumen && (effective.AlphaBlending || UsesRefraction(effective)) && effective.GlobalIllumination.Lumen.Translucency.Enabled ? Mask(Resource::TranslucentIrradiance) : ResourceMask(0);
    const bool hairIndirect=lumen && hair && effective.GlobalIllumination.Lumen.HairIndirect;
    // One SH field serves independent mesh and fiber receivers. Only enabled
    // transparent materials declare a read of it in later forward passes.
    const auto volumeIrradiance=(translucentIrradiance || hairIndirect)?Mask(Resource::TranslucentIrradiance):ResourceMask(0);
    const auto giHairInput=lumen && (hairIndirect || effective.GlobalIllumination.Lumen.HairTransport)?hairInput:ResourceMask(0);
    if (globalIllumination)
    {
        append(PassKind::GlobalIllumination, Mask(Resource::GBuffer) | N | Depth | C | HDR | giHairInput |
            (temporalGI ? Velocity | Mask(Resource::GlobalIlluminationHistory) : 0) |
            (screenGI ? 0 : rayGI ? AS | L : lumen ? Mask(Resource::SurfaceRadianceCache) : voxelGI ? Mask(Resource::Voxels) : Mask(Resource::BakedLighting)) |
            (lightmaps ? Mask(Resource::LightmapUV) : 0) | (lumenHardware?AS:0),
            HDR | Mask(Resource::IndirectRadiance) | volumeIrradiance | (bakedProbes ? Mask(Resource::Probes) : 0));
    }
    if (reflections)
        append(UsesHybridRendering(effective.Pipeline) ? PassKind::HybridReflections : PassKind::Reflections, Mask(Resource::GBuffer) | N | Depth | C | HDR |
            (environmentCapture ? Mask(Resource::Environment) : 0) | (planarCapture ? Mask(Resource::PlanarRadiance) : 0) |
            (rayReflections ? AS | L : 0) | (lumenReflections?Mask(Resource::SurfaceRadianceCache) | (lumenHardware?AS:0):0) |
            (temporalReflections?Velocity | Mask(Resource::ReflectionHistory):0),
            Mask(Resource::ReflectionRadiance) | HDR);
    if (effective.Reflections.Skybox || effective.Reflections.DiffuseIBL>0 || effective.Reflections.Atmosphere.Enabled) append(PassKind::Skybox,HDR | Depth | C | N | Mask(Resource::GBuffer) | Mask(Resource::Environment),HDR);
    if (fog)
    {
        const bool lighting = UsesVolumetricLighting(effective.VolumetricLighting);
        append(UsesVolumetricClouds(effective.VolumetricFog) ? PassKind::VolumetricClouds :
            UsesParticipatingMedia(effective.VolumetricFog) ? PassKind::ParticipatingMedia :
            lighting ? PassKind::VolumetricLighting : PassKind::VolumetricFog,
            HDR | Depth | C | (lighting ? L | (shadowPass ? shadowVisibility : 0) : 0), HDR | Mask(Resource::FogVolume));
    }
    if (UsesLightShafts(effective.LightShafts)) append(PassKind::LightShafts, HDR | Depth | C | L, HDR);
    if (RequiresMotionVectors(effective) && !earlyVelocity) appendVelocity();
    if (temporalAA) append(PassKind::TemporalAA, HDR | Temporal | C, HDR);
    if (effective.PostProcessing.MotionBlur.Enabled) append(PassKind::MotionBlur, HDR | Depth | Velocity | C, HDR);
    if (effective.PostProcessing.DepthOfField.Enabled)
        append(PassKind::DepthOfField, HDR | Depth | C, HDR);
    if (temporalSuperResolution) append(PassKind::SnapshotTSROpaque,HDR | Depth | Velocity,Mask(Resource::OpaqueHDRSnapshot));
    // Refractive materials sample opaque HDR. Alpha/OIT then handles its own
    // material set; separated demo objects do not require cross-method sorting.
    if (UsesRefraction(effective)) append(PassKind::Refraction, G | C | M | sceneColor | rasterDepth | translucentIrradiance, sceneColor);
    if (effective.AlphaBlending)
    {
        if (multisample && (subsurface || globalIllumination || ambientOcclusion || reflections || fog || UsesLightShafts(effective.LightShafts) || temporalAA || effective.PostProcessing.MotionBlur.Enabled || effective.PostProcessing.DepthOfField.Enabled))
            append(PassKind::RestoreColorSamples, sceneColor, rasterColor);
        const bool weighted = UsesWeightedTransparency(effective);
        const ResourceMask transparency = Mask(Resource::TransparencyStorage);
        if (UsesDepthPeeling(effective)) append(PassKind::DepthPeeling, Surface | lightLists | sceneColor | rasterDepth | translucentIrradiance |
            (shadowPass ? shadowVisibility : 0) | (fog ? Mask(Resource::FogVolume) : 0), sceneColor | transparency);
        else
        {
            if (weighted) append(PassKind::PrepareWeightedTransparency, rasterDepth, transparency);
            append(PassKind::TransparentForward, Surface | lightLists | translucentIrradiance | (weighted ? transparency : rasterColor) | rasterDepth | (shadowPass ? shadowVisibility : 0) |
                (fog ? Mask(Resource::FogVolume) : 0), weighted ? transparency : rasterColor);
            if (weighted) append(PassKind::ResolveWeightedTransparency, transparency | rasterColor, rasterColor);
        }
        if (multisample) append(PassKind::ResolveColorSamples, rasterColor, sceneColor);
    }
    if (effective.PostProcessing.TemporalEffects.Enabled) append(PassKind::TemporalEffects, HDR, HDR | Mask(Resource::ExposureHistory));
    result.HDRPasses = BuildHdrPlan(effective.PostProcessing.HDR);
    if (!result.HDRPasses.empty())
    {
        result.RhiResolveReads = HDR | (effective.PostProcessing.TemporalEffects.Enabled ? Mask(Resource::ExposureHistory) : 0);
        result.RhiResolveWrites = Color;
        if (temporalSuperResolution)
        {
            result.RhiResolveReads|=C | Depth | Velocity | Mask(Resource::OpaqueHDRSnapshot) | Mask(Resource::ReconstructionHistory);
            result.RhiResolveWrites|=Mask(Resource::PresentationHDR) | Mask(Resource::PresentationDepth) | Mask(Resource::PresentationVelocity) | Mask(Resource::ReconstructionHistory);
            result.AllocatedResources|=Mask(Resource::PresentationHDR);
        }
        if ((result.RhiResolveReads & ~visibleResources) != 0)
            result.Diagnostics.emplace_back("The RHI HDR resolve has no floating-point scene producer.");
        result.AllocatedResources |= result.RhiResolveWrites;
        visibleResources |= result.RhiResolveWrites;
    }
    if (outlineEnabled) append(PassKind::Outline, G | C | M | Color | Depth, Color);
    return result;
}

} // namespace Evans::Techniques
