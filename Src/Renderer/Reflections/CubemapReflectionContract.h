#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Greene, "Environment Mapping and Other Applications of World Projections"
    // (1986). Six HDR faces approximate directional incident radiance
    // at one probe. Reflect the camera ray about the receiver normal, convert
    // world Z to native cube Z, then apply Schlick Fresnel to the sampled RGB.
    // Ordinary mip averaging broadens rough reflections; it is not GGX importance
    // sampling or a split-sum BRDF integral. Nearby geometry retains parallax error.
    // An authored RGBE panorama or a six-view opaque scene capture supplies
    // radiance. Scene captures omit shadows, transparency and recursive reflections.
    // Lookup follows opaque AO and writes a private image before RGB-only HDR
    // addition. MSAA supplies nearest-depth materials, retaining per-sample depth.
    struct CubemapReflectionConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport; // extent XY, inverse extent ZW
        Float4 Sampling; // intensity, maximum mip level, sky intensity, background-only flag
        ShaderUInt4 Options; // phase, shading model, mip-reduction face (DX12/Vulkan), planar receiver mask
        Float4 ReceiverPlane;
        Float4 ReceiverMinimum; // xyz bounds, w plane tolerance
        Float4 ReceiverMaximum; // xyz bounds, w minimum normal cosine
        Float4 AtmospherePlanet; // radius km, thickness km, ground height m, aerial enable
        Float4 AtmosphereRayleigh; // RGB scattering / km, scale height km
        Float4 AtmosphereMie; // scattering / km, extinction / km, scale height km, anisotropy
        Float4 AtmosphereSun; // direction to sun XYZ, radiance multiplier
        Float4 AtmosphereSolar; // solar RGB irradiance, view samples
        Float4 AtmosphereQuality; // solar samples, reserved
    };
    static_assert(sizeof(CubemapReflectionConstants) == 256, "Cube reflection uniform layout changed");

    inline bool BuildCubemapReflectionConstants(const ReflectionDesc& desc,
        const Techniques::ReflectionSettings& requested, Techniques::ShadingModel shading,
        int width, int height, std::uint32_t mipCount, CubemapReflectionConstants& output)
    {
        if (width <= 0 || height <= 0 || mipCount == 0) return false;
        for (float value : desc.ViewProjection.Values) if (!std::isfinite(value)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        CubemapReflectionConstants candidate{};
        std::copy(inverse.Data(), inverse.Data() + 16, candidate.InverseViewProjection.Values.begin());
        for (float value : candidate.InverseViewProjection.Values) if (!std::isfinite(value)) return false;
        candidate.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        const auto settings=Techniques::SanitizeReflectionSettings(requested);
        candidate.Sampling = {desc.DrawSkybox ? settings.DiffuseIBL : settings.EnvironmentIntensity, float(mipCount - 1), desc.DrawSkybox && settings.Skybox ? settings.SkyboxIntensity : 0.0f, desc.DrawSkybox ? 1.0f : 0.0f};
        candidate.Options = {0, static_cast<std::uint32_t>(shading), 0, 0};
        const auto& receiver=settings.Planar;
        candidate.Options.W=receiver.Enabled ? (receiver.BoundedReceiver ? 2u : 1u) : 0u;
        candidate.ReceiverPlane={receiver.Plane[0],receiver.Plane[1],receiver.Plane[2],receiver.Plane[3]};
        candidate.ReceiverMinimum={receiver.ReceiverMinimum[0],receiver.ReceiverMinimum[1],receiver.ReceiverMinimum[2],receiver.ReceiverTolerance};
        candidate.ReceiverMaximum={receiver.ReceiverMaximum[0],receiver.ReceiverMaximum[1],receiver.ReceiverMaximum[2],receiver.MinimumNormalCosine};
        const auto& atmosphere=settings.Atmosphere;
        candidate.AtmospherePlanet={atmosphere.PlanetRadius*0.001f,atmosphere.Height*0.001f,atmosphere.GroundHeight,
            desc.DrawSkybox && atmosphere.Enabled && atmosphere.AerialPerspective ? 1.0f : 0.0f};
        const auto rayleigh=[&](unsigned i) { return 0.0135f*atmosphere.RayleighStrength*std::pow(550.0f/atmosphere.Wavelengths[i],4.0f); };
        candidate.AtmosphereRayleigh={rayleigh(0),rayleigh(1),rayleigh(2),atmosphere.RayleighHeight*0.001f};
        candidate.AtmosphereMie={0.004f*atmosphere.MieStrength,0.0044f*atmosphere.MieStrength,atmosphere.MieHeight*0.001f,atmosphere.MieAnisotropy};
        candidate.AtmosphereSun={atmosphere.SunDirection[0],atmosphere.SunDirection[1],atmosphere.SunDirection[2],atmosphere.Intensity};
        candidate.AtmosphereSolar={atmosphere.SunRadiance[0],atmosphere.SunRadiance[1],atmosphere.SunRadiance[2],float(atmosphere.ViewSamples)};
        candidate.AtmosphereQuality={float(atmosphere.LightSamples),0,0,0};
        output = candidate;
        return true;
    }
}
