#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Five RGBA16F attachments, nearest texel reads, and the opaque Depth32:
    // 0: linear diffuse reflectance + opacity; 1: world normal + roughness;
    // 2: specular reflectance + shininess; 3: ambient radiance + metallic;
    // 4: emission + material diffusion coverage. Keeping ambient separate lets AO attenuate ambient
    // without incorrectly darkening emissive surfaces. This teaching layout
    // favors explicit material semantics over bandwidth; MRT capability matters.
    // Negative roughness marks unlit wire overlays. Geometry owns normal mapping
    // and cutouts; sorted transparent objects use forward after lighting and keep
    // opaque depth. Gouraud is incompatible with deferred per-pixel illumination.
    // See Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
    // https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
    inline constexpr std::size_t GBufferAttachmentCount = 5;
    struct DeferredConstants
    {
        Matrix4 InverseViewProjection;
        Matrix4 LightViewProjection;
        Float4 CameraLightCount;
        Float4 ShadingParameters;
        // Shading model, shadow availability, shadow light index, DeferredOutputMode.
        ShaderUInt4 Options;
        Float4 BackgroundColor;
        std::array<ShaderLight, MaximumSceneLights> Lights;
    };
    static_assert(sizeof(DeferredConstants) == 448, "Deferred native uniform layout changed");
    static_assert(offsetof(DeferredConstants, Lights) == 192, "Deferred light offset changed");

    inline bool BuildDeferredConstants(const DeferredLightingDesc& desc,
        bool shadowAvailable, const Float4& background, DeferredConstants& result)
    {
        result = {};
        if (desc.Output != DeferredOutputMode::Scene && desc.Output != DeferredOutputMode::DirectDiffuse) return false;
        for (float value : desc.ViewProjection.Values) if (!std::isfinite(value)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        std::copy(inverse.Data(), inverse.Data() + result.InverseViewProjection.Values.size(),
            result.InverseViewProjection.Values.begin());
        for (float value : result.InverseViewProjection.Values) if (!std::isfinite(value)) return false;
        result.LightViewProjection = desc.LightViewProjection;
        PipelineState pipeline;
        pipeline.Lighting = desc.Lighting;
        const auto draw = BuildRealtimeDrawConstants(pipeline, Matrix4::Identity(),
            desc.ViewProjection, desc.LightViewProjection, PassType::Color, false, false, false, false);
        result.CameraLightCount = draw.CameraLightCount;
        result.Lights = draw.Lights;
        result.ShadingParameters = {desc.ToonBands, 0.0f, 0.0f, 0.0f};
        result.Options = {static_cast<std::uint32_t>(desc.Shading),
            desc.SamplesShadowMap && shadowAvailable ? 1u : 0u,
            static_cast<std::uint32_t>(desc.Lighting.ShadowLightIndex), static_cast<std::uint32_t>(desc.Output)};
        result.BackgroundColor = background;
        return true;
    }
}
