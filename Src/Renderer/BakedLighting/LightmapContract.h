#pragma once
#include "Renderer/BakedLighting/BakedLightmapCache.h"
#include "Renderer/FrameHistory.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    enum class LightmapPass : std::uint32_t { Surface = 0, Composite = 1 };
    struct LightmapDraw
    {
        std::uint32_t FirstVertex = 0, VertexCount = 0;
        Matrix4 World;
        Float4 DiffuseColor = {1,1,1,1}; // Linear instance * material color; alpha includes both opacities.
        TextureHandle DiffuseTexture, OpacityTexture;
        float AlphaCutoff = 0;
        bool BackFaceCulling = true;
        bool PackedMetallicRoughness = false;
        std::optional<float> Metallic;
    };
    struct LightmapFrame
    {
        std::shared_ptr<const LightmapBakeResult> Source;
        std::vector<LightmapDraw> Draws;
        // LESS depth writes retain the first coplanar surface; forward shading
        // over a depth prepass uses LESS_EQUAL and retains the last. A coverage
        // alpha latch reproduces the first case without reversing primitives.
        bool KeepFirstSurface = true;
    };
    struct LightmapConstants
    {
        Matrix4 World, ViewProjection;
        Float4 DiffuseColor;
        Float4 Appearance; // Intensity, alpha cutoff, diffuse BRDF scale, atlas side.
        ShaderUInt4 Options; // Surface: diffuse map, opacity map, back hemisphere, pass.
        // Composite: X enables screen AO; texture slot zero supplies visibility.
    };
    static_assert(sizeof(LightmapConstants) == 176 && offsetof(LightmapConstants,Options) == 160,
        "Lightmap native uniform ABI changed");

    inline void RecordLightmapDraws(const LightmapFrame& frame, FrameStatistics& statistics)
    {
        for (const auto& draw:frame.Draws) statistics.LightmapReceiverDrawCount += draw.VertexCount ? 1u : 0u;
        statistics.DrawCallCount += statistics.LightmapReceiverDrawCount;
    }

    inline bool BuildLightmapConstants(const LightmapFrame& frame, const LightmapDraw& draw, const Matrix4& projection,
        float intensity, Techniques::ShadingModel shading, float metallic, LightmapConstants& output)
    {
        if (!frame.Source || !frame.Source->Receivers || !frame.Source->Irradiance ||
            !frame.Source->Irradiance->Atlas || !FrameHistory::IsFinite(draw.World) || !FrameHistory::IsFinite(projection)) return false;
        const auto& vertices = frame.Source->Receivers->Vertices;
        if (draw.FirstVertex > vertices.size() || draw.VertexCount > vertices.size()-draw.FirstVertex || draw.VertexCount%3) return false;
        constexpr float DielectricReflectance = 0.04f;
        LightmapConstants constants;
        constants.World = draw.World; constants.ViewProjection = projection; constants.DiffuseColor = draw.DiffuseColor;
        const bool mapped = shading == Techniques::ShadingModel::PBR && draw.PackedMetallicRoughness && draw.OpacityTexture;
        metallic = std::clamp(draw.Metallic.value_or(metallic), 0.0f, 1.0f);
        constants.Appearance = {std::max(intensity,0.0f),std::clamp(draw.AlphaCutoff,0.0f,1.0f),
            mapped ? metallic : shading == Techniques::ShadingModel::PBR ? (1-DielectricReflectance)*(1-metallic) : 1.0f,
            float(frame.Source->Irradiance->Atlas->Settings.Resolution)};
        constants.Options = {draw.DiffuseTexture ? 1u : 0u,mapped ? 3u : draw.OpacityTexture ? 1u : 0u,
            frame.Source->Irradiance->BackIrradiance.empty() ? 0u : 1u,static_cast<std::uint32_t>(LightmapPass::Surface)};
        output = constants; return true;
    }
}
