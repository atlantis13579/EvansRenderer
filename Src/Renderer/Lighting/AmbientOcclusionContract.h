#pragma once

#include "Renderer/Pipelines/DeferredContract.h"
#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    // Screen-space AO consumes opaque depth, world normals and ambient radiance from the
    // material buffer. Forward records an extra material prepass; deferred reuses
    // its existing geometry. Neither changes the requested shading model.
    // SSAO projects a deterministic, rotated cosine hemisphere into opaque depth.
    // Offscreen/clear-depth samples are unoccluded; nearby occluders are weighted
    // by world-space distance. Two separable depth/normal bilateral passes remove
    // sampling noise without mixing across silhouettes. No noise texture is needed.
    // Mittring, "Finding Next Gen - CryEngine 2", SIGGRAPH 2007 course notes, 8.5.4.3.
    // https://advances.realtimerendering.com/s2007/Mittring-Finding_NextGen_CryEngine2(Siggraph07%20Course%20Notes).pdf
    // HBAO instead searches eight screen directions for increasing horizons.
    // Its weighted sine-angle increments are measured above the surface tangent.
    // Bavoil and Sainz, "Image-Space Horizon-Based Ambient Occlusion", SIGGRAPH 2008.
    // https://developer.download.nvidia.com/presentations/2008/SIGGRAPH/HBAO_SIG08b.pdf
    //
    // Composite outputs lost ambient = ambient * (1 - visibility). RGB-only
    // reverse-subtract blending applies it to the loaded HDR scene attachment.
    // This avoids sampling the render target and preserves direct light, emission
    // and alpha. Negative-roughness wire pixels and clear depth contribute zero.
    // Both AO ping-pong images are full-resolution RGBA16F; only R is consumed.
    // Texture order: opaque depth or SAO hierarchy, normal/roughness material
    // marker, ambient/metallic, previous AO. SAO uses only the normal buffer's
    // unlit marker; its estimate and bilateral weights derive geometry from depth.
    // SAO first builds an R32F mip chain of near-plane-relative linear depth.
    // Rotated-grid decimation preserves real depth samples; the shader recovers
    // each selected representative's base pixel before reconstructing position.
    // This sample hierarchy is not conservative Hi-Z: visibility culling and
    // ray traversal need a separate min/max reduction contract.
    // A separate render scratch and per-level copy avoid texture feedback on all
    // APIs. Nine spiral taps select levels by screen distance and derive normals
    // from depth, following McGuire, Mara and Luebke, "Scalable Ambient Obscurance"
    // (HPG 2012), with the updated appendix falloff and bounded edge-aware normals.
    // https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
    // GTAO analytically integrates cosine-weighted visibility between signed
    // horizons around the per-pixel view ray. Periodic clipping preserves the
    // supplied normal orientation; an unoccluded reference removes finite-slice
    // baseline error. Spatial filtering is independent of optional temporal reuse.
    // Texel snapping preserves the sample's height above the material tangent
    // plane when projecting it into a slice. A binary16-derived uncertainty cone
    // rejects slopes indistinguishable from the G-buffer's normal encoding error.
    // Jimenez et al., "Practical Realtime Strategies for Accurate Indirect
    // Occlusion", SIGGRAPH 2016, equations 5-8.
    // https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
    // Geometry must finish before AO; transparency follows its ambient composite.
    inline constexpr std::uint32_t AmbientOcclusionPassCount = 4;
    using Techniques::AmbientOcclusionDepthMipCount;
    struct AmbientOcclusionConstants
    {
        Matrix4 ViewProjection;
        Matrix4 InverseViewProjection;
        Float4 ViewportRadiusBias; // inverse width/height, world radius, world bias or HBAO angle radians
        Float4 ForwardIntensity; // normalized world camera forward, AO strength
        ShaderUInt4 Options; // pass kind, destination depth mip, maximum depth mip, reserved
    };
    static_assert(sizeof(AmbientOcclusionConstants) == 176, "AO native uniform layout changed");
    static_assert(offsetof(AmbientOcclusionConstants, Options) == 160, "AO options offset changed");

    inline bool BuildAmbientOcclusionConstants(const AmbientOcclusionDesc& desc,
        const Techniques::AmbientOcclusionSettings& requested, int width, int height,
        Techniques::AmbientOcclusionPassKind pass, AmbientOcclusionConstants& output)
    {
        if (width <= 0 || height <= 0) return false;
        DeferredLightingDesc camera;
        camera.ViewProjection = desc.ViewProjection;
        DeferredConstants inverse;
        if (!BuildDeferredConstants(camera, false, {}, inverse)) return false;
        const auto& values = inverse.InverseViewProjection.Values;
        constexpr float HomogeneousEpsilon = 1.0e-7f;
        constexpr float InteriorClipDepth = 0.5f;
        const float nearW = values[15];
        // An interior clip-depth point also supports an infinite far plane.
        const float interiorW = values[14] * InteriorClipDepth + values[15];
        if (std::abs(nearW) < HomogeneousEpsilon || std::abs(interiorW) < HomogeneousEpsilon) return false;
        Float3 forward;
        forward.X = (values[2] * InteriorClipDepth + values[3]) / interiorW - values[3] / nearW;
        forward.Y = (values[6] * InteriorClipDepth + values[7]) / interiorW - values[7] / nearW;
        forward.Z = (values[10] * InteriorClipDepth + values[11]) / interiorW - values[11] / nearW;
        if (requested.Mode == Techniques::AmbientOcclusionMode::HBAO ||
            requested.Mode == Techniques::AmbientOcclusionMode::SAO ||
            requested.Mode == Techniques::AmbientOcclusionMode::GTAO)
        {
            // Horizons and linear Z use optical eye Z. The center viewing ray is not
            // perpendicular to the near plane for an off-axis projection.
            const float horizontalW = values[12] + values[15];
            const float verticalW = values[13] + values[15];
            if (std::abs(horizontalW) < HomogeneousEpsilon || std::abs(verticalW) < HomogeneousEpsilon) return false;
            Float3 horizontal = {(values[0] + values[3]) / horizontalW - values[3] / nearW,
                (values[4] + values[7]) / horizontalW - values[7] / nearW,
                (values[8] + values[11]) / horizontalW - values[11] / nearW};
            Float3 vertical = {(values[1] + values[3]) / verticalW - values[3] / nearW,
                (values[5] + values[7]) / verticalW - values[7] / nearW,
                (values[9] + values[11]) / verticalW - values[11] / nearW};
            // Normalize before the cross product: its length must measure an
            // angle, not near-plane area. Scaling first also avoids underflow.
            const auto normalizeTangent = [](Float3& tangent)
            {
                if (!std::isfinite(tangent.X) || !std::isfinite(tangent.Y) || !std::isfinite(tangent.Z)) return false;
                const float scale = std::max({std::abs(tangent.X), std::abs(tangent.Y), std::abs(tangent.Z)});
                if (scale <= 0.0f) return false;
                tangent = {tangent.X / scale, tangent.Y / scale, tangent.Z / scale};
                const float length = std::sqrt(tangent.X * tangent.X + tangent.Y * tangent.Y + tangent.Z * tangent.Z);
                tangent = {tangent.X / length, tangent.Y / length, tangent.Z / length};
                return true;
            };
            if (!normalizeTangent(horizontal) || !normalizeTangent(vertical)) return false;
            Float3 axis = {horizontal.Y * vertical.Z - horizontal.Z * vertical.Y,
                horizontal.Z * vertical.X - horizontal.X * vertical.Z,
                horizontal.X * vertical.Y - horizontal.Y * vertical.X};
            const float orientation = axis.X * forward.X + axis.Y * forward.Y + axis.Z * forward.Z;
            if (orientation < 0.0f) axis = {-axis.X, -axis.Y, -axis.Z};
            forward = axis;
        }
        const float squared = forward.X * forward.X + forward.Y * forward.Y + forward.Z * forward.Z;
        if (!std::isfinite(squared) || squared <= HomogeneousEpsilon * HomogeneousEpsilon) return false;
        const float inverseLength = 1.0f / std::sqrt(squared);
        const auto settings = Techniques::SanitizeAmbientOcclusionSettings(requested);
        output.ViewProjection = desc.ViewProjection;
        output.InverseViewProjection = inverse.InverseViewProjection;
        constexpr float DegreesToRadians = 3.14159265358979323846f / 180.0f;
        const float bias = settings.Mode == Techniques::AmbientOcclusionMode::HBAO
            ? settings.HorizonAngleBiasDegrees * DegreesToRadians : settings.Bias;
        output.ViewportRadiusBias = {1.0f / width, 1.0f / height, settings.Radius, bias};
        output.ForwardIntensity = {forward.X * inverseLength, forward.Y * inverseLength,
            forward.Z * inverseLength, settings.Intensity};
        output.Options = {static_cast<std::uint32_t>(pass), 0, 0, 0};
        return true;
    }
}
