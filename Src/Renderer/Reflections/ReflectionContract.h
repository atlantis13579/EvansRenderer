#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing", JCGT (2014).
    // https://jcgt.org/published/0003/04/04/paper.pdf
    // Clip a world-space reflected ray to the view volume, then traverse its
    // projected major axis. Interpolating world/w and 1/w reconstructs each
    // ray interval without the oversampling of uniform world-distance steps.
    // A world-unit depth slab bounds hits; normals reject back-facing receivers.
    // This spatial mirror estimator attenuates rough surfaces, screen edges and
    // distant hits. It has no roughness convolution, hidden-surface fallback,
    // temporal accumulation or recursive reflections. Missing hits add zero.
    //
    // Forward captures materials; Deferred reuses its geometry. After AO, trace
    // reads pre-reflection HDR into a separate RGBA16F image. RGB-only additive
    // composition preserves alpha/depth before velocity, blur and transparency.
    // MSAA supplies nearest-depth materials; processed color must be restored
    // to sample storage before transparency. No external texture is required.
    // Texture order: scene HDR, opaque depth, normal/roughness, specular/shininess,
    // diffuse/opacity, ambient/metallic, reflection radiance. Phase 1 reads only
    // the last texture, so it never samples the active HDR render target.
    struct ReflectionConstants
    {
        Matrix4 ViewProjection;
        Matrix4 InverseViewProjection;
        Float4 Viewport; // extent XY, reciprocal extent ZW
        Float4 Trace; // world distance, depth thickness, normal bias, pixel stride
        Float4 Appearance; // intensity, roughness cutoff, edge fade pixels, reserved
        ShaderUInt4 Options; // phase, maximum steps, shading model, reserved
    };
    static_assert(sizeof(ReflectionConstants) == 192, "Reflection uniform layout changed");
    static_assert(offsetof(ReflectionConstants, Options) == 176, "Reflection options offset changed");

    inline bool BuildReflectionConstants(const ReflectionDesc& desc,
        const Techniques::ReflectionSettings& requested, Techniques::ShadingModel shading,
        int width, int height, ReflectionConstants& output)
    {
        if (width <= 0 || height <= 0) return false;
        for (float value : desc.ViewProjection.Values) if (!std::isfinite(value)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        ReflectionConstants candidate;
        candidate.ViewProjection = desc.ViewProjection;
        std::copy(inverse.Data(), inverse.Data() + 16, candidate.InverseViewProjection.Values.begin());
        for (float value : candidate.InverseViewProjection.Values) if (!std::isfinite(value)) return false;
        const auto s = Techniques::SanitizeReflectionSettings(requested);
        candidate.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        candidate.Trace = {s.MaximumDistance, s.Thickness, s.NormalBias, s.StridePixels};
        candidate.Appearance = {s.Intensity, s.MaximumRoughness, s.EdgeFadePixels, 0};
        candidate.Options = {0, s.MaximumSteps, static_cast<std::uint32_t>(shading), 0};
        output = candidate;
        return true;
    }
}
