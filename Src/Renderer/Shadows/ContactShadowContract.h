#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Shadows/ShadowProjection.h"

#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t MaximumContactShadowSteps = 256;
    struct ContactShadowParameters
    {
        double StartBias = 0.01; // world distance along the light ray
        double Thickness = 0.05; // camera-axis depth behind the visible layer
        std::uint32_t MaximumSteps = 64; // visited screen cells, including the receiver cell
    };

    struct ContactShadowDispatch
    {
        Matrix4 ViewProjection;
        SceneLight Light; // selected light; finite emitters use their centre
        ContactShadowParameters Parameters;
        float MaximumDistance = 0.25f;
    };
    // API-independent data layout; every native shader is authored separately.
    // One visibility layer per depth sample preserves MSAA coverage. Consumers
    // read their own sample; resolving the mask before lighting is incorrect.
    struct ContactShadowConstants
    {
        Matrix4 ViewProjection, InverseViewProjection;
        Float4 DepthMetric; // near, far, perspective flag, positive clip-W guard
        Float4 LightPositionType, LightDirection;
        Float4 TraceParameters; // maximum distance, start bias, depth thickness, reserved
        ShaderUInt4 ImageAndBudget; // width, height, maximum visited cells, samples
    };
    static_assert(sizeof(ContactShadowConstants) == 208 && offsetof(ContactShadowConstants, ImageAndBudget) == 192,
        "Contact-shadow native constant layout changed");

    inline bool BuildContactShadowConstants(const ContactShadowDispatch& desc, std::uint32_t width,
        std::uint32_t height, std::uint32_t samples, ContactShadowConstants& output, std::string& error)
    {
        output = {}; error.clear();
        const auto fail = [&](const char* text) { error = text; return false; };
        const auto nativeNonnegative = [](double value)
        {
            return std::isfinite(value) && value >= 0 && value <= std::numeric_limits<float>::max() &&
                (value == 0 || value >= std::numeric_limits<float>::min());
        };
        if (!width || !height || (samples != 1 && samples != 2 && samples != 4 && samples != 8) ||
            !desc.Parameters.MaximumSteps || desc.Parameters.MaximumSteps > MaximumContactShadowSteps ||
            !nativeNonnegative(desc.Parameters.StartBias) || !nativeNonnegative(desc.Parameters.Thickness) ||
            !nativeNonnegative(desc.MaximumDistance) || desc.MaximumDistance == 0)
            return fail("Contact-shadow dimensions, sample count and trace parameters must be representable and bounded");
        const bool directional = desc.Light.Type == LightType::Directional;
        if (desc.Light.Type > LightType::RectArea) return fail("Unknown contact-shadow light type");
        const Float3 location = directional ? desc.Light.Direction : desc.Light.Position;
        if (!std::isfinite(location.X) || !std::isfinite(location.Y) || !std::isfinite(location.Z))
            return fail("Contact-shadow light location must be finite");
        const double directionLength = directional ? std::hypot(double(location.X), double(location.Y), double(location.Z)) : 1;
        if (directionLength == 0) return fail("Contact-shadow light direction must be nonzero");
        ShadowProjection metric;
        if (!BuildShadowProjection(desc.ViewProjection, metric, error)) return false;
        ContactShadowConstants candidate;
        for (std::size_t index = 0; index < candidate.ViewProjection.Values.size(); ++index)
        {
            const double normalized = desc.ViewProjection.Values[index] / metric.ClipWScale;
            if (!std::isfinite(normalized) || std::abs(normalized) > std::numeric_limits<float>::max())
                return fail("Contact-shadow normalized projection is not representable");
            candidate.ViewProjection.Values[index] = float(normalized);
        }
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(candidate.ViewProjection.Values).Inverse(inverse))
            return fail("Contact-shadow projection cannot be inverted");
        std::copy(inverse.Data(), inverse.Data() + 16, candidate.InverseViewProjection.Values.begin());
        for (float value : candidate.InverseViewProjection.Values)
            if (!std::isfinite(value)) return fail("Contact-shadow inverse projection is not finite");
        constexpr double ClipGuardFraction = 0.5;
        candidate.DepthMetric = {float(metric.NearDistance), float(metric.FarDistance), metric.Perspective ? 1.0f : 0.0f,
            float((metric.Perspective ? metric.NearDistance : 1.0) * ClipGuardFraction)};
        candidate.LightPositionType = {location.X, location.Y, location.Z, directional ? 0.0f : 1.0f};
        candidate.LightDirection = {float(location.X / directionLength), float(location.Y / directionLength),
            float(location.Z / directionLength), 0};
        candidate.TraceParameters = {desc.MaximumDistance, float(desc.Parameters.StartBias), float(desc.Parameters.Thickness), 0};
        candidate.ImageAndBudget = {width, height, desc.Parameters.MaximumSteps, samples};
        output = candidate;
        return true;
    }
}
