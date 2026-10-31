#pragma once

#include "Renderer/RayTracing.h"

namespace Evans
{
namespace RayTracing
{
    class EVANS_RENDERER_API DxrRayTracingRHI final : public IRayTracingRHI
    {
    public:
        DxrRayTracingRHI();
        ~DxrRayTracingRHI() override;

        BackendInfo GetBackendInfo() const override;
        RenderResult Render(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationToken& cancellation = CancellationToken()) const override;
        RenderJob RenderAsync(Scene scene, Camera camera, RenderSettings settings) const override;
        RenderJob RenderAsync(
            std::shared_ptr<const Scene> scene,
            Camera camera,
            RenderSettings settings) const override;
        void WaitIdle() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
}
