#include "Editor/RenderAdapter.hpp"

#include <stdexcept>
#include <utility>

namespace Evans::Editor
{

CancellationToken::CancellationToken(std::shared_ptr<std::atomic<bool>> state)
    : State_(std::move(state))
{
}

bool CancellationToken::IsCancellationRequested() const noexcept
{
    return State_ != nullptr && State_->load(std::memory_order_relaxed);
}

CancellationSource::CancellationSource()
    : State_(std::make_shared<std::atomic<bool>>(false))
{
}

CancellationToken CancellationSource::Token() const
{
    return CancellationToken(State_);
}

void CancellationSource::Cancel() const noexcept
{
    State_->store(true, std::memory_order_relaxed);
}

CallbackRenderAdapter::CallbackRenderAdapter(
    BackendDescriptor descriptor,
    RenderCallback callback)
    : Descriptor_(std::move(descriptor)), Callback_(std::move(callback))
{
    if (Descriptor_.Id.empty())
    {
        throw std::invalid_argument("A render adapter must have a stable identifier");
    }
    if (!Callback_)
    {
        throw std::invalid_argument("A callback render adapter needs a callback");
    }
}

BackendDescriptor CallbackRenderAdapter::Descriptor() const
{
    return Descriptor_;
}

RenderResult CallbackRenderAdapter::Render(
    const RenderRequest& request,
    const CancellationToken& cancellation)
{
    return Callback_(request, cancellation);
}

const std::vector<BackendDescriptor>& StandardBackendCatalog()
{
    static const std::vector<BackendDescriptor> backends = {
        {"renderer.dx11", "Realtime / DirectX 11", RenderDomain::RealtimeRenderer,
            BackendPlatform::Windows},
        {"renderer.dx12", "Realtime / DirectX 12", RenderDomain::RealtimeRenderer,
            BackendPlatform::Windows},
        {"renderer.metal", "Realtime / Metal", RenderDomain::RealtimeRenderer,
            BackendPlatform::Apple},
        {"renderer.opengles", "Realtime / OpenGL ES", RenderDomain::RealtimeRenderer,
            BackendPlatform::Windows},
        {"renderer.vulkan", "Realtime / Vulkan", RenderDomain::RealtimeRenderer,
            BackendPlatform::Windows},
        {"rasterizer.cpu", "Software Rasterizer / CPU", RenderDomain::SoftwareRasterizer,
            BackendPlatform::Any},
        {"rasterizer.cuda", "Software Rasterizer / CUDA", RenderDomain::SoftwareRasterizer,
            BackendPlatform::Windows},
        {"rasterizer.compute", "Software Rasterizer / Compute Shader",
            RenderDomain::SoftwareRasterizer, BackendPlatform::Windows},
        {"rasterizer.metal", "Software Rasterizer / Metal Compute",
            RenderDomain::SoftwareRasterizer, BackendPlatform::Apple},
        {"renderer.dxr", "Realtime / DirectX Raytracing",
            RenderDomain::RealtimeRenderer,
            BackendPlatform::Windows}
    };
    return backends;
}

HostPlatform CurrentHostPlatform() noexcept
{
#if defined(_WIN32)
    return HostPlatform::Windows;
#elif defined(__APPLE__)
    return HostPlatform::Apple;
#else
    return HostPlatform::Other;
#endif
}

bool IsBackendSupportedOnPlatform(
    const BackendDescriptor& descriptor,
    HostPlatform platform) noexcept
{
    return descriptor.Platform == BackendPlatform::Any ||
        (descriptor.Platform == BackendPlatform::Windows &&
            platform == HostPlatform::Windows) ||
        (descriptor.Platform == BackendPlatform::Apple &&
            platform == HostPlatform::Apple);
}

} // namespace Evans::Editor
