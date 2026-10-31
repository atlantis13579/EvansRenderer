#include "Editor/EditorPresenter.h"

#if defined(_WIN32)
#include "Editor/EditorPresenterDX11.h"
#include "Editor/EditorPresenterDX12.h"
#include "Editor/EditorPresenterOpenGLES.h"
#include "Editor/EditorPresenterVulkan.h"
#endif

#if defined(__APPLE__)
#include "Editor/EditorPresenterMetal.h"
#endif

namespace Evans::Editor
{

std::unique_ptr<IEditorPresenter> CreateEditorPresenter(EditorPresentationAPI api)
{
#if defined(_WIN32)
    switch (api)
    {
    case EditorPresentationAPI::DX11:
        return std::make_unique<EditorPresenterDX11>();
    case EditorPresentationAPI::DX12:
        return std::make_unique<EditorPresenterDX12>();
    case EditorPresentationAPI::OpenGLES:
        return std::make_unique<EditorPresenterOpenGLES>();
    case EditorPresentationAPI::Vulkan:
        return std::make_unique<EditorPresenterVulkan>();
    case EditorPresentationAPI::Metal:
        return nullptr;
    }
#elif defined(__APPLE__)
    if (api == EditorPresentationAPI::Metal)
    {
        return std::make_unique<EditorPresenterMetal>();
    }
#else
    (void)api;
#endif
    return nullptr;
}

const char* ToString(EditorPresentationAPI api) noexcept
{
    switch (api)
    {
    case EditorPresentationAPI::DX11: return "DirectX 11";
    case EditorPresentationAPI::DX12: return "DirectX 12";
    case EditorPresentationAPI::Metal: return "Metal";
    case EditorPresentationAPI::OpenGLES: return "OpenGL ES";
    case EditorPresentationAPI::Vulkan: return "Vulkan";
    }
    return "Unknown";
}

EditorPresentationAPI ResolvePresentationAPI(
    std::string_view backendId,
    EditorPresentationAPI lastRealtimeAPI) noexcept
{
    if (backendId == "renderer.dx11") return EditorPresentationAPI::DX11;
    if (backendId == "renderer.dx12") return EditorPresentationAPI::DX12;
    if (backendId == "renderer.dxr")
    {
        return EditorPresentationAPI::DX12;
    }
    if (backendId == "renderer.metal") return EditorPresentationAPI::Metal;
    // OpenGL ES renders offscreen through ANGLE and publishes a CPU Image like
    // every other backend. Reuse the last window presenter so the UI and RHI do
    // not create independent contexts on ANGLE's process-global EGL display.
    if (backendId == "renderer.opengles") return lastRealtimeAPI;
    if (backendId == "renderer.vulkan") return EditorPresentationAPI::Vulkan;
    return lastRealtimeAPI;
}

} // namespace Evans::Editor
