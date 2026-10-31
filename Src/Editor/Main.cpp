#include "Editor/ModuleAdapters.h"
#include "Editor/ContentRoot.hpp"
#include "Editor/EvansEditor.hpp"
#include "Rasterizer/IRasterRHI.hpp"
#include "Renderer/EvansRenderer.h"
#include "Renderer/RealtimeShader.h"
#include "Renderer/RealtimeShaderContract.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
int RunMacEditor();
#endif

#if defined(_WIN32)
#include "Editor/EditorPresenter.h"
#include "Editor/Imgui.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Psapi.h>
#include <shellapi.h>
#include <windowsx.h>
#endif

namespace
{
using Evans::Editor::EditorCore;
using Evans::Editor::Image;
using Evans::Editor::Rect;

bool SavePPM(const Image& image, const std::filesystem::path& path)
{
    if (!image.IsValid())
    {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }
    output << "P6\n" << image.Width << ' ' << image.Height << "\n255\n";
    for (std::size_t pixel = 0; pixel < image.PixelsRGBA8.size(); pixel += 4)
    {
        output.write(reinterpret_cast<const char*>(&image.PixelsRGBA8[pixel]), 3);
    }
    return output.good();
}

std::unique_ptr<EditorCore> CreateEditor()
{
    auto editor = std::make_unique<EditorCore>(Evans::Editor::ResolveContentRoot());
    Evans::Editor::RegisterModuleAdapters(*editor);
    editor->SetPickHandler(Evans::Editor::CreatePickHandler());
    return editor;
}

bool HasBackend(const EditorCore& editor, std::string_view backend)
{
    const std::vector<Evans::Editor::BackendDescriptor> backends = editor.Backends();
    return std::any_of(
        backends.begin(),
        backends.end(),
        [&backend](const Evans::Editor::BackendDescriptor& descriptor)
        {
            return descriptor.Id == backend;
        });
}

const char* PlatformRealtimeBackend() noexcept
{
#if defined(__APPLE__)
    return "renderer.metal";
#elif defined(_WIN32)
    return "renderer.dx11";
#else
    return "rasterizer.cpu";
#endif
}

const char* PlatformRealtimeDisplayName() noexcept
{
#if defined(__APPLE__)
    return "Realtime / Metal";
#elif defined(_WIN32)
    return "Realtime / DirectX 11";
#else
    return "Software Rasterizer / CPU";
#endif
}

Evans::Editor::DemoId ParseDemo(std::string_view name)
{
    if (name == "teapot" || name == "utah-teapot") return Evans::Editor::DemoId::UtahTeapot;
    if (name == "sponza" || name == "sponza-atrium") return Evans::Editor::DemoId::SponzaAtrium;
    return Evans::Editor::DemoId::Triangle;
}

Evans::Editor::FillMode ParseFillMode(std::string_view name)
{
    if (name == "wireframe") return Evans::Editor::FillMode::Wireframe;
    if (name == "solid-wireframe" || name == "solid+wireframe")
    {
        return Evans::Editor::FillMode::SolidWireframe;
    }
    return Evans::Editor::FillMode::Solid;
}

std::string NarrowAscii(const std::wstring& text)
{
    std::string result;
    result.reserve(text.size());
    for (const wchar_t character : text)
    {
        result.push_back(character >= 0 && character <= 0x7f
            ? static_cast<char>(character)
            : '?');
    }
    return result;
}

bool WaitForDemo(
    EditorCore& editor,
    Evans::Editor::DemoId demo,
    const std::chrono::steady_clock::time_point deadline)
{
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor.Tick(0.0f);
        if (!editor.IsDemoLoading() && editor.SelectedDemo() == demo &&
            editor.Scene().Demo == demo)
        {
            return true;
        }
        if (editor.StatusText().rfind("Scene load failed", 0) == 0)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool WaitForPick(
    EditorCore& editor,
    const std::chrono::steady_clock::time_point deadline)
{
    while (editor.IsPickInProgress() && std::chrono::steady_clock::now() < deadline)
    {
        editor.Tick(0.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return !editor.IsPickInProgress() &&
        editor.StatusText().rfind("Pick failed", 0) != 0;
}

int CaptureRealtimeSemanticFrame(
    const std::string& backend,
    Evans::Editor::DemoId demo,
    Evans::Editor::FillMode fillMode,
    bool shadowsEnabled,
    bool outlineEnabled,
    bool selectFirstNode,
    Image& output)
{
    std::unique_ptr<EditorCore> editor = CreateEditor();
    if (!HasBackend(*editor, backend))
    {
        return 4;
    }
    editor->ResizeViewport(640, 360);
    editor->SelectDemo(demo);
    editor->SelectBackend(backend);
    editor->SetFillMode(fillMode);
    editor->SetShadowsEnabled(shadowsEnabled);
    editor->SetOutlineEnabled(outlineEnabled);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!WaitForDemo(*editor, demo, deadline))
    {
        return 2;
    }
    if (selectFirstNode)
    {
        if (editor->Scene().Nodes.empty())
        {
            return 5;
        }
        editor->SelectNode(editor->Scene().Nodes.front().Id);
    }

    const std::uint64_t previousRevision = editor->ViewportRevision();
    editor->RequestRender();
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor->Tick(0.0f);
        if (editor->ViewportRevision() > previousRevision &&
            editor->StatusText().rfind("Frame completed with ", 0) == 0)
        {
            output = editor->ViewportImage();
            return output.IsValid() ? 0 : 2;
        }
        if (editor->StatusText().rfind("Render failed", 0) == 0 ||
            editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 2;
}

[[nodiscard]] std::size_t ImagePixelCount(const Image& image)
{
    return image.IsValid() ? image.PixelsRGBA8.size() / 4U : 0U;
}

[[nodiscard]] bool ImagesHaveSameDimensions(
    const Image& left,
    const Image& right)
{
    return left.IsValid() && right.IsValid() &&
        left.Width == right.Width && left.Height == right.Height;
}

[[nodiscard]] std::size_t CountPixelsNearColor(
    const Image& image,
    const std::array<std::uint8_t, 3>& target,
    int tolerance)
{
    std::size_t result = 0;
    for (std::size_t pixel = 0; pixel + 3U < image.PixelsRGBA8.size(); pixel += 4U)
    {
        bool matches = true;
        for (std::size_t channel = 0; channel < target.size(); ++channel)
        {
            matches = matches &&
                std::abs(
                    static_cast<int>(image.PixelsRGBA8[pixel + channel]) -
                    static_cast<int>(target[channel])) <= tolerance;
        }
        result += matches ? 1U : 0U;
    }
    return result;
}

[[nodiscard]] std::size_t CountChangedPixels(
    const Image& left,
    const Image& right,
    int minimumRgbDifference)
{
    if (!ImagesHaveSameDimensions(left, right))
    {
        return 0;
    }
    std::size_t result = 0;
    for (std::size_t pixel = 0; pixel + 3U < left.PixelsRGBA8.size(); pixel += 4U)
    {
        const int difference =
            std::abs(
                static_cast<int>(left.PixelsRGBA8[pixel]) -
                static_cast<int>(right.PixelsRGBA8[pixel])) +
            std::abs(
                static_cast<int>(left.PixelsRGBA8[pixel + 1U]) -
                static_cast<int>(right.PixelsRGBA8[pixel + 1U])) +
            std::abs(
                static_cast<int>(left.PixelsRGBA8[pixel + 2U]) -
                static_cast<int>(right.PixelsRGBA8[pixel + 2U]));
        result += difference >= minimumRgbDifference ? 1U : 0U;
    }
    return result;
}

[[nodiscard]] std::size_t CountNonBackgroundPixels(
    const Image& image,
    int minimumRgbDifference)
{
    if (!image.IsValid())
    {
        return 0;
    }
    const std::array<int, 3> background = {
        image.PixelsRGBA8[0],
        image.PixelsRGBA8[1],
        image.PixelsRGBA8[2]};
    std::size_t result = 0;
    for (std::size_t pixel = 0; pixel + 3U < image.PixelsRGBA8.size(); pixel += 4U)
    {
        const int difference =
            std::abs(static_cast<int>(image.PixelsRGBA8[pixel]) - background[0]) +
            std::abs(static_cast<int>(image.PixelsRGBA8[pixel + 1U]) - background[1]) +
            std::abs(static_cast<int>(image.PixelsRGBA8[pixel + 2U]) - background[2]);
        result += difference >= minimumRgbDifference ? 1U : 0U;
    }
    return result;
}

[[nodiscard]] bool HasTrianglePrimaryColors(const Image& image)
{
    if (!image.IsValid())
    {
        return false;
    }
    std::array<std::size_t, 3> dominantCounts{};
    for (std::size_t pixel = 0; pixel + 3U < image.PixelsRGBA8.size(); pixel += 4U)
    {
        const int red = image.PixelsRGBA8[pixel];
        const int green = image.PixelsRGBA8[pixel + 1U];
        const int blue = image.PixelsRGBA8[pixel + 2U];
        dominantCounts[0] +=
            red >= 80 && red >= green + 24 && red >= blue + 24 ? 1U : 0U;
        dominantCounts[1] +=
            green >= 80 && green >= red + 24 && green >= blue + 24 ? 1U : 0U;
        dominantCounts[2] +=
            blue >= 80 && blue >= red + 24 && blue >= green + 24 ? 1U : 0U;
    }
    const std::size_t minimum =
        std::max<std::size_t>(32U, ImagePixelCount(image) / 2000U);
    return std::all_of(
        dominantCounts.begin(),
        dominantCounts.end(),
        [minimum](std::size_t count) { return count >= minimum; });
}

[[nodiscard]] std::uint32_t MeanRegionLuminance(
    const Image& image,
    std::uint32_t minimumXPercent,
    std::uint32_t maximumXPercent,
    std::uint32_t minimumYPercent,
    std::uint32_t maximumYPercent)
{
    if (!image.IsValid() ||
        minimumXPercent >= maximumXPercent ||
        minimumYPercent >= maximumYPercent)
    {
        return 0;
    }
    const std::uint32_t minimumX =
        image.Width * minimumXPercent / 100U;
    const std::uint32_t maximumX = std::min(
        image.Width,
        image.Width * maximumXPercent / 100U);
    const std::uint32_t minimumY =
        image.Height * minimumYPercent / 100U;
    const std::uint32_t maximumY = std::min(
        image.Height,
        image.Height * maximumYPercent / 100U);
    std::uint64_t total = 0;
    std::uint64_t samples = 0;
    for (std::uint32_t y = minimumY; y < maximumY; ++y)
    {
        for (std::uint32_t x = minimumX; x < maximumX; ++x)
        {
            const std::size_t pixel =
                (static_cast<std::size_t>(y) * image.Width + x) * 4U;
            total +=
                static_cast<std::uint32_t>(image.PixelsRGBA8[pixel]) * 54U +
                static_cast<std::uint32_t>(image.PixelsRGBA8[pixel + 1U]) * 183U +
                static_cast<std::uint32_t>(image.PixelsRGBA8[pixel + 2U]) * 19U;
            ++samples;
        }
    }
    return samples > 0 ? static_cast<std::uint32_t>(total / samples / 256U) : 0U;
}

[[nodiscard]] std::array<std::uint8_t, 3> ImagePixel(
    const Image& image,
    std::uint32_t x,
    std::uint32_t y)
{
    if (!image.IsValid())
    {
        return {};
    }
    x = std::min(x, image.Width - 1U);
    y = std::min(y, image.Height - 1U);
    const std::size_t pixel =
        (static_cast<std::size_t>(y) * image.Width + x) * 4U;
    return {
        image.PixelsRGBA8[pixel],
        image.PixelsRGBA8[pixel + 1U],
        image.PixelsRGBA8[pixel + 2U]};
}

[[nodiscard]] std::shared_ptr<const Evans::Renderer::ImageData> MakeRendererImage(
    std::uint32_t width,
    std::uint32_t height,
    std::initializer_list<std::uint8_t> pixels)
{
    auto image = std::make_shared<Evans::Renderer::ImageData>();
    image->Width = width;
    image->Height = height;
    image->PixelsRGBA8.assign(pixels);
    return image;
}

[[nodiscard]] Evans::Renderer::MeshData MakeRendererContractQuad(
    float halfWidth = 0.88f)
{
    Evans::Renderer::MeshData mesh;
    mesh.Id = "Realtime.ContractSurface";
    mesh.Color = {1.0f, 1.0f, 1.0f, 1.0f};
    mesh.Vertices = {
        {{-halfWidth, -0.78f, 0.5f}, {0.0f, 0.0f, 1.0f},
         {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
        {{halfWidth, -0.78f, 0.5f}, {0.0f, 0.0f, 1.0f},
         {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
        {{halfWidth, 0.78f, 0.5f}, {0.0f, 0.0f, 1.0f},
         {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
        {{-halfWidth, 0.78f, 0.5f}, {0.0f, 0.0f, 1.0f},
         {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}};
    mesh.Indices = {0U, 1U, 2U, 0U, 2U, 3U};
    return mesh;
}

int RunRealtimeRendererContractSmoke(Evans::Renderer::RHIBackend backend)
{
    namespace Renderer = Evans::Renderer;
    Renderer::Renderer renderer;
    Renderer::RendererCreateInfo createInfo;
    createInfo.Backend = backend;
    createInfo.Width = 192;
    createInfo.Height = 128;
    createInfo.ShadowMapSize = 256;
    createInfo.AutomaticViewProjection = false;
    createInfo.EnableValidation = true;
    if (!renderer.Initialize(createInfo))
    {
        return 2;
    }
    renderer.SetAutomaticViewProjection(false);
    renderer.SetViewProjection(Renderer::Matrix4::Identity());
    renderer.SetLightViewProjection(Renderer::Matrix4::Identity());
    renderer.SetShadowsEnabled(false);

    Renderer::LightingState ambientLighting;
    ambientLighting.LightCount = 0;
    ambientLighting.ShadowLightIndex = -1;
    ambientLighting.AmbientColor = {1.0f, 1.0f, 1.0f};
    ambientLighting.AmbientStrength = 1.0f;
    ambientLighting.CameraPosition = {0.0f, 0.0f, 2.0f};

    Renderer::LightingState directionalLighting = ambientLighting;
    directionalLighting.LightCount = 1;
    directionalLighting.Lights[0].Type = Renderer::LightType::Directional;
    directionalLighting.Lights[0].Direction = {0.0f, 0.0f, 1.0f};
    directionalLighting.Lights[0].Color = {1.0f, 1.0f, 1.0f};
    directionalLighting.Lights[0].Intensity = 1.0f;
    directionalLighting.Lights[0].CastShadows = false;
    directionalLighting.AmbientStrength = 0.0f;

    bool hasMesh = false;
    const auto capture = [&](
        const Renderer::MeshData& mesh,
        const Renderer::LightingState& lighting,
        Image& output)
    {
        const bool configured = hasMesh
            ? renderer.UpdateMesh(mesh)
            : renderer.AddMesh(mesh);
        if (!configured)
        {
            return false;
        }
        hasMesh = true;
        renderer.SetLighting(lighting);
        if (!renderer.RenderFrame())
        {
            return false;
        }
        const Renderer::FrameCapture frame = renderer.GetLastFrameCapture();
        if (!frame.HasColorOutput())
        {
            return false;
        }
        output.Width = frame.Width;
        output.Height = frame.Height;
        output.PixelsRGBA8 = frame.ColorRGBA8;
        return output.IsValid();
    };

    Image defaultTriangle;
    Renderer::MeshData triangle =
        Renderer::Renderer::CreateTriangle("Realtime.ContractSurface");
    if (!capture(triangle, ambientLighting, defaultTriangle) ||
        !HasTrianglePrimaryColors(defaultTriangle))
    {
        return 61;
    }

    Image diffuseFrame;
    Renderer::MeshData diffuse = MakeRendererContractQuad();
    diffuse.Material.AmbientColor = {1.0f, 1.0f, 1.0f, 1.0f};
    diffuse.Material.DiffuseImage = MakeRendererImage(
        2U,
        2U,
        {
            255U, 0U, 0U, 255U,
            0U, 255U, 0U, 255U,
            0U, 0U, 255U, 255U,
            255U, 255U, 255U, 255U});
    if (!capture(diffuse, ambientLighting, diffuseFrame) ||
        !HasTrianglePrimaryColors(diffuseFrame))
    {
        return 62;
    }

    Image srgbFrame;
    Renderer::MeshData srgb = MakeRendererContractQuad();
    srgb.Material.AmbientColor = {1.0f, 1.0f, 1.0f, 1.0f};
    srgb.Material.DiffuseImage = MakeRendererImage(
        1U,
        1U,
        {128U, 128U, 128U, 255U});
    if (!capture(srgb, ambientLighting, srgbFrame))
    {
        return 63;
    }
    const std::array<std::uint8_t, 3> srgbCenter =
        ImagePixel(srgbFrame, srgbFrame.Width / 2U, srgbFrame.Height / 2U);
    if (std::any_of(
            srgbCenter.begin(),
            srgbCenter.end(),
            [](std::uint8_t channel)
            {
                return channel < 120U || channel > 136U;
            }))
    {
        return 64;
    }

    Image normalBaseline;
    Renderer::MeshData normal = MakeRendererContractQuad();
    normal.Material.AmbientColor = {0.0f, 0.0f, 0.0f, 1.0f};
    normal.Material.BaseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    if (!capture(normal, directionalLighting, normalBaseline))
    {
        return 65;
    }
    Image normalFrame;
    normal.Material.NormalImage = MakeRendererImage(
        2U,
        1U,
        {
            128U, 128U, 255U, 255U,
            255U, 128U, 128U, 255U});
    if (!capture(normal, directionalLighting, normalFrame))
    {
        return 66;
    }
    const std::uint32_t normalLeft =
        MeanRegionLuminance(normalFrame, 12U, 42U, 25U, 75U);
    const std::uint32_t normalRight =
        MeanRegionLuminance(normalFrame, 58U, 88U, 25U, 75U);
    const std::uint32_t baselineLeft =
        MeanRegionLuminance(normalBaseline, 12U, 42U, 25U, 75U);
    const std::uint32_t baselineRight =
        MeanRegionLuminance(normalBaseline, 58U, 88U, 25U, 75U);
    if (std::abs(
            static_cast<int>(normalLeft) -
            static_cast<int>(normalRight)) < 40 ||
        std::abs(
            static_cast<int>(baselineLeft) -
            static_cast<int>(baselineRight)) > 12 ||
        std::max(normalLeft, normalRight) < 140U)
    {
        return 67;
    }

    Image opacityFrame;
    Renderer::MeshData opacity = MakeRendererContractQuad();
    opacity.Material.AmbientColor = {1.0f, 0.8f, 0.1f, 1.0f};
    opacity.Material.OpacityImage = MakeRendererImage(
        2U,
        1U,
        {
            0U, 0U, 0U, 255U,
            255U, 255U, 255U, 255U});
    opacity.Material.AlphaCutoff = 0.5f;
    if (!capture(opacity, ambientLighting, opacityFrame))
    {
        return 68;
    }
    const std::size_t opacityPixels =
        CountNonBackgroundPixels(opacityFrame, 24);
    if (opacityPixels < ImagePixelCount(opacityFrame) / 5U ||
        opacityPixels > ImagePixelCount(opacityFrame) * 3U / 5U)
    {
        return 69;
    }

    Image blendFrame;
    Renderer::MeshData blended = MakeRendererContractQuad();
    blended.Material.BaseColor = {1.0f, 1.0f, 1.0f, 0.5f};
    blended.Material.AmbientColor = {1.0f, 0.0f, 0.0f, 1.0f};
    if (!capture(blended, ambientLighting, blendFrame))
    {
        return 70;
    }
    const std::array<std::uint8_t, 3> background =
        ImagePixel(blendFrame, 0U, 0U);
    const std::array<std::uint8_t, 3> blendedCenter =
        ImagePixel(blendFrame, blendFrame.Width / 2U, blendFrame.Height / 2U);
    if (blendedCenter[0] <= background[0] + 60U ||
        blendedCenter[0] < 180U ||
        blendedCenter[0] > 198U ||
        blendedCenter[1] < 18U ||
        blendedCenter[1] > 34U ||
        blendedCenter[2] < 20U ||
        blendedCenter[2] > 38U)
    {
        return 71;
    }

    Image nonUniformFrame;
    Renderer::MeshData nonUniform = MakeRendererContractQuad(0.4f);
    constexpr float inverseSquareRootTwo = 0.70710678f;
    for (Renderer::Vertex& vertex : nonUniform.Vertices)
    {
        vertex.Normal = {
            inverseSquareRootTwo,
            0.0f,
            inverseSquareRootTwo};
    }
    nonUniform.Material.AmbientColor = {0.0f, 0.0f, 0.0f, 1.0f};
    nonUniform.WorldTransform.Values = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};
    if (!capture(nonUniform, directionalLighting, nonUniformFrame) ||
        MeanRegionLuminance(nonUniformFrame, 25U, 75U, 25U, 75U) < 235U)
    {
        return 72;
    }

    Image pointFrame;
    Renderer::MeshData pointSurface = MakeRendererContractQuad();
    pointSurface.Material.AmbientColor = {0.0f, 0.0f, 0.0f, 1.0f};
    Renderer::LightingState pointLighting = directionalLighting;
    pointLighting.Lights[0].Type = Renderer::LightType::Point;
    pointLighting.Lights[0].Position = {-0.65f, 0.0f, 1.5f};
    pointLighting.Lights[0].Intensity = 0.5f;
    if (!capture(pointSurface, pointLighting, pointFrame) ||
        MeanRegionLuminance(pointFrame, 12U, 42U, 25U, 75U) <=
            MeanRegionLuminance(pointFrame, 58U, 88U, 25U, 75U) + 15U)
    {
        return 73;
    }

    Image areaFrontFrame;
    Image areaBackFrame;
    Renderer::LightingState areaLighting = directionalLighting;
    areaLighting.Lights[0].Type = Renderer::LightType::RectArea;
    areaLighting.Lights[0].Position = {0.0f, 0.0f, 1.5f};
    areaLighting.Lights[0].Direction = {0.0f, 0.0f, -1.0f};
    areaLighting.Lights[0].Intensity = 0.6f;
    areaLighting.Lights[0].AreaSize = {1.0f, 1.0f};
    if (!capture(pointSurface, areaLighting, areaFrontFrame))
    {
        return 74;
    }
    areaLighting.Lights[0].Direction = {0.0f, 0.0f, 1.0f};
    if (!capture(pointSurface, areaLighting, areaBackFrame) ||
        MeanRegionLuminance(areaFrontFrame, 25U, 75U, 25U, 75U) <=
            MeanRegionLuminance(areaBackFrame, 25U, 75U, 25U, 75U) + 50U)
    {
        return 75;
    }

    directionalLighting.Lights[0].CastShadows = true;
    directionalLighting.ShadowLightIndex = 0;
    renderer.SetLighting(directionalLighting);
    renderer.SetShadowsEnabled(true);
    if (!renderer.RenderFrame())
    {
        return 76;
    }
    const Renderer::FrameCapture shadowFrame =
        renderer.GetLastFrameCapture();
    const auto shadowDepthRange = std::minmax_element(
        shadowFrame.ShadowDepth32.begin(),
        shadowFrame.ShadowDepth32.end());
    if (shadowDepthRange.first == shadowFrame.ShadowDepth32.end() ||
        *shadowDepthRange.first >= 0.9f ||
        *shadowDepthRange.second < 0.99f)
    {
        return 77;
    }

    Renderer::MeshData transparentShadow = pointSurface;
    transparentShadow.Material.BaseColor.W = 0.0f;
    if (!renderer.UpdateMesh(transparentShadow) ||
        !renderer.RenderFrame())
    {
        return 78;
    }
    const Renderer::FrameCapture transparentShadowFrame =
        renderer.GetLastFrameCapture();
    if (transparentShadowFrame.ShadowDepth32.empty() ||
        *std::min_element(
            transparentShadowFrame.ShadowDepth32.begin(),
            transparentShadowFrame.ShadowDepth32.end()) < 0.999f)
    {
        return 79;
    }

    renderer.SetShadowsEnabled(false);
    if (!renderer.RenderFrame())
    {
        return 80;
    }
    const Renderer::FrameCapture noShadowFrame =
        renderer.GetLastFrameCapture();
    if (shadowFrame.ShadowWidth == 0U ||
        shadowFrame.ShadowHeight == 0U ||
        shadowFrame.ShadowDepth32.empty() ||
        noShadowFrame.ShadowWidth != 0U ||
        noShadowFrame.ShadowHeight != 0U ||
        !noShadowFrame.ShadowDepth32.empty())
    {
        return 6;
    }

    if (!renderer.UpdateMesh(pointSurface) ||
        !renderer.Resize(320, 200, 512))
    {
        return 6;
    }
    renderer.SetShadowsEnabled(true);
    if (!renderer.RenderFrame())
    {
        return 81;
    }
    const Renderer::FrameCapture resizedFrame =
        renderer.GetLastFrameCapture();
    if (resizedFrame.Width != 320U ||
        resizedFrame.Height != 200U ||
        resizedFrame.ShadowWidth != 512U ||
        resizedFrame.ShadowHeight != 512U)
    {
        return 82;
    }

    Renderer::IRHI* rhi = renderer.GetRHI();
    Renderer::PassDesc interruptedPass;
    interruptedPass.Type = Renderer::PassType::Color;
    if (rhi == nullptr ||
        !rhi->BeginFrame() ||
        !rhi->BeginPass(interruptedPass))
    {
        return 83;
    }
    rhi->AbortFrame();
    if (!rhi->BeginFrame())
    {
        return 84;
    }
    rhi->AbortFrame();
    return 0;
}

int RunRealtimeSemanticSmoke(
    const std::filesystem::path& outputPath,
    const std::string& backend)
{
    Image triangleBaseline;
    Image triangleSelected;
    Image triangleWire;
    Image sponzaShadow;
    Image sponzaNoShadow;
    const std::array<int, 5> captures = {
        CaptureRealtimeSemanticFrame(
            backend,
            Evans::Editor::DemoId::Triangle,
            Evans::Editor::FillMode::Solid,
            true,
            false,
            false,
            triangleBaseline),
        CaptureRealtimeSemanticFrame(
            backend,
            Evans::Editor::DemoId::Triangle,
            Evans::Editor::FillMode::Solid,
            true,
            true,
            true,
            triangleSelected),
        CaptureRealtimeSemanticFrame(
            backend,
            Evans::Editor::DemoId::Triangle,
            Evans::Editor::FillMode::Wireframe,
            true,
            false,
            false,
            triangleWire),
        CaptureRealtimeSemanticFrame(
            backend,
            Evans::Editor::DemoId::SponzaAtrium,
            Evans::Editor::FillMode::Solid,
            true,
            false,
            false,
            sponzaShadow),
        CaptureRealtimeSemanticFrame(
            backend,
            Evans::Editor::DemoId::SponzaAtrium,
            Evans::Editor::FillMode::Solid,
            false,
            false,
            false,
            sponzaNoShadow)};
    const auto failedCapture =
        std::find_if(captures.begin(), captures.end(), [](int status) { return status != 0; });
    if (failedCapture != captures.end())
    {
        return *failedCapture;
    }
    if (!SavePPM(sponzaShadow, outputPath))
    {
        return 2;
    }

    const std::size_t trianglePixels = ImagePixelCount(triangleBaseline);
    const std::size_t minimumOutlineDifference =
        std::max<std::size_t>(32U, trianglePixels / 2000U);
    const std::size_t minimumOutlineColor =
        std::max<std::size_t>(16U, trianglePixels / 4000U);
    const std::size_t minimumWireColor =
        std::max<std::size_t>(32U, trianglePixels / 3000U);
    constexpr std::array<std::uint8_t, 3> outlineColor = {255U, 140U, 13U};
    constexpr std::array<std::uint8_t, 3> wireColor = {31U, 209U, 255U};
    const std::size_t wirePixels = CountNonBackgroundPixels(triangleWire, 24);
    const std::size_t fixedWirePixels =
        CountPixelsNearColor(triangleWire, wireColor, 24);
    if (!ImagesHaveSameDimensions(triangleBaseline, triangleSelected) ||
        !HasTrianglePrimaryColors(triangleBaseline) ||
        CountChangedPixels(triangleBaseline, triangleSelected, 12) <
            minimumOutlineDifference ||
        CountPixelsNearColor(triangleSelected, outlineColor, 28) <
            minimumOutlineColor ||
        fixedWirePixels < minimumWireColor ||
        wirePixels < minimumWireColor ||
        fixedWirePixels * 5U < wirePixels * 4U)
    {
        return 5;
    }

    const std::size_t sponzaPixels = ImagePixelCount(sponzaShadow);
    const std::size_t minimumShadowDifference =
        std::max<std::size_t>(64U, sponzaPixels / 400U);
    if (!ImagesHaveSameDimensions(sponzaShadow, sponzaNoShadow) ||
        CountChangedPixels(sponzaShadow, sponzaNoShadow, 24) <
            minimumShadowDifference)
    {
        return 5;
    }
    if (CountNonBackgroundPixels(sponzaNoShadow, 24) <
            ImagePixelCount(sponzaNoShadow) / 3U ||
        MeanRegionLuminance(sponzaNoShadow, 10U, 90U, 10U, 90U) < 35U)
    {
        return 5;
    }
    Evans::Renderer::RHIBackend rhiBackend;
    if (backend == "renderer.dx11")
    {
        rhiBackend = Evans::Renderer::RHIBackend::DX11;
    }
    else if (backend == "renderer.dx12")
    {
        rhiBackend = Evans::Renderer::RHIBackend::DX12;
    }
    else if (backend == "renderer.metal")
    {
        rhiBackend = Evans::Renderer::RHIBackend::Metal;
    }
    else if (backend == "renderer.opengles")
    {
        rhiBackend = Evans::Renderer::RHIBackend::OpenGLES;
    }
    else if (backend == "renderer.vulkan")
    {
        rhiBackend = Evans::Renderer::RHIBackend::Vulkan;
    }
    else
    {
        return 4;
    }
    return RunRealtimeRendererContractSmoke(rhiBackend);
}

int RunSmoke(
    const std::filesystem::path& outputPath,
    const std::string& backend,
    Evans::Editor::DemoId demo,
    Evans::Editor::FillMode fillMode,
    bool shadowsEnabled,
    bool outlineEnabled,
    bool selectFirstNode,
    bool pickCenter)
{
    std::unique_ptr<EditorCore> editor = CreateEditor();
    editor->ResizeViewport(640, 360);
    editor->SelectDemo(demo);
    if (!backend.empty())
    {
        if (!HasBackend(*editor, backend)) return 4;
        editor->SelectBackend(backend);
    }
    editor->SetFillMode(fillMode);
    editor->SetShadowsEnabled(shadowsEnabled);
    editor->SetOutlineEnabled(outlineEnabled);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!WaitForDemo(*editor, demo, deadline))
    {
        return 2;
    }
    if (selectFirstNode && !editor->Scene().Nodes.empty())
    {
        editor->SelectNode(editor->Scene().Nodes.front().Id);
    }
    if (pickCenter)
    {
        editor->HandleViewportPointer(320.0f, 180.0f, {0.0f, 0.0f, 640.0f, 360.0f});
        if (!WaitForPick(*editor, deadline))
        {
            return 2;
        }
    }
    editor->RequestRender();

    bool completed = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor->Tick(0.0f);
        if (!editor->IsDemoLoading() && editor->Scene().Demo == demo &&
            editor->StatusText().rfind("Frame completed with ", 0) == 0)
        {
            completed = true;
            break;
        }
        if (editor->StatusText().rfind("Render failed", 0) == 0 ||
            editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!completed)
    {
        return 2;
    }
    const Image& image = editor->ViewportImage();
    return SavePPM(image, outputPath) ? 0 : 2;
}

int RunSwitchSmoke(const std::filesystem::path& outputPath)
{
    std::unique_ptr<EditorCore> editor = CreateEditor();
    editor->ResizeViewport(640, 360);

    editor->SelectDemo(Evans::Editor::DemoId::SponzaAtrium);
    editor->SelectBackend("rasterizer.cpu");
    const auto sponzaDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!WaitForDemo(*editor, Evans::Editor::DemoId::SponzaAtrium, sponzaDeadline)) return 4;
    editor->RequestRender();

    // Exercise the production switching path while a software frame is active.
    // SelectBackend must only retire the old generation and enqueue the new one.
    const auto switchStarted = std::chrono::steady_clock::now();
    editor->SelectBackend(PlatformRealtimeBackend());
    const auto switchDuration = std::chrono::steady_clock::now() - switchStarted;
    if (switchDuration > std::chrono::milliseconds(250))
    {
        return 3;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor->Tick(0.0f);
        if (!editor->IsDemoLoading() &&
            editor->Scene().Demo == Evans::Editor::DemoId::SponzaAtrium &&
            editor->StatusText() ==
                std::string("Frame completed with ") + PlatformRealtimeDisplayName())
        {
            return SavePPM(editor->ViewportImage(), outputPath) ? 0 : 2;
        }
        if (editor->StatusText().rfind("Render failed", 0) == 0 ||
            editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 2;
}

int RunSwitchStressSmoke(const std::filesystem::path& outputPath)
{
    std::unique_ptr<EditorCore> editor = CreateEditor();
    editor->ResizeViewport(320, 180);

    // Demo loading is latest-only. These calls must stay responsive while the
    // first large OBJ load is still running.
    const std::array<Evans::Editor::DemoId, 4> demoSequence = {
        Evans::Editor::DemoId::UtahTeapot,
        Evans::Editor::DemoId::SponzaAtrium,
        Evans::Editor::DemoId::UtahTeapot,
        Evans::Editor::DemoId::SponzaAtrium};
    for (const Evans::Editor::DemoId demo : demoSequence)
    {
        const auto started = std::chrono::steady_clock::now();
        editor->SelectDemo(demo);
        if (std::chrono::steady_clock::now() - started > std::chrono::milliseconds(100))
        {
            return 3;
        }
    }
    const auto loadDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!WaitForDemo(*editor, Evans::Editor::DemoId::SponzaAtrium, loadDeadline))
    {
        return 4;
    }

    const std::vector<Evans::Editor::BackendDescriptor> backends = editor->Backends();
    if (backends.empty())
    {
        return 4;
    }
    for (std::size_t iteration = 0; iteration < 180; ++iteration)
    {
        const auto started = std::chrono::steady_clock::now();
        const Evans::Editor::BackendDescriptor& descriptor =
            backends[iteration % backends.size()];
        const std::string& backend = descriptor.Id;
        editor->SelectBackend(backend);
        editor->SetFillMode(static_cast<Evans::Editor::FillMode>(iteration % 3));
        editor->SetShadowsEnabled((iteration & 1U) == 0);
        editor->SetOutlineEnabled((iteration & 2U) == 0);
        editor->ResizeViewport(
            320U + static_cast<std::uint32_t>(iteration % 3),
            180U + static_cast<std::uint32_t>(iteration % 2));
        editor->Tick(0.0f);
        if (std::chrono::steady_clock::now() - started > std::chrono::milliseconds(100))
        {
            return 3;
        }
    }

    editor->SelectBackend(PlatformRealtimeBackend());
    editor->SetFillMode(Evans::Editor::FillMode::Solid);
    editor->SetShadowsEnabled(true);
    editor->SetOutlineEnabled(true);
    editor->ResizeViewport(640, 360);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor->Tick(0.0f);
        if (editor->StatusText() ==
            std::string("Frame completed with ") + PlatformRealtimeDisplayName())
        {
            return SavePPM(editor->ViewportImage(), outputPath) ? 0 : 2;
        }
        if (editor->StatusText().rfind("Render failed", 0) == 0 ||
            editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 2;
}

int RunCatalogSmoke()
{
    using namespace Evans::Editor;
    const auto& catalog = StandardBackendCatalog();
    if (catalog.size() != 10 || DemoCatalog::Descriptors().size() != 3) return 2;
    for (const auto& descriptor : catalog)
    {
        if (descriptor.Id.rfind("renderer.", 0) != 0 &&
            descriptor.Id.rfind("rasterizer.", 0) != 0) return 2;
    }
    auto editor = CreateEditor();
    if (editor->Backends().empty() || !HasBackend(*editor, "rasterizer.cpu")) return 2;
    for (const auto& descriptor : editor->Backends())
    {
        if (!IsBackendSupportedOnPlatform(descriptor, CurrentHostPlatform())) return 2;
    }
    const DemoCatalog demos(ResolveContentRoot());
    for (const DemoId demo : {DemoId::UtahTeapot, DemoId::SponzaAtrium})
    {
        const SceneSnapshot scene = demos.Load(demo);
        if (scene.Nodes.empty() || scene.Materials.empty()) return 2;
    }
    return 0;
}

int RunShaderParserSmoke()
{
    using namespace Evans::Renderer;
    const RealtimeShaderParseResult& builtin = BuiltinRealtimeShader();
    std::string error;
    if (!builtin ||
        !ValidateRealtimeShaderContract(
            builtin.Program,
            sizeof(RealtimeDrawConstants),
            MaximumSceneLights,
            error))
    {
        return 2;
    }

    std::string generated;
    if (!MaterializeRealtimeShader(
            builtin.Program,
            RealtimeShaderLanguage::HLSL,
            "{{VERTEX_ENTRY}} {{FRAGMENT_ENTRY}} {{SHADOW_ENTRY}} "
            "{{OUTLINE_ENTRY}} {{MAX_LIGHTS}} "
            "{{DIFFUSEMAP_VULKAN_BINDING}}",
            {
                RealtimeShaderStage::Vertex,
                RealtimeShaderStage::Fragment,
                RealtimeShaderStage::Shadow,
                RealtimeShaderStage::Outline},
            generated,
            error) ||
        generated.find("VSMain PSMain PSShadow PSOutline 4 2") ==
            std::string::npos ||
        generated.find("{{") != std::string::npos)
    {
        return 3;
    }

    const RealtimeShaderParseResult lexicalError =
        ParseRealtimeShader("shader Broken {\n version 1;\n @\n");
    if (lexicalError ||
        lexicalError.Diagnostics.empty() ||
        lexicalError.Diagnostics.front().Line != 3 ||
        lexicalError.Diagnostics.front().Column != 2)
    {
        return 4;
    }

    std::string missingOption = BuiltinRealtimeShaderManifest();
    const std::string optionLine =
        "    option shadow_strength = 0.55;\n";
    const std::size_t optionPosition =
        missingOption.find(optionLine);
    if (optionPosition == std::string::npos)
    {
        return 5;
    }
    missingOption.erase(optionPosition, optionLine.size());
    const RealtimeShaderParseResult missing =
        ParseRealtimeShader(missingOption);
    if (missing ||
        missing.ErrorText().find(
            "Missing required shader option 'shadow_strength'") ==
            std::string::npos)
    {
        return 6;
    }

    std::string duplicateBinding = BuiltinRealtimeShaderManifest();
    const std::string uniqueBinding = "vulkan_binding 2";
    const std::size_t bindingPosition =
        duplicateBinding.find(uniqueBinding);
    if (bindingPosition == std::string::npos)
    {
        return 7;
    }
    duplicateBinding.replace(
        bindingPosition,
        uniqueBinding.size(),
        "vulkan_binding 1");
    const RealtimeShaderParseResult duplicate =
        ParseRealtimeShader(duplicateBinding);
    if (duplicate ||
        duplicate.ErrorText().find("Duplicate Vulkan binding 1") ==
            std::string::npos)
    {
        return 8;
    }
    return 0;
}

int RunRealtimeSmoke(
    const std::filesystem::path& outputPath,
    const std::string& backend)
{
    std::unique_ptr<EditorCore> editor = CreateEditor();
    editor->ResizeViewport(320, 180);

    const std::vector<Evans::Editor::BackendDescriptor> backends = editor->Backends();
    const auto descriptor = std::find_if(
        backends.begin(),
        backends.end(),
        [&backend](const Evans::Editor::BackendDescriptor& candidate)
        {
            return candidate.Id == backend;
        });
    if (descriptor == backends.end())
    {
        return 4;
    }

    editor->SelectDemo(Evans::Editor::DemoId::UtahTeapot);
    editor->SelectBackend(backend);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!WaitForDemo(*editor, Evans::Editor::DemoId::UtahTeapot, deadline))
    {
        return 2;
    }

    const std::uint64_t initialViewportRevision = editor->ViewportRevision();
    const std::uint64_t initialSceneRevision = editor->Scene().Revision;
    std::uint64_t firstFrameRevision = 0;
    bool animationQueued = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        // A positive delta rotates the teapot. A realtime backend must publish
        // at least two independently rendered revisions without RequestRender.
        editor->Tick(animationQueued ? 0.0f : 1.0f / 60.0f);
        animationQueued = editor->Scene().Revision > initialSceneRevision;
        if (editor->StatusText().rfind("Render failed", 0) == 0 ||
            editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            return 2;
        }

        const std::uint64_t viewportRevision = editor->ViewportRevision();
        if (firstFrameRevision == 0 && viewportRevision > initialViewportRevision)
        {
            firstFrameRevision = viewportRevision;
        }
        else if (firstFrameRevision != 0 &&
                 viewportRevision > firstFrameRevision &&
                 editor->Scene().Revision > initialSceneRevision)
        {
            return SavePPM(editor->ViewportImage(), outputPath) ? 0 : 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 2;
}

#if defined(_WIN32)

constexpr wchar_t WindowClassName[] = L"EvansEditorWindow";
constexpr int EditorPanelX = 10;
constexpr int EditorPanelY = 10;
constexpr int EditorPanelWidth = 250;
constexpr int EditorPanelGap = 10;
constexpr int EditorViewportMargin = 10;

bool SaveBMP(const Image& image, const std::filesystem::path& path)
{
    if (!image.IsValid() || image.Width > static_cast<std::uint32_t>(INT_MAX) ||
        image.Height > static_cast<std::uint32_t>(INT_MAX))
    {
        return false;
    }

    constexpr std::uint32_t HeaderSize = 54u;
    const std::uint64_t pixelBytes64 =
        static_cast<std::uint64_t>(image.Width) * image.Height * 4u;
    if (pixelBytes64 > std::numeric_limits<std::uint32_t>::max() - HeaderSize)
    {
        return false;
    }
    const std::uint32_t pixelBytes = static_cast<std::uint32_t>(pixelBytes64);
    std::array<std::uint8_t, HeaderSize> header = {};
    const auto write16 = [&header](std::size_t offset, std::uint16_t value)
    {
        header[offset] = static_cast<std::uint8_t>(value);
        header[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
    };
    const auto write32 = [&header](std::size_t offset, std::uint32_t value)
    {
        header[offset] = static_cast<std::uint8_t>(value);
        header[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
        header[offset + 2] = static_cast<std::uint8_t>(value >> 16u);
        header[offset + 3] = static_cast<std::uint8_t>(value >> 24u);
    };

    header[0] = 'B';
    header[1] = 'M';
    write32(2, HeaderSize + pixelBytes);
    write32(10, HeaderSize);
    write32(14, 40u);
    write32(18, image.Width);
    write32(22, static_cast<std::uint32_t>(-static_cast<std::int32_t>(image.Height)));
    write16(26, 1u);
    write16(28, 32u);
    write32(34, pixelBytes);

    std::vector<std::uint8_t> bgra(image.PixelsRGBA8.size());
    for (std::size_t pixel = 0; pixel < bgra.size(); pixel += 4u)
    {
        bgra[pixel] = image.PixelsRGBA8[pixel + 2u];
        bgra[pixel + 1u] = image.PixelsRGBA8[pixel + 1u];
        bgra[pixel + 2u] = image.PixelsRGBA8[pixel];
        bgra[pixel + 3u] = image.PixelsRGBA8[pixel + 3u];
    }

    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    output.write(reinterpret_cast<const char*>(bgra.data()), bgra.size());
    return output.good();
}

struct WindowState
{
    std::unique_ptr<EditorCore> Editor;
    std::unique_ptr<Evans::Editor::IEditorPresenter> Presenter;
    Evans::Editor::EditorPresentationAPI LastRealtimePresentation =
        Evans::Editor::EditorPresentationAPI::DX11;
    std::string PresentationSelection;
    std::string PendingPresentationSelection;
    std::string FailedPresentationSelection;
    std::string PresentationError;
    void* NativeWindow = nullptr;
    std::chrono::steady_clock::time_point LastTick = std::chrono::steady_clock::now();
    int ClientWidth = 1280;
    int ClientHeight = 720;
    int MouseX = 0;
    int MouseY = 0;
    int MouseWheel = 0;
    int PanelScroll = 0;
    unsigned char MouseButtons = 0;
    bool CameraRotateCapture = false;
    std::ostream* PresentationTrace = nullptr;
    std::string QueuedBackendSelection;
    std::uint64_t PresentedFrames = 0;
    bool ReenterOnNextFrame = false;
    std::uint64_t PresentationFailures = 0;
    bool DrawingFrame = false;
    bool ClosePending = false;
    bool Closing = false;
    bool CloseOnNextFrame = false;
    bool ResizePending = false;
    int QueuedFillMode = 0;
    int QueuedEffectFlags = 0;
    std::uint32_t ReentryProbeCount = 0;
};

void TracePresentation(WindowState& state, const std::string& stage)
{
    if (state.PresentationTrace != nullptr)
        *state.PresentationTrace << GetTickCount64() << " " << stage << std::endl;
}

bool IsRealtimeBackend(std::string_view backendId)
{
    return backendId.rfind("renderer.", 0) == 0;
}

bool EnsurePresentationBackend(WindowState& state)
{
    const std::string selectedBackend = state.Editor->SelectedBackend();
    const Evans::Editor::EditorPresentationAPI desiredAPI =
        Evans::Editor::ResolvePresentationAPI(
            selectedBackend,
            state.LastRealtimePresentation);
    if (selectedBackend == state.PresentationSelection &&
        state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        return true;
    }
    if (selectedBackend == state.FailedPresentationSelection &&
        state.Presenter != nullptr)
    {
        return true;
    }
    if (state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend))
        {
            state.LastRealtimePresentation = desiredAPI;
        }
        return true;
    }

    // Keep the old window API alive while the new offscreen RHI initializes.
    // Concurrent DX12 teardown / Vulkan ICD initialization can fail inside the driver.
    if (state.Presenter != nullptr && state.Editor->ViewportBackend() != selectedBackend)
    {
        if (state.PendingPresentationSelection != selectedBackend)
            TracePresentation(state, "waiting for first frame: " + selectedBackend);
        state.PendingPresentationSelection = selectedBackend;
        return true;
    }
    state.PendingPresentationSelection.clear();
    TracePresentation(state, "switch begin: " + state.PresentationSelection + " -> " + selectedBackend);
    const bool hadPreviousPresenter = state.Presenter != nullptr;
    const Evans::Editor::EditorPresentationAPI previousAPI = hadPreviousPresenter
        ? state.Presenter->API()
        : Evans::Editor::EditorPresentationAPI::DX11;
    if (state.Presenter != nullptr)
    {
        // A Win32 window can only own one active flip-model swap chain. Release
        // it before initializing the newly selected native presentation API.
        TracePresentation(state, "previous presenter shutdown begin");
        state.Presenter->Shutdown();
        state.Presenter.reset();
        TracePresentation(state, "previous presenter shutdown complete");
    }
    TracePresentation(state, "new presenter initialize begin");

    std::unique_ptr<Evans::Editor::IEditorPresenter> candidate =
        Evans::Editor::CreateEditorPresenter(desiredAPI);
    if (candidate == nullptr)
    {
        state.PresentationError = std::string(Evans::Editor::ToString(desiredAPI))
            + " editor presentation is not available on this platform.";
    }
    else if (!candidate->Initialize(
                 state.NativeWindow,
                 state.ClientWidth,
                 state.ClientHeight))
    {
        state.PresentationError = candidate->LastError();
    }
    else
    {
        TracePresentation(state, "new presenter initialize complete");
        state.Presenter = std::move(candidate);
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend))
        {
            state.LastRealtimePresentation = desiredAPI;
        }
        return true;
    }
    // Release partially initialized swapchain resources before recreating the old API.
    ++state.PresentationFailures;
    TracePresentation(state, "FAIL initialize: " + state.PresentationError);
    candidate.reset();
    state.FailedPresentationSelection = selectedBackend;

    // Keep the editor controllable after an unavailable API selection, while
    // preserving the visible error and never reporting the fallback as native.
    if (hadPreviousPresenter)
    {
        std::unique_ptr<Evans::Editor::IEditorPresenter> previous =
            Evans::Editor::CreateEditorPresenter(previousAPI);
        if (previous != nullptr && previous->Initialize(
                state.NativeWindow,
                state.ClientWidth,
                state.ClientHeight))
        {
            state.Presenter = std::move(previous);
        }
    }
    return state.Presenter != nullptr;
}

Rect EditorViewportBounds(const WindowState& state)
{
    const float left = static_cast<float>(EditorPanelX + EditorPanelWidth + EditorPanelGap);
    const float top = static_cast<float>(EditorViewportMargin);
    return {
        left,
        top,
        std::max(1.0f, static_cast<float>(state.ClientWidth) - left - EditorViewportMargin),
        std::max(1.0f, static_cast<float>(state.ClientHeight) - top - EditorViewportMargin)};
}

int BackendSelectionIndex(
    const std::vector<Evans::Editor::BackendDescriptor>& backends,
    const std::string& selectedBackend)
{
    for (std::size_t index = 0; index < backends.size(); ++index)
    {
        if (backends[index].Id == selectedBackend)
        {
            return static_cast<int>(index);
        }
    }
    return 0;
}

const char* BackendUiLabel(const std::string& id)
{
    if (id == "renderer.dx11") return "Renderer / DX11";
    if (id == "renderer.dx12") return "Renderer / DX12";
    if (id == "renderer.metal") return "Renderer / Metal";
    if (id == "renderer.opengles") return "Renderer / OpenGL ES";
    if (id == "renderer.vulkan") return "Renderer / Vulkan";
    if (id == "rasterizer.cpu") return "Rasterizer / CPU";
    if (id == "rasterizer.cuda") return "Rasterizer / CUDA";
    if (id == "rasterizer.compute") return "Rasterizer / Compute";
    if (id == "rasterizer.metal") return "Rasterizer / Metal";
    if (id == "renderer.dxr") return "Renderer / DXR";
    return "Unknown RHI";
}

int DemoSelectionIndex(Evans::Editor::DemoId demo)
{
    const std::vector<Evans::Editor::DemoDescriptor>& descriptors =
        Evans::Editor::DemoCatalog::Descriptors();
    for (std::size_t index = 0; index < descriptors.size(); ++index)
    {
        if (descriptors[index].Id == demo)
        {
            return static_cast<int>(index);
        }
    }
    return 0;
}

void DrawEditorImgui(WindowState& state)
{
    if (state.CloseOnNextFrame)
    {
        state.CloseOnNextFrame = false;
        SendMessageW(static_cast<HWND>(state.NativeWindow), WM_CLOSE, 0, 0);
    }
    // Regression input follows the same mid-frame path as the backend combo.
    if (!state.QueuedBackendSelection.empty())
    {
        const std::string backend = std::move(state.QueuedBackendSelection);
        state.QueuedBackendSelection.clear();
        state.Editor->SelectBackend(backend);
        state.Editor->SetFillMode(static_cast<Evans::Editor::FillMode>(state.QueuedFillMode));
        state.Editor->SetShadowsEnabled((state.QueuedEffectFlags & 1) != 0);
        state.Editor->SetOutlineEnabled((state.QueuedEffectFlags & 2) != 0);
        state.Editor->RequestRender();
    }
    if (state.ReenterOnNextFrame)
    {
        state.ReenterOnNextFrame = false;
        TracePresentation(state, "reentrant paint requested");
        HWND window = static_cast<HWND>(state.NativeWindow);
        if ((++state.ReentryProbeCount % 24) == 0)
        {
            RECT bounds{};
            GetWindowRect(window, &bounds);
            const int delta = (state.ReentryProbeCount % 48) == 0 ? -32 : 32;
            SetWindowPos(window, nullptr, 0, 0, bounds.right - bounds.left + delta,
                bounds.bottom - bounds.top + delta, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
        InvalidateRect(window, nullptr, FALSE);
        UpdateWindow(window);
    }
    const int panelHeight = std::max(120, state.ClientHeight - EditorPanelY * 2);
    imguiBeginScrollArea(
        "Evans",
        EditorPanelX,
        EditorPanelY,
        EditorPanelWidth,
        panelHeight,
        &state.PanelScroll);

    const std::vector<Evans::Editor::BackendDescriptor> backends = state.Editor->Backends();
    std::vector<const char*> backendLabels;
    backendLabels.reserve(backends.size());
    for (const Evans::Editor::BackendDescriptor& backend : backends)
    {
        backendLabels.push_back(BackendUiLabel(backend.Id));
    }

    imguiLabel("RHI Backend");
    int backendIndex = BackendSelectionIndex(backends, state.Editor->SelectedBackend());
    if (!backendLabels.empty() &&
        imguiCombo("", &backendIndex, backendLabels.data(), static_cast<int>(backendLabels.size())))
    {
        TracePresentation(state, "UI backend: " + state.Editor->SelectedBackend() + " -> " +
            backends[static_cast<std::size_t>(backendIndex)].Id);
        state.Editor->SelectBackend(backends[static_cast<std::size_t>(backendIndex)].Id);
    }

    imguiSeparatorLine();
    imguiLabel("Scene");
    const std::vector<Evans::Editor::DemoDescriptor>& demos =
        Evans::Editor::DemoCatalog::Descriptors();
    std::vector<const char*> demoLabels;
    demoLabels.reserve(demos.size());
    for (const Evans::Editor::DemoDescriptor& demo : demos)
    {
        demoLabels.push_back(demo.Name.data());
    }
    int demoIndex = DemoSelectionIndex(state.Editor->SelectedDemo());
    if (!demoLabels.empty() &&
        imguiCombo("Demo", &demoIndex, demoLabels.data(), static_cast<int>(demoLabels.size())))
    {
        state.Editor->SelectDemo(demos[static_cast<std::size_t>(demoIndex)].Id);
    }

    imguiSeparatorLine();
    imguiLabel("Fill Mode");
    constexpr const char* fillModeLabels[] = {
        "Solid",
        "Wireframe",
        "Solid + Wireframe"};
    for (int modeIndex = 0; modeIndex < 3; ++modeIndex)
    {
        const Evans::Editor::FillMode mode =
            static_cast<Evans::Editor::FillMode>(modeIndex);
        if (imguiCheck(
                fillModeLabels[modeIndex],
                state.Editor->Settings().Mode == mode,
                true) &&
            state.Editor->Settings().Mode != mode)
        {
            state.Editor->SetFillMode(mode);
        }
    }

    imguiLabel("Features");
    if (imguiCheck("Shadow Map", state.Editor->Settings().ShadowsEnabled, true))
    {
        state.Editor->SetShadowsEnabled(!state.Editor->Settings().ShadowsEnabled);
    }
    if (imguiCheck("Selection Outline", state.Editor->Settings().OutlineEnabled, true))
    {
        state.Editor->SetOutlineEnabled(!state.Editor->Settings().OutlineEnabled);
    }

    imguiSeparatorLine();
    imguiLabel("Status");
    if (state.Presenter != nullptr)
    {
        const std::string presentation = std::string("Presentation: ") +
            Evans::Editor::ToString(state.Presenter->API());
        imguiValue(presentation.c_str());
    }
    if (!state.PresentationError.empty())
    {
        const std::string error = "Presentation error: " + state.PresentationError;
        imguiValue(error.c_str());
    }
    imguiValue(state.Editor->StatusText().c_str());
    imguiEndScrollArea();
}

bool DrawEditorFrame(
    WindowState& state,
    Image* capture,
    bool present,
    std::string* error = nullptr)
{
    // Native swapchain operations can synchronously dispatch window messages.
    // Never replace a presenter or begin another frame while one is recording.
    if (state.DrawingFrame)
    {
        TracePresentation(state, "deferred reentrant draw");
        if (capture != nullptr)
        {
            if (error != nullptr) *error = "Cannot capture a nested editor frame.";
            return false;
        }
        return true;
    }
    state.DrawingFrame = true;
    struct FrameGuard
    {
        WindowState& State;
        ~FrameGuard()
        {
            State.DrawingFrame = false;
            if (State.ClosePending)
            {
                State.ClosePending = false;
                PostMessageW(static_cast<HWND>(State.NativeWindow), WM_CLOSE, 0, 0);
            }
        }
    } guard{state};
    const auto resetFailedPresenter = [&state, error](const std::string& message)
    {
        ++state.PresentationFailures;
        TracePresentation(state, "FAIL draw: " + message);
        if (error != nullptr) *error = message;
        state.PresentationError = message;
        state.PresentationSelection.clear();
        if (state.Presenter != nullptr)
        {
            state.Presenter->Shutdown();
            state.Presenter.reset();
        }
        return false;
    };
    if (!EnsurePresentationBackend(state) || state.Presenter == nullptr ||
        !state.Presenter->IsInitialized())
    {
        if (error != nullptr) *error = state.PresentationError;
        return false;
    }

    if (state.ResizePending)
    {
        state.ResizePending = false;
        if (!state.Presenter->Resize(state.ClientWidth, state.ClientHeight))
            return resetFailedPresenter(state.Presenter->LastError());
    }
    const Rect viewport = EditorViewportBounds(state);
    if (!state.Presenter->BeginFrame(
            state.Editor->ViewportImage(),
            state.Editor->ViewportRevision(),
            static_cast<int>(viewport.X),
            static_cast<int>(viewport.Y),
            static_cast<int>(viewport.Width),
            static_cast<int>(viewport.Height)))
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }

    const int imguiMouseY = state.ClientHeight - 1 - state.MouseY;
    imguiBeginFrame(
        state.MouseX,
        imguiMouseY,
        state.MouseButtons,
        state.MouseWheel);
    state.MouseWheel = 0;
    DrawEditorImgui(state);
    imguiEndFrame();
    state.Presenter->DrawImgui();
    if (capture != nullptr && !state.Presenter->CaptureBackBuffer(*capture))
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    if (present && !state.Presenter->EndFrame())
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    if (present) ++state.PresentedFrames;
    return true;
}

void RenderEditorWindow(WindowState& state)
{
    if (state.Closing || IsIconic(static_cast<HWND>(state.NativeWindow))) return;
    (void)DrawEditorFrame(state, nullptr, true);
}

void ResizeEditor(WindowState& state, int width, int height)
{
    state.ClientWidth = std::max(width, 1);
    state.ClientHeight = std::max(height, 1);
    // Resize GPU resources at the next frame boundary, including during modal sizing.
    state.ResizePending = true;

    const Rect viewport = EditorViewportBounds(state);
    state.Editor->ResizeViewport(
        static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
        static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));
}

LRESULT CALLBACK WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        state = static_cast<WindowState*>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
    }
    if (state == nullptr)
    {
        return DefWindowProcW(window, message, wParam, lParam);
    }

    switch (message)
    {
    case WM_CREATE:
    {
        RECT client = {};
        GetClientRect(window, &client);
        state->ClientWidth = std::max<LONG>(client.right - client.left, 1);
        state->ClientHeight = std::max<LONG>(client.bottom - client.top, 1);
        state->NativeWindow = window;
        if (!EnsurePresentationBackend(*state))
        {
            MessageBoxA(
                window,
                state->PresentationError.c_str(),
                "Evans Editor",
                MB_OK | MB_ICONERROR);
            return -1;
        }
        ResizeEditor(*state, state->ClientWidth, state->ClientHeight);
        SetTimer(window, 1, 16, nullptr);
        return 0;
    }
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED)
        {
            ResizeEditor(*state, LOWORD(lParam), HIWORD(lParam));
            InvalidateRect(window, nullptr, FALSE);
        }
        return 0;
    case WM_TIMER:
    {
        const auto now = std::chrono::steady_clock::now();
        const float delta = std::chrono::duration<float>(now - state->LastTick).count();
        state->LastTick = now;
        state->Editor->Tick(std::min(delta, 0.1f));
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_MOUSEMOVE:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture && (wParam & MK_RBUTTON) != 0);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_LBUTTONDOWN:
    {
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_LEFT;
        SetCapture(window);
        const Rect viewport = EditorViewportBounds(*state);
        if (viewport.Contains(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY)))
        {
            state->Editor->HandleViewportPointer(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY),
                viewport);
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_LBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_LEFT);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONDOWN:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_RIGHT;
        state->CameraRotateCapture = EditorViewportBounds(*state).Contains(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY));
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture);
        SetCapture(window);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
        state->CameraRotateCapture = false;
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            false);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_CAPTURECHANGED:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->Editor->HandleCameraPointer(0.0f, 0.0f, false);
        return 0;
    case WM_KEYDOWN:
        state->Editor->SetCameraMovementKey(static_cast<char>(wParam), true);
        return 0;
    case WM_KEYUP:
        state->Editor->SetCameraMovementKey(static_cast<char>(wParam), false);
        return 0;
    case WM_KILLFOCUS:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->Editor->ClearCameraInput();
        return 0;
    case WM_MOUSEWHEEL:
    {
        POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        ScreenToClient(window, &cursor);
        state->MouseX = cursor.x;
        state->MouseY = cursor.y;
        state->MouseWheel += GET_WHEEL_DELTA_WPARAM(wParam) > 0 ? -1 : 1;
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT:
    {
        PAINTSTRUCT paint = {};
        BeginPaint(window, &paint);
        EndPaint(window, &paint);
        RenderEditorWindow(*state);
        return 0;
    }
    case WM_CLOSE:
        if (state->DrawingFrame)
        {
            state->ClosePending = true;
            return 0;
        }
        state->Closing = true;
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        state->Closing = true;
        KillTimer(window, 1);
        if (state->Presenter != nullptr)
        {
            state->Presenter->Shutdown();
            state->Presenter.reset();
        }
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

bool RegisterEditorWindowClass(HINSTANCE instance)
{
    WNDCLASSEXW windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.lpszClassName = WindowClassName;
    if (RegisterClassExW(&windowClass))
    {
        return true;
    }
    return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
}

int RunUiSmoke(
    HINSTANCE instance,
    const std::filesystem::path& outputPath,
    const std::string& initialBackend,
    Evans::Editor::DemoId initialDemo,
    bool shadowsEnabled)
{
    WindowState state;
    state.Editor = CreateEditor();
    state.Editor->SelectDemo(initialDemo);
    state.Editor->SetShadowsEnabled(shadowsEnabled);
    if (!initialBackend.empty())
    {
        if (!HasBackend(*state.Editor, initialBackend))
        {
            return 4;
        }
        state.Editor->SelectBackend(initialBackend);
    }
    if (!RegisterEditorWindowClass(instance))
    {
        return 3;
    }

    HWND window = CreateWindowExW(
        0,
        WindowClassName,
        L"Evans Renderer Editor UI Smoke",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1280,
        720,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 4;
    }
    // Creating the window sends the initial resize event. Start an explicit
    // capture frame after that event so the resize cannot retire it.
    state.Editor->RequestRender();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    bool completed = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        state.Editor->Tick(0.0f);
        if (!state.Editor->IsDemoLoading() && state.Editor->Scene().Demo == initialDemo &&
            state.Editor->StatusText().rfind("Frame completed with ", 0) == 0)
        {
            completed = true;
            break;
        }
        if (state.Editor->StatusText().rfind("Render failed", 0) == 0 ||
            state.Editor->StatusText().rfind("Scene load failed", 0) == 0)
        {
            DestroyWindow(window);
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!completed)
    {
        DestroyWindow(window);
        return 2;
    }

    if (initialBackend != "renderer.dx11")
    {
        Image capture;
        const Evans::Editor::EditorPresentationAPI expectedAPI =
            Evans::Editor::ResolvePresentationAPI(
                initialBackend,
                Evans::Editor::EditorPresentationAPI::DX11);
        std::string frameError;
        const bool drewFrame = DrawEditorFrame(state, &capture, false, &frameError);
        if (!drewFrame)
        {
            std::ofstream log(outputPath.string() + ".log", std::ios::binary);
            log << frameError;
        }
        const bool savedCapture = drewFrame && SaveBMP(capture, outputPath);
        const bool usedExpectedAPI = state.Presenter != nullptr &&
            state.Presenter->API() == expectedAPI;
        DestroyWindow(window);
        if (!drewFrame) return 5;
        if (!savedCapture) return 6;
        if (!usedExpectedAPI) return 7;
        return 0;
    }

    state.MouseX = 100;
    state.MouseY = 68;
    state.MouseButtons = 0;
    if (!DrawEditorFrame(state, nullptr, true))
    {
        DestroyWindow(window);
        return 2;
    }
    state.MouseButtons = IMGUI_MBUT_LEFT;
    if (!DrawEditorFrame(state, nullptr, true))
    {
        DestroyWindow(window);
        return 2;
    }
    state.MouseButtons = 0;
    Image capture;
    if (!DrawEditorFrame(state, &capture, false) || !SaveBMP(capture, outputPath))
    {
        DestroyWindow(window);
        return 2;
    }

    // Switch the live window from its default DX11 presenter to DX12.
    state.MouseY = 110;
    state.MouseButtons = 0;
    if (!DrawEditorFrame(state, nullptr, true))
    {
        DestroyWindow(window);
        return 2;
    }
    state.MouseButtons = IMGUI_MBUT_LEFT;
    if (!DrawEditorFrame(state, nullptr, true))
    {
        DestroyWindow(window);
        return 2;
    }
    state.MouseButtons = 0;
    if (!DrawEditorFrame(state, nullptr, true) ||
        state.Editor->SelectedBackend() != "renderer.dx12")
    {
        DestroyWindow(window);
        return 2;
    }
    Image switchedCapture;
    if (!DrawEditorFrame(state, &switchedCapture, false) ||
        state.Presenter == nullptr ||
        state.Presenter->API() != Evans::Editor::EditorPresentationAPI::DX12 ||
        !switchedCapture.IsValid())
    {
        DestroyWindow(window);
        return 2;
    }

    DestroyWindow(window);
    return 0;
}

int RunPresentationSwitchSmoke(HINSTANCE instance)
{
    constexpr std::array<std::string_view, 4> CandidateBackends = {
        "renderer.dx11",
        "renderer.dx12",
        "renderer.opengles",
        "renderer.vulkan"};
    constexpr std::size_t SwitchRounds = 2;

    WindowState state;
    state.Editor = CreateEditor();
    state.Editor->SelectDemo(Evans::Editor::DemoId::Triangle);

    const std::vector<Evans::Editor::BackendDescriptor> registered =
        state.Editor->Backends();
    std::vector<Evans::Editor::BackendDescriptor> backends;
    for (const std::string_view candidate : CandidateBackends)
    {
        const auto descriptor = std::find_if(
            registered.begin(),
            registered.end(),
            [candidate](const Evans::Editor::BackendDescriptor& value)
            {
                return value.Id == candidate;
            });
        if (descriptor != registered.end())
        {
            backends.push_back(*descriptor);
        }
    }
    if (backends.size() < 2U)
    {
        return 4;
    }

    state.Editor->SelectBackend(backends.front().Id);
    if (!RegisterEditorWindowClass(instance))
    {
        return 3;
    }

    HWND window = CreateWindowExW(
        0,
        WindowClassName,
        L"Evans Renderer Presentation Switch Smoke",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        960,
        540,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 4;
    }
    const auto finish = [window](int result)
    {
        if (IsWindow(window))
        {
            DestroyWindow(window);
        }
        return result;
    };
    const auto waitForFrame = [&state](
        const Evans::Editor::BackendDescriptor& descriptor,
        std::uint64_t previousRevision)
    {
        const std::string completed = "Frame completed with " + descriptor.DisplayName;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        auto nextPresentation = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < deadline)
        {
            state.Editor->Tick(0.0f);
            const auto now = std::chrono::steady_clock::now();
            if (now >= nextPresentation)
            {
                std::string frameError;
                if (!DrawEditorFrame(state, nullptr, true, &frameError))
                {
                    return false;
                }
                nextPresentation = now + std::chrono::milliseconds(16);
            }
            if (state.Editor->SelectedBackend() == descriptor.Id &&
                state.Editor->ViewportRevision() > previousRevision &&
                state.Editor->StatusText() == completed)
            {
                return true;
            }
            if (state.Editor->StatusText().rfind("Render failed", 0) == 0 ||
                state.Editor->StatusText().rfind("Scene load failed", 0) == 0)
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    };

    const auto loadDeadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    if (!WaitForDemo(
            *state.Editor,
            Evans::Editor::DemoId::Triangle,
            loadDeadline))
    {
        return finish(2);
    }
    const std::uint64_t initialRevision = state.Editor->ViewportRevision();
    state.Editor->RequestRender();
    if (!waitForFrame(backends.front(), initialRevision))
    {
        return finish(2);
    }

    // Exercise every registered realtime backend repeatedly on the same HWND.
    for (std::size_t round = 0; round < SwitchRounds; ++round)
    {
        for (const Evans::Editor::BackendDescriptor& descriptor : backends)
        {
            const Evans::Editor::EditorPresentationAPI expectedAPI =
                Evans::Editor::ResolvePresentationAPI(
                    descriptor.Id,
                    state.LastRealtimePresentation);
            state.Editor->SelectBackend(descriptor.Id);

            // Resolve the presentation path while the previous viewport is valid.
            std::string frameError;
            if (!DrawEditorFrame(state, nullptr, true, &frameError))
            {
                return finish(5);
            }

            const std::uint64_t previousRevision = state.Editor->ViewportRevision();
            state.Editor->RequestRender();
            if (!waitForFrame(descriptor, previousRevision))
            {
                return finish(2);
            }

            Image capture;
            if (!DrawEditorFrame(state, &capture, false, &frameError) ||
                state.Editor->SelectedBackend() != descriptor.Id ||
                state.Presenter == nullptr ||
                state.Presenter->API() != expectedAPI ||
                !capture.IsValid())
            {
                return finish(6);
            }
        }
    }

    // Software and DXR images must survive every presentation API and return
    // to the previous hardware backend on the same HWND.
    for (const Evans::Editor::BackendDescriptor& realtime : backends)
    {
        for (const Evans::Editor::BackendDescriptor& rendered : registered)
        {
            if (rendered.Domain != Evans::Editor::RenderDomain::SoftwareRasterizer &&
                rendered.Id != "renderer.dxr") continue;
            state.Editor->SelectBackend(realtime.Id);
            std::uint64_t previousRevision = state.Editor->ViewportRevision();
            state.Editor->RequestRender();
            if (!waitForFrame(realtime, previousRevision)) return finish(7);

            const auto expectedAPI = Evans::Editor::ResolvePresentationAPI(
                rendered.Id, state.LastRealtimePresentation);
            state.Editor->SelectBackend(rendered.Id);
            state.Editor->SetFillMode(Evans::Editor::FillMode::SolidWireframe);
            state.Editor->SetShadowsEnabled(true);
            state.Editor->SetOutlineEnabled(true);
            state.Editor->SelectNode(state.Editor->Scene().Nodes.front().Id);
            previousRevision = state.Editor->ViewportRevision();
            state.Editor->RequestRender();
            if (!waitForFrame(rendered, previousRevision)) return finish(8);
            Image capture;
            if (!DrawEditorFrame(state, &capture, false) || !capture.IsValid() ||
                !state.Presenter || state.Presenter->API() != expectedAPI)
                return finish(9);

            state.Editor->RequestRender();
            state.Editor->SelectBackend(realtime.Id);
            state.Editor->SetFillMode(Evans::Editor::FillMode::Solid);
            state.Editor->SetOutlineEnabled(false);
            previousRevision = state.Editor->ViewportRevision();
            state.Editor->RequestRender();
            if (!waitForFrame(realtime, previousRevision)) return finish(10);
        }
    }

    const auto findBackend = [&backends](std::string_view id)
        -> const Evans::Editor::BackendDescriptor*
    {
        const auto found = std::find_if(
            backends.begin(),
            backends.end(),
            [id](const Evans::Editor::BackendDescriptor& descriptor)
            {
                return descriptor.Id == id;
            });
        return found == backends.end() ? nullptr : &*found;
    };
    const Evans::Editor::BackendDescriptor* dx11 =
        findBackend("renderer.dx11");
    const Evans::Editor::BackendDescriptor* openGLES =
        findBackend("renderer.opengles");
    if (dx11 != nullptr && openGLES != nullptr)
    {
        // Reproduce the interactive sequence that previously escaped the
        // backend-only switch loop: DX11 -> OpenGL ES -> scene -> DX11.
        state.Editor->SelectBackend(openGLES->Id);
        std::uint64_t previousRevision = state.Editor->ViewportRevision();
        state.Editor->RequestRender();
        if (!waitForFrame(*openGLES, previousRevision))
        {
            return finish(2);
        }

        previousRevision = state.Editor->ViewportRevision();
        state.Editor->SelectDemo(Evans::Editor::DemoId::UtahTeapot);
        if (!waitForFrame(*openGLES, previousRevision))
        {
            return finish(2);
        }

        state.Editor->SelectBackend(dx11->Id);
        previousRevision = state.Editor->ViewportRevision();
        state.Editor->RequestRender();
        if (!waitForFrame(*dx11, previousRevision))
        {
            return finish(2);
        }

        // Also cover the fast UI path where the backend changes while the new
        // scene is still loading and the OpenGL frame is being retired.
        state.Editor->SelectBackend(openGLES->Id);
        previousRevision = state.Editor->ViewportRevision();
        state.Editor->RequestRender();
        if (!waitForFrame(*openGLES, previousRevision))
        {
            return finish(2);
        }
        previousRevision = state.Editor->ViewportRevision();
        state.Editor->SelectDemo(Evans::Editor::DemoId::UtahTeapot);
        state.Editor->SelectBackend(dx11->Id);
        if (!waitForFrame(*dx11, previousRevision))
        {
            return finish(2);
        }

        // Force a heavy OpenGL frame to remain active while both the scene and
        // backend selections change. This is the lifecycle ordering exercised
        // by normal UI timers but missed by sequential frame-completion tests.
        state.Editor->SelectBackend(openGLES->Id);
        previousRevision = state.Editor->ViewportRevision();
        state.Editor->SelectDemo(Evans::Editor::DemoId::SponzaAtrium);
        if (!waitForFrame(*openGLES, previousRevision))
        {
            return finish(2);
        }
        state.Editor->SetFillMode(Evans::Editor::FillMode::Wireframe);
        state.Editor->SelectDemo(Evans::Editor::DemoId::Triangle);
        state.Editor->SelectBackend(dx11->Id);
        state.Editor->SetFillMode(Evans::Editor::FillMode::Solid);
        previousRevision = state.Editor->ViewportRevision();
        if (!waitForFrame(*dx11, previousRevision))
        {
            return finish(2);
        }
    }

    return finish(0);
}

int RunPresentationLiveSmoke(HINSTANCE instance, const std::filesystem::path& output,
    Evans::Editor::DemoId demo, bool pairOnly)
{
    std::ofstream trace(output);
    if (!trace) return 1;
    std::atomic<ULONGLONG> heartbeat{GetTickCount64()};
    std::atomic<bool> finished{false};
    std::thread watchdog([&] {
        while (!finished.load())
        {
            if (GetTickCount64() - heartbeat.load() > 30000)
            {
                std::ofstream timeout(output, std::ios::app);
                timeout << "FAIL native message loop stalled for 30 seconds\n" << std::flush;
                std::_Exit(124);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    struct JoinWatchdog
    {
        std::atomic<bool>& Finished;
        std::thread& Thread;
        ~JoinWatchdog() { Finished = true; Thread.join(); }
    } join{finished, watchdog};
    WindowState state;
    state.PresentationTrace = &trace;
    state.Editor = CreateEditor();
    state.Editor->SelectDemo(demo);
    state.Editor->SetTeapotRotationSpeed(0.0f);
    if (!HasBackend(*state.Editor, "renderer.dx12") ||
        !HasBackend(*state.Editor, "renderer.vulkan")) return 4;
    state.Editor->SelectBackend("renderer.dx12");
    if (!RegisterEditorWindowClass(instance)) return 3;
    HWND window = CreateWindowExW(0, WindowClassName, L"Evans Live Switch Regression",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        pairOnly ? 1440 : 640, pairOnly ? 900 : 480,
        nullptr, nullptr, instance, &state);
    if (window == nullptr) return 4;
    // The first ShowWindow may honor a hidden launcher STARTUPINFO instead.
    // The regression must exercise a genuinely visible native swapchain.
    ShowWindow(window, SW_SHOW);
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    TracePresentation(state, "visible=" + std::to_string(IsWindowVisible(window)) +
        " client=" + std::to_string(state.ClientWidth) + "x" + std::to_string(state.ClientHeight));
    const auto finish = [&](int result) {
        TracePresentation(state, "finish " + std::to_string(result));
        if (IsWindow(window)) DestroyWindow(window);
        return result;
    };
    const auto wait = [&](const std::string& backend, std::uint64_t previousFrame,
                          std::uint64_t previousRevision, bool animated = false) {
        const ULONGLONG deadline = GetTickCount64() + 45000;
        while (GetTickCount64() < deadline)
        {
            heartbeat = GetTickCount64();
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT) return false;
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            if (state.Editor->StatusText().rfind("Render failed", 0) == 0)
            {
                TracePresentation(state, "FAIL " + state.Editor->StatusText());
                return false;
            }
            if (state.PresentationFailures != 0 || !state.PresentationError.empty())
            {
                TracePresentation(state, "FAIL " + state.PresentationError);
                return false;
            }
            if (state.PresentationSelection == backend &&
                state.PresentedFrames >= previousFrame + 6 &&
                state.Editor->ViewportRevision() > previousRevision &&
                (animated || state.Editor->StatusText().rfind("Frame completed with ", 0) == 0))
                return state.Editor->ViewportImage().IsValid() &&
                    state.Editor->ViewportBackend() == backend;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        TracePresentation(state, "FAIL frame timeout: " + state.Editor->StatusText());
        return false;
    };
    if (!IsWindowVisible(window)) return finish(7);
    state.Editor->RequestRender();
    if (!wait("renderer.dx12", state.PresentedFrames, 0)) return finish(5);
    if (!state.Editor->Scene().Nodes.empty())
        state.Editor->SelectNode(state.Editor->Scene().Nodes.front().Id);
    struct SelectionCase
    {
        std::string Backend;
        int Mode;
        int Flags;
        bool CheckPixels;
    };
    std::vector<SelectionCase> selections;
    std::size_t directedCases = 0;
    const auto appendPair = [&](const std::string& source, const std::string& destination,
                                int mode, int flags) {
        selections.push_back({source, mode, flags, false});
        selections.push_back({destination, mode, flags, true});
        ++directedCases;
    };
    for (int mode = 0; mode < 3; ++mode)
        for (int flags = 0; flags < 4; ++flags)
        {
            if (pairOnly)
            {
                for (int repeat = 0; repeat < 2; ++repeat)
                {
                    appendPair("renderer.dx12", "renderer.vulkan", mode, flags);
                    appendPair("renderer.vulkan", "renderer.dx12", mode, flags);
                }
            }
            else
            {
                for (const auto& source : state.Editor->Backends())
                    for (const auto& destination : state.Editor->Backends())
                        if (source.Id != destination.Id)
                            appendPair(source.Id, destination.Id, mode, flags);
            }
        }
    std::size_t captures = 0;
    for (const auto& selection : selections)
    {
        const auto expectedAPI = Evans::Editor::ResolvePresentationAPI(
            selection.Backend, state.LastRealtimePresentation);
        TracePresentation(state, "request " + state.Editor->SelectedBackend() + " -> " +
            selection.Backend + " mode=" + std::to_string(selection.Mode) +
            " flags=" + std::to_string(selection.Flags));
        const auto previousFrame = state.PresentedFrames;
        const auto previousRevision = state.Editor->ViewportRevision();
        // Start source work before applying the mid-frame UI transition.
        state.Editor->RequestRender();
        state.QueuedBackendSelection = selection.Backend;
        state.QueuedFillMode = selection.Mode;
        state.QueuedEffectFlags = selection.Flags;
        state.ReenterOnNextFrame = true;
        InvalidateRect(window, nullptr, FALSE);
        if (!wait(selection.Backend, previousFrame, previousRevision)) return finish(6);
        if (!state.Presenter || state.Presenter->API() != expectedAPI) return finish(8);
        if (selection.CheckPixels)
        {
            Image capture;
            std::string error;
            if (!DrawEditorFrame(state, &capture, false, &error))
            {
                TracePresentation(state, "FAIL capture: " + error);
                return finish(9);
            }
            const Image& expected = state.Editor->ViewportImage();
            const Rect viewport = EditorViewportBounds(state);
            if (!capture.IsValid() || expected.Width != static_cast<std::uint32_t>(viewport.Width) ||
                expected.Height != static_cast<std::uint32_t>(viewport.Height)) return finish(10);
            std::uint64_t difference = 0;
            std::size_t large = 0;
            for (std::uint32_t y = 0; y < expected.Height; ++y)
                for (std::uint32_t x = 0; x < expected.Width; ++x)
                {
                    const std::size_t source = (static_cast<std::size_t>(y) * expected.Width + x) * 4;
                    const std::size_t destination = ((static_cast<std::size_t>(y) +
                        static_cast<std::uint32_t>(viewport.Y)) * capture.Width + x +
                        static_cast<std::uint32_t>(viewport.X)) * 4;
                    if (destination + 3 >= capture.PixelsRGBA8.size()) return finish(11);
                    for (std::size_t channel = 0; channel < 3; ++channel)
                    {
                        const int delta = std::abs(int(expected.PixelsRGBA8[source + channel]) -
                            int(capture.PixelsRGBA8[destination + channel]));
                        difference += delta;
                        large += delta > 4 ? 1 : 0;
                    }
                }
            const double channels = static_cast<double>(expected.Width) * expected.Height * 3;
            if (difference / channels > 0.5 || large / channels > 0.005)
            {
                TracePresentation(state, "FAIL presented pixels: mean=" +
                    std::to_string(difference / channels) + " fraction=" + std::to_string(large / channels));
                SavePPM(expected, output.string() + ".expected.ppm");
                SavePPM(capture, output.string() + ".presented.ppm");
                return finish(12);
            }
            ++captures;
        }
        TracePresentation(state, "completed " + selection.Backend);
        PROCESS_MEMORY_COUNTERS_EX memory{};
        if (!K32GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) return finish(17);
        TracePresentation(state, "memory private=" + std::to_string(memory.PrivateUsage));
        // Stop runaway regressions before exhausting the interactive workstation.
        if (memory.PrivateUsage > 16ULL * 1024 * 1024 * 1024)
        {
            TracePresentation(state, "FAIL native switching exceeded 16 GiB of private memory");
            return finish(18);
        }
    }
    if (demo == Evans::Editor::DemoId::UtahTeapot)
    {
        state.Editor->SetTeapotRotationSpeed(90.0f);
        std::size_t animatedCases = 0;
        for (const auto& selection : selections)
        {
            if (selection.Mode != 2 || selection.Flags != 3) continue;
            const auto previousFrame = state.PresentedFrames;
            const auto previousRevision = state.Editor->ViewportRevision();
            const auto expectedAPI = Evans::Editor::ResolvePresentationAPI(
                selection.Backend, state.LastRealtimePresentation);
            TracePresentation(state, "animated request " + selection.Backend);
            state.QueuedBackendSelection = selection.Backend;
            state.QueuedFillMode = selection.Mode;
            state.QueuedEffectFlags = selection.Flags;
            state.ReenterOnNextFrame = true;
            InvalidateRect(window, nullptr, FALSE);
            // Realtime animation may start the next job as soon as a frame
            // completes, so publication/ownership matters instead of idle status.
            if (!wait(selection.Backend, previousFrame, previousRevision, true) ||
                !state.Presenter || state.Presenter->API() != expectedAPI) return finish(21);
            if (selection.CheckPixels) ++animatedCases;
        }
        state.Editor->SetTeapotRotationSpeed(0.0f);
        TracePresentation(state, "PASS " + std::to_string(animatedCases) + " animated directed cases");
    }
    std::vector<std::string> rapidBackends;
    if (pairOnly) rapidBackends = {"renderer.dx12", "renderer.vulkan"};
    else
        for (const auto& backend : state.Editor->Backends()) rapidBackends.push_back(backend.Id);
    std::size_t rapidSelections = 0;
    for (int mode = 0; mode < 3; ++mode)
        for (int flags = 0; flags < 4; ++flags)
        {
            for (const auto& backend : rapidBackends)
            {
                heartbeat = GetTickCount64();
                TracePresentation(state, "rapid request " + backend);
                state.QueuedBackendSelection = backend;
                state.QueuedFillMode = mode;
                state.QueuedEffectFlags = flags;
                state.ReenterOnNextFrame = true;
                InvalidateRect(window, nullptr, FALSE);
                UpdateWindow(window);
                MSG message{};
                while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
                {
                    if (message.message == WM_QUIT) return finish(13);
                    TranslateMessage(&message);
                    DispatchMessageW(&message);
                }
                if (!state.QueuedBackendSelection.empty() || state.PresentationFailures != 0 ||
                    !state.PresentationError.empty() ||
                    state.Editor->StatusText().rfind("Render failed", 0) == 0) return finish(14);
                ++rapidSelections;
            }
            ShowWindow(window, SW_MINIMIZE);
            const auto minimizedFrames = state.PresentedFrames;
            InvalidateRect(window, nullptr, FALSE);
            UpdateWindow(window);
            if (state.PresentedFrames != minimizedFrames) return finish(15);
            ShowWindow(window, SW_RESTORE);
            // Restore can resize and retire the currently requested frame.
            const auto previousRevision = state.Editor->ViewportRevision();
            state.Editor->RequestRender();
            if (!wait(rapidBackends.back(), state.PresentedFrames, previousRevision)) return finish(16);
        }
    TracePresentation(state, "PASS " + std::to_string(rapidSelections) +
        " rapid native selections and 12 minimize/restore cycles");
    state.CloseOnNextFrame = true;
    state.Editor->RequestRender();
    InvalidateRect(window, nullptr, FALSE);
    UpdateWindow(window);
    if (!IsWindow(window) || state.PresentationFailures != 0) return finish(19);
    MSG closeMessage{};
    while (PeekMessageW(&closeMessage, nullptr, 0, 0, PM_REMOVE))
    {
        if (closeMessage.message == WM_QUIT) break;
        TranslateMessage(&closeMessage);
        DispatchMessageW(&closeMessage);
    }
    if (IsWindow(window)) return finish(20);
    TracePresentation(state, "PASS deferred close with a pending renderer frame");
    TracePresentation(state, "PASS " + std::to_string(directedCases) +
        " native directed feature cases, " + std::to_string(captures) + " presented-image comparisons");
    return finish(0);
}

int RunWindow(HINSTANCE instance, int showCommand, const std::filesystem::path& tracePath)
{
    std::ofstream trace;
    if (!tracePath.empty())
    {
        trace.open(tracePath);
        if (!trace) return 1;
    }
    WindowState state;
    if (trace.is_open()) state.PresentationTrace = &trace;
    state.Editor = CreateEditor();

    if (!RegisterEditorWindowClass(instance))
    {
        return 3;
    }

    HWND window = CreateWindowExW(
        0,
        WindowClassName,
        L"Evans Renderer Editor",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1440,
        900,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 4;
    }
    ShowWindow(window, showCommand);
    UpdateWindow(window);

    MSG message = {};
    while (GetMessageW(&message, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}

#endif
} // namespace

#if defined(_WIN32)
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    int argumentCount = 0;
    wchar_t** arguments = CommandLineToArgvW(GetCommandLineW(), &argumentCount);
    bool smoke = false;
    bool switchSmoke = false;
    bool switchStressSmoke = false;
    bool realtimeSmoke = false;
    bool dx11SemanticSmoke = false;
    bool realtimeSemanticSmoke = false;
    bool shaderParserSmoke = false;
    bool catalogSmoke = false;
    bool uiSmoke = false;
    bool presentationSwitchSmoke = false;
    bool presentationLiveSmoke = false;
    bool presentationPairSmoke = false;
    std::filesystem::path presentationTrace;
    std::filesystem::path output = L"EvansTriangle.ppm";
    bool outputExplicit = false;
    std::string backend = PlatformRealtimeBackend();
    Evans::Editor::DemoId demo = Evans::Editor::DemoId::Triangle;
    Evans::Editor::FillMode fillMode = Evans::Editor::FillMode::Solid;
    bool shadowsEnabled = true;
    bool outlineEnabled = true;
    bool selectFirstNode = false;
    bool pickCenter = false;
    for (int index = 1; index < argumentCount; ++index)
    {
        const std::wstring argument = arguments[index];
        if (argument == L"--smoke")
        {
            smoke = true;
        }
        else if (argument == L"--switch-smoke")
        {
            switchSmoke = true;
        }
        else if (argument == L"--switch-stress-smoke")
        {
            switchStressSmoke = true;
        }
        else if (argument == L"--realtime-smoke")
        {
            realtimeSmoke = true;
        }
        else if (argument == L"--dx11-semantic-smoke")
        {
            dx11SemanticSmoke = true;
        }
        else if (argument == L"--rhi-semantic-smoke")
        {
            realtimeSemanticSmoke = true;
        }
        else if (argument == L"--shader-parser-smoke")
        {
            shaderParserSmoke = true;
        }
        else if (argument == L"--catalog-smoke")
        {
            catalogSmoke = true;
        }
        else if (argument == L"--ui-smoke")
        {
            uiSmoke = true;
        }
        else if (argument == L"--presentation-pair-smoke")
        {
            presentationPairSmoke = true;
            presentationLiveSmoke = true;
        }
        else if (argument == L"--presentation-live-smoke")
        {
            presentationLiveSmoke = true;
        }
        else if (argument == L"--presentation-switch-smoke")
        {
            presentationSwitchSmoke = true;
        }
        else if (argument == L"--presentation-trace" && index + 1 < argumentCount)
        {
            presentationTrace = arguments[++index];
        }
        else if (argument == L"--output" && index + 1 < argumentCount)
        {
            output = arguments[++index];
            outputExplicit = true;
        }
        else if (argument == L"--backend" && index + 1 < argumentCount)
        {
            backend = NarrowAscii(arguments[++index]);
        }
        else if (argument == L"--demo" && index + 1 < argumentCount)
        {
            demo = ParseDemo(NarrowAscii(arguments[++index]));
        }
        else if (argument == L"--fill" && index + 1 < argumentCount)
        {
            fillMode = ParseFillMode(NarrowAscii(arguments[++index]));
        }
        else if (argument == L"--no-shadows")
        {
            shadowsEnabled = false;
        }
        else if (argument == L"--no-outline")
        {
            outlineEnabled = false;
        }
        else if (argument == L"--select-first")
        {
            selectFirstNode = true;
        }
        else if (argument == L"--pick-center")
        {
            pickCenter = true;
        }
    }
    LocalFree(arguments);
    if (shaderParserSmoke)
    {
        return RunShaderParserSmoke();
    }
    if (dx11SemanticSmoke)
    {
        backend = "renderer.dx11";
        if (!outputExplicit)
        {
            output = L"EvansDX11Semantic.ppm";
        }
        return RunRealtimeSemanticSmoke(output, backend);
    }
    if (realtimeSemanticSmoke)
    {
        if (!outputExplicit)
        {
            output = L"EvansRealtimeSemantic.ppm";
        }
        return RunRealtimeSemanticSmoke(output, backend);
    }
    if (uiSmoke)
    {
        if (!outputExplicit)
        {
            output = L"EvansEditorUi.bmp";
        }
        return RunUiSmoke(instance, output, backend, demo, shadowsEnabled);
    }
    if (presentationLiveSmoke)
    {
        return RunPresentationLiveSmoke(instance, output, demo, presentationPairSmoke);
    }
    if (presentationSwitchSmoke)
    {
        return RunPresentationSwitchSmoke(instance);
    }
    if (switchSmoke)
    {
        return RunSwitchSmoke(output);
    }
    if (switchStressSmoke)
    {
        return RunSwitchStressSmoke(output);
    }
    if (realtimeSmoke)
    {
        return RunRealtimeSmoke(output, backend);
    }
    if (catalogSmoke)
    {
        return RunCatalogSmoke();
    }
    return smoke
        ? RunSmoke(
              output,
              backend,
              demo,
              fillMode,
              shadowsEnabled,
              outlineEnabled,
              selectFirstNode,
              pickCenter)
        : RunWindow(instance, showCommand, presentationTrace);
}
#else
int main(int argumentCount, char** arguments)
{
    bool smoke = false;
    std::filesystem::path output = "EvansTriangle.ppm";
    std::string backend = PlatformRealtimeBackend();
    Evans::Editor::DemoId demo = Evans::Editor::DemoId::Triangle;
    Evans::Editor::FillMode fillMode = Evans::Editor::FillMode::Solid;
    bool shadowsEnabled = true;
    bool outlineEnabled = true;
    bool selectFirstNode = false;
    bool pickCenter = false;
    bool switchSmoke = false;
    bool switchStressSmoke = false;
    bool realtimeSmoke = false;
    bool dx11SemanticSmoke = false;
    bool realtimeSemanticSmoke = false;
    bool shaderParserSmoke = false;
    bool catalogSmoke = false;
    for (int index = 1; index < argumentCount; ++index)
    {
        const std::string argument = arguments[index];
        if (argument == "--smoke")
        {
            smoke = true;
        }
        else if (argument == "--switch-smoke")
        {
            switchSmoke = true;
        }
        else if (argument == "--switch-stress-smoke")
        {
            switchStressSmoke = true;
        }
        else if (argument == "--realtime-smoke")
        {
            realtimeSmoke = true;
        }
        else if (argument == "--dx11-semantic-smoke")
        {
            dx11SemanticSmoke = true;
        }
        else if (argument == "--rhi-semantic-smoke")
        {
            realtimeSemanticSmoke = true;
        }
        else if (argument == "--shader-parser-smoke")
        {
            shaderParserSmoke = true;
        }
        else if (argument == "--catalog-smoke")
        {
            catalogSmoke = true;
        }
        else if (argument == "--output" && index + 1 < argumentCount)
        {
            output = arguments[++index];
        }
        else if (argument == "--backend" && index + 1 < argumentCount)
        {
            backend = arguments[++index];
        }
        else if (argument == "--demo" && index + 1 < argumentCount)
        {
            demo = ParseDemo(arguments[++index]);
        }
        else if (argument == "--fill" && index + 1 < argumentCount)
        {
            fillMode = ParseFillMode(arguments[++index]);
        }
        else if (argument == "--no-shadows")
        {
            shadowsEnabled = false;
        }
        else if (argument == "--no-outline")
        {
            outlineEnabled = false;
        }
        else if (argument == "--select-first")
        {
            selectFirstNode = true;
        }
        else if (argument == "--pick-center")
        {
            pickCenter = true;
        }
    }
    if (shaderParserSmoke)
    {
        return RunShaderParserSmoke();
    }
    if (dx11SemanticSmoke)
    {
        return RunRealtimeSemanticSmoke(output, "renderer.dx11");
    }
    if (realtimeSemanticSmoke)
    {
        return RunRealtimeSemanticSmoke(output, backend);
    }
    if (switchSmoke)
    {
        return RunSwitchSmoke(output);
    }
    if (switchStressSmoke)
    {
        return RunSwitchStressSmoke(output);
    }
    if (realtimeSmoke)
    {
        return RunRealtimeSmoke(output, backend);
    }
    if (catalogSmoke)
    {
        return RunCatalogSmoke();
    }
#if defined(__APPLE__)
    if (!smoke)
    {
        return RunMacEditor();
    }
#else
    (void)smoke;
#endif
    return RunSmoke(
        output,
        backend,
        demo,
        fillMode,
        shadowsEnabled,
        outlineEnabled,
        selectFirstNode,
        pickCenter);
}
#endif
