#include "Renderer/Types.h"
#include "Renderer/D3DShaderCache.h"
#include "Editor/EditorPresenterDX11.h"

#if defined(_WIN32)

#include "Editor/ImguiRenderDX11.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>

namespace Evans::Editor
{
namespace
{

template <class T>
void SafeRelease(T*& value) noexcept
{
    if (value != nullptr)
    {
        value->Release();
        value = nullptr;
    }
}

std::string HResultMessage(const char* operation, HRESULT result)
{
    std::ostringstream stream;
    stream << operation << " failed with HRESULT 0x"
           << std::hex << std::uppercase << static_cast<unsigned long>(result);
    return stream.str();
}

struct ViewportVertex
{
    float Position[2];
    float TexCoord[2];
};

} // namespace

struct EditorPresenterDX11::Impl
{
    ~Impl()
    {
        Shutdown();
    }

    bool Fail(const std::string& message)
    {
        LastError = message;
        return false;
    }

    bool Initialize(HWND window, int width, int height)
    {
        Shutdown();
        if (window == nullptr || width <= 0 || height <= 0)
        {
            return Fail("The editor presenter requires a valid window and dimensions.");
        }

        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        swapChainDesc.BufferDesc.Width = static_cast<UINT>(width);
        swapChainDesc.BufferDesc.Height = static_cast<UINT>(height);
        swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.BufferCount = 2;
        swapChainDesc.OutputWindow = window;
        swapChainDesc.Windowed = TRUE;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        constexpr std::array<D3D_FEATURE_LEVEL, 4> featureLevels = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0};
        constexpr std::array<D3D_DRIVER_TYPE, 3> driverTypes = {
            D3D_DRIVER_TYPE_HARDWARE,
            D3D_DRIVER_TYPE_WARP,
            D3D_DRIVER_TYPE_REFERENCE};

        HRESULT result = E_FAIL;
        for (D3D_DRIVER_TYPE driverType : driverTypes)
        {
            result = D3D11CreateDeviceAndSwapChain(
                nullptr,
                driverType,
                nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                featureLevels.data(),
                static_cast<UINT>(featureLevels.size()),
                D3D11_SDK_VERSION,
                &swapChainDesc,
                &SwapChain,
                &Device,
                &FeatureLevel,
                &Context);
            if (result == E_INVALIDARG)
            {
                result = D3D11CreateDeviceAndSwapChain(
                    nullptr,
                    driverType,
                    nullptr,
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    featureLevels.data() + 1,
                    static_cast<UINT>(featureLevels.size() - 1),
                    D3D11_SDK_VERSION,
                    &swapChainDesc,
                    &SwapChain,
                    &Device,
                    &FeatureLevel,
                    &Context);
            }
            if (SUCCEEDED(result))
            {
                break;
            }
            SafeRelease(Context);
            SafeRelease(Device);
            SafeRelease(SwapChain);
        }
        if (FAILED(result))
        {
            return Fail(HResultMessage("D3D11CreateDeviceAndSwapChain", result));
        }

        Window = window;
        Width = width;
        Height = height;
        if (!CreateBackBuffer() || !CreateViewportPipeline())
        {
            return false;
        }
        if (!ImguiRenderer.Init(Device, Context))
        {
            return Fail("The Riemann-style ImGui renderer could not initialize.");
        }
        LastError.clear();
        return true;
    }

    void Shutdown() noexcept
    {
        ImguiRenderer.Shutdown();
        if (Context != nullptr)
        {
            ID3D11ShaderResourceView* nullView = nullptr;
            Context->PSSetShaderResources(0, 1, &nullView);
            Context->OMSetRenderTargets(0, nullptr, nullptr);
            Context->ClearState();
        }
        SafeRelease(ImageView);
        SafeRelease(ImageTexture);
        SafeRelease(ViewportSampler);
        SafeRelease(ViewportDepthState);
        SafeRelease(ViewportRasterizerState);
        SafeRelease(ViewportVertexBuffer);
        SafeRelease(ViewportInputLayout);
        SafeRelease(ViewportPixelShader);
        SafeRelease(ViewportVertexShader);
        SafeRelease(BackBufferView);
        SafeRelease(SwapChain);
        // Flush deferred destruction before another API creates a swapchain
        // for this window; ClearState above releases all context-held views.
        if (Context != nullptr) Context->Flush();
        SafeRelease(Context);
        SafeRelease(Device);
        Window = nullptr;
        Width = 0;
        Height = 0;
        ImageWidth = 0;
        ImageHeight = 0;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    }

    bool CreateBackBuffer()
    {
        ID3D11Texture2D* backBuffer = nullptr;
        const HRESULT result = SwapChain->GetBuffer(
            0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
        if (FAILED(result))
        {
            return Fail(HResultMessage("IDXGISwapChain::GetBuffer", result));
        }
        const HRESULT viewResult = Device->CreateRenderTargetView(backBuffer, nullptr, &BackBufferView);
        backBuffer->Release();
        if (FAILED(viewResult))
        {
            return Fail(HResultMessage("CreateRenderTargetView", viewResult));
        }
        return true;
    }

    bool CreateViewportPipeline()
    {
        static constexpr const char* shaderSource = R"HLSL(
Texture2D ViewportTexture : register(t0);
SamplerState ViewportSampler : register(s0);

struct VSInput
{
    float2 Position : POSITION;
    float2 TexCoord : TEXCOORD0;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VSOutput VS(VSInput input)
{
    VSOutput output;
    output.Position = float4(input.Position, 0.0f, 1.0f);
    output.TexCoord = input.TexCoord;
    return output;
}

float4 PS(VSOutput input) : SV_TARGET
{
    return ViewportTexture.Sample(ViewportSampler, input.TexCoord);
}
)HLSL";

        ID3DBlob* vertexBlob = nullptr;
        ID3DBlob* pixelBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;
        HRESULT result = Evans::Renderer::CompileCachedD3DShader(
            shaderSource,
            std::strlen(shaderSource),
            "EvansEditorPresenter",
            nullptr,
            nullptr,
            "VS",
            "vs_4_0",
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &vertexBlob,
            &errorBlob);
        if (FAILED(result))
        {
            std::string error = errorBlob != nullptr
                ? static_cast<const char*>(errorBlob->GetBufferPointer())
                : HResultMessage("Evans::Renderer::CompileCachedD3DShader(VS)", result);
            SafeRelease(errorBlob);
            return Fail(error);
        }
        SafeRelease(errorBlob);

        result = Evans::Renderer::CompileCachedD3DShader(
            shaderSource,
            std::strlen(shaderSource),
            "EvansEditorPresenter",
            nullptr,
            nullptr,
            "PS",
            "ps_4_0",
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &pixelBlob,
            &errorBlob);
        if (FAILED(result))
        {
            std::string error = errorBlob != nullptr
                ? static_cast<const char*>(errorBlob->GetBufferPointer())
                : HResultMessage("Evans::Renderer::CompileCachedD3DShader(PS)", result);
            SafeRelease(errorBlob);
            SafeRelease(vertexBlob);
            return Fail(error);
        }
        SafeRelease(errorBlob);

        result = Device->CreateVertexShader(
            vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), nullptr, &ViewportVertexShader);
        if (SUCCEEDED(result))
        {
            result = Device->CreatePixelShader(
                pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, &ViewportPixelShader);
        }
        if (SUCCEEDED(result))
        {
            const D3D11_INPUT_ELEMENT_DESC elements[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0,
                    static_cast<UINT>(offsetof(ViewportVertex, Position)),
                    D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0,
                    static_cast<UINT>(offsetof(ViewportVertex, TexCoord)),
                    D3D11_INPUT_PER_VERTEX_DATA, 0}};
            result = Device->CreateInputLayout(
                elements,
                ARRAYSIZE(elements),
                vertexBlob->GetBufferPointer(),
                vertexBlob->GetBufferSize(),
                &ViewportInputLayout);
        }
        SafeRelease(vertexBlob);
        SafeRelease(pixelBlob);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport shaders", result));
        }

        D3D11_BUFFER_DESC vertexBufferDesc = {};
        vertexBufferDesc.ByteWidth = static_cast<UINT>(sizeof(ViewportVertex) * 6u);
        vertexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        result = Device->CreateBuffer(&vertexBufferDesc, nullptr, &ViewportVertexBuffer);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport vertex buffer", result));
        }

        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        result = Device->CreateSamplerState(&samplerDesc, &ViewportSampler);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport sampler", result));
        }

        D3D11_RASTERIZER_DESC rasterizerDesc = {};
        rasterizerDesc.FillMode = D3D11_FILL_SOLID;
        rasterizerDesc.CullMode = D3D11_CULL_NONE;
        rasterizerDesc.DepthClipEnable = TRUE;
        result = Device->CreateRasterizerState(&rasterizerDesc, &ViewportRasterizerState);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport rasterizer state", result));
        }

        D3D11_DEPTH_STENCIL_DESC depthDesc = {};
        depthDesc.DepthEnable = FALSE;
        depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        depthDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
        result = Device->CreateDepthStencilState(&depthDesc, &ViewportDepthState);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport depth state", result));
        }
        return true;
    }

    bool Resize(int width, int height)
    {
        if (SwapChain == nullptr || width <= 0 || height <= 0)
        {
            return false;
        }
        if (width == Width && height == Height && BackBufferView != nullptr)
        {
            return true;
        }

        Context->OMSetRenderTargets(0, nullptr, nullptr);
        SafeRelease(BackBufferView);
        const HRESULT result = SwapChain->ResizeBuffers(
            0, static_cast<UINT>(width), static_cast<UINT>(height), DXGI_FORMAT_UNKNOWN, 0);
        if (FAILED(result))
        {
            return Fail(HResultMessage("IDXGISwapChain::ResizeBuffers", result));
        }
        Width = width;
        Height = height;
        return CreateBackBuffer();
    }

    bool EnsureImageTexture(const Image& image)
    {
        if (!image.IsValid())
        {
            return false;
        }
        if (ImageTexture != nullptr && ImageWidth == image.Width && ImageHeight == image.Height && GpuImageTexture == bool(image.GpuColor))
        {
            return true;
        }

        SafeRelease(ImageView);
        SafeRelease(ImageTexture);
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = image.Width;
        desc.Height = image.Height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = image.GpuColor ? D3D11_USAGE_DEFAULT : D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = image.GpuColor ? 0 : D3D11_CPU_ACCESS_WRITE;
        HRESULT result = Device->CreateTexture2D(&desc, nullptr, &ImageTexture);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport texture", result));
        }
        result = Device->CreateShaderResourceView(ImageTexture, nullptr, &ImageView);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport shader resource view", result));
        }
        GpuImageTexture = bool(image.GpuColor);
        ImageWidth = image.Width;
        ImageHeight = image.Height;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
        return true;
    }

    bool UploadImage(const Image& image, std::uint64_t revision)
    {
        if (!EnsureImageTexture(image))
        {
            return false;
        }
        if (UploadedRevision == revision)
        {
            return true;
        }

        if (image.GpuColor)
        {
            Microsoft::WRL::ComPtr<ID3D11Texture2D> shared;
            HRESULT status = Device->OpenSharedResource(image.GpuColor->SharedHandle,
                __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(shared.GetAddressOf()));
            if (FAILED(status)) return Fail(HResultMessage("Open GPU viewport surface", status));
            Microsoft::WRL::ComPtr<IDXGIKeyedMutex> mutex;
            status = shared.As(&mutex);
            if (FAILED(status)) return Fail(HResultMessage("Query GPU viewport mutex", status));
            status = mutex->AcquireSync(0, 5000);
            if (status != S_OK) return Fail(HResultMessage("Acquire GPU viewport consumer", status));
            ID3D11ShaderResourceView* empty = nullptr;
            Context->PSSetShaderResources(0, 1, &empty);
            Context->CopyResource(ImageTexture, shared.Get());
            status = mutex->ReleaseSync(0);
            Context->Flush();
            if (FAILED(status)) return Fail(HResultMessage("Release GPU viewport consumer", status));
            UploadedRevision = revision;
            return true;
        }
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        const HRESULT result = Context->Map(ImageTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Map viewport texture", result));
        }
        const std::size_t sourcePitch = static_cast<std::size_t>(image.Width) * 4u;
        for (std::uint32_t row = 0; row < image.Height; ++row)
        {
            std::memcpy(
                static_cast<std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(row) * mapped.RowPitch,
                image.PixelsRGBA8.data() + static_cast<std::size_t>(row) * sourcePitch,
                sourcePitch);
        }
        Context->Unmap(ImageTexture, 0);
        UploadedRevision = revision;
        return true;
    }

    void DrawViewport(int x, int y, int width, int height)
    {
        if (ImageView == nullptr || width <= 0 || height <= 0 || Width <= 0 || Height <= 0)
        {
            return;
        }

        const float left = static_cast<float>(x) * 2.0f / static_cast<float>(Width) - 1.0f;
        const float right = static_cast<float>(x + width) * 2.0f / static_cast<float>(Width) - 1.0f;
        const float top = 1.0f - static_cast<float>(y) * 2.0f / static_cast<float>(Height);
        const float bottom = 1.0f - static_cast<float>(y + height) * 2.0f / static_cast<float>(Height);
        const ViewportVertex vertices[6] = {
            {{left, top}, {0.0f, 0.0f}},
            {{right, top}, {1.0f, 0.0f}},
            {{right, bottom}, {1.0f, 1.0f}},
            {{left, top}, {0.0f, 0.0f}},
            {{right, bottom}, {1.0f, 1.0f}},
            {{left, bottom}, {0.0f, 1.0f}}};

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (FAILED(Context->Map(ViewportVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        {
            return;
        }
        std::memcpy(mapped.pData, vertices, sizeof(vertices));
        Context->Unmap(ViewportVertexBuffer, 0);

        D3D11_VIEWPORT viewport = {};
        viewport.Width = static_cast<float>(Width);
        viewport.Height = static_cast<float>(Height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        Context->RSSetViewports(1, &viewport);
        Context->RSSetState(ViewportRasterizerState);
        Context->OMSetDepthStencilState(ViewportDepthState, 0);
        Context->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
        Context->IASetInputLayout(ViewportInputLayout);
        Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const UINT stride = sizeof(ViewportVertex);
        const UINT offset = 0;
        Context->IASetVertexBuffers(0, 1, &ViewportVertexBuffer, &stride, &offset);
        Context->VSSetShader(ViewportVertexShader, nullptr, 0);
        Context->PSSetShader(ViewportPixelShader, nullptr, 0);
        Context->PSSetShaderResources(0, 1, &ImageView);
        Context->PSSetSamplers(0, 1, &ViewportSampler);
        Context->Draw(6, 0);
    }

    bool BeginFrame(
        const Image& image,
        std::uint64_t revision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight)
    {
        if (Context == nullptr || BackBufferView == nullptr)
        {
            return Fail("The editor presenter is not initialized.");
        }
        Context->OMSetRenderTargets(1, &BackBufferView, nullptr);
        constexpr float clearColor[4] = {0.035f, 0.043f, 0.059f, 1.0f};
        Context->ClearRenderTargetView(BackBufferView, clearColor);
        if (image.IsValid())
        {
            if (!UploadImage(image, revision))
            {
                return false;
            }
            DrawViewport(viewportX, viewportY, viewportWidth, viewportHeight);
        }
        return true;
    }

    void DrawImgui()
    {
        if (Context == nullptr || BackBufferView == nullptr)
        {
            return;
        }
        Context->OMSetRenderTargets(1, &BackBufferView, nullptr);
        ImguiRenderer.Draw(Width, Height);
    }

    bool EndFrame()
    {
        if (SwapChain == nullptr)
        {
            return false;
        }
        const HRESULT result = SwapChain->Present(0, 0);
        return SUCCEEDED(result) || Fail(HResultMessage("IDXGISwapChain::Present", result));
    }

    HWND Window = nullptr;
    int Width = 0;
    int Height = 0;
    D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_10_0;
    ID3D11Device* Device = nullptr;
    ID3D11DeviceContext* Context = nullptr;
    IDXGISwapChain* SwapChain = nullptr;
    ID3D11RenderTargetView* BackBufferView = nullptr;
    ID3D11VertexShader* ViewportVertexShader = nullptr;
    ID3D11PixelShader* ViewportPixelShader = nullptr;
    ID3D11InputLayout* ViewportInputLayout = nullptr;
    ID3D11Buffer* ViewportVertexBuffer = nullptr;
    ID3D11RasterizerState* ViewportRasterizerState = nullptr;
    ID3D11DepthStencilState* ViewportDepthState = nullptr;
    ID3D11SamplerState* ViewportSampler = nullptr;
    bool GpuImageTexture = false;
    ID3D11Texture2D* ImageTexture = nullptr;
    ID3D11ShaderResourceView* ImageView = nullptr;
    std::uint32_t ImageWidth = 0;
    std::uint32_t ImageHeight = 0;
    std::uint64_t UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    ImguiRenderDX11 ImguiRenderer;
    std::string LastError;
};

EditorPresenterDX11::EditorPresenterDX11()
    : Impl_(std::make_unique<Impl>())
{
}

EditorPresenterDX11::~EditorPresenterDX11() = default;

bool EditorPresenterDX11::Initialize(void* nativeWindow, int width, int height)
{
    return Impl_->Initialize(static_cast<HWND>(nativeWindow), width, height);
}

void EditorPresenterDX11::Shutdown() noexcept
{
    Impl_->Shutdown();
}

bool EditorPresenterDX11::Resize(int width, int height)
{
    return Impl_->Resize(width, height);
}

bool EditorPresenterDX11::BeginFrame(
    const Image& viewportImage,
    std::uint64_t viewportRevision,
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight)
{
    return Impl_->BeginFrame(
        viewportImage,
        viewportRevision,
        viewportX,
        viewportY,
        viewportWidth,
        viewportHeight);
}

void EditorPresenterDX11::DrawImgui()
{
    Impl_->DrawImgui();
}

bool EditorPresenterDX11::EndFrame()
{
    return Impl_->EndFrame();
}

bool EditorPresenterDX11::IsInitialized() const noexcept
{
    return Impl_->Device != nullptr && Impl_->SwapChain != nullptr && Impl_->BackBufferView != nullptr;
}

const std::string& EditorPresenterDX11::LastError() const noexcept
{
    return Impl_->LastError;
}

} // namespace Evans::Editor

#endif
