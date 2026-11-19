#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#include "Editor/ImguiRenderDX12.h"

#if defined(_WIN32)

#include "Editor/Imgui.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

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

constexpr float Pi = 3.14159265358979323846f;
constexpr float QueueScale = 1.0f / 8.0f;
constexpr float TabStops[] = {150.0f, 210.0f, 270.0f, 330.0f};

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept
{
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC BufferDescription(UINT64 size) noexcept
{
    D3D12_RESOURCE_DESC description = {};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Alignment = 0;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return description;
}

} // namespace

struct ImguiRenderDX12::Impl
{
    struct Glyph
    {
        float U0 = 0.0f;
        float V0 = 0.0f;
        float U1 = 0.0f;
        float V1 = 0.0f;
        float Width = 0.0f;
        float Height = 0.0f;
        float Advance = 0.0f;
    };

    struct Vertex
    {
        float Position[2] = {};
        float TexCoord[2] = {};
        std::uint32_t Color = 0;
    };

    enum class Texture : std::uint8_t
    {
        White,
        Font
    };

    struct Batch
    {
        UINT FirstVertex = 0;
        UINT VertexCount = 0;
        Texture SourceTexture = Texture::White;
        D3D12_RECT Scissor = {};
    };

    struct FrameBuffer
    {
        ID3D12Resource* Resource = nullptr;
        std::uint8_t* Mapped = nullptr;
        UINT Capacity = 0;
    };

    ~Impl()
    {
        Shutdown();
    }

    void Shutdown() noexcept
    {
        for (FrameBuffer& frame : FrameBuffers)
        {
            if (frame.Resource != nullptr && frame.Mapped != nullptr)
            {
                frame.Resource->Unmap(0, nullptr);
            }
            frame.Mapped = nullptr;
            frame.Capacity = 0;
            SafeRelease(frame.Resource);
        }
        FrameBuffers.clear();
        SafeRelease(FontUpload);
        SafeRelease(WhiteUpload);
        SafeRelease(FontTexture);
        SafeRelease(WhiteTexture);
        SafeRelease(TextureHeap);
        SafeRelease(PipelineState);
        SafeRelease(RootSignature);
        SafeRelease(Device);
        Vertices.clear();
        Batches.clear();
        Width = 1;
        Height = 1;
    }

    bool Init(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commands,
        DXGI_FORMAT renderTargetFormat,
        UINT frameCount)
    {
        Shutdown();
        if (device == nullptr || commands == nullptr || frameCount == 0)
        {
            return false;
        }
        Device = device;
        Device->AddRef();
        FrameBuffers.resize(frameCount);
        return CreateRootSignature() &&
               CreatePipeline(renderTargetFormat) &&
               CreateTextureHeap() &&
               CreateWhiteTexture(commands) &&
               CreateFontTexture(commands);
    }

    bool CreateRootSignature()
    {
        D3D12_DESCRIPTOR_RANGE textureRange = {};
        textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        textureRange.NumDescriptors = 1;
        textureRange.BaseShaderRegister = 0;
        textureRange.RegisterSpace = 0;
        textureRange.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER parameters[2] = {};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.ShaderRegister = 0;
        parameters[0].Constants.RegisterSpace = 0;
        parameters[0].Constants.Num32BitValues = 2;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[1].DescriptorTable.NumDescriptorRanges = 1;
        parameters[1].DescriptorTable.pDescriptorRanges = &textureRange;
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_STATIC_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MipLODBias = 0.0f;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderRegister = 0;
        sampler.RegisterSpace = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC description = {};
        description.NumParameters = ARRAYSIZE(parameters);
        description.pParameters = parameters;
        description.NumStaticSamplers = 1;
        description.pStaticSamplers = &sampler;
        description.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        ID3DBlob* signature = nullptr;
        ID3DBlob* error = nullptr;
        const HRESULT serializeResult = D3D12SerializeRootSignature(
            &description, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(serializeResult))
        {
            if (error != nullptr)
            {
                OutputDebugStringA(static_cast<const char*>(error->GetBufferPointer()));
            }
            SafeRelease(error);
            SafeRelease(signature);
            return false;
        }
        SafeRelease(error);
        const HRESULT result = Evans::Renderer::CreateCachedRootSignature(Device,
            0,
            signature->GetBufferPointer(),
            signature->GetBufferSize(),
            IID_PPV_ARGS(&RootSignature));
        SafeRelease(signature);
        return SUCCEEDED(result);
    }

    bool CreatePipeline(DXGI_FORMAT renderTargetFormat)
    {
        static constexpr const char* ShaderSource = R"HLSL(
cbuffer ImguiConstants : register(b0)
{
    float2 ScreenSize;
};
Texture2D UiTexture : register(t0);
SamplerState UiSampler : register(s0);

struct VSInput
{
    float2 Position : POSITION;
    float2 TexCoord : TEXCOORD0;
    float4 Color : COLOR0;
};
struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float4 Color : COLOR0;
};

VSOutput VS(VSInput input)
{
    VSOutput output;
    float2 p = input.Position / ScreenSize;
    output.Position = float4(p.x * 2.0f - 1.0f, 1.0f - p.y * 2.0f, 0.0f, 1.0f);
    output.TexCoord = input.TexCoord;
    output.Color = input.Color;
    return output;
}

float4 PS(VSOutput input) : SV_TARGET
{
    return input.Color * UiTexture.Sample(UiSampler, input.TexCoord);
}
)HLSL";

        ID3DBlob* vertexShader = nullptr;
        ID3DBlob* pixelShader = nullptr;
        ID3DBlob* error = nullptr;
        HRESULT result = Evans::Renderer::CompileCachedD3DShader(
            ShaderSource,
            std::strlen(ShaderSource),
            "EvansImguiDX12",
            nullptr,
            nullptr,
            "VS",
            "vs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &vertexShader,
            &error);
        if (FAILED(result))
        {
            if (error != nullptr)
            {
                OutputDebugStringA(static_cast<const char*>(error->GetBufferPointer()));
            }
            SafeRelease(error);
            return false;
        }
        SafeRelease(error);
        result = Evans::Renderer::CompileCachedD3DShader(
            ShaderSource,
            std::strlen(ShaderSource),
            "EvansImguiDX12",
            nullptr,
            nullptr,
            "PS",
            "ps_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &pixelShader,
            &error);
        if (FAILED(result))
        {
            if (error != nullptr)
            {
                OutputDebugStringA(static_cast<const char*>(error->GetBufferPointer()));
            }
            SafeRelease(error);
            SafeRelease(vertexShader);
            return false;
        }
        SafeRelease(error);

        const D3D12_INPUT_ELEMENT_DESC inputElements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0,
                static_cast<UINT>(offsetof(Vertex, Position)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0,
                static_cast<UINT>(offsetof(Vertex, TexCoord)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0,
                static_cast<UINT>(offsetof(Vertex, Color)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        D3D12_GRAPHICS_PIPELINE_STATE_DESC description = {};
        description.pRootSignature = RootSignature;
        description.VS = {vertexShader->GetBufferPointer(), vertexShader->GetBufferSize()};
        description.PS = {pixelShader->GetBufferPointer(), pixelShader->GetBufferSize()};
        description.BlendState.AlphaToCoverageEnable = FALSE;
        description.BlendState.IndependentBlendEnable = FALSE;
        D3D12_RENDER_TARGET_BLEND_DESC& blend = description.BlendState.RenderTarget[0];
        blend.BlendEnable = TRUE;
        blend.LogicOpEnable = FALSE;
        blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        blend.BlendOp = D3D12_BLEND_OP_ADD;
        blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blend.LogicOp = D3D12_LOGIC_OP_NOOP;
        blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        description.SampleMask = UINT_MAX;
        description.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        description.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        description.RasterizerState.FrontCounterClockwise = FALSE;
        description.RasterizerState.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        description.RasterizerState.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        description.RasterizerState.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        description.RasterizerState.DepthClipEnable = TRUE;
        description.RasterizerState.MultisampleEnable = FALSE;
        description.RasterizerState.AntialiasedLineEnable = FALSE;
        description.RasterizerState.ForcedSampleCount = 0;
        description.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        description.DepthStencilState.DepthEnable = FALSE;
        description.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        description.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        description.DepthStencilState.StencilEnable = FALSE;
        description.InputLayout = {inputElements, ARRAYSIZE(inputElements)};
        description.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        description.NumRenderTargets = 1;
        description.RTVFormats[0] = renderTargetFormat;
        description.SampleDesc.Count = 1;

        result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&description, IID_PPV_ARGS(&PipelineState));
        SafeRelease(pixelShader);
        SafeRelease(vertexShader);
        return SUCCEEDED(result);
    }

    bool CreateTextureHeap()
    {
        D3D12_DESCRIPTOR_HEAP_DESC description = {};
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        description.NumDescriptors = 2;
        description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&TextureHeap))))
        {
            return false;
        }
        DescriptorSize = Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }

    bool CreateTexture(
        ID3D12GraphicsCommandList* commands,
        UINT width,
        UINT height,
        const std::uint8_t* rgba,
        UINT descriptorIndex,
        ID3D12Resource*& texture,
        ID3D12Resource*& upload)
    {
        D3D12_RESOURCE_DESC textureDescription = {};
        textureDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDescription.Width = width;
        textureDescription.Height = height;
        textureDescription.DepthOrArraySize = 1;
        textureDescription.MipLevels = 1;
        textureDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDescription.SampleDesc.Count = 1;
        textureDescription.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

        const D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
        HRESULT result = Device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &textureDescription,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&texture));
        if (FAILED(result))
        {
            return false;
        }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT rowCount = 0;
        UINT64 rowSize = 0;
        UINT64 uploadSize = 0;
        Device->GetCopyableFootprints(
            &textureDescription, 0, 1, 0, &footprint, &rowCount, &rowSize, &uploadSize);
        const D3D12_HEAP_PROPERTIES uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_RESOURCE_DESC uploadDescription = BufferDescription(uploadSize);
        result = Device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &uploadDescription,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&upload));
        if (FAILED(result))
        {
            return false;
        }

        std::uint8_t* mapped = nullptr;
        D3D12_RANGE readRange = {0, 0};
        result = upload->Map(0, &readRange, reinterpret_cast<void**>(&mapped));
        if (FAILED(result))
        {
            return false;
        }
        const std::size_t sourcePitch = static_cast<std::size_t>(width) * 4u;
        for (UINT row = 0; row < height; ++row)
        {
            std::memcpy(
                mapped + footprint.Offset + static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
                rgba + static_cast<std::size_t>(row) * sourcePitch,
                sourcePitch);
        }
        upload->Unmap(0, nullptr);

        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = upload;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION destination = {};
        destination.pResource = texture;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commands->ResourceBarrier(1, &barrier);

        D3D12_SHADER_RESOURCE_VIEW_DESC view = {};
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = TextureHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(descriptorIndex) * DescriptorSize;
        Device->CreateShaderResourceView(texture, &view, handle);
        return true;
    }

    bool CreateWhiteTexture(ID3D12GraphicsCommandList* commands)
    {
        constexpr std::uint8_t white[] = {255, 255, 255, 255};
        return CreateTexture(commands, 1, 1, white, 0, WhiteTexture, WhiteUpload);
    }

    bool BuildFontAtlas(std::vector<std::uint8_t>& rgba, UINT& width, UINT& height)
    {
        width = 512;
        height = 512;
        BITMAPINFO bitmapInfo = {};
        bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfo.bmiHeader.biWidth = static_cast<LONG>(width);
        bitmapInfo.bmiHeader.biHeight = -static_cast<LONG>(height);
        bitmapInfo.bmiHeader.biPlanes = 1;
        bitmapInfo.bmiHeader.biBitCount = 32;
        bitmapInfo.bmiHeader.biCompression = BI_RGB;

        HDC dc = CreateCompatibleDC(nullptr);
        if (dc == nullptr)
        {
            return false;
        }
        void* bitmapPixels = nullptr;
        HBITMAP bitmap = CreateDIBSection(dc, &bitmapInfo, DIB_RGB_COLORS, &bitmapPixels, nullptr, 0);
        if (bitmap == nullptr || bitmapPixels == nullptr)
        {
            if (bitmap != nullptr)
            {
                DeleteObject(bitmap);
            }
            DeleteDC(dc);
            return false;
        }
        const HGDIOBJ oldBitmap = SelectObject(dc, bitmap);
        std::memset(bitmapPixels, 0, static_cast<std::size_t>(width) * height * 4u);

        HFONT font = CreateFontA(
            -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
        HGDIOBJ oldFont = nullptr;
        if (font != nullptr)
        {
            oldFont = SelectObject(dc, font);
        }
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255));
        TEXTMETRICA metrics = {};
        GetTextMetricsA(dc, &metrics);
        FontHeight = std::max(8, static_cast<int>(metrics.tmHeight));
        imguiSetFontHeight(FontHeight);

        int penX = 1;
        int penY = 1;
        const int rowHeight = FontHeight + 2;
        for (int character = 32; character < 128; ++character)
        {
            const char value = static_cast<char>(character);
            SIZE size = {};
            GetTextExtentPoint32A(dc, &value, 1, &size);
            const int glyphWidth = std::max(1, static_cast<int>(size.cx));
            if (penX + glyphWidth + 2 >= static_cast<int>(width))
            {
                penX = 1;
                penY += rowHeight;
            }
            if (penY + FontHeight + 2 >= static_cast<int>(height))
            {
                break;
            }
            Glyph& glyph = Glyphs[static_cast<std::size_t>(character - 32)];
            glyph.U0 = static_cast<float>(penX) / static_cast<float>(width);
            glyph.V0 = static_cast<float>(penY) / static_cast<float>(height);
            glyph.U1 = static_cast<float>(penX + glyphWidth) / static_cast<float>(width);
            glyph.V1 = static_cast<float>(penY + FontHeight) / static_cast<float>(height);
            glyph.Width = static_cast<float>(glyphWidth);
            glyph.Height = static_cast<float>(FontHeight);
            glyph.Advance = static_cast<float>(glyphWidth);
            TextOutA(dc, penX, penY, &value, 1);
            penX += glyphWidth + 1;
        }

        rgba.resize(static_cast<std::size_t>(width) * height * 4u);
        const std::uint8_t* source = static_cast<const std::uint8_t*>(bitmapPixels);
        for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(width) * height; ++pixel)
        {
            const std::uint8_t alpha = std::max(source[pixel * 4u],
                std::max(source[pixel * 4u + 1u], source[pixel * 4u + 2u]));
            rgba[pixel * 4u] = 255;
            rgba[pixel * 4u + 1u] = 255;
            rgba[pixel * 4u + 2u] = 255;
            rgba[pixel * 4u + 3u] = alpha;
        }

        if (oldFont != nullptr)
        {
            SelectObject(dc, oldFont);
        }
        if (font != nullptr)
        {
            DeleteObject(font);
        }
        SelectObject(dc, oldBitmap);
        DeleteObject(bitmap);
        DeleteDC(dc);
        return true;
    }

    bool CreateFontTexture(ID3D12GraphicsCommandList* commands)
    {
        std::vector<std::uint8_t> rgba;
        UINT width = 0;
        UINT height = 0;
        return BuildFontAtlas(rgba, width, height) &&
               CreateTexture(commands, width, height, rgba.data(), 1, FontTexture, FontUpload);
    }

    D3D12_RECT ClampScissor(int x, int y, int width, int height) const noexcept
    {
        D3D12_RECT rect = {};
        rect.left = std::max<LONG>(0, x);
        rect.top = std::max<LONG>(0, Height - (y + height));
        rect.right = std::min<LONG>(Width, x + width);
        rect.bottom = std::min<LONG>(Height, Height - y);
        rect.right = std::max(rect.left, rect.right);
        rect.bottom = std::max(rect.top, rect.bottom);
        return rect;
    }

    float ToScreenY(float value) const noexcept
    {
        return static_cast<float>(Height) - value;
    }

    void Append(const Vertex* source, UINT count, Texture texture)
    {
        if (source == nullptr || count == 0 ||
            Vertices.size() > static_cast<std::size_t>(UINT_MAX - count))
        {
            return;
        }
        const UINT first = static_cast<UINT>(Vertices.size());
        Vertices.insert(Vertices.end(), source, source + count);
        if (!Batches.empty())
        {
            Batch& previous = Batches.back();
            if (previous.SourceTexture == texture &&
                previous.FirstVertex + previous.VertexCount == first &&
                std::memcmp(&previous.Scissor, &CurrentScissor, sizeof(CurrentScissor)) == 0)
            {
                previous.VertexCount += count;
                return;
            }
        }
        Batches.push_back({first, count, texture, CurrentScissor});
    }

    void AddRect(float x, float y, float width, float height, std::uint32_t color)
    {
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const Vertex vertices[6] = {
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, top}, {0.0f, 0.0f}, color},
            {{right, bottom}, {0.0f, 0.0f}, color},
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, bottom}, {0.0f, 0.0f}, color},
            {{left, bottom}, {0.0f, 0.0f}, color}};
        Append(vertices, ARRAYSIZE(vertices), Texture::White);
    }

    void AddRoundedRect(float x, float y, float width, float height, float radius, std::uint32_t color)
    {
        radius = std::max(0.0f, std::min(radius, std::min(width, height) * 0.5f));
        if (radius <= 0.5f)
        {
            AddRect(x, y, width, height, color);
            return;
        }
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const float centers[4][2] = {
            {right - radius, top + radius},
            {right - radius, bottom - radius},
            {left + radius, bottom - radius},
            {left + radius, top + radius}};
        constexpr float startAngles[4] = {-Pi * 0.5f, 0.0f, Pi * 0.5f, Pi};
        constexpr int segments = 6;
        std::vector<Vertex> points;
        points.reserve(28);
        for (int corner = 0; corner < 4; ++corner)
        {
            for (int segment = 0; segment <= segments; ++segment)
            {
                const float angle = startAngles[corner] +
                    Pi * 0.5f * (static_cast<float>(segment) / static_cast<float>(segments));
                Vertex vertex = {};
                vertex.Position[0] = centers[corner][0] + std::cos(angle) * radius;
                vertex.Position[1] = centers[corner][1] + std::sin(angle) * radius;
                vertex.Color = color;
                points.push_back(vertex);
            }
        }
        std::vector<Vertex> fan;
        fan.reserve(points.size() * 3u);
        Vertex center = {};
        center.Position[0] = (left + right) * 0.5f;
        center.Position[1] = (top + bottom) * 0.5f;
        center.Color = color;
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            fan.push_back(center);
            fan.push_back(points[index]);
            fan.push_back(points[(index + 1u) % points.size()]);
        }
        Append(fan.data(), static_cast<UINT>(fan.size()), Texture::White);
    }

    void AddLine(float x0, float y0, float x1, float y1, float radius, std::uint32_t color)
    {
        y0 = ToScreenY(y0);
        y1 = ToScreenY(y1);
        float deltaX = x1 - x0;
        float deltaY = y1 - y0;
        const float length = std::sqrt(deltaX * deltaX + deltaY * deltaY);
        if (length <= 0.0001f)
        {
            return;
        }
        deltaX /= length;
        deltaY /= length;
        const float halfWidth = std::max(1.0f, radius) * 0.5f;
        const float normalX = -deltaY * halfWidth;
        const float normalY = deltaX * halfWidth;
        const Vertex vertices[6] = {
            {{x0 - normalX, y0 - normalY}, {0.0f, 0.0f}, color},
            {{x0 + normalX, y0 + normalY}, {0.0f, 0.0f}, color},
            {{x1 + normalX, y1 + normalY}, {0.0f, 0.0f}, color},
            {{x0 - normalX, y0 - normalY}, {0.0f, 0.0f}, color},
            {{x1 + normalX, y1 + normalY}, {0.0f, 0.0f}, color},
            {{x1 - normalX, y1 - normalY}, {0.0f, 0.0f}, color}};
        Append(vertices, ARRAYSIZE(vertices), Texture::White);
    }

    void AddTriangle(float x, float y, float width, float height, int flags, std::uint32_t color)
    {
        Vertex vertices[3] = {};
        if (flags == 1)
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y);
            vertices[1].Position[0] = x + width;
            vertices[1].Position[1] = ToScreenY(y + height * 0.5f);
            vertices[2].Position[0] = x;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        else
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y + height);
            vertices[1].Position[0] = x + width * 0.5f;
            vertices[1].Position[1] = ToScreenY(y);
            vertices[2].Position[0] = x + width;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        for (Vertex& vertex : vertices)
        {
            vertex.Color = color;
        }
        Append(vertices, ARRAYSIZE(vertices), Texture::White);
    }

    float MeasureText(const char* text) const noexcept
    {
        float x = 0.0f;
        float length = 0.0f;
        while (text != nullptr && *text != '\0')
        {
            const unsigned char character = static_cast<unsigned char>(*text);
            if (character == '\t')
            {
                for (const float stop : TabStops)
                {
                    if (x < stop)
                    {
                        x = stop;
                        break;
                    }
                }
                length = std::max(length, x);
            }
            else if (character >= 32 && character < 128)
            {
                const Glyph& glyph = Glyphs[character - 32u];
                length = std::max(length, x + glyph.Width);
                x += glyph.Advance;
            }
            ++text;
        }
        return length;
    }

    void AddText(float x, float y, int alignment, const char* text, std::uint32_t color)
    {
        if (text == nullptr)
        {
            return;
        }
        if (alignment == IMGUI_ALIGN_CENTER)
        {
            x -= MeasureText(text) * 0.5f;
        }
        else if (alignment == IMGUI_ALIGN_RIGHT)
        {
            x -= MeasureText(text);
        }
        const float originX = x;
        const float top = ToScreenY(y + static_cast<float>(FontHeight));
        std::vector<Vertex> vertices;
        vertices.reserve(std::strlen(text) * 6u);
        while (*text != '\0')
        {
            const unsigned char character = static_cast<unsigned char>(*text);
            if (character == '\t')
            {
                for (const float stop : TabStops)
                {
                    if (x < originX + stop)
                    {
                        x = originX + stop;
                        break;
                    }
                }
            }
            else if (character >= 32 && character < 128)
            {
                const Glyph& glyph = Glyphs[character - 32u];
                const float right = x + glyph.Width;
                const float bottom = top + glyph.Height;
                vertices.push_back({{x, top}, {glyph.U0, glyph.V0}, color});
                vertices.push_back({{right, top}, {glyph.U1, glyph.V0}, color});
                vertices.push_back({{right, bottom}, {glyph.U1, glyph.V1}, color});
                vertices.push_back({{x, top}, {glyph.U0, glyph.V0}, color});
                vertices.push_back({{right, bottom}, {glyph.U1, glyph.V1}, color});
                vertices.push_back({{x, bottom}, {glyph.U0, glyph.V1}, color});
                x += glyph.Advance;
            }
            ++text;
        }
        if (!vertices.empty())
        {
            Append(vertices.data(), static_cast<UINT>(vertices.size()), Texture::Font);
        }
    }

    void BuildRenderQueue()
    {
        Vertices.clear();
        Batches.clear();
        CurrentScissor = ClampScissor(0, 0, Width, Height);
        const imguiGfxCmd* commands = imguiGetRenderQueue();
        const int count = imguiGetRenderQueueSize();
        for (int index = 0; index < count; ++index)
        {
            const imguiGfxCmd& command = commands[index];
            switch (command.type)
            {
            case IMGUI_GFXCMD_RECT:
            {
                const float x = static_cast<float>(command.rect.x) * QueueScale + 0.5f;
                const float y = static_cast<float>(command.rect.y) * QueueScale + 0.5f;
                const float width = static_cast<float>(command.rect.w) * QueueScale - 1.0f;
                const float height = static_cast<float>(command.rect.h) * QueueScale - 1.0f;
                if (command.rect.r == 0)
                {
                    AddRect(x, y, width, height, command.col);
                }
                else
                {
                    AddRoundedRect(
                        x, y, width, height,
                        static_cast<float>(command.rect.r) * QueueScale,
                        command.col);
                }
                break;
            }
            case IMGUI_GFXCMD_LINE:
                AddLine(
                    static_cast<float>(command.line.x0) * QueueScale,
                    static_cast<float>(command.line.y0) * QueueScale,
                    static_cast<float>(command.line.x1) * QueueScale,
                    static_cast<float>(command.line.y1) * QueueScale,
                    static_cast<float>(command.line.r) * QueueScale,
                    command.col);
                break;
            case IMGUI_GFXCMD_TRIANGLE:
                AddTriangle(
                    static_cast<float>(command.rect.x) * QueueScale + 0.5f,
                    static_cast<float>(command.rect.y) * QueueScale + 0.5f,
                    static_cast<float>(command.rect.w) * QueueScale - 1.0f,
                    static_cast<float>(command.rect.h) * QueueScale - 1.0f,
                    command.flags,
                    command.col);
                break;
            case IMGUI_GFXCMD_TEXT:
                AddText(
                    static_cast<float>(command.text.x),
                    static_cast<float>(command.text.y),
                    command.text.align,
                    command.text.text,
                    command.col);
                break;
            case IMGUI_GFXCMD_SCISSOR:
                CurrentScissor = command.flags != 0
                    ? ClampScissor(command.rect.x, command.rect.y, command.rect.w, command.rect.h)
                    : ClampScissor(0, 0, Width, Height);
                break;
            default:
                break;
            }
        }
    }

    bool EnsureFrameBuffer(UINT frameIndex, UINT vertexCount)
    {
        if (frameIndex >= FrameBuffers.size() || vertexCount == 0)
        {
            return vertexCount == 0;
        }
        FrameBuffer& frame = FrameBuffers[frameIndex];
        if (frame.Resource != nullptr && frame.Capacity >= vertexCount)
        {
            return true;
        }
        if (frame.Resource != nullptr && frame.Mapped != nullptr)
        {
            frame.Resource->Unmap(0, nullptr);
        }
        frame.Mapped = nullptr;
        SafeRelease(frame.Resource);
        frame.Capacity = std::max(512u, vertexCount + 256u);
        const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_RESOURCE_DESC description = BufferDescription(
            static_cast<UINT64>(frame.Capacity) * sizeof(Vertex));
        HRESULT result = Device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&frame.Resource));
        if (FAILED(result))
        {
            frame.Capacity = 0;
            return false;
        }
        D3D12_RANGE readRange = {0, 0};
        result = frame.Resource->Map(0, &readRange, reinterpret_cast<void**>(&frame.Mapped));
        return SUCCEEDED(result);
    }

    void Draw(ID3D12GraphicsCommandList* commands, int width, int height, UINT frameIndex)
    {
        if (commands == nullptr || PipelineState == nullptr || RootSignature == nullptr ||
            width <= 0 || height <= 0)
        {
            return;
        }
        Width = width;
        Height = height;
        BuildRenderQueue();
        if (Vertices.empty() || Vertices.size() > UINT_MAX ||
            !EnsureFrameBuffer(frameIndex, static_cast<UINT>(Vertices.size())))
        {
            return;
        }
        FrameBuffer& frame = FrameBuffers[frameIndex];
        std::memcpy(frame.Mapped, Vertices.data(), Vertices.size() * sizeof(Vertex));

        const D3D12_VIEWPORT viewport = {
            0.0f, 0.0f, static_cast<float>(Width), static_cast<float>(Height), 0.0f, 1.0f};
        commands->RSSetViewports(1, &viewport);
        commands->SetPipelineState(PipelineState);
        commands->SetGraphicsRootSignature(RootSignature);
        ID3D12DescriptorHeap* heaps[] = {TextureHeap};
        commands->SetDescriptorHeaps(ARRAYSIZE(heaps), heaps);
        const float screenSize[2] = {static_cast<float>(Width), static_cast<float>(Height)};
        commands->SetGraphicsRoot32BitConstants(0, ARRAYSIZE(screenSize), screenSize, 0);
        commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vertexBuffer = {
            frame.Resource->GetGPUVirtualAddress(),
            static_cast<UINT>(Vertices.size() * sizeof(Vertex)),
            sizeof(Vertex)};
        commands->IASetVertexBuffers(0, 1, &vertexBuffer);

        const D3D12_GPU_DESCRIPTOR_HANDLE heapStart = TextureHeap->GetGPUDescriptorHandleForHeapStart();
        for (const Batch& batch : Batches)
        {
            commands->RSSetScissorRects(1, &batch.Scissor);
            D3D12_GPU_DESCRIPTOR_HANDLE texture = heapStart;
            if (batch.SourceTexture == Texture::Font)
            {
                texture.ptr += DescriptorSize;
            }
            commands->SetGraphicsRootDescriptorTable(1, texture);
            commands->DrawInstanced(batch.VertexCount, 1, batch.FirstVertex, 0);
        }
    }

    ID3D12Device* Device = nullptr;
    ID3D12RootSignature* RootSignature = nullptr;
    ID3D12PipelineState* PipelineState = nullptr;
    ID3D12DescriptorHeap* TextureHeap = nullptr;
    ID3D12Resource* WhiteTexture = nullptr;
    ID3D12Resource* FontTexture = nullptr;
    ID3D12Resource* WhiteUpload = nullptr;
    ID3D12Resource* FontUpload = nullptr;
    UINT DescriptorSize = 0;
    std::vector<FrameBuffer> FrameBuffers;
    Glyph Glyphs[96] = {};
    int FontHeight = 16;
    int Width = 1;
    int Height = 1;
    D3D12_RECT CurrentScissor = {};
    std::vector<Vertex> Vertices;
    std::vector<Batch> Batches;
};

ImguiRenderDX12::ImguiRenderDX12()
    : Impl_(std::make_unique<Impl>())
{
}

ImguiRenderDX12::~ImguiRenderDX12() = default;

bool ImguiRenderDX12::Init(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* initializationCommands,
    DXGI_FORMAT renderTargetFormat,
    unsigned int frameCount)
{
    return Impl_->Init(device, initializationCommands, renderTargetFormat, frameCount);
}

void ImguiRenderDX12::Shutdown() noexcept
{
    Impl_->Shutdown();
}

void ImguiRenderDX12::Draw(
    ID3D12GraphicsCommandList* commandList,
    int width,
    int height,
    unsigned int frameIndex)
{
    Impl_->Draw(commandList, width, height, frameIndex);
}

} // namespace Evans::Editor

#endif
