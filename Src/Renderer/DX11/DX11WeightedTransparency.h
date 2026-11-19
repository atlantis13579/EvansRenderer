#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Transparency/WeightedTransparencyContract.h"
#include "Renderer/TechniqueShader.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <array>
#include <cstring>
#include <climits>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    class DX11WeightedTransparency
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
        static constexpr UINT ConstantSlot = 4;
        static constexpr std::array<DXGI_FORMAT, 2> Formats {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_FLOAT};
    public:
        bool Prepared = false, Accumulated = false, Resolved = false;
        void Begin() noexcept { Prepared = Accumulated = Resolved = false; }
        static std::uint32_t SupportedCounts(ID3D11Device* device)
        {
            if (!device) return 0;
            std::uint32_t result = Techniques::AllMultisampleCounts;
            for (auto format : Formats)
            {
                UINT support = 0;
                constexpr UINT required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_RENDER_TARGET |
                    D3D11_FORMAT_SUPPORT_BLENDABLE | D3D11_FORMAT_SUPPORT_SHADER_LOAD;
                if (FAILED(device->CheckFormatSupport(format, &support)) || (support & required) != required) return 0;
                for (UINT count : {2u, 4u, 8u})
                {
                    UINT levels = 0;
                    constexpr UINT sampleFeatures = D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET | D3D11_FORMAT_SUPPORT_MULTISAMPLE_LOAD;
                    if ((support & sampleFeatures) != sampleFeatures ||
                        FAILED(device->CheckMultisampleQualityLevels(format, count, &levels)) || levels == 0) result &= ~count;
                }
            }
            return result;
        }
        ID3D11BlendState* AccumulationBlend() const { return Blend.Get(); }
        void BindTargets(ID3D11DeviceContext* context, ID3D11DepthStencilView* depth)
        {
            ID3D11RenderTargetView* targets[] {Surfaces[0].Target.Get(), Surfaces[1].Target.Get()};
            context->OMSetRenderTargets(2, targets, depth);
        }
        void BindSurface(ID3D11DeviceContext* context, bool enabled)
        {
            ID3D11Buffer* constants = enabled ? Constants.Get() : nullptr;
            context->PSSetConstantBuffers(ConstantSlot, 1, &constants);
        }
        bool Ensure(ID3D11Device* device, int width, int height, UINT samples, std::string& error)
        {
            const auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 weighted transparency ") + operation + " failed (" + std::to_string(result) + ")";
                return true;
            };
            if (!Surfaces[0].Texture || Width != width || Height != height || Samples != samples)
            {
                std::array<Surface, 2> candidates;
                for (std::size_t index = 0; index < candidates.size(); ++index)
                {
                    D3D11_TEXTURE2D_DESC texture {};
                    texture.Width = width; texture.Height = height; texture.MipLevels = texture.ArraySize = 1;
                    texture.Format = Formats[index]; texture.SampleDesc.Count = samples;
                    texture.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                    auto& candidate = candidates[index];
                    if (failed(device->CreateTexture2D(&texture, nullptr, candidate.Texture.GetAddressOf()), "target allocation") ||
                        failed(device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf()), "RTV") ||
                        failed(device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf()), "SRV")) return false;
                }
                Surfaces = std::move(candidates); Width = width; Height = height; Samples = samples;
            }
            if (Vertex && Pixel && SamplePixel && Constants && Blend && CompositeBlend && Raster && DepthDisabled) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", "WeightedBlendedOIT.hlsl", source, error)) return false;
            const char* entries[] {"VSMain", "PSMain", "PSResolveSamples"};
            for (int stage = 0; stage < 3; ++stage)
            {
                ComPtr<ID3DBlob> binary, messages;
                if (failed(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "WeightedBlendedOIT.hlsl", nullptr, nullptr, entries[stage],
                    stage == 0 ? "vs_5_0" : "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                    binary.GetAddressOf(), messages.GetAddressOf()), "shader compilation"))
                {
                    if (messages) error += std::string(static_cast<const char*>(messages->GetBufferPointer()), messages->GetBufferSize());
                    return false;
                }
                if (stage == 0)
                {
                    if (failed(device->CreateVertexShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr,
                        Vertex.ReleaseAndGetAddressOf()), "vertex shader")) return false;
                }
                else if (failed(device->CreatePixelShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr,
                    (stage == 1 ? Pixel : SamplePixel).ReleaseAndGetAddressOf()), "pixel shader")) return false;
            }
            D3D11_BUFFER_DESC buffer {};
            buffer.ByteWidth = sizeof(WeightedTransparencyConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer, nullptr, Constants.ReleaseAndGetAddressOf()), "constants")) return false;
            D3D11_BLEND_DESC blend {};
            blend.IndependentBlendEnable = TRUE;
            for (auto& target : blend.RenderTarget)
            {
                target.BlendEnable = TRUE;
                target.SrcBlend = target.DestBlend = target.SrcBlendAlpha = target.DestBlendAlpha = D3D11_BLEND_ONE;
                target.BlendOp = target.BlendOpAlpha = D3D11_BLEND_OP_ADD;
                target.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            }
            auto& revealage = blend.RenderTarget[1];
            revealage.SrcBlend = revealage.SrcBlendAlpha = D3D11_BLEND_ZERO;
            revealage.DestBlend = D3D11_BLEND_INV_SRC_COLOR; revealage.DestBlendAlpha = D3D11_BLEND_ONE;
            revealage.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED;
            if (failed(device->CreateBlendState(&blend, Blend.ReleaseAndGetAddressOf()), "accumulation blend")) return false;
            blend.IndependentBlendEnable = FALSE;
            auto& composite = blend.RenderTarget[0];
            composite.SrcBlend = D3D11_BLEND_SRC_ALPHA; composite.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            composite.SrcBlendAlpha = D3D11_BLEND_ONE; composite.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
            if (failed(device->CreateBlendState(&blend, CompositeBlend.ReleaseAndGetAddressOf()), "composition blend")) return false;
            D3D11_RASTERIZER_DESC raster {};
            raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE;
            raster.DepthClipEnable = TRUE; raster.MultisampleEnable = TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {}; depth.DepthEnable = FALSE;
            return !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster state") &&
                !failed(device->CreateDepthStencilState(&depth, DepthDisabled.ReleaseAndGetAddressOf()), "depth state");
        }
        bool Prepare(ID3D11DeviceContext* context, const Techniques::TransparencySettings& settings, std::string& error)
        {
            const auto constants = BuildWeightedTransparencyConstants(settings);
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            { error = "DX11 weighted transparency constant upload failed"; return false; }
            std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
            // Clear identity statistics even when the scene has no transparent meshes.
            ID3D11ShaderResourceView* empty[4] {};
            context->PSSetShaderResources(0, 4, empty); context->OMSetRenderTargets(0, nullptr, nullptr);
            const float zero[4] {}, one[4] {1, 1, 1, 1};
            context->ClearRenderTargetView(Surfaces[0].Target.Get(), zero);
            context->ClearRenderTargetView(Surfaces[1].Target.Get(), one);
            Prepared = true; return true;
        }
        void Resolve(ID3D11DeviceContext* context, ID3D11RenderTargetView* scene)
        {
            context->OMSetRenderTargets(1, &scene, nullptr);
            context->OMSetBlendState(CompositeBlend.Get(), nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthDisabled.Get(), 0);
            context->RSSetState(Raster.Get());
            const D3D11_VIEWPORT viewport {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
            context->RSSetViewports(1, &viewport);
            context->IASetInputLayout(nullptr); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertex.Get(), nullptr, 0);
            context->PSSetShader(Samples > 1 ? SamplePixel.Get() : Pixel.Get(), nullptr, 0);
            ID3D11ShaderResourceView* inputs[] {Surfaces[0].View.Get(), Surfaces[1].View.Get()};
            context->PSSetShaderResources(Samples > 1 ? 2 : 0, 2, inputs);
            context->Draw(3, 0);
            ID3D11ShaderResourceView* empty[4] {};
            context->PSSetShaderResources(0, 4, empty); context->OMSetRenderTargets(0, nullptr, nullptr);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            Resolved = true;
        }
    private:
        int Width = 0, Height = 0;
        UINT Samples = 1;
        std::array<Surface, 2> Surfaces;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> Pixel, SamplePixel;
        ComPtr<ID3D11BlendState> Blend, CompositeBlend;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthDisabled;
    };
}
#endif
