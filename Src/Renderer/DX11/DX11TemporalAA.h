#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/PostProcessing/TemporalAAContract.h"
#include "Renderer/TechniqueShader.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstring>
#include <climits>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    // Ping-pong history is private: later blur, transparent draws and HDR effects
    // modify a copy in scene color. Only successful presentation publishes writes;
    // an aborted resolve may overwrite the pending slot, never the committed one.
    class DX11TemporalAA
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
    public:
        bool Resolved = false;
        void Begin() noexcept { Resolved = false; }
        void Invalidate() noexcept { HistoryValid = false; Resolved = false; }
        void Commit() noexcept
        {
            if (Resolved) { ReadIndex = 1 - ReadIndex; HistoryValid = true; }
        }
        bool Prepare(ID3D11Device* device, int width, int height, std::string& error)
        {
            if (width != Width || height != Height)
            {
                Colors = {}; Depths = {}; Width = width; Height = height; Invalidate();
            }
            auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 TAA ") + operation + " failed: " + std::to_string(result); return true;
            };
            for (int index = 0; index < 2; ++index)
                for (int kind = 0; kind < 2; ++kind)
                {
                    auto& surface = kind ? Depths[index] : Colors[index];
                    if (surface.Texture) continue;
                    Surface candidate;
                    D3D11_TEXTURE2D_DESC desc {};
                    desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = 1;
                    desc.Format = kind ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
                    desc.SampleDesc.Count = 1; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
                    if (failed(device->CreateTexture2D(&desc, nullptr, candidate.Texture.GetAddressOf()), "history texture") ||
                        failed(device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf()), "history target") ||
                        failed(device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf()), "history view")) return false;
                    surface = std::move(candidate);
                }
            if (Vertex && Pixel && Constants && Sampler && Raster && DepthState) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", "TAA.hlsl", source, error)) return false;
            ComPtr<ID3DBlob> vertex, pixel, diagnostics;
            for (int stage = 0; stage < 2; ++stage)
            {
                auto& binary = stage ? pixel : vertex;
                const HRESULT result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "TAA.hlsl", nullptr, nullptr,
                    stage ? "PSMain" : "VSMain", stage ? "ps_5_0" : "vs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                    binary.GetAddressOf(), diagnostics.ReleaseAndGetAddressOf());
                if (failed(result, "shader compilation"))
                {
                    if (diagnostics) error += std::string(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize());
                    return false;
                }
            }
            if (failed(device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, Vertex.ReleaseAndGetAddressOf()), "vertex shader") ||
                failed(device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, Pixel.ReleaseAndGetAddressOf()), "pixel shader")) return false;
            D3D11_BUFFER_DESC buffer {};
            buffer.ByteWidth = sizeof(TemporalAAConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer, nullptr, Constants.ReleaseAndGetAddressOf()), "constants")) return false;
            D3D11_SAMPLER_DESC sampler {};
            sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sampler.MaxLOD = D3D11_FLOAT32_MAX;
            D3D11_RASTERIZER_DESC raster {};
            raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE; raster.DepthClipEnable = TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {};
            depth.DepthEnable = FALSE;
            return !failed(device->CreateSamplerState(&sampler, Sampler.ReleaseAndGetAddressOf()), "sampler") &&
                !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster state") &&
                !failed(device->CreateDepthStencilState(&depth, DepthState.ReleaseAndGetAddressOf()), "depth state");
        }
        bool Resolve(ID3D11DeviceContext* context, const TemporalAADesc& desc, const Techniques::TemporalAASettings& settings,
            ID3D11Texture2D* scene, ID3D11ShaderResourceView* color, ID3D11ShaderResourceView* depth,
            ID3D11ShaderResourceView* velocity, std::string& error)
        {
            TemporalAAConstants constants;
            if (!BuildTemporalAAConstants(desc, settings, Width, Height, HistoryValid, constants))
            { error = "TAA requires finite invertible previous projection and valid dimensions"; return false; }
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            { error = "DX11 TAA constants mapping failed"; return false; }
            std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
            ID3D11ShaderResourceView* empty[5] {};
            context->OMSetRenderTargets(0, nullptr, nullptr);
            context->PSSetShaderResources(0, 5, empty);
            const int write = 1 - ReadIndex;
            ID3D11RenderTargetView* outputs[] {Colors[write].Target.Get(), Depths[write].Target.Get()};
            context->OMSetRenderTargets(2, outputs, nullptr);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthState.Get(), 0);
            context->RSSetState(Raster.Get());
            D3D11_VIEWPORT viewport {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
            context->RSSetViewports(1, &viewport);
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertex.Get(), nullptr, 0); context->PSSetShader(Pixel.Get(), nullptr, 0);
            ID3D11Buffer* buffer = Constants.Get(); context->PSSetConstantBuffers(0, 1, &buffer);
            ID3D11SamplerState* sampler = Sampler.Get(); context->PSSetSamplers(0, 1, &sampler);
            ID3D11ShaderResourceView* inputs[] {color, depth, velocity,
                HistoryValid ? Colors[ReadIndex].View.Get() : nullptr, HistoryValid ? Depths[ReadIndex].View.Get() : nullptr};
            context->PSSetShaderResources(0, 5, inputs);
            context->Draw(3, 0);
            context->PSSetShaderResources(0, 5, empty); context->OMSetRenderTargets(0, nullptr, nullptr);
            context->CopyResource(scene, Colors[write].Texture.Get());
            Resolved = true;
            return true;
        }
    private:
        std::array<Surface, 2> Colors, Depths;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> Pixel;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11SamplerState> Sampler;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
        int Width = 0, Height = 0, ReadIndex = 0;
        bool HistoryValid = false;
    };
}
#endif
