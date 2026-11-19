#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Volumetrics/LightShaftContract.h"
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
    class DX11LightShafts
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

        bool Prepare(ID3D11Device* device, int width, int height, std::string& error)
        {
            auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 light shafts ") + operation + " failed: " + std::to_string(result); return true;
            };
            if (Width != width || Height != height || !Mask.Texture || !Scratch.Texture)
            {
                std::array<Surface, 2> candidates;
                D3D11_TEXTURE2D_DESC description {};
                description.Width = width; description.Height = height;
                description.MipLevels = description.ArraySize = description.SampleDesc.Count = 1;
                description.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                for (auto& surface : candidates)
                    if (failed(device->CreateTexture2D(&description, nullptr, surface.Texture.GetAddressOf()), "texture") ||
                        failed(device->CreateRenderTargetView(surface.Texture.Get(), nullptr, surface.Target.GetAddressOf()), "target") ||
                        failed(device->CreateShaderResourceView(surface.Texture.Get(), nullptr, surface.View.GetAddressOf()), "view")) return false;
                Mask = std::move(candidates[0]); Scratch = std::move(candidates[1]); Width = width; Height = height;
            }
            if (Vertex && Pixel && Constants && Raster && DepthState) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", "LightShafts.hlsl", source, error)) return false;
            ComPtr<ID3DBlob> vertex, pixel, diagnostics;
            for (int stage = 0; stage < 2; ++stage)
            {
                auto& binary = stage ? pixel : vertex;
                const HRESULT result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "LightShafts.hlsl", nullptr, nullptr,
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
            buffer.ByteWidth = sizeof(LightShaftConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            D3D11_RASTERIZER_DESC raster {};
            raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE; raster.DepthClipEnable = TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {};
            depth.DepthEnable = FALSE;
            return !failed(device->CreateBuffer(&buffer, nullptr, Constants.ReleaseAndGetAddressOf()), "constants") &&
                !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster state") &&
                !failed(device->CreateDepthStencilState(&depth, DepthState.ReleaseAndGetAddressOf()), "depth state");
        }

        bool Resolve(ID3D11DeviceContext* context, const LightShaftDesc& desc, const Techniques::LightShaftSettings& settings,
            ID3D11Texture2D* scene, ID3D11ShaderResourceView* color, ID3D11ShaderResourceView* depth, std::string& error)
        {
            LightShaftConstants constants;
            if (!BuildLightShaftConstants(desc, settings, Width, Height, constants))
            { error = "Light shafts require finite camera/light inputs and valid dimensions"; return false; }
            constexpr UINT InputCount = 3, FullscreenVertexCount = 3;
            ID3D11ShaderResourceView* empty[InputCount] {};
            context->OMSetRenderTargets(0, nullptr, nullptr);
            context->PSSetShaderResources(0, InputCount, empty);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthState.Get(), 0);
            context->RSSetState(Raster.Get());
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertex.Get(), nullptr, 0); context->PSSetShader(Pixel.Get(), nullptr, 0);
            ID3D11Buffer* buffer = Constants.Get(); context->PSSetConstantBuffers(0, 1, &buffer);
            const D3D11_VIEWPORT viewport {0, 0, float(Width), float(Height), 0, 1};
            context->RSSetViewports(1, &viewport);
            for (UINT pass = 0; pass < 2; ++pass)
            {
                constants.Control.Y = pass;
                D3D11_MAPPED_SUBRESOURCE mapped {};
                if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                { error = "DX11 light-shaft constants mapping failed"; return false; }
                std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
                ID3D11RenderTargetView* target = pass ? Scratch.Target.Get() : Mask.Target.Get();
                context->OMSetRenderTargets(1, &target, nullptr);
                ID3D11ShaderResourceView* inputs[InputCount] {color, depth, pass ? Mask.View.Get() : nullptr};
                context->PSSetShaderResources(0, InputCount, inputs);
                context->Draw(FullscreenVertexCount, 0);
                context->OMSetRenderTargets(0, nullptr, nullptr);
                context->PSSetShaderResources(0, InputCount, empty);
            }
            context->CopyResource(scene, Scratch.Texture.Get());
            Resolved = true;
            return true;
        }
    private:
        Surface Mask, Scratch;
        int Width = 0, Height = 0;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> Pixel;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
    };
}
#endif
