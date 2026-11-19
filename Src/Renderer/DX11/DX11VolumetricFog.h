#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Volumetrics/VolumetricFogContract.h"
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
    class DX11VolumetricFog
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
        static constexpr UINT ConstantSlot = 3, AtlasSlot = 14, SamplerSlot = 3;
        static constexpr int MaximumTextureSize = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    public:
        bool Resolved = false;
        void Begin() noexcept { Resolved = false; ResolvedAtlas.Reset(); }
        bool BindSurface(ID3D11Device* device, ID3D11DeviceContext* context, bool enabled,
            ID3D11SamplerState* fallbackSampler, std::string& error)
        {
            // Disabled consumers still declare these bindings. Supply explicit
            // zero controls and a valid sampler instead of relying on null state.
            if (!DisabledConstants)
            {
                const std::array<std::uint8_t, sizeof(VolumetricFogConstants)> zero {};
                D3D11_BUFFER_DESC desc {}; desc.ByteWidth = UINT(zero.size());
                desc.Usage = D3D11_USAGE_IMMUTABLE; desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
                D3D11_SUBRESOURCE_DATA initial {}; initial.pSysMem = zero.data();
                if (FAILED(device->CreateBuffer(&desc, &initial, DisabledConstants.GetAddressOf())))
                { error = "DX11 disabled fog constants allocation failed"; return false; }
            }
            ID3D11Buffer* buffer = enabled && Resolved ? Constants.Get() : DisabledConstants.Get();
            ID3D11ShaderResourceView* view = enabled && Resolved ? ResolvedAtlas.Get() : nullptr;
            ID3D11SamplerState* sampler = Sampler ? Sampler.Get() : fallbackSampler;
            context->PSSetConstantBuffers(ConstantSlot, 1, &buffer);
            context->PSSetShaderResources(AtlasSlot, 1, &view);
            context->PSSetSamplers(SamplerSlot, 1, &sampler);
            return true;
        }
        bool Prepare(ID3D11Device* device, int width, int height,
            const Techniques::VolumetricFogSettings& settings, std::string& error)
        {
            auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 volumetric fog ") + operation + " failed: " + std::to_string(result); return true;
            };
            const auto grid = BuildFogGrid(width,height,settings,MaximumTextureSize);
            if (grid.Width == 0) { error = "Invalid fog atlas dimensions"; return false; }
            auto surface = [&](Surface& output, int w, int h)
            {
                if (output.Texture)
                {
                    D3D11_TEXTURE2D_DESC existing {}; output.Texture->GetDesc(&existing);
                    if (existing.Width == static_cast<UINT>(w) && existing.Height == static_cast<UINT>(h)) return true;
                }
                Surface candidate;
                D3D11_TEXTURE2D_DESC desc {};
                desc.Width = w; desc.Height = h; desc.MipLevels = desc.ArraySize = 1;
                desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count = 1;
                desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
                if (failed(device->CreateTexture2D(&desc, nullptr, candidate.Texture.GetAddressOf()), "texture") ||
                    failed(device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf()), "target") ||
                    failed(device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf()), "view")) return false;
                output = std::move(candidate); return true;
            };
            if (!surface(Atlas, grid.AtlasWidth(), grid.AtlasHeight()) || !surface(Scratch, width, height)) return false;
            Width = width; Height = height; Grid = grid;
            const bool media = Techniques::UsesBoundedMedium(settings);
            const bool cloud = Techniques::UsesVolumetricClouds(settings);
            ProgramIndex = static_cast<std::size_t>(settings.MediumMode);
            auto& Vertex = Vertices[ProgramIndex]; auto& Pixel = Pixels[ProgramIndex];
            if (Vertex && Pixel && Constants && Sampler && Raster && DepthState) return true;
            std::string source;
            const char* file = cloud ? "VolumetricClouds.hlsl" : media ? "ParticipatingMedia.hlsl" : "VolumetricFog.hlsl";
            const D3D_SHADER_MACRO definitions[] {{cloud ? "EVANS_CLOUD_COMPOSITE" : "EVANS_MEDIA_COMPOSITE", "1"}, {nullptr, nullptr}};
            if (!LoadTechniqueShaderSource("DX11", file, source, error)) return false;
            ComPtr<ID3DBlob> vertex, pixel, diagnostics;
            for (int stage = 0; stage < 2; ++stage)
            {
                auto& binary = stage ? pixel : vertex;
                const HRESULT result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), file, media ? definitions : nullptr, nullptr,
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
            buffer.ByteWidth = sizeof(VolumetricFogConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
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
                !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster") &&
                !failed(device->CreateDepthStencilState(&depth, DepthState.ReleaseAndGetAddressOf()), "depth state");
        }
        bool Resolve(ID3D11DeviceContext* context, const VolumetricFogDesc& desc,
            const Techniques::VolumetricFogSettings& settings, ID3D11Texture2D* scene,
            ID3D11ShaderResourceView* color, ID3D11ShaderResourceView* depth, std::string& error,
            ID3D11ShaderResourceView* integratedAtlas = nullptr)
        {
            VolumetricFogConstants constants;
            if (!BuildVolumetricFogConstants(desc, settings, Width, Height, MaximumTextureSize, constants))
            { error = "Fog requires a finite invertible camera frustum and valid dimensions"; return false; }
            ID3D11ShaderResourceView* empty[AtlasSlot + 1] {};
            context->OMSetRenderTargets(0, nullptr, nullptr);
            context->PSSetShaderResources(0, AtlasSlot + 1, empty);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthState.Get(), 0);
            context->RSSetState(Raster.Get());
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertices[ProgramIndex].Get(), nullptr, 0); context->PSSetShader(Pixels[ProgramIndex].Get(), nullptr, 0);
            ID3D11Buffer* buffer = Constants.Get(); context->PSSetConstantBuffers(ConstantSlot, 1, &buffer);
            ID3D11SamplerState* sampler = Sampler.Get(); context->PSSetSamplers(SamplerSlot, 1, &sampler);
            for (UINT pass = integratedAtlas ? 1U : 0U; pass < 2; ++pass)
            {
                constants.Options.X = pass;
                D3D11_MAPPED_SUBRESOURCE mapped {};
                if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                { error = "DX11 fog constants mapping failed"; return false; }
                std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
                ID3D11RenderTargetView* target = pass ? Scratch.Target.Get() : Atlas.Target.Get();
                context->OMSetRenderTargets(1, &target, nullptr);
                D3D11_VIEWPORT viewport {0, 0, static_cast<float>(pass ? Width : Grid.AtlasWidth()),
                    static_cast<float>(pass ? Height : Grid.AtlasHeight()), 0, 1};
                context->RSSetViewports(1, &viewport);
                if (pass)
                {
                    ID3D11ShaderResourceView* inputs[] {color, depth};
                    context->PSSetShaderResources(0, 2, inputs);
                    ID3D11ShaderResourceView* atlas = integratedAtlas ? integratedAtlas : Atlas.View.Get();
                    context->PSSetShaderResources(AtlasSlot, 1, &atlas);
                }
                context->Draw(3, 0);
                context->OMSetRenderTargets(0, nullptr, nullptr);
            }
            context->PSSetShaderResources(0, AtlasSlot + 1, empty);
            context->CopyResource(scene, Scratch.Texture.Get());
            ResolvedAtlas = integratedAtlas ? integratedAtlas : Atlas.View.Get();
            Resolved = true;
            return true;
        }
    private:
        Surface Atlas, Scratch;
        ComPtr<ID3D11ShaderResourceView> ResolvedAtlas;
        std::array<ComPtr<ID3D11VertexShader>, 3> Vertices;
        std::array<ComPtr<ID3D11PixelShader>, 3> Pixels;
        ComPtr<ID3D11Buffer> Constants, DisabledConstants;
        ComPtr<ID3D11SamplerState> Sampler;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
        FogGrid Grid;
        std::size_t ProgramIndex = 0;
        int Width = 0, Height = 0;
    };
}
#endif
