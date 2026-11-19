#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Volumetrics/VolumetricLightingContract.h"
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
    class DX11VolumetricLighting
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
        static constexpr UINT MediumSlot = 3, LightingSlot = 4;
        static constexpr UINT CompleteLightSlot = 16;
        static constexpr int MaximumTextureSize = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    public:
        ID3D11ShaderResourceView* View() const { return Atlas.View.Get(); }
        bool Prepare(ID3D11Device* device, int width, int height,
            const Techniques::VolumetricFogSettings& medium, std::string& error)
        {
            const auto grid = BuildFogGrid(width,height,medium,MaximumTextureSize);
            if (grid.Width == 0) { error = "Invalid volume-light atlas dimensions"; return false; }
            auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 volumetric lighting ") + operation + " failed: " + std::to_string(result); return true;
            };
            if (!Atlas.Texture || grid.AtlasWidth() != Grid.AtlasWidth() || grid.AtlasHeight() != Grid.AtlasHeight())
            {
                std::array<Surface, 3> candidate;
                for (std::size_t index = 0; index < candidate.size(); ++index)
                {
                    auto& surface = candidate[index];
                    D3D11_TEXTURE2D_DESC desc {};
                    desc.Width = grid.AtlasWidth(); desc.Height = grid.AtlasHeight();
                    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
                    // Keep prefix transport in FP32. Rounding each segment's T
                    // to half precision can erase extinction in a thin medium.
                    desc.Format = index < 2 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
                    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                    if (failed(device->CreateTexture2D(&desc, nullptr, surface.Texture.GetAddressOf()), "texture") ||
                        failed(device->CreateRenderTargetView(surface.Texture.Get(), nullptr, surface.Target.GetAddressOf()), "target") ||
                        failed(device->CreateShaderResourceView(surface.Texture.Get(), nullptr, surface.View.GetAddressOf()), "view")) return false;
                }
                Scratch[0] = std::move(candidate[0]); Scratch[1] = std::move(candidate[1]); Atlas = std::move(candidate[2]);
            }
            Grid = grid; Width = width; Height = height;
            const bool media = Techniques::UsesBoundedMedium(medium);
            const bool cloud = Techniques::UsesVolumetricClouds(medium);
            ProgramIndex = static_cast<std::size_t>(medium.MediumMode);
            auto& Vertex = Vertices[ProgramIndex]; auto& Pixels = PixelPrograms[ProgramIndex];
            if (Vertex && Pixels[0] && Pixels[1] && MediumBuffer && LightingBuffer && EmptySceneView && Raster && DepthState) return true;
            std::string source;
            const char* file = cloud ? "VolumetricClouds.hlsl" : media ? "ParticipatingMedia.hlsl" : "VolumetricLighting.hlsl";
            if (!LoadTechniqueShaderSource("DX11", file, source, error)) return false;
            const std::array<const char*, 3> entries {"VSMain", "PSInject", "PSPrefix"};
            for (std::size_t stage = 0; stage < entries.size(); ++stage)
            {
                ComPtr<ID3DBlob> binary, diagnostics;
                if (failed(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), file, nullptr, nullptr,
                    entries[stage], stage == 0 ? "vs_5_0" : "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                    binary.GetAddressOf(), diagnostics.GetAddressOf()), "shader compilation"))
                {
                    if (diagnostics) error += std::string(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize());
                    return false;
                }
                if (stage == 0)
                {
                    if (failed(device->CreateVertexShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr,
                        Vertex.ReleaseAndGetAddressOf()), "vertex shader")) return false;
                }
                else if (failed(device->CreatePixelShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr,
                    Pixels[stage - 1].ReleaseAndGetAddressOf()), "pixel shader")) return false;
            }
            D3D11_BUFFER_DESC buffer {};
            buffer.ByteWidth = sizeof(VolumetricFogConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer, nullptr, MediumBuffer.ReleaseAndGetAddressOf()), "medium constants")) return false;
            buffer.ByteWidth = sizeof(VolumetricLightingConstants);
            if (failed(device->CreateBuffer(&buffer, nullptr, LightingBuffer.ReleaseAndGetAddressOf()), "light constants")) return false;
            const ShaderLight emptyLight {};
            D3D11_BUFFER_DESC emptyDesc {};
            emptyDesc.ByteWidth = emptyDesc.StructureByteStride = sizeof(ShaderLight);
            emptyDesc.Usage = D3D11_USAGE_IMMUTABLE; emptyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            emptyDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            D3D11_SUBRESOURCE_DATA emptyData {&emptyLight, 0, 0};
            ComPtr<ID3D11Buffer> emptyBuffer;
            if (failed(device->CreateBuffer(&emptyDesc, &emptyData, emptyBuffer.GetAddressOf()), "empty light buffer") ||
                failed(device->CreateShaderResourceView(emptyBuffer.Get(), nullptr, EmptySceneView.ReleaseAndGetAddressOf()), "empty light view")) return false;
            D3D11_RASTERIZER_DESC raster {};
            raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE; raster.DepthClipEnable = TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {};
            depth.DepthEnable = FALSE; depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO; depth.DepthFunc = D3D11_COMPARISON_ALWAYS;
            return !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster state") &&
                !failed(device->CreateDepthStencilState(&depth, DepthState.ReleaseAndGetAddressOf()), "depth state");
        }
        bool Generate(ID3D11DeviceContext* context, const VolumetricFogDesc& desc,
            const Techniques::VolumetricFogSettings& medium, const Techniques::VolumetricLightingSettings& lighting,
            const CascadedShadowState& cascades, const Techniques::ShadowFilterSettings& filter,
            const std::array<ID3D11ShaderResourceView*, 5>& shadows, std::uint32_t& passCount, std::string& error,
            ID3D11PixelShader* injectionOverride = nullptr, ID3D11ShaderResourceView* completeLights = nullptr,
            const VirtualShadowSnapshot* virtualAtlas = nullptr, const VirtualShadowSamplingFrame* virtualFrame = nullptr,
            ID3D11ShaderResourceView* density = nullptr)
        {
            VolumetricFogConstants fog;
            VolumetricLightingConstants lights;
            if (!BuildVolumetricFogConstants(desc, medium, Width, Height, MaximumTextureSize, fog) ||
                !BuildVolumetricLightingConstants(desc, lighting, cascades, filter, lights, completeLights != nullptr, &medium,
                    virtualAtlas, virtualFrame, {fog.ViewportGrid.Z, fog.ViewportGrid.W}))
            { error = "Volume lighting requires finite medium, lights and camera/shadow transforms"; return false; }
            const auto passes = BuildVolumetricLightingPasses(Grid.Slices);
            if (passes.empty()) { error = "Invalid volume transport scan"; return false; }
            const auto upload = [&](ID3D11Buffer* buffer, const void* bytes, std::size_t count)
            {
                D3D11_MAPPED_SUBRESOURCE mapped {};
                if (FAILED(context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                { error = "DX11 volume light constant upload failed"; return false; }
                std::memcpy(mapped.pData, bytes, count); context->Unmap(buffer, 0); return true;
            };
            if (!upload(MediumBuffer.Get(), &fog, sizeof(fog))) return false;
            context->OMSetRenderTargets(0, nullptr, nullptr);
            constexpr UINT SurfaceFogAtlasSlot = 14;
            ID3D11ShaderResourceView* previousFrameInputs[SurfaceFogAtlasSlot + 1] {};
            context->PSSetShaderResources(0, SurfaceFogAtlasSlot + 1, previousFrameInputs);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthState.Get(), 0);
            context->RSSetState(Raster.Get());
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertices[ProgramIndex].Get(), nullptr, 0);
            ID3D11Buffer* mediumBuffer = MediumBuffer.Get();
            ID3D11Buffer* lightBuffer = LightingBuffer.Get();
            context->PSSetConstantBuffers(MediumSlot, 1, &mediumBuffer);
            context->PSSetConstantBuffers(LightingSlot, 1, &lightBuffer);
            auto* scene = completeLights ? completeLights : EmptySceneView.Get();
            context->PSSetShaderResources(CompleteLightSlot, 1, &scene);
            constexpr UINT DensitySlot=6;
            context->PSSetShaderResources(DensitySlot,1,&density);
            const D3D11_VIEWPORT viewport {0, 0, float(Grid.AtlasWidth()), float(Grid.AtlasHeight()), 0, 1};
            context->RSSetViewports(1, &viewport);
            ID3D11ShaderResourceView* empty[6] {};
            for (std::size_t index = 0; index < passes.size(); ++index)
            {
                const bool inject = index == 0, last = index + 1 == passes.size();
                lights.Control.W = passes[index].PrefixStride;
                if (!upload(LightingBuffer.Get(), &lights, sizeof(lights))) return false;
                context->PSSetShaderResources(0, 6, empty);
                auto& destination = last ? Atlas : Scratch[index % Scratch.size()];
                ID3D11RenderTargetView* target = destination.Target.Get();
                context->OMSetRenderTargets(1, &target, nullptr);
                context->PSSetShader(inject && injectionOverride ? injectionOverride : PixelPrograms[ProgramIndex][inject ? 0 : 1].Get(), nullptr, 0);
                if (inject) context->PSSetShaderResources(1, static_cast<UINT>(shadows.size()), shadows.data());
                else
                {
                    ID3D11ShaderResourceView* previous = Scratch[(index - 1) % Scratch.size()].View.Get();
                    context->PSSetShaderResources(0, 1, &previous);
                }
                constexpr UINT FullscreenVertexCount = 3;
                context->Draw(FullscreenVertexCount, 0);
                context->OMSetRenderTargets(0, nullptr, nullptr);
            }
            context->PSSetShaderResources(0, 6, empty);
            context->PSSetShaderResources(CompleteLightSlot, 1, empty);
            context->PSSetShaderResources(DensitySlot,1,empty);
            passCount = static_cast<std::uint32_t>(passes.size());
            return true;
        }
    private:
        std::array<Surface, 2> Scratch;
        Surface Atlas;
        std::array<ComPtr<ID3D11VertexShader>, 3> Vertices;
        std::array<std::array<ComPtr<ID3D11PixelShader>, 2>, 3> PixelPrograms;
        ComPtr<ID3D11Buffer> MediumBuffer, LightingBuffer;
        ComPtr<ID3D11ShaderResourceView> EmptySceneView;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
        FogGrid Grid;
        std::size_t ProgramIndex = 0;
        int Width = 0, Height = 0;
    };
}
#endif
