#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Lighting/GlobalIlluminationContract.h"
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
    // Four fullscreen stages own their outputs. The scene is copied only after
    // composition, so neither ray hits nor temporal history contain feedback.
    class DX11GlobalIllumination : public GlobalIlluminationHistory
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
    public:
        bool Prepare(ID3D11Device* device, int width, int height, std::string& error)
        {
            if (width != Width || height != Height)
            {
                Raw = {}; Filtered = {}; Composite = {}; Colors = {}; Depths = {}; Normals = {};
                Width = width; Height = height; Invalidate();
            }
            auto failed = [&](HRESULT result, const char* operation)
            {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 SSGI ") + operation + " failed: " + std::to_string(result); return true;
            };
            auto allocate = [&](Surface& surface, DXGI_FORMAT format)
            {
                if (surface.Texture) return true;
                Surface candidate;
                D3D11_TEXTURE2D_DESC desc {};
                desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = 1;
                desc.Format = format; desc.SampleDesc.Count = 1;
                desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
                if (failed(device->CreateTexture2D(&desc, nullptr, candidate.Texture.GetAddressOf()), "texture") ||
                    failed(device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf()), "target") ||
                    failed(device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf()), "view")) return false;
                surface = std::move(candidate); return true;
            };
            constexpr auto ColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (!allocate(Raw, ColorFormat) || !allocate(Filtered, ColorFormat) || !allocate(Composite, ColorFormat)) return false;
            for (unsigned i = 0; i < Colors.size(); ++i)
                if (!allocate(Colors[i], ColorFormat) || !allocate(Depths[i], DXGI_FORMAT_R32_FLOAT) ||
                    !allocate(Normals[i], ColorFormat)) return false;
            if (Vertex && Pixel && TemporalPixel && Constants && Raster && DepthState) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", "SSGI.hlsl", source, error)) return false;
            const char* entries[] {"VSMain", "PSMain", "PSTemporal"};
            for (unsigned stage = 0; stage < std::size(entries); ++stage)
            {
                ComPtr<ID3DBlob> binary, diagnostics;
                if (failed(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "SSGI.hlsl", nullptr, nullptr, entries[stage],
                    stage ? "ps_5_0" : "vs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
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
                    (stage == 1 ? Pixel : TemporalPixel).ReleaseAndGetAddressOf()), "pixel shader")) return false;
            }
            D3D11_BUFFER_DESC buffer {};
            buffer.ByteWidth = sizeof(GlobalIlluminationConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer, nullptr, Constants.ReleaseAndGetAddressOf()), "constants")) return false;
            D3D11_RASTERIZER_DESC raster {};
            raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE; raster.DepthClipEnable = TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {};
            return !failed(device->CreateRasterizerState(&raster, Raster.ReleaseAndGetAddressOf()), "raster state") &&
                !failed(device->CreateDepthStencilState(&depth, DepthState.ReleaseAndGetAddressOf()), "depth state");
        }

        bool Resolve(ID3D11DeviceContext* context, const GlobalIlluminationDesc& desc,
            const Techniques::GlobalIlluminationSettings& settings, Techniques::ShadingModel shading,
            ID3D11Texture2D* scene, ID3D11ShaderResourceView* color, ID3D11ShaderResourceView* depth,
            ID3D11ShaderResourceView* normal, ID3D11ShaderResourceView* diffuse, ID3D11ShaderResourceView* ambient,
            ID3D11ShaderResourceView* velocity, std::string& error)
        {
            GlobalIlluminationConstants constants;
            if (!BuildGlobalIlluminationConstants(desc, settings, shading, Width, Height, Valid, constants))
            { error = "SSGI requires finite invertible current/previous projections"; return false; }
            constexpr UINT InputCount = 10;
            ID3D11ShaderResourceView* empty[InputCount] {};
            context->OMSetRenderTargets(0, nullptr, nullptr);
            context->PSSetShaderResources(0, InputCount, empty);
            context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(DepthState.Get(), 0);
            context->RSSetState(Raster.Get());
            D3D11_VIEWPORT viewport {0, 0, float(Width), float(Height), 0, 1};
            context->RSSetViewports(1, &viewport);
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertex.Get(), nullptr, 0);
            ID3D11Buffer* buffer = Constants.Get(); context->PSSetConstantBuffers(0, 1, &buffer);
            const unsigned write = WriteIndex();
            using Phase = Techniques::GlobalIlluminationPassKind;
            for (auto phase : {Phase::Trace, Phase::Spatial, Phase::Temporal, Phase::Composite})
            {
                constants.Options.X = static_cast<unsigned>(phase);
                D3D11_MAPPED_SUBRESOURCE mapped {};
                if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                { error = "DX11 SSGI constants mapping failed"; return false; }
                std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
                ID3D11RenderTargetView* outputs[] {phase == Phase::Trace ? Raw.Target.Get() : phase == Phase::Spatial ?
                    Filtered.Target.Get() : phase == Phase::Temporal ? Colors[write].Target.Get() : Composite.Target.Get(),
                    Depths[write].Target.Get(), Normals[write].Target.Get()};
                context->OMSetRenderTargets(phase == Phase::Temporal ? 3 : 1, outputs, nullptr);
                context->PSSetShader(phase == Phase::Temporal ? TemporalPixel.Get() : Pixel.Get(), nullptr, 0);
                ID3D11ShaderResourceView* inputs[] {color, depth, normal, diffuse, ambient, velocity,
                    phase == Phase::Spatial ? Raw.View.Get() : phase == Phase::Temporal ? Filtered.View.Get() :
                    phase == Phase::Composite ? Colors[write].View.Get() : nullptr,
                    Colors[ReadIndex].View.Get(), Depths[ReadIndex].View.Get(), Normals[ReadIndex].View.Get()};
                context->PSSetShaderResources(0, InputCount, inputs);
                context->Draw(3, 0);
                context->PSSetShaderResources(0, InputCount, empty);
                context->OMSetRenderTargets(0, nullptr, nullptr);
            }
            context->CopyResource(scene, Composite.Texture.Get());
            Resolved = true; return true;
        }
    private:
        Surface Raw, Filtered, Composite;
        std::array<Surface, 2> Colors, Depths, Normals;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> Pixel, TemporalPixel;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
        int Width = 0, Height = 0;
    };
}
#endif
