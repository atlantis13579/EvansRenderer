#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/TechniqueShader.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstring>
#include <unordered_map>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    class DX11VSM
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        static constexpr std::size_t SurfaceCount = 5, ProgramCount = 10, ConstantSlot = 5;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
            int Width = 0, Height = 0;
        };
        struct Map { Surface Raw, Filtered; bool Ready = false, RawWritten = false; };
    public:
        static constexpr std::size_t Deferred = 5, DeferredMultisample = 6, Volume = 7, Caster = 8, Filter = 9;
        static bool Supported(ID3D11Device* device)
        {
            UINT support = 0;
            constexpr UINT Required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_RENDER_TARGET | D3D11_FORMAT_SUPPORT_SHADER_LOAD;
            return SUCCEEDED(device->CheckFormatSupport(DXGI_FORMAT_R32G32_FLOAT, &support)) && (support & Required) == Required;
        }
        void BeginFrame()
        { for (auto& pair : Maps) pair.second.Ready = pair.second.RawWritten = false; }
        void Release(TextureHandle handle) { Maps.erase(handle.Value); }
        ID3D11ShaderResourceView* View(TextureHandle handle) const
        {
            const auto found = Maps.find(handle.Value);
            return found != Maps.end() && found->second.Ready ? found->second.Filtered.View.Get() : nullptr;
        }
        bool Prepare(ID3D11Device* device, std::size_t program, bool weighted, std::string& error)
        {
            if (program >= ProgramCount || (weighted && program >= SurfaceCount))
            { error = "Invalid DX11 VSM program"; return false; }
            auto& pixel = Pixels[weighted ? ProgramCount + program : program];
            if (!pixel)
            {
                ComPtr<ID3DBlob> code;
                if (!Compile(program, weighted ? "PSWeighted" : "PSMain", "ps_5_0", code, error)) return false;
                if (FAILED(device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, pixel.GetAddressOf())))
                { error = "DX11 VSM pixel shader creation failed"; return false; }
            }
            if (!Constants)
            {
                D3D11_BUFFER_DESC desc {};
                desc.ByteWidth = sizeof(VSMConstants); desc.Usage = D3D11_USAGE_DYNAMIC;
                desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                if (FAILED(device->CreateBuffer(&desc, nullptr, Constants.GetAddressOf())))
                { error = "DX11 VSM constant allocation failed"; return false; }
            }
            return true;
        }
        ID3D11PixelShader* Pixel(std::size_t program, bool weighted = false) const
        { return Pixels[weighted ? ProgramCount + program : program].Get(); }
        ID3D11VertexShader* CasterVertex(bool instanced) const { return instanced ? InstanceVertex.Get() : Vertex.Get(); }
        bool BeginMap(ID3D11Device* device, ID3D11DeviceContext* context, TextureHandle handle,
            int width, int height, bool clear, ID3D11RenderTargetView*& target, std::string& error)
        {
            target = nullptr;
            if (!Supported(device)) { error = "DX11 VSM requires renderable, loadable RG32_FLOAT moments"; return false; }
            if (!Prepare(device, Caster, false, error) || !Prepare(device, Filter, false, error) ||
                !PrepareVertex(device, Caster, "VSMain", Vertex, error) ||
                !PrepareVertex(device, Caster, "VSInstanced", InstanceVertex, error) ||
                !PrepareVertex(device, Filter, "VSMain", FilterVertex, error)) return false;
            if (!FilterRaster)
            {
                D3D11_RASTERIZER_DESC desc {}; desc.FillMode = D3D11_FILL_SOLID; desc.CullMode = D3D11_CULL_NONE; desc.DepthClipEnable = TRUE;
                if (FAILED(device->CreateRasterizerState(&desc, FilterRaster.GetAddressOf())))
                { error = "DX11 VSM filter raster state allocation failed"; return false; }
            }
            if (!FilterDepth)
            {
                D3D11_DEPTH_STENCIL_DESC desc {}; desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
                if (FAILED(device->CreateDepthStencilState(&desc, FilterDepth.GetAddressOf())))
                { error = "DX11 VSM filter depth state allocation failed"; return false; }
            }
            auto& map = Maps[handle.Value];
            if (!clear && !map.RawWritten)
            { error = "DX11 VSM cannot load a caster map before its first clear in this frame"; return false; }
            if (!CreateSurface(device, width, height, map.Raw, error) || !CreateSurface(device, width, height, map.Filtered, error) ||
                !CreateSurface(device, width, height, Horizontal, error)) return false;
            map.Ready = false;
            target = map.Raw.Target.Get();
            if (clear) { const float unoccluded[] {1, 1, 0, 0}; context->ClearRenderTargetView(target, unoccluded); }
            return true;
        }
        bool BindCaster(ID3D11DeviceContext* context, const Matrix4& projection,
            const Techniques::VSMSettings& settings, std::string& error)
        {
            auto constants = BuildVSMParameters(settings);
            if (!BuildVSMEncoding(projection, constants.DepthEncoding[0], error)) return false;
            return Upload(context, constants, error);
        }
        bool BindReceiver(ID3D11DeviceContext* context, const Matrix4& projection, const LightingState& lighting,
            const CascadedShadowState& cascades, const Techniques::VSMSettings& settings, bool shadows, std::string& error)
        {
            VSMConstants constants;
            return BuildVSMConstants(projection, lighting, cascades, settings, shadows, constants, error) && Upload(context, constants, error);
        }
        bool FilterMap(ID3D11DeviceContext* context, TextureHandle handle, const Techniques::VSMSettings& settings, std::string& error)
        {
            auto found = Maps.find(handle.Value);
            if (found == Maps.end()) { error = "DX11 VSM filter has no caster map"; return false; }
            auto& map = found->second;
            auto constants = BuildVSMParameters(settings);
            context->OMSetRenderTargets(0, nullptr, nullptr);
            context->IASetInputLayout(nullptr); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(FilterVertex.Get(), nullptr, 0); context->PSSetShader(Pixel(Filter), nullptr, 0);
            context->RSSetState(FilterRaster.Get()); context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
            context->OMSetDepthStencilState(FilterDepth.Get(), 0);
            const D3D11_VIEWPORT viewport {0, 0, float(map.Raw.Width), float(map.Raw.Height), 0, 1};
            context->RSSetViewports(1, &viewport);
            for (UINT axis = 0; axis < 2; ++axis)
            {
                constants.Control.W = axis;
                if (!Upload(context, constants, error)) return false;
                auto* input = axis == 0 ? map.Raw.View.Get() : Horizontal.View.Get();
                auto* output = axis == 0 ? Horizontal.Target.Get() : map.Filtered.Target.Get();
                context->OMSetRenderTargets(1, &output, nullptr);
                context->PSSetShaderResources(0, 1, &input); context->Draw(3, 0);
                input = nullptr; context->PSSetShaderResources(0, 1, &input);
                context->OMSetRenderTargets(0, nullptr, nullptr);
            }
            map.RawWritten = map.Ready = true;
            return true;
        }
    private:
        bool Compile(std::size_t program, const char* entry, const char* profile, ComPtr<ID3DBlob>& code, std::string& error)
        {
            if (Source.empty() && !LoadTechniqueShaderSource("DX11", "VSM.hlsl", Source, error)) return false;
            const auto programText = std::to_string(program);
            const D3D_SHADER_MACRO defines[] {{"EVANS_VSM_PROGRAM", programText.c_str()}, {nullptr, nullptr}};
            ComPtr<ID3DBlob> diagnostics;
            const HRESULT result = Evans::Renderer::CompileCachedD3DShader(Source.data(), Source.size(), "DX11/VSM.hlsl", defines, nullptr, entry, profile,
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                0, code.GetAddressOf(), diagnostics.GetAddressOf());
            if (SUCCEEDED(result)) return true;
            error = diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize()) :
                "DX11 VSM shader compilation failed";
            return false;
        }
        bool PrepareVertex(ID3D11Device* device, std::size_t program, const char* entry,
            ComPtr<ID3D11VertexShader>& vertex, std::string& error)
        {
            if (vertex) return true;
            ComPtr<ID3DBlob> code;
            if (!Compile(program, entry, "vs_5_0", code, error)) return false;
            if (SUCCEEDED(device->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, vertex.GetAddressOf()))) return true;
            error = "DX11 VSM vertex shader creation failed"; return false;
        }
        bool CreateSurface(ID3D11Device* device, int width, int height, Surface& surface, std::string& error)
        {
            if (surface.Texture && surface.Width == width && surface.Height == height) return true;
            Surface candidate; candidate.Width = width; candidate.Height = height;
            D3D11_TEXTURE2D_DESC desc {}; desc.Width = UINT(width); desc.Height = UINT(height);
            desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1; desc.Format = DXGI_FORMAT_R32G32_FLOAT;
            desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(device->CreateTexture2D(&desc, nullptr, candidate.Texture.GetAddressOf())) ||
                FAILED(device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf())) ||
                FAILED(device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf())))
            { error = "DX11 VSM moment target allocation failed"; return false; }
            surface = std::move(candidate); return true;
        }
        bool Upload(ID3D11DeviceContext* context, const VSMConstants& constants, std::string& error)
        {
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            { error = "DX11 VSM constants upload failed"; return false; }
            std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
            ID3D11Buffer* buffer = Constants.Get();
            context->VSSetConstantBuffers(ConstantSlot, 1, &buffer); context->PSSetConstantBuffers(ConstantSlot, 1, &buffer);
            return true;
        }
        std::string Source;
        std::array<ComPtr<ID3D11PixelShader>, ProgramCount + SurfaceCount> Pixels;
        ComPtr<ID3D11VertexShader> Vertex, InstanceVertex, FilterVertex;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11RasterizerState> FilterRaster;
        ComPtr<ID3D11DepthStencilState> FilterDepth;
        std::unordered_map<std::uint32_t, Map> Maps;
        Surface Horizontal;
    };
}
#endif
