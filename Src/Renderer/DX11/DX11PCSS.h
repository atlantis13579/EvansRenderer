#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/TechniqueShader.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstring>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    class DX11PCSS
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        static constexpr std::size_t SurfaceCount = 5, ProgramCount = 8, ReceiverSlot = 5;
    public:
        static constexpr std::size_t Deferred = 5, DeferredMultisample = 6, Volume = 7;
        bool Prepare(ID3D11Device* device, std::size_t program, bool weighted, std::string& error)
        {
            if (program >= ProgramCount || (weighted && program >= SurfaceCount))
            { error = "Invalid DX11 PCSS receiver program"; return false; }
            auto& pixel = Pixels[weighted ? ProgramCount + program : program];
            if (!pixel)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("DX11", "PCSS.hlsl", source, error)) return false;
                const auto programText = std::to_string(program);
                const D3D_SHADER_MACRO defines[] {{"EVANS_PCSS_PROGRAM", programText.c_str()}, {nullptr, nullptr}};
                ComPtr<ID3DBlob> code, diagnostics;
                const HRESULT compiled = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX11/PCSS.hlsl", defines, nullptr,
                    weighted ? "PSWeighted" : "PSMain", "ps_5_0",
                    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                    0, code.GetAddressOf(), diagnostics.GetAddressOf());
                if (FAILED(compiled))
                {
                    error = diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize()) :
                        "DX11 PCSS compilation failed";
                    return false;
                }
                if (FAILED(device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, pixel.GetAddressOf())))
                { error = "DX11 PCSS pixel shader creation failed"; return false; }
            }
            if (!Constants)
            {
                D3D11_BUFFER_DESC desc {};
                desc.ByteWidth = sizeof(PCSSConstants); desc.Usage = D3D11_USAGE_DYNAMIC;
                desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                if (FAILED(device->CreateBuffer(&desc, nullptr, Constants.GetAddressOf())))
                { error = "DX11 PCSS constant-buffer allocation failed"; return false; }
            }
            return true;
        }
        ID3D11PixelShader* Pixel(std::size_t program, bool weighted = false) const
        { return Pixels[weighted ? ProgramCount + program : program].Get(); }
        bool Bind(ID3D11DeviceContext* context, const Matrix4& lightProjection, const LightingState& lighting,
            const CascadedShadowState& cascades, const Techniques::PCSSSettings& settings, bool shadowAvailable, std::string& error)
        {
            PCSSConstants constants;
            if (!BuildPCSSConstants(lightProjection, lighting, cascades, settings, shadowAvailable, constants, error)) return false;
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            { error = "DX11 PCSS receiver upload failed"; return false; }
            // WRITE_DISCARD renames storage for queued draws; every receiver
            // retains the constants and projection metrics recorded for it.
            std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
            ID3D11Buffer* buffer = Constants.Get();
            context->PSSetConstantBuffers(ReceiverSlot, 1, &buffer);
            return true;
        }
    private:
        std::array<ComPtr<ID3D11PixelShader>, ProgramCount + SurfaceCount> Pixels;
        ComPtr<ID3D11Buffer> Constants;
    };
}
#endif
