#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)

#include "Renderer/Shadows/ContactShadowContract.h"
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
    // Depth writers and previous visibility readers must be unbound by the caller.
    // Generate runs outside a raster pass and owns CS b0/t0/u0 and the CS shader;
    // these bindings are cleared before return. Geometry, lighting and presentation
    // remain the frame graph's responsibility. BeginFrame/failed dispatch hide stale
    // results, so a consumer cannot accidentally sample the previous camera frame.
    class DX11ContactShadows
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        static constexpr UINT ThreadsPerAxis = 8;
    public:
        void BeginFrame() noexcept { Ready = false; }
        ID3D11ShaderResourceView* View() const noexcept { return Ready ? VisibilityView.Get() : nullptr; }
        bool PrepareDepth(ID3D11Device* device, std::string& error)
        {
            if (DepthVertex && DepthPixel) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", "ContactShadows.hlsl", source, error)) return false;
            const D3D_SHADER_MACRO defines[] {{"EVANS_CONTACT_DEPTH", "1"}, {nullptr, nullptr}};
            for (unsigned stage = 0; stage < 2; ++stage)
            {
                ComPtr<ID3DBlob> code, diagnostics;
                const auto compiled = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX11/ContactShadows.hlsl", defines, nullptr,
                    stage == 0 ? "VSDepth" : "PSDepth", stage == 0 ? "vs_5_0" : "ps_5_0",
                    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                    0, code.GetAddressOf(), diagnostics.GetAddressOf());
                if (FAILED(compiled))
                { error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "Contact depth compilation failed"; return false; }
                const auto created = stage == 0 ? device->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                    DepthVertex.ReleaseAndGetAddressOf()) : device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                    DepthPixel.ReleaseAndGetAddressOf());
                if (FAILED(created)) { error = "Contact depth shader creation failed"; return false; }
            }
            return true;
        }
        ID3D11VertexShader* DepthVS() const noexcept { return DepthVertex.Get(); }
        ID3D11PixelShader* DepthPS() const noexcept { return DepthPixel.Get(); }
        ID3D11PixelShader* Receiver(ID3D11Device* device, Techniques::ShadingModel model,
            Techniques::ShadowFilterMode filter, bool deferred, bool multisample, std::string& error)
        {
            constexpr unsigned ProgramsPerFilter = 7;
            const unsigned group = filter == Techniques::ShadowFilterMode::PCSS ? 1 : filter == Techniques::ShadowFilterMode::VSM ? 2 : 0;
            const unsigned program = deferred ? (multisample ? 6 : 5) : unsigned(model);
            if (program >= ProgramsPerFilter) { error = "Invalid contact receiver model"; return nullptr; }
            auto& pixel = Receivers[group * ProgramsPerFilter + program];
            if (pixel) return pixel.Get();
            const char* name = group == 1 ? "PCSS" : group == 2 ? "VSM" : deferred ? "Deferred" : ShadingTechniqueFile(model, true);
            std::string source;
            if (!name || !LoadTechniqueShaderSource("DX11", (std::string(name) + ".hlsl").c_str(), source, error)) return nullptr;
            const std::string programText = std::to_string(program);
            std::vector<D3D_SHADER_MACRO> defines {{"EVANS_CONTACT_RECEIVER", "1"}};
            if (group) defines.push_back({group == 1 ? "EVANS_PCSS_PROGRAM" : "EVANS_VSM_PROGRAM", programText.c_str()});
            if (!group && deferred && multisample) defines.push_back({"EVANS_MULTISAMPLE", "1"});
            defines.push_back({nullptr, nullptr});
            ComPtr<ID3DBlob> code, diagnostics;
            const char* entry = deferred ? (group ? "PSMain" : "PSLighting") : "PSContact";
            // Match the existing surface compiler's strict-language policy.
            // FXC emits X4000 for some complete dynamic texture-return branches
            // in the original surface shaders, including without contact tracing.
            const auto compiled = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), name, defines.data(), nullptr, entry, "ps_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0, code.GetAddressOf(), diagnostics.GetAddressOf());
            if (FAILED(compiled))
            { error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "Contact receiver compilation failed"; return nullptr; }
            if (FAILED(device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, pixel.GetAddressOf())))
            { error = "Contact receiver creation failed"; return nullptr; }
            return pixel.Get();
        }
        bool BindReceiver(ID3D11Device* device, ID3D11DeviceContext* context, bool enabled, std::string& error)
        {
            if (enabled && !View()) { error = "Contact receiver requires current visibility"; return false; }
            if (!ReceiverConstants)
            {
                D3D11_BUFFER_DESC desc {}; desc.ByteWidth = sizeof(ShaderUInt4); desc.Usage = D3D11_USAGE_DYNAMIC;
                desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                if (FAILED(device->CreateBuffer(&desc, nullptr, ReceiverConstants.GetAddressOf())))
                { error = "Contact receiver constant allocation failed"; return false; }
            }
            const ShaderUInt4 control {enabled ? 1u : 0u, 0, 0, 0};
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(ReceiverConstants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            { error = "Contact receiver constant upload failed"; return false; }
            std::memcpy(mapped.pData, &control, sizeof(control)); context->Unmap(ReceiverConstants.Get(), 0);
            ID3D11Buffer* constants = ReceiverConstants.Get(); ID3D11ShaderResourceView* visibility = enabled ? View() : nullptr;
            context->PSSetConstantBuffers(6, 1, &constants); context->PSSetShaderResources(15, 1, &visibility);
            return true;
        }
        static bool Supported(ID3D11Device* device)
        {
            if (!device || device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) return false;
            UINT flags = 0;
            constexpr UINT Required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_SHADER_LOAD |
                D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW;
            return SUCCEEDED(device->CheckFormatSupport(DXGI_FORMAT_R32_FLOAT, &flags)) && (flags & Required) == Required;
        }
        bool Generate(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11ShaderResourceView* depth,
            const ContactShadowDispatch& desc, std::string& error)
        {
            Ready = false; error.clear();
            const auto fail = [&](const char* message) { error = message; return false; };
            if (!context || !depth || !Supported(device)) return fail("DX11 contact visibility requires FL11, R32F UAV and a depth view");
            ComPtr<ID3D11Device> contextDevice; context->GetDevice(contextDevice.GetAddressOf());
            ComPtr<ID3D11Resource> resource; depth->GetResource(resource.GetAddressOf());
            ComPtr<ID3D11Device> inputDevice; resource->GetDevice(inputDevice.GetAddressOf());
            if (contextDevice.Get() != device || inputDevice.Get() != device || (Owner && Owner.Get() != device))
                return fail("DX11 contact resources and context must belong to the same device");
            ComPtr<ID3D11Texture2D> source;
            if (FAILED(resource.As(&source))) return fail("DX11 contact depth must be a 2D texture");
            D3D11_TEXTURE2D_DESC input {}; source->GetDesc(&input);
            D3D11_SHADER_RESOURCE_VIEW_DESC view {}; depth->GetDesc(&view);
            const bool multisample = input.SampleDesc.Count > 1;
            if (input.ArraySize != 1 || view.Format != DXGI_FORMAT_R32_FLOAT ||
                (!multisample && (view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || view.Texture2D.MostDetailedMip != 0)) ||
                (multisample && view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2DMS))
                return fail("DX11 contact depth requires the base R32F 2D/MSAA view, without an array");
            ContactShadowConstants constants;
            if (!BuildContactShadowConstants(desc, input.Width, input.Height, input.SampleDesc.Count, constants, error)) return false;
            if (!Prepare(device, input.Width, input.Height, input.SampleDesc.Count, multisample, error)) return false;
            D3D11_MAPPED_SUBRESOURCE mapped {};
            if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                return fail("DX11 contact constant upload failed");
            std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
            ID3D11Buffer* buffer = Constants.Get(); ID3D11UnorderedAccessView* target = VisibilityTarget.Get();
            context->CSSetShader(Programs[multisample ? 1 : 0].Get(), nullptr, 0);
            context->CSSetConstantBuffers(0, 1, &buffer); context->CSSetShaderResources(0, 1, &depth);
            context->CSSetUnorderedAccessViews(0, 1, &target, nullptr);
            context->Dispatch((input.Width + ThreadsPerAxis - 1) / ThreadsPerAxis,
                (input.Height + ThreadsPerAxis - 1) / ThreadsPerAxis, input.SampleDesc.Count);
            target = nullptr; depth = nullptr; buffer = nullptr;
            context->CSSetUnorderedAccessViews(0, 1, &target, nullptr);
            context->CSSetShaderResources(0, 1, &depth); context->CSSetConstantBuffers(0, 1, &buffer);
            context->CSSetShader(nullptr, nullptr, 0);
            Ready = true;
            return true;
        }
    private:
        bool Prepare(ID3D11Device* device, UINT width, UINT height, UINT samples, bool multisample, std::string& error)
        {
            auto& program = Programs[multisample ? 1 : 0];
            const auto checked = [&](HRESULT result, const char* message)
            {
                if (SUCCEEDED(result)) return true;
                error = std::string("DX11 contact ") + message + " failed: " + std::to_string(result); return false;
            };
            if (!program)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("DX11", "ContactShadows.hlsl", source, error)) return false;
                const D3D_SHADER_MACRO defines[] {{"EVANS_CONTACT_MULTISAMPLE", multisample ? "1" : "0"}, {nullptr, nullptr}};
                ComPtr<ID3DBlob> code, diagnostics;
                if (!checked(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX11/ContactShadows.hlsl", defines, nullptr,
                    "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 |
                    D3DCOMPILE_WARNINGS_ARE_ERRORS, 0, code.GetAddressOf(), diagnostics.GetAddressOf()), "shader compilation"))
                {
                    if (diagnostics) error += std::string(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize());
                    return false;
                }
                if (!checked(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                    program.GetAddressOf()), "compute shader")) return false;
                Owner = device;
            }
            if (!Constants)
            {
                D3D11_BUFFER_DESC buffer {}; buffer.ByteWidth = sizeof(ContactShadowConstants);
                buffer.Usage = D3D11_USAGE_DYNAMIC; buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
                buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                if (!checked(device->CreateBuffer(&buffer, nullptr, Constants.GetAddressOf()), "constant buffer")) return false;
            }
            if (!Visibility || Width != width || Height != height || Samples != samples)
            {
                ComPtr<ID3D11Texture2D> texture;
                ComPtr<ID3D11ShaderResourceView> view;
                ComPtr<ID3D11UnorderedAccessView> target;
                D3D11_TEXTURE2D_DESC image {}; image.Width = width; image.Height = height;
                image.MipLevels = image.SampleDesc.Count = 1; image.ArraySize = samples;
                image.Format = DXGI_FORMAT_R32_FLOAT; image.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
                D3D11_SHADER_RESOURCE_VIEW_DESC sampled {}; sampled.Format = image.Format;
                sampled.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
                sampled.Texture2DArray.MipLevels = 1; sampled.Texture2DArray.ArraySize = samples;
                D3D11_UNORDERED_ACCESS_VIEW_DESC storage {}; storage.Format = image.Format;
                storage.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY; storage.Texture2DArray.ArraySize = samples;
                if (!checked(device->CreateTexture2D(&image, nullptr, texture.GetAddressOf()), "visibility texture") ||
                    !checked(device->CreateShaderResourceView(texture.Get(), &sampled, view.GetAddressOf()), "visibility SRV") ||
                    !checked(device->CreateUnorderedAccessView(texture.Get(), &storage, target.GetAddressOf()), "visibility UAV")) return false;
                Visibility = std::move(texture); VisibilityView = std::move(view); VisibilityTarget = std::move(target);
                Width = width; Height = height; Samples = samples;
            }
            return true;
        }
        ComPtr<ID3D11Device> Owner;
        ComPtr<ID3D11VertexShader> DepthVertex;
        ComPtr<ID3D11PixelShader> DepthPixel;
        std::array<ComPtr<ID3D11PixelShader>, 21> Receivers;
        ComPtr<ID3D11Buffer> ReceiverConstants;
        std::array<ComPtr<ID3D11ComputeShader>, 2> Programs;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11Texture2D> Visibility;
        ComPtr<ID3D11ShaderResourceView> VisibilityView;
        ComPtr<ID3D11UnorderedAccessView> VisibilityTarget;
        UINT Width = 0, Height = 0, Samples = 0;
        bool Ready = false;
    };
}
#endif
