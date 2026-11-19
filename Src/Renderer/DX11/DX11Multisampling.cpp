#include "Renderer/D3DShaderCache.h"
#include "Renderer/DX11/DX11Multisampling.h"
#if defined(_WIN32)
#include "Renderer/TechniqueShader.h"
#include <d3dcompiler.h>

namespace Evans::Renderer
{
    namespace
    {
        bool Check(HRESULT result, const char* operation, std::string& error)
        {
            if (SUCCEEDED(result)) return true;
            error = std::string(operation) + " failed: " + std::to_string(result);
            return false;
        }
    }

    std::uint32_t DX11Multisampling::SupportedCounts(ID3D11Device* device)
    {
        std::uint32_t counts = 1;
        if (!device) return counts;
        // Quality zero works across every attachment format. Restrict the GUI
        // to the intersection, so enabling Deferred or motion blur stays valid.
        for (const auto format : {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT})
        {
            UINT flags = 0;
            if (FAILED(device->CheckFormatSupport(format, &flags)) ||
                !(flags & D3D11_FORMAT_SUPPORT_MULTISAMPLE_LOAD)) return counts;
        }
        for (const auto format : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT})
        {
            UINT flags = 0;
            constexpr UINT Required = D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET | D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE;
            if (FAILED(device->CheckFormatSupport(format, &flags)) || (flags & Required) != Required) return counts;
        }
        for (const UINT count : {2u, 4u, 8u})
        {
            bool supported = true;
            for (const auto format : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_D32_FLOAT})
            {
                UINT levels = 0;
                supported &= SUCCEEDED(device->CheckMultisampleQualityLevels(format, count, &levels)) && levels > 0;
            }
            if (supported) counts |= count;
        }
        return counts;
    }

    bool DX11Multisampling::CreateSurface(ID3D11Device* device, Surface& surface,
        DXGI_FORMAT format, std::string& error)
    {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = Width; desc.Height = Height;
        desc.MipLevels = desc.ArraySize = 1;
        desc.SampleDesc.Count = SampleCount;
        desc.Format = format;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        return Check(device->CreateTexture2D(&desc, nullptr, surface.Texture.GetAddressOf()), "Create MSAA color", error) &&
            Check(device->CreateRenderTargetView(surface.Texture.Get(), nullptr, surface.Target.GetAddressOf()), "Create MSAA RTV", error) &&
            Check(device->CreateShaderResourceView(surface.Texture.Get(), nullptr, surface.View.GetAddressOf()), "Create MSAA SRV", error);
    }

    bool DX11Multisampling::CreatePrograms(ID3D11Device* device, std::string& error)
    {
        std::string source;
        if (!LoadTechniqueShaderSource("DX11", "MSAA.hlsl", source, error)) return false;
        const auto compile = [&](const char* entry, const char* profile, ComPtr<ID3DBlob>& code)
        {
            ComPtr<ID3DBlob> messages;
            const auto result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX11/MSAA.hlsl", nullptr, nullptr,
                entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                code.GetAddressOf(), messages.GetAddressOf());
            if (FAILED(result) && messages)
            {
                error.assign(static_cast<const char*>(messages->GetBufferPointer()), messages->GetBufferSize());
                return false;
            }
            return Check(result, "Compile MSAA", error);
        };
        ComPtr<ID3DBlob> vertex;
        if (!compile("VSMain", "vs_5_0", vertex) || !Check(device->CreateVertexShader(vertex->GetBufferPointer(),
            vertex->GetBufferSize(), nullptr, Vertex.GetAddressOf()), "Create MSAA VS", error)) return false;
        const char* entries[] = {"PSDepth", "PSMaterial", "PSVelocity", "PSRestoreColor"};
        ComPtr<ID3D11PixelShader>* shaders[] = {&DepthResolve, &MaterialResolve, &VelocityResolve, &ColorRestore};
        for (std::size_t index = 0; index < std::size(entries); ++index)
        {
            ComPtr<ID3DBlob> code;
            if (!compile(entries[index], "ps_5_0", code) || !Check(device->CreatePixelShader(code->GetBufferPointer(),
                code->GetBufferSize(), nullptr, shaders[index]->GetAddressOf()), "Create MSAA PS", error)) return false;
        }
        D3D11_DEPTH_STENCIL_DESC depth {};
        depth.DepthEnable = TRUE;
        depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        depth.DepthFunc = D3D11_COMPARISON_ALWAYS;
        if (!Check(device->CreateDepthStencilState(&depth, DepthWrite.GetAddressOf()), "Create resolve depth state", error)) return false;
        depth.DepthEnable = FALSE;
        depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        if (!Check(device->CreateDepthStencilState(&depth, DepthDisabled.GetAddressOf()), "Create restore depth state", error)) return false;
        D3D11_RASTERIZER_DESC raster {};
        raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE;
        raster.DepthClipEnable = TRUE; raster.MultisampleEnable = TRUE;
        return Check(device->CreateRasterizerState(&raster, Rasterizer.GetAddressOf()), "Create MSAA raster state", error);
    }

    bool DX11Multisampling::Prepare(ID3D11Device* device, int width, int height, UINT samples,
        bool hdr, bool materials, bool velocity, std::string& error)
    {
        const auto format = hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        if (Depth && Width == width && Height == height && SampleCount == samples && ColorFormat == format &&
            bool(Materials[0].Texture) == materials && bool(Velocity.Texture) == velocity) return true;
        // Publish a complete set only after allocation and shader compilation
        // succeed; failed reconfiguration never mixes sample counts or sizes.
        DX11Multisampling candidate;
        candidate.Width = width; candidate.Height = height;
        candidate.SampleCount = samples; candidate.ColorFormat = format;
        if (!candidate.CreateSurface(device, candidate.Color, format, error)) return false;
        if (materials) for (auto& surface : candidate.Materials)
            if (!candidate.CreateSurface(device, surface, DXGI_FORMAT_R16G16B16A16_FLOAT, error)) return false;
        if (velocity && !candidate.CreateSurface(device, candidate.Velocity, DXGI_FORMAT_R16G16B16A16_FLOAT, error)) return false;
        D3D11_TEXTURE2D_DESC depth {};
        depth.Width = width; depth.Height = height;
        depth.MipLevels = depth.ArraySize = 1;
        depth.SampleDesc.Count = samples; depth.Format = DXGI_FORMAT_R32_TYPELESS;
        depth.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        if (!Check(device->CreateTexture2D(&depth, nullptr, candidate.Depth.GetAddressOf()), "Create MSAA depth", error)) return false;
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv {};
        dsv.Format = DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
        if (!Check(device->CreateDepthStencilView(candidate.Depth.Get(), &dsv, candidate.DepthTarget.GetAddressOf()), "Create MSAA DSV", error)) return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.Format = DXGI_FORMAT_R32_FLOAT; srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
        if (!Check(device->CreateShaderResourceView(candidate.Depth.Get(), &srv, candidate.DepthView.GetAddressOf()), "Create MSAA depth SRV", error)) return false;
        if (Vertex)
        {
            candidate.Vertex = Vertex; candidate.DepthResolve = DepthResolve; candidate.MaterialResolve = MaterialResolve;
            candidate.VelocityResolve = VelocityResolve; candidate.ColorRestore = ColorRestore;
            candidate.DepthWrite = DepthWrite; candidate.DepthDisabled = DepthDisabled; candidate.Rasterizer = Rasterizer;
        }
        else if (!candidate.CreatePrograms(device, error)) return false;
        *this = std::move(candidate);
        return true;
    }

    void DX11Multisampling::Process(ID3D11DeviceContext* context, Techniques::MultisamplePassKind pass,
        const ResolveTargets& targets)
    {
        using Pass = Techniques::MultisamplePassKind;
        constexpr UINT ShaderInputCount = 14; // Includes the previous lighting pass's cascade maps.
        ID3D11ShaderResourceView* inputs[ShaderInputCount] {};
        context->PSSetShaderResources(0, ShaderInputCount, inputs);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        // Hardware color resolve averages in linear space, including sRGB RTVs.
        // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
        if (pass == Pass::OpaqueResolve || pass == Pass::ColorResolve)
            context->ResolveSubresource(targets.Color, 0, Color.Texture.Get(), 0, ColorFormat);
        if (pass == Pass::ColorResolve) return;

        const bool opaque = pass == Pass::OpaqueResolve;
        const bool material = opaque && Materials[0].Texture;
        ID3D11PixelShader* pixel = opaque ? (material ? MaterialResolve.Get() : DepthResolve.Get()) :
            (pass == Pass::VelocityResolve ? VelocityResolve.Get() : ColorRestore.Get());
        ID3D11RenderTargetView* target = pass == Pass::VelocityResolve ? targets.Velocity : Color.Target.Get();
        if (opaque) context->OMSetRenderTargets(material ? GBufferAttachmentCount : 0,
            material ? targets.Materials.data() : nullptr, targets.Depth);
        else context->OMSetRenderTargets(1, &target, nullptr);
        if (pass == Pass::ColorRestore) inputs[0] = targets.ColorView;
        else
        {
            inputs[1] = DepthView.Get();
            if (material) for (std::size_t i = 0; i < Materials.size(); ++i) inputs[2 + i] = Materials[i].View.Get();
            if (pass == Pass::VelocityResolve) inputs[7] = Velocity.View.Get();
        }
        context->PSSetShaderResources(0, ShaderInputCount, inputs);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(Vertex.Get(), nullptr, 0);
        context->PSSetShader(pixel, nullptr, 0);
        context->OMSetDepthStencilState(opaque ? DepthWrite.Get() : DepthDisabled.Get(), 0);
        context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
        context->RSSetState(Rasterizer.Get());
        const D3D11_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(Width), static_cast<float>(Height), 0.0f, 1.0f};
        context->RSSetViewports(1, &viewport);
        constexpr UINT FullscreenTriangleVertices = 3;
        context->Draw(FullscreenTriangleVertices, 0);
        for (auto& input : inputs) input = nullptr;
        context->PSSetShaderResources(0, ShaderInputCount, inputs);
        context->OMSetRenderTargets(0, nullptr, nullptr);
    }
}
#endif
