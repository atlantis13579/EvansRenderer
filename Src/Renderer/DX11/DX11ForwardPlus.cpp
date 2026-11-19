#include "Renderer/D3DShaderCache.h"
#include "Renderer/DX11/DX11ForwardPlus.h"
#if defined(_WIN32)
#include "Renderer/TechniqueShader.h"

#include <cstring>
#include <d3dcompiler.h>

namespace Evans::Renderer
{
    void DX11ForwardPlus::Shutdown() noexcept
    {
        Ready = false; CurrentGrid = {}; AllocatedTiles = AllocatedIndices = 0;
        for (auto& program : Programs) program.Reset();
        SurfaceVertex.Reset(); LightingVertex.Reset();
        for (auto& filter : LightingPixels) for (auto& program : filter) program.Reset();
        for (auto& filter : SurfacePixels) for (auto& program : filter) program.Reset();
        SceneView.Reset(); SceneBuffer.Reset(); SurfaceConstants.Reset(); UploadedLights.reset();
        EmptyHeaderView.Reset(); EmptyHeader.Reset(); EmptyIndexView.Reset(); EmptyIndex.Reset();
        Constants.Reset(); BoundsView.Reset(); BoundsBuffer.Reset();
        HeaderView.Reset(); HeaderTarget.Reset(); HeaderBuffer.Reset();
        IndexView.Reset(); IndexTarget.Reset(); IndexBuffer.Reset(); Owner.Reset();
    }
    bool DX11ForwardPlus::UploadScene(ID3D11Device* device, ID3D11DeviceContext* context,
        const std::shared_ptr<const SceneLightCollection>& lights, std::string& error)
    {
        error.clear();
        if (!device || !context || !lights || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
        { error = "DX11 Forward+ scene upload requires complete lights and an immediate context"; return false; }
        ComPtr<ID3D11Device> contextDevice; context->GetDevice(contextDevice.GetAddressOf());
        if (contextDevice.Get() != device || (Owner && Owner.Get() != device))
        { error = "DX11 Forward+ scene resources must share one device"; return false; }
        Owner = device;
        const auto checked = [&](HRESULT code, const char* operation)
        {
            if (SUCCEEDED(code)) return true;
            error = std::string("DX11 Forward+ ") + operation + " failed: " + std::to_string(code); return false;
        };
        const auto storage = [&](UINT count, UINT stride, const void* initial,
            ComPtr<ID3D11Buffer>& buffer, ComPtr<ID3D11ShaderResourceView>& view)
        {
            D3D11_BUFFER_DESC desc {}; desc.ByteWidth = count * stride;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            desc.StructureByteStride = stride;
            D3D11_SUBRESOURCE_DATA data {}; data.pSysMem = initial;
            ComPtr<ID3D11Buffer> candidate; ComPtr<ID3D11ShaderResourceView> candidateView;
            if (!checked(device->CreateBuffer(&desc, initial ? &data : nullptr, candidate.GetAddressOf()), "material storage allocation")) return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC srv {}; srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements = count;
            if (!checked(device->CreateShaderResourceView(candidate.Get(), &srv, candidateView.GetAddressOf()), "material storage view")) return false;
            buffer = std::move(candidate); view = std::move(candidateView); return true;
        };
        const ForwardPlusTile emptyHeader {};
        const std::uint32_t emptyIndex = 0;
        if ((!SceneView && !storage(MaximumForwardPlusLights, sizeof(ShaderLight), nullptr, SceneBuffer, SceneView)) ||
            (!EmptyHeaderView && !storage(1, sizeof(emptyHeader), &emptyHeader, EmptyHeader, EmptyHeaderView)) ||
            (!EmptyIndexView && !storage(1, sizeof(emptyIndex), &emptyIndex, EmptyIndex, EmptyIndexView))) return false;
        if (!SurfaceConstants)
        {
            D3D11_BUFFER_DESC desc {}; desc.ByteWidth = sizeof(LightListSurfaceConstants);
            desc.Usage = D3D11_USAGE_DYNAMIC; desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (!checked(device->CreateBuffer(&desc, nullptr, SurfaceConstants.GetAddressOf()), "material constants")) return false;
        }
        if (UploadedLights != lights && !lights->GetShaderLights().empty())
        {
            D3D11_BOX region {}; region.right = UINT(lights->GetShaderLights().size() * sizeof(ShaderLight));
            region.bottom = region.back = 1;
            context->UpdateSubresource(SceneBuffer.Get(), 0, &region, lights->GetShaderLights().data(), 0, 0);
        }
        UploadedLights = lights;
        return true;
    }

    bool DX11ForwardPlus::PrepareMaterial(ID3D11Device* device, Techniques::ShadowFilterMode filter,
        SurfaceProgram program, std::string& error)
    {
        error.clear();
        if (!device || (Owner && Owner.Get() != device) || static_cast<std::size_t>(filter) >= ShadowFilterCount || program >= SurfaceProgramCount)
        { error = "DX11 Forward+ material program or device is invalid"; return false; }
        Owner = device;
        auto& pixel = SurfacePixels[static_cast<std::size_t>(filter)][program];
        if (SurfaceVertex && pixel) return true;
        std::string source;
        if (!LoadTechniqueShaderSource("DX11", Shader == TiledLightShader::TiledDeferred ? "TiledDeferred.hlsl" : "ForwardPlus.hlsl", source, error)) return false;
        const auto filterName = std::to_string(static_cast<std::uint32_t>(filter));
        const D3D_SHADER_MACRO defines[] {{Shader == TiledLightShader::TiledDeferred ? "EVANS_TILED_DEFERRED_SURFACE" : "EVANS_FORWARD_PLUS_SURFACE", "1"}, {Shader == TiledLightShader::TiledDeferred ? "EVANS_TILED_DEFERRED_FILTER" : "EVANS_FORWARD_PLUS_FILTER", filterName.c_str()}, {nullptr, nullptr}};
        const auto compile = [&](const char* entry, const char* profile, ComPtr<ID3DBlob>& binary)
        {
            ComPtr<ID3DBlob> diagnostics;
            const auto result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), Shader == TiledLightShader::TiledDeferred ? "DX11/TiledDeferred.hlsl" : "DX11/ForwardPlus.hlsl", defines, nullptr,
                entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS |
                D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS, 0,
                binary.GetAddressOf(), diagnostics.GetAddressOf());
            if (SUCCEEDED(result)) return true;
            error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "DX11 Forward+ material compilation failed";
            return false;
        };
        if (!SurfaceVertex)
        {
            ComPtr<ID3DBlob> binary;
            if (!compile("VSMain", "vs_5_0", binary)) return false;
            if (FAILED(device->CreateVertexShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr, SurfaceVertex.GetAddressOf())))
            { error = "DX11 Forward+ material vertex creation failed"; return false; }
        }
        constexpr const char* entries[] {"PSMain", "PSContact", "PSWeighted", "PSDepth"};
        ComPtr<ID3DBlob> binary;
        if (!compile(entries[program], "ps_5_0", binary)) return false;
        if (FAILED(device->CreatePixelShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr, pixel.GetAddressOf())))
        { error = "DX11 Forward+ material pixel creation failed"; return false; }
        return true;
    }

    bool DX11ForwardPlus::PrepareDeferred(ID3D11Device* device, Techniques::ShadowFilterMode filter,
        bool multisample, std::string& error)
    {
        error.clear();
        if (Shader != TiledLightShader::TiledDeferred || !device || (Owner && Owner.Get() != device) ||
            static_cast<std::size_t>(filter) >= ShadowFilterCount)
        { error = "DX11 tiled lighting requires its own source and a valid device"; return false; }
        Owner = device;
        auto& pixel = LightingPixels[static_cast<std::size_t>(filter)][multisample ? 1 : 0];
        if (LightingVertex && pixel) return true;
        std::string source;
        if (!LoadTechniqueShaderSource("DX11", "TiledDeferred.hlsl", source, error)) return false;
        const auto filterName = std::to_string(static_cast<std::uint32_t>(filter));
        const D3D_SHADER_MACRO defines[] {{"EVANS_TILED_DEFERRED_SURFACE", "1"}, {"EVANS_TILED_DEFERRED_LIGHTING", "1"},
            {"EVANS_TILED_DEFERRED_FILTER", filterName.c_str()}, {multisample ? "EVANS_MULTISAMPLE" : nullptr, "1"}, {nullptr, nullptr}};
        const auto compile = [&](const char* entry, const char* profile, ComPtr<ID3DBlob>& binary)
        {
            ComPtr<ID3DBlob> diagnostics;
            const auto result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX11/TiledDeferred.hlsl", defines, nullptr,
                entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS |
                D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS, 0,
                binary.GetAddressOf(), diagnostics.GetAddressOf());
            if (SUCCEEDED(result)) return true;
            error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "DX11 tiled lighting compilation failed";
            return false;
        };
        if (!LightingVertex)
        {
            ComPtr<ID3DBlob> binary;
            if (!compile("VSLighting", "vs_5_0", binary)) return false;
            if (FAILED(device->CreateVertexShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr, LightingVertex.GetAddressOf())))
            { error = "DX11 tiled lighting vertex creation failed"; return false; }
        }
        ComPtr<ID3DBlob> binary;
        if (!compile("PSLighting", "ps_5_0", binary)) return false;
        if (FAILED(device->CreatePixelShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr, pixel.GetAddressOf())))
        { error = "DX11 tiled lighting pixel creation failed"; return false; }
        return true;
    }

    bool DX11ForwardPlus::BindSurface(ID3D11DeviceContext* context,
        const LightListSurfaceConstants& constants, std::string& error)
    {
        error.clear();
        const bool tiled = constants.LightLayout.Z != static_cast<std::uint32_t>(LightListSelection::FullScene);
        if (!context || !UploadedLights || !SurfaceConstants || constants.LightLayout.X != UploadedLights->GetShaderLights().size() ||
            (tiled && !Ready))
        { error = "DX11 Forward+ surface inputs are not ready"; return false; }
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(context->Map(SurfaceConstants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        { error = "DX11 Forward+ material constants map failed"; return false; }
        std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(SurfaceConstants.Get(), 0);
        ID3D11Buffer* control = SurfaceConstants.Get();
        context->PSSetConstantBuffers(MaterialConstantsSlot, 1, &control);
        ID3D11ShaderResourceView* views[] {SceneView.Get(), tiled ? HeaderView.Get() : EmptyHeaderView.Get(),
            tiled ? IndexView.Get() : EmptyIndexView.Get()};
        context->PSSetShaderResources(MaterialLightSlot, 3, views);
        return true;
    }

    bool DX11ForwardPlus::Prepare(ID3D11Device* device, const ForwardPlusGrid& grid, std::string& error)
    {
        Owner = device;
        const auto checked = [&](HRESULT code, const char* operation)
        {
            if (SUCCEEDED(code)) return true;
            error = std::string("DX11 Forward+ failed to create ") + operation + ": " + std::to_string(code); return false;
        };
        auto& program = Programs[grid.Samples > 1 ? 1 : 0];
        if (!program)
        {
            std::string source;
            if (!LoadTechniqueShaderSource("DX11", Shader == TiledLightShader::TiledDeferred ? "TiledDeferred.hlsl" : "ForwardPlus.hlsl", source, error)) return false;
            const D3D_SHADER_MACRO defines[] {{Shader == TiledLightShader::TiledDeferred ? "EVANS_TILED_DEFERRED_MULTISAMPLE" : "EVANS_FORWARD_PLUS_MULTISAMPLE", grid.Samples > 1 ? "1" : "0"}, {nullptr, nullptr}};
            ComPtr<ID3DBlob> binary, diagnostics;
            const auto compiled = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), Shader == TiledLightShader::TiledDeferred ? "DX11/TiledDeferred.hlsl" : "DX11/ForwardPlus.hlsl", defines, nullptr,
                "CSAssign", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS |
                D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS, 0,
                binary.GetAddressOf(), diagnostics.GetAddressOf());
            if (FAILED(compiled))
            { error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "DX11 Forward+ compilation failed"; return false; }
            if (!checked(device->CreateComputeShader(binary->GetBufferPointer(), binary->GetBufferSize(), nullptr,
                    program.GetAddressOf()), "list assignment shader")) return false;
        }
        if (!Constants)
        {
            D3D11_BUFFER_DESC desc {}; desc.ByteWidth = sizeof(ForwardPlusConstants);
            desc.Usage = D3D11_USAGE_DYNAMIC; desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (!checked(device->CreateBuffer(&desc, nullptr, Constants.GetAddressOf()), "constant buffer")) return false;
        }
        if (!BoundsView)
        {
            ComPtr<ID3D11Buffer> buffer; ComPtr<ID3D11ShaderResourceView> view;
            D3D11_BUFFER_DESC desc {}; desc.ByteWidth = MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds);
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            desc.StructureByteStride = sizeof(ForwardPlusGpuBounds);
            if (!checked(device->CreateBuffer(&desc, nullptr, buffer.GetAddressOf()), "light bounds") ||
                !checked(device->CreateShaderResourceView(buffer.Get(), nullptr, view.GetAddressOf()), "bounds view")) return false;
            BoundsBuffer = std::move(buffer); BoundsView = std::move(view);
        }
        if (AllocatedTiles != grid.TileCount || AllocatedIndices != grid.IndexCount)
        {
            ComPtr<ID3D11Buffer> headers, indices;
            ComPtr<ID3D11ShaderResourceView> headerView, indexView;
            ComPtr<ID3D11UnorderedAccessView> headerTarget, indexTarget;
            const auto allocate = [&](UINT count, UINT stride, ComPtr<ID3D11Buffer>& buffer,
                ComPtr<ID3D11ShaderResourceView>& view, ComPtr<ID3D11UnorderedAccessView>& target)
            {
                D3D11_BUFFER_DESC desc {}; desc.ByteWidth = count * stride;
                desc.StructureByteStride = stride; desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
                return checked(device->CreateBuffer(&desc, nullptr, buffer.GetAddressOf()), "tile storage") &&
                    checked(device->CreateShaderResourceView(buffer.Get(), nullptr, view.GetAddressOf()), "tile SRV") &&
                    checked(device->CreateUnorderedAccessView(buffer.Get(), nullptr, target.GetAddressOf()), "tile UAV");
            };
            if (!allocate(grid.TileCount, sizeof(ForwardPlusTile), headers, headerView, headerTarget) ||
                !allocate(grid.IndexCount, sizeof(std::uint32_t), indices, indexView, indexTarget)) return false;
            HeaderBuffer = std::move(headers); IndexBuffer = std::move(indices);
            HeaderView = std::move(headerView); IndexView = std::move(indexView);
            HeaderTarget = std::move(headerTarget); IndexTarget = std::move(indexTarget);
            AllocatedTiles = grid.TileCount; AllocatedIndices = grid.IndexCount;
        }
        return true;
    }
    bool DX11ForwardPlus::Generate(ID3D11Device* device, ID3D11DeviceContext* context,
        ID3D11ShaderResourceView* depth, const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights,
        const ForwardPlusParameters& parameters, std::string& error)
    {
        Ready = false; error.clear();
        const auto fail = [&](const char* message) { error = message; return false; };
        if (!device || !context || !depth || device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0 ||
            context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
            return fail("DX11 Forward+ requires FL11, an immediate context and sampled camera depth");
        ComPtr<ID3D11Device> contextDevice, depthDevice;
        context->GetDevice(contextDevice.GetAddressOf()); depth->GetDevice(depthDevice.GetAddressOf());
        if (contextDevice.Get() != device || depthDevice.Get() != device || (Owner && Owner.Get() != device))
            return fail("DX11 Forward+ resources must belong to the same device; Shutdown before changing devices");
        ComPtr<ID3D11Resource> resource; depth->GetResource(resource.GetAddressOf());
        ComPtr<ID3D11Texture2D> image;
        if (FAILED(resource.As(&image))) return fail("DX11 Forward+ depth must be a 2D image");
        D3D11_TEXTURE2D_DESC desc {}; image->GetDesc(&desc);
        D3D11_SHADER_RESOURCE_VIEW_DESC view {}; depth->GetDesc(&view);
        const bool multisample = desc.SampleDesc.Count > 1;
        if (desc.ArraySize != 1 || desc.Format != DXGI_FORMAT_R32_TYPELESS || !(desc.BindFlags & D3D11_BIND_DEPTH_STENCIL) ||
            view.Format != DXGI_FORMAT_R32_FLOAT ||
            (!multisample && (view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || view.Texture2D.MostDetailedMip != 0)) ||
            (multisample && view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2DMS))
            return fail("DX11 Forward+ requires a base D32 depth view, including original MSAA samples");
        ForwardPlusGrid grid; ForwardPlusConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildForwardPlusGrid(desc.Width, desc.Height, desc.SampleDesc.Count, lights.size(), parameters, grid, error) ||
            !BuildForwardPlusConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error) ||
            !Prepare(device, grid, error)) return false;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(context->Map(Constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) return fail("DX11 Forward+ constant upload failed");
        std::memcpy(mapped.pData, &constants, sizeof(constants)); context->Unmap(Constants.Get(), 0);
        if (!bounds.empty())
        {
            const D3D11_BOX region {0, 0, 0, UINT(bounds.size() * sizeof(ForwardPlusGpuBounds)), 1, 1};
            context->UpdateSubresource(BoundsBuffer.Get(), 0, &region, bounds.data(), 0, 0);
        }
        ID3D11Buffer* uniform = Constants.Get();
        ID3D11ShaderResourceView* inputs[] {depth, BoundsView.Get()};
        ID3D11UnorderedAccessView* outputs[] {HeaderTarget.Get(), IndexTarget.Get()};
        context->CSSetShader(Programs[multisample ? 1 : 0].Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, &uniform); context->CSSetShaderResources(0, 2, inputs);
        context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
        context->Dispatch(grid.TilesX, grid.TilesY, 1);
        ID3D11ShaderResourceView* noInputs[2] {}; ID3D11UnorderedAccessView* noOutputs[2] {}; uniform = nullptr;
        context->CSSetUnorderedAccessViews(0, 2, noOutputs, nullptr); context->CSSetShaderResources(0, 2, noInputs);
        context->CSSetConstantBuffers(0, 1, &uniform); context->CSSetShader(nullptr, nullptr, 0);
        CurrentGrid = grid; Ready = true;
        return true;
    }
}
#endif
