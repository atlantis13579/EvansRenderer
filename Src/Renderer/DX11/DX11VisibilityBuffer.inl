// D3D11 uses an immutable structured triangle table. ID/barycentric RTVs are
// unbound before SRV reads; material reconstruction keeps the original depth.
std::array<HdrSurface, 2> VisibilityImages;
ComPtr<ID3D11Buffer> VisibilityGeometry, VisibilityUniforms;
ComPtr<ID3D11ShaderResourceView> VisibilityGeometryView;
std::array<ComPtr<ID3D11VertexShader>, 2> VisibilityVertices;
std::array<ComPtr<ID3D11PixelShader>, 2> VisibilityPixels;
std::shared_ptr<const VisibilityScene> UploadedVisibility;

void DestroyVisibilityBuffer()
{
    DestroyNaniteMaterials();
    DestroyNaniteFrameRaster();
    DestroyNaniteVisibilityGeometry();
    VisibilityImages = {}; VisibilityGeometry.Reset(); VisibilityGeometryView.Reset(); VisibilityUniforms.Reset();
    VisibilityVertices = {}; VisibilityPixels = {}; UploadedVisibility.reset();
}
bool SupportsVisibilityBuffer() const
{
    if (!Device) return false;
    for (auto format : {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT})
    {
        UINT support = 0;
        const UINT required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_RENDER_TARGET | D3D11_FORMAT_SUPPORT_SHADER_LOAD;
        if (FAILED(Device->CheckFormatSupport(format, &support)) || (support & required) != required) return false;
    }
    return true;
}
bool PrepareVisibilityBuffer(const std::shared_ptr<const VisibilityScene>& scene)
{
    if (!scene || !SupportsVisibilityBuffer()) return Fail("DX11 visibility buffer requires sampled integer/float attachments");
    if (scene->HasNativeGeometry())
    {
        if (!PrepareNaniteVisibilityGeometry(*scene)) return false;
    }
    else if (UploadedVisibility != scene)
    {
        if (!ValidateVisibilityScene(*scene, LastError)) return false;
        const auto count = std::max(scene->Triangles.size(), std::size_t{1});
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = static_cast<UINT>(count * sizeof(VisibilityTriangle)); desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(VisibilityTriangle);
        const VisibilityTriangle empty{};
        D3D11_SUBRESOURCE_DATA input {scene->Triangles.empty() ? &empty : scene->Triangles.data()};
        ComPtr<ID3D11Buffer> buffer; ComPtr<ID3D11ShaderResourceView> view;
        auto result = Device->CreateBuffer(&desc, &input, buffer.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(visibility triangles)", result));
        D3D11_SHADER_RESOURCE_VIEW_DESC resource {};
        resource.ViewDimension = D3D11_SRV_DIMENSION_BUFFER; resource.Buffer.NumElements = static_cast<UINT>(count);
        result = Device->CreateShaderResourceView(buffer.Get(), &resource, view.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(visibility triangles)", result));
        VisibilityGeometry = std::move(buffer); VisibilityGeometryView = std::move(view); UploadedVisibility = scene;
    }
    const DXGI_FORMAT formats[] {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT};
    for (std::size_t index = 0; index < VisibilityImages.size(); ++index)
    {
        auto& image = VisibilityImages[index];
        if (image.Texture && image.Width == CreateInfo.Width && image.Height == CreateInfo.Height) continue;
        HdrSurface candidate;
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = CreateInfo.Width; desc.Height = CreateInfo.Height; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = formats[index]; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        auto result = Device->CreateTexture2D(&desc, nullptr, candidate.Texture.GetAddressOf());
        if (SUCCEEDED(result)) result = Device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf());
        if (SUCCEEDED(result)) result = Device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(visibility payload)", result));
        candidate.Width = CreateInfo.Width; candidate.Height = CreateInfo.Height; image = std::move(candidate);
    }
    if (!VisibilityUniforms)
    {
        D3D11_BUFFER_DESC desc {}; desc.ByteWidth = sizeof(VisibilityConstants); desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        const auto result = Device->CreateBuffer(&desc, nullptr, VisibilityUniforms.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(visibility constants)", result));
    }
    if (!VisibilityVertices[0] || !VisibilityVertices[1] || !VisibilityPixels[0] || !VisibilityPixels[1])
    {
        if (!LoadTechniqueShaderSource("DX11", "VisibilityBuffer.hlsl", ShaderCode, LastError)) return false;
        const char* vertexEntries[] {"VSRaster", "VSResolve"}; const char* pixelEntries[] {"PSRaster", "PSResolve"};
        for (std::size_t phase = 0; phase < 2; ++phase)
        {
            ComPtr<ID3DBlob> vertex, pixel;
            if (!CompileShader(vertexEntries[phase], "vs_5_0", vertex, nullptr, true) ||
                !CompileShader(pixelEntries[phase], "ps_5_0", pixel, nullptr, true)) return false;
            auto result = Device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, VisibilityVertices[phase].ReleaseAndGetAddressOf());
            if (SUCCEEDED(result)) result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, VisibilityPixels[phase].ReleaseAndGetAddressOf());
            if (FAILED(result)) return Fail(HResultMessage("CreateShader(visibility stages)", result));
        }
    }
    return true;
}
bool DrawVisibilityBuffer(const VisibilityBufferDesc& frame)
{
    ID3D11ShaderResourceView* empty[8]{};
    Context->PSSetShaderResources(0, 8, empty); Context->VSSetShaderResources(0, 8, empty);
    ID3D11RenderTargetView* targets[] {VisibilityImages[0].Target.Get(), VisibilityImages[1].Target.Get()};
    const float clear[4]{};
    for (auto* target : targets) Context->ClearRenderTargetView(target, clear);
    Context->OMSetRenderTargets(2, targets, DepthTarget.Get());
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
    ID3D11Buffer* uniform = VisibilityUniforms.Get();
    Context->VSSetConstantBuffers(0, 1, &uniform); Context->PSSetConstantBuffers(0, 1, &uniform);
    ID3D11ShaderResourceView* geometry = VisibilityGeometryView.Get(); Context->VSSetShaderResources(0, 1, &geometry);
    const auto bind = [&](const VisibilityMaterialDraw& draw, bool resolve)
    {
        ID3D11ShaderResourceView* inputs[] {geometry, resolve ? VisibilityImages[0].View.Get() : nullptr,
            resolve ? VisibilityImages[1].View.Get() : nullptr, nullptr, nullptr, nullptr,
            MaterialPlan.Active()?MaterialMap->View.Get():nullptr,MaterialPlan.Active()?MaterialWork->View.Get():nullptr};
        const TextureHandle handles[] {draw.Pipeline.DiffuseTexture, draw.Pipeline.NormalTexture, draw.Pipeline.OpacityTexture};
        for (std::size_t index = 0; index < 3; ++index) if (handles[index])
        {
            const auto found = Textures.find(handles[index].Value);
            if (found == Textures.end() || !found->second.ShaderResourceView || found->second.Description.Format != TextureFormat::RGBA8)
                return Fail("DX11 visibility material texture is invalid");
            inputs[index + 3] = found->second.Description.ColorSpace == TextureColorSpace::SRGB ?
                found->second.SRGBShaderResourceView.Get() : found->second.ShaderResourceView.Get();
        }
        VisibilityConstants constants;
        if (!BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, CreateInfo.Width, CreateInfo.Height,
            frame.Scene->TriangleCount(), frame.Scene->Draws[draw.Range], inputs[3], inputs[4], inputs[5], constants))
            return Fail("DX11 visibility material constants are invalid");
        constants.MaterialTiles=MaterialPlan.Resolve(unsigned(&draw-frame.Draws.data()));
        D3D11_MAPPED_SUBRESOURCE mapped{};
        const auto result = Context->Map(uniform, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(result)) return Fail(HResultMessage("Map(visibility constants)", result));
        std::memcpy(mapped.pData, &constants, sizeof(constants)); Context->Unmap(uniform, 0);
        Context->VSSetConstantBuffers(0, 1, &uniform); Context->PSSetConstantBuffers(0, 1, &uniform);
        Context->PSSetShaderResources(0, 8, inputs);
        return true;
    };
    ID3D11SamplerState* sampler = MaterialSampler(); Context->PSSetSamplers(0, 1, &sampler);
    Context->VSSetShader(VisibilityVertices[0].Get(), nullptr, 0); Context->PSSetShader(VisibilityPixels[0].Get(), nullptr, 0);
    if (frame.NaniteRaster) { if (!DrawNaniteFrameRaster(frame)) return false; }
    else
    for (const auto& draw : frame.Draws)
    {
        if (!bind(draw, false)) return false;
        BoundPipeline = draw.Pipeline; BoundWorld = draw.World;
        auto* raster = GetOrientedRasterizer(); if (!raster) return false; Context->RSSetState(raster);
        Context->OMSetDepthStencilState(!draw.Pipeline.DepthTest ? DepthDisabledState.Get() :
            draw.Pipeline.DepthWrite ? DepthWriteState.Get() : DepthReadState.Get(), 0);
        Context->Draw(frame.Scene->Draws[draw.Range].TriangleCount * VisibilityTriangleCorners, 0);
    }
    Context->OMSetRenderTargets(0, nullptr, nullptr);
    if (!ClassifyNaniteMaterials()) return false;
    std::array<ID3D11RenderTargetView*, GBufferAttachmentCount> materials{};
    for (std::size_t index = 0; index < materials.size(); ++index) materials[index] = GBufferSurfaces[index].Target.Get();
    Context->OMSetRenderTargets(static_cast<UINT>(materials.size()), materials.data(), nullptr);
    Context->VSSetShader(VisibilityVertices[1].Get(), nullptr, 0); Context->PSSetShader(VisibilityPixels[1].Get(), nullptr, 0);
    Context->OMSetDepthStencilState(DepthDisabledState.Get(), 0); Context->RSSetState(SolidRasterState.Get());
    Context->VSSetShaderResources(0, 1, &geometry);
    if (MaterialPlan.Active())
    {
        ID3D11ShaderResourceView* work=MaterialWork->View.Get(); Context->VSSetShaderResources(7,1,&work);
        for (unsigned bin=0;bin<MaterialPlan.Representatives.size();++bin)
        {
            if (!bind(frame.Draws[MaterialPlan.Representatives[bin]],true)) return false;
            Context->DrawInstancedIndirect(MaterialWork->Buffer.Get(),bin*NaniteMaterialBinWords*sizeof(unsigned));
        }
    }
    else for (const auto& draw : frame.Draws) { if (!bind(draw, true)) return false; Context->Draw(3, 0); }
    Context->PSSetShaderResources(0, 8, empty); Context->VSSetShaderResources(0, 8, empty); Context->OMSetRenderTargets(0, nullptr, nullptr);
    return true;
}
