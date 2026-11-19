// Immutable scene radiance storage is shared by diffuse GI and specular
// reflections. Each effect retains independent shaders, constants and targets.
RayRadianceFrame FrameRayRadiance;
Techniques::Settings FrameRayRadianceSettings;
std::shared_ptr<const RaySceneShadingUpload> UploadedRayShading;
std::shared_ptr<const SceneLightCollection> UploadedRayLights;
std::array<ComPtr<ID3D11Buffer>,RayRadianceBufferCount> RayRadianceBuffers;
std::array<ComPtr<ID3D11ShaderResourceView>,RayRadianceBufferCount> RayRadianceViews;

void DestroyRayRadiance()
{
    RayRadianceViews={}; RayRadianceBuffers={};
    UploadedRayShading.reset(); UploadedRayLights.reset(); FrameRayRadiance={};
}

bool EnsureRayRadiance()
{
    const auto& frame=FrameRayRadiance;
    if (!frame.Geometry || frame.Geometry!=FrameRayScene || !frame.Shading || !frame.Lights)
        return Fail("DX11 ray radiance requires matched scene/material/light snapshots");
    if (!EnsureRayScene()) return false;
    const bool materials=frame.Shading!=UploadedRayShading,lights=frame.Lights!=UploadedRayLights;
    if (!materials && !lights) return true;
    if (materials && !ValidateRaySceneShadingUpload(*frame.Geometry,*frame.Shading,LastError)) return false;
    const auto data=GetRayRadianceBufferData(frame);
    auto buffers=RayRadianceBuffers; auto views=RayRadianceViews;
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (!(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        if (data[slot].Bytes>UINT_MAX || data[slot].Bytes/data[slot].Stride>
            (std::size_t{1}<<D3D11_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP))
            return Fail("DX11 ray shading exceeds structured storage limits");
        D3D11_BUFFER_DESC create{}; create.ByteWidth=static_cast<UINT>(data[slot].Bytes);
        create.StructureByteStride=data[slot].Stride; create.Usage=D3D11_USAGE_IMMUTABLE;
        create.BindFlags=D3D11_BIND_SHADER_RESOURCE; create.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data[slot].Data;
        buffers[slot].Reset(); views[slot].Reset();
        auto result=Device->CreateBuffer(&create,&initial,buffers[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(ray shading)",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements=static_cast<UINT>(data[slot].Bytes/data[slot].Stride);
        result=Device->CreateShaderResourceView(buffers[slot].Get(),&view,views[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(ray shading)",result));
    }
    RayRadianceBuffers=std::move(buffers); RayRadianceViews=std::move(views);
    UploadedRayShading=frame.Shading; UploadedRayLights=frame.Lights; return true;
}
