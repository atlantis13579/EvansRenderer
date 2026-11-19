// Immutable structured storage for native shader BVH traversal. Publish all
// four buffers together, so allocation failure retains the previous snapshot.
std::shared_ptr<const RaySceneUpload> FrameRayScene, UploadedRayScene;
std::array<ComPtr<ID3D11Buffer>,RaySceneBufferCount> RaySceneBuffers;
std::array<ComPtr<ID3D11ShaderResourceView>,RaySceneBufferCount> RaySceneViews;

void DestroyRayScene()
{
    DestroyRayRadiance();
    RaySceneViews={}; RaySceneBuffers={}; FrameRayScene.reset(); UploadedRayScene.reset();
}

bool EnsureRayScene()
{
    if (!FrameRayScene) return Fail("DX11 ray AO requires a scene snapshot");
    if (FrameRayScene==UploadedRayScene) return true;
    if (!ValidateRaySceneUpload(*FrameRayScene,LastError)) return false;
    const auto data=GetRaySceneBufferData(*FrameRayScene);
    std::array<ComPtr<ID3D11Buffer>,RaySceneBufferCount> buffers;
    std::array<ComPtr<ID3D11ShaderResourceView>,RaySceneBufferCount> views;
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (data[slot].Bytes>UINT_MAX || data[slot].Bytes/data[slot].Stride>
            (std::size_t{1}<<D3D11_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP))
            return Fail("DX11 ray-scene storage exceeds structured-buffer limits");
        D3D11_BUFFER_DESC buffer{};
        buffer.ByteWidth=static_cast<UINT>(data[slot].Bytes); buffer.StructureByteStride=data[slot].Stride;
        buffer.Usage=D3D11_USAGE_IMMUTABLE; buffer.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data[slot].Data;
        auto result=Device->CreateBuffer(&buffer,&initial,buffers[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(ray scene)",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{};
        view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements=static_cast<UINT>(data[slot].Bytes/data[slot].Stride);
        result=Device->CreateShaderResourceView(buffers[slot].Get(),&view,views[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(ray scene)",result));
    }
    RaySceneBuffers=std::move(buffers); RaySceneViews=std::move(views);
    UploadedRayScene=FrameRayScene; return true;
}
