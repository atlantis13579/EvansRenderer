// Snapshot buffers use immutable upload-heap storage. They survive aborted
// command lists, and change only between the renderer's synchronized frames.
// A future device-local upload can replace this storage without changing the ABI.
std::shared_ptr<const RaySceneUpload> FrameRayScene, UploadedRayScene;
std::array<ComPtr<ID3D12Resource>,RaySceneBufferCount> RaySceneBuffers;
std::array<UINT,RaySceneBufferCount> RaySceneViews {InvalidDescriptor,InvalidDescriptor,InvalidDescriptor,InvalidDescriptor};

void DestroyRayScene()
{
    DestroyRayRadiance();
    for (auto& view:RaySceneViews) { SRVArena.Release(view); view=InvalidDescriptor; }
    RaySceneBuffers={}; FrameRayScene.reset(); UploadedRayScene.reset();
}

bool EnsureRayScene()
{
    if (!FrameRayScene) return Fail("DX12 ray AO requires a scene snapshot");
    if (FrameRayScene==UploadedRayScene) return true;
    if (!ValidateRaySceneUpload(*FrameRayScene,LastError)) return false;
    const auto data=GetRaySceneBufferData(*FrameRayScene);
    std::array<ComPtr<ID3D12Resource>,RaySceneBufferCount> buffers;
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (data[slot].Bytes/data[slot].Stride>UINT_MAX)
            return Fail("DX12 ray-scene storage exceeds structured-view element limits");
        if (RaySceneViews[slot]==InvalidDescriptor) RaySceneViews[slot]=SRVArena.Allocate();
        if (RaySceneViews[slot]==InvalidDescriptor) return Fail("DX12 ray-scene descriptor arena exhausted");
        if (!CreateBufferResource(data[slot].Bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,buffers[slot])) return false;
        void* mapped=nullptr;
        const D3D12_RANGE noRead {0,0};
        const auto result=buffers[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map(ray scene)",result);
        std::memcpy(mapped,data[slot].Data,data[slot].Bytes);
        const D3D12_RANGE written {0,data[slot].Bytes}; buffers[slot]->Unmap(0,&written);
    }
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.NumElements=static_cast<UINT>(data[slot].Bytes/data[slot].Stride);
        view.Buffer.StructureByteStride=data[slot].Stride;
        Device->CreateShaderResourceView(buffers[slot].Get(),&view,SRVArena.Cpu(RaySceneViews[slot]));
    }
    RaySceneBuffers=std::move(buffers); UploadedRayScene=FrameRayScene; return true;
}
