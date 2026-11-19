// Immutable scene radiance storage is shared by diffuse GI and specular
// reflections. Each effect retains independent shaders, constants and targets.
RayRadianceFrame FrameRayRadiance;
Techniques::Settings FrameRayRadianceSettings;
std::shared_ptr<const RaySceneShadingUpload> UploadedRayShading;
std::shared_ptr<const SceneLightCollection> UploadedRayLights;
std::array<ComPtr<ID3D12Resource>,RayRadianceBufferCount> RayRadianceBuffers;
std::array<UINT,RayRadianceBufferCount> RayRadianceViews {InvalidDescriptor,InvalidDescriptor,InvalidDescriptor,InvalidDescriptor};

void DestroyRayRadiance()
{
    for (auto& view:RayRadianceViews) { SRVArena.Release(view); view=InvalidDescriptor; }
    RayRadianceBuffers={};
    UploadedRayShading.reset(); UploadedRayLights.reset(); FrameRayRadiance={};
}

bool EnsureRayRadiance()
{
    const auto& frame=FrameRayRadiance;
    if (!frame.Geometry || frame.Geometry!=FrameRayScene || !frame.Shading || !frame.Lights)
        return Fail("DX12 ray radiance requires matched scene/material/light snapshots");
    if (!EnsureRayScene()) return false;
    const bool materials=frame.Shading!=UploadedRayShading,lights=frame.Lights!=UploadedRayLights;
    if (!materials && !lights) return true;
    if (materials && !ValidateRaySceneShadingUpload(*frame.Geometry,*frame.Shading,LastError)) return false;
    const auto data=GetRayRadianceBufferData(frame); auto buffers=RayRadianceBuffers;
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (!(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        if (data[slot].Bytes/data[slot].Stride>UINT_MAX) return Fail("DX12 ray shading exceeds structured-view element limits");
        if (RayRadianceViews[slot]==InvalidDescriptor) RayRadianceViews[slot]=SRVArena.Allocate();
        if (RayRadianceViews[slot]==InvalidDescriptor) return Fail("DX12 ray reflection descriptor arena exhausted");
        buffers[slot].Reset();
        if (!CreateBufferResource(data[slot].Bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,buffers[slot])) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead {0,0};
        const auto result=buffers[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map(ray reflection storage)",result);
        std::memcpy(mapped,data[slot].Data,data[slot].Bytes);
        const D3D12_RANGE written {0,data[slot].Bytes}; buffers[slot]->Unmap(0,&written);
    }
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (!(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.NumElements=static_cast<UINT>(data[slot].Bytes/data[slot].Stride); view.Buffer.StructureByteStride=data[slot].Stride;
        Device->CreateShaderResourceView(buffers[slot].Get(),&view,SRVArena.Cpu(RayRadianceViews[slot]));
    }
    RayRadianceBuffers=std::move(buffers); UploadedRayShading=frame.Shading; UploadedRayLights=frame.Lights; return true;
}
