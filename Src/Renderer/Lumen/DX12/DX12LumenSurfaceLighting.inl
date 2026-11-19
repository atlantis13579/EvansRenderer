struct LumenLightingRecord
{
    std::vector<unsigned> CardInstances; LumenLightingInfo Info; ComPtr<ID3D12Resource> Buffer;
    std::shared_ptr<LumenSurfaceRecord> Surface; std::shared_ptr<LumenDistanceRecord> Distance;
};
struct LumenLightingWork
{
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands; ComPtr<ID3D12DescriptorHeap> Views;
    ComPtr<ID3D12Resource> Input,Uniform,Readback; std::shared_ptr<LumenLightingRecord> Record,Previous;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenLightingRecord>> LumenLightingCaches;
std::shared_ptr<LumenLightingWork> LumenLightingPending;
ComPtr<ID3D12RootSignature> LumenLightingRoot; ComPtr<ID3D12PipelineState> LumenSurfaceLighting,LumenSurfaceRadiosity;
void DestroyLumenLightingCaches()
{ DestroyLumenLightingAsync(); LumenLightingPending.reset(); LumenLightingCaches.clear(); LumenSurfaceLighting.Reset(); LumenSurfaceRadiosity.Reset(); LumenLightingRoot.Reset(); }
bool EnsureLumenSurfaceLighting(bool indirect=false,bool hardware=false)
{
    if (!LumenLightingRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=LumenSurfacePlanes;
        // A root CBV keeps the expanded transport frame below the 64-DWORD root limit.
        D3D12_ROOT_PARAMETER parameters[8]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        parameters[2].ParameterType=parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[2].Descriptor.ShaderRegister=3; parameters[3].Descriptor.ShaderRegister=4; parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[5].Descriptor.ShaderRegister=5;
        LumenHardwareRoots(parameters+6);
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=8; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen lighting root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenLightingRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen lighting root",status);
    }
    if (hardware) return EnsureLumenHardwarePipeline(indirect?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct,LumenLightingRoot.Get());
    auto& pipeline=indirect?LumenSurfaceRadiosity:LumenSurfaceLighting;
    if (pipeline) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_LIGHTING_STAGE","1"},{indirect?"EVANS_LUMEN_RADIOSITY":nullptr,"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSLightSurface","cs_5_0",binary,macros,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenLightingRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(pipeline.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen lighting pipeline",status);
}
bool SelectLumenPreviousLighting(const LumenSurfaceLightingUpload& upload,const std::shared_ptr<LumenSurfaceRecord>& surface,
    const std::shared_ptr<LumenDistanceRecord>& distance,std::shared_ptr<LumenLightingRecord>& output)
{
    if (!upload.PreviousLighting) return true;
    const auto found=LumenLightingCaches.find(upload.PreviousLighting.Value);
    if (found==LumenLightingCaches.end() || found->second->Surface!=surface || found->second->Distance!=distance || found->second->CardInstances!=upload.CardInstances)
        return Fail("Lumen radiosity source is expired or belongs to different native inputs");
    if (!ValidateLumenRadiositySource(surface->Identity,distance->Info.Identity,upload,found->second->Info,LastError)) return false;
    output=found->second; return true;
}
bool UploadLumenLightingFrame(LumenLightingWork& work,const LumenSurfaceLightingFrame& frame)
{
    constexpr auto alignment=D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    constexpr auto bytes=(sizeof(frame)+alignment-1)/alignment*alignment;
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work.Uniform)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto status=work.Uniform->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Upload Lumen lighting frame",status);
    std::memcpy(mapped,&frame,sizeof(frame)); work.Uniform->Unmap(0,nullptr); return true;
}
bool BeginLumenLightingWork(LumenLightingWork& work)
{
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen lighting commands",status);
}
bool SubmitLumenLightingWork(const std::shared_ptr<LumenLightingWork>& work)
{
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen lighting commands",status);
    LumenLightingPending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false; LumenLightingPending.reset(); return true;
}
void LumenLightingTransition(LumenLightingWork& work,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={work.Record->Buffer.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; work.Commands->ResourceBarrier(1,&barrier);
}
bool ReadLumenLightingRecord(const std::shared_ptr<LumenLightingRecord>& record,std::vector<LumenLightingTexel>& output)
{
    if (!RecoverSubmittedWork("Read Lumen lighting")) return false; LumenLightingPending.reset();
    auto work=std::make_shared<LumenLightingWork>(); work->Record=record;
    const auto count=std::size_t(record->Info.Layout.Width)*record->Info.Layout.Height,bytes=count*sizeof(LumenLightingTexel);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback) || !BeginLumenLightingWork(*work)) return false;
    LumenLightingTransition(*work,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    work->Commands->CopyBufferRegion(work->Readback.Get(),0,record->Buffer.Get(),0,bytes);
    LumenLightingTransition(*work,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenLightingWork(work)) return false;
    void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; const auto status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Read Lumen lighting staging",status);
    std::vector<LumenLightingTexel> result(count); std::memcpy(result.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenLightingTexels(record->Info,result,LastError)) return false; output=std::move(result); return true;
}
bool RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    std::shared_ptr<LumenLightingRecord> previous;
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,previous)) return false;
    LumenSurfaceLightingInput input;
    if (!BuildLumenSurfaceLightingInput(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,input,LastError) ||
        !RecoverSubmittedWork("Relight Lumen surface") || !EnsureLumenSurfaceLighting(bool(previous),bool(occluders->second->Hardware))) return false;
    LumenLightingPending.reset(); auto work=std::make_shared<LumenLightingWork>(); auto record=std::make_shared<LumenLightingRecord>(); work->Record=record; work->Previous=previous;
    record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=previous->Info.Bounce+1; }
    const auto inputBytes=input.Words.size()*sizeof(std::uint32_t);
    if (!CreateBufferResource(inputBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Input)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; auto status=work->Input->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Upload Lumen lighting inputs",status);
    std::memcpy(mapped,input.Words.data(),inputBytes); work->Input->Unmap(0,nullptr);
    const auto count=std::size_t(record->Info.Layout.Width)*record->Info.Layout.Height; auto desc=BufferResourceDesc(count*sizeof(LumenLightingTexel));
    desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; const auto heapProperties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    status=Device->CreateCommittedResource(&heapProperties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(record->Buffer.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create resident Lumen radiance",status);
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=LumenSurfacePlanes; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen lighting views",status);
    auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart(); const auto increment=Device->GetDescriptorHandleIncrementSize(heap.Type);
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(record->Surface->Images[plane].Get(),&view,cpu); cpu.ptr+=increment;
    }
    if (!UploadLumenLightingFrame(*work,input.Frame) || !BeginLumenLightingWork(*work)) return false;
    work->Commands->SetPipelineState((record->Distance->Hardware?LumenHardwarePipelines[unsigned(work->Previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:work->Previous?LumenSurfaceRadiosity:LumenSurfaceLighting).Get()); work->Commands->SetComputeRootSignature(LumenLightingRoot.Get());
    ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; work->Commands->SetDescriptorHeaps(1,heaps);
    work->Commands->SetComputeRootConstantBufferView(0,work->Uniform->GetGPUVirtualAddress());
    work->Commands->SetComputeRootDescriptorTable(1,work->Views->GetGPUDescriptorHandleForHeapStart());
    work->Commands->SetComputeRootShaderResourceView(2,record->Distance->Scene->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(3,work->Input->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(4,record->Buffer->GetGPUVirtualAddress());
    if (work->Previous) work->Commands->SetComputeRootShaderResourceView(5,work->Previous->Buffer->GetGPUVirtualAddress());
    BindLumenHardware(work->Commands.Get(),*record->Distance,6);
    const auto groups=LumenLightingGroups(input.Frame);
    work->Commands->Dispatch(std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1);
    LumenLightingTransition(*work,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenLightingWork(work)) return false;
    LumenSurfaceLightingResult result; result.Info=record->Info;
    if (upload.Readback && !ReadLumenLightingRecord(record,result.Texels)) return false;
    result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
    LumenLightingCaches.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    const auto found=LumenLightingCaches.find(handle.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen lighting cache");
    return ReadLumenLightingRecord(found->second,output);
}
bool DestroyLumenLighting(LumenLightingHandle handle)
{ return LumenLightingCaches.erase(handle.Value)!=0 || Fail("Unknown Lumen lighting cache"); }
