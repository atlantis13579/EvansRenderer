struct LumenRadianceRecord
{
    std::shared_ptr<LumenDistanceRecord> TraceDistance;
    std::shared_ptr<LumenRadianceRecord> WorldCache;
    LumenRadianceInfo Info; ComPtr<ID3D12Resource> Buffer,Origins; std::shared_ptr<LumenLightingRecord> Lighting;
};
struct LumenRadianceWork
{
    std::vector<std::shared_ptr<LumenRadianceRecord>> CopySources;
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands; ComPtr<ID3D12DescriptorHeap> Views;
    ComPtr<ID3D12Resource> Rays,Mapping,Hits,Readback,Pyramid,WorldUniform; std::shared_ptr<LumenRadianceRecord> Record;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenRadianceRecord>> LumenRadianceTraces;
LumenFrameTransaction FrameLumen;
std::shared_ptr<LumenRadianceWork> LumenRadiancePending;
std::vector<std::shared_ptr<LumenRadianceWork>> LumenFrameRadianceWork;
ComPtr<ID3D12RootSignature> LumenRadianceRoot,LumenProbeTraceRoot,LumenScreenRoot,LumenHybridRoot,LumenScreenDepthRoot; ComPtr<ID3D12PipelineState> LumenRadianceTrace,LumenRadianceResolve,LumenProbeTrace,LumenScreenPlacement,LumenHybridTrace,LumenScreenDepth,LumenReflectionRayShader;
ComPtr<ID3D12RootSignature> LumenWorldRoot; ComPtr<ID3D12PipelineState> LumenWorldLookup;
bool EnsureLumenWorldLookup()
{
    if (!LumenWorldRoot)
    {
        // A CBV keeps four cascade descriptors below the 64-DWORD root limit.
        D3D12_ROOT_PARAMETER parameters[5]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[4].Descriptor.ShaderRegister=1;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=5; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen world lookup root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenWorldRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen world lookup root",status);
    }
    if (LumenWorldLookup) return true;
    if (!LoadTechniqueShaderSource("DX12","LumenWorldCache.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> binary; if (!CompileShader("CSWorldLookup","cs_5_0",binary,nullptr,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenWorldRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenWorldLookup.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen world lookup pipeline",status);
}
void DestroyLumenRadianceTraces()
{ LumenWorldLookup.Reset(); LumenWorldRoot.Reset(); LumenRadiancePending.reset(); LumenFrameRadianceWork.clear(); LumenRadianceTraces.clear(); LumenRadianceTrace.Reset(); LumenRadianceResolve.Reset(); LumenProbeTrace.Reset(); LumenRadianceRoot.Reset(); LumenProbeTraceRoot.Reset(); LumenScreenPlacement.Reset(); LumenScreenRoot.Reset(); LumenHybridTrace.Reset(); LumenHybridRoot.Reset(); LumenScreenDepthRoot.Reset(); LumenScreenDepth.Reset(); LumenReflectionRayShader.Reset(); }
bool EnsureLumenRadianceTrace(bool hardware=false)
{
    if (!EnsureLumenDistanceTrace()) return false;
    if (!LumenScreenDepthRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=1;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.Num32BitValues=sizeof(LumenScreenDepthFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=3; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen depth root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenScreenDepthRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen depth root",status);
    }
    if (!LumenHybridRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=3;
        D3D12_ROOT_PARAMETER parameters[6]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants.Num32BitValues=sizeof(LumenScreenTraceFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=3;
        parameters[3].ParameterType=parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[4].Descriptor.ShaderRegister=1;
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[5].Descriptor.ShaderRegister=4;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen hybrid root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenHybridRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen hybrid root",status);
    }
    if (!LumenScreenRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=2;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.Num32BitValues=sizeof(LumenScreenProbeFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=3; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen screen placement root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenScreenRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen screen placement root",status);
    }
    if (!LumenProbeTraceRoot)
    {
        D3D12_ROOT_PARAMETER parameters[6]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants.Num32BitValues=sizeof(LumenProbeTraceFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        LumenHardwareRoots(parameters+4);
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen probe trace root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenProbeTraceRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen probe trace root",status);
    }
    if (!LumenRadianceRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=1;
        D3D12_ROOT_PARAMETER parameters[6]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.Num32BitValues=sizeof(LumenRadianceResolveFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        for (unsigned slot=2;slot<5;++slot) { parameters[slot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot].Descriptor.ShaderRegister=slot-1; }
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen radiance root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenRadianceRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen radiance root",status);
    }
    const auto nativeReady=[&]() { return !hardware || (EnsureLumenHardwarePipeline(LumenHardwareProgram::Radiance,LumenDistanceRoot.Get()) && EnsureLumenHardwarePipeline(LumenHardwareProgram::Probes,LumenProbeTraceRoot.Get())); };
    if (LumenRadianceTrace && LumenRadianceResolve && LumenProbeTrace && LumenScreenPlacement && LumenHybridTrace && LumenScreenDepth && LumenReflectionRayShader) return nativeReady();
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const char* stages[]{"EVANS_LUMEN_RADIANCE_TRACE_STAGE","EVANS_LUMEN_RADIANCE_RESOLVE_STAGE","EVANS_LUMEN_PROBE_TRACE_STAGE","EVANS_LUMEN_SCREEN_PROBES_STAGE","EVANS_LUMEN_SCREEN_TRACE_STAGE","EVANS_LUMEN_SCREEN_DEPTH_STAGE","EVANS_LUMEN_REFLECTION_RAYS_STAGE"}; const char* entries[]{"CSTraceDistance","CSResolveRadiance","CSTraceDistance","CSPlaceScreenProbes","CSTraceScreen","CSBuildScreenDepth","CSReflectionRays"};
    ID3D12RootSignature* roots[]{LumenDistanceRoot.Get(),LumenRadianceRoot.Get(),LumenProbeTraceRoot.Get(),LumenScreenRoot.Get(),LumenHybridRoot.Get(),LumenScreenDepthRoot.Get(),LumenScreenRoot.Get()}; ComPtr<ID3D12PipelineState>* targets[]{&LumenRadianceTrace,&LumenRadianceResolve,&LumenProbeTrace,&LumenScreenPlacement,&LumenHybridTrace,&LumenScreenDepth,&LumenReflectionRayShader};
    for (unsigned stage=0;stage<std::size(stages);++stage)
    {
        if (*targets[stage]) continue;
        const D3D_SHADER_MACRO macros[]{{stages[stage],"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
        if (!CompileShader(entries[stage],"cs_5_0",binary,macros,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=roots[stage]; desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(targets[stage]->GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen radiance pipeline",status);
    }
    return nativeReady();
}
bool BeginLumenRadianceWork(LumenRadianceWork& work)
{
    if (FrameOpen) { work.Commands=CommandList; return true; }
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen radiance commands",status);
}
bool SubmitLumenRadianceWork(const std::shared_ptr<LumenRadianceWork>& work)
{
    if (FrameOpen)
    {
        // Root SRVs do not retain resources. Keep uploads and the private view
        // heap alive until this frame's submission completes or is discarded.
        LumenFrameRadianceWork.push_back(work);
        ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
        return true;
    }
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen radiance commands",status);
    LumenRadiancePending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false;
    LumenRadiancePending.reset(); return true;
}
void LumenRadianceTransition(LumenRadianceWork& work,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; work.Commands->ResourceBarrier(1,&barrier);
}
bool CreateLumenRadianceUpload(const void* source,std::size_t bytes,ComPtr<ID3D12Resource>& output)
{
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,output)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto status=output->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Lumen radiance input",status);
    std::memcpy(mapped,source,bytes); output->Unmap(0,nullptr); return true;
}
bool CreateLumenRadianceOutput(std::size_t bytes,ComPtr<ID3D12Resource>& output)
{
    auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    const auto status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(output.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen radiance output",status);
}
bool ReadLumenOriginRecord(const std::shared_ptr<LumenRadianceRecord>& record,std::vector<LumenProbeOrigin>& output)
{
    if (!LumenRadianceProbeCount(record->Info)) return Fail("This Lumen trace has no probe-origin buffer");
    if (!RecoverSubmittedWork("Read Lumen radiance")) return false; LumenRadiancePending.reset(); auto work=std::make_shared<LumenRadianceWork>(); work->Record=record;
    const auto bytes=std::size_t(LumenRadianceProbeCount(record->Info))*sizeof(LumenProbeOrigin);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback) || !BeginLumenRadianceWork(*work)) return false;
    if (record->Info.Screen.Grid.Y) LumenRadianceTransition(*work,record->Origins.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    work->Commands->CopyBufferRegion(work->Readback.Get(),0,record->Origins.Get(),0,bytes);
    if (record->Info.Screen.Grid.Y) LumenRadianceTransition(*work,record->Origins.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenRadianceWork(work)) return false;
    void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; const auto status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Read Lumen radiance staging",status);
    std::vector<LumenProbeOrigin> samples(LumenRadianceProbeCount(record->Info)); std::memcpy(samples.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenRadianceOrigins(record->Info,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool ReadLumenRadianceRecord(const std::shared_ptr<LumenRadianceRecord>& record,std::vector<LumenRadianceSample>& output)
{
    if (!RecoverSubmittedWork("Read Lumen radiance")) return false; LumenRadiancePending.reset(); auto work=std::make_shared<LumenRadianceWork>(); work->Record=record;
    const auto bytes=std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback) || !BeginLumenRadianceWork(*work)) return false;
    LumenRadianceTransition(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    work->Commands->CopyBufferRegion(work->Readback.Get(),0,record->Buffer.Get(),0,bytes);
    LumenRadianceTransition(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenRadianceWork(work)) return false;
    void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; const auto status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Read Lumen radiance staging",status);
    std::vector<LumenRadianceSample> samples(record->Info.RayCount); std::memcpy(samples.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenRadianceSamples(record->Info,record->Lighting->Distance->Info,record->Lighting->CardInstances,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (upload.Hybrid.Enabled && (!FrameOpen || (!upload.Reflections.Enabled && (GI.Resolved || FrameLumen.HasGathered())) || !HdrSceneActive || !SceneColorWritten || !GBufferWritten ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (DeferredActive && !DeferredResolved) || (AmbientOcclusionActive && !AmbientOcclusionApplied) || ReflectionApplied ||
        FogResolved || LightShaftResolved || TemporalResolved || MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete()))
        return Fail("Lumen hybrid tracing requires resolved opaque HDR before secondary composition");
    if (upload.Hybrid.Enabled && LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen hybrid tracing must precede transparent draws");
    if (upload.Screen.Enabled && (!FrameOpen || !GBufferWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved) ||
        upload.Screen.Width!=unsigned(CreateInfo.Width) || upload.Screen.Height!=unsigned(CreateInfo.Height)))
        return Fail("Lumen screen placement needs this frame's resolved G-buffer and matching viewport");
    const auto found=LumenLightingCaches.find(lighting.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen radiance source lighting");
    const auto source=found->second; LumenRadianceInput input;
    auto distance=source->Distance;
    if (upload.GlobalDistance)
    {
        const auto global=LumenDistanceScenes.find(upload.GlobalDistance.Value);
        if (global==LumenDistanceScenes.end() || global->second->Base!=source->Distance) return Fail("Global distance scene has a different or expired mesh source");
        distance=global->second;
    }
    if (!BuildLumenRadianceInput(source->Info,distance->Info,source->Surface->Captures,source->CardInstances,upload,input,LastError) ||
        !RecoverSubmittedWork("Trace Lumen radiance") || (FrameOpen && !FrameLumen.CanTrace(input.Resolve.Counts.X,LastError)) || !EnsureLumenRadianceTrace(bool(distance->Hardware))) return false;
    std::shared_ptr<LumenRadianceRecord> worldCache; LumenWorldLookupFrame worldFrame{};
    if (upload.WorldCache.Radiance)
    {
        const auto cache=LumenRadianceTraces.find(upload.WorldCache.Radiance.Value);
        if (cache==LumenRadianceTraces.end() || cache->second->Lighting!=source) return Fail("World radiance cache has a different or expired lighting source");
        worldCache=cache->second;
        if (!BuildLumenWorldLookupFrame(worldCache->Info,upload,input,worldFrame,LastError) || !EnsureLumenWorldLookup()) return false;
    }
    LumenRadiancePending.reset(); auto work=std::make_shared<LumenRadianceWork>(); auto record=std::make_shared<LumenRadianceRecord>(); work->Record=record;
    record->Info=MakeLumenRadianceInfo(source->Info.Identity,upload,input,worldCache?unsigned(worldFrame.Query.W):0); record->Info.HardwareTracing=bool(distance->Hardware); record->Lighting=source; record->WorldCache=worldCache; record->TraceDistance=distance; const auto count=std::size_t(input.Resolve.Counts.X); const auto sources=LumenRadianceSources(upload,input),traceFrame=LumenRadianceFrameData(input);
    if (!(upload.Screen.Enabled?CreateLumenRadianceOutput(sources.Bytes,work->Rays):CreateLumenRadianceUpload(sources.Data,sources.Bytes,work->Rays)) ||
        !CreateLumenRadianceUpload(input.Words.data(),input.Words.size()*sizeof(std::uint32_t),work->Mapping) ||
        !CreateLumenRadianceOutput(count*sizeof(LumenDistanceHit),work->Hits) || !CreateLumenRadianceOutput(count*sizeof(LumenRadianceSample),record->Buffer)) return false;
    record->Origins=work->Rays;
    if (worldCache)
    {
        // Root CBVs keep aligned storage when the portable frame ABI grows.
        constexpr auto alignment=D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
        constexpr auto bytes=(sizeof(worldFrame)+alignment-1)/alignment*alignment;
        std::array<std::uint8_t,bytes> constants{};
        std::memcpy(constants.data(),&worldFrame,sizeof(worldFrame));
        if (!CreateLumenRadianceUpload(constants.data(),constants.size(),work->WorldUniform)) return false;
    }
    if (input.Depth.Bytes && !CreateLumenRadianceOutput(input.Depth.Bytes,work->Pyramid)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=upload.Hybrid.Enabled?4:upload.Screen.Enabled?3:1; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    const auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen radiance texture view",status);
    D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
    Device->CreateShaderResourceView(source->Surface->Images[1].Get(),&view,work->Views->GetCPUDescriptorHandleForHeapStart()); if (!BeginLumenRadianceWork(*work)) return false;
    const auto groups=unsigned((count+LumenRadianceThreads-1)/LumenRadianceThreads),x=std::min(groups,LumenRadianceColumns),y=(groups+LumenRadianceColumns-1)/LumenRadianceColumns;
    if (upload.Screen.Enabled)
    {
        const auto stride=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart(); cpu.ptr+=stride;
        // Shader-visible heaps are CPU write-only; create these private views
        // directly rather than copying from the global shader-visible arena.
        D3D12_SHADER_RESOURCE_VIEW_DESC screenView{}; screenView.Format=DXGI_FORMAT_R32_FLOAT;
        screenView.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; screenView.Texture2D.MipLevels=1; screenView.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        Device->CreateShaderResourceView(MainDepth.Get(),&screenView,cpu); cpu.ptr+=stride;
        screenView.Format=GBufferSurfaces[1].Resource->GetDesc().Format; Device->CreateShaderResourceView(GBufferSurfaces[1].Resource.Get(),&screenView,cpu);
        const auto depthState=MainDepthState,normalState=GBufferSurfaces[1].State;
        Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; work->Commands->SetDescriptorHeaps(1,heaps);
        work->Commands->SetPipelineState((upload.Reflections.Enabled?LumenReflectionRayShader:LumenScreenPlacement).Get()); work->Commands->SetComputeRootSignature(LumenScreenRoot.Get());
        const auto placement=LumenRadiancePlacementData(input);
        work->Commands->SetComputeRoot32BitConstants(0,UINT(placement.Bytes/sizeof(std::uint32_t)),placement.Data,0);
        auto gpu=work->Views->GetGPUDescriptorHandleForHeapStart(); gpu.ptr+=stride; work->Commands->SetComputeRootDescriptorTable(1,gpu);
        work->Commands->SetComputeRootUnorderedAccessView(2,record->Origins->GetGPUVirtualAddress());
        work->Commands->Dispatch((input.Screen.Grid.Y+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
        LumenRadianceTransition(*work,record->Origins.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (upload.Hybrid.Enabled)
        {
            if (input.Depth.Bytes)
            {
                work->Commands->SetPipelineState(LumenScreenDepth.Get()); work->Commands->SetComputeRootSignature(LumenScreenDepthRoot.Get());
                work->Commands->SetComputeRootDescriptorTable(1,gpu); work->Commands->SetComputeRootUnorderedAccessView(2,work->Pyramid->GetGPUVirtualAddress());
                for (const auto& level:input.Depth.Levels)
                {
                    work->Commands->SetComputeRoot32BitConstants(0,sizeof(level)/sizeof(std::uint32_t),&level,0);
                    const auto levelGroups=(level.Destination.X*level.Destination.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
                    work->Commands->Dispatch(std::min(levelGroups,LumenRadianceColumns),(levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns,1);
                    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=work->Pyramid.Get(); work->Commands->ResourceBarrier(1,&barrier);
                }
                LumenRadianceTransition(*work,work->Pyramid.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            auto& scene=HdrSurfaces[HdrScene]; const auto sceneState=scene.State;
            cpu.ptr+=stride; screenView.Format=scene.Resource->GetDesc().Format; Device->CreateShaderResourceView(scene.Resource.Get(),&screenView,cpu);
            Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            work->Commands->SetPipelineState(LumenHybridTrace.Get()); work->Commands->SetComputeRootSignature(LumenHybridRoot.Get());
            work->Commands->SetComputeRoot32BitConstants(0,sizeof(input.Hybrid)/sizeof(std::uint32_t),&input.Hybrid,0);
            work->Commands->SetComputeRootDescriptorTable(1,gpu); work->Commands->SetComputeRootShaderResourceView(2,record->Origins->GetGPUVirtualAddress());
            work->Commands->SetComputeRootUnorderedAccessView(3,work->Hits->GetGPUVirtualAddress()); work->Commands->SetComputeRootUnorderedAccessView(4,record->Buffer->GetGPUVirtualAddress());
            work->Commands->SetComputeRootShaderResourceView(5,(work->Pyramid?work->Pyramid:record->Origins)->GetGPUVirtualAddress());
            work->Commands->Dispatch(x,y,1);
            D3D12_RESOURCE_BARRIER barriers[2]{};
            for (auto& barrier:barriers) barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[0].UAV.pResource=work->Hits.Get(); barriers[1].UAV.pResource=record->Buffer.Get(); work->Commands->ResourceBarrier(2,barriers);
            Transition(scene.Resource.Get(),scene.State,sceneState);
        }
        Transition(MainDepth.Get(),MainDepthState,depthState); Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,normalState);
    }
    if (worldCache)
    {
        work->Commands->SetPipelineState(LumenWorldLookup.Get()); work->Commands->SetComputeRootSignature(LumenWorldRoot.Get());
        work->Commands->SetComputeRootConstantBufferView(0,work->WorldUniform->GetGPUVirtualAddress());
        work->Commands->SetComputeRootShaderResourceView(1,worldCache->Buffer->GetGPUVirtualAddress());
        work->Commands->SetComputeRootShaderResourceView(2,record->Origins->GetGPUVirtualAddress());
        work->Commands->SetComputeRootUnorderedAccessView(3,work->Hits->GetGPUVirtualAddress());
        work->Commands->SetComputeRootUnorderedAccessView(4,record->Buffer->GetGPUVirtualAddress()); work->Commands->Dispatch(x,y,1);
        D3D12_RESOURCE_BARRIER barriers[2]{}; for (auto& barrier:barriers) barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers[0].UAV.pResource=work->Hits.Get(); barriers[1].UAV.pResource=record->Buffer.Get(); work->Commands->ResourceBarrier(2,barriers);
    }
    work->Commands->SetPipelineState((distance->Hardware?LumenHardwarePipelines[unsigned(input.ProbeTrace.Sampling.X?LumenHardwareProgram::Probes:LumenHardwareProgram::Radiance)]:input.ProbeTrace.Sampling.X?LumenProbeTrace:LumenRadianceTrace).Get());
    work->Commands->SetComputeRootSignature((input.ProbeTrace.Sampling.X?LumenProbeTraceRoot:LumenDistanceRoot).Get());
    BindLumenHardware(work->Commands.Get(),*distance,4);
    work->Commands->SetComputeRoot32BitConstants(0,unsigned(traceFrame.Bytes/sizeof(std::uint32_t)),traceFrame.Data,0);
    work->Commands->SetComputeRootShaderResourceView(1,distance->Scene->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(2,work->Rays->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(3,work->Hits->GetGPUVirtualAddress()); work->Commands->Dispatch(x,y,1);
    LumenRadianceTransition(*work,work->Hits.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    work->Commands->SetPipelineState(LumenRadianceResolve.Get()); work->Commands->SetComputeRootSignature(LumenRadianceRoot.Get()); ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; work->Commands->SetDescriptorHeaps(1,heaps);
    work->Commands->SetComputeRoot32BitConstants(0,sizeof(input.Resolve)/sizeof(std::uint32_t),&input.Resolve,0); work->Commands->SetComputeRootDescriptorTable(1,work->Views->GetGPUDescriptorHandleForHeapStart());
    work->Commands->SetComputeRootShaderResourceView(2,work->Hits->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(3,source->Buffer->GetGPUVirtualAddress());
    work->Commands->SetComputeRootShaderResourceView(4,work->Mapping->GetGPUVirtualAddress()); work->Commands->SetComputeRootUnorderedAccessView(5,record->Buffer->GetGPUVirtualAddress()); work->Commands->Dispatch(x,y,1);
    LumenRadianceTransition(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenRadianceWork(work)) return false;
    if (FrameOpen && distance->Hardware) PendingFrame.Statistics.LumenHardwareRayCount+=input.Resolve.Counts.X;
    LumenRadianceTraceResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenRadianceRecord(record,result.Samples)) return false;
    result.Handle=NewLumenRadianceHandle(); if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,result.Info.RayCount);
    output=std::move(result); return true;
}
bool AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!ValidateLumenRadianceAssembly(assembly,LastError) || !RecoverSubmittedWork("Assemble Lumen world radiance")) return false;
    auto work=std::make_shared<LumenRadianceWork>();
    for (const auto& copy:assembly.Copies)
    {
        const auto source=LumenRadianceTraces.find(copy.Source.Value);
        if (source==LumenRadianceTraces.end()) return Fail("World radiance copy source expired");
        if (!ValidateLumenRadianceCopy(source->second->Info,assembly,copy,LastError)) return false;
        if (!work->CopySources.empty() && work->CopySources.front()->Lighting!=source->second->Lighting) return Fail("World radiance copy crosses lighting generations");
        work->CopySources.push_back(source->second);
    }
    auto record=std::make_shared<LumenRadianceRecord>(); work->Record=record; record->Lighting=work->CopySources.front()->Lighting;
    record->Info=work->CopySources.front()->Info; record->Info.Probes=assembly.Probes;
    record->Info.RayCount=unsigned(assembly.Probes.Origins.size())*assembly.Probes.RaysPerProbe*assembly.Probes.DepthLayers;
    if (!CreateLumenRadianceUpload(assembly.Probes.Origins.data(),assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin),record->Origins) ||
        !CreateLumenRadianceOutput(std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample),record->Buffer) || !BeginLumenRadianceWork(*work)) return false;
    LumenRadianceTransition(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
    const auto probeBytes=std::uint64_t(assembly.Probes.RaysPerProbe)*assembly.Probes.DepthLayers*sizeof(LumenRadianceSample);
    for (unsigned range=0;range<assembly.Copies.size();++range)
    {
        const auto& copy=assembly.Copies[range]; auto* source=work->CopySources[range]->Buffer.Get();
        LumenRadianceTransition(*work,source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        work->Commands->CopyBufferRegion(record->Buffer.Get(),copy.DestinationProbe*probeBytes,source,copy.SourceProbe*probeBytes,copy.ProbeCount*probeBytes);
        LumenRadianceTransition(*work,source,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    LumenRadianceTransition(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenRadianceWork(work)) return false;
    LumenRadianceTraceResult result; result.Info=record->Info; result.Handle=NewLumenRadianceHandle();
    if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    if (!LumenRadianceProbeCount(found->second->Info)) return Fail("Lumen trace has no probe origins");
    return ReadLumenOriginRecord(found->second,output);
}
bool ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    return ReadLumenRadianceRecord(found->second,output);
}
bool DestroyLumenRadiance(LumenRadianceHandle handle) { return LumenRadianceTraces.erase(handle.Value)!=0 || Fail("Unknown Lumen radiance trace"); }
