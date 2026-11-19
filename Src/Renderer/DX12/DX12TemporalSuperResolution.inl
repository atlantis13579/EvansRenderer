// Structured buffer views carry element counts. Unlike root buffer descriptors,
// descriptor-table accesses retain native bounds handling for partial groups.
// https://learn.microsoft.com/windows/win32/direct3d12/root-signatures-overview
// Dedicated direct-queue work retains all resources across a failed fence wait.
// Submitted buffers decay to COMMON; later history reads promote them again.
struct TSRNativeHistory
{
    Impl& Owner; TSRHistoryInfo Info; std::array<ComPtr<ID3D12Resource>,2> Buffers; HdrSurface Color; ComPtr<ID3D12DescriptorHeap> ExportViews;
    explicit TSRNativeHistory(Impl& owner):Owner(owner) {}
    ~TSRNativeHistory() { Owner.RTVArena.Release(Color.RTV); Owner.SRVArena.Release(Color.SRV); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<TSRNativeHistory>> TSRHistories;
TSRSceneState TSRScene;
HdrSurface TSROpaque;
std::shared_ptr<TSRNativeHistory> TSRPending;
struct TSRNativeImages { std::array<ComPtr<ID3D12Resource>,TSRSceneImageCount> Images; std::array<D3D12_RESOURCE_STATES,TSRSceneImageCount> States{}; std::uint32_t Flags=0; };
static constexpr UINT TSRCoreInputView=TSRSceneImageCount,TSRCoreHistoryView=TSRCoreInputView+1;
static constexpr UINT TSRPackOutputView=TSRCoreHistoryView+1,TSRCoreOutputView=TSRPackOutputView+1,TSRViewCount=TSRCoreOutputView+2;
struct TSRImageWork { TSRNativeImages Native; std::array<ComPtr<ID3D12Resource>,TSRSceneImageCount> Uploads; ComPtr<ID3D12DescriptorHeap> Views; };
std::shared_ptr<TSRImageWork> TSRPendingImages;
struct TSRDisplayWork;
std::shared_ptr<TSRDisplayWork> TSRPendingDisplay;
ComPtr<ID3D12RootSignature> TSRRoot;
ComPtr<ID3D12PipelineState> TSRPipeline;
ComPtr<ID3DBlob> TSRPackCode;
ComPtr<ID3D12RootSignature> TSRPackRoot;
ComPtr<ID3D12PipelineState> TSRPackPipeline;
ComPtr<ID3D12RootSignature> TSRExportRoot;
ComPtr<ID3D12PipelineState> TSRExportPipeline;
ReadbackResource TSRExportReadback;
ComPtr<ID3D12CommandAllocator> TSRAllocator;
ComPtr<ID3D12GraphicsCommandList> TSRCommands;
std::array<ComPtr<ID3D12Resource>,4> TSRScratch;
void DestroyTSR()
{ TSRScene.Reset(); RTVArena.Release(TSROpaque.RTV); SRVArena.Release(TSROpaque.SRV); TSROpaque={}; TSRPendingDisplay.reset(); TSRHistories.clear(); TSRPending.reset(); TSRPendingImages.reset(); for (auto& buffer:TSRScratch) buffer.Reset(); TSRExportReadback={}; TSRCommands.Reset(); TSRAllocator.Reset(); TSRPipeline.Reset(); TSRPackPipeline.Reset(); TSRPackRoot.Reset(); TSRPackCode.Reset(); TSRRoot.Reset(); TSRExportPipeline.Reset(); TSRExportRoot.Reset(); }
bool DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!RecoverSubmittedWork("Release TSR history") || !WaitForGPU()) return false;
    return (handle && TSRHistories.erase(handle.Value)) || Fail("DX12 TSR history handle is invalid or belongs to another RHI");
}
bool EnsureTSR()
{
    if (TSRPipeline && TSRAllocator && TSRCommands && TSRPackCode) return true;
    TSRCommands.Reset(); TSRAllocator.Reset(); TSRPipeline.Reset(); TSRRoot.Reset();
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors=2;
    ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors=2;
    D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants={0,0,sizeof(TSRConstants)/sizeof(std::uint32_t)};
    for (UINT table=0;table<2;++table) { parameters[table+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[table+1].DescriptorTable={1,&ranges[table]}; }
    D3D12_ROOT_SIGNATURE_DESC root{}; root.NumParameters=3; root.pParameters=parameters; ComPtr<ID3DBlob> binary,diagnostics;
    auto status=SerializeRootSignatureFunction(&root,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),diagnostics.GetAddressOf()); if (FAILED(status)) return FailHRESULT("Serialize TSR root",status);
    status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(TSRRoot.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR root",status);
    if (!LoadTechniqueShaderSource("DX12","TSR.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> code; if (!CompileShader("CSMain","cs_5_0",code,nullptr,true)) return false;
    const D3D_SHADER_MACRO packDefines[]{{"EVANS_TSR_PACK_STAGE","1"},{nullptr,nullptr}};
    if (!CompileShader("CSPack","cs_5_0",TSRPackCode,packDefines,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=TSRRoot.Get(); pipeline.CS={code->GetBufferPointer(),code->GetBufferSize()};
    status=Evans::Renderer::CreateCachedComputePipelineState(Device,&pipeline,IID_PPV_ARGS(TSRPipeline.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR pipeline",status);
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(TSRAllocator.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR allocator",status);
    status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,TSRAllocator.Get(),nullptr,IID_PPV_ARGS(TSRCommands.GetAddressOf()));
    if (SUCCEEDED(status)) status=TSRCommands->Close(); return SUCCEEDED(status) || FailHRESULT("Create TSR commands",status);
}
bool ExportTSR(const std::shared_ptr<TSRNativeHistory>& record,TSRReadback& result,bool readback=true)
{
    if (!TSRExportPipeline)
    {
        TSRExportRoot.Reset(); D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants={0,0,sizeof(TSRTextureConstants)/sizeof(std::uint32_t)};
        D3D12_DESCRIPTOR_RANGE historyRange{}; historyRange.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; historyRange.NumDescriptors=1;
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&historyRange};
        for (auto& p:parameters) p.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC root{}; root.NumParameters=2; root.pParameters=parameters; ComPtr<ID3DBlob> binary,diagnostics;
        auto status=SerializeRootSignatureFunction(&root,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),diagnostics.GetAddressOf()); if (FAILED(status)) return FailHRESULT("Serialize TSR export root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(TSRExportRoot.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR export root",status);
        if (!LoadTechniqueShaderSource("DX12","TSR.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO defines[]{{"EVANS_TSR_EXPORT_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader("VSExport","vs_5_0",vertex,defines,true) || !CompileShader("PSExport","ps_5_0",pixel,defines,true)) return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=TSRExportRoot.Get(); desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
        auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
        desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
        auto& face=desc.DepthStencilState.FrontFace; face.StencilFailOp=face.StencilDepthFailOp=face.StencilPassOp=D3D12_STENCIL_OP_KEEP; face.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=face;
        desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=1;
        status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(TSRExportPipeline.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR HDR export pipeline",status);
    }
    if (!CreateHdrSurface(record->Color,record->Info.Width,record->Info.Height) || (readback && !CreateReadback(record->Color.Resource.Get(),TSRExportReadback))) return false;
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{}; heapDesc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heapDesc.NumDescriptors=1; heapDesc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    auto viewStatus=Device->CreateDescriptorHeap(&heapDesc,IID_PPV_ARGS(record->ExportViews.GetAddressOf())); if (FAILED(viewStatus)) return FailHRESULT("Create TSR export view",viewStatus);
    D3D12_SHADER_RESOURCE_VIEW_DESC historyView{}; historyView.ViewDimension=D3D12_SRV_DIMENSION_BUFFER; historyView.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    historyView.Buffer.NumElements=record->Info.Width*record->Info.Height*TSRHistoryVectors; historyView.Buffer.StructureByteStride=sizeof(Float4);
    Device->CreateShaderResourceView(record->Buffers[0].Get(),&historyView,record->ExportViews->GetCPUDescriptorHandleForHeapStart());
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(record->Info.Width,record->Info.Height,0,constants,LastError)) return false;
    auto status=TSRAllocator->Reset(); if (SUCCEEDED(status)) status=TSRCommands->Reset(TSRAllocator.Get(),TSRExportPipeline.Get()); if (FAILED(status)) return FailHRESULT("Reset TSR HDR export",status);
    TSRCommands->SetGraphicsRootSignature(TSRExportRoot.Get()); TSRCommands->SetGraphicsRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
    auto* exportHeap=record->ExportViews.Get(); TSRCommands->SetDescriptorHeaps(1,&exportHeap);
    TSRCommands->SetGraphicsRootDescriptorTable(1,record->ExportViews->GetGPUDescriptorHandleForHeapStart());
    const auto target=RTVArena.Cpu(record->Color.RTV); TSRCommands->OMSetRenderTargets(1,&target,FALSE,nullptr);
    const D3D12_VIEWPORT viewport{0,0,float(record->Info.Width),float(record->Info.Height),0,1}; const D3D12_RECT scissor{0,0,LONG(record->Info.Width),LONG(record->Info.Height)};
    TSRCommands->RSSetViewports(1,&viewport); TSRCommands->RSSetScissorRects(1,&scissor); TSRCommands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); TSRCommands->DrawInstanced(3,1,0,0);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; barrier.Transition={record->Color.Resource.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COPY_SOURCE}; TSRCommands->ResourceBarrier(1,&barrier);
    if(readback){
    D3D12_TEXTURE_COPY_LOCATION source{},destination{}; source.pResource=record->Color.Resource.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.pResource=TSRExportReadback.Buffer.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; destination.PlacedFootprint=TSRExportReadback.Footprint;
    TSRCommands->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
    }
    record->Color.State=D3D12_RESOURCE_STATE_COPY_SOURCE;
    status=TSRCommands->Close(); if (FAILED(status)) return FailHRESULT("Close TSR HDR export",status);
    TSRPending=record; ID3D12CommandList* commands[]{TSRCommands.Get()}; CommandQueue->ExecuteCommandLists(1,commands); SubmittedWorkPending=true;
    if (!SignalAndWait()) { LastError="TSR HDR export: "+LastError; return false; } TSRPending.reset();
    if(!readback)return true;
    void* mapped=nullptr; const D3D12_RANGE read{0,SIZE_T(TSRExportReadback.TotalSize)}; status=TSRExportReadback.Buffer->Map(0,&read,&mapped); if (FAILED(status)) return FailHRESULT("Map TSR HDR image",status);
    result.PresentationHDR.resize(std::size_t(record->Info.Width)*record->Info.Height);
    const auto* start=static_cast<const std::uint8_t*>(mapped)+TSRExportReadback.Footprint.Offset;
    for (unsigned y=0;y<record->Info.Height;++y)
    {
        const auto* row=reinterpret_cast<const std::uint16_t*>(start+std::size_t(y)*TSRExportReadback.Footprint.Footprint.RowPitch);
        for (unsigned x=0;x<record->Info.Width;++x) result.PresentationHDR[std::size_t(y)*record->Info.Width+x]=DecodeTSRHalfPixel(row+x*4);
    }
    const D3D12_RANGE noWrite{0,0}; TSRExportReadback.Buffer->Unmap(0,&noWrite); return true;
}
bool PackTSRImages(const TSRFrame& frame,TSRImageWork& work,const TSRNativeHistory& candidate,const TSRNativeHistory* previous)
{
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(frame.InputWidth,frame.InputHeight,work.Native.Flags,constants,LastError)) return false;
    for (unsigned image=0;image<TSRSceneImageCount;++image) if (TSRImageRequired(image,work.Native.Flags))
    {
        auto* source=work.Native.Images[image].Get(); if (!source) return Fail("DX12 TSR is missing a required scene image");
        const auto desc=source->GetDesc(); ComPtr<ID3D12Device> device; const auto status=source->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
        if (FAILED(status) || device.Get()!=Device.Get() || desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width!=frame.InputWidth || desc.Height!=frame.InputHeight || desc.SampleDesc.Count!=1 || desc.DepthOrArraySize!=1)
            return Fail("DX12 TSR requires resolved render-sized images on this device");
        for (unsigned previousImage=0;previousImage<image;++previousImage)
            if (TSRImageRequired(previousImage,work.Native.Flags) && work.Native.Images[previousImage].Get()==source && work.Native.States[previousImage]!=work.Native.States[image])
                return Fail("DX12 aliased TSR images disagree about resource state");
    }
    if (!TSRPackPipeline)
    {
        TSRPackRoot.Reset(); D3D12_DESCRIPTOR_RANGE images{}; images.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; images.NumDescriptors=TSRSceneImageCount;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants={0,0,sizeof(TSRTextureConstants)/sizeof(std::uint32_t)};
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&images}; parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        D3D12_DESCRIPTOR_RANGE packed{}; packed.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; packed.NumDescriptors=1; parameters[2].DescriptorTable={1,&packed};
        D3D12_ROOT_SIGNATURE_DESC root{}; root.NumParameters=3; root.pParameters=parameters; ComPtr<ID3DBlob> binary,diagnostics;
        auto status=SerializeRootSignatureFunction(&root,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),diagnostics.GetAddressOf()); if (FAILED(status)) return FailHRESULT("Serialize TSR image packing root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(TSRPackRoot.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR image packing root",status);
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=TSRPackRoot.Get(); desc.CS={TSRPackCode->GetBufferPointer(),TSRPackCode->GetBufferSize()};
        status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(TSRPackPipeline.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR image packing pipeline",status);
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=TSRViewCount; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    const auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work.Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR image descriptor heap",status);
    const auto increment=Device->GetDescriptorHandleIncrementSize(heap.Type); auto cpu=work.Views->GetCPUDescriptorHandleForHeapStart();
    for (unsigned image=0;image<TSRSceneImageCount;++image)
    {
        const auto sourceImage=TSRImageRequired(image,work.Native.Flags)?image:TSRSceneFinal; auto* source=work.Native.Images[sourceImage].Get(); const auto desc=source->GetDesc();
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=desc.Format==DXGI_FORMAT_R32_TYPELESS || desc.Format==DXGI_FORMAT_D32_FLOAT?DXGI_FORMAT_R32_FLOAT:desc.Format; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(source,&view,cpu); cpu.ptr+=increment;
        if (!TSRImageRequired(image,work.Native.Flags)) continue;
        // Logical input slots may share a resource; transition it only once.
        bool duplicate=false;
        for (unsigned previousImage=0;previousImage<image;++previousImage) if (TSRImageRequired(previousImage,work.Native.Flags) && work.Native.Images[previousImage].Get()==source)
        { work.Native.States[image]=work.Native.States[previousImage]; duplicate=true; break; }
        if (duplicate) continue;
        auto& state=work.Native.States[image]; if (state==D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) continue;
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; barrier.Transition={source,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}; TSRCommands->ResourceBarrier(1,&barrier); state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    const auto bufferSRV=[&](ID3D12Resource* buffer,UINT index)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER; view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.NumElements=static_cast<UINT>(buffer->GetDesc().Width/sizeof(Float4)); view.Buffer.StructureByteStride=sizeof(Float4);
        auto handle=work.Views->GetCPUDescriptorHandleForHeapStart(); handle.ptr+=SIZE_T(index)*increment; Device->CreateShaderResourceView(buffer,&view,handle);
    };
    const auto bufferUAV=[&](ID3D12Resource* buffer,UINT index)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements=static_cast<UINT>(buffer->GetDesc().Width/sizeof(Float4)); view.Buffer.StructureByteStride=sizeof(Float4);
        auto handle=work.Views->GetCPUDescriptorHandleForHeapStart(); handle.ptr+=SIZE_T(index)*increment; Device->CreateUnorderedAccessView(buffer,nullptr,&view,handle);
    };
    bufferSRV(TSRScratch[0].Get(),TSRCoreInputView); bufferSRV(previous?previous->Buffers[0].Get():TSRScratch[1].Get(),TSRCoreHistoryView);
    bufferUAV(TSRScratch[0].Get(),TSRPackOutputView);
    for (UINT slot=0;slot<candidate.Buffers.size();++slot) bufferUAV(candidate.Buffers[slot].Get(),TSRCoreOutputView+slot);
    auto* descriptors=work.Views.Get(); TSRCommands->SetDescriptorHeaps(1,&descriptors); TSRCommands->SetPipelineState(TSRPackPipeline.Get()); TSRCommands->SetComputeRootSignature(TSRPackRoot.Get());
    TSRCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0); TSRCommands->SetComputeRootDescriptorTable(1,work.Views->GetGPUDescriptorHandleForHeapStart()); auto packedHandle=work.Views->GetGPUDescriptorHandleForHeapStart(); packedHandle.ptr+=UINT64(TSRPackOutputView)*increment; TSRCommands->SetComputeRootDescriptorTable(2,packedHandle);
    TSRCommands->Dispatch((frame.InputWidth+TSRGroupSide-1)/TSRGroupSide,(frame.InputHeight+TSRGroupSide-1)/TSRGroupSide,1);
    D3D12_RESOURCE_BARRIER ready{}; ready.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; ready.Transition={TSRScratch[0].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}; TSRCommands->ResourceBarrier(1,&ready); return true;
}
bool ReconstructTSRImages(const TSRFrame& frame,TSRNativeImages& images,TSRHistoryHandle previous,TSRResult& output)
{
    std::shared_ptr<TSRNativeHistory> old;
    if (previous) { const auto found=TSRHistories.find(previous.Value); if (found==TSRHistories.end()) return Fail("DX12 previous TSR history is invalid"); old=found->second; }
    TSRConstants constants; if (!BuildTSRConstants(frame,old?&old->Info:nullptr,constants,LastError) || !RecoverSubmittedWork("Reconstruct TSR") || !WaitForGPU() || !EnsureTSR()) return false;
    TSRPending.reset(); TSRPendingImages.reset(); for (auto& resource:TSRScratch) resource.Reset();
    auto imageWork=std::make_shared<TSRImageWork>(); imageWork->Native=images;
    const std::array<Float4,TSRHistoryVectors> empty{};
    {
        const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(std::size_t(frame.InputWidth)*frame.InputHeight*(TSRInputVectors*sizeof(Float4))); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(TSRScratch[0].GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR GPU input packet",status);
    }
    if (!CreateBufferResource(sizeof(empty),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,TSRScratch[1])) return false;
    void* dummy=nullptr; const D3D12_RANGE noRead{0,0}; auto mappedStatus=TSRScratch[1]->Map(0,&noRead,&dummy); if (FAILED(mappedStatus)) return FailHRESULT("Map TSR dummy history",mappedStatus); std::memcpy(dummy,empty.data(),sizeof(empty)); TSRScratch[1]->Unmap(0,nullptr);
    auto record=std::make_shared<TSRNativeHistory>(*this); record->Info=TSRHistoryIdentity(frame);
    const auto bytes=std::size_t(record->Info.Width)*record->Info.Height*TSRHistoryVectors*sizeof(Float4);
    for (unsigned slot=0;slot<record->Buffers.size();++slot)
    {
        const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(record->Buffers[slot].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create TSR output",status);
        if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,TSRScratch[slot+2])) return false;
    }
    TSRPending=record; auto status=TSRAllocator->Reset(); if (SUCCEEDED(status)) status=TSRCommands->Reset(TSRAllocator.Get(),TSRPipeline.Get());
    if (FAILED(status)) return FailHRESULT("Reset TSR recording",status);
    TSRPendingImages=imageWork; if (!PackTSRImages(frame,*imageWork,*record,constants.Options.X?old.get():nullptr)) return false;
    TSRCommands->SetPipelineState(TSRPipeline.Get());
    TSRCommands->SetComputeRootSignature(TSRRoot.Get()); TSRCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto inputs=imageWork->Views->GetGPUDescriptorHandleForHeapStart(),outputs=inputs;
    inputs.ptr+=UINT64(TSRCoreInputView)*increment; outputs.ptr+=UINT64(TSRCoreOutputView)*increment;
    TSRCommands->SetComputeRootDescriptorTable(1,inputs); TSRCommands->SetComputeRootDescriptorTable(2,outputs);
    TSRCommands->Dispatch((constants.Extent.Z+TSRGroupSide-1)/TSRGroupSide,(constants.Extent.W+TSRGroupSide-1)/TSRGroupSide,1);
    for (unsigned slot=0;slot<record->Buffers.size();++slot)
    {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={record->Buffers[slot].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE}; TSRCommands->ResourceBarrier(1,&barrier);
        if(frame.CaptureDiagnostics)TSRCommands->CopyResource(TSRScratch[slot+2].Get(),record->Buffers[slot].Get());
    }
    status=TSRCommands->Close(); if (FAILED(status)) return FailHRESULT("Close TSR recording",status);
    ID3D12CommandList* commands[]{TSRCommands.Get()}; CommandQueue->ExecuteCommandLists(1,commands); SubmittedWorkPending=true; images.States=imageWork->Native.States;
    if (!SignalAndWait()) { LastError="TSR image packing/reconstruction: "+LastError; return false; }
    TSRPending.reset(); TSRResult candidate; candidate.HistoryUsed=constants.Options.X!=0; candidate.Info=record->Info; std::vector<Float4>* targets[]{&candidate.Readback.History,&candidate.Readback.Auxiliary};
    for (unsigned slot=0;frame.CaptureDiagnostics && slot<record->Buffers.size();++slot)
    {
        void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; status=TSRScratch[slot+2]->Map(0,&read,&mapped);
        if (FAILED(status)) return FailHRESULT("Map TSR readback",status); targets[slot]->resize(bytes/sizeof(Float4)); std::memcpy(targets[slot]->data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; TSRScratch[slot+2]->Unmap(0,&noWrite);
    }
    if (!ExportTSR(record,candidate.Readback,frame.CaptureDiagnostics) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,candidate.Readback,LastError))) return false;
    candidate.Handle=NewTSRHistoryHandle(); if (!candidate.Handle) return Fail("TSR history handle space is exhausted");
    TSRHistories.emplace(candidate.Handle.Value,std::move(record)); output=std::move(candidate); return true;
}
struct TSRDisplayWork
{
    Impl& Owner; int Width=0,Height=0; bool Hdr=false,Submitted=false; FrameCapture Capture;
    std::shared_ptr<TSRNativeHistory> Source; D3D12_RESOURCE_STATES SourceState=D3D12_RESOURCE_STATE_COMMON;
    ComPtr<ID3D12Resource> Color,Post; ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands;
    D3D12_RESOURCE_STATES ColorState=D3D12_RESOURCE_STATE_RENDER_TARGET,PostState=D3D12_RESOURCE_STATE_RENDER_TARGET;
    UINT RTV=InvalidDescriptor,SRV=InvalidDescriptor,PostRTV=InvalidDescriptor,PostSRV=InvalidDescriptor;
    std::array<HdrSurface,HdrSurfaceCount> HDR; ReadbackResource Readback;
    explicit TSRDisplayWork(Impl& owner):Owner(owner) {}
    ~TSRDisplayWork()
    {
        Owner.RTVArena.Release(RTV); Owner.RTVArena.Release(PostRTV); Owner.SRVArena.Release(SRV); Owner.SRVArena.Release(PostSRV);
        for (const auto& image:HDR) { Owner.RTVArena.Release(image.RTV); Owner.SRVArena.Release(image.SRV); }
    }
    void Swap()
    {
        std::swap(Width,Owner.CreateInfo.Width); std::swap(Height,Owner.CreateInfo.Height); std::swap(Hdr,Owner.HdrSceneActive); std::swap(Capture,Owner.PendingFrame);
        Color.Swap(Owner.MainColor); Post.Swap(Owner.PostColor); Commands.Swap(Owner.CommandList); HDR.swap(Owner.HdrSurfaces);
        std::swap(ColorState,Owner.MainColorState); std::swap(PostState,Owner.PostColorState); std::swap(RTV,Owner.MainColorRTV); std::swap(SRV,Owner.MainColorSRV); std::swap(PostRTV,Owner.PostColorRTV); std::swap(PostSRV,Owner.PostColorSRV);
    }
};
bool PrepareTSRScene(const TSRSceneFrame& requested,const Techniques::PostProcessSettings& post)
{
    const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value);
    const auto found=TSRHistories.find(TSRScene.Committed.Value);
    return TSRScene.Begin(requested,CreateInfo.Width,CreateInfo.Height,post,found==TSRHistories.end()?nullptr:&found->second->Info,LastError);
}
void AbortTSRScene() { const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value); }
void CommitTSRScene() { const auto old=TSRScene.Commit(); if (old) TSRHistories.erase(old.Value); }
bool RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& requested,FrameCapture& output)
{
    const auto found=TSRHistories.find(handle.Value); if (found==TSRHistories.end()) return Fail("DX12 TSR display history is invalid");
    Techniques::PostProcessSettings settings; if (!BuildTSRDisplaySettings(requested,settings,LastError) || !RecoverSubmittedWork("Render TSR display") || !WaitForGPU()) return false;
    TSRPendingDisplay.reset(); auto work=std::make_shared<TSRDisplayWork>(*this); work->Source=found->second; work->SourceState=work->Source->Color.State;
    work->Width=work->Source->Info.Width; work->Height=work->Source->Info.Height;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=work->Width; desc.Height=work->Height; desc.DepthOrArraySize=desc.MipLevels=1; desc.SampleDesc.Count=1; desc.Format=DXGI_FORMAT_R8G8B8A8_TYPELESS; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_CLEAR_VALUE clear{}; clear.Format=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_RENDER_TARGET,&clear,IID_PPV_ARGS(work->Color.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR display color",status);
    work->RTV=RTVArena.Allocate(); if (work->RTV==InvalidDescriptor || !CreateReadback(work->Color.Get(),work->Readback)) return Fail("DX12 TSR display allocation failed");
    D3D12_RENDER_TARGET_VIEW_DESC view{}; view.Format=clear.Format; view.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE2D; Device->CreateRenderTargetView(work->Color.Get(),&view,RTVArena.Cpu(work->RTV));
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create TSR display commands",status);
    {
        struct Restore
        {
            TSRDisplayWork& Work;
            explicit Restore(TSRDisplayWork& work):Work(work){Work.Swap();}
            ~Restore(){Work.Swap(); if (!Work.Submitted) Work.Source->Color.State=Work.SourceState;}
        } restore(*work);
        if (!PrepareHdrFrame(settings.HDR)) return false;
        auto& source=work->Source->Color; auto& scene=HdrSurfaces[HdrScene]; Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(scene.Resource.Get(),source.Resource.Get());
        BindSceneDescriptorHeaps(); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        bool main=true; if (!ApplyHdr(settings.HDR) || !ApplyPostProcessing(settings,main)) return false;
        auto* result=main?MainColor.Get():PostColor.Get(); auto& state=main?MainColorState:PostColorState; Transition(result,state,D3D12_RESOURCE_STATE_COPY_SOURCE); if(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics){if(!PublishGpuColor(result))return false;}else CopyToReadback(result,work->Readback);
        status=CommandList->Close(); if (FAILED(status)) return FailHRESULT("Close TSR display commands",status);
        TSRPendingDisplay=work; ID3D12CommandList* commands[]{CommandList.Get()}; CommandQueue->ExecuteCommandLists(1,commands); work->Submitted=true; SubmittedWorkPending=true;
        if (!SignalAndWait() || (!(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics) && !ReadColor(work->Readback,CreateInfo.Width,CreateInfo.Height,PendingFrame.ColorRGBA8))) return false;
        PendingFrame.Width=work->Source->Info.Width; PendingFrame.Height=work->Source->Info.Height; PendingFrame.Statistics.PassCount=PendingFrame.Statistics.PostProcessPassCount;
    }
    TSRPendingDisplay.reset(); output=std::move(work->Capture); return true;
}
bool SnapshotTSROpaque()
{
    if (!TSRScene.Active || TSRScene.OpaqueReady || !OpaqueEffectsComplete() || !SceneColorWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved)) return Fail("TSR snapshot requires completed opaque effects exactly once");
    if (!CreateHdrSurface(TSROpaque,CreateInfo.Width,CreateInfo.Height)) return false;
    // Restore both states so discarding this unsubmitted snapshot is coherent.
    auto& source=HdrSurfaces[HdrScene]; const auto original=source.State,targetOriginal=TSROpaque.State; Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(TSROpaque.Resource.Get(),TSROpaque.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(TSROpaque.Resource.Get(),source.Resource.Get()); Transition(source.Resource.Get(),source.State,original); Transition(TSROpaque.Resource.Get(),TSROpaque.State,targetOriginal); TSRScene.OpaqueReady=true; return true;
}
bool FinishTSRScene()
{
    if (!TSRScene.Active) return true;
    TSRScene.MotionReady=VelocityCaptured && (FrameSampleCount==1 || SamplesVelocityResolved);
    if (!TSRScene.OpaqueReady || !TSRScene.MotionReady) return Fail("TSR presentation requires opaque snapshot and resolved physical motion");
    struct Restore { bool& Value; explicit Restore(bool& value):Value(value){Value=false;} ~Restore(){Value=true;} } restore(FrameOpen);
    TSRNativeImages images; images.Flags=TSRTextureMotion;
    images.Images[TSRSceneFinal]=HdrSurfaces[HdrScene].Resource; images.States[TSRSceneFinal]=HdrSurfaces[HdrScene].State;
    images.Images[TSRSceneOpaque]=TSROpaque.Resource; images.States[TSRSceneOpaque]=TSROpaque.State;
    images.Images[TSRSceneMotion]=MotionSurfaces[MotionVelocity].Resource; images.States[TSRSceneMotion]=MotionSurfaces[MotionVelocity].State;
    images.Images[TSRSceneDepth]=MainDepth; images.States[TSRSceneDepth]=MainDepthState;
    TSRResult candidate; const bool reconstructed=ReconstructTSRImages(TSRScene.Frame,images,TSRScene.Committed,candidate);
    HdrSurfaces[HdrScene].State=images.States[TSRSceneFinal]; TSROpaque.State=images.States[TSRSceneOpaque]; MotionSurfaces[MotionVelocity].State=images.States[TSRSceneMotion]; MainDepthState=images.States[TSRSceneDepth];
    if (!reconstructed) { LastError="TSR scene reconstruction: "+LastError; return false; } TSRScene.Pending=candidate.Handle;
    FrameCapture display; if (!RenderTSRDisplay(candidate.Handle,FramePostProcessing,display)) { LastError="TSR scene display: "+LastError; return false; }
    return ComposeTSRSceneCapture(TSRScene.Frame,candidate,std::move(display),PendingFrame,LastError);
}
