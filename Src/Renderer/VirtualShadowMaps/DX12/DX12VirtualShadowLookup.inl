// The frame retains the immutable atlas until subsequent native work can safely
// replace its bindings. Releasing a public handle cannot invalidate encoded draws.
std::shared_ptr<VirtualShadowAtlasRecord> FrameVirtualShadowAtlas;
VirtualShadowSamplingFrame FrameVirtualShadowSampling;
std::array<UINT,2> FrameVirtualShadowViews{InvalidDescriptor,InvalidDescriptor};
bool PrepareVirtualShadowSampling(const VirtualShadowSamplingFrame& frame,const CascadedShadowState& cascades)
{
    if (!ValidateVirtualShadowSamplingFrame(frame,LastError)) return false;
    std::shared_ptr<VirtualShadowAtlasRecord> candidate;
    if (frame.Atlas)
    {
        if (cascades.Frame.CascadeCount) return Fail("Virtual and cascaded shadow representations are mutually exclusive");
        if (FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::Hard && FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::PCF)
            return Fail("Virtual scene sampling currently requires hard or PCF filtering");
        const auto found=VirtualShadowAtlases.find(frame.Atlas.Value);
        if (found==VirtualShadowAtlases.end()) return Fail("Virtual scene sampling requires a current atlas handle");
        candidate=found->second;
        for (unsigned index=0;index<FrameVirtualShadowViews.size();++index)
        {
            auto& descriptor=FrameVirtualShadowViews[index]; if (descriptor==InvalidDescriptor) descriptor=SRVArena.Allocate();
            if (descriptor==InvalidDescriptor) return Fail("DX12 virtual receiver descriptors are exhausted");
            D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
            Device->CreateShaderResourceView(index?candidate->Directory.Get():candidate->Depth.Get(),&view,SRVArena.Cpu(descriptor));
        }
    }
    FrameVirtualShadowAtlas=std::move(candidate); FrameVirtualShadowSampling=frame; return true;
}
ComPtr<ID3D12PipelineState> VirtualShadowLookupPipeline;
void DestroyVirtualShadowLookup() { FrameVirtualShadowAtlas.reset(); FrameVirtualShadowSampling={}; for (auto& descriptor:FrameVirtualShadowViews) { SRVArena.Release(descriptor); descriptor=InvalidDescriptor; } VirtualShadowLookupPipeline.Reset(); }
bool CreateVirtualShadowDirectory(const std::shared_ptr<VirtualShadowAtlasRecord>& record)
{
    VirtualShadowDirectory directory; if (!BuildVirtualShadowDirectory(record->Info,directory,LastError)) return false;
    auto work=std::make_shared<VirtualShadowAtlasWork>(); work->Candidate=record;
    if (!CreateVirtualAtlasWork(*work,1,0)) return false;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=directory.Width; desc.Height=directory.Height; desc.DepthOrArraySize=desc.MipLevels=1; desc.SampleDesc.Count=1; desc.Format=DXGI_FORMAT_R32_FLOAT;
    auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(record->Directory.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create immutable virtual page directory",status);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes=0; Device->GetCopyableFootprints(&desc,0,1,0,&footprint,nullptr,nullptr,&bytes);
    ComPtr<ID3D12Resource> staging; if (!CreateBufferResource(static_cast<std::size_t>(bytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,staging)) return false;
    const D3D12_RANGE noRead{0,0}; void* mapped=nullptr; status=staging->Map(0,&noRead,&mapped); if (FAILED(status)) return FailHRESULT("Map virtual directory upload",status);
    for (unsigned row=0;row<directory.Height;++row)
        std::memcpy(static_cast<std::uint8_t*>(mapped)+footprint.Offset+std::size_t(row)*footprint.Footprint.RowPitch,
            directory.Depth.data()+std::size_t(row)*directory.Width,directory.Width*sizeof(float));
    staging->Unmap(0,nullptr); work->Retained.push_back(staging);
    D3D12_TEXTURE_COPY_LOCATION source{},target{}; source.pResource=staging.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint=footprint;
    target.pResource=record->Directory.Get(); target.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    work->Commands->CopyTextureRegion(&target,0,0,0,&source,nullptr); VirtualAtlasBarrier(*work,record->Directory.Get(),D3D12_RESOURCE_STATE_COPY_DEST,VirtualAtlasSampleState);
    return CompleteVirtualAtlasWork(work,nullptr);
}
bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Virtual lookup atlas handle is invalid");
    VirtualShadowLookupConstants constants;
    if (!BuildVirtualShadowLookupConstants(found->second->Info,upload,constants,LastError) || !WaitForGPU() || !EnsureVirtualShadowAtlas()) return false;
    VirtualShadowAtlasPending.reset();
    if (!VirtualShadowLookupPipeline)
    {
        if (!LoadTechniqueShaderSource("DX12","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO definitions[]{{"EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE","1"},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code; if (!CompileShader("CSLookup","cs_5_0",code,definitions,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=VirtualAtlasRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(VirtualShadowLookupPipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create virtual shadow lookup pipeline",status);
    }
    auto work=std::make_shared<VirtualShadowAtlasWork>(); work->Candidate=found->second; const auto bytes=upload.Samples.size()*sizeof(Float4);
    if (!CreateVirtualAtlasWork(*work,1,bytes/sizeof(float))) return false;
    ComPtr<ID3D12Resource> queries; if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,queries)) return false;
    const D3D12_RANGE noRead{0,0}; void* mapped=nullptr; auto status=queries->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Map virtual lookup queries",status);
    std::memcpy(mapped,upload.Samples.data(),bytes); queries->Unmap(0,nullptr); work->Retained.push_back(queries);
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart(); D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format=DXGI_FORMAT_R32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
    Device->CreateShaderResourceView(work->Candidate->Depth.Get(),&view,cpu); cpu.ptr+=increment;
    Device->CreateShaderResourceView(work->Candidate->Directory.Get(),&view,cpu); cpu.ptr+=increment;
    view={}; view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER; view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.NumElements=static_cast<UINT>(upload.Samples.size()); view.Buffer.StructureByteStride=sizeof(Float4); Device->CreateShaderResourceView(queries.Get(),&view,cpu); cpu.ptr+=increment;
    D3D12_UNORDERED_ACCESS_VIEW_DESC resultView{}; resultView.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    resultView.Buffer.NumElements=static_cast<UINT>(upload.Samples.size()); resultView.Buffer.StructureByteStride=sizeof(Float4); Device->CreateUnorderedAccessView(work->Feedback.Get(),nullptr,&resultView,cpu);
    auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); commands->SetDescriptorHeaps(1,&views);
    commands->SetComputeRootSignature(VirtualAtlasRoot.Get()); commands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0); BindVirtualAtlasTable(*work,0,true);
    commands->SetPipelineState(VirtualShadowLookupPipeline.Get()); const auto groups=(constants.Filtering.Z+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    commands->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    VirtualAtlasBarrier(*work,work->Feedback.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); commands->CopyResource(work->Readback.Get(),work->Feedback.Get());
    if (!CompleteVirtualAtlasWork(work,nullptr)) return false;
    const D3D12_RANGE read{0,bytes},noWrite{0,0}; status=work->Readback->Map(0,&read,&mapped); if (FAILED(status)) return FailHRESULT("Read virtual shadow lookup",status);
    VirtualShadowLookupResult result; result.Samples.resize(upload.Samples.size()); std::memcpy(result.Samples.data(),mapped,bytes); work->Readback->Unmap(0,&noWrite);
    if (!ValidateVirtualShadowLookupResult(work->Candidate->Info,upload,result,LastError)) return false; output=std::move(result); return true;
}
