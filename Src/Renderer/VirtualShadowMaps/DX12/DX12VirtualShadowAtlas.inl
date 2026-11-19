// Candidate atlases own depth and descriptors. Borrowed caster resources are
// retained by each submission; the public handle is published after its fence.
struct VirtualShadowAtlasRecord
{
    VirtualShadowSnapshot Info;
    ComPtr<ID3D12Resource> Depth,Directory;
    ComPtr<ID3D12DescriptorHeap> DepthView;
};
struct VirtualShadowAtlasWork
{
    ComPtr<ID3D12CommandAllocator> Allocator;
    ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12DescriptorHeap> Views;
    ComPtr<ID3D12Resource> Feedback,Readback;
    std::vector<ComPtr<ID3D12Resource>> Retained;
    ComPtr<ID3D12Resource> Constants;
    std::uint8_t* ConstantsMapped=nullptr;
    std::size_t ConstantsUsed=0;
    ~VirtualShadowAtlasWork() { if (ConstantsMapped) Constants->Unmap(0,nullptr); }
    std::shared_ptr<VirtualShadowAtlasRecord> Candidate,Parent;
    std::unordered_map<std::uint32_t,D3D12_RESOURCE_STATES> BufferStates,TextureStates;
};
std::unordered_map<std::uint64_t,std::shared_ptr<VirtualShadowAtlasRecord>> VirtualShadowAtlases;
std::shared_ptr<VirtualShadowAtlasWork> VirtualShadowAtlasPending;
ComPtr<ID3D12RootSignature> VirtualAtlasRoot;
std::array<ComPtr<ID3D12PipelineState>,3> VirtualAtlasRaster;
ComPtr<ID3D12PipelineState> VirtualAtlasCopy,VirtualAtlasRead;
static constexpr auto VirtualAtlasSampleState=static_cast<D3D12_RESOURCE_STATES>(
    unsigned(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)|unsigned(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
static constexpr unsigned VirtualAtlasViewsPerDraw=4;
void DestroyVirtualShadowAtlases()
{
    DestroyVirtualShadowLookup();
    VirtualShadowAtlasPending.reset(); VirtualShadowAtlases.clear(); VirtualAtlasRoot.Reset();
    for (auto& pipeline:VirtualAtlasRaster) pipeline.Reset(); VirtualAtlasCopy.Reset(); VirtualAtlasRead.Reset();
}
bool EnsureVirtualShadowAtlas()
{
    if (!VirtualAtlasRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors=3;
        ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors=1;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[0].Descriptor={0,0};
        for (unsigned slot=0;slot<2;++slot)
        { parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+1].DescriptorTable={1,&ranges[slot]}; }
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; sampler.MaxAnisotropy=1;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=3; desc.pParameters=parameters; desc.NumStaticSamplers=1; desc.pStaticSamplers=&sampler;
        desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize virtual atlas root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(VirtualAtlasRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create virtual atlas root",status);
    }
    if (VirtualAtlasCopy && VirtualAtlasRead && VirtualAtlasRaster[0] && VirtualAtlasRaster[1] && VirtualAtlasRaster[2]) return true;
    if (!LoadTechniqueShaderSource("DX12","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_VIRTUAL_SHADOW_RASTER_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> vertex,pixel,copyVertex,copyPixel,read;
    if (!CompileShader("VSDepth","vs_5_0",vertex,macros,true) || !CompileShader("PSDepth","ps_5_0",pixel,macros,true) ||
        !CompileShader("VSCopy","vs_5_0",copyVertex,macros,true) || !CompileShader("PSCopy","ps_5_0",copyPixel,macros,true) || !CompileShader("CSRead","cs_5_0",read,macros,true)) return false;
    const D3D12_INPUT_ELEMENT_DESC elements[]{{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=VirtualAtlasRoot.Get(); desc.SampleMask=UINT_MAX;
    desc.SampleDesc.Count=1; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.DSVFormat=DXGI_FORMAT_D32_FLOAT;
    desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthEnable=TRUE; desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL;
    desc.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS};
    desc.DepthStencilState.BackFace=desc.DepthStencilState.FrontFace;
    for (auto& blend:desc.BlendState.RenderTarget)
    { blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO; blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; }
    desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()}; desc.InputLayout={elements,3};
    for (unsigned index=0;index<VirtualAtlasRaster.size();++index)
    {
        desc.RasterizerState.CullMode=index?D3D12_CULL_MODE_BACK:D3D12_CULL_MODE_NONE; desc.RasterizerState.FrontCounterClockwise=index==2?FALSE:TRUE;
        const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(VirtualAtlasRaster[index].ReleaseAndGetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create virtual atlas depth pipeline",status);
    }
    desc.VS={copyVertex->GetBufferPointer(),copyVertex->GetBufferSize()}; desc.PS={copyPixel->GetBufferPointer(),copyPixel->GetBufferSize()}; desc.InputLayout={};
    desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(VirtualAtlasCopy.ReleaseAndGetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create virtual atlas copy pipeline",status);
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute{}; compute.pRootSignature=VirtualAtlasRoot.Get(); compute.CS={read->GetBufferPointer(),read->GetBufferSize()};
    status=Evans::Renderer::CreateCachedComputePipelineState(Device,&compute,IID_PPV_ARGS(VirtualAtlasRead.ReleaseAndGetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create virtual atlas read pipeline",status);
}
void VirtualAtlasBarrier(VirtualShadowAtlasWork& work,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
{
    if (before==after) return;
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; work.Commands->ResourceBarrier(1,&barrier);
}
bool CreateVirtualAtlasWork(VirtualShadowAtlasWork& work,unsigned tables,std::size_t feedbackCount,bool readback=true)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors=tables*VirtualAtlasViewsPerDraw; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work.Views.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create virtual atlas submission views",status);
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create virtual atlas commands",status);
    if (!feedbackCount) return true;
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(feedbackCount*sizeof(float)); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work.Feedback.GetAddressOf()));
    return (SUCCEEDED(status) || FailHRESULT("Create virtual atlas feedback",status)) &&
        (!readback || CreateBufferResource(feedbackCount*sizeof(float),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work.Readback));
}
void VirtualAtlasViews(VirtualShadowAtlasWork& work,unsigned table,ID3D12Resource* diffuse,ID3D12Resource* opacity,ID3D12Resource* depth)
{
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=work.Views->GetCPUDescriptorHandleForHeapStart(); cpu.ptr+=std::size_t(table)*VirtualAtlasViewsPerDraw*increment;
    ID3D12Resource* sources[]{diffuse,opacity,depth};
    for (unsigned slot=0;slot<3;++slot)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{}; desc.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        desc.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; desc.Texture2D.MipLevels=1;
        desc.Format=slot==2?DXGI_FORMAT_R32_FLOAT:DXGI_FORMAT_R8G8B8A8_UNORM;
        Device->CreateShaderResourceView(sources[slot],&desc,cpu); cpu.ptr+=increment;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC target{}; target.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; target.Buffer.StructureByteStride=sizeof(float);
    target.Buffer.NumElements=work.Feedback?static_cast<UINT>(work.Feedback->GetDesc().Width/sizeof(float)):1;
    Device->CreateUnorderedAccessView(work.Feedback.Get(),nullptr,&target,cpu);
}
void BindVirtualAtlasTable(VirtualShadowAtlasWork& work,unsigned table,bool compute)
{
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto gpu=work.Views->GetGPUDescriptorHandleForHeapStart(); gpu.ptr+=std::uint64_t(table)*VirtualAtlasViewsPerDraw*increment;
    if (compute) work.Commands->SetComputeRootDescriptorTable(1,gpu); else work.Commands->SetGraphicsRootDescriptorTable(1,gpu);
    gpu.ptr+=3*increment;
    if (compute) work.Commands->SetComputeRootDescriptorTable(2,gpu); else work.Commands->SetGraphicsRootDescriptorTable(2,gpu);
}
bool PrepareVirtualAtlasCasterResources(VirtualShadowAtlasWork& work,const VirtualShadowCaster& caster)
{
    // Upload only dirty native resources. GPU-generated vertex/index buffers
    // already own their authoritative contents and are consumed directly.
    const D3D12_RANGE noRead{0,0};
    for (auto handle:{caster.Vertices,caster.Indices}) if (!work.BufferStates.count(handle.Value))
    {
        auto& source=Buffers.at(handle.Value); auto state=source.State; work.Retained.push_back(source.Resource);
        if (source.Dirty)
        {
            ComPtr<ID3D12Resource> staging; if (!CreateBufferResource(source.Bytes.size(),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,staging)) return false;
            void* mapped=nullptr; const auto status=staging->Map(0,&noRead,&mapped); if (FAILED(status)) return FailHRESULT("Map virtual caster upload",status);
            std::memcpy(mapped,source.Bytes.data(),source.Bytes.size()); staging->Unmap(0,nullptr);
            VirtualAtlasBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_COPY_DEST); state=D3D12_RESOURCE_STATE_COPY_DEST;
            work.Commands->CopyBufferRegion(source.Resource.Get(),0,staging.Get(),0,source.Bytes.size()); work.Retained.push_back(staging);
        }
        const auto ready=FinalBufferState(source.Description.Usage); VirtualAtlasBarrier(work,source.Resource.Get(),state,ready); work.BufferStates.emplace(handle.Value,ready);
    }
    for (auto handle:{caster.Diffuse,caster.Opacity}) if (handle && !work.TextureStates.count(handle.Value))
    {
        auto& source=Textures.at(handle.Value); auto state=source.State; work.Retained.push_back(source.Resource);
        if (source.Dirty)
        {
            ComPtr<ID3D12Resource> staging; if (!CreateBufferResource(static_cast<std::size_t>(source.Readback.TotalSize),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,staging)) return false;
            void* mapped=nullptr; const auto status=staging->Map(0,&noRead,&mapped); if (FAILED(status)) return FailHRESULT("Map virtual coverage upload",status);
            const auto rowBytes=std::size_t(source.Description.Width)*4;
            for (int row=0;row<source.Description.Height;++row)
                std::memcpy(static_cast<std::uint8_t*>(mapped)+source.Readback.Footprint.Offset+std::size_t(row)*source.Readback.Footprint.Footprint.RowPitch,source.Bytes.data()+std::size_t(row)*rowBytes,rowBytes);
            staging->Unmap(0,nullptr); VirtualAtlasBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_COPY_DEST); state=D3D12_RESOURCE_STATE_COPY_DEST;
            D3D12_TEXTURE_COPY_LOCATION target{},input{}; target.pResource=source.Resource.Get(); target.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            input.pResource=staging.Get(); input.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; input.PlacedFootprint=source.Readback.Footprint;
            work.Commands->CopyTextureRegion(&target,0,0,0,&input,nullptr); work.Retained.push_back(staging);
        }
        VirtualAtlasBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); work.TextureStates.emplace(handle.Value,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    return true;
}
// Per-draw wind matrices exceed the 64-DWORD root budget. Suballocate aligned
// CBVs in retained upload pages rather than allocating a resource per caster.
bool BindVirtualAtlasConstants(VirtualShadowAtlasWork& work,const VirtualShadowRasterConstants& constants,bool compute)
{
    constexpr std::size_t Capacity=64*1024;
    constexpr std::size_t Stride=(sizeof(VirtualShadowRasterConstants)+255)&~std::size_t(255);
    if (!work.ConstantsMapped || work.ConstantsUsed+Stride>Capacity)
    {
        if (work.ConstantsMapped) { work.Constants->Unmap(0,nullptr); work.ConstantsMapped=nullptr; }
        work.Constants.Reset();
        if (!CreateBufferResource(Capacity,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work.Constants)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0};
        const auto status=work.Constants->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map virtual atlas constants",status);
        work.ConstantsMapped=static_cast<std::uint8_t*>(mapped); work.ConstantsUsed=0;
        work.Retained.push_back(work.Constants);
    }
    std::memcpy(work.ConstantsMapped+work.ConstantsUsed,&constants,sizeof(constants));
    const auto address=work.Constants->GetGPUVirtualAddress()+work.ConstantsUsed; work.ConstantsUsed+=Stride;
    if (compute) work.Commands->SetComputeRootConstantBufferView(0,address);
    else work.Commands->SetGraphicsRootConstantBufferView(0,address);
    return true;
}
bool EncodeVirtualDepthPack(VirtualShadowAtlasWork& work,unsigned width,unsigned height,unsigned table)
{
    auto* commands=work.Commands.Get(); VirtualShadowRasterConstants constants;
    constants.Extent={width,height,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr); commands->SetComputeRootSignature(VirtualAtlasRoot.Get());
    if (!BindVirtualAtlasConstants(work,constants,true)) return false; BindVirtualAtlasTable(work,table,true);
    commands->SetPipelineState(VirtualAtlasRead.Get()); const auto count=std::size_t(constants.Extent.X)*constants.Extent.Y;
    const auto groups=static_cast<unsigned>((count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth);
    commands->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    VirtualAtlasBarrier(work,work.Feedback.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (work.Readback) commands->CopyResource(work.Readback.Get(),work.Feedback.Get());
    return true;
}
bool EncodeVirtualAtlasRead(VirtualShadowAtlasWork& work,const VirtualShadowAtlasRecord& record,unsigned table)
{ return EncodeVirtualDepthPack(work,record.Info.Layout.AtlasWidth,record.Info.Layout.AtlasHeight,table); }
bool CompleteVirtualAtlasWork(const std::shared_ptr<VirtualShadowAtlasWork>& work,std::vector<float>* depth)
{
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close virtual atlas commands",status);
    VirtualShadowAtlasPending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    // Buffer resources decay to COMMON at this submission boundary. Texture
    // transitions persist. Publish these states even if the fence wait fails.
    for (const auto& state:work->BufferStates) { auto& resource=Buffers.at(state.first); resource.State=D3D12_RESOURCE_STATE_COMMON; resource.Dirty=false; }
    for (const auto& state:work->TextureStates) { auto& resource=Textures.at(state.first); resource.State=state.second; resource.Dirty=false; }
    if (!SignalAndWait()) return false;
    VirtualShadowAtlasPending.reset();
    if (depth)
    {
        const auto bytes=static_cast<std::size_t>(work->Readback->GetDesc().Width); const D3D12_RANGE range{0,bytes},noWrite{0,0}; void* mapped=nullptr;
        const auto readStatus=work->Readback->Map(0,&range,&mapped); if (FAILED(readStatus)) return FailHRESULT("Read virtual depth atlas",readStatus);
        std::vector<float> result(bytes/sizeof(float)); std::memcpy(result.data(),mapped,bytes); work->Readback->Unmap(0,&noWrite);
        if (!ValidateVirtualShadowAtlasDepth(work->Candidate->Info,result,LastError)) return false; *depth=std::move(result);
    }
    return true;
}
bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("D3D12 virtual atlas handle is invalid");
    if (!RecoverSubmittedWork("Read virtual atlas") || !WaitForGPU() || !EnsureVirtualShadowAtlas()) return false; VirtualShadowAtlasPending.reset();
    auto work=std::make_shared<VirtualShadowAtlasWork>(); work->Candidate=found->second;
    const auto& layout=work->Candidate->Info.Layout;
    if (!CreateVirtualAtlasWork(*work,1,std::size_t(layout.AtlasWidth)*layout.AtlasHeight)) return false;
    VirtualAtlasViews(*work,0,nullptr,nullptr,work->Candidate->Depth.Get()); auto* heap=work->Views.Get(); work->Commands->SetDescriptorHeaps(1,&heap);
    return EncodeVirtualAtlasRead(*work,*work->Candidate,0) && CompleteVirtualAtlasWork(work,&output);
}
bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!RecoverSubmittedWork("Release virtual atlas")) return false; VirtualShadowAtlasPending.reset();
    return (handle && VirtualShadowAtlases.erase(handle.Value)) || Fail("D3D12 virtual atlas handle is invalid or already released");
}
bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!RecoverSubmittedWork("Render virtual atlas") || !WaitForGPU() || !EnsureVirtualShadowAtlas()) return false; VirtualShadowAtlasPending.reset();
    auto work=std::make_shared<VirtualShadowAtlasWork>();
    if (previous) { const auto found=VirtualShadowAtlases.find(previous.Value); if (found==VirtualShadowAtlases.end()) return Fail("D3D12 virtual atlas parent is invalid"); work->Parent=found->second; }
    VirtualShadowPlan plan; if (!BuildVirtualShadowPlan(upload.Identity,upload.Settings,upload.Requests,upload.Invalidations,work->Parent?&work->Parent->Info:nullptr,plan,LastError)) return false;
    for (const auto& caster:upload.Casters)
    {
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Virtual atlas caster buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto image=Textures.find(texture.Value); if (image==Textures.end() || image->second.Description.Format!=TextureFormat::RGBA8) return Fail("Virtual atlas coverage requires existing color textures"); }
    }
    if (upload.Casters.size()>(D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1/VirtualAtlasViewsPerDraw)-2) return Fail("Virtual atlas caster descriptors exceed native capacity");
    work->Candidate=std::make_shared<VirtualShadowAtlasRecord>(); auto& record=*work->Candidate; record.Info=plan.Candidate; const auto& layout=record.Info.Layout;
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=layout.AtlasWidth; desc.Height=layout.AtlasHeight; desc.DepthOrArraySize=desc.MipLevels=1;
    desc.Format=DXGI_FORMAT_R32_TYPELESS; desc.SampleDesc.Count=1; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{}; clear.Format=DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth=1;
    auto status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(record.Depth.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create immutable virtual depth atlas",status);
    D3D12_DESCRIPTOR_HEAP_DESC targetHeap{}; targetHeap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; targetHeap.NumDescriptors=1;
    status=Device->CreateDescriptorHeap(&targetHeap,IID_PPV_ARGS(record.DepthView.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create virtual atlas target view",status);
    D3D12_DEPTH_STENCIL_VIEW_DESC target{}; target.Format=DXGI_FORMAT_D32_FLOAT; target.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto targetView=record.DepthView->GetCPUDescriptorHandleForHeapStart(); Device->CreateDepthStencilView(record.Depth.Get(),&target,targetView);
    const auto readTable=static_cast<unsigned>(upload.Casters.size())+1;
    if (!CreateVirtualAtlasWork(*work,readTable+1,upload.Readback?std::size_t(layout.AtlasWidth)*layout.AtlasHeight:0)) return false;
    for (const auto& caster:upload.Casters) if (!PrepareVirtualAtlasCasterResources(*work,caster)) return false;
    VirtualAtlasViews(*work,0,nullptr,nullptr,work->Parent?work->Parent->Depth.Get():nullptr);
    for (unsigned index=0;index<upload.Casters.size();++index)
    {
        const auto& caster=upload.Casters[index]; VirtualAtlasViews(*work,index+1,caster.Diffuse?Textures.at(caster.Diffuse.Value).Resource.Get():nullptr,
            caster.Opacity?Textures.at(caster.Opacity.Value).Resource.Get():nullptr,nullptr);
    }
    VirtualAtlasViews(*work,readTable,nullptr,nullptr,record.Depth.Get());
    auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); commands->SetDescriptorHeaps(1,&views); commands->SetGraphicsRootSignature(VirtualAtlasRoot.Get());
    commands->OMSetRenderTargets(0,nullptr,FALSE,&targetView); commands->ClearDepthStencilView(targetView,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr);
    commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
    const auto frame=[&](const VirtualShadowRasterConstants& constants)
    {
        const D3D12_VIEWPORT viewport{constants.Origins.X,constants.Origins.Y,float(stride),float(stride),0,1};
        const D3D12_RECT scissor{LONG(constants.Origins.X),LONG(constants.Origins.Y),LONG(constants.Origins.X)+LONG(stride),LONG(constants.Origins.Y)+LONG(stride)};
        commands->RSSetViewports(1,&viewport); commands->RSSetScissorRects(1,&scissor); return BindVirtualAtlasConstants(*work,constants,false);
    };
    commands->SetPipelineState(VirtualAtlasCopy.Get()); BindVirtualAtlasTable(*work,0,false);
    for (const auto& copy:plan.Copies) { if (!frame(BuildVirtualShadowCopyConstants(work->Parent->Info.Layout,layout,copy))) return false; commands->DrawInstanced(3,1,0,0); }
    VirtualShadowAtlasResult result;
    for (const auto& tile:plan.Dirty) for (unsigned index=0;index<upload.Casters.size();++index)
    {
        const auto& caster=upload.Casters[index]; VirtualShadowRasterConstants constants;
        if (!BuildVirtualShadowRasterConstants(record.Info,tile,caster,constants,LastError) || !frame(constants)) return false;
        const unsigned raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0;
        commands->SetPipelineState(VirtualAtlasRaster[raster].Get()); BindVirtualAtlasTable(*work,index+1,false);
        commands->IASetVertexBuffers(0,1,&Buffers.at(caster.Vertices.Value).VertexView); commands->IASetIndexBuffer(&Buffers.at(caster.Indices.Value).IndexView);
        commands->DrawIndexedInstanced(caster.IndexCount,1,caster.FirstIndex,0,0); ++result.DrawCalls;
    }
    commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr); VirtualAtlasBarrier(*work,record.Depth.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE,VirtualAtlasSampleState);
    if (upload.Readback && !EncodeVirtualAtlasRead(*work,record,readTable)) return false;
    if (!CompleteVirtualAtlasWork(work,upload.Readback?&result.Depth:nullptr)) return false;
    std::vector<unsigned> rendered; for (const auto& dirty:plan.Dirty) rendered.push_back(dirty.Page);
    if (!CompleteVirtualShadowPlan(plan,rendered,record.Info,LastError)) return false;
    if (!CreateVirtualShadowDirectory(work->Candidate)) return false;
    result.Handle=NewVirtualShadowAtlasHandle(); if (!result.Handle) return Fail("Virtual atlas identity space is exhausted");
    result.Info=record.Info; result.CopiedPages=static_cast<unsigned>(plan.Copies.size()); result.RenderedPages=static_cast<unsigned>(plan.Dirty.size());
    VirtualShadowAtlases.emplace(result.Handle.Value,work->Candidate); output=std::move(result); return true;
}
