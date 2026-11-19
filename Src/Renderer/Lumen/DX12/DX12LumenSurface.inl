struct LumenSurfaceRecord
{
    LumenSurfaceIdentity Identity; LumenSurfaceLayout Layout;
    std::vector<LumenSurfaceCapture> Captures;
    std::array<ComPtr<ID3D12Resource>,LumenSurfacePlanes> Images;
    ComPtr<ID3D12Resource> Depth; ComPtr<ID3D12DescriptorHeap> Targets,DepthView;
};
struct LumenSurfaceWork
{
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12DescriptorHeap> Views; ComPtr<ID3D12Resource> Constants,Feedback,Readback;
    std::shared_ptr<LumenSurfaceRecord> Candidate;
    std::vector<ComPtr<ID3D12Resource>> Retained;
    std::unordered_map<std::uint32_t,D3D12_RESOURCE_STATES> BufferStates,TextureStates;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenSurfaceRecord>> LumenSurfaces;
std::shared_ptr<LumenSurfaceWork> LumenSurfacePending;
ComPtr<ID3D12RootSignature> LumenSurfaceRoot;
std::array<ComPtr<ID3D12PipelineState>,4> LumenCapturePipelines;
ComPtr<ID3D12PipelineState> LumenMaterialRead;
static constexpr unsigned LumenViewsPerTable=4,LumenConstantStride=256;
static constexpr auto LumenSampleState=static_cast<D3D12_RESOURCE_STATES>(
    unsigned(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)|unsigned(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
void DestroyLumenSurfaces()
{
    DestroyLumenCaptureAsync();
    LumenSurfacePending.reset(); LumenSurfaces.clear(); LumenSurfaceRoot.Reset(); LumenMaterialRead.Reset();
    for (auto& pipeline:LumenCapturePipelines) pipeline.Reset();
}
bool EnsureLumenSurface()
{
    if (!LumenSurfaceRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{}; ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors=LumenSurfacePlanes;
        ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors=1;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (unsigned table=0;table<2;++table)
        { parameters[table+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[table+1].DescriptorTable={1,&ranges[table]}; }
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP; sampler.MaxLOD=D3D12_FLOAT32_MAX;
        sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; sampler.MaxAnisotropy=1;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=3; desc.pParameters=parameters; desc.NumStaticSamplers=1; desc.pStaticSamplers=&sampler;
        desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen material root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenSurfaceRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen material root",status);
    }
    if (LumenMaterialRead && std::all_of(LumenCapturePipelines.begin(),LumenCapturePipelines.end(),[](const auto& p){return bool(p);})) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel,read; const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_READ_STAGE","1"},{nullptr,nullptr}};
    if (!CompileShader("VSCapture","vs_5_0",vertex,nullptr,true) || !CompileShader("PSCapture","ps_5_0",pixel,nullptr,true) || !CompileShader("CSRead","cs_5_0",read,macros,true)) return false;
    const D3D12_INPUT_ELEMENT_DESC input[]{{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenSurfaceRoot.Get(); desc.SampleMask=UINT_MAX; desc.SampleDesc.Count=1;
    desc.NumRenderTargets=LumenSurfacePlanes; for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) desc.RTVFormats[slot]=DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.DSVFormat=DXGI_FORMAT_D32_FLOAT; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthEnable=TRUE; desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL;
    desc.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS}; desc.DepthStencilState.BackFace=desc.DepthStencilState.FrontFace;
    for (auto& blend:desc.BlendState.RenderTarget)
    { blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO; blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL; }
    desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()}; desc.InputLayout={input,4};
    for (unsigned mode=0;mode<LumenCapturePipelines.size();++mode)
    {
        desc.RasterizerState.CullMode=mode>=2?D3D12_CULL_MODE_BACK:D3D12_CULL_MODE_NONE; desc.RasterizerState.FrontCounterClockwise=(mode&1)?FALSE:TRUE;
        const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenCapturePipelines[mode].ReleaseAndGetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen material pipeline",status);
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute{}; compute.pRootSignature=LumenSurfaceRoot.Get(); compute.CS={read->GetBufferPointer(),read->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&compute,IID_PPV_ARGS(LumenMaterialRead.ReleaseAndGetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen material export pipeline",status);
}
void LumenBarrier(LumenSurfaceWork& work,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
{
    if (before==after) return;
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; work.Commands->ResourceBarrier(1,&barrier);
}
bool CreateLumenWork(LumenSurfaceWork& work,unsigned tables,const std::vector<std::uint8_t>& constants,bool readback)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=tables*LumenViewsPerTable; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work.Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen material descriptors",status);
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen capture commands",status);
    if (!CreateBufferResource(constants.size(),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work.Constants)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; status=work.Constants->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Lumen capture constants",status);
    std::memcpy(mapped,constants.data(),constants.size()); work.Constants->Unmap(0,nullptr);
    if (!readback) return true;
    const auto& layout=work.Candidate->Layout; const auto bytes=std::size_t(layout.Width)*layout.Height*sizeof(LumenSurfaceTexel);
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work.Feedback.GetAddressOf()));
    return (SUCCEEDED(status) || FailHRESULT("Create Lumen material feedback",status)) && CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work.Readback);
}
void LumenViews(LumenSurfaceWork& work,unsigned table,const std::array<ID3D12Resource*,LumenSurfacePlanes>& sources,
    const std::array<DXGI_FORMAT,LumenSurfacePlanes>& formats)
{
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=work.Views->GetCPUDescriptorHandleForHeapStart(); cpu.ptr+=std::size_t(table)*LumenViewsPerTable*increment;
    for (unsigned slot=0;slot<LumenSurfacePlanes;++slot)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{}; desc.Format=formats[slot]; desc.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        desc.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; desc.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(sources[slot],&desc,cpu); cpu.ptr+=increment;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; uav.Buffer.StructureByteStride=sizeof(Float4);
    uav.Buffer.NumElements=work.Feedback?UINT(work.Feedback->GetDesc().Width/sizeof(Float4)):1;
    Device->CreateUnorderedAccessView(work.Feedback.Get(),nullptr,&uav,cpu);
}
void BindLumenViews(LumenSurfaceWork& work,unsigned table,bool compute)
{
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto gpu=work.Views->GetGPUDescriptorHandleForHeapStart(); gpu.ptr+=std::uint64_t(table)*LumenViewsPerTable*increment;
    if (compute) work.Commands->SetComputeRootDescriptorTable(1,gpu); else work.Commands->SetGraphicsRootDescriptorTable(1,gpu);
    gpu.ptr+=LumenSurfacePlanes*increment;
    if (compute) work.Commands->SetComputeRootDescriptorTable(2,gpu); else work.Commands->SetGraphicsRootDescriptorTable(2,gpu);
}
bool PrepareLumenDraw(LumenSurfaceWork& work,const LumenSurfaceDraw& draw)
{
    const D3D12_RANGE noRead{0,0};
    for (auto handle:{draw.Vertices,draw.Indices}) if (!work.BufferStates.count(handle.Value))
    {
        auto& source=Buffers.at(handle.Value); auto state=source.State; work.Retained.push_back(source.Resource);
        if (source.Dirty)
        {
            ComPtr<ID3D12Resource> staging; if (!CreateBufferResource(source.Bytes.size(),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,staging)) return false;
            void* mapped=nullptr; const auto status=staging->Map(0,&noRead,&mapped); if (FAILED(status)) return FailHRESULT("Map Lumen geometry upload",status);
            std::memcpy(mapped,source.Bytes.data(),source.Bytes.size()); staging->Unmap(0,nullptr);
            LumenBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_COPY_DEST); state=D3D12_RESOURCE_STATE_COPY_DEST;
            work.Commands->CopyBufferRegion(source.Resource.Get(),0,staging.Get(),0,source.Bytes.size()); work.Retained.push_back(staging);
        }
        const auto ready=FinalBufferState(source.Description.Usage); LumenBarrier(work,source.Resource.Get(),state,ready); work.BufferStates.emplace(handle.Value,ready);
    }
    for (auto handle:{draw.Diffuse,draw.Normal,draw.Opacity}) if (handle && !work.TextureStates.count(handle.Value))
    {
        auto& source=Textures.at(handle.Value); auto state=source.State; work.Retained.push_back(source.Resource);
        if (source.Dirty)
        {
            ComPtr<ID3D12Resource> staging; if (!CreateBufferResource(std::size_t(source.Readback.TotalSize),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,staging)) return false;
            void* mapped=nullptr; const auto status=staging->Map(0,&noRead,&mapped); if (FAILED(status)) return FailHRESULT("Map Lumen material upload",status);
            const auto rowBytes=std::size_t(source.Description.Width)*4;
            for (int row=0;row<source.Description.Height;++row)
                std::memcpy(static_cast<std::uint8_t*>(mapped)+source.Readback.Footprint.Offset+std::size_t(row)*source.Readback.Footprint.Footprint.RowPitch,source.Bytes.data()+std::size_t(row)*rowBytes,rowBytes);
            staging->Unmap(0,nullptr); LumenBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_COPY_DEST); state=D3D12_RESOURCE_STATE_COPY_DEST;
            D3D12_TEXTURE_COPY_LOCATION destination{},input{}; destination.pResource=source.Resource.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            input.pResource=staging.Get(); input.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; input.PlacedFootprint=source.Readback.Footprint;
            work.Commands->CopyTextureRegion(&destination,0,0,0,&input,nullptr); work.Retained.push_back(staging);
        }
        LumenBarrier(work,source.Resource.Get(),state,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); work.TextureStates.emplace(handle.Value,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    return true;
}
void EncodeLumenRead(LumenSurfaceWork& work)
{
    auto* commands=work.Commands.Get(); const auto& layout=work.Candidate->Layout;
    commands->SetComputeRootSignature(LumenSurfaceRoot.Get()); commands->SetPipelineState(LumenMaterialRead.Get());
    commands->SetComputeRootConstantBufferView(0,work.Constants->GetGPUVirtualAddress()); BindLumenViews(work,0,true);
    const auto count=std::size_t(layout.Width)*layout.Height; const unsigned groups=unsigned((count+LumenSurfaceReadThreads-1)/LumenSurfaceReadThreads);
    commands->Dispatch(std::min(groups,LumenSurfaceReadColumns),(groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns,1);
    LumenBarrier(work,work.Feedback.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); commands->CopyResource(work.Readback.Get(),work.Feedback.Get());
}
bool CompleteLumenWork(const std::shared_ptr<LumenSurfaceWork>& work,std::vector<LumenSurfaceTexel>* texels)
{
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen capture commands",status);
    LumenSurfacePending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    for (const auto& entry:work->BufferStates) { auto& resource=Buffers.at(entry.first); resource.State=D3D12_RESOURCE_STATE_COMMON; resource.Dirty=false; }
    for (const auto& entry:work->TextureStates) { auto& resource=Textures.at(entry.first); resource.State=entry.second; resource.Dirty=false; }
    if (!SignalAndWait()) return false; LumenSurfacePending.reset();
    if (texels)
    {
        const auto bytes=std::size_t(work->Readback->GetDesc().Width); std::vector<LumenSurfaceTexel> result(bytes/sizeof(LumenSurfaceTexel));
        const D3D12_RANGE range{0,bytes},noWrite{0,0}; void* mapped=nullptr; const auto mappedStatus=work->Readback->Map(0,&range,&mapped);
        if (FAILED(mappedStatus)) return FailHRESULT("Read Lumen material atlas",mappedStatus);
        std::memcpy(result.data(),mapped,bytes); work->Readback->Unmap(0,&noWrite);
        if (!ValidateLumenSurfaceTexels(work->Candidate->Layout,result,LastError)) return false; *texels=std::move(result);
    }
    return true;
}
void LumenReadViews(LumenSurfaceWork& work)
{
    LumenViews(work,0,{work.Candidate->Images[0].Get(),work.Candidate->Images[1].Get(),work.Candidate->Images[2].Get()},
        {DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT});
}
bool CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!RecoverSubmittedWork("Capture Lumen surface") || !WaitForGPU() || !EnsureLumenSurface()) return false; LumenSurfacePending.reset();
    LumenSurfaceLayout layout; if (!ValidateLumenSurfaceUpload(upload,layout,LastError)) return false;
    auto work=std::make_shared<LumenSurfaceWork>(); work->Candidate=std::make_shared<LumenSurfaceRecord>(); auto& record=*work->Candidate;
    record.Layout=layout; record.Identity=upload.Identity; record.Captures=upload.Cards;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    std::vector<std::uint8_t> constants(LumenConstantStride); ShaderUInt4 extent{layout.Width,layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    std::memcpy(constants.data(),&extent,sizeof(extent));
    for (const auto& card:upload.Cards) for (unsigned draw=card.FirstDraw;draw<std::uint64_t(card.FirstDraw)+card.DrawCount;++draw)
    {
        LumenSurfaceConstants frame; if (!BuildLumenSurfaceConstants(layout,card,upload.Draws[draw],frame,LastError)) return false;
        const auto offset=constants.size(); constants.resize(offset+LumenConstantStride); std::memcpy(constants.data()+offset,&frame,sizeof(frame));
    }
    if (!CreateLumenWork(*work,unsigned(upload.Draws.size())+1,constants,upload.Readback)) return false;
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_RESOURCE_DESC image{}; image.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image.Width=layout.Width; image.Height=layout.Height; image.DepthOrArraySize=1; image.MipLevels=1; image.SampleDesc.Count=1;
    image.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; image.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap.NumDescriptors=LumenSurfacePlanes;
    auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(record.Targets.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen color target descriptors",status);
    auto rtv=record.Targets->GetCPUDescriptorHandleForHeapStart(); const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE targets[LumenSurfacePlanes]; const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        D3D12_CLEAR_VALUE value{}; value.Format=image.Format; std::copy(clear[plane],clear[plane]+4,value.Color);
        status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&image,D3D12_RESOURCE_STATE_RENDER_TARGET,&value,IID_PPV_ARGS(record.Images[plane].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create immutable Lumen material target",status);
        targets[plane]=rtv; Device->CreateRenderTargetView(record.Images[plane].Get(),nullptr,rtv); rtv.ptr+=increment;
    }
    image.Format=DXGI_FORMAT_D32_FLOAT; image.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL; D3D12_CLEAR_VALUE depthClear{}; depthClear.Format=image.Format; depthClear.DepthStencil.Depth=1;
    status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&image,D3D12_RESOURCE_STATE_DEPTH_WRITE,&depthClear,IID_PPV_ARGS(record.Depth.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen capture depth",status);
    heap={}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; heap.NumDescriptors=1;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(record.DepthView.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen depth descriptor",status);
    const auto depthView=record.DepthView->GetCPUDescriptorHandleForHeapStart(); Device->CreateDepthStencilView(record.Depth.Get(),nullptr,depthView);
    for (unsigned index=0;index<upload.Draws.size();++index)
    {
        const auto& draw=upload.Draws[index]; if (!PrepareLumenDraw(*work,draw)) return false;
        std::array<ID3D12Resource*,LumenSurfacePlanes> sources{}; std::array<DXGI_FORMAT,LumenSurfacePlanes> formats;
        const TextureHandle handles[]{draw.Diffuse,draw.Normal,draw.Opacity};
        for (unsigned slot=0;slot<LumenSurfacePlanes;++slot)
        {
            formats[slot]=DXGI_FORMAT_R8G8B8A8_UNORM;
            if (handles[slot]) { const auto& texture=Textures.at(handles[slot].Value); sources[slot]=texture.Resource.Get();
                if (texture.Description.ColorSpace==TextureColorSpace::SRGB) formats[slot]=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; }
        }
        LumenViews(*work,index+1,sources,formats);
    }
    LumenReadViews(*work); auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); commands->SetDescriptorHeaps(1,&views);
    commands->SetGraphicsRootSignature(LumenSurfaceRoot.Get()); commands->OMSetRenderTargets(LumenSurfacePlanes,targets,FALSE,&depthView);
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) commands->ClearRenderTargetView(targets[plane],clear[plane],0,nullptr);
    commands->ClearDepthStencilView(depthView,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr); commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2; unsigned constantSlot=1; LumenSurfaceResult result; result.Layout=layout; result.Identity=upload.Identity;
    for (unsigned card=0;card<upload.Cards.size();++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const D3D12_VIEWPORT viewport{float(x),float(y),float(stride),float(stride),0,1}; const D3D12_RECT scissor{LONG(x),LONG(y),LONG(x+stride),LONG(y+stride)};
        commands->RSSetViewports(1,&viewport); commands->RSSetScissorRects(1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const unsigned raster=(draw.BackFaceCulling?2u:0u)+(RealtimeWorldIsMirrored(draw.World)?1u:0u);
            commands->SetPipelineState(LumenCapturePipelines[raster].Get()); commands->SetGraphicsRootConstantBufferView(0,work->Constants->GetGPUVirtualAddress()+std::uint64_t(constantSlot++)*LumenConstantStride);
            BindLumenViews(*work,index+1,false); commands->IASetVertexBuffers(0,1,&Buffers.at(draw.Vertices.Value).VertexView); commands->IASetIndexBuffer(&Buffers.at(draw.Indices.Value).IndexView);
            commands->DrawIndexedInstanced(draw.IndexCount,1,draw.FirstIndex,0,0); ++result.DrawCalls;
        }
    }
    commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (auto& texture:record.Images) LumenBarrier(*work,texture.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,LumenSampleState);
    if (upload.Readback) EncodeLumenRead(*work);
    if (!CompleteLumenWork(work,upload.Readback?&result.Texels:nullptr)) return false;
    result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space is exhausted");
    LumenSurfaces.emplace(result.Handle.Value,work->Candidate); output=std::move(result); return true;
}
bool ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Lumen surface handle is invalid or expired");
    if (!RecoverSubmittedWork("Read Lumen surface") || !WaitForGPU() || !EnsureLumenSurface()) return false; LumenSurfacePending.reset();
    auto work=std::make_shared<LumenSurfaceWork>(); work->Candidate=found->second; const auto& layout=work->Candidate->Layout;
    std::vector<std::uint8_t> constants(LumenConstantStride); const ShaderUInt4 extent{layout.Width,layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    std::memcpy(constants.data(),&extent,sizeof(extent)); if (!CreateLumenWork(*work,1,constants,true)) return false;
    LumenReadViews(*work); auto* views=work->Views.Get(); work->Commands->SetDescriptorHeaps(1,&views); EncodeLumenRead(*work); return CompleteLumenWork(work,&output);
}
bool DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!RecoverSubmittedWork("Release Lumen surface")) return false; LumenSurfacePending.reset();
    return (handle && LumenSurfaces.erase(handle.Value)) || Fail("Lumen surface handle is invalid or already released");
}
