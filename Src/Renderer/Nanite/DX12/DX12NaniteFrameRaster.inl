// Hardware and compute coverage share projected corners and a depth/owner pair.
// Buffers remain GPU resident until the normal visibility material resolve.
NaniteFrameRasterWork NaniteRasterPlan;
std::shared_ptr<NaniteFrameBuffer> NaniteRasterDecisions;
NaniteDepthHandle NaniteRasterMainDepth;
NaniteVisibilityHandle NaniteRasterPost;
bool PrepareNaniteRasterVisibility(const VisibilityBufferDesc& frame)
{
    // Previous frame resources are retired only after the frame fence/abort.
    NaniteDepths.erase(NaniteRasterMainDepth.Value); NaniteDecisions.erase(NaniteRasterPost.Value);
    NaniteRasterMainDepth={}; NaniteRasterPost={}; NaniteRasterDecisions.reset();
    if (!NaniteFrameDummy) { const unsigned zero=0; if (!CreateNaniteFrameBuffer(1,&zero,false,NaniteFrameDummy)) return false; }
    NaniteRasterDecisions=NaniteFrameDummy;
    if (!frame.NaniteVisibility) return true;
    const auto found=NaniteDecisions.find(frame.NaniteVisibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Phase!=NaniteVisibilityPhase::Main ||
        found->second->Info.Frame!=FrameNanite.CurrentFrame() || found->second->Info.QueryCount!=frame.NaniteClusters.size() || found->second->Draws)
        return Fail("Nanite raster requires matching unconsumed main-phase cluster decisions");
    NaniteRasterDecisions=found->second->Data; return true;
}
bool RetestNaniteRaster(const VisibilityBufferDesc& frame)
{
    if (!CaptureNaniteDepth(NaniteRasterMainDepth,true) || !RetestNaniteVisibility(frame.NaniteVisibility,NaniteRasterMainDepth,NaniteRasterPost,true)) return false;
    NaniteRasterDecisions=NaniteDecisions.at(NaniteRasterPost.Value)->Data;
    for (auto& job:NaniteRasterPlan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    return true;
}
std::shared_ptr<NaniteFrameBuffer> NaniteRasterInput,NaniteRasterOutput;
ComPtr<ID3D12RootSignature> NaniteRasterComputeRoot,NaniteRasterGraphicsRoot;
std::array<ComPtr<ID3D12PipelineState>,6> NaniteRasterFramePipelines;
std::array<ComPtr<ID3DBlob>,2> NaniteRasterVertices,NaniteRasterPixels;
std::unordered_map<unsigned,ComPtr<ID3D12PipelineState>> NaniteRasterGraphics;
ComPtr<ID3D12DescriptorHeap> NaniteRasterCaptureViews;
void DestroyNaniteFrameRaster()
{
    DestroyNaniteHardware();
    NaniteRasterDecisions.reset(); NaniteRasterMainDepth={}; NaniteRasterPost={};
    NaniteRasterInput.reset(); NaniteRasterOutput.reset(); NaniteRasterPlan={};
    NaniteRasterComputeRoot.Reset(); NaniteRasterGraphicsRoot.Reset(); NaniteRasterFramePipelines={};
    NaniteRasterVertices={}; NaniteRasterPixels={}; NaniteRasterGraphics.clear(); NaniteRasterCaptureViews.Reset();
}
bool PrepareNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (!PrepareNaniteRasterVisibility(frame)) return false;
    NaniteFrameRasterWork plan;
    if (!BuildNaniteFrameRasterWork(frame,CreateInfo.Width,CreateInfo.Height,plan,LastError)) return false;
    if (!NaniteRasterComputeRoot)
    {
        D3D12_ROOT_PARAMETER p[5]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p[0].Constants.Num32BitValues=sizeof(NaniteFrameRasterConstants)/sizeof(unsigned);
        p[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,2,1,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        p[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable={1,&range};
        p[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[4].Descriptor.ShaderRegister=3;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=5; desc.pParameters=p; ComPtr<ID3DBlob> code,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(NaniteRasterComputeRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite frame compute root",status);
    }
    if (!NaniteRasterGraphicsRoot)
    {
        D3D12_ROOT_PARAMETER p[6]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p[0].Constants.Num32BitValues=sizeof(NaniteRasterDrawConstants)/sizeof(unsigned);
        for (unsigned i=1;i<=3;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[i].Descriptor.ShaderRegister=i-1; }
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        for (unsigned i=0;i<2;++i)
        { ranges[i]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,i+3,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}; p[i+4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[i+4].DescriptorTable={1,&ranges[i]}; p[i+4].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL; }
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=p; desc.NumStaticSamplers=1; desc.pStaticSamplers=&sampler;
        ComPtr<ID3DBlob> code,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(NaniteRasterGraphicsRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite frame graphics root",status);
    }
    if (!LoadTechniqueShaderSource("DX12","NaniteRaster.hlsl",ShaderCode,LastError)) return false;
    const char* entries[]{"CSSeedDepth","CSRaster","CSSeedOwner","CSClear","CSProject","CSCapture"};
    for (unsigned pass=0;pass<NaniteRasterFramePipelines.size();++pass) if (!NaniteRasterFramePipelines[pass])
    {
        const auto value=std::to_string(pass); const D3D_SHADER_MACRO macros[]{{"EVANS_NANITE_FRAME_PASS",value.c_str()},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,pass>=3?macros:nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=NaniteRasterComputeRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(NaniteRasterFramePipelines[pass].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite frame compute pipeline",status);
    }
    const D3D_SHADER_MACRO graphics[]{{"EVANS_NANITE_GRAPHICS","1"},{nullptr,nullptr}};
    const char* vertices[]{"VSHardware","VSExport"}; const char* pixels[]{"PSHardware","PSExport"};
    for (unsigned i=0;i<2;++i)
    {
        if (!NaniteRasterVertices[i] && !CompileShader(vertices[i],"vs_5_0",NaniteRasterVertices[i],graphics,true)) return false;
        if (!NaniteRasterPixels[i] && !CompileShader(pixels[i],"ps_5_0",NaniteRasterPixels[i],graphics,true)) return false;
    }
    if (!CreateNaniteFrameBuffer(plan.InputWords,nullptr,true,NaniteRasterInput) || !CreateNaniteFrameBuffer(plan.OutputWords,nullptr,true,NaniteRasterOutput)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=2; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    NaniteRasterCaptureViews.Reset(); const auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(NaniteRasterCaptureViews.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Nanite depth/reference views",status);
    auto handle=NaniteRasterCaptureViews->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
    Device->CreateShaderResourceView(MainDepth.Get(),&view,handle); handle.ptr+=Device->GetDescriptorHandleIncrementSize(heap.Type);
    view.Format=DXGI_FORMAT_R32_UINT; Device->CreateShaderResourceView(VisibilityImages[0].Resource.Get(),&view,handle);
    NaniteRasterPlan=std::move(plan); return true;
}
bool DispatchNaniteFrameRaster(const NaniteFrameRasterDispatch& job)
{
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    const bool core=job.Pass<3,capture=job.Pass==5;
    auto& destination=core?NaniteRasterOutput:NaniteRasterInput;
    const auto depthState=MainDepthState,idState=VisibilityImages[0].State;
    if (capture)
    {
        Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Transition(VisibilityImages[0].Resource.Get(),VisibilityImages[0].State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    ID3D12DescriptorHeap* computeHeaps[]{NaniteRasterCaptureViews.Get()}; CommandList->SetDescriptorHeaps(1,computeHeaps);
    Transition(destination->Buffer.Get(),destination->State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootSignature(NaniteRasterComputeRoot.Get()); CommandList->SetPipelineState(NaniteRasterFramePipelines[job.Pass].Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(job.Constants)/sizeof(unsigned),&job.Constants,0);
    CommandList->SetComputeRootShaderResourceView(1,core?NaniteRasterInput->Buffer->GetGPUVirtualAddress():VisibilityGeometry->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(2,destination->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootDescriptorTable(3,NaniteRasterCaptureViews->GetGPUDescriptorHandleForHeapStart());
    // Upload-heap dummy buffers must remain GENERIC_READ. Only native cull
    // outputs need an explicit state transition before projection.
    if (job.Pass==4 && NaniteRasterDecisions && NaniteRasterDecisions!=NaniteFrameDummy)
        Transition(NaniteRasterDecisions->Buffer.Get(),NaniteRasterDecisions->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    CommandList->SetComputeRootShaderResourceView(4,(NaniteRasterDecisions?NaniteRasterDecisions:NaniteRasterInput)->Buffer->GetGPUVirtualAddress());
    CommandList->Dispatch(job.GroupsX,job.GroupsY,1);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=destination->Buffer.Get(); CommandList->ResourceBarrier(1,&barrier);
    Transition(destination->Buffer.Get(),destination->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    if (capture)
    {
        Transition(MainDepth.Get(),MainDepthState,depthState); Transition(VisibilityImages[0].Resource.Get(),VisibilityImages[0].State,idState);
    }
    ID3D12DescriptorHeap* graphicsHeaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,graphicsHeaps);
    ++PendingFrame.Statistics.NaniteRasterDispatchCount; return true;
}
ID3D12PipelineState* NaniteRasterPipeline(bool cull,bool mirrored,bool exportPass)
{
    const unsigned key=unsigned(cull)|unsigned(mirrored)<<1|unsigned(exportPass)<<2|unsigned(bool(NaniteRasterPost))<<3;
    const auto found=NaniteRasterGraphics.find(key); if (found!=NaniteRasterGraphics.end()) return found->second.Get();
    const unsigned phase=exportPass?1:0; D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=NaniteRasterGraphicsRoot.Get();
    desc.VS={NaniteRasterVertices[phase]->GetBufferPointer(),NaniteRasterVertices[phase]->GetBufferSize()};
    desc.PS={NaniteRasterPixels[phase]->GetBufferPointer(),NaniteRasterPixels[phase]->GetBufferSize()};
    desc.NumRenderTargets=2; desc.BlendState.IndependentBlendEnable=TRUE;
    for (unsigned i=0;i<2;++i)
    {
        auto& blend=desc.BlendState.RenderTarget[i]; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
        desc.RTVFormats[i]=i?DXGI_FORMAT_R32G32_FLOAT:DXGI_FORMAT_R32_UINT;
    }
    desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=cull&&!exportPass?D3D12_CULL_MODE_BACK:D3D12_CULL_MODE_NONE;
    desc.RasterizerState.FrontCounterClockwise=mirrored?FALSE:TRUE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthEnable=TRUE; desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc=exportPass?D3D12_COMPARISON_FUNC_ALWAYS:NaniteRasterPost?D3D12_COMPARISON_FUNC_LESS_EQUAL:D3D12_COMPARISON_FUNC_LESS;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=stencil;
    desc.DSVFormat=DXGI_FORMAT_D32_FLOAT; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.SampleDesc.Count=1; desc.SampleMask=UINT_MAX;
    ComPtr<ID3D12PipelineState> pipeline; const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(pipeline.GetAddressOf()));
    if (FAILED(status)) { FailHRESULT("Create Nanite frame graphics pipeline",status); return nullptr; }
    auto* result=pipeline.Get(); NaniteRasterGraphics.emplace(key,std::move(pipeline)); return result;
}
bool DrawNaniteRasterPhase(const VisibilityBufferDesc& frame)
{
    if (frame.NaniteVisibility) ++PendingFrame.Statistics.NaniteOcclusionPhases;
    const auto bind=[&](const VisibilityMaterialDraw* draw)
    {
        NaniteRasterDrawConstants constants; if (!BuildNaniteRasterDrawConstants(frame,draw,CreateInfo.Width,CreateInfo.Height,constants)) return false;
        if (NaniteRasterPost) constants.Raster.Quality.W|=NaniteRasterPreserveMain;
        CommandList->SetGraphicsRoot32BitConstants(0,sizeof(constants)/sizeof(unsigned),&constants,0);
        CommandList->SetGraphicsRootShaderResourceView(1,VisibilityGeometry->GetGPUVirtualAddress());
        CommandList->SetGraphicsRootShaderResourceView(2,NaniteRasterInput->Buffer->GetGPUVirtualAddress());
        CommandList->SetGraphicsRootShaderResourceView(3,NaniteRasterOutput->Buffer->GetGPUVirtualAddress());
        const TextureHandle handles[]{draw?draw->Pipeline.DiffuseTexture:TextureHandle{},draw?draw->Pipeline.OpacityTexture:TextureHandle{}};
        for (unsigned i=0;i<2;++i)
        {
            UINT descriptor=NullColorSRV;
            if (handles[i])
            {
                const auto found=Textures.find(handles[i].Value);
                if (found==Textures.end() || found->second.SRV==InvalidDescriptor || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid Nanite material texture");
                Transition(found->second.Resource.Get(),found->second.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); descriptor=found->second.SRV;
            }
            CommandList->SetGraphicsRootDescriptorTable(i+4,SRVArena.Gpu(descriptor));
        }
        return true;
    };
    const auto targets=[&]()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE colors[]{RTVArena.Cpu(VisibilityImages[0].RTV),RTVArena.Cpu(VisibilityImages[1].RTV)}; const auto depth=DSVArena.Cpu(MainDepthDSV);
        CommandList->OMSetRenderTargets(2,colors,FALSE,&depth); CommandList->SetGraphicsRootSignature(NaniteRasterGraphicsRoot.Get());
        CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    };
    for (unsigned i=0;i<NaniteRasterPlan.HardwareBoundary;++i) if (!DispatchNaniteFrameRaster(NaniteRasterPlan.Dispatches[i])) return false;
    targets();
    if (frame.GpuHardware)
    { if (!DrawNaniteHardware(frame)) return false; }
    else
    {
        for (std::size_t order=0;order<frame.Draws.size();++order)
        {
            const auto& draw=frame.Draws[NaniteRasterPost?frame.Draws.size()-1-order:order];
            const auto& range=frame.Scene->Draws[draw.Range]; auto* pipeline=NaniteRasterPipeline(draw.Pipeline.BackFaceCulling,range.Mirrored,false);
            if (!pipeline || !bind(&draw)) return false; CommandList->SetPipelineState(pipeline); CommandList->DrawInstanced(range.TriangleCount*VisibilityTriangleCorners,1,0,0);
        }
    }
    for (std::size_t i=NaniteRasterPlan.HardwareBoundary;i<NaniteRasterPlan.Dispatches.size();++i) if (!DispatchNaniteFrameRaster(NaniteRasterPlan.Dispatches[i])) return false;
    targets(); auto* pipeline=NaniteRasterPipeline(false,false,true); if (!pipeline || !bind(nullptr)) return false;
    CommandList->SetPipelineState(pipeline); CommandList->DrawInstanced(3,1,0,0); ++PendingFrame.Statistics.NaniteRasterExportCount;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); return true;
}

bool DrawNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (frame.Scene->HasNativeGeometry() && !(frame.GpuInstances?DrawNaniteInstances():DrawNaniteVisibilityGeometry())) return false;
    PendingFrame.Statistics.NaniteOcclusionQueries+=unsigned(frame.NaniteClusters.size());
    if (!DrawNaniteRasterPhase(frame)) return false;
    if (frame.NaniteVisibility)
    {
        if (!RetestNaniteRaster(frame)) return false;
        if (!DrawNaniteRasterPhase(frame)) return false;
    }
    return true;
}
