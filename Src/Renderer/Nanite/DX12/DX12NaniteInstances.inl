// The upload is immutable; GPU-owned counts/prefixes share the final argument
// allocation. Transitions publish it to indirect dispatch and shader reads.
NaniteInstanceWork InstancePlan;
struct InstanceStorage { std::shared_ptr<NaniteFrameBuffer> Upload,Work; };
std::vector<InstanceStorage> InstanceBuffers;
ComPtr<ID3D12RootSignature> InstanceRoot;
ComPtr<ID3D12CommandSignature> InstanceDispatchSignature;
std::array<ComPtr<ID3D12PipelineState>,3> InstanceKernels;
void DestroyNaniteInstances()
{ InstancePlan={}; InstanceBuffers.clear(); InstanceKernels={}; InstanceRoot.Reset(); InstanceDispatchSignature.Reset(); }
bool PrepareNaniteInstances(const VisibilityBufferDesc& frame)
{
    std::vector<NaniteVisibilityGeometryConstants> constants;
    if (frame.GpuInstances) for (const auto& source:NaniteAssemblySources) constants.push_back(source.Constants);
    if (!BuildNaniteInstanceWork(frame,constants,InstancePlan,LastError)) return false;
    InstanceBuffers.clear(); if (InstancePlan.Batches.empty()) return true;
    for (const auto& batch:InstancePlan.Batches)
    {
        InstanceStorage buffers;
        if (!CreateNaniteFrameBuffer(unsigned(batch.Upload.size()),batch.Upload.data(),false,buffers.Upload) ||
            !CreateNaniteFrameBuffer(unsigned(batch.Upload.size()),nullptr,true,buffers.Work)) return false;
        InstanceBuffers.push_back(std::move(buffers));
    }
    if (!InstanceRoot)
    {
        D3D12_ROOT_PARAMETER p[6]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues=sizeof(ShaderUInt4)/sizeof(unsigned);
        for (unsigned i=1;i<=3;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[i].Descriptor.ShaderRegister=i-1; }
        for (unsigned i=4;i<6;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; p[i].Descriptor.ShaderRegister=i-4; }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=p; ComPtr<ID3DBlob> code,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(InstanceRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create instance schedule root",status);
    }
    if (!InstanceDispatchSignature)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argument{}; argument.Type=D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC desc{}; desc.ByteStride=sizeof(D3D12_DISPATCH_ARGUMENTS); desc.NumArgumentDescs=1; desc.pArgumentDescs=&argument;
        const auto status=Device->CreateCommandSignature(&desc,nullptr,IID_PPV_ARGS(InstanceDispatchSignature.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create instance indirect dispatch signature",status);
    }
    if (std::all_of(InstanceKernels.begin(),InstanceKernels.end(),[](const auto& pipeline){return bool(pipeline);})) return true;
    if (!LoadTechniqueShaderSource("DX12","NaniteInstances.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSCount","CSScan","CSAssemble"};
    for (unsigned pass=0;pass<InstanceKernels.size();++pass) if (!InstanceKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=InstanceRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(InstanceKernels[pass].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create instance schedule pipeline",status);
    }
    return true;
}
bool DrawNaniteInstances()
{
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); CommandList->SetComputeRootSignature(InstanceRoot.Get());
    Transition(VisibilityGeometry.Get(),NaniteAssemblyState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootUnorderedAccessView(4,VisibilityGeometry->GetGPUVirtualAddress());
    for (unsigned family=0;family<InstancePlan.Batches.size();++family)
    {
        const auto& batch=InstancePlan.Batches[family]; auto& buffers=InstanceBuffers[family]; auto& work=*buffers.Work;
        NaniteCopyLocks.push_back(batch.Vertex); NaniteCopyLocks.push_back(batch.Index);
        Transition(work.Buffer.Get(),work.State,D3D12_RESOURCE_STATE_COPY_DEST);
        CommandList->CopyBufferRegion(work.Buffer.Get(),0,buffers.Upload->Buffer.Get(),0,batch.Upload.size()*sizeof(unsigned));
        Transition(work.Buffer.Get(),work.State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CommandList->SetComputeRoot32BitConstants(0,sizeof(ShaderUInt4)/sizeof(unsigned),&batch.Parameters,0);
        CommandList->SetComputeRootShaderResourceView(1,NaniteRasterDecisions->Buffer->GetGPUVirtualAddress());
        CommandList->SetComputeRootUnorderedAccessView(5,work.Buffer->GetGPUVirtualAddress());
        for (unsigned pass=0;pass<2;++pass)
        {
            CommandList->SetPipelineState(InstanceKernels[pass].Get()); CommandList->Dispatch(pass?1:(batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads,1,1);
            D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=work.Buffer.Get(); CommandList->ResourceBarrier(1,&barrier);
        }
        Transition(work.Buffer.Get(),work.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        auto& vertex=Buffers.at(batch.Vertex.Value); auto& index=Buffers.at(batch.Index.Value); const auto vs=vertex.State,is=index.State;
        Transition(vertex.Resource.Get(),vertex.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); Transition(index.Resource.Get(),index.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CommandList->SetComputeRootShaderResourceView(1,vertex.Resource->GetGPUVirtualAddress()); CommandList->SetComputeRootShaderResourceView(2,index.Resource->GetGPUVirtualAddress());
        CommandList->SetComputeRootShaderResourceView(3,work.Buffer->GetGPUVirtualAddress()); CommandList->SetPipelineState(InstanceKernels[2].Get());
        CommandList->ExecuteIndirect(InstanceDispatchSignature.Get(),1,work.Buffer.Get(),0,nullptr,0);
        Transition(vertex.Resource.Get(),vertex.State,vs); Transition(index.Resource.Get(),index.State,is);
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount; PendingFrame.Statistics.NaniteInstanceDispatchCount+=unsigned(InstanceKernels.size());
    }
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=VisibilityGeometry.Get(); CommandList->ResourceBarrier(1,&barrier);
    Transition(VisibilityGeometry.Get(),NaniteAssemblyState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); return true;
}
