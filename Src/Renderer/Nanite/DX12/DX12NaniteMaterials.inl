// Root SRVs bind independently authored HLSL inputs; only the work buffer
// transitions through UAV, indirect argument and vertex/fragment read states.
NaniteMaterialWork MaterialPlan;
std::shared_ptr<NaniteFrameBuffer> MaterialMap,MaterialWork;
ComPtr<ID3D12RootSignature> MaterialRoot;
ComPtr<ID3D12CommandSignature> MaterialDrawSignature;
std::array<ComPtr<ID3D12PipelineState>,4> MaterialKernels;
void DestroyNaniteMaterials()
{
    MaterialPlan={}; MaterialMap.reset(); MaterialWork.reset(); MaterialRoot.Reset(); MaterialDrawSignature.Reset(); MaterialKernels={};
}
bool PrepareNaniteMaterials(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteMaterialWork(frame,CreateInfo.Width,CreateInfo.Height,MaterialPlan,LastError)) return false;
    if (!MaterialPlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(MaterialPlan.MaterialBins.size()),MaterialPlan.MaterialBins.data(),false,MaterialMap) ||
        !CreateNaniteFrameBuffer(MaterialPlan.OutputWords,nullptr,true,MaterialWork)) return false;
    if (!MaterialRoot)
    {
        D3D12_ROOT_PARAMETER parameters[5]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants.Num32BitValues=sizeof(NaniteMaterialConstants)/sizeof(unsigned);
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_DESCRIPTOR_RANGE image{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,2,0,0};
        parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[4].DescriptorTable={1,&image};
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=5; desc.pParameters=parameters; ComPtr<ID3DBlob> code,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize material schedule root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(MaterialRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create material schedule root",status);
    }
    if (!MaterialDrawSignature)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argument{}; argument.Type=D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        D3D12_COMMAND_SIGNATURE_DESC desc{}; desc.ByteStride=sizeof(D3D12_DRAW_ARGUMENTS); desc.NumArgumentDescs=1; desc.pArgumentDescs=&argument;
        const auto status=Device->CreateCommandSignature(&desc,nullptr,IID_PPV_ARGS(MaterialDrawSignature.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create material draw signature",status);
    }
    if (std::all_of(MaterialKernels.begin(),MaterialKernels.end(),[](const auto& kernel){return bool(kernel);})) return true;
    if (!LoadTechniqueShaderSource("DX12","NaniteMaterials.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSClear","CSClassify","CSAllocate","CSScatter"};
    for (unsigned pass=0;pass<MaterialKernels.size();++pass) if (!MaterialKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=MaterialRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(MaterialKernels[pass].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create material schedule pipeline",status);
    }
    return true;
}
bool ClassifyNaniteMaterials()
{
    if (!MaterialPlan.Active()) return true;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(VisibilityImages[0].Resource.Get(),VisibilityImages[0].State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(MaterialWork->Buffer.Get(),MaterialWork->State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
    CommandList->SetComputeRootSignature(MaterialRoot.Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(NaniteMaterialConstants)/sizeof(unsigned),&MaterialPlan.Constants,0);
    CommandList->SetComputeRootShaderResourceView(1,MaterialMap->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootShaderResourceView(2,VisibilityGeometry->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(3,MaterialWork->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootDescriptorTable(4,SRVArena.Gpu(VisibilityImages[0].SRV));
    for (unsigned pass=0;pass<MaterialKernels.size();++pass)
    {
        const auto& c=MaterialPlan.Constants;
        const unsigned x=pass==0?(c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads:pass==2?1:c.Layout.Y;
        const unsigned y=pass==0 || pass==2?1:c.Layout.Z;
        CommandList->SetPipelineState(MaterialKernels[pass].Get()); CommandList->Dispatch(x,y,1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=MaterialWork->Buffer.Get(); CommandList->ResourceBarrier(1,&barrier);
    }
    Transition(MaterialWork->Buffer.Get(),MaterialWork->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    auto& stats=PendingFrame.Statistics; stats.NaniteMaterialDispatchCount+=unsigned(MaterialKernels.size());
    stats.NaniteMaterialBinCount+=unsigned(MaterialPlan.Representatives.size()); stats.VisibilityMaterialDrawCount+=unsigned(MaterialPlan.Representatives.size());
    stats.IndirectDrawCallCount+=unsigned(MaterialPlan.Representatives.size()); return true;
}
