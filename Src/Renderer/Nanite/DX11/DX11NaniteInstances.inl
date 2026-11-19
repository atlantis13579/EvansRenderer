// Native count/scan kernels build dispatch dimensions without a host count.
NaniteInstanceWork InstancePlan;
std::vector<std::shared_ptr<NaniteFrameBuffer>> InstanceBuffers;
std::array<ComPtr<ID3D11ComputeShader>,3> InstanceKernels;
ComPtr<ID3D11Buffer> InstanceUniform;
void DestroyNaniteInstances()
{ InstancePlan={}; InstanceBuffers.clear(); InstanceKernels={}; InstanceUniform.Reset(); }
bool PrepareNaniteInstances(const VisibilityBufferDesc& frame)
{
    std::vector<NaniteVisibilityGeometryConstants> constants;
    if (frame.GpuInstances) for (const auto& source:NaniteAssemblySources) constants.push_back(source.Constants);
    if (!BuildNaniteInstanceWork(frame,constants,InstancePlan,LastError)) return false;
    InstanceBuffers.clear(); if (InstancePlan.Batches.empty()) return true;
    for (const auto& batch:InstancePlan.Batches)
    {
        std::shared_ptr<NaniteFrameBuffer> work;
        if (!CreateNaniteFrameBuffer(unsigned(batch.Upload.size()),batch.Upload.data(),true,work,true)) return false;
        InstanceBuffers.push_back(std::move(work));
    }
    if (!InstanceUniform)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(ShaderUInt4); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,InstanceUniform.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create instance schedule constants",status));
    }
    if (std::all_of(InstanceKernels.begin(),InstanceKernels.end(),[](const auto& shader){return bool(shader);})) return true;
    if (!LoadTechniqueShaderSource("DX11","NaniteInstances.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSCount","CSScan","CSAssemble"};
    for (unsigned pass=0;pass<InstanceKernels.size();++pass) if (!InstanceKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,InstanceKernels[pass].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create instance schedule kernel",status));
    }
    return true;
}
bool DrawNaniteInstances()
{
    ID3D11ShaderResourceView* empty[8]{}; ID3D11UnorderedAccessView* noTargets[2]{};
    Context->VSSetShaderResources(0,8,empty); Context->PSSetShaderResources(0,8,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* uniform=InstanceUniform.Get(); Context->CSSetConstantBuffers(0,1,&uniform);
    for (unsigned family=0;family<InstancePlan.Batches.size();++family)
    {
        const auto& batch=InstancePlan.Batches[family]; const auto& work=InstanceBuffers[family];
        const auto& source=NaniteAssemblySources[batch.Sources.front()];
        NaniteCopyLocks.push_back(batch.Vertex); NaniteCopyLocks.push_back(batch.Index);
        Context->UpdateSubresource(uniform,0,nullptr,&batch.Parameters,0,0);
        auto* decision=NaniteRasterDecisions->View.Get(); Context->CSSetShaderResources(0,1,&decision);
        auto* writable=work->Target.Get(); Context->CSSetUnorderedAccessViews(1,1,&writable,nullptr);
        Context->CSSetShader(InstanceKernels[0].Get(),nullptr,0); Context->Dispatch((batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads,1,1);
        Context->CSSetShader(InstanceKernels[1].Get(),nullptr,0); Context->Dispatch(1,1,1);
        // UAV unbinding publishes the prefix buffer as both SRV and arguments.
        Context->CSSetUnorderedAccessViews(0,2,noTargets,nullptr);
        ID3D11ShaderResourceView* inputs[]{source.Views[0].Get(),source.Views[1].Get(),work->View.Get()}; Context->CSSetShaderResources(0,3,inputs);
        auto* table=NaniteAssemblyTarget.Get(); Context->CSSetUnorderedAccessViews(0,1,&table,nullptr);
        Context->CSSetShader(InstanceKernels[2].Get(),nullptr,0); Context->DispatchIndirect(work->Buffer.Get(),0);
        Context->CSSetUnorderedAccessViews(0,2,noTargets,nullptr); Context->CSSetShaderResources(0,3,empty);
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount; PendingFrame.Statistics.NaniteInstanceDispatchCount+=unsigned(InstanceKernels.size());
    }
    Context->CSSetShader(nullptr,nullptr,0); return true;
}
