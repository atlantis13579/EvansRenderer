// Native UAV -> CBV/INDIRECT_ARGUMENT transitions make visibility independent
// of CPU readback. Reused allocations are fenced by BeginFrame; discarded
// command lists restore their recorded resource states before the next frame.
struct GpuBatch
{
    GpuDrivenBatchDesc Description;
    ComPtr<ID3D12Resource> Upload, Output, Arguments;
    D3D12_RESOURCE_STATES OutputState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES ArgumentState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES StartOutputState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES StartArgumentState = D3D12_RESOURCE_STATE_COMMON;
};
std::vector<GpuBatch> GpuBatches;
GpuBatch* ActiveGpuBatch = nullptr;
std::size_t GpuCursor = 0;
std::uint64_t GpuEpoch = 0;
ComPtr<ID3D12RootSignature> GpuCullRoot;
ComPtr<ID3D12PipelineState> GpuCullPipeline;
ComPtr<ID3D12CommandSignature> GpuDrawSignature;
static constexpr UINT GpuCullConstantRoot = 0, GpuInputRoot = 1, GpuOutputRoot = 2, GpuArgumentRoot = 3;
static constexpr std::size_t GpuInputOffset = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
static_assert(sizeof(GpuDrivenCullConstants) <= GpuInputOffset, "GPU culling upload header exceeds its slice");

void DestroyGpuDriven()
{
    ActiveGpuBatch = nullptr; GpuCursor = 0; GpuBatches.clear();
    GpuCullPipeline.Reset(); GpuCullRoot.Reset(); GpuDrawSignature.Reset();
}
void AbortGpuDriven()
{
    for (std::size_t index = 0; index < GpuCursor; ++index)
    {
        auto& batch = GpuBatches[index];
        batch.OutputState = batch.StartOutputState; batch.ArgumentState = batch.StartArgumentState;
    }
    ActiveGpuBatch = nullptr; GpuCursor = 0;
}
bool EnsureGpuCull()
{
    if (GpuCullPipeline && GpuDrawSignature) return true;
    if (!GpuCullRoot)
    {
        D3D12_ROOT_PARAMETER parameters[4] {};
        parameters[GpuCullConstantRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[GpuInputRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[GpuOutputRoot].ParameterType = parameters[GpuArgumentRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[GpuArgumentRoot].Descriptor.ShaderRegister = 1;
        D3D12_ROOT_SIGNATURE_DESC signature {}; signature.NumParameters = 4; signature.pParameters = parameters;
        ComPtr<ID3DBlob> serialized, errors;
        auto result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(GPU visibility)", result);
        result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(GpuCullRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(GPU visibility)", result);
    }
    if (!GpuCullPipeline)
    {
        if (!LoadTechniqueShaderSource("DX12", "GPUDrivenRendering.hlsl", ShaderCode, LastError)) return false;
        ComPtr<ID3DBlob> code;
        if (!CompileShader("CSCull", "cs_5_0", code, nullptr, true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline {};
        pipeline.pRootSignature = GpuCullRoot.Get(); pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        const auto result = Evans::Renderer::CreateCachedComputePipelineState(Device,&pipeline, IID_PPV_ARGS(GpuCullPipeline.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateComputePipelineState(GPU visibility)", result);
    }
    if (!GpuDrawSignature)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argument {}; argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
        D3D12_COMMAND_SIGNATURE_DESC signature {};
        signature.ByteStride = sizeof(GpuDrivenIndexedArguments); signature.NumArgumentDescs = 1; signature.pArgumentDescs = &argument;
        const auto result = Device->CreateCommandSignature(&signature, nullptr, IID_PPV_ARGS(GpuDrawSignature.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateCommandSignature(indexed instances)", result);
    }
    return true;
}
bool CreateGpuBatch(GpuBatch& batch)
{
    if (!CreateBufferResource(GpuInputOffset + sizeof(InstanceConstants) * MaximumDrawInstances,
        D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, batch.Upload)) return false;
    // Buffers start in COMMON on current D3D12 runtimes. Track that state and
    // issue the existing explicit transition before the first compute write.
    const auto buffer = [&](std::size_t bytes, ComPtr<ID3D12Resource>& output)
    {
        auto desc = BufferResourceDesc(bytes); desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
        const auto result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(output.GetAddressOf()));
        return SUCCEEDED(result) || FailHRESULT("CreateCommittedResource(GPU batch)", result);
    };
    return buffer(sizeof(InstanceConstantBlock), batch.Output) && buffer(sizeof(GpuDrivenIndexedArguments), batch.Arguments);
}
bool PrepareGpuBatch(const GpuDrivenBatchDesc& desc, std::uint32_t& index)
{
    GpuDrivenBatchUpload upload;
    if (!BuildGpuDrivenBatch(desc, upload)) return Fail("DX12 GPU batch contains invalid bounds, transforms or command data");
    if (!EnsureGpuCull()) return false;
    if (GpuCursor == GpuBatches.size())
    {
        GpuBatch batch;
        if (!CreateGpuBatch(batch)) return false;
        GpuBatches.push_back(std::move(batch));
    }
    auto& batch = GpuBatches[GpuCursor]; batch.Description = desc;
    const D3D12_RANGE noRead {0, 0}; std::uint8_t* mapped = nullptr;
    auto result = batch.Upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped));
    if (FAILED(result)) return FailHRESULT("Map(GPU candidates)", result);
    std::memcpy(mapped, &upload.Constants, sizeof(upload.Constants));
    std::memcpy(mapped + GpuInputOffset, upload.Candidates.data(), sizeof(upload.Candidates));
    const D3D12_RANGE written {0, GpuInputOffset + sizeof(upload.Candidates)}; batch.Upload->Unmap(0, &written);
    batch.StartOutputState = batch.OutputState; batch.StartArgumentState = batch.ArgumentState;
    Transition(batch.Output.Get(), batch.OutputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Transition(batch.Arguments.Get(), batch.ArgumentState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootSignature(GpuCullRoot.Get()); CommandList->SetPipelineState(GpuCullPipeline.Get());
    CommandList->SetComputeRootConstantBufferView(GpuCullConstantRoot, batch.Upload->GetGPUVirtualAddress());
    CommandList->SetComputeRootShaderResourceView(GpuInputRoot, batch.Upload->GetGPUVirtualAddress() + GpuInputOffset);
    CommandList->SetComputeRootUnorderedAccessView(GpuOutputRoot, batch.Output->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(GpuArgumentRoot, batch.Arguments->GetGPUVirtualAddress());
    CommandList->Dispatch(1, 1, 1);
    Transition(batch.Output.Get(), batch.OutputState, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    Transition(batch.Arguments.Get(), batch.ArgumentState, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    index = static_cast<std::uint32_t>(GpuCursor++);
    return true;
}
