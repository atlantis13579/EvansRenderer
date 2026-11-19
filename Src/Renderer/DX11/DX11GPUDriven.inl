// D3D11 cannot combine CONSTANT_BUFFER and UAV binding on one allocation.
// Compute writes raw storage; an ordered GPU copy supplies the vertex CB.
// Every prepared batch owns both buffers and its indirect command until reuse.
struct GpuBatch
{
    GpuDrivenBatchDesc Description;
    ComPtr<ID3D11Buffer> Constants, Input, Output, VertexConstants, Arguments;
    ComPtr<ID3D11ShaderResourceView> InputView;
    ComPtr<ID3D11UnorderedAccessView> OutputView, ArgumentView;
};
std::vector<GpuBatch> GpuBatches;
GpuBatch* ActiveGpuBatch = nullptr;
std::size_t GpuCursor = 0;
std::uint64_t GpuEpoch = 0;
ComPtr<ID3D11ComputeShader> GpuCullShader;

void DestroyGpuDriven()
{
    ActiveGpuBatch = nullptr; GpuCursor = 0;
    GpuBatches.clear(); GpuCullShader.Reset();
}
bool CreateGpuBatch(GpuBatch& batch)
{
    const auto buffer = [&](UINT bytes, UINT binds, UINT misc, ComPtr<ID3D11Buffer>& output)
    {
        D3D11_BUFFER_DESC info {};
        info.ByteWidth = bytes; info.Usage = D3D11_USAGE_DEFAULT;
        info.BindFlags = binds; info.MiscFlags = misc;
        const auto result = Device->CreateBuffer(&info, nullptr, output.ReleaseAndGetAddressOf());
        return SUCCEEDED(result) || Fail(HResultMessage("CreateBuffer(GPU-driven)", result));
    };
    if (!buffer(sizeof(GpuDrivenCullConstants), D3D11_BIND_CONSTANT_BUFFER, 0, batch.Constants) ||
        !buffer(sizeof(InstanceConstants) * MaximumDrawInstances, D3D11_BIND_SHADER_RESOURCE,
            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, batch.Input) ||
        !buffer(sizeof(InstanceConstantBlock), D3D11_BIND_UNORDERED_ACCESS,
            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, batch.Output) ||
        !buffer(sizeof(InstanceConstantBlock), D3D11_BIND_CONSTANT_BUFFER, 0, batch.VertexConstants) ||
        !buffer(sizeof(GpuDrivenIndexedArguments), D3D11_BIND_UNORDERED_ACCESS,
            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS, batch.Arguments)) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC source {};
    source.Format = DXGI_FORMAT_R32_TYPELESS; source.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    source.BufferEx.NumElements = sizeof(InstanceConstants) * MaximumDrawInstances / sizeof(std::uint32_t);
    source.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    auto result = Device->CreateShaderResourceView(batch.Input.Get(), &source, batch.InputView.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(GPU candidates)", result));
    D3D11_UNORDERED_ACCESS_VIEW_DESC target {};
    target.Format = DXGI_FORMAT_R32_TYPELESS; target.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    target.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    target.Buffer.NumElements = sizeof(InstanceConstantBlock) / sizeof(std::uint32_t);
    result = Device->CreateUnorderedAccessView(batch.Output.Get(), &target, batch.OutputView.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateUnorderedAccessView(GPU instances)", result));
    target.Buffer.NumElements = sizeof(GpuDrivenIndexedArguments) / sizeof(std::uint32_t);
    result = Device->CreateUnorderedAccessView(batch.Arguments.Get(), &target, batch.ArgumentView.GetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateUnorderedAccessView(indirect command)", result));
}
bool PrepareGpuBatch(const GpuDrivenBatchDesc& desc, std::uint32_t& index)
{
    GpuDrivenBatchUpload upload;
    if (!BuildGpuDrivenBatch(desc, upload)) return Fail("DX11 GPU batch contains invalid bounds, transforms or command data");
    if (!GpuCullShader)
    {
        if (!LoadTechniqueShaderSource("DX11", "GPUDrivenRendering.hlsl", ShaderCode, LastError)) return false;
        ComPtr<ID3DBlob> code;
        if (!CompileShader("CSCull", "cs_5_0", code, nullptr, true)) return false;
        const auto result = Device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, GpuCullShader.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateComputeShader(GPU culling)", result));
    }
    if (GpuCursor == GpuBatches.size())
    {
        GpuBatch batch;
        if (!CreateGpuBatch(batch)) return false;
        GpuBatches.push_back(std::move(batch));
    }
    auto& batch = GpuBatches[GpuCursor];
    batch.Description = desc;
    Context->UpdateSubresource(batch.Constants.Get(), 0, nullptr, &upload.Constants, 0, 0);
    Context->UpdateSubresource(batch.Input.Get(), 0, nullptr, upload.Candidates.data(), 0, 0);
    ID3D11Buffer* parameters = batch.Constants.Get();
    ID3D11ShaderResourceView* input = batch.InputView.Get();
    ID3D11UnorderedAccessView* outputs[] {batch.OutputView.Get(), batch.ArgumentView.Get()};
    Context->CSSetShader(GpuCullShader.Get(), nullptr, 0);
    Context->CSSetConstantBuffers(0, 1, &parameters);
    Context->CSSetShaderResources(0, 1, &input);
    Context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
    Context->Dispatch(1, 1, 1);
    outputs[0] = outputs[1] = nullptr; input = nullptr; parameters = nullptr;
    Context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
    Context->CSSetShaderResources(0, 1, &input);
    Context->CSSetConstantBuffers(0, 1, &parameters);
    Context->CSSetShader(nullptr, nullptr, 0);
    Context->CopyResource(batch.VertexConstants.Get(), batch.Output.Get());
    index = static_cast<std::uint32_t>(GpuCursor++);
    return true;
}
