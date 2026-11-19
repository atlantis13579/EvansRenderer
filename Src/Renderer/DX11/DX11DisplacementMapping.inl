// Frame-external preparation owns all intermediate allocations. The default
// immediate-context UAV ordering separates position writes from normal reads;
// staging Map completes readback before the drawable allocation is published.
std::array<ComPtr<ID3D11ComputeShader>,2> DisplacementShaders;
void DestroyDisplacement() { for (auto& shader:DisplacementShaders) shader.Reset(); }
bool EnsureDisplacement()
{
    if (DisplacementShaders[0] && DisplacementShaders[1]) return true;
    if (!LoadTechniqueShaderSource("DX11","DisplacementMapping.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSDisplace","CSNormals"};
    for (std::size_t phase=0;phase<DisplacementShaders.size();++phase) if (!DisplacementShaders[phase])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[phase],"cs_5_0",code,nullptr,true)) return false;
        const auto result=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,DisplacementShaders[phase].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create displacement compute shader",result));
    }
    return true;
}
bool CreateDisplaced(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    DisplacementConstants constants;
    if (!ValidateDisplacementUpload(geometry,settings,constants,LastError) || !EnsureDisplacement()) return false;
    const void* data[]{geometry.Vertices.data(),geometry.Directions.data(),geometry.Indices.data(),geometry.Neighborhoods.data(),geometry.AdjacentTriangles.data()};
    const std::size_t sizes[]{geometry.Vertices.size()*sizeof(Vertex),geometry.Directions.size()*sizeof(Float4),geometry.Indices.size()*sizeof(std::uint32_t),
        geometry.Neighborhoods.size()*sizeof(ShaderUInt4),geometry.AdjacentTriangles.size()*sizeof(std::uint32_t)};
    const UINT strides[]{0,sizeof(Float4),sizeof(std::uint32_t),sizeof(ShaderUInt4),sizeof(std::uint32_t)};
    std::array<ComPtr<ID3D11Buffer>,5> sources;
    std::array<ComPtr<ID3D11ShaderResourceView>,5> views;
    for (std::size_t slot=0;slot<sources.size();++slot)
    {
        if (!sizes[slot] || sizes[slot]>std::numeric_limits<UINT>::max()) return Fail("Displacement input exceeds D3D11 buffer addressing");
        D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=static_cast<UINT>(sizes[slot]); buffer.Usage=D3D11_USAGE_DEFAULT;
        buffer.BindFlags=D3D11_BIND_SHADER_RESOURCE; buffer.StructureByteStride=strides[slot];
        buffer.MiscFlags=slot?D3D11_RESOURCE_MISC_BUFFER_STRUCTURED:D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data[slot];
        auto result=Device->CreateBuffer(&buffer,&initial,sources[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create displacement source buffer",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format=slot?DXGI_FORMAT_UNKNOWN:DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension=slot?D3D11_SRV_DIMENSION_BUFFER:D3D11_SRV_DIMENSION_BUFFEREX;
        if (slot) view.Buffer.NumElements=static_cast<UINT>(sizes[slot]/strides[slot]);
        else { view.BufferEx.NumElements=static_cast<UINT>(sizes[slot]/sizeof(std::uint32_t)); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW; }
        result=Device->CreateShaderResourceView(sources[slot].Get(),&view,views[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create displacement source view",result));
    }
    BufferResource resource; resource.Description={BufferUsage::Vertex,sizes[0],false,"Displaced vertices"}; resource.Bytes.resize(sizes[0]);
    DisplacedVertexBuffer candidate; candidate.Vertices.resize(geometry.Vertices.size());
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=static_cast<UINT>(sizes[0]); buffer.Usage=D3D11_USAGE_DEFAULT;
    buffer.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_VERTEX_BUFFER; buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    auto result=Device->CreateBuffer(&buffer,nullptr,resource.Buffer.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create displacement vertex/UAV allocation",result));
    D3D11_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
    view.Buffer.NumElements=buffer.ByteWidth/sizeof(std::uint32_t); view.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
    ComPtr<ID3D11UnorderedAccessView> target;
    result=Device->CreateUnorderedAccessView(resource.Buffer.Get(),&view,target.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create displacement vertex UAV",result));
    ComPtr<ID3D11Buffer> readback,parameters;
    buffer.BindFlags=buffer.MiscFlags=0; buffer.Usage=D3D11_USAGE_STAGING; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    result=Device->CreateBuffer(&buffer,nullptr,readback.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create displacement staging readback",result));
    buffer={}; buffer.ByteWidth=sizeof(constants); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&constants;
    result=Device->CreateBuffer(&buffer,&initial,parameters.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create displacement parameter buffer",result));
    ID3D11ShaderResourceView* inputs[5]{}; for (std::size_t slot=0;slot<views.size();++slot) inputs[slot]=views[slot].Get();
    auto* uniform=parameters.Get(); auto* unordered=target.Get();
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,5,inputs); Context->CSSetUnorderedAccessViews(0,1,&unordered,nullptr);
    const UINT groups=static_cast<UINT>((geometry.Vertices.size()+DisplacementThreadCount-1)/DisplacementThreadCount);
    for (const auto& shader:DisplacementShaders) { Context->CSSetShader(shader.Get(),nullptr,0); Context->Dispatch(groups,1,1); }
    for (auto& input:inputs) input=nullptr; uniform=nullptr; unordered=nullptr;
    Context->CSSetUnorderedAccessViews(0,1,&unordered,nullptr); Context->CSSetShaderResources(0,5,inputs); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShader(nullptr,nullptr,0); Context->CopyResource(readback.Get(),resource.Buffer.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{}; result=Context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped);
    if (FAILED(result)) return Fail(HResultMessage("Read displacement results",result));
    std::memcpy(candidate.Vertices.data(),mapped.pData,sizes[0]); std::memcpy(resource.Bytes.data(),mapped.pData,sizes[0]); Context->Unmap(readback.Get(),0);
    if (!ValidateTriangleGeometry(candidate.Vertices,geometry.Indices,LastError)) return false;
    candidate.Buffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(DisplacementShaders.size());
    Buffers.emplace(candidate.Buffer.Value,std::move(resource)); output=std::move(candidate); return true;
}
