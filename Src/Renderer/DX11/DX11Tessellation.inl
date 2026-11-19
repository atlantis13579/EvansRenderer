// Geometry preparation runs between frames. D3D11 orders immediate-context UAV
// accesses; staging Map completes both native streams before they are published.
std::array<ComPtr<ID3D11ComputeShader>,3> TessellationShaders;
void DestroyTessellation() { for (auto& shader:TessellationShaders) shader.Reset(); }
bool EnsureTessellation()
{
    if (TessellationShaders[0] && TessellationShaders[1] && TessellationShaders[2]) return true;
    if (!LoadTechniqueShaderSource("DX11","Tessellation.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSEdges","CSVertices","CSIndices"};
    for (std::size_t phase=0;phase<TessellationShaders.size();++phase) if (!TessellationShaders[phase])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[phase],"cs_5_0",code,nullptr,true)) return false;
        const auto result=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,TessellationShaders[phase].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN tessellation compute shader",result));
    }
    return true;
}
bool CreateTessellated(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    TessellationConstants constants;
    if (!ValidateTessellationUpload(geometry,settings,constants,LastError) || !EnsureTessellation()) return false;
    const void* data[]{geometry.Vertices.data(),geometry.Patches.data(),geometry.PatchEdges.data(),geometry.AttributeEdges.data(),geometry.GeometricEdges.data()};
    const std::size_t sizes[]{geometry.Vertices.size()*sizeof(Vertex),geometry.Patches.size()*sizeof(ShaderUInt4),geometry.PatchEdges.size()*sizeof(ShaderUInt4),
        geometry.AttributeEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*sizeof(ShaderUInt4)};
    std::array<ComPtr<ID3D11Buffer>,5> sources;
    std::array<ComPtr<ID3D11ShaderResourceView>,5> views;
    for (std::size_t slot=0;slot<sources.size();++slot)
    {
        if (!sizes[slot] || sizes[slot]>std::numeric_limits<UINT>::max()) return Fail("PN source exceeds D3D11 buffer addressing");
        D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=static_cast<UINT>(sizes[slot]); buffer.Usage=D3D11_USAGE_DEFAULT;
        buffer.BindFlags=D3D11_BIND_SHADER_RESOURCE; buffer.StructureByteStride=slot?sizeof(ShaderUInt4):0;
        buffer.MiscFlags=slot?D3D11_RESOURCE_MISC_BUFFER_STRUCTURED:D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data[slot];
        auto result=Device->CreateBuffer(&buffer,&initial,sources[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN source buffer",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=slot?DXGI_FORMAT_UNKNOWN:DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension=slot?D3D11_SRV_DIMENSION_BUFFER:D3D11_SRV_DIMENSION_BUFFEREX;
        if (slot) view.Buffer.NumElements=static_cast<UINT>(sizes[slot]/sizeof(ShaderUInt4));
        else { view.BufferEx.NumElements=static_cast<UINT>(sizes[slot]/sizeof(std::uint32_t)); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW; }
        result=Device->CreateShaderResourceView(sources[slot].Get(),&view,views[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN source view",result));
    }
    const std::size_t vertexBytes=std::size_t(constants.Outputs.X)*sizeof(Vertex),indexBytes=std::size_t(constants.Outputs.Y)*sizeof(std::uint32_t);
    std::array<BufferResource,2> resources;
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"PN tessellated vertices"};
    resources[1].Description={BufferUsage::Index,indexBytes,false,"PN tessellated indices"};
    std::array<ComPtr<ID3D11Buffer>,3> targets;
    std::array<ComPtr<ID3D11UnorderedAccessView>,3> unordered;
    for (std::size_t slot=0;slot<targets.size();++slot)
    {
        const std::size_t bytes=slot==0?geometry.GeometricEdges.size()*2*sizeof(Float4):resources[slot-1].Description.SizeInBytes;
        if (bytes>std::numeric_limits<UINT>::max()) return Fail("PN output exceeds D3D11 buffer addressing");
        D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=static_cast<UINT>(bytes); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
        if (slot==0) { buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; buffer.StructureByteStride=sizeof(Float4); }
        if (slot==1) { buffer.BindFlags|=D3D11_BIND_VERTEX_BUFFER; buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS; }
        if (slot==2) buffer.BindFlags|=D3D11_BIND_INDEX_BUFFER;
        auto result=Device->CreateBuffer(&buffer,nullptr,targets[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN control or drawable output",result));
        D3D11_UNORDERED_ACCESS_VIEW_DESC view{}; view.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
        view.Format=slot==0?DXGI_FORMAT_UNKNOWN:slot==1?DXGI_FORMAT_R32_TYPELESS:DXGI_FORMAT_R32_UINT;
        view.Buffer.NumElements=static_cast<UINT>(bytes/(slot==0?sizeof(Float4):sizeof(std::uint32_t)));
        if (slot==1) view.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
        result=Device->CreateUnorderedAccessView(targets[slot].Get(),&view,unordered[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN output UAV",result));
    }
    std::array<ComPtr<ID3D11Buffer>,2> readback;
    for (std::size_t slot=0;slot<readback.size();++slot)
    {
        D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=static_cast<UINT>(resources[slot].Description.SizeInBytes);
        buffer.Usage=D3D11_USAGE_STAGING; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        const auto result=Device->CreateBuffer(&buffer,nullptr,readback[slot].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create PN staging stream",result));
    }
    ComPtr<ID3D11Buffer> parameters; D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth=sizeof(constants); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&constants;
    auto result=Device->CreateBuffer(&buffer,&initial,parameters.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create PN constants",result));
    ID3D11ShaderResourceView* inputs[5]{}; ID3D11UnorderedAccessView* outputs[3]{};
    for (std::size_t slot=0;slot<views.size();++slot) inputs[slot]=views[slot].Get();
    for (std::size_t slot=0;slot<unordered.size();++slot) outputs[slot]=unordered[slot].Get();
    auto* uniform=parameters.Get(); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,5,inputs); Context->CSSetUnorderedAccessViews(0,3,outputs,nullptr);
    const unsigned counts[]{constants.Inputs.W,constants.Outputs.X,constants.Outputs.Y/3};
    for (std::size_t phase=0;phase<TessellationShaders.size();++phase)
    { Context->CSSetShader(TessellationShaders[phase].Get(),nullptr,0); Context->Dispatch((counts[phase]+TessellationThreadCount-1)/TessellationThreadCount,1,1); }
    for (auto& input:inputs) input=nullptr; for (auto& target:outputs) target=nullptr; uniform=nullptr;
    Context->CSSetUnorderedAccessViews(0,3,outputs,nullptr); Context->CSSetShaderResources(0,5,inputs);
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
    TessellatedGeometryBuffers candidate; candidate.Vertices.resize(constants.Outputs.X); candidate.Indices.resize(constants.Outputs.Y);
    void* mirrors[]{candidate.Vertices.data(),candidate.Indices.data()};
    for (std::size_t slot=0;slot<resources.size();++slot)
    {
        auto& resource=resources[slot]; const auto bytes=resource.Description.SizeInBytes; resource.Bytes.resize(bytes);
        Context->CopyResource(readback[slot].Get(),targets[slot+1].Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
        result=Context->Map(readback[slot].Get(),0,D3D11_MAP_READ,0,&mapped);
        if (FAILED(result)) return Fail(HResultMessage("Read PN output stream",result));
        std::memcpy(mirrors[slot],mapped.pData,bytes); std::memcpy(resource.Bytes.data(),mapped.pData,bytes); Context->Unmap(readback[slot].Get(),0);
        resource.Buffer=targets[slot+1];
    }
    if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,LastError)) return false;
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=3;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    output=std::move(candidate); return true;
}
