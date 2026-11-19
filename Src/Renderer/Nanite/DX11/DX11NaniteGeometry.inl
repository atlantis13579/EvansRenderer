// Immediate-context command order and staging Map complete the two transactions.
// Prefix feedback determines exact output allocation sizes. Published handles own
// the original UAV vertex/index resources; no CPU decoding or re-upload is used.
std::array<ComPtr<ID3D11ComputeShader>,5> NaniteGeometryShaders;
void DestroyNaniteGeometry() { DestroyNaniteAsync(); DestroyNaniteCacheAsync(); DestroyNaniteCaches(); for (auto& shader:NaniteGeometryShaders) shader.Reset(); }
bool EnsureNaniteGeometry()
{
    if (!LoadTechniqueShaderSource("DX11","NaniteGeometry.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSEvaluate","CSRefine","CSClusters","CSBlocks","CSDecode"};
    for (std::size_t stage=0;stage<NaniteGeometryShaders.size();++stage) if (!NaniteGeometryShaders[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[stage],"cs_5_0",code,nullptr,true)) return false;
        const auto result=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,NaniteGeometryShaders[stage].GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create Nanite geometry compute shader",result));
    }
    return true;
}
bool CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output,NaniteCacheHandle cache={})
{
    std::vector<NaniteGpuDispatch> jobs;
    if (!BuildNaniteGpuSchedule(upload,jobs,LastError) || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { const auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("DX11 Nanite cache handle is invalid"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    struct ClearBindings
    {
        ID3D11DeviceContext* Context;
        ~ClearBindings()
        {
            ID3D11ShaderResourceView* sources[3]{}; ID3D11UnorderedAccessView* targets[3]{}; ID3D11Buffer* uniform=nullptr;
            Context->CSSetUnorderedAccessViews(0,3,targets,nullptr); Context->CSSetShaderResources(0,3,sources);
            Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
        }
    } clear{Context.Get()};
    const void* sourceData[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data()};
    const std::size_t sourceBytes[]{upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage)};
    std::array<ComPtr<ID3D11Buffer>,3> sources,targets;
    std::array<ComPtr<ID3D11ShaderResourceView>,3> views; std::array<ComPtr<ID3D11UnorderedAccessView>,3> unordered;
    for (std::size_t slot=0;slot<sources.size();++slot)
    {
        if (slot==2 && cached) { sources[slot]=cached->Pool->Buffer; views[slot]=cached->Pool->View; continue; }
        if (sourceBytes[slot]>UINT32_MAX) return Fail("Nanite geometry source exceeds D3D11 byte addressing");
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(sourceBytes[slot]); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags=slot==1?D3D11_RESOURCE_MISC_BUFFER_STRUCTURED:D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS; desc.StructureByteStride=slot==1?sizeof(ShaderUInt4):0;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=sourceData[slot];
        auto result=Device->CreateBuffer(&desc,&initial,sources[slot].GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite source",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=slot==1?DXGI_FORMAT_UNKNOWN:DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension=slot==1?D3D11_SRV_DIMENSION_BUFFER:D3D11_SRV_DIMENSION_BUFFEREX;
        if (slot==1) view.Buffer.NumElements=desc.ByteWidth/sizeof(ShaderUInt4);
        else { view.BufferEx.NumElements=desc.ByteWidth/sizeof(std::uint32_t); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW; }
        result=Device->CreateShaderResourceView(sources[slot].Get(),&view,views[slot].GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite source view",result));
    }
    auto target=[&](unsigned slot,std::size_t bytes,const void* data)
    {
        if (!bytes || bytes>UINT32_MAX) return Fail("Nanite geometry UAV exceeds D3D11 addressing");
        targets[slot].Reset(); unordered[slot].Reset(); D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE|(slot==1?D3D11_BIND_VERTEX_BUFFER:slot==2?D3D11_BIND_INDEX_BUFFER:0);
        if (slot!=2) desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data;
        auto result=Device->CreateBuffer(&desc,data?&initial:nullptr,targets[slot].GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite UAV",result));
        D3D11_UNORDERED_ACCESS_VIEW_DESC view{}; view.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; view.Format=slot==2?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_R32_TYPELESS;
        view.Buffer.NumElements=desc.ByteWidth/sizeof(std::uint32_t); if (slot!=2) view.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
        result=Device->CreateUnorderedAccessView(targets[slot].Get(),&view,unordered[slot].GetAddressOf()); return SUCCEEDED(result) || Fail(HResultMessage("Create Nanite UAV view",result));
    };
    const std::size_t stateBytes=upload.InitialState.size()*sizeof(std::uint32_t);
    if (!target(0,stateBytes,upload.InitialState.data()) || !target(1,sizeof(std::uint32_t),nullptr) || !target(2,sizeof(std::uint32_t),nullptr)) return false;
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC uniformDesc{}; uniformDesc.ByteWidth=sizeof(NaniteGpuConstants); uniformDesc.Usage=D3D11_USAGE_DEFAULT; uniformDesc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    auto result=Device->CreateBuffer(&uniformDesc,nullptr,uniform.GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite constants",result));
    auto bindTargets=[&](){ID3D11UnorderedAccessView* values[]{unordered[0].Get(),unordered[1].Get(),unordered[2].Get()}; Context->CSSetUnorderedAccessViews(0,3,values,nullptr);};
    auto unbindTargets=[&](){ID3D11UnorderedAccessView* values[3]{}; Context->CSSetUnorderedAccessViews(0,3,values,nullptr);};
    ID3D11ShaderResourceView* inputs[]{views[0].Get(),views[1].Get(),views[2].Get()}; auto* cb=uniform.Get();
    Context->CSSetShaderResources(0,3,inputs); Context->CSSetConstantBuffers(0,1,&cb); bindTargets();
    auto dispatch=[&](const NaniteGpuDispatch& job,const NaniteGpuConstants& constants)
    {
        Context->UpdateSubresource(uniform.Get(),0,nullptr,&constants,0,0); Context->CSSetShader(NaniteGeometryShaders[static_cast<unsigned>(job.Pass)].Get(),nullptr,0);
        Context->Dispatch(job.GroupsX,job.GroupsY,1);
    };
    for (std::size_t stage=0;stage+1<jobs.size();++stage) dispatch(jobs[stage],jobs[stage].Constants);
    unbindTargets(); NaniteGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size());
    auto read=[&](ID3D11Buffer* source,std::size_t bytes,void* destination)
    {
        ComPtr<ID3D11Buffer> staging; D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite readback",status));
        Context->CopyResource(staging.Get(),source); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
        if (FAILED(status)) return Fail(HResultMessage("Map Nanite feedback",status)); std::memcpy(destination,mapped.pData,bytes); Context->Unmap(staging.Get(),0); return true;
    };
    if (!read(targets[0].Get(),stateBytes,candidate.Readback.State.data()) || !ValidateNaniteGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (!target(1,vertexBytes,nullptr) || !target(2,indexBytes,nullptr)) return false;
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount;
    bindTargets(); dispatch(jobs.back(),constants); unbindTargets(); candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    if (!read(targets[0].Get(),stateBytes,candidate.Readback.State.data()) || !read(targets[1].Get(),vertexBytes,candidate.Readback.Vertices.data()) ||
        !read(targets[2].Get(),indexBytes,candidate.Readback.Indices.data()) || !ValidateNaniteGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    { auto& resource=resources[stream]; resource.NaniteGeometry=true; resource.Buffer=targets[stream+1]; resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size()); }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(jobs.size());
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    output=std::move(candidate); return true;
}
