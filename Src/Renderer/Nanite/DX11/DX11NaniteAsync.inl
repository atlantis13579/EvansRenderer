// One immediate-context submission, followed by a non-flushing event query.
// Staging maps occur only after completion; cancellation keeps the query and
// every referenced allocation alive until the event is observed.
struct NaniteAsyncWork
{
    NaniteGpuUpload Upload; NaniteAsyncPlan Plan;
    std::array<ComPtr<ID3D11Buffer>,3> Sources,Targets,Staging;
    std::array<ComPtr<ID3D11ShaderResourceView>,3> Views;
    std::array<ComPtr<ID3D11UnorderedAccessView>,3> Unordered;
    ComPtr<ID3D11Buffer> Uniform; ComPtr<ID3D11Query> Completion;
};
NaniteAsyncQueue<NaniteAsyncWork> NaniteAsyncTasks;
void DestroyNaniteAsync() { NaniteAsyncTasks.Clear(); }
bool ProbeNaniteAsync(NaniteAsyncWork& work,bool& ready)
{
    BOOL done=FALSE; const auto status=Context->GetData(work.Completion.Get(),&done,sizeof(done),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (FAILED(status)) return Fail(HResultMessage("Poll Nanite completion",status));
    ready=status==S_OK && done; return true;
}
bool CollectNaniteAsync() { return NaniteAsyncTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteAsync(work,ready);}); }
bool CancelNaniteGeometry(NaniteGeometryTask task)
{ return (NaniteAsyncTasks.Cancel(task) || Fail("Invalid Nanite task")) && CollectNaniteAsync(); }
bool SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& output)
{
    auto work=std::make_shared<NaniteAsyncWork>();
    if (!CollectNaniteAsync() || !BuildNaniteAsyncPlan(upload,work->Plan,LastError) || !NaniteAsyncTasks.CanSubmit(work->Plan,LastError) || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite cache"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    work->Upload=upload;
    const void* inputs[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data()};
    const std::size_t inputBytes[]{upload.Metadata.size()*sizeof(unsigned),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage)};
    for (unsigned i=0;i<3;++i)
    {
        if (i==2 && cached) { work->Sources[i]=cached->Pool->Buffer; work->Views[i]=cached->Pool->View; continue; }
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(inputBytes[i]); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags=i==1?D3D11_RESOURCE_MISC_BUFFER_STRUCTURED:D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS; desc.StructureByteStride=i==1?sizeof(ShaderUInt4):0;
        D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=inputs[i];
        auto status=Device->CreateBuffer(&desc,&data,work->Sources[i].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite input",status));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=i==1?DXGI_FORMAT_UNKNOWN:DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=i==1?D3D11_SRV_DIMENSION_BUFFER:D3D11_SRV_DIMENSION_BUFFEREX;
        if (i==1) view.Buffer.NumElements=desc.ByteWidth/sizeof(ShaderUInt4);
        else { view.BufferEx.NumElements=desc.ByteWidth/sizeof(unsigned); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW; }
        status=Device->CreateShaderResourceView(work->Sources[i].Get(),&view,work->Views[i].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite SRV",status));
    }
    const std::size_t bytes[]{upload.InitialState.size()*sizeof(unsigned),work->Plan.VertexBytes,work->Plan.IndexBytes};
    for (unsigned i=0;i<3;++i)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(bytes[i]); desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE|(i==1?D3D11_BIND_VERTEX_BUFFER:i==2?D3D11_BIND_INDEX_BUFFER:0);
        desc.MiscFlags=i==2?0:D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=upload.InitialState.data();
        auto status=Device->CreateBuffer(&desc,i==0?&data:nullptr,work->Targets[i].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite target",status));
        D3D11_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format=i==2?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements=desc.ByteWidth/sizeof(unsigned); view.Buffer.Flags=i==2?0:D3D11_BUFFER_UAV_FLAG_RAW;
        status=Device->CreateUnorderedAccessView(work->Targets[i].Get(),&view,work->Unordered[i].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite UAV",status));
        if (i==0 || upload.ReadbackGeometry)
        {
            desc.BindFlags=desc.MiscFlags=0; desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            status=Device->CreateBuffer(&desc,nullptr,work->Staging[i].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite staging",status));
        }
    }
    D3D11_BUFFER_DESC uniform{}; uniform.ByteWidth=sizeof(NaniteGpuConstants); uniform.Usage=D3D11_USAGE_DEFAULT; uniform.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    auto status=Device->CreateBuffer(&uniform,nullptr,work->Uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create async Nanite constants",status));
    D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,work->Completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite completion",status));
    const auto task=NaniteAsyncTasks.Insert(work);
    ID3D11ShaderResourceView* views[]{work->Views[0].Get(),work->Views[1].Get(),work->Views[2].Get()};
    ID3D11UnorderedAccessView* targets[]{work->Unordered[0].Get(),work->Unordered[1].Get(),work->Unordered[2].Get()}; auto* cb=work->Uniform.Get();
    Context->CSSetShaderResources(0,3,views); Context->CSSetUnorderedAccessViews(0,3,targets,nullptr); Context->CSSetConstantBuffers(0,1,&cb);
    for (const auto& job:work->Plan.Dispatches)
    { Context->UpdateSubresource(cb,0,nullptr,&job.Constants,0,0); Context->CSSetShader(NaniteGeometryShaders[unsigned(job.Pass)].Get(),nullptr,0); Context->Dispatch(job.GroupsX,job.GroupsY,1); }
    ID3D11ShaderResourceView* clearViews[3]{}; ID3D11UnorderedAccessView* clearTargets[3]{}; cb=nullptr;
    Context->CSSetShaderResources(0,3,clearViews); Context->CSSetUnorderedAccessViews(0,3,clearTargets,nullptr); Context->CSSetConstantBuffers(0,1,&cb); Context->CSSetShader(nullptr,nullptr,0);
    for (unsigned i=0;i<3;++i) if (i==0 || upload.ReadbackGeometry) Context->CopyResource(work->Staging[i].Get(),work->Targets[i].Get());
    Context->End(work->Completion.Get()); Context->Flush(); output=task; return true;
}
bool PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    ready=false; auto work=NaniteAsyncTasks.Find(task); if (!work) return Fail("Invalid Nanite task");
    bool complete=false; if (!ProbeNaniteAsync(*work,complete)) return false; if (!complete) return true;
    const auto read=[&](unsigned i,void* target,std::size_t bytes)
    {
        D3D11_MAPPED_SUBRESOURCE mapped{}; const auto status=Context->Map(work->Staging[i].Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
        if (FAILED(status)) return Fail(HResultMessage("Map completed Nanite result",status));
        std::memcpy(target,mapped.pData,bytes); Context->Unmap(work->Staging[i].Get(),0); return true;
    };
    NaniteGeometryBuffers result; result.Readback.State.resize(work->Upload.InitialState.size());
    if (!read(0,result.Readback.State.data(),result.Readback.State.size()*sizeof(unsigned)) || !ValidateNaniteGpuState(work->Upload,result.Readback.State,result.Selection,LastError)) return false;
    const auto vertexBytes=std::size_t(result.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(result.Selection.IndexCount)*sizeof(unsigned);
    if (work->Upload.ReadbackGeometry)
    {
        result.Readback.Vertices.resize(result.Selection.VertexCount); result.Readback.Indices.resize(result.Selection.IndexCount);
        if (!read(1,result.Readback.Vertices.data(),vertexBytes) || !read(2,result.Readback.Indices.data(),indexBytes) || !ValidateNaniteGpuReadback(work->Upload,result.Readback,result.Selection,LastError)) return false;
    }
    std::array<BufferResource,2> resources; const void* mirrors[]{result.Readback.Vertices.data(),result.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite async vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite async indices"};
    for (unsigned i=0;i<2;++i)
    { auto& resource=resources[i]; resource.NaniteGeometry=true; resource.Buffer=work->Targets[i+1]; if (work->Upload.ReadbackGeometry) { resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[i],resource.Bytes.size()); } }
    result.VertexBuffer={NextBuffer++}; result.IndexBuffer={NextBuffer++}; result.DispatchCount=unsigned(work->Plan.Dispatches.size());
    try { Buffers.emplace(result.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(result.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(result.VertexBuffer.Value); throw; }
    NaniteAsyncTasks.Erase(task); output=std::move(result); ready=true; return true;
}
