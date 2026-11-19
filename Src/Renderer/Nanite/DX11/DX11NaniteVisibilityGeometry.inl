// Assembly owns only views/constants; the decoded source allocations stay in
// Buffers and are locked until Present/Abort. No CPU geometry mirror is read.
struct NaniteAssemblySource
{
    NaniteVisibilityGeometryConstants Constants; unsigned GroupsX=0,GroupsY=0;
    BufferHandle Vertex,Index; std::array<ComPtr<ID3D11ShaderResourceView>,2> Views;
};
std::vector<NaniteAssemblySource> NaniteAssemblySources;
ComPtr<ID3D11ComputeShader> NaniteAssemblyShader;
ComPtr<ID3D11Buffer> NaniteAssemblyUniform;
ComPtr<ID3D11UnorderedAccessView> NaniteAssemblyTarget;
void DestroyNaniteVisibilityGeometry()
{ DestroyNaniteInstances();
    NaniteAssemblySources.clear(); NaniteAssemblyShader.Reset(); NaniteAssemblyUniform.Reset(); NaniteAssemblyTarget.Reset();
}
bool PrepareNaniteVisibilityGeometry(const VisibilityScene& scene)
{
    if (!ValidateVisibilityScene(scene,LastError)) return false;
    std::vector<NaniteAssemblySource> sources; sources.reserve(scene.NativeMeshes.size());
    for (unsigned i=0;i<scene.NativeMeshes.size();++i)
    {
        const auto& mesh=scene.NativeMeshes[i]; const auto vertex=Buffers.find(mesh.VertexBuffer.Value),index=Buffers.find(mesh.IndexBuffer.Value);
        if (vertex==Buffers.end() || index==Buffers.end() || !vertex->second.NaniteGeometry || !index->second.NaniteGeometry)
            return Fail("DX11 visibility assembly requires native decoded geometry buffers");
        NaniteAssemblySource source; source.Vertex=mesh.VertexBuffer; source.Index=mesh.IndexBuffer;
        if (!BuildNaniteVisibilityGeometryConstants(scene,i,vertex->second.Description,index->second.Description,source.Constants,source.GroupsX,source.GroupsY,LastError)) return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
        view.Format=DXGI_FORMAT_R32_TYPELESS; view.BufferEx.NumElements=UINT(vertex->second.Description.SizeInBytes/sizeof(unsigned)); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
        auto status=Device->CreateShaderResourceView(vertex->second.Buffer.Get(),&view,source.Views[0].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite decoded vertex view",status));
        view={}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Format=DXGI_FORMAT_R32_UINT; view.Buffer.NumElements=UINT(index->second.Description.SizeInBytes/sizeof(unsigned));
        status=Device->CreateShaderResourceView(index->second.Buffer.Get(),&view,source.Views[1].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite decoded index view",status));
        sources.push_back(std::move(source));
    }
    if (!NaniteAssemblyShader)
    {
        if (!LoadTechniqueShaderSource("DX11","NaniteRaster.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO defines[]{{"EVANS_NANITE_VISIBILITY_GEOMETRY","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (!CompileShader("CSVisibilityGeometry","cs_5_0",code,defines,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,NaniteAssemblyShader.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite visibility assembly shader",status));
    }
    if (!NaniteAssemblyUniform)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(NaniteVisibilityGeometryConstants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,NaniteAssemblyUniform.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite assembly constants",status));
    }
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(scene.TriangleCount()*sizeof(VisibilityTriangle)); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(VisibilityTriangle);
    ComPtr<ID3D11Buffer> table; ComPtr<ID3D11ShaderResourceView> view; ComPtr<ID3D11UnorderedAccessView> target;
    auto status=Device->CreateBuffer(&desc,nullptr,table.GetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateShaderResourceView(table.Get(),nullptr,view.GetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateUnorderedAccessView(table.Get(),nullptr,target.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Nanite assembled triangle table",status));
    VisibilityGeometry=std::move(table); VisibilityGeometryView=std::move(view); NaniteAssemblyTarget=std::move(target);
    NaniteAssemblySources=std::move(sources); UploadedVisibility.reset(); return true;
}
bool DrawNaniteVisibilityGeometry()
{
    NaniteCopyLocks.reserve(NaniteCopyLocks.size()+NaniteAssemblySources.size()*2);
    for (const auto& source:NaniteAssemblySources) { NaniteCopyLocks.push_back(source.Vertex); NaniteCopyLocks.push_back(source.Index); }
    ID3D11ShaderResourceView* empty[6]{}; Context->VSSetShaderResources(0,6,empty); Context->PSSetShaderResources(0,6,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* uniform=NaniteAssemblyUniform.Get(); auto* target=NaniteAssemblyTarget.Get();
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShader(NaniteAssemblyShader.Get(),nullptr,0);
    for (const auto& source:NaniteAssemblySources)
    {
        Context->UpdateSubresource(uniform,0,nullptr,&source.Constants,0,0);
        ID3D11ShaderResourceView* views[]{source.Views[0].Get(),source.Views[1].Get()}; Context->CSSetShaderResources(0,2,views);
        Context->Dispatch(source.GroupsX,source.GroupsY,1); ++PendingFrame.Statistics.NaniteGeometryAssemblyCount;
    }
    target=nullptr; Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShaderResources(0,2,empty); Context->CSSetShader(nullptr,nullptr,0); return true;
}
