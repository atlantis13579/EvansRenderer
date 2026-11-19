// Root descriptors address decoded GPU streams directly. Assembly runs inside
// the frame command list and publishes the world table before raster compute.
struct NaniteAssemblySource
{
    NaniteVisibilityGeometryConstants Constants; unsigned GroupsX=0,GroupsY=0; BufferHandle Vertex,Index;
};
std::vector<NaniteAssemblySource> NaniteAssemblySources;
ComPtr<ID3D12RootSignature> NaniteAssemblyRoot;
ComPtr<ID3D12PipelineState> NaniteAssemblyPipeline;
D3D12_RESOURCE_STATES NaniteAssemblyState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
void DestroyNaniteVisibilityGeometry()
{ DestroyNaniteInstances(); NaniteAssemblySources.clear(); NaniteAssemblyPipeline.Reset(); NaniteAssemblyRoot.Reset(); }
bool PrepareNaniteVisibilityGeometry(const VisibilityScene& scene)
{
    if (!ValidateVisibilityScene(scene,LastError)) return false;
    std::vector<NaniteAssemblySource> sources; sources.reserve(scene.NativeMeshes.size());
    for (unsigned i=0;i<scene.NativeMeshes.size();++i)
    {
        const auto& mesh=scene.NativeMeshes[i]; const auto vertex=Buffers.find(mesh.VertexBuffer.Value),index=Buffers.find(mesh.IndexBuffer.Value);
        if (vertex==Buffers.end() || index==Buffers.end() || !vertex->second.NaniteGeometry || !index->second.NaniteGeometry)
            return Fail("DX12 visibility assembly requires native decoded buffers");
        NaniteAssemblySource source; source.Vertex=mesh.VertexBuffer; source.Index=mesh.IndexBuffer;
        if (!BuildNaniteVisibilityGeometryConstants(scene,i,vertex->second.Description,index->second.Description,source.Constants,source.GroupsX,source.GroupsY,LastError)) return false;
        sources.push_back(source);
    }
    if (!NaniteAssemblyRoot)
    {
        D3D12_ROOT_PARAMETER p[4]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues=sizeof(NaniteVisibilityGeometryConstants)/sizeof(unsigned);
        p[1].ParameterType=p[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[2].Descriptor.ShaderRegister=1;
        p[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=4; desc.pParameters=p; ComPtr<ID3DBlob> code,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(NaniteAssemblyRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite assembly root",status);
    }
    if (!NaniteAssemblyPipeline)
    {
        if (!LoadTechniqueShaderSource("DX12","NaniteRaster.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO defines[]{{"EVANS_NANITE_VISIBILITY_GEOMETRY","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (!CompileShader("CSVisibilityGeometry","cs_5_0",code,defines,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=NaniteAssemblyRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(NaniteAssemblyPipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite assembly pipeline",status);
    }
    auto desc=BufferResourceDesc(scene.TriangleCount()*sizeof(VisibilityTriangle)); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); ComPtr<ID3D12Resource> table;
    const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(table.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Nanite GPU world table",status);
    VisibilityGeometry=std::move(table); NaniteAssemblyState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    NaniteAssemblySources=std::move(sources); UploadedVisibility.reset(); return true;
}
bool DrawNaniteVisibilityGeometry()
{
    NaniteCopyLocks.reserve(NaniteCopyLocks.size()+NaniteAssemblySources.size()*2);
    for (const auto& source:NaniteAssemblySources) { NaniteCopyLocks.push_back(source.Vertex); NaniteCopyLocks.push_back(source.Index); }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    CommandList->SetComputeRootSignature(NaniteAssemblyRoot.Get()); CommandList->SetPipelineState(NaniteAssemblyPipeline.Get());
    Transition(VisibilityGeometry.Get(),NaniteAssemblyState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootUnorderedAccessView(3,VisibilityGeometry->GetGPUVirtualAddress());
    for (const auto& source:NaniteAssemblySources)
    {
        auto& vertex=Buffers.at(source.Vertex.Value); auto& index=Buffers.at(source.Index.Value); const auto vs=vertex.State,is=index.State;
        Transition(vertex.Resource.Get(),vertex.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); Transition(index.Resource.Get(),index.State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CommandList->SetComputeRootShaderResourceView(1,vertex.Resource->GetGPUVirtualAddress()); CommandList->SetComputeRootShaderResourceView(2,index.Resource->GetGPUVirtualAddress());
        CommandList->SetComputeRoot32BitConstants(0,sizeof(source.Constants)/sizeof(unsigned),&source.Constants,0);
        CommandList->Dispatch(source.GroupsX,source.GroupsY,1); ++PendingFrame.Statistics.NaniteGeometryAssemblyCount;
        Transition(vertex.Resource.Get(),vertex.State,vs); Transition(index.Resource.Get(),index.State,is);
    }
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=VisibilityGeometry.Get(); CommandList->ResourceBarrier(1,&barrier);
    Transition(VisibilityGeometry.Get(),NaniteAssemblyState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); return true;
}
