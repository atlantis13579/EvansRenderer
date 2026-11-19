ComPtr<ID3D12Device5> LumenRayDevice;
enum class LumenHardwareProgram:unsigned { Distance,Radiance,Probes,Direct,Bounce,Visibility,Count };
std::array<ComPtr<ID3D12PipelineState>,unsigned(LumenHardwareProgram::Count)> LumenHardwarePipelines;
void LumenHardwareRoots(D3D12_ROOT_PARAMETER* parameters)
{
    for (unsigned slot=0;slot<2;++slot)
    { parameters[slot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot].Descriptor.ShaderRegister=8+slot; }
}
bool EnsureLumenHardwarePipeline(LumenHardwareProgram program,ID3D12RootSignature* root)
{
    auto& pipeline=LumenHardwarePipelines[unsigned(program)]; if (pipeline) return true;
    const wchar_t* stages[]{L"EVANS_LUMEN_DISTANCE_TRACE_STAGE=1",L"EVANS_LUMEN_RADIANCE_TRACE_STAGE=1",L"EVANS_LUMEN_PROBE_TRACE_STAGE=1",
        L"EVANS_LUMEN_LIGHTING_STAGE=1",L"EVANS_LUMEN_LIGHTING_STAGE=1",L"EVANS_LUMEN_GATHER_VISIBILITY_STAGE=1"};
    const wchar_t* entries[]{L"CSTraceDistance",L"CSTraceDistance",L"CSTraceDistance",L"CSLightSurface",L"CSLightSurface",L"CSProbeVisibility"};
    ComPtr<IDxcBlob> binary;
    if (!CompileDXILCode("Lumen.hlsl",entries[unsigned(program)],L"cs_6_5",{stages[unsigned(program)],L"EVANS_LUMEN_HARDWARE_RAYS=1",
        program==LumenHardwareProgram::Bounce?L"EVANS_LUMEN_RADIOSITY=1":nullptr},binary,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC create{}; create.pRootSignature=root; create.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&create,IID_PPV_ARGS(pipeline.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen inline-ray pipeline",status);
}
void BindLumenHardware(ID3D12GraphicsCommandList* commands,const LumenDistanceRecord& distance,unsigned root)
{
    if (!distance.Hardware) return;
    commands->SetComputeRootShaderResourceView(root,distance.Hardware->Top->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(root+1,distance.Hardware->Words->GetGPUVirtualAddress());
}
bool SupportsLumenHardwareRayTracing() const
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{}; ComPtr<ID3D12Device5> device;
    return SupportsDXIL65() && SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,&options,sizeof(options))) &&
        options.RaytracingTier>=D3D12_RAYTRACING_TIER_1_1 && SUCCEEDED(Device.As(&device));
}
bool LumenHardwareUpload(LumenDistanceWork& work,const void* source,std::size_t bytes,ComPtr<ID3D12Resource>& result)
{
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,result)) return false;
    void* mapped=nullptr; const D3D12_RANGE none{0,0}; const auto status=result->Map(0,&none,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Lumen hardware input",status);
    std::memcpy(mapped,source,bytes); result->Unmap(0,nullptr); work.HardwareInputs.push_back(result); return true;
}
bool LumenHardwareStorage(UINT64 bytes,D3D12_RESOURCE_STATES state,ComPtr<ID3D12Resource>& result)
{
    if (!bytes) return Fail("DX12 hardware acceleration structure reported zero storage");
    auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(result.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Allocate Lumen acceleration structure",status);
}
bool BuildLumenHardwareAcceleration(LumenDistanceWork& work)
{
    const auto& snapshot=work.Record->Info.Hardware;
    if (!snapshot || snapshot->Instances.empty() || !SupportsLumenHardwareRayTracing()) return true;
    if (!LumenRayDevice && FAILED(Device.As(&LumenRayDevice))) return Fail("DX12 raytracing device interface is unavailable");
    ComPtr<ID3D12GraphicsCommandList4> commands;
    if (FAILED(work.Commands.As(&commands))) return Fail("DX12 acceleration-structure command interface is unavailable");
    auto record=std::make_shared<LumenHardwareRecord>();
    const unsigned empty=0; ComPtr<ID3D12Resource> words;
    const auto wordBytes=std::max(std::size_t{1},snapshot->Words.size())*sizeof(unsigned);
    if (!LumenHardwareUpload(work,snapshot->Words.empty()?&empty:snapshot->Words.data(),wordBytes,words) ||
        !CreateBufferResource(wordBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,record->Words)) return false;
    work.Commands->CopyBufferRegion(record->Words.Get(),0,words.Get(),0,wordBytes);
    D3D12_RESOURCE_BARRIER copied{}; copied.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copied.Transition={record->Words.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    work.Commands->ResourceBarrier(1,&copied);
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometry(snapshot->Meshes.size());
    std::vector<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO> sizes(snapshot->Meshes.size());
    UINT64 scratchSize=0;
    for (unsigned mesh=0;mesh<snapshot->Meshes.size();++mesh)
    {
        const auto& triangles=snapshot->Meshes[mesh]->Triangles; std::vector<Float3> vertices; vertices.reserve(triangles.size()*3);
        for (const auto& triangle:triangles) for (auto p:{triangle.A,triangle.B,triangle.C}) vertices.push_back({p.X,p.Y,p.Z});
        ComPtr<ID3D12Resource> input; if (!LumenHardwareUpload(work,vertices.data(),vertices.size()*sizeof(Float3),input)) return false;
        auto& desc=geometry[mesh]; desc.Type=D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        // Non-opaque geometry always exposes candidates to the material filter.
        desc.Triangles.VertexFormat=DXGI_FORMAT_R32G32B32_FLOAT; desc.Triangles.VertexCount=UINT(vertices.size());
        desc.Triangles.VertexBuffer={input->GetGPUVirtualAddress(),sizeof(Float3)};
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS build{}; build.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        build.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE; build.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;
        build.NumDescs=1; build.pGeometryDescs=&desc; LumenRayDevice->GetRaytracingAccelerationStructurePrebuildInfo(&build,&sizes[mesh]);
        scratchSize=std::max(scratchSize,sizes[mesh].ScratchDataSizeInBytes);
    }
    record->Bottom.resize(snapshot->Meshes.size()); std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(std::max(std::size_t{1},snapshot->Instances.size()));
    // The custom ID is the shared distance/card instance index. GI and shadow
    // masks are independent; per-triangle caster flags are checked by the query.
    constexpr unsigned GeometryMask=1,ShadowMask=2;
    for (unsigned mesh=0;mesh<record->Bottom.size();++mesh)
        if (!LumenHardwareStorage(sizes[mesh].ResultDataMaxSizeInBytes,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,record->Bottom[mesh])) return false;
    for (unsigned index=0;index<snapshot->Instances.size();++index)
    {
        const auto& source=snapshot->Instances[index]; auto& target=instances[index];
        std::memcpy(target.Transform,source.World.Values.data(),sizeof(target.Transform)); target.InstanceID=index;
        target.InstanceMask=GeometryMask|(source.CastShadow?ShadowMask:0); target.Flags=D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        target.AccelerationStructure=record->Bottom[source.Mesh]->GetGPUVirtualAddress();
    }
    ComPtr<ID3D12Resource> instanceInput; if (!LumenHardwareUpload(work,instances.data(),instances.size()*sizeof(instances[0]),instanceInput)) return false;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS top{}; top.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    top.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE; top.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;
    top.NumDescs=UINT(snapshot->Instances.size()); top.InstanceDescs=instanceInput->GetGPUVirtualAddress();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topSize{}; LumenRayDevice->GetRaytracingAccelerationStructurePrebuildInfo(&top,&topSize);
    scratchSize=std::max(scratchSize,topSize.ScratchDataSizeInBytes); ComPtr<ID3D12Resource> scratch;
    if (!LumenHardwareStorage(scratchSize,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,scratch) ||
        !LumenHardwareStorage(topSize.ResultDataMaxSizeInBytes,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,record->Top)) return false;
    work.HardwareInputs.push_back(scratch); work.Record->Hardware=record;
    D3D12_RESOURCE_BARRIER ready{}; ready.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; // Global barrier also protects reused scratch.
    for (unsigned mesh=0;mesh<geometry.size();++mesh)
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{}; build.Inputs.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        build.Inputs.Flags=top.Flags; build.Inputs.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY; build.Inputs.NumDescs=1; build.Inputs.pGeometryDescs=&geometry[mesh];
        build.DestAccelerationStructureData=record->Bottom[mesh]->GetGPUVirtualAddress(); build.ScratchAccelerationStructureData=scratch->GetGPUVirtualAddress();
        commands->BuildRaytracingAccelerationStructure(&build,0,nullptr); commands->ResourceBarrier(1,&ready);
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{}; build.Inputs=top; build.DestAccelerationStructureData=record->Top->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData=scratch->GetGPUVirtualAddress(); commands->BuildRaytracingAccelerationStructure(&build,0,nullptr); commands->ResourceBarrier(1,&ready);
    return true;
}
