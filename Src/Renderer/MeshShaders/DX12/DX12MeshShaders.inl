// D3D12 mesh pipelines use native DXIL, compiled from readable HLSL at runtime.
// Dispatch dimensions stay GPU-owned for Nanite; no queue count is read back.
bool MeshBarycentricChecked=false,MeshNativeBarycentrics=false;
bool UseNativeMeshBarycentrics()
{
    if (!MeshBarycentricChecked)
    {
        MeshBarycentricChecked=true; D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
        MeshNativeBarycentrics=SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3,&options,sizeof(options))) && options.BarycentricsSupported;
    }
    return MeshNativeBarycentrics;
}
ComPtr<ID3D12GraphicsCommandList6> MeshCommandList;
std::array<ComPtr<IDxcBlob>,3> MeshBytecode; // Direct, Nanite, task-fed ordinary mesh.
ComPtr<IDxcBlob> MeshTaskBytecode;
std::array<ComPtr<IDxcBlob>,3> MeshPixelBytecode;
ComPtr<ID3D12PipelineState> MeshArgumentPipeline;
ComPtr<ID3D12CommandSignature> MeshDispatchSignature;
std::shared_ptr<NaniteFrameBuffer> MeshDispatchArguments;
std::unordered_map<unsigned,ComPtr<ID3D12PipelineState>> MeshPipelines;

bool SupportsMeshShaders() const
{
    if (!Device) return false;
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    return SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7,&options,sizeof(options))) &&
        options.MeshShaderTier!=D3D12_MESH_SHADER_TIER_NOT_SUPPORTED && SupportsDXIL65();
}
void DestroyMeshShaders()
{
    MeshPipelines.clear(); MeshBytecode={}; MeshTaskBytecode.Reset(); MeshPixelBytecode={}; MeshCommandList.Reset();
    MeshArgumentPipeline.Reset(); MeshDispatchSignature.Reset(); MeshDispatchArguments.reset();
    MeshBarycentricChecked=MeshNativeBarycentrics=false;
}
bool CompileMeshCode(const char* file,const wchar_t* entry,const wchar_t* profile,
    const wchar_t* define,ComPtr<IDxcBlob>& output,const wchar_t* extraDefine=nullptr)
{
    if (!SupportsMeshShaders()) return Fail("DX12 mesh stages require SM 6.5, mesh tier 1 and the runtime DXC compiler");
    return CompileDXILCode(file,entry,profile,{define,extraDefine},output);
}
bool PrepareMeshShaders(const VisibilityBufferDesc& frame)
{
    if (!frame.MeshShaders) return true;
    if (!SupportsMeshShaders()) return Fail("DX12 native mesh stages are unavailable on this device");
    if (!MeshCommandList && FAILED(CommandList.As(&MeshCommandList))) return Fail("DX12 mesh command-list interface is unavailable");
    if (frame.NaniteRaster && !HardwarePlan.Active()) return true;
    const wchar_t* nativeBary=!frame.NaniteRaster && UseNativeMeshBarycentrics()?L"EVANS_MESH_NATIVE_BARYCENTRICS=1":nullptr;
    bool needed[3]{false,frame.NaniteRaster,false};
    if (!frame.NaniteRaster) for (const auto& draw:frame.Draws)
        needed[UseMeshTaskStage(frame.MeshDispatch,frame.Scene->Draws[draw.Range].TriangleCount,true)?2:0]=true;
    for (unsigned variant=0;variant<3;++variant)
        if (needed[variant] && !MeshBytecode[variant] && !CompileMeshCode("MeshShaders.hlsl",L"MSMesh",L"ms_6_5",
            variant==1?L"EVANS_MESH_NANITE=1":nativeBary,MeshBytecode[variant],variant==2?L"EVANS_MESH_TASK_INPUT=1":nullptr)) return false;
    if (needed[2] && !MeshTaskBytecode && !CompileMeshCode("MeshShaders.hlsl",L"ASMesh",L"as_6_5",
        L"EVANS_MESH_TASK_STAGE=1",MeshTaskBytecode)) return false;
    if (!frame.NaniteRaster)
        return MeshPixelBytecode[0] || CompileMeshCode("VisibilityBuffer.hlsl",L"PSMeshRaster",L"ps_6_5",nativeBary,MeshPixelBytecode[0]);
    for (unsigned phase=0;phase<2;++phase)
        if (!MeshPixelBytecode[phase+1] && !CompileMeshCode("NaniteHardware.hlsl",phase?L"PSOwner":L"PSDepth",
            L"ps_6_5",L"EVANS_HARDWARE_GRAPHICS=1",MeshPixelBytecode[phase+1])) return false;
    if (!MeshArgumentPipeline)
    {
        ComPtr<IDxcBlob> code;
        if (!CompileMeshCode("MeshShaders.hlsl",L"CSArguments",L"cs_6_5",L"EVANS_MESH_ARGUMENTS=1",code)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=HardwareComputeRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(MeshArgumentPipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create mesh dispatch preparation pipeline",status);
    }
    if (!MeshDispatchSignature)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argument{}; argument.Type=D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        D3D12_COMMAND_SIGNATURE_DESC desc{}; desc.ByteStride=sizeof(MeshShaderGrid); desc.NumArgumentDescs=1; desc.pArgumentDescs=&argument;
        const auto status=Device->CreateCommandSignature(&desc,nullptr,IID_PPV_ARGS(MeshDispatchSignature.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create mesh indirect signature",status);
    }
    return !HardwarePlan.Active() || CreateNaniteFrameBuffer(unsigned(HardwarePlan.Bins.size())*MeshShaderArgumentWords,nullptr,true,MeshDispatchArguments);
}
// D3D12 requires pointer-aligned stream subobjects, including trailing padding.
#pragma warning(push)
#pragma warning(disable:4324)
template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind,class Value>
struct alignas(void*) MeshStatePart { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type=Kind; Value Data{}; };
#pragma warning(pop)
ID3D12PipelineState* MeshPipeline(const PipelineState& state,bool mirrored,bool nanite,unsigned phase=0,bool task=false)
{
    const unsigned key=unsigned(state.BackFaceCulling)|(unsigned(mirrored)<<1)|(unsigned(state.DepthWrite)<<2)|
        (unsigned(state.DepthTest)<<3)|(unsigned(state.Fill==FillMode::Wireframe)<<4)|(unsigned(nanite)<<5)|(phase<<6)|(unsigned(task)<<8);
    const auto found=MeshPipelines.find(key); if (found!=MeshPipelines.end()) return found->second.Get();
    struct Stream
    {
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,ID3D12RootSignature*> Root;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,D3D12_SHADER_BYTECODE> Mesh;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,D3D12_SHADER_BYTECODE> Task;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,D3D12_SHADER_BYTECODE> Pixel;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,D3D12_BLEND_DESC> Blend;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,UINT> Mask;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,D3D12_RASTERIZER_DESC> Raster;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,D3D12_DEPTH_STENCIL_DESC> Depth;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,D3D12_PRIMITIVE_TOPOLOGY_TYPE> Topology;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,D3D12_RT_FORMAT_ARRAY> Targets;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,DXGI_FORMAT> DepthFormat;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,DXGI_SAMPLE_DESC> Samples;
        MeshStatePart<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,D3D12_CACHED_PIPELINE_STATE> Cached;
    } stream;
    stream.Root.Data=nanite?HardwareGraphicsRoot.Get():VisibilityRoot.Get();
    const auto& mesh=MeshBytecode[nanite?1:task?2:0]; const auto& pixel=MeshPixelBytecode[nanite?phase+1:0];
    stream.Mesh.Data={mesh->GetBufferPointer(),mesh->GetBufferSize()}; stream.Pixel.Data={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    if (task) stream.Task.Data={MeshTaskBytecode->GetBufferPointer(),MeshTaskBytecode->GetBufferSize()};
    stream.Blend.Data.IndependentBlendEnable=TRUE;
    for (unsigned target=0;target<2;++target)
    {
        auto& blend=stream.Blend.Data.RenderTarget[target]; blend.RenderTargetWriteMask=nanite?0:D3D12_COLOR_WRITE_ENABLE_ALL;
        blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
    }
    auto& raster=stream.Raster.Data; raster.FillMode=state.Fill==FillMode::Wireframe?D3D12_FILL_MODE_WIREFRAME:D3D12_FILL_MODE_SOLID;
    raster.CullMode=state.BackFaceCulling?D3D12_CULL_MODE_BACK:D3D12_CULL_MODE_NONE;
    raster.FrontCounterClockwise=mirrored?FALSE:TRUE; raster.DepthClipEnable=TRUE;
    auto& depth=stream.Depth.Data; depth.DepthEnable=nanite || state.DepthTest;
    depth.DepthWriteMask=(nanite?phase==0:state.DepthWrite)?D3D12_DEPTH_WRITE_MASK_ALL:D3D12_DEPTH_WRITE_MASK_ZERO;
    depth.DepthFunc=nanite?(phase?D3D12_COMPARISON_FUNC_EQUAL:D3D12_COMPARISON_FUNC_LESS):state.DepthWrite?D3D12_COMPARISON_FUNC_LESS:D3D12_COMPARISON_FUNC_LESS_EQUAL;
    auto& stencil=depth.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; depth.BackFace=stencil;
    stream.Topology.Data=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; stream.Targets.Data.NumRenderTargets=2;
    stream.Targets.Data.RTFormats[0]=DXGI_FORMAT_R32_UINT; stream.Targets.Data.RTFormats[1]=DXGI_FORMAT_R32G32_FLOAT;
    stream.DepthFormat.Data=DXGI_FORMAT_D32_FLOAT; stream.Mask.Data=UINT_MAX; stream.Samples.Data={1,0};
    ComPtr<ID3D12Device2> device; auto status=Device.As(&device);
    if (FAILED(status)) { FailHRESULT("Query mesh pipeline device",status); return nullptr; }
    D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream),&stream}; ComPtr<ID3D12PipelineState> pipeline;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC canonical{};
    canonical.pRootSignature=stream.Root.Data;canonical.PS=stream.Pixel.Data;canonical.BlendState=stream.Blend.Data;
    canonical.SampleMask=stream.Mask.Data;canonical.RasterizerState=stream.Raster.Data;canonical.DepthStencilState=stream.Depth.Data;
    canonical.PrimitiveTopologyType=stream.Topology.Data;canonical.NumRenderTargets=stream.Targets.Data.NumRenderTargets;
    std::copy(std::begin(stream.Targets.Data.RTFormats),std::end(stream.Targets.Data.RTFormats),std::begin(canonical.RTVFormats));
    canonical.DSVFormat=stream.DepthFormat.Data;canonical.SampleDesc=stream.Samples.Data;
    status=CreateCachedMeshPipelineState(device.Get(),desc,stream.Cached.Data,canonical,stream.Mesh.Data,stream.Task.Data,IID_PPV_ARGS(pipeline.GetAddressOf()));
    if (FAILED(status)) { FailHRESULT("Create native mesh pipeline",status); return nullptr; }
    auto* result=pipeline.Get(); MeshPipelines.emplace(key,std::move(pipeline)); return result;
}
void BuildMeshDispatchArguments()
{
    Transition(HardwareArguments->Buffer.Get(),HardwareArguments->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    Transition(MeshDispatchArguments->Buffer.Get(),MeshDispatchArguments->State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootSignature(HardwareComputeRoot.Get()); CommandList->SetPipelineState(MeshArgumentPipeline.Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(NaniteHardwareConstants)/sizeof(unsigned),&HardwarePlan.Constants,0);
    CommandList->SetComputeRootShaderResourceView(1,HardwareArguments->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(3,MeshDispatchArguments->Buffer->GetGPUVirtualAddress());
    const auto grid=BuildMeshShaderGrid(unsigned(HardwarePlan.Bins.size())); CommandList->Dispatch(grid.X,grid.Y,grid.Z);
    Transition(MeshDispatchArguments->Buffer.Get(),MeshDispatchArguments->State,D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
}
