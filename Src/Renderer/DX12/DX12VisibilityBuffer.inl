// D3D12 reads an immutable upload-heap triangle table. Per-draw CBV slices and
// texture descriptors stay immutable until the frame fence completes. Payload
// RTVs transition to pixel reads before the material pass; depth is preserved.
std::array<HdrSurface, 2> VisibilityImages;
ComPtr<ID3D12Resource> VisibilityGeometry;
std::shared_ptr<const VisibilityScene> UploadedVisibility;
static constexpr UINT VisibilityBufferSamplerRoot = 9;
ComPtr<ID3D12RootSignature> VisibilityRoot;
std::array<ComPtr<ID3DBlob>, 2> VisibilityVertices, VisibilityPixels;
std::unordered_map<UINT, ComPtr<ID3D12PipelineState>> VisibilityPipelines;
static constexpr UINT VisibilityConstantRoot = 0, VisibilityGeometryRoot = 1, VisibilityImageRoot = 2;

void DestroyVisibilityBuffer()
{
    DestroyMeshShaders();
    DestroyNaniteMaterials();
    DestroyNaniteFrameRaster();
    DestroyNaniteVisibilityGeometry();
    for (auto& image : VisibilityImages) { RTVArena.Release(image.RTV); SRVArena.Release(image.SRV); image = {}; }
    VisibilityGeometry.Reset(); UploadedVisibility.reset(); VisibilityRoot.Reset();
    VisibilityVertices = {}; VisibilityPixels = {}; VisibilityPipelines.clear();
}
bool SupportsVisibilityBuffer() const
{
    if (!Device) return false;
    for (auto format : {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT})
    {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support {format};
        const UINT required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
        if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
            (support.Support1 & required) != required) return false;
    }
    return true;
}
bool PrepareVisibilityBuffer(const std::shared_ptr<const VisibilityScene>& scene)
{
    if (!scene || !SupportsVisibilityBuffer()) return Fail("DX12 visibility buffer requires sampled integer/float attachments");
    if (scene->HasNativeGeometry())
    {
        if (!PrepareNaniteVisibilityGeometry(*scene)) return false;
    }
    else if (UploadedVisibility != scene)
    {
        if (!ValidateVisibilityScene(*scene, LastError)) return false;
        std::vector<VisibilityTriangle> bounds;
        if (!BuildMeshShaderBounds(*scene,bounds,LastError)) return false;
        const auto geometryBytes=scene->Triangles.size()*sizeof(VisibilityTriangle);
        const auto bytes=std::max(scene->Triangles.size()+bounds.size(),std::size_t{1})*sizeof(VisibilityTriangle);
        ComPtr<ID3D12Resource> candidate;
        if (!CreateBufferResource(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, candidate)) return false;
        void* mapped = nullptr; const D3D12_RANGE noRead{0, 0};
        const auto result = candidate->Map(0, &noRead, &mapped);
        if (FAILED(result)) return FailHRESULT("Map(visibility geometry)", result);
        if (geometryBytes) std::memcpy(mapped,scene->Triangles.data(),geometryBytes); else std::memset(mapped,0,bytes);
        if (!bounds.empty()) std::memcpy(static_cast<std::byte*>(mapped)+geometryBytes,bounds.data(),bounds.size()*sizeof(VisibilityTriangle));
        const D3D12_RANGE written{0, bytes}; candidate->Unmap(0, &written);
        VisibilityGeometry = std::move(candidate); UploadedVisibility = scene;
    }
    const float clear[4]{};
    if (!CreateHdrSurface(VisibilityImages[0], CreateInfo.Width, CreateInfo.Height, DXGI_FORMAT_R32_UINT, 1, clear) ||
        !CreateHdrSurface(VisibilityImages[1], CreateInfo.Width, CreateInfo.Height, DXGI_FORMAT_R32G32_FLOAT, 1, clear)) return false;
    if (!VisibilityRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[5]{}; D3D12_ROOT_PARAMETER parameters[VisibilityBufferSamplerRoot + 1]{};
        parameters[VisibilityConstantRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[VisibilityGeometryRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        for (UINT image = 0; image < 5; ++image)
        {
            ranges[image] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, image + 1, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[VisibilityImageRoot + image].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[VisibilityImageRoot + image].DescriptorTable = {1, &ranges[image]};
            parameters[VisibilityImageRoot + image].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
        DescribeMaterialSampler(parameters[VisibilityBufferSamplerRoot], materialSamplerRange);
        parameters[7].ParameterType=parameters[8].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[7].Descriptor.ShaderRegister=6; parameters[8].Descriptor.ShaderRegister=7;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = VisibilityBufferSamplerRoot + 1; desc.pParameters = parameters;

        ComPtr<ID3DBlob> serialized, errors;
        auto result = SerializeRootSignatureFunction(&desc, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(visibility)", result);
        result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(VisibilityRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(visibility)", result);
    }
    if (!VisibilityVertices[0] || !VisibilityVertices[1] || !VisibilityPixels[0] || !VisibilityPixels[1])
    {
        if (!LoadTechniqueShaderSource("DX12", "VisibilityBuffer.hlsl", ShaderCode, LastError)) return false;
        const char* vertex[] {"VSRaster", "VSResolve"}; const char* pixel[] {"PSRaster", "PSResolve"};
        for (std::size_t phase = 0; phase < 2; ++phase)
            if (!CompileShader(vertex[phase], "vs_5_0", VisibilityVertices[phase], nullptr, true) ||
                !CompileShader(pixel[phase], "ps_5_0", VisibilityPixels[phase], nullptr, true)) return false;
    }
    return true;
}
ID3D12PipelineState* VisibilityPipeline(const PipelineState& state, bool mirrored, bool resolve)
{
    const UINT key = (resolve ? 1u : 0u) | (state.Fill == FillMode::Wireframe ? 2u : 0u) |
        (state.BackFaceCulling ? 4u : 0u) | (mirrored ? 8u : 0u) | (state.DepthTest ? 16u : 0u) | (state.DepthWrite ? 32u : 0u);
    const auto found = VisibilityPipelines.find(key); if (found != VisibilityPipelines.end()) return found->second.Get();
    const auto phase = resolve ? 1u : 0u;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature = VisibilityRoot.Get();
    desc.VS = {VisibilityVertices[phase]->GetBufferPointer(), VisibilityVertices[phase]->GetBufferSize()};
    desc.PS = {VisibilityPixels[phase]->GetBufferPointer(), VisibilityPixels[phase]->GetBufferSize()};
    desc.NumRenderTargets = resolve ? GBufferAttachmentCount : 2; desc.BlendState.IndependentBlendEnable = TRUE;
    for (UINT index = 0; index < desc.NumRenderTargets; ++index)
    {
        auto& blend = desc.BlendState.RenderTarget[index]; blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE; blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
        blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD; blend.LogicOp = D3D12_LOGIC_OP_NOOP;
        desc.RTVFormats[index] = resolve ? DXGI_FORMAT_R16G16B16A16_FLOAT : index == 0 ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R32G32_FLOAT;
    }
    desc.RasterizerState.FillMode = !resolve && state.Fill == FillMode::Wireframe ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = !resolve && state.BackFaceCulling ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
    desc.RasterizerState.FrontCounterClockwise = mirrored ? FALSE : TRUE; desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = !resolve && state.DepthTest;
    desc.DepthStencilState.DepthWriteMask = !resolve && state.DepthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    desc.DepthStencilState.DepthFunc = !resolve && state.DepthWrite ? D3D12_COMPARISON_FUNC_LESS : D3D12_COMPARISON_FUNC_LESS_EQUAL;
    auto& stencil = desc.DepthStencilState.FrontFace;
    stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace = stencil;
    desc.DSVFormat = resolve ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_D32_FLOAT;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.SampleDesc.Count = 1; desc.SampleMask = UINT_MAX;
    ComPtr<ID3D12PipelineState> pipeline;
    const auto result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc, IID_PPV_ARGS(pipeline.GetAddressOf()));
    if (FAILED(result)) { FailHRESULT("CreateGraphicsPipelineState(visibility)", result); return nullptr; }
    auto* native = pipeline.Get(); VisibilityPipelines.emplace(key, std::move(pipeline)); return native;
}
bool DrawVisibilityBuffer(const VisibilityBufferDesc& frame)
{
    struct Binding { D3D12_GPU_VIRTUAL_ADDRESS Constant; std::array<UINT, 3> Images; };
    std::vector<Binding> bindings; bindings.reserve(frame.Draws.size());
    for (const auto& draw : frame.Draws)
    {
        Binding binding{}; binding.Images.fill(NullColorSRV);
        const TextureHandle handles[] {draw.Pipeline.DiffuseTexture, draw.Pipeline.NormalTexture, draw.Pipeline.OpacityTexture};
        for (std::size_t index = 0; index < 3; ++index) if (handles[index])
        {
            const auto found = Textures.find(handles[index].Value);
            if (found == Textures.end() || found->second.SRV == InvalidDescriptor || found->second.Description.Format != TextureFormat::RGBA8)
                return Fail("DX12 visibility material texture is invalid");
            Transition(found->second.Resource.Get(), found->second.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            binding.Images[index] = found->second.SRV;
        }
        VisibilityConstants constants;
        if (!BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, CreateInfo.Width, CreateInfo.Height, frame.Scene->TriangleCount(),
            frame.Scene->Draws[draw.Range], bool(handles[0]), bool(handles[1]), bool(handles[2]), constants)) return Fail("Invalid DX12 visibility material");
        constants.MaterialTiles=MaterialPlan.Resolve(unsigned(&draw-frame.Draws.data()));
        ConstantOffset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        if (ConstantOffset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - ConstantOffset)
            return Fail("DX12 visibility constants exceed the frame upload arena");
        std::memcpy(ConstantData + ConstantOffset, &constants, sizeof(constants));
        binding.Constant = ConstantBuffer->GetGPUVirtualAddress() + ConstantOffset; ConstantOffset += sizeof(constants); bindings.push_back(binding);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE targets[2]; const float clear[4]{};
    for (std::size_t image = 0; image < 2; ++image)
    {
        auto& surface = VisibilityImages[image]; Transition(surface.Resource.Get(), surface.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
        targets[image] = RTVArena.Cpu(surface.RTV); CommandList->ClearRenderTargetView(targets[image], clear, 0, nullptr);
    }
    const auto depth = DSVArena.Cpu(MainDepthDSV); CommandList->OMSetRenderTargets(2, targets, FALSE, &depth);
    CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); CommandList->SetGraphicsRootSignature(VisibilityRoot.Get());
    BindMaterialSampler(VisibilityBufferSamplerRoot);
    CommandList->SetGraphicsRootShaderResourceView(VisibilityGeometryRoot, VisibilityGeometry->GetGPUVirtualAddress());
    const auto bind = [&](std::size_t index)
    {
        CommandList->SetGraphicsRootConstantBufferView(VisibilityConstantRoot, bindings[index].Constant);
        for (UINT image = 0; image < 3; ++image)
            CommandList->SetGraphicsRootDescriptorTable(VisibilityImageRoot + 2 + image, SRVArena.Gpu(bindings[index].Images[image]));
    };
    if (frame.NaniteRaster) { if (!DrawNaniteFrameRaster(frame)) return false; }
    else
    for (std::size_t index = 0; index < frame.Draws.size(); ++index)
    {
        const auto& draw = frame.Draws[index]; const auto& range = frame.Scene->Draws[draw.Range];
        const bool task=frame.MeshShaders && UseMeshTaskStage(frame.MeshDispatch,range.TriangleCount,true);
        auto* pipeline=frame.MeshShaders?MeshPipeline(draw.Pipeline,range.Mirrored,false,0,task):VisibilityPipeline(draw.Pipeline,range.Mirrored,false);
        if (!pipeline) return false; bind(index); CommandList->SetPipelineState(pipeline);
        if (frame.MeshShaders)
        {
            const auto grid=task?BuildMeshTaskGrid(range.TriangleCount):BuildMeshShaderGrid(range.TriangleCount);
            MeshCommandList->DispatchMesh(grid.X,grid.Y,grid.Z);
            if (task) ++PendingFrame.Statistics.MeshShaderObjectDispatchCount;
        }
        else CommandList->DrawInstanced(range.TriangleCount*VisibilityTriangleCorners,1,0,0);
    }
    CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
    if (!ClassifyNaniteMaterials()) return false;
    for (auto& image : VisibilityImages) Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, GBufferAttachmentCount> materials;
    for (std::size_t index = 0; index < materials.size(); ++index) materials[index] = RTVArena.Cpu(GBufferSurfaces[index].RTV);
    CommandList->OMSetRenderTargets(static_cast<UINT>(materials.size()), materials.data(), FALSE, nullptr);
    CommandList->SetGraphicsRootSignature(VisibilityRoot.Get());
    BindMaterialSampler(VisibilityBufferSamplerRoot);
    CommandList->SetGraphicsRootShaderResourceView(VisibilityGeometryRoot, VisibilityGeometry->GetGPUVirtualAddress());
    for (UINT image = 0; image < 2; ++image) CommandList->SetGraphicsRootDescriptorTable(VisibilityImageRoot + image, SRVArena.Gpu(VisibilityImages[image].SRV));
    auto* resolve = VisibilityPipeline({}, false, true); if (!resolve) return false;
    CommandList->SetPipelineState(resolve);
    CommandList->SetGraphicsRootShaderResourceView(7,MaterialPlan.Active()?MaterialMap->Buffer->GetGPUVirtualAddress():VisibilityGeometry->GetGPUVirtualAddress());
    CommandList->SetGraphicsRootShaderResourceView(8,MaterialPlan.Active()?MaterialWork->Buffer->GetGPUVirtualAddress():VisibilityGeometry->GetGPUVirtualAddress());
    if (MaterialPlan.Active())
    {
        for (unsigned bin=0;bin<MaterialPlan.Representatives.size();++bin)
        {
            bind(MaterialPlan.Representatives[bin]);
            CommandList->ExecuteIndirect(MaterialDrawSignature.Get(),1,MaterialWork->Buffer.Get(),bin*NaniteMaterialBinWords*sizeof(unsigned),nullptr,0);
        }
    }
    else for (std::size_t index = 0; index < frame.Draws.size(); ++index) { bind(index); CommandList->DrawInstanced(3, 1, 0, 0); }
    return true;
}
