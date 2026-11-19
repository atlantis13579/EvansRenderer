// DX12RHI::Impl members. Native UV2 replay uses the opaque depth buffer and
// a per-sample coverage latch, then resolves and adds indirect light to HDR.
static constexpr std::size_t LightmapFront=0,LightmapBack=1,LightmapCovered=2,LightmapResolved=3,LightmapComposite=4;
std::array<HdrSurface,5> LightmapSurfaces;
ComPtr<ID3D12Resource> LightmapVertices;
D3D12_VERTEX_BUFFER_VIEW LightmapVertexView{};
static constexpr UINT LightmapsSamplerRoot = 7;
ComPtr<ID3D12RootSignature> LightmapRootSignature;
std::array<ComPtr<ID3D12PipelineState>,8> LightmapSurfacePipelines;
ComPtr<ID3D12PipelineState> LightmapCompositePipeline;
std::shared_ptr<const LightmapBakeResult> UploadedLightmap,PendingUploadedLightmap;
LightmapFrame FrameLightmaps;
Techniques::Settings FrameLightmapSettings;
UINT LightmapPipelineSamples=0;
bool LightmapActive=false;

void ResetLightmaps(bool complete=false)
{
    for (auto& surface:LightmapSurfaces)
    { surface.Resource.Reset(); surface.State=surface.FrameStartState=D3D12_RESOURCE_STATE_RENDER_TARGET; }
    LightmapVertices.Reset(); LightmapVertexView={}; UploadedLightmap.reset(); PendingUploadedLightmap.reset();
    FrameLightmaps={};
    if (!complete) return;
    LightmapSurfaces={}; LightmapRootSignature.Reset(); LightmapSurfacePipelines={}; LightmapCompositePipeline.Reset();
    LightmapPipelineSamples=0; LightmapActive=false;
}

bool EnsureLightmapResources()
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) return true;
    const auto& source=*FrameLightmaps.Source;
    if (!source.Receivers || !source.Irradiance || !source.Irradiance->Atlas || source.Receivers->Vertices.size()<3)
        return Fail("DX12 lightmaps require complete receiver and irradiance data");
    const auto side=source.Irradiance->Atlas->Settings.Resolution;
    const float clear[4]{};
    if (!CreateHdrSurface(LightmapSurfaces[LightmapFront],side,side,DXGI_FORMAT_R32G32B32A32_FLOAT) ||
        !CreateHdrSurface(LightmapSurfaces[LightmapBack],side,side,DXGI_FORMAT_R32G32B32A32_FLOAT) ||
        !CreateHdrSurface(LightmapSurfaces[LightmapCovered],CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R16G16B16A16_FLOAT,FrameSampleCount,clear) ||
        !CreateHdrSurface(LightmapSurfaces[LightmapResolved],CreateInfo.Width,CreateInfo.Height) ||
        !CreateHdrSurface(LightmapSurfaces[LightmapComposite],CreateInfo.Width,CreateInfo.Height)) return false;
    if (!LightmapRootSignature)
    {
        constexpr UINT TextureCount=6;
        std::array<D3D12_DESCRIPTOR_RANGE,TextureCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER, LightmapsSamplerRoot + 1> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[0].Descriptor={0,0}; parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        for (UINT slot=0;slot<TextureCount;++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1;
            ranges[slot].BaseShaderRegister=slot; ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[slot+1].DescriptorTable={1,&ranges[slot]}; parameters[slot+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
        DescribeMaterialSampler(parameters[LightmapsSamplerRoot], materialSamplerRange);
        D3D12_ROOT_SIGNATURE_DESC signature{};
        signature.NumParameters=static_cast<UINT>(parameters.size()); signature.pParameters=parameters.data();

        signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> bytes,errors;
        auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,bytes.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result))
        {
            if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize());
            else FailHRESULT("SerializeRootSignature(lightmaps)",result);
            return false;
        }
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,bytes->GetBufferPointer(),bytes->GetBufferSize(),IID_PPV_ARGS(LightmapRootSignature.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(lightmaps)",result);
    }
    if (LightmapPipelineSamples!=FrameSampleCount) { LightmapSurfacePipelines={}; LightmapPipelineSamples=FrameSampleCount; }
    if (!LightmapSurfacePipelines[0] || !LightmapCompositePipeline)
    {
        if (!LoadTechniqueShaderSource("DX12","Lightmaps.hlsl",ShaderCode,LastError)) return false;
        ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader("VSMain","vs_5_0",vertex) || !CompileShader("PSMain","ps_5_0",pixel)) return false;
        const D3D12_INPUT_ELEMENT_DESC elements[] {
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(LightmapReceiverVertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(LightmapReceiverVertex,MaterialUV),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(LightmapReceiverVertex,AtlasUV),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(LightmapReceiverVertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
        for (UINT index=0;index<=LightmapSurfacePipelines.size();++index)
        {
            const bool composite=index==LightmapSurfacePipelines.size();
            auto& output=composite ? LightmapCompositePipeline : LightmapSurfacePipelines[index];
            if (output) continue;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
            pipeline.pRootSignature=LightmapRootSignature.Get();
            pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
            pipeline.InputLayout={elements,static_cast<UINT>(std::size(elements))};
            auto& blend=pipeline.BlendState.RenderTarget[0];
            const bool first=!composite && (index&4)!=0;
            blend.BlendEnable=first;
            blend.SrcBlend=first ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_ONE;
            blend.DestBlend=first ? D3D12_BLEND_ONE : D3D12_BLEND_ZERO;
            blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlendAlpha=D3D12_BLEND_ZERO;
            blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
            blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL; pipeline.SampleMask=UINT_MAX;
            pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
            pipeline.RasterizerState.CullMode=!composite && (index&1) ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
            pipeline.RasterizerState.FrontCounterClockwise=(index&2)==0;
            pipeline.RasterizerState.DepthClipEnable=TRUE; pipeline.RasterizerState.MultisampleEnable=!composite && FrameSampleCount>1;
            pipeline.DepthStencilState.DepthEnable=!composite; pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
            pipeline.DepthStencilState.DepthFunc=composite ? D3D12_COMPARISON_FUNC_ALWAYS : D3D12_COMPARISON_FUNC_EQUAL;
            pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK;
            pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
            pipeline.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS};
            pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
            pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT;
            pipeline.DSVFormat=composite ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_D32_FLOAT;
            pipeline.SampleDesc.Count=composite ? 1 : FrameSampleCount;
            const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(output.GetAddressOf()));
            if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(lightmaps)",result);
        }
    }
    return true;
}

bool UploadLightmapSnapshot()
{
    if (UploadedLightmap==FrameLightmaps.Source || PendingUploadedLightmap==FrameLightmaps.Source) return true;
    const auto& bake=*FrameLightmaps.Source->Irradiance;
    const auto side=bake.Atlas->Settings.Resolution;
    if (bake.FrontIrradiance.size()!=static_cast<std::size_t>(side)*side ||
        (!bake.BackIrradiance.empty() && bake.BackIrradiance.size()!=bake.FrontIrradiance.size())) return Fail("DX12 lightmap upload size is invalid");
    const auto& vertices=FrameLightmaps.Source->Receivers->Vertices;
    const auto vertexBytes=vertices.size()*sizeof(LightmapReceiverVertex);
    if (vertexBytes>UINT_MAX) return Fail("DX12 lightmap vertex buffer exceeds the native size limit");
    ComPtr<ID3D12Resource> vertexBuffer, vertexUpload;
    if (!CreateBufferResource(vertexBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, vertexBuffer) ||
        !CreateBufferResource(vertexBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, vertexUpload)) return false;
    void* mapping=nullptr; const D3D12_RANGE noRead{0,0};
    auto result=vertexUpload->Map(0,&noRead,&mapping);
    if (FAILED(result)) return FailHRESULT("Map(lightmap receivers)",result);
    std::memcpy(mapping,vertices.data(),vertexBytes); vertexUpload->Unmap(0,nullptr);
    CommandList->CopyBufferRegion(vertexBuffer.Get(), 0, vertexUpload.Get(), 0, vertexBytes);
    auto state = D3D12_RESOURCE_STATE_COPY_DEST;
    Transition(vertexBuffer.Get(), state, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    FrameUploads.push_back(std::move(vertexUpload));
    for (auto index:{LightmapFront,LightmapBack})
    {
        auto& surface=LightmapSurfaces[index]; const auto texture=surface.Resource->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes=0;
        Device->GetCopyableFootprints(&texture,0,1,0,&footprint,nullptr,nullptr,&bytes);
        ComPtr<ID3D12Resource> upload;
        if (!CreateBufferResource(static_cast<std::size_t>(bytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,upload)) return false;
        result=upload->Map(0,&noRead,&mapping);
        if (FAILED(result)) return FailHRESULT("Map(lightmap irradiance)",result);
        const auto& texels=index==LightmapFront || bake.BackIrradiance.empty() ? bake.FrontIrradiance : bake.BackIrradiance;
        const auto rowBytes=static_cast<std::size_t>(side)*sizeof(Float4);
        for (UINT row=0;row<side;++row) std::memcpy(static_cast<std::uint8_t*>(mapping)+footprint.Offset+
            static_cast<std::size_t>(row)*footprint.Footprint.RowPitch,texels.data()+static_cast<std::size_t>(row)*side,rowBytes);
        upload->Unmap(0,nullptr); Transition(surface.Resource.Get(),surface.State,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{},to{};
        from.pResource=upload.Get(); from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint=footprint;
        to.pResource=surface.Resource.Get(); to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        CommandList->CopyTextureRegion(&to,0,0,0,&from,nullptr); FrameUploads.push_back(std::move(upload));
    }
    LightmapVertices = std::move(vertexBuffer);
    LightmapVertexView={LightmapVertices->GetGPUVirtualAddress(),static_cast<UINT>(vertexBytes),sizeof(LightmapReceiverVertex)};
    // Publish only after successful submission. An aborted command list must
    // re-upload both hemispheres on the next frame, even if CPU data is reused.
    PendingUploadedLightmap=FrameLightmaps.Source; UploadedLightmap.reset();
    return true;
}

bool ResolveLightmaps(const GlobalIlluminationDesc& desc)
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) { GI.Resolved=true; return true; }
    if (!UploadLightmapSnapshot()) return false;
    constexpr auto Sampled=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (auto index:{LightmapFront,LightmapBack}) Transition(LightmapSurfaces[index].Resource.Get(),LightmapSurfaces[index].State,Sampled);
    auto& covered=LightmapSurfaces[LightmapCovered]; auto& resolved=LightmapSurfaces[LightmapResolved];
    auto& composite=LightmapSurfaces[LightmapComposite]; auto& scene=HdrSurfaces[HdrScene];
    Transition(covered.Resource.Get(),covered.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const bool multisample=FrameSampleCount>1;
    auto* depth=multisample ? MultisampleDepth.Resource.Get() : MainDepth.Get();
    auto& depthState=multisample ? MultisampleDepth.State : MainDepthState;
    Transition(depth,depthState,D3D12_RESOURCE_STATE_DEPTH_READ);
    const auto target=RTVArena.Cpu(covered.RTV),depthTarget=DSVArena.Cpu(multisample ? MultisampleDepthReadDSV : MainDepthReadOnlyDSV);
    const float clear[4]{}; CommandList->ClearRenderTargetView(target,clear,0,nullptr);
    CommandList->OMSetRenderTargets(1,&target,FALSE,&depthTarget);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
    CommandList->SetGraphicsRootSignature(LightmapRootSignature.Get());
    BindMaterialSampler(LightmapsSamplerRoot);
    CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->IASetVertexBuffers(0,1,&LightmapVertexView);
    auto constants=[&](const LightmapConstants& value) {
        const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        if (offset>ConstantBufferCapacity || sizeof(value)>ConstantBufferCapacity-offset) return Fail("DX12 lightmaps exhausted the frame constant upload");
        std::memcpy(ConstantData+offset,&value,sizeof(value)); ConstantOffset=offset+sizeof(value);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset); return true;
    };
    auto bind=[&](const std::array<UINT,6>& inputs) {
        for (UINT slot=0;slot<inputs.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot]));
    };
    auto texture=[&](TextureHandle handle,UINT& descriptor) {
        descriptor=NullColorSRV; if (!handle) return true;
        const auto found=Textures.find(handle.Value);
        if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8 || !found->second.Resource)
            return Fail("DX12 lightmap receiver references an invalid material texture");
        Transition(found->second.Resource.Get(),found->second.State,Sampled); descriptor=found->second.SRV; return true;
    };
    for (const auto& draw:FrameLightmaps.Draws)
    {
        if (!draw.VertexCount) continue;
        LightmapConstants value;
        if (!BuildLightmapConstants(FrameLightmaps,draw,desc.ViewProjection,FrameGI.Intensity,FrameLightmapSettings.Shading,FrameLightmapSettings.Metallic,value))
            return Fail("DX12 lightmap receiver uniforms are invalid");
        UINT diffuse,opacity;
        if (!texture(draw.DiffuseTexture,diffuse) || !texture(draw.OpacityTexture,opacity) || !constants(value)) return false;
        bind({LightmapSurfaces[LightmapFront].SRV,LightmapSurfaces[LightmapBack].SRV,diffuse,opacity,NullColorSRV,NullColorSRV});
        const UINT pipeline=(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)|(FrameLightmaps.KeepFirstSurface ? 4u : 0u);
        CommandList->SetPipelineState(LightmapSurfacePipelines[pipeline].Get()); CommandList->DrawInstanced(draw.VertexCount,1,draw.FirstVertex,0);
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (multisample)
    {
        Transition(covered.Resource.Get(),covered.State,D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        Transition(resolved.Resource.Get(),resolved.State,D3D12_RESOURCE_STATE_RESOLVE_DEST);
        CommandList->ResolveSubresource(resolved.Resource.Get(),0,covered.Resource.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    auto& contribution=multisample ? resolved : covered;
    Transition(contribution.Resource.Get(),contribution.State,Sampled); Transition(scene.Resource.Get(),scene.State,Sampled);
    Transition(composite.Resource.Get(),composite.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto compositeTarget=RTVArena.Cpu(composite.RTV); CommandList->OMSetRenderTargets(1,&compositeTarget,FALSE,nullptr);
    LightmapConstants value{}; value.Options.W=static_cast<unsigned>(LightmapPass::Composite);
    value.Options.X=AmbientOcclusionApplied ? 1u : 0u;
    if (AmbientOcclusionApplied) Transition(AmbientOcclusionSurfaces[0].Resource.Get(),AmbientOcclusionSurfaces[0].State,Sampled);
    if (!constants(value)) return false;
    bind({AmbientOcclusionApplied ? AmbientOcclusionSurfaces[0].SRV : NullColorSRV,
        LightmapSurfaces[LightmapBack].SRV,NullColorSRV,NullColorSRV,scene.SRV,contribution.SRV});
    CommandList->SetPipelineState(LightmapCompositePipeline.Get()); CommandList->DrawInstanced(3,1,0,0);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(composite.Resource.Get(),composite.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(scene.Resource.Get(),composite.Resource.Get());
    GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=2;
    RecordLightmapDraws(FrameLightmaps,PendingFrame.Statistics); return true;
}
