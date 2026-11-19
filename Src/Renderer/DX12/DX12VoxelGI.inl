// Texture uploads belong to the frame transaction. The CPU publication becomes
// uploaded only after submission/presentation; abort restores speculative states.
VoxelGIFrame FrameVoxelGI;
std::shared_ptr<const VoxelRadianceVolume> UploadedVoxelVolume,PendingVoxelVolume;
ComPtr<ID3D12Resource> VoxelVolume;
UINT VoxelVolumeSRV=InvalidDescriptor;
D3D12_RESOURCE_STATES VoxelVolumeState=D3D12_RESOURCE_STATE_COPY_DEST,VoxelVolumeFrameState=D3D12_RESOURCE_STATE_COPY_DEST;
std::array<HdrSurface,2> VoxelGIImages;
ComPtr<ID3D12RootSignature> VoxelGIRoot;
std::array<ComPtr<ID3D12PipelineState>,2> VoxelGIPipelines;
static constexpr UINT VoxelGIInputCount=7;
bool SupportsVoxelGI() const
{
    D3D12_FEATURE_DATA_FORMAT_SUPPORT format{DXGI_FORMAT_R32G32B32A32_FLOAT};
    constexpr UINT Required=D3D12_FORMAT_SUPPORT1_TEXTURE2D|D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
    return Device && SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&format,sizeof(format))) &&
        (format.Support1&Required)==Required;
}
void DestroyVoxelGI()
{
    for (auto& image:VoxelGIImages) { SRVArena.Release(image.SRV); RTVArena.Release(image.RTV); image={}; }
    SRVArena.Release(VoxelVolumeSRV); VoxelVolumeSRV=InvalidDescriptor; VoxelVolume.Reset();
    VoxelGIRoot.Reset(); VoxelGIPipelines={}; FrameVoxelGI={}; UploadedVoxelVolume.reset(); PendingVoxelVolume.reset();
}
bool EnsureVoxelGIResources()
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants({},frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("DX12 voxel GI publication is invalid");
    const auto& data=*frame.Volume;
    if (!VoxelVolume || VoxelVolume->GetDesc().Width!=data.Width || VoxelVolume->GetDesc().Height!=data.Height)
    {
        D3D12_RESOURCE_DESC texture{}; texture.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width=data.Width; texture.Height=data.Height; texture.DepthOrArraySize=texture.MipLevels=1;
        texture.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; texture.SampleDesc.Count=1;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> image;
        const auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&texture,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(image.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateCommittedResource(voxel radiance)",result);
        if (VoxelVolumeSRV==InvalidDescriptor) VoxelVolumeSRV=SRVArena.Allocate();
        if (VoxelVolumeSRV==InvalidDescriptor) return Fail("DX12 voxel GI descriptor arena exhausted");
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=texture.Format; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(image.Get(),&view,SRVArena.Cpu(VoxelVolumeSRV)); VoxelVolume=std::move(image);
        VoxelVolumeState=VoxelVolumeFrameState=D3D12_RESOURCE_STATE_COPY_DEST; UploadedVoxelVolume.reset();
    }
    for (auto& image:VoxelGIImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (!VoxelGIRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,VoxelGIInputCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER,VoxelGIInputCount+1> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor.ShaderRegister=0;
        parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        for (UINT slot=0;slot<ranges.size();++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1;
            ranges[slot].BaseShaderRegister=slot; ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            auto& parameter=parameters[slot+1]; parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable.NumDescriptorRanges=1; parameter.DescriptorTable.pDescriptorRanges=&ranges[slot];
            parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters=static_cast<UINT>(parameters.size());
        signature.pParameters=parameters.data(); signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> bytes,errors;
        auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,bytes.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(voxel GI)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,bytes->GetBufferPointer(),bytes->GetBufferSize(),IID_PPV_ARGS(VoxelGIRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(voxel GI)",result);
    }
    if (VoxelGIPipelines[0] && VoxelGIPipelines[1]) return true;
    if (!LoadTechniqueShaderSource("DX12","VoxelGlobalIllumination.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    for (std::size_t phase=0;phase<VoxelGIPipelines.size();++phase)
    {
        if (VoxelGIPipelines[phase]) continue;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=VoxelGIRoot.Get();
        pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
        auto& color=pipeline.BlendState.RenderTarget[0]; color.BlendEnable=phase!=0;
        color.SrcBlend=D3D12_BLEND_ONE; color.DestBlend=phase ? D3D12_BLEND_ONE:D3D12_BLEND_ZERO;
        color.SrcBlendAlpha=D3D12_BLEND_ZERO; color.DestBlendAlpha=D3D12_BLEND_ONE;
        color.BlendOp=color.BlendOpAlpha=D3D12_BLEND_OP_ADD; color.LogicOp=D3D12_LOGIC_OP_NOOP;
        color.RenderTargetWriteMask=phase ? D3D12_COLOR_WRITE_ENABLE_RED|D3D12_COLOR_WRITE_ENABLE_GREEN|D3D12_COLOR_WRITE_ENABLE_BLUE:D3D12_COLOR_WRITE_ENABLE_ALL;
        pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
        pipeline.RasterizerState.DepthClipEnable=TRUE; pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
        pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        auto& stencil=pipeline.DepthStencilState.FrontFace;
        stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
        stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; pipeline.DepthStencilState.BackFace=stencil;
        pipeline.SampleMask=UINT_MAX; pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; pipeline.SampleDesc.Count=1;
        const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(VoxelGIPipelines[phase].GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(voxel GI)",result);
    }
    return true;
}
bool UploadVoxelRadiance()
{
    if (UploadedVoxelVolume==FrameVoxelGI.Volume) return true;
    const auto& volume=*FrameVoxelGI.Volume; const auto texture=VoxelVolume->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes=0;
    Device->GetCopyableFootprints(&texture,0,1,0,&footprint,nullptr,nullptr,&bytes);
    ComPtr<ID3D12Resource> upload;
    if (!CreateBufferResource(static_cast<std::size_t>(bytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,upload)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=upload->Map(0,&noRead,&mapped);
    if (FAILED(result)) return FailHRESULT("Map(voxel radiance upload)",result);
    const auto rowBytes=std::size_t(volume.Width)*sizeof(Float4);
    for (std::uint32_t row=0;row<volume.Height;++row)
        std::memcpy(static_cast<std::uint8_t*>(mapped)+footprint.Offset+std::size_t(row)*footprint.Footprint.RowPitch,
            volume.Texels.data()+std::size_t(row)*volume.Width,rowBytes);
    upload->Unmap(0,nullptr); Transition(VoxelVolume.Get(),VoxelVolumeState,D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource=upload.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint=footprint;
    D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource=VoxelVolume.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    CommandList->CopyTextureRegion(&destination,0,0,0,&source,nullptr); FrameUploads.push_back(std::move(upload));
    PendingVoxelVolume=FrameVoxelGI.Volume; return true;
}
bool ApplyVoxelGI(const GlobalIlluminationDesc& desc)
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants(desc.ViewProjection,frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("DX12 voxel GI camera is invalid");
    const auto first=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),stride=AlignUp(sizeof(constants),D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const auto required=(VoxelGIPasses.size()-1)*stride+sizeof(constants);
    if (first>ConstantBufferCapacity || required>ConstantBufferCapacity-first) return Fail("DX12 voxel GI constant arena exhausted");
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (!UploadVoxelRadiance()) return false;
    Transition(VoxelVolume.Get(),VoxelVolumeState,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    for (auto& image:GBufferSurfaces) Transition(image.Resource.Get(),image.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    CommandList->SetGraphicsRootSignature(VoxelGIRoot.Get()); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT rectangle{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&rectangle);
    for (const auto pass:VoxelGIPasses)
    {
        const auto index=static_cast<UINT>(pass); const bool composite=pass==Techniques::VoxelGIPass::Composite;
        auto& destination=composite ? HdrSurfaces[HdrScene]:VoxelGIImages[index%2];
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
        if (index) Transition(VoxelGIImages[(index-1)%2].Resource.Get(),VoxelGIImages[(index-1)%2].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (!composite) Transition(HdrSurfaces[HdrScene].Resource.Get(),HdrSurfaces[HdrScene].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
        const std::array<UINT,VoxelGIInputCount> textures{MainDepthSRV,GBufferSurfaces[1].SRV,GBufferSurfaces[0].SRV,GBufferSurfaces[3].SRV,
            VoxelVolumeSRV,index ? VoxelGIImages[(index-1)%2].SRV:NullColorSRV,composite ? NullColorSRV:HdrSurfaces[HdrScene].SRV};
        for (UINT slot=0;slot<textures.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(textures[slot]));
        constants.Options.X=index; const auto offset=first+index*stride; std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
        const auto target=RTVArena.Cpu(destination.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
        CommandList->SetPipelineState(VoxelGIPipelines[composite ? 1:0].Get()); CommandList->DrawInstanced(3,1,0,0);
        PendingFrame.VoxelGIPasses.push_back(pass); ++PendingFrame.Statistics.GlobalIlluminationPassCount;
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); GI.Resolved=true; return true;
}
