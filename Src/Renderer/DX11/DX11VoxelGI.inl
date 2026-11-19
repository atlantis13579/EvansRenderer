// The CPU publication is immutable. A complete RGBA32F texture replaces the
// previous volume before recording; each fullscreen stage owns its HDR output.
VoxelGIFrame FrameVoxelGI;
std::shared_ptr<const VoxelRadianceVolume> UploadedVoxelVolume;
ComPtr<ID3D11Texture2D> VoxelVolume;
ComPtr<ID3D11ShaderResourceView> VoxelVolumeView;
std::array<HdrSurface,2> VoxelGIImages;
ComPtr<ID3D11VertexShader> VoxelGIVertex;
ComPtr<ID3D11PixelShader> VoxelGIPixel;
ComPtr<ID3D11Buffer> VoxelGIUniforms;
ComPtr<ID3D11BlendState> VoxelGIBlend;
int VoxelGIWidth=0,VoxelGIHeight=0;
bool SupportsVoxelGI() const
{
    UINT support=0; constexpr UINT Required=D3D11_FORMAT_SUPPORT_TEXTURE2D|D3D11_FORMAT_SUPPORT_SHADER_LOAD;
    return Device && SUCCEEDED(Device->CheckFormatSupport(DXGI_FORMAT_R32G32B32A32_FLOAT,&support)) && (support&Required)==Required;
}
void DestroyVoxelGI()
{
    VoxelGIImages={}; VoxelVolume.Reset(); VoxelVolumeView.Reset(); UploadedVoxelVolume.reset(); FrameVoxelGI={};
    VoxelGIVertex.Reset(); VoxelGIPixel.Reset(); VoxelGIUniforms.Reset(); VoxelGIBlend.Reset(); VoxelGIWidth=VoxelGIHeight=0;
}
bool EnsureVoxelGIResources()
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants({},frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("DX11 voxel GI publication is invalid");
    if (frame.Volume!=UploadedVoxelVolume)
    {
        const auto& data=*frame.Volume; D3D11_TEXTURE2D_DESC texture{};
        texture.Width=data.Width; texture.Height=data.Height; texture.MipLevels=texture.ArraySize=1;
        texture.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; texture.SampleDesc.Count=1;
        texture.Usage=D3D11_USAGE_IMMUTABLE; texture.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data.Texels.data(); initial.SysMemPitch=data.Width*sizeof(Float4);
        ComPtr<ID3D11Texture2D> image; ComPtr<ID3D11ShaderResourceView> view;
        auto result=Device->CreateTexture2D(&texture,&initial,image.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(voxel radiance)",result));
        result=Device->CreateShaderResourceView(image.Get(),nullptr,view.GetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(voxel radiance)",result));
        VoxelVolume=std::move(image); VoxelVolumeView=std::move(view); UploadedVoxelVolume=frame.Volume;
    }
    if (VoxelGIWidth!=CreateInfo.Width || VoxelGIHeight!=CreateInfo.Height)
    { VoxelGIImages={}; VoxelGIWidth=CreateInfo.Width; VoxelGIHeight=CreateInfo.Height; }
    for (auto& image:VoxelGIImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (VoxelGIVertex && VoxelGIPixel && VoxelGIUniforms && VoxelGIBlend) return true;
    if (!LoadTechniqueShaderSource("DX11","VoxelGlobalIllumination.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,VoxelGIVertex.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(voxel GI)",result));
    result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,VoxelGIPixel.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(voxel GI)",result));
    D3D11_BUFFER_DESC uniform{}; uniform.ByteWidth=sizeof(VoxelGIConstants); uniform.Usage=D3D11_USAGE_DYNAMIC;
    uniform.BindFlags=D3D11_BIND_CONSTANT_BUFFER; uniform.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    result=Device->CreateBuffer(&uniform,nullptr,VoxelGIUniforms.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(voxel GI constants)",result));
    D3D11_BLEND_DESC blend{}; auto& color=blend.RenderTarget[0]; color.BlendEnable=TRUE;
    color.SrcBlend=color.DestBlend=D3D11_BLEND_ONE; color.BlendOp=color.BlendOpAlpha=D3D11_BLEND_OP_ADD;
    color.SrcBlendAlpha=D3D11_BLEND_ZERO; color.DestBlendAlpha=D3D11_BLEND_ONE;
    color.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED|D3D11_COLOR_WRITE_ENABLE_GREEN|D3D11_COLOR_WRITE_ENABLE_BLUE;
    result=Device->CreateBlendState(&blend,VoxelGIBlend.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateBlendState(voxel GI)",result));
}
bool ApplyVoxelGI(const GlobalIlluminationDesc& desc)
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants(desc.ViewProjection,frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("DX11 voxel GI camera is invalid");
    constexpr UINT InputCount=7; ID3D11ShaderResourceView* empty[InputCount]{};
    Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(VoxelGIVertex.Get(),nullptr,0); Context->PSSetShader(VoxelGIPixel.Get(),nullptr,0);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    ID3D11Buffer* uniform=VoxelGIUniforms.Get(); Context->PSSetConstantBuffers(0,1,&uniform);
    for (const auto pass:VoxelGIPasses)
    {
        const auto index=static_cast<UINT>(pass); const bool composite=pass==Techniques::VoxelGIPass::Composite;
        constants.Options.X=index; D3D11_MAPPED_SUBRESOURCE mapping{};
        const auto result=Context->Map(uniform,0,D3D11_MAP_WRITE_DISCARD,0,&mapping);
        if (FAILED(result)) return Fail(HResultMessage("Map(voxel GI constants)",result));
        std::memcpy(mapping.pData,&constants,sizeof(constants)); Context->Unmap(uniform,0);
        Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
        ID3D11RenderTargetView* destination=composite ? HdrSurfaces[HdrScene].Target.Get():VoxelGIImages[index%2].Target.Get();
        Context->OMSetRenderTargets(1,&destination,nullptr); Context->OMSetBlendState(composite ? VoxelGIBlend.Get():nullptr,nullptr,UINT_MAX);
        ID3D11ShaderResourceView* textures[InputCount]{};
        if (index) textures[5]=VoxelGIImages[(index-1)%2].View.Get();
        if (!composite)
        {
            textures[0]=DepthShaderView.Get(); textures[1]=GBufferSurfaces[1].View.Get(); textures[2]=GBufferSurfaces[0].View.Get();
            textures[3]=GBufferSurfaces[3].View.Get(); textures[4]=VoxelVolumeView.Get(); textures[6]=HdrSurfaces[HdrScene].View.Get();
        }
        Context->PSSetShaderResources(0,InputCount,textures); Context->Draw(3,0);
        PendingFrame.VoxelGIPasses.push_back(pass); ++PendingFrame.Statistics.GlobalIlluminationPassCount;
    }
    Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX); GI.Resolved=true; return true;
}
