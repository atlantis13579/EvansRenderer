NaniteHardwareWork HardwarePlan;
std::shared_ptr<NaniteFrameBuffer> HardwareRanges,HardwareArguments,HardwareQueue,HardwareOwners;
std::array<ComPtr<ID3D11ComputeShader>,4> HardwareKernels;
std::array<ComPtr<ID3D11VertexShader>,2> HardwareVertices;
std::array<ComPtr<ID3D11PixelShader>,3> HardwarePixels;
std::array<ComPtr<ID3D11DepthStencilState>,3> HardwareDepth;
ComPtr<ID3D11Buffer> HardwareComputeUniform,HardwareDrawUniform;
ComPtr<ID3D11BlendState> HardwareNoColor;
void DestroyNaniteHardware()
{
    HardwarePlan={}; HardwareRanges.reset(); HardwareArguments.reset(); HardwareQueue.reset(); HardwareOwners.reset();
    HardwareKernels={}; HardwareVertices={}; HardwarePixels={}; HardwareDepth={}; HardwareComputeUniform.Reset(); HardwareDrawUniform.Reset(); HardwareNoColor.Reset();
}
bool PrepareNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteHardwareWork(frame,CreateInfo.Width,CreateInfo.Height,HardwarePlan,LastError)) return false;
    if (!HardwarePlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Ranges.size()*sizeof(ShaderUInt4)/sizeof(unsigned)),HardwarePlan.Ranges.data(),false,HardwareRanges) ||
        !CreateNaniteFrameBuffer(HardwarePlan.ArgumentWords,nullptr,true,HardwareArguments,true) ||
        !CreateNaniteFrameBuffer(HardwarePlan.QueueWords,nullptr,true,HardwareQueue) ||
        !CreateNaniteFrameBuffer(HardwarePlan.Constants.Counts.W,nullptr,true,HardwareOwners)) return false;
    for (unsigned slot=0;slot<2;++slot)
    {
        auto& buffer=slot?HardwareDrawUniform:HardwareComputeUniform; if (buffer) continue;
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=slot?sizeof(NaniteHardwareDrawConstants):sizeof(NaniteHardwareConstants);
        desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,buffer.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create hardware schedule constants",status));
    }
    for (unsigned phase=0;phase<HardwareDepth.size();++phase) if (!HardwareDepth[phase])
    {
        D3D11_DEPTH_STENCIL_DESC desc{}; desc.DepthEnable=TRUE; desc.DepthWriteMask=phase?D3D11_DEPTH_WRITE_MASK_ZERO:D3D11_DEPTH_WRITE_MASK_ALL;
        desc.DepthFunc=phase==0?D3D11_COMPARISON_LESS:phase==1?D3D11_COMPARISON_EQUAL:D3D11_COMPARISON_ALWAYS;
        const auto status=Device->CreateDepthStencilState(&desc,HardwareDepth[phase].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create hardware ownership depth state",status));
    }
    if (!HardwareNoColor)
    {
        D3D11_BLEND_DESC desc{}; desc.IndependentBlendEnable=TRUE;
        for (auto& target:desc.RenderTarget)
        { target.SrcBlend=target.SrcBlendAlpha=D3D11_BLEND_ONE; target.DestBlend=target.DestBlendAlpha=D3D11_BLEND_ZERO; target.BlendOp=target.BlendOpAlpha=D3D11_BLEND_OP_ADD; }
        const auto status=Device->CreateBlendState(&desc,HardwareNoColor.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create hardware depth-only color state",status));
    }
    if (HardwareKernels.back() && HardwareVertices.back() && HardwarePixels.back()) return true;
    if (!LoadTechniqueShaderSource("DX11","NaniteHardware.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* kernels[]{"CSClear","CSCount","CSScan","CSScatter"};
    for (unsigned pass=0;pass<HardwareKernels.size();++pass) if (!HardwareKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(kernels[pass],"cs_5_0",code,nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,HardwareKernels[pass].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create hardware queue kernel",status));
    }
    const D3D_SHADER_MACRO graphics[]{{"EVANS_HARDWARE_GRAPHICS","1"},{nullptr,nullptr}};
    constexpr const char* vertices[]{"VSQueue","VSResolve"}; constexpr const char* pixels[]{"PSDepth","PSOwner","PSResolve"};
    for (unsigned stage=0;stage<HardwareVertices.size();++stage) if (!HardwareVertices[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(vertices[stage],"vs_5_0",code,graphics,true)) return false;
        const auto status=Device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,HardwareVertices[stage].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create hardware queue vertex stage",status));
    }
    for (unsigned stage=0;stage<HardwarePixels.size();++stage) if (!HardwarePixels[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(pixels[stage],"ps_5_0",code,graphics,true)) return false;
        const auto status=Device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,HardwarePixels[stage].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create hardware ownership fragment stage",status));
    }
    return true;
}
bool DrawNaniteHardware(const VisibilityBufferDesc&)
{
    if (!HardwarePlan.Active()) return true;
    ID3D11ShaderResourceView* empty[8]{}; ID3D11UnorderedAccessView* noWrites[3]{};
    Context->VSSetShaderResources(0,8,empty); Context->PSSetShaderResources(0,8,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* uniform=HardwareComputeUniform.Get(); Context->UpdateSubresource(uniform,0,nullptr,&HardwarePlan.Constants,0,0); Context->CSSetConstantBuffers(0,1,&uniform);
    ID3D11ShaderResourceView* reads[]{HardwareRanges->View.Get(),NaniteRasterOutput->View.Get()}; Context->CSSetShaderResources(0,2,reads);
    ID3D11UnorderedAccessView* writes[]{HardwareArguments->Target.Get(),HardwareQueue->Target.Get(),HardwareOwners->Target.Get()}; Context->CSSetUnorderedAccessViews(0,3,writes,nullptr);
    for (unsigned pass=0;pass<HardwareKernels.size();++pass)
    {
        const auto& c=HardwarePlan.Constants; const unsigned invocations=pass==0?std::max(c.Counts.Z,c.Counts.W):c.Counts.X;
        const unsigned groups=(invocations+NaniteHardwareThreads-1)/NaniteHardwareThreads;
        Context->CSSetShader(HardwareKernels[pass].Get(),nullptr,0);
        Context->Dispatch(pass==2?1:std::min(groups,NaniteHardwareColumns),pass==2?1:(groups+NaniteHardwareColumns-1)/NaniteHardwareColumns,1);
    }
    Context->CSSetUnorderedAccessViews(0,3,noWrites,nullptr); Context->CSSetShaderResources(0,2,empty); Context->CSSetShader(nullptr,nullptr,0);
    uniform=HardwareDrawUniform.Get(); Context->VSSetConstantBuffers(0,1,&uniform); Context->PSSetConstantBuffers(0,1,&uniform);
    ID3D11RenderTargetView* targets[]{VisibilityImages[0].Target.Get(),VisibilityImages[1].Target.Get()};
    auto* sampler=SurfaceSampler.Get(); Context->PSSetSamplers(0,1,&sampler);
    const auto bind=[&](const NaniteHardwareBin& bin,bool resolve)
    {
        Context->UpdateSubresource(uniform,0,nullptr,&bin.Constants,0,0);
        ID3D11ShaderResourceView* views[]{VisibilityGeometryView.Get(),NaniteRasterInput->View.Get(),HardwareQueue->View.Get(),nullptr,nullptr,resolve?HardwareOwners->View.Get():nullptr};
        const TextureHandle textures[]{bin.Diffuse,bin.Opacity};
        for (unsigned i=0;i<2;++i) if (textures[i])
        {
            const auto found=Textures.find(textures[i].Value);
            if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid hardware coverage texture");
            views[i+3]=found->second.Description.ColorSpace==TextureColorSpace::SRGB?found->second.SRGBShaderResourceView.Get():found->second.ShaderResourceView.Get();
        }
        Context->VSSetShaderResources(0,3,views); Context->PSSetShaderResources(0,6,views); return true;
    };
    for (unsigned phase=0;phase<2;++phase)
    {
        if (phase==1) { auto* owner=HardwareOwners->Target.Get(); Context->OMSetRenderTargetsAndUnorderedAccessViews(2,targets,DepthTarget.Get(),2,1,&owner,nullptr); }
        else Context->OMSetRenderTargets(2,targets,DepthTarget.Get());
        Context->OMSetBlendState(HardwareNoColor.Get(),nullptr,UINT_MAX); Context->OMSetDepthStencilState(HardwareDepth[phase].Get(),0);
        Context->VSSetShader(HardwareVertices[0].Get(),nullptr,0); Context->PSSetShader(HardwarePixels[phase].Get(),nullptr,0);
        for (unsigned bin=0;bin<HardwarePlan.Bins.size();++bin)
        {
            const auto& item=HardwarePlan.Bins[bin]; if (!bind(item,false)) return false;
            BoundPipeline={}; BoundPipeline.BackFaceCulling=item.Cull; BoundWorld={}; BoundWorld.Values[0]=item.Mirrored?-1.0f:1.0f;
            auto* raster=GetOrientedRasterizer(); if (!raster) return false; Context->RSSetState(raster);
            Context->DrawInstancedIndirect(HardwareArguments->Buffer.Get(),bin*NaniteHardwareBinWords*sizeof(unsigned));
        }
    }
    // Release the pixel UAV before reading ownership to export stable IDs.
    Context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,2,1,noWrites,nullptr);
    Context->OMSetRenderTargets(2,targets,DepthTarget.Get()); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    Context->OMSetDepthStencilState(HardwareDepth[2].Get(),0); Context->RSSetState(SolidRasterState.Get());
    Context->VSSetShader(HardwareVertices[1].Get(),nullptr,0); Context->PSSetShader(HardwarePixels[2].Get(),nullptr,0);
    if (!bind(HardwarePlan.Bins.front(),true)) return false; Context->Draw(3,0);
    Context->VSSetShaderResources(0,8,empty); Context->PSSetShaderResources(0,8,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto& stats=PendingFrame.Statistics; stats.NaniteHardwareBinCount+=unsigned(HardwarePlan.Bins.size()); stats.NaniteHardwareDispatchCount+=unsigned(HardwareKernels.size());
    stats.NaniteHardwareIndirectCount+=unsigned(HardwarePlan.Bins.size()*2); stats.IndirectDrawCallCount+=unsigned(HardwarePlan.Bins.size()*2); return true;
}
