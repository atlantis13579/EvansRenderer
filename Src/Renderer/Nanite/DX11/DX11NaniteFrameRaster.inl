// Persistent scratch is used once per visibility frame. The ordinary frame
// lifecycle completes GPU work before resize/reuse; no raster result is mapped.
NaniteFrameRasterWork NaniteRasterPlan;
std::shared_ptr<NaniteFrameBuffer> NaniteRasterDecisions;
NaniteDepthHandle NaniteRasterMainDepth;
NaniteVisibilityHandle NaniteRasterPost;
bool PrepareNaniteRasterVisibility(const VisibilityBufferDesc& frame)
{
    // Previous frame resources are retired only after the frame fence/abort.
    NaniteDepths.erase(NaniteRasterMainDepth.Value); NaniteDecisions.erase(NaniteRasterPost.Value);
    NaniteRasterMainDepth={}; NaniteRasterPost={}; NaniteRasterDecisions.reset();
    if (!NaniteFrameDummy) { const unsigned zero=0; if (!CreateNaniteFrameBuffer(1,&zero,false,NaniteFrameDummy)) return false; }
    NaniteRasterDecisions=NaniteFrameDummy;
    if (!frame.NaniteVisibility) return true;
    const auto found=NaniteDecisions.find(frame.NaniteVisibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Phase!=NaniteVisibilityPhase::Main ||
        found->second->Info.Frame!=FrameNanite.CurrentFrame() || found->second->Info.QueryCount!=frame.NaniteClusters.size() || found->second->Draws)
        return Fail("Nanite raster requires matching unconsumed main-phase cluster decisions");
    NaniteRasterDecisions=found->second->Data; return true;
}
bool RetestNaniteRaster(const VisibilityBufferDesc& frame)
{
    if (!CaptureNaniteDepth(NaniteRasterMainDepth,true) || !RetestNaniteVisibility(frame.NaniteVisibility,NaniteRasterMainDepth,NaniteRasterPost,true)) return false;
    NaniteRasterDecisions=NaniteDecisions.at(NaniteRasterPost.Value)->Data;
    for (auto& job:NaniteRasterPlan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    return true;
}
std::shared_ptr<NaniteFrameBuffer> NaniteRasterInput,NaniteRasterOutput;
std::array<ComPtr<ID3D11ComputeShader>,6> NaniteRasterFrameShaders;
std::array<ComPtr<ID3D11VertexShader>,2> NaniteRasterVertices;
std::array<ComPtr<ID3D11PixelShader>,2> NaniteRasterPixels;
ComPtr<ID3D11Buffer> NaniteRasterFrameUniform,NaniteRasterDrawUniform;
ComPtr<ID3D11DepthStencilState> NaniteRasterExportDepth,NaniteRasterPostDepth;
void DestroyNaniteFrameRaster()
{
    DestroyNaniteHardware();
    NaniteRasterDecisions.reset(); NaniteRasterMainDepth={}; NaniteRasterPost={};
    NaniteRasterInput.reset(); NaniteRasterOutput.reset(); NaniteRasterPlan={};
    NaniteRasterFrameShaders={}; NaniteRasterVertices={}; NaniteRasterPixels={};
    NaniteRasterFrameUniform.Reset(); NaniteRasterDrawUniform.Reset(); NaniteRasterExportDepth.Reset(); NaniteRasterPostDepth.Reset();
}
bool PrepareNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (!PrepareNaniteRasterVisibility(frame)) return false;
    NaniteFrameRasterWork plan; if (!DepthShaderView || !BuildNaniteFrameRasterWork(frame,CreateInfo.Width,CreateInfo.Height,plan,LastError)) return false;
    if (!LoadTechniqueShaderSource("DX11","NaniteRaster.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSSeedDepth","CSRaster","CSSeedOwner","CSClear","CSProject","CSCapture"};
    for (unsigned pass=0;pass<NaniteRasterFrameShaders.size();++pass) if (!NaniteRasterFrameShaders[pass])
    {
        const auto value=std::to_string(pass); const D3D_SHADER_MACRO defines[]{{"EVANS_NANITE_FRAME_PASS",value.c_str()},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,pass>=3?defines:nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,NaniteRasterFrameShaders[pass].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame raster kernel",status));
    }
    constexpr const char* vertices[]{"VSHardware","VSExport"}; constexpr const char* pixels[]{"PSHardware","PSExport"};
    const D3D_SHADER_MACRO graphics[]{{"EVANS_NANITE_GRAPHICS","1"},{nullptr,nullptr}};
    for (unsigned stage=0;stage<2;++stage) if (!NaniteRasterVertices[stage] || !NaniteRasterPixels[stage])
    {
        ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader(vertices[stage],"vs_5_0",vertex,graphics,true) || !CompileShader(pixels[stage],"ps_5_0",pixel,graphics,true)) return false;
        auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,NaniteRasterVertices[stage].ReleaseAndGetAddressOf());
        if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,NaniteRasterPixels[stage].ReleaseAndGetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame raster graphics",status));
    }
    for (unsigned slot=0;slot<2;++slot)
    {
        auto& buffer=slot?NaniteRasterDrawUniform:NaniteRasterFrameUniform; if (buffer) continue;
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=slot?sizeof(NaniteRasterDrawConstants):sizeof(NaniteFrameRasterConstants);
        desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,buffer.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame constants",status));
    }
    if (!NaniteRasterExportDepth)
    {
        D3D11_DEPTH_STENCIL_DESC desc{}; desc.DepthEnable=TRUE; desc.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; desc.DepthFunc=D3D11_COMPARISON_ALWAYS;
        const auto status=Device->CreateDepthStencilState(&desc,NaniteRasterExportDepth.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite depth export state",status));
    }
    if (!NaniteRasterPostDepth)
    {
        D3D11_DEPTH_STENCIL_DESC desc{}; desc.DepthEnable=TRUE; desc.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; desc.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        const auto status=Device->CreateDepthStencilState(&desc,NaniteRasterPostDepth.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite post depth state",status));
    }
    std::shared_ptr<NaniteFrameBuffer> input,output;
    if (!CreateNaniteFrameBuffer(plan.InputWords,nullptr,true,input) || !CreateNaniteFrameBuffer(plan.OutputWords,nullptr,true,output)) return false;
    NaniteRasterInput=std::move(input); NaniteRasterOutput=std::move(output); NaniteRasterPlan=std::move(plan); return true;
}
void DispatchNaniteRasterFrame(const NaniteFrameRasterDispatch& job)
{
    ID3D11ShaderResourceView* empty[5]{};
    Context->VSSetShaderResources(0,5,empty); Context->PSSetShaderResources(0,5,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* uniform=NaniteRasterFrameUniform.Get(); Context->UpdateSubresource(uniform,0,nullptr,&job.Constants,0,0);
    ID3D11ShaderResourceView* inputs[]{job.Pass<3?NaniteRasterInput->View.Get():VisibilityGeometryView.Get(),
        job.Pass==5?DepthShaderView.Get():nullptr,job.Pass==5?VisibilityImages[0].View.Get():nullptr,
        job.Pass==4 && NaniteRasterDecisions?NaniteRasterDecisions->View.Get():nullptr};
    auto* target=job.Pass<3?NaniteRasterOutput->Target.Get():NaniteRasterInput->Target.Get();
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,4,inputs); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    Context->CSSetShader(NaniteRasterFrameShaders[job.Pass].Get(),nullptr,0); Context->Dispatch(job.GroupsX,job.GroupsY,1);
    target=nullptr; Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShaderResources(0,4,empty); Context->CSSetShader(nullptr,nullptr,0);
    ++PendingFrame.Statistics.NaniteRasterDispatchCount;
}
bool DrawNaniteRasterPhase(const VisibilityBufferDesc& frame)
{
    if (frame.NaniteVisibility) ++PendingFrame.Statistics.NaniteOcclusionPhases;
    const auto bind=[&](const VisibilityMaterialDraw* draw)
    {
        NaniteRasterDrawConstants constants;
        if (!BuildNaniteRasterDrawConstants(frame,draw,CreateInfo.Width,CreateInfo.Height,constants)) return Fail("Invalid Nanite raster draw constants");
        if (NaniteRasterPost) constants.Raster.Quality.W|=NaniteRasterPreserveMain;
        auto* uniform=NaniteRasterDrawUniform.Get(); Context->UpdateSubresource(uniform,0,nullptr,&constants,0,0);
        Context->VSSetConstantBuffers(0,1,&uniform); Context->PSSetConstantBuffers(0,1,&uniform);
        ID3D11ShaderResourceView* inputs[]{VisibilityGeometryView.Get(),NaniteRasterInput->View.Get(),NaniteRasterOutput->View.Get(),nullptr,nullptr};
        if (draw)
        {
            const TextureHandle textures[]{draw->Pipeline.DiffuseTexture,draw->Pipeline.OpacityTexture};
            for (unsigned i=0;i<2;++i) if (textures[i])
            {
                const auto found=Textures.find(textures[i].Value);
                if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid Nanite raster material image");
                inputs[i+3]=found->second.Description.ColorSpace==TextureColorSpace::SRGB?found->second.SRGBShaderResourceView.Get():found->second.ShaderResourceView.Get();
            }
        }
        Context->VSSetShaderResources(0,3,inputs); Context->PSSetShaderResources(0,5,inputs); return true;
    };
    for (unsigned job=0;job<NaniteRasterPlan.HardwareBoundary;++job) DispatchNaniteRasterFrame(NaniteRasterPlan.Dispatches[job]);
    ID3D11RenderTargetView* targets[]{VisibilityImages[0].Target.Get(),VisibilityImages[1].Target.Get()};
    Context->OMSetRenderTargets(2,targets,DepthTarget.Get()); Context->OMSetDepthStencilState(NaniteRasterPost?NaniteRasterPostDepth.Get():DepthWriteState.Get(),0);
    Context->VSSetShader(NaniteRasterVertices[0].Get(),nullptr,0); Context->PSSetShader(NaniteRasterPixels[0].Get(),nullptr,0);
    auto* sampler=SurfaceSampler.Get(); Context->PSSetSamplers(0,1,&sampler);
    if (frame.GpuHardware)
    { if (!DrawNaniteHardware(frame)) return false; }
    else
    {
        for (std::size_t order=0;order<frame.Draws.size();++order)
        {
            const auto& draw=frame.Draws[NaniteRasterPost?frame.Draws.size()-1-order:order];
            if (!bind(&draw)) return false; BoundPipeline=draw.Pipeline; BoundWorld=Matrix4{};
            BoundWorld.Values[0]=frame.Scene->Draws[draw.Range].Mirrored?-1.0f:1.0f;
            auto* raster=GetOrientedRasterizer(); if (!raster) return false; Context->RSSetState(raster);
            Context->Draw(frame.Scene->Draws[draw.Range].TriangleCount*VisibilityTriangleCorners,0);
        }
    }
    for (unsigned job=NaniteRasterPlan.HardwareBoundary;job<NaniteRasterPlan.Dispatches.size();++job) DispatchNaniteRasterFrame(NaniteRasterPlan.Dispatches[job]);
    Context->OMSetRenderTargets(2,targets,DepthTarget.Get()); Context->OMSetDepthStencilState(NaniteRasterExportDepth.Get(),0); Context->RSSetState(SolidRasterState.Get());
    Context->VSSetShader(NaniteRasterVertices[1].Get(),nullptr,0); Context->PSSetShader(NaniteRasterPixels[1].Get(),nullptr,0);
    if (!bind(nullptr)) return false; Context->Draw(3,0); ++PendingFrame.Statistics.NaniteRasterExportCount;
    ID3D11ShaderResourceView* empty[5]{}; Context->VSSetShaderResources(0,5,empty); Context->PSSetShaderResources(0,5,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    return true;
}

bool DrawNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (frame.Scene->HasNativeGeometry() && !(frame.GpuInstances?DrawNaniteInstances():DrawNaniteVisibilityGeometry())) return false;
    PendingFrame.Statistics.NaniteOcclusionQueries+=unsigned(frame.NaniteClusters.size());
    if (!DrawNaniteRasterPhase(frame)) return false;
    if (frame.NaniteVisibility)
    {
        if (!RetestNaniteRaster(frame)) return false;
        if (!DrawNaniteRasterPhase(frame)) return false;
    }
    return true;
}
