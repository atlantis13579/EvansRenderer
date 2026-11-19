struct LumenCaptureWork : LumenCaptureJob
{
    std::shared_ptr<LumenSurfaceRecord> Record;
    ComPtr<ID3D11Buffer> Uniform; ComPtr<ID3D11Query> Completion;
};
LumenCaptureQueue<LumenCaptureWork> LumenCaptureTasks;
bool CollectLumenCapture()
{ return LumenCaptureTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenCapture(work,ready); }); }
bool CancelLumenCaptureUpdate(LumenCaptureTask task)
{ return (LumenCaptureTasks.Cancel(task) || Fail("Invalid Lumen capture task")) && CollectLumenCapture(); }
template<class Handle> bool RetireLumenCaptureInput(Handle handle)
{
    if (!handle || LumenCaptureTasks.Empty()) return true;
    // Source deletion/mutation is exceptional. Drain only submissions borrowing
    // this resource; ordinary frame advances never wait for their capture work.
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    if (!LumenCaptureTasks.Visit([&](auto& work)
    {
        if (!work.Uses(handle)) return true;
        if (FrameOpen) return Fail("Lumen capture inputs cannot change inside a render frame");
        work.Cancelled=true;
        while (work.InFlight)
        {
            bool ready=false; if (!ProbeLumenCapture(work,ready)) return false;
            if (ready) { work.InFlight=false; break; }
            if (std::chrono::steady_clock::now()>=deadline) return Fail("Retiring Lumen capture input timed out");
            std::this_thread::yield();
        }
        return true;
    })) return false;
    return CollectLumenCapture();
}
void DestroyLumenCaptureAsync() { LumenCaptureTasks.Clear(); }
bool ProbeLumenCapture(LumenCaptureWork& work,bool& ready)
{
    BOOL complete=FALSE; const auto status=Context->GetData(work.Completion.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (FAILED(status)) return Fail(HResultMessage("Poll Lumen capture completion",status)); ready=status==S_OK && complete; return true;
}
bool BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& output)
{
    if (!CollectLumenCapture() || !EnsureLumenSurface()) return false;
    auto job=std::make_shared<LumenCaptureWork>();
    if (!job->Prepare(upload,LastError) || !LumenCaptureTasks.CanSubmit(*job,LastError)) return false;
    const auto& layout=job->Layout;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    auto record=std::make_shared<LumenSurfaceRecord>(); record->Layout=layout; record->Identity=upload.Identity; record->Captures=upload.Cards;
    D3D11_TEXTURE2D_DESC image{}; image.Width=layout.Width; image.Height=layout.Height; image.MipLevels=image.ArraySize=image.SampleDesc.Count=1;
    image.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; image.Usage=D3D11_USAGE_DEFAULT; image.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        auto status=Device->CreateTexture2D(&image,nullptr,record->Images[plane].GetAddressOf());
        if (SUCCEEDED(status)) status=Device->CreateRenderTargetView(record->Images[plane].Get(),nullptr,record->Targets[plane].GetAddressOf());
        if (SUCCEEDED(status)) status=Device->CreateShaderResourceView(record->Images[plane].Get(),nullptr,record->Views[plane].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create immutable Lumen material image",status));
    }
    image.Format=DXGI_FORMAT_D32_FLOAT; image.BindFlags=D3D11_BIND_DEPTH_STENCIL;
    auto status=Device->CreateTexture2D(&image,nullptr,record->Depth.GetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateDepthStencilView(record->Depth.Get(),nullptr,record->DepthView.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture depth",status));
    job->Record=record; auto& uniform=job->Uniform; D3D11_BUFFER_DESC cb{}; cb.ByteWidth=sizeof(LumenSurfaceConstants); cb.Usage=D3D11_USAGE_DEFAULT; cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    status=Device->CreateBuffer(&cb,nullptr,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture frame",status));
    const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,job->Completion.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create capture slice event",status));
    const auto task=LumenCaptureTasks.Insert(job); if (!task) return Fail("Lumen capture task space exhausted"); output=task; return true;
}
bool SubmitLumenCaptureSlice(LumenCaptureWork& work,const LumenCaptureJob::SliceFrame& slice)
{
    const auto& upload=work.Upload; const auto& layout=work.Layout; const auto& record=work.Record; auto& uniform=work.Uniform;
    struct Unbind
    {
        ID3D11DeviceContext* Context;
        ~Unbind() { ID3D11ShaderResourceView* empty[LumenSurfacePlanes]{}; Context->PSSetShaderResources(0,LumenSurfacePlanes,empty); Context->OMSetRenderTargets(0,nullptr,nullptr); }
    } unbind{Context.Get()};
    const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    ID3D11RenderTargetView* targets[]{record->Targets[0].Get(),record->Targets[1].Get(),record->Targets[2].Get()};
    Context->OMSetRenderTargets(LumenSurfacePlanes,targets,record->DepthView.Get());
    if (!work.Initialized) for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) Context->ClearRenderTargetView(targets[plane],clear[plane]);
    if (!work.Initialized) Context->ClearDepthStencilView(record->DepthView.Get(),D3D11_CLEAR_DEPTH,1,0);
    Context->OMSetDepthStencilState(LumenDepthState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); Context->IASetInputLayout(LumenVertexLayout.Get());
    Context->VSSetShader(LumenCaptureVertex.Get(),nullptr,0); Context->PSSetShader(LumenCapturePixel.Get(),nullptr,0);
    auto* frame=uniform.Get(); auto* sampler=SurfaceSampler.Get(); Context->VSSetConstantBuffers(0,1,&frame); Context->PSSetConstantBuffers(0,1,&frame); Context->PSSetSamplers(0,1,&sampler);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const D3D11_VIEWPORT viewport{float(x),float(y),float(stride),float(stride),0,1}; const D3D11_RECT scissor{LONG(x),LONG(y),LONG(x+stride),LONG(y+stride)};
        Context->RSSetViewports(1,&viewport); Context->RSSetScissorRects(1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const auto& constants=work.Constants[work.FirstConstant[card]+index-capture.FirstDraw];
            Context->UpdateSubresource(uniform.Get(),0,nullptr,&constants,0,0);
            const unsigned raster=(draw.BackFaceCulling?2u:0u)+(RealtimeWorldIsMirrored(draw.World)?1u:0u); Context->RSSetState(LumenRasterStates[raster].Get());
            const UINT vertexStride=sizeof(Vertex),offset=0; auto* vertex=Buffers.at(draw.Vertices.Value).Buffer.Get();
            Context->IASetVertexBuffers(0,1,&vertex,&vertexStride,&offset); Context->IASetIndexBuffer(Buffers.at(draw.Indices.Value).Buffer.Get(),DXGI_FORMAT_R32_UINT,0);
            const auto view=[&](TextureHandle handle)->ID3D11ShaderResourceView*
            {
                if (!handle) return nullptr; const auto& texture=Textures.at(handle.Value);
                return texture.Description.ColorSpace==TextureColorSpace::SRGB?texture.SRGBShaderResourceView.Get():texture.ShaderResourceView.Get();
            };
            ID3D11ShaderResourceView* images[]{view(draw.Diffuse),view(draw.Normal),view(draw.Opacity)};
            Context->PSSetShaderResources(0,LumenSurfacePlanes,images); Context->DrawIndexed(draw.IndexCount,draw.FirstIndex,0);
        }
    }
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->End(work.Completion.Get()); work.InFlight=true; Context->Flush(); return true;
}
bool AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    if (!CollectLumenCapture()) return false;
    return LumenCaptureTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenCapture(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenCaptureSlice(work,frame); },
        [&](auto& work,LumenSurfaceResult& result)
        {
            result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space exhausted");
            result.Layout=work.Layout; result.Identity=work.Upload.Identity; result.DrawCalls=work.Statistics.DrawCalls;
            LumenSurfaces.emplace(result.Handle.Value,work.Record); return true;
        });
}
