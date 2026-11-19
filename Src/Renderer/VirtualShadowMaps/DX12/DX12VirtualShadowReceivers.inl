// A closed-frame native prepass uses the same material coverage and generated
// geometry buffers as depth-atlas drawing. Only page requests cross to the CPU.
bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    VirtualShadowRequestConstants request;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,request,LastError) || !RecoverSubmittedWork("Virtual receiver depth") ||
        !WaitForGPU() || !EnsureVirtualShadowAtlas()) return false;
    VirtualShadowAtlasPending.reset();
    const auto width=upload.Frame.Width,height=upload.Frame.Height;
    if (width>D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || height>D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        upload.Receivers.size()>D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1/VirtualAtlasViewsPerDraw-1) return Fail("Receiver depth exceeds D3D12 image or descriptor capacity");
    std::vector<VirtualShadowRasterConstants> draws(upload.Receivers.size());
    for (std::size_t draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw];
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Receiver buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError) ||
            !BuildVirtualShadowReceiverConstants(upload.Frame,caster,draws[draw],LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Receiver coverage texture is invalid"); }
    }
    auto work=std::make_shared<VirtualShadowAtlasWork>(); work->Candidate=std::make_shared<VirtualShadowAtlasRecord>();
    auto& target=*work->Candidate; // This private depth target has no residency snapshot or public atlas handle.
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_RESOURCE_DESC depth{};
    depth.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; depth.Width=width; depth.Height=height; depth.DepthOrArraySize=depth.MipLevels=1;
    depth.Format=DXGI_FORMAT_R32_TYPELESS; depth.SampleDesc.Count=1; depth.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{}; clear.Format=DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth=1;
    auto status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&depth,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(target.Depth.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create current receiver depth",status);
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; heap.NumDescriptors=1;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(target.DepthView.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create receiver depth view",status);
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto view=target.DepthView->GetCPUDescriptorHandleForHeapStart(); Device->CreateDepthStencilView(target.Depth.Get(),&dsv,view);
    const auto readTable=static_cast<unsigned>(draws.size());
    if (!CreateVirtualAtlasWork(*work,readTable+1,std::size_t(width)*height,false)) return false;
    for (unsigned draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw]; if (!PrepareVirtualAtlasCasterResources(*work,caster)) return false;
        VirtualAtlasViews(*work,draw,caster.Diffuse?Textures.at(caster.Diffuse.Value).Resource.Get():nullptr,
            caster.Opacity?Textures.at(caster.Opacity.Value).Resource.Get():nullptr,nullptr);
    }
    VirtualAtlasViews(*work,readTable,nullptr,nullptr,target.Depth.Get());
    auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); commands->SetDescriptorHeaps(1,&views); commands->SetGraphicsRootSignature(VirtualAtlasRoot.Get());
    commands->OMSetRenderTargets(0,nullptr,FALSE,&view); commands->ClearDepthStencilView(view,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr);
    const D3D12_VIEWPORT viewport{0,0,float(width),float(height),0,1}; const D3D12_RECT scissor{0,0,LONG(width),LONG(height)};
    commands->RSSetViewports(1,&viewport); commands->RSSetScissorRects(1,&scissor); commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    for (unsigned draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw]; const unsigned raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0;
        commands->SetPipelineState(VirtualAtlasRaster[raster].Get()); BindVirtualAtlasTable(*work,draw,false);
        if (!BindVirtualAtlasConstants(*work,draws[draw],false)) return false;
        commands->IASetVertexBuffers(0,1,&Buffers.at(caster.Vertices.Value).VertexView); commands->IASetIndexBuffer(&Buffers.at(caster.Indices.Value).IndexView);
        commands->DrawIndexedInstanced(caster.IndexCount,1,caster.FirstIndex,0,0);
    }
    commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr); VirtualAtlasBarrier(*work,target.Depth.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE,VirtualAtlasSampleState);
    if (!EncodeVirtualDepthPack(*work,width,height,readTable)) return false;
    if (!CompleteVirtualAtlasWork(work,nullptr)) return false;
    // The completed buffer decays to COMMON before the next submission reads it.
    return BuildVirtualShadowRequests({upload.Frame,{}},output,work->Feedback.Get());
}
