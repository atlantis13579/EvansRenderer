// Current-camera receiver depth is produced and packed entirely on the native
// queue. The CPU sees only the binary page directory used for residency.
bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    VirtualShadowRequestConstants request;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,request,LastError) || !EnsureVirtualShadowAtlas()) return false;
    const auto width=upload.Frame.Width,height=upload.Frame.Height;
    if (width>D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || height>D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) return Fail("Receiver depth exceeds D3D11 image dimensions");
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
    VirtualShadowAtlasRecord target;
    D3D11_TEXTURE2D_DESC depth{}; depth.Width=width; depth.Height=height; depth.MipLevels=depth.ArraySize=depth.SampleDesc.Count=1;
    depth.Format=DXGI_FORMAT_R32_TYPELESS; depth.Usage=D3D11_USAGE_DEFAULT; depth.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
    auto status=Device->CreateTexture2D(&depth,nullptr,target.Depth.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create current receiver depth",status));
    D3D11_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
    status=Device->CreateDepthStencilView(target.Depth.Get(),&dsv,target.Target.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Target current receiver depth",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels=1;
    status=Device->CreateShaderResourceView(target.Depth.Get(),&srv,target.View.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Sample current receiver depth",status));
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC cb{}; cb.ByteWidth=sizeof(VirtualShadowRasterConstants); cb.Usage=D3D11_USAGE_DEFAULT; cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    status=Device->CreateBuffer(&cb,nullptr,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create receiver frame",status));
    struct Cleanup
    {
        ID3D11DeviceContext* Context;
        ~Cleanup() { ID3D11ShaderResourceView* empty[3]{}; Context->PSSetShaderResources(0,3,empty); Context->OMSetRenderTargets(0,nullptr,nullptr); }
    } cleanup{Context.Get()};
    Context->OMSetRenderTargets(0,nullptr,target.Target.Get()); Context->ClearDepthStencilView(target.Target.Get(),D3D11_CLEAR_DEPTH,1,0);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX); Context->OMSetDepthStencilState(VirtualAtlasDepthState.Get(),0);
    const D3D11_VIEWPORT viewport{0,0,float(width),float(height),0,1}; const D3D11_RECT scissor{0,0,LONG(width),LONG(height)};
    Context->RSSetViewports(1,&viewport); Context->RSSetScissorRects(1,&scissor); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetInputLayout(VirtualAtlasLayout.Get()); Context->VSSetShader(VirtualAtlasVertex.Get(),nullptr,0); Context->PSSetShader(VirtualAtlasPixel.Get(),nullptr,0);
    auto* constants=uniform.Get(); Context->VSSetConstantBuffers(0,1,&constants); Context->PSSetConstantBuffers(0,1,&constants);
    auto* sampler=SurfaceSampler.Get(); Context->PSSetSamplers(0,1,&sampler);
    for (std::size_t draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw]; Context->UpdateSubresource(uniform.Get(),0,nullptr,&draws[draw],0,0);
        const unsigned raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0; Context->RSSetState(VirtualAtlasRaster[raster].Get());
        const UINT stride=sizeof(Vertex),offset=0; auto* vertex=Buffers.at(caster.Vertices.Value).Buffer.Get();
        Context->IASetVertexBuffers(0,1,&vertex,&stride,&offset); Context->IASetIndexBuffer(Buffers.at(caster.Indices.Value).Buffer.Get(),DXGI_FORMAT_R32_UINT,0);
        ID3D11ShaderResourceView* images[]{caster.Diffuse?Textures.at(caster.Diffuse.Value).ShaderResourceView.Get():nullptr,caster.Opacity?Textures.at(caster.Opacity.Value).ShaderResourceView.Get():nullptr};
        Context->PSSetShaderResources(0,2,images); Context->DrawIndexed(caster.IndexCount,caster.FirstIndex,0);
    }
    ComPtr<ID3D11Buffer> packed;
    if (!PackVirtualShadowDepth(target.View.Get(),width,height,packed)) return false;
    return BuildVirtualShadowRequests({upload.Frame,{}},output,packed.Get());
}
