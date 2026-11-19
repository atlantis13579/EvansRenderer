struct LumenSurfaceRecord
{
    LumenSurfaceIdentity Identity; LumenSurfaceLayout Layout;
    std::vector<LumenSurfaceCapture> Captures;
    std::array<ComPtr<ID3D11Texture2D>,LumenSurfacePlanes> Images;
    std::array<ComPtr<ID3D11RenderTargetView>,LumenSurfacePlanes> Targets;
    std::array<ComPtr<ID3D11ShaderResourceView>,LumenSurfacePlanes> Views;
    ComPtr<ID3D11Texture2D> Depth; ComPtr<ID3D11DepthStencilView> DepthView;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenSurfaceRecord>> LumenSurfaces;
ComPtr<ID3D11VertexShader> LumenCaptureVertex;
ComPtr<ID3D11PixelShader> LumenCapturePixel;
ComPtr<ID3D11ComputeShader> LumenMaterialRead;
ComPtr<ID3D11InputLayout> LumenVertexLayout;
ComPtr<ID3D11DepthStencilState> LumenDepthState;
std::array<ComPtr<ID3D11RasterizerState>,4> LumenRasterStates;
void DestroyLumenSurfaces()
{
    DestroyLumenCaptureAsync();
    LumenSurfaces.clear(); LumenCaptureVertex.Reset(); LumenCapturePixel.Reset(); LumenMaterialRead.Reset(); LumenVertexLayout.Reset(); LumenDepthState.Reset();
    for (auto& raster:LumenRasterStates) raster.Reset();
}
bool EnsureLumenSurface()
{
    if (LumenCaptureVertex && LumenCapturePixel && LumenMaterialRead && LumenVertexLayout && LumenDepthState &&
        LumenRasterStates[0] && LumenRasterStates[1] && LumenRasterStates[2] && LumenRasterStates[3]) return true;
    if (Device->GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0) return Fail("Lumen material capture requires D3D11 compute and floating MRT targets");
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel,read; const D3D_SHADER_MACRO options[]{{"EVANS_LUMEN_READ_STAGE","1"},{nullptr,nullptr}};
    if (!CompileShader("VSCapture","vs_5_0",vertex,nullptr,true) || !CompileShader("PSCapture","ps_5_0",pixel,nullptr,true) || !CompileShader("CSRead","cs_5_0",read,options,true)) return false;
    auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,LumenCaptureVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,LumenCapturePixel.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateComputeShader(read->GetBufferPointer(),read->GetBufferSize(),nullptr,LumenMaterialRead.ReleaseAndGetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen material capture shaders",status));
    const D3D11_INPUT_ELEMENT_DESC attributes[]{{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D11_INPUT_PER_VERTEX_DATA,0}};
    status=Device->CreateInputLayout(attributes,4,vertex->GetBufferPointer(),vertex->GetBufferSize(),LumenVertexLayout.ReleaseAndGetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen material vertex layout",status));
    D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    status=Device->CreateDepthStencilState(&depth,LumenDepthState.ReleaseAndGetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture depth state",status));
    for (unsigned mode=0;mode<LumenRasterStates.size();++mode)
    {
        D3D11_RASTERIZER_DESC raster{}; raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=mode>=2?D3D11_CULL_BACK:D3D11_CULL_NONE;
        raster.FrontCounterClockwise=(mode&1)?FALSE:TRUE; raster.DepthClipEnable=TRUE; raster.ScissorEnable=TRUE;
        status=Device->CreateRasterizerState(&raster,LumenRasterStates[mode].ReleaseAndGetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture raster state",status));
    }
    return true;
}
bool ReadLumenSurfaceRecord(const LumenSurfaceRecord& record,std::vector<LumenSurfaceTexel>& output)
{
    const auto count=std::size_t(record.Layout.Width)*record.Layout.Height,bytes=count*sizeof(LumenSurfaceTexel);
    ComPtr<ID3D11Buffer> data,staging,uniform; ComPtr<ID3D11UnorderedAccessView> target;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(Float4);
    auto status=Device->CreateBuffer(&desc,nullptr,data.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen material export",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=UINT(count*LumenSurfacePlanes);
    status=Device->CreateUnorderedAccessView(data.Get(),&uav,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen material export",status));
    ShaderUInt4 extent{record.Layout.Width,record.Layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    desc={}; desc.ByteWidth=sizeof(extent); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&extent;
    status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen export frame",status));
    Context->OMSetRenderTargets(0,nullptr,nullptr); ID3D11ShaderResourceView* images[]{record.Views[0].Get(),record.Views[1].Get(),record.Views[2].Get()};
    auto* buffer=uniform.Get(); auto* view=target.Get(); Context->CSSetShader(LumenMaterialRead.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&buffer);
    Context->CSSetShaderResources(0,LumenSurfacePlanes,images); Context->CSSetUnorderedAccessViews(0,1,&view,nullptr);
    const unsigned groups=unsigned((count+LumenSurfaceReadThreads-1)/LumenSurfaceReadThreads);
    Context->Dispatch(std::min(groups,LumenSurfaceReadColumns),(groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns,1);
    ID3D11ShaderResourceView* empty[LumenSurfacePlanes]{}; view=nullptr; buffer=nullptr;
    Context->CSSetShaderResources(0,LumenSurfacePlanes,empty); Context->CSSetUnorderedAccessViews(0,1,&view,nullptr); Context->CSSetConstantBuffers(0,1,&buffer); Context->CSSetShader(nullptr,nullptr,0);
    desc={}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen export staging",status));
    Context->CopyResource(staging.Get(),data.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; std::vector<LumenSurfaceTexel> result(count);
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read Lumen material export",status));
    std::memcpy(result.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenSurfaceTexels(record.Layout,result,LastError)) return false;
    output=std::move(result); return true;
}
bool CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    LumenSurfaceLayout layout; if (!ValidateLumenSurfaceUpload(upload,layout,LastError) || !EnsureLumenSurface()) return false;
    if (layout.Width>D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || layout.Height>D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) return Fail("Lumen material atlas exceeds D3D11 image dimensions");
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
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC cb{}; cb.ByteWidth=sizeof(LumenSurfaceConstants); cb.Usage=D3D11_USAGE_DEFAULT; cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    status=Device->CreateBuffer(&cb,nullptr,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture frame",status));
    struct Unbind
    {
        ID3D11DeviceContext* Context;
        ~Unbind() { ID3D11ShaderResourceView* empty[LumenSurfacePlanes]{}; Context->PSSetShaderResources(0,LumenSurfacePlanes,empty); Context->OMSetRenderTargets(0,nullptr,nullptr); }
    } unbind{Context.Get()};
    const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    ID3D11RenderTargetView* targets[]{record->Targets[0].Get(),record->Targets[1].Get(),record->Targets[2].Get()};
    Context->OMSetRenderTargets(LumenSurfacePlanes,targets,record->DepthView.Get());
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) Context->ClearRenderTargetView(targets[plane],clear[plane]);
    Context->ClearDepthStencilView(record->DepthView.Get(),D3D11_CLEAR_DEPTH,1,0);
    Context->OMSetDepthStencilState(LumenDepthState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); Context->IASetInputLayout(LumenVertexLayout.Get());
    Context->VSSetShader(LumenCaptureVertex.Get(),nullptr,0); Context->PSSetShader(LumenCapturePixel.Get(),nullptr,0);
    auto* frame=uniform.Get(); auto* sampler=SurfaceSampler.Get(); Context->VSSetConstantBuffers(0,1,&frame); Context->PSSetConstantBuffers(0,1,&frame); Context->PSSetSamplers(0,1,&sampler);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2; LumenSurfaceResult result; result.Layout=layout; result.Identity=upload.Identity;
    for (unsigned card=0;card<upload.Cards.size();++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const D3D11_VIEWPORT viewport{float(x),float(y),float(stride),float(stride),0,1}; const D3D11_RECT scissor{LONG(x),LONG(y),LONG(x+stride),LONG(y+stride)};
        Context->RSSetViewports(1,&viewport); Context->RSSetScissorRects(1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; LumenSurfaceConstants constants;
            if (!BuildLumenSurfaceConstants(layout,capture,draw,constants,LastError)) return false;
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
            Context->PSSetShaderResources(0,LumenSurfacePlanes,images); Context->DrawIndexed(draw.IndexCount,draw.FirstIndex,0); ++result.DrawCalls;
        }
    }
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    if (upload.Readback) { if (!ReadLumenSurfaceRecord(*record,result.Texels)) return false; }
    else
    {
        ComPtr<ID3D11Query> completion; const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0};
        status=Device->CreateQuery(&query,completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen capture completion query",status));
        Context->End(completion.Get()); Context->Flush(); constexpr ULONGLONG TimeoutMilliseconds=10000;
        const auto deadline=GetTickCount64()+TimeoutMilliseconds;
        while ((status=Context->GetData(completion.Get(),nullptr,0,0))==S_FALSE)
        { if (GetTickCount64()>=deadline) return Fail("Lumen material capture timed out"); std::this_thread::yield(); }
        if (FAILED(status)) return Fail(HResultMessage("Complete Lumen material capture",status));
    }
    result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space is exhausted");
    LumenSurfaces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Lumen surface handle is invalid or expired");
    return ReadLumenSurfaceRecord(*found->second,output);
}
bool DestroyLumenSurface(LumenSurfaceHandle handle)
{ return (handle && LumenSurfaces.erase(handle.Value)) || Fail("Lumen surface handle is invalid or already released"); }
