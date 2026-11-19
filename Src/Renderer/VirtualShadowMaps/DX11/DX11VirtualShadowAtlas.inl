struct VirtualShadowAtlasRecord
{
    VirtualShadowSnapshot Info;
    ComPtr<ID3D11Texture2D> Depth,Directory;
    ComPtr<ID3D11DepthStencilView> Target;
    ComPtr<ID3D11ShaderResourceView> View,DirectoryView;
};
std::unordered_map<std::uint64_t,std::shared_ptr<VirtualShadowAtlasRecord>> VirtualShadowAtlases;
ComPtr<ID3D11VertexShader> VirtualAtlasVertex,VirtualAtlasCopyVertex;
ComPtr<ID3D11PixelShader> VirtualAtlasPixel,VirtualAtlasCopyPixel;
ComPtr<ID3D11ComputeShader> VirtualAtlasRead;
ComPtr<ID3D11InputLayout> VirtualAtlasLayout;
std::array<ComPtr<ID3D11RasterizerState>,3> VirtualAtlasRaster;
ComPtr<ID3D11DepthStencilState> VirtualAtlasDepthState,VirtualAtlasCopyState;
void DestroyVirtualShadowAtlases()
{
    DestroyVirtualShadowLookup();
    VirtualShadowAtlases.clear(); VirtualAtlasVertex.Reset(); VirtualAtlasCopyVertex.Reset(); VirtualAtlasPixel.Reset(); VirtualAtlasCopyPixel.Reset();
    VirtualAtlasRead.Reset(); VirtualAtlasLayout.Reset(); for (auto& state:VirtualAtlasRaster) state.Reset(); VirtualAtlasDepthState.Reset(); VirtualAtlasCopyState.Reset();
}
bool EnsureVirtualShadowAtlas()
{
    if (VirtualAtlasVertex && VirtualAtlasPixel && VirtualAtlasCopyVertex && VirtualAtlasCopyPixel && VirtualAtlasRead && VirtualAtlasLayout && VirtualAtlasDepthState && VirtualAtlasCopyState && VirtualAtlasRaster[0] && VirtualAtlasRaster[1] && VirtualAtlasRaster[2]) return true;
    if (!LoadTechniqueShaderSource("DX11","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO options[]{{"EVANS_VIRTUAL_SHADOW_RASTER_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> vertex,pixel,copyVertex,copyPixel,read;
    if (!CompileShader("VSDepth","vs_5_0",vertex,options,true) || !CompileShader("PSDepth","ps_5_0",pixel,options,true) ||
        !CompileShader("VSCopy","vs_5_0",copyVertex,options,true) || !CompileShader("PSCopy","ps_5_0",copyPixel,options,true) || !CompileShader("CSRead","cs_5_0",read,options,true)) return false;
    auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,VirtualAtlasVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,VirtualAtlasPixel.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateVertexShader(copyVertex->GetBufferPointer(),copyVertex->GetBufferSize(),nullptr,VirtualAtlasCopyVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(copyPixel->GetBufferPointer(),copyPixel->GetBufferSize(),nullptr,VirtualAtlasCopyPixel.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateComputeShader(read->GetBufferPointer(),read->GetBufferSize(),nullptr,VirtualAtlasRead.ReleaseAndGetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create virtual atlas shaders",status));
    const D3D11_INPUT_ELEMENT_DESC elements[]{{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D11_INPUT_PER_VERTEX_DATA,0},{"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D11_INPUT_PER_VERTEX_DATA,0}};
    status=Device->CreateInputLayout(elements,3,vertex->GetBufferPointer(),vertex->GetBufferSize(),VirtualAtlasLayout.ReleaseAndGetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create virtual atlas vertex layout",status));
    for (unsigned cull=0;cull<VirtualAtlasRaster.size();++cull)
    {
        D3D11_RASTERIZER_DESC desc{}; desc.FillMode=D3D11_FILL_SOLID; desc.CullMode=cull?D3D11_CULL_BACK:D3D11_CULL_NONE; desc.DepthClipEnable=TRUE; desc.ScissorEnable=TRUE;
        desc.FrontCounterClockwise=cull==2?FALSE:TRUE;
        status=Device->CreateRasterizerState(&desc,VirtualAtlasRaster[cull].ReleaseAndGetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create virtual atlas raster state",status));
    }
    D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    status=Device->CreateDepthStencilState(&depth,VirtualAtlasDepthState.ReleaseAndGetAddressOf()); depth.DepthFunc=D3D11_COMPARISON_ALWAYS;
    if (SUCCEEDED(status)) status=Device->CreateDepthStencilState(&depth,VirtualAtlasCopyState.ReleaseAndGetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create virtual atlas depth state",status));
}
bool PackVirtualShadowDepth(ID3D11ShaderResourceView* source,unsigned width,unsigned height,ComPtr<ID3D11Buffer>& packed)
{
    if (!EnsureVirtualShadowAtlas()) return false;
    const auto count=std::size_t(width)*height,bytes=count*sizeof(float);
    ComPtr<ID3D11Buffer> data,uniform; ComPtr<ID3D11UnorderedAccessView> view;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(float);
    auto status=Device->CreateBuffer(&desc,nullptr,data.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create atlas depth feedback",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=static_cast<UINT>(count);
    status=Device->CreateUnorderedAccessView(data.Get(),&uav,view.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View atlas depth feedback",status));
    VirtualShadowRasterConstants constants; constants.Extent={width,height,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    desc={}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&constants;
    status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create atlas feedback frame",status));
    Context->OMSetRenderTargets(0,nullptr,nullptr); auto* target=view.Get(); auto* cb=uniform.Get();
    Context->CSSetShader(VirtualAtlasRead.Get(),nullptr,0); Context->CSSetShaderResources(2,1,&source); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetConstantBuffers(0,1,&cb);
    const auto groups=static_cast<unsigned>((count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth);
    Context->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    ID3D11ShaderResourceView* emptySource=nullptr; ID3D11UnorderedAccessView* emptyTarget=nullptr; Context->CSSetShaderResources(2,1,&emptySource); Context->CSSetUnorderedAccessViews(0,1,&emptyTarget,nullptr); Context->CSSetShader(nullptr,nullptr,0);
    packed=std::move(data); return true;
}
bool ReadVirtualShadowAtlasRecord(const VirtualShadowAtlasRecord& record,std::vector<float>& output)
{
    const auto count=std::size_t(record.Info.Layout.AtlasWidth)*record.Info.Layout.AtlasHeight,bytes=count*sizeof(float);
    ComPtr<ID3D11Buffer> data,staging;
    if (!PackVirtualShadowDepth(record.View.Get(),record.Info.Layout.AtlasWidth,record.Info.Layout.AtlasHeight,data)) return false;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create atlas depth staging",status));
    Context->CopyResource(staging.Get(),data.Get()); std::vector<float> result(count); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Map atlas depth",status));
    std::memcpy(result.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateVirtualShadowAtlasDepth(record.Info,result,LastError)) return false; output=std::move(result); return true;
}
bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("D3D11 virtual atlas handle is invalid");
    return ReadVirtualShadowAtlasRecord(*found->second,output);
}
bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{ return (handle && VirtualShadowAtlases.erase(handle.Value)) || Fail("D3D11 virtual atlas handle is invalid or already released"); }
bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    std::shared_ptr<VirtualShadowAtlasRecord> parent;
    if (previous) { const auto found=VirtualShadowAtlases.find(previous.Value); if (found==VirtualShadowAtlases.end()) return Fail("D3D11 virtual atlas parent is invalid"); parent=found->second; }
    VirtualShadowPlan plan;
    if (!BuildVirtualShadowPlan(upload.Identity,upload.Settings,upload.Requests,upload.Invalidations,parent?&parent->Info:nullptr,plan,LastError) || !EnsureVirtualShadowAtlas()) return false;
    for (const auto& caster:upload.Casters)
    {
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Virtual atlas caster buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto image=Textures.find(texture.Value); if (image==Textures.end() || image->second.Description.Format!=TextureFormat::RGBA8) return Fail("Virtual atlas coverage requires existing color textures"); }
    }
    auto record=std::make_shared<VirtualShadowAtlasRecord>(); record->Info=plan.Candidate;
    const auto& layout=record->Info.Layout; D3D11_TEXTURE2D_DESC desc{}; desc.Width=layout.AtlasWidth; desc.Height=layout.AtlasHeight;
    desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1; desc.Format=DXGI_FORMAT_R32_TYPELESS; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
    auto status=Device->CreateTexture2D(&desc,nullptr,record->Depth.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create immutable virtual depth atlas",status));
    D3D11_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
    status=Device->CreateDepthStencilView(record->Depth.Get(),&dsv,record->Target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Target virtual depth atlas",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels=1;
    status=Device->CreateShaderResourceView(record->Depth.Get(),&srv,record->View.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Sample virtual depth atlas",status));
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC cbdesc{}; cbdesc.ByteWidth=sizeof(VirtualShadowRasterConstants); cbdesc.Usage=D3D11_USAGE_DEFAULT; cbdesc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    status=Device->CreateBuffer(&cbdesc,nullptr,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create virtual atlas draw frame",status));
    struct Cleanup
    {
        ID3D11DeviceContext* Context;
        ~Cleanup() { ID3D11ShaderResourceView* empty[3]{}; Context->PSSetShaderResources(0,3,empty); Context->OMSetRenderTargets(0,nullptr,nullptr); }
    } cleanup{Context.Get()};
    Context->OMSetRenderTargets(0,nullptr,record->Target.Get()); Context->ClearDepthStencilView(record->Target.Get(),D3D11_CLEAR_DEPTH,1,0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); auto* cb=uniform.Get(); Context->VSSetConstantBuffers(0,1,&cb); Context->PSSetConstantBuffers(0,1,&cb);
    const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
    const auto viewport=[&](const VirtualShadowRasterConstants& constants)
    {
        const D3D11_VIEWPORT area{constants.Origins.X,constants.Origins.Y,float(stride),float(stride),0,1};
        const D3D11_RECT scissor{LONG(constants.Origins.X),LONG(constants.Origins.Y),LONG(constants.Origins.X)+LONG(stride),LONG(constants.Origins.Y)+LONG(stride)};
        Context->RSSetViewports(1,&area); Context->RSSetScissorRects(1,&scissor); Context->UpdateSubresource(uniform.Get(),0,nullptr,&constants,0,0);
    };
    Context->RSSetState(VirtualAtlasRaster[0].Get()); Context->OMSetDepthStencilState(VirtualAtlasCopyState.Get(),0); Context->IASetInputLayout(nullptr);
    Context->VSSetShader(VirtualAtlasCopyVertex.Get(),nullptr,0); Context->PSSetShader(VirtualAtlasCopyPixel.Get(),nullptr,0);
    if (parent) { auto* source=parent->View.Get(); Context->PSSetShaderResources(2,1,&source); }
    for (const auto& copy:plan.Copies) { viewport(BuildVirtualShadowCopyConstants(parent->Info.Layout,layout,copy)); Context->Draw(3,0); }
    ID3D11ShaderResourceView* empty=nullptr; Context->PSSetShaderResources(2,1,&empty);
    Context->VSSetShader(VirtualAtlasVertex.Get(),nullptr,0); Context->PSSetShader(VirtualAtlasPixel.Get(),nullptr,0); Context->IASetInputLayout(VirtualAtlasLayout.Get()); Context->OMSetDepthStencilState(VirtualAtlasDepthState.Get(),0);
    auto* sampler=SurfaceSampler.Get(); Context->PSSetSamplers(0,1,&sampler); VirtualShadowAtlasResult result;
    for (const auto& tile:plan.Dirty) for (const auto& caster:upload.Casters)
    {
        VirtualShadowRasterConstants constants; if (!BuildVirtualShadowRasterConstants(record->Info,tile,caster,constants,LastError)) return false; viewport(constants);
        const unsigned raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0;
        Context->RSSetState(VirtualAtlasRaster[raster].Get()); const UINT vertexStride=sizeof(Vertex),offset=0; auto* vertex=Buffers.at(caster.Vertices.Value).Buffer.Get();
        Context->IASetVertexBuffers(0,1,&vertex,&vertexStride,&offset); Context->IASetIndexBuffer(Buffers.at(caster.Indices.Value).Buffer.Get(),DXGI_FORMAT_R32_UINT,0);
        ID3D11ShaderResourceView* images[]{caster.Diffuse?Textures.at(caster.Diffuse.Value).ShaderResourceView.Get():nullptr,caster.Opacity?Textures.at(caster.Opacity.Value).ShaderResourceView.Get():nullptr};
        Context->PSSetShaderResources(0,2,images); Context->DrawIndexed(caster.IndexCount,caster.FirstIndex,0); ++result.DrawCalls;
    }
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    if (upload.Readback && !ReadVirtualShadowAtlasRecord(*record,result.Depth)) return false;
    else if (!upload.Readback)
    {
        // Publication also waits when the caller does not request a CPU mirror.
        ComPtr<ID3D11Query> completed; const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0};
        status=Device->CreateQuery(&query,completed.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create virtual atlas completion query",status));
        Context->End(completed.Get()); Context->Flush();
        constexpr ULONGLONG CompletionTimeoutMilliseconds=10000;
        const auto deadline=GetTickCount64()+CompletionTimeoutMilliseconds;
        while ((status=Context->GetData(completed.Get(),nullptr,0,0))==S_FALSE)
        {
            if (GetTickCount64()>=deadline) return Fail("D3D11 virtual atlas completion timed out");
            std::this_thread::yield();
        }
        if (FAILED(status)) return Fail(HResultMessage("Complete virtual atlas",status));
    }
    std::vector<unsigned> rendered; for (const auto& dirty:plan.Dirty) rendered.push_back(dirty.Page);
    if (!CompleteVirtualShadowPlan(plan,rendered,record->Info,LastError)) return false;
    if (!CreateVirtualShadowDirectory(*record)) return false;
    result.Handle=NewVirtualShadowAtlasHandle(); if (!result.Handle) return Fail("Virtual atlas identity space is exhausted"); result.Info=record->Info;
    result.CopiedPages=static_cast<unsigned>(plan.Copies.size()); result.RenderedPages=static_cast<unsigned>(plan.Dirty.size()); VirtualShadowAtlases.emplace(result.Handle.Value,record); output=std::move(result); return true;
}
