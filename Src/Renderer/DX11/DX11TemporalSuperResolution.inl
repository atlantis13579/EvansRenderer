// Immutable GPU history versions keep candidate writes separate from the last
// presented frame. Staging Map completes native execution before publication.
struct TSRNativeSurface
{
    ComPtr<ID3D11Buffer> Buffer;
    ComPtr<ID3D11ShaderResourceView> View;
    ComPtr<ID3D11UnorderedAccessView> Unordered;
};
struct TSRNativeHistory { TSRHistoryInfo Info; std::array<TSRNativeSurface,2> Surfaces; HdrSurface Color; };
std::unordered_map<std::uint64_t,TSRNativeHistory> TSRHistories;
TSRSceneState TSRScene;
HdrSurface TSROpaque;
ComPtr<ID3D11ComputeShader> TSRShader;
ComPtr<ID3D11ComputeShader> TSRPackShader;
ComPtr<ID3D11VertexShader> TSRExportVertex;
ComPtr<ID3D11PixelShader> TSRExportPixel;
void DestroyTSR() { TSRScene.Reset(); TSROpaque={}; TSRHistories.clear(); TSRShader.Reset(); TSRPackShader.Reset(); TSRExportVertex.Reset(); TSRExportPixel.Reset(); }
bool DestroyTSRHistory(TSRHistoryHandle handle)
{ return (handle && TSRHistories.erase(handle.Value)) || Fail("DX11 TSR history handle is invalid or belongs to another RHI"); }
bool ExportTSR(TSRNativeHistory& record,TSRReadback& result,bool readback=true)
{
    if (!TSRExportVertex || !TSRExportPixel)
    {
        if (!LoadTechniqueShaderSource("DX11","TSR.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO defines[]{{"EVANS_TSR_EXPORT_STAGE","1"},{nullptr,nullptr}};
        ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader("VSExport","vs_5_0",vertex,defines,true) || !CompileShader("PSExport","ps_5_0",pixel,defines,true)) return false;
        auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,TSRExportVertex.ReleaseAndGetAddressOf());
        if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,TSRExportPixel.ReleaseAndGetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create TSR HDR export shaders",status));
    }
    if (!CreateHdrSurface(record.Color,record.Info.Width,record.Info.Height)) return false;
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(record.Info.Width,record.Info.Height,0,constants,LastError)) return false;
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&constants;
    auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create TSR export frame",status));
    Context->CSSetShader(nullptr,nullptr,0); ID3D11ShaderResourceView* empty=nullptr; Context->CSSetShaderResources(0,1,&empty);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    D3D11_VIEWPORT viewport{0,0,float(record.Info.Width),float(record.Info.Height),0,1}; Context->RSSetViewports(1,&viewport);
    auto* target=record.Color.Target.Get(); auto* view=record.Surfaces[0].View.Get(); auto* cb=uniform.Get();
    Context->OMSetRenderTargets(1,&target,nullptr); Context->PSSetShaderResources(0,1,&view); Context->PSSetConstantBuffers(0,1,&cb);
    Context->VSSetShader(TSRExportVertex.Get(),nullptr,0); Context->PSSetShader(TSRExportPixel.Get(),nullptr,0); Context->Draw(3,0);
    Context->PSSetShaderResources(0,1,&empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    if(!readback)return true;
    ComPtr<ID3D11Texture2D> staging; D3D11_TEXTURE2D_DESC read{}; record.Color.Texture->GetDesc(&read);
    read.BindFlags=0; read.Usage=D3D11_USAGE_STAGING; read.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateTexture2D(&read,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create TSR HDR readback",status));
    Context->CopyResource(staging.Get(),record.Color.Texture.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
    if (FAILED(status)) return Fail(HResultMessage("Read TSR HDR export",status));
    result.PresentationHDR.resize(std::size_t(record.Info.Width)*record.Info.Height);
    for (unsigned y=0;y<record.Info.Height;++y)
    {
        const auto* row=reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(mapped.pData)+std::size_t(y)*mapped.RowPitch);
        for (unsigned x=0;x<record.Info.Width;++x) result.PresentationHDR[std::size_t(y)*record.Info.Width+x]=DecodeTSRHalfPixel(row+x*4);
    }
    Context->Unmap(staging.Get(),0); return true;
}
struct TSRNativeImages { std::array<ComPtr<ID3D11ShaderResourceView>,TSRSceneImageCount> Views; std::uint32_t Flags=0; };
bool PackTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRNativeSurface& input)
{
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(frame.InputWidth,frame.InputHeight,images.Flags,constants,LastError)) return false;
    std::array<ID3D11ShaderResourceView*,TSRSceneImageCount> sources{};
    for (unsigned image=0;image<TSRSceneImageCount;++image)
    {
        if (!TSRImageRequired(image,images.Flags)) continue;
        auto* view=images.Views[image].Get(); if (!view) return Fail("DX11 TSR is missing a required scene image");
        ComPtr<ID3D11Device> device; view->GetDevice(device.GetAddressOf());
        ComPtr<ID3D11Resource> resource; view->GetResource(resource.GetAddressOf()); ComPtr<ID3D11Texture2D> texture;
        if (device.Get()!=Device.Get() || FAILED(resource.As(&texture))) return Fail("DX11 TSR source is not an image on this device");
        D3D11_TEXTURE2D_DESC desc{}; texture->GetDesc(&desc); D3D11_SHADER_RESOURCE_VIEW_DESC vd{}; view->GetDesc(&vd);
        if (desc.Width!=frame.InputWidth || desc.Height!=frame.InputHeight || desc.SampleDesc.Count!=1 || vd.ViewDimension!=D3D11_SRV_DIMENSION_TEXTURE2D || vd.Texture2D.MostDetailedMip!=0)
            return Fail("DX11 TSR requires resolved render-sized level-zero images");
        sources[image]=view;
    }
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&constants;
    const auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create TSR image pack frame",status));
    struct Clear
    {
        ID3D11DeviceContext* Context;
        ~Clear() { ID3D11ShaderResourceView* sources[TSRSceneImageCount]{}; ID3D11UnorderedAccessView* target=nullptr; Context->CSSetShaderResources(0,TSRSceneImageCount,sources); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); }
    } clear{Context.Get()};
    Context->OMSetRenderTargets(0,nullptr,nullptr); auto* cb=uniform.Get(); auto* target=input.Unordered.Get();
    Context->CSSetShader(TSRPackShader.Get(),nullptr,0); Context->CSSetShaderResources(0,TSRSceneImageCount,sources.data()); Context->CSSetConstantBuffers(0,1,&cb); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    Context->Dispatch((frame.InputWidth+TSRGroupSide-1)/TSRGroupSide,(frame.InputHeight+TSRGroupSide-1)/TSRGroupSide,1); return true;
}
bool ReconstructTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRHistoryHandle previous,TSRResult& output)
{
    const TSRNativeHistory* old=nullptr;
    if (previous) { const auto found=TSRHistories.find(previous.Value); if (found==TSRHistories.end()) return Fail("DX11 previous TSR history is invalid"); old=&found->second; }
    TSRConstants constants; if (!BuildTSRConstants(frame,old?&old->Info:nullptr,constants,LastError)) return false;
    if (!TSRShader || !TSRPackShader)
    {
        if (!LoadTechniqueShaderSource("DX11","TSR.hlsl",ShaderCode,LastError)) return false;
        ComPtr<ID3DBlob> code; if (!CompileShader("CSMain","cs_5_0",code,nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,TSRShader.ReleaseAndGetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create TSR compute shader",status));
        const D3D_SHADER_MACRO packDefines[]{{"EVANS_TSR_PACK_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> packed;
        if (!CompileShader("CSPack","cs_5_0",packed,packDefines,true)) return false;
        const auto packStatus=Device->CreateComputeShader(packed->GetBufferPointer(),packed->GetBufferSize(),nullptr,TSRPackShader.ReleaseAndGetAddressOf());
        if (FAILED(packStatus)) return Fail(HResultMessage("Create TSR image packing shader",packStatus));
    }
    const auto allocate=[&](std::size_t bytes,const void* data,bool writable,TSRNativeSurface& surface)
    {
        if (!bytes || bytes>UINT32_MAX) return Fail("DX11 TSR buffer exceeds native addressing");
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0);
        desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(Float4);
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data;
        auto status=Device->CreateBuffer(&desc,data?&initial:nullptr,surface.Buffer.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create TSR storage",status));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.NumElements=desc.ByteWidth/sizeof(Float4);
        status=Device->CreateShaderResourceView(surface.Buffer.Get(),&view,surface.View.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create TSR source view",status));
        if (writable)
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC target{}; target.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; target.Buffer.NumElements=view.Buffer.NumElements;
            status=Device->CreateUnorderedAccessView(surface.Buffer.Get(),&target,surface.Unordered.GetAddressOf());
            if (FAILED(status)) return Fail(HResultMessage("Create TSR output view",status));
        }
        return true;
    };
    TSRNativeSurface input,dummy; const std::array<Float4,TSRHistoryVectors> empty{};
    if (!allocate(std::size_t(frame.InputWidth)*frame.InputHeight*(TSRInputVectors*sizeof(Float4)),nullptr,true,input) || !allocate(sizeof(empty),empty.data(),false,dummy) || !PackTSRImages(frame,images,input)) return false;
    TSRNativeHistory record; record.Info=TSRHistoryIdentity(frame);
    const auto bytes=std::size_t(record.Info.Width)*record.Info.Height*TSRHistoryVectors*sizeof(Float4);
    for (auto& surface:record.Surfaces) if (!allocate(bytes,nullptr,true,surface)) return false;
    ComPtr<ID3D11Buffer> uniform; D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&constants;
    auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create TSR constants",status));
    struct Clear
    {
        ID3D11DeviceContext* Context;
        ~Clear() { ID3D11ShaderResourceView* views[2]{}; ID3D11UnorderedAccessView* targets[2]{}; ID3D11Buffer* uniform=nullptr; Context->CSSetUnorderedAccessViews(0,2,targets,nullptr); Context->CSSetShaderResources(0,2,views); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0); }
    } clear{Context.Get()};
    ID3D11ShaderResourceView* sources[]{input.View.Get(),constants.Options.X?old->Surfaces[0].View.Get():dummy.View.Get()};
    ID3D11UnorderedAccessView* targets[]{record.Surfaces[0].Unordered.Get(),record.Surfaces[1].Unordered.Get()}; auto* cb=uniform.Get();
    Context->CSSetShader(TSRShader.Get(),nullptr,0); Context->CSSetShaderResources(0,2,sources); Context->CSSetUnorderedAccessViews(0,2,targets,nullptr); Context->CSSetConstantBuffers(0,1,&cb);
    Context->Dispatch((constants.Extent.Z+TSRGroupSide-1)/TSRGroupSide,(constants.Extent.W+TSRGroupSide-1)/TSRGroupSide,1);
    ID3D11UnorderedAccessView* unbound[2]{}; Context->CSSetUnorderedAccessViews(0,2,unbound,nullptr);
    TSRResult candidate; candidate.HistoryUsed=constants.Options.X!=0; candidate.Info=record.Info;
    std::vector<Float4>* mirrors[]{&candidate.Readback.History,&candidate.Readback.Auxiliary};
    for (unsigned surface=0;frame.CaptureDiagnostics && surface<record.Surfaces.size();++surface)
    {
        ComPtr<ID3D11Buffer> staging; D3D11_BUFFER_DESC read{}; read.ByteWidth=static_cast<UINT>(bytes); read.Usage=D3D11_USAGE_STAGING; read.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        status=Device->CreateBuffer(&read,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create TSR readback",status));
        Context->CopyResource(staging.Get(),record.Surfaces[surface].Buffer.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
        if (FAILED(status)) return Fail(HResultMessage("Map TSR result",status)); mirrors[surface]->resize(bytes/sizeof(Float4)); std::memcpy(mirrors[surface]->data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    }
    if (!ExportTSR(record,candidate.Readback,frame.CaptureDiagnostics) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,candidate.Readback,LastError))) return false;
    candidate.Handle=NewTSRHistoryHandle(); if (!candidate.Handle) return Fail("TSR history handle space is exhausted");
    TSRHistories.emplace(candidate.Handle.Value,std::move(record)); output=std::move(candidate); return true;
}
bool PrepareTSRScene(const TSRSceneFrame& requested,const Techniques::PostProcessSettings& post)
{
    const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value);
    const auto found=TSRHistories.find(TSRScene.Committed.Value);
    return TSRScene.Begin(requested,CreateInfo.Width,CreateInfo.Height,post,found==TSRHistories.end()?nullptr:&found->second.Info,LastError);
}
void AbortTSRScene() { const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value); }
void CommitTSRScene() { const auto old=TSRScene.Commit(); if (old) TSRHistories.erase(old.Value); }
bool RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& requested,FrameCapture& output)
{
    const auto found=TSRHistories.find(handle.Value); if (found==TSRHistories.end()) return Fail("DX11 TSR display history is invalid");
    Techniques::PostProcessSettings settings; if (!BuildTSRDisplaySettings(requested,settings,LastError)) return false;
    const auto& history=found->second;
    // A typed, closed-frame scope gives existing HDR/display passes a complete
    // presentation resource set. Scene depth, motion and histories stay owned by
    // the scene. The source HDR image is copied before any mutable post pass.
    struct Targets
    {
        int Width=0,Height=0; bool HdrActive=false; FrameCapture Capture;
        ComPtr<ID3D11Texture2D> Color,Staging,Post;
        ComPtr<ID3D11RenderTargetView> Target,PostTarget;
        ComPtr<ID3D11ShaderResourceView> SceneView,PostView;
        std::array<HdrSurface,HdrSurfaceCount> HDR;
        void Swap(Impl& owner)
        {
            std::swap(Width,owner.CreateInfo.Width); std::swap(Height,owner.CreateInfo.Height); std::swap(HdrActive,owner.HdrSceneActive); std::swap(Capture,owner.PendingFrame);
            Color.Swap(owner.ColorTexture); Staging.Swap(owner.ColorStaging); Post.Swap(owner.PostTexture); Target.Swap(owner.ColorTarget); PostTarget.Swap(owner.PostTarget);
            SceneView.Swap(owner.SceneColorView); PostView.Swap(owner.PostView); HDR.swap(owner.HdrSurfaces);
        }
    } targets;
    targets.Width=history.Info.Width; targets.Height=history.Info.Height;
    D3D11_TEXTURE2D_DESC desc{}; desc.Width=history.Info.Width; desc.Height=history.Info.Height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_R8G8B8A8_TYPELESS; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    auto status=Device->CreateTexture2D(&desc,nullptr,targets.Color.GetAddressOf());
    D3D11_RENDER_TARGET_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; view.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2D;
    if (SUCCEEDED(status)) status=Device->CreateRenderTargetView(targets.Color.Get(),&view,targets.Target.GetAddressOf());
    desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=0; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    if (SUCCEEDED(status)) status=Device->CreateTexture2D(&desc,nullptr,targets.Staging.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create TSR display targets",status));
    {
        struct Restore { Impl& Owner; Targets& Data; Restore(Impl& owner,Targets& data):Owner(owner),Data(data){Data.Swap(Owner);} ~Restore(){Data.Swap(Owner);} } restore(*this,targets);
        if (!PrepareHdrFrame(settings.HDR)) return false;
        Context->OMSetRenderTargets(0,nullptr,nullptr); Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),history.Color.Texture.Get());
        ID3D11Texture2D* color=nullptr;
        if (!ApplyHdr(settings.HDR) || !ApplyPostProcessing(settings,color)) return false;
        if(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics){if(!PublishGpuColor(color))return false;}
        else if(!ReadColor(color,PendingFrame.ColorRGBA8))return false;
        PendingFrame.Width=history.Info.Width; PendingFrame.Height=history.Info.Height; PendingFrame.Statistics.PassCount=PendingFrame.Statistics.PostProcessPassCount;
    }
    output=std::move(targets.Capture); return true;
}
bool SnapshotTSROpaque()
{
    if (!TSRScene.Active || TSRScene.OpaqueReady || !OpaqueEffectsComplete() || !SceneColorWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved)) return Fail("TSR snapshot requires completed opaque effects exactly once");
    if (!CreateHdrSurface(TSROpaque,CreateInfo.Width,CreateInfo.Height)) return false;
    Context->OMSetRenderTargets(0,nullptr,nullptr); Context->CopyResource(TSROpaque.Texture.Get(),HdrSurfaces[HdrScene].Texture.Get()); TSRScene.OpaqueReady=true; return true;
}
bool FinishTSRScene()
{
    if (!TSRScene.Active) return true;
    TSRScene.MotionReady=VelocityCaptured && (FrameSampleCount==1 || SamplesVelocityResolved);
    if (!TSRScene.OpaqueReady || !TSRScene.MotionReady) return Fail("TSR presentation requires opaque snapshot and resolved physical motion");
    struct Restore { bool& Value; explicit Restore(bool& value):Value(value){Value=false;} ~Restore(){Value=true;} } restore(FrameOpen);
    TSRNativeImages images; images.Flags=TSRTextureMotion; images.Views[TSRSceneFinal]=HdrSurfaces[HdrScene].View; images.Views[TSRSceneOpaque]=TSROpaque.View; images.Views[TSRSceneMotion]=MotionSurfaces[MotionVelocity].View; images.Views[TSRSceneDepth]=DepthShaderView;
    if (HasOutlineMaskThisFrame)
    {
        const auto status=Device->CreateShaderResourceView(OutlineMaskTexture.Get(),nullptr,images.Views[TSRSceneOutline].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Read TSR outline input",status)); images.Flags|=TSRTextureOutline;
    }
    TSRResult candidate; if (!ReconstructTSRImages(TSRScene.Frame,images,TSRScene.Committed,candidate)) return false; TSRScene.Pending=candidate.Handle;
    FrameCapture display; if (!RenderTSRDisplay(candidate.Handle,FramePostProcessing,display) || !ComposeTSRSceneCapture(TSRScene.Frame,candidate,std::move(display),PendingFrame,LastError)) return false;
    if (HasOutlineMaskThisFrame && TSRScene.Frame.CaptureDiagnostics)
    {
        constexpr float UNormMaximum=65535; std::vector<std::uint16_t> mask(std::size_t(PendingFrame.Width)*PendingFrame.Height*4);
        for (std::size_t pixel=0;pixel<mask.size()/4;++pixel) { const auto p=candidate.Readback.Auxiliary[pixel*TSRAuxiliaryVectors+1]; const float values[]{p.X,p.Y,p.Z,p.W}; for (unsigned c=0;c<4;++c) mask[pixel*4+c]=static_cast<std::uint16_t>(std::lround(std::clamp(values[c],0.0f,1.0f)*UNormMaximum)); }
        struct Extent { Impl& Owner; int Width,Height; explicit Extent(Impl& owner):Owner(owner),Width(owner.CreateInfo.Width),Height(owner.CreateInfo.Height){Owner.CreateInfo.Width=Owner.PendingFrame.Width;Owner.CreateInfo.Height=Owner.PendingFrame.Height;} ~Extent(){Owner.CreateInfo.Width=Width;Owner.CreateInfo.Height=Height;} } extent(*this);
        ApplyOutline(mask,PendingFrame.ColorRGBA8);
    }
    return true;
}
