std::shared_ptr<const RayShadowImage> FrameRayShadowImage,CachedRayShadowImage;
ComPtr<ID3D12Resource> RayShadowTexture,RayShadowDummyTexture;
UINT RayShadowDescriptor=InvalidDescriptor,RayShadowDummyDescriptor=InvalidDescriptor;
bool RayShadowUploadedThisFrame=false,RayShadowDummyUploadedThisFrame=false;
std::array<ComPtr<ID3DBlob>,13> RayShadowPixelBytecodes;

// The copy belongs to this recording frame. Abort invalidates its cache marker;
// the next frame cannot reuse an image whose upload was never submitted.
bool CreateRayShadowTexture(const RayShadowImage& image,ComPtr<ID3D12Resource>& texture,UINT& descriptor)
{
    if (!image.Width || !image.Height || image.Texels.size()!=std::size_t(image.Width)*image.Height)
        return Fail("DX12 ray-shadow image has invalid dimensions or payload");
    if (descriptor==InvalidDescriptor) descriptor=SRVArena.Allocate();
    if (descriptor==InvalidDescriptor) return Fail("DX12 ray-shadow descriptor arena is exhausted");
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=image.Width; desc.Height=image.Height; desc.DepthOrArraySize=1; desc.MipLevels=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_UINT; desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); ComPtr<ID3D12Resource> candidate,upload;
    auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(candidate.GetAddressOf()));
    if (FAILED(result)) return FailHRESULT("Create DX12 ray-shadow image",result);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT rows=0; UINT64 rowBytes=0,total=0;
    Device->GetCopyableFootprints(&desc,0,1,0,&footprint,&rows,&rowBytes,&total);
    if (!CreateBufferResource(std::size_t(total),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,upload)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; result=upload->Map(0,&noRead,&mapped);
    if (FAILED(result)) return FailHRESULT("Map DX12 ray-shadow upload",result);
    const auto pitch=std::size_t(image.Width)*sizeof(RayShadowTexel);
    for (unsigned row=0;row<image.Height;++row)
        std::memcpy(static_cast<std::uint8_t*>(mapped)+footprint.Offset+std::size_t(row)*footprint.Footprint.RowPitch,
            reinterpret_cast<const std::uint8_t*>(image.Texels.data())+std::size_t(row)*pitch,pitch);
    upload->Unmap(0,nullptr);
    D3D12_TEXTURE_COPY_LOCATION destination{},source{};
    destination.pResource=candidate.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.pResource=upload.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint=footprint;
    CommandList->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
    auto state=D3D12_RESOURCE_STATE_COPY_DEST;
    Transition(candidate.Get(),state,D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=desc.Format; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
    Device->CreateShaderResourceView(candidate.Get(),&view,SRVArena.Cpu(descriptor));
    if (texture) FrameUploads.push_back(texture);
    FrameUploads.push_back(std::move(upload)); texture=std::move(candidate); return true;
}
bool PrepareRayShadowImage(std::shared_ptr<const RayShadowImage> image)
{
    RayShadowUploadedThisFrame=RayShadowDummyUploadedThisFrame=false;
    if (!RayShadowDummyTexture)
    {
        RayShadowImage empty; empty.Width=RayShadowHeaderTexels; empty.Height=1; empty.Texels.resize(RayShadowHeaderTexels);
        if (!CreateRayShadowTexture(empty,RayShadowDummyTexture,RayShadowDummyDescriptor)) return false;
        RayShadowDummyUploadedThisFrame=true;
    }
    if (image && image!=CachedRayShadowImage)
    {
        if (!CreateRayShadowTexture(*image,RayShadowTexture,RayShadowDescriptor)) return false;
        CachedRayShadowImage=image; RayShadowUploadedThisFrame=true;
    }
    FrameRayShadowImage=std::move(image); return true;
}
void FinishRayShadowImage(bool publish)
{
    if (!publish && RayShadowUploadedThisFrame) CachedRayShadowImage.reset();
    if (!publish && RayShadowDummyUploadedThisFrame) RayShadowDummyTexture.Reset();
    RayShadowUploadedThisFrame=RayShadowDummyUploadedThisFrame=false; FrameRayShadowImage.reset();
}
struct RayShadowPresentPublication
{
    Impl& Owner; bool Published=false;
    ~RayShadowPresentPublication()
    { if (Published || !Owner.FrameOpen || Owner.LumenCommandsEnded()) Owner.FinishRayShadowImage(Published); }
};
void BindRayShadowImage(UINT root,bool compute=false)
{
    const auto descriptor=SRVArena.Gpu(FrameRayShadowImage?RayShadowDescriptor:RayShadowDummyDescriptor);
    if (compute) CommandList->SetComputeRootDescriptorTable(root,descriptor);
    else CommandList->SetGraphicsRootDescriptorTable(root,descriptor);
}
bool EnsureRayShadowProgram(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms)) return Fail("DX12 ray-shadow material program is invalid");
    auto& shader=RayShadowPixelBytecodes[weighted?ProgramCount+program:program];
    if (shader) return true;
    if (!LoadTechniqueShaderSource("DX12","RayTracedShadows.hlsl",ShaderCode,LastError)) return false;
    const auto index=std::to_string(program);
    const D3D_SHADER_MACRO definitions[]{{"EVANS_RAY_SHADOW_PROGRAM",index.c_str()},{nullptr,nullptr}};
    return CompileShader(weighted?"PSWeighted":"PSMain","ps_5_0",shader,definitions,true);
}
void DestroyRayShadowImage()
{
    FrameRayShadowImage.reset(); CachedRayShadowImage.reset(); RayShadowTexture.Reset(); RayShadowDummyTexture.Reset();
    SRVArena.Release(RayShadowDescriptor); SRVArena.Release(RayShadowDummyDescriptor);
    RayShadowDescriptor=RayShadowDummyDescriptor=InvalidDescriptor; RayShadowPixelBytecodes={};
}
