// Typed integer storage preserves BVH addresses and cutout floats exactly.
// Immutable scene images are retained across camera/light motion; the renderer
// publishes a new image only when geometry, light kind or trace controls change.
static constexpr UINT RayShadowImageSlot=20;
std::shared_ptr<const RayShadowImage> FrameRayShadowImage,CachedRayShadowImage;
ComPtr<ID3D11Texture2D> RayShadowTexture,RayShadowDummyTexture;
ComPtr<ID3D11ShaderResourceView> RayShadowView,RayShadowDummyView;
std::array<ComPtr<ID3D11PixelShader>,13> RayShadowPrograms;

bool CreateRayShadowTexture(const RayShadowImage& image,ComPtr<ID3D11Texture2D>& texture,
    ComPtr<ID3D11ShaderResourceView>& view)
{
    if (!image.Width || !image.Height || image.Texels.size()!=std::size_t(image.Width)*image.Height)
        return Fail("DX11 ray-shadow image has invalid dimensions or payload");
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width=image.Width; desc.Height=image.Height; desc.MipLevels=1; desc.ArraySize=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_UINT; desc.SampleDesc.Count=1;
    desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=image.Texels.data(); data.SysMemPitch=image.Width*sizeof(RayShadowTexel);
    ComPtr<ID3D11Texture2D> candidate; ComPtr<ID3D11ShaderResourceView> candidateView;
    auto result=Device->CreateTexture2D(&desc,&data,candidate.GetAddressOf());
    if (SUCCEEDED(result)) result=Device->CreateShaderResourceView(candidate.Get(),nullptr,candidateView.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create DX11 ray-shadow integer image",result));
    texture=std::move(candidate); view=std::move(candidateView); return true;
}
bool PrepareRayShadowImage(std::shared_ptr<const RayShadowImage> image)
{
    if (!RayShadowDummyView)
    {
        RayShadowImage empty; empty.Width=RayShadowHeaderTexels; empty.Height=1; empty.Texels.resize(RayShadowHeaderTexels);
        if (!CreateRayShadowTexture(empty,RayShadowDummyTexture,RayShadowDummyView)) return false;
    }
    if (image && image!=CachedRayShadowImage)
    {
        if (!CreateRayShadowTexture(*image,RayShadowTexture,RayShadowView)) return false;
        CachedRayShadowImage=image;
    }
    FrameRayShadowImage=std::move(image); return true;
}
void BindRayShadowImage()
{
    auto* view=FrameRayShadowImage?RayShadowView.Get():RayShadowDummyView.Get();
    Context->PSSetShaderResources(RayShadowImageSlot,1,&view);
    Context->VSSetShaderResources(RayShadowImageSlot,1,&view);
}
ID3D11PixelShader* RayShadowPixel(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms))
    { Fail("DX11 ray-shadow material program is invalid"); return nullptr; }
    auto& shader=RayShadowPrograms[weighted?ProgramCount+program:program];
    if (!shader)
    {
        std::string source; if (!LoadTechniqueShaderSource("DX11","RayTracedShadows.hlsl",source,LastError)) return nullptr;
        const auto text=std::to_string(program);
        const D3D_SHADER_MACRO definitions[]{{"EVANS_RAY_SHADOW_PROGRAM",text.c_str()},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code,diagnostics;
        const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),"DX11/RayTracedShadows.hlsl",definitions,nullptr,
            weighted?"PSWeighted":"PSMain","ps_5_0",D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,code.GetAddressOf(),diagnostics.GetAddressOf());
        if (FAILED(result))
        { Fail(diagnostics?std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()):"DX11 ray-shadow shader compilation failed"); return nullptr; }
        if (FAILED(Device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,shader.GetAddressOf())))
        { Fail("DX11 ray-shadow pixel shader creation failed"); return nullptr; }
    }
    return shader.Get();
}
void DestroyRayShadowImage()
{
    FrameRayShadowImage.reset(); CachedRayShadowImage.reset(); RayShadowView.Reset(); RayShadowTexture.Reset();
    RayShadowDummyView.Reset(); RayShadowDummyTexture.Reset(); for (auto& shader:RayShadowPrograms) shader.Reset();
}
