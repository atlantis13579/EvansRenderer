// The standard mesh constant prefix and material SRVs remain unchanged. Only
// material capture selects the independent POM shaders and private b1 block.
ComPtr<ID3D11VertexShader> ParallaxVertex;
ComPtr<ID3D11PixelShader> ParallaxPixel;
ComPtr<ID3D11Buffer> ParallaxUniforms;
void DestroyParallax()
{ ParallaxVertex.Reset(); ParallaxPixel.Reset(); ParallaxUniforms.Reset(); }
bool BindParallax()
{
    if (!ParallaxVertex || !ParallaxPixel || !ParallaxUniforms)
    {
        if (!LoadTechniqueShaderSource("DX11","ParallaxOcclusionMapping.hlsl",ShaderCode,LastError)) return false;
        ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
        auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,ParallaxVertex.ReleaseAndGetAddressOf());
        if (SUCCEEDED(result)) result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,ParallaxPixel.ReleaseAndGetAddressOf());
        D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(ParallaxConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
        buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        if (SUCCEEDED(result)) result=Device->CreateBuffer(&buffer,nullptr,ParallaxUniforms.ReleaseAndGetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("Create POM shaders/constants",result));
    }
    ParallaxConstants constants;
    if (!BuildParallaxConstants(BoundPipeline.Parallax,BoundPipeline.ParallaxBoundsMinimum,BoundPipeline.ParallaxBoundsMaximum,constants))
        return Fail("POM requires finite nonempty local object bounds");
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(Context->Map(ParallaxUniforms.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return Fail("Map POM constants failed");
    std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(ParallaxUniforms.Get(),0);
    auto* buffer=ParallaxUniforms.Get(); Context->PSSetConstantBuffers(1,1,&buffer);
    Context->VSSetShader(ParallaxVertex.Get(),nullptr,0); Context->PSSetShader(ParallaxPixel.Get(),nullptr,0); return true;
}
