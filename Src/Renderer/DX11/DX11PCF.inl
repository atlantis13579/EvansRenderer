std::array<ComPtr<ID3D11PixelShader>,13> PCFPixels;
ComPtr<ID3D11Buffer> PCFBuffer;
ID3D11PixelShader* PCFPixel(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms))
    { Fail("DX11 PCF material program is invalid"); return nullptr; }
    auto& shader=PCFPixels[weighted?ProgramCount+program:program];
    if (!shader)
    {
        std::string source; if (!LoadTechniqueShaderSource("DX11","PCF.hlsl",source,LastError)) return nullptr;
        const auto text=std::to_string(program);
        const D3D_SHADER_MACRO definitions[]{{"EVANS_PCF_PROGRAM",text.c_str()},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code,diagnostics;
        const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),"DX11/PCF.hlsl",definitions,nullptr,
            weighted?"PSWeighted":"PSMain","ps_5_0",D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,code.GetAddressOf(),diagnostics.GetAddressOf());
        if (FAILED(result))
        { Fail(diagnostics?std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()):"DX11 PCF shader compilation failed"); return nullptr; }
        if (FAILED(Device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,shader.GetAddressOf())))
        { Fail("DX11 PCF pixel shader creation failed"); return nullptr; }
    }
    return shader.Get();
}

bool BindPCF(bool enabled)
{
    if (!PCFBuffer)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(PCFConstants); desc.Usage=D3D11_USAGE_DYNAMIC;
        desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        if (FAILED(Device->CreateBuffer(&desc,nullptr,PCFBuffer.GetAddressOf()))) return Fail("DX11 PCF buffer allocation failed");
    }
    const auto constants=BuildPCFConstants(FrameShadowFilter,enabled);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(Context->Map(PCFBuffer.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return Fail("DX11 PCF control upload failed");
    std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(PCFBuffer.Get(),0);
    constexpr UINT ReceiverSlot=5; auto* buffer=PCFBuffer.Get(); Context->PSSetConstantBuffers(ReceiverSlot,1,&buffer); return true;
}
void DestroyPCF() { PCFPixels={}; PCFBuffer.Reset(); }
