std::array<ComPtr<ID3D11PixelShader>,MaterialProgramCount> MaterialPixels;
ID3D11PixelShader* MaterialPixel(MaterialTechnique technique,unsigned model,bool weighted,bool geometry)
{
    if (unsigned(technique)>=MaterialTechniqueCount || model>=MaterialSurfaceModels) { Fail("Invalid DX11 material program"); return nullptr; }
    auto& pixel=MaterialPixels[MaterialProgramIndex(technique,model,weighted,geometry)];
    if (pixel) return pixel.Get();
    const auto filename=std::string(MaterialTechniqueFile(technique))+".hlsl";
    std::string source; if (!LoadTechniqueShaderSource("DX11",filename.c_str(),source,LastError)) return nullptr;
    const auto program=std::to_string(model);
    const D3D_SHADER_MACRO definitions[]{{"EVANS_MATERIAL_PROGRAM",program.c_str()},{geometry?"EVANS_MATERIAL_GBUFFER":nullptr,"1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> code,errors;
    const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),filename.c_str(),definitions,nullptr,weighted&&!geometry?"PSWeighted":"PSMain","ps_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,code.GetAddressOf(),errors.GetAddressOf());
    if (FAILED(result)) { Fail(errors?std::string(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize()):"DX11 material compilation failed"); return nullptr; }
    if (FAILED(Device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,pixel.GetAddressOf()))) { Fail("DX11 material shader creation failed"); return nullptr; }
    return pixel.Get();
}
void DestroyMaterials() { MaterialPixels={}; }
