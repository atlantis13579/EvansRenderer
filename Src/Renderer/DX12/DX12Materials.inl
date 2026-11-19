std::array<ComPtr<ID3DBlob>,MaterialProgramCount> MaterialPixels;
bool EnsureMaterialPixel(MaterialTechnique technique,unsigned model,bool weighted,bool geometry)
{
    if (unsigned(technique)>=MaterialTechniqueCount || model>=MaterialSurfaceModels) return Fail("Invalid DX12 material program");
    auto& pixel=MaterialPixels[MaterialProgramIndex(technique,model,weighted,geometry)];
    if (pixel) return true;
    const auto filename=std::string(MaterialTechniqueFile(technique))+".hlsl";
    std::string source;
    if (!LoadTechniqueShaderSource("DX12",filename.c_str(),source,LastError)) return false;
    const auto program=std::to_string(model);
    const D3D_SHADER_MACRO definitions[]{{"EVANS_MATERIAL_PROGRAM",program.c_str()},{geometry?"EVANS_MATERIAL_GBUFFER":nullptr,"1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> diagnostics;
    const UINT flags=D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS|
        D3DCOMPILE_WARNINGS_ARE_ERRORS|D3DCOMPILE_OPTIMIZATION_LEVEL3;
    const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),filename.c_str(),definitions,nullptr,
        weighted&&!geometry?"PSWeighted":"PSMain","ps_5_0",flags,0,pixel.GetAddressOf(),diagnostics.GetAddressOf());
    if (FAILED(result)) return Fail(diagnostics?std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),
        diagnostics->GetBufferSize()):"DX12 material compilation failed: "+filename);
    return true;
}
void DestroyMaterials() { MaterialPixels={}; }
