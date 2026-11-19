std::array<ComPtr<ID3DBlob>,13> PCFPixelBytecodes;
bool EnsurePCFProgram(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms)) return Fail("DX12 PCF material program is invalid");
    auto& shader=PCFPixelBytecodes[weighted?ProgramCount+program:program];
    if (shader) return true;
    if (!LoadTechniqueShaderSource("DX12","PCF.hlsl",ShaderCode,LastError)) return false;
    const auto index=std::to_string(program);
    const D3D_SHADER_MACRO definitions[]{{"EVANS_PCF_PROGRAM",index.c_str()},{nullptr,nullptr}};
    return CompileShader(weighted?"PSWeighted":"PSMain","ps_5_0",shader,definitions,true);
}

bool BindPCF(UINT root,bool enabled)
{
    const auto constants=BuildPCFConstants(FrameShadowFilter,enabled);
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(constants)>ConstantBufferCapacity-offset) return Fail("DX12 PCF upload arena is exhausted");
    std::memcpy(ConstantData+offset,&constants,sizeof(constants));
    CommandList->SetGraphicsRootConstantBufferView(root,ConstantBuffer->GetGPUVirtualAddress()+offset);
    ConstantOffset=offset+sizeof(constants); return true;
}
void DestroyPCF() { PCFPixelBytecodes={}; }
