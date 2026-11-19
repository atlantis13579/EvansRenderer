// Mesh stages and inline ray queries share DXC, not shader source. Shader-model
// support and feature-specific tiers are negotiated independently.
mutable HMODULE DXILCompilerModule=nullptr;
mutable bool DXILCompilerChecked=false;
bool SupportsDXIL65() const
{
    if (!Device) return false;
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_5};
    if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model))) || model.HighestShaderModel<D3D_SHADER_MODEL_6_5) return false;
    if (!DXILCompilerChecked)
    {
        DXILCompilerModule=LoadLibraryExW(L"dxcompiler.dll",nullptr,LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        DXILCompilerChecked=true;
    }
    return DXILCompilerModule && GetProcAddress(DXILCompilerModule,"DxcCreateInstance");
}
void DestroyDXILCompiler()
{
    if (DXILCompilerModule) FreeLibrary(DXILCompilerModule);
    DXILCompilerModule=nullptr; DXILCompilerChecked=false;
}
bool CompileDXILCode(const char* file,const wchar_t* entry,const wchar_t* profile,
    const std::vector<const wchar_t*>& definitions,ComPtr<IDxcBlob>& output,bool strict=false)
{
    if (!SupportsDXIL65()) return Fail("DX12 shader requires SM 6.5 and the runtime DXC compiler");
    std::string source; if (!LoadTechniqueShaderSource("DX12",file,source,LastError)) return false;
    const auto create=reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(DXILCompilerModule,"DxcCreateInstance"));
    const DxcBuffer input{source.data(),source.size(),DXC_CP_UTF8};
    std::vector<const wchar_t*> arguments{L"-E",entry,L"-T",profile,L"-HV",L"2021",L"-Ges",L"-O3"};
    if (strict) arguments.push_back(L"-Gis");
    for (auto value:definitions) if (value) { arguments.push_back(L"-D"); arguments.push_back(value); }
    return CompileCachedDXIL(create,input,arguments.data(),UINT32(arguments.size()),output.ReleaseAndGetAddressOf(),LastError);
}
