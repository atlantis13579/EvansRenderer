#include "Renderer/D3DShaderCache.h"
#include "Renderer/ShaderCache.h"
#if defined(_WIN32)
#include <cstring>
#include <wrl/client.h>
namespace Evans::Renderer
{
    namespace
    {
        std::string CompilerIdentity()
        {
            // D3DCompile's address may be a linker-generated import thunk in
            // the executable. Hash the loaded compiler DLL, not that thunk's
            // module, so unrelated executable rebuilds preserve the cache.
            const auto module=GetModuleHandleW(D3DCOMPILER_DLL_W);wchar_t path[32768]{};
            if(!module || !GetProcAddress(module,"D3DCompile") || !GetModuleFileNameW(module,path,32768))return {};
            return ShaderCache::FileIdentity(path);
        }
    }
    HRESULT CompileCachedD3DShader(LPCVOID source,SIZE_T size,LPCSTR name,const D3D_SHADER_MACRO* definitions,
        ID3DInclude* include,LPCSTR entry,LPCSTR target,UINT flags1,UINT flags2,ID3DBlob** code,ID3DBlob** errors)
    {
        static const auto identity=CompilerIdentity();
        // Include handlers may resolve arbitrary mutable content. Until their
        // dependencies are supplied explicitly, preserve the compiler contract.
        if(include || !source || !code || identity.empty())
            return D3DCompile(source,size,name,definitions,include,entry,target,flags1,flags2,code,errors);
        ShaderCache::Key key("d3d-shader-v1");key.Add(identity).Add(source,size).Add(name?name:"").Add(entry?entry:"").Add(target?target:"").Value(flags1).Value(flags2);
        if(definitions)for(auto macro=definitions;macro->Name;++macro)
            key.Add(macro->Name).Value(macro->Definition!=nullptr).Add(macro->Definition?macro->Definition:"");
        const auto digest=key.Finish();std::vector<std::uint8_t> artifact;
        if(ShaderCache::Read("d3d-shader",digest,artifact) && artifact.size()>=8)
        {
            std::uint64_t codeSize=0;for(unsigned i=0;i<8;++i)codeSize|=std::uint64_t(artifact[i])<<(8*i);
            if(codeSize && codeSize<=artifact.size()-8)
            {
                Microsoft::WRL::ComPtr<ID3DBlob> binary,messages;
                const auto messageSize=artifact.size()-8-std::size_t(codeSize);
                if(SUCCEEDED(D3DCreateBlob(SIZE_T(codeSize),&binary)) && (!errors || !messageSize || SUCCEEDED(D3DCreateBlob(messageSize,&messages))))
                {
                    std::memcpy(binary->GetBufferPointer(),artifact.data()+8,SIZE_T(codeSize));
                    if(messages)std::memcpy(messages->GetBufferPointer(),artifact.data()+8+codeSize,messageSize);
                    *code=binary.Detach();if(errors)*errors=messages.Detach();return S_OK;
                }
            }
        }
        Microsoft::WRL::ComPtr<ID3DBlob> binary,messages;
        const auto result=D3DCompile(source,size,name,definitions,include,entry,target,flags1,flags2,&binary,&messages);
        if(SUCCEEDED(result) && binary)
        {
            const auto codeSize=binary->GetBufferSize(),messageSize=messages?messages->GetBufferSize():0;
            artifact.resize(8+codeSize+messageSize);
            for(unsigned i=0;i<8;++i)artifact[i]=std::uint8_t(std::uint64_t(codeSize)>>(8*i));
            std::memcpy(artifact.data()+8,binary->GetBufferPointer(),codeSize);
            if(messageSize)std::memcpy(artifact.data()+8+codeSize,messages->GetBufferPointer(),messageSize);
            ShaderCache::Write("d3d-shader",digest,artifact.data(),artifact.size());
        }
        *code=binary.Detach();if(errors)*errors=messages.Detach();return result;
    }
}
#endif
