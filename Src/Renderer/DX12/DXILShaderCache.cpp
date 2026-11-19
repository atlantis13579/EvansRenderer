#include "Renderer/DX12/DXILShaderCache.h"
#include "Renderer/ShaderCache.h"
#if defined(_WIN32)
#include <wrl/client.h>
#include <map>
#include <mutex>
namespace Evans::Renderer
{
    bool CompileCachedDXIL(DxcCreateInstanceProc create,const DxcBuffer& source,
        const wchar_t** arguments,UINT32 count,IDxcBlob** output,std::string& error)
    {
        using Microsoft::WRL::ComPtr;
        HMODULE module=nullptr;wchar_t path[32768]{};std::string identity;
        if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(create),&module) && GetModuleFileNameW(module,path,32768))
        {
            static std::mutex mutex;static std::map<std::wstring,std::string> identities;
            std::lock_guard<std::mutex> lock(mutex);auto& entry=identities[path];
            if(entry.empty())
            {
                const auto compiler=ShaderCache::FileIdentity(path);
                if(!compiler.empty())entry=compiler+":"+ShaderCache::FileIdentity(std::filesystem::path(path).parent_path()/"dxil.dll");
            }
            identity=entry;
        }
        ShaderCache::Key key("dxil-shader-v1");key.Add(identity).Add(source.Ptr,source.Size).Value(source.Encoding).Value(count);
        for(UINT32 i=0;i<count;++i)key.Add(arguments[i],std::wcslen(arguments[i])*sizeof(wchar_t));
        const auto digest=key.Finish();std::vector<std::uint8_t> data;
        ComPtr<IDxcUtils> utils;auto status=create(CLSID_DxcUtils,IID_PPV_ARGS(&utils));
        if(FAILED(status)){error="Cannot create DXC utilities: "+std::to_string(status);return false;}
        if(!identity.empty() && ShaderCache::Read("dxil-shader",digest,data))
        {
            ComPtr<IDxcBlobEncoding> blob;
            if(SUCCEEDED(utils->CreateBlob(data.data(),UINT32(data.size()),DXC_CP_ACP,&blob)) &&
                SUCCEEDED(blob->QueryInterface(__uuidof(IDxcBlob),reinterpret_cast<void**>(output))))return true;
        }
        ComPtr<IDxcCompiler3> compiler;status=create(CLSID_DxcCompiler,IID_PPV_ARGS(&compiler));
        if(FAILED(status)){error="Cannot create DXC compiler: "+std::to_string(status);return false;}
        ComPtr<IDxcResult> result;
        status=compiler->Compile(&source,arguments,count,nullptr,IID_PPV_ARGS(&result));
        if(FAILED(status)){error="DXC invocation failed: "+std::to_string(status);return false;}
        result->GetStatus(&status);
        if(FAILED(status))
        {
            ComPtr<IDxcBlobUtf8> messages;result->GetOutput(DXC_OUT_ERRORS,IID_PPV_ARGS(&messages),nullptr);
            error=messages?messages->GetStringPointer():"DXIL compilation failed";return false;
        }
        status=result->GetOutput(DXC_OUT_OBJECT,__uuidof(IDxcBlob),reinterpret_cast<void**>(output),nullptr);
        if(FAILED(status) || !*output){error="DXC object retrieval failed: "+std::to_string(status);return false;}
        if(!identity.empty())ShaderCache::Write("dxil-shader",digest,(*output)->GetBufferPointer(),(*output)->GetBufferSize());
        return true;
    }
}
#endif
