#pragma once
#if defined(_WIN32)
#include "Renderer/D3DShaderCache.h"
#include <dxcapi.h>
#include <string>
namespace Evans::Renderer
{
    bool CompileCachedDXIL(DxcCreateInstanceProc create,const DxcBuffer& source,
        const wchar_t** arguments,UINT32 count,IDxcBlob** output,std::string& error);
}
#endif
