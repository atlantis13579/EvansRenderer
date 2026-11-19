#pragma once
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3dcompiler.h>
namespace Evans::Renderer
{
    HRESULT CompileCachedD3DShader(LPCVOID source,SIZE_T size,LPCSTR name,const D3D_SHADER_MACRO* definitions,
        ID3DInclude* include,LPCSTR entry,LPCSTR target,UINT flags1,UINT flags2,ID3DBlob** code,ID3DBlob** errors);
}
#endif
