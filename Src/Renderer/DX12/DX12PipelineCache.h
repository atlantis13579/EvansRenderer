#pragma once
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <wrl/client.h>
namespace Evans::Renderer
{
    HRESULT CreateCachedRootSignature(ID3D12Device* device,UINT nodeMask,const void* bytes,SIZE_T size,REFIID iid,void** result);
    HRESULT CreateCachedGraphicsPipelineState(ID3D12Device* device,const D3D12_GRAPHICS_PIPELINE_STATE_DESC* description,REFIID iid,void** result);
    HRESULT CreateCachedComputePipelineState(ID3D12Device* device,const D3D12_COMPUTE_PIPELINE_STATE_DESC* description,REFIID iid,void** result);
    HRESULT CreateCachedMeshPipelineState(ID3D12Device2* device,const D3D12_PIPELINE_STATE_STREAM_DESC& stream,
        D3D12_CACHED_PIPELINE_STATE& cached,const D3D12_GRAPHICS_PIPELINE_STATE_DESC& state,
        const D3D12_SHADER_BYTECODE& mesh,const D3D12_SHADER_BYTECODE& task,REFIID iid,void** result);
    template<class T,class... Args> HRESULT CreateCachedRootSignature(const Microsoft::WRL::ComPtr<T>& device,Args&&... args)
    { return CreateCachedRootSignature(device.Get(),static_cast<Args&&>(args)...); }
    template<class T,class... Args> HRESULT CreateCachedGraphicsPipelineState(const Microsoft::WRL::ComPtr<T>& device,Args&&... args)
    { return CreateCachedGraphicsPipelineState(device.Get(),static_cast<Args&&>(args)...); }
    template<class T,class... Args> HRESULT CreateCachedComputePipelineState(const Microsoft::WRL::ComPtr<T>& device,Args&&... args)
    { return CreateCachedComputePipelineState(device.Get(),static_cast<Args&&>(args)...); }
}
#endif
