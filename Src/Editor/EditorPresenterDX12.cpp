#include "Renderer/Types.h"
#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#include "Editor/EditorPresenterDX12.h"

#if defined(_WIN32)

#include "Editor/ImguiRenderDX12.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace Evans::Editor
{
namespace
{

constexpr UINT FrameCount = 2;
constexpr DXGI_FORMAT BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;

template <class T>
void SafeRelease(T*& value) noexcept
{
    if (value != nullptr)
    {
        value->Release();
        value = nullptr;
    }
}

std::string HResultMessage(const char* operation, HRESULT result)
{
    std::ostringstream stream;
    stream << operation << " failed with HRESULT 0x"
           << std::hex << std::uppercase << static_cast<unsigned long>(result);
    return stream.str();
}

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept
{
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC BufferDescription(UINT64 size) noexcept
{
    D3D12_RESOURCE_DESC description = {};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return description;
}

} // namespace

struct EditorPresenterDX12::Impl
{
    struct FrameResources
    {
        ID3D12CommandAllocator* Allocator = nullptr;
        UINT64 FenceValue = 0;
        std::vector<IUnknown*> DeferredReleases;
        std::vector<std::shared_ptr<const Renderer::GpuImage>> GpuLeases;
    };

    ~Impl()
    {
        Shutdown();
    }

    bool Fail(const std::string& message)
    {
        LastError = message;
        return false;
    }

    bool FailDeviceOperation(const char* operation, HRESULT result)
    {
        std::string message = HResultMessage(operation, result);
        if (Device != nullptr &&
            (result == DXGI_ERROR_DEVICE_REMOVED ||
             result == DXGI_ERROR_DEVICE_RESET))
        {
            message += "; removal reason ";
            message += HResultMessage(
                "ID3D12Device::GetDeviceRemovedReason",
                Device->GetDeviceRemovedReason());
        }
        return Fail(message);
    }

    void DeferRelease(IUnknown* resource)
    {
        if (resource != nullptr)
        {
            Frames[CurrentFrameIndex].DeferredReleases.push_back(resource);
        }
    }

    static void ReleaseDeferred(FrameResources& frame) noexcept
    {
        for (IUnknown* resource : frame.DeferredReleases)
        {
            if (resource != nullptr)
            {
                resource->Release();
            }
        }
        frame.DeferredReleases.clear();
        frame.GpuLeases.clear();
    }

    bool CreateDevice(IDXGIFactory4* factory)
    {
        IDXGIAdapter1* adapter = nullptr;
        for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index)
        {
            DXGI_ADAPTER_DESC1 description = {};
            adapter->GetDesc1(&description);
            if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
            {
                ID3D12Device* device = nullptr;
                const HRESULT result = D3D12CreateDevice(
                    adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
                if (SUCCEEDED(result))
                {
                    Device = device;
                    SafeRelease(adapter);
                    return true;
                }
            }
            SafeRelease(adapter);
        }

        IDXGIAdapter* warpAdapter = nullptr;
        HRESULT result = factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
        if (SUCCEEDED(result))
        {
            result = D3D12CreateDevice(
                warpAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&Device));
        }
        SafeRelease(warpAdapter);
        return SUCCEEDED(result) || Fail(HResultMessage("D3D12CreateDevice", result));
    }

    bool Initialize(HWND window, int width, int height)
    {
        Shutdown();
        if (window == nullptr || width <= 0 || height <= 0)
        {
            return Fail("The DX12 editor presenter requires a valid window and dimensions.");
        }

        IDXGIFactory4* factory = nullptr;
        HRESULT result = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        if (FAILED(result))
        {
            return Fail(HResultMessage("CreateDXGIFactory2", result));
        }
        if (!CreateDevice(factory))
        {
            SafeRelease(factory);
            return false;
        }

        D3D12_COMMAND_QUEUE_DESC queueDescription = {};
        queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDescription.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        result = Device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&CommandQueue));
        if (FAILED(result))
        {
            SafeRelease(factory);
            return Fail(HResultMessage("CreateCommandQueue", result));
        }

        BOOL tearing = FALSE;
        IDXGIFactory5* factory5 = nullptr;
        if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory5))))
        {
            AllowTearing = SUCCEEDED(factory5->CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))) && tearing;
            SafeRelease(factory5);
        }

        DXGI_SWAP_CHAIN_DESC1 swapChainDescription = {};
        swapChainDescription.Flags = AllowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
        swapChainDescription.Width = static_cast<UINT>(width);
        swapChainDescription.Height = static_cast<UINT>(height);
        swapChainDescription.Format = BackBufferFormat;
        swapChainDescription.SampleDesc.Count = 1;
        swapChainDescription.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDescription.BufferCount = FrameCount;
        swapChainDescription.Scaling = DXGI_SCALING_STRETCH;
        swapChainDescription.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDescription.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

        IDXGISwapChain1* swapChain = nullptr;
        result = factory->CreateSwapChainForHwnd(
            CommandQueue, window, &swapChainDescription, nullptr, nullptr, &swapChain);
        if (SUCCEEDED(result))
        {
            result = swapChain->QueryInterface(IID_PPV_ARGS(&SwapChain));
        }
        if (SUCCEEDED(result))
        {
            result = factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
        }
        SafeRelease(swapChain);
        SafeRelease(factory);
        if (FAILED(result))
        {
            return Fail(HResultMessage("CreateSwapChainForHwnd", result));
        }

        D3D12_DESCRIPTOR_HEAP_DESC renderTargetHeapDescription = {};
        renderTargetHeapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        renderTargetHeapDescription.NumDescriptors = FrameCount;
        result = Device->CreateDescriptorHeap(
            &renderTargetHeapDescription, IID_PPV_ARGS(&RenderTargetHeap));
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create render target descriptor heap", result));
        }
        RenderTargetDescriptorSize = Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        for (FrameResources& frame : Frames)
        {
            result = Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.Allocator));
            if (FAILED(result))
            {
                return Fail(HResultMessage("CreateCommandAllocator", result));
            }
        }
        result = Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            Frames[0].Allocator,
            nullptr,
            IID_PPV_ARGS(&CommandList));
        if (FAILED(result))
        {
            return Fail(HResultMessage("CreateCommandList", result));
        }
        result = Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence));
        if (FAILED(result))
        {
            return Fail(HResultMessage("CreateFence", result));
        }
        FenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (FenceEvent == nullptr)
        {
            return Fail("CreateEventW failed for the DX12 fence.");
        }

        Window = window;
        Width = width;
        Height = height;
        CurrentFrameIndex = SwapChain->GetCurrentBackBufferIndex();
        if (!CreateBackBuffers() || !CreateViewportPipeline())
        {
            return false;
        }
        if (!ImguiRenderer.Init(Device, CommandList, BackBufferFormat, FrameCount))
        {
            return Fail("The DX12 Riemann-style ImGui renderer could not initialize.");
        }
        result = CommandList->Close();
        if (FAILED(result))
        {
            return Fail(HResultMessage("Close initialization command list", result));
        }
        ID3D12CommandList* initializationLists[] = {CommandList};
        CommandQueue->ExecuteCommandLists(ARRAYSIZE(initializationLists), initializationLists);
        if (!WaitForGpu())
        {
            return false;
        }
        LastError.clear();
        return true;
    }

    void Shutdown() noexcept
    {
        if (CommandList != nullptr && FrameRecording)
        {
            (void)CommandList->Close();
            FrameRecording = false;
        }
        if (CommandQueue != nullptr && Fence != nullptr && FenceEvent != nullptr)
        {
            (void)WaitForGpu();
        }
        ImguiRenderer.Shutdown();
        for (FrameResources& frame : Frames)
        {
            ReleaseDeferred(frame);
        }
        SafeRelease(OverlayTexture);SafeRelease(OverlayHeap);SafeRelease(OverlayPipelineState);
        OverlayWidth=OverlayHeight=0;OverlayState=D3D12_RESOURCE_STATE_COMMON;OverlayRevision=std::numeric_limits<std::uint64_t>::max();
        SafeRelease(ImageTexture);
        SafeRelease(ViewportTextureHeap);
        SafeRelease(ViewportPipelineState);
        SafeRelease(ViewportRootSignature);
        for (ID3D12Resource*& backBuffer : BackBuffers)
        {
            SafeRelease(backBuffer);
        }
        for (FrameResources& frame : Frames)
        {
            SafeRelease(frame.Allocator);
            frame.FenceValue = 0;
        }
        SafeRelease(RenderTargetHeap);
        SafeRelease(CommandList);
        SafeRelease(Fence);
        if (FenceEvent != nullptr)
        {
            CloseHandle(FenceEvent);
            FenceEvent = nullptr;
        }
        SafeRelease(SwapChain);
        AllowTearing = false;
        SafeRelease(CommandQueue);
        SafeRelease(Device);
        Window = nullptr;
        Width = 0;
        Height = 0;
        CurrentFrameIndex = 0;
        NextFenceValue = 0;
        FrameRecording = false;
        ImageWidth = 0;
        ImageHeight = 0;
        ImageState = D3D12_RESOURCE_STATE_COMMON;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    }

    bool WaitForFence(UINT64 value)
    {
        const UINT64 completed = Fence->GetCompletedValue();
        if (completed == std::numeric_limits<UINT64>::max())
            return FailDeviceOperation("DX12 presentation fence", DXGI_ERROR_DEVICE_REMOVED);
        if (value == 0 || completed >= value)
        {
            return true;
        }
        HRESULT result = Fence->SetEventOnCompletion(value, FenceEvent);
        if (FAILED(result))
        {
            return Fail(HResultMessage("ID3D12Fence::SetEventOnCompletion", result));
        }
        const DWORD waitResult =
            WaitForSingleObject(FenceEvent, GpuWaitTimeoutMilliseconds);
        if (waitResult == WAIT_OBJECT_0)
        {
            const UINT64 completedAfterWait = Fence->GetCompletedValue();
            if (completedAfterWait == std::numeric_limits<UINT64>::max())
                return FailDeviceOperation("DX12 presentation fence", DXGI_ERROR_DEVICE_REMOVED);
            return completedAfterWait >= value ||
                Fail("The DX12 presentation fence event signaled before completion.");
        }
        const HRESULT removedReason = Device->GetDeviceRemovedReason();
        return FAILED(removedReason)
            ? Fail(HResultMessage("DX12 presentation device removal", removedReason))
            : Fail(waitResult == WAIT_TIMEOUT
                ? "Waiting for the DX12 presentation fence timed out."
                : "Waiting for the DX12 presentation fence failed.");
    }

    bool WaitForGpu()
    {
        if (CommandQueue == nullptr || Fence == nullptr)
        {
            return true;
        }
        const UINT64 value = ++NextFenceValue;
        const HRESULT result = CommandQueue->Signal(Fence, value);
        return SUCCEEDED(result)
            ? WaitForFence(value)
            : Fail(HResultMessage("ID3D12CommandQueue::Signal", result));
    }

    bool SignalFrame(FrameResources& frame)
    {
        const UINT64 value = ++NextFenceValue;
        const HRESULT result = CommandQueue->Signal(Fence, value);
        if (FAILED(result))
        {
            return Fail(HResultMessage("ID3D12CommandQueue::Signal", result));
        }
        frame.FenceValue = value;
        return true;
    }

    bool BeginRecording()
    {
        FrameResources& frame = Frames[CurrentFrameIndex];
        if (!WaitForFence(frame.FenceValue))
        {
            return false;
        }
        ReleaseDeferred(frame);
        HRESULT result = frame.Allocator->Reset();
        if (SUCCEEDED(result))
        {
            result = CommandList->Reset(frame.Allocator, nullptr);
        }
        if (FAILED(result))
        {
            return Fail(HResultMessage("Reset DX12 frame commands", result));
        }
        FrameRecording = true;
        return true;
    }

    bool ExecuteRecording()
    {
        if (!FrameRecording)
        {
            return false;
        }
        HRESULT result = CommandList->Close();
        if (FAILED(result))
        {
            return Fail(HResultMessage("ID3D12GraphicsCommandList::Close", result));
        }
        ID3D12CommandList* lists[] = {CommandList};
        CommandQueue->ExecuteCommandLists(ARRAYSIZE(lists), lists);
        FrameRecording = false;
        return SignalFrame(Frames[CurrentFrameIndex]);
    }

    bool CreateBackBuffers()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = RenderTargetHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT index = 0; index < FrameCount; ++index)
        {
            HRESULT result = SwapChain->GetBuffer(index, IID_PPV_ARGS(&BackBuffers[index]));
            if (FAILED(result))
            {
                return Fail(HResultMessage("IDXGISwapChain3::GetBuffer", result));
            }
            Device->CreateRenderTargetView(BackBuffers[index], nullptr, handle);
            handle.ptr += RenderTargetDescriptorSize;
        }
        return true;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE CurrentRenderTargetHandle() const noexcept
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = RenderTargetHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(CurrentFrameIndex) * RenderTargetDescriptorSize;
        return handle;
    }

    static void Transition(
        ID3D12GraphicsCommandList* commands,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
    {
        if (before == after)
        {
            return;
        }
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        commands->ResourceBarrier(1, &barrier);
    }

    bool CreateViewportPipeline()
    {
        D3D12_DESCRIPTOR_RANGE textureRange = {};
        textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        textureRange.NumDescriptors = 1;
        textureRange.BaseShaderRegister = 0;
        textureRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_ROOT_PARAMETER parameter = {};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable.NumDescriptorRanges = 1;
        parameter.DescriptorTable.pDescriptorRanges = &textureRange;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_STATIC_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderRegister = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rootDescription = {};
        rootDescription.NumParameters = 1;
        rootDescription.pParameters = &parameter;
        rootDescription.NumStaticSamplers = 1;
        rootDescription.pStaticSamplers = &sampler;
        rootDescription.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                                D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                                D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                                D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
        ID3DBlob* signature = nullptr;
        ID3DBlob* error = nullptr;
        HRESULT result = D3D12SerializeRootSignature(
            &rootDescription, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(result))
        {
            if (error != nullptr)
            {
                LastError = static_cast<const char*>(error->GetBufferPointer());
            }
            SafeRelease(error);
            SafeRelease(signature);
            return false;
        }
        SafeRelease(error);
        result = Evans::Renderer::CreateCachedRootSignature(Device,
            0, signature->GetBufferPointer(), signature->GetBufferSize(),
            IID_PPV_ARGS(&ViewportRootSignature));
        SafeRelease(signature);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport root signature", result));
        }

        static constexpr const char* ShaderSource = R"HLSL(
Texture2D ViewportTexture : register(t0);
SamplerState ViewportSampler : register(s0);
struct VSOutput { float4 Position : SV_POSITION; float2 TexCoord : TEXCOORD0; };
VSOutput VS(uint vertex : SV_VertexID)
{
    VSOutput output;
    output.TexCoord = float2((vertex << 1) & 2, vertex & 2);
    output.Position = float4(output.TexCoord * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float4 PS(VSOutput input) : SV_TARGET
{
    return ViewportTexture.Sample(ViewportSampler, input.TexCoord);
}
)HLSL";
        ID3DBlob* vertexShader = nullptr;
        ID3DBlob* pixelShader = nullptr;
        result = Evans::Renderer::CompileCachedD3DShader(
            ShaderSource, std::strlen(ShaderSource), "EvansEditorDX12Viewport",
            nullptr, nullptr, "VS", "vs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
            &vertexShader, &error);
        if (FAILED(result))
        {
            if (error != nullptr)
            {
                LastError = static_cast<const char*>(error->GetBufferPointer());
            }
            SafeRelease(error);
            return false;
        }
        SafeRelease(error);
        result = Evans::Renderer::CompileCachedD3DShader(
            ShaderSource, std::strlen(ShaderSource), "EvansEditorDX12Viewport",
            nullptr, nullptr, "PS", "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
            &pixelShader, &error);
        if (FAILED(result))
        {
            if (error != nullptr)
            {
                LastError = static_cast<const char*>(error->GetBufferPointer());
            }
            SafeRelease(error);
            SafeRelease(vertexShader);
            return false;
        }
        SafeRelease(error);

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline = {};
        pipeline.pRootSignature = ViewportRootSignature;
        pipeline.VS = {vertexShader->GetBufferPointer(), vertexShader->GetBufferSize()};
        pipeline.PS = {pixelShader->GetBufferPointer(), pixelShader->GetBufferSize()};
        pipeline.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pipeline.SampleMask = UINT_MAX;
        pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pipeline.RasterizerState.DepthClipEnable = TRUE;
        pipeline.DepthStencilState.DepthEnable = FALSE;
        pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets = 1;
        pipeline.RTVFormats[0] = BackBufferFormat;
        pipeline.SampleDesc.Count = 1;
        result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(&ViewportPipelineState));
        if(SUCCEEDED(result))
        {
            auto& blend=pipeline.BlendState.RenderTarget[0];blend.BlendEnable=TRUE;
            blend.SrcBlend=D3D12_BLEND_SRC_ALPHA;blend.DestBlend=D3D12_BLEND_INV_SRC_ALPHA;blend.BlendOp=D3D12_BLEND_OP_ADD;
            blend.SrcBlendAlpha=D3D12_BLEND_ONE;blend.DestBlendAlpha=D3D12_BLEND_INV_SRC_ALPHA;blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;
            result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(&OverlayPipelineState));
        }
        SafeRelease(pixelShader);
        SafeRelease(vertexShader);
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create viewport pipeline state", result));
        }

        D3D12_DESCRIPTOR_HEAP_DESC heap = {};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 1;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        return SUCCEEDED(Device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&ViewportTextureHeap))) ||
               Fail("Create viewport texture descriptor heap failed.");
    }

    bool Resize(int width, int height)
    {
        if (SwapChain == nullptr || width <= 0 || height <= 0)
        {
            return false;
        }
        if (width == Width && height == Height)
        {
            return true;
        }
        if (FrameRecording)
        {
            (void)CommandList->Close();
            FrameRecording = false;
        }
        if (!WaitForGpu())
        {
            return false;
        }
        for (FrameResources& frame : Frames)
        {
            ReleaseDeferred(frame);
            frame.FenceValue = 0;
        }
        for (ID3D12Resource*& backBuffer : BackBuffers)
        {
            SafeRelease(backBuffer);
        }
        const HRESULT result = SwapChain->ResizeBuffers(
            FrameCount,
            static_cast<UINT>(width),
            static_cast<UINT>(height),
            BackBufferFormat,
            AllowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
        if (FAILED(result))
        {
            return Fail(HResultMessage("IDXGISwapChain3::ResizeBuffers", result));
        }
        Width = width;
        Height = height;
        CurrentFrameIndex = SwapChain->GetCurrentBackBufferIndex();
        return CreateBackBuffers();
    }

    bool CreateImageResources(const Image& image)
    {
        ID3D12Resource* texture = nullptr;
        ID3D12DescriptorHeap* heap = nullptr;
        D3D12_RESOURCE_DESC textureDescription = {};
        textureDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDescription.Width = image.Width;
        textureDescription.Height = image.Height;
        textureDescription.DepthOrArraySize = 1;
        textureDescription.MipLevels = 1;
        textureDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDescription.SampleDesc.Count = 1;
        textureDescription.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        const D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
        HRESULT result = Device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &textureDescription,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&texture));
        if (FAILED(result))
        {
            const bool failed = FailDeviceOperation(
                "Create DX12 viewport texture",
                result);
            LastError += " for " + std::to_string(image.Width) + "x" +
                         std::to_string(image.Height);
            return failed;
        }
        D3D12_DESCRIPTOR_HEAP_DESC heapDescription = {};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = 1;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        result = Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&heap));
        if (FAILED(result))
        {
            SafeRelease(texture);
            return Fail(HResultMessage("Create DX12 viewport descriptor heap", result));
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC view = {};
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        Device->CreateShaderResourceView(texture, &view, heap->GetCPUDescriptorHandleForHeapStart());

        DeferRelease(ImageTexture);
        DeferRelease(ViewportTextureHeap);
        ImageTexture = texture;
        ViewportTextureHeap = heap;
        ImageWidth = image.Width;
        ImageHeight = image.Height;
        ImageState = D3D12_RESOURCE_STATE_COPY_DEST;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
        return true;
    }

    bool UploadImage(const Image& image, std::uint64_t revision)
    {
        if (ImageTexture == nullptr || ImageWidth != image.Width || ImageHeight != image.Height)
        {
            if (!CreateImageResources(image))
            {
                return false;
            }
        }
        if (UploadedRevision == revision)
        {
            return true;
        }

        if (image.GpuColor)
        {
            if (image.GpuColor->Backend!=Renderer::RHIBackend::DX12) return Fail("Incompatible GPU viewport backend");
            ID3D12Resource* source=nullptr;
            auto status=Device->OpenSharedHandle(image.GpuColor->SharedHandle,IID_PPV_ARGS(&source));
            if (FAILED(status)) return FailDeviceOperation("Import DX12 viewport",status);
            Transition(CommandList,source,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(CommandList,ImageTexture,ImageState,D3D12_RESOURCE_STATE_COPY_DEST);
            CommandList->CopyResource(ImageTexture,source);
            Transition(CommandList,source,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
            Transition(CommandList,ImageTexture,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            ImageState=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            Frames[CurrentFrameIndex].GpuLeases.push_back(image.GpuColor);
            DeferRelease(source); UploadedRevision=revision; return true;
        }
        const D3D12_RESOURCE_DESC textureDescription = ImageTexture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT64 uploadSize = 0;
        Device->GetCopyableFootprints(
            &textureDescription, 0, 1, 0, &footprint, nullptr, nullptr, &uploadSize);
        const D3D12_HEAP_PROPERTIES uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_RESOURCE_DESC uploadDescription = BufferDescription(uploadSize);
        ID3D12Resource* upload = nullptr;
        HRESULT result = Device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &uploadDescription,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&upload));
        if (FAILED(result))
        {
            return Fail(HResultMessage("Create DX12 viewport upload buffer", result));
        }
        std::uint8_t* mapped = nullptr;
        const D3D12_RANGE readRange = {0, 0};
        result = upload->Map(0, &readRange, reinterpret_cast<void**>(&mapped));
        if (FAILED(result))
        {
            SafeRelease(upload);
            return Fail(HResultMessage("Map DX12 viewport upload buffer", result));
        }
        const std::size_t sourcePitch = static_cast<std::size_t>(image.Width) * 4u;
        for (std::uint32_t row = 0; row < image.Height; ++row)
        {
            std::memcpy(
                mapped + footprint.Offset + static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
                image.PixelsRGBA8.data() + static_cast<std::size_t>(row) * sourcePitch,
                sourcePitch);
        }
        upload->Unmap(0, nullptr);

        Transition(CommandList, ImageTexture, ImageState, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = upload;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION destination = {};
        destination.pResource = ImageTexture;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        Transition(
            CommandList,
            ImageTexture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        ImageState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        UploadedRevision = revision;
        DeferRelease(upload);
        return true;
    }

    void DrawViewport(int x, int y, int width, int height, bool overlay = false)
    {
        if (ImageTexture == nullptr || ViewportTextureHeap == nullptr || width <= 0 || height <= 0)
        {
            return;
        }
        const D3D12_VIEWPORT viewport = {
            static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
        const D3D12_RECT scissor = {x, y, x + width, y + height};
        CommandList->RSSetViewports(1, &viewport);
        CommandList->RSSetScissorRects(1, &scissor);
        CommandList->SetPipelineState(overlay?OverlayPipelineState:ViewportPipelineState);
        CommandList->SetGraphicsRootSignature(ViewportRootSignature);
        ID3D12DescriptorHeap* heaps[] = {ViewportTextureHeap};
        CommandList->SetDescriptorHeaps(ARRAYSIZE(heaps), heaps);
        CommandList->SetGraphicsRootDescriptorTable(
            0, ViewportTextureHeap->GetGPUDescriptorHandleForHeapStart());
        CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        CommandList->DrawInstanced(3, 1, 0, 0);
    }

    bool BeginFrame(
        const Image& image,
        std::uint64_t revision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight)
    {
        if (Device == nullptr || SwapChain == nullptr)
        {
            return Fail("The DX12 editor presenter is not initialized.");
        }
        if (FrameRecording)
        {
            (void)CommandList->Close();
            FrameRecording = false;
        }
        if (!BeginRecording())
        {
            return false;
        }
        ID3D12Resource* backBuffer = BackBuffers[CurrentFrameIndex];
        Transition(CommandList, backBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        const D3D12_CPU_DESCRIPTOR_HANDLE renderTarget = CurrentRenderTargetHandle();
        CommandList->OMSetRenderTargets(1, &renderTarget, FALSE, nullptr);
        constexpr float clearColor[4] = {0.035f, 0.043f, 0.059f, 1.0f};
        CommandList->ClearRenderTargetView(renderTarget, clearColor, 0, nullptr);
        if (image.IsValid())
        {
            if (!UploadImage(image, revision))
            {
                return false;
            }
            DrawViewport(viewportX, viewportY, viewportWidth, viewportHeight);
            if(image.Overlay && image.Overlay->IsValid())
            {
                const auto swap=[&]() {
                    std::swap(ImageTexture,OverlayTexture);std::swap(ViewportTextureHeap,OverlayHeap);
                    std::swap(ImageState,OverlayState);std::swap(ImageWidth,OverlayWidth);std::swap(ImageHeight,OverlayHeight);
                    std::swap(UploadedRevision,OverlayRevision);
                };
                swap();const bool uploaded=UploadImage(*image.Overlay,revision);
                if(uploaded)DrawViewport(viewportX,viewportY,viewportWidth,viewportHeight,true);
                swap();if(!uploaded)return false;
            }
        }
        return true;
    }

    void DrawImgui()
    {
        if (!FrameRecording)
        {
            return;
        }
        const D3D12_CPU_DESCRIPTOR_HANDLE renderTarget = CurrentRenderTargetHandle();
        CommandList->OMSetRenderTargets(1, &renderTarget, FALSE, nullptr);
        ImguiRenderer.Draw(CommandList, Width, Height, CurrentFrameIndex);
    }

    bool EndFrame()
    {
        if (SwapChain == nullptr)
        {
            return false;
        }
        if (!FrameRecording) return Fail("The DX12 editor frame has not been recorded.");
        Transition(CommandList, BackBuffers[CurrentFrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT);
        if (!ExecuteRecording()) return false;
        const HRESULT result = SwapChain->Present(0, AllowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (FAILED(result))
        {
            return Fail(HResultMessage("IDXGISwapChain3::Present", result));
        }
        CurrentFrameIndex = SwapChain->GetCurrentBackBufferIndex();
        return true;
    }

    HWND Window = nullptr;
    int Width = 0;
    int Height = 0;
    ID3D12Device* Device = nullptr;
    ID3D12CommandQueue* CommandQueue = nullptr;
    IDXGISwapChain3* SwapChain = nullptr;
    bool AllowTearing = false;
    ID3D12DescriptorHeap* RenderTargetHeap = nullptr;
    std::array<ID3D12Resource*, FrameCount> BackBuffers = {};
    std::array<FrameResources, FrameCount> Frames = {};
    ID3D12GraphicsCommandList* CommandList = nullptr;
    ID3D12Fence* Fence = nullptr;
    HANDLE FenceEvent = nullptr;
    UINT RenderTargetDescriptorSize = 0;
    UINT CurrentFrameIndex = 0;
    UINT64 NextFenceValue = 0;
    bool FrameRecording = false;

    ID3D12RootSignature* ViewportRootSignature = nullptr;
    ID3D12PipelineState* ViewportPipelineState = nullptr;
    ID3D12DescriptorHeap* ViewportTextureHeap = nullptr;
    ID3D12Resource* ImageTexture = nullptr;
    D3D12_RESOURCE_STATES ImageState = D3D12_RESOURCE_STATE_COMMON;
    std::uint32_t ImageWidth = 0;
    std::uint32_t ImageHeight = 0;
    std::uint64_t UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    ID3D12Resource* OverlayTexture=nullptr;
    ID3D12DescriptorHeap* OverlayHeap=nullptr;
    ID3D12PipelineState* OverlayPipelineState=nullptr;
    D3D12_RESOURCE_STATES OverlayState=D3D12_RESOURCE_STATE_COMMON;
    std::uint32_t OverlayWidth=0,OverlayHeight=0;
    std::uint64_t OverlayRevision=std::numeric_limits<std::uint64_t>::max();
    ImguiRenderDX12 ImguiRenderer;
    std::string LastError;
};

EditorPresenterDX12::EditorPresenterDX12()
    : Impl_(std::make_unique<Impl>())
{
}

EditorPresenterDX12::~EditorPresenterDX12() = default;

bool EditorPresenterDX12::Initialize(void* nativeWindow, int width, int height)
{
    return Impl_->Initialize(static_cast<HWND>(nativeWindow), width, height);
}

void EditorPresenterDX12::Shutdown() noexcept
{
    Impl_->Shutdown();
}

bool EditorPresenterDX12::Resize(int width, int height)
{
    return Impl_->Resize(width, height);
}

bool EditorPresenterDX12::BeginFrame(
    const Image& viewportImage,
    std::uint64_t viewportRevision,
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight)
{
    return Impl_->BeginFrame(
        viewportImage,
        viewportRevision,
        viewportX,
        viewportY,
        viewportWidth,
        viewportHeight);
}

void EditorPresenterDX12::DrawImgui()
{
    Impl_->DrawImgui();
}

bool EditorPresenterDX12::EndFrame()
{
    return Impl_->EndFrame();
}

bool EditorPresenterDX12::IsInitialized() const noexcept
{
    return Impl_->Device != nullptr && Impl_->SwapChain != nullptr && Impl_->RenderTargetHeap != nullptr;
}

const std::string& EditorPresenterDX12::LastError() const noexcept
{
    return Impl_->LastError;
}

} // namespace Evans::Editor

#endif
