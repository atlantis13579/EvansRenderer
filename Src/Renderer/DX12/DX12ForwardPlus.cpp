#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#include "Renderer/DX12/DX12ForwardPlus.h"
#if defined(_WIN32)
#include "Renderer/TechniqueShader.h"

#include <cstring>
#include <d3dcompiler.h>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        constexpr auto ReadStates = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        bool Checked(HRESULT result, const char* operation, std::string& error)
        {
            if (SUCCEEDED(result)) return true;
            error = std::string("DX12 Forward+ ") + operation + " failed: " + std::to_string(result); return false;
        }
        D3D12_RESOURCE_DESC BufferDescription(UINT64 bytes, bool writable)
        {
            D3D12_RESOURCE_DESC desc {}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = bytes; desc.Height = desc.SampleDesc.Count = 1;
            desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (writable) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            return desc;
        }
        void Transition(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource,
            D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES target)
        {
            if (state == target) return;
            D3D12_RESOURCE_BARRIER barrier {}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, target};
            commands->ResourceBarrier(1, &barrier); state = target;
        }
    }
    DX12ForwardPlus::~DX12ForwardPlus()
    {
        // A null event requests a synchronous wait. Retry an event-allocation
        // failure instead of releasing resources that are still in GPU use.
        // https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-seteventoncompletion
        while (CompletionFence && CompletionFence->GetCompletedValue() < CompletionValue)
            if (FAILED(CompletionFence->SetEventOnCompletion(CompletionValue, nullptr))) Sleep(1);
    }
    bool DX12ForwardPlus::Completed(std::string& error) const
    {
        if (!CompletionFence) return true;
        const auto completed = CompletionFence->GetCompletedValue();
        if (completed == std::numeric_limits<UINT64>::max())
        { error = "DX12 Forward+ completion device was removed"; return false; }
        if (completed < CompletionValue)
        { error = "DX12 Forward+ prior submission still owns its buffers and descriptors"; return false; }
        return true;
    }
    bool DX12ForwardPlus::BeginFrame(std::string& error)
    {
        Ready = false; error.clear();
        if (Frame == Phase::Open || Frame == Phase::Recorded)
        { error = "DX12 Forward+ requires submission or abort before another frame"; return false; }
        if (!Completed(error)) return false;
        // Buffers decay to COMMON when ExecuteCommandLists completes, even
        // after explicit transitions. Within the submitted list they remained
        // readable by both graphics and compute consumers.
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
        if (Frame == Phase::Submitted) HeaderState = IndexState = D3D12_RESOURCE_STATE_COMMON;
        CompletionFence.Reset(); CompletionValue = 0; RecordedCommands.Reset(); InputDepth.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        StartHeaderState = HeaderState; StartIndexState = IndexState; Frame = Phase::Open;
        return true;
    }
    bool DX12ForwardPlus::AbortFrame(std::string& error)
    {
        error.clear();
        if (Frame != Phase::Open && Frame != Phase::Recorded)
        { error = "DX12 Forward+ can only abort unsubmitted work"; return false; }
        HeaderState = StartHeaderState; IndexState = StartIndexState;
        Ready = false; Frame = Phase::Idle; RecordedCommands.Reset(); InputDepth.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        return true;
    }
    bool DX12ForwardPlus::MarkSubmitted(ID3D12Fence* fence, std::uint64_t value, std::string& error)
    {
        error.clear(); ComPtr<ID3D12Device> fenceDevice;
        if (Frame != Phase::Recorded || !fence || !value || value == std::numeric_limits<UINT64>::max() ||
            FAILED(fence->GetDevice(IID_PPV_ARGS(fenceDevice.GetAddressOf()))) || fenceDevice.Get() != Owner.Get())
        { error = "DX12 Forward+ submission requires recorded work and its device's queue-completion fence"; return false; }
        CompletionFence = fence; CompletionValue = value; Frame = Phase::Submitted; RecordedCommands.Reset();
        return true;
    }
    bool DX12ForwardPlus::Shutdown(std::string& error)
    {
        Ready = false; error.clear();
        // Device removal is terminal: no queued work can consume these resources.
        // Release the old owner so RHI reinitialization can use a replacement device.
        const bool deviceRemoved = Owner && FAILED(Owner->GetDeviceRemovedReason());
        if (!deviceRemoved)
        {
            if (Frame == Phase::Open || Frame == Phase::Recorded)
            { error = "DX12 Forward+ must submit or abort recorded work before shutdown"; return false; }
            if (!Completed(error)) return false;
        }
        CompletionFence.Reset(); CompletionValue = 0; RecordedCommands.Reset(); InputDepth.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        for (auto& program : Programs) program.Reset();
        Root.Reset(); DispatchHeap.Reset(); ResultHeap.Reset(); BoundsUpload.Reset(); HeaderBuffer.Reset(); IndexBuffer.Reset(); Owner.Reset();
        CurrentGrid = {}; AllocatedTiles = AllocatedIndices = DescriptorStride = 0; Frame = Phase::Idle;
        HeaderState = IndexState = StartHeaderState = StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        return true;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ForwardPlus::Cpu(UINT slot) const noexcept
    { auto handle = DispatchHeap->GetCPUDescriptorHandleForHeapStart(); handle.ptr += SIZE_T(slot) * DescriptorStride; return handle; }
    D3D12_GPU_DESCRIPTOR_HANDLE DX12ForwardPlus::Gpu(UINT slot) const noexcept
    { auto handle = DispatchHeap->GetGPUDescriptorHandleForHeapStart(); handle.ptr += UINT64(slot) * DescriptorStride; return handle; }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ForwardPlus::HeaderView() const noexcept
    { return Ready ? ResultHeap->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE {}; }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ForwardPlus::IndexView() const noexcept
    { auto handle = HeaderView(); if (handle.ptr) handle.ptr += DescriptorStride; return handle; }

    bool DX12ForwardPlus::Prepare(ID3D12Device* device, const ForwardPlusGrid& grid, std::string& error)
    {
        Owner = device;
        if (!Root)
        {
            static_assert(ConstantWords + 4 <= 64, "Forward+ root signature exceeds the native DWORD budget");
            std::array<D3D12_ROOT_PARAMETER, 4> parameters {};
            parameters[CameraRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameters[CameraRoot].Constants = {0, 0, ConstantWords};
            parameters[BoundsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            parameters[BoundsRoot].Descriptor = {1, 0};
            std::array<D3D12_DESCRIPTOR_RANGE, 2> ranges {};
            ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[DepthRoot].ParameterType = parameters[OutputRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[DepthRoot].DescriptorTable = {1, &ranges[0]}; parameters[OutputRoot].DescriptorTable = {1, &ranges[1]};
            D3D12_ROOT_SIGNATURE_DESC desc {}; desc.NumParameters = UINT(parameters.size()); desc.pParameters = parameters.data();
            ComPtr<ID3DBlob> serialized, diagnostic;
            if (!Checked(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), diagnostic.GetAddressOf()),
                    "root serialization", error))
            { if (diagnostic) error += static_cast<const char*>(diagnostic->GetBufferPointer()); return false; }
            if (!Checked(Evans::Renderer::CreateCachedRootSignature(device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(Root.GetAddressOf())), "root creation", error)) return false;
        }
        auto& program = Programs[grid.Samples > 1 ? 1 : 0];
        if (!program)
        {
            std::string source;
            if (!LoadTechniqueShaderSource("DX12", Shader == TiledLightShader::TiledDeferred ? "TiledDeferred.hlsl" : "ForwardPlus.hlsl", source, error)) return false;
            const D3D_SHADER_MACRO macros[] {{Shader == TiledLightShader::TiledDeferred ? "EVANS_TILED_DEFERRED_MULTISAMPLE" : "EVANS_FORWARD_PLUS_MULTISAMPLE", grid.Samples > 1 ? "1" : "0"}, {nullptr, nullptr}};
            ComPtr<ID3DBlob> code, diagnostic;
            if (!Checked(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), Shader == TiledLightShader::TiledDeferred ? "DX12/TiledDeferred.hlsl" : "DX12/ForwardPlus.hlsl", macros, nullptr, "CSAssign", "cs_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                0, code.GetAddressOf(), diagnostic.GetAddressOf()), "shader compilation", error))
            { if (diagnostic) error += static_cast<const char*>(diagnostic->GetBufferPointer()); return false; }
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc {}; desc.pRootSignature = Root.Get(); desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            if (!Checked(Evans::Renderer::CreateCachedComputePipelineState(device,&desc, IID_PPV_ARGS(program.GetAddressOf())), "compute pipeline", error)) return false;
        }
        if (!DispatchHeap)
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc {}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = 3; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (!Checked(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(DispatchHeap.GetAddressOf())), "dispatch descriptors", error)) return false;
            DescriptorStride = device->GetDescriptorHandleIncrementSize(desc.Type);
        }
        if (!ResultHeap)
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc {}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; desc.NumDescriptors = 2;
            if (!Checked(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(ResultHeap.GetAddressOf())), "copyable result descriptors", error)) return false;
        }
        const auto allocate = [&](UINT64 bytes, D3D12_HEAP_TYPE type, ComPtr<ID3D12Resource>& buffer)
        {
            D3D12_HEAP_PROPERTIES heap {}; heap.Type = type; heap.CreationNodeMask = heap.VisibleNodeMask = 1;
            const bool upload = type == D3D12_HEAP_TYPE_UPLOAD;
            const auto desc = BufferDescription(bytes, !upload);
            return Checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(buffer.GetAddressOf())), "list storage", error);
        };
        if (!BoundsUpload && !allocate(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds), D3D12_HEAP_TYPE_UPLOAD, BoundsUpload)) return false;
        if (AllocatedTiles != grid.TileCount || AllocatedIndices != grid.IndexCount)
        {
            ComPtr<ID3D12Resource> headers, indices;
            if (!allocate(UINT64(grid.TileCount) * sizeof(ForwardPlusTile), D3D12_HEAP_TYPE_DEFAULT, headers) ||
                !allocate(UINT64(grid.IndexCount) * sizeof(std::uint32_t), D3D12_HEAP_TYPE_DEFAULT, indices)) return false;
            const auto views = [&](ID3D12Resource* buffer, UINT elements, UINT stride, UINT slot)
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC storage {}; storage.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                storage.Buffer.NumElements = elements; storage.Buffer.StructureByteStride = stride;
                device->CreateUnorderedAccessView(buffer, nullptr, &storage, Cpu(slot));
                D3D12_SHADER_RESOURCE_VIEW_DESC sampled {}; sampled.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                sampled.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                sampled.Buffer.NumElements = elements; sampled.Buffer.StructureByteStride = stride;
                auto destination = ResultHeap->GetCPUDescriptorHandleForHeapStart(); destination.ptr += SIZE_T(slot - HeaderSlot) * DescriptorStride;
                device->CreateShaderResourceView(buffer, &sampled, destination);
            };
            views(headers.Get(), grid.TileCount, sizeof(ForwardPlusTile), HeaderSlot);
            views(indices.Get(), grid.IndexCount, sizeof(std::uint32_t), IndexSlot);
            HeaderBuffer = std::move(headers); IndexBuffer = std::move(indices);
            AllocatedTiles = grid.TileCount; AllocatedIndices = grid.IndexCount;
            HeaderState = IndexState = StartHeaderState = StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        }
        return true;
    }
    bool DX12ForwardPlus::Record(ID3D12Device* device, ID3D12GraphicsCommandList* commands, ID3D12Resource* depth,
        D3D12_RESOURCE_STATES& depthState, const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights,
        const ForwardPlusParameters& parameters, std::string& error)
    {
        Ready = false; error.clear();
        const auto fail = [&](const char* text) { error = text; return false; };
        if (Frame != Phase::Open || !device || !commands || !depth || commands->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
            return fail("DX12 Forward+ requires an open frame, camera depth and a direct command list");
        ComPtr<ID3D12Device> commandDevice, depthDevice;
        if (FAILED(commands->GetDevice(IID_PPV_ARGS(commandDevice.GetAddressOf()))) ||
            FAILED(depth->GetDevice(IID_PPV_ARGS(depthDevice.GetAddressOf()))) || commandDevice.Get() != device || depthDevice.Get() != device ||
            (Owner && Owner.Get() != device)) return fail("DX12 Forward+ commands and resources must share their device");
        const auto image = depth->GetDesc();
        if (image.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || image.DepthOrArraySize != 1 || image.MipLevels != 1 ||
            image.Format != DXGI_FORMAT_R32_TYPELESS || !(image.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ||
            (image.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) || image.Width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            image.Height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return fail("DX12 Forward+ requires original sampleable D32 camera depth without arrays or mipmaps");
        ForwardPlusGrid grid; ForwardPlusConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildForwardPlusGrid(UINT(image.Width), image.Height, image.SampleDesc.Count, lights.size(), parameters, grid, error) ||
            !BuildForwardPlusConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error) ||
            !Prepare(device, grid, error)) return false;
        if (!bounds.empty())
        {
            const D3D12_RANGE noRead {}; void* memory = nullptr;
            if (!Checked(BoundsUpload->Map(0, &noRead, &memory), "light upload map", error)) return false;
            const D3D12_RANGE written {0, bounds.size() * sizeof(ForwardPlusGpuBounds)};
            std::memcpy(memory, bounds.data(), written.End); BoundsUpload->Unmap(0, &written);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC view {}; view.Format = DXGI_FORMAT_R32_FLOAT;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.ViewDimension = grid.Samples > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
        if (grid.Samples == 1) view.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(depth, &view, Cpu(DepthSlot));
        Transition(commands, depth, depthState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Transition(commands, HeaderBuffer.Get(), HeaderState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(commands, IndexBuffer.Get(), IndexState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto* heap = DispatchHeap.Get(); commands->SetDescriptorHeaps(1, &heap);
        commands->SetComputeRootSignature(Root.Get()); commands->SetPipelineState(Programs[grid.Samples > 1 ? 1 : 0].Get());
        commands->SetComputeRoot32BitConstants(CameraRoot, ConstantWords, &constants, 0);
        commands->SetComputeRootShaderResourceView(BoundsRoot, BoundsUpload->GetGPUVirtualAddress());
        commands->SetComputeRootDescriptorTable(DepthRoot, Gpu(DepthSlot));
        commands->SetComputeRootDescriptorTable(OutputRoot, Gpu(HeaderSlot));
        commands->Dispatch(grid.TilesX, grid.TilesY, 1);
        // Transition barriers order UAV stores before all later shader reads;
        // the next dispatch transitions back before writing these buffers again.
        Transition(commands, HeaderBuffer.Get(), HeaderState, ReadStates);
        Transition(commands, IndexBuffer.Get(), IndexState, ReadStates);
        InputDepth = depth; RecordedCommands = commands; CurrentGrid = grid; Frame = Phase::Recorded; Ready = true;
        return true;
    }
    bool DX12ForwardPlus::RecordReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* headers,
        ID3D12Resource* indices, std::string& error)
    {
        error.clear();
        if (!Ready || Frame != Phase::Recorded || !commands || commands != RecordedCommands.Get() || !headers || !indices || ReadbackTargets[0])
        { error = "DX12 Forward+ permits one readback after assignment on its unsubmitted command list"; return false; }
        ID3D12Resource* destinations[] {headers, indices};
        const UINT64 bytes[] {UINT64(CurrentGrid.TileCount) * sizeof(ForwardPlusTile), UINT64(CurrentGrid.IndexCount) * sizeof(std::uint32_t)};
        for (unsigned index = 0; index < 2; ++index)
        {
            ComPtr<ID3D12Device> owner; D3D12_HEAP_PROPERTIES heap {}; D3D12_HEAP_FLAGS flags {};
            const auto desc = destinations[index]->GetDesc();
            if (FAILED(destinations[index]->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) || owner.Get() != Owner.Get() ||
                FAILED(destinations[index]->GetHeapProperties(&heap, &flags)) || heap.Type != D3D12_HEAP_TYPE_READBACK ||
                desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < bytes[index] || headers == indices)
            { error = "DX12 Forward+ requires distinct, sufficiently large readback buffers on its device"; return false; }
        }
        Transition(commands, HeaderBuffer.Get(), HeaderState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(commands, IndexBuffer.Get(), IndexState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        commands->CopyBufferRegion(headers, 0, HeaderBuffer.Get(), 0, bytes[0]);
        commands->CopyBufferRegion(indices, 0, IndexBuffer.Get(), 0, bytes[1]);
        Transition(commands, HeaderBuffer.Get(), HeaderState, ReadStates);
        Transition(commands, IndexBuffer.Get(), IndexState, ReadStates);
        ReadbackTargets[0] = headers; ReadbackTargets[1] = indices;
        return true;
    }
}
#endif
