#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#include "Renderer/DX12/DX12ClusteredForward.h"
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
            error = std::string("DX12 Clustered Forward ") + operation + " failed: " + std::to_string(result); return false;
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
    DX12ClusteredForward::~DX12ClusteredForward()
    {
        // A null event requests a synchronous wait. Retry an event-allocation
        // failure instead of releasing resources that are still in GPU use.
        // https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-seteventoncompletion
        while (CompletionFence && CompletionFence->GetCompletedValue() < CompletionValue)
            if (FAILED(CompletionFence->SetEventOnCompletion(CompletionValue, nullptr))) Sleep(1);
    }
    bool DX12ClusteredForward::Completed(std::string& error) const
    {
        if (!CompletionFence) return true;
        const auto completed = CompletionFence->GetCompletedValue();
        if (completed == std::numeric_limits<UINT64>::max())
        { error = "DX12 Clustered Forward completion device was removed"; return false; }
        if (completed < CompletionValue)
        { error = "DX12 Clustered Forward prior submission still owns its buffers and descriptors"; return false; }
        return true;
    }
    bool DX12ClusteredForward::BeginFrame(std::string& error)
    {
        Ready = false; error.clear();
        if (Frame == Phase::Open || Frame == Phase::Recorded)
        { error = "DX12 Clustered Forward requires submission or abort before another frame"; return false; }
        if (!Completed(error)) return false;
        // Buffers decay to COMMON when ExecuteCommandLists completes, even
        // after explicit transitions. Within the submitted list they remained
        // readable by both graphics and compute consumers.
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
        if (Frame == Phase::Submitted) HeaderState = IndexState = D3D12_RESOURCE_STATE_COMMON;
        CompletionFence.Reset(); CompletionValue = 0; RecordedCommands.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        StartHeaderState = HeaderState; StartIndexState = IndexState; Frame = Phase::Open;
        return true;
    }
    bool DX12ClusteredForward::AbortFrame(std::string& error)
    {
        error.clear();
        if (Frame != Phase::Open && Frame != Phase::Recorded)
        { error = "DX12 Clustered Forward can only abort unsubmitted work"; return false; }
        HeaderState = StartHeaderState; IndexState = StartIndexState;
        Ready = false; Frame = Phase::Idle; RecordedCommands.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        return true;
    }
    bool DX12ClusteredForward::MarkSubmitted(ID3D12Fence* fence, std::uint64_t value, std::string& error)
    {
        error.clear(); ComPtr<ID3D12Device> fenceDevice;
        if (Frame != Phase::Recorded || !fence || !value || value == std::numeric_limits<UINT64>::max() ||
            FAILED(fence->GetDevice(IID_PPV_ARGS(fenceDevice.GetAddressOf()))) || fenceDevice.Get() != Owner.Get())
        { error = "DX12 Clustered Forward submission requires recorded work and its device's queue-completion fence"; return false; }
        CompletionFence = fence; CompletionValue = value; Frame = Phase::Submitted; RecordedCommands.Reset();
        return true;
    }
    bool DX12ClusteredForward::Shutdown(std::string& error)
    {
        Ready = false; error.clear();
        // Device removal is terminal: no queued work can consume these resources.
        // Release the old owner so RHI reinitialization can use a replacement device.
        const bool deviceRemoved = Owner && FAILED(Owner->GetDeviceRemovedReason());
        if (!deviceRemoved)
        {
            if (Frame == Phase::Open || Frame == Phase::Recorded)
            { error = "DX12 Clustered Forward must submit or abort recorded work before shutdown"; return false; }
            if (!Completed(error)) return false;
        }
        CompletionFence.Reset(); CompletionValue = 0; RecordedCommands.Reset();
        for (auto& target : ReadbackTargets) target.Reset();
        Program.Reset();
        Root.Reset(); DispatchHeap.Reset(); ResultHeap.Reset(); BoundsUpload.Reset(); ConstantUpload.Reset(); HeaderBuffer.Reset(); IndexBuffer.Reset(); Owner.Reset();
        CurrentGrid = {}; AllocatedClusters = AllocatedIndices = DescriptorStride = 0; Frame = Phase::Idle;
        HeaderState = IndexState = StartHeaderState = StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        return true;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ClusteredForward::Cpu(UINT slot) const noexcept
    { auto handle = DispatchHeap->GetCPUDescriptorHandleForHeapStart(); handle.ptr += SIZE_T(slot) * DescriptorStride; return handle; }
    D3D12_GPU_DESCRIPTOR_HANDLE DX12ClusteredForward::Gpu(UINT slot) const noexcept
    { auto handle = DispatchHeap->GetGPUDescriptorHandleForHeapStart(); handle.ptr += UINT64(slot) * DescriptorStride; return handle; }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ClusteredForward::HeaderView() const noexcept
    { return Ready ? ResultHeap->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE {}; }
    D3D12_CPU_DESCRIPTOR_HANDLE DX12ClusteredForward::IndexView() const noexcept
    { auto handle = HeaderView(); if (handle.ptr) handle.ptr += DescriptorStride; return handle; }

    bool DX12ClusteredForward::Prepare(ID3D12Device* device, const ClusteredForwardGrid& grid, std::string& error)
    {
        Owner = device;
        if (!Root)
        {
            std::array<D3D12_ROOT_PARAMETER, 3> parameters {};
            parameters[CameraRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[CameraRoot].Descriptor = {0, 0};
            parameters[BoundsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            parameters[BoundsRoot].Descriptor = {0, 0};
            const D3D12_DESCRIPTOR_RANGE outputs {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[OutputRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[OutputRoot].DescriptorTable = {1, &outputs};
            D3D12_ROOT_SIGNATURE_DESC desc {}; desc.NumParameters = UINT(parameters.size()); desc.pParameters = parameters.data();
            ComPtr<ID3DBlob> serialized, diagnostic;
            if (!Checked(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), diagnostic.GetAddressOf()),
                    "root serialization", error))
            { if (diagnostic) error += static_cast<const char*>(diagnostic->GetBufferPointer()); return false; }
            if (!Checked(Evans::Renderer::CreateCachedRootSignature(device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(Root.GetAddressOf())), "root creation", error)) return false;
        }
        auto& program = Program;
        if (!program)
        {
            std::string source;
            if (!LoadTechniqueShaderSource("DX12", "ClusteredForward.hlsl", source, error)) return false;
            ComPtr<ID3DBlob> code, diagnostic;
            if (!Checked(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX12/ClusteredForward.hlsl", nullptr, nullptr, "CSAssign", "cs_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                0, code.GetAddressOf(), diagnostic.GetAddressOf()), "shader compilation", error))
            { if (diagnostic) error += static_cast<const char*>(diagnostic->GetBufferPointer()); return false; }
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc {}; desc.pRootSignature = Root.Get(); desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            if (!Checked(Evans::Renderer::CreateCachedComputePipelineState(device,&desc, IID_PPV_ARGS(program.GetAddressOf())), "compute pipeline", error)) return false;
        }
        if (!DispatchHeap)
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc {}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = 2; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
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
        constexpr UINT ConstantAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
        constexpr UINT ConstantBytes = (sizeof(ClusteredForwardConstants) + ConstantAlignment - 1) / ConstantAlignment * ConstantAlignment;
        if (!ConstantUpload && !allocate(ConstantBytes, D3D12_HEAP_TYPE_UPLOAD, ConstantUpload)) return false;
        if (!BoundsUpload && !allocate(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds), D3D12_HEAP_TYPE_UPLOAD, BoundsUpload)) return false;
        if (AllocatedClusters != grid.ClusterCount || AllocatedIndices != grid.IndexCount)
        {
            ComPtr<ID3D12Resource> headers, indices;
            if (!allocate(UINT64(grid.ClusterCount + ClusteredBoundarySlots) * sizeof(ClusteredForwardHeader), D3D12_HEAP_TYPE_DEFAULT, headers) ||
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
            views(headers.Get(), grid.ClusterCount + ClusteredBoundarySlots, sizeof(ClusteredForwardHeader), HeaderSlot);
            views(indices.Get(), grid.IndexCount, sizeof(std::uint32_t), IndexSlot);
            HeaderBuffer = std::move(headers); IndexBuffer = std::move(indices);
            AllocatedClusters = grid.ClusterCount; AllocatedIndices = grid.IndexCount;
            HeaderState = IndexState = StartHeaderState = StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        }
        return true;
    }
    bool DX12ClusteredForward::Record(ID3D12Device* device, ID3D12GraphicsCommandList* commands,
        std::uint32_t width, std::uint32_t height, const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights,
        const ClusteredForwardParameters& parameters, std::string& error)
    {
        Ready = false; error.clear();
        const auto fail = [&](const char* text) { error = text; return false; };
        if (Frame != Phase::Open || !device || !commands || commands->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
            return fail("DX12 clustered assignment requires an open frame and a direct command list");
        ComPtr<ID3D12Device> commandDevice;
        if (FAILED(commands->GetDevice(IID_PPV_ARGS(commandDevice.GetAddressOf()))) || commandDevice.Get() != device ||
            (Owner && Owner.Get() != device)) return fail("DX12 clustered commands and resources must share their device");
        ClusteredForwardGrid grid; ClusteredForwardConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildClusteredForwardGrid(width, height, lights.size(), parameters, grid, error) ||
            !BuildClusteredForwardConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error)) return false;
        if (grid.TilesX > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            grid.TilesY > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
            return fail("DX12 clustered dispatch exceeds the device group limit");
        if (!Prepare(device, grid, error)) return false;
        const D3D12_RANGE noRead {}; void* uniform = nullptr;
        if (!Checked(ConstantUpload->Map(0, &noRead, &uniform), "constant upload map", error)) return false;
        std::memcpy(uniform, &constants, sizeof(constants));
        const D3D12_RANGE constantRange {0, sizeof(constants)}; ConstantUpload->Unmap(0, &constantRange);
        if (!bounds.empty())
        {
            void* memory = nullptr;
            if (!Checked(BoundsUpload->Map(0, &noRead, &memory), "light upload map", error)) return false;
            const D3D12_RANGE written {0, bounds.size() * sizeof(ForwardPlusGpuBounds)};
            std::memcpy(memory, bounds.data(), written.End); BoundsUpload->Unmap(0, &written);
        }
        Transition(commands, HeaderBuffer.Get(), HeaderState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(commands, IndexBuffer.Get(), IndexState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto* heap = DispatchHeap.Get(); commands->SetDescriptorHeaps(1, &heap);
        commands->SetComputeRootSignature(Root.Get()); commands->SetPipelineState(Program.Get());
        commands->SetComputeRootConstantBufferView(CameraRoot, ConstantUpload->GetGPUVirtualAddress());
        commands->SetComputeRootShaderResourceView(BoundsRoot, BoundsUpload->GetGPUVirtualAddress());
        commands->SetComputeRootDescriptorTable(OutputRoot, Gpu(HeaderSlot));
        commands->Dispatch(grid.TilesX, grid.TilesY, grid.DepthSlices);
        // Transition barriers order UAV stores before all later shader reads;
        // the next dispatch transitions back before writing these buffers again.
        Transition(commands, HeaderBuffer.Get(), HeaderState, ReadStates);
        Transition(commands, IndexBuffer.Get(), IndexState, ReadStates);
        RecordedCommands = commands; CurrentGrid = grid; Frame = Phase::Recorded; Ready = true;
        return true;
    }
    bool DX12ClusteredForward::RecordReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* headers,
        ID3D12Resource* indices, std::string& error)
    {
        error.clear();
        if (!Ready || Frame != Phase::Recorded || !commands || commands != RecordedCommands.Get() || !headers || !indices || ReadbackTargets[0])
        { error = "DX12 Clustered Forward permits one readback after assignment on its unsubmitted command list"; return false; }
        ID3D12Resource* destinations[] {headers, indices};
        const UINT64 bytes[] {UINT64(CurrentGrid.ClusterCount + ClusteredBoundarySlots) * sizeof(ClusteredForwardHeader), UINT64(CurrentGrid.IndexCount) * sizeof(std::uint32_t)};
        for (unsigned index = 0; index < 2; ++index)
        {
            ComPtr<ID3D12Device> owner; D3D12_HEAP_PROPERTIES heap {}; D3D12_HEAP_FLAGS flags {};
            const auto desc = destinations[index]->GetDesc();
            if (FAILED(destinations[index]->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) || owner.Get() != Owner.Get() ||
                FAILED(destinations[index]->GetHeapProperties(&heap, &flags)) || heap.Type != D3D12_HEAP_TYPE_READBACK ||
                desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < bytes[index] || headers == indices)
            { error = "DX12 Clustered Forward requires distinct, sufficiently large readback buffers on its device"; return false; }
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
