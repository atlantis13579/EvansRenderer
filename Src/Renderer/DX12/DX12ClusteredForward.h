#pragma once
#if defined(_WIN32)
#include "Renderer/Lighting/ClusteredForwardContract.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    // One assignment per frame on a direct command list. BeginFrame only permits
    // reuse after the registered completion fence. After executing this list and
    // queue-signaling its completion fence, the caller must call MarkSubmitted.
    // Discard an unsubmitted command list before AbortFrame; this producer
    // restores its output states. No camera depth resource is required.
    // Root state and the active descriptor heap are replaced, so later passes
    // rebind them. Result SRVs are CPU-only descriptors suitable for copying into
    // the frame's lighting heap. Inputs/uploads/outputs stay owned until completion.
    class DX12ClusteredForward
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        enum class Phase { Idle, Open, Recorded, Submitted };
        static constexpr UINT CameraRoot = 0, BoundsRoot = 1, OutputRoot = 2;
        static constexpr UINT HeaderSlot = 0, IndexSlot = 1;
    public:
        DX12ClusteredForward() = default;
        ~DX12ClusteredForward(); // Waits for any registered outstanding submission.
        DX12ClusteredForward(const DX12ClusteredForward&) = delete;
        DX12ClusteredForward& operator=(const DX12ClusteredForward&) = delete;
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool MarkSubmitted(ID3D12Fence* fence, std::uint64_t value, std::string& error);
        bool Shutdown(std::string& error);
        bool Record(ID3D12Device* device, ID3D12GraphicsCommandList* commands, std::uint32_t width, std::uint32_t height, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error);
        // Optional single diagnostic copy; the destinations are retained through
        // submission, and output buffers return to their shader-readable states.
        bool RecordReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* headers,
            ID3D12Resource* indices, std::string& error);
        ID3D12Resource* Headers() const noexcept { return Ready ? HeaderBuffer.Get() : nullptr; }
        ID3D12Resource* Indices() const noexcept { return Ready ? IndexBuffer.Get() : nullptr; }
        D3D12_CPU_DESCRIPTOR_HANDLE HeaderView() const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE IndexView() const noexcept;
        ClusteredForwardGrid Grid() const noexcept { return Ready ? CurrentGrid : ClusteredForwardGrid {}; }
    private:
        bool Completed(std::string& error) const;
        bool Prepare(ID3D12Device* device, const ClusteredForwardGrid& grid, std::string& error);
        D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT slot) const noexcept;
        D3D12_GPU_DESCRIPTOR_HANDLE Gpu(UINT slot) const noexcept;
        ComPtr<ID3D12Device> Owner;
        ComPtr<ID3D12RootSignature> Root;
        ComPtr<ID3D12PipelineState> Program;
        ComPtr<ID3D12DescriptorHeap> DispatchHeap, ResultHeap;
        ComPtr<ID3D12Resource> BoundsUpload, ConstantUpload, HeaderBuffer, IndexBuffer;
        std::array<ComPtr<ID3D12Resource>, 2> ReadbackTargets;
        ComPtr<ID3D12GraphicsCommandList> RecordedCommands;
        ComPtr<ID3D12Fence> CompletionFence;
        std::uint64_t CompletionValue = 0;
        D3D12_RESOURCE_STATES HeaderState = D3D12_RESOURCE_STATE_COMMON, IndexState = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES StartHeaderState = D3D12_RESOURCE_STATE_COMMON, StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        ClusteredForwardGrid CurrentGrid;
        UINT DescriptorStride = 0, AllocatedClusters = 0, AllocatedIndices = 0;
        Phase Frame = Phase::Idle;
        bool Ready = false;
    };
}
#endif
