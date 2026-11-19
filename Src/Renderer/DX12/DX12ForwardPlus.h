#pragma once
#if defined(_WIN32)
#include "Renderer/Lighting/ForwardPlusContract.h"
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
    // Discard an unsubmitted command list before AbortFrame; the caller restores
    // its depth-state snapshot, while this producer restores its output states.
    // Root state and the active descriptor heap are replaced, so later passes
    // rebind them. Result SRVs are CPU-only descriptors suitable for copying into
    // the frame's lighting heap. Inputs/uploads/outputs stay owned until completion.
    class DX12ForwardPlus
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        enum class Phase { Idle, Open, Recorded, Submitted };
        static constexpr UINT ConstantWords = sizeof(ForwardPlusConstants) / sizeof(std::uint32_t);
        static constexpr UINT CameraRoot = 0, BoundsRoot = 1, DepthRoot = 2, OutputRoot = 3;
        static constexpr UINT DepthSlot = 0, HeaderSlot = 1, IndexSlot = 2;
    public:
        explicit DX12ForwardPlus(TiledLightShader shader = TiledLightShader::ForwardPlus) : Shader(shader) {}
        ~DX12ForwardPlus(); // Waits for any registered outstanding submission.
        DX12ForwardPlus(const DX12ForwardPlus&) = delete;
        DX12ForwardPlus& operator=(const DX12ForwardPlus&) = delete;
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool MarkSubmitted(ID3D12Fence* fence, std::uint64_t value, std::string& error);
        bool Shutdown(std::string& error);
        bool Record(ID3D12Device* device, ID3D12GraphicsCommandList* commands, ID3D12Resource* depth,
            D3D12_RESOURCE_STATES& depthState, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ForwardPlusParameters& parameters, std::string& error);
        // Optional single diagnostic copy; the destinations are retained through
        // submission, and output buffers return to their shader-readable states.
        bool RecordReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* headers,
            ID3D12Resource* indices, std::string& error);
        ID3D12Resource* Headers() const noexcept { return Ready ? HeaderBuffer.Get() : nullptr; }
        ID3D12Resource* Indices() const noexcept { return Ready ? IndexBuffer.Get() : nullptr; }
        D3D12_CPU_DESCRIPTOR_HANDLE HeaderView() const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE IndexView() const noexcept;
        ForwardPlusGrid Grid() const noexcept { return Ready ? CurrentGrid : ForwardPlusGrid {}; }
    private:
        const TiledLightShader Shader;
        bool Completed(std::string& error) const;
        bool Prepare(ID3D12Device* device, const ForwardPlusGrid& grid, std::string& error);
        D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT slot) const noexcept;
        D3D12_GPU_DESCRIPTOR_HANDLE Gpu(UINT slot) const noexcept;
        ComPtr<ID3D12Device> Owner;
        ComPtr<ID3D12RootSignature> Root;
        std::array<ComPtr<ID3D12PipelineState>, 2> Programs;
        ComPtr<ID3D12DescriptorHeap> DispatchHeap, ResultHeap;
        ComPtr<ID3D12Resource> BoundsUpload, HeaderBuffer, IndexBuffer, InputDepth;
        std::array<ComPtr<ID3D12Resource>, 2> ReadbackTargets;
        ComPtr<ID3D12GraphicsCommandList> RecordedCommands;
        ComPtr<ID3D12Fence> CompletionFence;
        std::uint64_t CompletionValue = 0;
        D3D12_RESOURCE_STATES HeaderState = D3D12_RESOURCE_STATE_COMMON, IndexState = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES StartHeaderState = D3D12_RESOURCE_STATE_COMMON, StartIndexState = D3D12_RESOURCE_STATE_COMMON;
        ForwardPlusGrid CurrentGrid;
        UINT DescriptorStride = 0, AllocatedTiles = 0, AllocatedIndices = 0;
        Phase Frame = Phase::Idle;
        bool Ready = false;
    };
}
#endif
