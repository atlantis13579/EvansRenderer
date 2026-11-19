#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#pragma once

#include "Renderer/Shadows/ContactShadowContract.h"
#include "Renderer/TechniqueShader.h"
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    // Records one independent DX12 visibility dispatch per frame. The caller
    // waits for prior GPU work before BeginFrame or destruction and retains the
    // input depth until that fence. Abort restores this producer's unsubmitted
    // output state; the caller restores depth state through its frame snapshot.
    // Root constants snapshot all 52 words; two descriptor tables bring this
    // compute signature to 54 DWORDs, below DX12's 64-DWORD limit.
    // https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-constants-directly-in-the-root-signature
    class DX12ContactShadows
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        static constexpr UINT ConstantsRoot = 0, DepthRoot = 1, VisibilityRoot = 2;
        static constexpr UINT DepthDescriptor = 0, StorageDescriptor = 1, DescriptorCount = 2;
        static constexpr UINT ConstantWords = sizeof(ContactShadowConstants) / sizeof(std::uint32_t);
        static constexpr UINT GroupWidth = 8;
    public:
        bool PrepareDepth(std::string& error)
        {
            if (DepthVertex && DepthPixel) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("DX12", "ContactShadows.hlsl", source, error)) return false;
            const D3D_SHADER_MACRO macros[] {{"EVANS_CONTACT_DEPTH", "1"}, {nullptr, nullptr}};
            for (unsigned stage = 0; stage < 2; ++stage)
            {
                auto& destination = stage == 0 ? DepthVertex : DepthPixel;
                ComPtr<ID3DBlob> diagnostics;
                const auto result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX12/ContactShadows.hlsl", macros, nullptr,
                    stage == 0 ? "VSDepth" : "PSDepth", stage == 0 ? "vs_5_0" : "ps_5_0",
                    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                    0, destination.ReleaseAndGetAddressOf(), diagnostics.GetAddressOf());
                if (FAILED(result))
                { error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "DX12 contact depth compilation failed"; return false; }
            }
            return true;
        }
        ID3DBlob* DepthVS() const noexcept { return DepthVertex.Get(); }
        ID3DBlob* DepthPS() const noexcept { return DepthPixel.Get(); }
        ID3DBlob* Receiver(Techniques::ShadingModel model, Techniques::ShadowFilterMode filter,
            bool deferred, bool multisample, std::string& error)
        {
            constexpr unsigned ProgramsPerFilter = 7;
            const unsigned family = filter == Techniques::ShadowFilterMode::PCSS ? 1 : filter == Techniques::ShadowFilterMode::VSM ? 2 : 0;
            const unsigned program = deferred ? (multisample ? 6 : 5) : unsigned(model);
            if (program >= ProgramsPerFilter) { error = "Invalid DX12 contact receiver"; return nullptr; }
            auto& bytecode = Receivers[family * ProgramsPerFilter + program];
            if (bytecode) return bytecode.Get();
            const char* technique = family == 1 ? "PCSS" : family == 2 ? "VSM" : deferred ? "Deferred" : ShadingTechniqueFile(model, true);
            std::string source;
            if (!technique || !LoadTechniqueShaderSource("DX12", (std::string(technique) + ".hlsl").c_str(), source, error)) return nullptr;
            const auto programText = std::to_string(program);
            std::vector<D3D_SHADER_MACRO> macros {{"EVANS_CONTACT_RECEIVER", "1"}};
            if (family) macros.push_back({family == 1 ? "EVANS_PCSS_PROGRAM" : "EVANS_VSM_PROGRAM", programText.c_str()});
            if (!family && deferred && multisample) macros.push_back({"EVANS_MULTISAMPLE", "1"});
            macros.push_back({nullptr, nullptr});
            const char* entry = deferred ? (family ? "PSMain" : "PSLighting") : "PSContact";
            ComPtr<ID3DBlob> diagnostics;
            // Preserve the native surface compiler's strict-language policy.
            const auto result = Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), technique, macros.data(), nullptr, entry, "ps_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, bytecode.GetAddressOf(), diagnostics.GetAddressOf());
            if (FAILED(result))
            { error = diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer()) : "DX12 contact receiver compilation failed"; return nullptr; }
            return bytecode.Get();
        }
        void BeginFrame() noexcept { Ready = Recorded = false; FrameStartState = State; }
        void AbortFrame() noexcept { Ready = Recorded = false; State = FrameStartState; }
        ID3D12Resource* Result() const noexcept { return Ready ? Visibility.Get() : nullptr; }
        D3D12_CPU_DESCRIPTOR_HANDLE ResultView() const noexcept
        { return Ready ? ReadDescriptor->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE{}; }
        static bool Supported(ID3D12Device* device)
        {
            if (!device) return false;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT format {DXGI_FORMAT_R32_FLOAT};
            return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format))) &&
                (format.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD) &&
                (format.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
        }
        bool Record(ID3D12Device* device, ID3D12GraphicsCommandList* commands, ID3D12Resource* depth,
            D3D12_RESOURCE_STATES& depthState, const ContactShadowDispatch& description, std::string& error)
        {
            Ready = false; error.clear();
            const auto fail = [&](const char* text) { error = text; return false; };
            if (!device || !commands || !depth || Recorded)
                return fail("DX12 contact visibility requires valid resources and one dispatch per frame");
            ComPtr<ID3D12Device> depthDevice, commandDevice;
            if (FAILED(depth->GetDevice(IID_PPV_ARGS(depthDevice.GetAddressOf()))) ||
                FAILED(commands->GetDevice(IID_PPV_ARGS(commandDevice.GetAddressOf()))) ||
                depthDevice.Get() != device || commandDevice.Get() != device || (Owner && Owner.Get() != device) ||
                commands->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
                return fail("DX12 contact resources and direct command list must share their device");
            const auto image = depth->GetDesc();
            if (image.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || image.DepthOrArraySize != 1 || image.MipLevels != 1 ||
                (image.Format != DXGI_FORMAT_R32_TYPELESS && image.Format != DXGI_FORMAT_R32_FLOAT) ||
                (image.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
                image.Width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || image.Height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
                return fail("DX12 contact depth must be one sampled R32 image without array layers or mipmaps");
            ContactShadowConstants constants;
            if (!BuildContactShadowConstants(description, UINT(image.Width), image.Height, image.SampleDesc.Count, constants, error)) return false;
            if (!Prepare(device, UINT(image.Width), image.Height, image.SampleDesc.Count, error)) return false;
            D3D12_SHADER_RESOURCE_VIEW_DESC input {};
            input.Format = DXGI_FORMAT_R32_FLOAT; input.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            input.ViewDimension = Samples > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
            if (Samples == 1) input.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(depth, &input, Cpu(DepthDescriptor));
            // A state transition supplies ordering between depth writes, compute
            // reads, UAV writes and subsequent pixel reads. No implicit hazard
            // handling is assumed, including when the same allocation is reused.
            // https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
            Transition(commands, depth, depthState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Transition(commands, Visibility.Get(), State, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ID3D12DescriptorHeap* heaps[] {Descriptors.Get()}; commands->SetDescriptorHeaps(1, heaps);
            commands->SetComputeRootSignature(Root.Get());
            commands->SetPipelineState(Programs[Samples > 1 ? 1 : 0].Get());
            commands->SetComputeRoot32BitConstants(ConstantsRoot, ConstantWords, &constants, 0);
            commands->SetComputeRootDescriptorTable(DepthRoot, Gpu(DepthDescriptor));
            commands->SetComputeRootDescriptorTable(VisibilityRoot, Gpu(StorageDescriptor));
            commands->Dispatch((Width + GroupWidth - 1) / GroupWidth, (Height + GroupWidth - 1) / GroupWidth, Samples);
            Transition(commands, Visibility.Get(), State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            Ready = Recorded = true;
            return true;
        }
    private:
        D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT index) const noexcept
        {
            auto handle = Descriptors->GetCPUDescriptorHandleForHeapStart(); handle.ptr += SIZE_T(index) * Stride; return handle;
        }
        D3D12_GPU_DESCRIPTOR_HANDLE Gpu(UINT index) const noexcept
        {
            auto handle = Descriptors->GetGPUDescriptorHandleForHeapStart(); handle.ptr += UINT64(index) * Stride; return handle;
        }
        static void Transition(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource,
            D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES next)
        {
            if (state == next) return;
            D3D12_RESOURCE_BARRIER barrier {}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, next};
            commands->ResourceBarrier(1, &barrier); state = next;
        }
        bool Prepare(ID3D12Device* device, UINT width, UINT height, UINT samples, std::string& error)
        {
            const auto checked = [&](HRESULT result, const char* text)
            {
                if (SUCCEEDED(result)) return true;
                error = std::string("DX12 contact ") + text + " failed: " + std::to_string(result); return false;
            };
            if (!Supported(device)) { error = "DX12 contact visibility requires R32 shader loads and typed UAV stores"; return false; }
            if (!Root)
            {
                std::array<D3D12_DESCRIPTOR_RANGE, 2> ranges {};
                ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                std::array<D3D12_ROOT_PARAMETER, 3> parameters {};
                parameters[ConstantsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
                parameters[ConstantsRoot].Constants = {0, 0, ConstantWords};
                for (UINT index = 0; index < ranges.size(); ++index)
                {
                    ranges[index].NumDescriptors = 1;
                    ranges[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                    parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    parameters[index + 1].DescriptorTable = {1, &ranges[index]};
                }
                D3D12_ROOT_SIGNATURE_DESC desc {}; desc.NumParameters = UINT(parameters.size()); desc.pParameters = parameters.data();
                ComPtr<ID3DBlob> serialized, diagnostics;
                if (!checked(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                    serialized.GetAddressOf(), diagnostics.GetAddressOf()), "root serialization"))
                { if (diagnostics) error += static_cast<const char*>(diagnostics->GetBufferPointer()); return false; }
                if (!checked(Evans::Renderer::CreateCachedRootSignature(device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(Root.GetAddressOf())), "root creation")) return false;
                Owner = device;
            }
            auto& program = Programs[samples > 1 ? 1 : 0];
            if (!program)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("DX12", "ContactShadows.hlsl", source, error)) return false;
                const D3D_SHADER_MACRO defines[] {{"EVANS_CONTACT_MULTISAMPLE", samples > 1 ? "1" : "0"}, {nullptr, nullptr}};
                ComPtr<ID3DBlob> code, diagnostics;
                if (!checked(Evans::Renderer::CompileCachedD3DShader(source.data(), source.size(), "DX12/ContactShadows.hlsl", defines, nullptr, "CSMain", "cs_5_0",
                    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                    0, code.GetAddressOf(), diagnostics.GetAddressOf()), "shader compilation"))
                { if (diagnostics) error += static_cast<const char*>(diagnostics->GetBufferPointer()); return false; }
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc {}; desc.pRootSignature = Root.Get();
                desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
                if (!checked(Evans::Renderer::CreateCachedComputePipelineState(device,&desc, IID_PPV_ARGS(program.GetAddressOf())), "compute pipeline")) return false;
            }
            if (!Descriptors)
            {
                D3D12_DESCRIPTOR_HEAP_DESC desc {}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                desc.NumDescriptors = DescriptorCount; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
                if (!checked(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(Descriptors.GetAddressOf())), "descriptor heap")) return false;
                Stride = device->GetDescriptorHandleIncrementSize(desc.Type);
            }
            if (!ReadDescriptor)
            {
                // CopyDescriptorsSimple sources must be non-shader-visible.
                // The scene copies this SRV into its own active lighting heap.
                D3D12_DESCRIPTOR_HEAP_DESC desc {}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; desc.NumDescriptors = 1;
                if (!checked(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(ReadDescriptor.GetAddressOf())), "result descriptor")) return false;
            }
            if (!Visibility || Width != width || Height != height || Samples != samples)
            {
                D3D12_HEAP_PROPERTIES heap {}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
                heap.CreationNodeMask = heap.VisibleNodeMask = 1;
                D3D12_RESOURCE_DESC desc {}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                desc.Width = width; desc.Height = height; desc.DepthOrArraySize = UINT16(samples);
                desc.MipLevels = 1; desc.SampleDesc.Count = 1; desc.Format = DXGI_FORMAT_R32_FLOAT;
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                ComPtr<ID3D12Resource> candidate;
                if (!checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                    nullptr, IID_PPV_ARGS(candidate.GetAddressOf())), "visibility allocation")) return false;
                D3D12_UNORDERED_ACCESS_VIEW_DESC storage {}; storage.Format = desc.Format;
                storage.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY; storage.Texture2DArray.ArraySize = samples;
                device->CreateUnorderedAccessView(candidate.Get(), nullptr, &storage, Cpu(StorageDescriptor));
                D3D12_SHADER_RESOURCE_VIEW_DESC view {}; view.Format = desc.Format;
                view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                view.Texture2DArray.ArraySize = samples; view.Texture2DArray.MipLevels = 1;
                device->CreateShaderResourceView(candidate.Get(), &view, ReadDescriptor->GetCPUDescriptorHandleForHeapStart());
                Visibility = std::move(candidate); Width = width; Height = height; Samples = samples;
                State = FrameStartState = D3D12_RESOURCE_STATE_COMMON;
            }
            return true;
        }
        ComPtr<ID3D12Device> Owner;
        ComPtr<ID3DBlob> DepthVertex, DepthPixel;
        std::array<ComPtr<ID3DBlob>, 21> Receivers;
        ComPtr<ID3D12RootSignature> Root;
        std::array<ComPtr<ID3D12PipelineState>, 2> Programs;
        ComPtr<ID3D12DescriptorHeap> Descriptors, ReadDescriptor;
        ComPtr<ID3D12Resource> Visibility;
        D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON, FrameStartState = D3D12_RESOURCE_STATE_COMMON;
        UINT Width = 0, Height = 0, Samples = 0, Stride = 0;
        bool Ready = false, Recorded = false;
    };
}
#endif
