#pragma once

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include <dxgi1_4.h>

#include <memory>

namespace Evans::Editor
{

// Renders the Riemann immediate-mode pull queue with a native D3D12 pipeline.
class ImguiRenderDX12 final
{
public:
    ImguiRenderDX12();
    ~ImguiRenderDX12();

    ImguiRenderDX12(const ImguiRenderDX12&) = delete;
    ImguiRenderDX12& operator=(const ImguiRenderDX12&) = delete;

    [[nodiscard]] bool Init(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* initializationCommands,
        DXGI_FORMAT renderTargetFormat,
        unsigned int frameCount);
    void Shutdown() noexcept;
    void Draw(
        ID3D12GraphicsCommandList* commandList,
        int width,
        int height,
        unsigned int frameIndex);

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};

} // namespace Evans::Editor

#endif
