#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Renderer/Types.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>

namespace Evans::Rasterizer::Detail
{
// Converts native float buffers to a leased, display-encoded shared texture.
// CUDA maps the input buffers directly; no host pixel storage is involved.
class D3D11GpuOutput
{
public:
    void Initialize(ID3D11Device* device);
    void Shutdown();
    ID3D11Device* Device() const { return Device_.Get(); }
    Microsoft::WRL::ComPtr<ID3D11Buffer> CreateInput(std::uint32_t count, std::uint32_t stride);
    std::shared_ptr<Renderer::GpuImage> Publish(ID3D11Buffer* colors, std::uint32_t width, std::uint32_t height);

private:
    struct Surface;
    Microsoft::WRL::ComPtr<ID3D11Device> Device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> Context_;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> Shader_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> Input_, Constants_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> InputView_;
    std::vector<std::shared_ptr<Surface>> Pool_;
};
}
#endif
