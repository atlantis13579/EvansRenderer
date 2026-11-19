#pragma once

#if defined(_WIN32)
#include "Renderer/Pipelines/DeferredContract.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <wrl/client.h>
#include <array>
#include <string>

namespace Evans::Renderer
{
    // Native sample storage is separate from the single-sample post-processing
    // surfaces. All raster attachments use one count and quality level.
    class DX11Multisampling
    {
    public:
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
        struct ResolveTargets
        {
            ID3D11Texture2D* Color = nullptr;
            ID3D11ShaderResourceView* ColorView = nullptr;
            ID3D11DepthStencilView* Depth = nullptr;
            std::array<ID3D11RenderTargetView*, GBufferAttachmentCount> Materials {};
            ID3D11RenderTargetView* Velocity = nullptr;
        };

        static std::uint32_t SupportedCounts(ID3D11Device* device);
        bool Prepare(ID3D11Device* device, int width, int height, UINT samples,
            bool hdr, bool materials, bool velocity, std::string& error);
        void Process(ID3D11DeviceContext* context, Techniques::MultisamplePassKind pass,
            const ResolveTargets& targets);

        Surface Color, Velocity;
        std::array<Surface, GBufferAttachmentCount> Materials;
        ComPtr<ID3D11Texture2D> Depth;
        ComPtr<ID3D11DepthStencilView> DepthTarget;
        ComPtr<ID3D11ShaderResourceView> DepthView;
        UINT SampleCount = 1;

    private:
        bool CreateSurface(ID3D11Device* device, Surface& surface, DXGI_FORMAT format, std::string& error);
        bool CreatePrograms(ID3D11Device* device, std::string& error);
        int Width = 0, Height = 0;
        DXGI_FORMAT ColorFormat = DXGI_FORMAT_UNKNOWN;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> DepthResolve, MaterialResolve, VelocityResolve, ColorRestore;
        ComPtr<ID3D11DepthStencilState> DepthWrite, DepthDisabled;
        ComPtr<ID3D11RasterizerState> Rasterizer;
    };
}
#endif
