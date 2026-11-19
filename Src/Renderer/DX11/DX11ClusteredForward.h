#pragma once
#if defined(_WIN32)
#include "Renderer/Lighting/ClusteredForwardContract.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    // Immediate-context assignment. Caller unbinds previous list readers first;
    // this producer clears CS b0/t0/u0-u1 after dispatch. No depth image is bound.
    class DX11ClusteredForward
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    public:
        void BeginFrame() noexcept { Ready = false; }
        void Shutdown() noexcept;
        bool Generate(ID3D11Device* device, ID3D11DeviceContext* context, std::uint32_t width, std::uint32_t height,
            const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights,
            const ClusteredForwardParameters& parameters, std::string& error);
        enum SurfaceProgram : std::size_t { OpaqueSurface, ContactSurface, WeightedSurface, DepthSurface, SurfaceProgramCount };
        bool UploadScene(ID3D11Device* device, ID3D11DeviceContext* context,
            const std::shared_ptr<const SceneLightCollection>& lights, std::string& error);
        bool PrepareMaterial(ID3D11Device* device, Techniques::ShadowFilterMode filter, SurfaceProgram program, std::string& error);
        bool BindSurface(ID3D11DeviceContext* context, const LightListSurfaceConstants& constants, std::string& error);
        ID3D11VertexShader* MaterialVertex() const noexcept { return SurfaceVertex.Get(); }
        ID3D11PixelShader* MaterialPixel(Techniques::ShadowFilterMode filter, SurfaceProgram program) const noexcept
        { return SurfacePixels[static_cast<std::size_t>(filter)][program].Get(); }
        ID3D11ShaderResourceView* Headers() const noexcept { return Ready ? HeaderView.Get() : nullptr; }
        ID3D11ShaderResourceView* Indices() const noexcept { return Ready ? IndexView.Get() : nullptr; }
        ID3D11ShaderResourceView* SceneLights() const noexcept { return SceneView.Get(); }
        ClusteredForwardGrid Grid() const noexcept { return Ready ? CurrentGrid : ClusteredForwardGrid {}; }
    private:
        bool Prepare(ID3D11Device* device, const ClusteredForwardGrid& grid, std::string& error);
        ComPtr<ID3D11Device> Owner;
        ComPtr<ID3D11ComputeShader> Program;
        ComPtr<ID3D11Buffer> Constants, BoundsBuffer, HeaderBuffer, IndexBuffer;
        ComPtr<ID3D11ShaderResourceView> BoundsView, HeaderView, IndexView;
        ComPtr<ID3D11UnorderedAccessView> HeaderTarget, IndexTarget;
        ClusteredForwardGrid CurrentGrid;
        std::uint32_t AllocatedClusters = 0, AllocatedIndices = 0;
        bool Ready = false;
        ComPtr<ID3D11VertexShader> SurfaceVertex;
        static constexpr std::size_t ShadowFilterCount = static_cast<std::size_t>(Techniques::ShadowFilterMode::VSM) + 1;
        static constexpr UINT MaterialLightSlot = 16, MaterialConstantsSlot = 7;
        std::array<std::array<ComPtr<ID3D11PixelShader>, SurfaceProgramCount>, ShadowFilterCount> SurfacePixels;
        ComPtr<ID3D11Buffer> SceneBuffer, SurfaceConstants, EmptyHeader, EmptyIndex;
        ComPtr<ID3D11ShaderResourceView> SceneView, EmptyHeaderView, EmptyIndexView;
        std::shared_ptr<const SceneLightCollection> UploadedLights;

    };
}
#endif
