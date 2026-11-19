#pragma once
#if defined(_WIN32)

#include "Renderer/Lighting/ForwardPlusContract.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <wrl/client.h>

namespace Evans::Renderer
{
    // List producer and independent material programs. The RHI owns pass order.
    // Caller unbinds depth writers and previous list readers before Generate.
    // It accepts the immediate context and base D32 2D/MSAA depth from its device,
    // owns CS b0/t0-t1/u0-u1 and the compute program, and clears these bindings.
    // BeginFrame and every failed Generate hide stale output. Consumers rebind
    // their own state and obey the overflow fallback in ForwardPlus.h.
    class DX11ForwardPlus
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    public:
        explicit DX11ForwardPlus(TiledLightShader shader = TiledLightShader::ForwardPlus) : Shader(shader) {}
        DX11ForwardPlus(const DX11ForwardPlus&) = delete;
        DX11ForwardPlus& operator=(const DX11ForwardPlus&) = delete;
        void BeginFrame() noexcept { Ready = false; }
        void Shutdown() noexcept;
        bool Generate(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11ShaderResourceView* depth,
            const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights,
            const ForwardPlusParameters& parameters, std::string& error);
        enum SurfaceProgram : std::size_t { OpaqueSurface, ContactSurface, WeightedSurface, DepthSurface, SurfaceProgramCount };
        bool UploadScene(ID3D11Device* device, ID3D11DeviceContext* context,
            const std::shared_ptr<const SceneLightCollection>& lights, std::string& error);
        bool PrepareMaterial(ID3D11Device* device, Techniques::ShadowFilterMode filter, SurfaceProgram program, std::string& error);
        bool PrepareDeferred(ID3D11Device* device, Techniques::ShadowFilterMode filter, bool multisample, std::string& error);
        ID3D11VertexShader* DeferredVertex() const noexcept { return LightingVertex.Get(); }
        ID3D11PixelShader* DeferredPixel(Techniques::ShadowFilterMode filter, bool multisample) const noexcept
        { return LightingPixels[static_cast<std::size_t>(filter)][multisample ? 1 : 0].Get(); }
        bool BindSurface(ID3D11DeviceContext* context, const LightListSurfaceConstants& constants, std::string& error);
        ID3D11VertexShader* MaterialVertex() const noexcept { return SurfaceVertex.Get(); }
        ID3D11PixelShader* MaterialPixel(Techniques::ShadowFilterMode filter, SurfaceProgram program) const noexcept
        { return SurfacePixels[static_cast<std::size_t>(filter)][program].Get(); }
        ID3D11ShaderResourceView* Headers() const noexcept { return Ready ? HeaderView.Get() : nullptr; }
        ID3D11ShaderResourceView* Indices() const noexcept { return Ready ? IndexView.Get() : nullptr; }
        ID3D11ShaderResourceView* SceneLights() const noexcept { return SceneView.Get(); }
        ForwardPlusGrid Grid() const noexcept { return Ready ? CurrentGrid : ForwardPlusGrid {}; }
    private:
        const TiledLightShader Shader;
        bool Prepare(ID3D11Device* device, const ForwardPlusGrid& grid, std::string& error);
        ComPtr<ID3D11Device> Owner;
        std::array<ComPtr<ID3D11ComputeShader>, 2> Programs;
        ComPtr<ID3D11Buffer> Constants, BoundsBuffer, HeaderBuffer, IndexBuffer;
        ComPtr<ID3D11ShaderResourceView> BoundsView, HeaderView, IndexView;
        ComPtr<ID3D11UnorderedAccessView> HeaderTarget, IndexTarget;
        ForwardPlusGrid CurrentGrid;
        std::uint32_t AllocatedTiles = 0, AllocatedIndices = 0;
        bool Ready = false;
        ComPtr<ID3D11VertexShader> SurfaceVertex;
        static constexpr std::size_t ShadowFilterCount = static_cast<std::size_t>(Techniques::ShadowFilterMode::VSM) + 1;
        static constexpr UINT MaterialLightSlot = 16, MaterialConstantsSlot = 7;
        std::array<std::array<ComPtr<ID3D11PixelShader>, SurfaceProgramCount>, ShadowFilterCount> SurfacePixels;
        ComPtr<ID3D11VertexShader> LightingVertex;
        std::array<std::array<ComPtr<ID3D11PixelShader>, 2>, ShadowFilterCount> LightingPixels;
        ComPtr<ID3D11Buffer> SceneBuffer, SurfaceConstants, EmptyHeader, EmptyIndex;
        ComPtr<ID3D11ShaderResourceView> SceneView, EmptyHeaderView, EmptyIndexView;
        std::shared_ptr<const SceneLightCollection> UploadedLights;
    };
}
#endif
