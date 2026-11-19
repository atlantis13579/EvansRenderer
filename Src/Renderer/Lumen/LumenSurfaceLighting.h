#pragma once

#include "Renderer/Lumen/LumenSurfaceLookup.h"
#include "Renderer/Lumen/LumenDistanceScene.h"
#include "Renderer/Lighting/AreaLightSampling.h"
#include "Renderer/Lumen/LumenEnvironment.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenLightingThreads=64,LumenLightingColumns=1024,LumenLightingMaximumSamples=1024;
    inline constexpr unsigned LumenLightingTileSide=8;
    inline constexpr std::size_t LumenLightingMaximumSampleWork=64*1024*1024;
    struct LumenLightingIdentity
    {
        LumenSurfaceIdentity Surface;
        LumenDistanceSceneIdentity Occlusion;
        std::uint64_t Lighting=1;
    };
    struct LumenLightingHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(LumenLightingHandle other) const noexcept { return Value==other.Value; }
    };
    struct LumenSurfaceLightingUpload
    {
        std::vector<SceneLight> Lights;
        // The scene instance represented by each card. Invalid means that the
        // source is absent from the occluder scene and uses only explicit bias.
        std::vector<unsigned> CardInstances;
        std::uint64_t LightingRevision=1;
        unsigned MaximumSteps=256;
        float HitToleranceCells=0.1f,StepScale=0.9f;
        float SurfaceBias=0.002f,VoxelBiasScale=1.25f,MaximumShadowDistance=1000,MinimumLightDistance=0.001f;
        // One Jacobi radiosity iteration reads an immutable previous atlas of
        // this exact material/occluder/light generation. An empty handle seeds
        // direct lighting; the scene scheduler owns iteration and publication.
        LumenLightingHandle PreviousLighting;
        unsigned IndirectRays=0,IndirectSequence=0;
        float IndirectMaximumDistance=25;
        Float3 IndirectSky;
        bool Readback=false;
        LumenEnvironmentLobe Environment;
    };
    struct alignas(16) LumenLightingCard
    {
        Matrix4 InverseProjection;
        Float4 Origin,Backward;
        ShaderUInt4 TileInstance; // Tile XY, source scene instance, original card.
    };
    struct alignas(16) LumenLightingEmitter
    {
        Float4 PositionKind; // Position, or direction toward a directional light; source LightType in W.
        Float4 NormalRange;
        Float4 RadianceWeight; // Emitted RGB and quadrature measure.
        ShaderUInt4 Metadata; // Cast-shadow flag, original light, sample index, reserved W.
    };
    struct alignas(16) LumenSurfaceLightingFrame
    {
        LumenDistanceTraceFrame Trace;
        ShaderUInt4 Atlas,Work; // Width/height/card side/columns; cards/samples/emitter vector offset/tile stride.
        Float4 Lighting,Numeric; // Explicit bias/voxel bias/shadow horizon/distance floor squared; roundoff/Lambert scale/reserved ZW.
        ShaderUInt4 Dispatch; // First tile, tile count, tiles per row, tile side.
        ShaderUInt4 Radiosity; // Rays, sequence, ownership-map vector base, relative lookup-card vector offset.
        Float4 SkyDistance; // Constant sky radiance, world-space indirect horizon.
        LumenEnvironmentFrame Environment;
    };
    inline unsigned LumenLightingGroups(const LumenSurfaceLightingFrame& frame)
    { return (frame.Dispatch.Y*frame.Dispatch.W*frame.Dispatch.W+LumenLightingThreads-1)/LumenLightingThreads; }
    struct LumenLightingTask
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
    };
    struct LumenLightingStatistics
    {
        unsigned TotalTiles=0,CompletedTiles=0,LastSubmittedTiles=0;
        unsigned PendingTiles() const noexcept { return TotalTiles-CompletedTiles; }
    };
    enum class LumenLightingStatus:unsigned { Uncovered=0,Complete=1,Unresolved=2,InvalidArithmetic=3 };
    struct alignas(16) LumenLightingTexel
    {
        Float4 RadianceCoverage{0,0,0,0};
        ShaderUInt4 Metadata; // Status, direct+indirect samples, blocked direct rays, unresolved samples.
    };
    static_assert(sizeof(LumenLightingCard)==112 && sizeof(LumenLightingEmitter)==64,"Lumen lighting input vector ABI changed");
    static_assert(sizeof(LumenSurfaceLightingFrame)==224 && sizeof(LumenLightingTexel)==32,"Lumen lighting frame/texel ABI changed");
    struct LumenSurfaceLightingInput
    {
        LumenSurfaceLightingFrame Frame;
        std::vector<std::uint32_t> Words; // Cards, emitters and optional instance-to-card lookup, retaining integer metadata.
    };
    struct LumenLightingInfo
    {
        LumenLightingIdentity Identity;
        LumenSurfaceLayout Layout;
        unsigned EmitterSamples=0,IndirectSamples=0,Bounce=0;
        Float3 HairSky{};
    };
    struct LumenSurfaceLightingResult
    {
        LumenLightingHandle Handle;
        LumenLightingInfo Info;
        std::vector<LumenLightingTexel> Texels;
    };

    // View-independent diffuse transport: emission + rho * (direct / pi +
    // mean previous radiance). Cosine hemisphere sampling cancels Lambert's pi.
    // Each Jacobi iteration recomputes direct light instead of accumulating it.
    // SDF visibility and finite card coverage remain geometric approximations;
    // unresolved traces/cards retain explicit uncertainty and no invented energy.
    bool BuildLumenSurfaceLightingInput(const LumenSurfaceIdentity& surface,const LumenSurfaceLayout& layout,
        const std::vector<LumenSurfaceCapture>& captures,const LumenDistanceSceneInfo& distance,
        const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingInput& output,std::string& error);
    bool ValidateLumenLightingTexels(const LumenLightingInfo& info,const std::vector<LumenLightingTexel>& texels,std::string& error);
    inline constexpr unsigned LumenMaximumIndirectBounces=8,LumenMaximumIndirectRays=64;
    bool ValidateLumenRadiositySource(const LumenSurfaceIdentity& surface,const LumenDistanceSceneIdentity& distance,
        const LumenSurfaceLightingUpload& upload,const LumenLightingInfo& previous,std::string& error);
    LumenLightingHandle NewLumenLightingHandle() noexcept;
}
