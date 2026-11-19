#pragma once

#include "Renderer/Lumen/LumenSurfaceLighting.h"
#include "Renderer/Lumen/LumenProbeSampling.h"
#include "Renderer/Lumen/LumenReflections.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenRadianceThreads=64,LumenRadianceColumns=1024;
    // Separate from mesh-SDF status: XY contain a screen pixel, never an instance.
    inline constexpr unsigned LumenRadianceScreenSurface=3,LumenRadianceWorldCache=4;
    struct alignas(16) LumenScreenTraceFrame
    {
        Matrix4 ViewProjection,InverseViewProjection;
        ShaderUInt4 Counts,Viewport; // Rays/rays per probe/sequence/dispatch stride; width/height/node budget/hierarchy levels.
        Float4 Anchor; // Camera-relative depth origin and conservative float roundoff.
        Float4 Forward,Parameters; // Unit camera-forward/reserved; horizon/world thickness/pixel stride/minimum W.
    };
    static_assert(sizeof(LumenScreenTraceFrame)==208,"Lumen screen trace ABI changed");
    inline constexpr unsigned LumenScreenDepthLevels=16,LumenScreenDepthHeaderWords=4*LumenScreenDepthLevels;
    struct alignas(16) LumenScreenDepthFrame
    {
        Matrix4 Inverse;
        Float4 Anchor,Forward;
        ShaderUInt4 Source,Destination,Dispatch; // Width/height/word offset/level; dispatch stride/reserved.
        Float4 Numeric; // Minimum homogeneous W, empty depth, unknown depth, reserved.
    };
    static_assert(sizeof(LumenScreenDepthFrame)==160,"Lumen screen depth ABI changed");
    struct LumenScreenDepthPlan
    {
        std::vector<LumenScreenDepthFrame> Levels;
        std::size_t Bytes=0;
    };
    // Ceil-halved levels retain odd edge pixels. GPU-written headers and depth
    // share one storage binding; invalid reconstruction forces leaf traversal.
    bool BuildLumenScreenDepthPlan(const LumenScreenTraceFrame& frame,LumenScreenDepthPlan& output,std::string& error);
    struct LumenRadianceHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(LumenRadianceHandle other) const noexcept { return Value==other.Value; }
    };
    inline constexpr unsigned LumenWorldMaximumCascades=4;
    struct LumenWorldCacheLayout
    {
        Float4 OriginCell; // First probe position and lattice spacing.
        ShaderUInt4 Dimensions; // XYZ probe counts and square directional side.
        Float4 Distances; // Probe near distance, required local trace, far distance, reciprocal cell size.
    };
    static_assert(sizeof(LumenWorldCacheLayout)==48,"World radiance-cache layout ABI changed");
    struct LumenWorldCacheBinding
    {
        LumenRadianceHandle Radiance;
        LumenWorldCacheLayout Layout{};
        Float3 SkyRadiance{};
        bool Parallax=true;
        std::vector<LumenWorldCacheLayout> Coarser;
        float BlendCells=1; // Boundary width in cells of the finer cascade; zero disables blending.
        LumenEnvironmentLobe Environment;
        unsigned AngularFilterRadius=0;
    };
    struct alignas(16) LumenWorldLookupFrame
    {
        LumenWorldCacheLayout Layout;
        std::array<LumenWorldCacheLayout,LumenWorldMaximumCascades-1> Coarser{};
        ShaderUInt4 Work; // Rays, rays per probe, sequence, dispatch row stride.
        Float4 Sky; // Explicit miss radiance, sphere parallax toggle.
        Float4 Query; // Query horizon, screen-first flag, numeric guard, cascade count.
        Float4 Transition; // Boundary width in finer cells, extra angular tent radius, source depth layers; reserved W.
        LumenEnvironmentFrame Environment;
    };
    static_assert(sizeof(LumenWorldLookupFrame)==288,"World cache lookup ABI changed");
    struct LumenRadianceTraceUpload
    {
        LumenDistanceTraceUpload Trace;
        LumenProbeRaySet Probes; // Nonempty origins select GPU ray generation; Trace.Rays must then be empty.
        LumenScreenProbePlacement Screen; // Current-frame G-buffer placement; CPU origins and explicit rays must be empty.
        LumenReflectionRays Reflections; // Screen enabled: generate reflected rays instead of diffuse probes.
        Techniques::LumenScreenTraceSettings Hybrid; // Optional screen-first transport; misses retain the full mesh-SDF ray.
        LumenWorldCacheBinding WorldCache;
        LumenDistanceSceneHandle GlobalDistance;
        float MinimumFacingCosine=0.1f,MinimumNormalCosine=0.25f;
        float PlaneTolerance=0,VoxelToleranceScale=2;
        bool Readback=false;
    };
    enum class LumenRadianceStatus:unsigned { Unavailable=0,Complete=1,Partial=2,Invalid=3 };
    struct alignas(16) LumenRadianceSample
    {
        // Layered directional sources retain the normal-plane SDF footprint in
        // Hit.Uncertainty.Z. Resolved query samples keep that component zero.
        // A validated layered world-cache surface instead carries its predicted
        // position/normal and equal normal-plane footprints in uncertainty XY.
        LumenDistanceHit Hit;
        Float4 RadianceCoverage{0,0,0,0};
        // Dominant card, direction groups, cache status, unreliable groups.
        // A layered world-cache surface uses X for its deepest contributing
        // source ordinal; YW remain zero and no card identity is implied.
        ShaderUInt4 Cache{LumenLookupMissingCard,0,0,0};
    };
    struct alignas(16) LumenRadianceResolveFrame
    {
        ShaderUInt4 Counts,Atlas,Mapping; // Queries/side/stride/columns; width/height/instances/dispatch stride; card-vector offset/cards/retain plane footprint/reserved.
        Float4 Parameters,Precision; // Facing/normal agreement/explicit radius/voxel radius scale; coordinate/world roundoff/reserved ZW.
    };
    static_assert(sizeof(LumenRadianceSample)==112 && sizeof(LumenRadianceResolveFrame)==80,"Lumen ray-radiance ABI changed");
    struct LumenRadianceInput
    {
        LumenDistanceTraceFrame Trace;
        LumenProbeTraceFrame ProbeTrace;
        LumenScreenProbeFrame Screen;
        LumenReflectionRayFrame Reflection;
        LumenScreenTraceFrame Hybrid;
        LumenScreenDepthPlan Depth;
        LumenRadianceResolveFrame Resolve;
        // Instance ranges point to packed card indices; the aligned tail holds
        // full lookup cards. Noncontiguous input cards retain their ownership.
        std::vector<std::uint32_t> Words;
    };
    struct LumenRadianceInfo
    {
        LumenLightingIdentity Identity;
        LumenProbeRaySet Probes;
        unsigned RayCount=0;
        bool Hybrid=false,WorldCache=false,HardwareTracing=false;
        bool HairTransport=false;
        unsigned WorldCacheCascades=0,GlobalMaximumSteps=0;
        LumenScreenProbeFrame Screen;
        LumenReflectionRayFrame Reflection;
    };
    LumenRadianceInfo MakeLumenRadianceInfo(const LumenLightingIdentity&,const LumenRadianceTraceUpload&,
        const LumenRadianceInput&,unsigned worldCascades=0);
    inline unsigned LumenRadianceProbeCount(const LumenRadianceInfo& info)
    { return info.Reflection.Work.Y?0:info.Screen.Grid.Y?info.Screen.Grid.Y:unsigned(info.Probes.Origins.size()); }
    struct LumenRadianceSourceData { const void* Data=nullptr; std::size_t Bytes=0; };
    inline LumenRadianceSourceData LumenRadianceSources(const LumenRadianceTraceUpload& upload,const LumenRadianceInput& input)
    {
        if (input.Screen.Grid.Y) return {nullptr,std::size_t(input.Screen.Grid.Y)*sizeof(LumenProbeOrigin)};
        return upload.Probes.Origins.empty()?LumenRadianceSourceData{upload.Trace.Rays.data(),upload.Trace.Rays.size()*sizeof(LumenDistanceRay)}:
            LumenRadianceSourceData{upload.Probes.Origins.data(),upload.Probes.Origins.size()*sizeof(LumenProbeOrigin)};
    }
    inline LumenRadianceSourceData LumenRadiancePlacementData(const LumenRadianceInput& input)
    { return input.Reflection.Work.Y?LumenRadianceSourceData{&input.Reflection,sizeof(input.Reflection)}:LumenRadianceSourceData{&input.Screen,sizeof(input.Screen)}; }
    inline LumenRadianceSourceData LumenRadianceFrameData(const LumenRadianceInput& input)
    { return input.ProbeTrace.Sampling.X?LumenRadianceSourceData{&input.ProbeTrace,sizeof(input.ProbeTrace)}:LumenRadianceSourceData{&input.Trace,sizeof(input.Trace)}; }
    struct LumenRadianceTraceResult
    {
        LumenRadianceHandle Handle;
        LumenRadianceInfo Info;
        std::vector<LumenRadianceSample> Samples;
    };
    struct LumenRadianceCopy
    {
        LumenRadianceHandle Source;
        unsigned SourceProbe=0,DestinationProbe=0,ProbeCount=0;
    };
    struct LumenRadianceAssembly
    {
        LumenProbeRaySet Probes;
        std::vector<LumenRadianceCopy> Copies; // Ordered, complete destination coverage.
    };
    // Fixed equal-area directions are independent of probe IDs. Exact origin,
    // interval and lighting identity are still required before copying samples.
    bool ValidateLumenRadianceAssembly(const LumenRadianceAssembly&,std::string& error);
    bool ValidateLumenRadianceCopy(const LumenRadianceInfo&,const LumenRadianceAssembly&,const LumenRadianceCopy&,std::string& error);
    // Native SDF hits feed depth-gated radiance lookup without a CPU feedback
    // boundary. A hit is not evidence of cache coverage, and a cache with unknown
    // shadow rays retains its partial status. Misses never synthesize sky light.
    bool BuildLumenRadianceInput(const LumenLightingInfo& lighting,const LumenDistanceSceneInfo& distance,
        const std::vector<LumenSurfaceCapture>& captures,const std::vector<unsigned>& cardInstances,
        const LumenRadianceTraceUpload& upload,LumenRadianceInput& output,std::string& error);
    bool BuildLumenReflectionQuery(const ReflectionDesc& camera,const Techniques::ReflectionSettings& settings,
        unsigned width,unsigned height,LumenRadianceTraceUpload& output,std::string& error);
    bool ValidateLumenRadianceSamples(const LumenRadianceInfo& info,const LumenDistanceSceneInfo& distance,
        const std::vector<unsigned>& cardInstances,const std::vector<LumenRadianceSample>& samples,std::string& error);
    bool ValidateLumenRadianceOrigins(const LumenRadianceInfo& info,const std::vector<LumenProbeOrigin>& origins,std::string& error);
    // Reject forged layouts and incompatible sampling. Native callers additionally
    // require the same retained lighting snapshot before recording commands.
    bool BuildLumenWorldLookupFrame(const LumenRadianceInfo& cached,const LumenRadianceTraceUpload& query,
        const LumenRadianceInput& input,LumenWorldLookupFrame& output,std::string& error);
    LumenRadianceHandle NewLumenRadianceHandle() noexcept;
}
