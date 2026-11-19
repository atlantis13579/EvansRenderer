#pragma once

#include "Editor/Types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Evans::Editor
{

enum class RenderDomain
{
    RealtimeRenderer,
    SoftwareRasterizer
};

enum class HostPlatform
{
    Windows,
    Apple,
    Other
};

enum class BackendPlatform
{
    Any,
    Windows,
    Apple
};

struct BackendDescriptor
{
    std::string Id;
    std::string DisplayName;
    RenderDomain Domain = RenderDomain::RealtimeRenderer;
    BackendPlatform Platform = BackendPlatform::Any;
};

enum class RenderStatus
{
    Completed,
    Cancelled,
    Failed
};

class CancellationToken
{
public:
    [[nodiscard]] bool IsCancellationRequested() const noexcept;
    [[nodiscard]] const std::shared_ptr<std::atomic<bool>>& SharedState() const noexcept
    {
        return State_;
    }

private:
    explicit CancellationToken(std::shared_ptr<std::atomic<bool>> state);
    std::shared_ptr<std::atomic<bool>> State_;
    friend class CancellationSource;
};

class CancellationSource
{
public:
    CancellationSource();
    [[nodiscard]] CancellationToken Token() const;
    void Cancel() const noexcept;

private:
    std::shared_ptr<std::atomic<bool>> State_;
};

struct RenderRequest
{
    std::shared_ptr<const SceneSnapshot> Scene;
    ViewportSettings Settings;
    std::uint64_t Generation = 0;
    // Changes on discontinuities such as switching render backends or scenes.
    std::uint64_t HistoryEpoch = 0;
};

struct GeometryStatistics
{
    std::uint32_t LodReducedMeshes = 0;
    bool Available = false;
    std::uint32_t MeshDrawCalls = 0;
    std::uint32_t InstancedDrawCalls = 0;
    std::uint32_t IndirectDrawCalls = 0;
    std::uint64_t GpuCandidateInstances = 0;
    std::uint32_t VisibilityRasterDraws = 0, VisibilityMaterialDraws = 0;
    std::uint32_t HybridPasses = 0;
    std::uint32_t SubsurfacePasses = 0;
    std::uint32_t VoxelPasses = 0;
    std::uint32_t HairPasses = 0;
    std::uint32_t ParallaxDraws = 0;
    std::uint32_t DisplacedMeshes=0,DisplacedVertices=0,DisplacementPreparationDispatches=0;
    std::uint64_t DisplacementGeneration=0;
    std::uint32_t TessellatedMeshes=0,TessellatedVertices=0,TessellationPreparationDispatches=0,GeometryPreparationDispatches=0;
    std::uint64_t PreparedGeometryGeneration=0;
    std::uint32_t VirtualMeshCount=0,VirtualNodeCount=0;
    std::uint64_t VirtualResidentPages=0,VirtualPendingPages=0,VirtualUploadedPages=0,VirtualCopiedPages=0,VirtualBootstrapPages=0;
    std::uint64_t VirtualUploadedBytes=0,VirtualCopiedBytes=0,VirtualPoolBytes=0,VirtualGeometryGeneration=0;
    std::uint32_t NaniteMeshCount=0,NaniteSelectedClusters=0,NanitePendingPages=0,NaniteUploadedPages=0,NaniteGeometryAssemblyCount=0,NaniteRasterExportCount=0,NaniteOcclusionQueries=0,NaniteOcclusionPhases=0;
    std::uint64_t NaniteGeometryGeneration=0;
    std::uint32_t NaniteMaterialBins=0;
    std::uint32_t RenderWidth=0,RenderHeight=0,TSRPasses=0;
    std::uint64_t TSRSerial=0;
    bool TSRHistoryValid=false;
    std::uint32_t TSRForegroundPixels=0,TSRValidMotionPixels=0,TSRAccumulatedPixels=0;
    std::uint64_t VirtualShadowSerial=0;
    std::uint32_t VirtualShadowRequestedPages=0,VirtualShadowResidentPages=0,VirtualShadowRenderedPages=0,VirtualShadowCopiedPages=0,VirtualShadowDrawCalls=0;
    std::uint64_t NaniteReadbackBytes=0;
    std::uint32_t NaniteHostRebuiltVertices=0;
    std::uint32_t NaniteScenePageBudget=0,NaniteFamilyCount=0,NaniteViewCount=0,NaniteBudgetBlockedGroups=0,NaniteCookRequests=0;
    std::uint64_t NaniteRootPages=0,NaniteAllocatedPages=0,NaniteCookSourceBytes=0,NanitePreparationMicroseconds=0;
    bool NaniteBudgetLimited=false;
    double GpuFrameMilliseconds=0;
    bool GpuFrameTimingAvailable=false;
    std::uint32_t MeshShaderDispatches=0,NaniteHardwareDispatches=0,LumenHardwareRaySlots=0;
    std::uint32_t RasterWorkerThreads = 0;
    std::uint64_t RasterComputeGroups = 0, RasterComputeInvocations = 0;
    double RasterPreparationMilliseconds = 0, RasterExecutionMilliseconds = 0;
    std::string RasterExecutionPath;
};

struct RenderResult
{
    RenderStatus Status = RenderStatus::Failed;
    Image Output;
    std::string Error;
    bool NeedsTemporalFrame = false;
    std::uint32_t PathTracingSamples = 0;
    GeometryStatistics Geometry;
    std::string BakedLightingStatus;
    std::string VoxelGIStatus;
            std::string LumenGIStatus;
    std::string HairFurStatus;
    bool HairGeometryPending = false;
};

class IRenderAdapter
{
public:
    virtual ~IRenderAdapter() = default;
    [[nodiscard]] virtual BackendDescriptor Descriptor() const = 0;
    [[nodiscard]] virtual bool IsAvailable() const { return true; }
    [[nodiscard]] virtual Techniques::Support QueryTechniqueSupport(Techniques::Technique technique) const
    {
        return Techniques::QuerySupport(technique, Techniques::BackendFromId(Descriptor().Id));
    }
    [[nodiscard]] virtual std::uint32_t QueryMultisampleCounts() const { return 1; }
    [[nodiscard]] virtual std::uint32_t QueryWeightedTransparencySampleCounts() const { return 0; }
    [[nodiscard]] virtual bool QueryNaniteHardwareScheduling() const { return false; }
    [[nodiscard]] virtual bool QueryLumenHardwareRayTracing() const { return false; }
    [[nodiscard]] virtual bool QueryMaterialAnisotropicFiltering() const { return false; }
    [[nodiscard]] virtual std::uint64_t QuerySceneStorageBlockBytes() const { return 0; }
    [[nodiscard]] virtual RenderResult Render(
        const RenderRequest& request,
        const CancellationToken& cancellation) = 0;
};

using RenderCallback = std::function<RenderResult(const RenderRequest&, const CancellationToken&)>;

class CallbackRenderAdapter final : public IRenderAdapter
{
public:
    CallbackRenderAdapter(BackendDescriptor descriptor, RenderCallback callback);
    [[nodiscard]] BackendDescriptor Descriptor() const override;
    [[nodiscard]] RenderResult Render(
        const RenderRequest& request,
        const CancellationToken& cancellation) override;

private:
    BackendDescriptor Descriptor_;
    RenderCallback Callback_;
};

[[nodiscard]] const std::vector<BackendDescriptor>& StandardBackendCatalog();
[[nodiscard]] HostPlatform CurrentHostPlatform() noexcept;
[[nodiscard]] bool IsBackendSupportedOnPlatform(
    const BackendDescriptor& descriptor,
    HostPlatform platform) noexcept;

struct PickRequest
{
    float ViewportX = 0.0f;
    float ViewportY = 0.0f;
    float ViewportWidth = 0.0f;
    float ViewportHeight = 0.0f;
    float NormalizedX = 0.0f;
    float NormalizedY = 0.0f;
};

using PickHandler = std::function<SceneNodeId(const SceneSnapshot&, const PickRequest&)>;

} // namespace Evans::Editor
