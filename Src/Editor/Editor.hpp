#pragma once

#include "Editor/Demos.hpp"
#include "Editor/RenderAdapter.hpp"
#include "Renderer/ShaderPreparationProgress.h"
#include "Renderer/BakedLighting/ArvoLightmap.h"
#include "Editor/NeuralWorkspace.h"
#include <mutex>

#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Evans::Editor
{

class EditorCore
{
public:
    explicit EditorCore(std::filesystem::path contentRoot);
    ~EditorCore();

    EditorCore(const EditorCore&) = delete;
    EditorCore& operator=(const EditorCore&) = delete;

    void RegisterRenderAdapter(std::shared_ptr<IRenderAdapter> adapter);
    void SetPickHandler(PickHandler handler);

    void SelectDemo(DemoId demo);
    void SelectBackend(const std::string& backendId);
    void SetFillMode(FillMode mode);
    void SetRenderBuffer(Techniques::RenderBuffer buffer);
    void SetShadowsEnabled(bool enabled);
    void SetOutlineEnabled(bool enabled);
    void SetBigWorldSettings(const struct BigWorldSettings& settings);
    void SetTechniqueSettings(const Evans::Techniques::Settings& settings);
    void SetLight(std::size_t index,const Light& light);
    void SelectLight(int index);
    void SetShowLights(bool show);
    void SetContinuousRendering(bool enabled) noexcept { Settings_.ContinuousRendering=enabled; }
    void SetGpuPresentation(bool enabled) noexcept { Settings_.GpuPresentation=enabled; }
    float ViewportFramesPerSecond() const noexcept { return ViewportFPS_; }
    int SelectedLight() const noexcept { return SelectedLight_; }
    std::optional<Float2> LightScreenPosition(std::size_t index,const Rect& viewport) const;
    bool HandleLightPointer(float x,float y,const Rect& viewport,bool down);
    void SetLightRange(std::size_t index, float range);
    void AddLocalLight(LightType type);
    void RemoveLocalLight(std::size_t index);
    void SetTeapotRotationSpeed(float degreesPerSecond);
    void ResizeViewport(std::uint32_t width, std::uint32_t height);
    void SelectNode(SceneNodeId node);
    void HandleViewportPointer(float x, float y, const Rect& viewportBounds);
    void SetCameraMovementKey(char key, bool down) noexcept;
    void HandleCameraPointer(float x, float y, bool rotateButtonDown);
    void ClearCameraInput() noexcept;
    void HandleDemoPointer(float x,float y,bool insideViewport);

    // Tick never waits for background work. Completed jobs are collected here.
    void Tick(float deltaSeconds);
    void RequestRender();
    Renderer::ArvoBakeOptions& LightmapOptions() noexcept { return LightmapOptions_; }
    std::filesystem::path LightmapBakeDirectory() const;
    void StartLightmapBake(const Techniques::Settings& settings);
    void CancelLightmapBake();
    void LoadLightmapOutput(const Techniques::Settings& settings);
    void PollLightmapBake();
    bool IsLightmapBaking() const noexcept { return LightmapTask_ != nullptr; }
    std::string LightmapBakeMessage() const;
    void SaveLightmapPreferences() const;
    void LoadLightmapPreferences(const std::filesystem::path& contentRoot);
    NeuralWorkspace& NeuralRendering() noexcept { return *Neural_; }
    bool HasNeuralAsset() const { return HasLoadedSelectedDemo() && Scene_.Demo==DemoId::UtahTeapot && Neural_->HasAsset(); }

    [[nodiscard]] const SceneSnapshot& Scene() const noexcept { return Scene_; }
    [[nodiscard]] const ViewportSettings& Settings() const noexcept { return Settings_; }
    [[nodiscard]] const Image& ViewportImage() const noexcept { return ViewportImage_; }
    [[nodiscard]] std::uint32_t ViewportPathTracingSamples() const noexcept { return ViewportPathTracingSamples_; }
    [[nodiscard]] const GeometryStatistics& ViewportGeometry() const noexcept { return ViewportGeometry_; }
    [[nodiscard]] const std::string& ViewportBakedLightingStatus() const noexcept { return ViewportBakedLightingStatus_; }
    [[nodiscard]] const std::string& ViewportLumenGIStatus() const noexcept { return ViewportLumenGIStatus_; }
    [[nodiscard]] const std::string& ViewportVoxelGIStatus() const noexcept { return ViewportVoxelGIStatus_; }
    [[nodiscard]] const std::string& ViewportHairFurStatus() const noexcept { return ViewportHairFurStatus_; }
    [[nodiscard]] std::uint64_t ViewportRevision() const noexcept { return ViewportRevision_; }
    [[nodiscard]] const std::string& ViewportBackend() const noexcept { return ViewportBackend_; }
    [[nodiscard]] DemoId SelectedDemo() const noexcept { return SelectedDemo_; }
    [[nodiscard]] bool IsDemoLoading() const noexcept { return ActiveDemoLoad_ != nullptr; }
    [[nodiscard]] bool IsRendering() const noexcept { return ActiveJob_ != nullptr; }
    struct PreparationProgress
    {
        Renderer::ShaderPreparationStatus Shaders;
        double Seconds = 0;
        bool Active = false;
        bool WaitingForPreviousFrame = false;
    };
    [[nodiscard]] PreparationProgress PendingPreparation() const;
    [[nodiscard]] bool IsPickInProgress() const noexcept { return ActivePickJob_ != nullptr; }
    [[nodiscard]] const std::string& SelectedBackend() const noexcept { return SelectedBackend_; }
    [[nodiscard]] const std::string& StatusText() const noexcept { return StatusText_; }
    [[nodiscard]] std::uint64_t Generation() const noexcept { return Generation_; }
    [[nodiscard]] std::vector<BackendDescriptor> Backends() const;
    [[nodiscard]] std::uint32_t QueryMultisampleCounts() const;
    [[nodiscard]] std::uint32_t QueryWeightedTransparencySampleCounts() const;
    [[nodiscard]] bool QueryNaniteHardwareScheduling() const;
    [[nodiscard]] bool QueryLumenHardwareRayTracing() const;
    [[nodiscard]] bool QueryMaterialAnisotropicFiltering() const;
    [[nodiscard]] Techniques::Support QueryTechniqueSupport(Techniques::Technique technique) const;

private:
    std::unique_ptr<NeuralWorkspace> Neural_;
    struct LightmapTask
    {
        std::atomic<bool> Cancelled{false};
        mutable std::mutex Mutex;
        std::string Message;
        std::future<std::string> Future;
        DemoId Demo;
        std::uint64_t Revision=0,TransformRevision=0,LightingRevision=0;
        std::filesystem::path Directory;
        Techniques::LightmapSettings Atlas;
        Techniques::Settings Settings;
    };
    std::unique_ptr<LightmapTask> LightmapTask_;
    Renderer::ArvoBakeOptions LightmapOptions_;
    std::filesystem::path LightmapPreferences_;
    std::string LightmapMessage_;
    struct RenderJob
    {
        std::chrono::steady_clock::time_point Started=std::chrono::steady_clock::now();
        std::uint64_t Generation = 0;
        std::string BackendId;
        RenderDomain Domain = RenderDomain::RealtimeRenderer;
        CancellationSource Cancellation;
        std::shared_ptr<Renderer::ShaderPreparationProgress> Preparation =
            std::make_shared<Renderer::ShaderPreparationProgress>();
        std::future<RenderResult> Future;
        std::shared_ptr<const SceneSnapshot> NextWorldFrame;
    };

    struct DemoLoadResult
    {
        SceneSnapshot Scene;
        std::string Error;
    };

    struct DemoLoadJob
    {
        std::chrono::steady_clock::time_point Started = std::chrono::steady_clock::now();
        DemoId Demo = DemoId::Triangle;
        std::future<DemoLoadResult> Future;
    };

    struct PickJob
    {
        std::uint64_t Request = 0;
        std::uint64_t DemoRequest = 0;
        std::future<SceneNodeId> Future;
    };

    void InvalidateAndRender();
    void QueueRender();
    [[nodiscard]] bool UpdateCameraMovement(float deltaSeconds);
    void StartRender(std::shared_ptr<const SceneSnapshot> prepared={});
    void RetireActiveJob();
    void CollectFinishedJobs();
    void CollectRetiredJobs();
    [[nodiscard]] bool HasBlockingRetiredRenderJob(
        const std::string& backendId,
        RenderDomain domain) const noexcept;
    void StartDemoLoad(DemoId demo);
    void CollectFinishedDemoLoad();
    void RetirePickJob();
    void CollectFinishedPick();
    void CollectRetiredPicks();
    void ApplyResult(RenderResult result, std::uint64_t generation, const std::string& backendId);
    [[nodiscard]] bool HasLoadedSelectedDemo() const noexcept;
    [[nodiscard]] std::shared_ptr<IRenderAdapter> FindAdapter(const std::string& id) const;
    [[nodiscard]] BackendDescriptor FindDescriptor(const std::string& id) const;
    DemoCatalog Demos_;
    SceneSnapshot Scene_;
    ViewportSettings Settings_;
    struct DemoSettingsRestore
    {
        Techniques::Settings Techniques;
        FillMode Mode;
        bool OutlineEnabled;
    };
    std::optional<DemoSettingsRestore> BeforeDedicatedDemo_;
    std::unordered_map<std::string, std::shared_ptr<IRenderAdapter>> Adapters_;
    PickHandler PickHandler_;
    DemoId SelectedDemo_ = DemoId::Triangle;
    std::string SelectedBackend_ = "renderer.dx11";
    std::string StatusText_ = "No renderer adapter registered";
    Image ViewportImage_;
    GeometryStatistics ViewportGeometry_;
    std::string ViewportBakedLightingStatus_;
    std::string ViewportVoxelGIStatus_;
    std::string ViewportLumenGIStatus_;
    std::string ViewportHairFurStatus_;
    std::uint32_t ViewportPathTracingSamples_ = 0;
    std::string ViewportBackend_;
    std::uint64_t ViewportRevision_ = 0;
    std::chrono::steady_clock::time_point FPSWindowStart_{};
    unsigned FPSFrames_=0;
    float ViewportFPS_=0;
    std::uint64_t RenderHistoryEpoch_ = 0;
    std::uint64_t Generation_ = 1;
    std::uint64_t DemoRequest_ = 1;
    std::uint64_t AppliedDemoRequest_ = 1;
    std::uint64_t PickRequest_ = 0;
    bool RenderDirty_ = false;
    bool PendingRenderAfterLoad_ = false;
    std::array<bool, 4> CameraMovementKeys_{};
    int SelectedLight_=-1;
    bool LightDragging_=false;
    float LightPointerX_=0,LightPointerY_=0;
    bool CameraPointerActive_ = false;
    bool DemoPointerActive_=false;
    float PreviousDemoPointerX_=0,PreviousDemoPointerY_=0;
    float PreviousCameraPointerX_ = 0.0f;
    float PreviousCameraPointerY_ = 0.0f;
    std::unique_ptr<RenderJob> ActiveJob_;
    std::vector<std::unique_ptr<RenderJob>> RetiredJobs_;
    std::unique_ptr<DemoLoadJob> ActiveDemoLoad_;
    std::unique_ptr<PickJob> ActivePickJob_;
    std::vector<std::unique_ptr<PickJob>> RetiredPickJobs_;
};

} // namespace Evans::Editor
