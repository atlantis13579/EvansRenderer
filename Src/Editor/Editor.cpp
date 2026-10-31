#include "Editor/Editor.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Evans::Editor
{
namespace
{

template <typename Result>
[[nodiscard]] bool IsReady(const std::future<Result>& future)
{
    return future.valid() &&
           future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

template <typename Result>
void DrainFuture(std::future<Result>& future) noexcept
{
    if (!future.valid())
    {
        return;
    }
    try
    {
        (void)future.get();
    }
    catch (...)
    {
        // Shutdown only needs to join outstanding worker tasks.
    }
}

[[nodiscard]] std::string RenderingStatus(const BackendDescriptor& descriptor)
{
    return "Rendering with " + descriptor.DisplayName + "...";
}

[[nodiscard]] std::string DemoName(DemoId demo)
{
    const auto& descriptors = DemoCatalog::Descriptors();
    const auto found = std::find_if(
        descriptors.begin(), descriptors.end(),
        [demo](const DemoDescriptor& descriptor) { return descriptor.Id == demo; });
    return found == descriptors.end() ? "Unknown Demo" : std::string(found->Name);
}

[[nodiscard]] std::string LoadingStatus(DemoId demo, bool renderQueued)
{
    std::string status = "Loading " + DemoName(demo) + "...";
    if (renderQueued)
    {
        status += " Render queued.";
    }
    return status;
}

[[nodiscard]] float Length(const Float3& value)
{
    return std::sqrt(value.X * value.X + value.Y * value.Y + value.Z * value.Z);
}

[[nodiscard]] Float3 Subtract(const Float3& left, const Float3& right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

void Translate(Float3& value, const Float3& offset)
{
    value.X += offset.X;
    value.Y += offset.Y;
    value.Z += offset.Z;
}

} // namespace

EditorCore::EditorCore(std::filesystem::path contentRoot)
    : Demos_(std::move(contentRoot)), Scene_(Demos_.Load(DemoId::Triangle))
{
}

EditorCore::~EditorCore()
{
    if (ActiveJob_)
    {
        ActiveJob_->Cancellation.Cancel();
        DrainFuture(ActiveJob_->Future);
        ActiveJob_.reset();
    }
    for (const std::unique_ptr<RenderJob>& job : RetiredJobs_)
    {
        job->Cancellation.Cancel();
    }
    for (const std::unique_ptr<RenderJob>& job : RetiredJobs_)
    {
        DrainFuture(job->Future);
    }
    if (ActiveDemoLoad_)
    {
        DrainFuture(ActiveDemoLoad_->Future);
    }
    if (ActivePickJob_)
    {
        DrainFuture(ActivePickJob_->Future);
    }
    for (const std::unique_ptr<PickJob>& job : RetiredPickJobs_)
    {
        DrainFuture(job->Future);
    }
}

void EditorCore::RegisterRenderAdapter(std::shared_ptr<IRenderAdapter> adapter)
{
    if (!adapter)
    {
        throw std::invalid_argument("Cannot register a null render adapter");
    }
    const BackendDescriptor descriptor = adapter->Descriptor();
    if (descriptor.Id.empty())
    {
        throw std::invalid_argument("A render adapter must have a stable identifier");
    }
    Adapters_[descriptor.Id] = std::move(adapter);
    if (descriptor.Id == SelectedBackend_)
    {
        InvalidateAndRender();
    }
}

void EditorCore::SetPickHandler(PickHandler handler)
{
    ++PickRequest_;
    RetirePickJob();
    PickHandler_ = std::move(handler);
}

void EditorCore::SelectDemo(DemoId demo)
{
    if ((ActiveDemoLoad_ && SelectedDemo_ == demo) ||
        (!ActiveDemoLoad_ && HasLoadedSelectedDemo() && Scene_.Demo == demo))
    {
        return;
    }
    SelectedDemo_ = demo;
    ClearCameraInput();
    ++DemoRequest_;
    ++Generation_;
    ++PickRequest_;
    PendingRenderAfterLoad_ = false;
    RenderDirty_ = false;
    Settings_.SelectedNode = 0;
    ViewportImage_ = {};
    ViewportBackend_.clear();
    ++ViewportRevision_;
    RetireActiveJob();
    RetirePickJob();
    if (ActiveDemoLoad_)
    {
        StatusText_ = LoadingStatus(demo, false);
    }
    else
    {
        StartDemoLoad(demo);
    }
}

void EditorCore::SelectBackend(const std::string& backendId)
{
    const std::string canonicalBackend =
        backendId;
    (void)FindDescriptor(canonicalBackend);
    if (canonicalBackend == SelectedBackend_)
    {
        return;
    }
    SelectedBackend_ = canonicalBackend;
    InvalidateAndRender();
}

void EditorCore::SetFillMode(FillMode mode)
{
    if (Settings_.Mode != mode)
    {
        Settings_.Mode = mode;
        InvalidateAndRender();
    }
}

void EditorCore::SetShadowsEnabled(bool enabled)
{
    if (Settings_.ShadowsEnabled != enabled)
    {
        Settings_.ShadowsEnabled = enabled;
        InvalidateAndRender();
    }
}

void EditorCore::SetOutlineEnabled(bool enabled)
{
    if (Settings_.OutlineEnabled != enabled)
    {
        Settings_.OutlineEnabled = enabled;
        InvalidateAndRender();
    }
}

void EditorCore::SetTeapotRotationSpeed(float degreesPerSecond)
{
    Settings_.TeapotRotationDegreesPerSecond = std::clamp(degreesPerSecond, 0.0f, 360.0f);
}

void EditorCore::ResizeViewport(std::uint32_t width, std::uint32_t height)
{
    if (width == 0 || height == 0)
    {
        throw std::invalid_argument("The editor viewport must have non-zero dimensions");
    }
    if (Settings_.Width != width || Settings_.Height != height)
    {
        Settings_.Width = width;
        Settings_.Height = height;
        InvalidateAndRender();
    }
}

void EditorCore::SelectNode(SceneNodeId node)
{
    if (node != 0)
    {
        const auto found = std::find_if(
            Scene_.Nodes.begin(),
            Scene_.Nodes.end(),
            [node](const SceneNode& candidate)
            {
                return candidate.Id == node && candidate.Selectable;
            });
        if (found == Scene_.Nodes.end())
        {
            return;
        }
    }
    if (Settings_.SelectedNode != node)
    {
        Settings_.SelectedNode = node;
        InvalidateAndRender();
    }
}

void EditorCore::HandleViewportPointer(float x, float y, const Rect& viewportBounds)
{
    if (!PickHandler_ || viewportBounds.Width <= 0.0f || viewportBounds.Height <= 0.0f ||
        !viewportBounds.Contains(x, y) || !HasLoadedSelectedDemo())
    {
        return;
    }
    PickRequest request;
    request.ViewportX = x - viewportBounds.X;
    request.ViewportY = y - viewportBounds.Y;
    request.ViewportWidth = viewportBounds.Width;
    request.ViewportHeight = viewportBounds.Height;
    request.NormalizedX = request.ViewportX / viewportBounds.Width;
    request.NormalizedY = request.ViewportY / viewportBounds.Height;

    ++PickRequest_;
    RetirePickJob();
    const std::shared_ptr<const SceneSnapshot> scene =
        std::make_shared<SceneSnapshot>(Scene_);
    const PickHandler handler = PickHandler_;
    auto job = std::make_unique<PickJob>();
    job->Request = PickRequest_;
    job->DemoRequest = AppliedDemoRequest_;
    try
    {
        job->Future = std::async(
            std::launch::async,
            [handler, scene, request]() { return handler(*scene, request); });
    }
    catch (const std::exception& exception)
    {
        StatusText_ = std::string("Pick failed: ") + exception.what();
        return;
    }
    catch (...)
    {
        StatusText_ = "Pick failed: could not start background work";
        return;
    }
    ActivePickJob_ = std::move(job);
}

void EditorCore::SetCameraMovementKey(char key, bool down) noexcept
{
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(key))))
    {
    case 'w': CameraMovementKeys_[0] = down; break;
    case 'a': CameraMovementKeys_[1] = down; break;
    case 's': CameraMovementKeys_[2] = down; break;
    case 'd': CameraMovementKeys_[3] = down; break;
    default: break;
    }
}

void EditorCore::HandleCameraPointer(float x, float y, bool rotateButtonDown)
{
    if (!rotateButtonDown || !HasLoadedSelectedDemo())
    {
        CameraPointerActive_ = false;
        return;
    }
    if (!CameraPointerActive_)
    {
        CameraPointerActive_ = true;
        PreviousCameraPointerX_ = x;
        PreviousCameraPointerY_ = y;
        return;
    }

    const float deltaX = x - PreviousCameraPointerX_;
    const float deltaY = y - PreviousCameraPointerY_;
    PreviousCameraPointerX_ = x;
    PreviousCameraPointerY_ = y;
    if (deltaX == 0.0f && deltaY == 0.0f)
    {
        return;
    }

    Camera& camera = Scene_.SceneCamera;
    const Float3 direction = Subtract(camera.Target, camera.Position);
    const float radius = Length(direction);
    if (radius <= 1.0e-6f)
    {
        return;
    }
    float yaw = std::atan2(direction.X, direction.Z) - deltaX * 0.01f;
    float pitch = std::asin(std::clamp(direction.Y / radius, -1.0f, 1.0f));
    pitch = std::clamp(pitch - deltaY * 0.01f, -1.5f, 1.5f);
    const float horizontalRadius = radius * std::cos(pitch);
    camera.Target = {
        camera.Position.X + std::sin(yaw) * horizontalRadius,
        camera.Position.Y + std::sin(pitch) * radius,
        camera.Position.Z + std::cos(yaw) * horizontalRadius};
    ++Scene_.CameraRevision;
    InvalidateCameraAndRender();
}

void EditorCore::ClearCameraInput() noexcept
{
    CameraMovementKeys_.fill(false);
    CameraPointerActive_ = false;
}

void EditorCore::Tick(float deltaSeconds)
{
    CollectFinishedDemoLoad();
    CollectFinishedPick();
    CollectFinishedJobs();
    CollectRetiredJobs();
    CollectRetiredPicks();

    if (!HasLoadedSelectedDemo())
    {
        return;
    }
    const bool cameraChanged = UpdateCameraMovement(std::max(deltaSeconds, 0.0f));
    const std::uint64_t previousRevision = Scene_.Revision;
    Demos_.Update(
        Scene_,
        std::max(deltaSeconds, 0.0f),
        Settings_.TeapotRotationDegreesPerSecond);
    const bool sceneChanged = Scene_.Revision != previousRevision;

    if (cameraChanged)
    {
        InvalidateCameraAndRender();
    }
    if (sceneChanged && !cameraChanged)
    {
        if (ActiveJob_)
        {
            RenderDirty_ = true;
        }
        else
        {
            StartRender();
        }
    }
}

bool EditorCore::UpdateCameraMovement(float deltaSeconds)
{
    if (deltaSeconds <= 0.0f ||
        std::none_of(
            CameraMovementKeys_.begin(),
            CameraMovementKeys_.end(),
            [](bool down) { return down; }))
    {
        return false;
    }

    Camera& camera = Scene_.SceneCamera;
    Float3 viewForward = Subtract(camera.Target, camera.Position);
    const float viewForwardLength = Length(viewForward);
    if (viewForwardLength > 1.0e-6f)
    {
        viewForward.X /= viewForwardLength;
        viewForward.Y /= viewForwardLength;
        viewForward.Z /= viewForwardLength;
    }
    else
    {
        viewForward = {0.0f, 0.0f, 1.0f};
    }

    Float3 planarForward = viewForward;
    planarForward.Y = 0.0f;
    const float planarForwardLength = Length(planarForward);
    if (planarForwardLength > 1.0e-6f)
    {
        planarForward.X /= planarForwardLength;
        planarForward.Z /= planarForwardLength;
    }
    else
    {
        planarForward = {0.0f, 0.0f, 1.0f};
    }

    Float3 movement;
    if (CameraMovementKeys_[0])
    {
        movement.X += viewForward.X;
        movement.Y += viewForward.Y;
        movement.Z += viewForward.Z;
    }
    if (CameraMovementKeys_[2])
    {
        movement.X -= viewForward.X;
        movement.Y -= viewForward.Y;
        movement.Z -= viewForward.Z;
    }
    if (CameraMovementKeys_[1])
    {
        movement.X += planarForward.Z;
        movement.Z -= planarForward.X;
    }
    if (CameraMovementKeys_[3])
    {
        movement.X -= planarForward.Z;
        movement.Z += planarForward.X;
    }
    const float movementLength = Length(movement);
    if (movementLength <= 1.0e-6f)
    {
        return false;
    }

    const float radius = std::max(Length(Subtract(camera.Position, camera.Target)), 1.0e-4f);
    const float distance = radius * 1.5f * deltaSeconds;
    movement.X *= distance / movementLength;
    movement.Y *= distance / movementLength;
    movement.Z *= distance / movementLength;
    Translate(camera.Target, movement);
    Translate(camera.Position, movement);
    ++Scene_.CameraRevision;
    return true;
}

void EditorCore::RequestRender()
{
    ++Generation_;
    RenderDirty_ = false;
    RetireActiveJob();
    if (ActiveDemoLoad_)
    {
        PendingRenderAfterLoad_ = true;
        StatusText_ = LoadingStatus(SelectedDemo_, true);
        return;
    }
    if (!HasLoadedSelectedDemo())
    {
        return;
    }
    StartRender();
}

std::vector<BackendDescriptor> EditorCore::Backends() const
{
    std::vector<BackendDescriptor> result;
    result.reserve(Adapters_.size());
    const auto& catalog = StandardBackendCatalog();
    for (const BackendDescriptor& descriptor : catalog)
    {
        const auto adapter = Adapters_.find(descriptor.Id);
        if (adapter != Adapters_.end() && adapter->second->IsAvailable())
        {
            result.push_back(adapter->second->Descriptor());
        }
    }
    for (const auto& [id, adapter] : Adapters_)
    {
        if (!adapter->IsAvailable())
        {
            continue;
        }
        const auto found = std::find_if(
            result.begin(),
            result.end(),
            [&id](const BackendDescriptor& descriptor) { return descriptor.Id == id; });
        if (found == result.end())
        {
            result.push_back(adapter->Descriptor());
        }
    }
    return result;
}

void EditorCore::InvalidateAndRender()
{
    ++Generation_;
    RenderDirty_ = false;
    RetireActiveJob();
    if (ActiveDemoLoad_)
    {
        StatusText_ = LoadingStatus(SelectedDemo_, PendingRenderAfterLoad_);
        return;
    }
    if (!HasLoadedSelectedDemo())
    {
        return;
    }
    StartRender();
}

void EditorCore::InvalidateCameraAndRender()
{
    if (ActiveDemoLoad_ || !HasLoadedSelectedDemo())
    {
        return;
    }
    if (ActiveJob_)
    {
        RenderDirty_ = true;
        return;
    }
    StartRender();
}

void EditorCore::StartRender()
{
    if (ActiveDemoLoad_ || !HasLoadedSelectedDemo())
    {
        return;
    }
    if (ActiveJob_)
    {
        RenderDirty_ = true;
        return;
    }
    const std::shared_ptr<IRenderAdapter> adapter = FindAdapter(SelectedBackend_);
    if (!adapter)
    {
        StatusText_ = "No adapter registered for " + FindDescriptor(SelectedBackend_).DisplayName;
        return;
    }
    const BackendDescriptor descriptor = adapter->Descriptor();
    if (HasBlockingRetiredRenderJob(SelectedBackend_, descriptor.Domain))
    {
        RenderDirty_ = true;
        StatusText_ = "Waiting for the previous render frame to stop...";
        return;
    }
    RenderRequest request;
    request.Scene = std::make_shared<SceneSnapshot>(Scene_);
    request.Settings = Settings_;
    request.Generation = Generation_;

    auto job = std::make_unique<RenderJob>();
    job->Generation = Generation_;
    job->BackendId = SelectedBackend_;
    job->Domain = descriptor.Domain;
    const CancellationToken token = job->Cancellation.Token();
    try
    {
        job->Future = std::async(
            std::launch::async,
            [adapter, request = std::move(request), token]() mutable
            {
                if (token.IsCancellationRequested())
                {
                    return RenderResult{RenderStatus::Cancelled};
                }
                try
                {
                    return adapter->Render(request, token);
                }
                catch (const std::exception& exception)
                {
                    RenderResult result;
                    result.Status = RenderStatus::Failed;
                    result.Error = exception.what();
                    return result;
                }
                catch (...)
                {
                    RenderResult result;
                    result.Status = RenderStatus::Failed;
                    result.Error = "Unknown renderer exception";
                    return result;
                }
            });
    }
    catch (const std::exception& exception)
    {
        StatusText_ = std::string("Render could not start: ") + exception.what();
        return;
    }
    catch (...)
    {
        StatusText_ = "Render could not start: background worker creation failed";
        return;
    }
    StatusText_ = RenderingStatus(descriptor);
    ActiveJob_ = std::move(job);
}

void EditorCore::RetireActiveJob()
{
    if (!ActiveJob_)
    {
        return;
    }
    ActiveJob_->Cancellation.Cancel();
    RetiredJobs_.push_back(std::move(ActiveJob_));
}

void EditorCore::CollectFinishedJobs()
{
    if (!ActiveJob_ || !IsReady(ActiveJob_->Future))
    {
        return;
    }

    std::unique_ptr<RenderJob> completed = std::move(ActiveJob_);
    RenderResult result;
    try
    {
        result = completed->Future.get();
    }
    catch (const std::exception& exception)
    {
        result.Status = RenderStatus::Failed;
        result.Error = exception.what();
    }
    catch (...)
    {
        result.Status = RenderStatus::Failed;
        result.Error = "Unknown renderer future exception";
    }
    ApplyResult(std::move(result), completed->Generation, completed->BackendId);

    if (RenderDirty_)
    {
        RenderDirty_ = false;
        StartRender();
    }
}

void EditorCore::CollectRetiredJobs()
{
    auto iterator = RetiredJobs_.begin();
    while (iterator != RetiredJobs_.end())
    {
        RenderJob& job = **iterator;
        if (!IsReady(job.Future))
        {
            ++iterator;
            continue;
        }
        try
        {
            job.Future.get();
        }
        catch (...)
        {
            // Results from retired generations are intentionally discarded.
        }
        iterator = RetiredJobs_.erase(iterator);
    }
    if (!ActiveJob_ && RenderDirty_ && HasLoadedSelectedDemo())
    {
        const RenderDomain selectedDomain = FindDescriptor(SelectedBackend_).Domain;
        if (!HasBlockingRetiredRenderJob(SelectedBackend_, selectedDomain))
        {
            RenderDirty_ = false;
            StartRender();
        }
    }
}

bool EditorCore::HasBlockingRetiredRenderJob(
    const std::string& backendId,
    RenderDomain domain) const noexcept
{
    return std::any_of(
        RetiredJobs_.begin(),
        RetiredJobs_.end(),
        [&backendId, domain](const std::unique_ptr<RenderJob>& job)
        {
            return job->BackendId == backendId ||
                (job->Domain == RenderDomain::RealtimeRenderer &&
                 domain == RenderDomain::RealtimeRenderer);
        });
}

void EditorCore::StartDemoLoad(DemoId demo)
{
    auto job = std::make_unique<DemoLoadJob>();
    job->Demo = demo;
    const DemoCatalog catalog = Demos_;
    try
    {
        job->Future = std::async(
            std::launch::async,
            [catalog, demo]() mutable
            {
                DemoLoadResult result;
                try
                {
                    result.Scene = catalog.Load(demo);
                }
                catch (const std::exception& exception)
                {
                    result.Error = exception.what();
                }
                catch (...)
                {
                    result.Error = "Unknown scene loading exception";
                }
                return result;
            });
    }
    catch (const std::exception& exception)
    {
        StatusText_ = std::string("Scene load failed: ") + exception.what();
        return;
    }
    catch (...)
    {
        StatusText_ = "Scene load failed: could not start background work";
        return;
    }
    StatusText_ = LoadingStatus(demo, PendingRenderAfterLoad_);
    ActiveDemoLoad_ = std::move(job);
}

void EditorCore::CollectFinishedDemoLoad()
{
    if (!ActiveDemoLoad_ || !IsReady(ActiveDemoLoad_->Future))
    {
        return;
    }
    std::unique_ptr<DemoLoadJob> completed = std::move(ActiveDemoLoad_);
    DemoLoadResult result;
    try
    {
        result = completed->Future.get();
    }
    catch (const std::exception& exception)
    {
        result.Error = exception.what();
    }
    catch (...)
    {
        result.Error = "Unknown scene loading future exception";
    }
    if (completed->Demo != SelectedDemo_)
    {
        StartDemoLoad(SelectedDemo_);
        return;
    }
    if (!result.Error.empty() || result.Scene.Demo != completed->Demo)
    {
        PendingRenderAfterLoad_ = false;
        StatusText_ = result.Error.empty()
            ? "Scene load failed: loader returned the wrong demo"
            : "Scene load failed: " + result.Error;
        return;
    }

    Scene_ = std::move(result.Scene);
    AppliedDemoRequest_ = DemoRequest_;
    Settings_.SelectedNode = 0;
    PendingRenderAfterLoad_ = false;
    StartRender();
}

void EditorCore::RetirePickJob()
{
    if (ActivePickJob_)
    {
        RetiredPickJobs_.push_back(std::move(ActivePickJob_));
    }
}

void EditorCore::CollectFinishedPick()
{
    if (!ActivePickJob_ || !IsReady(ActivePickJob_->Future))
    {
        return;
    }
    std::unique_ptr<PickJob> completed = std::move(ActivePickJob_);
    SceneNodeId node = 0;
    std::string error;
    try
    {
        node = completed->Future.get();
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
    }
    catch (...)
    {
        error = "Unknown picking future exception";
    }
    if (completed->Request != PickRequest_ ||
        completed->DemoRequest != AppliedDemoRequest_ || !HasLoadedSelectedDemo())
    {
        return;
    }
    if (!error.empty())
    {
        StatusText_ = "Pick failed: " + error;
        return;
    }
    SelectNode(node);
}

void EditorCore::CollectRetiredPicks()
{
    auto iterator = RetiredPickJobs_.begin();
    while (iterator != RetiredPickJobs_.end())
    {
        if (!IsReady((*iterator)->Future))
        {
            ++iterator;
            continue;
        }
        DrainFuture((*iterator)->Future);
        iterator = RetiredPickJobs_.erase(iterator);
    }
}

bool EditorCore::HasLoadedSelectedDemo() const noexcept
{
    return ActiveDemoLoad_ == nullptr && AppliedDemoRequest_ == DemoRequest_ &&
        Scene_.Demo == SelectedDemo_;
}

void EditorCore::ApplyResult(
    RenderResult result,
    std::uint64_t generation,
    const std::string& backendId)
{
    if (generation != Generation_ || backendId != SelectedBackend_)
    {
        return;
    }
    switch (result.Status)
    {
    case RenderStatus::Completed:
        if (!result.Output.IsValid())
        {
            StatusText_ = "Renderer returned an invalid image";
            return;
        }
        ViewportImage_ = std::move(result.Output);
        ViewportBackend_ = backendId;
        ++ViewportRevision_;
        StatusText_ = "Frame completed with " + FindDescriptor(backendId).DisplayName;
        break;
    case RenderStatus::Cancelled:
        StatusText_ = "Render cancelled";
        break;
    case RenderStatus::Failed:
        StatusText_ = result.Error.empty() ? "Render failed" : "Render failed: " + result.Error;
        break;
    }
}

std::shared_ptr<IRenderAdapter> EditorCore::FindAdapter(const std::string& id) const
{
    const auto found = Adapters_.find(id);
    return found == Adapters_.end() ? nullptr : found->second;
}

BackendDescriptor EditorCore::FindDescriptor(const std::string& id) const
{
    const auto adapter = FindAdapter(id);
    if (adapter)
    {
        if (!adapter->IsAvailable())
        {
            throw std::invalid_argument(
                "Editor render backend is unavailable: " + id);
        }
        return adapter->Descriptor();
    }
    const auto& descriptors = StandardBackendCatalog();
    const auto found = std::find_if(
        descriptors.begin(),
        descriptors.end(),
        [&id](const BackendDescriptor& descriptor) { return descriptor.Id == id; });
    if (found == descriptors.end())
    {
        throw std::invalid_argument("Unknown editor render backend: " + id);
    }
    throw std::invalid_argument(
        "Editor render backend is unavailable on this platform or device: " + id);
}

} // namespace Evans::Editor
