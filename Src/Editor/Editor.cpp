#include "Techniques/LightRange.h"
#include "Renderer/Types.h"
#include "Renderer/Lighting/ForwardPlus.h"
#include "Renderer/RayTracing/RayTracedShadowScene.h"
#include "Renderer/Pipelines/SceneStorageBudget.h"
#include "Editor/Editor.hpp"
#include "Editor/BigWorld.hpp"

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

void SelectMaterialLibrary(Techniques::Settings& settings, SceneSnapshot& scene)
{
    if (scene.AlternateMaterials.empty())
    {
        if (settings.Shading == Techniques::ShadingModel::PBR) settings.Shading = Techniques::ShadingModel::Phong;
        return;
    }
    const bool pbr = settings.Shading == Techniques::ShadingModel::PBR;
    if (scene.Demo == DemoId::SponzaAtrium)
    {
        using Techniques::ReflectionMode;
        auto& mode = settings.Reflections.Mode;
        if (mode != ReflectionMode::Cubemap) scene.UsingDefaultPbrReflection = false;
        if (pbr && !scene.UsingPbrMaterials && mode == ReflectionMode::None)
        {
            // Metals in the shaded arcade need incoming specular sky radiance.
            mode = ReflectionMode::Cubemap;
            scene.UsingDefaultPbrReflection = true;
        }
        else if (!pbr && scene.UsingDefaultPbrReflection)
        {
            mode = ReflectionMode::None;
            scene.UsingDefaultPbrReflection = false;
        }
    }
    scene.UsePbrMaterials(pbr);
}

void ConstrainSponzaAtmosphere(Techniques::Settings& settings,const SceneSnapshot& scene)
{
    if (scene.Demo!=DemoId::SponzaAtrium) return;
    settings.VolumetricFog.Enabled=false;
    settings.VolumetricLighting.Enabled=false;
    settings.LightShafts.Enabled=false;
    settings.Reflections.Atmosphere.Enabled=false;
    settings.Reflections.Planar.Enabled=false;
    settings.GlobalIllumination.Lumen.SkyRadiance={0,0,0};
    settings.GlobalIllumination.Lumen.Environment.Enabled=false;
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
    SelectedLight_=-1; LightDragging_=false;
    ++RenderHistoryEpoch_;
    SelectedDemo_ = demo;
    ClearCameraInput();
    DemoPointerActive_=false;
    ++DemoRequest_;
    ++Generation_;
    ++PickRequest_;
    PendingRenderAfterLoad_ = false;
    RenderDirty_ = false;
    Settings_.SelectedNode = 0;
    ViewportImage_ = {};
    FPSWindowStart_={};FPSFrames_=0;ViewportFPS_=0;
    ViewportGeometry_ = {};
    ViewportBakedLightingStatus_.clear();
    ViewportVoxelGIStatus_.clear(); ViewportLumenGIStatus_.clear();
    ViewportHairFurStatus_.clear();
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
    if (!DemoCatalog::SupportsBackend(SelectedDemo_,Techniques::BackendFromId(canonicalBackend))) return;
    if (canonicalBackend == SelectedBackend_)
    {
        return;
    }
    ++RenderHistoryEpoch_;
    SelectedBackend_ = canonicalBackend;
    FPSWindowStart_={};FPSFrames_=0;ViewportFPS_=0;
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

void EditorCore::SetBigWorldSettings(const BigWorldSettings& settings)
{
    if (!Scene_.World || Scene_.Demo!=DemoId::BigWorld) return;
    ConfigureBigWorld(Scene_,settings);
    ++RenderHistoryEpoch_;
    // Publish the in-flight frame, then render the most recent slider values.
    QueueRender();
}

void EditorCore::SetTechniqueSettings(const Evans::Techniques::Settings& settings)
{
    auto sanitized = Evans::Techniques::SanitizeSettings(settings);
    if (sanitized.Shadows==Techniques::ShadowRepresentation::RayTraced &&
        QueryTechniqueSupport(Techniques::Technique::RayTracedShadows).State==Techniques::Availability::SceneLimit)
        sanitized.Shadows=Techniques::ShadowRepresentation::SingleMap;
    if (const auto adapter=FindAdapter(SelectedBackend_); adapter && adapter->QuerySceneStorageBlockBytes())
    {
        sanitized=Renderer::ConstrainSceneTriangleStorage(sanitized,Scene_.TriangleCountAtLoad,adapter->QuerySceneStorageBlockBytes());
    }
    if (!Scene_.Environment)
    {
        sanitized.Reflections.Skybox=false;
        sanitized.Reflections.DiffuseIBL=0;
        if (Techniques::UsesEnvironmentCapture(sanitized.Reflections)) sanitized.Reflections.Mode=Techniques::ReflectionMode::None;
    }
    if (Scene_.Demo!=DemoId::FurCube) sanitized.Hair.Enabled=false;
    if (Scene_.Demo!=DemoId::SponzaAtrium && (sanitized.GlobalIllumination.Mode==Techniques::GlobalIlluminationMode::Lightmaps ||
        sanitized.GlobalIllumination.Mode==Techniques::GlobalIlluminationMode::BakedProbes)) sanitized.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::None;
    if (Scene_.Demo==DemoId::SponzaAtrium && sanitized.GlobalIllumination.Mode==Techniques::GlobalIlluminationMode::Lightmaps &&
        !DemoCatalog::SupportsSponzaLightmaps(sanitized))
        sanitized.GlobalIllumination.Mode=QueryTechniqueSupport(Techniques::Technique::LightProbes).Available() ?
            Techniques::GlobalIlluminationMode::BakedProbes : Techniques::GlobalIlluminationMode::None;
    if (Scene_.Demo!=DemoId::StanfordDragon)
    {
        sanitized.Subsurface.Enabled=false; sanitized.Refraction.Enabled=false;
        sanitized.Transparency.Mode=Techniques::TransparencyMode::Sorted;
        if (Scene_.Demo!=DemoId::BigWorld) sanitized.Reflections.Planar.Enabled=false;
    }
    SelectMaterialLibrary(sanitized, Scene_);
    if (sanitized.Shading!=Techniques::ShadingModel::PBR && !Techniques::UsesPathTracing(sanitized.Pipeline))
        sanitized.AreaLights.Enabled=false;
    if (!DemoCatalog::IsVolumeDemo(Scene_.Demo))
    {
        sanitized.VolumetricFog.Enabled=false; sanitized.VolumetricLighting.Enabled=false; sanitized.LightShafts.Enabled=false;
    }
    if (Scene_.Demo==DemoId::SponzaAtrium && sanitized.AreaLights.Enabled!=Settings_.Techniques.AreaLights.Enabled)
    {
        // Keep the area emitters synchronized with the selected technique.
        for (auto& light:Scene_.Lights) if (light.Type==LightType::Area) light.Enabled=sanitized.AreaLights.Enabled;
        ++Scene_.LightingRevision; ++RenderHistoryEpoch_;
    }
    ConstrainSponzaAtmosphere(sanitized,Scene_);
    if (Settings_.Techniques != sanitized)
    {
        Settings_.Techniques = sanitized;
        InvalidateAndRender();
    }
}

void EditorCore::AddLocalLight(LightType type)
{
    const bool pathLights = Techniques::UsesPathTracing(Settings_.Techniques.Pipeline) && QueryTechniqueSupport(Techniques::Technique::PathTracing).Available();
    const auto capacity = pathLights || Techniques::UsesLightLists(Settings_.Techniques.Pipeline) &&
        QueryTechniqueSupport(Settings_.Techniques.Pipeline == Techniques::RenderingPipeline::ClusteredForward ?
            Techniques::Technique::ClusteredForward : Settings_.Techniques.Pipeline == Techniques::RenderingPipeline::TiledDeferred ?
            Techniques::Technique::TiledDeferred : Techniques::Technique::ForwardPlus).Available() ? Renderer::MaximumForwardPlusLights : Renderer::MaximumSceneLights;
    if (!HasLoadedSelectedDemo() || (type != LightType::Point && type != LightType::Area) ||
        Scene_.Lights.size() >= capacity) return;
    const auto& camera = Scene_.SceneCamera;
    Light light;
    light.Type = type; light.Position = camera.Position;
    if (type==LightType::Area) light.Enabled=Settings_.Techniques.AreaLights.Enabled;
    light.Direction = {camera.Target.X - camera.Position.X, camera.Target.Y - camera.Position.Y, camera.Target.Z - camera.Position.Z};
    const float distanceSquared = light.Direction.X * light.Direction.X + light.Direction.Y * light.Direction.Y + light.Direction.Z * light.Direction.Z;
    constexpr float InitialTargetIrradiance = 0.5f;
    light.Intensity = std::max(distanceSquared, 1.0f) * InitialTargetIrradiance;
    light.CastShadows = false;
    const float d=std::sqrt(std::max(distanceSquared,0.0001f));
    light.Position={camera.Position.X+light.Direction.X/d*std::max(2.0f,d*0.3f),camera.Position.Y+light.Direction.Y/d*std::max(2.0f,d*0.3f),camera.Position.Z+light.Direction.Z/d*std::max(2.0f,d*0.3f)};
    Scene_.Lights.push_back(light); SelectLight(int(Scene_.Lights.size()-1));
    ++Scene_.LightingRevision; ++RenderHistoryEpoch_;
    InvalidateAndRender();
}

void EditorCore::RemoveLocalLight(std::size_t index)
{
    if (!HasLoadedSelectedDemo() || index >= Scene_.Lights.size() || Scene_.Lights[index].Type == LightType::Directional) return;
    Scene_.Lights.erase(Scene_.Lights.begin() + index);
    SelectLight(-1);
    ConstrainSponzaAtmosphere(Settings_.Techniques,Scene_);
    ++Scene_.LightingRevision; ++RenderHistoryEpoch_;
    InvalidateAndRender();
}

namespace
{
Float3 LightSub(Float3 a,Float3 b) { return {a.X-b.X,a.Y-b.Y,a.Z-b.Z}; }
Float3 LightUnit(Float3 a) { float n=std::sqrt(a.X*a.X+a.Y*a.Y+a.Z*a.Z);return n>0.00001f?Float3{a.X/n,a.Y/n,a.Z/n}:Float3{0,0,-1}; }
Float3 LightCross(Float3 a,Float3 b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
float LightDot(Float3 a,Float3 b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
}
void EditorCore::SelectLight(int index)
{
    ++PickRequest_; RetirePickJob();
    SelectedLight_=index>=0 && std::size_t(index)<Scene_.Lights.size()?index:-1;
    if (Settings_.SelectedNode) {Settings_.SelectedNode=0;InvalidateAndRender();}
}
void EditorCore::SetLight(std::size_t index,const Light& input)
{
    if (index>=Scene_.Lights.size() || !HasLoadedSelectedDemo()) return;
    Light light=input; light.EditorOverride=true;
    const auto finite=[](Float3 p){return std::isfinite(p.X)&&std::isfinite(p.Y)&&std::isfinite(p.Z);};
    if (!finite(light.Position)||!finite(light.Direction)||!std::isfinite(light.Intensity)) return;
    light.Intensity=std::max(0.0f,light.Intensity);light.Range=Techniques::CanonicalLightRange(light.Range);
    light.Direction=LightUnit(light.Direction);
    Scene_.Lights[index]=light;
    ++Scene_.LightingRevision; ++RenderHistoryEpoch_; InvalidateAndRender();
}
void EditorCore::SetShowLights(bool show)
{
    Settings_.ShowLights=show;
    if (!show) LightDragging_=false;
}
std::optional<Float2> EditorCore::LightScreenPosition(std::size_t index,const Rect& viewport) const
{
    if (index>=Scene_.Lights.size() || viewport.Height<=0) return {};
    const auto& camera=Scene_.SceneCamera;const auto forward=LightUnit(LightSub(camera.Target,camera.Position));
    const auto right=LightUnit(LightCross(forward,{0,1,0})),up=LightCross(right,forward);
    const auto& light=Scene_.Lights[index]; auto position=light.Position;
    if (light.Type==LightType::Directional)
    { const float radius=std::sqrt(LightDot(LightSub(camera.Target,camera.Position),LightSub(camera.Target,camera.Position)))*0.35f;
      position={camera.Target.X-light.Direction.X*radius,camera.Target.Y-light.Direction.Y*radius,camera.Target.Z-light.Direction.Z*radius}; }
    const auto delta=LightSub(position,camera.Position);const float depth=LightDot(delta,forward);
    if (depth<=0.01f) return {};
    const float scale=viewport.Height/(2*std::tan(camera.VerticalFieldOfViewDegrees*3.14159265f/360));
    const Float2 result{viewport.X+viewport.Width*0.5f+LightDot(delta,right)*scale/depth,viewport.Y+viewport.Height*0.5f-LightDot(delta,up)*scale/depth};
    return viewport.Contains(result.X,result.Y)?std::optional<Float2>(result):std::nullopt;
}
bool EditorCore::HandleLightPointer(float x,float y,const Rect& viewport,bool down)
{
    if (!Settings_.ShowLights) { LightDragging_=false; return false; }
    if (!down) { const bool was=LightDragging_; LightDragging_=false;return was; }
    if (!LightDragging_)
    {
        float best=16*16;int selected=-1;
        for (std::size_t i=0;i<Scene_.Lights.size();++i) if (auto p=LightScreenPosition(i,viewport))
        { const float d=(p->X-x)*(p->X-x)+(p->Y-y)*(p->Y-y);if (d<best) {best=d;selected=int(i);} }
        if (selected<0) return false;
        SelectLight(selected);LightDragging_=true;LightPointerX_=x;LightPointerY_=y;return true;
    }
    if (SelectedLight_<0 || std::size_t(SelectedLight_)>=Scene_.Lights.size()) return false;
    auto light=Scene_.Lights[SelectedLight_];const auto& camera=Scene_.SceneCamera;
    const auto forward=LightUnit(LightSub(camera.Target,camera.Position)),right=LightUnit(LightCross(forward,{0,1,0})),up=LightCross(right,forward);
    const float depth=std::max(1.0f,LightDot(LightSub(light.Position,camera.Position),forward));
    float scale=2*std::tan(camera.VerticalFieldOfViewDegrees*3.14159265f/360)*depth/std::max(1.0f,viewport.Height);
    if (light.Type==LightType::Directional) scale=0.005f;
    const float dx=(x-LightPointerX_)*scale,dy=-(y-LightPointerY_)*scale;
    auto& p=light.Type==LightType::Directional?light.Direction:light.Position;
    const float sign=light.Type==LightType::Directional?-1.0f:1.0f;
    p={p.X+sign*(right.X*dx+up.X*dy),p.Y+sign*(right.Y*dx+up.Y*dy),p.Z+sign*(right.Z*dx+up.Z*dy)};
    LightPointerX_=x;LightPointerY_=y;SetLight(std::size_t(SelectedLight_),light);return true;
}

void EditorCore::SetRenderBuffer(Techniques::RenderBuffer buffer)
{ if (Settings_.Buffer!=buffer) {Settings_.Buffer=buffer; InvalidateAndRender();} }

void EditorCore::SetLightRange(std::size_t index, float range)
{
    if (!HasLoadedSelectedDemo() || index >= Scene_.Lights.size() || Scene_.Lights[index].Type == LightType::Directional) return;
    range = Techniques::CanonicalLightRange(range);
    if (Scene_.Lights[index].Range == range) return;
    Scene_.Lights[index].Range = range;
    // Invalidate illumination snapshots while retaining resident raster geometry.
    ++Scene_.LightingRevision;
    ++RenderHistoryEpoch_;
    InvalidateAndRender();
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
    if (HandleLightPointer(x,y,viewportBounds,true)) return;
    SelectLight(-1);
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
    const bool orbit=Scene_.Demo==DemoId::FurCube;
    const Float3 direction = orbit ? Subtract(camera.Position,camera.Target) : Subtract(camera.Target,camera.Position);
    const float radius = Length(direction);
    if (radius <= 1.0e-6f)
    {
        return;
    }
    float yaw = std::atan2(direction.X, direction.Z) - deltaX * 0.01f;
    float pitch = std::asin(std::clamp(direction.Y / radius, -1.0f, 1.0f));
    pitch = std::clamp(pitch - deltaY * 0.01f, -1.5f, 1.5f);
    const float horizontalRadius = radius * std::cos(pitch);
    const Float3 offset{std::sin(yaw)*horizontalRadius,std::sin(pitch)*radius,std::cos(yaw)*horizontalRadius};
    if (orbit)
        camera.Position={camera.Target.X+offset.X,camera.Target.Y+offset.Y,camera.Target.Z+offset.Z};
    else
        camera.Target={camera.Position.X+offset.X,camera.Position.Y+offset.Y,camera.Position.Z+offset.Z};
    ++Scene_.CameraRevision;
    QueueRender();
}

void EditorCore::ClearCameraInput() noexcept
{
    CameraMovementKeys_.fill(false);
    CameraPointerActive_ = false;
}

void EditorCore::HandleDemoPointer(float x,float y,bool insideViewport)
{
    if (!insideViewport || !HasLoadedSelectedDemo() || Scene_.Demo!=DemoId::FurCube)
    { DemoPointerActive_=false; return; }
    if (!DemoPointerActive_)
    { DemoPointerActive_=true; PreviousDemoPointerX_=x; PreviousDemoPointerY_=y; return; }
    const float dx=x-PreviousDemoPointerX_,dy=y-PreviousDemoPointerY_;
    PreviousDemoPointerX_=x; PreviousDemoPointerY_=y;
    if (dx==0 && dy==0) return;
    for (auto& node:Scene_.Nodes)
    {
        node.LocalTransform.RotationDegrees.Y=std::fmod(node.LocalTransform.RotationDegrees.Y+dx*0.45f,360.0f);
        node.LocalTransform.RotationDegrees.X=std::clamp(node.LocalTransform.RotationDegrees.X+dy*0.35f,-70.0f,70.0f);
    }
    ++Scene_.TransformRevision;
    QueueRender();
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
    const float elapsed = std::isfinite(deltaSeconds) ? std::max(deltaSeconds, 0.0f) : 0.0f;
    Scene_.SampleTimeSeconds += elapsed;
    const bool cameraChanged = UpdateCameraMovement(elapsed);
    const std::uint64_t previousRevision = Scene_.TransformRevision;
    Demos_.Update(
        Scene_,
        elapsed,
        Settings_.TeapotRotationDegreesPerSecond);
    const bool sceneChanged = Scene_.TransformRevision != previousRevision;

    if (cameraChanged)
    {
        QueueRender();
    }
    const auto& techniques=Settings_.Techniques;
    const bool animated=elapsed>0 && ((Scene_.World && (Scene_.World->Settings.DayNight || Scene_.World->Settings.AnimateWater || Scene_.World->Settings.AnimateVegetation || Scene_.World->Settings.Weather!=WorldWeather::Clear)) || (Techniques::UsesHairFur(techniques.Hair) && techniques.Hair.Animate) ||
        (techniques.VolumetricFog.Enabled && techniques.VolumetricFog.MediumMode==Techniques::VolumeMediumMode::Clouds &&
         techniques.VolumetricFog.Clouds.Animate && techniques.VolumetricFog.Clouds.AnimationSpeed>0));
    if ((sceneChanged || animated || (Settings_.ContinuousRendering && elapsed>0)) && !cameraChanged)
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
    // The renderer owns an immutable snapshot. Advance the next simulation
    // frame while that snapshot is drawing, at most once per in-flight frame.
    if (ActiveJob_ && RenderDirty_ && Scene_.World && elapsed>0 && !ActiveJob_->NextWorldFrame &&
        (Scene_.SampleTimeSeconds-Scene_.World->LastUpdate>=BigWorldUpdateInterval || Scene_.CameraRevision!=Scene_.World->CameraRevision))
    {
        UpdateBigWorld(Scene_,Settings_.Techniques);
        ActiveJob_->NextWorldFrame=std::make_shared<SceneSnapshot>(Scene_);
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
    const float distance = (Scene_.World?Scene_.World->Settings.CameraSpeed:radius*1.5f)*deltaSeconds;
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
        if (!DemoCatalog::SupportsBackend(SelectedDemo_,Techniques::BackendFromId(descriptor.Id))) continue;
        const auto adapter = Adapters_.find(descriptor.Id);
        if (adapter != Adapters_.end() && adapter->second->IsAvailable())
        {
            result.push_back(adapter->second->Descriptor());
        }
    }
    for (const auto& [id, adapter] : Adapters_)
    {
        if (!adapter->IsAvailable() || !DemoCatalog::SupportsBackend(SelectedDemo_,Techniques::BackendFromId(id)))
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

void EditorCore::QueueRender()
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

EditorCore::PreparationProgress EditorCore::PendingPreparation() const
{
    PreparationProgress progress;
    const auto now = std::chrono::steady_clock::now();
    if (ActiveDemoLoad_)
    {
        progress.Active = true;
        progress.Seconds = std::chrono::duration<double>(now - ActiveDemoLoad_->Started).count();
        return progress;
    }
    const RenderJob* job = ActiveJob_.get();
    if (!job && RenderDirty_)
    {
        const auto domain = FindDescriptor(SelectedBackend_).Domain;
        for (const auto& retired : RetiredJobs_)
            if (retired->BackendId == SelectedBackend_ ||
                (retired->Domain == RenderDomain::RealtimeRenderer && domain == RenderDomain::RealtimeRenderer))
            {
                job = retired.get();
                progress.WaitingForPreviousFrame = true;
                break;
            }
    }
    if (job)
    {
        progress.Active = true;
        progress.Seconds = std::chrono::duration<double>(now - job->Started).count();
        progress.Shaders = job->Preparation->Read();
    }
    return progress;
}

void EditorCore::StartRender(std::shared_ptr<const SceneSnapshot> prepared)
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
    if (!prepared)
    {
        if (Scene_.World) UpdateBigWorld(Scene_,Settings_.Techniques);
        prepared=std::make_shared<SceneSnapshot>(Scene_);
    }
    RenderRequest request;
    request.Scene = std::move(prepared);
    request.Settings = Settings_;
    request.Generation = Generation_;
    request.HistoryEpoch = RenderHistoryEpoch_;

    auto job = std::make_unique<RenderJob>();
    job->Generation = Generation_;
    job->BackendId = SelectedBackend_;
    job->Domain = descriptor.Domain;
    const CancellationToken token = job->Cancellation.Token();
    try
    {
        job->Future = std::async(
            std::launch::async,
            [adapter, request = std::move(request), token, preparation = job->Preparation]() mutable
            {
                const Renderer::ShaderPreparationScope preparationScope(preparation);
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
        auto prepared=std::move(completed->NextWorldFrame);
        if (prepared && prepared->World!=Scene_.World) prepared.reset();
        // Keep the camera paired with the water and vegetation LOD it selected.
        // A newer input still gets its own frame, including after movement stops.
        if (prepared && prepared->CameraRevision!=Scene_.CameraRevision) RenderDirty_=true;
        StartRender(std::move(prepared));
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

    const auto& environmentPreferences=!Scene_.Environment && BeforeDedicatedDemo_ ?
        BeforeDedicatedDemo_->Techniques.Reflections : Settings_.Techniques.Reflections;
    const auto environmentMode=environmentPreferences.Mode;
    const bool environmentEnabled=Techniques::UsesEnvironmentCapture(environmentPreferences);
    const auto environmentIntensity=environmentPreferences.EnvironmentIntensity;
    Scene_ = std::move(result.Scene);
    AppliedDemoRequest_ = DemoRequest_;
    Settings_.SelectedNode = 0;
    // Restore common preferences before applying the next dedicated scene's
    // technique setup, including transitions directly between dedicated demos.
    if (BeforeDedicatedDemo_)
    {
        Settings_.Techniques=BeforeDedicatedDemo_->Techniques; Settings_.Mode=BeforeDedicatedDemo_->Mode;
        Settings_.OutlineEnabled=BeforeDedicatedDemo_->OutlineEnabled; BeforeDedicatedDemo_.reset();
    }
    if (Scene_.Demo==DemoId::FurCube || Scene_.Demo==DemoId::SponzaAtrium || Scene_.Demo==DemoId::StanfordDragon || DemoCatalog::IsVolumeDemo(Scene_.Demo))
        BeforeDedicatedDemo_=DemoSettingsRestore{Settings_.Techniques,Settings_.Mode,Settings_.OutlineEnabled};
    if (Scene_.Demo==DemoId::FurCube)
    {
        DemoCatalog::ApplyHairAppearance(Scene_,Settings_.Techniques);
        Settings_.Mode=FillMode::Solid; Settings_.OutlineEnabled=false;
    }
    else
    {
        Settings_.Techniques.Hair.Enabled=false;
    }
    if (Scene_.Demo==DemoId::SponzaAtrium)
    {
        DemoCatalog::ApplySponzaAppearance(Settings_.Techniques);
        Settings_.Mode=FillMode::Solid; Settings_.OutlineEnabled=true;
    }
    else Settings_.Techniques.AreaLights.Enabled=false;
    if (Scene_.Demo == DemoId::StanfordDragon)
    {
        DemoCatalog::ApplyDragonAppearance(Settings_.Techniques);
        Settings_.Mode=FillMode::Solid; Settings_.OutlineEnabled=false;
    }
    if (DemoCatalog::IsVolumeDemo(Scene_.Demo))
    {
        DemoCatalog::ApplyVolumeAppearance(Scene_,Settings_.Techniques);
        Settings_.Mode=FillMode::Solid; Settings_.OutlineEnabled=false;
    }
    Settings_.Techniques.Pipeline = Techniques::RenderingPipeline::Forward;
    Settings_.Techniques.Nanite.Enabled = false;
    Settings_.Techniques.MeshShaders = false;
    Settings_.Techniques.Parallax.Enabled = false;
    // Filter by implementation support while device capabilities are still
    // initializing; a dedicated demo must not silently lose its defining effect.
    if (!DemoCatalog::SupportsBackend(Scene_.Demo,Techniques::BackendFromId(SelectedBackend_)))
    {
        const auto backends=Backends();
        if (!backends.empty()) SelectedBackend_=backends.front().Id;
    }
    if (Scene_.Demo!=DemoId::BigWorld && Scene_.Demo!=DemoId::SponzaAtrium)
    {
        Settings_.Techniques.Reflections.EnvironmentIntensity=environmentIntensity;
        if (environmentEnabled && Scene_.Environment) Settings_.Techniques.Reflections.Mode=environmentMode;
        else if (Techniques::UsesEnvironmentCapture(Settings_.Techniques.Reflections))
            Settings_.Techniques.Reflections.Mode=Techniques::ReflectionMode::None;
    }
    if (Scene_.Demo!=DemoId::SponzaAtrium)
        Settings_.Techniques.Reflections.Skybox=Scene_.Demo==DemoId::UtahTeapot || Scene_.Demo==DemoId::BigWorld;
    ConstrainSponzaAtmosphere(Settings_.Techniques,Scene_);
    if (Scene_.World) UpdateBigWorld(Scene_,Settings_.Techniques,true);
    if (!Scene_.Environment)
    {
        Settings_.Techniques.Reflections.Skybox=false;
        if (Techniques::UsesEnvironmentCapture(Settings_.Techniques.Reflections)) Settings_.Techniques.Reflections.Mode=Techniques::ReflectionMode::None;
    }
    SelectMaterialLibrary(Settings_.Techniques, Scene_);
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
        ViewportHairFurStatus_ = std::move(result.HairFurStatus);
        // Background preparation and temporal effects request followup frames
        // even when the camera and scene are stationary.
        RenderDirty_ |= result.NeedsTemporalFrame;
        // The dedicated fur scene is incomplete until its groom is published.
        // Keep the previous image during shape edits and request the next frame;
        // a bare substrate must not be presented as a completed Fur Cube demo.
        if (Scene_.Demo==DemoId::FurCube && result.HairGeometryPending)
        {
            RenderDirty_=true;
            StatusText_="Preparing fur...";
            return;
        }
        {
            const auto frameTime=std::chrono::steady_clock::now();
            if (FPSWindowStart_==std::chrono::steady_clock::time_point{}) FPSWindowStart_=frameTime;
            else ++FPSFrames_;
            const float fpsElapsed=std::chrono::duration<float>(frameTime-FPSWindowStart_).count();
            if (fpsElapsed>=0.5f)
            {
                ViewportFPS_=float(FPSFrames_)/fpsElapsed;
                FPSFrames_=0;FPSWindowStart_=frameTime;
            }
        }
        ViewportImage_ = std::move(result.Output);
        ViewportGeometry_ = result.Geometry;
        ViewportPathTracingSamples_ = result.PathTracingSamples;
        ViewportBakedLightingStatus_ = std::move(result.BakedLightingStatus);
        ViewportVoxelGIStatus_ = std::move(result.VoxelGIStatus);
        ViewportLumenGIStatus_=std::move(result.LumenGIStatus);
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

std::uint32_t EditorCore::QueryMultisampleCounts() const
{
    const auto adapter = FindAdapter(SelectedBackend_);
    return adapter ? adapter->QueryMultisampleCounts() : 1;
}

std::uint32_t EditorCore::QueryWeightedTransparencySampleCounts() const
{
    const auto adapter = FindAdapter(SelectedBackend_);
    return adapter ? adapter->QueryWeightedTransparencySampleCounts() : 0;
}

bool EditorCore::QueryMaterialAnisotropicFiltering() const
{
    const auto adapter = FindAdapter(SelectedBackend_);
    return adapter && adapter->QueryMaterialAnisotropicFiltering();
}

bool EditorCore::QueryLumenHardwareRayTracing() const
{ const auto adapter=FindAdapter(SelectedBackend_); return adapter && adapter->QueryLumenHardwareRayTracing(); }
bool EditorCore::QueryNaniteHardwareScheduling() const
{ const auto adapter=FindAdapter(SelectedBackend_); return adapter && adapter->QueryNaniteHardwareScheduling(); }

Techniques::Support EditorCore::QueryTechniqueSupport(Techniques::Technique technique) const
{
    const auto adapter=FindAdapter(SelectedBackend_);
    const auto backend=Techniques::BackendFromId(SelectedBackend_);
    const auto support=adapter ? adapter->QueryTechniqueSupport(technique) : Techniques::QuerySupport(technique,backend);
    if (support.Available() && Scene_.AlternateMaterials.empty() &&
        (technique == Techniques::Technique::PBR || (technique == Techniques::Technique::AreaLights && !Techniques::UsesPathTracing(Settings_.Techniques.Pipeline))))
        return {Techniques::Availability::SceneLimit, "This scene has no authored PBR texture set."};
    if (!support.Available() || backend==Techniques::Backend::DXR) return support;
    const auto storageLimit=adapter ? adapter->QuerySceneStorageBlockBytes() : 0;
    if (storageLimit && !Renderer::SceneTriangleStorageFits(technique,Scene_.TriangleCountAtLoad,storageLimit))
        return {Techniques::Availability::SceneLimit,"Scene triangle table exceeds this device's storage-block limit"};
    if (technique==Techniques::Technique::RayTracedShadows)
    {
        // The sampled shadow image expands instances into world-space triangles.
        // Bound the binary BVH by two nodes per triangle before offering the
        // option; dense scenes need an instanced BVH to fit this upload budget.
        std::uint64_t bytes=Renderer::RayShadowHeaderTexels*sizeof(Renderer::RayShadowTexel);
        bytes+=Scene_.TriangleCountAtLoad*(sizeof(Renderer::RaySceneTriangle)+2*sizeof(Renderer::RaySceneNode));
        for (const auto& material:Scene_.Materials)
        {
            bytes+=sizeof(Renderer::RaySceneMaterial);
            for (const auto& image:{material.DiffuseImage,material.OpacityImage})
                if (image) bytes+=std::uint64_t(image->Width)*image->Height*sizeof(float);
        }
        if (bytes>Renderer::RayShadowMaximumImageBytes)
            return {Techniques::Availability::SceneLimit,
                "Scene exceeds the conservative ray-shadow image budget; an instanced BVH is required"};
    }
    return support;
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
