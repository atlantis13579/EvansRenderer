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
};

struct RenderResult
{
    RenderStatus Status = RenderStatus::Failed;
    Image Output;
    std::string Error;
};

class IRenderAdapter
{
public:
    virtual ~IRenderAdapter() = default;
    [[nodiscard]] virtual BackendDescriptor Descriptor() const = 0;
    [[nodiscard]] virtual bool IsAvailable() const { return true; }
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
