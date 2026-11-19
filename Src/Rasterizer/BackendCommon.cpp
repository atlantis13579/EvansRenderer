#include "BackendCommon.hpp"
#include "SoftwareRasterCore.hpp"
#include "Rasterizer/ParallelExecutor.h"

#include <cmath>
#include <vector>

namespace Evans::Rasterizer::Detail
{
void ApplyGpuSelectionOutline(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation)
{
    if (!settings.OutlineEnabled || externalCancellation.IsCancellationRequested() ||
        backendCancellation.load(std::memory_order_relaxed) ||
        std::none_of(scene.Triangles.begin(), scene.Triangles.end(),
            [](const RasterTriangle& triangle) { return triangle.Selected; }))
        return;
    CpuRasterResources resources;
    RasterStatistics statistics;
    (void)SoftwareRasterCore::RenderCpuTiled(scene, settings, target,
        externalCancellation, backendCancellation, 8, resources, statistics,
        CpuRasterPass::SelectionOutline);
}

namespace
{

constexpr float ClipEpsilon = 1.0e-6f;

struct PreparedVertex
{
    Float4 ViewClip;
    Float4 ShadowClip;
    Float3 WorldPosition;
    Float3 WorldNormal;
    Color VertexColor;
    Float2 TexCoord;
};

[[nodiscard]] Float3 Normalize(const Float3& value) noexcept
{
    const float lengthSquared =
        value.X * value.X + value.Y * value.Y + value.Z * value.Z;
    if (lengthSquared <= ClipEpsilon * ClipEpsilon)
    {
        return {0.0f, 0.0f, 1.0f};
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return {
        value.X * inverseLength,
        value.Y * inverseLength,
        value.Z * inverseLength};
}

[[nodiscard]] Float3 TransformNormal(
    const Matrix4& matrix,
    const Float3& normal) noexcept
{
    const float a00 = matrix(0, 0);
    const float a01 = matrix(0, 1);
    const float a02 = matrix(0, 2);
    const float a10 = matrix(1, 0);
    const float a11 = matrix(1, 1);
    const float a12 = matrix(1, 2);
    const float a20 = matrix(2, 0);
    const float a21 = matrix(2, 1);
    const float a22 = matrix(2, 2);
    const float determinant =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    if (std::abs(determinant) <= ClipEpsilon)
    {
        return Normalize(normal);
    }
    const float inverseDeterminant = 1.0f / determinant;
    return Normalize({
        ((a11 * a22 - a12 * a21) * normal.X +
         (a12 * a20 - a10 * a22) * normal.Y +
         (a10 * a21 - a11 * a20) * normal.Z) *
            inverseDeterminant,
        ((a02 * a21 - a01 * a22) * normal.X +
         (a00 * a22 - a02 * a20) * normal.Y +
         (a01 * a20 - a00 * a21) * normal.Z) *
            inverseDeterminant,
        ((a01 * a12 - a02 * a11) * normal.X +
         (a02 * a10 - a00 * a12) * normal.Y +
         (a00 * a11 - a01 * a10) * normal.Z) *
            inverseDeterminant});
}

[[nodiscard]] PreparedVertex TransformVertex(
    const RasterVertex& vertex,
    const RasterSettings& settings) noexcept
{
    PreparedVertex result;
    result.VertexColor = vertex.VertexColor;
    result.TexCoord = vertex.TexCoord;
    if (settings.InputSpace == VertexInputSpace::Clip)
    {
        result.ViewClip = vertex.ClipPosition;
        result.ShadowClip = vertex.ShadowClipPosition;
        result.WorldPosition = vertex.Position;
        result.WorldNormal = Normalize(vertex.Normal);
        return result;
    }

    Float4 world = settings.World * Float4{
        vertex.Position.X,
        vertex.Position.Y,
        vertex.Position.Z,
        1.0f};
    if (std::abs(world.W) > ClipEpsilon &&
        std::abs(world.W - 1.0f) > ClipEpsilon)
    {
        const float inverseW = 1.0f / world.W;
        world.X *= inverseW;
        world.Y *= inverseW;
        world.Z *= inverseW;
        world.W = 1.0f;
    }
    result.WorldPosition = {world.X, world.Y, world.Z};
    result.WorldNormal = TransformNormal(settings.World, vertex.Normal);
    result.ViewClip = settings.Projection * (settings.View * world);
    result.ShadowClip = settings.LightViewProjection * world;
    return result;
}

[[nodiscard]] PreparedVertex Interpolate(
    const PreparedVertex& left,
    const PreparedVertex& right,
    float amount) noexcept
{
    const float inverse = 1.0f - amount;
    return {
        {left.ViewClip.X * inverse + right.ViewClip.X * amount,
         left.ViewClip.Y * inverse + right.ViewClip.Y * amount,
         left.ViewClip.Z * inverse + right.ViewClip.Z * amount,
         left.ViewClip.W * inverse + right.ViewClip.W * amount},
        {left.ShadowClip.X * inverse + right.ShadowClip.X * amount,
         left.ShadowClip.Y * inverse + right.ShadowClip.Y * amount,
         left.ShadowClip.Z * inverse + right.ShadowClip.Z * amount,
         left.ShadowClip.W * inverse + right.ShadowClip.W * amount},
        {left.WorldPosition.X * inverse + right.WorldPosition.X * amount,
         left.WorldPosition.Y * inverse + right.WorldPosition.Y * amount,
         left.WorldPosition.Z * inverse + right.WorldPosition.Z * amount},
        {left.WorldNormal.X * inverse + right.WorldNormal.X * amount,
         left.WorldNormal.Y * inverse + right.WorldNormal.Y * amount,
         left.WorldNormal.Z * inverse + right.WorldNormal.Z * amount},
        {left.VertexColor.R * inverse + right.VertexColor.R * amount,
         left.VertexColor.G * inverse + right.VertexColor.G * amount,
         left.VertexColor.B * inverse + right.VertexColor.B * amount,
         left.VertexColor.A * inverse + right.VertexColor.A * amount},
        {left.TexCoord.X * inverse + right.TexCoord.X * amount,
         left.TexCoord.Y * inverse + right.TexCoord.Y * amount}};
}

[[nodiscard]] float PlaneDistance(
    const PreparedVertex& vertex,
    int plane,
    ClipDepthRange depthRange,
    bool shadowPass) noexcept
{
    const Float4& clip = shadowPass ? vertex.ShadowClip : vertex.ViewClip;
    switch (plane)
    {
    case 0: return clip.X + clip.W;
    case 1: return clip.W - clip.X;
    case 2: return clip.Y + clip.W;
    case 3: return clip.W - clip.Y;
    case 4:
        return depthRange == ClipDepthRange::ZeroToOne
                   ? clip.Z
                   : clip.Z + clip.W;
    case 5: return clip.W - clip.Z;
    default: return -1.0f;
    }
}

void ClipAgainstPlane(
    const std::vector<PreparedVertex>& input,
    std::vector<PreparedVertex>& output,
    int plane,
    ClipDepthRange depthRange,
    bool shadowPass)
{
    output.clear();
    if (input.empty())
    {
        return;
    }

    PreparedVertex previous = input.back();
    float previousDistance =
        PlaneDistance(previous, plane, depthRange, shadowPass);
    bool previousInside = previousDistance >= 0.0f;
    for (const PreparedVertex& current : input)
    {
        const float currentDistance =
            PlaneDistance(current, plane, depthRange, shadowPass);
        const bool currentInside = currentDistance >= 0.0f;
        if (previousInside != currentInside)
        {
            const float denominator = previousDistance - currentDistance;
            const float amount = std::abs(denominator) > ClipEpsilon
                                     ? previousDistance / denominator
                                     : 0.0f;
            output.push_back(Interpolate(previous, current, amount));
        }
        if (currentInside)
        {
            output.push_back(current);
        }
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
}

[[nodiscard]] RasterVertex ToRasterVertex(
    const PreparedVertex& source) noexcept
{
    RasterVertex result;
    result.ClipPosition = source.ViewClip;
    result.ShadowClipPosition = source.ShadowClip;
    result.Position = source.WorldPosition;
    result.Normal = source.WorldNormal;
    result.VertexColor = source.VertexColor;
    result.TexCoord = source.TexCoord;
    return result;
}

void PreparePass(
    const RasterScene& source,
    const RasterSettings& settings,
    bool shadowPass,
    RasterScene& destination,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation,
    bool& cancelled, std::size_t first, std::size_t last)
{
    std::vector<PreparedVertex> polygon;
    std::vector<PreparedVertex> temporary;
    polygon.reserve(10);
    temporary.reserve(10);
    destination.Triangles.reserve(last-first);

    for (std::size_t triangleIndex = first;
         triangleIndex < last;
         ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U &&
            (externalCancellation.IsCancellationRequested() ||
             backendCancellation.load(std::memory_order_relaxed)))
        {
            cancelled = true;
            return;
        }

        const RasterTriangle& sourceTriangle =
            source.Triangles[triangleIndex];
        polygon = {
            TransformVertex(sourceTriangle.Vertices[0], settings),
            TransformVertex(sourceTriangle.Vertices[1], settings),
            TransformVertex(sourceTriangle.Vertices[2], settings)};
        unsigned outside=0,common=63;
        for(const auto& vertex:polygon)
        {
            unsigned mask=0;
            for(int plane=0;plane<6;++plane)if(!(PlaneDistance(vertex, plane, settings.DepthRange, shadowPass)>=0))mask|=1u<<plane;
            outside|=mask;common&=mask;
        }
        if(common)continue;
        // Most triangles are wholly inside or outside the view. Only boundary
        // triangles need the polygon clipper and its repeated vertex copies.
        for (int plane = 0; outside && plane < 6 && !polygon.empty(); ++plane)
        {
            ClipAgainstPlane(
                polygon,
                temporary,
                plane,
                settings.DepthRange,
                shadowPass);
            polygon.swap(temporary);
        }
        for (std::size_t vertexIndex = 1;
             vertexIndex + 1 < polygon.size();
             ++vertexIndex)
        {
            RasterTriangle output = sourceTriangle;
            output.Vertices[0] = ToRasterVertex(polygon[0]);
            output.Vertices[1] = ToRasterVertex(polygon[vertexIndex]);
            output.Vertices[2] = ToRasterVertex(polygon[vertexIndex + 1]);
            destination.Triangles.push_back(std::move(output));
        }
    }
}

void PreparePassParallel(const RasterScene& source,const RasterSettings& settings,bool shadow,
    RasterScene& destination,const RasterCancellationToken& cancellation,const std::atomic<bool>& backendCancellation,
    bool& cancelled, ParallelExecutor* executor)
{
    if(!executor)
    {
        PreparePass(source,settings,shadow,destination,cancellation,backendCancellation,cancelled,0,source.Triangles.size());return;
    }
    constexpr std::size_t BatchSize=256;
    struct Batch {RasterScene Scene;bool Cancelled=false;};
    std::vector<Batch> batches((source.Triangles.size()+BatchSize-1)/BatchSize);
    executor->Run(static_cast<std::uint32_t>(batches.size()),0,[&](std::uint32_t index) {
        const auto first=std::size_t(index)*BatchSize;
        PreparePass(source,settings,shadow,batches[index].Scene,cancellation,backendCancellation,batches[index].Cancelled,
            first,std::min(first+BatchSize,source.Triangles.size()));
    });
    std::size_t count=0;for(const auto& batch:batches){count+=batch.Scene.Triangles.size();cancelled|=batch.Cancelled;}
    destination.Triangles.reserve(count);
    for(auto& batch:batches)destination.Triangles.insert(destination.Triangles.end(),
        std::make_move_iterator(batch.Scene.Triangles.begin()),std::make_move_iterator(batch.Scene.Triangles.end()));
}

} // namespace

bool ExecutionGate::Initialize(std::uint32_t requestedWorkerThreads)
{
    Shutdown();
    std::lock_guard<std::mutex> lock(StateMutex_);
    WorkerThreads_ = requestedWorkerThreads == 0
                         ? std::max(std::thread::hardware_concurrency(), 1U)
                         : requestedWorkerThreads;
    Initialized_ = true;
    ShuttingDown_ = false;
    return true;
}

void ExecutionGate::Shutdown() noexcept
{
    {
        std::lock_guard<std::mutex> lock(StateMutex_);
        if (!Initialized_)
        {
            return;
        }
        ShuttingDown_ = true;
        if (ActiveCancellation_)
        {
            ActiveCancellation_->store(true, std::memory_order_relaxed);
        }
    }
    WaitIdle();
    std::lock_guard<std::mutex> lock(StateMutex_);
    Initialized_ = false;
    ShuttingDown_ = false;
}

bool ExecutionGate::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(StateMutex_);
    return Initialized_ && !ShuttingDown_;
}

std::uint32_t ExecutionGate::WorkerThreads() const noexcept
{
    std::lock_guard<std::mutex> lock(StateMutex_);
    return WorkerThreads_;
}

void ExecutionGate::Cancel() noexcept
{
    std::lock_guard<std::mutex> lock(StateMutex_);
    if (ActiveCancellation_)
    {
        ActiveCancellation_->store(true, std::memory_order_relaxed);
    }
}

void ExecutionGate::WaitIdle() noexcept
{
    std::unique_lock<std::mutex> lock(StateMutex_);
    IdleCondition_.wait(lock, [&]() { return PendingSubmissions_ == 0; });
}

void ExecutionGate::FinishSubmission(
    const std::shared_ptr<std::atomic<bool>>& cancellation) noexcept
{
    std::lock_guard<std::mutex> lock(StateMutex_);
    if (ActiveCancellation_ == cancellation)
    {
        ActiveCancellation_.reset();
    }
    --PendingSubmissions_;
    if (PendingSubmissions_ == 0)
    {
        IdleCondition_.notify_all();
    }
}

void StatisticsStore::Set(RasterStatistics statistics)
{
    std::lock_guard<std::mutex> lock(Mutex_);
    Statistics_ = std::move(statistics);
}

RasterStatistics StatisticsStore::Get() const
{
    std::lock_guard<std::mutex> lock(Mutex_);
    return Statistics_;
}

void ValidateRenderTarget(const RenderTarget& target)
{
    if (!target.IsValid())
    {
        throw std::invalid_argument("Cannot rasterize into an invalid render target");
    }
}

PreparedGpuRasterScenes PrepareGpuRasterScenes(
    const RasterScene& scene,
    const RasterSettings& settings,
    bool prepareShadowScene,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation, ParallelExecutor* workers)
{
    PreparedGpuRasterScenes result;
    result.Settings = settings;
    result.Settings.InputSpace = VertexInputSpace::Clip;
    result.Main.Textures = scene.Textures;
    PreparePassParallel(
        scene,
        settings,
        false,
        result.Main,
        externalCancellation,
        backendCancellation,
        result.Cancelled, workers);
    if (!result.Cancelled && prepareShadowScene)
    {
        result.Shadow.Textures = scene.Textures;
        PreparePassParallel(
            scene,
            settings,
            true,
            result.Shadow,
            externalCancellation,
            backendCancellation,
            result.Cancelled, workers);
    }
    return result;
}

} // namespace Evans::Rasterizer::Detail
