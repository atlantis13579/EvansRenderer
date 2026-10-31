#include "Rasterizer/CudaRasterRHI.h"

#include "BackendCommon.hpp"
#include "CudaRasterPtx.h"
#include "GpuRasterCommon.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace Evans::Rasterizer
{
namespace
{

using GpuTriangle = Detail::GpuRasterTriangle;
using GpuTexture = Detail::GpuRasterTexture;
using GpuConstants = Detail::GpuRasterConstants;

[[nodiscard]] std::vector<GpuTriangle> BuildGpuTriangles(const RasterScene& scene)
{
    return Detail::BuildGpuRasterTriangles(scene);
}

[[nodiscard]] GpuConstants BuildConstants(
    const RasterScene& scene,
    const RasterSettings& settings,
    const RenderTarget& target,
    bool shadowsEnabled,
    std::uint32_t shadowWidth,
    std::uint32_t shadowHeight)
{
    return Detail::BuildGpuRasterConstants(
        scene,
        settings,
        target,
        shadowsEnabled,
        shadowWidth,
        shadowHeight);
}

struct TileBins
{
    std::vector<std::uint32_t> Offsets;
    std::vector<std::uint32_t> TriangleIndices;
    std::uint32_t TilesX = 0;
    std::uint32_t TilesY = 0;
    std::uint32_t MaximumListLength = 0;
};

[[nodiscard]] Float4 TransformForBinning(
    const RasterVertex& vertex,
    const RasterSettings& settings,
    bool lightPass) noexcept
{
    if (settings.InputSpace == VertexInputSpace::Clip)
    {
        return lightPass ? vertex.ShadowClipPosition : vertex.ClipPosition;
    }
    const Float4 local{vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1.0f};
    const Float4 world = settings.World * local;
    return lightPass
               ? settings.LightViewProjection * world
               : settings.Projection * (settings.View * world);
}

[[nodiscard]] TileBins BuildTileBins(
    const RasterScene& scene,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    bool lightPass)
{
    constexpr std::uint32_t tileSize = 16U;
    TileBins result;
    result.TilesX = (width + tileSize - 1U) / tileSize;
    result.TilesY = (height + tileSize - 1U) / tileSize;
    const std::uint32_t tileCount = result.TilesX * result.TilesY;
    std::vector<std::uint32_t> counts(tileCount, 0U);
    struct Bounds { int MinimumX; int MinimumY; int MaximumX; int MaximumY; bool Valid; };
    std::vector<Bounds> bounds(scene.Triangles.size());
    for (std::size_t triangleIndex = 0; triangleIndex < scene.Triangles.size(); ++triangleIndex)
    {
        float screenX[3]{};
        float screenY[3]{};
        bool valid = true;
        for (std::size_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const Float4 clip = TransformForBinning(
                scene.Triangles[triangleIndex].Vertices[vertexIndex], settings, lightPass);
            if (!std::isfinite(clip.W) || clip.W <= 1.0e-6f)
            {
                valid = false;
                break;
            }
            const float inverseW = 1.0f / clip.W;
            screenX[vertexIndex] = (clip.X * inverseW * 0.5f + 0.5f) * width;
            screenY[vertexIndex] = (0.5f - clip.Y * inverseW * 0.5f) * height;
        }
        if (!valid)
        {
            bounds[triangleIndex] = {0, 0, -1, -1, false};
            continue;
        }
        const int minimumX = std::max(
            0, static_cast<int>(std::floor(std::min({screenX[0], screenX[1], screenX[2]}))));
        const int minimumY = std::max(
            0, static_cast<int>(std::floor(std::min({screenY[0], screenY[1], screenY[2]}))));
        const int maximumX = std::min(
            static_cast<int>(width) - 1,
            static_cast<int>(std::ceil(std::max({screenX[0], screenX[1], screenX[2]}))));
        const int maximumY = std::min(
            static_cast<int>(height) - 1,
            static_cast<int>(std::ceil(std::max({screenY[0], screenY[1], screenY[2]}))));
        valid = minimumX <= maximumX && minimumY <= maximumY;
        bounds[triangleIndex] = {minimumX, minimumY, maximumX, maximumY, valid};
        if (!valid) continue;
        for (std::uint32_t tileY = static_cast<std::uint32_t>(minimumY) / tileSize;
             tileY <= static_cast<std::uint32_t>(maximumY) / tileSize;
             ++tileY)
        {
            for (std::uint32_t tileX = static_cast<std::uint32_t>(minimumX) / tileSize;
                 tileX <= static_cast<std::uint32_t>(maximumX) / tileSize;
                 ++tileX)
            {
                ++counts[tileY * result.TilesX + tileX];
            }
        }
    }
    result.Offsets.resize(static_cast<std::size_t>(tileCount) + 1U, 0U);
    for (std::uint32_t tile = 0; tile < tileCount; ++tile)
    {
        result.MaximumListLength = std::max(result.MaximumListLength, counts[tile]);
        result.Offsets[tile + 1U] = result.Offsets[tile] + counts[tile];
    }
    result.TriangleIndices.resize(result.Offsets.back());
    std::vector<std::uint32_t> writeOffsets = result.Offsets;
    for (std::uint32_t triangleIndex = 0;
         triangleIndex < static_cast<std::uint32_t>(bounds.size());
         ++triangleIndex)
    {
        const Bounds& triangleBounds = bounds[triangleIndex];
        if (!triangleBounds.Valid) continue;
        for (std::uint32_t tileY = static_cast<std::uint32_t>(triangleBounds.MinimumY) / tileSize;
             tileY <= static_cast<std::uint32_t>(triangleBounds.MaximumY) / tileSize;
             ++tileY)
        {
            for (std::uint32_t tileX = static_cast<std::uint32_t>(triangleBounds.MinimumX) / tileSize;
                 tileX <= static_cast<std::uint32_t>(triangleBounds.MaximumX) / tileSize;
                 ++tileX)
            {
                const std::uint32_t tile = tileY * result.TilesX + tileX;
                result.TriangleIndices[writeOffsets[tile]++] = triangleIndex;
            }
        }
    }
    return result;
}

[[nodiscard]] const std::string& EmbeddedCudaRasterPtx()
{
    static const std::string ptx = []
    {
        std::string combined;
        for (const char* chunk : Detail::CudaRasterPtxChunks) combined += chunk;
        return combined;
    }();
    return ptx;
}

#if defined(_WIN32)

using CuResult = int;
using CuDevice = int;
using CuContext = void*;
using CuModule = void*;
using CuFunction = void*;
using CuDevicePointer = unsigned long long;
constexpr CuResult CuSuccess = 0;

class CudaRuntime
{
public:
    ~CudaRuntime() { Shutdown(); }

    [[nodiscard]] bool Initialize(std::string& error)
    {
        Shutdown();
        DriverLibrary_ = LoadLibraryA("nvcuda.dll");
        if (DriverLibrary_ == nullptr)
        {
            error = "CUDA Driver API library nvcuda.dll is unavailable";
            Shutdown();
            return false;
        }
        if (!LoadFunctions(error))
        {
            Shutdown();
            return false;
        }
        int deviceCount = 0;
        if (!Check(CuInit_(0), "cuInit", error) ||
            !Check(CuDeviceGetCount_(&deviceCount), "cuDeviceGetCount", error) ||
            deviceCount <= 0)
        {
            if (error.empty()) error = "No CUDA-capable device is available";
            Shutdown();
            return false;
        }
        if (!Check(CuDeviceGet_(&Device_, 0), "cuDeviceGet", error) ||
            !Check(CuCtxCreate_(&Context_, 0, Device_), "cuCtxCreate", error))
        {
            Shutdown();
            return false;
        }

        if (!Check(
                CuModuleLoadData_(&Module_, EmbeddedCudaRasterPtx().c_str()),
                "cuModuleLoadData(embedded PTX)",
                error) ||
            !Check(CuModuleGetFunction_(&Kernel_, Module_, "RasterKernel"), "cuModuleGetFunction(RasterKernel)", error) ||
            !Check(CuModuleGetFunction_(&ShadowKernel_, Module_, "ShadowDepthKernel"), "cuModuleGetFunction(ShadowDepthKernel)", error))
        {
            Shutdown();
            return false;
        }
        (void)CuCtxSetCurrent_(nullptr);
        return true;
    }

    void Shutdown() noexcept
    {
        if (Context_ != nullptr && CuCtxSetCurrent_ != nullptr)
        {
            (void)CuCtxSetCurrent_(Context_);
        }
        if (Module_ != nullptr && CuModuleUnload_ != nullptr) (void)CuModuleUnload_(Module_);
        if (Context_ != nullptr && CuCtxDestroy_ != nullptr) (void)CuCtxDestroy_(Context_);
        Module_ = nullptr;
        Kernel_ = nullptr;
        ShadowKernel_ = nullptr;
        Context_ = nullptr;
        Device_ = 0;
        if (DriverLibrary_ != nullptr) FreeLibrary(DriverLibrary_);
        DriverLibrary_ = nullptr;
        ResetFunctions();
    }

    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        RasterStatistics& statistics)
    {
        if (externalCancellation.IsCancellationRequested() ||
            backendCancellation.load(std::memory_order_relaxed))
        {
            return RasterStatus::Cancelled;
        }

        const bool hasExternalShadowMap =
            settings.ShadowsEnabled && settings.SceneShadowMap != nullptr &&
            settings.SceneShadowMap->IsValid();
        const bool generateShadowMap =
            settings.ShadowsEnabled && !hasExternalShadowMap &&
            settings.AutoGenerateShadowMap;
        const bool shadowsEnabled = hasExternalShadowMap || generateShadowMap;
        const std::uint32_t shadowWidth = hasExternalShadowMap
                                              ? settings.SceneShadowMap->Width
                                              : (generateShadowMap
                                                     ? std::clamp(settings.ShadowMapSize, 16U, 4096U)
                                                     : 0U);
        const std::uint32_t shadowHeight = hasExternalShadowMap
                                               ? settings.SceneShadowMap->Height
                                               : shadowWidth;

        const auto transformBegin = std::chrono::steady_clock::now();
        Detail::PreparedGpuRasterScenes prepared = Detail::PrepareGpuRasterScenes(
            scene,
            settings,
            generateShadowMap,
            externalCancellation,
            backendCancellation);
        if (prepared.Cancelled ||
            externalCancellation.IsCancellationRequested() ||
            backendCancellation.load(std::memory_order_relaxed))
        {
            return RasterStatus::Cancelled;
        }
        const RasterScene& mainScene = prepared.Main;
        const RasterScene& shadowScene = prepared.Shadow;
        const RasterSettings& gpuSettings = prepared.Settings;
        const auto triangles = BuildGpuTriangles(mainScene);
        const auto shadowTriangles = generateShadowMap
                                         ? BuildGpuTriangles(shadowScene)
                                         : std::vector<GpuTriangle>{};
        const Detail::GpuRasterTextureData textureData =
            Detail::BuildGpuRasterTextures(mainScene);
        const TileBins mainBins =
            BuildTileBins(
                mainScene,
                gpuSettings,
                target.Width(),
                target.Height(),
                false);
        const TileBins shadowBins = generateShadowMap
                                        ? BuildTileBins(
                                              shadowScene,
                                              gpuSettings,
                                              shadowWidth,
                                              shadowHeight,
                                              true)
                                        : TileBins{};
        const GpuConstants constants = BuildConstants(
            mainScene,
            gpuSettings,
            target,
            shadowsEnabled,
            shadowWidth,
            shadowHeight);
        statistics.TransformMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - transformBegin)
                .count();

        std::string error;
        if (!Check(CuCtxSetCurrent_(Context_), "cuCtxSetCurrent", error))
        {
            throw std::runtime_error(error);
        }

        const std::size_t triangleBytes =
            std::max<std::size_t>(triangles.size() * sizeof(GpuTriangle), 1U);
        const std::size_t shadowTriangleBytes = std::max<std::size_t>(
            shadowTriangles.size() * sizeof(GpuTriangle),
            1U);
        const std::size_t mainOffsetBytes =
            mainBins.Offsets.size() * sizeof(std::uint32_t);
        const std::size_t mainIndexBytes = std::max<std::size_t>(
            mainBins.TriangleIndices.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t));
        const std::size_t shadowOffsetBytes = std::max<std::size_t>(
            shadowBins.Offsets.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t));
        const std::size_t shadowIndexBytes = std::max<std::size_t>(
            shadowBins.TriangleIndices.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t));
        const std::size_t pixelCount =
            static_cast<std::size_t>(target.Width()) * target.Height();
        const std::size_t colorBytes = pixelCount * sizeof(float) * 4U;
        const std::size_t depthBytes = pixelCount * sizeof(float);
        const std::size_t shadowBytes = std::max<std::size_t>(
            static_cast<std::size_t>(shadowWidth) * shadowHeight * sizeof(float),
            sizeof(float));
        const std::size_t textureDescriptorBytes = std::max<std::size_t>(
            textureData.Descriptors.size() * sizeof(GpuTexture),
            sizeof(GpuTexture));
        const std::size_t textureTexelBytes = std::max<std::size_t>(
            textureData.TexelsRGBA8.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t));

        CuDevicePointer deviceTriangles = 0;
        CuDevicePointer deviceShadowTriangles = 0;
        CuDevicePointer deviceConstants = 0;
        CuDevicePointer deviceMainOffsets = 0;
        CuDevicePointer deviceMainIndices = 0;
        CuDevicePointer deviceShadowOffsets = 0;
        CuDevicePointer deviceShadowIndices = 0;
        CuDevicePointer deviceShadowDepths = 0;
        CuDevicePointer deviceTextureDescriptors = 0;
        CuDevicePointer deviceTextureTexels = 0;
        CuDevicePointer deviceColors = 0;
        CuDevicePointer deviceDepths = 0;
        const auto release = [&]() noexcept
        {
            if (deviceDepths != 0) (void)CuMemFree_(deviceDepths);
            if (deviceColors != 0) (void)CuMemFree_(deviceColors);
            if (deviceTextureTexels != 0) (void)CuMemFree_(deviceTextureTexels);
            if (deviceTextureDescriptors != 0)
            {
                (void)CuMemFree_(deviceTextureDescriptors);
            }
            if (deviceShadowDepths != 0) (void)CuMemFree_(deviceShadowDepths);
            if (deviceShadowIndices != 0) (void)CuMemFree_(deviceShadowIndices);
            if (deviceShadowOffsets != 0) (void)CuMemFree_(deviceShadowOffsets);
            if (deviceMainIndices != 0) (void)CuMemFree_(deviceMainIndices);
            if (deviceMainOffsets != 0) (void)CuMemFree_(deviceMainOffsets);
            if (deviceConstants != 0) (void)CuMemFree_(deviceConstants);
            if (deviceShadowTriangles != 0)
            {
                (void)CuMemFree_(deviceShadowTriangles);
            }
            if (deviceTriangles != 0) (void)CuMemFree_(deviceTriangles);
            (void)CuCtxSetCurrent_(nullptr);
        };
        try
        {
            if (!Check(CuMemAlloc_(&deviceTriangles, triangleBytes), "cuMemAlloc(triangles)", error) ||
                (generateShadowMap &&
                 !Check(
                     CuMemAlloc_(
                         &deviceShadowTriangles,
                         shadowTriangleBytes),
                     "cuMemAlloc(shadow triangles)",
                     error)) ||
                !Check(CuMemAlloc_(&deviceConstants, sizeof(constants)), "cuMemAlloc(constants)", error) ||
                !Check(CuMemAlloc_(&deviceMainOffsets, mainOffsetBytes), "cuMemAlloc(tile offsets)", error) ||
                !Check(CuMemAlloc_(&deviceMainIndices, mainIndexBytes), "cuMemAlloc(tile indices)", error) ||
                !Check(CuMemAlloc_(&deviceShadowDepths, shadowBytes), "cuMemAlloc(shadow depths)", error) ||
                !Check(
                    CuMemAlloc_(
                        &deviceTextureDescriptors,
                        textureDescriptorBytes),
                    "cuMemAlloc(texture descriptors)",
                    error) ||
                !Check(
                    CuMemAlloc_(&deviceTextureTexels, textureTexelBytes),
                    "cuMemAlloc(texture texels)",
                    error) ||
                !Check(CuMemAlloc_(&deviceColors, colorBytes), "cuMemAlloc(colors)", error) ||
                !Check(CuMemAlloc_(&deviceDepths, depthBytes), "cuMemAlloc(depths)", error))
            {
                throw std::runtime_error(error);
            }
            if (!triangles.empty() &&
                !Check(
                    CuMemcpyHtoD_(
                        deviceTriangles,
                        triangles.data(),
                        triangles.size() * sizeof(GpuTriangle)),
                    "cuMemcpyHtoD(triangles)",
                    error))
            {
                throw std::runtime_error(error);
            }
            if (generateShadowMap && !shadowTriangles.empty() &&
                !Check(
                    CuMemcpyHtoD_(
                        deviceShadowTriangles,
                        shadowTriangles.data(),
                        shadowTriangles.size() * sizeof(GpuTriangle)),
                    "cuMemcpyHtoD(shadow triangles)",
                    error))
            {
                throw std::runtime_error(error);
            }
            if (!Check(
                    CuMemcpyHtoD_(deviceConstants, &constants, sizeof(constants)),
                    "cuMemcpyHtoD(constants)",
                    error) ||
                (!textureData.Descriptors.empty() &&
                 !Check(
                     CuMemcpyHtoD_(
                         deviceTextureDescriptors,
                         textureData.Descriptors.data(),
                         textureData.Descriptors.size() * sizeof(GpuTexture)),
                     "cuMemcpyHtoD(texture descriptors)",
                     error)) ||
                (!textureData.TexelsRGBA8.empty() &&
                 !Check(
                     CuMemcpyHtoD_(
                         deviceTextureTexels,
                         textureData.TexelsRGBA8.data(),
                         textureData.TexelsRGBA8.size() *
                             sizeof(std::uint32_t)),
                     "cuMemcpyHtoD(texture texels)",
                     error)) ||
                !Check(
                    CuMemcpyHtoD_(
                        deviceMainOffsets,
                        mainBins.Offsets.data(),
                        mainOffsetBytes),
                    "cuMemcpyHtoD(tile offsets)",
                    error) ||
                (!mainBins.TriangleIndices.empty() &&
                 !Check(
                     CuMemcpyHtoD_(
                         deviceMainIndices,
                         mainBins.TriangleIndices.data(),
                         mainBins.TriangleIndices.size() * sizeof(std::uint32_t)),
                     "cuMemcpyHtoD(tile indices)",
                     error)))
            {
                throw std::runtime_error(error);
            }

            const auto begin = std::chrono::steady_clock::now();
            if (hasExternalShadowMap)
            {
                if (!Check(
                        CuMemcpyHtoD_(
                            deviceShadowDepths,
                            settings.SceneShadowMap->Depth.data(),
                            static_cast<std::size_t>(shadowWidth) * shadowHeight *
                                sizeof(float)),
                        "cuMemcpyHtoD(shadow map)",
                        error))
                {
                    throw std::runtime_error(error);
                }
            }
            else if (generateShadowMap)
            {
                if (!Check(
                        CuMemAlloc_(&deviceShadowOffsets, shadowOffsetBytes),
                        "cuMemAlloc(shadow tile offsets)",
                        error) ||
                    !Check(
                        CuMemAlloc_(&deviceShadowIndices, shadowIndexBytes),
                        "cuMemAlloc(shadow tile indices)",
                        error) ||
                    !Check(
                        CuMemcpyHtoD_(
                            deviceShadowOffsets,
                            shadowBins.Offsets.data(),
                            shadowBins.Offsets.size() * sizeof(std::uint32_t)),
                        "cuMemcpyHtoD(shadow tile offsets)",
                        error) ||
                    (!shadowBins.TriangleIndices.empty() &&
                     !Check(
                         CuMemcpyHtoD_(
                             deviceShadowIndices,
                             shadowBins.TriangleIndices.data(),
                             shadowBins.TriangleIndices.size() *
                                 sizeof(std::uint32_t)),
                         "cuMemcpyHtoD(shadow tile indices)",
                         error)))
                {
                    throw std::runtime_error(error);
                }
                void* shadowArguments[] = {
                    &deviceShadowTriangles,
                    &deviceConstants,
                    &deviceShadowOffsets,
                    &deviceShadowIndices,
                    &deviceShadowDepths,
                    &deviceTextureDescriptors,
                    &deviceTextureTexels};
                const unsigned int shadowGridX = (shadowWidth + 7U) / 8U;
                const unsigned int shadowGridY = (shadowHeight + 7U) / 8U;
                if (!Check(
                        CuLaunchKernel_(
                            ShadowKernel_,
                            shadowGridX,
                            shadowGridY,
                            1,
                            8,
                            8,
                            1,
                            0,
                            nullptr,
                            shadowArguments,
                            nullptr),
                        "cuLaunchKernel(ShadowDepthKernel)",
                        error))
                {
                    throw std::runtime_error(error);
                }
            }

            void* arguments[] = {
                &deviceTriangles,
                &deviceConstants,
                &deviceMainOffsets,
                &deviceMainIndices,
                &deviceShadowDepths,
                &deviceTextureDescriptors,
                &deviceTextureTexels,
                &deviceColors,
                &deviceDepths};
            const unsigned int gridX = (target.Width() + 7U) / 8U;
            const unsigned int gridY = (target.Height() + 7U) / 8U;
            if (!Check(
                    CuLaunchKernel_(
                        Kernel_,
                        gridX,
                        gridY,
                        1,
                        8,
                        8,
                        1,
                        0,
                        nullptr,
                        arguments,
                        nullptr),
                    "cuLaunchKernel(RasterKernel)",
                    error) ||
                !Check(CuCtxSynchronize_(), "cuCtxSynchronize", error))
            {
                throw std::runtime_error(error);
            }

            std::vector<float> colors(pixelCount * 4U);
            std::vector<float> depths(pixelCount);
            if (!Check(
                    CuMemcpyDtoH_(colors.data(), deviceColors, colorBytes),
                    "cuMemcpyDtoH(colors)",
                    error) ||
                !Check(
                    CuMemcpyDtoH_(depths.data(), deviceDepths, depthBytes),
                    "cuMemcpyDtoH(depths)",
                    error))
            {
                throw std::runtime_error(error);
            }
            for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
            {
                target.MutableColors()[pixel] = {
                    colors[pixel * 4U],
                    colors[pixel * 4U + 1U],
                    colors[pixel * 4U + 2U],
                    colors[pixel * 4U + 3U]};
                target.MutableDepths()[pixel] = depths[pixel];
            }

            Detail::ApplyGpuSelectionOutline(mainScene, gpuSettings, target,
                externalCancellation, backendCancellation);

            statistics.InputTriangles = scene.Triangles.size();
            statistics.PostClipTriangles = mainScene.Triangles.size();
            statistics.DispatchedTiles =
                static_cast<std::uint64_t>(mainBins.TilesX) * mainBins.TilesY +
                static_cast<std::uint64_t>(shadowBins.TilesX) * shadowBins.TilesY;
            statistics.WorkerThreads = 64;
            statistics.ExecutionPath =
                "Native CUDA Driver API embedded PTX, 16x16 bins=" +
                std::to_string(mainBins.TilesX * mainBins.TilesY) +
                ", max-list=" + std::to_string(mainBins.MaximumListLength) +
                (generateShadowMap ? ", GPU shadow-depth pass" :
                 hasExternalShadowMap ? ", uploaded shadow map" : "");
            statistics.RasterMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin)
                    .count();
        }
        catch (...)
        {
            release();
            throw;
        }
        release();
        return externalCancellation.IsCancellationRequested() ||
                       backendCancellation.load(std::memory_order_relaxed)
                   ? RasterStatus::Cancelled
                   : RasterStatus::Completed;
    }

private:
    template <typename Function>
    [[nodiscard]] bool LoadDriver(Function& function, const char* name)
    {
        function = reinterpret_cast<Function>(GetProcAddress(DriverLibrary_, name));
        return function != nullptr;
    }

    [[nodiscard]] bool LoadFunctions(std::string& error)
    {
        const bool driverLoaded =
            LoadDriver(CuInit_, "cuInit") &&
            LoadDriver(CuDeviceGetCount_, "cuDeviceGetCount") &&
            LoadDriver(CuDeviceGet_, "cuDeviceGet") &&
            LoadDriver(CuCtxCreate_, "cuCtxCreate_v2") &&
            LoadDriver(CuCtxDestroy_, "cuCtxDestroy_v2") &&
            LoadDriver(CuCtxSetCurrent_, "cuCtxSetCurrent") &&
            LoadDriver(CuModuleLoadData_, "cuModuleLoadData") &&
            LoadDriver(CuModuleUnload_, "cuModuleUnload") &&
            LoadDriver(CuModuleGetFunction_, "cuModuleGetFunction") &&
            LoadDriver(CuMemAlloc_, "cuMemAlloc_v2") &&
            LoadDriver(CuMemFree_, "cuMemFree_v2") &&
            LoadDriver(CuMemcpyHtoD_, "cuMemcpyHtoD_v2") &&
            LoadDriver(CuMemcpyDtoH_, "cuMemcpyDtoH_v2") &&
            LoadDriver(CuLaunchKernel_, "cuLaunchKernel") &&
            LoadDriver(CuCtxSynchronize_, "cuCtxSynchronize");
        if (!driverLoaded)
        {
            error = "CUDA Driver API is missing required entry points";
            return false;
        }
        return true;
    }

    [[nodiscard]] static bool Check(CuResult result, const char* operation, std::string& error)
    {
        if (result == CuSuccess) return true;
        error = std::string(operation) + " failed with CUDA error " + std::to_string(result);
        return false;
    }

    void ResetFunctions() noexcept
    {
        CuInit_ = nullptr; CuDeviceGetCount_ = nullptr; CuDeviceGet_ = nullptr;
        CuCtxCreate_ = nullptr; CuCtxDestroy_ = nullptr;
        CuCtxSetCurrent_ = nullptr; CuModuleLoadData_ = nullptr; CuModuleUnload_ = nullptr;
        CuModuleGetFunction_ = nullptr; CuMemAlloc_ = nullptr; CuMemFree_ = nullptr;
        CuMemcpyHtoD_ = nullptr; CuMemcpyDtoH_ = nullptr; CuLaunchKernel_ = nullptr;
        CuCtxSynchronize_ = nullptr;
    }

    HMODULE DriverLibrary_ = nullptr;
    CuDevice Device_ = 0;
    CuContext Context_ = nullptr;
    CuModule Module_ = nullptr;
    CuFunction Kernel_ = nullptr;
    CuFunction ShadowKernel_ = nullptr;

    CuResult (*CuInit_)(unsigned int) = nullptr;
    CuResult (*CuDeviceGetCount_)(int*) = nullptr;
    CuResult (*CuDeviceGet_)(CuDevice*, int) = nullptr;
    CuResult (*CuCtxCreate_)(CuContext*, unsigned int, CuDevice) = nullptr;
    CuResult (*CuCtxDestroy_)(CuContext) = nullptr;
    CuResult (*CuCtxSetCurrent_)(CuContext) = nullptr;
    CuResult (*CuModuleLoadData_)(CuModule*, const void*) = nullptr;
    CuResult (*CuModuleUnload_)(CuModule) = nullptr;
    CuResult (*CuModuleGetFunction_)(CuFunction*, CuModule, const char*) = nullptr;
    CuResult (*CuMemAlloc_)(CuDevicePointer*, std::size_t) = nullptr;
    CuResult (*CuMemFree_)(CuDevicePointer) = nullptr;
    CuResult (*CuMemcpyHtoD_)(CuDevicePointer, const void*, std::size_t) = nullptr;
    CuResult (*CuMemcpyDtoH_)(void*, CuDevicePointer, std::size_t) = nullptr;
    CuResult (*CuLaunchKernel_)(CuFunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**) = nullptr;
    CuResult (*CuCtxSynchronize_)() = nullptr;

};

#else

class CudaRuntime
{
public:
    [[nodiscard]] bool Initialize(std::string& error)
    {
        error = "CUDA raster backend is only available on Windows in this build";
        return false;
    }
    void Shutdown() noexcept {}
};

#endif

} // namespace

class CudaRasterRHI::Context
{
public:
    Detail::ExecutionGate Execution;
    Detail::StatisticsStore Statistics;
    CudaRuntime Runtime;
    mutable std::mutex StateMutex;
    std::string Error;
    bool NativeAvailable = false;
};

CudaRasterRHI::CudaRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

CudaRasterRHI::~CudaRasterRHI()
{
    Shutdown();
}

bool CudaRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    Shutdown();
    std::string error;
    const bool available = Context_->Runtime.Initialize(error);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = available;
        Context_->Error = std::move(error);
    }
    return available && Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void CudaRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
    Context_->Runtime.Shutdown();
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    Context_->NativeAvailable = false;
}

bool CudaRasterRHI::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return Context_->NativeAvailable && Context_->Execution.IsInitialized();
}

RasterBackend CudaRasterRHI::Backend() const noexcept
{
    return RasterBackend::CUDA;
}

std::string_view CudaRasterRHI::Name() const noexcept
{
    return "CUDA Driver Raster";
}

RasterBackendCapabilities CudaRasterRHI::Capabilities() const
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return {
        RasterBackend::CUDA,
        Context_->NativeAvailable,
        Context_->NativeAvailable,
        8,
        8,
        Context_->NativeAvailable
            ? "Native CUDA Driver API with embedded PTX, tiled raster and shadow kernels"
            : "Unsupported: " + Context_->Error};
}

RasterStatus CudaRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        if (!Context_->NativeAvailable)
        {
            throw std::runtime_error("CUDA raster backend is unsupported: " + Context_->Error);
        }
    }
    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
#if defined(_WIN32)
            return Context_->Runtime.Render(
                scene, settings, target, cancellation, backendCancellation, statistics);
#else
            (void)scene; (void)settings; (void)target; (void)cancellation; (void)backendCancellation;
            return RasterStatus::Cancelled;
#endif
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void CudaRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void CudaRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics CudaRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
