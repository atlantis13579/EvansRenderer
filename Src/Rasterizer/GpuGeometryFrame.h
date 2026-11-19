#pragma once

#include "GpuRasterCommon.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Evans::Rasterizer::Detail
{
struct GpuBinLayout
{
    std::uint32_t TileSize = 8;
    std::size_t Words = 0;
    std::size_t StorageWords() const noexcept { return Words + 1; }
    GpuBinLayout(std::uint32_t width, std::uint32_t height, std::uint32_t triangles)
    {
        const auto maskWords = [](std::size_t count) { return (count + 31) / 32; };
        const auto fine = maskWords(triangles), middle = maskWords(fine);
        const auto stride = fine + middle + maskWords(middle);
        for (;; TileSize *= 2)
        {
            Words = stride * ((std::size_t(width) + TileSize - 1) / TileSize) *
                ((std::size_t(height) + TileSize - 1) / TileSize);
            if (Words <= 32u * 1024u * 1024u) break;
            if (TileSize >= std::max(width,height)) throw std::length_error("GPU bin mask exceeds resource limits");
        }
    }
};

struct GpuGeometryConstants : GpuRasterConstants
{
    float NormalRows[3][4]{};
    std::uint32_t MainTileSize, ShadowTileSize, GroupsX, CollectCounts = 0;
};
static_assert(sizeof(GpuGeometryConstants) == sizeof(GpuRasterConstants) + 64);

struct GpuGeometryFrame
{
    bool ExternalShadow, GenerateShadow;
    std::uint32_t ShadowWidth, ShadowHeight, TriangleCount, GroupsX, GroupsY;
    GpuBinLayout MainBins, ShadowBins;
    GpuGeometryConstants Constants;

    GpuGeometryFrame(const RasterScene& scene, const RasterSettings& settings, const RenderTarget& target)
        : ExternalShadow(settings.ShadowsEnabled && settings.SceneShadowMap && settings.SceneShadowMap->IsValid()),
          GenerateShadow(settings.ShadowsEnabled && !ExternalShadow && settings.AutoGenerateShadowMap),
          ShadowWidth(ExternalShadow ? settings.SceneShadowMap->Width : GenerateShadow ? std::clamp(settings.ShadowMapSize,16u,4096u) : 0u),
          ShadowHeight(ExternalShadow ? settings.SceneShadowMap->Height : ShadowWidth),
          TriangleCount(static_cast<std::uint32_t>(scene.Triangles.size())),
          GroupsX(std::max(1u,std::min(65535u,(TriangleCount+63)/64))),
          GroupsY(((TriangleCount+63)/64+GroupsX-1)/GroupsX),
          MainBins(target.Width(),target.Height(),TriangleCount), ShadowBins(ShadowWidth,ShadowHeight,TriangleCount)
    {
        if (scene.Triangles.size() > UINT32_MAX / 9u) throw std::length_error("GPU geometry exceeds index range");
        static_cast<GpuRasterConstants&>(Constants) = BuildGpuRasterConstants(scene,settings,target,
            GenerateShadow || ExternalShadow,ShadowWidth,ShadowHeight);
        Constants.MainTileSize=MainBins.TileSize; Constants.ShadowTileSize=ShadowBins.TileSize; Constants.GroupsX=GroupsX;
        Constants.CollectCounts=!settings.GpuPresentation;
        Constants.OutlineEnabled=settings.OutlineEnabled ? 1u : 0u;
        if (settings.OutlineEnabled)
        {
            Constants.OutlineWidth=std::clamp(std::ceil(settings.OutlineWidth),1.0f,
                static_cast<float>(std::max(target.Width(),target.Height()))*2.0f);
            const auto color=settings.OutlineColor.Clamped();
            Constants.OutlineColor[0]=color.R; Constants.OutlineColor[1]=color.G;
            Constants.OutlineColor[2]=color.B; Constants.OutlineColor[3]=color.A;
        }
        const auto& matrix=settings.World;
        for (unsigned row=0;row<3;++row) for (unsigned col=0;col<3;++col)
            Constants.NormalRows[row][col]=matrix((row+1)%3,(col+1)%3)*matrix((row+2)%3,(col+2)%3)-
                matrix((row+1)%3,(col+2)%3)*matrix((row+2)%3,(col+1)%3);
        float det=0;
        for(unsigned col=0;col<3;++col) det+=matrix(0,col)*Constants.NormalRows[0][col];
        for(unsigned row=0;row<3;++row) for(unsigned col=0;col<3;++col)
            Constants.NormalRows[row][col]=std::abs(det)>1.0e-6f ? Constants.NormalRows[row][col]/det : float(row==col);
    }

    void Report(RasterStatistics& stats, bool uploaded, std::string path) const
    {
        const auto pixelGroups=[](unsigned width,unsigned height) {return std::uint64_t((width+7)/8)*((height+7)/8);};
        stats.InputTriangles=TriangleCount;
        stats.ComputeGroups=pixelGroups(Constants.Width,Constants.Height)*(1+Constants.OutlineEnabled)+(GenerateShadow?pixelGroups(ShadowWidth,ShadowHeight):0)+
            std::uint64_t(GroupsX)*GroupsY*(GenerateShadow?3:2);
        stats.ComputeInvocations=stats.ComputeGroups*64; stats.WorkerThreads=0;
        stats.DispatchedTiles=std::uint64_t((Constants.Width+MainBins.TileSize-1)/MainBins.TileSize)*((Constants.Height+MainBins.TileSize-1)/MainBins.TileSize);
        if(GenerateShadow) stats.DispatchedTiles+=std::uint64_t((ShadowWidth+ShadowBins.TileSize-1)/ShadowBins.TileSize)*((ShadowHeight+ShadowBins.TileSize-1)/ShadowBins.TileSize);
        stats.ExecutionPath=std::move(path)+(uploaded?", source uploaded":", resident source")+", GPU transform/clip/bin"+
            (GenerateShadow?", GPU shadow-depth pass":"");
    }
};
}
