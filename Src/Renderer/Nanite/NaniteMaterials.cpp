#include "Renderer/Nanite/NaniteMaterials.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace Evans::Renderer
{
    bool BuildNaniteMaterialWork(const VisibilityBufferDesc& frame,int width,int height,
        NaniteMaterialWork& output,std::string& error)
    {
        NaniteMaterialWork result;
        if (!frame.MaterialTiles || frame.Draws.empty()) { output=std::move(result); error.clear(); return true; }
        if (!ValidateVisibilityFrame(frame,width,height,error)) return false;
        const std::uint64_t columns=(std::uint64_t(width)+NaniteMaterialTileSize-1)/NaniteMaterialTileSize;
        const std::uint64_t rows=(std::uint64_t(height)+NaniteMaterialTileSize-1)/NaniteMaterialTileSize;
        const std::uint64_t capacity=columns*rows*NaniteMaterialThreads;
        std::uint64_t materialCount=0;
        for (const auto& range:frame.Scene->Draws) materialCount=std::max(materialCount,std::uint64_t(range.Material)+1);
        if (capacity>NaniteMaterialMaximumBytes/(2*sizeof(unsigned)))
        { error="Nanite material image exceeds the bounded frame budget"; return false; }
        const auto withinBudget=[&](std::uint64_t bins)
        { return (materialCount+2*capacity+bins*NaniteMaterialBinWords)*sizeof(unsigned)<=NaniteMaterialMaximumBytes; };
        if (!withinBudget(frame.Draws.size())) { error="Nanite material tile scratch exceeds the bounded frame budget"; return false; }
        result.MaterialBins.resize(std::size_t(materialCount),UINT32_MAX);
        std::unordered_map<std::string,unsigned> bins;
        for (unsigned index=0;index<frame.Draws.size();++index)
        {
            const auto& draw=frame.Draws[index]; const auto& p=draw.Pipeline;
            VisibilityConstants constants;
            if (!BuildVisibilityConstants(frame.ViewProjection,p,width,height,frame.Scene->TriangleCount(),frame.Scene->Draws[draw.Range],
                bool(p.DiffuseTexture),bool(p.NormalTexture),bool(p.OpacityTexture),constants)) return false;
            // Compare packed shader values, never C++ object padding. Geometry,
            // transforms and raster state have already determined the winning ID.
            std::string key(reinterpret_cast<const char*>(&constants),offsetof(VisibilityConstants,Draw));
            const std::uint64_t textures[]{p.DiffuseTexture.Value,p.NormalTexture.Value,p.OpacityTexture.Value};
            key.append(reinterpret_cast<const char*>(textures),sizeof(textures));
            const auto inserted=bins.emplace(std::move(key),unsigned(result.Representatives.size()));
            const unsigned bin=inserted.first->second;
            if (inserted.second) result.Representatives.push_back(index);
            auto& mapped=result.MaterialBins[frame.Scene->Draws[draw.Range].Material];
            if (mapped!=UINT32_MAX && mapped!=bin)
            { error="Nanite material tiles require a unique shader material per visibility source"; return false; }
            mapped=bin; result.DrawBins.push_back(bin);
        }
        const auto headers=result.Representatives.size()*NaniteMaterialBinWords;
        result.OutputWords=unsigned(headers+2*capacity);
        result.Constants.Counts={unsigned(width),unsigned(height),unsigned(frame.Scene->TriangleCount()),unsigned(materialCount)};
        result.Constants.Layout={unsigned(result.Representatives.size()),unsigned(columns),unsigned(rows),unsigned(headers)};
        result.Constants.Storage={unsigned(capacity),unsigned(headers+capacity),result.OutputWords,0};
        output=std::move(result); error.clear(); return true;
    }
}
