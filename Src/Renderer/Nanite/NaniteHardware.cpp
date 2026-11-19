#include "Renderer/Nanite/NaniteHardware.h"

#include <unordered_map>
#include <unordered_set>

namespace Evans::Renderer
{
    bool BuildNaniteHardwareWork(const VisibilityBufferDesc& frame,unsigned width,unsigned height,
        NaniteHardwareWork& output,std::string& error)
    {
        NaniteHardwareWork result;
        if (!frame.GpuHardware || frame.Draws.empty()) { output=std::move(result); error.clear(); return true; }
        if (!frame.NaniteRaster) { error="Nanite hardware scheduling requires hybrid visibility rasterization"; return false; }
        if (!ValidateVisibilityFrame(frame,int(width),int(height),error)) return false;
        const std::uint64_t pixels=std::uint64_t(width)*height;
        if (!width || !height || width>NaniteRasterMaximumSide || height>NaniteRasterMaximumSide || pixels>NaniteHardwareMaximumBytes/sizeof(unsigned))
        { error="Nanite hardware ownership image exceeds the frame budget"; return false; }
        std::unordered_map<std::string,unsigned> bins; std::unordered_set<unsigned> submitted;
        std::uint64_t candidates=0;
        for (const auto& draw:frame.Draws)
        {
            if (!submitted.insert(draw.Range).second) { error="Nanite hardware scheduling requires unique submitted ranges"; return false; }
            NaniteHardwareBin bin; const auto& range=frame.Scene->Draws[draw.Range];
            if (!BuildNaniteRasterDrawConstants(frame,&draw,width,height,bin.Constants.Raster))
            { error="Invalid Nanite hardware coverage material"; return false; }
            bin.Cull=draw.Pipeline.BackFaceCulling; bin.Mirrored=range.Mirrored;
            bin.Diffuse=draw.Pipeline.DiffuseTexture; bin.Opacity=draw.Pipeline.OpacityTexture;
            // Opaque coverage does not depend on material colors or textures.
            // Cutout coverage keeps the original opacity, cutoff and map handles.
            auto& c=bin.Constants.Raster;
            if (c.Material.Y<=0) { c.Material={1,0,0,0}; c.Images={}; bin.Diffuse={}; bin.Opacity={}; }
            std::string key(reinterpret_cast<const char*>(&c.Material),sizeof(c.Material));
            key.append(reinterpret_cast<const char*>(&c.Images),sizeof(c.Images));
            const std::uint64_t images[]{bin.Diffuse.Value,bin.Opacity.Value}; key.append(reinterpret_cast<const char*>(images),sizeof(images));
            const unsigned flags[]{unsigned(bin.Cull),unsigned(bin.Mirrored)}; key.append(reinterpret_cast<const char*>(flags),sizeof(flags));
            const auto inserted=bins.emplace(std::move(key),unsigned(result.Bins.size())); const unsigned index=inserted.first->second;
            if (inserted.second) result.Bins.push_back(std::move(bin));
            result.Ranges.push_back({range.FirstTriangle,range.TriangleCount,index,unsigned(candidates)}); candidates+=range.TriangleCount;
            if (candidates>NaniteRasterMaximumTriangles) { error="Nanite hardware candidates exceed the triangle budget"; return false; }
        }
        const auto binCount=result.Bins.size();
        const auto words=pixels+binCount*NaniteHardwareBinWords+binCount+2*candidates+result.Ranges.size()*sizeof(ShaderUInt4)/sizeof(unsigned);
        if (words>NaniteHardwareMaximumBytes/sizeof(unsigned)) { error="Nanite hardware schedule exceeds the frame budget"; return false; }
        result.Constants.Counts={unsigned(candidates),unsigned(result.Ranges.size()),unsigned(binCount),unsigned(pixels)};
        result.Constants.Layout={unsigned(pixels*2),unsigned(binCount),unsigned(binCount+candidates),NaniteHardwareColumns};
        result.Constants.Images={width,height,0,0};
        result.ArgumentWords=unsigned(binCount*NaniteHardwareBinWords); result.QueueWords=unsigned(binCount+2*candidates);
        for (unsigned bin=0;bin<result.Bins.size();++bin)
            result.Bins[bin].Constants.Schedule={bin,unsigned(binCount),unsigned(candidates),0};
        output=std::move(result); error.clear(); return true;
    }
}
