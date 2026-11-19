#pragma once

#include "Renderer/Lumen/LumenSurfaceCache.h"
#include "Renderer/Lumen/LumenUpdateQueue.h"
#include <chrono>
#include <thread>

namespace Evans::Renderer
{
    // Freeze projections/materials before submission. Native input resources
    // remain borrowed; mutation/deletion cancels and drains their submitted slice.
    struct LumenCaptureJob : LumenUpdateJob
    {
        LumenSurfaceUpload Upload;
        LumenSurfaceLayout Layout;
        std::vector<LumenSurfaceConstants> Constants;
        std::vector<unsigned> FirstConstant;
        std::vector<std::uint32_t> InputBuffers,InputTextures;
        LumenCaptureStatistics Statistics;
        struct SliceFrame { ShaderUInt4 Dispatch; };
        static std::uint64_t NewTaskValue() { return NewLumenSurfaceHandle().Value; }
        unsigned RemainingUnits() const { return Statistics.TotalCards-Statistics.CompletedCards; }
        void RetireUnits(unsigned count)
        {
            const auto previous=Statistics.CompletedCards; Statistics.CompletedCards+=count;
            Statistics.DrawCalls+=FirstConstant[Statistics.CompletedCards]-FirstConstant[previous];
        }
        void Submitted(unsigned count) { Statistics.LastSubmittedCards=count; }
        SliceFrame Slice(unsigned budget) const
        { return {{Statistics.CompletedCards,budget?std::min(budget,RemainingUnits()):RemainingUnits(),0,0}}; }
        bool Uses(BufferHandle handle) const
        { return std::binary_search(InputBuffers.begin(),InputBuffers.end(),handle.Value); }
        bool Uses(TextureHandle handle) const
        { return std::binary_search(InputTextures.begin(),InputTextures.end(),handle.Value); }
        bool Prepare(const LumenSurfaceUpload& upload,std::string& error)
        {
            if (upload.Readback) { error="Lumen capture jobs publish resident atlases; read texels after completion"; return false; }
            if (!ValidateLumenSurfaceUpload(upload,Layout,error)) return false;
            Upload=upload; Statistics.TotalCards=Layout.CardCount;
            for (const auto& card:Upload.Cards)
            {
                FirstConstant.push_back(unsigned(Constants.size()));
                for (unsigned index=card.FirstDraw;index<card.FirstDraw+card.DrawCount;++index)
                {
                    LumenSurfaceConstants frame;
                    if (!BuildLumenSurfaceConstants(Layout,card,Upload.Draws[index],frame,error)) return false;
                    Constants.push_back(frame);
                }
            }
            FirstConstant.push_back(unsigned(Constants.size()));
            for (const auto& draw:Upload.Draws)
            {
                InputBuffers.push_back(draw.Vertices.Value); InputBuffers.push_back(draw.Indices.Value);
                for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture) InputTextures.push_back(texture.Value);
            }
            for (auto* handles:{&InputBuffers,&InputTextures})
            { std::sort(handles->begin(),handles->end()); handles->erase(std::unique(handles->begin(),handles->end()),handles->end()); }
            ReservedBytes=std::size_t(Layout.Width)*Layout.Height*(LumenSurfacePlanes*sizeof(Float4)+sizeof(float))+
                Constants.size()*sizeof(LumenSurfaceConstants)*2+Upload.Draws.size()*sizeof(LumenSurfaceDraw)+
                Upload.Cards.size()*sizeof(LumenSurfaceCapture)+FirstConstant.size()*sizeof(unsigned)+
                (InputBuffers.capacity()+InputTextures.capacity())*sizeof(std::uint32_t);
            return true;
        }
    };
    template<class Work> using LumenCaptureQueue=LumenUpdateQueue<Work,LumenCaptureTask>;
}
