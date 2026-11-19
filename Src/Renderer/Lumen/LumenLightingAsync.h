#pragma once

#include "Renderer/Lumen/LumenSurfaceLighting.h"
#include "Renderer/Lumen/LumenUpdateQueue.h"

namespace Evans::Renderer
{
    struct LumenLightingJob : LumenUpdateJob
    {
        LumenSurfaceLightingInput Input;
        LumenLightingStatistics Statistics;
        static std::uint64_t NewTaskValue() { return NewLumenLightingHandle().Value; }
        unsigned RemainingUnits() const { return Statistics.PendingTiles(); }
        void RetireUnits(unsigned count) { Statistics.CompletedTiles+=count; }
        void Submitted(unsigned count) { Statistics.LastSubmittedTiles=count; }
        LumenSurfaceLightingFrame Slice(unsigned budget) const
        {
            auto frame=Input.Frame; frame.Dispatch.X=Statistics.CompletedTiles;
            frame.Dispatch.Y=budget?std::min(budget,RemainingUnits()):RemainingUnits(); return frame;
        }
        bool Prepare(const LumenSurfaceIdentity& surface,const LumenSurfaceLayout& layout,
            const std::vector<LumenSurfaceCapture>& captures,const LumenDistanceSceneInfo& distance,
            const LumenSurfaceLightingUpload& upload,std::string& error)
        {
            if (upload.Readback) { error="Lumen tile jobs publish resident lighting; read texels after completion"; return false; }
            if (!BuildLumenSurfaceLightingInput(surface,layout,captures,distance,upload,Input,error)) return false;
            Statistics.TotalTiles=Input.Frame.Dispatch.Y;
            ReservedBytes=std::size_t(layout.Width)*layout.Height*(LumenSurfacePlanes*sizeof(Float4)+sizeof(LumenLightingTexel)*(upload.PreviousLighting?2:1))+
                (std::size_t(distance.Counts.W)+Input.Words.size()+Input.Words.capacity())*sizeof(std::uint32_t)+sizeof(Input.Frame);
            return true;
        }
    };
    template<class Work> using LumenLightingQueue=LumenUpdateQueue<Work,LumenLightingTask>;
}
