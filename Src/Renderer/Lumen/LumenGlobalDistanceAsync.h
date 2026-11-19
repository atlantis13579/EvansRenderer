#pragma once

#include "Renderer/Lumen/LumenGlobalDistance.h"
#include "Renderer/Lumen/LumenUpdateQueue.h"

namespace Evans::Renderer
{
    // One immutable destination is filled over several queue submissions. Each
    // completed slice advances the cursor; partial fields are never published.
    struct LumenGlobalDistanceJob : LumenUpdateJob
    {
        std::shared_ptr<LumenGlobalDistancePlan> Plan;
        LumenGlobalDistanceUpdate Update;
        LumenGlobalDistanceStatistics Statistics;
        static std::uint64_t NewTaskValue() { return NewLumenDistanceSceneHandle().Value; }
        unsigned RemainingUnits() const { return unsigned(Update.Compose.size())-Statistics.ComposedVoxels; }
        void RetireUnits(unsigned count) { Statistics.ComposedVoxels+=count; }
        void Submitted(unsigned count) { Statistics.LastSubmittedVoxels=count; }
        LumenGlobalDistanceFrame Slice(unsigned budget) const
        {
            auto frame=Plan->Frame; const unsigned remaining=RemainingUnits();
            frame.Dispatch={Statistics.ComposedVoxels,budget?std::min(budget,remaining):remaining,0,0}; return frame;
        }
        bool Prepare(const LumenDistanceSceneInfo& source,const LumenGlobalDistanceSettings& settings,
            const LumenGlobalDistancePlan* previous,std::string& error)
        {
            Plan=std::make_shared<LumenGlobalDistancePlan>();
            if (!BuildLumenGlobalDistancePlan(source,settings,*Plan,error) || !BuildLumenGlobalDistanceUpdate(source,*Plan,previous,Update,error)) return false;
            Statistics.TotalVoxels=Plan->Frame.Header.Layout.Z; Statistics.ReusedVoxels=Update.ReusedVoxels;
            ReservedBytes=(std::size_t(Plan->WordCount)+source.Counts.W+(previous?previous->WordCount:0))*sizeof(std::uint32_t)+
                (std::max<std::size_t>(1,Update.Compose.size())+Update.Compose.capacity())*sizeof(std::uint32_t)+
                Update.Copies.capacity()*sizeof(LumenGlobalDistanceCopy)+2*sizeof(LumenGlobalDistanceFrame);
            return true;
        }
    };
    template<class Work> using LumenGlobalDistanceQueue=LumenUpdateQueue<Work,LumenGlobalDistanceTask>;
}
