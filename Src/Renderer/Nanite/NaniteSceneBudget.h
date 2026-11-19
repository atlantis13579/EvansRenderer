#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Evans::Renderer
{
    struct NaniteSceneDemand { std::uint32_t LogicalPages=0,RootPages=0; };
    struct NaniteSceneBudget
    {
        std::vector<std::uint32_t> Capacities;
        std::uint64_t RootPages=0,AssignedPages=0;
        bool Feasible=true;
    };
    // Reserve complete pinned roots first. Distribute the remaining scene pool
    // proportionally to each family's remaining logical pages, using largest
    // remainders and stable family order for integer ties. Camera motion never
    // changes capacities, so streaming can finish admitted dependency bundles.
    // An infeasible budget returns an empty allocation and preserves source
    // rendering; it must never publish a partially covered scene generation.
    [[nodiscard]] bool PlanNaniteSceneBudget(const std::vector<NaniteSceneDemand>&,
        std::uint32_t familyLimit,std::uint32_t sceneLimit,NaniteSceneBudget&,std::string& error);
}
