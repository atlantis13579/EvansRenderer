#include "Renderer/Nanite/NaniteSceneBudget.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace Evans::Renderer
{
    bool PlanNaniteSceneBudget(const std::vector<NaniteSceneDemand>& demands,unsigned familyLimit,
        unsigned sceneLimit,NaniteSceneBudget& output,std::string& error)
    {
        if (demands.size()>std::numeric_limits<unsigned>::max())
        { error="Nanite scene has too many geometry families"; return false; }
        NaniteSceneBudget result; std::vector<unsigned> maximum;
        std::uint64_t remainingDemand=0;
        for (const auto& demand:demands)
        {
            if (!demand.LogicalPages || !demand.RootPages || demand.RootPages>demand.LogicalPages)
            { error="Nanite scene has invalid logical or pinned page counts"; return false; }
            const auto limit=familyLimit?std::min(familyLimit,demand.LogicalPages):demand.LogicalPages;
            result.RootPages+=demand.RootPages;
            if (limit<demand.RootPages) result.Feasible=false;
            maximum.push_back(std::max(limit,demand.RootPages));
            remainingDemand+=maximum.back()-demand.RootPages;
        }
        if (result.RootPages>sceneLimit) result.Feasible=false;
        if (!result.Feasible) { output=std::move(result); error.clear(); return true; }
        result.AssignedPages=result.RootPages;
        const auto available=std::min<std::uint64_t>(std::uint64_t(sceneLimit)-result.RootPages,remainingDemand);
        std::vector<std::uint64_t> remainders(demands.size());
        for (unsigned i=0;i<demands.size();++i)
        {
            const auto numerator=available*(maximum[i]-demands[i].RootPages);
            const auto extra=remainingDemand?numerator/remainingDemand:0;
            result.Capacities.push_back(demands[i].RootPages+static_cast<unsigned>(extra));
            result.AssignedPages+=extra; remainders[i]=remainingDemand?numerator%remainingDemand:0;
        }
        std::vector<unsigned> order(demands.size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return remainders[a]>remainders[b];});
        const auto target=result.RootPages+available;
        for (auto i:order) if (result.AssignedPages<target && result.Capacities[i]<maximum[i])
        { ++result.Capacities[i]; ++result.AssignedPages; }
        if (result.AssignedPages!=target) { error="Nanite scene page apportionment lost its integer remainder"; return false; }
        output=std::move(result); error.clear(); return true;
    }
}
