#include "Renderer/Lumen/LumenScene.h"
#include "Renderer/Lumen/LumenSurfaceLookup.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace Evans::Renderer
{
    bool AllocateLumenSceneMeshParts(const std::vector<LumenSceneMeshDemand>& meshes,const LumenMeshBuildSettings& settings,
        unsigned cardSide,const LumenSceneBudget& budget,std::vector<LumenMeshBuildSettings>& output,std::string& error)
    {
        if (!settings.Partition.MaximumParts || settings.Partition.MaximumParts>LumenPartitionMaximumParts ||
            !settings.Cards.MaximumCards || settings.Cards.MaximumCards>LumenLookupMaximumCandidates ||
            !budget.MaximumCards || budget.MaximumCards>LumenSurfaceMaximumCards ||
            !budget.MaximumInstances || budget.MaximumInstances>LumenDistanceMaximumInstances ||
            !budget.MaximumAtlasTexels || budget.MaximumAtlasTexels>LumenSurfaceMaximumTexels || meshes.size()>LumenDistanceMaximumFields)
        { error="Lumen scene allocation needs bounded domain, card and instance budgets"; return false; }
        unsigned cardCapacity=0;
        for (unsigned cards=1;cards<=budget.MaximumCards;++cards)
        {
            LumenSurfaceLayout layout; std::string reason;
            if (!BuildLumenSurfaceLayout(cards,cardSide,layout,reason) || std::size_t(layout.Width)*layout.Height>budget.MaximumAtlasTexels) break;
            cardCapacity=cards;
        }
        if (!cardCapacity) { error="Lumen scene allocation cannot fit even one atlas tile"; return false; }
        std::unordered_set<std::uint64_t> keys; unsigned fields=unsigned(meshes.size()),instances=0;
        std::vector<LumenMeshBuildSettings> result(meshes.size(),settings); std::vector<unsigned> limits;
        for (unsigned i=0;i<meshes.size();++i)
        {
            const auto& mesh=meshes[i];
            if (!mesh.Key || !keys.insert(mesh.Key).second || !mesh.Instances || mesh.Instances>LumenDistanceMaximumInstances ||
                !mesh.MaximumParts || mesh.MaximumParts>LumenPartitionMaximumParts || !mesh.Triangles || mesh.Triangles>LumenPartitionMaximumTriangles)
            { error="Lumen scene allocation requires unique source keys, triangle counts and bounded multiplicity"; return false; }
            result[i].Partition.MaximumParts=1; instances+=mesh.Instances;
            limits.push_back(std::min({mesh.MaximumParts,settings.Partition.MaximumParts,unsigned(mesh.Triangles)}));
        }
        if (instances>budget.MaximumInstances || instances>cardCapacity)
        { error="Lumen scene budget cannot reserve one proxy domain for every source instance"; return false; }
        // A maximum card count is a quality ceiling, not a minimum reservation.
        // Large instanced landscapes first retain every object, then share the
        // finite atlas across their local representations.
        const unsigned cardsPerPart=std::min(settings.Cards.MaximumCards,cardCapacity/std::max(1u,instances));
        for (auto& allocation:result) allocation.Cards.MaximumCards=cardsPerPart;
        const unsigned placementCapacity=std::min(budget.MaximumInstances,cardCapacity/cardsPerPart);
        // Greedy density allocation is independent of input order: stable source
        // keys break ties. A whole component is never clipped to earn a budget.
        while (fields<LumenDistanceMaximumFields)
        {
            unsigned selected=LumenDistanceInvalid;
            for (unsigned i=0;i<meshes.size();++i)
            {
                const auto next=result[i].Partition.MaximumParts+1;
                if (next>limits[i] || meshes[i].Instances>placementCapacity-instances) continue;
                if (selected==LumenDistanceInvalid) { selected=i; continue; }
                const auto other=result[selected].Partition.MaximumParts+1;
                const auto left=std::uint64_t(meshes[i].Triangles)*other,right=std::uint64_t(meshes[selected].Triangles)*next;
                if (left>right || (left==right && meshes[i].Key<meshes[selected].Key)) selected=i;
            }
            if (selected==LumenDistanceInvalid) break;
            ++result[selected].Partition.MaximumParts; ++fields; instances+=meshes[selected].Instances;
        }
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenScenePlan(const LumenSceneUpload& upload,LumenScenePlan& output,std::string& error)
    {
        const auto& budget=upload.Budget; const auto& id=upload.Identity;
        if (!id.Epoch || !id.Geometry || !id.Material || !id.Occlusion ||
            !budget.MaximumCards || budget.MaximumCards>LumenSurfaceMaximumCards ||
            !budget.MaximumInstances || budget.MaximumInstances>LumenDistanceMaximumInstances ||
            !budget.MaximumAtlasTexels || budget.MaximumAtlasTexels>LumenSurfaceMaximumTexels ||
            !budget.MaximumDistanceWords || budget.MaximumDistanceWords>LumenDistanceMaximumWords ||
            !budget.MaximumCaptureTriangles || upload.Instances.size()>LumenDistanceMaximumInstances)
        { error="Lumen scene requires nonzero identities and bounded scene budgets"; return false; }
        LumenScenePlan result; result.Surface.CardSide=upload.CardSide;
        result.Surface.Identity={id.Epoch,id.Material,id.Geometry,id.Material};
        LumenDistanceSceneUpload distance; distance.Identity={id.Epoch,id.Occlusion,id.Geometry}; distance.Hair=upload.Hair;
        std::vector<LumenHardwarePlacement> hardware;
        std::unordered_map<const LumenDistanceField*,unsigned> fields;
        std::unordered_set<std::uint64_t> keys;
        for (const auto& source:upload.Instances)
        {
            if (!source.Key || !keys.insert(source.Key).second)
            { error="Lumen scene instance keys must be nonzero and unique"; return false; }
            if (!source.Visible) continue;
            if (!source.Mesh) { error="Visible Lumen instance has no immutable mesh representation"; return false; }
            LumenMeshPlacement placement;
            if (!BuildLumenMeshPlacement(*source.Mesh,source.World,source.Draws,upload.CardSide,placement,error)) return false;
            if (result.Surface.Cards.size()+placement.Surface.Cards.size()>budget.MaximumCards ||
                distance.Instances.size()+placement.Distance.Instances.size()>budget.MaximumInstances ||
                result.Parts.size()+placement.Parts.size()>LumenSurfaceMaximumDraws)
            {
                error="Lumen scene exceeds aggregate budget: cards "+std::to_string(result.Surface.Cards.size()+placement.Surface.Cards.size())+"/"+std::to_string(budget.MaximumCards)+
                    ", instances "+std::to_string(distance.Instances.size()+placement.Distance.Instances.size())+"/"+std::to_string(budget.MaximumInstances)+
                    ", parts "+std::to_string(result.Parts.size()+placement.Parts.size())+"/"+std::to_string(LumenSurfaceMaximumDraws); return false;
            }
            const auto firstCard=unsigned(result.Surface.Cards.size()),firstDraw=unsigned(result.Surface.Draws.size());
            const auto firstInstance=unsigned(distance.Instances.size()),firstPart=unsigned(result.Parts.size());
            for (auto instance:placement.Distance.Instances)
            {
                const auto& field=placement.Distance.Fields[instance.Field];
                auto [entry,inserted]=fields.emplace(field.get(),unsigned(distance.Fields.size()));
                if (inserted) distance.Fields.push_back(field);
                instance.Field=entry->second; instance.UserId+=firstPart; instance.CastShadow=source.CastShadow;
                distance.Instances.push_back(instance);
            }
            for (unsigned index=0;index<placement.Parts.size();++index)
            {
                auto range=placement.Parts[index]; range.FirstCard+=firstCard;
                if (range.DistanceInstance!=LumenDistanceInvalid) range.DistanceInstance+=firstInstance;
                result.Parts.push_back({source.Key,index,range});
                if (upload.HardwareRayTracing && range.DistanceInstance!=LumenDistanceInvalid)
                {
                    if (source.Mesh->Hardware.size()!=source.Mesh->Geometry.Parts.size() || !source.Mesh->Hardware[index])
                    { error="Lumen hardware transport requires exact triangle exports for every occupied part"; return false; }
                    hardware.push_back({source.Mesh->Hardware[index],source.World,range.DistanceInstance,source.CastShadow});
                }
                for (unsigned card=0;card<range.CardCount;++card) result.CardInstances.push_back(range.DistanceInstance);
            }
            for (auto capture:placement.Surface.Cards)
            {
                for (unsigned draw=0;draw<capture.DrawCount;++draw)
                    result.CaptureTriangles+=placement.Surface.Draws[capture.FirstDraw+draw].IndexCount/3;
                capture.FirstDraw+=firstDraw; result.Surface.Cards.push_back(capture);
            }
            if (result.CaptureTriangles>budget.MaximumCaptureTriangles)
            { error="Lumen scene exceeds its capture-triangle work budget"; return false; }
            result.Surface.Draws.insert(result.Surface.Draws.end(),placement.Surface.Draws.begin(),placement.Surface.Draws.end());
        }
        if (result.Surface.Cards.empty())
        { result.Surface.Cards.resize(1); result.CardInstances.push_back(LumenDistanceInvalid); }
        std::vector<LumenLookupCard> lookup;
        if (!ValidateLumenSurfaceUpload(result.Surface,result.Layout,error) ||
            !BuildLumenLookupCards(result.Layout,result.Surface.Cards,lookup,error)) return false;
        if (std::size_t(result.Layout.Width)*result.Layout.Height>budget.MaximumAtlasTexels)
        { error="Lumen scene exceeds its atlas texel budget, including gutters and unused tiles"; return false; }
        if (!BuildLumenDistanceScene(distance,result.Distance,error)) return false;
        if (upload.HardwareRayTracing)
        {
            auto snapshot=std::make_shared<LumenHardwareScene>();
            if (!BuildLumenHardwareScene(hardware,distance.Identity,*snapshot,error)) return false;
            result.Distance.Info.Hardware=std::move(snapshot);
        }
        if (result.Distance.Words.size()>budget.MaximumDistanceWords)
        { error="Lumen scene exceeds its shared distance-field and BVH storage budget"; return false; }
        if (!ValidateLumenScenePlan(result,error)) return false;
        output=std::move(result); error.clear(); return true;
    }
    bool ValidateLumenScenePlan(const LumenScenePlan& scene,std::string& error)
    {
        LumenSurfaceLayout layout;
        if (!ValidateLumenSurfaceUpload(scene.Surface,layout,error) || !ValidateLumenDistanceScene(scene.Distance,error)) return false;
        if (layout.CardCount!=scene.Layout.CardCount || layout.CardSide!=scene.Layout.CardSide || layout.Columns!=scene.Layout.Columns ||
            layout.Rows!=scene.Layout.Rows || layout.Width!=scene.Layout.Width || layout.Height!=scene.Layout.Height ||
            scene.Parts.size()>LumenSurfaceMaximumDraws || scene.CardInstances.size()!=layout.CardCount ||
            scene.Surface.Identity.Epoch!=scene.Distance.Info.Identity.Epoch || scene.Surface.Identity.Geometry!=scene.Distance.Info.Identity.Geometry)
        { error="Lumen scene plan has incompatible identities, layout or correspondence counts"; return false; }
        std::vector<unsigned> cards(layout.CardCount,LumenDistanceInvalid);
        std::vector<bool> instances(scene.Distance.Info.Counts.Y);
        std::unordered_map<std::uint64_t,std::unordered_set<unsigned>> seen;
        for (unsigned index=0;index<scene.Parts.size();++index)
        {
            const auto& part=scene.Parts[index]; const auto& range=part.Range;
            if (!part.InstanceKey || !seen[part.InstanceKey].insert(part.Part).second || range.CardCount>LumenLookupMaximumCandidates ||
                std::uint64_t(range.FirstCard)+range.CardCount>cards.size())
            { error="Lumen scene part lost its stable identity or bounded card range"; return false; }
            if (range.DistanceInstance==LumenDistanceInvalid)
            { if (range.CardCount) { error="Lumen scene assigned cards to a missing distance instance"; return false; } continue; }
            if (range.DistanceInstance>=instances.size() || instances[range.DistanceInstance] || scene.Distance.Info.UserIds[range.DistanceInstance]!=index)
            { error="Lumen scene distance instance lost its unique source part"; return false; }
            instances[range.DistanceInstance]=true;
            for (unsigned card=range.FirstCard;card<range.FirstCard+range.CardCount;++card)
            {
                if (cards[card]!=LumenDistanceInvalid) { error="Lumen scene card belongs to multiple instances"; return false; }
                cards[card]=range.DistanceInstance;
            }
        }
        if (cards!=scene.CardInstances || std::find(instances.begin(),instances.end(),false)!=instances.end())
        { error="Lumen scene card/part correspondence omits or aliases an instance"; return false; }
        std::uint64_t work=0;
        for (unsigned index=0;index<cards.size();++index)
        {
            const auto& capture=scene.Surface.Cards[index];
            if (cards[index]==LumenDistanceInvalid && capture.DrawCount)
            { error="Lumen scene draws a card without a source distance instance"; return false; }
            for (unsigned draw=0;draw<capture.DrawCount;++draw) work+=scene.Surface.Draws[capture.FirstDraw+draw].IndexCount/3;
        }
        if (work!=scene.CaptureTriangles) { error="Lumen scene capture work accounting is stale"; return false; }
        error.clear(); return true;
    }
}
