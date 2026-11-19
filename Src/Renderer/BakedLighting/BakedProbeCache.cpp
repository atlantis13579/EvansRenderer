#include "Renderer/BakedLighting/BakedProbeCache.h"
#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/Lighting/SceneLightCollection.h"

namespace Evans::Renderer
{
    namespace
    {
        bool SameMaterials(const Techniques::Settings& a, const Techniques::Settings& b)
        {
            return a.TextureMapping == b.TextureMapping && a.NormalMapping == b.NormalMapping && a.AlphaTesting == b.AlphaTesting &&
                a.AlphaBlending == b.AlphaBlending && a.Refraction.Enabled==b.Refraction.Enabled && a.Shading == b.Shading && a.Metallic == b.Metallic;
        }
        bool SameSamples(const BakedProbeSettings& a, const BakedProbeSettings& b)
        {
            return a.Minimum.X == b.Minimum.X && a.Minimum.Y == b.Minimum.Y && a.Minimum.Z == b.Minimum.Z &&
                a.Maximum.X == b.Maximum.X && a.Maximum.Y == b.Maximum.Y && a.Maximum.Z == b.Maximum.Z &&
                a.Counts == b.Counts && a.RayCount == b.RayCount && a.MaximumDistance == b.MaximumDistance &&
                a.SurfaceBias == b.SurfaceBias && a.MaximumBackfaceFraction == b.MaximumBackfaceFraction && a.Shadows == b.Shadows;
        }
    }

    bool BakedProbeCache::Update(const std::vector<SceneRayMeshView>& meshes, std::uint64_t sceneRevision,
        const Techniques::Settings& requested, const LightingState& lighting, bool shadows, std::string& error,
        const std::filesystem::path& archive)
    {
        const auto settings = Techniques::SanitizeSettings(requested);
        const auto& probes = settings.GlobalIllumination.Probes;
        const auto& prior=RequestedProbes;
        if (Output && sceneRevision==Revision && SameMaterials(settings,MaterialSettings) && SameSceneLights(lighting,Lighting) &&
            shadows==Output->Settings.Shadows && archive==Archive && probes.AutomaticBounds==prior.AutomaticBounds &&
            probes.BoundsMinimum==prior.BoundsMinimum && probes.BoundsMaximum==prior.BoundsMaximum &&
            probes.GridAxisCount==prior.GridAxisCount && probes.RayCount==prior.RayCount && probes.MaximumBackfaceFraction==prior.MaximumBackfaceFraction) return true;
        if (!archive.empty())
            if (auto loaded=LoadProbeAsset(archive,ProbeAssetKey(meshes,settings,lighting,shadows)))
            {
                Query.reset(); Output=std::move(loaded); Revision=sceneRevision; MaterialSettings=settings; Lighting=lighting;
                RequestedProbes=probes; Archive=archive; Loaded=true; error.clear(); return true;
            }
        const bool rebuild = !Query || sceneRevision != Revision || !SameMaterials(settings, MaterialSettings);
        auto query = Query;
        if (rebuild)
        {
            query = SceneRayQuery::Create(meshes, settings, error);
            if (!query) return false;
        }
        BakedProbeSettings bake;
        bake.Counts.fill(probes.GridAxisCount); bake.RayCount = probes.RayCount;
        bake.MaximumBackfaceFraction = probes.MaximumBackfaceFraction; bake.Shadows = shadows;
        Maths::Vector3 minimum(probes.BoundsMinimum[0],probes.BoundsMinimum[1],probes.BoundsMinimum[2]);
        Maths::Vector3 maximum(probes.BoundsMaximum[0],probes.BoundsMaximum[1],probes.BoundsMaximum[2]);
        const auto& bounds = query->GetBounds();
        if (probes.AutomaticBounds && bounds.IsValid())
        {
            constexpr float PaddingFraction = 0.05f, MinimumHalfExtent = 0.05f;
            const auto center = bounds.Min + (bounds.Max-bounds.Min)*0.5f;
            auto halfExtent = (bounds.Max-bounds.Min)*0.5f;
            const float padding = std::max({halfExtent.x,halfExtent.y,halfExtent.z,MinimumHalfExtent})*PaddingFraction;
            halfExtent = Maths::Vector3::Max(halfExtent+Maths::Vector3(padding),Maths::Vector3(MinimumHalfExtent));
            minimum = center-halfExtent; maximum = center+halfExtent;
        }
        bake.Minimum = {minimum.x,minimum.y,minimum.z}; bake.Maximum = {maximum.x,maximum.y,maximum.z};
        Maths::AABB traceBounds(minimum,maximum);
        if (bounds.IsValid()) traceBounds.Expand(bounds);
        constexpr float TraceExtentMultiplier = 2, RelativeRayBias = 0.000001f, MinimumRayBias = 0.00001f;
        bake.MaximumDistance = (traceBounds.Max-traceBounds.Min).Length()*TraceExtentMultiplier;
        bake.SurfaceBias = std::max(bake.MaximumDistance*RelativeRayBias,MinimumRayBias);
        if (!rebuild && Output && SameSamples(bake,Output->Settings) && SameSceneLights(lighting,Lighting))
        { RequestedProbes=probes; Archive=archive; return true; }
        auto result = std::make_shared<BakedProbeSamples>();
        if (!BakeLightProbeSamples(*query,lighting,bake,*result,error)) return false;
        Query = std::move(query); Output = std::move(result); Revision = sceneRevision;
        MaterialSettings = settings; Lighting = lighting; RequestedProbes=probes; Archive=archive; Loaded=false;
        ++Bakes; if (rebuild) ++QueryBuilds;
        return true;
    }
}
