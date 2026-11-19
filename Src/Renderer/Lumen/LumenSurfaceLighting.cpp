#include "Renderer/Lumen/LumenSurfaceLighting.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

namespace Evans::Renderer
{
    /* Surface-cache material capture is separate from lighting in Lumen.
       This direct-light stage evaluates Lambert transport at reconstructed
       card texels and retains geometry/material/occlusion/light identities.
       Rectangle quadrature uses the renderer's existing prepared area-light
       units; each API evaluates attenuation, visibility and BRDF independently.
       It is a radiance source for later GI, not screen-space color accumulation.
       Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5"
       (SIGGRAPH 2022); PBRT 4e, "Diffuse Reflection" and "Area Lights".
    */
    namespace
    {
        std::atomic<std::uint64_t> NextLighting{1};
        constexpr float Pi=3.14159265358979323846f;
        constexpr std::size_t VectorBytes=16;
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        Float3 Unit(Float3 value)
        {
            const double length=std::hypot(double(value.X),double(value.Y),double(value.Z));
            return length>0?Float3{float(value.X/length),float(value.Y/length),float(value.Z/length)}:Float3{0,1,0};
        }
    }
    bool BuildLumenSurfaceLightingInput(const LumenSurfaceIdentity& surface,const LumenSurfaceLayout& layout,
        const std::vector<LumenSurfaceCapture>& captures,const LumenDistanceSceneInfo& distance,
        const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingInput& output,std::string& error)
    {
        if (!surface.Epoch || !surface.Serial || !surface.Geometry || !surface.Material || !upload.LightingRevision ||
            surface.Epoch!=distance.Identity.Epoch || surface.Geometry!=distance.Identity.Geometry || upload.CardInstances.size()!=layout.CardCount)
        { error="Lumen relighting requires matched material/occlusion geometry versions and one source instance per card"; return false; }
        if (!std::isfinite(upload.SurfaceBias) || upload.SurfaceBias<0 || !std::isfinite(upload.VoxelBiasScale) || upload.VoxelBiasScale<1 || upload.VoxelBiasScale>8 ||
            !std::isfinite(upload.MaximumShadowDistance) || upload.MaximumShadowDistance<=0 || upload.SurfaceBias>=upload.MaximumShadowDistance ||
            !std::isfinite(upload.MinimumLightDistance) || upload.MinimumLightDistance<=0 ||
            !std::isfinite(upload.MinimumLightDistance*upload.MinimumLightDistance) || upload.MinimumLightDistance*upload.MinimumLightDistance<std::numeric_limits<float>::min())
        { error="Lumen relighting needs finite bias, shadow horizon and representable light-distance regularization"; return false; }
        const auto rays=upload.IndirectRays; const auto& sky=upload.IndirectSky;
        if ((rays && (rays<4 || rays>LumenMaximumIndirectRays || (rays&(rays-1)))) || (upload.PreviousLighting && !rays) ||
            !std::isfinite(upload.IndirectMaximumDistance) || (rays && upload.IndirectMaximumDistance<=0) ||
            !std::isfinite(sky.X) || !std::isfinite(sky.Y) || !std::isfinite(sky.Z) || sky.X<0 || sky.Y<0 || sky.Z<0)
        { error="Lumen radiosity needs bounded power-of-two samples, finite sky energy and a positive trace interval"; return false; }
        std::vector<LumenLookupCard> lookup;
        if (!BuildLumenLookupCards(layout,captures,lookup,error)) return false;
        LumenDistanceTraceUpload trace; trace.MaximumSteps=upload.MaximumSteps; trace.HitToleranceCells=upload.HitToleranceCells; trace.StepScale=upload.StepScale;
        trace.ShadowCastersOnly=true;
        const auto& environment=upload.Environment; const float mean=1/(2*(environment.Exponent+1));
        trace.HairSky={sky.X+environment.Radiance.X*mean,sky.Y+environment.Radiance.Y*mean,sky.Z+environment.Radiance.Z*mean};
        trace.Rays={{{0,0,0,0},{0,0,1,upload.MaximumShadowDistance}}}; LumenSurfaceLightingInput result;
        if (!BuildLumenEnvironmentFrame(upload.Environment,result.Frame.Environment,error) || !BuildLumenDistanceTraceFrame(distance,trace,result.Frame.Trace,error)) return false;
        std::vector<LumenLightingCard> cards; cards.reserve(lookup.size());
        for (unsigned index=0;index<lookup.size();++index)
        {
            const auto instance=upload.CardInstances[index];
            if (instance!=LumenDistanceInvalid && instance>=distance.Counts.Y) { error="Lumen relighting card addresses an absent occluder instance"; return false; }
            const auto& source=lookup[index]; LumenLightingCard card;
            card.InverseProjection=source.InverseProjection; card.Origin=source.Origin;
            card.Backward={source.BackwardScale.X,source.BackwardScale.Y,source.BackwardScale.Z,0}; card.TileInstance={source.Tile.X,source.Tile.Y,instance,index}; cards.push_back(card);
        }
        const auto collection=SceneLightCollection::Create(upload.Lights,-1,error); if (!collection) return false;
        std::vector<LumenLightingEmitter> emitters;
        for (unsigned index=0;index<collection->GetLights().size();++index)
        {
            const auto& light=collection->GetLights()[index];
            const Float3 radiance{std::max(light.Color.X,0.0f)*std::max(light.Intensity,0.0f),std::max(light.Color.Y,0.0f)*std::max(light.Intensity,0.0f),std::max(light.Color.Z,0.0f)*std::max(light.Intensity,0.0f)};
            const auto append=[&](Float3 position,Float3 normal,float weight,unsigned sample)
            {
                LumenLightingEmitter emitter;
                emitter.PositionKind={position.X,position.Y,position.Z,float(light.Type)}; emitter.NormalRange={normal.X,normal.Y,normal.Z,light.Range};
                emitter.RadianceWeight={radiance.X,radiance.Y,radiance.Z,weight}; emitter.Metadata={light.CastShadows?1u:0u,index,sample,0};
                if (emitters.size()>=LumenLightingMaximumSamples || !Finite(emitter.PositionKind) || !Finite(emitter.NormalRange) || !Finite(emitter.RadianceWeight) || weight<0)
                { error="Lumen relighting exceeds finite emitter sampling/storage limits"; return false; }
                emitters.push_back(emitter); return true;
            };
            if (light.Type==LightType::RectArea)
            {
                const auto quadrature=BuildRectangleEmitterQuadrature(light);
                for (unsigned y=0;y<quadrature.SamplesPerAxis;++y) for (unsigned x=0;x<quadrature.SamplesPerAxis;++x)
                {
                    const auto p=quadrature.Position(x,y),n=quadrature.Normal;
                    if (!append({p.x,p.y,p.z},{n.x,n.y,n.z},quadrature.Weight,y*quadrature.SamplesPerAxis+x)) return false;
                }
            }
            else if (!append(light.Type==LightType::Directional?Unit(light.Direction):light.Position,{},1,0)) return false;
        }
        const auto texels=std::size_t(layout.Width)*layout.Height;
        if (texels*(emitters.size()+rays)>LumenLightingMaximumSampleWork) { error="Lumen relighting exceeds its texel/lighting-sample work budget"; return false; }
        const auto cardBytes=cards.size()*sizeof(LumenLightingCard),emitterBytes=emitters.size()*sizeof(LumenLightingEmitter);
        result.Words.resize((cardBytes+emitterBytes)/sizeof(std::uint32_t)); std::memcpy(result.Words.data(),cards.data(),cardBytes);
        if (emitterBytes) std::memcpy(reinterpret_cast<std::uint8_t*>(result.Words.data())+cardBytes,emitters.data(),emitterBytes);
        if (rays)
        {
            LumenRadianceMapping mapping;
            if (!BuildLumenRadianceMapping(layout,captures,distance.Counts.Y,upload.CardInstances,mapping,error)) return false;
            result.Frame.Radiosity={rays,upload.IndirectSequence,unsigned(result.Words.size()*sizeof(std::uint32_t)/VectorBytes),mapping.CardVectorOffset};
            result.Frame.SkyDistance={sky.X,sky.Y,sky.Z,upload.IndirectMaximumDistance};
            result.Words.insert(result.Words.end(),mapping.Words.begin(),mapping.Words.end());
        }
        result.Frame.Trace.Counts.X=unsigned(texels); result.Frame.Trace.Traversal.Y=LumenLightingThreads*LumenLightingColumns;
        result.Frame.Atlas={layout.Width,layout.Height,layout.CardSide,layout.Columns};
        result.Frame.Work={layout.CardCount,unsigned(emitters.size()),unsigned(cardBytes/VectorBytes),layout.CardSide+LumenSurfaceBorder*2};
        result.Frame.Lighting={upload.SurfaceBias,upload.VoxelBiasScale,upload.MaximumShadowDistance,upload.MinimumLightDistance*upload.MinimumLightDistance};
        constexpr float PositionRoundoff=8*std::numeric_limits<float>::epsilon();
        result.Frame.Numeric={PositionRoundoff,1/Pi,0,0};
        const unsigned columns=(layout.Width+LumenLightingTileSide-1)/LumenLightingTileSide,rows=(layout.Height+LumenLightingTileSide-1)/LumenLightingTileSide;
        result.Frame.Dispatch={0,columns*rows,columns,LumenLightingTileSide}; output=std::move(result); error.clear(); return true;
    }
    bool ValidateLumenLightingTexels(const LumenLightingInfo& info,const std::vector<LumenLightingTexel>& texels,std::string& error)
    {
        if (texels.size()!=std::size_t(info.Layout.Width)*info.Layout.Height) { error="Lumen lighting readback has the wrong atlas extent"; return false; }
        for (const auto& texel:texels)
        {
            const auto& radiance=texel.RadianceCoverage; const auto& m=texel.Metadata;
            if (!Finite(radiance) || radiance.X<0 || radiance.Y<0 || radiance.Z<0 || m.X>unsigned(LumenLightingStatus::InvalidArithmetic) ||
                std::uint64_t(m.Y)>std::uint64_t(info.EmitterSamples)+info.IndirectSamples || std::uint64_t(m.Z)+m.W>m.Y)
            { error="Lumen lighting returned invalid radiance or transport accounting"; return false; }
            if (!m.X)
            {
                if (radiance.X || radiance.Y || radiance.Z || radiance.W || m.Y || m.Z || m.W)
                { error="Uncovered Lumen lighting retained stale radiance"; return false; }
            }
            else if (radiance.W!=1 || (m.X==unsigned(LumenLightingStatus::Complete) && m.W) ||
                (m.X==unsigned(LumenLightingStatus::Unresolved) && !m.W) ||
                (m.X==unsigned(LumenLightingStatus::InvalidArithmetic) && (radiance.X || radiance.Y || radiance.Z)))
            { error="Lumen lighting status does not match its coverage or uncertainty"; return false; }
        }
        error.clear(); return true;
    }
    bool ValidateLumenRadiositySource(const LumenSurfaceIdentity& surface,const LumenDistanceSceneIdentity& distance,
        const LumenSurfaceLightingUpload& upload,const LumenLightingInfo& previous,std::string& error)
    {
        const auto& a=previous.Identity.Surface; const auto& b=previous.Identity.Occlusion;
        if (!upload.PreviousLighting || !upload.IndirectRays || previous.Bounce>=LumenMaximumIndirectBounces ||
            a.Epoch!=surface.Epoch || a.Serial!=surface.Serial || a.Geometry!=surface.Geometry || a.Material!=surface.Material ||
            b.Epoch!=distance.Epoch || b.Serial!=distance.Serial || b.Geometry!=distance.Geometry || previous.Identity.Lighting!=upload.LightingRevision)
        { error="Lumen radiosity requires a completed earlier bounce of the exact material, geometry and light generation"; return false; }
        error.clear(); return true;
    }
    LumenLightingHandle NewLumenLightingHandle() noexcept
    {
        auto value=NextLighting.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX) if (NextLighting.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
