#include "Renderer/Lumen/LumenRadianceTrace.h"

#include <atomic>
#include <cstring>
#include <limits>

namespace Evans::Renderer
{
    /* Lumen separates mesh-SDF intersection from surface-cache lighting lookup
       (Wright et al., "Lumen", SIGGRAPH 2022). The optional screen-first pass
       preserves trusted HDR hits through both subsequent world stages. All
       intermediate hits and final radiance stay on the GPU. Four storage bindings in
       the resolve pass fit the ES 3.1 minimum: hits, lighting, mapping and output.
       Source instance membership is immutable; nearby geometry cannot borrow an
       unrelated cache tile to hide missing coverage. Normal-direction SDF error
       uses the inverse-transpose metric rather than the ray-direction metric.
    */
    namespace
    {
        std::atomic<std::uint64_t> NextRadiance{1};
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        bool UnitInterval(float x) { return std::isfinite(x) && x>=0 && x<=1; }
    }
    bool BuildLumenScreenDepthPlan(const LumenScreenTraceFrame& frame,LumenScreenDepthPlan& output,std::string& error)
    {
        constexpr std::size_t MaximumBytes=128u*1024u*1024u;
        LumenScreenDepthPlan result; std::size_t words=LumenScreenDepthHeaderWords;
        unsigned width=frame.Viewport.X,height=frame.Viewport.Y;
        if (!width || !height) { error="Lumen screen depth requires a nonempty viewport"; return false; }
        while (true)
        {
            const auto cells=std::uint64_t(width)*height;
            if (result.Levels.size()>=LumenScreenDepthLevels || cells>MaximumBytes/sizeof(std::uint32_t)-words)
            { error="Lumen screen depth exceeds its level or storage budget"; return false; }
            LumenScreenDepthFrame level{}; level.Inverse=frame.InverseViewProjection; level.Anchor=frame.Anchor; level.Forward=frame.Forward;
            if (!result.Levels.empty()) level.Source=result.Levels.back().Destination;
            level.Destination={width,height,unsigned(words),unsigned(result.Levels.size())};
            level.Dispatch={LumenRadianceThreads*LumenRadianceColumns,0,0,0};
            level.Numeric={frame.Parameters.W,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),0};
            result.Levels.push_back(level); words+=std::size_t(cells);
            if (width==1 && height==1) break;
            width=width/2+width%2; height=height/2+height%2;
        }
        result.Bytes=words*sizeof(std::uint32_t); output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenRadianceInput(const LumenLightingInfo& lighting,const LumenDistanceSceneInfo& distance,
        const std::vector<LumenSurfaceCapture>& captures,const std::vector<unsigned>& cardInstances,
        const LumenRadianceTraceUpload& upload,LumenRadianceInput& output,std::string& error)
    {
        const auto& identity=lighting.Identity;
        if (!identity.Lighting || !identity.Surface.Serial || !identity.Surface.Material || !identity.Surface.Epoch || !identity.Surface.Geometry ||
            identity.Surface.Epoch!=distance.Identity.Epoch || identity.Surface.Geometry!=distance.Identity.Geometry ||
            identity.Occlusion.Epoch!=distance.Identity.Epoch || identity.Occlusion.Serial!=distance.Identity.Serial || identity.Occlusion.Geometry!=distance.Identity.Geometry ||
            cardInstances.size()!=lighting.Layout.CardCount)
        { error="Lumen ray radiance requires the exact captured material, occluder and light versions"; return false; }
        if (!UnitInterval(upload.MinimumFacingCosine) || upload.MinimumFacingCosine==0 || !UnitInterval(upload.MinimumNormalCosine) ||
            !std::isfinite(upload.PlaneTolerance) || upload.PlaneTolerance<0 || !std::isfinite(upload.VoxelToleranceScale) || upload.VoxelToleranceScale<1 || upload.VoxelToleranceScale>8)
        { error="Lumen radiance lookup requires finite plane tolerances and valid angular thresholds"; return false; }
        if (upload.Probes.DirectionalSide && (upload.Probes.Origins.empty() || upload.Screen.Enabled || upload.Hybrid.Enabled))
        { error="Directional world cache generation requires explicit world probes and no screen inputs"; return false; }
        if (upload.Probes.DepthLayers!=1 && (!upload.Probes.DirectionalSide || upload.Reflections.Enabled))
        { error="Layered radiance generation requires directional world probes"; return false; }
        LumenRadianceInput result;
        if (upload.Reflections.Enabled)
        {
            if (!upload.Trace.Rays.empty() || upload.WorldCache.Radiance)
            { error="Reflected GPU rays exclude caller rays and diffuse world-probe shortcuts"; return false; }
            if (!BuildLumenReflectionRayFrame(upload.Screen,upload.Probes,upload.Reflections,result.Reflection,error)) return false;
            auto trace=upload.Trace; trace.Rays={{{0,0,0,0},{0,0,1,1}}};
            if (!BuildLumenDistanceTraceFrame(distance,trace,result.Trace,error)) return false;
            result.Trace.Counts.X=result.Reflection.Work.Y;
            result.Trace.Traversal.W|=LumenGeneratedReflectionRays|(upload.Hybrid.Enabled?LumenReflectionScreenFirst:0);
            result.Screen.InverseViewProjection=result.Reflection.Inverse; result.Screen.Viewport=result.Reflection.Viewport;
            result.Screen.Grid={result.Reflection.Work.X,result.Reflection.Work.Y,result.Reflection.Viewport.Z,0};
            result.Screen.Parameters={upload.Screen.NormalBias,0,result.Reflection.Numeric.X,result.Reflection.Numeric.Y};
        }
        else if (upload.Probes.Origins.empty() && !upload.Screen.Enabled)
        { if (!BuildLumenDistanceTraceFrame(distance,upload.Trace,result.Trace,error)) return false; }
        else
        {
            if (!upload.Trace.Rays.empty()) { error="Lumen probe generation and explicit rays are mutually exclusive"; return false; }
            unsigned probeCount=unsigned(upload.Probes.Origins.size());
            if (upload.Screen.Enabled)
            { if (!BuildLumenScreenProbeFrame(upload.Screen,upload.Probes,result.Screen,error)) return false; probeCount=result.Screen.Grid.Y; }
            else if (!ValidateLumenProbeRaySet(upload.Probes,error)) return false;
            // Reuse the scene/step contract without manufacturing CPU probe rays.
            auto trace=upload.Trace; trace.Rays={{{0,0,0,0},{0,0,1,1}}};
            if (!BuildLumenDistanceTraceFrame(distance,trace,result.Trace,error)) return false;
            result.Trace.Counts.X=probeCount*upload.Probes.RaysPerProbe;
            result.ProbeTrace.Trace=result.Trace;
            result.ProbeTrace.Sampling={upload.Probes.RaysPerProbe,probeCount,upload.Probes.Sequence,0};
            // Depth stacks resolve crossings at SDF precision. Subvoxel
            // separation still requires a finer geometry proxy.
            constexpr float LayerCoordinateRoundoff=32*std::numeric_limits<float>::epsilon();
            result.ProbeTrace.Range={upload.Probes.MaximumDistance,float(upload.Probes.DirectionalSide),float(upload.Probes.DepthLayers),LayerCoordinateRoundoff};
        }
        result.Trace.HairSky={lighting.HairSky.X,lighting.HairSky.Y,lighting.HairSky.Z,0};
        if (result.ProbeTrace.Sampling.X) result.ProbeTrace.Trace.HairSky=result.Trace.HairSky;
        if (upload.Hybrid.Enabled)
        {
            const auto& settings=upload.Hybrid;
            if (!upload.Screen.Enabled || !settings.MaximumSteps || settings.MaximumSteps>1024 ||
                !std::isfinite(settings.Thickness) || settings.Thickness<0 || settings.Thickness>1000 ||
                !std::isfinite(settings.PixelStride) || settings.PixelStride<1 || settings.PixelStride>4)
            { error="Lumen hybrid tracing needs screen probes, bounded pixel traversal and finite world thickness"; return false; }
            auto& hybrid=result.Hybrid; hybrid.ViewProjection=upload.Screen.ViewProjection; hybrid.InverseViewProjection=result.Screen.InverseViewProjection;
            const auto& inverse=hybrid.InverseViewProjection.Values;
            const auto center=[&](float depth) { const float w=inverse[14]*depth+inverse[15]; return Float3{(inverse[2]*depth+inverse[3])/w,(inverse[6]*depth+inverse[7])/w,(inverse[10]*depth+inverse[11])/w}; };
            const auto near=center(0),middle=center(0.5f); const double x=double(middle.X)-near.X,y=double(middle.Y)-near.Y,z=double(middle.Z)-near.Z;
            const double length=std::sqrt(x*x+y*y+z*z);
            if (!std::isfinite(length) || length<=0) { error="Lumen screen camera has no finite forward direction"; return false; }
            hybrid.Forward={float(x/length),float(y/length),float(z/length),upload.Reflections.Enabled?1.0f:0.0f};
            constexpr float DepthRoundoff=32*std::numeric_limits<float>::epsilon();
            hybrid.Anchor={near.X,near.Y,near.Z,DepthRoundoff};
            hybrid.Counts={result.Trace.Counts.X,upload.Reflections.Enabled?1u:upload.Probes.RaysPerProbe,upload.Probes.Sequence,LumenRadianceThreads*LumenRadianceColumns};
            hybrid.Viewport={upload.Screen.Width,upload.Screen.Height,settings.MaximumSteps,0};
            hybrid.Parameters={upload.Probes.MaximumDistance,settings.Thickness,settings.PixelStride,result.Screen.Parameters.W}; result.ProbeTrace.Sampling.W=1;
            if (settings.Hierarchical)
            {
                if (!BuildLumenScreenDepthPlan(hybrid,result.Depth,error)) return false;
                hybrid.Viewport.W=unsigned(result.Depth.Levels.size());
            }
        }
        if (upload.WorldCache.Radiance)
        {
            if (!result.ProbeTrace.Sampling.X || upload.Probes.DirectionalSide)
            { error="World radiance lookup requires generated Hammersley probe rays"; return false; }
            result.ProbeTrace.Sampling.W|=2;
        }
        LumenRadianceMapping mapping;
        if (!BuildLumenRadianceMapping(lighting.Layout,captures,distance.Counts.Y,cardInstances,mapping,error)) return false;
        result.Words=std::move(mapping.Words);
        const auto& layout=lighting.Layout; auto& frame=result.Resolve;
        frame.Counts={result.Trace.Counts.X*upload.Probes.DepthLayers,layout.CardSide,layout.CardSide+LumenSurfaceBorder*2,layout.Columns};
        frame.Atlas={layout.Width,layout.Height,distance.Counts.Y,LumenRadianceThreads*LumenRadianceColumns};
        frame.Mapping={mapping.CardVectorOffset,layout.CardCount,upload.Probes.DepthLayers>1?1u:0u,0};
        frame.Parameters={upload.MinimumFacingCosine,upload.MinimumNormalCosine,upload.PlaneTolerance,upload.VoxelToleranceScale};
        constexpr float CoordinateRoundoff=16*std::numeric_limits<float>::epsilon(),WorldRoundoff=0.5f*std::numeric_limits<float>::epsilon();
        frame.Precision={CoordinateRoundoff,WorldRoundoff,0,0}; output=std::move(result); error.clear(); return true;
    }
    LumenRadianceInfo MakeLumenRadianceInfo(const LumenLightingIdentity& identity,const LumenRadianceTraceUpload& upload,
        const LumenRadianceInput& input,unsigned worldCascades)
    {
        LumenRadianceInfo result; result.Identity=identity; result.Probes=upload.Probes; result.RayCount=input.Resolve.Counts.X;
        result.Hybrid=upload.Hybrid.Enabled; result.WorldCache=worldCascades!=0; result.WorldCacheCascades=worldCascades;
        result.GlobalMaximumSteps=input.Trace.Offsets.W?upload.Trace.GlobalMaximumSteps:0;
        result.HairTransport=(input.Trace.Traversal.W>>LumenHairPointerShift)!=0;
        result.Screen=input.Screen; result.Reflection=input.Reflection; return result;
    }
    bool ValidateLumenRadianceSamples(const LumenRadianceInfo& info,const LumenDistanceSceneInfo& distance,
        const std::vector<unsigned>& cardInstances,const std::vector<LumenRadianceSample>& samples,std::string& error)
    {
        if (samples.size()!=info.RayCount || info.Identity.Occlusion.Epoch!=distance.Identity.Epoch || info.Identity.Occlusion.Serial!=distance.Identity.Serial || info.Identity.Occlusion.Geometry!=distance.Identity.Geometry)
        { error="Lumen radiance feedback has a different source identity or ray count"; return false; }
        if (info.HardwareTracing && !distance.Hardware)
        { error="Lumen hardware radiance has no matching triangle snapshot"; return false; }
        constexpr double NormalTolerance=0.002;
        const bool layeredSource=info.Probes.DirectionalSide && info.Probes.DepthLayers>1;
        for (const auto& sample:samples)
        {
            const auto& hit=sample.Hit; const auto& c=sample.Cache; const auto& r=sample.RadianceCoverage;
            if (!Finite(hit.PositionDistance) || !Finite(hit.NormalResidual) || !Finite(hit.Uncertainty) || !Finite(hit.Medium) || !Finite(r) ||
                hit.Medium.X<0 || hit.Medium.Y<0 || hit.Medium.Z<0 || !UnitInterval(hit.Medium.W) ||
                hit.PositionDistance.W<0 || hit.Uncertainty.X<0 || hit.Uncertainty.Y<hit.Uncertainty.X || hit.Uncertainty.Z<0 ||
                (hit.Uncertainty.Z && !(layeredSource && hit.Metadata.Z==unsigned(LumenDistanceTraceStatus::Surface))) ||
                hit.Uncertainty.W<0 || hit.Uncertainty.W>float(info.GlobalMaximumSteps) || std::floor(hit.Uncertainty.W)!=hit.Uncertainty.W ||
                hit.Metadata.Z>LumenRadianceWorldCache || r.X<0 || r.Y<0 || r.Z<0 || !UnitInterval(r.W) ||
                c.Z>unsigned(LumenRadianceStatus::Invalid) || c.Y>LumenCardDirectionCount || c.W>c.Y)
            { error="Lumen radiance returned invalid metrics, energy, coverage or status"; return false; }
            const double squared=double(hit.NormalResidual.X)*hit.NormalResidual.X+double(hit.NormalResidual.Y)*hit.NormalResidual.Y+double(hit.NormalResidual.Z)*hit.NormalResidual.Z;
            if (hit.Metadata.Z==LumenRadianceWorldCache)
            {
                if (!info.WorldCache || !info.WorldCacheCascades || info.WorldCacheCascades>LumenWorldMaximumCascades || hit.Metadata.W>=info.WorldCacheCascades ||
                    hit.NormalResidual.W || hit.Uncertainty.W || hit.PositionDistance.W<=0 ||
                    hit.PositionDistance.W>info.Probes.MaximumDistance || c.Y || c.W ||
                    c.Z!=unsigned(LumenRadianceStatus::Complete) || r.W<=0)
                { error="Lumen world cache lost its local visibility or directional radiance provenance"; return false; }
                if (hit.Metadata.X!=LumenDistanceInvalid)
                {
                    if (hit.Metadata.X>=distance.UserIds.size() || hit.Metadata.Y!=distance.UserIds[hit.Metadata.X] || std::abs(squared-1)>NormalTolerance ||
                        hit.Uncertainty.X!=hit.Uncertainty.Y || c.X>=LumenMaximumWorldDepthLayers)
                    { error="Layered world radiance lost its validated surface or depth provenance"; return false; }
                }
                else if (hit.Metadata.Y!=LumenDistanceInvalid || squared || hit.Uncertainty.X || hit.Uncertainty.Y || c.X!=LumenLookupMissingCard)
                { error="Lumen world sky/sphere reconstruction retained a surface identity"; return false; }
                continue;
            }
            if (hit.Metadata.Z==LumenRadianceScreenSurface)
            {
                if (!info.Hybrid || !info.Screen.Grid.Y || hit.Metadata.X>=info.Screen.Viewport.X || hit.Metadata.Y>=info.Screen.Viewport.Y ||
                    !hit.Metadata.W || hit.Metadata.W>1024 || std::abs(squared-1)>NormalTolerance || hit.NormalResidual.W || hit.Uncertainty.X || hit.Uncertainty.Y ||
                    hit.PositionDistance.W>info.Probes.MaximumDistance || c.X!=LumenLookupMissingCard || c.Y || c.Z!=unsigned(LumenRadianceStatus::Complete) || c.W || r.W!=1)
                { error="Lumen screen hit lost its pixel, finite surface or independent radiance provenance"; return false; }
                continue;
            }
            if (hit.Metadata.Z==unsigned(LumenDistanceTraceStatus::Surface))
            {
                const bool traversalValid=info.HardwareTracing?hit.Metadata.W==0 && hit.NormalResidual.W==0 &&
                    hit.Uncertainty.X==0 && hit.Uncertainty.Y==0 && hit.Uncertainty.Z==0 && hit.Uncertainty.W==0:hit.Metadata.W!=0;
                if (hit.Metadata.X>=distance.UserIds.size() || hit.Metadata.Y!=distance.UserIds[hit.Metadata.X] || std::abs(squared-1)>NormalTolerance || !traversalValid)
                { error="Lumen radiance surface lost its native instance or unit normal"; return false; }
            }
            else
            {
                if (squared || hit.NormalResidual.W || hit.Uncertainty.X || hit.Uncertainty.Y || c.Z)
                { error="Lumen radiance assigned cache light to an unresolved or missed surface"; return false; }
                if (hit.Metadata.Z==unsigned(LumenDistanceTraceStatus::Miss))
                { if (hit.Metadata.X!=LumenDistanceInvalid || hit.Metadata.Y!=LumenDistanceInvalid) { error="Lumen radiance miss retained a source instance"; return false; } }
                else if (info.Screen.Grid.Y && hit.Metadata.X==LumenDistanceInvalid && hit.Metadata.Y==LumenDistanceInvalid &&
                    !hit.Metadata.W && !hit.PositionDistance.X && !hit.PositionDistance.Y && !hit.PositionDistance.Z && !hit.PositionDistance.W) {}
                else if (hit.Metadata.X>=distance.UserIds.size() || hit.Metadata.Y!=distance.UserIds[hit.Metadata.X])
                { error="Lumen radiance unresolved hit lost its source instance"; return false; }
            }
            if (c.Y)
            {
                if (c.X>=cardInstances.size() || cardInstances[c.X]!=hit.Metadata.X || !c.Z)
                { error="Lumen radiance borrowed another instance's surface cache"; return false; }
            }
            else if (c.X!=LumenLookupMissingCard || c.W || c.Z==unsigned(LumenRadianceStatus::Complete) || c.Z==unsigned(LumenRadianceStatus::Partial))
            { error="Lumen radiance claimed a valid cache without matching directions"; return false; }
            if (c.Z==unsigned(LumenRadianceStatus::Unavailable) || c.Z==unsigned(LumenRadianceStatus::Invalid))
            { if (r.X || r.Y || r.Z || r.W || (!c.Z && c.Y)) { error="Unavailable Lumen radiance retained stale light"; return false; } }
            else if (r.W<=0 || (c.Z==unsigned(LumenRadianceStatus::Complete) && c.W) || (c.Z==unsigned(LumenRadianceStatus::Partial) && !c.W))
            { error="Lumen radiance validity disagrees with lighting uncertainty"; return false; }
        }
        if (info.Probes.DepthLayers!=1)
        {
            const auto& probes=info.Probes;
            if (info.Screen.Grid.Y || info.Reflection.Work.Y || info.Hybrid || info.WorldCache ||
                !ValidateLumenProbeRaySet(probes,error) || info.RayCount!=probes.Origins.size()*probes.RaysPerProbe*probes.DepthLayers)
            { error="Layered radiance feedback has incompatible source sampling"; return false; }
            for (unsigned probe=0;probe<probes.Origins.size();++probe) for (unsigned ray=0;ray<probes.RaysPerProbe;++ray)
                for (unsigned layer=1;layer<probes.DepthLayers;++layer)
                {
                    const auto index=(probe*probes.DepthLayers+layer)*probes.RaysPerProbe+ray;
                    const auto& before=samples[index-probes.RaysPerProbe]; const auto& after=samples[index];
                    if (after.Hit.PositionDistance.W>probes.MaximumDistance || after.Hit.PositionDistance.W<before.Hit.PositionDistance.W ||
                        (before.Hit.Metadata.Z==unsigned(LumenDistanceTraceStatus::Surface) && after.Hit.Metadata.Z==unsigned(LumenDistanceTraceStatus::Surface) &&
                            after.Hit.PositionDistance.W<=before.Hit.PositionDistance.W) ||
                        (before.Hit.Metadata.Z!=unsigned(LumenDistanceTraceStatus::Surface) && std::memcmp(&before,&after,sizeof(after))))
                    { error="World radiance layers lost ordered crossings or terminal provenance"; return false; }
                }
        }
        error.clear(); return true;
    }
    bool ValidateLumenRadianceOrigins(const LumenRadianceInfo& info,const std::vector<LumenProbeOrigin>& origins,std::string& error)
    {
        if (info.Reflection.Work.Y || origins.empty() || origins.size()!=LumenRadianceProbeCount(info)) { error="Lumen origin feedback has a different probe count"; return false; }
        if (!info.Screen.Grid.Y)
        {
            if (std::memcmp(origins.data(),info.Probes.Origins.data(),origins.size()*sizeof(LumenProbeOrigin)))
            { error="Resident Lumen origins changed after upload"; return false; }
        }
        else
        {
            const auto& screen=info.Screen; const auto& viewport=screen.Viewport; const auto& grid=screen.Grid;
            constexpr unsigned MaximumViewport=16384;
            if (!viewport.X || !viewport.Y || viewport.X>MaximumViewport || viewport.Y>MaximumViewport || !viewport.Z ||
                !viewport.W || !grid.X || !ValidLumenScreenProbeLayout(grid) ||
                viewport.W!=(std::uint64_t(viewport.X)+viewport.Z-1)/viewport.Z || grid.X!=(std::uint64_t(viewport.Y)+viewport.Z-1)/viewport.Z ||
                std::uint64_t(viewport.W)*grid.X*(grid.W+1)!=grid.Y)
            { error="Lumen screen origin feedback has an invalid reserved-slot layout"; return false; }
        }
        for (unsigned index=0;info.Screen.Grid.Y && index<origins.size();++index)
        {
            const auto& p=origins[index].PositionMinimum; const auto& tag=origins[index].Metadata; const auto& screen=info.Screen;
            if (!Finite(p) || tag.X!=index || tag.Y>2 || (!tag.Y && (p.W!=screen.Parameters.Y ||
                std::max({std::abs(p.X),std::abs(p.Y),std::abs(p.Z)})>screen.Parameters.Z)) ||
                (tag.Y && (p.X || p.Y || p.Z || p.W)) || (tag.Y==1 && (tag.Z || tag.W)))
            { error="Lumen screen origin lost its identity or validity"; return false; }
            const auto slots=screen.Grid.W+1,tile=index/slots,child=index%slots;
            unsigned left=(tile%screen.Viewport.W)*screen.Viewport.Z,top=(tile/screen.Viewport.W)*screen.Viewport.Z;
            unsigned right=unsigned(std::min(std::uint64_t(left)+screen.Viewport.Z,std::uint64_t(screen.Viewport.X)));
            unsigned bottom=unsigned(std::min(std::uint64_t(top)+screen.Viewport.Z,std::uint64_t(screen.Viewport.Y)));
            std::array<unsigned,Techniques::MaximumLumenAdaptiveProbeLevels> path{}; unsigned depth=0;
            for (auto node=child;node;node=(node-1)/LumenAdaptiveProbeChildren)
                path[depth++]=(node-1)%LumenAdaptiveProbeChildren;
            while (depth)
            {
                const auto quadrant=path[--depth],middleX=(left+right)/2,middleY=(top+bottom)/2;
                if (quadrant&1) left=middleX; else right=middleX;
                if (quadrant&2) top=middleY; else bottom=middleY;
            }
            if (tag.Y!=1 && (tag.Z<left || tag.Z>=right || tag.W<top || tag.W>=bottom))
            { error="Lumen screen origin selected a pixel outside its tile or refinement quadrant"; return false; }
        }
        error.clear(); return true;
    }
    bool ValidateLumenRadianceAssembly(const LumenRadianceAssembly& assembly,std::string& error)
    {
        if (!assembly.Probes.DirectionalSide || !ValidateLumenProbeRaySet(assembly.Probes,error))
        { error="Radiance assembly requires a fixed directional atlas"; return false; }
        std::uint64_t covered=0;
        for (const auto& copy:assembly.Copies)
        {
            if (!copy.Source || !copy.ProbeCount || copy.DestinationProbe!=covered)
            { error="Radiance assembly has missing, overlapping or unordered ranges"; return false; }
            covered+=copy.ProbeCount;
            if (covered>assembly.Probes.Origins.size()) { error="Radiance assembly exceeds its destination"; return false; }
        }
        if (covered!=assembly.Probes.Origins.size()) { error="Radiance assembly leaves uninitialized probes"; return false; }
        error.clear(); return true;
    }
    bool ValidateLumenRadianceCopy(const LumenRadianceInfo& source,const LumenRadianceAssembly& assembly,const LumenRadianceCopy& copy,std::string& error)
    {
        const auto& from=source.Probes; const auto& to=assembly.Probes;
        if (source.HairTransport || source.WorldCache || source.WorldCacheCascades || source.Hybrid || source.Screen.Grid.Y || !from.DirectionalSide || from.DirectionalSide!=to.DirectionalSide ||
            from.RaysPerProbe!=to.RaysPerProbe || from.DepthLayers!=to.DepthLayers || from.Sequence || to.Sequence || from.MaximumDistance!=to.MaximumDistance ||
            source.RayCount!=from.Origins.size()*from.RaysPerProbe*from.DepthLayers || copy.SourceProbe>=from.Origins.size() ||
            copy.ProbeCount>from.Origins.size()-copy.SourceProbe || copy.DestinationProbe>=to.Origins.size() || copy.ProbeCount>to.Origins.size()-copy.DestinationProbe)
        { error="Radiance assembly source has incompatible sampling or range"; return false; }
        for (unsigned probe=0;probe<copy.ProbeCount;++probe)
            if (std::memcmp(&from.Origins[copy.SourceProbe+probe].PositionMinimum,&to.Origins[copy.DestinationProbe+probe].PositionMinimum,sizeof(Float4)))
            { error="Radiance assembly cannot reuse a different origin or near interval"; return false; }
        error.clear(); return true;
    }
    bool BuildLumenReflectionQuery(const ReflectionDesc& camera,const Techniques::ReflectionSettings& settings,
        unsigned width,unsigned height,LumenRadianceTraceUpload& output,std::string& error)
    {
        LumenRadianceTraceUpload query;
        query.Screen.Enabled=true; query.Screen.ViewProjection=camera.ViewProjection;
        query.Screen.Width=width; query.Screen.Height=height; query.Screen.NormalBias=settings.NormalBias;
        query.Probes.Sequence=settings.Lumen.Temporal.Enabled?camera.Sampling.SampleIndex:0;
        query.Probes.MaximumDistance=settings.MaximumDistance;
        query.Reflections={true,settings.Lumen.ResolutionDivisor,settings.MaximumRoughness};
        query.Trace.MaximumSteps=settings.MaximumSteps;
        query.Hybrid.Enabled=settings.Lumen.ScreenTracing; query.Hybrid.MaximumSteps=settings.MaximumSteps;
        query.Hybrid.Thickness=settings.Thickness; query.Hybrid.PixelStride=settings.StridePixels;
        LumenReflectionRayFrame rays;
        if (!BuildLumenReflectionRayFrame(query.Screen,query.Probes,query.Reflections,rays,error)) return false;
        LumenReflectionResolveFrame reconstruction; const auto& filter=settings.Lumen;
        if (!BuildLumenReflectionResolveFrame(rays,{filter.FilterRadius,filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance},
            {},Techniques::ShadingModel::PBR,reconstruction,error)) return false;
        output=std::move(query); error.clear(); return true;
    }
    LumenRadianceHandle NewLumenRadianceHandle() noexcept
    {
        auto value=NextRadiance.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX) if (NextRadiance.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
