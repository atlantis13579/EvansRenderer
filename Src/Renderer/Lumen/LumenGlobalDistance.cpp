#include "Renderer/Lumen/LumenGlobalDistance.h"
#include "Renderer/IRHI.h"

#include <cstring>

namespace Evans::Renderer
{
    bool BuildLumenGlobalDistancePlan(const LumenDistanceSceneInfo& scene,const LumenGlobalDistanceSettings& settings,LumenGlobalDistancePlan& output,std::string& error)
    {
        const auto fail=[&](const char* message) { error=message; return false; };
        if (scene.Global) return fail("Global distance composition requires an original mesh scene");
        LumenDistanceTraceUpload query; query.Rays.push_back({{0,0,0,0},{0,0,1,1}});
        query.HitToleranceCells=0; LumenDistanceTraceFrame validated;
        if (!BuildLumenDistanceTraceFrame(scene,query,validated,error)) return false;
        if (settings.GridSide<4 || settings.GridSide>LumenGlobalDistanceMaximumSide || settings.CascadeCount<1 ||
            settings.CascadeCount>LumenGlobalDistanceMaximumCascades || !std::isfinite(settings.HitToleranceCells) || settings.HitToleranceCells<0 || settings.HitToleranceCells>=scene.MaximumHitToleranceCells || !std::isfinite(settings.CellSize) || settings.CellSize<=0)
            return fail("Global distance clipmaps require bounded dimensions and positive finite spacing");
        const double camera[]{settings.CameraPosition.X,settings.CameraPosition.Y,settings.CameraPosition.Z};
        for (double value:camera) if (!std::isfinite(value)) return fail("Global distance camera must be finite");
        constexpr unsigned VectorWords=4,HeaderWords=sizeof(LumenGlobalDistanceHeader)/sizeof(std::uint32_t);
        constexpr float BandCells=8,CoordinateHeadroom=4,MinimumCellsPerRoundoff=16,Roundoff=32*std::numeric_limits<float>::epsilon();
        LumenGlobalDistancePlan candidate; candidate.Settings=settings;
        auto& frame=candidate.Frame; frame.Counts=scene.Counts; frame.Offsets=scene.Offsets;
        auto& header=frame.Header;
        const unsigned headerWord=((std::max(1u,scene.Counts.W)+VectorWords-1)/VectorWords)*VectorWords;
        const unsigned count=settings.GridSide*settings.GridSide*settings.GridSide;
        header.Layout={settings.CascadeCount,settings.GridSide,count*settings.CascadeCount,headerWord};
        header.Numeric={settings.HitToleranceCells,Roundoff,0,0};
        candidate.WordCount=headerWord+HeaderWords+header.Layout.Z*VectorWords;
        for (unsigned level=0;level<settings.CascadeCount;++level)
        {
            auto& grid=header.Grids[level]; const double cell=std::ldexp(double(settings.CellSize),int(level));
            float origin[3]{}; double magnitude=cell;
            for (unsigned axis=0;axis<3;++axis)
            {
                const double first=(std::floor(camera[axis]/cell)-double(settings.GridSide/2))*cell;
                const double last=first+(settings.GridSide-1)*cell;
                if (!std::isfinite(first) || !std::isfinite(last) || std::max(std::abs(first),std::abs(last))>std::numeric_limits<float>::max()/CoordinateHeadroom)
                    return fail("Global distance bounds exceed native coordinate range");
                origin[axis]=float(first); magnitude=std::max({magnitude,std::abs(first),std::abs(last)});
            }
            if (!std::isfinite(cell*BandCells) || cell*BandCells>std::numeric_limits<float>::max()/CoordinateHeadroom ||
                Roundoff*magnitude>cell/MinimumCellsPerRoundoff || float(cell)<=0)
                return fail("Global distance spacing is too fine for its world coordinate precision");
            grid.OriginCell={origin[0],origin[1],origin[2],float(cell)};
            grid.Metrics={std::nextafter(float(std::sqrt(3.0)*cell*0.5+Roundoff*magnitude),std::numeric_limits<float>::infinity()),
                float(cell*BandCells),float(Roundoff*magnitude),0};
            grid.Storage={headerWord+HeaderWords+level*count*VectorWords,count,0,0};
        }
        output=std::move(candidate); error.clear(); return true;
    }
    bool ValidateLumenGlobalDistancePlan(const LumenDistanceSceneInfo& scene,const LumenGlobalDistancePlan& plan,std::string& error)
    {
        auto base=scene; base.Global.reset(); LumenGlobalDistancePlan expected;
        if (!BuildLumenGlobalDistancePlan(base,plan.Settings,expected,error)) return false;
        if (expected.WordCount!=plan.WordCount || std::memcmp(&expected.Frame,&plan.Frame,sizeof(plan.Frame)))
        { error="Global distance layout does not match its source scene and clipmap settings"; return false; }
        return true;
    }
    namespace
    {
        bool SameFloatBits(float a,float b) { return !std::memcmp(&a,&b,sizeof(float)); }
        bool StableGlobalCoordinate(float origin,float spacing,unsigned coordinate,float& result)
        {
            // Different drivers may contract multiply/add. Reuse across changed
            // origins only when both evaluations agree bit-for-bit, so copying
            // never moves an SDF sample to a merely nearby world position.
            volatile float offset=float(coordinate)*spacing;
            result=origin+offset;
            return SameFloatBits(result,std::fma(float(coordinate),spacing,origin));
        }
    }
    bool BuildLumenGlobalDistanceUpdate(const LumenDistanceSceneInfo& scene,const LumenGlobalDistancePlan& next,
        const LumenGlobalDistancePlan* previous,LumenGlobalDistanceUpdate& output,std::string& error)
    {
        if (!ValidateLumenGlobalDistancePlan(scene,next,error) || (previous && !ValidateLumenGlobalDistancePlan(scene,*previous,error))) return false;
        const auto& settings=next.Settings;
        const bool compatible=previous && settings.GridSide==previous->Settings.GridSide && settings.CascadeCount==previous->Settings.CascadeCount &&
            settings.CellSize==previous->Settings.CellSize && settings.HitToleranceCells==previous->Settings.HitToleranceCells;
        const unsigned side=settings.GridSide,volume=side*side*side;
        constexpr unsigned VoxelWords=sizeof(Float4)/sizeof(std::uint32_t);
        LumenGlobalDistanceUpdate candidate; candidate.Compose.reserve(next.Frame.Header.Layout.Z);
        for (unsigned level=0;level<settings.CascadeCount;++level)
        {
            const auto& grid=next.Frame.Header.Grids[level];
            const auto* old=compatible?&previous->Frame.Header.Grids[level]:nullptr;
            std::array<std::array<int,LumenGlobalDistanceMaximumSide>,3> mapping;
            for (auto& axis:mapping) axis.fill(-1);
            if (old)
            {
                const float current[]{grid.OriginCell.X,grid.OriginCell.Y,grid.OriginCell.Z};
                const float prior[]{old->OriginCell.X,old->OriginCell.Y,old->OriginCell.Z};
                for (unsigned axis=0;axis<3;++axis)
                for (unsigned i=0;i<side;++i)
                {
                    if (SameFloatBits(current[axis],prior[axis])) { mapping[axis][i]=int(i); continue; }
                    float point=0; if (!StableGlobalCoordinate(current[axis],grid.OriginCell.W,i,point)) continue;
                    for (unsigned j=0;j<side;++j)
                    {
                        float former=0;
                        if (StableGlobalCoordinate(prior[axis],old->OriginCell.W,j,former) && SameFloatBits(point,former))
                        { mapping[axis][i]=int(j); break; }
                    }
                }
            }
            for (unsigned z=0;z<side;++z) for (unsigned y=0;y<side;++y) for (unsigned x=0;x<side;++x)
            {
                const unsigned voxel=(z*side+y)*side+x;
                const int sx=mapping[0][x],sy=mapping[1][y],sz=mapping[2][z];
                if (sx<0 || sy<0 || sz<0) { candidate.Compose.push_back(level*volume+voxel); continue; }
                const unsigned source=old->Storage.X+((unsigned(sz)*side+unsigned(sy))*side+unsigned(sx))*VoxelWords;
                const unsigned destination=grid.Storage.X+voxel*VoxelWords;
                if (!candidate.Copies.empty() && candidate.Copies.back().SourceWord+candidate.Copies.back().WordCount==source &&
                    candidate.Copies.back().DestinationWord+candidate.Copies.back().WordCount==destination)
                    candidate.Copies.back().WordCount+=VoxelWords;
                else candidate.Copies.push_back({source,destination,VoxelWords});
                ++candidate.ReusedVoxels;
            }
        }
        output=std::move(candidate); error.clear(); return true;
    }
    namespace
    {
        bool SameGlobalQuality(const LumenGlobalDistanceSettings& a,const LumenGlobalDistanceSettings& b)
        { return a.GridSide==b.GridSide && a.CascadeCount==b.CascadeCount && a.CellSize==b.CellSize && a.HitToleranceCells==b.HitToleranceCells; }
        bool SameGlobalCells(const LumenGlobalDistanceSettings& a,const LumenGlobalDistanceSettings& b)
        {
            if (!SameGlobalQuality(a,b)) return false;
            const double current[]{a.CameraPosition.X,a.CameraPosition.Y,a.CameraPosition.Z};
            const double previous[]{b.CameraPosition.X,b.CameraPosition.Y,b.CameraPosition.Z};
            for (unsigned level=0;level<a.CascadeCount;++level)
            {
                const double spacing=std::ldexp(double(a.CellSize),int(level));
                for (unsigned axis=0;axis<3;++axis) if (std::floor(current[axis]/spacing)!=std::floor(previous[axis]/spacing)) return false;
            }
            return true;
        }
    }
    LumenGlobalDistanceCache::~LumenGlobalDistanceCache() { Cancel(); if (Published) RHI.DestroyLumenDistanceScene(Published); }
    void LumenGlobalDistanceCache::Cancel()
    {
        if (Pending) RHI.CancelLumenGlobalDistanceUpdate(Pending);
        Pending={}; PendingSource={}; State=Published?LumenGlobalDistanceState::Ready:LumenGlobalDistanceState::Empty;
    }
    bool LumenGlobalDistanceCache::CanUseSnapshot(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings) const
    { return Published && Source==source && SameGlobalQuality(Settings,settings); }
    bool LumenGlobalDistanceCache::Request(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,std::string& error)
    {
        if (!source || !std::isfinite(settings.CellSize) || settings.CellSize<=0 || !std::isfinite(settings.CameraPosition.X) ||
            !std::isfinite(settings.CameraPosition.Y) || !std::isfinite(settings.CameraPosition.Z))
        { error="Global distance updates require a source, finite camera and positive spacing"; return false; }
        // Finish a frozen camera generation before following the latest request.
        // Geometry/quality changes cancel it; an older source is never rebound.
        if (Pending && (!(PendingSource==source) || !SameGlobalQuality(PendingSettings,settings))) Cancel();
        LatestSettings=settings;
        if (Pending) { error.clear(); return true; }
        if (Published && Source==source && SameGlobalCells(Settings,settings))
        { State=LumenGlobalDistanceState::Ready; error.clear(); return true; }
        LumenGlobalDistanceTask task;
        if (!RHI.BeginLumenGlobalDistanceUpdate(source,settings,Source==source?Published:LumenDistanceSceneHandle{},task))
        { State=LumenGlobalDistanceState::Failed; error=RHI.GetLastError(); return false; }
        Pending=task; PendingSource=source; PendingSettings=settings; Progress={}; State=LumenGlobalDistanceState::Updating;
        error.clear(); return true;
    }
    bool LumenGlobalDistanceCache::Advance(unsigned voxelBudget,std::string& error)
    {
        if (!Pending) { error.clear(); return true; }
        bool ready=false; LumenDistanceSceneHandle candidate;
        if (!RHI.AdvanceLumenGlobalDistanceUpdate(Pending,voxelBudget,ready,candidate,Progress))
        { State=LumenGlobalDistanceState::Failed; error=RHI.GetLastError(); return false; }
        if (!ready) { error.clear(); return true; }
        Pending={};
        if (Published && !RHI.DestroyLumenDistanceScene(Published))
        { error=RHI.GetLastError(); RHI.DestroyLumenDistanceScene(candidate); State=LumenGlobalDistanceState::Failed; return false; }
        Published=candidate; Source=PendingSource; Settings=PendingSettings; PendingSource={};
        State=SameGlobalCells(Settings,LatestSettings)?LumenGlobalDistanceState::Ready:LumenGlobalDistanceState::Updating;
        error.clear(); return true;
    }
    bool LumenGlobalDistanceCache::Update(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,std::string& error)
    {
        Cancel();
        bool same=bool(Published) && source==Source && settings.GridSide==Settings.GridSide && settings.CascadeCount==Settings.CascadeCount &&
            settings.CellSize==Settings.CellSize && settings.HitToleranceCells==Settings.HitToleranceCells;
        const double current[]{settings.CameraPosition.X,settings.CameraPosition.Y,settings.CameraPosition.Z};
        const double previous[]{Settings.CameraPosition.X,Settings.CameraPosition.Y,Settings.CameraPosition.Z};
        for (unsigned level=0;same && level<settings.CascadeCount;++level)
        {
            const double spacing=std::ldexp(double(settings.CellSize),int(level));
            for (unsigned axis=0;axis<3;++axis) if (std::floor(current[axis]/spacing)!=std::floor(previous[axis]/spacing)) same=false;
        }
        if (same) { error.clear(); return true; }
        LumenDistanceSceneHandle next;
        if (!RHI.CreateLumenGlobalDistanceScene(source,settings,next,source==Source?Published:LumenDistanceSceneHandle{})) { error=RHI.GetLastError(); return false; }
        if (Published && !RHI.DestroyLumenDistanceScene(Published))
        { error=RHI.GetLastError(); RHI.DestroyLumenDistanceScene(next); return false; }
        Published=next; Source=source; Settings=settings; State=LumenGlobalDistanceState::Ready; error.clear(); return true;
    }
}
