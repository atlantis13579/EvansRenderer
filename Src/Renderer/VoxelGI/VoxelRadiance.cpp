#include "Renderer/VoxelGI/VoxelRadiance.h"
#include "Renderer/BakedLighting/BakedDiffuseLighting.h"
#include "Renderer/Lighting/SceneLightCollection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        using Maths::Vector3;
        constexpr std::size_t ClipCapacity=12,CoverageSamples=4;
        constexpr double MinimumArea=1.0e-16;
        constexpr float VectorTolerance=1.0e-16f,MinimumOpacity=1.0e-6f;
        constexpr float ShadowBiasCells=0.02f;
        constexpr std::uint32_t BoundsPaddingCells=1;
        Vector3 Vector(Float3 v) { return {v.X,v.Y,v.Z}; }
        Float3 Native(Vector3 v) { return {v.x,v.y,v.z}; }
        bool Finite(Float3 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z); }
        Vector3 Unit(Vector3 value,Vector3 fallback)
        { return value.LengthSquared()>VectorTolerance ? value.Normalized() : fallback; }
        std::uint32_t Index(std::uint32_t x,std::uint32_t y,std::uint32_t z,std::uint32_t size)
        { return x+size*(y+size*z); }
        struct ClipVertex { std::array<double,3> Position{},Barycentric{}; };
        struct Polygon { std::array<ClipVertex,ClipCapacity> Vertices{}; std::size_t Count=0; };
        Polygon Clip(Polygon polygon,unsigned axis,bool upper)
        {
            Polygon output; if (!polygon.Count) return output;
            auto previous=polygon.Vertices[polygon.Count-1];
            auto distance=[&](const ClipVertex& v){return upper ? 1-v.Position[axis] : v.Position[axis];};
            double previousDistance=distance(previous);
            for (std::size_t index=0;index<polygon.Count;++index)
            {
                const auto current=polygon.Vertices[index]; const double currentDistance=distance(current);
                if ((currentDistance>=0)!=(previousDistance>=0))
                {
                    const double fraction=previousDistance/(previousDistance-currentDistance); ClipVertex intersection;
                    for (unsigned component=0;component<3;++component)
                    {
                        intersection.Position[component]=previous.Position[component]+fraction*(current.Position[component]-previous.Position[component]);
                        intersection.Barycentric[component]=previous.Barycentric[component]+fraction*(current.Barycentric[component]-previous.Barycentric[component]);
                    }
                    if (output.Count<ClipCapacity) output.Vertices[output.Count++]=intersection;
                }
                if (currentDistance>=0 && output.Count<ClipCapacity) output.Vertices[output.Count++]=current;
                previous=current; previousDistance=currentDistance;
            }
            return output;
        }
        double Area(const ClipVertex& a,const ClipVertex& b,const ClipVertex& c)
        {
            std::array<double,3> u{},v{};
            for (unsigned axis=0;axis<3;++axis) { u[axis]=b.Position[axis]-a.Position[axis]; v[axis]=c.Position[axis]-a.Position[axis]; }
            return std::hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])*0.5;
        }
        struct Accumulator
        {
            double Weight=0;
            std::array<double,3> Position{},Normal{},Geometric{},Diffuse{},Emission{};
            void Add(const SceneRayHit& hit,double weight)
            {
                Weight+=weight; const Vector3 values[]{Vector(hit.Position),Vector(hit.Normal),Vector(hit.GeometricNormal),Vector(hit.Diffuse),Vector(hit.Emission)};
                std::array<double,3>* fields[]{&Position,&Normal,&Geometric,&Diffuse,&Emission};
                for (unsigned field=0;field<std::size(values);++field)
                    for (unsigned axis=0;axis<3;++axis) (*fields[field])[axis]+=values[field][axis]*weight;
            }
            SceneRayHit Surface(unsigned face) const
            {
                auto mean=[&](const std::array<double,3>& v){return Vector3{float(v[0]/Weight),float(v[1]/Weight),float(v[2]/Weight)};};
                Vector3 fallback; fallback[face/2]=face%2 ? -1.0f : 1.0f;
                SceneRayHit hit; hit.Position=Native(mean(Position)); hit.Normal=Native(Unit(mean(Normal),fallback));
                hit.GeometricNormal=Native(Unit(mean(Geometric),fallback)); hit.Diffuse=Native(mean(Diffuse)); hit.Emission=Native(mean(Emission));
                return hit;
            }
        };
        bool ValidGrid(VoxelGrid grid)
        {
            const float side=grid.CellSize*grid.Resolution;
            return Finite(grid.Origin) && Finite(Native(Vector(grid.Origin)+Vector3(side))) && std::isfinite(grid.CellSize) && grid.CellSize>0 &&
                std::isfinite(grid.CellSize*grid.Resolution) && grid.Resolution>=Techniques::MinimumVoxelResolution &&
                grid.Resolution<=Techniques::MaximumVoxelResolution && (grid.Resolution&(grid.Resolution-1))==0;
        }
        Float4 Over(Float4 front,Float4 back)
        { const float through=1-front.W; return {front.X+through*back.X,front.Y+through*back.Y,front.Z+through*back.Z,front.W+through*back.W}; }
    }

    bool VoxelizeScene(std::shared_ptr<const SceneRayQuery> scene,const Techniques::VoxelGISettings& request,
        VoxelizedScene& output,std::string& error,const VoxelCancellation& cancelled)
    {
        error.clear(); const auto fail=[&](const char* message){error=message;return false;};
        if (!scene) return fail("Voxelization requires an immutable scene query");
        if (cancelled && cancelled()) return fail("Voxelization cancelled");
        const auto settings=Techniques::SanitizeVoxelGISettings(request);
        Vector3 minimum,maximum;
        if (settings.AutomaticBounds && scene->GetTriangleCount()) { minimum=scene->GetBounds().Min; maximum=scene->GetBounds().Max; }
        else if (settings.AutomaticBounds) { minimum=Vector3(-1); maximum=Vector3(1); }
        else { minimum={settings.BoundsMinimum[0],settings.BoundsMinimum[1],settings.BoundsMinimum[2]}; maximum={settings.BoundsMaximum[0],settings.BoundsMaximum[1],settings.BoundsMaximum[2]}; }
        const auto extent=maximum-minimum; const float longest=std::max({extent.x,extent.y,extent.z});
        VoxelizedScene result; result.Query=std::move(scene); auto& grid=result.Grid; grid.Resolution=settings.Resolution;
        grid.CellSize=longest/float(grid.Resolution-2*BoundsPaddingCells);
        grid.Origin=Native((minimum+maximum)*0.5f-Vector3(grid.CellSize*grid.Resolution*0.5f));
        if (!ValidGrid(grid)) return fail("Voxel grid has invalid or unrepresentable world bounds");
        const auto cells=grid.CellCount(); const auto total=std::size_t(cells)*Techniques::VoxelDirectionCount;
        std::vector<double> coverage(total,0); std::unordered_map<std::uint32_t,Accumulator> accumulators;
        // Four equal-area subtriangle centroids integrate alpha-test coverage.
        // Clipping carries original barycentrics; UV seams and mirrored instances
        // therefore use the same material query as all other scene-ray producers.
        constexpr std::array<std::array<double,3>,CoverageSamples> quadrature{{{2.0/3,1.0/6,1.0/6},{1.0/6,2.0/3,1.0/6},{1.0/6,1.0/6,2.0/3},{1.0/3,1.0/3,1.0/3}}};
        for (std::size_t triangle=0;triangle<result.Query->GetTriangleCount();++triangle)
        {
            if (cancelled && cancelled()) return fail("Voxelization cancelled");
            SceneRayTriangle source; if (!result.Query->GetTriangle(static_cast<std::uint32_t>(triangle),source)) return fail("Voxel triangle lookup failed");
            std::array<std::array<double,3>,3> positions{};
            std::array<int,3> first{},last{}; bool outside=false;
            for (unsigned axis=0;axis<3;++axis)
            {
                double low=std::numeric_limits<double>::infinity(),high=-low;
                for (unsigned corner=0;corner<3;++corner)
                {
                    const auto p=Vector(source.Corners[corner].Position); const auto origin=Vector(grid.Origin);
                    const double value=(double(p[axis])-origin[axis])/grid.CellSize;
                    positions[corner][axis]=value; low=std::min(low,value); high=std::max(high,value);
                }
                outside=outside || high<0 || low>grid.Resolution;
                first[axis]=std::clamp(int(std::floor(std::clamp(low,-1.0,double(grid.Resolution)))),0,int(grid.Resolution)-1);
                last[axis]=std::clamp(int(std::floor(std::clamp(high,-1.0,double(grid.Resolution)))),0,int(grid.Resolution)-1);
            }
            if (outside) continue;
            for (int z=first[2];z<=last[2];++z) for (int y=first[1];y<=last[1];++y) for (int x=first[0];x<=last[0];++x)
            {
                ++result.TriangleCellTests;
                if (cancelled && (result.TriangleCellTests%1024)==0 && cancelled()) return fail("Voxelization cancelled");
                Polygon polygon; polygon.Count=3; const int cell[]{x,y,z};
                for (unsigned corner=0;corner<3;++corner) for (unsigned axis=0;axis<3;++axis)
                { polygon.Vertices[corner].Position[axis]=positions[corner][axis]-cell[axis]; polygon.Vertices[corner].Barycentric[axis]=corner==axis ? 1.0 : 0.0; }
                for (unsigned axis=0;axis<3 && polygon.Count;++axis) { polygon=Clip(polygon,axis,false); polygon=Clip(polygon,axis,true); }
                if (polygon.Count<3) continue;
                const auto cellIndex=Index(std::uint32_t(x),std::uint32_t(y),std::uint32_t(z),grid.Resolution);
                bool clipped=false;
                for (std::size_t fan=1;fan+1<polygon.Count;++fan)
                {
                    const ClipVertex vertices[]{polygon.Vertices[0],polygon.Vertices[fan],polygon.Vertices[fan+1]};
                    const double area=Area(vertices[0],vertices[1],vertices[2]); if (!(area>MinimumArea)) continue; clipped=true;
                    for (auto location:quadrature)
                    {
                        Vector3 barycentric;
                        for (unsigned axis=0;axis<3;++axis) for (unsigned corner=0;corner<3;++corner)
                            barycentric[axis]+=float(vertices[corner].Barycentric[axis]*location[corner]);
                        barycentric=Vector3::Max(barycentric,Vector3::Zero()); barycentric/=barycentric.x+barycentric.y+barycentric.z;
                        SceneRayHit hit; if (!result.Query->SampleSurface(static_cast<std::uint32_t>(triangle),Native(barycentric),true,hit)) continue;
                        ++result.AcceptedSamples; const auto normal=Vector(hit.GeometricNormal);
                        for (unsigned axis=0;axis<3;++axis)
                        {
                            const double projected=area*std::abs(normal[axis])/CoverageSamples; if (!(projected>0)) continue;
                            const auto positive=(axis*2)*cells+cellIndex,negative=(axis*2+1)*cells+cellIndex;
                            coverage[positive]+=projected; coverage[negative]+=projected;
                            accumulators[normal[axis]>=0 ? positive : negative].Add(hit,projected);
                        }
                    }
                }
                if (clipped) ++result.ClippedTriangles;
            }
        }
        result.Opacity.resize(total);
        for (std::size_t index=0;index<total;++index) result.Opacity[index]=float(std::min(coverage[index],1.0));
        std::vector<std::uint32_t> keys; keys.reserve(accumulators.size()); for (const auto& entry:accumulators) keys.push_back(entry.first);
        std::sort(keys.begin(),keys.end()); result.Samples.reserve(keys.size());
        for (auto key:keys)
        {
            const auto& accumulated=accumulators.at(key); VoxelSurfaceSample sample;
            sample.FaceCell=key; sample.FrontFraction=float(std::clamp(accumulated.Weight/coverage[key],0.0,1.0));
            sample.Surface=accumulated.Surface(key/cells); result.Samples.push_back(sample);
        }
        if (cancelled && cancelled()) return fail("Voxelization cancelled");
        output=std::move(result); return true;
    }

    static bool BuildVoxelRadianceMips(VoxelGrid grid,const std::vector<Float4>& base,VoxelRadianceVolume& output,std::string& error)
    {
        error.clear(); if (!ValidGrid(grid) || base.size()!=std::size_t(grid.CellCount())*Techniques::VoxelDirectionCount)
        { error="Voxel radiance needs a valid grid and six complete directional faces"; return false; }
        for (auto value:base) if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z) || !std::isfinite(value.W) ||
            value.W<0 || value.W>1 || value.X<0 || value.Y<0 || value.Z<0 ||
            std::max({value.X,value.Y,value.Z})>MaximumBakedRadiance*value.W+MinimumOpacity)
        { error="Voxel radiance is not finite nonnegative premultiplied coverage"; return false; }
        VoxelRadianceVolume result; result.Grid=grid; std::size_t count=0;
        for (std::uint32_t dimension=grid.Resolution;dimension;dimension/=2)
        {
            auto& level=result.Levels[result.MipCount++]; level={dimension,static_cast<std::uint32_t>(count),dimension*dimension*dimension,0};
            count+=std::size_t(level.FaceStride)*Techniques::VoxelDirectionCount;
        }
        result.Height=static_cast<std::uint32_t>((count+result.Width-1)/result.Width);
        result.Texels.resize(std::size_t(result.Width)*result.Height,Float4{0,0,0,0}); std::copy(base.begin(),base.end(),result.Texels.begin());
        for (std::uint32_t mip=1;mip<result.MipCount;++mip)
        {
            const auto source=result.Levels[mip-1],destination=result.Levels[mip];
            for (unsigned face=0;face<Techniques::VoxelDirectionCount;++face)
                for (unsigned z=0;z<destination.Dimension;++z) for (unsigned y=0;y<destination.Dimension;++y) for (unsigned x=0;x<destination.Dimension;++x)
                {
                    Float4 sum{0,0,0,0}; const unsigned axis=face/2,other0=(axis+1)%3,other1=(axis+2)%3;
                    for (unsigned first=0;first<2;++first) for (unsigned second=0;second<2;++second)
                    {
                        unsigned p[]{x*2,y*2,z*2}; p[other0]+=first; p[other1]+=second; p[axis]+=face%2 ? 0u : 1u;
                        const auto front=result.Texels[source.Offset+face*source.FaceStride+Index(p[0],p[1],p[2],source.Dimension)];
                        p[axis]=p[axis]^1u;
                        const auto back=result.Texels[source.Offset+face*source.FaceStride+Index(p[0],p[1],p[2],source.Dimension)];
                        const auto value=Over(front,back); sum.X+=value.X; sum.Y+=value.Y; sum.Z+=value.Z; sum.W+=value.W;
                    }
                    constexpr float TransverseRayWeight=0.25f;
                    result.Texels[destination.Offset+face*destination.FaceStride+Index(x,y,z,destination.Dimension)]={sum.X*TransverseRayWeight,sum.Y*TransverseRayWeight,sum.Z*TransverseRayWeight,sum.W*TransverseRayWeight};
                }
        }
        output=std::move(result); return true;
    }
    bool InjectVoxelRadiance(const VoxelizedScene& scene,const LightingState& lighting,bool shadows,
        VoxelRadianceVolume& output,std::string& error,const VoxelCancellation& cancelled)
    {
        error.clear(); if (!scene.Query || !ValidGrid(scene.Grid) || scene.Opacity.size()!=std::size_t(scene.Grid.CellCount())*Techniques::VoxelDirectionCount)
        { error="Voxel injection requires a complete immutable voxel scene"; return false; }
        if (cancelled && cancelled()) { error="Voxel radiance injection cancelled"; return false; }
        std::vector<SceneLight> lights;
        if (lighting.SceneLights) lights=lighting.SceneLights->GetLights();
        else lights.assign(lighting.Lights.begin(),lighting.Lights.begin()+std::min(lighting.LightCount,MaximumSceneLights));
        std::string lightError; if (!SceneLightCollection::Create(lights,-1,lightError)) { error=lightError; return false; }
        std::vector<Float4> base(scene.Opacity.size());
        for (std::size_t index=0;index<base.size();++index) base[index].W=scene.Opacity[index];
        BakedDiffuseLightingSettings settings; settings.Shadows=shadows; settings.SurfaceBias=scene.Grid.CellSize*ShadowBiasCells;
        std::uint64_t queries=0;
        for (const auto& sample:scene.Samples)
        {
            if (cancelled && cancelled()) { error="Voxel radiance injection cancelled"; return false; }
            if (sample.FaceCell>=base.size() || !std::isfinite(sample.FrontFraction) || sample.FrontFraction<0 || sample.FrontFraction>1)
            { error="Voxel surface representative is invalid"; return false; }
            auto radiance=EvaluateBakedDiffuse(*scene.Query,sample.Surface,lights,settings,queries);
            if (!Finite(Native(radiance))) { error="Voxel diffuse lighting produced nonfinite radiance"; return false; }
            radiance=Vector3::Min(Vector3::Max(radiance,Vector3::Zero()),Vector3(MaximumBakedRadiance));
            const float coverage=base[sample.FaceCell].W; radiance*=coverage*sample.FrontFraction;
            base[sample.FaceCell]={radiance.x,radiance.y,radiance.z,coverage};
        }
        VoxelRadianceVolume result; if (!BuildVoxelRadianceMips(scene.Grid,base,result,error)) return false;
        if (cancelled && cancelled()) { error="Voxel radiance injection cancelled"; return false; }
        result.ShadowQueries=queries; output=std::move(result); return true;
    }
}
