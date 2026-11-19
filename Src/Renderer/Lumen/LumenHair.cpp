#include "Renderer/Lumen/LumenHair.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        using Vector=std::array<double,3>;
        constexpr unsigned CornersPerSegment=6,CoordinateGuardUlps=4,WordAlignment=4;
        constexpr double MomentTolerance=64*std::numeric_limits<float>::epsilon();
        struct Segment { const HairVertex *A=nullptr,*B=nullptr; double Length=0; Vector Direction{}; bool Caster=false; };
        struct Accumulator
        {
            double Area=0,CasterArea=0;
            std::array<double,6> Moment{},CasterMoment{};
            Vector Albedo{},Emission{};
        };
        Vector Components(Float3 p) { return {p.X,p.Y,p.Z}; }
        bool Finite(Float4 p) { return std::isfinite(p.X)&&std::isfinite(p.Y)&&std::isfinite(p.Z)&&std::isfinite(p.W); }
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        bool Same(Float4 a,Float4 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; }
        bool RibbonPair(const HairVertex& a,const HairVertex& b)
        {
            return Same(a.Position,b.Position) && Same(a.Tangent,b.Tangent) && Same(a.Reflectance,b.Reflectance) && Same(a.Emission,b.Emission) &&
                a.Ribbon.X==-b.Ribbon.X && a.Ribbon.X<=0 && a.Ribbon.Y==-1 && b.Ribbon.Y==1 && a.Ribbon.Z==b.Ribbon.Z && a.Ribbon.W==b.Ribbon.W;
        }
        bool ReadSegment(const HairStrandGeometry& source,std::size_t at,Segment& output)
        {
            const auto* ids=source.Indices.data()+at;
            for (unsigned corner=0;corner<CornersPerSegment;++corner) if (ids[corner]>=source.Vertices.size()) return false;
            const auto &a=source.Vertices[ids[0]],&b=source.Vertices[ids[1]],&c=source.Vertices[ids[2]],&d=source.Vertices[ids[5]];
            if (ids[3]!=ids[2] || ids[4]!=ids[1] || !RibbonPair(a,b) || !RibbonPair(c,d) ||
                a.Ribbon.Z>c.Ribbon.Z || (a.Ribbon.W<0)!=(c.Ribbon.W<0)) return false;
            const auto p=Components(a.Position),q=Components(c.Position);
            Vector delta{q[0]-p[0],q[1]-p[1],q[2]-p[2]}; const double length=std::hypot(delta[0],delta[1],delta[2]);
            if (!std::isfinite(length)) return false;
            output={&a,&c,length,{},a.Ribbon.W>=0};
            if (length>0) for (unsigned axis=0;axis<3;++axis) output.Direction[axis]=delta[axis]/length;
            return true;
        }
        double Interpolate(double a,double b,double t) { return a+(b-a)*t; }
        float RoundDown(double value)
        {
            float result=float(value);
            if (double(result)>value) result=std::nextafter(result,-std::numeric_limits<float>::infinity());
            return result;
        }
        float RoundUp(double value)
        {
            float result=float(value);
            if (double(result)<value) result=std::nextafter(result,std::numeric_limits<float>::infinity());
            return result;
        }
        std::size_t Address(const std::array<unsigned,3>& cell,const ShaderUInt4& grid)
        { return cell[0]+std::size_t(grid.X)*(cell[1]+std::size_t(grid.Y)*cell[2]); }
        // Averaged covariance is PSD with unit trace. Check all principal minors,
        // allowing only float packing roundoff, not arbitrary forged directions.
        bool PositiveSemidefinite(double x,double y,double z,double xy,double xz,double yz)
        {
            return x>=-MomentTolerance && y>=-MomentTolerance && z>=-MomentTolerance &&
                x*y-xy*xy>=-MomentTolerance && x*z-xz*xz>=-MomentTolerance && y*z-yz*yz>=-MomentTolerance &&
                x*y*z+2*xy*xz*yz-x*yz*yz-y*xz*xz-z*xy*xy>=-MomentTolerance;
        }
        bool ValidMoment(Float4 a,Float4 b,bool active)
        {
            if (!Finite(a) || !Finite(b) || b.Z || b.W) return false;
            if (!active) return a.X==0 && a.Y==0 && a.Z==0 && a.W==0 && b.X==0 && b.Y==0;
            const double x=a.X,y=a.Y,z=a.Z,xy=a.W,xz=b.X,yz=b.Y;
            return x>=0 && y>=0 && z>=0 && std::abs(x+y+z-1)<=MomentTolerance && PositiveSemidefinite(x,y,z,xy,xz,yz);
        }
        bool ValidCasterSubset(const LumenHairVoxel& voxel)
        {
            if (voxel.EmissionCaster.W==0) return true;
            const double fraction=double(voxel.EmissionCaster.W)/voxel.DensityAlbedo.X;
            const auto a=voxel.Orientation,b=voxel.Cross,s=voxel.CasterOrientation,t=voxel.CasterCross;
            // rho * (I-C) is the projected-area tensor. The noncaster difference
            // must also be PSD; a subset cannot create extra directional opacity.
            return PositiveSemidefinite(1-a.X-fraction*(1-s.X),1-a.Y-fraction*(1-s.Y),1-a.Z-fraction*(1-s.Z),
                -a.W+fraction*s.W,-b.X+fraction*t.X,-b.Y+fraction*t.Y);
        }
        bool RepresentableDensity(double value,float& output)
        {
            output=float(value);
            return std::isfinite(output) && output>=0 && (value==0 || output>=std::numeric_limits<float>::min());
        }
        bool ValidOpticalVoxel(const LumenHairVoxel& voxel)
        {
            const auto d=voxel.DensityAlbedo,e=voxel.EmissionCaster;
            return Finite(d) && Finite(e) && d.X>=0 && (d.X==0 || d.X>=std::numeric_limits<float>::min()) &&
                d.Y>=0 && d.Y<=1 && d.Z>=0 && d.Z<=1 && d.W>=0 && d.W<=1 && e.X>=0 && e.Y>=0 && e.Z>=0 && e.W>=0 && e.W<=d.X &&
                (e.W==0 || e.W>=std::numeric_limits<float>::min()) && ValidMoment(voxel.Orientation,voxel.Cross,d.X>0) &&
                ValidMoment(voxel.CasterOrientation,voxel.CasterCross,e.W>0) && ValidCasterSubset(voxel) &&
                (d.X>0 || (d.Y==0 && d.Z==0 && d.W==0 && e.X==0 && e.Y==0 && e.Z==0));
        }
    }

    bool BuildLumenHairVolume(const HairStrandGeometry& source,const Techniques::HairFurSettings& requested,
        const LumenHairBuildSettings& settings,LumenHairVolume& output,std::string& error,const std::function<bool()>& cancelled)
    {
        error.clear(); const auto fail=[&](const char* message) { error=message; return false; };
        if (settings.GridSide<LumenHairMinimumSide || settings.GridSide>LumenHairMaximumSide ||
            !settings.MaximumCells || settings.MaximumCells>LumenHairMaximumCells) return fail("Invalid Lumen hair grid budget");
        constexpr std::size_t MaximumIndices=std::size_t(Techniques::MaximumHairStrands)*Techniques::MaximumHairSegments*CornersPerSegment;
        constexpr std::size_t MaximumVertices=std::size_t(Techniques::MaximumHairStrands)*(Techniques::MaximumHairSegments+1)*2;
        if (source.Indices.size()>MaximumIndices || source.Vertices.size()>MaximumVertices || source.Indices.size()%CornersPerSegment)
            return fail("Lumen hair topology exceeds ribbon limits");
        const auto grooming=Techniques::SanitizeHairFurSettings(requested);
        LumenHairVolume candidate;
        Vector minimum{},maximum{}; bool haveBounds=false;
        for (std::size_t at=0;at<source.Indices.size();at+=CornersPerSegment)
        {
            if (cancelled && cancelled()) return fail("Lumen hair build cancelled");
            const auto* ids=source.Indices.data()+at;
            for (unsigned corner=0;corner<CornersPerSegment;++corner)
            {
                if (ids[corner]>=source.Vertices.size()) return fail("Lumen hair index exceeds its vertex packet");
                const auto& v=source.Vertices[ids[corner]];
                if (!Finite({v.Position.X,v.Position.Y,v.Position.Z,0}) || !Finite({v.Tangent.X,v.Tangent.Y,v.Tangent.Z,0}) ||
                    !Finite(v.Ribbon) || v.Ribbon.Z<0 || v.Ribbon.Z>1 || !Finite(v.Reflectance) || !Finite(v.Emission) ||
                    v.Reflectance.X<0 || v.Reflectance.Y<0 || v.Reflectance.Z<0 || v.Emission.X<0 || v.Emission.Y<0 || v.Emission.Z<0)
                    return fail("Lumen hair ribbon has invalid optical attributes");
            }
            Segment segment; if (!ReadSegment(source,at,segment)) return fail("Lumen hair requires paired centerline ribbons");
            if (segment.Length==0 || (segment.A->Ribbon.X==0 && segment.B->Ribbon.X==0)) continue;
            for (const auto* v:{segment.A,segment.B})
            {
                const auto p=Components(v->Position); const double radius=std::abs(double(v->Ribbon.X));
                for (unsigned axis=0;axis<3;++axis)
                {
                    const double lo=p[axis]-radius,hi=p[axis]+radius;
                    if (!haveBounds) { minimum[axis]=lo; maximum[axis]=hi; }
                    else { minimum[axis]=std::min(minimum[axis],lo); maximum[axis]=std::max(maximum[axis],hi); }
                }
                haveBounds=true;
            }
            ++candidate.Segments;
        }
        if (!haveBounds) { if (cancelled && cancelled()) return fail("Lumen hair build cancelled"); output=std::move(candidate); return true; }
        Vector origin{}; double cell=0;
        for (unsigned axis=0;axis<3;++axis)
        {
            const float lo=RoundDown(minimum[axis]),hi=RoundUp(maximum[axis]);
            if (!std::isfinite(lo) || !std::isfinite(hi)) return fail("Lumen hair bounds exceed native coordinates");
            origin[axis]=lo; maximum[axis]=hi;
            const double spacing=std::max(double(std::nextafter(lo,std::numeric_limits<float>::infinity()))-lo,
                double(hi)-std::nextafter(hi,-std::numeric_limits<float>::infinity()));
            cell=std::max(cell,std::max((maximum[axis]-origin[axis])/settings.GridSide,spacing*CoordinateGuardUlps));
        }
        const float packedCell=RoundUp(cell);
        if (!std::isfinite(packedCell) || !(packedCell>=std::numeric_limits<float>::min())) return fail("Lumen hair cell size is unrepresentable");
        candidate.Header.OriginCell={float(origin[0]),float(origin[1]),float(origin[2]),packedCell}; cell=packedCell;
        std::array<unsigned,3> dimensions{};
        for (unsigned axis=0;axis<3;++axis)
        {
            dimensions[axis]=std::max(1u,unsigned(std::ceil((maximum[axis]-origin[axis])/cell)));
            if (dimensions[axis]>settings.GridSide || !std::isfinite(float(origin[axis]+dimensions[axis]*cell)))
                return fail("Lumen hair grid exceeds its coordinate or axis budget");
        }
        candidate.Header.Grid={dimensions[0],dimensions[1],dimensions[2],0};
        const auto cells=std::size_t(dimensions[0])*dimensions[1]*dimensions[2];
        if (cells>settings.MaximumCells) return fail("Lumen hair grid exceeds its cell budget");
        if (cancelled && cancelled()) return fail("Lumen hair build cancelled");
        std::vector<Accumulator> accumulation(cells); candidate.Voxels.resize(cells);
        const double inverseVolume=1/(cell*cell*cell),quadratureOffset=1/std::sqrt(3.0);
        for (std::size_t at=0;at<source.Indices.size();at+=CornersPerSegment)
        {
            if (cancelled && cancelled()) return fail("Lumen hair build cancelled");
            Segment segment; if (!ReadSegment(source,at,segment)) return fail("Lumen hair packet changed during construction");
            const auto& a=*segment.A; const auto& b=*segment.B;
            if (segment.Length==0 || (a.Ribbon.X==0 && b.Ribbon.X==0)) continue;
            const auto p=Components(a.Position),q=Components(b.Position);
            const Vector delta{q[0]-p[0],q[1]-p[1],q[2]-p[2]};
            const double radiusA=std::abs(double(a.Ribbon.X)),radiusB=std::abs(double(b.Ribbon.X));
            const auto& d=segment.Direction;
            const std::array<double,6> moment{d[0]*d[0],d[1]*d[1],d[2]*d[2],d[0]*d[1],d[0]*d[2],d[1]*d[2]};
            std::array<unsigned,3> coordinate{}; std::array<int,3> step{}; Vector crossing{};
            for (unsigned axis=0;axis<3;++axis)
            {
                coordinate[axis]=std::min(unsigned(std::max(0.0,std::floor((p[axis]-origin[axis])/cell))),dimensions[axis]-1);
                step[axis]=delta[axis]>0?1:delta[axis]<0?-1:0;
                crossing[axis]=std::numeric_limits<double>::infinity();
                if (step[axis])
                {
                    const double plane=origin[axis]+(coordinate[axis]+unsigned(step[axis]>0))*cell;
                    crossing[axis]=std::max(0.0,(plane-p[axis])/delta[axis]);
                }
            }
            double start=0; unsigned visits=0; const unsigned maximumVisits=dimensions[0]+dimensions[1]+dimensions[2]+1;
            while (start<1)
            {
                if (++visits>maximumVisits) return fail("Lumen hair cell traversal did not converge");
                const double end=std::min({1.0,crossing[0],crossing[1],crossing[2]});
                if (end<start) return fail("Lumen hair cell traversal moved backwards");
                const double middle=(start+end)/2,half=(end-start)/2;
                auto& voxel=accumulation[Address(coordinate,candidate.Header.Grid)];
                const double area=segment.Length*(end-start)*(Interpolate(radiusA,radiusB,start)+Interpolate(radiusA,radiusB,end));
                voxel.Area+=area; candidate.ProjectedArea+=area;
                if (segment.Caster) { voxel.CasterArea+=area; candidate.CasterProjectedArea+=area; }
                for (unsigned component=0;component<moment.size();++component)
                { voxel.Moment[component]+=area*moment[component]; if (segment.Caster) voxel.CasterMoment[component]+=area*moment[component]; }
                const Vector reflectanceA{a.Reflectance.X,a.Reflectance.Y,a.Reflectance.Z},reflectanceB{b.Reflectance.X,b.Reflectance.Y,b.Reflectance.Z};
                const Vector emissionA{a.Emission.X,a.Emission.Y,a.Emission.Z},emissionB{b.Emission.X,b.Emission.Y,b.Emission.Z};
                for (double sign:{-1.0,1.0})
                {
                    const double t=middle+sign*half*quadratureOffset,weight=segment.Length*(end-start)*Interpolate(radiusA,radiusB,t);
                    const double height=Interpolate(a.Ribbon.Z,b.Ribbon.Z,t);
                    for (unsigned channel=0;channel<3;++channel)
                    {
                        const double tint=Interpolate(grooming.RootTint[channel],grooming.TipTint[channel],height);
                        voxel.Albedo[channel]+=weight*std::clamp(Interpolate(reflectanceA[channel],reflectanceB[channel],t)*tint,0.0,1.0);
                        voxel.Emission[channel]+=weight*Interpolate(emissionA[channel],emissionB[channel],t);
                    }
                }
                start=end; if (start==1) break;
                for (unsigned axis=0;axis<3;++axis) if (crossing[axis]<=end)
                {
                    const int next=int(coordinate[axis])+step[axis];
                    if (next<0 || next>=int(dimensions[axis])) return fail("Lumen hair centerline escaped its grid");
                    coordinate[axis]=unsigned(next);
                    const double plane=origin[axis]+(coordinate[axis]+unsigned(step[axis]>0))*cell;
                    crossing[axis]=(plane-p[axis])/delta[axis];
                }
            }
        }
        for (std::size_t index=0;index<cells;++index)
        {
            if (cancelled && cancelled()) return fail("Lumen hair build cancelled");
            const auto& sum=accumulation[index]; auto& value=candidate.Voxels[index]; if (sum.Area==0) continue;
            float density=0,caster=0;
            if (!RepresentableDensity(sum.Area*inverseVolume,density) || !RepresentableDensity(sum.CasterArea*inverseVolume,caster))
                return fail("Lumen hair extinction is unrepresentable");
            value.DensityAlbedo={density,float(sum.Albedo[0]/sum.Area),float(sum.Albedo[1]/sum.Area),float(sum.Albedo[2]/sum.Area)};
            value.EmissionCaster={float(sum.Emission[0]/sum.Area),float(sum.Emission[1]/sum.Area),float(sum.Emission[2]/sum.Area),caster};
            const auto pack=[](const std::array<double,6>& m,double weight,Float4& main,Float4& cross)
            { if (weight>0) { main={float(m[0]/weight),float(m[1]/weight),float(m[2]/weight),float(m[3]/weight)}; cross={float(m[4]/weight),float(m[5]/weight),0,0}; } };
            pack(sum.Moment,sum.Area,value.Orientation,value.Cross);
            pack(sum.CasterMoment,sum.CasterArea,value.CasterOrientation,value.CasterCross); ++candidate.ActiveCells;
        }
        if (!ValidateLumenHairVolume(candidate,error)) return false;
        if (cancelled && cancelled()) return fail("Lumen hair build cancelled");
        output=std::move(candidate); return true;
    }

    bool ValidateLumenHairVolume(const LumenHairVolume& volume,std::string& error)
    {
        error.clear(); const auto fail=[&](const char* message) { error=message; return false; };
        const auto grid=volume.Header.Grid; const auto origin=volume.Header.OriginCell;
        if (volume.Voxels.empty())
        {
            if (grid.X || grid.Y || grid.Z || grid.W || !Same(origin,{0,0,0,0}) || volume.Segments || volume.ActiveCells || volume.ProjectedArea || volume.CasterProjectedArea)
                return fail("Empty Lumen hair volume has nonempty metadata");
            return true;
        }
        if (!grid.X || !grid.Y || !grid.Z || grid.X>LumenHairMaximumSide || grid.Y>LumenHairMaximumSide || grid.Z>LumenHairMaximumSide || grid.W ||
            !Finite(origin) || !(origin.W>=std::numeric_limits<float>::min()) || std::size_t(grid.X)*grid.Y*grid.Z!=volume.Voxels.size() ||
            !volume.Segments || volume.Segments>Techniques::MaximumHairStrands*Techniques::MaximumHairSegments ||
            !volume.ActiveCells || volume.ActiveCells>volume.Voxels.size() || !std::isfinite(volume.ProjectedArea) || !(volume.ProjectedArea>0) ||
            !std::isfinite(volume.CasterProjectedArea) || volume.CasterProjectedArea<0 || volume.CasterProjectedArea>volume.ProjectedArea)
            return fail("Invalid Lumen hair grid metadata");
        const Vector corner{double(origin.X)+grid.X*double(origin.W),double(origin.Y)+grid.Y*double(origin.W),double(origin.Z)+grid.Z*double(origin.W)};
        for (double value:corner) if (!std::isfinite(float(value))) return fail("Lumen hair grid corner exceeds native coordinates");
        unsigned active=0; double total=0,casters=0;
        for (const auto& voxel:volume.Voxels)
        {
            const auto d=voxel.DensityAlbedo,e=voxel.EmissionCaster;
            if (!ValidOpticalVoxel(voxel)) return fail("Invalid Lumen hair optical voxel");
            active+=d.X>0; total+=d.X; casters+=e.W;
        }
        const double cellVolume=double(origin.W)*origin.W*origin.W;
        if (active!=volume.ActiveCells || std::abs(total*cellVolume-volume.ProjectedArea)>MomentTolerance*volume.ProjectedArea ||
            std::abs(casters*cellVolume-volume.CasterProjectedArea)>MomentTolerance*volume.ProjectedArea)
            return fail("Lumen hair voxel density does not conserve projected area");
        return true;
    }

    bool AppendLumenHairVolume(const LumenHairVolume& volume,std::size_t maximumWords,
        std::vector<std::uint32_t>& words,unsigned& headerOffset,std::string& error)
    {
        if (!ValidateLumenHairVolume(volume,error)) return false;
        if (volume.Voxels.empty()) { headerOffset=0; return true; }
        constexpr auto HeaderWords=sizeof(LumenHairHeader)/sizeof(std::uint32_t),VoxelWords=sizeof(LumenHairVoxel)/sizeof(std::uint32_t);
        const auto limit=std::min(maximumWords,std::size_t(UINT32_MAX));
        if (words.size()>limit || limit-words.size()<WordAlignment+HeaderWords)
        { error="Lumen hair word header exceeds the transport budget"; return false; }
        const auto first=std::max(std::size_t(WordAlignment),(words.size()+WordAlignment-1)/WordAlignment*WordAlignment),data=first+HeaderWords;
        if (data>limit || volume.Voxels.size()>(limit-data)/VoxelWords)
        { error="Lumen hair cells exceed the transport word budget"; return false; }
        const auto end=data+volume.Voxels.size()*VoxelWords;
        auto header=volume.Header; header.Grid.W=unsigned(data);
        words.reserve(end); words.resize(end,0);
        std::memcpy(words.data()+first,&header,sizeof(header));
        std::memcpy(words.data()+data,volume.Voxels.data(),volume.Voxels.size()*sizeof(LumenHairVoxel));
        headerOffset=unsigned(first); return true;
    }
    bool ValidateLumenHairWords(const std::vector<std::uint32_t>& words,unsigned headerOffset,std::size_t prefixWords,std::string& error)
    {
        error.clear(); const auto fail=[&](const char* message) { error=message; return false; };
        if (prefixWords>words.size()) return fail("Hair prefix exceeds word storage");
        if (!headerOffset) return prefixWords==words.size() || fail("Transport scene has unowned trailing words");
        constexpr auto HeaderWords=sizeof(LumenHairHeader)/sizeof(std::uint32_t),VoxelWords=sizeof(LumenHairVoxel)/sizeof(std::uint32_t);
        const auto expected=std::max(std::size_t(WordAlignment),(prefixWords+WordAlignment-1)/WordAlignment*WordAlignment);
        if (headerOffset!=expected || expected>words.size() || words.size()-expected<HeaderWords)
            return fail("Invalid packed hair header address");
        for (auto at=prefixWords;at<expected;++at) if (words[at]) return fail("Nonzero hair alignment padding");
        LumenHairHeader header; std::memcpy(&header,words.data()+expected,sizeof(header));
        const auto grid=header.Grid; const auto origin=header.OriginCell;
        if (!grid.X || !grid.Y || !grid.Z || grid.X>LumenHairMaximumSide || grid.Y>LumenHairMaximumSide || grid.Z>LumenHairMaximumSide ||
            grid.W!=expected+HeaderWords || !Finite(origin) || origin.W<std::numeric_limits<float>::min()) return fail("Invalid packed hair grid");
        const std::size_t count=std::size_t(grid.X)*grid.Y*grid.Z;
        if (count>(words.size()-grid.W)/VoxelWords || grid.W+count*VoxelWords!=words.size()) return fail("Packed hair cells do not cover their word range");
        for (double corner:{double(origin.X)+grid.X*double(origin.W),double(origin.Y)+grid.Y*double(origin.W),double(origin.Z)+grid.Z*double(origin.W)})
            if (!std::isfinite(float(corner))) return fail("Packed hair corner exceeds native coordinates");
        bool active=false;
        for (std::size_t index=0;index<count;++index)
        {
            LumenHairVoxel voxel; std::memcpy(&voxel,words.data()+grid.W+index*VoxelWords,sizeof(voxel));
            if (!ValidOpticalVoxel(voxel)) return fail("Invalid packed hair optical cell");
            active|=voxel.DensityAlbedo.X>0;
        }
        return active || fail("Packed hair grid contains no optical cells");
    }
}
