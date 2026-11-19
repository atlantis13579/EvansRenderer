#include "Renderer/Lumen/LumenDistanceField.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Evans::Renderer
{
    /* Mesh signed distance fields separate Lumen's intersection representation
       from the material surface cache. This independent producer uses exact
       triangle point distances within a truncated band, and an antipodal ray
       vote for sign. Sparse bricks retain constant exterior/interior regions
       without storing their samples; brick halos support trilinear filtering.
       Daniel Wright, "Dynamic Occlusion with Signed Distance Fields" (2015);
       Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5"
       (2022). Neither Unreal nor Embree is a build/runtime dependency.
       https://dev.epicgames.com/documentation/en-us/unreal-engine/mesh-distance-fields-in-unreal-engine
    */
    namespace
    {
        using Vec=Maths::Vector3;
        using Point=std::array<double,3>;
        constexpr unsigned MinimumResolution=4,MaximumResolution=128,MaximumSignRays=128;
        constexpr double Pi=3.14159265358979323846;
        bool Finite(Float3 value) { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }
        Point Convert(Vec p) { return {p.x,p.y,p.z}; }
        Point Subtract(Point a,Point b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
        double Dot(Point a,Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
        double SegmentSquared(Point p,Point a,Point b)
        {
            const auto ab=Subtract(b,a),ap=Subtract(p,a); const double length=Dot(ab,ab);
            const double t=length>0?std::clamp(Dot(ap,ab)/length,0.0,1.0):0;
            const Point delta{ap[0]-t*ab[0],ap[1]-t*ab[1],ap[2]-t*ab[2]}; return Dot(delta,delta);
        }
        double TriangleSquared(Point p,Vec va,Vec vb,Vec vc)
        {
            const auto a=Convert(va),b=Convert(vb),c=Convert(vc),ab=Subtract(b,a),ac=Subtract(c,a),ap=Subtract(p,a);
            const double aa=Dot(ab,ab),bb=Dot(ac,ac),cross=Dot(ab,ac),pa=Dot(ap,ab),pc=Dot(ap,ac),determinant=aa*bb-cross*cross;
            if (determinant>std::numeric_limits<double>::epsilon()*aa*bb)
            {
                const double u=(pa*bb-pc*cross)/determinant,v=(pc*aa-pa*cross)/determinant;
                if (u>=0 && v>=0 && u+v<=1)
                { const Point delta{ap[0]-u*ab[0]-v*ac[0],ap[1]-u*ab[1]-v*ac[1],ap[2]-u*ab[2]-v*ac[2]}; return Dot(delta,delta); }
            }
            return std::min({SegmentSquared(p,a,b),SegmentSquared(p,b,c),SegmentSquared(p,c,a)});
        }
        double BoxSquared(Point p,const Maths::AABB& box)
        {
            double result=0;
            for (unsigned axis=0;axis<3;++axis)
            { const double delta=std::max({double(box.Min[axis])-p[axis],0.0,p[axis]-box.Max[axis]}); result+=delta*delta; }
            return result;
        }
        double NearestSquared(const Maths::TriMesh& mesh,Vec point,double bandSquared,std::uint64_t& tests)
        {
            const auto& tree=mesh.GetAcceleration(); const auto& nodes=tree.GetNodes(); const auto& indices=tree.GetPrimitiveIndices();
            struct Visit { unsigned Node; double Distance; };
            std::vector<Visit> stack; stack.reserve(64); const auto p=Convert(point);
            stack.push_back({0,BoxSquared(p,nodes[0].Bounds)}); double best=bandSquared;
            while (!stack.empty())
            {
                const auto visit=stack.back(); stack.pop_back(); if (visit.Distance>=best) continue;
                const auto& node=nodes[visit.Node];
                if (node.IsLeaf())
                {
                    for (unsigned i=0;i<node.PrimitiveCount;++i)
                    {
                        Vec a,b,c; mesh.GetTriangle(indices[node.FirstPrimitive+i],a,b,c); ++tests;
                        best=std::min(best,TriangleSquared(p,a,b,c));
                    }
                }
                else
                {
                    Visit first{node.Left,BoxSquared(p,nodes[node.Left].Bounds)},second{node.Right,BoxSquared(p,nodes[node.Right].Bounds)};
                    if (first.Distance>second.Distance) std::swap(first,second);
                    if (second.Distance<best) stack.push_back(second); if (first.Distance<best) stack.push_back(first);
                }
            }
            return best;
        }
        std::size_t Address(unsigned x,unsigned y,unsigned z,const std::array<unsigned,3>& dimensions)
        { return (std::size_t(z)*dimensions[1]+y)*dimensions[0]+x; }
        bool Interrupted(const LumenDistanceCancellation& cancelled,std::string& error)
        { if (cancelled && cancelled()) { error="Lumen distance-field build cancelled"; return true; } return false; }
        bool LayoutValid(const LumenDistanceField& field)
        {
            if (!Finite(field.Center) || !Finite(field.Minimum) || !std::isfinite(field.MeshScale) || field.MeshScale<=0 ||
                !std::isfinite(field.CellSize) || field.CellSize<1.0f/MaximumResolution || field.CellSize>1.0f/MinimumResolution ||
                !std::isfinite(field.MaximumDistance) || field.MaximumDistance<field.CellSize || field.MaximumDistance>field.CellSize*8 ||
                !std::isfinite(field.MeshScale*field.MaximumDistance) || field.MeshScale*field.CellSize<=0 ||
                (field.Sign!=LumenDistanceSign::BackfaceVote && field.Sign!=LumenDistanceSign::TwoSided)) return false;
            std::size_t samples=1,bricks=1;
            for (unsigned axis=0;axis<3;++axis)
            {
                if (!field.Dimensions[axis] || field.Dimensions[axis]>LumenDistanceMaximumAxis ||
                    field.BrickDimensions[axis]!=(field.Dimensions[axis]+LumenDistanceBrickInterior-1)/LumenDistanceBrickInterior) return false;
                samples*=field.Dimensions[axis]; bricks*=field.BrickDimensions[axis];
            }
            return samples<=LumenDistanceMaximumVoxels && field.Directory.size()==bricks &&
                field.Bricks.size()%LumenDistanceBrickSamples==0 && field.Bricks.size()/LumenDistanceBrickSamples<=bricks;
        }
    }
    bool BuildLumenDistanceField(const SceneRayQuery& source,const LumenDistanceBuildSettings& settings,
        LumenDistanceField& output,std::string& error,const LumenDistanceCancellation& cancelled)
    {
        if (settings.Resolution<MinimumResolution || settings.Resolution>MaximumResolution || settings.SignRays<2 || settings.SignRays>MaximumSignRays || settings.SignRays%2 ||
            !std::isfinite(settings.BandCells) || settings.BandCells<1 || settings.BandCells>8 ||
            (settings.Sign!=LumenDistanceSign::BackfaceVote && settings.Sign!=LumenDistanceSign::TwoSided) ||
            !source.GetTriangleCount() || source.GetTriangleCount()>LumenDistanceMaximumTriangles || !source.GetBounds().IsValid())
        { error="Lumen distance fields require bounded nonempty geometry, resolution, band and paired sign rays"; return false; }
        if (Interrupted(cancelled,error)) return false;
        LumenDistanceField result; result.Sign=settings.Sign; const auto bounds=source.GetBounds();
        const double center[]{(double(bounds.Min.x)+bounds.Max.x)/2,(double(bounds.Min.y)+bounds.Max.y)/2,(double(bounds.Min.z)+bounds.Max.z)/2};
        const double scale=std::max({double(bounds.Max.x)-bounds.Min.x,double(bounds.Max.y)-bounds.Min.y,double(bounds.Max.z)-bounds.Min.z});
        result.Center={float(center[0]),float(center[1]),float(center[2])}; result.MeshScale=float(scale);
        if (!Finite(result.Center) || !std::isfinite(result.MeshScale) || result.MeshScale<=0)
        { error="Lumen distance geometry has a degenerate or unrepresentable extent"; return false; }
        std::vector<Vec> vertices; vertices.reserve(source.GetTriangleCount()*3); std::vector<std::uint32_t> indices(source.GetTriangleCount()*3);
        std::iota(indices.begin(),indices.end(),0u);
        for (unsigned index=0;index<source.GetTriangleCount();++index)
        {
            if (index%1024==0 && Interrupted(cancelled,error)) return false;
            SceneRayTriangle triangle; if (!source.GetTriangle(index,triangle)) { error="Lumen distance source triangle disappeared"; return false; }
            for (const auto& vertex:triangle.Corners)
                vertices.push_back({float((double(vertex.Position.X)-result.Center.X)/result.MeshScale),float((double(vertex.Position.Y)-result.Center.Y)/result.MeshScale),float((double(vertex.Position.Z)-result.Center.Z)/result.MeshScale)});
        }
        Maths::TriMesh mesh;
        if (!mesh.SetGeometry(std::move(vertices),std::move(indices),&error) || !mesh.BuildAcceleration()) return false;
        result.CellSize=1.0f/settings.Resolution; result.MaximumDistance=settings.BandCells*result.CellSize;
        const unsigned padding=unsigned(std::ceil(settings.BandCells))+1; const auto unitBounds=mesh.GetBounds();
        const auto unitCenter=unitBounds.Center(); const auto size=unitBounds.Max-unitBounds.Min; float minimum[3]{};
        std::size_t count=1;
        for (unsigned axis=0;axis<3;++axis)
        {
            result.Dimensions[axis]=std::max(1u,unsigned(std::ceil(double(size[axis])/result.CellSize)))+padding*2;
            if (result.Dimensions[axis]>LumenDistanceMaximumAxis) { error="Lumen distance grid exceeds its axis budget"; return false; }
            result.BrickDimensions[axis]=(result.Dimensions[axis]+LumenDistanceBrickInterior-1)/LumenDistanceBrickInterior;
            minimum[axis]=unitCenter[axis]-result.Dimensions[axis]*result.CellSize*0.5f; count*=result.Dimensions[axis];
        }
        if (count>LumenDistanceMaximumVoxels) { error="Lumen distance grid exceeds its voxel budget"; return false; }
        result.Minimum={minimum[0],minimum[1],minimum[2]}; std::vector<std::int16_t> dense(count);
        std::vector<Vec> directions; const unsigned half=settings.SignRays/2; const double goldenAngle=Pi*(3-std::sqrt(5.0));
        for (unsigned i=0;i<half;++i)
        {
            const double z=(i+0.5)/half,radius=std::sqrt(1-z*z),angle=goldenAngle*i;
            const Vec direction{float(radius*std::cos(angle)),float(radius*std::sin(angle)),float(z)}; directions.push_back(direction); directions.push_back(-direction);
        }
        Maths::TriMeshHitOptions options; options.MinimumDistance=0; options.MaximumDistance=std::numeric_limits<float>::infinity();
        for (unsigned z=0;z<result.Dimensions[2];++z) for (unsigned y=0;y<result.Dimensions[1];++y)
        {
            if (Interrupted(cancelled,error)) return false;
            for (unsigned x=0;x<result.Dimensions[0];++x)
            {
                const Vec p{minimum[0]+(x+0.5f)*result.CellSize,minimum[1]+(y+0.5f)*result.CellSize,minimum[2]+(z+0.5f)*result.CellSize};
                double distance=std::sqrt(NearestSquared(mesh,p,double(result.MaximumDistance)*result.MaximumDistance,result.DistanceTriangleTests));
                if (settings.Sign==LumenDistanceSign::BackfaceVote && unitBounds.Contains(p))
                {
                    unsigned backfaces=0;
                    for (const auto& direction:directions)
                    { Maths::TriMeshHit hit; ++result.SignQueries; if (mesh.RayCast({p,direction},hit,options) && !hit.FrontFace) ++backfaces; }
                    // Misses remain part of the denominator. Opposite direction
                    // pairs keep an isolated sheet from becoming a solid halfspace.
                    if (backfaces*2>directions.size()) { distance=-distance; ++result.InsideVoxels; }
                    if (backfaces && backfaces<directions.size()) ++result.AmbiguousVoxels;
                }
                dense[Address(x,y,z,result.Dimensions)]=std::int16_t(std::clamp(std::lround(distance/result.MaximumDistance*LumenDistanceQuantization),
                    -long(LumenDistanceQuantization),long(LumenDistanceQuantization)));
            }
        }
        result.Directory.reserve(std::size_t(result.BrickDimensions[0])*result.BrickDimensions[1]*result.BrickDimensions[2]);
        for (unsigned z=0;z<result.BrickDimensions[2];++z) for (unsigned y=0;y<result.BrickDimensions[1];++y) for (unsigned x=0;x<result.BrickDimensions[0];++x)
        {
            if (Interrupted(cancelled,error)) return false;
            std::array<std::int16_t,LumenDistanceBrickSamples> brick; bool outside=true,inside=true; unsigned destination=0;
            for (unsigned bz=0;bz<LumenDistanceBrickSide;++bz) for (unsigned by=0;by<LumenDistanceBrickSide;++by) for (unsigned bx=0;bx<LumenDistanceBrickSide;++bx)
            {
                const unsigned coordinate[]{x*LumenDistanceBrickInterior+bx,y*LumenDistanceBrickInterior+by,z*LumenDistanceBrickInterior+bz}; unsigned sample[3]{};
                for (unsigned axis=0;axis<3;++axis) sample[axis]=unsigned(std::clamp(int(coordinate[axis])-int(LumenDistanceBrickBorder),0,int(result.Dimensions[axis])-1));
                const auto value=dense[Address(sample[0],sample[1],sample[2],result.Dimensions)]; brick[destination++]=value;
                outside&=value==LumenDistanceQuantization; inside&=value==-LumenDistanceQuantization;
            }
            if (outside || inside) result.Directory.push_back(outside?LumenDistanceOutside:LumenDistanceInside);
            else
            {
                result.Directory.push_back(unsigned(result.Bricks.size()/LumenDistanceBrickSamples)+LumenDistanceFirstBrick);
                result.Bricks.insert(result.Bricks.end(),brick.begin(),brick.end());
            }
        }
        if (!ValidateLumenDistanceField(result,error)) return false; output=std::move(result); error.clear(); return true;
    }
    bool ValidateLumenDistanceField(const LumenDistanceField& field,std::string& error)
    {
        if (!LayoutValid(field)) { error="Lumen distance field has invalid bounds, dimensions or brick storage"; return false; }
        for (auto entry:field.Directory) if (entry>=LumenDistanceFirstBrick && entry-LumenDistanceFirstBrick>=field.Bricks.size()/LumenDistanceBrickSamples)
        { error="Lumen distance directory addresses a missing brick"; return false; }
        for (auto sample:field.Bricks) if (sample==std::numeric_limits<std::int16_t>::min())
        { error="Lumen distance brick contains an invalid signed encoding"; return false; }
        error.clear(); return true;
    }
    bool SampleLumenDistanceField(const LumenDistanceField& field,Float3 position,LumenDistanceSample& output) noexcept
    {
        if (!Finite(position) || !LayoutValid(field)) return false;
        const double point[]{(double(position.X)-field.Center.X)/field.MeshScale,(double(position.Y)-field.Center.Y)/field.MeshScale,(double(position.Z)-field.Center.Z)/field.MeshScale};
        const double minimum[]{field.Minimum.X,field.Minimum.Y,field.Minimum.Z}; unsigned cell[3]{},local[3]{}; double weight[3]{};
        for (unsigned axis=0;axis<3;++axis)
        {
            const double grid=(point[axis]-minimum[axis])/field.CellSize;
            if (grid<0 || grid>field.Dimensions[axis]) return false;
            const double coordinate=std::clamp(grid-0.5,0.0,double(field.Dimensions[axis]-1)); cell[axis]=unsigned(std::floor(coordinate)); weight[axis]=coordinate-cell[axis];
            local[axis]=cell[axis]%LumenDistanceBrickInterior+LumenDistanceBrickBorder;
        }
        const auto entry=field.Directory[Address(cell[0]/LumenDistanceBrickInterior,cell[1]/LumenDistanceBrickInterior,cell[2]/LumenDistanceBrickInterior,field.BrickDimensions)];
        double value=entry==LumenDistanceInside?-1.0:1.0;
        if (entry>=LumenDistanceFirstBrick)
        {
            const auto offset=std::size_t(entry-LumenDistanceFirstBrick)*LumenDistanceBrickSamples;
            if (offset+LumenDistanceBrickSamples>field.Bricks.size()) return false; value=0;
            for (unsigned z=0;z<2;++z) for (unsigned y=0;y<2;++y) for (unsigned x=0;x<2;++x)
            {
                const auto encoded=field.Bricks[offset+((local[2]+z)*LumenDistanceBrickSide+local[1]+y)*LumenDistanceBrickSide+local[0]+x];
                if (encoded==std::numeric_limits<std::int16_t>::min()) return false;
                const double contribution=(x?weight[0]:1-weight[0])*(y?weight[1]:1-weight[1])*(z?weight[2]:1-weight[2]);
                value+=double(encoded)/LumenDistanceQuantization*contribution;
            }
        }
        LumenDistanceSample result; result.Distance=float(value*field.MaximumDistance*field.MeshScale);
        // Unsigned distance is 1-Lipschitz. Trilinear corner weights give a
        // maximum RMS point-to-corner distance of sqrt(3)/2 cells. Quantization
        // adds half an encoding step; truncation and sign cancellation only
        // reduce magnitude. Sign itself is not guaranteed for open geometry.
        constexpr double NormalizationRoundoff=4*std::numeric_limits<float>::epsilon();
        result.ErrorBound=float((std::sqrt(3.0)*field.CellSize*0.5+double(field.MaximumDistance)/(LumenDistanceQuantization*2)+NormalizationRoundoff)*field.MeshScale);
        result.ConservativeDistance=std::max(0.0f,std::abs(result.Distance)-result.ErrorBound);
        if (!std::isfinite(result.Distance) || !std::isfinite(result.ErrorBound)) return false; output=result; return true;
    }
}
