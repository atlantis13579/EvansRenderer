#include "Renderer/Hair/HairStrands.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace Evans::Renderer
{
    namespace
    {
        using Maths::Vector3;
        constexpr double Pi=3.14159265358979323846;
        constexpr float DirectionTolerance=1e-12f,ParallelThreshold=0.9f;
        Vector3 Vector(Float3 value) { return {value.X,value.Y,value.Z}; }
        Float3 Native(Vector3 value) { return {value.x,value.y,value.z}; }
        bool Finite(Vector3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
        Vector3 Unit(Vector3 value,Vector3 fallback)
        { return value.LengthSquared()>DirectionTolerance ? value.Normalized():fallback; }
        double Area(const SceneRayTriangle& triangle)
        {
            double edge0[3],edge1[3];
            for (unsigned axis=0;axis<3;++axis)
            {
                const auto p0=Vector(triangle.Corners[0].Position),p1=Vector(triangle.Corners[1].Position),p2=Vector(triangle.Corners[2].Position);
                edge0[axis]=double(p1[axis])-p0[axis]; edge1[axis]=double(p2[axis])-p0[axis];
            }
            return std::hypot(edge0[1]*edge1[2]-edge0[2]*edge1[1],edge0[2]*edge1[0]-edge0[0]*edge1[2],edge0[0]*edge1[1]-edge0[1]*edge1[0])*0.5;
        }
    }
    bool BuildHairStrands(const SceneRayQuery& scene,const Techniques::HairFurSettings& request,
        HairStrandGeometry& output,std::string& error,const std::function<bool()>& cancelled)
    {
        error.clear(); const auto fail=[&](const char* text){error=text; return false;};
        if (cancelled && cancelled()) return fail("Hair strand construction cancelled");
        const auto settings=Techniques::SanitizeHairFurSettings(request); HairStrandGeometry result;
        if (!scene.GetTriangleCount() || !settings.StrandCount || settings.LengthFraction<=0 || settings.RadiusRatio<=0)
        { output=std::move(result); return true; }
        std::vector<double> cumulative; std::vector<std::uint32_t> triangles;
        cumulative.reserve(scene.GetTriangleCount()); triangles.reserve(scene.GetTriangleCount()); double totalArea=0;
        for (std::size_t index=0;index<scene.GetTriangleCount();++index)
        {
            if (cancelled && (index%256)==0 && cancelled()) return fail("Hair strand construction cancelled");
            SceneRayTriangle triangle; if (!scene.GetTriangle(static_cast<std::uint32_t>(index),triangle)) return fail("Hair source triangle lookup failed");
            const double area=Area(triangle); if (!(area>0)) continue;
            if (!std::isfinite(area) || !std::isfinite(totalArea+area)) return fail("Hair source area is unrepresentable");
            totalArea+=area; cumulative.push_back(totalArea); triangles.push_back(static_cast<std::uint32_t>(index));
        }
        if (triangles.empty()) { output=std::move(result); return true; }
        const auto extent=scene.GetBounds().Max-scene.GetBounds().Min;
        result.GrowthLength=std::max({extent.x,extent.y,extent.z})*settings.LengthFraction;
        result.RootRadius=result.GrowthLength*settings.RadiusRatio;
        if (!std::isfinite(result.GrowthLength) || !std::isfinite(result.RootRadius) || !(result.RootRadius>0)) return fail("Hair growth dimensions are unrepresentable");
        const auto verticesPerStrand=2*(settings.Segments+1);
        result.Vertices.reserve(std::size_t(settings.StrandCount)*verticesPerStrand);
        result.Indices.reserve(std::size_t(settings.StrandCount)*settings.Segments*6);
        std::mt19937 random(settings.Seed);
        // Explicit top-24-bit conversion avoids implementation-dependent
        // uniform_real_distribution mappings and retains seeded reproducibility.
        const auto unit=[&] { constexpr unsigned DiscardedBits=8; constexpr double UnitScale=1.0/16777216;
            return double(random()>>DiscardedBits)*UnitScale; };
        Vector3 minimum(std::numeric_limits<float>::max()),maximum(-std::numeric_limits<float>::max());
        for (std::uint32_t candidate=0;candidate<settings.StrandCount;++candidate)
        {
            if (cancelled && (candidate%256)==0 && cancelled()) return fail("Hair strand construction cancelled");
            ++result.CandidateRoots;
            const auto selected=std::upper_bound(cumulative.begin(),cumulative.end(),unit()*totalArea)-cumulative.begin();
            const auto triangle=triangles[std::min<std::size_t>(selected,triangles.size()-1)];
            const double root=std::sqrt(unit()),along=unit(); const Float3 barycentric{float(1-root),float(root*(1-along)),float(root*along)};
            const float azimuth=float(2*Pi*unit()),lengthScale=1+settings.LengthVariation*float(2*unit()-1);
            SceneRayHit hit;
            if (!scene.SampleSurface(triangle,barycentric,true,hit)) { ++result.RejectedRoots; continue; }
            SceneRayTriangle source; if (!scene.GetTriangle(triangle,source)) return fail("Hair source flags are unavailable");
            constexpr float NoShadowFlag=-1.0f;
            auto normal=Unit(Vector(hit.Normal),Vector(hit.GeometricNormal)); const auto outward=Vector(hit.GeometricNormal);
            if (normal.Dot(outward)<0) normal=-normal;
            const auto helper=std::abs(normal.y)<ParallelThreshold ? Vector3::UnitY():Vector3::UnitX();
            const auto tangent=Unit(helper.Cross(normal),Vector3::UnitX()),bitangent=normal.Cross(tangent);
            const auto bend=tangent*std::cos(azimuth)+bitangent*std::sin(azimuth);
            const float length=result.GrowthLength*lengthScale,radius=result.RootRadius*lengthScale;
            const auto origin=Vector(hit.Position)+normal*(result.GrowthLength*settings.RootOffset);
            const auto first=static_cast<std::uint32_t>(result.Vertices.size());
            for (std::uint32_t segment=0;segment<=settings.Segments;++segment)
            {
                const float height=float(segment)/settings.Segments;
                const auto center=origin+normal*(length*height)+bend*(length*settings.Bend*height*height);
                const auto direction=Unit(normal+bend*(2*settings.Bend*height),normal);
                const float width=radius*(1-height);
                if (!Finite(center) || !Finite(direction) || !Finite(center-Vector3(width)) || !Finite(center+Vector3(width)))
                    return fail("Hair curve or ribbon bounds are unrepresentable");
                minimum=Vector3::Min(minimum,center-Vector3(width)); maximum=Vector3::Max(maximum,center+Vector3(width));
                for (float side:{-1.0f,1.0f})
                {
                    HairVertex vertex; vertex.Position=Native(center); vertex.Tangent=Native(direction);
                    vertex.Reflectance={hit.Diffuse.X,hit.Diffuse.Y,hit.Diffuse.Z,1}; vertex.Emission={hit.Emission.X,hit.Emission.Y,hit.Emission.Z,1};
                    vertex.Ribbon={side*width,side,height,source.CastShadow?0.0f:NoShadowFlag}; result.Vertices.push_back(vertex);
                    result.VertexSources.push_back(source.MeshIndex);
                }
            }
            for (std::uint32_t segment=0;segment<settings.Segments;++segment)
            {
                const auto a=first+segment*2,b=a+2;
                result.Indices.insert(result.Indices.end(),{a,a+1,b,b,a+1,b+1});
            }
            ++result.StrandCount;
        }
        if (result.StrandCount) { result.BoundsMinimum=Native(minimum); result.BoundsMaximum=Native(maximum); }
        if (cancelled && cancelled()) return fail("Hair strand construction cancelled");
        output=std::move(result); return true;
    }
}
