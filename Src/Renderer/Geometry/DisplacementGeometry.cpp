#include "Renderer/Geometry/DisplacementGeometry.h"
#include "Renderer/Geometry/HeightField.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        struct Vector
        {
            double X=0,Y=0,Z=0;
            Vector()=default;
            Vector(double x,double y,double z):X(x),Y(y),Z(z) {}
            explicit Vector(Float3 value):X(value.X),Y(value.Y),Z(value.Z) {}
            Vector operator+(Vector b) const { return {X+b.X,Y+b.Y,Z+b.Z}; }
            Vector operator-(Vector b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
            Vector operator*(double scale) const { return {X*scale,Y*scale,Z*scale}; }
            double Dot(Vector b) const { return X*b.X+Y*b.Y+Z*b.Z; }
            Vector Cross(Vector b) const { return {Y*b.Z-Z*b.Y,Z*b.X-X*b.Z,X*b.Y-Y*b.X}; }
            double Length() const { return std::hypot(X,Y,Z); }
            Vector Unit() const { const double length=Length(); return length>0?*this*(1/length):Vector{}; }
            Float3 Native() const { return {float(X),float(Y),float(Z)}; }
        };
        using Key=std::array<std::uint32_t,6>;
        struct KeyHash
        {
            std::size_t operator()(const Key& key) const noexcept
            {
                // FNV-1a over numeric IEEE words. Equality resolves collisions.
                std::uint64_t hash=14695981039346656037ull;
                for (auto word:key) { hash^=word; hash*=1099511628211ull; }
                return static_cast<std::size_t>(hash);
            }
        };
        Key Identity(const Vertex& vertex,bool shading)
        {
            Key key{};
            const float values[]{vertex.Position.X,vertex.Position.Y,vertex.Position.Z,vertex.Normal.X,vertex.Normal.Y,vertex.Normal.Z};
            for (unsigned i=0;i<(shading?6u:3u);++i)
                if (values[i]!=0) std::memcpy(&key[i],&values[i],sizeof(float)); // signed zeros share a position
            return key;
        }
        Vector Face(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,std::size_t triangle,double scale)
        {
            const Vector a(vertices[indices[triangle*3]].Position),b(vertices[indices[triangle*3+1]].Position),c(vertices[indices[triangle*3+2]].Position);
            return ((b-a)*scale).Cross((c-a)*scale);
        }
        bool Finite(Float3 p) { return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z); }
    }
    bool PrepareDisplacementGeometry(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const GeometryPreparationLimits& limits,DisplacementGeometry& output,std::string& error)
    {
        if (!ValidateTriangleGeometry(vertices,indices,error)) return false;
        if (vertices.size()>limits.MaximumVertices || indices.size()>limits.MaximumIndices || indices.size()>std::numeric_limits<std::uint32_t>::max() || vertices.size()>std::numeric_limits<std::uint32_t>::max()/sizeof(Vertex))
        { error="Displacement input exceeds the geometry budget or native adjacency range"; return false; }
        DisplacementGeometry candidate; candidate.BoundsMinimum=candidate.BoundsMaximum=vertices.front().Position;
        for (const auto& vertex:vertices)
        {
            auto& lo=candidate.BoundsMinimum; auto& hi=candidate.BoundsMaximum; const auto p=vertex.Position;
            lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)};
            hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)};
        }
        const double extent=std::max({double(candidate.BoundsMaximum.X)-candidate.BoundsMinimum.X,
            double(candidate.BoundsMaximum.Y)-candidate.BoundsMinimum.Y,double(candidate.BoundsMaximum.Z)-candidate.BoundsMinimum.Z});
        if (!(extent>0) || !std::isnormal(float(1/extent)))
        { error="Displacement needs a nonzero local extent representable by its height-coordinate scale"; return false; }
        std::unordered_map<Key,std::uint32_t,KeyHash> positionGroups,normalGroups;
        std::vector<std::uint32_t> positionOf(vertices.size()),normalOf(vertices.size());
        for (std::size_t vertex=0;vertex<vertices.size();++vertex)
        {
            positionOf[vertex]=positionGroups.emplace(Identity(vertices[vertex],false),static_cast<std::uint32_t>(positionGroups.size())).first->second;
            normalOf[vertex]=normalGroups.emplace(Identity(vertices[vertex],true),static_cast<std::uint32_t>(normalGroups.size())).first->second;
        }
        std::vector<Vector> direction(positionGroups.size()),authored(positionGroups.size()),largest(positionGroups.size());
        std::vector<std::vector<std::uint32_t>> adjacency(normalGroups.size());
        for (std::size_t vertex=0;vertex<vertices.size();++vertex)
            authored[positionOf[vertex]]=authored[positionOf[vertex]]+Vector(vertices[vertex].Normal).Unit();
        for (std::size_t triangle=0;triangle<indices.size()/3;++triangle)
        {
            const auto a=indices[triangle*3],b=indices[triangle*3+1],c=indices[triangle*3+2];
            auto face=Face(vertices,indices,triangle,1/extent);
            const auto expected=Vector(vertices[a].Normal)+Vector(vertices[b].Normal)+Vector(vertices[c].Normal);
            if (face.Dot(expected)<0) face=face*(-1);
            const std::uint32_t positions[]{positionOf[a],positionOf[b],positionOf[c]},normals[]{normalOf[a],normalOf[b],normalOf[c]};
            for (unsigned corner=0;corner<3;++corner)
            {
                const auto p=positions[corner],n=normals[corner];
                if ((corner==0 || p!=positions[0]) && (corner<2 || p!=positions[1]))
                {
                    direction[p]=direction[p]+face;
                    if (face.Length()>largest[p].Length()) largest[p]=face;
                }
                if ((corner==0 || n!=normals[0]) && (corner<2 || n!=normals[1])) adjacency[n].push_back(static_cast<std::uint32_t>(triangle));
            }
        }
        std::vector<ShaderUInt4> ranges(adjacency.size());
        for (std::size_t group=0;group<adjacency.size();++group)
        {
            ranges[group]={static_cast<unsigned>(candidate.AdjacentTriangles.size()),static_cast<unsigned>(adjacency[group].size()),0,0};
            candidate.AdjacentTriangles.insert(candidate.AdjacentTriangles.end(),adjacency[group].begin(),adjacency[group].end());
        }
        candidate.Directions.reserve(vertices.size()); candidate.Neighborhoods.reserve(vertices.size());
        for (std::size_t vertex=0;vertex<vertices.size();++vertex)
        {
            const auto group=positionOf[vertex];
            auto normal=direction[group].Unit();
            if (normal.Length()==0) normal=authored[group].Unit();
            if (normal.Length()==0) normal=largest[group].Unit();
            candidate.Directions.push_back({float(normal.X),float(normal.Y),float(normal.Z),0});
            candidate.Neighborhoods.push_back(ranges[normalOf[vertex]]);
        }
        candidate.Vertices=vertices; candidate.Indices=indices; output=std::move(candidate); return true;
    }
    bool BuildDisplacementConstants(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& requested,DisplacementConstants& output) noexcept
    {
        const auto lo=geometry.BoundsMinimum,hi=geometry.BoundsMaximum;
        if (!Finite(lo) || !Finite(hi) || lo.X>hi.X || lo.Y>hi.Y || lo.Z>hi.Z || geometry.Vertices.empty() ||
            geometry.Vertices.size()>std::numeric_limits<std::uint32_t>::max()/sizeof(Vertex) || geometry.Indices.empty() || geometry.Indices.size()%3 ||
            geometry.Indices.size()>std::numeric_limits<std::uint32_t>::max() || geometry.Directions.size()!=geometry.Vertices.size() || geometry.Neighborhoods.size()!=geometry.Vertices.size()) return false;
        const double x=double(hi.X)-lo.X,y=double(hi.Y)-lo.Y,z=double(hi.Z)-lo.Z,extent=std::max({x,y,z});
        if (!(extent>0) || !std::isnormal(float(1/extent)) || !(float(1/extent)>0)) return false;
        const auto settings=Techniques::SanitizeDisplacementMappingSettings(requested);
        auto projection=settings.Projection;
        if (projection==Techniques::DisplacementProjection::Automatic)
            projection=z<=std::min(x,y)?Techniques::DisplacementProjection::XY:y<=x?Techniques::DisplacementProjection::XZ:Techniques::DisplacementProjection::YZ;
        const double amplitude=Techniques::UsesDisplacementMapping(settings)?settings.HeightFraction*extent:0;
        const double excursion=amplitude*std::max(settings.Midlevel,1-settings.Midlevel);
        const double coordinateLimit=std::numeric_limits<float>::max();
        if (std::max({std::abs(double(lo.X)),std::abs(double(lo.Y)),std::abs(double(lo.Z)),std::abs(double(hi.X)),std::abs(double(hi.Y)),std::abs(double(hi.Z))})+excursion>coordinateLimit) return false;
        output={{lo.X,lo.Y,lo.Z,float(1/extent)},{float(amplitude),settings.Tiling,settings.Midlevel,0},
            {static_cast<unsigned>(geometry.Vertices.size()),static_cast<unsigned>(geometry.Indices.size()/3),static_cast<unsigned>(projection),0}};
        return true;
    }
    bool ValidateDisplacementUpload(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacementConstants& constants,std::string& error)
    {
        if (!BuildDisplacementConstants(geometry,settings,constants)) { error="Displacement upload has invalid bounds, counts or dimensions"; return false; }
        // The shared one-dimensional dispatch fits the guaranteed 65,535 group
        // limit. Per-device storage range checks still precede native allocation.
        constexpr std::size_t MaximumGroups=65535;
        if (geometry.Vertices.size()>MaximumGroups*DisplacementThreadCount)
        { error="Displacement upload exceeds the portable one-dimensional dispatch range"; return false; }
        for (const auto& range:geometry.Neighborhoods) if (range.X>geometry.AdjacentTriangles.size() || range.Y>geometry.AdjacentTriangles.size()-range.X)
        { error="Displacement upload contains an invalid adjacency range"; return false; }
        for (auto triangle:geometry.AdjacentTriangles) if (triangle>=geometry.Indices.size()/3) { error="Displacement upload contains an invalid adjacent triangle"; return false; }
        if (!ValidateTriangleGeometry(geometry.Vertices,geometry.Indices,error)) return false;
        for (std::size_t vertex=0;vertex<geometry.Vertices.size();++vertex)
        {
            const auto d=geometry.Directions[vertex]; const auto p=geometry.Vertices[vertex].Position;
            constexpr double DirectionRoundoff=32*std::numeric_limits<float>::epsilon();
            if (!std::isfinite(d.X) || !std::isfinite(d.Y) || !std::isfinite(d.Z) || std::hypot(double(d.X),double(d.Y),double(d.Z))>1+DirectionRoundoff)
            { error="Displacement upload contains a non-finite or non-unit-bounded direction"; return false; }
            if (p.X<geometry.BoundsMinimum.X || p.Y<geometry.BoundsMinimum.Y || p.Z<geometry.BoundsMinimum.Z ||
                p.X>geometry.BoundsMaximum.X || p.Y>geometry.BoundsMaximum.Y || p.Z>geometry.BoundsMaximum.Z)
            { error="Displacement source positions exceed their declared local bounds"; return false; }
        }
        return true;
    }
}
