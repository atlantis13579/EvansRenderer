#include "Renderer/Geometry/PNTessellation.h"
#include "Maths/ColorSpace.h"

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
            explicit Vector(Float3 p):X(p.X),Y(p.Y),Z(p.Z) {}
            Vector operator+(Vector p) const { return {X+p.X,Y+p.Y,Z+p.Z}; }
            Vector operator-(Vector p) const { return {X-p.X,Y-p.Y,Z-p.Z}; }
            Vector operator*(double a) const { return {X*a,Y*a,Z*a}; }
            double Dot(Vector p) const { return X*p.X+Y*p.Y+Z*p.Z; }
            Vector Cross(Vector p) const { return {Y*p.Z-Z*p.Y,Z*p.X-X*p.Z,X*p.Y-Y*p.X}; }
            double Length() const { return std::hypot(X,Y,Z); }
            Vector Unit(Vector fallback) const { const auto length=Length(); return length>0?*this*(1/length):fallback; }
            Vector Unit() const { return Unit(Vector(0,0,0)); }
            Float3 Native() const { return {float(X),float(Y),float(Z)}; }
        };
        using PointKey=std::array<std::uint32_t,3>;
        struct PointHash
        {
            std::size_t operator()(const PointKey& key) const noexcept
            {
                std::uint64_t hash=14695981039346656037ull;
                for (auto value:key) { hash^=value; hash*=1099511628211ull; } return static_cast<std::size_t>(hash);
            }
        };
        PointKey Identity(Float3 p)
        {
            PointKey key{}; const float values[]{p.X,p.Y,p.Z};
            for (unsigned axis=0;axis<3;++axis) if (values[axis]!=0) std::memcpy(&key[axis],&values[axis],sizeof(float));
            return key;
        }
        std::uint64_t EdgeKey(std::uint32_t a,std::uint32_t b) { return (std::uint64_t(std::min(a,b))<<32)|std::max(a,b); }
        Vector NormalProjection(Vector edge,Float3 normal)
        { const auto n=Vector(normal).Unit(); return n*n.Dot(edge); }
        bool Finite(Float3 p) { return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z); }
    }
    bool PreparePNTessellation(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const GeometryPreparationLimits& limits,PNTessellationGeometry& output,std::string& error)
    {
        if (!ValidateTriangleGeometry(vertices,indices,error)) return false;
        if (vertices.size()>limits.MaximumVertices || indices.size()>limits.MaximumIndices || indices.size()>std::numeric_limits<std::uint32_t>::max())
        { error="PN source geometry exceeds its preparation budget or native address range"; return false; }
        PNTessellationGeometry candidate; candidate.BoundsMinimum=candidate.BoundsMaximum=vertices.front().Position;
        std::unordered_map<PointKey,std::uint32_t,PointHash> positions; std::vector<std::uint32_t> groups;
        for (const auto& vertex:vertices)
        {
            const auto p=vertex.Position; auto& lo=candidate.BoundsMinimum; auto& hi=candidate.BoundsMaximum;
            lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)}; hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)};
            groups.push_back(positions.emplace(Identity(p),static_cast<std::uint32_t>(positions.size())).first->second);
        }
        std::unordered_map<std::uint64_t,std::uint32_t> attributes,geometric;
        for (std::size_t face=0;face<indices.size();face+=3)
        {
            const auto a=indices[face],b=indices[face+1],c=indices[face+2];
            const auto area=(Vector(vertices[b].Position)-Vector(vertices[a].Position)).Cross(Vector(vertices[c].Position)-Vector(vertices[a].Position));
            candidate.Patches.push_back({a,b,c,area.Length()==0?PNDegeneratePatch:0}); ShaderUInt4 edges{}; std::uint32_t* slots[]{&edges.X,&edges.Y,&edges.Z};
            for (unsigned corner=0;corner<3;++corner)
            {
                const auto first=indices[face+corner],second=indices[face+(corner+1)%3];
                const auto low=groups[first]<=groups[second]?first:second,high=groups[first]<=groups[second]?second:first;
                const auto inserted=geometric.emplace(EdgeKey(groups[low],groups[high]),static_cast<std::uint32_t>(candidate.GeometricEdges.size()));
                if (inserted.second) candidate.GeometricEdges.push_back({low,high,groups[low]==groups[high]?PNLinearEdge:0,0});
                else
                {
                    auto& representative=candidate.GeometricEdges[inserted.first->second];
                    const auto edge=Vector(vertices[high].Position)-Vector(vertices[low].Position);
                    constexpr double ProjectionTolerance=0.00001;
                    if ((NormalProjection(edge,vertices[low].Normal)-NormalProjection(edge,vertices[representative.X].Normal)).Length()>ProjectionTolerance*edge.Length() ||
                        (NormalProjection(edge,vertices[high].Normal)-NormalProjection(edge,vertices[representative.Y].Normal)).Length()>ProjectionTolerance*edge.Length()) representative.Z|=PNLinearEdge;
                }
                const auto attribute=attributes.emplace(EdgeKey(first,second),static_cast<std::uint32_t>(candidate.AttributeEdges.size()));
                if (attribute.second)
                {
                    const auto sourceLow=std::min(first,second),sourceHigh=std::max(first,second);
                    candidate.AttributeEdges.push_back({sourceLow,sourceHigh,inserted.first->second,groups[sourceLow]<=groups[sourceHigh]?0:PNReverseEdge});
                }
                *slots[corner]=attribute.first->second;
            }
            candidate.PatchEdges.push_back(edges);
        }
        candidate.Vertices=vertices; TessellationConstants constants;
        if (!BuildTessellationConstants(candidate,{},limits,constants)) { error="PN geometry has unrepresentable bounds or output dimensions"; return false; }
        output=std::move(candidate); return true;
    }
    bool BuildTessellationConstants(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& requested,
        const GeometryPreparationLimits& limits,TessellationConstants& output) noexcept
    {
        const auto lo=geometry.BoundsMinimum,hi=geometry.BoundsMaximum;
        const auto sourceMaximum=std::numeric_limits<std::uint32_t>::max();
        if (geometry.Vertices.size()>sourceMaximum || geometry.Patches.size()>sourceMaximum || geometry.AttributeEdges.size()>sourceMaximum || geometry.GeometricEdges.size()>sourceMaximum) return false;
        if (!Finite(lo) || !Finite(hi) || lo.X>hi.X || lo.Y>hi.Y || lo.Z>hi.Z || geometry.Vertices.empty() || geometry.Patches.empty() ||
            geometry.PatchEdges.size()!=geometry.Patches.size() || geometry.AttributeEdges.empty() || geometry.GeometricEdges.empty()) return false;
        const double extent=std::max({double(hi.X)-lo.X,double(hi.Y)-lo.Y,double(hi.Z)-lo.Z});
        if (!(extent>0) || !std::isnormal(float(1/extent))) return false;
        const auto settings=Techniques::SanitizeTessellationSettings(requested);
        const std::uint64_t n=Techniques::UsesTessellation(settings)?settings.Segments:1;
        const std::uint64_t interior=n>1?(n-1)*(n-2)/2:0;
        const std::uint64_t vertices=geometry.Vertices.size()+std::uint64_t(geometry.AttributeEdges.size())*(n-1)+geometry.Patches.size()*interior;
        const std::uint64_t indices=std::uint64_t(geometry.Patches.size())*n*n*3;
        constexpr std::uint64_t MaximumGroups=65535;
        const auto nativeMaximum=std::numeric_limits<std::uint32_t>::max();
        if (vertices>limits.MaximumVertices || indices>limits.MaximumIndices || vertices>nativeMaximum/sizeof(Vertex) || indices>nativeMaximum/sizeof(std::uint32_t) ||
            vertices>MaximumGroups*TessellationThreadCount || indices/3>MaximumGroups*TessellationThreadCount || geometry.GeometricEdges.size()>MaximumGroups*TessellationThreadCount) return false;
        output={{static_cast<unsigned>(geometry.Vertices.size()),static_cast<unsigned>(geometry.Patches.size()),static_cast<unsigned>(geometry.AttributeEdges.size()),static_cast<unsigned>(geometry.GeometricEdges.size())},
            {static_cast<unsigned>(vertices),static_cast<unsigned>(indices),static_cast<unsigned>(n),static_cast<unsigned>(interior)},
            {Techniques::UsesTessellation(settings)?settings.Curvature:0,0,0,0},{lo.X,lo.Y,lo.Z,float(1/extent)}};
        return true;
    }
    bool ValidatePNTessellationGeometry(const PNTessellationGeometry& geometry,std::string& error)
    {
        TessellationConstants constants;
        GeometryPreparationLimits nativeLimits{std::numeric_limits<std::uint32_t>::max(),std::numeric_limits<std::uint32_t>::max()};
        if (!BuildTessellationConstants(geometry,{},nativeLimits,constants)) { error="PN source dimensions or bounds are invalid"; return false; }
        const auto sourceCount=geometry.Vertices.size();
        std::vector<std::uint32_t> originalIndices; originalIndices.reserve(geometry.Patches.size()*3);
        for (const auto& patch:geometry.Patches) originalIndices.insert(originalIndices.end(),{patch.X,patch.Y,patch.Z});
        if (!ValidateTriangleGeometry(geometry.Vertices,originalIndices,error)) return false;
        for (const auto& vertex:geometry.Vertices)
        {
            const auto p=vertex.Position,lo=geometry.BoundsMinimum,hi=geometry.BoundsMaximum;
            if (p.X<lo.X || p.Y<lo.Y || p.Z<lo.Z || p.X>hi.X || p.Y>hi.Y || p.Z>hi.Z) { error="PN source position is outside its declared bounds"; return false; }
        }
        for (const auto& edge:geometry.GeometricEdges)
            if (edge.X>=sourceCount || edge.Y>=sourceCount || (edge.Z&~PNLinearEdge) || edge.W) { error="PN geometric edge has an invalid source vertex or flag"; return false; }
        for (const auto& edge:geometry.AttributeEdges)
        {
            if (edge.X>=sourceCount || edge.Y>=sourceCount || edge.X>edge.Y || edge.Z>=geometry.GeometricEdges.size() || (edge.W&~PNReverseEdge)) { error="PN attribute edge is outside the upload"; return false; }
            const auto representative=geometry.GeometricEdges[edge.Z]; const bool reverse=(edge.W&PNReverseEdge)!=0;
            if (Identity(geometry.Vertices[edge.X].Position)!=Identity(geometry.Vertices[reverse?representative.Y:representative.X].Position) ||
                Identity(geometry.Vertices[edge.Y].Position)!=Identity(geometry.Vertices[reverse?representative.X:representative.Y].Position)) { error="PN attribute edge does not match its canonical geometric edge"; return false; }
        }
        for (std::size_t face=0;face<geometry.Patches.size();++face)
        {
            const auto p=geometry.Patches[face],e=geometry.PatchEdges[face];
            if (e.X>=geometry.AttributeEdges.size() || e.Y>=geometry.AttributeEdges.size() || e.Z>=geometry.AttributeEdges.size() || e.W || (p.W&~PNDegeneratePatch)) { error="PN patch edge or flag is outside the upload"; return false; }
            const unsigned corners[]{p.X,p.Y,p.Z},edges[]{e.X,e.Y,e.Z};
            for (unsigned side=0;side<3;++side)
            {
                const auto edge=geometry.AttributeEdges[edges[side]];
                if (EdgeKey(edge.X,edge.Y)!=EdgeKey(corners[side],corners[(side+1)%3])) { error="PN patch boundary does not match its source corners"; return false; }
            }
            const auto area=(Vector(geometry.Vertices[p.Y].Position)-Vector(geometry.Vertices[p.X].Position)).Cross(Vector(geometry.Vertices[p.Z].Position)-Vector(geometry.Vertices[p.X].Position));
            if ((area.Length()==0)!=((p.W&PNDegeneratePatch)!=0)) { error="PN degenerate-patch flag does not match its source area"; return false; }
        }
        error.clear(); return true;
    }
    bool ValidateTessellationUpload(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,
        TessellationConstants& constants,std::string& error)
    {
        if (!ValidatePNTessellationGeometry(geometry,error)) return false;
        if (!BuildTessellationConstants(geometry,settings,{},constants)) { error="PN output exceeds geometry budgets or native dispatch/address limits"; return false; }
        error.clear(); return true;
    }
}
