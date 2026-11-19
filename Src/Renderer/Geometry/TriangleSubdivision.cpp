#include "Renderer/Geometry/TriangleSubdivision.h"
#include "Maths/ColorSpace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        std::uint64_t Edge(std::uint32_t a,std::uint32_t b)
        { return (std::uint64_t(std::min(a,b))<<32)|std::max(a,b); }
        Float3 Unit(Float3 normal,Float3 fallback)
        {
            const double length=std::hypot(double(normal.X),double(normal.Y),double(normal.Z));
            if (length==0) return fallback;
            return {float(normal.X/length),float(normal.Y/length),float(normal.Z/length)};
        }
        Vertex Interpolate(const Vertex& a,const Vertex& b,const Vertex& c,unsigned wa,unsigned wb,unsigned wc,unsigned denominator)
        {
            // Integer numerators make oppositely oriented duplicate seam edges
            // evaluate the same convex combination, including non-power-of-two n.
            const auto mix=[&](float x,float y,float z){return float((double(wa)*x+double(wb)*y+double(wc)*z)/denominator);};
            Vertex result;
            result.Position={mix(a.Position.X,b.Position.X,c.Position.X),mix(a.Position.Y,b.Position.Y,c.Position.Y),mix(a.Position.Z,b.Position.Z,c.Position.Z)};
            result.Normal={mix(a.Normal.X,b.Normal.X,c.Normal.X),mix(a.Normal.Y,b.Normal.Y,c.Normal.Y),mix(a.Normal.Z,b.Normal.Z,c.Normal.Z)};
            const auto& dominant=wa>=wb && wa>=wc?a:wb>=wc?b:c;
            result.Normal=Unit(result.Normal,Unit(dominant.Normal,{0,0,1}));
            // Native vertex stages decode authored display-sRGB before raster
            // interpolation. Refined vertices must represent that same linear
            // color field, encoded again for the unchanged vertex format.
            const auto color=[&](float x,float y,float z)
            { return x==y && y==z?x:Maths::LinearToSRGB(mix(Maths::SRGBToLinear(x),Maths::SRGBToLinear(y),Maths::SRGBToLinear(z))); };
            result.Color={color(a.Color.X,b.Color.X,c.Color.X),color(a.Color.Y,b.Color.Y,c.Color.Y),color(a.Color.Z,b.Color.Z,c.Color.Z),mix(a.Color.W,b.Color.W,c.Color.W)};
            result.TexCoord={mix(a.TexCoord.X,b.TexCoord.X,c.TexCoord.X),mix(a.TexCoord.Y,b.TexCoord.Y,c.TexCoord.Y)};
            return result;
        }
    }
    bool SubdivideTriangles(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,unsigned segments,
        const GeometryPreparationLimits& limits,SubdividedTriangles& output,std::string& error)
    {
        if (!segments || segments>MaximumTriangleSegments)
        { error="Triangle subdivision segment count is outside the supported integer range"; return false; }
        if (!ValidateTriangleGeometry(vertices,indices,error)) return false;
        const std::uint64_t triangles=indices.size()/3;
        const std::uint64_t growth=std::uint64_t(segments)*segments;
        if (indices.size()>std::min<std::uint64_t>(limits.MaximumIndices,std::numeric_limits<std::uint32_t>::max())/growth || vertices.size()>limits.MaximumVertices)
        { error="Triangle subdivision exceeds the requested geometry budget"; return false; }
        const std::uint64_t indexCount=indices.size()*growth;
        SubdividedTriangles candidate;
        if (segments==1)
        { candidate.Vertices=vertices; candidate.Indices=indices; output=std::move(candidate); return true; }
        std::unordered_map<std::uint64_t,std::uint32_t> edges;
        for (std::size_t triangle=0;triangle<indices.size();triangle+=3)
            for (unsigned corner=0;corner<3;++corner)
                edges.emplace(Edge(indices[triangle+corner],indices[triangle+(corner+1)%3]),0);
        const std::uint64_t interiorPerTriangle=std::uint64_t(segments-1)*(segments-2)/2;
        const std::uint64_t vertexCount=vertices.size()+std::uint64_t(edges.size())*(segments-1)+triangles*interiorPerTriangle;
        if (vertexCount>limits.MaximumVertices || vertexCount>std::numeric_limits<std::uint32_t>::max())
        { error="Triangle subdivision vertices exceed the geometry budget or native index range"; return false; }
        candidate.Vertices=vertices; candidate.Vertices.reserve(static_cast<std::size_t>(vertexCount));
        candidate.Indices.reserve(static_cast<std::size_t>(indexCount));
        // Assign edges in source encounter order, independently of hash-table
        // iteration order. Reversed input winding uses the same stored samples.
        for (std::size_t triangle=0;triangle<indices.size();triangle+=3)
            for (unsigned corner=0;corner<3;++corner)
            {
                const auto a=indices[triangle+corner],b=indices[triangle+(corner+1)%3];
                auto& first=edges.at(Edge(a,b)); if (first) continue;
                first=static_cast<std::uint32_t>(candidate.Vertices.size());
                const auto& low=vertices[std::min(a,b)]; const auto& high=vertices[std::max(a,b)];
                for (unsigned step=1;step<segments;++step)
                    candidate.Vertices.push_back(Interpolate(low,high,low,segments-step,step,0,segments));
            }
        const auto edgePoint=[&](std::uint32_t a,std::uint32_t b,unsigned step)
        { return edges.at(Edge(a,b))+(a<=b?step:segments-step)-1; };
        const std::size_t latticeSize=std::size_t(segments+1)*(segments+2)/2;
        std::vector<std::uint32_t> lattice(latticeSize);
        const auto row=[&](unsigned i){return std::size_t(i)*(segments+1)-std::size_t(i)*(i?i-1:0)/2;};
        for (std::size_t triangle=0;triangle<indices.size();triangle+=3)
        {
            const auto a=indices[triangle],b=indices[triangle+1],c=indices[triangle+2];
            for (unsigned i=0;i<=segments;++i) for (unsigned j=0;j<=segments-i;++j)
            {
                std::uint32_t index;
                if (!i && !j) index=a;
                else if (i==segments) index=b;
                else if (j==segments) index=c;
                else if (!j) index=edgePoint(a,b,i);
                else if (!i) index=edgePoint(a,c,j);
                else if (i+j==segments) index=edgePoint(b,c,j);
                else
                {
                    index=static_cast<std::uint32_t>(candidate.Vertices.size());
                    candidate.Vertices.push_back(Interpolate(vertices[a],vertices[b],vertices[c],segments-i-j,i,j,segments));
                }
                lattice[row(i)+j]=index;
            }
            for (unsigned i=0;i<segments;++i) for (unsigned j=0;j<segments-i;++j)
            {
                const auto p=lattice[row(i)+j],q=lattice[row(i+1)+j],r=lattice[row(i)+j+1];
                candidate.Indices.insert(candidate.Indices.end(),{p,q,r});
                if (i+j+1<segments) candidate.Indices.insert(candidate.Indices.end(),{q,lattice[row(i+1)+j+1],r});
            }
        }
        output=std::move(candidate); return true;
    }
}
