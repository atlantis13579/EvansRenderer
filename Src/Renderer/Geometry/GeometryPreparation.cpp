#include "Renderer/Geometry/GeometryPreparation.h"

#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    bool ValidateTriangleGeometry(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,std::string& error)
    {
        if (vertices.empty() || vertices.size()>std::numeric_limits<std::uint32_t>::max() || indices.empty() || indices.size()%3)
        { error="Geometry preparation requires nonempty indexed triangles and 32-bit vertex addresses"; return false; }
        for (const auto& v:vertices)
        {
            const float attributes[]{v.Position.X,v.Position.Y,v.Position.Z,v.Normal.X,v.Normal.Y,v.Normal.Z,
                v.Color.X,v.Color.Y,v.Color.Z,v.Color.W,v.TexCoord.X,v.TexCoord.Y};
            for (float value:attributes) if (!std::isfinite(value))
            { error="Geometry preparation requires finite positions, normals, colors and coordinates"; return false; }
        }
        for (auto index:indices) if (index>=vertices.size())
        { error="Geometry preparation found a vertex index outside the source allocation"; return false; }
        return true;
    }
}
