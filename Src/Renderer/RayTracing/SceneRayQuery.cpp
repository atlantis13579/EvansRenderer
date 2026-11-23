#include "Renderer/RayTracing/SceneRayQuery.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Maths/ColorSpace.h"
#include "Maths/Matrix4.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    // The existing median-split AABB tree owns traversal and triangle tests.
    // Acceptance is evaluated before updating the closest distance, so alpha
    // rejection does not skip another surface at the same depth. No fixed
    // "maximum transparent layers" or repeated-ray epsilon is necessary.
    //
    // Texture evaluation follows the raster material contract: bilinear repeat,
    // flipped OBJ V, sRGB diffuse RGB decoded BEFORE filtering, linear opacity
    // red and normal RGB. Ray footprints use mip zero; no derivative LOD exists.
    // Blend surfaces are excluded while blending is enabled. Cutout thresholds
    // remain authored material classification even when alpha testing is off.
    namespace
    {
        using Maths::Vector3;
        constexpr float ByteMaximum = 255.0f;
        constexpr std::size_t PixelComponents = 4;
        constexpr float TexelCenter = 0.5f;
        constexpr float VectorSquaredEpsilon = 1.0e-16f;
        constexpr float TangentDeterminantEpsilon = 1.0e-8f;
        constexpr float ParallelAxisLimit = 0.999f;
        constexpr std::size_t TriangleCorners = 3;

        Vector3 Vector(Float3 v) { return {v.X, v.Y, v.Z}; }
        Float3 Native(Vector3 v) { return {v.x, v.y, v.z}; }
        bool Finite(Float3 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z); }
        bool Finite(Float4 v) { return Finite(Float3{v.X, v.Y, v.Z}) && std::isfinite(v.W); }
        Vector3 Unit(Vector3 value, Vector3 fallback)
        { return value.LengthSquared() > VectorSquaredEpsilon ? value.Normalized() : fallback; }
        Float4 Multiply(Float4 a, Float4 b) { return {a.X*b.X, a.Y*b.Y, a.Z*b.Z, a.W*b.W}; }
        Float4 Interpolate(Float4 a, Float4 b, float weight)
        { return {a.X+(b.X-a.X)*weight, a.Y+(b.Y-a.Y)*weight, a.Z+(b.Z-a.Z)*weight, a.W+(b.W-a.W)*weight}; }
        Float4 Sample(const std::shared_ptr<const ImageData>& image, Float2 uv, bool srgb)
        {
            if (!image || !image->IsValid()) return {1, 1, 1, 1};
            // Reduce to [0,1) before conversion, including negative/large UVs.
            const double u = double(uv.X) - std::floor(double(uv.X));
            const double v = -double(uv.Y) - std::floor(-double(uv.Y));
            const double x = u * image->Width - TexelCenter, y = v * image->Height - TexelCenter;
            const auto left = static_cast<std::int64_t>(std::floor(x)), top = static_cast<std::int64_t>(std::floor(y));
            auto read = [&](std::int64_t px, std::int64_t py) {
                const auto wrap = [](std::int64_t value, std::uint32_t size) { return (value % size + size) % size; };
                const auto offset = (static_cast<std::size_t>(wrap(py, image->Height)) * image->Width +
                    static_cast<std::size_t>(wrap(px, image->Width))) * PixelComponents;
                const auto* texel = image->PixelsRGBA8.data() + offset;
                Float4 result {texel[0]/ByteMaximum, texel[1]/ByteMaximum, texel[2]/ByteMaximum, texel[3]/ByteMaximum};
                if (srgb) { result.X = Maths::SRGBToLinear(result.X); result.Y = Maths::SRGBToLinear(result.Y); result.Z = Maths::SRGBToLinear(result.Z); }
                return result;
            };
            const float horizontal = static_cast<float>(x - left), vertical = static_cast<float>(y - top);
            return Interpolate(Interpolate(read(left, top), read(left+1, top), horizontal),
                Interpolate(read(left, top+1), read(left+1, top+1), horizontal), vertical);
        }
    }

    std::shared_ptr<const SceneRayQuery> SceneRayQuery::Create(const std::vector<SceneRayMeshView>& meshes,
        const Techniques::Settings& settings, std::string& error)
    {
        error.clear();
        auto result = std::shared_ptr<SceneRayQuery>(new SceneRayQuery);
        result->TextureMapping = settings.TextureMapping;
        result->NormalMapping = settings.NormalMapping && settings.Shading != Techniques::ShadingModel::Flat &&
            settings.Shading != Techniques::ShadingModel::Gouraud;
        result->AlphaTesting = settings.AlphaTesting;
        result->FlatShading = settings.Shading == Techniques::ShadingModel::Flat;
        result->PBR = settings.Shading == Techniques::ShadingModel::PBR;
        result->Metallic = std::isfinite(settings.Metallic) ? std::clamp(settings.Metallic, 0.0f, 1.0f) : 0;
        std::vector<Vector3> positions;
        std::vector<std::uint32_t> indices;
        auto fail = [&error](std::size_t index, const char* message) -> std::shared_ptr<const SceneRayQuery> {
            error = "Scene ray mesh " + std::to_string(index) + ": " + message; return {};
        };
        for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
        {
            const auto& source = meshes[meshIndex];
            if (!source.Visible) continue;
            if (!source.Vertices || !source.Indices || source.Vertices->empty() || source.Indices->empty() ||
                source.Indices->size() % TriangleCorners != 0) return fail(meshIndex, "triangle-list geometry is required");
            if (meshIndex > std::numeric_limits<std::uint32_t>::max() ||
                source.Vertices->size() > std::numeric_limits<std::uint32_t>::max() - positions.size() ||
                source.Indices->size()/TriangleCorners > std::numeric_limits<std::uint32_t>::max()-result->TriangleMeshes.size())
                return fail(meshIndex, "geometry exceeds the 32-bit query capacity");
            if (!Finite(source.Color) || !Finite(source.Material.BaseColor) || !Finite(source.Material.EmissiveColor) ||
                !std::isfinite(source.Material.AlphaCutoff)) return fail(meshIndex, "non-finite material data");
            const auto& world = source.World.Values;
            if (!std::all_of(world.begin(), world.end(), [](float v) { return std::isfinite(v); }) ||
                world[12] != 0 || world[13] != 0 || world[14] != 0 || world[15] != 1)
                return fail(meshIndex, "world transform must be finite and affine");
            for (const auto& vertex : *source.Vertices)
                if (!Finite(vertex.Position) || !Finite(vertex.Normal) || !Finite(vertex.Color) ||
                    !std::isfinite(vertex.TexCoord.X) || !std::isfinite(vertex.TexCoord.Y))
                    return fail(meshIndex, "non-finite vertex data");
            if ((settings.AlphaBlending && IsMeshTranslucent(*source.Vertices, source.Color, source.Material)) ||
                (Techniques::UsesRefraction(settings) && source.Material.Refractive)) continue;
            const Maths::Matrix4 transform(world), normalTransform(BuildRealtimeNormalMatrix(source.World).Values);
            const auto vertexBase = static_cast<std::uint32_t>(positions.size());
            const auto materialIndex = static_cast<std::uint32_t>(result->Meshes.size());
            const bool mirrored = RealtimeWorldIsMirrored(source.World);
            result->Meshes.push_back({source.Color, source.Material, static_cast<std::uint32_t>(meshIndex), source.CastShadow,
                vertexBase, static_cast<std::uint32_t>(result->TriangleMeshes.size()), mirrored});
            for (const auto& vertex : *source.Vertices)
            {
                auto transformed = vertex;
                transformed.Position = Native(transform.TransformPoint(Vector(vertex.Position)));
                transformed.Normal = Native(normalTransform.TransformVector(Vector(vertex.Normal)));
                if (!Finite(transformed.Position) || !Finite(transformed.Normal)) return fail(meshIndex, "transformed vertex overflow");
                positions.push_back(Vector(transformed.Position));
                result->Vertices.push_back(transformed);
                // Native vertex shaders decode authored RGB before interpolation.
                // Keep original colors for source correspondence, and cache the
                // decoded values once instead of repeating transfer functions
                // in every material/alpha query during a bake.
                result->LinearVertexColors.push_back({Maths::SRGBToLinear(std::clamp(vertex.Color.X,0.0f,1.0f)),
                    Maths::SRGBToLinear(std::clamp(vertex.Color.Y,0.0f,1.0f)),Maths::SRGBToLinear(std::clamp(vertex.Color.Z,0.0f,1.0f))});
            }
            for (std::size_t first = 0; first < source.Indices->size(); first += TriangleCorners)
            {
                for (std::size_t corner = 0; corner < TriangleCorners; ++corner)
                {
                    const auto sourceCorner = mirrored && corner != 0 ? TriangleCorners - corner : corner;
                    const auto index = (*source.Indices)[first + sourceCorner];
                    if (index >= source.Vertices->size()) return fail(meshIndex, "index outside the vertex array");
                    indices.push_back(vertexBase + index);
                }
                result->TriangleMeshes.push_back(materialIndex);
            }
        }
        if (!indices.empty())
        {
            if (!result->Geometry.SetGeometry(std::move(positions), std::move(indices), &error)) return {};
            const auto& bounds = result->Geometry.GetBounds();
            if (!std::isfinite((bounds.Max - bounds.Min).LengthSquared()))
            { error = "Scene ray bounds exceed finite world-distance arithmetic"; return {}; }
            if (!result->Geometry.BuildAcceleration()) { error = "Scene ray acceleration build failed"; return {}; }
        }
        return result;
    }

    SceneRayHit SceneRayQuery::ReadHit(std::uint32_t triangle, const Maths::RayTriangleHit& hit, bool shading) const
    {
        const auto first = static_cast<std::size_t>(triangle) * TriangleCorners;
        const auto& indices = Geometry.GetIndices();
        const auto& a = Vertices[indices[first]], &b = Vertices[indices[first+1]], &c = Vertices[indices[first+2]];
        const auto& mesh = Meshes[TriangleMeshes[triangle]];
        const auto& material = mesh.Material;
        const auto weights = hit.Barycentric;
        auto blend = [&](float x, float y, float z) { return weights.x*x + weights.y*y + weights.z*z; };
        SceneRayHit result;
        result.Position = Native(hit.Position); result.GeometricNormal = Native(hit.Normal);
        result.Distance = hit.Distance; result.FrontFace = hit.FrontFace;
        result.MeshIndex = mesh.SourceIndex; result.TriangleIndex = triangle;
        result.TexCoord = {blend(a.TexCoord.X, b.TexCoord.X, c.TexCoord.X), blend(a.TexCoord.Y, b.TexCoord.Y, c.TexCoord.Y)};
        const auto& colorA = LinearVertexColors[indices[first]], &colorB = LinearVertexColors[indices[first+1]], &colorC = LinearVertexColors[indices[first+2]];
        Float4 vertexColor {blend(colorA.X, colorB.X, colorC.X), blend(colorA.Y, colorB.Y, colorC.Y),
            blend(colorA.Z, colorB.Z, colorC.Z), blend(a.Color.W, b.Color.W, c.Color.W)};
        const auto texture = TextureMapping ? Sample(material.DiffuseImage, result.TexCoord, true) : Float4{1,1,1,1};
        const auto base = Multiply(Multiply(vertexColor, mesh.Color), Multiply(texture, material.BaseColor));
        result.Opacity = std::clamp(base.W * Sample(material.OpacityImage, result.TexCoord, false).X, 0.0f, 1.0f);
        if (!shading) return result;
        if (PBR) result.Metallic = std::clamp(material.Metallic.value_or(Metallic) *
            Sample(material.MetallicRoughnessImage, result.TexCoord, false).Z, 0.0f, 1.0f);
        result.Diffuse = {std::max(base.X, 0.0f), std::max(base.Y, 0.0f), std::max(base.Z, 0.0f)};
        result.Emission = {std::max(material.EmissiveColor.X, 0.0f), std::max(material.EmissiveColor.Y, 0.0f), std::max(material.EmissiveColor.Z, 0.0f)};
        Vector3 normal = FlatShading ? hit.Normal : Unit(Vector(a.Normal)*weights.x + Vector(b.Normal)*weights.y + Vector(c.Normal)*weights.z, hit.Normal);
        if (!hit.FrontFace) normal = -normal;
        if (NormalMapping && material.NormalImage && material.NormalImage->IsValid())
        {
            const auto edgeB = Vector(b.Position) - Vector(a.Position), edgeC = Vector(c.Position) - Vector(a.Position);
            const Float2 uvB {b.TexCoord.X-a.TexCoord.X, a.TexCoord.Y-b.TexCoord.Y};
            const Float2 uvC {c.TexCoord.X-a.TexCoord.X, a.TexCoord.Y-c.TexCoord.Y};
            const float determinant = uvB.X*uvC.Y - uvB.Y*uvC.X;
            if (std::abs(determinant) >= TangentDeterminantEpsilon)
            {
                auto tangent = (edgeB*uvC.Y - edgeC*uvB.Y) / determinant;
                const auto sourceBitangent = (edgeC*uvB.X - edgeB*uvC.X) / determinant;
                const auto helper = std::abs(normal.y) < ParallelAxisLimit ? Vector3::UnitY() : Vector3::UnitX();
                tangent = Unit(tangent - normal*normal.Dot(tangent), Unit(helper.Cross(normal), Vector3::UnitX()));
                const auto bitangent = normal.Cross(tangent) * (normal.Cross(tangent).Dot(sourceBitangent) < 0 ? -1.0f : 1.0f);
                const auto sampled = Sample(material.NormalImage, result.TexCoord, false);
                normal = Unit(tangent*(sampled.X*2-1) + bitangent*(sampled.Y*2-1) + normal*(sampled.Z*2-1), normal);
            }
        }
        result.Normal = Native(normal);
        return result;
    }

    bool SceneRayQuery::GetTriangleMaterial(std::uint32_t triangle, MaterialData& material, Float4& color) const
    {
        if (triangle >= TriangleMeshes.size()) return false;
        const auto& mesh = Meshes[TriangleMeshes[triangle]];
        material = mesh.Material; color = mesh.Color;
        if (!TextureMapping) material.DiffuseImage.reset();
        if (!NormalMapping) material.NormalImage.reset();
        if (!AlphaTesting) material.AlphaCutoff = 0;
        material.Metallic = PBR ? material.Metallic.value_or(Metallic) : 0;
        if (!PBR) material.MetallicRoughnessImage.reset();
        return true;
    }

    bool SceneRayQuery::GetTriangle(std::uint32_t triangle, SceneRayTriangle& output) const noexcept
    {
        if (triangle >= Geometry.GetTriangleCount()) return false;
        const auto first = static_cast<std::size_t>(triangle) * TriangleCorners;
        const auto& indices = Geometry.GetIndices();
        const auto& mesh = Meshes[TriangleMeshes[triangle]];
        for (std::size_t corner = 0; corner < TriangleCorners; ++corner)
        {
            output.Corners[corner] = Vertices[indices[first+corner]];
            output.SourceVertexIndices[corner] = indices[first+corner]-mesh.VertexBase;
            output.SourceCorners[corner] = static_cast<std::uint32_t>(mesh.Mirrored && corner != 0 ? TriangleCorners-corner : corner);
        }
        output.MeshIndex = mesh.SourceIndex;
        output.SourceTriangleIndex = triangle-mesh.FirstTriangle;
        output.CastShadow = mesh.CastShadow;
        return true;
    }

    bool SceneRayQuery::SampleSurface(std::uint32_t triangle, Float3 barycentric,
        bool frontFace, SceneRayHit& output) const
    {
        output = {};
        constexpr float BarycentricSumTolerance = 1.0e-5f;
        auto weights = Vector(barycentric);
        const float sum = weights.x + weights.y + weights.z;
        if (!Finite(barycentric) || weights.x < 0 || weights.y < 0 || weights.z < 0 ||
            std::abs(sum - 1) > BarycentricSumTolerance) return false;
        weights /= sum;
        Vector3 a, b, c;
        if (!Geometry.GetTriangle(triangle, a, b, c)) return false;
        // A directly addressed texel has no ray determinant to normalize. Use
        // double edge products so tiny or large valid triangles retain normals.
        const double abX = double(b.x)-a.x, abY = double(b.y)-a.y, abZ = double(b.z)-a.z;
        const double acX = double(c.x)-a.x, acY = double(c.y)-a.y, acZ = double(c.z)-a.z;
        const double normalX = abY*acZ-abZ*acY, normalY = abZ*acX-abX*acZ, normalZ = abX*acY-abY*acX;
        const double normalLength = std::hypot(normalX,normalY,normalZ);
        if (!(normalLength > 0) || !std::isfinite(normalLength)) return false;
        Maths::RayTriangleHit hit;
        hit.Barycentric = weights; hit.Position = a*weights.x + b*weights.y + c*weights.z;
        hit.Normal = {float(normalX/normalLength),float(normalY/normalLength),float(normalZ/normalLength)}; hit.FrontFace = frontFace;
        const auto& material = Meshes[TriangleMeshes[triangle]].Material;
        if (AlphaTesting && material.AlphaCutoff > 0 && ReadHit(triangle, hit, false).Opacity < material.AlphaCutoff) return false;
        output = ReadHit(triangle, hit, true);
        return true;
    }

    bool SceneRayQuery::Trace(Float3 origin, Float3 direction, float minimumDistance,
        float maximumDistance, SceneRayHit& hit, bool shadowCastersOnly) const
    {
        hit = {};
        Maths::TriMeshHitOptions options;
        options.MinimumDistance = minimumDistance; options.MaximumDistance = maximumDistance;
        options.AcceptHit = [&](std::uint32_t triangle, const Maths::RayTriangleHit& candidate) {
            const auto& mesh = Meshes[TriangleMeshes[triangle]];
            if (shadowCastersOnly && !mesh.CastShadow) return false;
            return !AlphaTesting || mesh.Material.AlphaCutoff <= 0 ||
                ReadHit(triangle, candidate, false).Opacity >= mesh.Material.AlphaCutoff;
        };
        Maths::TriMeshHit accepted;
        if (!Geometry.RayCast({Vector(origin), Vector(direction)}, accepted, options)) return false;
        hit = ReadHit(accepted.TriangleIndex, accepted, true);
        return true;
    }
}
