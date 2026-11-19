#include "Renderer/Pipelines/VisibilityBuffer.h"
#include "Maths/ColorSpace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace Evans::Renderer
{
    namespace
    {
        constexpr double HomogeneousTolerance = 1.0e-12;
        constexpr float NormalSquaredTolerance = 1.0e-12f;
        bool Finite(Float4 value)
        { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z) && std::isfinite(value.W); }
        Float4 Transform(const Matrix4& matrix, Float3 value, float w)
        {
            const auto& m = matrix.Values;
            return {m[0]*value.X+m[1]*value.Y+m[2]*value.Z+m[3]*w,
                m[4]*value.X+m[5]*value.Y+m[6]*value.Z+m[7]*w,
                m[8]*value.X+m[9]*value.Y+m[10]*value.Z+m[11]*w,
                m[12]*value.X+m[13]*value.Y+m[14]*value.Z+m[15]*w};
        }
    }

    bool BuildVisibilityScene(const std::vector<VisibilityMeshView>& meshes, VisibilityScene& output, std::string& error)
    {
        error.clear(); VisibilityScene candidate;
        const auto fail = [&](const char* message) { error = message; return false; };
        if (meshes.size() > UINT32_MAX) return fail("Visibility scene exceeds material addressing");
        for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
        {
            const auto& mesh = meshes[meshIndex];
            if (!mesh.Visible) continue;
            if (!mesh.Vertices || !mesh.Indices || mesh.Indices->size() % VisibilityTriangleCorners)
                return fail("Visibility mesh requires complete indexed triangles");
            const auto triangleCount = mesh.Indices->size() / VisibilityTriangleCorners;
            if (triangleCount > VisibilityStorageCapacity - candidate.Triangles.size())
                return fail("Visibility scene exceeds its packed triangle reference");
            for (float value : mesh.World.Values) if (!std::isfinite(value)) return fail("Visibility world matrix is not finite");
            const auto& world = mesh.World.Values;
            if (world[12] != 0 || world[13] != 0 || world[14] != 0 || world[15] != 1)
                return fail("Visibility world matrix must be affine");
            const auto normalMatrix = BuildRealtimeNormalMatrix(mesh.World);
            VisibilityDrawRange range {static_cast<std::uint32_t>(candidate.Triangles.size()),
                static_cast<std::uint32_t>(triangleCount), static_cast<std::uint32_t>(meshIndex), RealtimeWorldIsMirrored(mesh.World)};
            for (std::size_t primitive = 0; primitive < triangleCount; ++primitive)
            {
                VisibilityTriangle triangle{}; std::array<Float2, 3> uv;
                for (std::size_t corner = 0; corner < VisibilityTriangleCorners; ++corner)
                {
                    const auto index = (*mesh.Indices)[primitive * VisibilityTriangleCorners + corner];
                    if (index >= mesh.Vertices->size()) return fail("Visibility index exceeds its vertex buffer");
                    const auto& vertex = (*mesh.Vertices)[index];
                    auto position = Transform(mesh.World, vertex.Position, 1);
                    auto normal = Transform(normalMatrix, vertex.Normal, 0);
                    if (!Finite(position) || !Finite(normal) || !Finite(vertex.Color) ||
                        !std::isfinite(vertex.TexCoord.X) || !std::isfinite(vertex.TexCoord.Y))
                        return fail("Visibility corner attributes are not finite");
                    const double length = std::hypot(double(normal.X), double(normal.Y), double(normal.Z));
                    if (length * length > NormalSquaredTolerance)
                        normal = {float(normal.X / length), float(normal.Y / length), float(normal.Z / length), 0};
                    else normal = {0, 1, 0, 0};
                    triangle.Position[corner] = position; triangle.Normal[corner] = normal;
                    triangle.Color[corner] = {Maths::SRGBToLinear(std::clamp(vertex.Color.X, 0.0f, 1.0f)),
                        Maths::SRGBToLinear(std::clamp(vertex.Color.Y, 0.0f, 1.0f)),
                        Maths::SRGBToLinear(std::clamp(vertex.Color.Z, 0.0f, 1.0f)), vertex.Color.W};
                    uv[corner] = vertex.TexCoord;
                }
                triangle.UVAB = {uv[0].X, uv[0].Y, uv[1].X, uv[1].Y}; triangle.UVC = {uv[2].X, uv[2].Y, 0, 0};
                triangle.Source = {range.Material, static_cast<std::uint32_t>(primitive), 0, 0};
                candidate.Triangles.push_back(triangle);
            }
            if (triangleCount) candidate.Draws.push_back(range);
        }
        output = std::move(candidate); return true;
    }

    bool BuildVisibilityConstants(const Matrix4& camera, const PipelineState& pipeline, int width, int height,
        std::size_t triangleCount, const VisibilityDrawRange& range, bool diffuseImage, bool normalImage,
        bool opacityImage, VisibilityConstants& output)
    {
        output = {};
        if (width <= 0 || height <= 0 || triangleCount > VisibilityStorageCapacity ||
            range.FirstTriangle > triangleCount || range.TriangleCount > triangleCount - range.FirstTriangle ||
            range.TriangleCount > UINT32_MAX / VisibilityTriangleCorners || pipeline.BlendEnabled || pipeline.OutlineExtrusion ||
            pipeline.Shading == Techniques::ShadingModel::Gouraud)
            return false;
        for (float value : camera.Values) if (!std::isfinite(value)) return false;
        const auto draw = BuildRealtimeDrawConstants(pipeline, {}, camera, {}, PassType::GBuffer,
            false, diffuseImage, normalImage, opacityImage, true);
        VisibilityConstants candidate;
        candidate.ViewProjection = camera;
        candidate.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        candidate.PipelineColor = draw.PipelineColor; candidate.Diffuse = draw.MaterialDiffuseColor;
        candidate.Ambient = draw.MaterialAmbientColor; candidate.Specular = draw.MaterialSpecularColor;
        candidate.Emission = draw.MaterialEmissiveColor; candidate.Material = draw.MaterialParameters;
        candidate.Material.W = pipeline.WireframeOverlay ? 2.0f : pipeline.Fill == FillMode::Wireframe ? 1.0f : 0.0f;
        candidate.AmbientLighting = draw.AmbientColorStrength; candidate.Shading = draw.ShadingParameters;
        candidate.Options = draw.TextureFlags;
        candidate.Draw = {static_cast<std::uint32_t>(triangleCount), range.FirstTriangle, range.TriangleCount, range.Material};
        for (const auto value : {candidate.PipelineColor, candidate.Diffuse, candidate.Ambient, candidate.Specular,
            candidate.Emission, candidate.Material, candidate.AmbientLighting, candidate.Shading})
            if (!Finite(value)) return false;
        output = candidate; return true;
    }

    bool ValidateVisibilityScene(const VisibilityScene& scene, std::string& error)
    {
        error.clear();
        const auto fail = [&](const char* message) { error = message; return false; };
        if (scene.HasNativeGeometry())
        {
            if (!scene.Triangles.empty() || scene.NativeMeshes.size()!=scene.Draws.size() || scene.TriangleCount()>VisibilityStorageCapacity)
                return fail("Native visibility scene must contain only matching GPU ranges");
            std::size_t first=0;
            for (std::size_t i=0;i<scene.NativeMeshes.size();++i)
            {
                const auto& source=scene.NativeMeshes[i]; const auto& range=scene.Draws[i];
                if (!source.VertexBuffer || !source.IndexBuffer || !source.Visible || !source.IndexCount || source.IndexCount%VisibilityTriangleCorners ||
                    range.FirstTriangle!=first || range.TriangleCount!=source.IndexCount/VisibilityTriangleCorners || range.Material!=i ||
                    range.Mirrored!=RealtimeWorldIsMirrored(source.World)) return fail("Native visibility range is inconsistent");
                for (float value:source.World.Values) if (!std::isfinite(value)) return fail("Native visibility transform is not finite");
                const auto& m=source.World.Values;
                if (m[12]!=0 || m[13]!=0 || m[14]!=0 || m[15]!=1) return fail("Native visibility transform must be affine");
                first+=range.TriangleCount;
            }
            return first==scene.NativeTriangleCount || fail("Native visibility ranges do not cover their table");
        }
        if (scene.NativeTriangleCount) return fail("Native visibility count has no GPU sources");
        if (scene.Triangles.size() > VisibilityStorageCapacity || scene.Draws.size() > UINT32_MAX)
            return fail("Visibility table exceeds native addressing");
        std::size_t first = 0;
        for (const auto& range : scene.Draws)
        {
            if (range.FirstTriangle != first || !range.TriangleCount || range.TriangleCount > scene.Triangles.size() - first)
                return fail("Visibility ranges do not partition their triangle table");
            for (std::size_t local = 0; local < range.TriangleCount; ++local)
            {
                const auto& triangle = scene.Triangles[first + local];
                if (triangle.Source.X != range.Material || triangle.Source.Y != local || triangle.Source.Z || triangle.Source.W ||
                    !Finite(triangle.UVAB) || !Finite(triangle.UVC)) return fail("Visibility triangle has invalid source metadata or UVs");
                for (std::size_t corner = 0; corner < VisibilityTriangleCorners; ++corner)
                    if (!Finite(triangle.Position[corner]) || triangle.Position[corner].W != 1 ||
                        !Finite(triangle.Normal[corner]) || !Finite(triangle.Color[corner]))
                        return fail("Visibility triangle has invalid corner attributes");
            }
            first += range.TriangleCount;
        }
        return first == scene.Triangles.size() || fail("Visibility ranges leave unreferenced triangles");
    }

    bool ValidateVisibilityFrame(const VisibilityBufferDesc& frame, int width, int height, std::string& error)
    {
        error.clear();
        if (!frame.Scene || !Finite(frame.Background)) { error = "Visibility frame has no scene or finite background"; return false; }
        if (unsigned(frame.MeshDispatch)>unsigned(Techniques::MeshShaderDispatchMode::Task))
        { error="Invalid mesh dispatch mode"; return false; }
        if (frame.MeshShaders && frame.NaniteRaster && !frame.GpuHardware)
        { error="Nanite mesh stages require GPU hardware-bin scheduling"; return false; }
        if (frame.Scene->HasNativeGeometry() && !frame.NaniteRaster)
        { error="Native decoded visibility requires the Nanite frame producer"; return false; }
        VisibilityConstants constants;
        for (float value : frame.ViewProjection.Values)
            if (!std::isfinite(value)) { error = "Visibility camera is not finite"; return false; }
        std::unordered_set<unsigned> hybridRanges;
        for (const auto& draw : frame.Draws)
        {
            if (frame.NaniteRaster)
            {
                const auto& p=draw.Pipeline;
                if (!p.DepthTest || !p.DepthWrite || p.Fill!=FillMode::Solid || p.Parallax.Enabled || !hybridRanges.insert(draw.Range).second)
                { error="Nanite visibility requires unique opaque solid depth-writing ranges"; return false; }
            }
            if (draw.Range >= frame.Scene->Draws.size() || !BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, width, height,
                frame.Scene->TriangleCount(), frame.Scene->Draws[draw.Range], false, false, false, constants))
            { error = "Visibility draw has an invalid range or material"; return false; }
        }
        if (width <= 0 || height <= 0) { error = "Visibility extent is empty"; return false; }
        return true;
    }

    void RecordVisibilityDraws(FrameCapture& capture, const VisibilityBufferDesc& frame)
    {
        auto& statistics = capture.Statistics;
        for (const auto& draw : frame.Draws)
        {
            const auto& range = frame.Scene->Draws[draw.Range];
            DrawCallRecord record;
            record.Pipeline = draw.Pipeline; record.WorldTransform = draw.World;
            record.VertexBuffer = draw.VertexBuffer; record.IndexBuffer = draw.IndexBuffer;
            record.IndexCount = range.TriangleCount * VisibilityTriangleCorners; record.FirstIndex=draw.FirstIndex;
            record.MeshShader=frame.MeshShaders;
            record.VisibilityBuffer = true; record.VisibilityFirstTriangle = range.FirstTriangle;
            capture.Passes.back().DrawCalls.push_back(record);
            ++statistics.DrawCallCount; ++statistics.VisibilityRasterDrawCount;
            if (frame.MeshShaders && !frame.NaniteRaster) ++statistics.MeshShaderDispatchCount;
            if (!frame.MaterialTiles) ++statistics.VisibilityMaterialDrawCount;
            statistics.SubmittedTriangleCount += range.TriangleCount;
        }
    }

}
