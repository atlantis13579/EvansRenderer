#include "SoftwareRasterCore.hpp"

#include "Maths/ColorSpace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace Evans::Rasterizer
{
namespace
{

using Clock = std::chrono::steady_clock;
constexpr float Epsilon = 1.0e-6f;
constexpr float EdgeTolerance = 1.0e-5f;

struct ClipVertex
{
    Float4 Clip;
    Float3 WorldPosition;
    Float3 WorldNormal;
    Color VertexColor;
    Float4 ShadowClip;
    Float2 TexCoord;
};

struct ScreenVertex
{
    float X = 0.0f;
    float Y = 0.0f;
    float Depth = 0.0f;
    float InverseW = 1.0f;
    Float3 WorldPosition;
    Float3 WorldNormal;
    Color VertexColor;
    Float4 ShadowClip;
    Float2 TexCoord;
};

struct ScreenTriangle
{
    ScreenVertex Vertices[3];
    RasterMaterial Material;
    float SignedArea = 0.0f;
    int MinimumX = 0;
    int MinimumY = 0;
    int MaximumX = -1;
    int MaximumY = -1;
    bool Selected = false;
};

struct PreparedFrame
{
    std::vector<ScreenTriangle> Triangles;
    ShadowMap GeneratedShadowMap;
    const ShadowMap* ActiveShadowMap = nullptr;
};

struct AtomicCounters
{
    std::atomic<std::uint64_t> Shaded{0};
    std::atomic<std::uint64_t> DepthRejected{0};
};

struct TileRect
{
    int MinimumX = 0;
    int MinimumY = 0;
    int MaximumX = -1;
    int MaximumY = -1;
};

[[nodiscard]] Float3 Add(const Float3& left, const Float3& right) noexcept
{
    return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
}

[[nodiscard]] Float3 Subtract(const Float3& left, const Float3& right) noexcept
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] Float3 Multiply(const Float3& value, float scalar) noexcept
{
    return {value.X * scalar, value.Y * scalar, value.Z * scalar};
}

[[nodiscard]] float Dot(const Float3& left, const Float3& right) noexcept
{
    return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
}

[[nodiscard]] Float3 Cross(const Float3& left, const Float3& right) noexcept
{
    return {
        left.Y * right.Z - left.Z * right.Y,
        left.Z * right.X - left.X * right.Z,
        left.X * right.Y - left.Y * right.X};
}

[[nodiscard]] Float3 Normalize(const Float3& value) noexcept
{
    const float lengthSquared = Dot(value, value);
    if (lengthSquared <= Epsilon * Epsilon)
    {
        return {0.0f, 0.0f, 1.0f};
    }
    return Multiply(value, 1.0f / std::sqrt(lengthSquared));
}

[[nodiscard]] Float3 NormalizeOr(
    const Float3& value,
    const Float3& fallback) noexcept
{
    const float lengthSquared = Dot(value, value);
    return lengthSquared <= Epsilon * Epsilon
        ? fallback
        : Multiply(value, 1.0f / std::sqrt(lengthSquared));
}

[[nodiscard]] Float3 TransformNormal(const Matrix4& matrix, const Float3& normal) noexcept
{
    const float a00 = matrix(0, 0);
    const float a01 = matrix(0, 1);
    const float a02 = matrix(0, 2);
    const float a10 = matrix(1, 0);
    const float a11 = matrix(1, 1);
    const float a12 = matrix(1, 2);
    const float a20 = matrix(2, 0);
    const float a21 = matrix(2, 1);
    const float a22 = matrix(2, 2);
    const float determinant =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    if (std::abs(determinant) <= Epsilon)
    {
        return Normalize(normal);
    }
    const float inverseDeterminant = 1.0f / determinant;
    // Cofactors form the inverse-transpose directly.
    return Normalize({
        ((a11 * a22 - a12 * a21) * normal.X +
         (a12 * a20 - a10 * a22) * normal.Y +
         (a10 * a21 - a11 * a20) * normal.Z) * inverseDeterminant,
        ((a02 * a21 - a01 * a22) * normal.X +
         (a00 * a22 - a02 * a20) * normal.Y +
         (a01 * a20 - a00 * a21) * normal.Z) * inverseDeterminant,
        ((a01 * a12 - a02 * a11) * normal.X +
         (a02 * a10 - a00 * a12) * normal.Y +
         (a00 * a11 - a01 * a10) * normal.Z) * inverseDeterminant});
}

[[nodiscard]] bool IsCancelled(
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation) noexcept
{
    return externalCancellation.IsCancellationRequested() ||
           backendCancellation.load(std::memory_order_relaxed);
}

[[nodiscard]] float Edge(
    float ax,
    float ay,
    float bx,
    float by,
    float px,
    float py) noexcept
{
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

[[nodiscard]] ClipVertex Interpolate(
    const ClipVertex& left,
    const ClipVertex& right,
    float amount) noexcept
{
    const float inverse = 1.0f - amount;
    return {
        {left.Clip.X * inverse + right.Clip.X * amount,
         left.Clip.Y * inverse + right.Clip.Y * amount,
         left.Clip.Z * inverse + right.Clip.Z * amount,
         left.Clip.W * inverse + right.Clip.W * amount},
        {left.WorldPosition.X * inverse + right.WorldPosition.X * amount,
         left.WorldPosition.Y * inverse + right.WorldPosition.Y * amount,
         left.WorldPosition.Z * inverse + right.WorldPosition.Z * amount},
        {left.WorldNormal.X * inverse + right.WorldNormal.X * amount,
         left.WorldNormal.Y * inverse + right.WorldNormal.Y * amount,
         left.WorldNormal.Z * inverse + right.WorldNormal.Z * amount},
        {left.VertexColor.R * inverse + right.VertexColor.R * amount,
         left.VertexColor.G * inverse + right.VertexColor.G * amount,
         left.VertexColor.B * inverse + right.VertexColor.B * amount,
         left.VertexColor.A * inverse + right.VertexColor.A * amount},
        {left.ShadowClip.X * inverse + right.ShadowClip.X * amount,
         left.ShadowClip.Y * inverse + right.ShadowClip.Y * amount,
         left.ShadowClip.Z * inverse + right.ShadowClip.Z * amount,
         left.ShadowClip.W * inverse + right.ShadowClip.W * amount},
        {left.TexCoord.X * inverse + right.TexCoord.X * amount,
         left.TexCoord.Y * inverse + right.TexCoord.Y * amount}};
}

[[nodiscard]] float PlaneDistance(
    const ClipVertex& vertex,
    int plane,
    ClipDepthRange depthRange) noexcept
{
    switch (plane)
    {
    case 0: return vertex.Clip.X + vertex.Clip.W;
    case 1: return vertex.Clip.W - vertex.Clip.X;
    case 2: return vertex.Clip.Y + vertex.Clip.W;
    case 3: return vertex.Clip.W - vertex.Clip.Y;
    case 4:
        return depthRange == ClipDepthRange::ZeroToOne
                   ? vertex.Clip.Z
                   : vertex.Clip.Z + vertex.Clip.W;
    case 5: return vertex.Clip.W - vertex.Clip.Z;
    default: return -1.0f;
    }
}

void ClipAgainstPlane(
    const std::vector<ClipVertex>& input,
    std::vector<ClipVertex>& output,
    int plane,
    ClipDepthRange depthRange)
{
    output.clear();
    if (input.empty())
    {
        return;
    }
    ClipVertex previous = input.back();
    float previousDistance = PlaneDistance(previous, plane, depthRange);
    bool previousInside = previousDistance >= 0.0f;
    for (const ClipVertex& current : input)
    {
        const float currentDistance = PlaneDistance(current, plane, depthRange);
        const bool currentInside = currentDistance >= 0.0f;
        if (previousInside != currentInside)
        {
            const float denominator = previousDistance - currentDistance;
            const float amount = std::abs(denominator) > Epsilon
                                     ? previousDistance / denominator
                                     : 0.0f;
            output.push_back(Interpolate(previous, current, amount));
        }
        if (currentInside)
        {
            output.push_back(current);
        }
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
}

[[nodiscard]] ClipVertex TransformVertex(
    const RasterVertex& vertex,
    const RasterSettings& settings,
    bool lightPass) noexcept
{
    ClipVertex result;
    result.VertexColor = vertex.VertexColor;
    result.TexCoord = vertex.TexCoord;
    if (settings.InputSpace == VertexInputSpace::Clip)
    {
        result.Clip = lightPass ? vertex.ShadowClipPosition : vertex.ClipPosition;
        result.ShadowClip = vertex.ShadowClipPosition;
        result.WorldPosition = vertex.Position;
        result.WorldNormal = Normalize(vertex.Normal);
        return result;
    }

    Float4 worldPosition = settings.World * Float4{
        vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1.0f};
    if (std::abs(worldPosition.W) > Epsilon && std::abs(worldPosition.W - 1.0f) > Epsilon)
    {
        const float inverseW = 1.0f / worldPosition.W;
        worldPosition.X *= inverseW;
        worldPosition.Y *= inverseW;
        worldPosition.Z *= inverseW;
        worldPosition.W = 1.0f;
    }
    result.WorldPosition = {worldPosition.X, worldPosition.Y, worldPosition.Z};
    result.WorldNormal = TransformNormal(settings.World, vertex.Normal);
    result.ShadowClip = settings.LightViewProjection * worldPosition;
    result.Clip = lightPass
                      ? result.ShadowClip
                      : settings.Projection * (settings.View * worldPosition);
    return result;
}

[[nodiscard]] bool ToScreenTriangle(
    const ClipVertex& vertex0,
    const ClipVertex& vertex1,
    const ClipVertex& vertex2,
    const RasterTriangle& source,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    ScreenTriangle& output) noexcept
{
    const ClipVertex sourceVertices[] = {vertex0, vertex1, vertex2};
    for (std::size_t index = 0; index < 3; ++index)
    {
        const ClipVertex& input = sourceVertices[index];
        if (!std::isfinite(input.Clip.W) || input.Clip.W <= Epsilon)
        {
            return false;
        }
        const float inverseW = 1.0f / input.Clip.W;
        const float ndcX = input.Clip.X * inverseW;
        const float ndcY = input.Clip.Y * inverseW;
        const float ndcZ = input.Clip.Z * inverseW;
        if (!std::isfinite(ndcX) || !std::isfinite(ndcY) || !std::isfinite(ndcZ))
        {
            return false;
        }
        ScreenVertex& screen = output.Vertices[index];
        screen.X = (ndcX * 0.5f + 0.5f) * static_cast<float>(width);
        screen.Y = (0.5f - ndcY * 0.5f) * static_cast<float>(height);
        screen.Depth = settings.DepthRange == ClipDepthRange::ZeroToOne
                           ? ndcZ
                           : ndcZ * 0.5f + 0.5f;
        screen.InverseW = inverseW;
        screen.WorldPosition = input.WorldPosition;
        screen.WorldNormal = input.WorldNormal;
        screen.VertexColor = input.VertexColor;
        screen.ShadowClip = input.ShadowClip;
        screen.TexCoord = input.TexCoord;
    }

    output.SignedArea = Edge(
        output.Vertices[0].X,
        output.Vertices[0].Y,
        output.Vertices[1].X,
        output.Vertices[1].Y,
        output.Vertices[2].X,
        output.Vertices[2].Y);
    if (std::abs(output.SignedArea) <= EdgeTolerance)
    {
        return false;
    }
    if (settings.BackFaceCulling)
    {
        const bool frontFacing = settings.FrontFaceCounterClockwise
                                     ? output.SignedArea < 0.0f
                                     : output.SignedArea > 0.0f;
        if (!frontFacing)
        {
            return false;
        }
    }

    output.MinimumX = std::max(0, static_cast<int>(std::floor(std::min({
                                      output.Vertices[0].X,
                                      output.Vertices[1].X,
                                      output.Vertices[2].X}))));
    output.MinimumY = std::max(0, static_cast<int>(std::floor(std::min({
                                      output.Vertices[0].Y,
                                      output.Vertices[1].Y,
                                      output.Vertices[2].Y}))));
    output.MaximumX = std::min(
        static_cast<int>(width) - 1,
        static_cast<int>(std::ceil(std::max({
            output.Vertices[0].X, output.Vertices[1].X, output.Vertices[2].X}))));
    output.MaximumY = std::min(
        static_cast<int>(height) - 1,
        static_cast<int>(std::ceil(std::max({
            output.Vertices[0].Y, output.Vertices[1].Y, output.Vertices[2].Y}))));
    output.Material = source.Material;
    output.Selected = source.Selected;
    return output.MinimumX <= output.MaximumX && output.MinimumY <= output.MaximumY;
}

[[nodiscard]] std::vector<ScreenTriangle> PrepareTriangles(
    const RasterScene& scene,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    bool lightPass,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation,
    std::uint64_t& postClipTriangleCount)
{
    std::vector<ScreenTriangle> triangles;
    triangles.reserve(scene.Triangles.size());
    std::vector<ClipVertex> polygon;
    std::vector<ClipVertex> temporary;
    polygon.reserve(10);
    temporary.reserve(10);
    for (std::size_t triangleIndex = 0; triangleIndex < scene.Triangles.size(); ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U && IsCancelled(externalCancellation, backendCancellation))
        {
            break;
        }
        const RasterTriangle& source = scene.Triangles[triangleIndex];
        polygon = {
            TransformVertex(source.Vertices[0], settings, lightPass),
            TransformVertex(source.Vertices[1], settings, lightPass),
            TransformVertex(source.Vertices[2], settings, lightPass)};
        for (int plane = 0; plane < 6 && !polygon.empty(); ++plane)
        {
            ClipAgainstPlane(polygon, temporary, plane, settings.DepthRange);
            polygon.swap(temporary);
        }
        if (polygon.size() < 3)
        {
            continue;
        }
        for (std::size_t index = 1; index + 1 < polygon.size(); ++index)
        {
            ScreenTriangle output;
            if (ToScreenTriangle(
                    polygon[0],
                    polygon[index],
                    polygon[index + 1],
                    source,
                    settings,
                    width,
                    height,
                    output))
            {
                triangles.push_back(std::move(output));
                ++postClipTriangleCount;
            }
        }
    }
    return triangles;
}

[[nodiscard]] bool IsTriangleSurfaceDiscarded(
    const ScreenTriangle& triangle,
    const std::vector<RasterTexture>& textures,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept;

void RasterizeShadowDepth(
    const std::vector<ScreenTriangle>& triangles,
    const std::vector<RasterTexture>& textures,
    ShadowMap& shadowMap,
    float slopeScaledBias,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation)
{
    for (std::size_t triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
    {
        if ((triangleIndex & 127U) == 0U && IsCancelled(externalCancellation, backendCancellation))
        {
            return;
        }
        const ScreenTriangle& triangle = triangles[triangleIndex];
        float depthBias = 0.0f;
        if (slopeScaledBias > 0.0f)
        {
            const auto& a = triangle.Vertices[0];
            const auto& b = triangle.Vertices[1];
            const auto& c = triangle.Vertices[2];
            const float dz1 = b.Depth - a.Depth;
            const float dz2 = c.Depth - a.Depth;
            const float slopeX = (dz1 * (c.Y - a.Y) - dz2 * (b.Y - a.Y)) / triangle.SignedArea;
            const float slopeY = ((b.X - a.X) * dz2 - (c.X - a.X) * dz1) / triangle.SignedArea;
            depthBias = slopeScaledBias * std::max(std::abs(slopeX), std::abs(slopeY));
        }
        for (int y = triangle.MinimumY; y <= triangle.MaximumY; ++y)
        {
            for (int x = triangle.MinimumX; x <= triangle.MaximumX; ++x)
            {
                const float sampleX = static_cast<float>(x) + 0.5f;
                const float sampleY = static_cast<float>(y) + 0.5f;
                const float weight0 = Edge(
                                          triangle.Vertices[1].X,
                                          triangle.Vertices[1].Y,
                                          triangle.Vertices[2].X,
                                          triangle.Vertices[2].Y,
                                          sampleX,
                                          sampleY) /
                                      triangle.SignedArea;
                const float weight1 = Edge(
                                          triangle.Vertices[2].X,
                                          triangle.Vertices[2].Y,
                                          triangle.Vertices[0].X,
                                          triangle.Vertices[0].Y,
                                          sampleX,
                                          sampleY) /
                                      triangle.SignedArea;
                const float weight2 = 1.0f - weight0 - weight1;
                if (weight0 < -EdgeTolerance || weight1 < -EdgeTolerance ||
                    weight2 < -EdgeTolerance)
                {
                    continue;
                }
                float depth = weight0 * triangle.Vertices[0].Depth +
                              weight1 * triangle.Vertices[1].Depth +
                              weight2 * triangle.Vertices[2].Depth;
                if (depth < 0.0f || depth > 1.0f)
                {
                    continue;
                }
                depth = std::min(depth + depthBias, 1.0f);
                const std::size_t pixel = static_cast<std::size_t>(y) * shadowMap.Width +
                                          static_cast<std::size_t>(x);
                if (depth >= shadowMap.Depth[pixel])
                {
                    continue;
                }
                const float p0 = weight0 * triangle.Vertices[0].InverseW;
                const float p1 = weight1 * triangle.Vertices[1].InverseW;
                const float p2 = weight2 * triangle.Vertices[2].InverseW;
                const float denominator = p0 + p1 + p2;
                if (std::abs(denominator) <= Epsilon ||
                    IsTriangleSurfaceDiscarded(
                        triangle,
                        textures,
                        p0,
                        p1,
                        p2,
                        denominator))
                {
                    continue;
                }
                shadowMap.Depth[pixel] = depth;
            }
        }
    }
}

[[nodiscard]] PreparedFrame PrepareFrame(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation,
    RasterStatistics& statistics)
{
    PreparedFrame frame;
    const Clock::time_point begin = Clock::now();
    statistics.InputTriangles = scene.Triangles.size();
    frame.Triangles = PrepareTriangles(
        scene,
        settings,
        target.Width(),
        target.Height(),
        false,
        externalCancellation,
        backendCancellation,
        statistics.PostClipTriangles);

    if (settings.ShadowsEnabled && settings.SceneShadowMap != nullptr &&
        settings.SceneShadowMap->IsValid())
    {
        frame.ActiveShadowMap = settings.SceneShadowMap;
    }
    else if (settings.ShadowsEnabled && settings.AutoGenerateShadowMap &&
             !scene.Triangles.empty() &&
             !IsCancelled(externalCancellation, backendCancellation))
    {
        const std::uint32_t size = std::clamp(settings.ShadowMapSize, 16U, 4096U);
        frame.GeneratedShadowMap.Width = size;
        frame.GeneratedShadowMap.Height = size;
        frame.GeneratedShadowMap.Depth.assign(
            static_cast<std::size_t>(size) * size,
            std::numeric_limits<float>::infinity());
        std::uint64_t ignoredCount = 0;
        std::vector<ScreenTriangle> lightTriangles = PrepareTriangles(
            scene,
            settings,
            size,
            size,
            true,
            externalCancellation,
            backendCancellation,
            ignoredCount);
        RasterizeShadowDepth(
            lightTriangles,
            scene.Textures,
            frame.GeneratedShadowMap,
            settings.ShadowSlopeScaledBias,
            externalCancellation,
            backendCancellation);
        frame.ActiveShadowMap = &frame.GeneratedShadowMap;
    }
    statistics.TransformMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return frame;
}

[[nodiscard]] Float3 InterpolateFloat3(
    const Float3& value0,
    const Float3& value1,
    const Float3& value2,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    return {
        (value0.X * p0 + value1.X * p1 + value2.X * p2) / denominator,
        (value0.Y * p0 + value1.Y * p1 + value2.Y * p2) / denominator,
        (value0.Z * p0 + value1.Z * p1 + value2.Z * p2) / denominator};
}

[[nodiscard]] Color InterpolateColor(
    const ScreenTriangle& triangle,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    const Color& color0 = triangle.Vertices[0].VertexColor;
    const Color& color1 = triangle.Vertices[1].VertexColor;
    const Color& color2 = triangle.Vertices[2].VertexColor;
    return {
        (color0.R * p0 + color1.R * p1 + color2.R * p2) / denominator,
        (color0.G * p0 + color1.G * p1 + color2.G * p2) / denominator,
        (color0.B * p0 + color1.B * p1 + color2.B * p2) / denominator,
        (color0.A * p0 + color1.A * p1 + color2.A * p2) / denominator};
}

[[nodiscard]] Float2 InterpolateFloat2(
    const Float2& value0,
    const Float2& value1,
    const Float2& value2,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    return {
        (value0.X * p0 + value1.X * p1 + value2.X * p2) / denominator,
        (value0.Y * p0 + value1.Y * p1 + value2.Y * p2) / denominator};
}

[[nodiscard]] Float4 InterpolateFloat4(
    const Float4& value0,
    const Float4& value1,
    const Float4& value2,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    return {
        (value0.X * p0 + value1.X * p1 + value2.X * p2) / denominator,
        (value0.Y * p0 + value1.Y * p1 + value2.Y * p2) / denominator,
        (value0.Z * p0 + value1.Z * p1 + value2.Z * p2) / denominator,
        (value0.W * p0 + value1.W * p1 + value2.W * p2) / denominator};
}

enum class TextureSampleConversion
{
    Preserve,
    DecodeSRGB
};

[[nodiscard]] float RepeatCoordinate(float value) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    return value - std::floor(value);
}

[[nodiscard]] std::uint32_t WrapTexel(
    std::int64_t coordinate,
    std::uint32_t size) noexcept
{
    const std::int64_t signedSize = static_cast<std::int64_t>(size);
    std::int64_t wrapped = coordinate % signedSize;
    if (wrapped < 0)
    {
        wrapped += signedSize;
    }
    return static_cast<std::uint32_t>(wrapped);
}

[[nodiscard]] Color SampleRasterTexture(
    const std::vector<RasterTexture>& textures,
    std::uint32_t textureIndex,
    const Float2& texCoord,
    TextureSampleConversion conversion) noexcept
{
    if (textureIndex >= textures.size() ||
        !textures[textureIndex].IsValid())
    {
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }

    const RasterTexture& texture = textures[textureIndex];
    const float x = RepeatCoordinate(texCoord.X) *
                        static_cast<float>(texture.Width) -
                    0.5f;
    const float y = RepeatCoordinate(1.0f - texCoord.Y) *
                        static_cast<float>(texture.Height) -
                    0.5f;
    const float floorX = std::floor(x);
    const float floorY = std::floor(y);
    const std::int64_t x0 = static_cast<std::int64_t>(floorX);
    const std::int64_t y0 = static_cast<std::int64_t>(floorY);
    const std::uint32_t sampleX[2] = {
        WrapTexel(x0, texture.Width),
        WrapTexel(x0 + 1, texture.Width)};
    const std::uint32_t sampleY[2] = {
        WrapTexel(y0, texture.Height),
        WrapTexel(y0 + 1, texture.Height)};
    const float blendX = x - floorX;
    const float blendY = y - floorY;
    const std::vector<std::uint8_t>& pixels = *texture.PixelsRGBA8;
    const auto load = [&](std::uint32_t pixelX, std::uint32_t pixelY)
    {
        const std::size_t byte =
            (static_cast<std::size_t>(pixelY) * texture.Width + pixelX) * 4U;
        Color result{
            static_cast<float>(pixels[byte]) / 255.0f,
            static_cast<float>(pixels[byte + 1U]) / 255.0f,
            static_cast<float>(pixels[byte + 2U]) / 255.0f,
            static_cast<float>(pixels[byte + 3U]) / 255.0f};
        if (conversion == TextureSampleConversion::DecodeSRGB)
        {
            result.R = Maths::SRGBToLinear(result.R);
            result.G = Maths::SRGBToLinear(result.G);
            result.B = Maths::SRGBToLinear(result.B);
        }
        return result;
    };
    const auto lerp = [](const Color& left, const Color& right, float amount)
    {
        const float inverse = 1.0f - amount;
        return Color{
            left.R * inverse + right.R * amount,
            left.G * inverse + right.G * amount,
            left.B * inverse + right.B * amount,
            left.A * inverse + right.A * amount};
    };
    const Color top = lerp(
        load(sampleX[0], sampleY[0]),
        load(sampleX[1], sampleY[0]),
        blendX);
    const Color bottom = lerp(
        load(sampleX[0], sampleY[1]),
        load(sampleX[1], sampleY[1]),
        blendX);
    return lerp(top, bottom, blendY);
}

struct SurfaceSample
{
    Color Albedo;
    bool Discarded = false;
};

[[nodiscard]] SurfaceSample EvaluateSurface(
    const RasterMaterial& material,
    const std::vector<RasterTexture>& textures,
    const Color& vertexColor,
    const Float2& texCoord) noexcept
{
    const Color diffuse = SampleRasterTexture(
        textures,
        material.DiffuseTextureIndex,
        texCoord,
        TextureSampleConversion::DecodeSRGB);
    const Color opacity = SampleRasterTexture(
        textures,
        material.OpacityTextureIndex,
        texCoord,
        TextureSampleConversion::Preserve);
    const float alpha = std::clamp(
        vertexColor.A * material.BaseColor.A * diffuse.A * opacity.R,
        0.0f,
        1.0f);
    return {
        {vertexColor.R * material.BaseColor.R * diffuse.R,
         vertexColor.G * material.BaseColor.G * diffuse.G,
         vertexColor.B * material.BaseColor.B * diffuse.B,
         alpha},
        material.AlphaCutoff > 0.0f && alpha < material.AlphaCutoff};
}

[[nodiscard]] SurfaceSample SampleTriangleSurface(
    const ScreenTriangle& triangle,
    const std::vector<RasterTexture>& textures,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    return EvaluateSurface(
        triangle.Material,
        textures,
        InterpolateColor(triangle, p0, p1, p2, denominator),
        InterpolateFloat2(
            triangle.Vertices[0].TexCoord,
            triangle.Vertices[1].TexCoord,
            triangle.Vertices[2].TexCoord,
            p0,
            p1,
            p2,
            denominator));
}

bool IsTriangleSurfaceDiscarded(
    const ScreenTriangle& triangle,
    const std::vector<RasterTexture>& textures,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    return SampleTriangleSurface(
               triangle,
               textures,
               p0,
               p1,
               p2,
               denominator)
        .Discarded;
}

[[nodiscard]] float SampleShadowVisibility(
    const ScreenTriangle& triangle,
    const RasterSettings& settings,
    const ShadowMap* shadowMap,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    if (!settings.ShadowsEnabled || shadowMap == nullptr || !shadowMap->IsValid())
    {
        return 1.0f;
    }

    const Float4 shadowClip = InterpolateFloat4(
        triangle.Vertices[0].ShadowClip,
        triangle.Vertices[1].ShadowClip,
        triangle.Vertices[2].ShadowClip,
        p0,
        p1,
        p2,
        denominator);
    if (!std::isfinite(shadowClip.W) || shadowClip.W <= Epsilon)
    {
        return 1.0f;
    }

    const float inverseW = 1.0f / shadowClip.W;
    const float ndcX = shadowClip.X * inverseW;
    const float ndcY = shadowClip.Y * inverseW;
    const float ndcZ = shadowClip.Z * inverseW;
    const float depth = settings.DepthRange == ClipDepthRange::ZeroToOne
                            ? ndcZ
                            : ndcZ * 0.5f + 0.5f;
    const float textureX = ndcX * 0.5f + 0.5f;
    const float textureY = 0.5f - ndcY * 0.5f;
    if (textureX < 0.0f || textureX > 1.0f || textureY < 0.0f || textureY > 1.0f ||
        depth < 0.0f || depth > 1.0f)
    {
        return 1.0f;
    }

    const int centerX = std::clamp(
        static_cast<int>(textureX * static_cast<float>(shadowMap->Width)),
        0,
        static_cast<int>(shadowMap->Width) - 1);
    const int centerY = std::clamp(
        static_cast<int>(textureY * static_cast<float>(shadowMap->Height)),
        0,
        static_cast<int>(shadowMap->Height) - 1);
    const int radius = static_cast<int>(std::min(settings.ShadowPCFRadius, 8U));
    std::uint32_t occludedSamples = 0;
    std::uint32_t validSamples = 0;
    for (int offsetY = -radius; offsetY <= radius; ++offsetY)
    {
        const int sampleY = centerY + offsetY;
        if (sampleY < 0 || sampleY >= static_cast<int>(shadowMap->Height))
        {
            continue;
        }
        for (int offsetX = -radius; offsetX <= radius; ++offsetX)
        {
            const int sampleX = centerX + offsetX;
            if (sampleX < 0 || sampleX >= static_cast<int>(shadowMap->Width))
            {
                continue;
            }
            ++validSamples;
            if (depth - settings.ShadowBias >
                shadowMap->Depth[static_cast<std::size_t>(sampleY) * shadowMap->Width +
                                 static_cast<std::size_t>(sampleX)])
            {
                ++occludedSamples;
            }
        }
    }
    if (validSamples == 0)
    {
        return 1.0f;
    }
    const float occlusion =
        static_cast<float>(occludedSamples) / static_cast<float>(validSamples);
    return 1.0f - std::clamp(settings.ShadowStrength, 0.0f, 1.0f) * occlusion;
}

[[nodiscard]] Color Shade(
    const ScreenTriangle& triangle,
    const RasterSettings& settings,
    const ShadowMap* shadowMap,
    const Color& albedo,
    float p0,
    float p1,
    float p2,
    float denominator) noexcept
{
    const auto prepareDisplayLinear = [](const Color& color) noexcept
    {
        const float scale = Maths::DisplayLinearScale(color.R, color.G, color.B);
        return Color{
            std::clamp(color.R * scale, 0.0f, 1.0f),
            std::clamp(color.G * scale, 0.0f, 1.0f),
            std::clamp(color.B * scale, 0.0f, 1.0f),
            std::clamp(color.A, 0.0f, 1.0f)};
    };
    if (settings.Shading == ShadingModel::Unlit)
    {
        return prepareDisplayLinear({
            albedo.R + triangle.Material.EmissiveColor.R,
            albedo.G + triangle.Material.EmissiveColor.G,
            albedo.B + triangle.Material.EmissiveColor.B,
            albedo.A});
    }

    Float3 normal = Normalize(InterpolateFloat3(
        triangle.Vertices[0].WorldNormal,
        triangle.Vertices[1].WorldNormal,
        triangle.Vertices[2].WorldNormal,
        p0,
        p1,
        p2,
        denominator));
    const bool frontFacing = settings.FrontFaceCounterClockwise
        ? triangle.SignedArea < 0.0f
        : triangle.SignedArea > 0.0f;
    if (!frontFacing)
    {
        normal = Multiply(normal, -1.0f);
    }
    const Float3 worldPosition = InterpolateFloat3(
        triangle.Vertices[0].WorldPosition,
        triangle.Vertices[1].WorldPosition,
        triangle.Vertices[2].WorldPosition,
        p0,
        p1,
        p2,
        denominator);
    const Float3 viewDirection = NormalizeOr(
        Subtract(settings.CameraPosition, worldPosition),
        normal);
    Float3 diffuseLight;
    Float3 specularLight;
    const auto accumulateSample = [&](
                                      const Float3& lightDirection,
                                      const Float3& radiance,
                                      float visibility)
    {
        const float directionLengthSquared = Dot(lightDirection, lightDirection);
        if (directionLengthSquared <= Epsilon * Epsilon)
        {
            return;
        }
        const Float3 direction = Multiply(
            lightDirection,
            1.0f / std::sqrt(directionLengthSquared));
        const float normalToLight = std::max(Dot(normal, direction), 0.0f);
        if (normalToLight <= 0.0f)
        {
            return;
        }
        const Float3 visibleRadiance = Multiply(radiance, visibility);
        diffuseLight = Add(diffuseLight, Multiply(visibleRadiance, normalToLight));
        if (settings.Shading == ShadingModel::Phong)
        {
            const Float3 halfVector = NormalizeOr(Add(direction, viewDirection), normal);
            const float specularAmount = std::pow(
                std::max(Dot(normal, halfVector), 0.0f),
                std::max(triangle.Material.Shininess, 1.0f));
            specularLight = Add(
                specularLight,
                Multiply(visibleRadiance, specularAmount));
        }
    };

    const std::uint32_t lightCount = EffectiveRasterLightCount(settings);
    for (std::uint32_t lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        const RasterLight light = EffectiveRasterLight(settings, lightIndex);
        const float visibility =
            static_cast<std::int32_t>(lightIndex) == settings.ShadowLightIndex &&
                    light.CastShadows
                ? SampleShadowVisibility(
                      triangle,
                      settings,
                      shadowMap,
                      p0,
                      p1,
                      p2,
                      denominator)
                : 1.0f;
        const Float3 radiance{
            std::max(light.LightColor.R, 0.0f) * std::max(light.Intensity, 0.0f),
            std::max(light.LightColor.G, 0.0f) * std::max(light.Intensity, 0.0f),
            std::max(light.LightColor.B, 0.0f) * std::max(light.Intensity, 0.0f)};
        if (light.Type == RasterLightType::Directional)
        {
            accumulateSample(light.Direction, radiance, visibility);
            continue;
        }
        if (light.Type == RasterLightType::Point)
        {
            const Float3 toLight = Subtract(light.Position, worldPosition);
            const float distanceSquared = std::max(Dot(toLight, toLight), 1.0e-6f);
            accumulateSample(
                toLight,
                Multiply(radiance, 1.0f / distanceSquared),
                visibility);
            continue;
        }

        const Float3 emissionDirection = NormalizeOr(
            light.Direction,
            {0.0f, -1.0f, 0.0f});
        const Float3 helper = std::abs(emissionDirection.Y) < 0.999f
            ? Float3{0.0f, 1.0f, 0.0f}
            : Float3{1.0f, 0.0f, 0.0f};
        const Float3 tangent = NormalizeOr(
            Cross(helper, emissionDirection),
            {1.0f, 0.0f, 0.0f});
        const Float3 bitangent = NormalizeOr(
            Cross(emissionDirection, tangent),
            {0.0f, 0.0f, 1.0f});
        for (std::uint32_t sampleY = 0; sampleY < 2; ++sampleY)
        {
            for (std::uint32_t sampleX = 0; sampleX < 2; ++sampleX)
            {
                const float offsetX = sampleX == 0 ? -0.25f : 0.25f;
                const float offsetY = sampleY == 0 ? -0.25f : 0.25f;
                const Float3 samplePosition = Add(
                    light.Position,
                    Add(
                        Multiply(tangent, offsetX * light.AreaSize.X),
                        Multiply(bitangent, offsetY * light.AreaSize.Y)));
                const Float3 toLight = Subtract(samplePosition, worldPosition);
                const float distanceSquared = std::max(Dot(toLight, toLight), 1.0e-6f);
                const Float3 direction = Multiply(toLight, 1.0f / std::sqrt(distanceSquared));
                const float emitterCosine = std::max(
                    Dot(emissionDirection, Multiply(direction, -1.0f)),
                    0.0f);
                accumulateSample(
                    direction,
                    Multiply(radiance, 0.25f * emitterCosine / distanceSquared),
                    visibility);
            }
        }
    }

    const Color result{
        albedo.R *
                (settings.AmbientColor.R * settings.AmbientStrength + diffuseLight.X) +
            triangle.Material.SpecularColor.R * specularLight.X +
            triangle.Material.EmissiveColor.R,
        albedo.G *
                (settings.AmbientColor.G * settings.AmbientStrength + diffuseLight.Y) +
            triangle.Material.SpecularColor.G * specularLight.Y +
            triangle.Material.EmissiveColor.G,
        albedo.B *
                (settings.AmbientColor.B * settings.AmbientStrength + diffuseLight.Z) +
            triangle.Material.SpecularColor.B * specularLight.Z +
            triangle.Material.EmissiveColor.B,
        albedo.A};
    return prepareDisplayLinear(result);
}

void RasterizeSolidInTile(
    const ScreenTriangle& triangle,
    const TileRect& tile,
    const RasterSettings& settings,
    const std::vector<RasterTexture>& textures,
    const ShadowMap* shadowMap,
    RenderTarget& target,
    AtomicCounters& counters)
{
    const int minimumX = std::max(triangle.MinimumX, tile.MinimumX);
    const int minimumY = std::max(triangle.MinimumY, tile.MinimumY);
    const int maximumX = std::min(triangle.MaximumX, tile.MaximumX);
    const int maximumY = std::min(triangle.MaximumY, tile.MaximumY);
    for (int y = minimumY; y <= maximumY; ++y)
    {
        for (int x = minimumX; x <= maximumX; ++x)
        {
            const float sampleX = static_cast<float>(x) + 0.5f;
            const float sampleY = static_cast<float>(y) + 0.5f;
            const float weight0 = Edge(
                                      triangle.Vertices[1].X,
                                      triangle.Vertices[1].Y,
                                      triangle.Vertices[2].X,
                                      triangle.Vertices[2].Y,
                                      sampleX,
                                      sampleY) /
                                  triangle.SignedArea;
            const float weight1 = Edge(
                                      triangle.Vertices[2].X,
                                      triangle.Vertices[2].Y,
                                      triangle.Vertices[0].X,
                                      triangle.Vertices[0].Y,
                                      sampleX,
                                      sampleY) /
                                  triangle.SignedArea;
            const float weight2 = 1.0f - weight0 - weight1;
            if (weight0 < -EdgeTolerance || weight1 < -EdgeTolerance ||
                weight2 < -EdgeTolerance)
            {
                continue;
            }

            const float depth = weight0 * triangle.Vertices[0].Depth +
                                weight1 * triangle.Vertices[1].Depth +
                                weight2 * triangle.Vertices[2].Depth;
            if (depth < 0.0f || depth > 1.0f)
            {
                continue;
            }
            const std::size_t pixel = static_cast<std::size_t>(y) * target.Width() +
                                      static_cast<std::size_t>(x);
            if (settings.DepthTest && depth >= target.MutableDepths()[pixel])
            {
                counters.DepthRejected.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const float perspective0 = weight0 * triangle.Vertices[0].InverseW;
            const float perspective1 = weight1 * triangle.Vertices[1].InverseW;
            const float perspective2 = weight2 * triangle.Vertices[2].InverseW;
            const float denominator = perspective0 + perspective1 + perspective2;
            if (std::abs(denominator) <= Epsilon)
            {
                continue;
            }
            const SurfaceSample surface = SampleTriangleSurface(
                triangle,
                textures,
                perspective0,
                perspective1,
                perspective2,
                denominator);
            if (surface.Discarded)
            {
                continue;
            }
            target.MutableColors()[pixel] = Shade(
                triangle,
                settings,
                shadowMap,
                surface.Albedo,
                perspective0,
                perspective1,
                perspective2,
                denominator);
            target.MutableDepths()[pixel] = depth;
            counters.Shaded.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void RasterizeLineInTile(
    const ScreenVertex& start,
    const ScreenVertex& end,
    const RasterMaterial& material,
    const std::vector<RasterTexture>& textures,
    float width,
    const Color& color,
    float depthTolerance,
    const TileRect& tile,
    const RasterSettings& settings,
    RenderTarget& target,
    AtomicCounters& counters)
{
    const float halfWidth = std::max(width, 1.0f) * 0.5f;
    const int minimumX = std::max(
        tile.MinimumX, static_cast<int>(std::floor(std::min(start.X, end.X) - halfWidth)));
    const int minimumY = std::max(
        tile.MinimumY, static_cast<int>(std::floor(std::min(start.Y, end.Y) - halfWidth)));
    const int maximumX = std::min(
        tile.MaximumX, static_cast<int>(std::ceil(std::max(start.X, end.X) + halfWidth)));
    const int maximumY = std::min(
        tile.MaximumY, static_cast<int>(std::ceil(std::max(start.Y, end.Y) + halfWidth)));
    const float deltaX = end.X - start.X;
    const float deltaY = end.Y - start.Y;
    const float lengthSquared = deltaX * deltaX + deltaY * deltaY;
    if (lengthSquared <= Epsilon)
    {
        return;
    }
    for (int y = minimumY; y <= maximumY; ++y)
    {
        for (int x = minimumX; x <= maximumX; ++x)
        {
            const float sampleX = static_cast<float>(x) + 0.5f;
            const float sampleY = static_cast<float>(y) + 0.5f;
            const float amount = std::clamp(
                ((sampleX - start.X) * deltaX + (sampleY - start.Y) * deltaY) /
                    lengthSquared,
                0.0f,
                1.0f);
            const float distanceX = sampleX - (start.X + amount * deltaX);
            const float distanceY = sampleY - (start.Y + amount * deltaY);
            if (distanceX * distanceX + distanceY * distanceY > halfWidth * halfWidth)
            {
                continue;
            }
            const float depth = start.Depth + amount * (end.Depth - start.Depth);
            if (depth < 0.0f || depth > 1.0f)
            {
                continue;
            }
            const std::size_t pixel = static_cast<std::size_t>(y) * target.Width() +
                                      static_cast<std::size_t>(x);
            if (settings.DepthTest && depth > target.MutableDepths()[pixel] + depthTolerance)
            {
                counters.DepthRejected.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const float p0 = (1.0f - amount) * start.InverseW;
            const float p1 = amount * end.InverseW;
            const float denominator = p0 + p1;
            if (std::abs(denominator) <= Epsilon)
            {
                continue;
            }
            const float inverseDenominator = 1.0f / denominator;
            const Color vertexColor{
                (start.VertexColor.R * p0 + end.VertexColor.R * p1) *
                    inverseDenominator,
                (start.VertexColor.G * p0 + end.VertexColor.G * p1) *
                    inverseDenominator,
                (start.VertexColor.B * p0 + end.VertexColor.B * p1) *
                    inverseDenominator,
                (start.VertexColor.A * p0 + end.VertexColor.A * p1) *
                    inverseDenominator};
            const Float2 texCoord{
                (start.TexCoord.X * p0 + end.TexCoord.X * p1) *
                    inverseDenominator,
                (start.TexCoord.Y * p0 + end.TexCoord.Y * p1) *
                    inverseDenominator};
            if (EvaluateSurface(material, textures, vertexColor, texCoord).Discarded)
            {
                continue;
            }
            target.MutableColors()[pixel] = color.Clamped();
            target.MutableDepths()[pixel] = std::min(target.MutableDepths()[pixel], depth);
            counters.Shaded.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void RasterizeEdgesInTile(
    const ScreenTriangle& triangle,
    const std::vector<RasterTexture>& textures,
    float width,
    const Color& color,
    float depthTolerance,
    const TileRect& tile,
    const RasterSettings& settings,
    RenderTarget& target,
    AtomicCounters& counters)
{
    for (std::size_t edgeIndex = 0; edgeIndex < 3; ++edgeIndex)
    {
        RasterizeLineInTile(
            triangle.Vertices[edgeIndex],
            triangle.Vertices[(edgeIndex + 1U) % 3U],
            triangle.Material,
            textures,
            width,
            color,
            depthTolerance,
            tile,
            settings,
            target,
            counters);
    }
}

void RasterizeVisibilityMaskInTile(
    const ScreenTriangle& triangle,
    const std::vector<RasterTexture>& textures,
    const TileRect& tile,
    std::uint32_t targetWidth,
    std::vector<float>& visibilityDepth,
    std::vector<std::uint8_t>& selectionMask)
{
    const int minimumX = std::max(triangle.MinimumX, tile.MinimumX);
    const int minimumY = std::max(triangle.MinimumY, tile.MinimumY);
    const int maximumX = std::min(triangle.MaximumX, tile.MaximumX);
    const int maximumY = std::min(triangle.MaximumY, tile.MaximumY);
    for (int y = minimumY; y <= maximumY; ++y)
    {
        for (int x = minimumX; x <= maximumX; ++x)
        {
            const float sampleX = static_cast<float>(x) + 0.5f;
            const float sampleY = static_cast<float>(y) + 0.5f;
            const float weight0 = Edge(
                                      triangle.Vertices[1].X,
                                      triangle.Vertices[1].Y,
                                      triangle.Vertices[2].X,
                                      triangle.Vertices[2].Y,
                                      sampleX,
                                      sampleY) /
                                  triangle.SignedArea;
            const float weight1 = Edge(
                                      triangle.Vertices[2].X,
                                      triangle.Vertices[2].Y,
                                      triangle.Vertices[0].X,
                                      triangle.Vertices[0].Y,
                                      sampleX,
                                      sampleY) /
                                  triangle.SignedArea;
            const float weight2 = 1.0f - weight0 - weight1;
            if (weight0 < -EdgeTolerance || weight1 < -EdgeTolerance ||
                weight2 < -EdgeTolerance)
            {
                continue;
            }

            const float depth = weight0 * triangle.Vertices[0].Depth +
                                weight1 * triangle.Vertices[1].Depth +
                                weight2 * triangle.Vertices[2].Depth;
            if (depth < 0.0f || depth > 1.0f)
            {
                continue;
            }
            const std::size_t pixel = static_cast<std::size_t>(y) * targetWidth +
                                      static_cast<std::size_t>(x);
            if (depth >= visibilityDepth[pixel])
            {
                continue;
            }
            const float p0 = weight0 * triangle.Vertices[0].InverseW;
            const float p1 = weight1 * triangle.Vertices[1].InverseW;
            const float p2 = weight2 * triangle.Vertices[2].InverseW;
            const float denominator = p0 + p1 + p2;
            if (std::abs(denominator) <= Epsilon ||
                SampleTriangleSurface(
                    triangle,
                    textures,
                    p0,
                    p1,
                    p2,
                    denominator)
                    .Discarded)
            {
                continue;
            }
            visibilityDepth[pixel] = depth;
            selectionMask[pixel] = triangle.Selected ? 1U : 0U;
        }
    }
}

void BuildVisibilityMaskInTile(
    const std::vector<ScreenTriangle>& triangles,
    const std::vector<std::uint32_t>& triangleIndices,
    const std::vector<RasterTexture>& textures,
    const TileRect& tile,
    std::uint32_t targetWidth,
    std::vector<float>& visibilityDepth,
    std::vector<std::uint8_t>& selectionMask)
{
    for (const std::uint32_t triangleIndex : triangleIndices)
    {
        RasterizeVisibilityMaskInTile(
            triangles[triangleIndex],
            textures,
            tile,
            targetWidth,
            visibilityDepth,
            selectionMask);
    }
}

void ApplySelectionOutlineInTile(
    const TileRect& tile,
    const RasterSettings& settings,
    const std::vector<float>& visibilityDepth,
    const std::vector<std::uint8_t>& selectionMask,
    RenderTarget& target,
    AtomicCounters& counters)
{
    const int radius = std::max(1, static_cast<int>(std::ceil(settings.OutlineWidth)));
    const int radiusSquared = radius * radius;
    const int width = static_cast<int>(target.Width());
    const int height = static_cast<int>(target.Height());
    for (int y = tile.MinimumY; y <= tile.MaximumY; ++y)
    {
        for (int x = tile.MinimumX; x <= tile.MaximumX; ++x)
        {
            const std::size_t pixel = static_cast<std::size_t>(y) * target.Width() +
                                      static_cast<std::size_t>(x);
            if (selectionMask[pixel] != 0U)
            {
                continue;
            }

            float nearestSelectedDepth = std::numeric_limits<float>::infinity();
            for (int offsetY = -radius; offsetY <= radius; ++offsetY)
            {
                const int sampleY = y + offsetY;
                if (sampleY < 0 || sampleY >= height)
                {
                    continue;
                }
                for (int offsetX = -radius; offsetX <= radius; ++offsetX)
                {
                    if (offsetX * offsetX + offsetY * offsetY > radiusSquared)
                    {
                        continue;
                    }
                    const int sampleX = x + offsetX;
                    if (sampleX < 0 || sampleX >= width)
                    {
                        continue;
                    }
                    const std::size_t samplePixel = static_cast<std::size_t>(sampleY) *
                                                        target.Width() +
                                                    static_cast<std::size_t>(sampleX);
                    if (selectionMask[samplePixel] != 0U)
                    {
                        nearestSelectedDepth = std::min(
                            nearestSelectedDepth, visibilityDepth[samplePixel]);
                    }
                }
            }

            if (!std::isfinite(nearestSelectedDepth) ||
                nearestSelectedDepth > visibilityDepth[pixel] + 2.0e-3f)
            {
                continue;
            }
            target.MutableColors()[pixel] = settings.OutlineColor.Clamped();
            counters.Shaded.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void RasterizeTriangleStages(
    const std::vector<ScreenTriangle>& triangles,
    const std::vector<std::uint32_t>* triangleIndices,
    const std::vector<RasterTexture>& textures,
    const TileRect& tile,
    const RasterSettings& settings,
    const ShadowMap* shadowMap,
    RenderTarget& target,
    AtomicCounters& counters)
{
    const auto forEachTriangle = [&](const auto& function)
    {
        if (triangleIndices != nullptr)
        {
            for (const std::uint32_t index : *triangleIndices)
            {
                function(triangles[index]);
            }
        }
        else
        {
            for (const ScreenTriangle& triangle : triangles)
            {
                if (triangle.MaximumX >= tile.MinimumX && triangle.MinimumX <= tile.MaximumX &&
                    triangle.MaximumY >= tile.MinimumY && triangle.MinimumY <= tile.MaximumY)
                {
                    function(triangle);
                }
            }
        }
    };

    if (settings.Mode == FillMode::Solid || settings.Mode == FillMode::SolidWireframe)
    {
        forEachTriangle([&](const ScreenTriangle& triangle)
        {
            RasterizeSolidInTile(
                triangle,
                tile,
                settings,
                textures,
                shadowMap,
                target,
                counters);
        });
    }
    if (settings.Mode == FillMode::Wireframe || settings.Mode == FillMode::SolidWireframe)
    {
        forEachTriangle([&](const ScreenTriangle& triangle)
        {
            RasterizeEdgesInTile(
                triangle,
                textures,
                settings.WireframeWidth,
                settings.WireframeColor,
                1.0e-4f,
                tile,
                settings,
                target,
                counters);
        });
    }
}

[[nodiscard]] std::uint32_t ResolveWorkerCount(std::uint32_t requested, std::uint32_t jobs) noexcept
{
    const std::uint32_t hardware = std::max(std::thread::hardware_concurrency(), 1U);
    const std::uint32_t workers = requested == 0 ? hardware : requested;
    return std::max(1U, std::min(workers, std::max(jobs, 1U)));
}

template <typename Function>
void ParallelDispatch(
    std::uint32_t jobCount,
    std::uint32_t workerCount,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation,
    Function&& function)
{
    std::atomic<std::uint32_t> nextJob{0};
    const auto worker = [&]()
    {
        while (!IsCancelled(externalCancellation, backendCancellation))
        {
            const std::uint32_t job = nextJob.fetch_add(1, std::memory_order_relaxed);
            if (job >= jobCount)
            {
                break;
            }
            function(job);
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workerCount > 0 ? workerCount - 1U : 0U);
    for (std::uint32_t index = 1; index < workerCount; ++index)
    {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

[[nodiscard]] TileRect TileBounds(
    std::uint32_t tileIndex,
    std::uint32_t tilesX,
    std::uint32_t tileWidth,
    std::uint32_t tileHeight,
    const RenderTarget& target) noexcept
{
    const std::uint32_t tileX = tileIndex % tilesX;
    const std::uint32_t tileY = tileIndex / tilesX;
    return {
        static_cast<int>(tileX * tileWidth),
        static_cast<int>(tileY * tileHeight),
        std::min(static_cast<int>((tileX + 1U) * tileWidth), static_cast<int>(target.Width())) - 1,
        std::min(static_cast<int>((tileY + 1U) * tileHeight), static_cast<int>(target.Height())) - 1};
}

void FinalizeStatistics(
    RasterStatistics& statistics,
    const AtomicCounters& counters,
    Clock::time_point rasterBegin) noexcept
{
    statistics.ShadedFragments = counters.Shaded.load(std::memory_order_relaxed);
    statistics.DepthRejectedFragments = counters.DepthRejected.load(std::memory_order_relaxed);
    statistics.RasterMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - rasterBegin).count();
}

} // namespace

RasterStatus SoftwareRasterCore::RenderCpuTiled(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation,
    std::uint32_t workerThreads,
    CpuRasterResources& resources,
    RasterStatistics& statistics,
    CpuRasterPass pass)
{
    const bool overlayOnly = pass != CpuRasterPass::Full;
    const bool wireOverlay = pass == CpuRasterPass::PresentationOverlay &&
        settings.Mode != FillMode::Solid;
    if (!overlayOnly)
        target.Clear(settings.ClearColor.Clamped(), std::numeric_limits<float>::infinity());
    RasterSettings preparationSettings = settings;
    if (overlayOnly) preparationSettings.ShadowsEnabled = false;
    PreparedFrame frame = PrepareFrame(
        scene, preparationSettings, target, externalCancellation, backendCancellation, statistics);
    if (IsCancelled(externalCancellation, backendCancellation))
    {
        return RasterStatus::Cancelled;
    }

    constexpr std::uint32_t tileWidth = 16;
    constexpr std::uint32_t tileHeight = 16;
    const std::uint32_t tilesX = (target.Width() + tileWidth - 1U) / tileWidth;
    const std::uint32_t tilesY = (target.Height() + tileHeight - 1U) / tileHeight;
    const std::uint32_t tileCount = tilesX * tilesY;
    resources.TileBins.assign(tileCount, {});
    resources.SelectionMask.assign(target.Colors().size(), 0U);
    const bool renderOutline = settings.OutlineEnabled &&
                               std::any_of(
                                   frame.Triangles.begin(),
                                   frame.Triangles.end(),
                                   [](const ScreenTriangle& triangle)
                                   {
                                       return triangle.Selected;
                                   });
    const bool renderVisibility = renderOutline || wireOverlay;
    if (overlayOnly && !renderVisibility) return RasterStatus::Completed;
    std::vector<float> visibilityDepth;
    if (renderVisibility)
    {
        visibilityDepth.assign(
            target.Depths().size(), std::numeric_limits<float>::infinity());
    }
    for (std::uint32_t triangleIndex = 0;
         triangleIndex < static_cast<std::uint32_t>(frame.Triangles.size());
         ++triangleIndex)
    {
        const ScreenTriangle& triangle = frame.Triangles[triangleIndex];
        const std::uint32_t firstX = static_cast<std::uint32_t>(triangle.MinimumX) / tileWidth;
        const std::uint32_t firstY = static_cast<std::uint32_t>(triangle.MinimumY) / tileHeight;
        const std::uint32_t lastX = static_cast<std::uint32_t>(triangle.MaximumX) / tileWidth;
        const std::uint32_t lastY = static_cast<std::uint32_t>(triangle.MaximumY) / tileHeight;
        for (std::uint32_t y = firstY; y <= lastY; ++y)
        {
            for (std::uint32_t x = firstX; x <= lastX; ++x)
            {
                resources.TileBins[y * tilesX + x].push_back(triangleIndex);
            }
        }
    }

    const Clock::time_point rasterBegin = Clock::now();
    AtomicCounters counters;
    const std::uint32_t workers = ResolveWorkerCount(workerThreads, tileCount);
    statistics.WorkerThreads = workers;
    statistics.DispatchedTiles = tileCount;
    statistics.ExecutionPath = "CPU parallel 16x16 tiled rasterizer";
    if (renderVisibility)
    {
        ParallelDispatch(
            tileCount,
            workers,
            externalCancellation,
            backendCancellation,
            [&](std::uint32_t tileIndex)
            {
                BuildVisibilityMaskInTile(
                    frame.Triangles,
                    resources.TileBins[tileIndex],
                    scene.Textures,
                    TileBounds(tileIndex, tilesX, tileWidth, tileHeight, target),
                    target.Width(),
                    visibilityDepth,
                    resources.SelectionMask);
            });
    }
    // Solid overlays need surface depths even though they preserve the base colors.
    if (wireOverlay) target.MutableDepths() = visibilityDepth;
    RasterSettings drawingSettings = settings;
    if (wireOverlay) drawingSettings.Mode = FillMode::Wireframe;
    if (!overlayOnly || wireOverlay) ParallelDispatch(
        tileCount,
        workers,
        externalCancellation,
        backendCancellation,
        [&](std::uint32_t tileIndex)
        {
            RasterizeTriangleStages(
                frame.Triangles,
                &resources.TileBins[tileIndex],
                scene.Textures,
                TileBounds(tileIndex, tilesX, tileWidth, tileHeight, target),
                drawingSettings,
                frame.ActiveShadowMap,
                target,
                counters);
        });
    if (renderOutline && !IsCancelled(externalCancellation, backendCancellation))
    {
        ParallelDispatch(
            tileCount,
            workers,
            externalCancellation,
            backendCancellation,
            [&](std::uint32_t tileIndex)
            {
                ApplySelectionOutlineInTile(
                    TileBounds(tileIndex, tilesX, tileWidth, tileHeight, target),
                    settings,
                    visibilityDepth,
                    resources.SelectionMask,
                    target,
                    counters);
            });
    }
    FinalizeStatistics(statistics, counters, rasterBegin);
    return IsCancelled(externalCancellation, backendCancellation)
               ? RasterStatus::Cancelled
               : RasterStatus::Completed;
}

} // namespace Evans::Rasterizer
