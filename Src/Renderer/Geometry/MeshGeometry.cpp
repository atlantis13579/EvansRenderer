#include "Renderer/Geometry/MeshGeometry.h"

#include <cstring>
#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    bool ValidateIndexedGeometryDraw(const BufferDesc& vertex,const BufferDesc& index,
        const std::vector<std::uint8_t>& indices,bool nativeDecoded,std::uint32_t firstIndex,
        std::uint32_t indexCount,int baseVertex,std::string& error, IndexedDrawRange* lastValidRange)
    {
        const auto count=index.SizeInBytes/sizeof(std::uint32_t),vertices=vertex.SizeInBytes/sizeof(Vertex);
        if (vertex.Usage!=BufferUsage::Vertex || index.Usage!=BufferUsage::Index || !vertices ||
            vertex.SizeInBytes%sizeof(Vertex) || index.SizeInBytes%sizeof(std::uint32_t) || indexCount%3 ||
            firstIndex>count || indexCount>count-firstIndex)
        { error="Indexed draw has incompatible buffers or triangle range"; return false; }
        if (nativeDecoded)
        {
            if (baseVertex) { error="Native decoded geometry requires base vertex zero"; return false; }
            error.clear(); return true;
        }
        if (indices.size()!=index.SizeInBytes)
        { error="Indexed draw has no complete host index mirror"; return false; }
        if (lastValidRange && lastValidRange->VertexBytes == vertex.SizeInBytes &&
            lastValidRange->FirstIndex == firstIndex && lastValidRange->IndexCount == indexCount &&
            lastValidRange->BaseVertex == baseVertex)
        {
            error.clear();
            return true;
        }
        for (std::size_t at=firstIndex;at<std::size_t(firstIndex)+indexCount;++at)
        {
            std::uint32_t value; std::memcpy(&value,indices.data()+at*sizeof(value),sizeof(value));
            const auto resolved=std::int64_t(value)+baseVertex;
            if (resolved<0 || std::uint64_t(resolved)>=vertices)
            { error="Indexed draw references a vertex outside its buffer"; return false; }
        }
        if (lastValidRange) *lastValidRange = {firstIndex, indexCount, baseVertex, vertex.SizeInBytes};
        error.clear(); return true;
    }

    /*
    Instancing prerequisites
    -------------------------
    Repeated scene objects can reuse one immutable vertex/index allocation.
    Exact identity includes position, normal, vertex RGBA, UV and index order;
    it must not weld seams, infer equivalence from bounds or merge materials.
    FNV-1a narrows the lookup. Full stream comparison resolves hash collisions.
    Float bits are preserved (including signed zero), and struct padding is never
    hashed. This is conservative sharing, not geometry compression.

    Sharing buffers alone does not implement instanced GPU draws. Future batches
    must additionally agree on the pass, material/pipeline, selected LOD and
    winding. Current/previous transforms and velocity validity remain per object.
    Sorted transparency must preserve draw order; an unordered geometry bucket
    is not a valid transparent batch. CPU outlines likewise retain object identity.
    */
    namespace
    {
        std::uint32_t Bits(float value) noexcept
        {
            std::uint32_t result;
            static_assert(sizeof(result) == sizeof(value), "Geometry identity requires 32-bit floats");
            std::memcpy(&result, &value, sizeof(result));
            return result;
        }
        std::array<std::uint32_t, 12> Components(const Vertex& v) noexcept
        {
            return {Bits(v.Position.X), Bits(v.Position.Y), Bits(v.Position.Z),
                Bits(v.Normal.X), Bits(v.Normal.Y), Bits(v.Normal.Z),
                Bits(v.Color.X), Bits(v.Color.Y), Bits(v.Color.Z), Bits(v.Color.W),
                Bits(v.TexCoord.X), Bits(v.TexCoord.Y)};
        }
    }
    std::uint64_t HashMeshGeometry(const std::vector<Vertex>& vertices, const std::vector<std::uint32_t>& indices) noexcept
    {
        constexpr std::uint64_t FnvOffsetBasis = 14695981039346656037ull;
        constexpr std::uint64_t FnvPrime = 1099511628211ull;
        constexpr unsigned BitsPerByte = 8;
        constexpr std::uint64_t ByteMask = 255;
        std::uint64_t hash = FnvOffsetBasis;
        const auto append = [&](std::uint64_t value, unsigned bytes)
        {
            for (unsigned byte = 0; byte < bytes; ++byte)
            {
                hash = (hash ^ (value & ByteMask)) * FnvPrime;
                value >>= BitsPerByte;
            }
        };
        append(vertices.size(), sizeof(std::uint64_t));
        append(indices.size(), sizeof(std::uint64_t));
        for (const auto& vertex : vertices)
            for (auto component : Components(vertex)) append(component, sizeof(component));
        for (auto index : indices) append(index, sizeof(index));
        return hash;
    }
    bool EqualMeshGeometry(const std::vector<Vertex>& leftVertices, const std::vector<std::uint32_t>& leftIndices,
        const std::vector<Vertex>& rightVertices, const std::vector<std::uint32_t>& rightIndices) noexcept
    {
        if (leftVertices.size() != rightVertices.size() || leftIndices != rightIndices) return false;
        for (std::size_t i = 0; i < leftVertices.size(); ++i)
            if (Components(leftVertices[i]) != Components(rightVertices[i])) return false;
        return true;
    }

    bool IsMeshTranslucent(const std::vector<Vertex>& vertices, Float4 color, const MaterialData& material) noexcept
    {
        // Explicit alpha thresholds remain binary cutouts. Without a
        // threshold, both diffuse alpha and opacity-map red are blended.
        // Feature switches disable testing/blending within this authored
        // material mode; disabling cutout testing does not convert it to Blend.
        if (material.AlphaCutoff > 0.0f)
        {
            return false;
        }

        float minimumAlpha = color.W * material.BaseColor.W;
        float minimumVertexAlpha = 1.0f;
        for (const Vertex& vertex : vertices)
        {
            minimumVertexAlpha = std::min(
                minimumVertexAlpha,
                vertex.Color.W);
        }
        minimumAlpha *= minimumVertexAlpha;
        if (material.DiffuseImage &&
            material.DiffuseImage->IsValid())
        {
            const std::vector<std::uint8_t>& pixels =
                material.DiffuseImage->PixelsRGBA8;
            std::uint8_t minimumTextureAlpha = 255;
            for (std::size_t offset = 3; offset < pixels.size(); offset += 4)
            {
                minimumTextureAlpha = std::min(
                    minimumTextureAlpha,
                    pixels[offset]);
            }
            minimumAlpha *=
                static_cast<float>(minimumTextureAlpha) / 255.0f;
        }
        if (material.OpacityImage &&
            material.OpacityImage->IsValid())
        {
            const std::vector<std::uint8_t>& pixels =
                material.OpacityImage->PixelsRGBA8;
            std::uint8_t minimumOpacity = 255;
            for (std::size_t offset = 0; offset < pixels.size(); offset += 4)
            {
                minimumOpacity = std::min(minimumOpacity, pixels[offset]);
            }
            minimumAlpha *=
                static_cast<float>(minimumOpacity) / 255.0f;
        }
        return !std::isfinite(minimumAlpha) ||
            minimumAlpha < 1.0f - 1.0e-6f;
    }
}
