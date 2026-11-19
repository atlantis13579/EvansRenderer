#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Nanite/NaniteFrameVisibility.h"

namespace Evans::Renderer
{
    // A visibility sample stores a triangle reference and two perspective-correct
    // barycentrics. Material reconstruction fetches attributes only after depth
    // and cutout visibility have been decided. Zero is the cleared background;
    // the high bit records the rasterizer's back-facing result.
    // Burns and Hunt, "The Visibility Buffer: A Cache-Friendly Approach to
    // Deferred Shading", JCGT 2(2), 2013: https://jcgt.org/published/0002/02/04/
    inline constexpr std::uint32_t VisibilityBackFace = 0x80000000u;
    inline constexpr std::uint32_t VisibilityTriangleMask = ~VisibilityBackFace;
    inline constexpr std::size_t VisibilityTriangleCapacity = VisibilityTriangleMask;
    inline constexpr std::size_t VisibilityTriangleCorners = 3;

    [[nodiscard]] inline std::uint32_t EncodeVisibility(std::uint32_t triangle, bool backFace)
    { return triangle < VisibilityTriangleCapacity ? (triangle + 1) | (backFace ? VisibilityBackFace : 0u) : 0u; }
    [[nodiscard]] inline bool DecodeVisibility(std::uint32_t value, std::uint32_t& triangle, bool& backFace)
    {
        const auto reference = value & VisibilityTriangleMask;
        triangle = reference ? reference - 1 : 0; backFace = reference && (value & VisibilityBackFace);
        return reference != 0;
    }

    // World-space table, independent of any ray-query/BVH ordering. Positions
    // have homogeneous W=1. Corner normals match the native vertex stage's unit
    // inverse-transpose normals; RGB is decoded before interpolation. UVs retain
    // authored OBJ coordinates: each shader flips V exactly once.
    struct alignas(16) VisibilityTriangle
    {
        std::array<Float4, 3> Position, Normal, Color;
        Float4 UVAB, UVC;
        ShaderUInt4 Source; // Material/mesh index, source triangle, reserved, reserved.
    };
    static_assert(sizeof(VisibilityTriangle) == 192 && offsetof(VisibilityTriangle, UVAB) == 144 &&
        offsetof(VisibilityTriangle, Source) == 176, "Visibility triangle storage ABI changed");
    // DX12 byte-address loads use 32-bit byte offsets. Native resource limits
    // can be smaller; reject the common addressing overflow before allocation.
    inline constexpr std::size_t VisibilityStorageCapacity = UINT32_MAX / sizeof(VisibilityTriangle);

    struct VisibilityMeshView
    {
        const std::vector<Vertex>* Vertices = nullptr;
        const std::vector<std::uint32_t>* Indices = nullptr;
        Matrix4 World;
        bool Visible = true;
        VisibilityMeshView() = default;
        explicit VisibilityMeshView(const MeshData& mesh)
            : Vertices(&mesh.Vertices), Indices(&mesh.Indices), World(mesh.WorldTransform), Visible(mesh.Visible) {}
    };
    struct VisibilityDrawRange
    {
        // Non-indexed raster draw: vertex count = TriangleCount * 3, first vertex
        // = 0. The shader adds FirstTriangle to its local vertex-ID/3 reference.
        std::uint32_t FirstTriangle = 0, TriangleCount = 0, Material = 0;
        bool Mirrored = false; // Native front-face state must match ordinary raster submission.
    };
    // Native decoded streams remain in the RHI. The scene snapshot carries
    // only indexed ranges and affine instance transforms, never vertex mirrors.
    struct VisibilityNativeMeshView
    {
        BufferHandle VertexBuffer,IndexBuffer;
        unsigned FirstIndex=0,IndexCount=0;
        Matrix4 World;
        bool Visible=true;
    };
    struct VisibilityScene
    {
        std::vector<VisibilityTriangle> Triangles;
        std::vector<VisibilityDrawRange> Draws;
        std::vector<VisibilityNativeMeshView> NativeMeshes;
        unsigned NativeTriangleCount=0;
        [[nodiscard]] bool HasNativeGeometry() const { return !NativeMeshes.empty(); }
        [[nodiscard]] std::size_t TriangleCount() const { return HasNativeGeometry()?NativeTriangleCount:Triangles.size(); }
    };
    struct VisibilityMaterialDraw
    {
        std::uint32_t Range = 0;
        PipelineState Pipeline;
        // Source handles/world are retained for frame inspection. Native raster
        // stages pull the immutable world-space table instead of these buffers.
        BufferHandle VertexBuffer, IndexBuffer;
        Matrix4 World;
        std::uint32_t FirstIndex=0;
    };
    // Ordered, complete partitions of the submitted material ranges. Entry i
    // consumes GPU culling decision i; triangle references remain stable in both phases.
    struct NaniteRasterClusterRange
    { unsigned Range=0,FirstTriangle=0,TriangleCount=0; };
    struct VisibilityBufferDesc
    {
        std::shared_ptr<const VisibilityScene> Scene;
        std::vector<VisibilityMaterialDraw> Draws; // Ordered subset; CPU culling may omit ranges.
        Matrix4 ViewProjection;
        Float4 Background = {0.02f, 0.02f, 0.025f, 1};
        // GPU projection, hybrid coverage and depth/visibility export precede material resolve.
        bool NaniteRaster=false;
        bool MeshShaders=false; // Native mesh stage; Nanite requires GPU hardware bins.
        Techniques::MeshShaderDispatchMode MeshDispatch=Techniques::MeshShaderDispatchMode::Automatic;
        bool GpuHardware=false; // GPU raster bins and order-independent hardware ownership.
        bool GpuInstances=false; // GPU count/scan and indirect assembly of shared decoded streams.
        bool MaterialTiles=false; // GPU tile classification and indirect material resolve.
        NaniteVisibilityHandle NaniteVisibility;
        std::vector<NaniteRasterClusterRange> NaniteClusters;
        unsigned MaximumSoftwareExtent=8,SubpixelBits=8;
    };
    [[nodiscard]] bool ValidateVisibilityScene(const VisibilityScene& scene, std::string& error);
    [[nodiscard]] bool ValidateVisibilityFrame(const VisibilityBufferDesc& frame, int width, int height, std::string& error);
    void RecordVisibilityDraws(FrameCapture& capture, const VisibilityBufferDesc& frame);
    // Borrow inputs only during construction. No culling, sorting, BVH reorder,
    // texture generation or geometry welding occurs here. Failure preserves output.
    [[nodiscard]] bool BuildVisibilityScene(const std::vector<VisibilityMeshView>& meshes,
        VisibilityScene& output, std::string& error);

    struct VisibilityConstants
    {
        Matrix4 ViewProjection;
        Float4 Viewport; // Width, height, reciprocal width, reciprocal height.
        Float4 PipelineColor, Diffuse, Ambient, Specular, Emission;
        Float4 Material; // Shininess, opacity, alpha cutoff, unlit wire style.
        Float4 AmbientLighting, Shading;
        ShaderUInt4 Options; // Diffuse, normal, opacity image presence; shading model.
        ShaderUInt4 Draw; // Total triangles, first triangle, draw count, material/mesh index.
        ShaderUInt4 MaterialTiles; // Bin, tile columns, tile width, enabled.
    };
    static_assert(sizeof(VisibilityConstants) == 256 && offsetof(VisibilityConstants, Options) == 208 &&
        offsetof(VisibilityConstants, Draw) == 224, "Visibility uniform ABI changed");
    // Per-pixel opaque/cutout materials and unlit wire overlays. Gouraud, ordered
    // blending and selection outlines require their separate submission paths.
    [[nodiscard]] bool BuildVisibilityConstants(const Matrix4& camera, const PipelineState& pipeline,
        int width, int height, std::size_t triangleCount, const VisibilityDrawRange& range,
        bool diffuseImage, bool normalImage, bool opacityImage, VisibilityConstants& output);

}
