#pragma once

#include "Renderer/Pipelines/VisibilityBuffer.h"

namespace Evans::Renderer
{
    // Small triangles use half-space tests, one compute lane per triangle.
    // Karis, Stubbe and Wihlidal, "A Deep Dive into Nanite Virtualized Geometry",
    // SIGGRAPH 2021, slides 83--86. The published 64-bit depth/payload atomic is
    // replaced by separate depth and ownership epochs for portable 32-bit atomics.
    // Hardware winners seed the depth epoch and win exact ties; compute ties use
    // the lowest triangle reference. A barrier separates every epoch.
    inline constexpr unsigned NaniteRasterThreads=64, NaniteRasterDispatchColumns=64;
    inline constexpr unsigned NaniteRasterMaximumSide=8192, NaniteRasterMaximumTriangles=1024*1024;
    inline constexpr std::size_t NaniteRasterMaximumBytes=512ull*1024*1024;
    inline constexpr unsigned NaniteRasterCullBackFace=1, NaniteRasterFrontClockwise=2, NaniteRasterForceHardware=4;
    // Frame ownership keys reserve the high bit for compute coverage, keeping
    // hardware priority when equal-depth winners cross a phase boundary.
    inline constexpr unsigned NaniteRasterPriorityOrdering=1,NaniteRasterPreserveMain=2;
    enum class NaniteRasterRoute : unsigned { Rejected=0, Compute=1, Hardware=2 };
    struct alignas(16) NaniteRasterTriangle
    {
        std::array<Float4,3> Clip;
        ShaderUInt4 Flags; // Cull/front-winding/force-hardware flags, reserved, reserved, reserved.
    };
    static_assert(sizeof(NaniteRasterTriangle)==64,"Nanite raster input ABI changed");
    struct NaniteRasterConstants
    {
        ShaderUInt4 Counts;   // Width, height, triangle count, pixel count.
        ShaderUInt4 Layout;   // Input depth/reference/triangle offsets, output route offset.
        ShaderUInt4 Dispatch; // Group columns, invocation count, phase, reserved.
        ShaderUInt4 Quality;  // Maximum software extent, subpixel bits, accepted cull state, phase flags.
    };
    static_assert(sizeof(NaniteRasterConstants)==64,"Nanite raster constants ABI changed");
    struct NaniteFrameRasterConstants
    {
        NaniteRasterConstants Raster;
        Matrix4 ViewProjection;
        ShaderUInt4 Range; // First triangle, count, cull/winding/material flags, cull query + 1 (zero disables).
    };
    static_assert(sizeof(NaniteFrameRasterConstants)==144,"Nanite frame input ABI changed");
    struct NaniteFrameRasterDispatch
    {
        NaniteFrameRasterConstants Constants;
        unsigned Pass=0,GroupsX=0,GroupsY=0;
    };
    struct NaniteFrameRasterWork
    {
        unsigned InputWords=0,OutputWords=0,HardwareBoundary=0;
        std::vector<NaniteFrameRasterDispatch> Dispatches;
    };
    // Clip/viewport crossings and large triangles go to hardware. The compute
    // route snaps XY to the specified subpixel grid, uses the top-left rule and
    // interpolates Z/W linearly. Hardware integration must use the same snapped
    // positions for inside corners, including shared edges of clipped triangles. No CPU raster fallback.
    // No triangle data or depth is read back to plan these dispatches. Capture
    // and projection populate the ordinary raster input ABI directly on the GPU.
    [[nodiscard]] bool BuildNaniteFrameRasterWork(const VisibilityBufferDesc&,unsigned width,unsigned height,
        NaniteFrameRasterWork&,std::string&);
    struct NaniteRasterDrawConstants
    {
        NaniteRasterConstants Raster;
        ShaderUInt4 Range; // First triangle, count, reserved, reserved.
        Float4 Material; // Opacity, alpha cutoff, reserved, reserved.
        ShaderUInt4 Images; // Diffuse and opacity map presence, reserved, reserved.
    };
    static_assert(sizeof(NaniteRasterDrawConstants)==112,"Nanite raster draw ABI changed");
    [[nodiscard]] bool BuildNaniteRasterDrawConstants(const VisibilityBufferDesc&,const VisibilityMaterialDraw*,
        unsigned width,unsigned height,NaniteRasterDrawConstants&);
    struct NaniteVisibilityGeometryConstants
    {
        Matrix4 World,Normal;
        ShaderUInt4 Range; // First output triangle, triangle count, first index, vertex count.
        ShaderUInt4 Dispatch; // Group columns, invocation count, unique material/draw identity, reserved.
    };
    static_assert(sizeof(NaniteVisibilityGeometryConstants)==160 && sizeof(Vertex)==48,"Native visibility assembly ABI changed");
    // Build only metadata. Native kernels read the decoded vertex/index buffers
    // and produce the world-space table consumed by hybrid coverage and shading.
    [[nodiscard]] bool BuildNaniteVisibilityScene(const std::vector<VisibilityNativeMeshView>&,VisibilityScene&,std::string&);
    [[nodiscard]] bool BuildNaniteVisibilityGeometryConstants(const VisibilityScene&,unsigned source,
        const BufferDesc& vertices,const BufferDesc& indices,NaniteVisibilityGeometryConstants&,unsigned& groupsX,unsigned& groupsY,std::string&);
}
