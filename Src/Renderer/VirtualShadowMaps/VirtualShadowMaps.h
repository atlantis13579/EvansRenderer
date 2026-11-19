#pragma once

#include "Renderer/Types.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Techniques/VirtualShadowMaps.h"

namespace Evans::Renderer
{
    inline constexpr unsigned VirtualShadowMaximumMipCount=9,VirtualShadowPageBorder=1;
    inline constexpr unsigned VirtualShadowGroupWidth=64,VirtualShadowGroupColumns=1024,VirtualShadowMaximumRequestRadius=128;
    inline constexpr std::uint32_t VirtualShadowMaximumReceiverPixels=8*1024*1024;
    inline constexpr unsigned VirtualShadowDirectoryWidth=256,VirtualShadowMaximumLookupQueries=65536,VirtualShadowMaximumLookupRadius=8;
    struct VirtualShadowLayout
    {
        std::uint32_t Resolution=0,PageSide=0,MipCount=0,PageCount=0;
        std::uint32_t PhysicalPages=0,AtlasColumns=0,AtlasRows=0,AtlasWidth=0,AtlasHeight=0;
        std::array<std::uint32_t,VirtualShadowMaximumMipCount> MipOffsets{},PagesPerSide{};
        [[nodiscard]] std::uint32_t Root() const noexcept { return PageCount?PageCount-1:0; }
    };
    struct VirtualShadowPage { std::uint32_t Id=0,Mip=0,X=0,Y=0; };
    struct VirtualShadowRectangle { Float2 Minimum,Maximum; };
    struct VirtualShadowSlot { std::uint32_t Page=0,Slot=0; std::uint64_t LastRequested=0; };
    struct VirtualShadowCopy { std::uint32_t Source=0,Destination=0; };
    struct VirtualShadowIdentity
    {
        Matrix4 LightViewProjection;
        std::uint64_t Epoch=1,Serial=1,PreviousSerial=0,CasterRevision=1;
    };
    struct VirtualShadowSnapshot
    {
        VirtualShadowLayout Layout;
        VirtualShadowIdentity Identity;
        // Zero means unmapped. Nonzero entries contain physical slot + one.
        std::vector<std::uint32_t> PageTable;
        std::vector<VirtualShadowSlot> Residents;
    };
    struct VirtualShadowPlan
    {
        VirtualShadowSnapshot Candidate;
        std::vector<VirtualShadowCopy> Copies;
        std::vector<VirtualShadowSlot> Dirty;
        std::uint32_t RequestedPages=0,UnservedPages=0,SourcePhysicalPages=0;
        bool ParentCompatible=false;
    };
    // Matrices use canonical increasing [0,1] hardware depth and top-left UVs.
    // Native depth-image packing must preserve that convention at its boundary.
    struct VirtualShadowRequestFrame
    {
        Matrix4 CameraViewProjection,LightViewProjection;
        std::uint32_t Width=0,Height=0,OutputWidth=0,OutputHeight=0,FilterRadius=1;
        Techniques::VirtualShadowSettings Settings;
    };
    struct alignas(16) VirtualShadowRequestConstants
    {
        Matrix4 CameraInverse,LightViewProjection;
        ShaderUInt4 Extent; // receiver width/height, virtual resolution, page side
        ShaderUInt4 Limits; // mip count, page count, filter radius, dispatch row stride
        Float4 Quality; // mip bias, homogeneous epsilon, page-boundary roundoff, mip roundoff
        Float4 Sampling; // one output pixel in input-pixel XY units, reserved ZW
    };
    static_assert(sizeof(VirtualShadowRequestConstants)==192 && offsetof(VirtualShadowRequestConstants,Extent)==128,"Virtual shadow demand ABI is twelve vectors");
    struct VirtualShadowRequestUpload { VirtualShadowRequestFrame Frame; std::vector<float> Depth; };
    struct VirtualShadowRequestResult { std::vector<std::uint32_t> Pages; };
    struct VirtualShadowCaster
    {
        BufferHandle Vertices,Indices;
        std::uint32_t FirstIndex=0,IndexCount=0;
        Matrix4 World;
        TextureHandle Diffuse,Opacity;
        float Coverage=1,Cutoff=0;
        bool Dithered=false,BackFaceCulling=false;
        Float4 Wind{0,0,0,0}, Gust{0,0,0,0}, VegetationHistory{0,0,0,0};
    };
    // Receiver geometry is rasterized for this camera sample. The resulting
    // depth stays on the GPU through packing and page marking; only the page
    // directory is returned to the residency policy.
    struct VirtualShadowReceiverUpload
    {
        VirtualShadowRequestFrame Frame;
        std::vector<VirtualShadowCaster> Receivers;
    };
    struct VirtualShadowAtlasUpload
    {
        VirtualShadowIdentity Identity;
        Techniques::VirtualShadowSettings Settings;
        std::vector<std::uint32_t> Requests;
        std::vector<VirtualShadowRectangle> Invalidations;
        std::vector<VirtualShadowCaster> Casters;
        bool Readback=true;
    };
    struct VirtualShadowAtlasHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(VirtualShadowAtlasHandle other) const noexcept { return Value==other.Value; }
        bool operator!=(VirtualShadowAtlasHandle other) const noexcept { return Value!=other.Value; }
    };
    struct VirtualShadowAtlasResult
    {
        VirtualShadowAtlasHandle Handle;
        VirtualShadowSnapshot Info;
        std::vector<float> Depth;
        std::uint32_t CopiedPages=0,RenderedPages=0,DrawCalls=0;
    };
    struct VirtualShadowDirectory
    {
        unsigned Width=0,Height=0,Denominator=0;
        // D32 holds (slot + 1) / (physical budget + 1); zero is unmapped.
        // Nearest loads and round restore the bounded integer exactly. This
        // lets mutually exclusive CSM and virtual shadows share depth slots,
        // including Metal depth2d bindings and ES's sixteen-texture budget.
        std::vector<float> Depth;
    };
    struct alignas(16) VirtualShadowLookupConstants
    {
        Matrix4 LightViewProjection;
        ShaderUInt4 VirtualExtent; // resolution, page side, mip count, directory row width
        ShaderUInt4 PhysicalExtent; // atlas width, atlas height, atlas columns, directory denominator
        ShaderUInt4 Filtering; // PCF radius, kernel sample count, query count, dispatch row stride
        Float4 Reserved{0,0,0,0};
    };
    static_assert(sizeof(VirtualShadowLookupConstants)==128,"Virtual shadow lookup ABI is eight vectors");
    struct VirtualShadowLookupUpload
    {
        // Canonical light UV, comparison depth, integer requested mip.
        std::vector<Float4> Samples;
        unsigned Radius=0;
    };
    struct VirtualShadowLookupResult
    {
        // Visibility [0,1], nearest stored depth, resolved mip, physical slot+1.
        // Outside the light volume: (1,1,-1,0).
        std::vector<Float4> Samples;
    };
    struct alignas(16) VirtualShadowRasterConstants
    {
        Matrix4 WorldViewProjection;
        Float4 Material; // scalar coverage, cutoff, stochastic coverage, reserved
        ShaderUInt4 Textures; // diffuse/opacity presence, reserved ZW
        Float4 Origins; // destination tile XY, logical mip-space origin XY (copy: source tile XY)
        ShaderUInt4 Extent; // atlas width/height; copy source width/height or readback row stride
        Matrix4 VegetationWorld, VegetationNormalWorld;
        Float4 Wind{0,0,0,0}, Gust{0,0,0,0}, VegetationHistory{0,0,0,0};
    };
    static_assert(sizeof(VirtualShadowRasterConstants)==304,"Virtual shadow raster ABI includes vegetation deformation");

    [[nodiscard]] bool BuildVirtualShadowLayout(const Techniques::VirtualShadowSettings&,VirtualShadowLayout&,std::string&);
    [[nodiscard]] bool DecodeVirtualShadowPage(const VirtualShadowLayout&,std::uint32_t,VirtualShadowPage&) noexcept;
    [[nodiscard]] std::uint32_t VirtualShadowParent(const VirtualShadowLayout&,const VirtualShadowPage&) noexcept;
    [[nodiscard]] bool BuildVirtualShadowRequestConstants(const VirtualShadowRequestFrame&,VirtualShadowRequestConstants&,std::string&);
    [[nodiscard]] bool ValidateVirtualShadowRequestUpload(const VirtualShadowRequestUpload&,std::string&);
    [[nodiscard]] bool ValidateVirtualShadowRequests(const VirtualShadowLayout&,const std::vector<std::uint32_t>&,std::string&);
    // The candidate owns a distinct atlas. Copies retain valid parent tiles;
    // Dirty tiles may reuse slot numbers without modifying the parent resource.
    // Publish only after every Dirty tile and the complete presentation succeed.
    [[nodiscard]] bool BuildVirtualShadowPlan(const VirtualShadowIdentity&,const Techniques::VirtualShadowSettings&,
        const std::vector<std::uint32_t>& requests,const std::vector<VirtualShadowRectangle>& invalidations,
        const VirtualShadowSnapshot* previous,VirtualShadowPlan&,std::string&);
    [[nodiscard]] bool ValidateVirtualShadowSnapshot(const VirtualShadowSnapshot&,std::string&);
    [[nodiscard]] bool CompleteVirtualShadowPlan(const VirtualShadowPlan&,const std::vector<std::uint32_t>& renderedPages,VirtualShadowSnapshot&,std::string&);
    [[nodiscard]] bool FindVirtualShadowPage(const VirtualShadowSnapshot&,Float2 uv,std::uint32_t requestedMip,
        VirtualShadowPage&,std::uint32_t& physicalSlot) noexcept;
    [[nodiscard]] Float2 VirtualShadowAtlasUV(const VirtualShadowLayout&,const VirtualShadowPage&,std::uint32_t slot,Float2 uv) noexcept;
    [[nodiscard]] Matrix4 VirtualShadowPageProjection(const VirtualShadowLayout&,const VirtualShadowPage&,const Matrix4&) noexcept;
    [[nodiscard]] bool ValidateVirtualShadowCaster(const VirtualShadowCaster&,const BufferDesc&,const BufferDesc&,
        const std::vector<std::uint8_t>& indexMirror,bool naniteGeometry,std::string&);
    [[nodiscard]] bool BuildVirtualShadowRasterConstants(const VirtualShadowSnapshot&,const VirtualShadowSlot&,const VirtualShadowCaster&,
        VirtualShadowRasterConstants&,std::string&);
    [[nodiscard]] bool BuildVirtualShadowReceiverConstants(const VirtualShadowRequestFrame&,const VirtualShadowCaster&,
        VirtualShadowRasterConstants&,std::string&);
    [[nodiscard]] VirtualShadowRasterConstants BuildVirtualShadowCopyConstants(const VirtualShadowLayout& source,
        const VirtualShadowLayout& destination,const VirtualShadowCopy&) noexcept;
    [[nodiscard]] bool ValidateVirtualShadowAtlasDepth(const VirtualShadowSnapshot&,const std::vector<float>&,std::string&);
    [[nodiscard]] bool BuildVirtualShadowDirectory(const VirtualShadowSnapshot&,VirtualShadowDirectory&,std::string&);
    [[nodiscard]] bool BuildVirtualShadowLookupConstants(const VirtualShadowSnapshot&,const VirtualShadowLookupUpload&,VirtualShadowLookupConstants&,std::string&);
    [[nodiscard]] bool ValidateVirtualShadowLookupResult(const VirtualShadowSnapshot&,const VirtualShadowLookupUpload&,const VirtualShadowLookupResult&,std::string&);
    [[nodiscard]] VirtualShadowAtlasHandle NewVirtualShadowAtlasHandle() noexcept;
}
