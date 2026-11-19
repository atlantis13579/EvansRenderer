#pragma once

#include "Renderer/Lumen/LumenMeshCards.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenSurfacePlanes=3,LumenSurfaceBorder=1,LumenSurfaceReadThreads=64,LumenSurfaceReadColumns=1024;
    inline constexpr unsigned LumenSurfaceMaximumCards=4096,LumenSurfaceMaximumSide=512,LumenSurfaceMaximumAtlasSide=16384;
    inline constexpr unsigned LumenSurfaceMaximumDraws=65536;
    inline constexpr std::size_t LumenSurfaceMaximumTexels=4*1024*1024;
    struct LumenSurfaceIdentity { std::uint64_t Epoch=1,Serial=1,Geometry=1,Material=1; };
    struct LumenSurfaceLayout { unsigned CardCount=0,CardSide=0,Columns=0,Rows=0,Width=0,Height=0; };
    struct LumenSurfaceDraw
    {
        BufferHandle Vertices,Indices;
        unsigned FirstIndex=0,IndexCount=0;
        Matrix4 World;
        Float4 BaseColor{1,1,1,1}; // Linear reflectance and material coverage.
        Float3 Emission;
        float Roughness=0.5f,Metallic=0,Cutoff=0;
        TextureHandle Diffuse,Normal,Opacity;
        bool BackFaceCulling=false;
    };
    struct LumenSurfaceCapture
    {
        Matrix4 ViewProjection; // Acts on capture-relative world coordinates; canonical [0,1] depth.
        unsigned FirstDraw=0,DrawCount=0;
        Float3 ViewOrigin; // Removed before projection to preserve thin layers at large coordinates.
        unsigned ProjectionGroup=0; // One of the six local mesh directions; layers share a group.
    };
    struct LumenSurfaceUpload
    {
        LumenSurfaceIdentity Identity;
        unsigned CardSide=64;
        std::vector<LumenSurfaceCapture> Cards;
        std::vector<LumenSurfaceDraw> Draws;
        bool Readback=false;
    };
    struct LumenSurfaceHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(LumenSurfaceHandle other) const noexcept { return Value==other.Value; }
    };
    struct LumenCaptureTask
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
    };
    struct LumenCaptureStatistics
    {
        unsigned TotalCards=0,CompletedCards=0,LastSubmittedCards=0;
        std::uint64_t DrawCalls=0;
    };
    struct LumenSurfaceTexel
    {
        // A zero normal marks uncovered texels, including legitimate black materials.
        Float4 AlbedoMetallic,NormalDepth,EmissionRoughness;
    };
    static_assert(sizeof(LumenSurfaceTexel)==LumenSurfacePlanes*sizeof(Float4),"Lumen material readback uses three contiguous vectors");
    struct LumenSurfaceResult
    {
        LumenSurfaceHandle Handle;
        LumenSurfaceIdentity Identity;
        LumenSurfaceLayout Layout;
        std::vector<LumenSurfaceTexel> Texels;
        std::uint64_t DrawCalls=0;
    };
    struct alignas(16) LumenSurfaceConstants
    {
        Matrix4 RelativeWorld,NormalWorld,ViewProjection;
        Float4 BaseColor,EmissionRoughness,Surface; // Surface: metallic, cutoff, reserved ZW.
        ShaderUInt4 Textures; // Diffuse, normal, opacity presence; reserved W.
    };
    static_assert(sizeof(LumenSurfaceConstants)==256,"Lumen material capture ABI is sixteen vectors");

    // Closed-frame immutable material capture. Surface cache relighting must
    // retain this exact geometry/material version; it must not sample a mutable
    // camera G-buffer or replace a presented cache with an incomplete upload.
    // Asynchronous capture freezes draw parameters and borrows native inputs.
    // Input mutation/deletion cancels affected jobs and retires submitted reads.
    bool BuildLumenSurfaceLayout(unsigned cardCount,unsigned cardSide,LumenSurfaceLayout& output,std::string& error);
    LumenSurfaceCapture LumenCardSurfaceCapture(const LumenMeshCard& card,unsigned firstDraw,unsigned drawCount) noexcept;
    bool ValidateLumenSurfaceUpload(const LumenSurfaceUpload& upload,LumenSurfaceLayout& layout,std::string& error);
    bool ValidateLumenSurfaceDraw(const LumenSurfaceDraw& draw,const BufferDesc& vertices,const BufferDesc& indices,
        const std::vector<std::uint8_t>& indexMirror,std::string& error);
    bool BuildLumenSurfaceConstants(const LumenSurfaceLayout& layout,const LumenSurfaceCapture& capture,
        const LumenSurfaceDraw& draw,LumenSurfaceConstants& output,std::string& error);
    bool ValidateLumenSurfaceTexels(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceTexel>& texels,std::string& error);
    LumenSurfaceHandle NewLumenSurfaceHandle() noexcept;
}
