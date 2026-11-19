#pragma once

#include "Renderer/Nanite/NaniteFrameVisibility.h"

namespace Evans::Renderer
{
    struct NaniteDrawHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const { return Value!=0; }
    };
    struct NaniteClusterDraw
    {
        BufferHandle VertexBuffer,IndexBuffer;
        unsigned FirstIndex=0,IndexCount=0;
        Matrix4 World,ViewProjection,LightViewProjection;
        PipelineState Pipeline;
    };
    inline constexpr unsigned NaniteDrawArgumentWords=5,NaniteDrawRangeWords=2,NaniteMaximumDrawBatches=8,NaniteDrawDispatchColumns=64;
    inline constexpr std::size_t NaniteMaximumDrawBytes=128ull*1024*1024;
    inline constexpr unsigned NaniteBinCandidateWords=4,NaniteBinBlockWords=4,NaniteBinDescriptorWords=8;
    struct NaniteDrawBin
    {
        unsigned FirstDraw=0,DrawCount=0,FirstIndex=0,IndexCount=0;
        unsigned OutputIndex=0,FirstBlock=0,BlockCount=0;
    };
    struct NaniteBinLayout
    {
        std::vector<NaniteDrawBin> Bins;
        std::vector<unsigned> Metadata;
        // Counts: candidates/blocks/bins/phase. Layout: metadata offsets.
        // Source: local prefix/block total/block prefix/arguments.
        // Target: packed indices/capacity/dispatch jobs/dispatch columns.
        NaniteVisibilityConstants Constants;
        unsigned OutputWords=0,SourceIndexWords=0;
        unsigned AlignmentPadding[2]{};
    };

    // Hardware-raster branch of Karis et al., "A Deep Dive into Nanite
    // Virtualized Geometry" (SIGGRAPH 2021). A GPU classification gates each
    // indexed command; zero instance count rejects without a CPU readback.
    // Decoded indices are absolute, so base vertex and first instance are zero.
    // The single-cluster path retains stable diagnostic commands. The binned
    // path scans visibility in 64-cluster blocks, scans block totals per bin,
    // then scatters visible indices to one indirect range per compatible bin.
    // Ordinary material bindings preserve alpha testing and existing shading.
    // Only consecutive compatible draws merge: grouping nonadjacent materials
    // would reorder coplanar depth ties. No visible count returns to the CPU.
    [[nodiscard]] bool BuildNaniteDrawBins(const std::vector<NaniteClusterDraw>&,NaniteBinLayout&,std::string&);
    bool NaniteDrawPipelineSupported(const PipelineState&);
    bool SameNaniteDraw(const NaniteClusterDraw&,const NaniteClusterDraw&);
    [[nodiscard]] bool ValidateNaniteClusterDraw(const NaniteClusterDraw&,const NaniteVisibilityQuery&,
        const BufferDesc&,const std::vector<std::uint8_t>& vertices,
        const BufferDesc&,const std::vector<std::uint8_t>& indices,std::string&);
    // Queries used for ordinary material draws have WORLD-space bounds and
    // uncomposed view-projection matrices. Union current/previous bounds and
    // include world-transform rounding before the existing clip error test.
    [[nodiscard]] bool BuildNaniteDrawQuery(Float3 minimum,Float3 maximum,
        const Matrix4& world,const Matrix4& previousWorld,const Matrix4& viewProjection,
        const Matrix4& previousViewProjection,bool historyValid,unsigned cluster,unsigned instance,
        NaniteVisibilityQuery&,std::string&);
}
