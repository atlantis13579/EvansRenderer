#pragma once
#include "Renderer/Geometry/MeshGeometry.h"

#include "Renderer/Transparency/DepthPeeling.h"
#include "Renderer/FrameHistory.h"
#include "Renderer/RealtimeShaderContract.h"
#include <cstring>

namespace Evans::Renderer
{
    enum class DepthPeelingPass { Capture, Accumulate, Compose };
    struct DepthPeelingDraw
    {
        BufferHandle VertexBuffer, IndexBuffer;
        std::uint32_t FirstIndex = 0, IndexCount = 0;
        std::int32_t BaseVertex = 0;
        Matrix4 World;
        TextureHandle DiffuseTexture, OpacityTexture;
        float Opacity = 1, AlphaCutoff = 0;
        bool BackFaceCulling = false;
        std::uint64_t SourceKey = 0;
    };
    struct DepthPeelingFrame
    {
        bool Enabled = false;
        Matrix4 ViewProjection;
        Techniques::OrderIndependentTransparencySettings Settings;
        std::vector<DepthPeelingDraw> Draws;
    };
    // Capture and material replay consume the same immutable frame geometry.
    // Increasing source keys define coplanar blending order independently of
    // submission order and of unordered scene-storage iteration.
    inline bool ValidateDepthPeelingFrame(const DepthPeelingFrame& frame,std::string& error)
    {
        error.clear();
        if (!frame.Enabled) return true;
        if (!FrameHistory::IsFinite(frame.ViewProjection) ||
            frame.Settings.LayerCount < Techniques::MinimumPeelingLayers ||
            frame.Settings.LayerCount > Techniques::MaximumPeelingLayers)
        { error="Invalid depth-peeling projection or layer budget"; return false; }
        std::uint64_t previous=0; bool first=true;
        for (const auto& draw:frame.Draws)
        {
            if (!draw.VertexBuffer || !draw.IndexBuffer || draw.IndexCount%3 ||
                !FrameHistory::IsFinite(draw.World) || !std::isfinite(draw.Opacity) || !std::isfinite(draw.AlphaCutoff) ||
                draw.Opacity<0 || draw.Opacity>1 || draw.AlphaCutoff<0 || draw.AlphaCutoff>1 ||
                (!first && draw.SourceKey<=previous))
            { error="Depth-peeling draws require valid geometry, materials and increasing unique source keys"; return false; }
            first=false; previous=draw.SourceKey;
        }
        return true;
    }
    inline bool ValidateDepthPeelingBuffers(const DepthPeelingDraw& draw,const BufferDesc& vertex,
        const BufferDesc& index,std::size_t vertexBytes,const std::vector<std::uint8_t>& indices,std::string& error,bool nativeDecoded=false)
    {
        if (!nativeDecoded && vertexBytes!=vertex.SizeInBytes)
        { error="Depth-peeling capture has no complete host vertex mirror"; return false; }
        return ValidateIndexedGeometryDraw(vertex,index,indices,nativeDecoded,draw.FirstIndex,draw.IndexCount,draw.BaseVertex,error);
    }
    inline bool ValidateDepthPeelingPass(const PassDesc& pass,const DepthPeelingFrame& frame,
        const DepthPeelingSequence& sequence,std::string& error)
    {
        if (pass.DepthPeelingLayer)
        {
            if (!frame.Enabled || sequence.Stage()!=DepthPeelingStage::NeedShading ||
                pass.Type!=PassType::Color || pass.Target || pass.ClearColorEnabled || pass.ClearDepthEnabled)
            { error="OIT lighting requires the captured private layer and preserves opaque attachments"; return false; }
        }
        else if (frame.Enabled && sequence.Stage()!=DepthPeelingStage::Complete &&
            !(sequence.Stage()==DepthPeelingStage::NeedCapture && sequence.Step().FirstLayer))
        { error="Complete the active depth-peeling layer before starting another raster pass"; return false; }
        return true;
    }
    inline bool ValidateDepthPeelingPipeline(const PipelineState& pipeline,std::string& error)
    {
        if (pipeline.Fill!=FillMode::Solid || !pipeline.DepthTest || pipeline.DepthWrite ||
            !pipeline.ColorWrite || !pipeline.BlendEnabled || pipeline.OutlineExtrusion)
        { error="OIT lighting requires solid depth-EQUAL replay, depth writes off and source-over blending"; return false; }
        return true;
    }
    inline void RecordDepthPeelingPass(const DepthPeelingFrame& frame,DepthPeelingPass pass,FrameStatistics& statistics)
    {
        ++statistics.DepthPeelingPassCount;
        if (pass!=DepthPeelingPass::Capture) return;
        for (const auto& draw:frame.Draws) if (draw.IndexCount)
        {
            ++statistics.DepthPeelingDrawCount; ++statistics.DrawCallCount;
            statistics.SubmittedTriangleCount+=draw.IndexCount/3;
        }
    }

    // Independent OIT shaders own depth capture, accumulation and final resolve.
    // Material lighting is replayed by the selected Forward shader against the
    // captured D32 depth using EQUAL, depth writes off, and source-over RGBA.
    // Capture/replay must use identical clip positions and raster coverage. A
    // depth epsilon is not a substitute: it merges adjacent representable layers.
    // Ordinary material and OIT vertex stages preserve position arithmetic with
    // precise/invariant outputs. Replay must use the same matrices, viewport,
    // culling, sample state and original geometry, without outline extrusion.
    // Instanced producers are outside this contract; use ordinary draws for OIT.
    // Slots: opaque depth, previous depth, diffuse, opacity, shaded premultiplied
    // layer, previous accumulation, opaque HDR, current capture/probe depth.
    // Accumulation images clear to (0,0,0,1), layer color to transparent black,
    // and both depth images to 1. No sampled image is an active write target.
    struct DepthPeelingConstants
    {
        Matrix4 World, ViewProjection;
        Float4 Material; // Combined scalar opacity, alpha cutoff, reserved.
        Float4 Viewport; // Extent XY, reciprocal ZW.
        ShaderUInt4 Options; // First layer, diffuse map, opacity map, show overflow.
    };
    static_assert(sizeof(DepthPeelingConstants) == 176 && offsetof(DepthPeelingConstants,Options) == 160,
        "Depth-peeling uniform layout changed");

    inline bool BuildDepthPeelingConstants(const Matrix4& projection,int width,int height,
        bool firstLayer,bool showOverflow,DepthPeelingConstants& output)
    {
        output = {};
        if (!FrameHistory::IsFinite(projection) || width <= 0 || height <= 0) return false;
        output.ViewProjection = projection; output.Material = {1,0,0,0};
        output.Viewport = {float(width),float(height),1.0f/width,1.0f/height};
        output.Options = {firstLayer ? 1u : 0u,0,0,showOverflow ? 1u : 0u}; return true;
    }

    inline bool BuildDepthPeelingDrawConstants(const DepthPeelingConstants& base,const Matrix4& world,
        float opacity,float cutoff,bool diffuseMap,bool opacityMap,DepthPeelingConstants& output)
    {
        DepthPeelingConstants value = base; output = {};
        if (!FrameHistory::IsFinite(world) || !std::isfinite(opacity) || !std::isfinite(cutoff)) return false;
        value.World = world; value.Material = {std::clamp(opacity,0.0f,1.0f),std::clamp(cutoff,0.0f,1.0f),0,0};
        value.Options.Y = diffuseMap ? 1 : 0; value.Options.Z = opacityMap ? 1 : 0; output = value; return true;
    }
}
