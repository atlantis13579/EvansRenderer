#pragma once
#include "Renderer/Transparency/RefractionGeometry.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/FrameHistory.h"
#include "Maths/Matrix4.h"
#include "Techniques/Refraction.h"

namespace Evans::Renderer
{
    struct RefractionDesc
    {
        Matrix4 ViewProjection;
        Float3 Environment;
    };
    struct RefractionDraw
    {
        std::uint32_t SourceIndex = 0, LodLevel = 0;
        Matrix4 World;
        Float4 Color = {1,1,1,1}; // Linear instance * material color.
        TextureHandle DiffuseTexture, NormalTexture, OpacityTexture;
        float AlphaCutoff = 0;
        bool ForceCoverage = false, BackFaceCulling = true;
    };
    struct RefractionFrame
    {
        std::shared_ptr<const RefractionGeometry> Geometry;
        std::vector<RefractionDraw> Draws;
    };
    inline void RecordRefractionDraws(const RefractionFrame& frame,FrameStatistics& statistics)
    {
        statistics.RefractionPassCount = 2;
        for (const auto& draw:frame.Draws)
            if (frame.Geometry->Sources[draw.SourceIndex][draw.LodLevel].IndexCount) ++statistics.RefractionDrawCount;
        statistics.DrawCallCount += statistics.RefractionDrawCount;
    }

    // The capture shader writes RGBA16F world normals and tint/coverage against
    // private D32 nearest-layer depth, rejecting fragments behind opaque depth.
    // Composition reads a frozen HDR background, never its active output target.
    // Texture slots: opaque depth, diffuse, normal, opacity, frozen scene color,
    // captured depth, captured normal, captured tint/coverage. Capture only binds
    // the first four; composition never binds its private depth as an attachment.
    // Macro EVANS_REFRACTION_CAPTURE selects geometry stages in each native file.
    struct RefractionConstants
    {
        Matrix4 World, NormalWorld, ViewProjection, InverseViewProjection;
        Float4 Color, Material; // Material: alpha cutoff, forced coverage, reserved.
        Float4 Optics; // Exterior IOR, slab IOR, perpendicular thickness, tint distance.
        Float4 Absorption, Viewport; // RGB inverse world units; extent XY, reciprocal ZW.
        Float4 Trace; // Maximum distance, depth-slab thickness, surface bias, pixel stride.
        Float4 Appearance, Environment; // Edge fade pixels; uniform miss radiance RGB.
        ShaderUInt4 Textures; // Diffuse, normal, opacity, material tint.
        ShaderUInt4 Options; // Maximum steps, reserved.
    };
    static_assert(sizeof(RefractionConstants) == 416 && offsetof(RefractionConstants,Textures) == 384 &&
        offsetof(RefractionConstants,Options) == 400,"Refraction native uniform ABI changed");

    inline bool BuildRefractionConstants(const Matrix4& viewProjection,Float3 missRadiance,
        Techniques::RefractionSettings requested,int width,int height,RefractionConstants& output)
    {
        output = {};
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(viewProjection) ||
            !std::isfinite(missRadiance.X) || !std::isfinite(missRadiance.Y) || !std::isfinite(missRadiance.Z)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(viewProjection.Values).Inverse(inverse)) return false;
        RefractionConstants constants;
        constants.ViewProjection = viewProjection;
        std::copy(inverse.Data(),inverse.Data()+16,constants.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(constants.InverseViewProjection)) return false;
        const auto s = Techniques::SanitizeRefractionSettings(requested);
        constants.Color = {1,1,1,1}; constants.Material = {0,0,0,0};
        constants.Optics = {s.ExteriorIOR,s.MaterialIOR,s.Thickness,s.TintDistance};
        constants.Absorption = {s.Absorption[0],s.Absorption[1],s.Absorption[2],0};
        constants.Viewport = {float(width),float(height),1.0f/width,1.0f/height};
        constants.Trace = {s.MaximumDistance,s.DepthThickness,s.SurfaceBias,s.StridePixels};
        constants.Appearance = {s.EdgeFadePixels,0,0,0};
        constants.Environment = {std::max(missRadiance.X,0.0f),std::max(missRadiance.Y,0.0f),std::max(missRadiance.Z,0.0f),0};
        constants.Textures.W = s.MaterialTint ? 1 : 0; constants.Options.X = s.MaximumSteps;
        output = constants; return true;
    }

    inline bool BuildRefractionDrawConstants(const RefractionFrame& frame,const RefractionDraw& draw,
        const RefractionConstants& composition,RefractionConstants& output,RefractionGeometryRange& range)
    {
        RefractionConstants constants = composition;
        output = {}; range = {};
        if (!frame.Geometry || draw.SourceIndex >= frame.Geometry->Sources.size() ||
            draw.LodLevel >= frame.Geometry->Sources[draw.SourceIndex].size() || !FrameHistory::IsFinite(draw.World) ||
            !std::isfinite(draw.Color.X) || !std::isfinite(draw.Color.Y) || !std::isfinite(draw.Color.Z) ||
            !std::isfinite(draw.Color.W) || !std::isfinite(draw.AlphaCutoff)) return false;
        const auto candidate = frame.Geometry->Sources[draw.SourceIndex][draw.LodLevel];
        if (candidate.FirstIndex > frame.Geometry->Indices.size() || candidate.IndexCount%3 ||
            candidate.IndexCount > frame.Geometry->Indices.size()-candidate.FirstIndex) return false;
        constants.World = draw.World; constants.NormalWorld = BuildRealtimeNormalMatrix(draw.World);
        constants.Color = draw.Color; constants.Material = {std::clamp(draw.AlphaCutoff,0.0f,1.0f),draw.ForceCoverage ? 1.0f : 0.0f,0,0};
        constants.Textures.X = draw.DiffuseTexture ? 1 : 0;
        constants.Textures.Y = draw.NormalTexture ? 1 : 0;
        constants.Textures.Z = draw.OpacityTexture ? 1 : 0;
        output = constants; range = candidate; return true;
    }
}
