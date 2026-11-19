#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"

#include <functional>

namespace Evans::Renderer
{
    inline constexpr unsigned LumenDistanceBrickInterior=6,LumenDistanceBrickBorder=1,LumenDistanceBrickSide=8;
    inline constexpr unsigned LumenDistanceBrickSamples=LumenDistanceBrickSide*LumenDistanceBrickSide*LumenDistanceBrickSide;
    inline constexpr unsigned LumenDistanceMaximumAxis=160;
    inline constexpr std::size_t LumenDistanceMaximumVoxels=4*1024*1024,LumenDistanceMaximumTriangles=2*1024*1024;
    inline constexpr std::uint32_t LumenDistanceOutside=0,LumenDistanceInside=1,LumenDistanceFirstBrick=2;
    inline constexpr std::int16_t LumenDistanceQuantization=32767;
    enum class LumenDistanceSign { BackfaceVote,TwoSided };
    struct LumenDistanceBuildSettings
    {
        unsigned Resolution=32,SignRays=16;
        float BandCells=4;
        LumenDistanceSign Sign=LumenDistanceSign::BackfaceVote;
    };
    struct LumenDistanceField
    {
        // Geometry is normalized around Center before preprocessing. GPU queries
        // must subtract Center and divide by MeshScale before sampling the grid.
        Float3 Center,Minimum;
        float MeshScale=1,CellSize=1,MaximumDistance=1;
        std::array<unsigned,3> Dimensions{},BrickDimensions{};
        LumenDistanceSign Sign=LumenDistanceSign::BackfaceVote;
        // Entries 0/1 encode uniform positive/negative saturated distance.
        // Other entries address independent 8^3 bricks with a one-cell halo.
        std::vector<std::uint32_t> Directory;
        std::vector<std::int16_t> Bricks;
        std::uint64_t DistanceTriangleTests=0,SignQueries=0,InsideVoxels=0,AmbiguousVoxels=0;
    };
    struct LumenDistanceSample
    {
        float Distance=0,ConservativeDistance=0,ErrorBound=0; // Original mesh units.
    };
    using LumenDistanceCancellation=std::function<bool()>;

    // An immutable geometric proxy, not final GI. Cutout texture detail is not
    // represented by the distance volume; two-sided/open surfaces can request
    // unsigned distance. The sign vote is a heuristic, not a watertightness proof.
    bool BuildLumenDistanceField(const SceneRayQuery& mesh,const LumenDistanceBuildSettings& settings,
        LumenDistanceField& output,std::string& error,const LumenDistanceCancellation& cancelled={});
    bool ValidateLumenDistanceField(const LumenDistanceField& field,std::string& error);
    // Samples outside the padded domain return false without changing output.
    // The conservative magnitude removes interpolation and quantization error;
    // a zero step means unresolved within the voxel footprint, not an exact hit.
    bool SampleLumenDistanceField(const LumenDistanceField& field,Float3 position,LumenDistanceSample& output) noexcept;
}
