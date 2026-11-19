#pragma once

#include "Renderer/Nanite/NaniteGpu.h"

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteVisibilityThreads=64,NaniteVisibilityColumns=1024;
    inline constexpr unsigned NaniteVisibilityMaximumSide=8192,NaniteVisibilityMaximumQueries=1024*1024;
    enum class NaniteVisibilityPhase : unsigned { Main,Post };
    enum class NaniteVisibilityState : unsigned { Unclassified,FrustumRejected,MainVisible,Deferred,PostVisible,Occluded };
    struct alignas(16) NaniteVisibilityQuery
    {
        Float4 Minimum,Maximum; // Object bounds, already expanded for page quantization; W must be zero.
        Matrix4 Current,Previous; // Row-major object-to-clip, standard depth [0,1].
        ShaderUInt4 Identity; // History valid, main-pass state, original cluster ID, instance ID.
    };
    static_assert(sizeof(NaniteVisibilityQuery)==176,"Nanite visibility query ABI changed");
    struct alignas(16) NaniteVisibilityConstants
    {
        ShaderUInt4 Counts; // Query count, base width, base height, phase.
        ShaderUInt4 Layout; // Query input offset, classification output offset, mip count, reserved.
        ShaderUInt4 Source; // Previous mip offset, width, height, mip level (zero reads source depth).
        ShaderUInt4 Target; // Destination offset, width, height, dispatch columns.
    };
    struct NaniteVisibilityDispatch
    {
        NaniteVisibilityConstants Constants;
        unsigned Pass=0,GroupsX=0,GroupsY=0; // Pass zero reduces depth; pass one classifies bounds.
    };
    struct NaniteVisibilityWork
    {
        std::vector<unsigned> Input;
        std::vector<NaniteVisibilityDispatch> Dispatches;
        unsigned PyramidWords=0,OutputWords=0;
    };
    [[nodiscard]] bool ValidateNaniteVisibilityQueries(const std::vector<NaniteVisibilityQuery>&,std::string&);
    [[nodiscard]] bool BuildNaniteVisibilityQueries(const NaniteHierarchy&,const NanitePages&,const NaniteGpuSelection&,
        const Matrix4& current,const Matrix4& previous,bool historyValid,unsigned instance,
        std::vector<NaniteVisibilityQuery>&,std::string&);
}
