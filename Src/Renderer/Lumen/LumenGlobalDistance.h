#pragma once

#include "Renderer/Lumen/LumenDistanceScene.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenGlobalDistanceMaximumCascades=4;
    inline constexpr unsigned LumenGlobalDistanceMaximumSide=64;
    struct LumenGlobalDistanceSettings
    {
        Float3 CameraPosition{};
        float CellSize=0.25f,HitToleranceCells=0.25f;
        unsigned GridSide=16,CascadeCount=3;
    };
    struct alignas(16) LumenGlobalDistanceGrid
    {
        Float4 OriginCell; // First sample center and world-space cell size.
        Float4 Metrics; // Interpolation error, distance band, coordinate guard, reserved.
        ShaderUInt4 Storage; // Absolute voxel word offset, voxel count, reserved ZW.
    };
    struct alignas(16) LumenGlobalDistanceHeader
    {
        ShaderUInt4 Layout; // Cascade count, grid side, total voxels, header word offset.
        Float4 Numeric; // Source hit tolerance bound, float roundoff, reserved ZW.
        std::array<LumenGlobalDistanceGrid,LumenGlobalDistanceMaximumCascades> Grids{};
    };
    struct alignas(16) LumenGlobalDistanceFrame
    {
        ShaderUInt4 Counts,Offsets; // Immutable mesh scene prefix.
        LumenGlobalDistanceHeader Header;
        ShaderUInt4 Dispatch; // First dirty-list index, slice length, reserved ZW.
    };
    static_assert(sizeof(LumenGlobalDistanceGrid)==48 && sizeof(LumenGlobalDistanceHeader)==224 && sizeof(LumenGlobalDistanceFrame)==272,"Global distance-field ABI changed");
    struct LumenGlobalDistancePlan
    {
        LumenGlobalDistanceSettings Settings;
        LumenGlobalDistanceFrame Frame{};
        unsigned WordCount=0;
    };
    struct LumenGlobalDistanceCopy
    {
        unsigned SourceWord=0,DestinationWord=0,WordCount=0;
    };
    struct LumenGlobalDistanceUpdate
    {
        std::vector<LumenGlobalDistanceCopy> Copies;
        std::vector<std::uint32_t> Compose; // Destination voxel indices, never CPU voxel values.
        unsigned ReusedVoxels=0;
    };
    // Each level snaps independently. The native producer copies the immutable
    // mesh prefix and appends signed union/empty-space bounds for GI and shadows.
    // See Wright et al., Lumen (SIGGRAPH 2022), software ray tracing clipmaps.
    bool BuildLumenGlobalDistancePlan(const LumenDistanceSceneInfo& scene,const LumenGlobalDistanceSettings& settings,
        LumenGlobalDistancePlan& output,std::string& error);
    bool ValidateLumenGlobalDistancePlan(const LumenDistanceSceneInfo& scene,const LumenGlobalDistancePlan& plan,std::string& error);
    // The native caller must establish exact source-record identity before
    // supplying a previous plan. Numeric revision counters alone are insufficient.
    bool BuildLumenGlobalDistanceUpdate(const LumenDistanceSceneInfo& scene,const LumenGlobalDistancePlan& next,
        const LumenGlobalDistancePlan* previous,LumenGlobalDistanceUpdate& output,std::string& error);
    struct LumenGlobalDistanceTask
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
    };
    enum class LumenGlobalDistanceState { Empty,Updating,Ready,Failed };
    struct LumenGlobalDistanceStatistics
    {
        unsigned TotalVoxels=0,ReusedVoxels=0,ComposedVoxels=0,LastSubmittedVoxels=0;
        unsigned PendingVoxels() const noexcept { return TotalVoxels-ReusedVoxels-ComposedVoxels; }
    };
    class IRHI;
    class LumenGlobalDistanceCache
    {
    public:
        explicit LumenGlobalDistanceCache(IRHI& rhi):RHI(rhi) {}
        ~LumenGlobalDistanceCache();
        LumenGlobalDistanceCache(const LumenGlobalDistanceCache&)=delete;
        LumenGlobalDistanceCache& operator=(const LumenGlobalDistanceCache&)=delete;
        bool Update(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,std::string& error);
        LumenDistanceSceneHandle Handle() const { return Published; }
        bool Request(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,std::string& error);
        bool Advance(unsigned voxelBudget,std::string& error);
        void Cancel();
        bool CanUseSnapshot(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&) const;
        LumenGlobalDistanceState GetState() const { return State; }
        const LumenGlobalDistanceStatistics& Statistics() const { return Progress; }
    private:
        IRHI& RHI;
        LumenDistanceSceneHandle Source,Published;
        LumenGlobalDistanceSettings Settings,PendingSettings,LatestSettings;
        LumenGlobalDistanceTask Pending;
        LumenDistanceSceneHandle PendingSource;
        LumenGlobalDistanceState State=LumenGlobalDistanceState::Empty;
        LumenGlobalDistanceStatistics Progress;
    };
}
