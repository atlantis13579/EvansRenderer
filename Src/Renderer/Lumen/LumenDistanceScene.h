#pragma once

#include "Renderer/Lumen/LumenDistanceField.h"
#include "Renderer/Lumen/LumenHair.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenDistanceTraceThreads=64,LumenDistanceTraceColumns=1024;
    inline constexpr unsigned LumenDistanceMaximumFields=1024,LumenDistanceMaximumInstances=4096,LumenDistanceMaximumTraceSteps=4096;
    inline constexpr std::size_t LumenDistanceMaximumWords=16*1024*1024,LumenDistanceMaximumRays=1024*1024;
    inline constexpr std::uint32_t LumenDistanceInvalid=UINT32_MAX;
    inline constexpr unsigned LumenDistanceShadowCaster=1;
    inline constexpr unsigned LumenHairPointerShift=2; // Low traversal bits retain reflected-ray controls.
    struct LumenDistanceSceneIdentity { std::uint64_t Epoch=1,Serial=1,Geometry=1; };
    struct LumenDistanceInstance
    {
        unsigned Field=0,UserId=0;
        Matrix4 World;
        bool CastShadow=true; // GI visibility and direct-shadow visibility are independent.
    };
    struct LumenDistanceSceneUpload
    {
        LumenDistanceSceneIdentity Identity;
        std::vector<std::shared_ptr<const LumenDistanceField>> Fields;
        std::vector<LumenDistanceInstance> Instances;
        std::shared_ptr<const LumenHairVolume> Hair;
    };
    // Word offsets address one immutable uint buffer, keeping the native trace
    // contract within ES 3.1's minimum storage-binding count. Integer metadata
    // and packed signed samples are never converted through float addresses.
    struct alignas(16) LumenDistanceGPUField
    {
        ShaderUInt4 Dimensions,BrickDimensions,Storage; // Storage: directory word, brick word, brick count, reserved.
        Float4 MinimumCell,Distances; // Distance band, interpolation error; reserved ZW.
    };
    struct alignas(16) LumenDistanceGPUInstance
    {
        Float4 Center; // World-space location of the normalized field origin.
        std::array<Float4,3> WorldToUnit;
        ShaderUInt4 Metadata; // Field, caller identifier, visibility flags, reserved W.
    };
    struct alignas(16) LumenDistanceGPUNode
    {
        Float4 Minimum,Maximum;
        ShaderUInt4 Links; // First child, instance, leaf flag, escape; all links move forward.
    };
    static_assert(sizeof(LumenDistanceGPUField)==80 && sizeof(LumenDistanceGPUInstance)==80 && sizeof(LumenDistanceGPUNode)==48,"Lumen distance scene word ABI changed");
    struct LumenGlobalDistancePlan;
    struct LumenHardwareScene;
    struct LumenDistanceSceneInfo
    {
        LumenDistanceSceneIdentity Identity;
        std::shared_ptr<const LumenGlobalDistancePlan> Global;
        std::shared_ptr<const LumenHardwareScene> Hardware; // Optional immutable triangle transport snapshot.
        ShaderUInt4 Counts; // Fields, instances, nodes, words.
        ShaderUInt4 Offsets; // Field records, instance records, node records; reserved W.
        float MaximumHitToleranceCells=2;
        unsigned HairHeaderWord=0;
        std::vector<unsigned> UserIds;
        std::vector<unsigned> InstanceFlags;
    };
    struct LumenDistanceSceneData { LumenDistanceSceneInfo Info; std::vector<std::uint32_t> Words; };
    struct LumenDistanceSceneHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(LumenDistanceSceneHandle other) const noexcept { return Value==other.Value; }
    };
    struct alignas(16) LumenDistanceRay
    {
        Float4 OriginMinimum,DirectionMaximum; // Unit world direction; distance interval in world units.
    };
    struct LumenDistanceTraceUpload
    {
        std::vector<LumenDistanceRay> Rays;
        unsigned MaximumSteps=256; // Per candidate instance, with explicit exhaustion feedback.
        float HitToleranceCells=0.1f,StepScale=0.9f;
        float GlobalNearDistance=2;
        unsigned GlobalMaximumSteps=128;
        bool ShadowCastersOnly=false;
        bool HairTransport=true;
        Float3 HairSky{}; // Isotropic single-scattering source; emission is stored in the volume.
    };
    enum class LumenDistanceTraceStatus:std::uint32_t { Miss=0,Surface=1,Unresolved=2 };
    struct alignas(16) LumenDistanceHit
    {
        Float4 PositionDistance,NormalResidual;
        // Projected sphere radii in ray-distance units, not intersection-time
        // error bounds at grazing angles. Interpolation error / hit threshold; Z reserved, W global steps.
        Float4 Uncertainty;
        ShaderUInt4 Metadata; // Instance, caller identifier, status, accumulated sphere-trace steps.
        Float4 Medium{0,0,0,1}; // Integrated medium source RGB and scalar transmittance.
    };
    static_assert(sizeof(LumenDistanceRay)==32 && sizeof(LumenDistanceHit)==80,"Lumen distance query ABI changed");
    struct alignas(16) LumenDistanceTraceFrame
    {
        ShaderUInt4 Counts,Offsets,Traversal; // Rays/nodes/instances/fields; offsets; steps/row stride/required flags/hair pointer and ray controls.
        Float4 Parameters; // Hit tolerance in cells, step scale, mesh-detail distance, global step budget.
        Float4 HairSky{0,0,0,0};
    };
    static_assert(sizeof(LumenDistanceTraceFrame)==80,"Lumen distance trace frame ABI changed");
    struct LumenDistanceTraceResult
    {
        LumenDistanceSceneIdentity Identity;
        std::vector<LumenDistanceHit> Hits;
        bool HardwareTracing=false; // Actual native query path; false also covers unsupported-device fallback.
    };

    bool BuildLumenDistanceScene(const LumenDistanceSceneUpload& upload,LumenDistanceSceneData& output,std::string& error);
    bool ValidateLumenDistanceScene(const LumenDistanceSceneData& scene,std::string& error);
    bool BuildLumenDistanceTraceFrame(const LumenDistanceSceneInfo& scene,const LumenDistanceTraceUpload& upload,
        LumenDistanceTraceFrame& output,std::string& error);
    bool ValidateLumenDistanceTraceResult(const LumenDistanceSceneInfo& scene,const LumenDistanceTraceUpload& upload,
        const LumenDistanceTraceResult& result,std::string& error);
    LumenDistanceSceneHandle NewLumenDistanceSceneHandle() noexcept;
}
