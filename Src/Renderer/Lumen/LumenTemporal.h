#pragma once

#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Renderer/Lumen/LumenSurfaceLighting.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenTemporalThreads=64,LumenTemporalColumns=1024;
    inline constexpr std::size_t LumenTemporalMaximumBytes=128u*1024u*1024u;
    // RG and B/count use binary16 pairs. The normal uses signed octahedral
    // coordinates; device depth keeps all 32 bits for distant disocclusions.
    struct alignas(16) LumenTemporalPixel { ShaderUInt4 Packed; };
    struct alignas(16) LumenTemporalFrame
    {
        Matrix4 InverseViewProjection,PreviousInverseViewProjection;
        Float4 PreviousDepthPlane;
        ShaderUInt4 Extent; // Width, height, pixel count, dispatch row stride.
        Float4 Temporal; // History weight, relative depth tolerance, normal agreement, plane tolerance.
        Float4 Clip; // Relative color allowance, maximum history samples; reserved ZW.
        Float4 Jitter; // Current-minus-previous pixels, composition intensity, history valid.
        Float4 Numeric; // Absolute optical-depth tolerance, minimum clip extent, homogeneous floor, half maximum.
    };
    static_assert(sizeof(LumenTemporalPixel)==16 && sizeof(LumenTemporalFrame)==224,"Lumen temporal ABI changed");
    bool BuildLumenTemporalFrame(const GlobalIlluminationDesc& camera,const Techniques::GlobalIlluminationSettings& settings,
        int width,int height,bool historyValid,LumenTemporalFrame& output,std::string& error);

    bool BuildLumenReflectionTemporalFrame(const ReflectionDesc& camera,const Techniques::ReflectionSettings& settings,
        int width,int height,bool historyValid,LumenTemporalFrame& output,std::string& error);

    struct LumenTemporalKey
    {
        LumenLightingHandle Lighting;
        Techniques::GlobalIlluminationSettings Settings;
        Techniques::ShadingModel Shading=Techniques::ShadingModel::Phong;
        unsigned Width=0,Height=0;
        Techniques::ReflectionSettings Reflections;
        bool Reflected=false;
        bool operator==(const LumenTemporalKey& other) const noexcept
        { return Lighting==other.Lighting && Settings==other.Settings && Shading==other.Shading && Width==other.Width && Height==other.Height && Reflected==other.Reflected && Reflections==other.Reflections; }
    };
    // Karis, "High-Quality Temporal Supersampling", SIGGRAPH 2014:
    // motion reprojection and neighborhood clipping stabilize a noisy estimate.
    // Each owner accumulates diffuse GI or specular reflection before HDR composition. A
    // rejected current visibility estimate cannot recover light from history.
    // The pending slot may be overwritten on cancellation; publication, source
    // identity and the read slot change only after successful presentation.
    class LumenTemporalHistory
    {
    public:
        unsigned ReadIndex() const noexcept { return Read; }
        unsigned WriteIndex() const noexcept { return 1u-Read; }
        bool CanReuse(const LumenTemporalKey& key) const noexcept { return Valid && Committed==key; }
        void Record(const LumenTemporalKey& key) noexcept { Pending=key; Ready=true; }
        void Finish(bool publish) noexcept
        {
            if (publish)
            {
                if (Ready) { Read=WriteIndex(); Committed=Pending; Valid=true; }
                else Valid=false;
            }
            Ready=false;
        }
        void Invalidate() noexcept { Valid=Ready=false; }
    private:
        LumenTemporalKey Committed,Pending;
        unsigned Read=0;
        bool Valid=false,Ready=false;
    };
}
