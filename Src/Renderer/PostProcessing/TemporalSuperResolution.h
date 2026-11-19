#pragma once

#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Techniques/TemporalSuperResolution.h"

namespace Evans::Renderer
{
    inline constexpr unsigned TSRGroupSide=8,TSRInputVectors=5,TSRHistoryVectors=2,TSRAuxiliaryVectors=2;
    inline constexpr std::uint64_t TSRMaximumPixels=8*1024*1024;
    inline constexpr std::uint32_t TSRMaximumExtent=16*1024;
    enum TSRTextureFlags : std::uint32_t { TSRTextureMotion=1,TSRTextureOutline=2,TSRTextureReactive=4,TSRTextureAll=7 };
    enum TSRSceneImage : unsigned { TSRSceneFinal,TSRSceneOpaque,TSRSceneMotion,TSRSceneDepth,TSRSceneOutline,TSRSceneReactive,TSRSceneImageCount };
    constexpr bool TSRImageRequired(unsigned image,std::uint32_t flags) noexcept
    { return image==TSRSceneMotion?(flags&TSRTextureMotion)!=0:image==TSRSceneOutline?(flags&TSRTextureOutline)!=0:image==TSRSceneReactive?(flags&TSRTextureReactive)!=0:true; }
    struct alignas(16) TSRTextureConstants
    {
        // Pack uses the render extent; export uses the presentation extent.
        ShaderUInt4 ExtentFlags;
    };
    static_assert(sizeof(TSRTextureConstants)==sizeof(Float4),"TSR texture bridge is one vector");
    [[nodiscard]] bool BuildTSRTextureConstants(std::uint32_t width,std::uint32_t height,
        std::uint32_t flags,TSRTextureConstants& output,std::string& error);
    [[nodiscard]] bool BuildTSRRenderExtent(int width,int height,const Techniques::TemporalSuperResolutionSettings&,ShaderUInt4& output,std::string& error);
    // Consumes HDR/display fields after scene effects. A second temporal AA
    // resolve cannot consume an already reconstructed presentation image.
    [[nodiscard]] bool BuildTSRDisplaySettings(const Techniques::PostProcessSettings& requested,
        Techniques::PostProcessSettings& output,std::string& error);
    struct alignas(16) TSRConstants
    {
        Matrix4 CurrentInverse,PreviousInverse;
        Float4 CurrentDepthPlane,PreviousDepthPlane;
        ShaderUInt4 Extent; // render XY, presentation ZW
        Float4 JitterQuality; // current render-pixel jitter XY, maximum feedback, relative depth tolerance
        Float4 Controls; // variance gamma, reactive scale, maximum accumulated sample weight, reserved
        ShaderUInt4 Options; // valid history, static-background correspondence, reserved XY
    };
    static_assert(sizeof(TSRConstants)==224 && offsetof(TSRConstants,Extent)==160,"TSR constants are fourteen vectors");
    struct TSRFrame
    {
        std::uint32_t InputWidth=0,InputHeight=0,OutputWidth=0,OutputHeight=0;
        Matrix4 CurrentViewProjection,PreviousViewProjection;
        Float2 JitterPixels;
        Techniques::TemporalSuperResolutionSettings Settings;
        std::uint64_t Epoch=1,Serial=1,PreviousSerial=0;
        bool Correspondence=false;
        bool CaptureDiagnostics=true;
    };
    struct TSRHistoryInfo
    {
        std::uint32_t Width=0,Height=0;
        Matrix4 ViewProjection;
        std::uint64_t Epoch=0,Serial=0;
    };
    struct TSRHistoryHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(TSRHistoryHandle other) const noexcept { return Value==other.Value; }
        bool operator!=(TSRHistoryHandle other) const noexcept { return Value!=other.Value; }
    };
    struct TSRReadback
    {
        // Two vectors per presentation pixel: HDR/alpha, then hardware depth,
        // sample weight, optical depth and reactivity. Auxiliary pairs contain
        // presentation-pixel motion and the current overlay respectively.
        std::vector<Float4> History,Auxiliary;
        // Optional authoritative RGBA16F export mirror, for native image checks.
        std::vector<Float4> PresentationHDR;
    };
    struct TSRResult { TSRHistoryHandle Handle; TSRHistoryInfo Info; TSRReadback Readback; bool HistoryUsed=false; };
    struct TSRSceneFrame { bool Enabled=false; TSRFrame Reconstruction; };
    // Scene candidates publish only after every presentation/diagnostic step.
    // Aborting drops the candidate while retaining the last presented version.
    class TSRSceneState
    {
    public:
        bool Active=false,OpaqueReady=false,MotionReady=false;
        TSRFrame Frame;
        TSRHistoryHandle Committed,Pending;
        [[nodiscard]] bool Begin(const TSRSceneFrame&,int,int,const Techniques::PostProcessSettings&,const TSRHistoryInfo*,std::string&);
        [[nodiscard]] TSRHistoryHandle Abort() noexcept { const auto discarded=Pending; Pending={}; Active=OpaqueReady=MotionReady=false; return discarded; }
        [[nodiscard]] TSRHistoryHandle Commit() noexcept { const auto old=Committed; Committed=Active?Pending:TSRHistoryHandle{}; Pending={}; Active=OpaqueReady=MotionReady=false; return old; }
        void Reset() noexcept { Committed={}; (void)Abort(); }
    };
    [[nodiscard]] bool ComposeTSRSceneCapture(const TSRFrame&,const TSRResult&,FrameCapture,FrameCapture&,std::string&);

    // Lanczos-2 reconstruction uses the actual jittered render sample positions.
    // Nearest-depth motion dilation, optical-depth rejection and YCoCg variance
    // clipping condition high-resolution history. Opaque/final color differences
    // suppress history for transparency without motion/depth. Alpha and overlays
    // always belong to the current frame. History must publish after Present.
    // References: Salvi, "An Excursion in Temporal Supersampling", GDC 2016,
    // https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
    // AMD, "Temporal Upscaling: Past, Present, and Future", GDC 2023,
    // https://gpuopen.com/download/GDC-2023-Temporal-Upscaling.pdf
    // This is an independent teaching algorithm, not FSR, DLSS or Unreal TSR.
    [[nodiscard]] bool BuildTSRConstants(const TSRFrame& frame,const TSRHistoryInfo* previous,TSRConstants& output,std::string& error);
    [[nodiscard]] bool ValidateTSRReadback(const TSRFrame& frame,const TSRReadback& readback,std::string& error);
    [[nodiscard]] TSRHistoryInfo TSRHistoryIdentity(const TSRFrame& frame) noexcept;
    [[nodiscard]] TSRHistoryHandle NewTSRHistoryHandle() noexcept;
    [[nodiscard]] Float4 DecodeTSRHalfPixel(const std::uint16_t* pixel) noexcept;
}
