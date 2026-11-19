#include "Renderer/PostProcessing/TemporalSuperResolution.h"

#include <atomic>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        std::atomic<std::uint64_t> NextHistory{1};
        constexpr double PixelCenter=0.5,MaximumRadiance=65504;
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        bool Extents(const TSRFrame& f)
        { return f.InputWidth && f.InputHeight && f.OutputWidth>=f.InputWidth && f.OutputHeight>=f.InputHeight && std::uint64_t(f.InputWidth)*2>=f.OutputWidth && std::uint64_t(f.InputHeight)*2>=f.OutputHeight && f.OutputWidth<=TSRMaximumExtent && f.OutputHeight<=TSRMaximumExtent && std::uint64_t(f.InputWidth)*f.InputHeight<=TSRMaximumPixels && std::uint64_t(f.OutputWidth)*f.OutputHeight<=TSRMaximumPixels; }

    }
    TSRHistoryHandle NewTSRHistoryHandle() noexcept
    {
        auto value=NextHistory.load(std::memory_order_relaxed);
        while (value!=UINT64_MAX) if (NextHistory.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
    TSRHistoryInfo TSRHistoryIdentity(const TSRFrame& frame) noexcept
    { return {frame.OutputWidth,frame.OutputHeight,frame.CurrentViewProjection,frame.Epoch,frame.Serial}; }
    Float4 DecodeTSRHalfPixel(const std::uint16_t* pixel) noexcept
    {
        const auto decode=[](std::uint16_t bits)
        {
            constexpr unsigned FractionBits=10,ExponentMask=31,FractionMask=1023,HiddenBit=1024,SignBit=32768;
            constexpr int SubnormalExponent=-24,NormalizedExponentOffset=25;
            const auto exponent=(bits>>FractionBits)&ExponentMask,fraction=bits&FractionMask;
            if (exponent==ExponentMask) return fraction?std::numeric_limits<float>::quiet_NaN():
                (bits&SignBit?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity());
            const float magnitude=exponent?std::ldexp(float(fraction+HiddenBit),int(exponent)-NormalizedExponentOffset):std::ldexp(float(fraction),SubnormalExponent);
            return bits&SignBit?-magnitude:magnitude;
        };
        return {decode(pixel[0]),decode(pixel[1]),decode(pixel[2]),decode(pixel[3])};
    }
    bool BuildTSRTextureConstants(std::uint32_t width,std::uint32_t height,std::uint32_t flags,TSRTextureConstants& output,std::string& error)
    {
        constexpr std::uint32_t knownFlags=TSRTextureAll;
        if (!width || !height || width>TSRMaximumExtent || height>TSRMaximumExtent ||
            std::uint64_t(width)*height>TSRMaximumPixels || (flags&~knownFlags))
        { error="TSR texture bridge requires bounded extents and known optional inputs"; return false; }
        output.ExtentFlags={width,height,flags,0}; error.clear(); return true;
    }
    bool BuildTSRConstants(const TSRFrame& f,const TSRHistoryInfo* previous,TSRConstants& output,std::string& error)
    {
        if (!Extents(f) || !f.Epoch || !f.Serial || !std::isfinite(f.JitterPixels.X) || !std::isfinite(f.JitterPixels.Y) || std::abs(f.JitterPixels.X)>PixelCenter || std::abs(f.JitterPixels.Y)>PixelCenter)
        { error="TSR requires bounded render/output extents, finite subpixel jitter and a versioned frame"; return false; }
        MotionBlurDesc current,old; current.ViewProjection=f.CurrentViewProjection; old.ViewProjection=f.PreviousViewProjection;
        MotionBlurConstants a,b;
        if (!BuildMotionBlurConstants(current,{},f.InputWidth,f.InputHeight,a) || !BuildMotionBlurConstants(old,{},f.InputWidth,f.InputHeight,b))
        { error="TSR camera matrices must be finite and invertible"; return false; }
        TSRConstants result; result.CurrentInverse=a.InverseViewProjection; result.PreviousInverse=b.InverseViewProjection;
        result.CurrentDepthPlane=a.DepthPlane; result.PreviousDepthPlane=b.DepthPlane;
        result.Extent={f.InputWidth,f.InputHeight,f.OutputWidth,f.OutputHeight};
        const auto settings=Techniques::SanitizeTemporalSuperResolutionSettings(f.Settings);
        result.JitterQuality={f.JitterPixels.X,f.JitterPixels.Y,settings.MaximumHistoryWeight,settings.DepthTolerance};
        result.Controls={settings.VarianceGamma,settings.ReactiveScale,settings.MaximumSamples,0};
        const bool compatible=previous && previous->Epoch==f.Epoch && previous->Width==f.OutputWidth && previous->Height==f.OutputHeight;
        if (compatible && f.Correspondence && (previous->Serial!=f.PreviousSerial || f.Serial<=previous->Serial || previous->ViewProjection.Values!=f.PreviousViewProjection.Values))
        { error="TSR history serial/camera does not match the previous presented frame"; return false; }
        result.Options={compatible && f.Correspondence?1u:0u,f.CurrentViewProjection.Values==f.PreviousViewProjection.Values?1u:0u,0,0};
        output=result; error.clear(); return true;
    }
    bool BuildTSRDisplaySettings(const Techniques::PostProcessSettings& requested,Techniques::PostProcessSettings& output,std::string& error)
    {
        const auto result=Techniques::SanitizePostProcessSettings(requested);
        if (requested.AntiAliasing!=Techniques::AntiAliasingMode::None && requested.AntiAliasing!=Techniques::AntiAliasingMode::FXAA && requested.AntiAliasing!=Techniques::AntiAliasingMode::TSR)
        { error="TSR display accepts only spatial display stages after reconstruction"; return false; }
        if (!Techniques::RequiresHDR(result.HDR)) { error="TSR display requires an HDR tone-mapping stage"; return false; }
        output=result; if (Techniques::UsesTemporalSuperResolution(output)) output.AntiAliasing=Techniques::AntiAliasingMode::None; error.clear(); return true;
    }
    bool BuildTSRRenderExtent(int width,int height,const Techniques::TemporalSuperResolutionSettings& requested,ShaderUInt4& output,std::string& error)
    {
        if (width<=0 || height<=0 || width>int(TSRMaximumExtent) || height>int(TSRMaximumExtent) || std::uint64_t(width)*height>TSRMaximumPixels)
        { error="TSR presentation extent exceeds the reconstruction contract"; return false; }
        const auto settings=Techniques::SanitizeTemporalSuperResolutionSettings(requested);
        // Round to the nearest pixel: ceil would turn an exact 2/3 scale into
        // an extra pixel when its float representation is slightly above 2/3.
        output={unsigned(std::max(1l,std::lround(double(width)*settings.RenderScale))),unsigned(std::max(1l,std::lround(double(height)*settings.RenderScale))),unsigned(width),unsigned(height)};
        error.clear(); return true;
    }
    bool TSRSceneState::Begin(const TSRSceneFrame& requested,int width,int height,const Techniques::PostProcessSettings& post,const TSRHistoryInfo* previous,std::string& error)
    {
        if (Techniques::UsesTemporalSuperResolution(post) && !requested.Enabled) { error="TSR requires a frozen scene reconstruction request"; return false; }
        if (requested.Enabled)
        {
            TSRConstants constants; Techniques::PostProcessSettings display;
            if (requested.Reconstruction.InputWidth!=unsigned(width) || requested.Reconstruction.InputHeight!=unsigned(height) ||
                (post.AntiAliasing!=Techniques::AntiAliasingMode::None && !Techniques::UsesTemporalSuperResolution(post)) || !BuildTSRDisplaySettings(post,display,error) || !BuildTSRConstants(requested.Reconstruction,previous,constants,error))
            { if (error.empty()) error="TSR scene requires matching render extents, HDR and exclusive reconstruction"; return false; }
        }
        Active=requested.Enabled; Frame=requested.Reconstruction; OpaqueReady=MotionReady=false; Pending={}; error.clear(); return true;
    }
    bool ComposeTSRSceneCapture(const TSRFrame& frame,const TSRResult& reconstructed,FrameCapture display,FrameCapture& scene,std::string& error)
    {
        if (!reconstructed.Handle || reconstructed.Info.Width!=frame.OutputWidth || reconstructed.Info.Height!=frame.OutputHeight || reconstructed.Info.Epoch!=frame.Epoch || reconstructed.Info.Serial!=frame.Serial ||
            reconstructed.Info.ViewProjection.Values!=frame.CurrentViewProjection.Values || display.Width!=frame.OutputWidth || display.Height!=frame.OutputHeight || (!display.HasColorOutput() && !display.GpuColor) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,reconstructed.Readback,error)))
        { if (error.empty()) error="TSR presentation does not cover its output grid"; return false; }
        const auto count=std::size_t(frame.OutputWidth)*frame.OutputHeight; std::vector<float> depth(frame.CaptureDiagnostics?count:0); std::vector<Float4> motion(frame.CaptureDiagnostics?count:0);
        scene.Statistics.TemporalSuperResolutionForegroundPixels=0;
        scene.Statistics.TemporalSuperResolutionValidMotionPixels=0;
        scene.Statistics.TemporalSuperResolutionAccumulatedPixels=0;
        for (std::size_t pixel=0;frame.CaptureDiagnostics && pixel<count;++pixel)
        {
            const auto meta=reconstructed.Readback.History[pixel*TSRHistoryVectors+1];
            depth[pixel]=meta.X; motion[pixel]=reconstructed.Readback.Auxiliary[pixel*TSRAuxiliaryVectors];
            if (meta.X<1)
            {
                ++scene.Statistics.TemporalSuperResolutionForegroundPixels;
                if (motion[pixel].W>0) ++scene.Statistics.TemporalSuperResolutionValidMotionPixels;
                // A fresh observation contributes at most one sample. More than
                // one proves per-pixel history reuse, not just a bound history.
                if (meta.Y>1 && frame.Settings.MaximumHistoryWeight>0)
                    ++scene.Statistics.TemporalSuperResolutionAccumulatedPixels;
            }
        }
        scene.RenderWidth=frame.InputWidth; scene.RenderHeight=frame.InputHeight; scene.Width=frame.OutputWidth; scene.Height=frame.OutputHeight;
        scene.GpuColor=std::move(display.GpuColor);
        scene.ColorRGBA8=std::move(display.ColorRGBA8); scene.Depth32=std::move(depth); scene.MotionVectors=std::move(motion);
        scene.HDRPasses=std::move(display.HDRPasses); scene.PostProcessingPasses=std::move(display.PostProcessingPasses);
        constexpr unsigned ReconstructionPasses=3; scene.Statistics.TemporalSuperResolutionPassCount=ReconstructionPasses;
        scene.Statistics.TemporalSuperResolutionHistoryValid=reconstructed.HistoryUsed; scene.Statistics.TemporalSuperResolutionSerial=frame.Serial;
        scene.Statistics.PostProcessPassCount+=display.Statistics.PostProcessPassCount; scene.Statistics.PassCount+=display.Statistics.PostProcessPassCount+ReconstructionPasses;
        error.clear(); return true;
    }
    bool ValidateTSRReadback(const TSRFrame& f,const TSRReadback& r,std::string& error)
    {
        if (!Extents(f) || r.History.size()!=std::size_t(f.OutputWidth)*f.OutputHeight*TSRHistoryVectors || r.Auxiliary.size()!=r.History.size())
        { error="TSR output does not cover the presentation grid"; return false; }
        const auto maxSamples=Techniques::SanitizeTemporalSuperResolutionSettings(f.Settings).MaximumSamples;
        if (!r.PresentationHDR.empty() && r.PresentationHDR.size()!=r.History.size()/TSRHistoryVectors)
        { error="TSR HDR image does not cover the presentation grid"; return false; }
        for (std::size_t pixel=0;pixel<r.History.size();pixel+=TSRHistoryVectors)
        {
            const auto color=r.History[pixel],meta=r.History[pixel+1];
            if (!Finite(color) || !Finite(meta) || !Finite(r.Auxiliary[pixel]) || !Finite(r.Auxiliary[pixel+1]) || color.X<0 || color.Y<0 || color.Z<0 || color.X>MaximumRadiance || color.Y>MaximumRadiance || color.Z>MaximumRadiance || color.W<0 || color.W>1 || meta.X<0 || meta.X>1 || meta.Y<=0 || meta.Y>maxSamples || meta.W<0 || meta.W>1)
            { error="TSR GPU output contains invalid radiance or history metadata"; return false; }
            if (!r.PresentationHDR.empty())
            {
                const auto exported=r.PresentationHDR[pixel/TSRHistoryVectors];
                constexpr float HalfRelativeTolerance=0.001f,HalfAbsoluteTolerance=0.000001f;
                const auto agrees=[](float a,float b){return std::abs(a-b)<=HalfAbsoluteTolerance+HalfRelativeTolerance*std::abs(a);};
                if (!Finite(exported) || !agrees(color.X,exported.X) || !agrees(color.Y,exported.Y) || !agrees(color.Z,exported.Z) || !agrees(color.W,exported.W))
                { error="TSR GPU HDR export disagrees with the reconstruction buffer"; return false; }
            }
        }
        error.clear(); return true;
    }
}
