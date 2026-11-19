#pragma once

#include "Renderer/Hair/HairLighting.h"
#include "Renderer/Hair/HairScattering.h"
#include "Renderer/Hair/HairStrands.h"
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/Hair/HairShadowReceiver.h"
#include <memory>

namespace Evans::Renderer
{
    // Publications are immutable and frozen before BeginFrame. Geometry and
    // expanded emitter samples have independent upload identities.
    struct HairFurFrame
    {
        std::shared_ptr<const HairStrandGeometry> Geometry;
        std::shared_ptr<const HairLightSamples> Lights;
        Techniques::HairFurSettings Settings;
        double SampleTimeSeconds=0;
    };
    struct HairFurDesc
    {
        Matrix4 ViewProjection;
        LightingState Lighting;
        Float3 Forward{0,0,1};
        Float4 ClipPlane{0,0,0,0};
        bool Orthographic=false;
        Matrix4 LightViewProjection;
        bool SamplesShadowMap=false;
    };
    struct HairFurVelocityDesc
    {
        HairFurDesc View;
        Float2 JitterDeltaPixels;
        bool HistoryValid=false; // Scene correspondence, independently of shutter interval.
    };
    enum class HairFurRasterPhase : unsigned { Surface,CameraDepth,ShadowDepth,ShadowMoments,Capture,Velocity,Count };
    inline constexpr unsigned HairFurPhaseCount=static_cast<unsigned>(HairFurRasterPhase::Count);
    inline unsigned HairFurPhaseIndex(HairFurRasterPhase phase) { return static_cast<unsigned>(phase); }
    inline unsigned HairFurPipelineKey(HairFurRasterPhase phase,unsigned samples,bool contact)
    {
        constexpr unsigned ContactReceiverBit=1u<<8;
        return samples|(phase==HairFurRasterPhase::Surface && contact?ContactReceiverBit:0);
    }
    inline bool HairFurReplayPass(PassType pass)
    { return pass==PassType::Shadow || pass==PassType::ContactDepth || pass==PassType::ForwardPlusDepth || pass==PassType::EnvironmentCapture || pass==PassType::PlanarCapture || pass==PassType::Velocity; }
    inline bool HairFurCameraDepthPass(PassType pass) { return pass==PassType::ContactDepth || pass==PassType::ForwardPlusDepth; }
    inline HairFurRasterPhase HairFurReplayPhase(PassType pass,bool variance)
    {
        return pass==PassType::Shadow ? (variance?HairFurRasterPhase::ShadowMoments:HairFurRasterPhase::ShadowDepth) :
            HairFurCameraDepthPass(pass)?HairFurRasterPhase::CameraDepth:pass==PassType::Velocity?HairFurRasterPhase::Velocity:HairFurRasterPhase::Capture;
    }
    inline bool SameHairFurRasterView(const HairFurDesc& a,const HairFurDesc& b)
    {
        const auto eyeA=a.Lighting.CameraPosition,eyeB=b.Lighting.CameraPosition;
        return a.ViewProjection.Values==b.ViewProjection.Values && a.Orthographic==b.Orthographic &&
            eyeA.X==eyeB.X && eyeA.Y==eyeB.Y && eyeA.Z==eyeB.Z && a.Forward.X==b.Forward.X && a.Forward.Y==b.Forward.Y && a.Forward.Z==b.Forward.Z &&
            a.ClipPlane.X==b.ClipPlane.X && a.ClipPlane.Y==b.ClipPlane.Y && a.ClipPlane.Z==b.ClipPlane.Z && a.ClipPlane.W==b.ClipPlane.W;
    }
    inline bool UsesHairFurFrame(const HairFurFrame& frame)
    { return frame.Geometry && frame.Lights && Techniques::UsesHairFur(frame.Settings); }
    inline bool HairFurPrecedesLaterPasses(const FrameCapture& frame)
    {
        const auto& stats=frame.Statistics;
        return !(stats.SubsurfacePassCount || stats.AmbientOcclusionPassCount || stats.GlobalIlluminationPassCount || stats.ReflectionPassCount ||
            stats.DepthOfFieldPassCount || stats.MotionBlurPassCount || stats.TemporalAAPassCount || stats.VolumetricFogPassCount ||
            stats.VolumetricLightingPassCount || stats.ParticipatingMediaPassCount || stats.VolumetricCloudPassCount || stats.LightShaftPassCount);
    }
    inline bool ValidateHairFurPublication(const HairFurFrame& frame,std::string& error)
    {
        if (!frame.Geometry && !frame.Lights) return true;
        if (!frame.Geometry || !frame.Lights) { error="Hair requires both immutable geometry and light publications"; return false; }
        const auto& geometry=*frame.Geometry; const auto& lights=*frame.Lights;
        constexpr std::size_t VertexLimit=Techniques::MaximumHairStrands*(Techniques::MaximumHairSegments+1ull)*2;
        constexpr std::size_t IndexLimit=Techniques::MaximumHairStrands*Techniques::MaximumHairSegments*6ull;
        if (geometry.Vertices.size()>VertexLimit || geometry.Indices.size()>IndexLimit || geometry.Indices.size()%3 ||
            geometry.Vertices.empty()!=geometry.Indices.empty() || lights.Samples.size()>UINT32_MAX ||
            lights.ShadowBegin>lights.ShadowEnd || lights.ShadowEnd>lights.Samples.size())
        { error="Hair publication exceeds native topology or lighting limits"; return false; }
        const auto finite=[](Float4 value){return std::isfinite(value.X)&&std::isfinite(value.Y)&&std::isfinite(value.Z)&&std::isfinite(value.W);};
        for (const auto& vertex:geometry.Vertices)
        {
            const auto p=vertex.Position,t=vertex.Tangent;
            if (!finite({p.X,p.Y,p.Z,0}) || !finite({t.X,t.Y,t.Z,0}) ||
                !(std::hypot(double(t.X),double(t.Y),double(t.Z))>0) || !finite(vertex.Reflectance) || !finite(vertex.Emission) ||
                !finite(vertex.Ribbon) || std::abs(vertex.Ribbon.Y)>1 || vertex.Ribbon.Z<0 || vertex.Ribbon.Z>1)
            { error="Hair publication contains invalid vertex attributes"; return false; }
        }
        for (auto index:geometry.Indices) if (index>=geometry.Vertices.size())
        { error="Hair publication contains an out-of-range vertex index"; return false; }
        for (const auto& light:lights.Samples)
        {
            const auto kind=light.PositionKind.W;
            if (!finite(light.PositionKind) || !finite(light.DirectionRange) || !finite(light.Radiance) ||
                (kind!=0 && kind!=1 && kind!=2) || (kind!=0 && !(light.DirectionRange.W>0)) ||
                light.Radiance.X<0 || light.Radiance.Y<0 || light.Radiance.Z<0)
            { error="Hair publication contains invalid emitter samples"; return false; }
        }
        return true;
    }
    struct HairFurExecution
    {
        HairFurFrame Frame;
        bool Active=false,Resolved=false,LumenReceiver=false;
        bool VelocityRequested=false,VelocityResolved=false;
        bool HasPrimaryView=false;
        HairFurDesc PrimaryView;
        std::size_t ReplayedPass=0;
        std::vector<std::pair<TextureHandle,Matrix4>> ShadowViews;
        HairContactShadowState Contact;
        bool UsesContactReceiver() const
        { return Contact.Active && Frame.Lights && Frame.Lights->ShadowBegin<Frame.Lights->ShadowEnd; }
        bool ValidateContactProducer(const Matrix4& view,const LightingState& lighting,std::string& error) const
        {
            if (!Active || Frame.Geometry->Indices.empty()) return true;
            if (!HasPrimaryView || view.Values!=PrimaryView.ViewProjection.Values)
            { error="Fiber contact generation requires its matching camera-depth replay"; return false; }
            const SceneLight* source=nullptr; if (!ResolveHairShadowSource(*Frame.Lights,source,error)) return false;
            if (!source) return true;
            const auto index=lighting.ShadowLightIndex;
            if (index<0 || std::uint32_t(index)>=std::min(lighting.LightCount,MaximumSceneLights) || !SameHairContactSource(*source,lighting.Lights[index]))
            { error="Contact generation selected a different emitter than the frozen fiber packet"; return false; }
            return true;
        }
        void RecordContact(const Matrix4& view,const LightingState& lighting,bool active)
        {
            Contact={}; Contact.Generated=true; Contact.Active=active; Contact.ViewProjection=view;
            if (active) Contact.Source=lighting.Lights[lighting.ShadowLightIndex];
        }
        void RecordShadowView(TextureHandle map,const Matrix4& projection)
        {
            for (auto& entry:ShadowViews) if (entry.first==map) { entry.second=projection; return; }
            ShadowViews.emplace_back(map,projection);
        }
        bool HasShadowView(TextureHandle map,const Matrix4& projection) const
        {
            for (const auto& entry:ShadowViews) if (entry.first==map && entry.second.Values==projection.Values) return true;
            return false;
        }
        std::shared_ptr<const HairStrandGeometry> PresentedGeometry;
        HairFurDesc PresentedView;
        double PresentedTimeSeconds=0;
        int PresentedWidth=0,PresentedHeight=0;
        bool MatchesPrimaryView(const HairFurDesc& view) const { return !HasPrimaryView || SameHairFurRasterView(PrimaryView,view); }
        void RecordPrimaryView(const HairFurDesc& view) { PrimaryView=view; HasPrimaryView=true; }
        bool Complete() const { return !Active || Resolved; }
        bool VelocityComplete() const { return !VelocityRequested || VelocityResolved; }
        void ResetHistory() { PresentedGeometry.reset(); PresentedView={}; PresentedWidth=PresentedHeight=0; }
        // Frame history advances on presentation, never on command recording.
        // A changed world-space groom has no vertex correspondence. Its first
        // frame writes invalid velocity and establishes the next baseline.
        void CommitPresent(int width,int height)
        {
            if (!Active || !Resolved || !HasPrimaryView || Frame.Geometry->Indices.empty()) { ResetHistory(); return; }
            PresentedTimeSeconds=Frame.SampleTimeSeconds; PresentedGeometry=Frame.Geometry; PresentedView=PrimaryView; PresentedWidth=width; PresentedHeight=height;
        }
        bool CanReproject(const HairFurDesc& view,int width,int height) const
        {
            const auto a=view.ClipPlane,b=PresentedView.ClipPlane;
            return PresentedGeometry && PresentedGeometry==Frame.Geometry && PresentedWidth==width && PresentedHeight==height &&
                a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W;
        }
        bool Freeze(HairFurFrame request,const Techniques::Settings& settings,std::string& error)
        {
            Active=Resolved=LumenReceiver=HasPrimaryView=VelocityRequested=VelocityResolved=false; ReplayedPass=0; Frame={}; PrimaryView={}; ShadowViews.clear(); Contact={};
            if (bool(request.Geometry)!=bool(request.Lights))
            { error="Hair requires both immutable geometry and light publications"; return false; }
            request.Settings=Techniques::SanitizeHairFurSettings(request.Settings);
            if (!UsesHairFurFrame(request)) return true;
            if (Techniques::UsesPathTracing(settings.Pipeline) || !Techniques::RequiresHDR(settings.PostProcessing.HDR))
            { error="Hair ribbons require an HDR raster frame"; return false; }
            Frame=std::move(request); Active=true;
            LumenReceiver=!Frame.Geometry->Indices.empty() && Techniques::UsesLumen(settings.GlobalIllumination) && settings.GlobalIllumination.Lumen.HairIndirect;
            VelocityRequested=!Frame.Geometry->Indices.empty() &&
                Techniques::RequiresMotionVectors(settings);
            return true;
        }
    };
    struct HairFurConstants
    {
        Matrix4 ViewProjection;
        Float4 Eye; // XYZ position; W selects an orthographic view.
        Float4 Forward;
        FiberScatteringProfile Scattering;
        Float4 RootTint,TipTint,Ambient,ClipPlane;
        ShaderUInt4 Options; // Light sample count, shadow sample begin/end, shadow replay flag.
        Float4 ShadowDepth; // Linear clip-Z/W encoding, followed by derivative-variance cap.
        Matrix4 PreviousViewProjection;
        Float4 PreviousEye,PreviousForward;
        Float4 MotionViewport; // Extent XY; current-minus-previous jitter ZW, in top-left pixels.
        ShaderUInt4 MotionOptions; // Valid correspondence, deferred Lumen indirect, reserved ZW.
        HairShadowReceiverConstants Shadows;
        ShaderUInt4 ContactControl; // Enabled, visibility layers, reserved, reserved.
        Float4 Animation; // Current/previous phase, tip displacement, inverse growth length.
    };
    static_assert(sizeof(HairFurConstants)==1168 && offsetof(HairFurConstants,Scattering)==96 && offsetof(HairFurConstants,Options)==224 && offsetof(HairFurConstants,ShadowDepth)==240 &&
        offsetof(HairFurConstants,PreviousViewProjection)==256 && offsetof(HairFurConstants,MotionOptions)==368 && offsetof(HairFurConstants,Shadows)==384 && offsetof(HairFurConstants,ContactControl)==1136,
        "Hair native uniform ABI changed");
    inline void SetHairFurAnimation(const HairFurFrame& frame,double previousTime,HairFurConstants& output)
    {
        if (!frame.Settings.Animate || !frame.Geometry) return;
        const float length=frame.Geometry->GrowthLength;
        output.Animation={float(std::fmod(frame.SampleTimeSeconds,3600.0)),float(std::fmod(previousTime,3600.0)),
            length*frame.Settings.WindStrength,length>0?1.0f/length:0.0f};
    }
    inline bool BuildHairFurConstants(const Matrix4& camera,const LightingState& lighting,const Techniques::HairFurSettings& request,
        const HairLightSamples& samples,bool orthographic,Float3 forward,Float4 clipPlane,HairFurConstants& output)
    {
        output={};
        const auto finite=[](Float4 v){return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W);};
        const auto eye=lighting.CameraPosition;
        if (samples.Samples.size()>UINT32_MAX || samples.ShadowBegin>samples.ShadowEnd || samples.ShadowEnd>samples.Samples.size() ||
            !std::isfinite(eye.X) || !std::isfinite(eye.Y) || !std::isfinite(eye.Z) || !std::isfinite(lighting.AmbientStrength) ||
            !finite({lighting.AmbientColor.X,lighting.AmbientColor.Y,lighting.AmbientColor.Z,0}) || !finite(clipPlane)) return false;
        for (float value:camera.Values) if (!std::isfinite(value)) return false;
        if (!std::isfinite(forward.X) || !std::isfinite(forward.Y) || !std::isfinite(forward.Z)) return false;
        const double length=std::hypot(double(forward.X),double(forward.Y),double(forward.Z)); if (!(length>0)) return false;
        const auto settings=Techniques::SanitizeHairFurSettings(request);
        output.ViewProjection=camera; output.Eye={eye.X,eye.Y,eye.Z,orthographic?1.0f:0.0f};
        output.Forward={float(forward.X/length),float(forward.Y/length),float(forward.Z/length),0};
        output.Scattering=BuildFiberScatteringProfile(settings);
        output.RootTint={settings.RootTint[0],settings.RootTint[1],settings.RootTint[2],0};
        output.TipTint={settings.TipTint[0],settings.TipTint[1],settings.TipTint[2],0};
        output.Ambient={std::max(lighting.AmbientColor.X,0.0f),std::max(lighting.AmbientColor.Y,0.0f),std::max(lighting.AmbientColor.Z,0.0f),
            std::max(lighting.AmbientStrength,0.0f)};
        output.ClipPlane=clipPlane; output.Options={static_cast<std::uint32_t>(samples.Samples.size()),samples.ShadowBegin,samples.ShadowEnd,0};
        output.PreviousViewProjection=camera; output.PreviousEye=output.Eye; output.PreviousForward=output.Forward; return true;
    }
    inline bool BuildHairFurReplayConstants(const HairFurDesc& view,const HairFurFrame& frame,HairFurRasterPhase phase,
        const Techniques::VSMSettings& variance,HairFurConstants& output,std::string& error)
    {
        if (!BuildHairFurConstants(view.ViewProjection,view.Lighting,frame.Settings,*frame.Lights,view.Orthographic,view.Forward,view.ClipPlane,output))
        { error="Hair replay requires finite camera and clipping inputs"; return false; }
        SetHairFurAnimation(frame,frame.SampleTimeSeconds,output);
        output.Options.W=phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments?1u:0u;
        if (phase==HairFurRasterPhase::ShadowMoments)
        {
            if (!BuildVSMEncoding(view.ViewProjection,output.ShadowDepth,error)) return false;
            output.ShadowDepth.W=Techniques::SanitizeVSMSettings(variance).MaximumDerivativeVariance;
        }
        return true;
    }
    inline bool BuildHairFurVelocityConstants(const HairFurVelocityDesc& desc,const HairFurExecution& execution,
        int width,int height,HairFurConstants& output,std::string& error)
    {
        const auto& view=desc.View; const auto& frame=execution.Frame;
        if (!frame.Lights || width<=0 || height<=0 || !std::isfinite(desc.JitterDeltaPixels.X) || !std::isfinite(desc.JitterDeltaPixels.Y) ||
            !BuildHairFurConstants(view.ViewProjection,view.Lighting,frame.Settings,*frame.Lights,view.Orthographic,view.Forward,view.ClipPlane,output))
        { error="Hair velocity requires finite raster, extent and jitter inputs"; return false; }
        SetHairFurAnimation(frame,execution.PresentedTimeSeconds,output);
        const bool valid=desc.HistoryValid && execution.CanReproject(view,width,height);
        if (valid)
        {
            const auto& previous=execution.PresentedView; HairFurConstants packed;
            if (!BuildHairFurConstants(previous.ViewProjection,previous.Lighting,frame.Settings,*frame.Lights,previous.Orthographic,previous.Forward,previous.ClipPlane,packed))
            { error="Presented hair camera cannot produce finite velocity"; return false; }
            output.PreviousViewProjection=packed.ViewProjection; output.PreviousEye=packed.Eye; output.PreviousForward=packed.Forward;
        }
        output.MotionViewport={float(width),float(height),desc.JitterDeltaPixels.X,desc.JitterDeltaPixels.Y};
        output.MotionOptions={valid?1u:0u,0,0,0}; return true;
    }
}
