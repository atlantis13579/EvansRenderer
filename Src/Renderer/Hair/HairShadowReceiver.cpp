#include "Renderer/Hair/HairShadowReceiver.h"
#include "Renderer/Lighting/AreaLightSampling.h"
#include "Maths/Matrix4.h"

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float4 value)
        { return std::isfinite(value.X)&&std::isfinite(value.Y)&&std::isfinite(value.Z)&&std::isfinite(value.W); }
        bool ValidProjection(const Matrix4& matrix)
        {
            for (float value:matrix.Values) if (!std::isfinite(value)) return false;
            Maths::Matrix4 inverse;
            if (!Maths::Matrix4(matrix.Values).Inverse(inverse)) return false;
            for (unsigned entry=0;entry<16;++entry) if (!std::isfinite(inverse.Data()[entry])) return false;
            return true;
        }
        bool ValidCascades(const CascadedShadowState& state)
        {
            const auto& frame=state.Frame;
            if (frame.CascadeCount>MaximumShadowCascades) return false;
            if (!frame.CascadeCount) return true;
            if (state.MapResolution<=0 || !Finite(frame.EyeDepthPlane) ||
                !(std::hypot(double(frame.EyeDepthPlane.X),double(frame.EyeDepthPlane.Y),double(frame.EyeDepthPlane.Z))>0) ||
                !std::isfinite(frame.NearDistance) || !std::isfinite(frame.FarDistance) || frame.NearDistance<0 || !(frame.FarDistance>frame.NearDistance)) return false;
            float previous=frame.NearDistance;
            for (unsigned index=0;index<frame.CascadeCount;++index)
            {
                const auto& cascade=frame.Cascades[index];
                if (!ValidProjection(cascade.LightViewProjection) || !std::isfinite(cascade.FarDistance) || !std::isfinite(cascade.BlendStartDistance) ||
                    !(cascade.FarDistance>previous) || cascade.FarDistance>frame.FarDistance || cascade.BlendStartDistance<previous || cascade.BlendStartDistance>cascade.FarDistance) return false;
                previous=cascade.FarDistance;
            }
            return previous==frame.FarDistance;
        }
    }
    bool ResolveHairShadowSource(const HairLightSamples& lights,const SceneLight*& selectedSource,std::string& error)
    {
        selectedSource=nullptr; error.clear();
        if (lights.ShadowBegin>lights.ShadowEnd || lights.ShadowEnd>lights.Samples.size())
        { error="Hair shadow sample interval is outside the frozen light packet"; return false; }
        if (lights.ShadowBegin==lights.ShadowEnd)
        {
            if (lights.SourceLights && lights.SourceLights->GetShadowLightIndex()>=0)
            { error="Hair shadow packet omitted its selected source's sample interval"; return false; }
            return true;
        }
        if (!lights.SourceLights)
        { error="Hair shadow receivers require the frozen complete source-light collection"; return false; }
        const auto selected=lights.SourceLights->GetShadowLightIndex(); const auto& source=lights.SourceLights->GetLights();
        if (selected<0 || std::size_t(selected)>=source.size())
        { error="Hair shadow interval has no selected source light"; return false; }
        std::uint64_t firstSample=0,selectedEnd=0;
        for (int index=0;index<=selected;++index)
        {
            const auto& entry=source[index]; const auto side=entry.Type==LightType::RectArea?BuildRectangleEmitterQuadrature(entry).SamplesPerAxis:1u;
            const std::uint64_t next=firstSample+std::uint64_t(side)*side;
            if (index==selected) { selectedEnd=next; break; }
            firstSample=next;
        }
        if (firstSample!=lights.ShadowBegin || selectedEnd!=lights.ShadowEnd)
        { error="Hair shadow interval does not match its frozen source-light expansion"; return false; }
        if (source[selected].CastShadows) selectedSource=&source[selected];
        return true;
    }
    bool SameHairContactSource(const SceneLight& a,const SceneLight& b)
    {
        if (a.Type!=b.Type || a.CastShadows!=b.CastShadows) return false;
        const auto first=a.Type==LightType::Directional?a.Direction:a.Position,second=b.Type==LightType::Directional?b.Direction:b.Position;
        return first.X==second.X && first.Y==second.Y && first.Z==second.Z;
    }
    bool BuildHairContactControl(const Matrix4& view,const HairLightSamples& lights,bool requested,
        const HairContactShadowState& state,bool resultReady,std::uint32_t samples,ShaderUInt4& output,std::string& error)
    {
        output={}; error.clear(); if (!requested) return true;
        const SceneLight* source=nullptr; if (!ResolveHairShadowSource(lights,source,error)) return false;
        if (!source) return true;
        if (!state.Generated || !state.Active || !resultReady || view.Values!=state.ViewProjection.Values ||
            !SameHairContactSource(*source,state.Source) || (samples!=1 && samples!=2 && samples!=4 && samples!=8))
        { error="Fiber contact receiver requires current visibility from its frozen camera, selected emitter and sample layers"; return false; }
        output={1,samples,0,0}; return true;
    }
    bool BuildHairShadowReceiverConstants(const HairShadowReceiverRequest& request,const HairLightSamples& lights,
        const CascadedShadowState& cascades,const Techniques::ShadowFilterSettings& requested,
        const std::array<HairShadowMapReadiness,HairShadowMapCount>& maps,HairShadowReceiverConstants& output,std::string& error)
    {
        output={}; error.clear(); const auto filter=Techniques::SanitizeShadowFilterSettings(requested);
        HairShadowReceiverConstants candidate;
        candidate.ReceiverBias={HairShadowConstantBias,HairShadowSlopeBias,HairShadowMaximumBias,HairShadowMinimumCosine};
        candidate.Composition={HairShadowStrength,0,0,0}; candidate.Control={static_cast<std::uint32_t>(filter.Mode),0,0,0};
        candidate.Cascades=BuildCascadedShadowConstants({},filter);
        const auto inactive=[&](){output=candidate; return true;};
        if (!request.SampleShadows) return inactive();
        const SceneLight* source=nullptr; if (!ResolveHairShadowSource(lights,source,error)) return false;
        if (!source) return inactive(); const auto& light=*source;
        if (!ValidCascades(cascades) || (!cascades.Frame.CascadeCount && !ValidProjection(request.LightViewProjection)))
        { error="Hair shadow receiver requires finite ordered cascades or a nonsingular single projection"; return false; }
        if (cascades.Frame.CascadeCount && light.Type!=LightType::Directional)
        { error="Hair cascades require a directional selected light"; return false; }
        const auto first=cascades.Frame.CascadeCount?1u:0u;
        const auto count=cascades.Frame.CascadeCount?cascades.Frame.CascadeCount:1u;
        for (unsigned index=first;index<first+count;++index)
        {
            const auto& map=maps[index];
            if (!map.Width || !map.Height || !map.DepthReady || (filter.Mode==Techniques::ShadowFilterMode::VSM && !map.MomentsReady) ||
                (first && (map.Width!=std::uint32_t(cascades.MapResolution) || map.Height!=std::uint32_t(cascades.MapResolution))))
            { error="Hair shadow receiver is missing a current depth/moment map or matching cascade extent"; return false; }
            candidate.MapExtents[index]={float(map.Width),float(map.Height),1.0f/map.Width,1.0f/map.Height};
            candidate.Control.Y|=1u<<index;
        }
        // Existing CPU metric builders consume one compatibility light. Project
        // the selected full-scene source explicitly; never truncate then reselect.
        LightingState selectedLight; selectedLight.LightCount=1; selectedLight.ShadowLightIndex=0; selectedLight.Lights[0]=light;
        candidate.LightViewProjection=request.LightViewProjection; candidate.Cascades=BuildCascadedShadowConstants(cascades,filter);
        if (filter.Mode==Techniques::ShadowFilterMode::PCSS &&
            !BuildPCSSConstants(request.LightViewProjection,selectedLight,cascades,filter.PCSS,true,candidate.Soft,error)) return false;
        if (filter.Mode==Techniques::ShadowFilterMode::VSM &&
            !BuildVSMConstants(request.LightViewProjection,selectedLight,cascades,filter.VSM,true,candidate.Variance,error)) return false;
        candidate.Control.Z=1; output=candidate; return true;
    }
}
