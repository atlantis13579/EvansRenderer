#pragma once

#include "Renderer/Hair/HairLighting.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t HairShadowMapCount=MaximumShadowCascades+1;
    inline constexpr float HairShadowConstantBias=0.00035f,HairShadowSlopeBias=0.00060f;
    inline constexpr float HairShadowMaximumBias=0.0045f,HairShadowMinimumCosine=0.15f,HairShadowStrength=0.55f;
    // Slot zero is the single map; slots one through four are cascades. Native
    // adapters report resources from this frame, never a previous ready moment map.
    struct HairShadowMapReadiness
    {
        std::uint32_t Width=0,Height=0;
        bool DepthReady=false,MomentsReady=false;
    };
    struct HairShadowReceiverRequest
    {
        Matrix4 LightViewProjection;
        bool SampleShadows=false;
    };
    struct HairContactShadowState
    {
        Matrix4 ViewProjection;
        SceneLight Source;
        bool Generated=false,Active=false;
    };
    // Resolve the complete selected emitter and verify its expanded sample span.
    // An unlit/non-casting packet succeeds with a null source.
    bool ResolveHairShadowSource(const HairLightSamples& lights,const SceneLight*& source,std::string& error);
    bool SameHairContactSource(const SceneLight& a,const SceneLight& b);
    bool BuildHairContactControl(const Matrix4& view,const HairLightSamples& lights,bool requested,
        const HairContactShadowState& state,bool resultReady,std::uint32_t samples,ShaderUInt4& output,std::string& error);
    struct HairShadowReceiverConstants
    {
        Matrix4 LightViewProjection;
        CascadedShadowConstants Cascades;
        PCSSConstants Soft;
        VSMConstants Variance;
        std::array<Float4,HairShadowMapCount> MapExtents; // Width, height, reciprocal width/height.
        Float4 ReceiverBias; // Constant, slope, maximum, minimum absolute cosine.
        ShaderUInt4 Control; // Filter mode, current map bits, enabled, reserved.
        Float4 Composition; // Shadow strength, reserved, reserved, reserved.
    };
    static_assert(sizeof(HairShadowReceiverConstants)==752 && offsetof(HairShadowReceiverConstants,Cascades)==64 &&
        offsetof(HairShadowReceiverConstants,Soft)==400 && offsetof(HairShadowReceiverConstants,Variance)==512 &&
        offsetof(HairShadowReceiverConstants,MapExtents)==624 && offsetof(HairShadowReceiverConstants,Control)==720,
        "Hair shadow receiver ABI changed");

    // Build a standalone receiver block from the frozen complete light packet.
    // PCSS and VSM retain their existing orthographic/perspective light-axis
    // metrics. Every required depth/moment map must belong to the current frame.
    // Failure clears enable bits, so a partially validated map set cannot bind.
    bool BuildHairShadowReceiverConstants(const HairShadowReceiverRequest& request,const HairLightSamples& lights,
        const CascadedShadowState& cascades,const Techniques::ShadowFilterSettings& filter,
        const std::array<HairShadowMapReadiness,HairShadowMapCount>& maps,
        HairShadowReceiverConstants& output,std::string& error);
}
