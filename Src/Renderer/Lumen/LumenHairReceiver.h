#pragma once

#include "Renderer/Lumen/LumenTranslucency.h"

namespace Evans::Renderer
{
    // A view-independent low-frequency receiver, separate from direct fiber
    // scattering. The normalized projected-cylinder kernel is
    // sqrt(1-dot(t,w)^2) / pi^2. Its SH eigenvalues through band two are
    // {1,0,-1/8}; the incident field is stored after cosine convolution, so
    // evaluate band zero / pi and band two * (-1/(2*pi)). Odd bands vanish.
    // This is an ambient transport approximation, not full hair multiple scattering.
    struct alignas(16) LumenHairReceiverFrame
    {
        LumenTranslucencyFrame Volume;
        Float4 RootTint,TipTint; // Root W: fiber intensity; tip W reserved.
        Float4 Ambient; // Fallback ambient RGB, including its scalar strength; W=0.
        Float4 ClipPlane;
    };
    static_assert(sizeof(LumenHairReceiverFrame)==240 && offsetof(LumenHairReceiverFrame,RootTint)==176,
        "Lumen fiber receiver uniform ABI changed");

    // The primary draw defers ambient when this receiver is requested. A private
    // HDR target stores the replacement indirect component, including fallback.
    // Read-only primary depth and identical ribbon projection preserve per-sample
    // ownership. Resolve this target before composing it with post-processed HDR.
    // Rejected support retains the matching ambient fraction, without bright
    // probe renormalization. Failure leaves output unchanged.
    bool BuildLumenHairReceiverFrame(const LumenTranslucencyFrame&,
        const Techniques::HairFurSettings&,const LightingState&,Float4 clipPlane,
        LumenHairReceiverFrame&,std::string& error);
}
