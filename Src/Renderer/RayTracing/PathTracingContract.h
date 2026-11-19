#pragma once

#include "Renderer/FrameHistory.h"
#include "Renderer/RayTracing/RaySceneRadianceFrame.h"
#include "Maths/Matrix4.h"
#include "Maths/ColorSpace.h"
#include "Techniques/PathTracing.h"
#include <limits>

namespace Evans::Renderer
{
    // Camera rays and every surface bounce use the scene BVH. A GGX/Lambertian
    // mixture is weighted by the full mixture PDF; explicit analytic lights use
    // next-event estimation, while mesh emission is sampled by path hits. Those
    // emitter sets are disjoint, so their contributions are not averaged as MIS.
    // Analytic rectangle lights are lighting sources, not visible scene geometry.
    // Russian roulette preserves expectation within the configured depth bound.
    // PBRT 4e, A Better Path Tracer; Heitz, Sampling the GGX Distribution of
    // Visible Normals (2018): https://jcgt.org/published/0007/04/01/
    // https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer
    //
    // Trace writes an RGBA32F running mean plus R32F center-ray depth. Compose
    // copies that estimate to HDR and writes opaque depth. Mean alpha estimates
    // primary coverage; all misses sample the constant Background environment.
    // No raster color, screen-space hit lookup, ambient term or denoiser is used.
    // The native pipeline must exclude duplicate GI/reflection/AO integration,
    // raster MSAA and TAA. Thin opacity and dielectric reflection/refraction are
    // evaluated by the same BVH path integrator; participating media are separate.
    struct PathTracingConstants
    {
        Matrix4 InverseViewProjection,ViewProjection;
        Float4 Viewport; // Pixel extent XY, reciprocal extent ZW.
        Float4 Trace; // Segment distance, origin bias, directional shadow extent, reserved.
        Float4 Background; // Linear primary-miss RGB and coverage alpha.
        Float4 Material; // Per-scene roughness, metallic, reserved.
        ShaderUInt4 SceneCounts,ShadingCounts;
        ShaderUInt4 Sampling; // Committed sample count, new samples, surface depth, seed.
        ShaderUInt4 Options; // PathMaterial, surface flags, direct-light visibility, reserved.
    };
    static_assert(sizeof(PathTracingConstants)==256 && offsetof(PathTracingConstants,Sampling)==224 &&
        offsetof(PathTracingConstants,Options)==240,"Path-tracing uniform ABI changed");

    struct PathTracingIdentity
    {
        Matrix4 ViewProjection;
        RayRadianceFrame Scene;
        Float4 Background={0,0,0,1};
        float Roughness=0.5f,Metallic=0;
        int Width=0,Height=0;
        std::uint64_t Epoch=0,ResetSerial=0;
        bool DirectShadows=true;
    };
    inline bool SamePathTracingIdentity(const PathTracingIdentity& a,const PathTracingIdentity& b) noexcept
    {
        return a.ViewProjection.Values==b.ViewProjection.Values && a.Scene.Geometry==b.Scene.Geometry &&
            a.Scene.Shading==b.Scene.Shading && a.Scene.Lights==b.Scene.Lights &&
            a.Background.X==b.Background.X && a.Background.Y==b.Background.Y && a.Background.Z==b.Background.Z && a.Background.W==b.Background.W &&
            a.Roughness==b.Roughness && a.Metallic==b.Metallic && a.Width==b.Width && a.Height==b.Height && a.Epoch==b.Epoch &&
            a.ResetSerial==b.ResetSerial && a.DirectShadows==b.DirectShadows;
    }
    struct PathTracingSamplePlan
    {
        std::uint32_t PreviousSamples=0,NewSamples=0;
        unsigned ReadIndex=0,WriteIndex=1;
    };
    // Color history is transactional. Resets render into the pending slot too;
    // an aborted camera/material edit must never destroy the committed estimate.
    // Changing display tone mapping does not reset linear radiance. A reduced
    // budget below the current count starts over; a larger budget keeps samples.
    class PathTracingHistory
    {
    public:
        bool Begin(const PathTracingIdentity& identity,Techniques::PathTracingSettings settings) noexcept
        {
            if (Pending) return false;
            settings=Techniques::SanitizePathTracingSettings(settings);
            const bool reuse=Valid && settings.Accumulate && Samples<=settings.AccumulationLimit &&
                SamePathTracingIdentity(CommittedIdentity,identity) && Techniques::SamePathTransport(CommittedSettings,settings);
            PendingIdentity=identity; PendingSettings=settings;
            Plan.PreviousSamples=reuse ? Samples : 0;
            Plan.NewSamples=std::min(settings.SamplesPerFrame,settings.AccumulationLimit-Plan.PreviousSamples);
            Plan.ReadIndex=ReadIndex; Plan.WriteIndex=1u-ReadIndex;
            Pending=true; Resolved=false; return true;
        }
        const PathTracingSamplePlan& GetPlan() const noexcept { return Plan; }
        bool MarkResolved() noexcept
        { if (!Pending || Resolved) return false; Resolved=true; return true; }
        bool Commit() noexcept
        {
            if (!Pending || !Resolved) return false;
            CommittedIdentity=std::move(PendingIdentity); CommittedSettings=PendingSettings;
            Samples=Plan.PreviousSamples+Plan.NewSamples; ReadIndex=Plan.WriteIndex;
            Valid=true; Pending=Resolved=false; return true;
        }
        void Abort() noexcept { Pending=Resolved=false; PendingIdentity={}; }
        void Invalidate() noexcept
        { Abort(); CommittedIdentity={}; Valid=false; Samples=0; ReadIndex=0; Plan={}; }
        std::uint32_t SampleCount() const noexcept { return Valid ? Samples : 0; }
    private:
        PathTracingIdentity CommittedIdentity,PendingIdentity;
        Techniques::PathTracingSettings CommittedSettings,PendingSettings;
        PathTracingSamplePlan Plan;
        std::uint32_t Samples=0;
        unsigned ReadIndex=0;
        bool Valid=false,Pending=false,Resolved=false;
    };

    inline bool BuildPathTracingConstants(const PathTracingIdentity& identity,Techniques::PathTracingSettings requested,
        const PathTracingSamplePlan& plan,PathTracingConstants& output)
    {
        const auto settings=Techniques::SanitizePathTracingSettings(requested);
        if (identity.Width<=0 || identity.Height<=0 || !FrameHistory::IsFinite(identity.ViewProjection) ||
            !std::isfinite(identity.Roughness) || !std::isfinite(identity.Metallic) ||
            !std::isfinite(identity.Background.X) || !std::isfinite(identity.Background.Y) ||
            !std::isfinite(identity.Background.Z) || !std::isfinite(identity.Background.W) ||
            plan.PreviousSamples>settings.AccumulationLimit || plan.NewSamples>settings.SamplesPerFrame ||
            plan.NewSamples>settings.AccumulationLimit-plan.PreviousSamples || (!plan.PreviousSamples && !plan.NewSamples) ||
            plan.ReadIndex>1 || plan.WriteIndex!=1u-plan.ReadIndex) return false;
        const auto& frame=identity.Scene;
        if (!frame.Geometry || !frame.Shading || !frame.Lights) return false;
        const auto& geometry=*frame.Geometry; const auto& shading=*frame.Shading;
        if (geometry.Nodes.size()>UINT32_MAX || geometry.Triangles.size()>UINT32_MAX || geometry.Materials.size()>UINT32_MAX ||
            geometry.CoverageTexels.size()>UINT32_MAX || shading.Surfaces.size()!=geometry.Triangles.size() ||
            shading.Materials.size()!=geometry.Materials.size() || shading.Texels.size()>UINT32_MAX ||
            frame.Lights->GetShaderLights().size()>UINT32_MAX || (shading.Flags&~RaySceneFlatNormals)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(identity.ViewProjection.Values).Inverse(inverse)) return false;
        double diagonal=0;
        if (!geometry.Nodes.empty())
        {
            const auto& root=geometry.Nodes.front();
            diagonal=std::hypot(double(root.Maximum.X)-root.Minimum.X,double(root.Maximum.Y)-root.Minimum.Y,double(root.Maximum.Z)-root.Minimum.Z);
        }
        const double extent=diagonal+double(settings.NormalBias)*2;
        if (!std::isfinite(extent) || extent>std::numeric_limits<float>::max()) return false;
        PathTracingConstants candidate;
        std::copy(inverse.Data(),inverse.Data()+candidate.InverseViewProjection.Values.size(),candidate.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(candidate.InverseViewProjection)) return false;
        candidate.ViewProjection=identity.ViewProjection;
        candidate.Viewport={float(identity.Width),float(identity.Height),1.0f/identity.Width,1.0f/identity.Height};
        candidate.Trace={settings.MaximumDistance,settings.NormalBias,float(extent),0};
        candidate.Background={std::max(identity.Background.X,0.0f),std::max(identity.Background.Y,0.0f),
            std::max(identity.Background.Z,0.0f),std::clamp(identity.Background.W,0.0f,1.0f)};
        constexpr float MinimumRoughness=0.045f;
        candidate.Material={std::clamp(identity.Roughness,MinimumRoughness,1.0f),std::clamp(identity.Metallic,0.0f,1.0f),0,0};
        candidate.SceneCounts={std::uint32_t(geometry.Nodes.size()),std::uint32_t(geometry.Triangles.size()),
            std::uint32_t(geometry.Materials.size()),std::uint32_t(geometry.CoverageTexels.size())};
        candidate.ShadingCounts={std::uint32_t(shading.Surfaces.size()),std::uint32_t(shading.Materials.size()),
            std::uint32_t(shading.Texels.size()),std::uint32_t(frame.Lights->GetShaderLights().size())};
        candidate.Sampling={plan.PreviousSamples,plan.NewSamples,settings.MaximumDepth,settings.Seed};
        candidate.Options={std::uint32_t(settings.Material),shading.Flags,identity.DirectShadows ? 1u : 0u,0};
        output=candidate; return true;
    }

    // Resource allocation and command recording stay in each native backend.
    // This shared transaction only captures frame identity and publication rules.
    struct PathTracingFrameState
    {
        PathTracingHistory History;
        bool Active=false,Resolved=false;
        void BeginFrame(bool active) noexcept
        {
            History.Abort(); Active=active; Resolved=false;
            if (!Active) History.Invalidate();
        }
        bool Prepare(const PathTracingDesc& desc,const Techniques::Settings& settings,const RayRadianceFrame& scene,
            int width,int height,PathTracingConstants& constants)
        {
            if (!Active || Resolved) return false;
            PathTracingIdentity identity; identity.ViewProjection=desc.ViewProjection; identity.Scene=scene;
            identity.Background={Maths::SRGBToLinear(desc.DisplayBackground.X),Maths::SRGBToLinear(desc.DisplayBackground.Y),
                Maths::SRGBToLinear(desc.DisplayBackground.Z),desc.DisplayBackground.W};
            identity.Roughness=settings.Roughness; identity.Metallic=settings.Metallic;
            identity.Width=width; identity.Height=height; identity.Epoch=desc.Epoch; identity.ResetSerial=desc.ResetSerial;
            identity.DirectShadows=desc.DirectShadows;
            if (!History.Begin(identity,settings.PathTracing)) return false;
            if (BuildPathTracingConstants(identity,settings.PathTracing,History.GetPlan(),constants)) return true;
            History.Abort(); return false;
        }
        bool Resolve(FrameStatistics& statistics) noexcept
        {
            if (!History.MarkResolved()) return false;
            Resolved=true; statistics.PathTracingPassCount=2;
            statistics.PathTracingSampleCount=History.GetPlan().PreviousSamples+History.GetPlan().NewSamples;
            return true;
        }
        void Commit() noexcept { if (Active && Resolved) History.Commit(); }
        void Abort() noexcept { History.Abort(); Resolved=false; }
    };
}
