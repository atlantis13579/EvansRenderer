#include "Renderer/Renderer.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    bool Renderer::PrepareVirtualShadows()
    {
        DiscardVirtualShadows();
        if (m_Techniques.Shadows!=Techniques::ShadowRepresentation::Virtual || !m_ShadowsEnabled ||
            m_RenderMode==RenderMode::Wireframe || Techniques::UsesPathTracing(m_Techniques.Pipeline)) return true;
        const auto source=m_RenderLighting.ShadowLightIndex;
        if (source<0 || unsigned(source)>=std::min(m_RenderLighting.LightCount,MaximumSceneLights) ||
            !m_RenderLighting.Lights[source].CastShadows) return true;

        VirtualShadowReceiverUpload receivers;
        auto& frame=receivers.Frame; frame.CameraViewProjection=m_RasterViewProjection; frame.LightViewProjection=m_LightViewProjection;
        frame.Settings=m_Techniques.VirtualShadows; frame.FilterRadius=m_Techniques.ShadowFilter.Mode==Techniques::ShadowFilterMode::Hard?0:m_Techniques.ShadowFilter.Radius;
        frame.Width=unsigned(m_RenderWidth); frame.Height=unsigned(m_RenderHeight);
        frame.OutputWidth=unsigned(m_CreateInfo.Width); frame.OutputHeight=unsigned(m_CreateInfo.Height);
        // Demand uses the actual jittered camera. Very large outputs use a
        // bounded receiver grid while retaining final-output pixel footprints.
        const double pixels=double(frame.Width)*frame.Height;
        if (pixels>VirtualShadowMaximumReceiverPixels)
        {
            const double scale=std::sqrt(VirtualShadowMaximumReceiverPixels/pixels);
            frame.Width=std::max(1u,unsigned(std::floor(frame.Width*scale)));
            frame.Height=std::max(1u,unsigned(std::floor(frame.Height*scale)));
        }
        std::vector<VirtualShadowSceneCaster> casters;
        for (const auto& entry:m_Meshes)
        {
            const auto& mesh=entry.second;
            if (!mesh.Mesh.Visible || !mesh.Geometry || !mesh.SelectedIndexCount() || UsesPrivateTransparency(mesh)) continue;
            PipelineState material; ConfigureMaterialPipeline(mesh,material);
            VirtualShadowSceneCaster caster; caster.Key=entry.first; caster.GeometryRevision=mesh.Geometry->ShadowRevision;
            caster.PreparationRevision=m_PreparedGeometry?m_PreparedGeometry->Serial:0;
            caster.BoundsMinimum=mesh.BoundsMinimum; caster.BoundsMaximum=mesh.BoundsMaximum;
            auto& draw=caster.Draw; draw.Vertices=mesh.Geometry->VertexBuffer;
            draw.Indices=mesh.SelectedLod?mesh.Geometry->LodIndexBuffers[mesh.SelectedLod-1]:mesh.Geometry->IndexBuffer;
            draw.FirstIndex=mesh.SelectedFirstIndex(); draw.IndexCount=mesh.SelectedIndexCount(); draw.World=mesh.Mesh.WorldTransform;
            if (mesh.Mesh.Vegetation)
            {
                draw.Wind=m_VegetationWind; draw.Wind.Z=float(m_VegetationTime);
                draw.Gust=m_VegetationGust; draw.Gust.W=float(mesh.Mesh.Vegetation);
                draw.VegetationHistory={0,mesh.Mesh.Vegetation==1?1.0f:14.0f,0,0};
            }
            draw.Diffuse=material.DiffuseTexture; draw.Opacity=material.OpacityTexture;
            draw.Coverage=std::clamp(material.MaterialOpacity,0.0f,1.0f); draw.Cutoff=material.AlphaCutoff;
            draw.Dithered=mesh.Translucent && m_Techniques.AlphaBlending; draw.BackFaceCulling=m_Techniques.BackFaceCulling;
            // Keep off-camera casters. Camera rasterization rejects hidden
            // receivers; light-page rasterization clips every candidate caster.
            receivers.Receivers.push_back(draw);
            if (mesh.Mesh.CastShadow) casters.push_back(std::move(caster));
        }
        VirtualShadowSceneUpdate scene;
        if (!BuildVirtualShadowSceneUpdate(m_LightViewProjection,std::move(casters),
            m_VirtualShadowPublished.Handle?&m_VirtualShadowScene:nullptr,scene,m_LastError)) return false;
        VirtualShadowRequestResult demand;
        if (!m_RHI->BuildVirtualShadowReceiverRequests(receivers,demand)) return Fail(m_RHI->GetLastError().c_str());
        VirtualShadowAtlasUpload upload; upload.Identity=scene.Scene.Identity; upload.Settings=m_Techniques.VirtualShadows;
        upload.Requests=std::move(demand.Pages); upload.Invalidations=scene.Invalidations; upload.Readback=false;
        for (const auto& caster:scene.Scene.Casters) upload.Casters.push_back(caster.Draw);
        VirtualShadowAtlasResult candidate;
        if (!m_RHI->RenderVirtualShadowAtlas(upload,m_VirtualShadowPublished.Handle,candidate)) return Fail(m_RHI->GetLastError().c_str());
        m_VirtualShadowRequestedPages=unsigned(std::count(upload.Requests.begin(),upload.Requests.end(),1u));
        m_VirtualShadowCandidate=std::move(candidate); m_VirtualShadowCandidateScene=std::move(scene); m_VirtualShadowFrameActive=true;
        VirtualShadowSamplingFrame sampling; sampling.Atlas=m_VirtualShadowCandidate.Handle;
        sampling.MipBias=m_Techniques.VirtualShadows.MipBias;
        sampling.OutputPixelScale={float(m_RenderWidth)/m_CreateInfo.Width,float(m_RenderHeight)/m_CreateInfo.Height};
        m_RHI->SetVirtualShadowSamplingFrame(sampling); return true;
    }
    void Renderer::DiscardVirtualShadows()
    {
        if (m_RHI)
        {
            m_RHI->SetVirtualShadowSamplingFrame({});
            if (m_VirtualShadowCandidate.Handle) m_RHI->DestroyVirtualShadowAtlas(m_VirtualShadowCandidate.Handle);
        }
        m_VirtualShadowCandidate={}; m_VirtualShadowCandidateScene={}; m_VirtualShadowFrameActive=false; m_VirtualShadowRequestedPages=0;
    }
    void Renderer::CommitVirtualShadows()
    {
        if (!m_VirtualShadowCandidate.Handle) return;
        const auto previous=m_VirtualShadowPublished.Handle;
        m_VirtualShadowPublished=std::move(m_VirtualShadowCandidate); m_VirtualShadowCandidate={};
        m_VirtualShadowScene=std::move(m_VirtualShadowCandidateScene.Scene); m_VirtualShadowCandidateScene={};
        if (previous) m_RHI->DestroyVirtualShadowAtlas(previous);
    }
    void Renderer::DestroyVirtualShadows()
    {
        DiscardVirtualShadows();
        if (m_RHI && m_VirtualShadowPublished.Handle) m_RHI->DestroyVirtualShadowAtlas(m_VirtualShadowPublished.Handle);
        m_VirtualShadowPublished={}; m_VirtualShadowScene={};
    }
    void Renderer::CaptureVirtualShadowStatistics(FrameStatistics& statistics) const
    {
        if (!m_VirtualShadowFrameActive) return;
        statistics.VirtualShadowSerial=m_VirtualShadowPublished.Info.Identity.Serial;
        statistics.VirtualShadowRequestedPages=m_VirtualShadowRequestedPages;
        statistics.VirtualShadowResidentPages=unsigned(m_VirtualShadowPublished.Info.Residents.size());
        statistics.VirtualShadowRenderedPages=m_VirtualShadowPublished.RenderedPages;
        statistics.VirtualShadowCopiedPages=m_VirtualShadowPublished.CopiedPages;
        statistics.VirtualShadowDrawCalls=m_VirtualShadowPublished.DrawCalls;
    }
}
