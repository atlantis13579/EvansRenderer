#include "Techniques/LightRange.h"
#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/Renderer.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/Geometry/TriangleSubdivision.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Geometry/Instancing.h"
#include "Renderer/Geometry/FrustumCulling.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/RayTracing/SceneRayQuery.h"
#include "Renderer/Nanite/NaniteAsset.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <functional>
#include <map>
#include <cstring>
#include <utility>
#include <unordered_set>

namespace Evans::Renderer
{
	namespace
	{
		Float3 Add(const Float3& a, const Float3& b) { return { a.X + b.X, a.Y + b.Y, a.Z + b.Z }; }
		Float3 Subtract(const Float3& a, const Float3& b) { return { a.X - b.X, a.Y - b.Y, a.Z - b.Z }; }
		Float3 Multiply(const Float3& value, float scalar) { return { value.X * scalar, value.Y * scalar, value.Z * scalar }; }
		float Dot(const Float3& a, const Float3& b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
		Float3 Cross(const Float3& a, const Float3& b)
		{
			return { a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X };
		}
		Float3 Normalize(const Float3& value)
		{
			const float length = std::sqrt(Dot(value, value));
			return length > 1.0e-7f ? Multiply(value, 1.0f / length) : Float3 { 0.0f, 0.0f, 1.0f };
		}

		Matrix4 Multiply(const Matrix4& a, const Matrix4& b)
		{
			Matrix4 result;
			result.Values.fill(0.0f);
			for (int row = 0; row < 4; ++row)
			{
				for (int column = 0; column < 4; ++column)
				{
					for (int inner = 0; inner < 4; ++inner)
					{
						result.Values[static_cast<std::size_t>(row) * 4 + column]
							+= a.Values[static_cast<std::size_t>(row) * 4 + inner]
							* b.Values[static_cast<std::size_t>(inner) * 4 + column];
					}
				}
			}
			return result;
		}

		Float3 TransformPoint(const Matrix4& matrix, const Float3& point)
		{
			const float x = matrix.Values[0] * point.X + matrix.Values[1] * point.Y
				+ matrix.Values[2] * point.Z + matrix.Values[3];
			const float y = matrix.Values[4] * point.X + matrix.Values[5] * point.Y
				+ matrix.Values[6] * point.Z + matrix.Values[7];
			const float z = matrix.Values[8] * point.X + matrix.Values[9] * point.Y
				+ matrix.Values[10] * point.Z + matrix.Values[11];
			const float w = matrix.Values[12] * point.X + matrix.Values[13] * point.Y
				+ matrix.Values[14] * point.Z + matrix.Values[15];
			return std::fabs(w) > 1.0e-7f ? Float3 { x / w, y / w, z / w } : Float3 { x, y, z };
		}

		Matrix4 LookAt(const Float3& eye, const Float3& target, const Float3& up)
		{
			const Float3 backward = Normalize(Subtract(eye, target));
			const Float3 right = Normalize(Cross(up, backward));
			const Float3 correctedUp = Cross(backward, right);
			Matrix4 result;
			result.Values = {
				right.X, right.Y, right.Z, -Dot(right, eye),
				correctedUp.X, correctedUp.Y, correctedUp.Z, -Dot(correctedUp, eye),
				backward.X, backward.Y, backward.Z, -Dot(backward, eye),
				0.0f, 0.0f, 0.0f, 1.0f };
			return result;
		}

		Matrix4 Perspective(float verticalFov, float aspect, float nearPlane, float farPlane)
		{
			Matrix4 result;
			result.Values.fill(0.0f);
			const float cotangent = 1.0f / std::tan(verticalFov * 0.5f);
			result.Values[0] = cotangent / aspect;
			result.Values[5] = cotangent;
			result.Values[10] = farPlane / (nearPlane - farPlane);
			result.Values[11] = farPlane * nearPlane / (nearPlane - farPlane);
			result.Values[14] = -1.0f;
			return result;
		}

		Matrix4 PerspectiveOffCenter(
			float left,
			float right,
			float bottom,
			float top,
			float nearPlane,
			float farPlane)
		{
			Matrix4 result;
			result.Values.fill(0.0f);
			result.Values[0] = 2.0f * nearPlane / (right - left);
			result.Values[2] = (right + left) / (right - left);
			result.Values[5] = 2.0f * nearPlane / (top - bottom);
			result.Values[6] = (top + bottom) / (top - bottom);
			result.Values[10] = farPlane / (nearPlane - farPlane);
			result.Values[11] = farPlane * nearPlane / (nearPlane - farPlane);
			result.Values[14] = -1.0f;
			return result;
		}

		Matrix4 OrthographicBounds(const Float3& minimum, const Float3& maximum)
		{
			const float width = std::max(maximum.X - minimum.X, 1.0e-4f);
			const float height = std::max(maximum.Y - minimum.Y, 1.0e-4f);
			const float depth = std::max(maximum.Z - minimum.Z, 1.0e-4f);
			Matrix4 result;
			result.Values = {
				2.0f / width, 0.0f, 0.0f, -(maximum.X + minimum.X) / width,
				0.0f, 2.0f / height, 0.0f, -(maximum.Y + minimum.Y) / height,
				// Right-handed light-view space looks down -Z. The nearest bound is
				// therefore maximum.Z and must map to depth 0 for LESS testing.
				0.0f, 0.0f, -1.0f / depth, maximum.Z / depth,
				0.0f, 0.0f, 0.0f, 1.0f };
			return result;
		}


	}

#include "Renderer/Geometry/GeometryPreparation.inl"
#include "Renderer/VirtualGeometry/VirtualGeometryPreparation.inl"
#include "Renderer/Nanite/NanitePreparation.inl"

	Renderer::Renderer() = default;

	Renderer::Renderer(std::unique_ptr<IRHI> rhi)
		: m_RHI(std::move(rhi))
	{
	}

	Renderer::~Renderer()
	{
		Shutdown();
	}

	bool Renderer::Initialize(const RendererCreateInfo& createInfo)
	{
		if (m_Initialized)
		{
			return Fail("Renderer is already initialized");
		}
		if (createInfo.Width <= 0 || createInfo.Height <= 0 || createInfo.ShadowMapSize <= 0)
		{
			return Fail("Renderer dimensions must be positive");
		}

		if (!m_RHI)
		{
			m_RHI = CreateRHI(createInfo.Backend);
		}
		if (!m_RHI)
		{
			return Fail("Requested RHI backend is unavailable");
		}

		RHICreateInfo rhiCreateInfo;
		rhiCreateInfo.NativeWindow = createInfo.NativeWindow;
		rhiCreateInfo.Width = createInfo.Width;
		rhiCreateInfo.Height = createInfo.Height;
		rhiCreateInfo.CaptureDiagnosticBuffers = createInfo.CaptureDiagnosticBuffers;
		if (!m_RHI->Initialize(rhiCreateInfo))
		{
			m_LastError = m_RHI->GetLastError();
			return false;
		}

		TextureDesc shadowMapDesc;
		shadowMapDesc.Width = createInfo.ShadowMapSize;
		shadowMapDesc.Height = createInfo.ShadowMapSize;
		shadowMapDesc.Format = TextureFormat::Depth32;
		shadowMapDesc.RenderTarget = true;
		shadowMapDesc.DebugName = "Renderer.ShadowMap";
		m_ShadowMap = m_RHI->CreateTexture(shadowMapDesc, nullptr);
		if (!m_ShadowMap)
		{
			m_LastError = m_RHI->GetLastError();
			m_RHI->Shutdown();
			return false;
		}

		m_CreateInfo = createInfo;
        m_RenderWidth=createInfo.Width; m_RenderHeight=createInfo.Height;
		m_AutomaticViewProjection = createInfo.AutomaticViewProjection;
		m_AutomaticLightViewProjection = createInfo.AutomaticViewProjection;
		m_Initialized = true;
		SetTechniqueSettings(m_Techniques);
		m_LastFrameCapture = {};
		m_LastError.clear();
		return true;
	}

	bool Renderer::Initialize(const RendererCreateInfo& createInfo, std::unique_ptr<IRHI> rhi)
	{
		if (!rhi)
		{
			return Fail("A custom RHI must not be null");
		}
		Shutdown();
		m_RHI = std::move(rhi);
		return Initialize(createInfo);
	}

	void Renderer::Shutdown()
	{
        m_LumenGlobalDistance.reset(); m_LumenWorldCache.reset(); m_LumenGI.reset(); if (m_RHI) { m_RHI->SetLumenLighting({}); m_RHI->SetLumenWorldCache({}); m_RHI->SetLumenGlobalDistance({}); }
        m_VisibilitySources.clear(); m_VisibilityScene.reset(); m_VisibilitySceneRevision = m_VisibilityGeometryViewSerial = 0;
		ResetFrameHistory();
		m_NextSampleTimeSeconds.reset();
		if (m_RHI && m_RHI->IsInitialized())
		{
			ClearScene();
            DestroyVirtualShadows();
			for (auto texture : m_CascadeMaps) if (texture) m_RHI->DestroyTexture(texture);
			m_RHI->SetCascadedShadowState({});
			if (m_ShadowMap)
			{
				m_RHI->DestroyTexture(m_ShadowMap);
			}
			m_RHI->Shutdown();
		}
		else
		{
			m_Meshes.clear();
            m_PendingNaniteGeometry.reset(); m_NaniteGeometry.reset(); m_VirtualGeometry.reset(); m_PreparedGeometry.reset();
            m_RefractionGeometry.Reset();
            if (m_RHI) m_RHI->SetRefractionFrame({});
        ++m_ProbeSceneRevision; m_BakedProbes = {};
        m_BakedLightmaps.Reset(); m_LightmapStatus={}; m_LightmapInstances.clear(); m_LightmapSourceRevision=0;
        m_VoxelGI.Reset(); m_VoxelStatus={}; m_VoxelInstances.clear(); m_VoxelSourceRevision=0; m_BoundVoxelVolume.reset();
        m_HairGeometry.Reset(); m_HairStatus={}; m_HairInstances.clear(); m_HairSourceRevision=0; m_BoundHairGeometry.reset(); m_HairLighting.Reset();
        m_HairInstanceIds.clear(); m_HairLastPose.clear(); m_HairPoseBase.reset(); m_HairPosedGeometry.reset();
        if (m_RHI) m_RHI->SetVoxelGIFrame({});
        if (m_RHI) m_RHI->SetHairFurFrame({});
        if (m_RHI) m_RHI->SetLightmapFrame({});
        if (m_RHI) m_RHI->SetBakedProbeSamples({});
        m_RayShadowImageSource.reset(); m_RayShadowImage.reset(); m_RayShadowFrame={};
        m_RayScene.reset(); m_RayShading.reset(); m_RayLights.reset(); m_RayLighting={}; m_RaySceneLods.clear();
        if (m_RHI) { m_RHI->SetRayScene({}); m_RHI->SetRayShadowImage({}); m_RHI->SetRayRadianceFrame({}); }
			m_GeometryCache.clear();
			m_TextureCache.clear();
		}
		m_ShadowMap = {};
        m_VirtualShadowScene={}; m_VirtualShadowCandidateScene={}; m_VirtualShadowPublished={}; m_VirtualShadowCandidate={};
        m_VirtualShadowFrameActive=false; m_VirtualShadowRequestedPages=0;
		m_CascadeMaps = {};
		m_CascadeFrame = {};
		m_CascadeCameraValid = false;
		m_LastFrameCapture = {};
		m_Initialized = false;
        m_RenderWidth=m_RenderHeight=m_PreviousPresentedWidth=m_PreviousPresentedHeight=0;
        m_PreviousPresentedJitter={}; m_PresentedSerial=0;
	}

	bool Renderer::IsInitialized() const
	{
		return m_Initialized;
	}

	RHIBackend Renderer::GetBackend() const
	{
		return m_RHI ? m_RHI->GetBackend() : m_CreateInfo.Backend;
	}

	const char* Renderer::GetBackendName() const
	{
		return m_RHI ? m_RHI->GetName() : "Uninitialized";
	}

	RHICapabilities Renderer::GetCapabilities() const
	{
		return m_RHI ? m_RHI->GetCapabilities() : RHICapabilities {};
	}

	bool Renderer::Resize(int width, int height)
	{
		return Resize(width, height, m_CreateInfo.ShadowMapSize);
	}

	bool Renderer::Resize(int width, int height, int shadowMapSize)
	{
		if (!m_Initialized)
		{
			return Fail("Renderer is not initialized");
		}
		if (width <= 0 || height <= 0 || shadowMapSize <= 0)
		{
			return Fail("Renderer dimensions must be positive");
		}
		const bool viewportChanged =
			width != m_CreateInfo.Width || height != m_CreateInfo.Height;
		const bool shadowMapChanged =
			shadowMapSize != m_CreateInfo.ShadowMapSize;
		if (!viewportChanged && !shadowMapChanged)
		{
			return true;
		}

        ShaderUInt4 extent{static_cast<unsigned>(width),static_cast<unsigned>(height),static_cast<unsigned>(width),static_cast<unsigned>(height)};
        if (Techniques::UsesTemporalSuperResolution(m_Techniques.PostProcessing) &&
            !BuildTSRRenderExtent(width,height,m_Techniques.PostProcessing.TemporalSuperResolution,extent,m_LastError)) return false;
		TextureHandle replacementShadowMap;
		if (shadowMapChanged)
		{
			TextureDesc shadowMapDesc;
			shadowMapDesc.Width = shadowMapSize;
			shadowMapDesc.Height = shadowMapSize;
			shadowMapDesc.Format = TextureFormat::Depth32;
			shadowMapDesc.RenderTarget = true;
			shadowMapDesc.DebugName = "Renderer.ShadowMap";
			replacementShadowMap =
				m_RHI->CreateTexture(shadowMapDesc, nullptr);
			if (!replacementShadowMap)
			{
				m_LastError = m_RHI->GetLastError();
				return false;
			}
		}
		if ((int(extent.X)!=m_RenderWidth || int(extent.Y)!=m_RenderHeight) && !m_RHI->Resize(int(extent.X), int(extent.Y)))
		{
			m_LastError = m_RHI->GetLastError();
			if (replacementShadowMap)
			{
				m_RHI->DestroyTexture(replacementShadowMap);
			}
			return false;
		}
		if (replacementShadowMap)
		{
			for (auto texture : m_CascadeMaps) if (texture) m_RHI->DestroyTexture(texture);
			m_CascadeMaps = {};
			m_RHI->SetCascadedShadowState({});
			const TextureHandle previousShadowMap = m_ShadowMap;
			m_ShadowMap = replacementShadowMap;
			if (previousShadowMap)
			{
				m_RHI->DestroyTexture(previousShadowMap);
			}
		}
        m_RenderWidth=int(extent.X); m_RenderHeight=int(extent.Y);
		m_CreateInfo.Width = width;
		m_CreateInfo.Height = height;
		m_CreateInfo.ShadowMapSize = shadowMapSize;
		if (viewportChanged) ResetFrameHistory();
		m_LastFrameCapture = {};
		m_LastError.clear();
		return true;
	}

	bool Renderer::AddMesh(const MeshData& mesh)
	{
		if (!m_Initialized)
		{
			return Fail("Renderer is not initialized");
		}
		if (mesh.Id.empty())
		{
			return Fail("A mesh requires an identifier and triangle-list geometry");
		}

		const auto inserted = m_Meshes.try_emplace(mesh.Id);
		if (!inserted.second)
		{
			return Fail("Mesh identifier already exists");
		}
		try
		{
			if (!CreateMeshResources(mesh, inserted.first->second))
			{
				m_Meshes.erase(inserted.first);
				return false;
			}
		}
		catch (...)
		{
			DestroyMeshResources(inserted.first->second);
			m_Meshes.erase(inserted.first);
			throw;
		}
		++m_ProbeSceneRevision;
		m_TemporalSampling.Reset();
		return true;
	}

	bool Renderer::UpdateMesh(const MeshData& mesh)
	{
		auto iterator = m_Meshes.find(mesh.Id);
		if (iterator == m_Meshes.end())
		{
			return Fail("Mesh identifier does not exist");
		}

        auto& resource=iterator->second;
        auto* geometry=resource.BaseGeometry;
        const auto& previousMaterial=resource.Mesh.Material;
        if (mesh.DynamicVertices && geometry && geometry->DynamicVertices && geometry->ReferenceCount==1 &&
            resource.Geometry==geometry && resource.SourceGeometry==geometry && !m_PreparedGeometry && !m_VirtualGeometry && !m_NaniteGeometry &&
            geometry->Vertices.size()==mesh.Vertices.size() && geometry->Indices==mesh.Indices &&
            previousMaterial.DiffuseImage==mesh.Material.DiffuseImage && previousMaterial.NormalImage==mesh.Material.NormalImage &&
            previousMaterial.OpacityImage==mesh.Material.OpacityImage &&
            previousMaterial.MetallicRoughnessImage==mesh.Material.MetallicRoughnessImage)
        {
            if (!FrameHistory::IsFinite(mesh.WorldTransform)) return Fail("Mesh world transform must be finite");
            const auto& vertices=mesh.Vertices;
            if (!m_RHI->UpdateBuffer(geometry->VertexBuffer,0,vertices.data(),vertices.size()*sizeof(Vertex)))
                return Fail(m_RHI->GetLastError().c_str());
            bool shadowChanged=false;
            bool alphaChanged=resource.Mesh.Color.W!=mesh.Color.W || previousMaterial.BaseColor.W!=mesh.Material.BaseColor.W ||
                previousMaterial.AlphaCutoff!=mesh.Material.AlphaCutoff;
            geometry->BoundsMinimum=geometry->BoundsMaximum=vertices.front().Position;
            for (std::size_t i=0;i<vertices.size();++i)
            {
                const auto& v=vertices[i]; const auto& old=geometry->Vertices[i];
                const auto p=v.Position;
                alphaChanged|=v.Color.W!=old.Color.W;
                shadowChanged|=p.X!=old.Position.X || p.Y!=old.Position.Y || p.Z!=old.Position.Z ||
                    v.TexCoord.X!=old.TexCoord.X || v.TexCoord.Y!=old.TexCoord.Y || v.Color.W!=old.Color.W;
                auto& lo=geometry->BoundsMinimum; auto& hi=geometry->BoundsMaximum;
                lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)};
                hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)};
            }
            if (shadowChanged) ++geometry->ShadowRevision;
            std::copy(vertices.begin(),vertices.end(),geometry->Vertices.begin());
            // These derived views contain vertex attributes. The stable Lumen
            // proxy deliberately survives water ripples and foliage bending.
            geometry->RefractionSource.reset(); geometry->HairSource.reset();
            for (auto buffer:geometry->LodIndexBuffers) if (buffer) m_RHI->DestroyBuffer(buffer);
            for (const auto& overlay:geometry->WireOverlays)
            { if (overlay.Vertices) m_RHI->DestroyBuffer(overlay.Vertices); if (overlay.Indices) m_RHI->DestroyBuffer(overlay.Indices); }
            geometry->LodIndexBuffers.clear(); geometry->LodChain={}; geometry->LodPrepared=false; geometry->WireOverlays.clear();
            if (alphaChanged) resource.Translucent=IsMeshTranslucent(mesh);
            resource.Mesh=MeshInstance(mesh);
            resource.BoundsMinimum=geometry->BoundsMinimum; resource.BoundsMaximum=geometry->BoundsMaximum;
            resource.BoundsCenter=Multiply(Add(resource.BoundsMinimum,resource.BoundsMaximum),0.5f);
            resource.SelectedLod=resource.PreviousLod=0; resource.PreviousTransformValid=false;
            ++m_ProbeSceneRevision; ++m_GeometryViewSerial;
            return true;
        }

		MeshResource replacement;
		if (!CreateMeshResources(mesh, replacement))
		{
			return false;
		}
		replacement.Outline = iterator->second.Outline;
        replacement.LumenKey = iterator->second.LumenKey;
		DestroyMeshResources(iterator->second);
		iterator->second = std::move(replacement);
		++m_ProbeSceneRevision;
		m_TemporalSampling.Reset();
		return true;
	}

	bool Renderer::SetWorldTransform(const std::string& id, const Matrix4& transform)
	{
		if (!FrameHistory::IsFinite(transform)) return Fail("World transform must be finite");
		auto iterator = m_Meshes.find(id);
		if (iterator == m_Meshes.end())
		{
			return Fail("Mesh identifier does not exist");
		}
		if (iterator->second.Mesh.WorldTransform.Values != transform.Values) { ++m_ProbeSceneRevision; ++m_HairPoseRevision; }
		iterator->second.Mesh.WorldTransform = transform;
		return true;
	}

	bool Renderer::SetOutline(const std::string& id, const OutlineSettings& outline)
	{
		auto iterator = m_Meshes.find(id);
		if (iterator == m_Meshes.end() || outline.Thickness < 0.0f)
		{
			return Fail("Invalid mesh outline settings");
		}
		iterator->second.Outline = outline;
		return true;
	}

	bool Renderer::RemoveMesh(const std::string& id)
	{
		auto iterator = m_Meshes.find(id);
		if (iterator == m_Meshes.end())
		{
			return Fail("Mesh identifier does not exist");
		}
		DestroyMeshResources(iterator->second);
		m_Meshes.erase(iterator);
		++m_ProbeSceneRevision;
		m_TemporalSampling.Reset();
		return true;
	}

	void Renderer::ClearScene()
	{
        SetDensityVolume({});
        ReleaseNaniteFrameVisibility();
        m_LumenGlobalDistance.reset(); m_LumenWorldCache.reset(); m_LumenGI.reset(); if (m_RHI) { m_RHI->SetLumenLighting({}); m_RHI->SetLumenWorldCache({}); m_RHI->SetLumenGlobalDistance({}); }
        m_VisibilitySources.clear(); m_VisibilityScene.reset();
		m_BatchInstances.clear(); m_BatchMesh = nullptr;
        m_PreparedMeshDraws.clear(); m_CapturingGpuPass = false; m_InstanceDraws.clear(); m_CollectInstanceDraws=false;
		ResetFrameHistory();
		if (m_RHI && m_RHI->IsInitialized())
		{
			for (auto& pair : m_Meshes)
			{
				DestroyMeshResources(pair.second);
			}
		}
		m_Meshes.clear();
        m_PendingNaniteGeometry.reset(); m_NaniteGeometry.reset(); m_VirtualGeometry.reset(); m_PreparedGeometry.reset();
        m_RefractionGeometry.Reset();
        if (m_RHI) m_RHI->SetRefractionFrame({});
        ++m_ProbeSceneRevision; m_BakedProbes = {};
        m_BakedLightmaps.Reset(); m_LightmapStatus={}; m_LightmapInstances.clear(); m_LightmapSourceRevision=0;
        m_VoxelGI.Reset(); m_VoxelStatus={}; m_VoxelInstances.clear(); m_VoxelSourceRevision=0; m_BoundVoxelVolume.reset();
        m_HairGeometry.Reset(); m_HairStatus={}; m_HairInstances.clear(); m_HairSourceRevision=0; m_BoundHairGeometry.reset(); m_HairLighting.Reset();
        m_HairInstanceIds.clear(); m_HairLastPose.clear(); m_HairPoseBase.reset(); m_HairPosedGeometry.reset();
        if (m_RHI) m_RHI->SetVoxelGIFrame({});
        if (m_RHI) m_RHI->SetHairFurFrame({});
        if (m_RHI) m_RHI->SetLightmapFrame({});
        if (m_RHI) m_RHI->SetBakedProbeSamples({});
        m_RayShadowImageSource.reset(); m_RayShadowImage.reset(); m_RayShadowFrame={};
        m_RayScene.reset(); m_RayShading.reset(); m_RayLights.reset(); m_RayLighting={}; m_RaySceneLods.clear();
        if (m_RHI) { m_RHI->SetRayScene({}); m_RHI->SetRayShadowImage({}); m_RHI->SetRayRadianceFrame({}); }
		m_GeometryCache.clear();
		m_TextureCache.clear();
	}

    void Renderer::SetDensityVolume(std::shared_ptr<const DensityVolume> source)
    {
        if (source==m_DensityVolume) return;
        if (m_DensityBinding.Texture && m_RHI && m_RHI->IsInitialized()) m_RHI->DestroyTexture(m_DensityBinding.Texture);
        m_DensityBinding={}; m_DensityVolume=std::move(source); ResetFrameHistory();
    }

    bool Renderer::PrepareDensityVolume()
    {
        if (!m_DensityVolume || !Techniques::UsesBoundedMedium(m_Techniques.VolumetricFog) || m_DensityBinding.Texture) return true;
        const auto& source=*m_DensityVolume;
        const auto width=source.AtlasWidth(),height=source.AtlasHeight();
        if (!width || !height || source.AtlasRGBA8.size()!=std::size_t(width)*height*4) return Fail("Invalid density atlas dimensions");
        TextureDesc texture; texture.Width=static_cast<int>(width); texture.Height=static_cast<int>(height);
        texture.DebugName="Volume density float-byte atlas";
        m_DensityBinding.Texture=m_RHI->CreateTexture(texture,source.AtlasRGBA8.data());
        if (!m_DensityBinding.Texture) return Fail(m_RHI->GetLastError().c_str());
        m_DensityBinding.Grid={float(source.Dimensions[0]),float(source.Dimensions[1]),float(source.Dimensions[2]),float(source.TileColumns)};
        return true;
    }

    void Renderer::SetEnvironmentRadiance(std::shared_ptr<const EnvironmentRadiance> source)
    {
        if (source==m_EnvironmentSource) return;
        m_EnvironmentSource=std::move(source); ResetFrameHistory();
    }

    void Renderer::SetBakedLightingDirectory(const std::filesystem::path& directory)
    {
        if (directory==m_BakedLightingDirectory) return;
        m_BakedLightingDirectory=directory; m_BakedLightmaps.Reset(); m_BakedProbes={}; m_LightmapStatus={}; ResetFrameHistory();
    }

	void Renderer::SetRenderMode(RenderMode mode)
	{
		if (mode != m_RenderMode)
		{
			ResetFrameHistory();
			// Adding an edge overlay does not change transport geometry or baked
			// receivers. Pure wireframe changes private transparency participation.
			if ((mode == RenderMode::Wireframe) != (m_RenderMode == RenderMode::Wireframe)) ++m_ProbeSceneRevision;
		}
		m_RenderMode = mode;
	}

	RenderMode Renderer::GetRenderMode() const
	{
		return m_RenderMode;
	}

	void Renderer::SetShadowsEnabled(bool enabled)
	{
		if (enabled != m_ShadowsEnabled) m_TemporalSampling.Reset();
		m_ShadowsEnabled = enabled;
	}

	bool Renderer::AreShadowsEnabled() const
	{
		return m_ShadowsEnabled;
	}

	void Renderer::SetTechniqueSettings(const Techniques::Settings& settings)
	{
		const auto previousPipeline = m_Techniques.Pipeline;
		const auto previousSamples = m_Techniques.SampleCount;
		const bool previousMotionBlur = m_Techniques.PostProcessing.MotionBlur.Enabled;
		const auto previousSettings = m_Techniques;
		m_Techniques = m_RHI ? m_RHI->ResolveTechniqueSettings(settings).Effective
			: Techniques::ResolveSettings(settings, TechniqueBackend(GetBackend())).Effective;
		if (m_Techniques.Pipeline != previousPipeline || m_Techniques.SampleCount != previousSamples) ResetFrameHistory();
		if (m_Techniques.PostProcessing.MotionBlur.Enabled != previousMotionBlur) ResetFrameHistory();
        if (!(m_Techniques.Subsurface == previousSettings.Subsurface)) ResetFrameHistory();
        if (m_Techniques.PostProcessing.AntiAliasing!=previousSettings.PostProcessing.AntiAliasing) ResetFrameHistory();
        auto historySettings=previousSettings;
        if (Techniques::UsesTemporalSuperResolution(previousSettings.PostProcessing) && Techniques::UsesTemporalSuperResolution(m_Techniques.PostProcessing))
            historySettings.PostProcessing.TemporalSuperResolution.RenderScale=m_Techniques.PostProcessing.TemporalSuperResolution.RenderScale;
        if (!(m_Techniques == historySettings)) m_TemporalSampling.Reset();
        if (Techniques::UsesRefraction(previousSettings) != Techniques::UsesRefraction(m_Techniques) ||
            Techniques::UsesDepthPeeling(previousSettings) != Techniques::UsesDepthPeeling(m_Techniques))
        { ++m_ProbeSceneRevision; ResetFrameHistory(); }
		// Gouraud interpolates vertex illumination; flat shading uses the
		// geometric face normal. Neither performs fragment normal-map lighting.
		if (m_Techniques.Shading == Techniques::ShadingModel::Gouraud ||
			m_Techniques.Shading == Techniques::ShadingModel::Flat)
		{
			m_Techniques.NormalMapping = false;
		}
	}
	const Techniques::Settings& Renderer::GetTechniqueSettings() const
	{
		return m_Techniques;
	}

	void Renderer::SetLighting(const LightingState& lighting)
	{
		const auto previousLighting = m_Lighting;
		m_Lighting = lighting;
		if (m_Lighting.SceneLights) m_Lighting.SceneLights->WriteLegacySubset(m_Lighting);
		m_Lighting.LightCount = std::min(
			m_Lighting.LightCount,
			MaximumSceneLights);
		for (auto& light : m_Lighting.Lights) light.Range = Techniques::CanonicalLightRange(light.Range);
		if (m_Lighting.ShadowLightIndex < 0 ||
			static_cast<std::uint32_t>(m_Lighting.ShadowLightIndex) >=
				m_Lighting.LightCount ||
			!m_Lighting.Lights[
				static_cast<std::size_t>(m_Lighting.ShadowLightIndex)]
					.CastShadows)
		{
			m_Lighting.ShadowLightIndex = -1;
		}
		if (!SameTemporalLighting(previousLighting, m_Lighting)) m_TemporalSampling.Reset();
	}

	const LightingState& Renderer::GetLighting() const
	{
		return m_Lighting;
	}

	void Renderer::SetViewProjection(const Matrix4& viewProjection)
	{
		m_ViewProjection = viewProjection;
		m_CascadeCameraValid = false;
		m_AutomaticViewProjection = false;
	}

	void Renderer::SetCameraMatrices(const Matrix4& view, const Matrix4& projection)
	{
		m_CascadeCamera = {view, projection};
		m_CascadeCameraValid = true;
		m_ViewProjection = Multiply(projection, view);
		m_AutomaticViewProjection = false;
	}

	void Renderer::SetLightViewProjection(const Matrix4& lightViewProjection)
	{
		m_LightViewProjection = lightViewProjection;
		m_AutomaticLightViewProjection = false;
	}

	void Renderer::SetAutomaticViewProjection(bool enabled)
	{
		if (enabled != m_AutomaticViewProjection) ResetFrameHistory();
		m_AutomaticViewProjection = enabled;
		m_AutomaticLightViewProjection = enabled;
	}

	bool Renderer::IsAutomaticViewProjection() const
	{
		return m_AutomaticViewProjection;
	}

	void Renderer::SetFrameTime(double sampleTimeSeconds, std::uint64_t epoch)
	{
		m_NextSampleTimeSeconds = sampleTimeSeconds;
		m_NextHistoryEpoch = epoch;
	}

	void Renderer::ResetFrameHistory() noexcept
	{
        m_NaniteDepthHistoryValid=false;
        ++m_PathResetSerial;
        ++m_TSRHistoryEpoch;
		m_FrameHistory.Reset();
		m_CurrentMotionFrame = {};
		m_TemporalSampling.Reset();
		m_TemporalFrame = {};
	}

    bool Renderer::PrepareRenderResolution()
    {
        ShaderUInt4 extent{static_cast<unsigned>(m_CreateInfo.Width),static_cast<unsigned>(m_CreateInfo.Height),0,0};
        if (Techniques::UsesTemporalSuperResolution(m_Techniques.PostProcessing) &&
            !BuildTSRRenderExtent(m_CreateInfo.Width,m_CreateInfo.Height,m_Techniques.PostProcessing.TemporalSuperResolution,extent,m_LastError)) return false;
        if (int(extent.X)==m_RenderWidth && int(extent.Y)==m_RenderHeight) return true;
        if (!m_RHI->Resize(int(extent.X),int(extent.Y))) { m_LastError=m_RHI->GetLastError(); return false; }
        m_RenderWidth=int(extent.X); m_RenderHeight=int(extent.Y);
        return true;
    }

	bool Renderer::RenderFrame()
	{
		if (!m_Initialized)
		{
			return Fail("Renderer is not initialized");
		}
        if (!PrepareMaterialTextures() || !PrepareRenderResolution() || !PrepareGeometry() || !PrepareHairFur()) return false;
		UpdateAutomaticViewProjections();
        if (!m_AreaLighting.Prepare(m_Lighting, m_Techniques.AreaLights, m_LastError)) return false;
        m_RenderLighting = m_AreaLighting.GetLighting();
        const double sampleTime = m_NextSampleTimeSeconds.value_or(
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
        m_RHI->SetHairFurFrame({}); m_HairCameraDepthWritten=false;
        if (m_BoundHairGeometry)
        {
            if (!m_HairLighting.Prepare(m_RenderLighting,m_LastError)) return false;
            HairFurFrame hair; hair.Geometry=m_BoundHairGeometry; hair.Lights=m_HairLighting.Samples(); hair.Settings=m_Techniques.Hair; hair.SampleTimeSeconds=sampleTime;
            m_RHI->SetHairFurFrame(std::move(hair));
        }
		if (!PrepareDensityVolume() || !PrepareCascadedShadows() || !PrepareEnvironmentCapture() || !PrepareStreamedGeometry() || !PrepareBakedLightProbes() || !PrepareVoxelGI() || !PrepareLumenGI()) return false;
		const std::uint64_t historyEpoch = m_NextSampleTimeSeconds ? m_NextHistoryEpoch : 0;
		m_NextSampleTimeSeconds.reset();
		if (!std::isfinite(sampleTime) || !FrameHistory::IsFinite(m_ViewProjection))
			return Fail("Frame sample time and view-projection must be finite");
		m_CurrentMotionFrame = m_FrameHistory.Describe(m_ViewProjection, sampleTime, historyEpoch);
		const bool temporalAA = Techniques::UsesTemporalReconstruction(m_Techniques.PostProcessing);
		const bool temporalHistory = temporalAA || Techniques::UsesTemporalGI(m_Techniques.GlobalIllumination) || Techniques::UsesTemporalReflections(m_Techniques.Reflections);
        const auto temporalCycle=Techniques::UsesTemporalSuperResolution(m_Techniques.PostProcessing) ?
            TemporalSampling::ReconstructionCycleLength(m_RenderWidth,m_RenderHeight,m_CreateInfo.Width,m_CreateInfo.Height) : TemporalSampling::CycleLength;
        m_TemporalFrame = temporalHistory ? m_TemporalSampling.Describe(m_CurrentMotionFrame.CorrespondenceValid,temporalAA,temporalCycle) : TemporalFrame{};
        // Settings can discard color history while physical camera correspondence
        // remains valid. Express the last presented jitter in this input grid.
        if (m_CurrentMotionFrame.CorrespondenceValid && m_PreviousPresentedWidth>0 && m_PreviousPresentedHeight>0)
            m_TemporalFrame.PreviousJitterPixels={m_PreviousPresentedJitter.X*m_RenderWidth/m_PreviousPresentedWidth,
                m_PreviousPresentedJitter.Y*m_RenderHeight/m_PreviousPresentedHeight};
		m_RasterViewProjection = TemporalSampling::Jitter(m_ViewProjection, m_TemporalFrame.JitterPixels,
			m_RenderWidth, m_RenderHeight);
		m_CullingViewProjection = m_ViewProjection;
		if (temporalAA)
		{
			// One stable pixel of guard band contains the complete jitter cycle.
			for (int column = 0; column < 4; ++column)
			{
				m_CullingViewProjection.Values[column] *= float(m_RenderWidth) / (m_RenderWidth + 2);
				m_CullingViewProjection.Values[4 + column] *= float(m_RenderHeight) / (m_RenderHeight + 2);
			}
		}
		m_FrustumCulledMeshCount = 0;
        if (!PreparePlanarCapture() || !PrepareMeshLods() || !PrepareLightmaps() || !PrepareRefraction() || !PrepareDepthPeeling() || !PrepareRayScene()) return false;
		const bool outlines = std::any_of(m_Meshes.begin(), m_Meshes.end(),
			[](const auto& entry) { return entry.second.Mesh.Visible && entry.second.Outline.Enabled; });
		m_RHI->SetDepthCaptureRequired(outlines && !m_GpuPresentation && !Techniques::UsesPathTracing(m_Techniques.Pipeline));
		const auto backend = [this]()
		{
			switch (GetBackend())
			{
			case RHIBackend::DX11: return Techniques::Backend::DX11;
			case RHIBackend::DX12: return Techniques::Backend::DX12;
			case RHIBackend::Metal: return Techniques::Backend::Metal;
			case RHIBackend::OpenGLES: return Techniques::Backend::OpenGLES;
			case RHIBackend::Vulkan: return Techniques::Backend::Vulkan;
			}
			return Techniques::Backend::Unknown;
		}();
		auto frameSettings = m_Techniques;
        auto& cloud=frameSettings.VolumetricFog.Clouds;
        if (cloud.Animate)
        {
            const double time=sampleTime*cloud.AnimationSpeed;
            const auto& bounds=cloud.Medium;
            // Bounded advection keeps the authored field inside its fixed medium.
            cloud.Offset[0]+=float(std::sin(time*0.18)*0.08*(bounds.BoundsMaximum[0]-bounds.BoundsMinimum[0]));
            cloud.Offset[1]+=float(std::sin(time*0.13)*0.025*(bounds.BoundsMaximum[1]-bounds.BoundsMinimum[1]));
            cloud.Offset[2]+=float(std::sin(time*0.11)*0.06*(bounds.BoundsMaximum[2]-bounds.BoundsMinimum[2]));
        }
        if (frameSettings.Nanite.Enabled && (!m_NaniteGeometry || m_NaniteGeometry->Bindings.empty()))
        {
            // Cook/bootstrap and root-budget fallback still draw ordinary source
            // meshes. Rasterize those directly into the G-buffer: expanding every
            // instance into a temporary visibility table can exceed a device's
            // storage-block limit before the shared resident cut becomes ready.
            frameSettings.Nanite.Enabled=false; frameSettings.MeshShaders=false;
            frameSettings.Pipeline=Techniques::RenderingPipeline::Deferred;
        }
        if (!m_BoundHairGeometry) frameSettings.Hair.Enabled=false;
        if (Techniques::UsesLumenReflections(frameSettings.Reflections) && (!GetLumenGIStatus().HasLighting() || m_RenderMode==RenderMode::Wireframe))
            frameSettings.Reflections.Mode=Techniques::ReflectionMode::None;
        if (Techniques::UsesLumen(frameSettings.GlobalIllumination) && !GetLumenGIStatus().HasLighting())
            frameSettings.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::None;
        if (Techniques::UsesVoxelGI(frameSettings.GlobalIllumination) && (m_RenderMode==RenderMode::Wireframe || !m_VoxelStatus.Current()))
            frameSettings.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::None;
        if (m_RenderMode == RenderMode::Wireframe)
        { frameSettings.Refraction.Enabled=false; frameSettings.Transparency.Mode=Techniques::TransparencyMode::Sorted; }
        const auto plan = Techniques::BuildPassPlan(frameSettings, backend,
			m_ShadowsEnabled, outlines, m_RenderMode == RenderMode::Wireframe);
		if (!plan.Valid()) return Fail(plan.Diagnostics.front().c_str());
		// Immutable uploads must finish before BeginFrame (in particular on DX12).
		// Visibility rasterization already supplies primitive barycentrics.
		if (m_RenderMode == RenderMode::SolidWireframe && m_RHI->GetCapabilities().NativeAPI &&
			!Techniques::UsesPathTracing(frameSettings.Pipeline))
			for (const auto& entry : m_Meshes)
			{
				const auto& mesh = entry.second;
				if (mesh.Mesh.Visible && (!UsesPrivateTransparency(mesh) || IsDepthPeeled(mesh)) &&
					(!Techniques::UsesVisibilityBuffer(frameSettings.Pipeline) || (mesh.Translucent && m_Techniques.AlphaBlending)) &&
					!PrepareWireOverlay(mesh)) return false;
			}
		m_RHI->SetTechniqueSettings(frameSettings);
        TSRSceneFrame reconstruction;
        reconstruction.Enabled=Techniques::UsesTemporalSuperResolution(frameSettings.PostProcessing);
        if (reconstruction.Enabled)
        {
            auto& frame=reconstruction.Reconstruction;
            frame.CaptureDiagnostics=!m_GpuPresentation || m_CreateInfo.CaptureDiagnosticBuffers || m_RenderBuffer!=Techniques::RenderBuffer::Color;
            frame.InputWidth=m_RenderWidth; frame.InputHeight=m_RenderHeight;
            frame.OutputWidth=m_CreateInfo.Width; frame.OutputHeight=m_CreateInfo.Height;
            frame.CurrentViewProjection=m_ViewProjection; frame.PreviousViewProjection=m_CurrentMotionFrame.PreviousViewProjection;
            frame.JitterPixels=m_TemporalFrame.JitterPixels; frame.Settings=frameSettings.PostProcessing.TemporalSuperResolution;
            frame.Epoch=m_TSRHistoryEpoch; frame.Serial=m_PresentedSerial+1; frame.PreviousSerial=m_PresentedSerial;
            frame.Correspondence=m_TemporalFrame.HistoryValid;
        }
        m_RHI->SetTSRSceneFrame(reconstruction);
        if (!PrepareVirtualShadows()) return false;
		if (!m_RHI->BeginFrame())
		{
			m_LastError = m_RHI->GetLastError();
            DiscardVirtualShadows();
			return false;
		}
		m_LastError.clear();
		const auto abortFrame = [this]()
		{
			m_BatchInstances.clear(); m_BatchMesh = nullptr;
            m_PreparedMeshDraws.clear(); m_CapturingGpuPass = false; m_InstanceDraws.clear(); m_CollectInstanceDraws=false;
			const std::string error = m_LastError.empty()
				? m_RHI->GetLastError()
				: m_LastError;
			m_RHI->AbortFrame();
            m_NaniteDepthCandidate={}; m_NaniteMainVisibility={};
            DiscardVirtualShadows();
			m_LastError = error;
			return false;
		};
		try
		{
            if (Techniques::UsesLightLists(m_Techniques.Pipeline))
            {
                LightListFrameDesc frame; frame.ViewProjection = m_RasterViewProjection; frame.Lights = m_RenderLighting.SceneLights;
                frame.Layout = m_Techniques.Pipeline == Techniques::RenderingPipeline::ClusteredForward ? LightListLayout::Clusters : LightListLayout::Tiles;
                if (!frame.Lights)
                    frame.Lights = SceneLightCollection::Create(
                        std::vector<SceneLight>(m_RenderLighting.Lights.begin(), m_RenderLighting.Lights.begin() + std::min(m_RenderLighting.LightCount, MaximumSceneLights)),
                        m_RenderLighting.ShadowLightIndex, m_LastError);
                if (!frame.Lights || !m_RHI->PrepareLightListFrame(frame)) return abortFrame();
            }
			for (const auto& pass : plan.Passes)
			{
				bool success = true;
				switch (pass.Kind)
				{
				case Techniques::PassKind::EnvironmentCapture: success = RenderEnvironmentCapture(); break;
				case Techniques::PassKind::PlanarCapture: success = RenderPlanarCapture(); break;
				case Techniques::PassKind::ShadowDepth: success = RenderShadowPass(); break;
                case Techniques::PassKind::ContactDepth: success = RenderCameraDepthPass(PassType::ContactDepth); break;
                case Techniques::PassKind::ForwardPlusDepth: success = RenderCameraDepthPass(PassType::ForwardPlusDepth); break;
                case Techniques::PassKind::TiledDeferredAssignment:
                case Techniques::PassKind::ClusteredAssignment:
                case Techniques::PassKind::ForwardPlusAssignment: success = m_RHI->GenerateLightLists(); break;
                case Techniques::PassKind::ContactVisibility:
                {
                    ContactShadowDesc contact; contact.ViewProjection = m_RasterViewProjection; contact.Lighting = m_RenderLighting;
                    success = m_RHI->GenerateContactShadows(contact);
                    break;
                }
				case Techniques::PassKind::OpaqueForward: success = RenderColorPass(false); break;
				case Techniques::PassKind::MaterialInputs: success = RenderColorPass(false, true); break;
				case Techniques::PassKind::DeferredGeometry: success = RenderColorPass(false); break;
                case Techniques::PassKind::VisibilityGeometry: success = RenderVisibilityGeometry(); break;
				case Techniques::PassKind::DeferredLighting:
				{
					DeferredLightingDesc lighting;
					lighting.ViewProjection = m_RasterViewProjection;
					lighting.LightViewProjection = m_LightViewProjection;
					lighting.Lighting = m_RenderLighting;
					lighting.Shading = m_Techniques.Shading;
					lighting.ToonBands = m_Techniques.ToonBands;
					lighting.SamplesShadowMap = m_ShadowsEnabled && m_RenderMode != RenderMode::Wireframe;
					success = m_RHI->DrawDeferredLighting(lighting);
					break;
				}
                case Techniques::PassKind::SubsurfaceScattering:
                {
                    SubsurfaceScatteringDesc diffusion; diffusion.Settings=m_Techniques.Subsurface;
                    success=m_RHI->ApplySubsurfaceScattering(diffusion); break;
                }
                case Techniques::PassKind::TemporalEffects:
                {
                    TemporalEffectsDesc exposure; exposure.SampleTimeSeconds=sampleTime; exposure.Epoch=historyEpoch; exposure.ResetSerial=m_PathResetSerial;
                    success=m_RHI->ApplyTemporalEffects(exposure); break;
                }
                case Techniques::PassKind::HairFur:
                {
                    auto view=HairView(m_RasterViewProjection); view.LightViewProjection=m_LightViewProjection;
                    view.SamplesShadowMap=m_ShadowsEnabled && m_RenderMode!=RenderMode::Wireframe;
                    success=m_RHI->DrawHairFur(view); break;
                }
				case Techniques::PassKind::DepthPeeling: success = RenderDepthPeeling(); break;
                case Techniques::PassKind::Refraction:
                {
                    const auto background = PassDesc{}.ClearColor;
                    RefractionDesc refraction; refraction.ViewProjection = m_RasterViewProjection;
                    refraction.Environment = {background.X,background.Y,background.Z};
                    success = m_RHI->ProcessRefraction(refraction); break;
                }
                case Techniques::PassKind::TransparentForward: success = RenderColorPass(true); break;
				case Techniques::PassKind::PrepareWeightedTransparency:
					success = m_RHI->ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind::Prepare); break;
				case Techniques::PassKind::ResolveWeightedTransparency:
					success = m_RHI->ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind::Resolve); break;
				case Techniques::PassKind::ResolveOpaqueSamples:
					success = m_RHI->ProcessMultisampling(Techniques::MultisamplePassKind::OpaqueResolve); break;
				case Techniques::PassKind::ResolveVelocitySamples:
					success = m_RHI->ProcessMultisampling(Techniques::MultisamplePassKind::VelocityResolve); break;
				case Techniques::PassKind::RestoreColorSamples:
					success = m_RHI->ProcessMultisampling(Techniques::MultisamplePassKind::ColorRestore); break;
				case Techniques::PassKind::ResolveColorSamples:
					success = m_RHI->ProcessMultisampling(Techniques::MultisamplePassKind::ColorResolve); break;
				case Techniques::PassKind::AmbientOcclusion:
				{
					AmbientOcclusionDesc occlusion;
					occlusion.ViewProjection = m_RasterViewProjection;
					success = m_RHI->ApplyAmbientOcclusion(occlusion);
					break;
				}
                case Techniques::PassKind::PathTracing:
                {
                    PathTracingDesc path; path.ViewProjection=m_ViewProjection; path.DisplayBackground=PassDesc{}.ClearColor;
                    path.DirectShadows=m_ShadowsEnabled; path.Epoch=historyEpoch; path.ResetSerial=m_PathResetSerial;
                    success=m_RHI->ApplyPathTracing(path); break;
                }
                case Techniques::PassKind::GlobalIllumination:
                {
                    GlobalIlluminationDesc illumination;
                    illumination.Translucency=(m_Techniques.AlphaBlending || Techniques::UsesRefraction(m_Techniques)) && std::any_of(m_Meshes.begin(),m_Meshes.end(),[&](const auto& entry) {
                        const auto& mesh=entry.second; return mesh.Mesh.Visible && (mesh.Translucent || UsesPrivateTransparency(mesh)); });
                    illumination.ViewProjection = m_RasterViewProjection;
                    illumination.Sampling = m_TemporalFrame;
                    illumination.PreviousViewProjection = m_TemporalFrame.HistoryValid ?
                        TemporalSampling::Jitter(m_CurrentMotionFrame.PreviousViewProjection, m_TemporalFrame.PreviousJitterPixels,
                            m_RenderWidth, m_RenderHeight) : m_RasterViewProjection;
                    success = m_RHI->ApplyGlobalIllumination(illumination);
                    break;
                }
				case Techniques::PassKind::HybridReflections:
                case Techniques::PassKind::Reflections:
                case Techniques::PassKind::Skybox:
				{
					ReflectionDesc reflections;
                    reflections.SkyboxOnly=pass.Kind==Techniques::PassKind::Skybox;
					reflections.ViewProjection = m_RasterViewProjection;
                    reflections.Sampling=m_TemporalFrame;
                    reflections.PreviousViewProjection=m_TemporalFrame.HistoryValid?
                        TemporalSampling::Jitter(m_CurrentMotionFrame.PreviousViewProjection,m_TemporalFrame.PreviousJitterPixels,m_RenderWidth,m_RenderHeight):m_RasterViewProjection;
					success = m_RHI->ApplyReflections(reflections);
					break;
				}
				case Techniques::PassKind::DepthOfField:
				{
					DepthOfFieldDesc lens;
					lens.ViewProjection = m_RasterViewProjection;
					success = m_RHI->ApplyDepthOfField(lens);
					break;
				}
				case Techniques::PassKind::VolumetricFog:
				case Techniques::PassKind::VolumetricLighting:
                case Techniques::PassKind::ParticipatingMedia:
                case Techniques::PassKind::VolumetricClouds:
				{
					VolumetricFogDesc fog;
                    if (m_DensityBinding.Texture) fog.Density=&m_DensityBinding;
					fog.ViewProjection = m_RasterViewProjection;
					fog.LightViewProjection = m_LightViewProjection;
					fog.Lighting = m_RenderLighting;
					fog.SamplesShadowMap = m_ShadowsEnabled && m_RenderMode != RenderMode::Wireframe;
					success = m_RHI->ApplyVolumetricFog(fog);
					break;
				}
				case Techniques::PassKind::Velocity: success = RenderVelocityPass(); break;
                case Techniques::PassKind::SnapshotTSROpaque: success=m_RHI->SnapshotTSROpaque(); break;
				case Techniques::PassKind::LightShafts:
				{
					LightShaftDesc shafts;
					shafts.ViewProjection = m_RasterViewProjection;
					shafts.Lighting = m_RenderLighting;
					success = m_RHI->ApplyLightShafts(shafts);
					break;
				}
				case Techniques::PassKind::TemporalAA:
				{
					TemporalAADesc temporal;
					temporal.Sampling = m_TemporalFrame;
					temporal.PreviousViewProjection = m_TemporalFrame.HistoryValid ?
						TemporalSampling::Jitter(m_CurrentMotionFrame.PreviousViewProjection, m_TemporalFrame.PreviousJitterPixels,
							m_RenderWidth, m_RenderHeight) : m_RasterViewProjection;
					success = m_RHI->ApplyTemporalAA(temporal);
					break;
				}
				case Techniques::PassKind::MotionBlur:
				{
					MotionBlurDesc motion;
					motion.ViewProjection = m_RasterViewProjection;
					motion.Motion = m_CurrentMotionFrame;
					success = m_RHI->ApplyMotionBlur(motion);
					break;
				}
				// GPU views draw outlines in the native frame; CPU captures composite after readback.
				case Techniques::PassKind::Outline: if(m_GpuPresentation) success=RenderOutlinePass(); break;
				}
				if (!success) return abortFrame();
			}
			if (!m_RHI->Present())
			{
				m_LastError = m_RHI->GetLastError();
				return abortFrame();
			}
            m_PreviousVegetationTime=m_VegetationTime;
            CommitNaniteFrameVisibility();
            ++m_PresentedSerial;
            CommitVirtualShadows();
            m_PreviousPresentedWidth=m_RenderWidth; m_PreviousPresentedHeight=m_RenderHeight;
            m_PreviousPresentedJitter=m_TemporalFrame.JitterPixels;
			m_LastFrameCapture = m_RHI->TakeLastFrameCapture();
            if (m_RenderBuffer==Techniques::RenderBuffer::Shadow && m_Techniques.Shadows==Techniques::ShadowRepresentation::Virtual && m_VirtualShadowPublished.Handle)
            {
                if (!m_RHI->ReadVirtualShadowAtlas(m_VirtualShadowPublished.Handle,m_LastFrameCapture.ShadowDepth32)) return Fail(m_RHI->GetLastError().c_str());
                m_LastFrameCapture.ShadowWidth=m_VirtualShadowPublished.Info.Layout.AtlasWidth;
                m_LastFrameCapture.ShadowHeight=m_VirtualShadowPublished.Info.Layout.AtlasHeight;
                ApplyDiagnosticCapture(m_LastFrameCapture,m_RenderBuffer);
            }
            CaptureVirtualShadowStatistics(m_LastFrameCapture.Statistics);
            if (Techniques::UsesLumenScene(m_Techniques))
            {
                const auto status=GetLumenGIStatus(); const char* names[]{"Idle","Preparing mesh proxies","Uploading distance scene","Capturing materials","Relighting surface cache","Ready","Failed"};
                m_LastFrameCapture.LumenGIStatus=status.HasLighting() && status.Pending() ?
                    std::string("Ready; updating ")+names[unsigned(status.State)] : names[unsigned(status.State)];
                if (!status.Error.empty()) m_LastFrameCapture.LumenGIStatus+=": "+status.Error;
                if (status.Cards && status.State!=LumenRenderState::Preparing)
                {
                    m_LastFrameCapture.LumenGIStatus+="; sampled texel coverage "+std::to_string(unsigned(std::clamp(status.SampledTexelCoverage,0.0,1.0)*100))+"%";
                    if (status.CardCoverageBudgetExhausted) m_LastFrameCapture.LumenGIStatus+=" (coverage ray budget exhausted)";
                }
                if (status.State==LumenRenderState::Capture && status.CaptureCards.TotalCards)
                    m_LastFrameCapture.LumenGIStatus+="; material cards "+std::to_string(status.CaptureCards.CompletedCards)+" / "+std::to_string(status.CaptureCards.TotalCards);
                if (status.TargetBounces) m_LastFrameCapture.LumenGIStatus+="; indirect bounces "+std::to_string(status.CompletedBounces)+" / "+std::to_string(status.TargetBounces);
                if (status.State==LumenRenderState::Lighting && status.LightingTiles.TotalTiles)
                    m_LastFrameCapture.LumenGIStatus+="; lighting tiles "+std::to_string(status.LightingTiles.CompletedTiles)+" / "+std::to_string(status.LightingTiles.TotalTiles);
                if (m_Techniques.GlobalIllumination.Lumen.GlobalDistance.Enabled && GetLumenGlobalDistanceState()==LumenGlobalDistanceState::Updating)
                {
                    const auto global=GetLumenGlobalDistanceStatistics();
                    m_LastFrameCapture.LumenGIStatus+="; global voxels "+std::to_string(global.ReusedVoxels+global.ComposedVoxels)+" / "+std::to_string(global.TotalVoxels);
                }
                if (m_Techniques.GlobalIllumination.Lumen.WorldCache.Enabled && GetLumenWorldCacheState()==LumenWorldCacheState::Queued)
                {
                    const auto world=GetLumenWorldCacheStatistics();
                    m_LastFrameCapture.LumenGIStatus+="; world probes "+std::to_string(world.ReusedProbes+world.TracedProbes)+" / "+std::to_string(world.TotalProbes);
                }
            }
            if (Techniques::UsesVoxelGI(m_Techniques.GlobalIllumination))
            {
                const char* states[]{"Idle","Queued","Preparing scene","Voxelizing geometry","Injecting radiance","Ready","Failed"};
                m_LastFrameCapture.VoxelGIStatus=states[static_cast<std::size_t>(m_VoxelStatus.State)];
                if (!m_VoxelStatus.Error.empty()) m_LastFrameCapture.VoxelGIStatus += ": " + m_VoxelStatus.Error;
            }
            {
                const char* hairStates[]{"Idle","Queued","Preparing source","Growing strands","Ready","Failed"};
                m_LastFrameCapture.HairFurStatus=hairStates[static_cast<std::size_t>(m_HairStatus.State)];
                m_LastFrameCapture.HairGeometryPending=m_HairStatus.Pending();
                if (m_HairStatus.Current()) m_LastFrameCapture.HairFurStatus+=": "+std::to_string(m_HairStatus.Published->Geometry->StrandCount)+" strands";
                if (!m_HairStatus.Error.empty()) m_LastFrameCapture.HairFurStatus+=": "+m_HairStatus.Error;
            }
            if (Techniques::UsesLightmaps(m_Techniques.GlobalIllumination))
            {
                const char* states[]{"Idle","Queued","Preparing UV2 receivers","Baking indirect light","Ready","Failed"};
                m_LastFrameCapture.BakedLightingStatus=states[static_cast<std::size_t>(m_LightmapStatus.State)];
                if (!m_BakedLightingDirectory.empty() && m_LightmapStatus.State!=LightmapBakeState::Failed)
                    m_LastFrameCapture.BakedLightingStatus="Loading baked resource";
                if (m_LightmapStatus.Current() && m_LightmapStatus.Published->LoadedFromAsset) m_LastFrameCapture.BakedLightingStatus="Loaded baked resource";
                if (!m_LightmapStatus.Error.empty()) m_LastFrameCapture.BakedLightingStatus += ": " + m_LightmapStatus.Error;
                m_LastFrameCapture.Statistics.LightmapReceiverBuildCount=m_LightmapStatus.ReceiverBuildCount;
                m_LastFrameCapture.Statistics.LightmapBakeCount=m_LightmapStatus.BakeCount;
                m_LastFrameCapture.Statistics.BakedLightingFromAsset=m_LightmapStatus.Current() && m_LightmapStatus.Published->LoadedFromAsset;
            }
            if (Techniques::UsesBakedProbes(m_Techniques.GlobalIllumination))
            {
                m_LastFrameCapture.BakedLightingStatus=m_BakedProbes.LoadedFromAsset()?"Loaded baked resource":"Ready (live bake)";
                m_LastFrameCapture.Statistics.ProbeBakeCount=m_BakedProbes.BakeCount();
                m_LastFrameCapture.Statistics.BakedLightingFromAsset=m_BakedProbes.LoadedFromAsset();
            }
            if (m_PreparedGeometry)
            {
                auto& statistics=m_LastFrameCapture.Statistics;
                statistics.PreparedGeometryGeneration=m_PreparedGeometry->Serial;
                statistics.GeometryPreparationDispatches=m_PreparedGeometry->DisplacementDispatches+m_PreparedGeometry->TessellationDispatches;
                if (m_PreparedGeometry->DisplacementDispatches)
                {
                    statistics.DisplacedMeshCount=static_cast<std::uint32_t>(m_PreparedGeometry->Bindings.size());
                    statistics.DisplacedVertexCount=m_PreparedGeometry->VertexCount;
                    statistics.DisplacementPreparationDispatches=m_PreparedGeometry->DisplacementDispatches;
                    statistics.DisplacementGeneration=m_PreparedGeometry->Serial;
                }
                if (m_PreparedGeometry->TessellationDispatches)
                {
                    statistics.TessellatedMeshCount=static_cast<std::uint32_t>(m_PreparedGeometry->Bindings.size());
                    statistics.TessellatedVertexCount=m_PreparedGeometry->VertexCount;
                    statistics.TessellationPreparationDispatches=m_PreparedGeometry->TessellationDispatches;
                }
            }
			m_LastFrameCapture.Statistics.FrustumCulledMeshCount = m_FrustumCulledMeshCount;
			m_LastFrameCapture.Statistics.LodReducedMeshCount = 0;
            m_LastFrameCapture.Statistics.NaniteMeshCount=m_NaniteStatistics.NaniteMeshCount;
            m_LastFrameCapture.Statistics.NaniteSelectedClusters=m_NaniteStatistics.NaniteSelectedClusters;
            m_LastFrameCapture.Statistics.NaniteResidentPages=m_NaniteStatistics.NaniteResidentPages;
            m_LastFrameCapture.Statistics.NanitePendingPages=m_NaniteStatistics.NanitePendingPages;
            m_LastFrameCapture.Statistics.NaniteUploadedPages=m_NaniteStatistics.NaniteUploadedPages;
            m_LastFrameCapture.Statistics.NaniteGeometryGeneration=m_NaniteStatistics.NaniteGeometryGeneration;
            m_LastFrameCapture.Statistics.NaniteReadbackBytes=m_NaniteStatistics.NaniteReadbackBytes;
            m_LastFrameCapture.Statistics.NaniteHostRebuiltVertices=m_NaniteStatistics.NaniteHostRebuiltVertices;
            m_LastFrameCapture.Statistics.NaniteScenePageBudget=m_NaniteStatistics.NaniteScenePageBudget;
            m_LastFrameCapture.Statistics.NaniteRootPages=m_NaniteStatistics.NaniteRootPages;
            m_LastFrameCapture.Statistics.NaniteAllocatedPages=m_NaniteStatistics.NaniteAllocatedPages;
            m_LastFrameCapture.Statistics.NaniteFamilyCount=m_NaniteStatistics.NaniteFamilyCount;
            m_LastFrameCapture.Statistics.NaniteViewCount=m_NaniteStatistics.NaniteViewCount;
            m_LastFrameCapture.Statistics.NaniteBudgetBlockedGroups=m_NaniteStatistics.NaniteBudgetBlockedGroups;
            m_LastFrameCapture.Statistics.NaniteCookRequests=m_NaniteStatistics.NaniteCookRequests;
            m_LastFrameCapture.Statistics.NaniteCookSourceBytes=m_NaniteStatistics.NaniteCookSourceBytes;
            m_LastFrameCapture.Statistics.NanitePreparationMicroseconds=m_NaniteStatistics.NanitePreparationMicroseconds;
            m_LastFrameCapture.Statistics.NaniteBudgetLimited=m_NaniteStatistics.NaniteBudgetLimited;
            m_LastFrameCapture.Statistics.VirtualMeshCount=m_VirtualStatistics.VirtualMeshCount;
            m_LastFrameCapture.Statistics.VirtualNodeCount=m_VirtualStatistics.VirtualNodeCount;
            m_LastFrameCapture.Statistics.VirtualResidentPages=m_VirtualStatistics.VirtualResidentPages;
            m_LastFrameCapture.Statistics.VirtualPendingPages=m_VirtualStatistics.VirtualPendingPages;
            m_LastFrameCapture.Statistics.VirtualUploadedPages=m_VirtualStatistics.VirtualUploadedPages;
            m_LastFrameCapture.Statistics.VirtualCopiedPages=m_VirtualStatistics.VirtualCopiedPages;
            m_LastFrameCapture.Statistics.VirtualBootstrapPages=m_VirtualStatistics.VirtualBootstrapPages;
            m_LastFrameCapture.Statistics.VirtualUploadedBytes=m_VirtualStatistics.VirtualUploadedBytes;
            m_LastFrameCapture.Statistics.VirtualCopiedBytes=m_VirtualStatistics.VirtualCopiedBytes;
            m_LastFrameCapture.Statistics.VirtualPoolBytes=m_VirtualStatistics.VirtualPoolBytes;
            m_LastFrameCapture.Statistics.VirtualGeometryGeneration=m_VirtualStatistics.VirtualGeometryGeneration;
			if (!Techniques::UsesPathTracing(m_Techniques.Pipeline)) ApplyOutlines(m_LastFrameCapture);
			m_LastFrameCapture.Motion = m_CurrentMotionFrame;
			m_LastFrameCapture.Temporal = m_TemporalFrame;
			const auto& motionBlur = m_Techniques.PostProcessing.MotionBlur;
			m_LastFrameCapture.NeedsTemporalFrame = temporalHistory || (motionBlur.Enabled && motionBlur.ShutterSeconds > 0.0f &&
				motionBlur.MaxRadiusPixels > 0.0f && m_CurrentMotionFrame.MotionDetected);
            m_LastFrameCapture.NeedsTemporalFrame |= Techniques::UsesPathTracing(m_Techniques.Pipeline) && m_Techniques.PathTracing.Accumulate &&
                m_LastFrameCapture.Statistics.PathTracingSampleCount < m_Techniques.PathTracing.AccumulationLimit;
            m_LastFrameCapture.NeedsTemporalFrame |= m_HairStatus.Pending() || m_VirtualPending || m_NanitePending;
            m_LastFrameCapture.NeedsTemporalFrame |= Techniques::UsesLumenScene(m_Techniques) && (GetLumenGIStatus().Pending() ||
                (m_Techniques.GlobalIllumination.Lumen.GlobalDistance.Enabled && GetLumenGlobalDistanceState()==LumenGlobalDistanceState::Updating) ||
                (m_Techniques.GlobalIllumination.Lumen.WorldCache.Enabled && GetLumenWorldCacheState()==LumenWorldCacheState::Queued));
            m_LastFrameCapture.NeedsTemporalFrame |= m_Techniques.PostProcessing.TemporalEffects.Enabled;
            m_LastFrameCapture.NeedsTemporalFrame |= m_VoxelStatus.State==VoxelUpdateState::Queued || m_VoxelStatus.State==VoxelUpdateState::Preparing ||
                m_VoxelStatus.State==VoxelUpdateState::Voxelizing || m_VoxelStatus.State==VoxelUpdateState::Injecting;
            m_LastFrameCapture.NeedsTemporalFrame |= m_LightmapStatus.State==LightmapBakeState::Queued ||
                m_LightmapStatus.State==LightmapBakeState::Preparing || m_LightmapStatus.State==LightmapBakeState::Baking;
			if (temporalHistory) m_TemporalSampling.Commit(m_TemporalFrame,temporalCycle);
			m_FrameHistory.Commit(m_ViewProjection, sampleTime, historyEpoch);
			for (auto& entry : m_Meshes)
			{
				entry.second.PreviousWorldTransform = entry.second.Mesh.WorldTransform;
				entry.second.PreviousTransformValid = entry.second.Mesh.Visible;
				entry.second.PreviousLod = entry.second.SelectedLod;
				m_LastFrameCapture.Statistics.LodReducedMeshCount += entry.second.Mesh.Visible && entry.second.SelectedLod != 0;
			}
		}
		catch (...)
		{
			try
			{
				m_BatchInstances.clear(); m_BatchMesh = nullptr;
                m_PreparedMeshDraws.clear(); m_CapturingGpuPass = false; m_InstanceDraws.clear(); m_CollectInstanceDraws=false;
				m_RHI->AbortFrame();
                m_NaniteDepthCandidate={}; m_NaniteMainVisibility={};
                DiscardVirtualShadows();
			}
			catch (...)
			{
				// Preserve the original rendering exception.
			}
			throw;
		}
		m_LastError.clear();
		return true;
	}

    void Renderer::ReleasePresentationContext() noexcept
    { if(m_RHI)m_RHI->ReleasePresentationContext(); }

    void Renderer::SetGpuPresentationEnabled(bool enabled)
    {
        m_GpuPresentation=enabled;
        if (m_RHI) m_RHI->SetGpuPresentationEnabled(enabled);
    }

	FrameCapture Renderer::GetLastFrameCapture() const
	{
		return m_LastFrameCapture;
	}

	FrameCapture Renderer::TakeLastFrameCapture() noexcept
	{
		return std::move(m_LastFrameCapture);
	}

	const FrameCapture& Renderer::GetLastFrameCaptureView() const noexcept
	{
		return m_LastFrameCapture;
	}

	std::string Renderer::GetLastError() const
	{
		return m_LastError;
	}

	std::size_t Renderer::GetMeshCount() const
	{
		return m_Meshes.size();
	}

	IRHI* Renderer::GetRHI()
	{
		return m_RHI.get();
	}

	const IRHI* Renderer::GetRHI() const
	{
		return m_RHI.get();
	}

	MeshData Renderer::CreateTriangle(const std::string& id)
	{
		MeshData mesh;
		mesh.Id = id;
		mesh.Vertices = {
			{ {  0.0f,  0.65f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.15f, 0.10f, 1.0f } },
			{ { -0.6f, -0.45f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.10f, 0.80f, 0.20f, 1.0f } },
			{ {  0.6f, -0.45f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.10f, 0.35f, 1.0f, 1.0f } }
		};
		mesh.Indices = { 0, 1, 2 };
		mesh.Color = { 1.0f, 1.0f, 1.0f, 1.0f };
		return mesh;
	}

	bool Renderer::CreateMeshResources(const MeshData& mesh, MeshResource& resource)
	{
		if (!FrameHistory::IsFinite(mesh.WorldTransform)) return Fail("Mesh world transform must be finite");
		if (mesh.Id.empty() || mesh.Vertices.empty() || mesh.Indices.empty() || mesh.Indices.size() % 3 != 0)
		{
			return Fail("A mesh requires an identifier and triangle-list geometry");
		}
		for (const std::uint32_t index : mesh.Indices)
		{
			if (index >= mesh.Vertices.size())
			{
				return Fail("Mesh index is outside the vertex array");
			}
		}

		MeshResource candidate;
		try
		{
			candidate.Mesh = MeshInstance(mesh);
			candidate.Translucent = IsMeshTranslucent(mesh);
			const auto rollback = [this, &candidate](std::string error)
			{
				DestroyMeshResources(candidate);
				m_LastError = std::move(error);
				return false;
			};

			if (!AcquireGeometry(mesh, candidate.Geometry)) return rollback(m_LastError);
            candidate.SourceGeometry=candidate.BaseGeometry=candidate.Geometry;
			candidate.BoundsMinimum = candidate.Geometry->BoundsMinimum;
			candidate.BoundsMaximum = candidate.Geometry->BoundsMaximum;
			if (candidate.Mesh.Vegetation)
            {
                const float margin=candidate.Mesh.Vegetation==1?5.0f:9.0f;
                candidate.BoundsMinimum=Add(candidate.BoundsMinimum,{-margin,-margin,-margin});
                candidate.BoundsMaximum=Add(candidate.BoundsMaximum,{margin,margin,margin});
            }
            candidate.BoundsCenter = Multiply(Add(candidate.BoundsMinimum, candidate.BoundsMaximum), 0.5f);

			if (!AcquireTexture(
					mesh.Material.DiffuseImage,
					TextureColorSpace::SRGB,
					mesh.Id + ".Diffuse",
					candidate.DiffuseTexture) ||
				!AcquireTexture(
					mesh.Material.NormalImage,
					TextureColorSpace::Linear,
					mesh.Id + ".Normal",
					candidate.NormalTexture) ||
				!AcquireTexture(
					mesh.Material.SurfaceImage(),
					TextureColorSpace::Linear,
					mesh.Id + ".Opacity",
					candidate.OpacityTexture))
			{
				return rollback(m_LastError);
			}

			resource = std::move(candidate);
			return true;
		}
		catch (...)
		{
			DestroyMeshResources(candidate);
			throw;
		}
	}

	void Renderer::DestroyMeshResources(MeshResource& resource)
	{
		ReleaseTexture(resource.DiffuseTexture);
		ReleaseTexture(resource.NormalTexture);
		ReleaseTexture(resource.OpacityTexture);
        resource.SourceGeometry=resource.Geometry=nullptr; ReleaseGeometry(resource.BaseGeometry);
		resource.SelectedLod = resource.PreviousLod = 0;
		resource.Mesh = {};
		resource.Outline = {};
	}

	bool Renderer::AcquireGeometry(const MeshData& mesh, MeshGeometryResource*& output)
	{
		const auto hash = mesh.DynamicVertices ? std::hash<std::string>{}(mesh.Id) : HashMeshGeometry(mesh.Vertices, mesh.Indices);
		const auto range = m_GeometryCache.equal_range(hash);
		for (auto entry = range.first; entry != range.second; ++entry)
		{
			auto& geometry = *entry->second;
			if (mesh.DynamicVertices || geometry.DynamicVertices || !EqualMeshGeometry(geometry.Vertices, geometry.Indices, mesh.Vertices, mesh.Indices)) continue;
			++geometry.ReferenceCount;
			output = &geometry;
			return true;
		}
		auto candidate = std::make_unique<MeshGeometryResource>();
		candidate->Hash = hash;
        candidate->DynamicVertices=mesh.DynamicVertices;
		candidate->Vertices = mesh.Vertices;
		candidate->Indices = mesh.Indices;
		candidate->BoundsMinimum = candidate->BoundsMaximum = mesh.Vertices.front().Position;
		for (const auto& vertex : mesh.Vertices)
		{
			auto& minimum = candidate->BoundsMinimum; auto& maximum = candidate->BoundsMaximum;
			minimum.X = std::min(minimum.X, vertex.Position.X); maximum.X = std::max(maximum.X, vertex.Position.X);
			minimum.Y = std::min(minimum.Y, vertex.Position.Y); maximum.Y = std::max(maximum.Y, vertex.Position.Y);
			minimum.Z = std::min(minimum.Z, vertex.Position.Z); maximum.Z = std::max(maximum.Z, vertex.Position.Z);
		}
		try
		{
			BufferDesc desc; desc.Usage = BufferUsage::Vertex; desc.Dynamic=mesh.DynamicVertices;
			desc.SizeInBytes = candidate->Vertices.size() * sizeof(Vertex); desc.DebugName = mesh.Id + ".Vertices";
			candidate->VertexBuffer = m_RHI->CreateBuffer(desc, candidate->Vertices.data());
			if (candidate->VertexBuffer)
			{
				desc.Usage = BufferUsage::Index; desc.Dynamic=false; desc.SizeInBytes = candidate->Indices.size() * sizeof(std::uint32_t);
				desc.DebugName = mesh.Id + ".Indices";
				candidate->IndexBuffer = m_RHI->CreateBuffer(desc, candidate->Indices.data());
			}
			if (!candidate->VertexBuffer || !candidate->IndexBuffer)
			{
				const auto error = m_RHI->GetLastError(); DestroyGeometryResources(*candidate);
				return Fail(error.c_str());
			}
			// Allocate the cache node before transferring GPU ownership. Failure in
			// emplace leaves candidate available for rollback, not a half-owned node.
			const auto entry = m_GeometryCache.emplace(hash, nullptr);
			entry->second = std::move(candidate);
			output = entry->second.get();
		}
		catch (...) { if (candidate) DestroyGeometryResources(*candidate); throw; }
		return true;
	}

	void Renderer::DestroyGeometryResources(MeshGeometryResource& geometry)
	{
		for (const auto& overlay : geometry.WireOverlays)
		{
			if (overlay.Vertices) m_RHI->DestroyBuffer(overlay.Vertices);
			if (overlay.Indices) m_RHI->DestroyBuffer(overlay.Indices);
		}
		geometry.WireOverlays.clear();
		for (auto buffer : geometry.LodIndexBuffers) if (buffer) m_RHI->DestroyBuffer(buffer);
		if (geometry.OwnsVertexBuffer && geometry.VertexBuffer) m_RHI->DestroyBuffer(geometry.VertexBuffer);
		if (geometry.OwnsIndexBuffer && geometry.IndexBuffer) m_RHI->DestroyBuffer(geometry.IndexBuffer);
		geometry.LodIndexBuffers.clear(); geometry.LodChain = {}; geometry.LodPrepared = false;
		geometry.VertexBuffer = {}; geometry.IndexBuffer = {};
	}

	bool Renderer::PrepareWireOverlay(const MeshResource& mesh)
	{
		auto& geometry = *mesh.Geometry;
		if (geometry.WireOverlays.size() <= mesh.SelectedLod) geometry.WireOverlays.resize(mesh.SelectedLod + 1);
		auto& overlay = geometry.WireOverlays[mesh.SelectedLod];
		if (overlay.Vertices && overlay.Indices) return true;
		const auto& source = mesh.SelectedIndices();
		if (!geometry.HostGeometryReady || source.size() != mesh.SelectedIndexCount() || source.size() % 3)
			return Fail("Wire overlay requires the selected triangle geometry");
		// Duplicate corners only in this immutable diagnostic view. Position,
		// normal, UV and opacity retain the exact solid-pass attributes.
		std::vector<Vertex> vertices; vertices.reserve(source.size());
		std::vector<std::uint32_t> indices; indices.reserve(source.size());
		for (std::size_t corner = 0; corner < source.size(); ++corner)
		{
			if (source[corner] >= geometry.Vertices.size()) return Fail("Invalid wire overlay vertex index");
			auto vertex = geometry.Vertices[source[corner]];
			vertex.Color = {corner % 3 == 0 ? 1.0f : 0.0f, corner % 3 == 1 ? 1.0f : 0.0f,
				corner % 3 == 2 ? 1.0f : 0.0f, vertex.Color.W};
			vertices.push_back(vertex); indices.push_back(static_cast<std::uint32_t>(corner));
		}
		BufferDesc desc; desc.Usage = BufferUsage::Vertex;
		desc.SizeInBytes = vertices.size() * sizeof(Vertex); desc.DebugName = mesh.Mesh.Id + ".WireOverlay.Vertices";
		auto vertexBuffer = m_RHI->CreateBuffer(desc, vertices.data());
		if (!vertexBuffer) return Fail(m_RHI->GetLastError().c_str());
		desc.Usage = BufferUsage::Index; desc.SizeInBytes = indices.size() * sizeof(std::uint32_t);
		desc.DebugName = mesh.Mesh.Id + ".WireOverlay.Indices";
		auto indexBuffer = m_RHI->CreateBuffer(desc, indices.data());
		if (!indexBuffer)
		{
			const auto error = m_RHI->GetLastError(); m_RHI->DestroyBuffer(vertexBuffer); return Fail(error.c_str());
		}
		overlay = {vertexBuffer, indexBuffer};
		return true;
	}

	void Renderer::ReleaseGeometry(MeshGeometryResource*& geometry)
	{
		if (!geometry) return;
		auto* released = geometry; geometry = nullptr;
		if (--released->ReferenceCount) return;
		const auto range = m_GeometryCache.equal_range(released->Hash);
		for (auto entry = range.first; entry != range.second; ++entry)
			if (entry->second.get() == released)
			{
				DestroyGeometryResources(*released);
				m_GeometryCache.erase(entry);
				return;
			}
	}

	bool Renderer::EnsureMeshLods(MeshResource& resource)
    {
        return EnsureGeometryLods(*resource.Geometry,resource.Mesh.Id);
    }

    bool Renderer::EnsureGeometryLods(MeshGeometryResource& geometry,const std::string& name)
	{
		if (geometry.LodPrepared) return true;
		MeshLodChain chain;
		if (!BuildMeshLods(geometry.Vertices, geometry.Indices, {}, chain, m_LastError)) return false;
		std::vector<BufferHandle> buffers(chain.Levels.size() - 1);
		const auto rollback = [&]()
		{
			for (auto buffer : buffers) if (buffer) m_RHI->DestroyBuffer(buffer);
		};
		try
		{
			// Prepare outside BeginFrame: DX12 upload commands cannot run inside
			// its recorded frame. Publish a complete resident chain atomically.
			for (std::size_t level = 1; level < chain.Levels.size(); ++level)
			{
				const auto& indices = chain.Levels[level].Indices;
				BufferDesc desc;
				desc.Usage = BufferUsage::Index;
				desc.SizeInBytes = indices.size() * sizeof(std::uint32_t);
				desc.DebugName = name + ".LOD" + std::to_string(level);
				buffers[level - 1] = m_RHI->CreateBuffer(desc, indices.data());
				if (!buffers[level - 1])
				{
					const auto error = m_RHI->GetLastError(); rollback();
					return Fail(error.c_str());
				}
			}
		}
		catch (...) { rollback(); throw; }
		geometry.LodChain = std::move(chain);
		geometry.LodIndexBuffers = std::move(buffers);
		geometry.LodPrepared = true;
		return true;
	}

	bool Renderer::PrepareMeshLods()
	{
		// One level satisfies every view that can rasterize this mesh. Sharing it
		// across material/color/velocity and CPU outlines prevents coverage cracks;
		// shadow/capture views can demand finer detail than the main camera.
		// Only Present commits PreviousLod. Failed uploads/draws leave hysteresis
		// and velocity correspondence at the last successfully presented geometry.
		for (auto& entry : m_Meshes)
		{
			auto& mesh = entry.second;
			mesh.SelectedLod = 0;
			if (!m_Techniques.Lod.Enabled || !mesh.Mesh.Visible || mesh.Mesh.ManagedLod || mesh.Mesh.Vegetation) continue;
			if (!EnsureMeshLods(mesh)) return false;
			const auto previous = mesh.PreviousTransformValid ? mesh.PreviousLod : 0;
			std::uint32_t selected = static_cast<std::uint32_t>(mesh.Geometry->LodChain.Levels.size() - 1);
			bool relevantView = false;
			const auto constrain = [&](const Matrix4& projection, std::uint32_t width, std::uint32_t height)
			{
				if (!IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum, mesh.Mesh.WorldTransform, projection)) return;
				LodView view;
				view.World = mesh.Mesh.WorldTransform; view.ViewProjection = projection;
				view.BoundsMinimum = mesh.BoundsMinimum; view.BoundsMaximum = mesh.BoundsMaximum;
				view.Width = width; view.Height = height;
				selected = std::min(selected, SelectMeshLod(mesh.Geometry->LodChain, view,
					m_Techniques.Lod.MaximumErrorPixels, m_Techniques.Lod.HysteresisFraction, previous));
				relevantView = true;
			};
			constrain(m_ViewProjection, m_CreateInfo.Width, m_CreateInfo.Height);
			if (m_ShadowsEnabled && m_RenderMode != RenderMode::Wireframe && mesh.Mesh.CastShadow && !UsesPrivateTransparency(mesh))
			{
				if (m_CascadeFrame.CascadeCount)
					for (std::uint32_t index = 0; index < m_CascadeFrame.CascadeCount; ++index)
						constrain(m_CascadeFrame.Cascades[index].LightViewProjection, m_CreateInfo.ShadowMapSize, m_CreateInfo.ShadowMapSize);
				else constrain(m_LightViewProjection, m_CreateInfo.ShadowMapSize, m_CreateInfo.ShadowMapSize);
			}
			if (!(mesh.Translucent && m_Techniques.AlphaBlending) && !UsesPrivateTransparency(mesh))
			{
				if (Techniques::UsesEnvironmentCapture(m_Techniques.Reflections))
					for (const auto& face : m_EnvironmentFrame.Faces)
						constrain(EnvironmentRasterProjection(face, GetBackend()), m_EnvironmentFrame.Resolution, m_EnvironmentFrame.Resolution);
				if (Techniques::UsesPlanarCapture(m_Techniques.Reflections) && m_PlanarFrame.Active())
					constrain(m_PlanarFrame.ClippedViewProjection, m_PlanarFrame.Width, m_PlanarFrame.Height);
			}
			mesh.SelectedLod = relevantView ? selected : 0;
		}
		return true;
	}

	bool Renderer::AcquireTexture(
		const std::shared_ptr<const ImageData>& image,
		TextureColorSpace colorSpace,
		const std::string& debugName,
		TextureCacheEntry*& texture)
	{
		texture = nullptr;
		if (!image)
		{
			return true;
		}
		if (!image->IsValid())
		{
			return Fail("Mesh material image data is invalid");
		}
		if (image->Width >
				static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
			image->Height >
				static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
		{
			return Fail("Mesh material image dimensions exceed renderer limits");
		}

		const TextureCacheKey key { image.get(), colorSpace };
		const auto cached = m_TextureCache.find(key);
		if (cached != m_TextureCache.end())
		{
			if (cached->second.ReferenceCount ==
				std::numeric_limits<std::size_t>::max())
			{
				return Fail("Mesh material texture reference count overflow");
			}
			++cached->second.ReferenceCount;
			texture = &cached->second;
			return true;
		}

		TextureDesc description;
		description.Width = static_cast<int>(image->Width);
		description.Height = static_cast<int>(image->Height);
		description.Format = TextureFormat::RGBA8;
		description.ColorSpace = colorSpace;
		description.RenderTarget = false;
		description.DebugName = debugName;
		if (WantsMaterialMips()) description.MipLevels = MaterialMipLevels(description);
		const TextureHandle created = UploadMaterialTexture(description, *image);
		if (!created)
		{
			m_LastError = m_RHI->GetLastError();
			return false;
		}

		TextureCacheEntry entry;
		entry.Image = image;
		entry.Description = std::move(description);
		entry.Texture = created;
		entry.ReferenceCount = 1;
		bool createdNeedsRollback = true;
		try
		{
			const auto inserted = m_TextureCache.emplace(key, std::move(entry));
			if (!inserted.second)
			{
				m_RHI->DestroyTexture(created);
				createdNeedsRollback = false;
				return Fail("Mesh material texture cache insertion failed");
			}
			createdNeedsRollback = false;
			texture = &inserted.first->second;
			return true;
		}
		catch (...)
		{
			if (createdNeedsRollback)
			{
				m_RHI->DestroyTexture(created);
			}
			throw;
		}
	}

	void Renderer::ReleaseTexture(TextureCacheEntry*& texture)
	{
		if (!texture) return;
		if (--texture->ReferenceCount == 0)
		{
			if (m_RHI && m_RHI->IsInitialized()) m_RHI->DestroyTexture(texture->Texture);
			m_TextureCache.erase({texture->Image.get(), texture->Description.ColorSpace});
		}
		texture = nullptr;
	}

	bool Renderer::WantsMaterialMips() const
	{
		return m_Techniques.AnisotropicFiltering && !Techniques::UsesPathTracing(m_Techniques.Pipeline) &&
			m_RHI->GetCapabilities().MaterialAnisotropicFiltering;
	}

	TextureHandle Renderer::UploadMaterialTexture(const TextureDesc& desc, const ImageData& image)
	{
		if (desc.MipLevels == 1) return m_RHI->CreateTexture(desc, image.PixelsRGBA8.data());
		const auto mips = BuildMaterialMipChain(desc, image.PixelsRGBA8);
		return m_RHI->CreateTexture(desc, mips.data());
	}

	bool Renderer::PrepareMaterialTextures()
	{
		const bool useMips = WantsMaterialMips();
		bool changingTextures = false;
		for (auto& entry : m_TextureCache)
		{
			auto& resource = entry.second;
			auto& desc = resource.Description;
			const int levels = useMips ? MaterialMipLevels(desc) : 1;
			if (desc.MipLevels == levels) continue;
			if (!changingTextures)
			{
				if (m_LumenGI) m_LumenGI->Cancel();
				ResetFrameHistory();
				changingTextures = true;
			}

			// The cache entry is shared by all meshes using this image and color space.
			// Keep the old texture until its replacement has been created.
			auto replacement = desc;
			replacement.MipLevels = levels;
			const auto texture = UploadMaterialTexture(replacement, *resource.Image);
			if (!texture) return Fail(m_RHI->GetLastError().c_str());
			if (!m_RHI->DestroyTexture(resource.Texture))
			{
				const auto error = m_RHI->GetLastError();
				m_RHI->DestroyTexture(texture);
				return Fail(error.c_str());
			}
			resource.Texture = texture;
			desc.MipLevels = levels;
		}
		return true;
	}

	void Renderer::ConfigureMaterialPipeline(
		const MeshResource& mesh,
		PipelineState& pipeline) const
	{
		const MaterialData& material = mesh.Mesh.Material;
		pipeline.MaterialDiffuseColor = material.BaseColor;
		pipeline.MaterialAmbientColor = material.AmbientColor;
		pipeline.MaterialSpecularColor = material.SpecularColor;
        pipeline.MaterialSpecularColor.W=std::clamp(material.DielectricReflectance,0.0f,1.0f);
		pipeline.MaterialEmissiveColor = material.EmissiveColor;
		pipeline.MaterialShininess = material.Shininess;
		pipeline.SubsurfaceWeight = material.SubsurfaceWeight;
		pipeline.MaterialOpacity =
			mesh.Mesh.Color.W * material.BaseColor.W;
		pipeline.AlphaCutoff = m_Techniques.AlphaTesting ? material.AlphaCutoff : 0.0f;
		pipeline.DiffuseTexture = m_Techniques.TextureMapping ? TextureCacheEntry::Handle(mesh.DiffuseTexture) : TextureHandle{};
		pipeline.NormalTexture = m_Techniques.NormalMapping ? TextureCacheEntry::Handle(mesh.NormalTexture) : TextureHandle{};
		pipeline.OpacityTexture = TextureCacheEntry::Handle(mesh.OpacityTexture);
        pipeline.PackedMetallicRoughness = material.MetallicRoughnessImage != nullptr;
		pipeline.Lighting = m_RenderLighting;
		pipeline.Shading = m_Techniques.Shading;
		pipeline.Roughness = material.Roughness.value_or(m_Techniques.Roughness);
		pipeline.Metallic = material.Metallic.value_or(m_Techniques.Metallic);
		pipeline.ToonBands = m_Techniques.ToonBands;
		pipeline.BackFaceCulling = m_Techniques.BackFaceCulling;
        pipeline.Parallax=m_Techniques.Parallax;
        pipeline.Parallax.Enabled &= m_CurrentPass==PassType::GBuffer && !mesh.Translucent && material.AlphaCutoff<=0 && !material.OpacityImage;
        pipeline.ParallaxBoundsMinimum=mesh.BoundsMinimum; pipeline.ParallaxBoundsMaximum=mesh.BoundsMaximum;
	}

	bool Renderer::PrepareCascadedShadows()
	{
		m_CascadeFrame = {};
		m_RHI->SetCascadedShadowState({});
		if (Techniques::UsesPathTracing(m_Techniques.Pipeline) || m_Meshes.empty() || !m_ShadowsEnabled || m_RenderMode == RenderMode::Wireframe ||
			m_Techniques.Shadows != Techniques::ShadowRepresentation::Cascaded) return true;
		if (std::none_of(m_Meshes.begin(), m_Meshes.end(),
			[](const auto& pair) { return pair.second.Mesh.Visible; })) return true;
		const auto lightIndex = m_RenderLighting.ShadowLightIndex;
		if (lightIndex < 0 || static_cast<std::uint32_t>(lightIndex) >= m_RenderLighting.LightCount ||
			m_RenderLighting.Lights[lightIndex].Type != LightType::Directional) return true;
		if (!m_CascadeCameraValid) return Fail("CSM requires SetCameraMatrices or an automatic camera");
		std::vector<ShadowCasterBounds> casters;
		for (const auto& pair : m_Meshes)
		{
			const auto& mesh = pair.second;
			if (!mesh.Mesh.Visible || !mesh.Mesh.CastShadow || UsesPrivateTransparency(mesh)) continue;
			const float infinity = std::numeric_limits<float>::infinity();
			ShadowCasterBounds box {{infinity, infinity, infinity}, {-infinity, -infinity, -infinity}};
			for (unsigned corner = 0; corner < 8; ++corner)
			{
				const Float3 local {(corner & 1) ? mesh.BoundsMaximum.X : mesh.BoundsMinimum.X,
					(corner & 2) ? mesh.BoundsMaximum.Y : mesh.BoundsMinimum.Y,
					(corner & 4) ? mesh.BoundsMaximum.Z : mesh.BoundsMinimum.Z};
				const Float3 world = TransformPoint(mesh.Mesh.WorldTransform, local);
				box.Minimum = {std::min(box.Minimum.X, world.X), std::min(box.Minimum.Y, world.Y), std::min(box.Minimum.Z, world.Z)};
				box.Maximum = {std::max(box.Maximum.X, world.X), std::max(box.Maximum.Y, world.Y), std::max(box.Maximum.Z, world.Z)};
			}
			casters.push_back(box);
		}
        if (m_BoundHairGeometry) casters.push_back({m_BoundHairGeometry->BoundsMinimum,m_BoundHairGeometry->BoundsMaximum});
        const float filterRadiusTexels = m_Techniques.ShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS ?
            (m_Techniques.ShadowFilter.PCSS.AngularRadiusDegrees > 0 ?
                PCSSFootprintLimit(float(m_Techniques.ShadowFilter.PCSS.MaximumSearchRadius), m_CreateInfo.ShadowMapSize) : 0.0f) :
            m_Techniques.ShadowFilter.Mode == Techniques::ShadowFilterMode::VSM ? float(m_Techniques.ShadowFilter.VSM.BlurRadius) :
            m_Techniques.ShadowFilter.Mode == Techniques::ShadowFilterMode::PCF ? float(m_Techniques.ShadowFilter.Radius) : 0.0f;
		if (!BuildCascadedShadowFrame(m_CascadeCamera, m_RenderLighting.Lights[lightIndex], casters,
			m_Techniques.Cascades, m_CreateInfo.ShadowMapSize, filterRadiusTexels, m_CascadeFrame, m_LastError)) return false;
		// Publish only a complete map set; a failed allocation cannot leak partial frame bindings.
		auto maps = m_CascadeMaps;
		for (std::uint32_t index = 0; index < m_CascadeFrame.CascadeCount; ++index)
		{
			if (maps[index]) continue;
			TextureDesc desc;
			desc.Width = desc.Height = m_CreateInfo.ShadowMapSize;
			desc.Format = TextureFormat::Depth32; desc.RenderTarget = true;
			desc.DebugName = "Renderer.Cascade" + std::to_string(index);
			maps[index] = m_RHI->CreateTexture(desc, nullptr);
			if (!maps[index])
			{
				for (std::size_t created = 0; created < maps.size(); ++created)
					if (maps[created] && !m_CascadeMaps[created]) m_RHI->DestroyTexture(maps[created]);
				return Fail(m_RHI->GetLastError().c_str());
			}
		}
		m_CascadeMaps = maps;
		m_RHI->SetCascadedShadowState({m_CascadeFrame, maps, m_CreateInfo.ShadowMapSize});
		return true;
	}

    bool Renderer::IsRefractive(const MeshResource& mesh) const
    {
        return Techniques::UsesRefraction(m_Techniques) && m_RenderMode != RenderMode::Wireframe &&
            mesh.Mesh.Material.Refractive;
    }

    bool Renderer::IsDepthPeeled(const MeshResource& mesh) const
    {
        return Techniques::UsesDepthPeeling(m_Techniques) && m_RenderMode != RenderMode::Wireframe &&
            mesh.Translucent && !IsRefractive(mesh);
    }

    bool Renderer::UsesPrivateTransparency(const MeshResource& mesh) const
    { return IsRefractive(mesh) || IsDepthPeeled(mesh); }

    void Renderer::SetVegetation(double time, Float4 wind, Float4 gust)
    {
        const auto same=[](Float4 a,Float4 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; };
        if (time!=m_VegetationTime || !same(wind,m_VegetationWind) || !same(gust,m_VegetationGust)) ++m_VegetationRevision;
        m_VegetationTime=time; m_VegetationWind=wind; m_VegetationGust=gust;
    }

    bool Renderer::PrepareRayScene()
    {
        m_RHI->SetRayScene({}); m_RHI->SetRayShadowImage({}); m_RHI->SetRayRadianceFrame({});
        const bool rayShadows=m_Techniques.Shadows==Techniques::ShadowRepresentation::RayTraced && m_ShadowsEnabled && m_RenderMode!=RenderMode::Wireframe;
        const bool pathTracing=Techniques::UsesPathTracing(m_Techniques.Pipeline);
        const bool radiance=pathTracing || m_Techniques.Reflections.Mode==Techniques::ReflectionMode::RayTraced ||
            Techniques::UsesRayTracedGI(m_Techniques.GlobalIllumination);
        if (!radiance && !rayShadows && m_Techniques.AmbientOcclusion.Mode!=Techniques::AmbientOcclusionMode::RayTraced) return true;
        std::vector<const MeshResource*> ordered;
        for (const auto& entry:m_Meshes)
        {
            const auto& mesh=entry.second;
            // Visibility rays must retain offscreen blockers and non-shadow-casting
            // geometry. Reuse the finalized raster LOD, but never camera-cull it.
            if (mesh.Mesh.Visible && !UsesPrivateTransparency(mesh) &&
                !(!pathTracing && m_Techniques.AlphaBlending && mesh.Translucent)) ordered.push_back(&mesh);
        }
        std::sort(ordered.begin(),ordered.end(),[](const auto* a,const auto* b) { return a->Mesh.Id<b->Mesh.Id; });
        std::vector<std::pair<std::string,std::uint32_t>> lods;
        for (const auto* mesh:ordered) lods.emplace_back(mesh->Mesh.Id,mesh->SelectedLod);
        const std::array<bool,3> coverage {m_Techniques.TextureMapping,m_Techniques.AlphaTesting,pathTracing || m_Techniques.AlphaBlending};
        const bool shadingChanged=radiance && (!m_RayShading || m_RayShadingModel!=m_Techniques.Shading || m_RayNormalMapping!=m_Techniques.NormalMapping);
        const bool vegetationChanged=m_RayVegetationRevision!=m_VegetationRevision &&
            std::any_of(ordered.begin(),ordered.end(),[](const auto* mesh){ return mesh->Mesh.Vegetation!=0; });
        if (!m_RayScene || vegetationChanged || m_RaySceneRevision!=m_ProbeSceneRevision || m_RayGeometryViewSerial!=m_GeometryViewSerial || m_RaySceneCoverage!=coverage || m_RaySceneLods!=lods || shadingChanged)
        {
            std::vector<SceneRayMeshView> meshes;
            // Pose only when a scene-ray consumer requests geometry. Raster wind
            // keeps immutable resident meshes and never performs this CPU work.
            std::vector<std::vector<Vertex>> posed; posed.reserve(ordered.size());
            for (const auto* mesh:ordered)
            {
                SceneRayMeshView view;
                view.Vertices=&mesh->Geometry->Vertices; view.Indices=&mesh->SelectedIndices();
                if (mesh->Mesh.Vegetation)
                {
                    posed.push_back(mesh->Geometry->Vertices); view.Vertices=&posed.back();
                    const auto& worldMatrix=mesh->Mesh.WorldTransform;
                    const auto normalMatrix=BuildRealtimeNormalMatrix(worldMatrix);
                    const auto transform=[](const Matrix4& matrix,Float3 value,bool transpose=false)
                    {
                        const auto& m=matrix.Values;
                        return transpose?Float3{m[0]*value.X+m[4]*value.Y+m[8]*value.Z,m[1]*value.X+m[5]*value.Y+m[9]*value.Z,m[2]*value.X+m[6]*value.Y+m[10]*value.Z}:
                            Float3{m[0]*value.X+m[1]*value.Y+m[2]*value.Z,m[4]*value.X+m[5]*value.Y+m[6]*value.Z,m[8]*value.X+m[9]*value.Y+m[10]*value.Z};
                    };
                    for (auto& vertex:posed.back())
                    {
                        const auto world=TransformPoint(worldMatrix,vertex.Position);
                        const float height=std::max(0.0f,mesh->Mesh.Vegetation==1?vertex.TexCoord.Y:vertex.Position.Y);
                        const float time=static_cast<float>(m_VegetationTime),reference=mesh->Mesh.Vegetation==1?1.0f:14.0f;
                        const auto wind=m_VegetationWind,gust=m_VegetationGust;
                        const float phase=(world.X*wind.X+world.Z*wind.Y)*6.2831853f/std::max(gust.X,1.0f)-time*1.7f;
                        const float wave=0.55f+gust.Y*(0.35f*std::sin(phase)+0.22f*std::sin(phase*0.47f+1.3f));
                        const float flutter=gust.Z*std::sin(time*7.1f+world.X*1.83f+world.Z*2.17f);
                        const float bend=std::clamp(wind.W*(wave+flutter)*height*height/reference,-height*0.6f,height*0.6f);
                        const Float3 offset{wind.X*bend,-std::abs(bend)*0.22f,wind.Y*bend};
                        vertex.Position=Add(vertex.Position,transform(normalMatrix,offset,true));
                        auto normal=Normalize(transform(normalMatrix,vertex.Normal));
                        normal.Y-=(normal.X*offset.X+normal.Z*offset.Z)*2/std::max(height,0.01f);
                        vertex.Normal=Normalize(transform(worldMatrix,Normalize(normal),true));
                    }
                }
                view.World=mesh->Mesh.WorldTransform; view.Color=mesh->Mesh.Color; view.Material=mesh->Mesh.Material;
                view.CastShadow=mesh->Mesh.CastShadow; meshes.push_back(std::move(view));
            }
            const auto query=SceneRayQuery::Create(meshes,m_Techniques,m_LastError);
            if (!query) return false;
            auto candidate=std::make_shared<RaySceneUpload>();
            std::shared_ptr<RaySceneShadingUpload> shading;
            if (radiance)
            {
                shading=std::make_shared<RaySceneShadingUpload>();
                if (!query->ExportRadiance(*candidate,*shading,m_LastError)) return false;
            }
            else if (!query->Export(*candidate,m_LastError)) return false;
            // Quality, lighting and projection changes do not invalidate geometry.
            // Publish only after the complete coverage-aware BVH is validated.
            m_RayScene=std::move(candidate); m_RaySceneRevision=m_ProbeSceneRevision; m_RayGeometryViewSerial=m_GeometryViewSerial;
            m_RayShading=std::move(shading); m_RayShadingModel=m_Techniques.Shading; m_RayNormalMapping=m_Techniques.NormalMapping;
            m_RaySceneCoverage=coverage; m_RaySceneLods=std::move(lods); m_RayVegetationRevision=m_VegetationRevision;
        }
        if (radiance)
        {
            auto lights=m_RenderLighting.SceneLights;
            if (!lights)
            {
                if (m_RenderLighting.LightCount>MaximumSceneLights) return Fail("Inline ray lighting exceeds its capacity");
                if (!m_RayLights || !SameSceneLights(m_RayLighting,m_RenderLighting))
                    m_RayLights=SceneLightCollection::Create({m_RenderLighting.Lights.begin(),m_RenderLighting.Lights.begin()+m_RenderLighting.LightCount},-1,m_LastError);
                lights=m_RayLights;
            }
            if (!lights) return false;
            m_RayLights=lights;
            m_RayLighting=m_RenderLighting;
            m_RHI->SetRayRadianceFrame({m_RayScene,m_RayShading,lights,m_RenderLighting});
        }
        if (rayShadows)
        {
            RayShadowFrame frame; frame.Enabled=true; frame.Settings=m_Techniques.RayShadows;
            const auto index=m_RenderLighting.ShadowLightIndex;
            if (index>=0 && unsigned(index)<m_RenderLighting.LightCount) frame.PrimaryLight=m_RenderLighting.Lights[unsigned(index)].Type;
            if (!m_RayShadowImage || m_RayShadowImageSource!=m_RayScene ||
                frame.PrimaryLight!=m_RayShadowFrame.PrimaryLight || !(frame.Settings==m_RayShadowFrame.Settings))
            {
                auto image=std::make_shared<RayShadowImage>();
                if (!BuildRayShadowImage(*m_RayScene,frame,m_RHI->GetCapabilities().MaximumTextureSize,*image,m_LastError)) return false;
                m_RayShadowImage=std::move(image); m_RayShadowImageSource=m_RayScene; m_RayShadowFrame=frame;
            }
            m_RHI->SetRayShadowImage(m_RayShadowImage);
        }
        m_RHI->SetRayScene(m_RayScene); return true;
    }

    bool Renderer::PrepareDepthPeeling()
    {
        m_PeelingMeshes.clear(); m_RHI->SetDepthPeelingFrame({});
        if (!Techniques::UsesDepthPeeling(m_Techniques) || m_RenderMode == RenderMode::Wireframe) return true;
        for (const auto& entry:m_Meshes)
        {
            const auto& mesh=entry.second;
            if (!mesh.Mesh.Visible || !IsDepthPeeled(mesh) || !mesh.SelectedIndexCount()) continue;
            if (m_Techniques.FrustumCulling && !IntersectsFrustum(mesh.BoundsMinimum,mesh.BoundsMaximum,
                mesh.Mesh.WorldTransform,m_CullingViewProjection)) { ++m_FrustumCulledMeshCount; continue; }
            m_PeelingMeshes.push_back(&mesh);
        }
        if (m_PeelingMeshes.empty()) return true;
        // Stable IDs define exact-depth ties; storage/insertion order and camera
        // distance must not change their source-over order. Capture and replay
        // share the same resident LOD buffers and raster projection.
        std::sort(m_PeelingMeshes.begin(),m_PeelingMeshes.end(),[](const auto* a,const auto* b) { return a->Mesh.Id<b->Mesh.Id; });
        DepthPeelingFrame frame; frame.Enabled=true; frame.ViewProjection=m_RasterViewProjection; frame.Settings=m_Techniques.OIT;
        for (const auto* mesh:m_PeelingMeshes)
        {
            PipelineState material; ConfigureMaterialPipeline(*mesh,material);
            DepthPeelingDraw draw; draw.SourceKey=frame.Draws.size(); draw.World=mesh->Mesh.WorldTransform;
            draw.VertexBuffer=mesh->Geometry->VertexBuffer;
            draw.IndexBuffer=mesh->SelectedLod ? mesh->Geometry->LodIndexBuffers[mesh->SelectedLod-1] : mesh->Geometry->IndexBuffer;
            draw.IndexCount=mesh->SelectedIndexCount(); draw.FirstIndex=mesh->SelectedFirstIndex();
            draw.Opacity=material.MaterialOpacity; draw.AlphaCutoff=material.AlphaCutoff;
            draw.DiffuseTexture=material.DiffuseTexture; draw.OpacityTexture=material.OpacityTexture;
            draw.BackFaceCulling=material.BackFaceCulling; frame.Draws.push_back(draw);
        }
        m_RHI->SetDepthPeelingFrame(std::move(frame)); return true;
    }

    bool Renderer::RenderDepthPeeling()
    {
        if (m_PeelingMeshes.empty()) return true;
        m_CurrentPass=PassType::Color;
        for (unsigned layer=0;layer<m_Techniques.OIT.LayerCount;++layer)
        {
            if (!m_RHI->ProcessDepthPeeling(DepthPeelingPass::Capture)) return Fail(m_RHI->GetLastError().c_str());
            PassDesc pass; pass.DepthPeelingLayer=true; pass.ClearColorEnabled=pass.ClearDepthEnabled=false;
            if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
            for (const auto* mesh:m_PeelingMeshes)
            {
                PipelineState material; ConfigureMaterialPipeline(*mesh,material);
                material.Color=mesh->Mesh.Color; material.DepthWrite=false; material.BlendEnabled=true;
                material.SamplesShadowMap=m_ShadowsEnabled;
                if (!DrawMesh(*mesh,material)) return false;
                if (m_RenderMode==RenderMode::SolidWireframe)
                {
                    // Replay only edges at this layer's captured depth. Other
                    // transparent layers retain their original peel ordering.
                    material.Color={0.12f,0.82f,1.0f,1.0f};
                    material.WireframeOverlay=m_RHI->GetCapabilities().NativeAPI;
                    material.Fill=material.WireframeOverlay?FillMode::Solid:FillMode::Wireframe;
                    if (!DrawMesh(*mesh,material)) return false;
                }
            }
            if (!EndMeshPass() || !m_RHI->ProcessDepthPeeling(DepthPeelingPass::Accumulate))
                return Fail(m_RHI->GetLastError().c_str());
        }
        if (!m_RHI->ProcessDepthPeeling(DepthPeelingPass::Capture) || !m_RHI->ProcessDepthPeeling(DepthPeelingPass::Compose))
            return Fail(m_RHI->GetLastError().c_str());
        return true;
    }

    bool Renderer::PrepareRefraction()
    {
        m_RHI->SetRefractionFrame({});
        if (!Techniques::UsesRefraction(m_Techniques) || m_RenderMode == RenderMode::Wireframe) return true;
        // Keep immutable local geometry separate from changing selection, transforms
        // and materials. All resident LODs share one native vertex/index allocation.
        std::vector<std::shared_ptr<const RefractionSourceGeometry>> sources;
        RefractionFrame frame;
        for (auto& entry : m_Meshes)
        {
            auto& mesh = entry.second;
            if (!EnsureMeshLods(mesh)) return false;
            auto& source = mesh.Geometry->RefractionSource;
            if (!source)
            {
                auto snapshot = std::make_shared<RefractionSourceGeometry>();
                snapshot->Vertices = mesh.Geometry->Vertices;
                for (const auto& level : mesh.Geometry->LodChain.Levels) snapshot->Levels.push_back(level.Indices);
                source = std::move(snapshot);
            }
            const auto index = static_cast<std::uint32_t>(sources.size()); sources.push_back(source);
            if (!mesh.Mesh.Visible || !IsRefractive(mesh) || !mesh.SelectedIndexCount()) continue;
            if (m_Techniques.FrustumCulling && !IntersectsFrustum(mesh.BoundsMinimum,mesh.BoundsMaximum,
                mesh.Mesh.WorldTransform,m_CullingViewProjection)) { ++m_FrustumCulledMeshCount; continue; }
            const auto& color = mesh.Mesh.Color; const auto& material = mesh.Mesh.Material;
            RefractionDraw draw; draw.SourceIndex = index; draw.LodLevel = mesh.SelectedLod;
            draw.World = mesh.Mesh.WorldTransform;
            draw.Color = {color.X*material.BaseColor.X,color.Y*material.BaseColor.Y,color.Z*material.BaseColor.Z,color.W*material.BaseColor.W};
            draw.DiffuseTexture = m_Techniques.TextureMapping ? TextureCacheEntry::Handle(mesh.DiffuseTexture) : TextureHandle{};
            draw.NormalTexture = m_Techniques.NormalMapping ? TextureCacheEntry::Handle(mesh.NormalTexture) : TextureHandle{};
            draw.OpacityTexture = TextureCacheEntry::Handle(mesh.OpacityTexture); draw.AlphaCutoff = m_Techniques.AlphaTesting ? material.AlphaCutoff : 0;
            draw.ForceCoverage = !mesh.Translucent; draw.BackFaceCulling = m_Techniques.BackFaceCulling;
            frame.Draws.push_back(draw);
        }
        if (!frame.Draws.empty())
        {
            if (!m_RefractionGeometry.Prepare(sources,m_LastError)) return false;
            frame.Geometry = m_RefractionGeometry.GetGeometry();
        }
        m_RHI->SetRefractionFrame(std::move(frame)); return true;
    }

    bool Renderer::PrepareLightmaps()
    {
        m_RHI->SetLightmapFrame({});
        if (!Techniques::UsesLightmaps(m_Techniques.GlobalIllumination) || m_RenderMode==RenderMode::Wireframe)
        {
            if (m_LightmapStatus.State!=LightmapBakeState::Idle) m_BakedLightmaps.Cancel();
            m_LightmapStatus={}; return true;
        }
        // Persistent UV2 ranges use scene identifiers, never hash-table order.
        std::vector<std::string> identifiers; identifiers.reserve(m_Meshes.size());
        for (const auto& entry:m_Meshes) identifiers.push_back(entry.first);
        std::sort(identifiers.begin(),identifiers.end());
        if (m_LightmapSourceRevision!=m_ProbeSceneRevision || m_LightmapInstances.size()!=m_Meshes.size() || m_LightmapVirtualSerial!=(m_VirtualGeometry?m_VirtualGeometry->Serial:0) || m_LightmapNaniteSerial!=(m_NaniteGeometry?m_NaniteGeometry->Serial:0))
        {
            // Share one immutable geometry snapshot across instances. Build all
            // endpoint LODs before requesting a bake so camera changes need no
            // new UV2 layout, worker restart, or transport-scene rebuild.
            std::unordered_map<const MeshGeometryResource*,std::shared_ptr<const LightmapSourceGeometry>> geometries;
            std::vector<LightmapSourceInstance> instances; instances.reserve(m_Meshes.size());
            constexpr float Largest=std::numeric_limits<float>::max();
            Float3 minimum{Largest,Largest,Largest},maximum{-Largest,-Largest,-Largest}; bool hasBounds=false;
            for (const auto& identifier:identifiers)
            {
                auto& resource=m_Meshes.at(identifier);
                std::size_t part=0; auto* family=VirtualFamilyFor(resource,part);
                auto& geometry=geometries[resource.SourceGeometry];
                if (family)
                {
                    if (!PrepareVirtualLightmapSources(*family)) return false;
                    geometry=family->LightmapSources[part];
                }
                else if (auto* nanite=NaniteFamilyFor(resource,part))
                {
                    if (!PrepareNaniteLightmapSources(*nanite)) return false;
                    geometry=nanite->LightmapSources[part];
                }
                else if (!geometry)
                {
                    if (!EnsureGeometryLods(*resource.SourceGeometry,resource.Mesh.Id)) return false;
                    auto snapshot=std::make_shared<LightmapSourceGeometry>(); snapshot->Vertices=resource.SourceGeometry->Vertices;
                    for (const auto& level:resource.SourceGeometry->LodChain.Levels) snapshot->Levels.push_back(level.Indices);
                    geometry=std::move(snapshot);
                }
                LightmapSourceInstance instance; instance.Geometry=geometry; instance.World=resource.Mesh.WorldTransform;
                instance.Color=resource.Mesh.Color; instance.Material=resource.Mesh.Material;
                instance.Visible=resource.Mesh.Visible && !UsesPrivateTransparency(resource); instance.CastShadow=resource.Mesh.CastShadow; instances.push_back(std::move(instance));
                if (resource.Mesh.Visible) for (unsigned corner=0;corner<8;++corner)
                {
                    const auto point=TransformPoint(resource.Mesh.WorldTransform,{corner&1 ? resource.BoundsMaximum.X : resource.BoundsMinimum.X,
                        corner&2 ? resource.BoundsMaximum.Y : resource.BoundsMinimum.Y,corner&4 ? resource.BoundsMaximum.Z : resource.BoundsMinimum.Z});
                    minimum={std::min(minimum.X,point.X),std::min(minimum.Y,point.Y),std::min(minimum.Z,point.Z)};
                    maximum={std::max(maximum.X,point.X),std::max(maximum.Y,point.Y),std::max(maximum.Z,point.Z)}; hasBounds=true;
                }
            }
            constexpr float TraceExtentMultiplier=2,MinimumTraceDistance=0.001f;
            const auto extent=Subtract(maximum,minimum);
            m_LightmapTraceDistance=hasBounds ? std::max(std::sqrt(Dot(extent,extent))*TraceExtentMultiplier,MinimumTraceDistance) : 1;
            m_LightmapInstances=std::move(instances); m_LightmapSourceRevision=m_ProbeSceneRevision;
            m_LightmapVirtualSerial=m_VirtualGeometry?m_VirtualGeometry->Serial:0; m_LightmapNaniteSerial=m_NaniteGeometry?m_NaniteGeometry->Serial:0; ++m_LightmapGeometryRevision;
        }
        auto request=MakeLightmapBakeRequest(m_LightmapInstances,m_Techniques,m_RenderLighting,m_ShadowsEnabled,m_LightmapTraceDistance);
        request.SceneRevision=m_LightmapGeometryRevision;
        if (!m_BakedLightingDirectory.empty()) request.Archive=m_BakedLightingDirectory/"irradiance.evlightmap";
        m_BakedLightmaps.Request(std::move(request)); m_LightmapStatus=m_BakedLightmaps.Status();
        if (!m_LightmapStatus.Current()) return true;
        LightmapFrame frame; frame.Source=m_LightmapStatus.Published;
        frame.KeepFirstSurface=Techniques::UsesDeferredLighting(m_Techniques.Pipeline) ||
            !(m_Techniques.ContactShadows.Enabled || m_Techniques.Pipeline==Techniques::RenderingPipeline::ForwardPlus);
        const auto& ranges=frame.Source->Receivers->Ranges;
        std::size_t rangeIndex=0; std::uint32_t instanceIndex=0;
        for (const auto& identifier:identifiers)
        {
            const auto& resource=m_Meshes.at(identifier); const auto index=instanceIndex++;
            while (rangeIndex<ranges.size() && ranges[rangeIndex].InstanceIndex<index) ++rangeIndex;
            if (!resource.Mesh.Visible || UsesPrivateTransparency(resource) || (resource.Translucent && m_Techniques.AlphaBlending) ||
                (m_Techniques.FrustumCulling && !IntersectsFrustum(resource.BoundsMinimum,resource.BoundsMaximum,
                    resource.Mesh.WorldTransform,m_CullingViewProjection))) continue;
            for (auto current=rangeIndex;current<ranges.size() && ranges[current].InstanceIndex==index;++current)
            {
                const auto& range=ranges[current]; if (!range.VertexCount) continue;
                std::size_t part=0; const auto* family=VirtualFamilyFor(resource,part);
                if (family)
                { if (range.Tag==LightmapUnsegmented || !std::binary_search(family->SelectedClusters.begin(),family->SelectedClusters.end(),range.Tag)) continue; }
                else if (const auto* nanite=NaniteFamilyFor(resource,part))
                { if (range.Tag==LightmapUnsegmented || !std::binary_search(nanite->SelectedClusters.begin(),nanite->SelectedClusters.end(),range.Tag)) continue; }
                else if (range.Tag!=LightmapUnsegmented || range.LodLevel!=resource.SelectedLod) continue;
                const auto& color=resource.Mesh.Color; const auto& material=resource.Mesh.Material;
                LightmapDraw draw; draw.FirstVertex=range.FirstVertex; draw.VertexCount=range.VertexCount;
                draw.World=resource.Mesh.WorldTransform;
                draw.DiffuseColor={color.X*material.BaseColor.X,color.Y*material.BaseColor.Y,color.Z*material.BaseColor.Z,color.W*material.BaseColor.W};
                draw.DiffuseTexture=m_Techniques.TextureMapping ? TextureCacheEntry::Handle(resource.DiffuseTexture) : TextureHandle{};
                draw.Metallic=material.Metallic; draw.PackedMetallicRoughness=bool(material.MetallicRoughnessImage);
                draw.OpacityTexture=TextureCacheEntry::Handle(resource.OpacityTexture); draw.AlphaCutoff=m_Techniques.AlphaTesting ? material.AlphaCutoff : 0;
                draw.BackFaceCulling=m_Techniques.BackFaceCulling; frame.Draws.push_back(draw);
            }
        }
        // Authored irradiance replaces the unoccluded ambient approximation.
        // Keeping that constant fill underneath the baked result masks its
        // spatial visibility and counts indirect illumination twice.
        if (frame.Source->LoadedFromAsset && !frame.Draws.empty()) m_RenderLighting.AmbientStrength=0;
        m_RHI->SetLightmapFrame(std::move(frame)); return true;
    }

    bool Renderer::PrepareHairFur()
    {
        const bool enabled=Techniques::UsesHairFur(m_Techniques.Hair) && m_RenderMode!=RenderMode::Wireframe;
        if (!enabled)
        {
            if (m_HairStatus.State!=HairUpdateState::Idle) m_HairGeometry.Cancel();
            m_HairStatus={};
            if (m_BoundHairGeometry) { m_BoundHairGeometry.reset(); ResetFrameHistory(); }
            return true;
        }
        const auto& settings=m_Techniques.Hair;
        const auto selected=settings.UseSelectedObject?m_HairSelectionGroup:0;
        const auto sourceRevision=m_ProbeSceneRevision-m_HairPoseRevision;
        std::vector<Matrix4> pose,deltas;
        bool rigid=true;
        for (std::size_t index=0;index<m_HairInstances.size();++index)
        {
            const auto found=m_Meshes.find(m_HairInstanceIds[index]);
            if (found==m_Meshes.end()) { rigid=false; break; }
            const auto world=found->second.Mesh.WorldTransform; pose.push_back(world);
            if (world.Values==m_HairInstances[index].World.Values) { deltas.push_back({}); continue; }
            Maths::Matrix4 inverse;
            if (!Maths::Matrix4(m_HairInstances[index].World.Values).Inverse(inverse)) { rigid=false; break; }
            Matrix4 baseInverse; std::copy(inverse.Data(),inverse.Data()+16,baseInverse.Values.begin());
            const auto delta=Multiply(world,baseInverse); deltas.push_back(delta);
            constexpr float RigidTolerance=0.0001f;
            for (unsigned row=0;row<3;++row) for (unsigned other=0;other<3;++other)
            {
                float product=0; for (unsigned column=0;column<3;++column) product+=delta.Values[row*4+column]*delta.Values[other*4+column];
                if (std::abs(product-(row==other?1.0f:0.0f))>RigidTolerance) rigid=false;
            }
        }
        if (!rigid || m_HairSourceRevision!=sourceRevision || m_HairSourceSelection!=selected ||
            m_HairSourceSelectedOnly!=settings.UseSelectedObject || m_HairSourceAlphaBlending!=m_Techniques.AlphaBlending)
        {
            std::vector<const MeshResource*> meshes;
            for (const auto& entry:m_Meshes)
            {
                const auto& mesh=entry.second;
                if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending) ||
                    (settings.UseSelectedObject && (!selected || mesh.Mesh.SelectionGroup!=selected))) continue;
                meshes.push_back(&mesh);
            }
            std::sort(meshes.begin(),meshes.end(),[](const auto* a,const auto* b){return a->Mesh.Id<b->Mesh.Id;});
            std::vector<HairSourceInstance> instances; instances.reserve(meshes.size()); m_HairInstanceIds.clear();
            for (const auto* mesh:meshes)
            {
                auto& source=mesh->SourceGeometry->HairSource;
                if (!source)
                {
                    auto copy=std::make_shared<HairSourceGeometry>(); copy->Vertices=mesh->SourceGeometry->Vertices; copy->Indices=mesh->SourceGeometry->Indices;
                    source=std::move(copy);
                }
                HairSourceInstance instance; instance.Geometry=source; instance.World=mesh->Mesh.WorldTransform;
                instance.Color=mesh->Mesh.Color; instance.Material=mesh->Mesh.Material; instance.CastShadow=mesh->Mesh.CastShadow;
                instances.push_back(std::move(instance)); m_HairInstanceIds.push_back(mesh->Mesh.Id);
            }
            pose.clear(); deltas.clear(); for (const auto& instance:instances) { pose.push_back(instance.World); deltas.push_back({}); }
            ++m_HairGroomSourceId;
            m_HairInstances=std::move(instances); m_HairSourceRevision=sourceRevision; m_HairSourceSelection=selected;
            m_HairSourceSelectedOnly=settings.UseSelectedObject; m_HairSourceAlphaBlending=m_Techniques.AlphaBlending;
        }
        HairGeometryRequest request; request.Instances=m_HairInstances; request.SceneRevision=m_HairGroomSourceId;
        request.SelectionGroup=selected; request.Groom=settings; request.Materials=m_Techniques;
        m_HairGeometry.Request(std::move(request)); m_HairStatus=m_HairGeometry.Status();
        const auto base=m_HairStatus.Current() && !m_HairStatus.Published->Geometry->Indices.empty()?m_HairStatus.Published->Geometry:nullptr;
        const bool replaced=base!=m_HairPoseBase;
        bool poseChanged=pose.size()!=m_HairLastPose.size();
        if (!poseChanged) for (std::size_t index=0;index<pose.size();++index) poseChanged|=pose[index].Values!=m_HairLastPose[index].Values;
        if (replaced || poseChanged)
        {
            m_HairPoseBase=base; m_HairLastPose=pose; m_HairPosedGeometry=base;
            bool moved=false; for (const auto& delta:deltas) moved|=delta.Values!=Matrix4{}.Values;
            if (base && moved)
            {
                auto posed=std::make_shared<HairStrandGeometry>(*base);
                if (posed->VertexSources.size()!=posed->Vertices.size()) return Fail("Posed groom has no complete source-instance correspondence");
                const float infinity=std::numeric_limits<float>::infinity();
                posed->BoundsMinimum={infinity,infinity,infinity}; posed->BoundsMaximum={-infinity,-infinity,-infinity};
                for (std::size_t index=0;index<posed->Vertices.size();++index)
                {
                    const auto source=posed->VertexSources[index]; if (source>=deltas.size()) return Fail("Posed groom source instance is missing");
                    auto& vertex=posed->Vertices[index]; const auto& delta=deltas[source]; vertex.Position=TransformPoint(delta,vertex.Position);
                    const auto tangent=vertex.Tangent;
                    vertex.Tangent=Normalize({delta.Values[0]*tangent.X+delta.Values[1]*tangent.Y+delta.Values[2]*tangent.Z,
                        delta.Values[4]*tangent.X+delta.Values[5]*tangent.Y+delta.Values[6]*tangent.Z,delta.Values[8]*tangent.X+delta.Values[9]*tangent.Y+delta.Values[10]*tangent.Z});
                    const float radius=std::abs(vertex.Ribbon.X); const auto p=vertex.Position;
                    posed->BoundsMinimum={std::min(posed->BoundsMinimum.X,p.X-radius),std::min(posed->BoundsMinimum.Y,p.Y-radius),std::min(posed->BoundsMinimum.Z,p.Z-radius)};
                    posed->BoundsMaximum={std::max(posed->BoundsMaximum.X,p.X+radius),std::max(posed->BoundsMaximum.Y,p.Y+radius),std::max(posed->BoundsMaximum.Z,p.Z+radius)};
                }
                m_HairPosedGeometry=std::move(posed);
            }
        }
        const auto geometry=base?m_HairPosedGeometry:nullptr;
        if (geometry!=m_BoundHairGeometry) { m_BoundHairGeometry=geometry; if (replaced) ResetFrameHistory(); }
        return true;
    }
    HairFurDesc Renderer::HairView(const Matrix4& projection) const
    {
        HairFurDesc view; view.ViewProjection=projection; view.Lighting=m_RenderLighting;
        view.Orthographic=projection.Values[12]==0 && projection.Values[13]==0 && projection.Values[14]==0;
        Maths::Matrix4 inverse;
        if (Maths::Matrix4(projection.Values).Inverse(inverse))
        {
            const auto unproject=[&](float z,float w)
            {
                const auto* matrix=inverse.Data(); const float divisor=matrix[14]*z+matrix[15]*w;
                return Float3{(matrix[2]*z+matrix[3]*w)/divisor,(matrix[6]*z+matrix[7]*w)/divisor,(matrix[10]*z+matrix[11]*w)/divisor};
            };
            view.Forward=Normalize(Subtract(unproject(1,1),unproject(0,1)));
            if (m_CurrentPass==PassType::Shadow && !view.Orthographic) view.Lighting.CameraPosition=unproject(1,0);
        }
        return view;
    }

    bool Renderer::PrepareLumenGI()
    {
        const auto previous=m_RHI->GetLumenLighting(); const auto previousWorld=m_RHI->GetLumenWorldCache().Radiance;
        const auto previousGlobal=m_RHI->GetLumenGlobalDistance();
        m_RHI->SetLumenGlobalDistance({});
        m_RHI->SetLumenWorldCache({});
        m_RHI->SetLumenLighting({});
        if (!Techniques::UsesLumenScene(m_Techniques) || m_RenderMode==RenderMode::Wireframe)
        { if (m_LumenGI) m_LumenGI->Cancel(); if (m_LumenWorldCache) m_LumenWorldCache->Cancel(); if (m_LumenGlobalDistance) m_LumenGlobalDistance->Cancel(); if (previous || previousWorld || previousGlobal) ResetFrameHistory(); return true; }
        if (!m_LumenGI) m_LumenGI=std::make_unique<LumenRenderCache>(*m_RHI);
        const auto geometrySerial=m_PreparedGeometry?m_PreparedGeometry->Serial:0;
        if (m_LumenSceneRevision!=m_ProbeSceneRevision || m_LumenGeometrySerial!=geometrySerial)
        { ++m_LumenRevision; m_LumenSceneRevision=m_ProbeSceneRevision; m_LumenGeometrySerial=geometrySerial; }
        std::vector<LumenSourceInstance> instances;
        for (auto& entry:m_Meshes)
        {
            auto& mesh=entry.second;
            if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (m_Techniques.AlphaBlending && mesh.Translucent)) continue;
            auto& geometry=*mesh.SourceGeometry;
            if (!geometry.LumenSource)
            { auto snapshot=std::make_shared<LumenSourceGeometry>(); snapshot->Vertices=geometry.Vertices; snapshot->Indices=geometry.Indices; geometry.LumenSource=std::move(snapshot); }
            if (!mesh.LumenKey) mesh.LumenKey=++m_NextLumenKey;
            LumenSourceInstance source; source.Key=mesh.LumenKey; source.Geometry=geometry.LumenSource; source.Material=mesh.Mesh.Material;
            source.Color=mesh.Mesh.Color; source.World=mesh.Mesh.WorldTransform; source.CastShadow=mesh.Mesh.CastShadow;
            PipelineState material; ConfigureMaterialPipeline(mesh,material); auto& draw=source.Capture;
            draw.BaseColor={source.Color.X*material.MaterialDiffuseColor.X,source.Color.Y*material.MaterialDiffuseColor.Y,
                source.Color.Z*material.MaterialDiffuseColor.Z,material.MaterialOpacity};
            draw.Emission={material.MaterialEmissiveColor.X,material.MaterialEmissiveColor.Y,material.MaterialEmissiveColor.Z};
            draw.Diffuse=material.DiffuseTexture; draw.Normal=material.NormalTexture; draw.Opacity=material.OpacityTexture; draw.Cutoff=material.AlphaCutoff;
            draw.Roughness=material.Roughness; draw.Metallic=material.Shading==Techniques::ShadingModel::PBR?material.Metallic:0; draw.BackFaceCulling=material.BackFaceCulling;
            instances.push_back(std::move(source));
        }
        std::sort(instances.begin(),instances.end(),[](const auto& a,const auto& b) { return a.Key<b.Key; });
        m_LumenGI->Request(std::move(instances),m_LumenRevision,m_Techniques,m_RenderLighting,m_ShadowsEnabled,m_BoundHairGeometry,m_HairPoseBase); m_LumenGI->Advance();
        const auto status=m_LumenGI->Status(); if (status.HasLighting()) m_RHI->SetLumenLighting(status.Lighting);
        const auto& gi=m_Techniques.GlobalIllumination; const auto& world=gi.Lumen.WorldCache;
        if (status.HasLighting() && gi.Lumen.GlobalDistance.Enabled)
        {
            if (!m_LumenGlobalDistance) m_LumenGlobalDistance=std::make_unique<LumenGlobalDistanceCache>(*m_RHI);
            const auto& global=gi.Lumen.GlobalDistance; LumenGlobalDistanceSettings settings;
            settings.CameraPosition=m_RenderLighting.CameraPosition; settings.CellSize=gi.MaximumDistance*global.CellScale;
            settings.GridSide=global.GridSide; settings.CascadeCount=global.CascadeCount;
            if (!m_LumenGlobalDistance->Request(status.Distance,settings,m_LastError) || !m_LumenGlobalDistance->Advance(global.VoxelsPerUpdate,m_LastError)) return false;
            if (m_LumenGlobalDistance->CanUseSnapshot(status.Distance,settings)) m_RHI->SetLumenGlobalDistance(m_LumenGlobalDistance->Handle());
        }
        else if (m_LumenGlobalDistance) m_LumenGlobalDistance->Cancel();
        if (status.HasLighting() && world.Enabled)
        {
            if (!m_LumenWorldCache) m_LumenWorldCache=std::make_unique<LumenWorldRadianceCache>(*m_RHI);
            LumenWorldCacheSettings settings; settings.CellSize=gi.MaximumDistance*world.CellScale;
            settings.Dimensions={world.GridSide,world.GridSide,world.GridSide}; settings.DirectionalSide=world.DirectionalSide; settings.CascadeCount=world.CascadeCount;
            settings.DepthLayers=world.DepthLayers;
            settings.MinimumDistance=2*settings.CellSize; settings.MaximumDistance=gi.MaximumDistance; settings.MaximumSteps=gi.MaximumSteps;
            settings.ScrollWithCamera=true; settings.CameraPosition=m_RenderLighting.CameraPosition; settings.ProbesPerUpdate=world.ProbesPerUpdate;
            if (!m_LumenWorldCache->Request(status.Lighting,settings,m_LastError) || !m_LumenWorldCache->Advance(m_LastError)) return false;
            const auto& snapshot=m_LumenWorldCache->Snapshot(); const auto& sky=gi.Lumen.SkyRadiance;
            if (m_LumenWorldCache->CanUseSnapshot(status.Lighting,settings))
                m_RHI->SetLumenWorldCache({snapshot.Radiance,snapshot.Layout,{sky[0],sky[1],sky[2]},world.Parallax,snapshot.Coarser,world.BlendCells,MakeLumenEnvironmentLobe(gi.Lumen),world.AngularFilterRadius});
        }
        else if (m_LumenWorldCache) m_LumenWorldCache->Cancel();
        // A pending or newly published cache changes radiance even when the
        // camera is stationary. Discard history before describing this frame.
        if (previous.Value!=m_RHI->GetLumenLighting().Value || previousWorld.Value!=m_RHI->GetLumenWorldCache().Radiance.Value || previousGlobal.Value!=m_RHI->GetLumenGlobalDistance().Value) ResetFrameHistory();
        return true;
    }

    bool Renderer::PrepareVoxelGI()
    {
        m_RHI->SetVoxelGIFrame({});
        const bool enabled=Techniques::UsesVoxelGI(m_Techniques.GlobalIllumination) && m_RenderMode!=RenderMode::Wireframe;
        if (!enabled)
        {
            if (m_VoxelStatus.State!=VoxelUpdateState::Idle) m_VoxelGI.Cancel();
            m_VoxelStatus={};
            if (m_BoundVoxelVolume) { m_BoundVoxelVolume.reset(); ResetFrameHistory(); }
            return true;
        }
        if (m_VoxelSourceRevision!=m_ProbeSceneRevision || m_VoxelInstances.size()!=m_Meshes.size())
        {
            // Camera-selected LOD and frustum culling never remove transport
            // geometry. Instances share immutable original topology snapshots.
            std::unordered_map<const MeshGeometryResource*,std::shared_ptr<const VoxelSourceGeometry>> geometries;
            std::vector<VoxelSourceInstance> instances; instances.reserve(m_Meshes.size());
            for (const auto& entry:m_Meshes)
            {
                const auto& resource=entry.second; auto& geometry=geometries[resource.SourceGeometry];
                if (!geometry)
                {
                    auto copy=std::make_shared<VoxelSourceGeometry>(); copy->Vertices=resource.SourceGeometry->Vertices; copy->Indices=resource.SourceGeometry->Indices;
                    geometry=std::move(copy);
                }
                VoxelSourceInstance instance; instance.Geometry=geometry; instance.World=resource.Mesh.WorldTransform; instance.Color=resource.Mesh.Color;
                instance.Material=resource.Mesh.Material; instance.Visible=resource.Mesh.Visible && !UsesPrivateTransparency(resource);
                instance.CastShadow=resource.Mesh.CastShadow; instances.push_back(std::move(instance));
            }
            m_VoxelInstances=std::move(instances); m_VoxelSourceRevision=m_ProbeSceneRevision;
        }
        const auto& settings=m_Techniques.GlobalIllumination;
        VoxelRadianceRequest request; request.Instances=m_VoxelInstances; request.SceneRevision=m_ProbeSceneRevision;
        request.Materials=m_Techniques; request.Grid=settings.Voxels; request.Lighting=m_RenderLighting;
        request.Shadows=m_ShadowsEnabled && settings.Voxels.Shadows;
        m_VoxelGI.Request(std::move(request)); m_VoxelStatus=m_VoxelGI.Status();
        const auto volume=m_VoxelStatus.Current() ? m_VoxelStatus.Published->Radiance:nullptr;
        // Reset before camera/temporal sampling is described for this frame.
        // Pending/failed requests never bind an older scene's completed volume.
        if (m_BoundVoxelVolume!=volume) { m_BoundVoxelVolume=volume; ResetFrameHistory(); }
        if (volume)
        {
            VoxelGIFrame frame; frame.Volume=volume; frame.Settings=settings.Voxels; frame.Intensity=settings.Intensity;
            frame.MaximumDistance=settings.MaximumDistance; frame.Shading=m_Techniques.Shading; m_RHI->SetVoxelGIFrame(std::move(frame));
        }
        return true;
    }

    bool Renderer::PrepareBakedLightProbes()
    {
        if (!Techniques::UsesBakedProbes(m_Techniques.GlobalIllumination))
        { m_RHI->SetBakedProbeSamples({}); return true; }
        std::vector<SceneRayMeshView> meshes;
        meshes.reserve(m_Meshes.size());
        std::vector<std::string> identifiers; identifiers.reserve(m_Meshes.size());
        for (const auto& entry:m_Meshes) identifiers.push_back(entry.first);
        std::sort(identifiers.begin(),identifiers.end());
        for (const auto& identifier:identifiers)
        {
            const auto& resource = m_Meshes.at(identifier);
            SceneRayMeshView view;
            view.Vertices = &resource.SourceGeometry->Vertices; view.Indices = &resource.SourceGeometry->Indices;
            view.World = resource.Mesh.WorldTransform; view.Color = resource.Mesh.Color;
            view.Material = resource.Mesh.Material; view.Visible = resource.Mesh.Visible && !UsesPrivateTransparency(resource); view.CastShadow = resource.Mesh.CastShadow;
            meshes.push_back(std::move(view));
        }
        const auto archive=m_BakedLightingDirectory.empty()?std::filesystem::path{}:m_BakedLightingDirectory/"irradiance.evprobes";
        if (!m_BakedProbes.Update(meshes,m_ProbeSceneRevision,m_Techniques,m_RenderLighting,m_ShadowsEnabled,m_LastError,archive)) return false;
        m_RHI->SetBakedProbeSamples(m_BakedProbes.Samples());
        return true;
    }

	bool Renderer::PrepareEnvironmentCapture()
	{
		m_EnvironmentFrame = {};
		m_RHI->SetEnvironmentCaptureFrame({});
		if (!Techniques::UsesEnvironmentTexture(m_Techniques.Reflections)) return true;
        if (m_EnvironmentSource)
        {
            m_EnvironmentFrame.Source=m_EnvironmentSource;
            m_EnvironmentFrame.Resolution=m_EnvironmentSource->Resolution;
            if (!ValidEnvironmentSource(m_EnvironmentFrame)) return Fail("Invalid environment radiance faces");
            for (auto extent=m_EnvironmentFrame.Resolution;extent;extent>>=1) ++m_EnvironmentFrame.MipCount;
            m_RHI->SetEnvironmentCaptureFrame(m_EnvironmentFrame);
            return true;
        }
		// Fit all visible capture geometry, including objects outside the main view.
		// The camera-centered probe avoids placing the default probe inside a demo mesh.
		EnvironmentCaptureBounds bounds {m_RenderLighting.CameraPosition, m_RenderLighting.CameraPosition};
		for (const auto& pair : m_Meshes)
		{
			const auto& mesh = pair.second;
			if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
			for (unsigned corner = 0; corner < 8; ++corner)
			{
				const Float3 local {(corner & 1) ? mesh.BoundsMaximum.X : mesh.BoundsMinimum.X,
					(corner & 2) ? mesh.BoundsMaximum.Y : mesh.BoundsMinimum.Y,
					(corner & 4) ? mesh.BoundsMaximum.Z : mesh.BoundsMinimum.Z};
				const auto world = TransformPoint(mesh.Mesh.WorldTransform, local);
				if (!std::isfinite(world.X) || !std::isfinite(world.Y) || !std::isfinite(world.Z))
					return Fail("Environment capture bounds must be finite");
				bounds.Minimum = {std::min(bounds.Minimum.X, world.X), std::min(bounds.Minimum.Y, world.Y), std::min(bounds.Minimum.Z, world.Z)};
				bounds.Maximum = {std::max(bounds.Maximum.X, world.X), std::max(bounds.Maximum.Y, world.Y), std::max(bounds.Maximum.Z, world.Z)};
			}
		}
		if (!BuildEnvironmentCaptureFrame(m_RenderLighting.CameraPosition, bounds, m_Techniques.Reflections.CaptureResolution,
			m_EnvironmentFrame, m_LastError)) return false;
		m_RHI->SetEnvironmentCaptureFrame(m_EnvironmentFrame);
		return true;
	}

	bool Renderer::RenderEnvironmentCapture()
	{
        if (m_EnvironmentFrame.Source)
            return m_RHI->CompleteEnvironmentCapture() || Fail(m_RHI->GetLastError().c_str());
		m_CurrentPass = PassType::EnvironmentCapture;
		for (std::uint32_t face = 0; face < EnvironmentFaceCount; ++face)
		{
			m_EnvironmentDrawProjection = EnvironmentRasterProjection(m_EnvironmentFrame.Faces[face], GetBackend());
			PassDesc pass;
			pass.Type = PassType::EnvironmentCapture;
			pass.EnvironmentFaceIndex = face;
			pass.ClearColor = {0, 0, 0, 1};
			if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
			for (const auto& pair : m_Meshes)
			{
				const auto& mesh = pair.second;
				if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
				if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
					mesh.Mesh.WorldTransform, m_EnvironmentDrawProjection)) continue;
				PipelineState pipeline;
				pipeline.Color = mesh.Mesh.Color;
				ConfigureMaterialPipeline(mesh, pipeline);
				pipeline.Lighting.CameraPosition = m_EnvironmentFrame.Position;
				// Capture direct/ambient/emissive radiance without recursive screen effects.
				// Main-view CSM coverage is unsuitable for the other five directions.
				pipeline.SamplesShadowMap = false;
				if (!DrawMesh(mesh, pipeline)) return false;
			}
			if (!EndMeshPass()) return false;
		}
		if (!m_RHI->CompleteEnvironmentCapture()) return Fail(m_RHI->GetLastError().c_str());
		return true;
	}

	bool Renderer::PreparePlanarCapture()
	{
		m_PlanarFrame = {};
		m_RHI->SetPlanarCaptureFrame({});
		if (!Techniques::UsesPlanarCapture(m_Techniques.Reflections)) return true;
		// Lookup reconstructs raster depth; capture must freeze the same jittered
        // main matrix after temporal sampling has selected this frame's camera.
		if (!BuildPlanarCaptureFrame(m_RasterViewProjection, m_RenderLighting.CameraPosition, m_RenderWidth, m_RenderHeight,
			m_Techniques.Reflections.Planar, m_PlanarFrame, m_LastError)) return false;
		m_RHI->SetPlanarCaptureFrame(m_PlanarFrame);
		return true;
	}

	bool Renderer::RenderPlanarCapture()
	{
		m_CurrentPass = PassType::PlanarCapture;
		PassDesc pass;
		pass.Type = PassType::PlanarCapture;
		pass.ClearColor = {0, 0, 0, 1};
        if (m_PlanarFrame.Active() && m_Techniques.Reflections.Skybox && m_EnvironmentFrame.Source)
        {
            // Geometry replaces this low-frequency sky background in the mirror.
            const auto& inverse=m_PlanarFrame.InverseMainViewProjection;
            const auto forward=Normalize(Subtract(TransformPoint(inverse,{0,0,0.5f}),TransformPoint(inverse,{0,0,0})));
            const auto& plane=m_PlanarFrame.ReceiverPlane;
            const Float3 normal{plane.X,plane.Y,plane.Z};
            const float elevation=Dot(forward,normal);
            const auto skyDirection=Add(Subtract(forward,Multiply(normal,elevation)),Multiply(normal,std::max(0.08f,-elevation)));
            EnvironmentLookup lookup;
            if (LocateEnvironmentDirection(skyDirection,lookup))
            {
                const auto& sky=*m_EnvironmentFrame.Source;
                const int side=int(sky.Resolution),radius=std::max(1,side/16);
                const int cx=int(lookup.UV.X*side),cy=int(lookup.UV.Y*side);
                Float3 sum{}; unsigned count=0;
                for (int y=std::max(0,cy-radius);y<=std::min(side-1,cy+radius);++y)
                    for (int x=std::max(0,cx-radius);x<=std::min(side-1,cx+radius);++x)
                    { sum=Add(sum,sky.Texel(lookup.Face,x,y)); ++count; }
                const auto radiance=Multiply(sum,m_Techniques.Reflections.SkyboxIntensity/count);
                pass.ClearColor={radiance.X,radiance.Y,radiance.Z,1};
            }
        }
		if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
		if (m_PlanarFrame.Active()) for (const auto& pair : m_Meshes)
		{
			const auto& mesh = pair.second;
			if (!mesh.Mesh.Visible || mesh.Mesh.PlanarReceiver || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
			if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
				mesh.Mesh.WorldTransform, m_PlanarFrame.ClippedViewProjection)) continue;
			PipelineState pipeline;
			pipeline.Color = mesh.Mesh.Color;
			ConfigureMaterialPipeline(mesh, pipeline);
			pipeline.Lighting.CameraPosition = m_PlanarFrame.Position;
			// Main-view shadow coverage and screen effects do not describe the
			// reflected camera. Clip the receiver rather than recursively shading it.
			pipeline.SamplesShadowMap = false;
			if (!DrawMesh(mesh, pipeline)) return false;
		}
		if (!EndMeshPass()) return false;
		if (!m_RHI->CompletePlanarCapture()) return Fail(m_RHI->GetLastError().c_str());
		return true;
	}

	bool Renderer::RenderShadowPass()
	{
		m_CurrentPass = PassType::Shadow;
		const std::uint32_t count = std::max(1u, m_CascadeFrame.CascadeCount);
		for (std::uint32_t index = 0; index < count; ++index)
		{
			m_ShadowDrawProjection = m_CascadeFrame.CascadeCount ? m_CascadeFrame.Cascades[index].LightViewProjection : m_LightViewProjection;
			PassDesc pass;
			pass.Type = PassType::Shadow;
			pass.Target = m_CascadeFrame.CascadeCount ? m_CascadeMaps[index] : m_ShadowMap;
			pass.ClearColorEnabled = false;
			if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
			for (const auto& pair : m_Meshes)
			{
				const auto& mesh = pair.second;
				if (!mesh.Mesh.Visible || !mesh.Mesh.CastShadow || UsesPrivateTransparency(mesh)) continue;
				// Every cascade tests the light frustum, including off-camera casters.
				if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
					mesh.Mesh.WorldTransform, m_ShadowDrawProjection)) continue;
				PipelineState pipeline;
				pipeline.Fill = FillMode::Solid; pipeline.ColorWrite = false;
				pipeline.BlendEnabled = mesh.Translucent && m_Techniques.AlphaBlending;
				pipeline.Color = mesh.Mesh.Color;
				ConfigureMaterialPipeline(mesh, pipeline);
				if (!DrawMesh(mesh, pipeline)) return false;
			}
			if (!EndMeshPass()) return false;
		}
		return true;
	}

    bool Renderer::RenderCameraDepthPass(PassType type)
    {
        m_CurrentPass = type;
        PassDesc pass; pass.Type = m_CurrentPass; pass.ClearColorEnabled = false;
        if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
        for (const auto& entry : m_Meshes)
        {
            const auto& mesh = entry.second;
            if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
            if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
                mesh.Mesh.WorldTransform, m_CullingViewProjection)) continue;
            PipelineState pipeline; pipeline.Color = mesh.Mesh.Color;
            pipeline.Fill = m_RenderMode == RenderMode::Wireframe ? FillMode::Wireframe : FillMode::Solid;
            ConfigureMaterialPipeline(mesh, pipeline);
            pipeline.ColorWrite = false; pipeline.SamplesShadowMap = false;
            if (!DrawMesh(mesh, pipeline)) return false;
        }
        return EndMeshPass();
    }
    bool Renderer::RenderVisibilityGeometry()
    {
        m_CurrentPass = PassType::GBuffer;
        std::vector<const MeshResource*> vegetation;
        std::vector<VisibilitySourceKey> keys;
        std::vector<VisibilityMeshView> views;
        std::vector<VisibilityNativeMeshView> nativeViews;
        VisibilityBufferDesc frame; frame.ViewProjection = m_RasterViewProjection;
        // Wire inspection uses the same decoded cut through ordinary hardware
        // visibility. Solid opaque coverage consumes the native GPU streams.
        frame.NaniteRaster=m_Techniques.Nanite.Enabled && m_NaniteGeometry && !m_NaniteGeometry->Bindings.empty() && m_RenderMode==RenderMode::Solid;
        frame.MeshShaders=m_Techniques.MeshShaders;
        frame.MeshDispatch=m_Techniques.MeshDispatch;
        frame.GpuHardware=frame.NaniteRaster && m_Techniques.Nanite.GpuHardware && m_RHI->GetCapabilities().NaniteHardwareScheduling;
        frame.GpuInstances=frame.NaniteRaster && m_Techniques.Nanite.GpuInstances;
        frame.MaterialTiles=frame.NaniteRaster && m_Techniques.Nanite.MaterialTiles;
        frame.MaximumSoftwareExtent=m_Techniques.Nanite.MaximumSoftwareExtent; frame.SubpixelBits=m_Techniques.Nanite.SubpixelBits;
        const auto append = [&](const MeshResource& mesh, PipelineState pipeline, bool visible)
        {
            VisibilityMeshView view;
            view.Vertices = &mesh.Geometry->Vertices; view.Indices = &mesh.SelectedIndices(); view.World = mesh.Mesh.WorldTransform;
            keys.push_back({&mesh, mesh.Geometry, mesh.Mesh.WorldTransform, mesh.SelectedLod}); views.push_back(view);
            if (frame.NaniteRaster)
            {
                VisibilityNativeMeshView native; native.VertexBuffer=mesh.Geometry->VertexBuffer; native.IndexBuffer=mesh.Geometry->IndexBuffer;
                native.FirstIndex=mesh.Geometry->FirstIndex; native.IndexCount=mesh.SelectedIndexCount(); native.World=mesh.Mesh.WorldTransform;
                nativeViews.push_back(native);
            }
            if (!visible) return;
            VisibilityMaterialDraw draw; draw.Range = static_cast<std::uint32_t>(views.size() - 1);
            draw.Pipeline = pipeline; draw.World = mesh.Mesh.WorldTransform;
            draw.VertexBuffer = mesh.Geometry->VertexBuffer;
            draw.IndexBuffer = mesh.SelectedLod ? mesh.Geometry->LodIndexBuffers[mesh.SelectedLod - 1] : mesh.Geometry->IndexBuffer;
            draw.FirstIndex=mesh.SelectedFirstIndex(); frame.Draws.push_back(draw);
        };
        // Preserve ordinary opaque submission order, including coplanar depth
        // ties. Camera culling changes only the draw subset, never table identity.
        for (const auto& entry : m_Meshes)
        {
            const auto& mesh = entry.second;
            if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
            const bool visible = !m_Techniques.FrustumCulling || IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
                mesh.Mesh.WorldTransform, m_CullingViewProjection);
            if (!visible) ++m_FrustumCulledMeshCount;
            if (mesh.Mesh.Vegetation || (frame.NaniteRaster && mesh.Mesh.DynamicVertices)) { if (visible) vegetation.push_back(&mesh); continue; }
            PipelineState solid; solid.Color = mesh.Mesh.Color; solid.SamplesShadowMap = m_ShadowsEnabled;
            ConfigureMaterialPipeline(mesh, solid);
            PipelineState wire = solid; wire.Fill = FillMode::Wireframe; wire.Color = {0.12f, 0.82f, 1, 1};
            if (m_RenderMode != RenderMode::Wireframe) append(mesh, solid, visible);
            if (m_RenderMode == RenderMode::SolidWireframe)
            { wire.DepthWrite = false; wire.Fill = FillMode::Solid; wire.WireframeOverlay = true; }
            if (m_RenderMode != RenderMode::Solid) append(mesh, wire, visible);
        }
        // Streaming cuts can reuse a geometry object's address after releasing
        // its GPU buffers. The publication serial distinguishes those lifetimes.
        if (!m_VisibilityScene || m_VisibilitySceneRevision != m_ProbeSceneRevision || m_VisibilityGeometryViewSerial != m_GeometryViewSerial || keys != m_VisibilitySources || m_VisibilityScene->HasNativeGeometry()!=frame.NaniteRaster)
        {
            auto candidate = std::make_shared<VisibilityScene>();
            if (frame.NaniteRaster)
            { if (!BuildNaniteVisibilityScene(nativeViews,*candidate,m_LastError)) return false; }
            else if (!BuildVisibilityScene(views, *candidate, m_LastError)) return false;
            m_VisibilityScene = std::move(candidate); m_VisibilitySources = std::move(keys); m_VisibilitySceneRevision = m_ProbeSceneRevision;
            m_VisibilityGeometryViewSerial = m_GeometryViewSerial;
        }
        frame.Scene = m_VisibilityScene;
        if (!PrepareNaniteFrameVisibility(frame)) return false;
        if (!m_RHI->RenderVisibilityBuffer(frame)) return Fail(m_RHI->GetLastError().c_str());
        if (frame.NaniteRaster && m_Techniques.Nanite.Occlusion && !m_RHI->CaptureNaniteDepth(m_NaniteDepthCandidate)) return Fail(m_RHI->GetLastError().c_str());
        if (!vegetation.empty())
        {
            PassDesc pass; pass.Type=PassType::GBuffer; pass.ClearColorEnabled=false; pass.ClearDepthEnabled=false;
            if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
            for (const auto* mesh:vegetation)
            {
                PipelineState material; material.Color=mesh->Mesh.Color; ConfigureMaterialPipeline(*mesh,material);
                if (!DrawMesh(*mesh,material)) return false;
            }
            if (!EndMeshPass()) return false;
        }
        return true;
    }
	bool Renderer::RenderColorPass(bool transparentPhase, bool materialInputs)
	{
		m_CurrentPass = materialInputs || (!transparentPhase && Techniques::UsesDeferredLighting(m_Techniques.Pipeline))
			? PassType::GBuffer : PassType::Color;
		PassDesc pass;
		pass.Type = m_CurrentPass;
		// Transparent draws load the opaque attachments and preserve opaque depth.
		pass.ClearColorEnabled = !transparentPhase;
		pass.ClearDepthEnabled = !transparentPhase && !m_HairCameraDepthWritten && !((m_Techniques.ContactShadows.Enabled || m_Techniques.Pipeline == Techniques::RenderingPipeline::ForwardPlus) && m_CurrentPass == PassType::Color);
		if (!BeginMeshPass(pass))
		{
			return Fail(m_RHI->GetLastError().c_str());
		}

		std::vector<const MeshResource*> opaqueMeshes;
		std::vector<const MeshResource*> translucentMeshes;
		opaqueMeshes.reserve(m_Meshes.size());
		translucentMeshes.reserve(m_Meshes.size());
		for (const auto& pair : m_Meshes)
		{
			const MeshResource& mesh = pair.second;
			if (!mesh.Mesh.Visible)
			{
				continue;
			}
			if (UsesPrivateTransparency(mesh)) continue;
            const bool translucent = mesh.Translucent && m_Techniques.AlphaBlending;
			if (translucent != transparentPhase) continue;
			if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(
				mesh.BoundsMinimum, mesh.BoundsMaximum,
				mesh.Mesh.WorldTransform, m_CullingViewProjection))
			{
				if (!materialInputs) ++m_FrustumCulledMeshCount;
				continue;
			}
			(mesh.Translucent && m_Techniques.AlphaBlending ? translucentMeshes : opaqueMeshes)
				.push_back(&mesh);
		}

		const Float3 cameraPosition = m_RenderLighting.CameraPosition;
		const auto distanceToCamera = [&cameraPosition](
			const MeshResource* mesh)
		{
			const Float3 center = TransformPoint(
				mesh->Mesh.WorldTransform,
				mesh->BoundsCenter);
			const Float3 offset = Subtract(center, cameraPosition);
			const float distanceSquared = Dot(offset, offset);
			return std::isfinite(distanceSquared) ? distanceSquared : 0.0f;
		};
		if (!Techniques::UsesWeightedTransparency(m_Techniques)) std::sort(
			translucentMeshes.begin(),
			translucentMeshes.end(),
			[&distanceToCamera](
				const MeshResource* left,
				const MeshResource* right)
			{
				const float leftDistance = distanceToCamera(left);
				const float rightDistance = distanceToCamera(right);
				if (leftDistance != rightDistance)
				{
					return leftDistance > rightDistance;
				}
				return left->Mesh.Id < right->Mesh.Id;
			});

		const auto drawForRenderMode = [this](
			const MeshResource& mesh,
			bool translucent)
		{
			PipelineState solid;
			solid.Fill = FillMode::Solid;
			solid.DepthWrite = !translucent && !((m_Techniques.ContactShadows.Enabled || m_Techniques.Pipeline == Techniques::RenderingPipeline::ForwardPlus) && m_CurrentPass == PassType::Color);
			solid.BlendEnabled = translucent;
			solid.SamplesShadowMap = m_ShadowsEnabled;
			solid.Color = mesh.Mesh.Color;
			ConfigureMaterialPipeline(mesh, solid);
			PipelineState wireframe = solid;
			wireframe.Fill = FillMode::Wireframe;
			wireframe.Color = { 0.12f, 0.82f, 1.0f, 1.0f };

			if (m_RenderMode == RenderMode::Solid && !DrawMesh(mesh, solid))
			{
				return false;
			}
			if (m_RenderMode == RenderMode::Wireframe && !DrawMesh(mesh, wireframe))
			{
				return false;
			}
			if (m_RenderMode == RenderMode::SolidWireframe)
			{
				if (!DrawMesh(mesh, solid))
				{
					return false;
				}
				wireframe.DepthWrite = false;
				if (m_RHI->GetCapabilities().NativeAPI)
				{ wireframe.Fill = FillMode::Solid; wireframe.WireframeOverlay = true; }
				if (!DrawMesh(mesh, wireframe))
				{
					return false;
				}
			}
			return true;
		};

		for (const MeshResource* mesh : opaqueMeshes)
		{
			if (!drawForRenderMode(*mesh, false))
			{
				return false;
			}
		}
		for (const MeshResource* mesh : translucentMeshes)
		{
			if (!drawForRenderMode(*mesh, true))
			{
				return false;
			}
		}

		if (!EndMeshPass()) return false;
		return true;
	}

	bool Renderer::RenderVelocityPass()
	{
		m_CurrentPass = PassType::Velocity;
		PassDesc pass;
		pass.Type = PassType::Velocity;
		pass.ClearColor = {0, 0, 0, 0};
		pass.ClearDepthEnabled = false;
		if (!BeginMeshPass(pass)) return Fail(m_RHI->GetLastError().c_str());
		for (const auto& entry : m_Meshes)
		{
			const auto& mesh = entry.second;
			if (!mesh.Mesh.Visible || UsesPrivateTransparency(mesh) || (mesh.Translucent && m_Techniques.AlphaBlending)) continue;
			if (NeedsCpuFrustumTest(mesh) && !IntersectsFrustum(mesh.BoundsMinimum, mesh.BoundsMaximum,
				mesh.Mesh.WorldTransform, m_CullingViewProjection)) continue;
			PipelineState pipeline;
			pipeline.Color = mesh.Mesh.Color;
			pipeline.Fill = m_RenderMode == RenderMode::Wireframe ? FillMode::Wireframe : FillMode::Solid;
			pipeline.DepthWrite = false;
			ConfigureMaterialPipeline(mesh, pipeline);
			if (!DrawMesh(mesh, pipeline)) return false;
		}
		return EndMeshPass();
	}

	bool Renderer::RenderOutlinePass()
	{
		bool hasOutline = false;
		for (const auto& pair : m_Meshes)
		{
			hasOutline = hasOutline || (pair.second.Mesh.Visible && pair.second.Outline.Enabled);
		}
		if (!hasOutline)
		{
			return true;
		}

		PassDesc pass;
		m_CurrentPass = PassType::Outline;
		pass.Type = PassType::Outline;
		pass.ClearColorEnabled = false;
		pass.ClearDepthEnabled = false;
		if (!BeginMeshPass(pass))
		{
			return Fail(m_RHI->GetLastError().c_str());
		}

		for (const auto& pair : m_Meshes)
		{
			const MeshResource& mesh = pair.second;
			if (!mesh.Mesh.Visible || !mesh.Outline.Enabled)
			{
				continue;
			}
			PipelineState outline;
			outline.Fill = FillMode::Solid;
			outline.DepthWrite = false;
			outline.OutlineExtrusion = true;
			outline.OutlineThickness = mesh.Outline.Thickness;
			outline.Color = mesh.Outline.Color;
			ConfigureMaterialPipeline(mesh, outline);
            if(m_GpuPresentation)outline.Shading=Techniques::ShadingModel::Phong;
			if (!DrawMesh(mesh, outline))
			{
				return false;
			}
		}

		if (!EndMeshPass()) return false;
		return true;
	}

	void Renderer::ApplyOutlines(FrameCapture& capture) const
	{
        const auto& outlineProjection=Techniques::UsesTemporalSuperResolution(m_Techniques.PostProcessing)?m_ViewProjection:m_RasterViewProjection;
		if (!capture.HasColorOutput() ||
			capture.Width == 0 ||
			capture.Height == 0)
		{
			return;
		}

		struct MaskPixel
		{
			Float3 Color;
			std::uint8_t Radius = 0;
			std::uint8_t Opacity = 0;
			float Depth = std::numeric_limits<float>::infinity();
		};
		struct ProjectedVertex
		{
			float X = 0.0f;
			float Y = 0.0f;
			float Depth = 1.0f;
			float InverseW = 0.0f;
			Float2 TexCoord;
			float Alpha = 1.0f;
			bool Valid = false;
		};

		const std::size_t pixelCount =
			static_cast<std::size_t>(capture.Width) * capture.Height;
		std::vector<MaskPixel> mask(pixelCount);
		const bool hasDepth = capture.Depth32.size() == pixelCount;
		std::uint32_t outlinedMeshCount = 0;

		const auto edge = [](
			const ProjectedVertex& a,
			const ProjectedVertex& b,
			float x,
			float y)
		{
			return (b.X - a.X) * (y - a.Y) -
				(b.Y - a.Y) * (x - a.X);
		};
		const auto sampleImage = [](
			const std::shared_ptr<const ImageData>& image,
			float u,
			float v,
			std::size_t channel)
		{
			if (!image || !image->IsValid())
			{
				return 1.0f;
			}
			u -= std::floor(u);
			v -= std::floor(v);
			const std::uint32_t x = std::min(
				static_cast<std::uint32_t>(
					u * static_cast<float>(image->Width)),
				image->Width - 1U);
			const std::uint32_t y = std::min(
				static_cast<std::uint32_t>(
					v * static_cast<float>(image->Height)),
				image->Height - 1U);
			const std::size_t offset =
				(static_cast<std::size_t>(y) * image->Width + x) * 4U +
				channel;
			return static_cast<float>(image->PixelsRGBA8[offset]) / 255.0f;
		};

		for (const auto& pair : m_Meshes)
		{
			const MeshResource& resource = pair.second;
			const auto& mesh = resource.Mesh;
			const auto& sourceVertices = resource.Geometry->Vertices;
			const auto& indices = resource.SelectedIndices();
			if (!mesh.Visible || !resource.Outline.Enabled ||
				sourceVertices.empty() || indices.size() < 3)
			{
				continue;
			}
			const int radius = static_cast<int>(std::lround(std::clamp(
				resource.Outline.Thickness,
				0.0f,
				255.0f)));
			const float outlineOpacity = std::clamp(
				resource.Outline.Color.W,
				0.0f,
				1.0f);
			if (radius <= 0 || outlineOpacity <= 0.0f)
			{
				continue;
			}
			++outlinedMeshCount;

			const Matrix4 transform = Multiply(
				outlineProjection,
				mesh.WorldTransform);
			std::vector<ProjectedVertex> vertices(sourceVertices.size());
			for (std::size_t vertexIndex = 0;
				vertexIndex < sourceVertices.size();
				++vertexIndex)
			{
				const Vertex& source = sourceVertices[vertexIndex];
				const Float3& position = source.Position;
				const float clipX =
					transform.Values[0] * position.X +
					transform.Values[1] * position.Y +
					transform.Values[2] * position.Z +
					transform.Values[3];
				const float clipY =
					transform.Values[4] * position.X +
					transform.Values[5] * position.Y +
					transform.Values[6] * position.Z +
					transform.Values[7];
				const float clipZ =
					transform.Values[8] * position.X +
					transform.Values[9] * position.Y +
					transform.Values[10] * position.Z +
					transform.Values[11];
				const float clipW =
					transform.Values[12] * position.X +
					transform.Values[13] * position.Y +
					transform.Values[14] * position.Z +
					transform.Values[15];
				ProjectedVertex& projected = vertices[vertexIndex];
				if (!std::isfinite(clipW) || clipW <= 1.0e-7f)
				{
					continue;
				}
				projected.InverseW = 1.0f / clipW;
				const float ndcX = clipX * projected.InverseW;
				const float ndcY = clipY * projected.InverseW;
				projected.Depth = clipZ * projected.InverseW;
				projected.X =
					(ndcX * 0.5f + 0.5f) *
					static_cast<float>(capture.Width);
				projected.Y =
					(0.5f - ndcY * 0.5f) *
					static_cast<float>(capture.Height);
				projected.TexCoord = {
					source.TexCoord.X,
					1.0f - source.TexCoord.Y };
				projected.Alpha = source.Color.W;
				projected.Valid =
					std::isfinite(projected.X) &&
					std::isfinite(projected.Y) &&
					std::isfinite(projected.Depth);
			}

			for (std::size_t triangle = 0;
				triangle + 2 < indices.size();
				triangle += 3)
			{
				const std::uint32_t index0 = indices[triangle];
				const std::uint32_t index1 = indices[triangle + 1];
				const std::uint32_t index2 = indices[triangle + 2];
				if (index0 >= vertices.size() ||
					index1 >= vertices.size() ||
					index2 >= vertices.size())
				{
					continue;
				}
				const ProjectedVertex& a = vertices[index0];
				const ProjectedVertex& b = vertices[index1];
				const ProjectedVertex& c = vertices[index2];
				if (!a.Valid || !b.Valid || !c.Valid)
				{
					continue;
				}
				const float area = edge(a, b, c.X, c.Y);
				if (!std::isfinite(area) || std::fabs(area) <= 1.0e-8f)
				{
					continue;
				}
				const bool frontFacing = RealtimeWorldIsMirrored(mesh.WorldTransform)
					? area > 0.0f : area < 0.0f;
				if (m_Techniques.BackFaceCulling && !frontFacing) continue;
				const int minimumX = std::max(
					0,
					static_cast<int>(std::floor(
						std::min({a.X, b.X, c.X}))));
				const int maximumX = std::min(
					static_cast<int>(capture.Width) - 1,
					static_cast<int>(std::ceil(
						std::max({a.X, b.X, c.X}))));
				const int minimumY = std::max(
					0,
					static_cast<int>(std::floor(
						std::min({a.Y, b.Y, c.Y}))));
				const int maximumY = std::min(
					static_cast<int>(capture.Height) - 1,
					static_cast<int>(std::ceil(
						std::max({a.Y, b.Y, c.Y}))));
				for (int y = minimumY; y <= maximumY; ++y)
				{
					for (int x = minimumX; x <= maximumX; ++x)
					{
						const float sampleX = static_cast<float>(x) + 0.5f;
						const float sampleY = static_cast<float>(y) + 0.5f;
						const float weight0 =
							edge(b, c, sampleX, sampleY) / area;
						const float weight1 =
							edge(c, a, sampleX, sampleY) / area;
						const float weight2 =
							edge(a, b, sampleX, sampleY) / area;
						if (weight0 < -1.0e-6f ||
							weight1 < -1.0e-6f ||
							weight2 < -1.0e-6f)
						{
							continue;
						}
						const float depth =
							weight0 * a.Depth +
							weight1 * b.Depth +
							weight2 * c.Depth;
						if (depth < 0.0f || depth > 1.0f)
						{
							continue;
						}
						const std::size_t pixel =
							static_cast<std::size_t>(y) * capture.Width +
							static_cast<std::size_t>(x);
						if (hasDepth &&
							depth > capture.Depth32[pixel] + 2.0e-4f)
						{
							continue;
						}

						const float perspectiveDenominator =
							weight0 * a.InverseW +
							weight1 * b.InverseW +
							weight2 * c.InverseW;
						if (perspectiveDenominator <= 1.0e-12f)
						{
							continue;
						}
						const float perspective0 =
							weight0 * a.InverseW /
							perspectiveDenominator;
						const float perspective1 =
							weight1 * b.InverseW /
							perspectiveDenominator;
						const float perspective2 =
							weight2 * c.InverseW /
							perspectiveDenominator;
						const float u =
							perspective0 * a.TexCoord.X +
							perspective1 * b.TexCoord.X +
							perspective2 * c.TexCoord.X;
						const float v =
							perspective0 * a.TexCoord.Y +
							perspective1 * b.TexCoord.Y +
							perspective2 * c.TexCoord.Y;
						const float alpha =
							(perspective0 * a.Alpha +
								perspective1 * b.Alpha +
								perspective2 * c.Alpha) *
							mesh.Color.W *
							mesh.Material.BaseColor.W *
							(m_Techniques.TextureMapping ? sampleImage(
								mesh.Material.DiffuseImage,
								u,
								v,
								3U) : 1.0f) *
							sampleImage(
								mesh.Material.OpacityImage,
								u,
								v,
								0U);
						if (m_Techniques.AlphaTesting && mesh.Material.AlphaCutoff > 0.0f &&
							alpha < mesh.Material.AlphaCutoff)
						{
							continue;
						}
						if (depth >= mask[pixel].Depth)
						{
							continue;
						}
						mask[pixel].Color = {
							std::clamp(resource.Outline.Color.X, 0.0f, 1.0f),
							std::clamp(resource.Outline.Color.Y, 0.0f, 1.0f),
							std::clamp(resource.Outline.Color.Z, 0.0f, 1.0f) };
						mask[pixel].Radius =
							static_cast<std::uint8_t>(radius);
						mask[pixel].Opacity =
							static_cast<std::uint8_t>(std::lround(
								outlineOpacity * 255.0f));
						mask[pixel].Depth = depth;
					}
				}
			}
		}
		if (outlinedMeshCount == 0)
		{
			return;
		}

		std::vector<std::uint32_t> bestDistance(
			pixelCount,
			std::numeric_limits<std::uint32_t>::max());
		std::vector<std::uint32_t> bestSource(
			pixelCount,
			std::numeric_limits<std::uint32_t>::max());
		for (int sourceY = 0;
			sourceY < static_cast<int>(capture.Height);
			++sourceY)
		{
			for (int sourceX = 0;
				sourceX < static_cast<int>(capture.Width);
				++sourceX)
			{
				const std::size_t sourcePixel =
					static_cast<std::size_t>(sourceY) * capture.Width +
					static_cast<std::size_t>(sourceX);
				const int radius = mask[sourcePixel].Radius;
				if (radius <= 0)
				{
					continue;
				}
				bool boundary = false;
				for (int neighborY = -1;
					neighborY <= 1 && !boundary;
					++neighborY)
				{
					for (int neighborX = -1;
						neighborX <= 1;
						++neighborX)
					{
						if (neighborX == 0 && neighborY == 0)
						{
							continue;
						}
						const int x = sourceX + neighborX;
						const int y = sourceY + neighborY;
						if (x < 0 ||
							x >= static_cast<int>(capture.Width) ||
							y < 0 ||
							y >= static_cast<int>(capture.Height) ||
							mask[
								static_cast<std::size_t>(y) *
									capture.Width +
								static_cast<std::size_t>(x)]
								.Radius == 0)
						{
							boundary = true;
							break;
						}
					}
				}
				if (!boundary)
				{
					continue;
				}

				const int minimumX = std::max(0, sourceX - radius);
				const int maximumX = std::min(
					static_cast<int>(capture.Width) - 1,
					sourceX + radius);
				const int minimumY = std::max(0, sourceY - radius);
				const int maximumY = std::min(
					static_cast<int>(capture.Height) - 1,
					sourceY + radius);
				const std::uint32_t radiusSquared =
					static_cast<std::uint32_t>(radius * radius);
				for (int targetY = minimumY;
					targetY <= maximumY;
					++targetY)
				{
					for (int targetX = minimumX;
						targetX <= maximumX;
						++targetX)
					{
						const int deltaX = targetX - sourceX;
						const int deltaY = targetY - sourceY;
						const std::uint32_t distance =
							static_cast<std::uint32_t>(
								deltaX * deltaX + deltaY * deltaY);
						if (distance > radiusSquared)
						{
							continue;
						}
						const std::size_t targetPixel =
							static_cast<std::size_t>(targetY) *
								capture.Width +
							static_cast<std::size_t>(targetX);
						if (mask[targetPixel].Radius != 0 ||
							distance >= bestDistance[targetPixel])
						{
							continue;
						}
						bestDistance[targetPixel] = distance;
						bestSource[targetPixel] =
							static_cast<std::uint32_t>(sourcePixel);
					}
				}
			}
		}

		const auto srgbToLinear = [](float value)
		{
			value = std::clamp(value, 0.0f, 1.0f);
			return value <= 0.04045f
				? value / 12.92f
				: std::pow(
					(value + 0.055f) / 1.055f,
					2.4f);
		};
		const auto linearToSrgb = [](float value)
		{
			value = std::max(value, 0.0f);
			return value <= 0.0031308f
				? value * 12.92f
				: 1.055f * std::pow(
					value,
					1.0f / 2.4f) -
					0.055f;
		};
		for (std::size_t targetPixel = 0;
			targetPixel < pixelCount;
			++targetPixel)
		{
			if (bestSource[targetPixel] ==
				std::numeric_limits<std::uint32_t>::max())
			{
				continue;
			}
			const MaskPixel& source = mask[bestSource[targetPixel]];
			const float opacity =
				static_cast<float>(source.Opacity) / 255.0f;
			const std::array<float, 3> sourceColor = {
				source.Color.X,
				source.Color.Y,
				source.Color.Z };
			for (std::size_t channel = 0; channel < 3; ++channel)
			{
				const float destination =
					static_cast<float>(
						capture.ColorRGBA8[targetPixel * 4 + channel]) /
					255.0f;
				const float blended =
					srgbToLinear(sourceColor[channel]) * opacity +
					srgbToLinear(destination) * (1.0f - opacity);
				capture.ColorRGBA8[targetPixel * 4 + channel] =
					static_cast<std::uint8_t>(std::lround(
						std::clamp(
							linearToSrgb(blended),
							0.0f,
							1.0f) *
						255.0f));
			}
			const float destinationAlpha =
				static_cast<float>(
					capture.ColorRGBA8[targetPixel * 4 + 3]) /
				255.0f;
			capture.ColorRGBA8[targetPixel * 4 + 3] =
				static_cast<std::uint8_t>(std::lround(
					std::clamp(
						opacity +
							destinationAlpha * (1.0f - opacity),
						0.0f,
						1.0f) *
					255.0f));
		}
		capture.Statistics.OutlineDrawCallCount = outlinedMeshCount;
	}

	bool Renderer::DrawMesh(const MeshResource& mesh, const PipelineState& pipeline)
	{
        if (m_CollectInstanceDraws && mesh.Mesh.ReorderForInstancing && !pipeline.BlendEnabled &&
            !pipeline.OutlineExtrusion && pipeline.DepthWrite && m_RenderMode==RenderMode::Solid &&
            m_CurrentPass!=PassType::Outline && !UsesParallaxMaterial(pipeline,m_CurrentPass))
        { m_InstanceDraws.push_back({&mesh,pipeline}); return true; }

		const Matrix4& viewProjection = m_CurrentPass == PassType::Shadow
			? m_ShadowDrawProjection : m_CurrentPass == PassType::PlanarCapture ? m_PlanarFrame.ClippedViewProjection :
				(m_CurrentPass == PassType::EnvironmentCapture ? m_EnvironmentDrawProjection : m_RasterViewProjection);
		MotionTransform motion;
		const bool temporalAA = Techniques::UsesTemporalReconstruction(m_Techniques.PostProcessing);
		// A lighting/settings reset discards color history, not rigid correspondence.
		// Retaining the actual previous raster jitter cancels it from physical
		// displacement, including changes to the TSR input resolution.
		motion.Valid = ((temporalAA || Techniques::UsesTemporalGI(m_Techniques.GlobalIllumination) || Techniques::UsesTemporalReflections(m_Techniques.Reflections)) ? m_CurrentMotionFrame.CorrespondenceValid : m_CurrentMotionFrame.Valid) &&
			mesh.PreviousTransformValid && mesh.SelectedLod == mesh.PreviousLod &&
			(m_CurrentPass == PassType::Color || m_CurrentPass == PassType::GBuffer || m_CurrentPass == PassType::Velocity ||
                m_CurrentPass == PassType::ForwardPlusDepth);
		motion.PreviousWorldViewProjection = motion.Valid
			? Multiply(TemporalSampling::Jitter(m_CurrentMotionFrame.PreviousViewProjection, m_TemporalFrame.PreviousJitterPixels,
				m_RenderWidth, m_RenderHeight), mesh.PreviousWorldTransform)
			: Multiply(viewProjection, mesh.Mesh.WorldTransform);
		motion.JitterDeltaPixels = {m_TemporalFrame.JitterPixels.X - m_TemporalFrame.PreviousJitterPixels.X,
			m_TemporalFrame.JitterPixels.Y - m_TemporalFrame.PreviousJitterPixels.Y};
		motion.DeltaSeconds = motion.Valid ? m_CurrentMotionFrame.DeltaSeconds : 0.0f;
		if (m_CurrentPass == PassType::Velocity && motion.Valid)
			m_CurrentMotionFrame.MotionDetected |= Multiply(m_CurrentMotionFrame.PreviousViewProjection, mesh.PreviousWorldTransform).Values !=
				Multiply(m_ViewProjection, mesh.Mesh.WorldTransform).Values;
		const bool batchable = (m_Techniques.Instancing || mesh.Mesh.Vegetation) &&
			!pipeline.BlendEnabled && !pipeline.OutlineExtrusion && m_CurrentPass != PassType::Outline && !UsesParallaxMaterial(pipeline,m_CurrentPass);
		if (!batchable)
			return FlushDrawBatch() && SubmitMesh(mesh, pipeline, viewProjection, motion, nullptr);
		if (m_BatchMesh && (m_BatchInstances.size() == MaximumDrawInstances ||
			m_BatchMesh->Geometry != mesh.Geometry || m_BatchMesh->SelectedLod != mesh.SelectedLod ||
			RealtimeWorldIsMirrored(m_BatchMesh->Mesh.WorldTransform) != RealtimeWorldIsMirrored(mesh.Mesh.WorldTransform) ||
			!SameInstancedPipeline(m_BatchPipeline, pipeline)))
			if (!FlushDrawBatch()) return false;
		if (!m_BatchMesh)
		{
			m_BatchMesh = &mesh; m_BatchPipeline = pipeline; m_BatchViewProjection = viewProjection;
		}
		DrawInstance instance{mesh.Mesh.WorldTransform, motion};
        if (mesh.Mesh.Vegetation)
        {
            instance.Wind=m_VegetationWind; instance.Wind.Z=float(m_VegetationTime);
            instance.Gust=m_VegetationGust; instance.Gust.W=float(mesh.Mesh.Vegetation);
            instance.VegetationHistory={float(m_PreviousVegetationTime),mesh.Mesh.Vegetation==1?1.0f:14.0f,0,0};
            if (motion.Valid && m_CurrentPass==PassType::Velocity && m_VegetationWind.W>0)
                m_CurrentMotionFrame.MotionDetected |= m_PreviousVegetationTime!=m_VegetationTime;
        }
        m_BatchInstances.push_back(instance);
		return true;
	}

	bool Renderer::FlushDrawBatch()
	{
		if (m_BatchInstances.empty()) { m_BatchMesh = nullptr; return true; }
		const auto* mesh = m_BatchMesh;
		m_BatchMesh = nullptr;
		const bool submitted = SubmitMesh(*mesh, m_BatchPipeline, m_BatchViewProjection, m_BatchInstances.front().Motion,
			(m_BatchInstances.size() > 1 || mesh->Mesh.Vegetation) ? &m_BatchInstances : nullptr);
		m_BatchInstances.clear();
		return submitted;
	}

    bool Renderer::BeginMeshPass(const PassDesc& pass)
    {
        m_PreparedMeshDraws.clear(); m_PreparedMeshPass = pass;
        m_InstanceDraws.clear(); m_CollectInstanceDraws=m_Techniques.Instancing;
        m_CapturingGpuPass = m_Techniques.GPUDriven && pass.Type != PassType::Outline && !pass.DepthPeelingLayer;
        // Material order is captured first. Vulkan and Metal require compute
        // outside the native render pass; all backends use the same ordering.
        return m_CapturingGpuPass || m_RHI->BeginPass(pass);
    }

    bool Renderer::NeedsCpuFrustumTest(const MeshResource& mesh) const
    {
        const bool parallax=m_CurrentPass==PassType::GBuffer && Techniques::UsesParallaxOcclusion(m_Techniques.Parallax) &&
            !mesh.Translucent && mesh.Mesh.Material.AlphaCutoff<=0 && !mesh.Mesh.Material.OpacityImage;
        return m_Techniques.FrustumCulling && (!m_CapturingGpuPass || parallax || (mesh.Translucent && m_Techniques.AlphaBlending));
    }

	bool Renderer::EndMeshPass()
	{
        m_CollectInstanceDraws=false;
        // Procedural foliage opts into reordering so its shared bark and leaf
        // sections form full native batches instead of alternating per tree.
        std::stable_sort(m_InstanceDraws.begin(),m_InstanceDraws.end(),[](const auto& a,const auto& b)
        { return std::less<const MeshGeometryResource*>{}(a.Mesh->Geometry,b.Mesh->Geometry); });
        for (const auto& draw:m_InstanceDraws) if (!DrawMesh(*draw.Mesh,draw.Pipeline)) return false;
        m_InstanceDraws.clear();
        if (!FlushDrawBatch()) return false;
        if (m_CapturingGpuPass)
        {
            m_CapturingGpuPass = false;
            for (auto& draw : m_PreparedMeshDraws) if (draw.GPU)
            {
                GpuDrivenBatchDesc desc;
                desc.BoundsMinimum = draw.Mesh->BoundsMinimum; desc.BoundsMaximum = draw.Mesh->BoundsMaximum;
                desc.RenderViewProjection = draw.ViewProjection;
                const bool auxiliaryView = m_CurrentPass == PassType::Shadow || m_CurrentPass == PassType::EnvironmentCapture ||
                    m_CurrentPass == PassType::PlanarCapture;
                desc.CullingViewProjection = auxiliaryView ? draw.ViewProjection : m_CullingViewProjection;
                desc.Material = {draw.Pipeline.Color.W * draw.Pipeline.MaterialOpacity, 0, 0, 0};
                desc.IndexCount = draw.Mesh->SelectedIndexCount(); desc.FirstIndex=draw.Pipeline.WireframeOverlay?0:draw.Mesh->SelectedFirstIndex();
                desc.FrustumCulling = m_Techniques.FrustumCulling;
                desc.Wireframe = draw.Pipeline.Fill == FillMode::Wireframe;
                desc.Instances = draw.Instances;
                if (!m_RHI->PrepareGpuDrivenBatch(desc, draw.Handle)) return Fail(m_RHI->GetLastError().c_str());
            }
            if (!m_RHI->BeginPass(m_PreparedMeshPass)) return Fail(m_RHI->GetLastError().c_str());
            for (const auto& draw : m_PreparedMeshDraws)
                if (!SubmitMesh(*draw.Mesh, draw.Pipeline, draw.ViewProjection, draw.Motion,
                    draw.Instances.empty() ? nullptr : &draw.Instances, draw.GPU ? &draw.Handle : nullptr)) return false;
            m_PreparedMeshDraws.clear();
        }
        if (m_BoundHairGeometry && HairFurReplayPass(m_CurrentPass) &&
            (m_CurrentPass!=PassType::PlanarCapture || m_PlanarFrame.Active()))
        {
            const auto projection=m_CurrentPass==PassType::Shadow?m_ShadowDrawProjection:m_CurrentPass==PassType::EnvironmentCapture?m_EnvironmentDrawProjection:
                m_CurrentPass==PassType::PlanarCapture?m_PlanarFrame.ClippedViewProjection:m_RasterViewProjection;
            auto view=HairView(projection);
            if (m_CurrentPass==PassType::EnvironmentCapture) { view.Lighting.CameraPosition=m_EnvironmentFrame.Position; view.Forward=m_EnvironmentFrame.Faces[m_PreparedMeshPass.EnvironmentFaceIndex].Forward; view.Orthographic=false; }
            if (m_CurrentPass==PassType::PlanarCapture) { view.Lighting.CameraPosition=m_PlanarFrame.Position; view.ClipPlane=m_PlanarFrame.ClipPlane; }
            if (m_CurrentPass==PassType::Velocity)
            {
                HairFurVelocityDesc velocity; velocity.View=view; velocity.HistoryValid=m_CurrentMotionFrame.CorrespondenceValid;
                velocity.JitterDeltaPixels={m_TemporalFrame.JitterPixels.X-m_TemporalFrame.PreviousJitterPixels.X,m_TemporalFrame.JitterPixels.Y-m_TemporalFrame.PreviousJitterPixels.Y};
                if (!m_RHI->DrawHairFurVelocity(velocity)) return Fail(m_RHI->GetLastError().c_str());
            }
            else if (!m_RHI->DrawHairFurInPass(view)) return Fail(m_RHI->GetLastError().c_str());
            if (HairFurCameraDepthPass(m_CurrentPass)) m_HairCameraDepthWritten=true;
        }
		return m_RHI->EndPass() || Fail(m_RHI->GetLastError().c_str());
	}

	bool Renderer::SubmitMesh(const MeshResource& mesh, const PipelineState& pipeline, const Matrix4& viewProjection,
		const MotionTransform& motion, const std::vector<DrawInstance>* instances, const GpuDrivenBatchHandle* gpu)
	{
        if (m_CapturingGpuPass)
        {
            PreparedMeshDraw draw;
            draw.Mesh = &mesh; draw.Pipeline = pipeline; draw.ViewProjection = viewProjection; draw.Motion = motion;
            draw.GPU = !pipeline.BlendEnabled && !pipeline.OutlineExtrusion && !UsesParallaxMaterial(pipeline,m_CurrentPass);
            if (instances) draw.Instances = *instances;
            else if (draw.GPU) draw.Instances.push_back({mesh.Mesh.WorldTransform, motion});
            m_PreparedMeshDraws.push_back(std::move(draw));
            return true;
        }
		m_RHI->SetMotionTransform(motion);
		const auto vertexBuffer = pipeline.WireframeOverlay ? mesh.Geometry->WireOverlays[mesh.SelectedLod].Vertices : mesh.Geometry->VertexBuffer;
		const auto indexBuffer = pipeline.WireframeOverlay ? mesh.Geometry->WireOverlays[mesh.SelectedLod].Indices :
			mesh.SelectedLod ? mesh.Geometry->LodIndexBuffers[mesh.SelectedLod - 1] : mesh.Geometry->IndexBuffer;
		const auto firstIndex = pipeline.WireframeOverlay ? 0 : mesh.SelectedFirstIndex();
		if (!m_RHI->BindPipeline(pipeline)
			|| !m_RHI->BindWorldTransform(mesh.Mesh.WorldTransform)
			|| !m_RHI->BindViewProjection(viewProjection)
			|| !m_RHI->BindLightViewProjection(m_LightViewProjection)
			|| !m_RHI->BindVertexBuffer(vertexBuffer)
			|| !m_RHI->BindIndexBuffer(indexBuffer)
			|| !(gpu ? m_RHI->DrawGpuDrivenBatch(*gpu) : instances ? m_RHI->DrawIndexedInstanced(firstIndex, mesh.SelectedIndexCount(), 0, *instances) :
				m_RHI->DrawIndexed(firstIndex, mesh.SelectedIndexCount(), 0)))
		{
			return Fail(m_RHI->GetLastError().c_str());
		}
		return true;
	}

	void Renderer::UpdateAutomaticViewProjections()
	{
		if (!m_AutomaticViewProjection && !m_AutomaticLightViewProjection) return;
		Float3 minimum {
			std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity() };
		Float3 maximum {
			-std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity() };
		bool hasGeometry = false;
		for (const auto& pair : m_Meshes)
		{
			const MeshResource& resource = pair.second;
			if (!resource.Mesh.Visible) continue;
			for (const Vertex& vertex : resource.SourceGeometry->Vertices)
			{
				const Float3 world = TransformPoint(resource.Mesh.WorldTransform, vertex.Position);
				minimum.X = std::min(minimum.X, world.X);
				minimum.Y = std::min(minimum.Y, world.Y);
				minimum.Z = std::min(minimum.Z, world.Z);
				maximum.X = std::max(maximum.X, world.X);
				maximum.Y = std::max(maximum.Y, world.Y);
				maximum.Z = std::max(maximum.Z, world.Z);
				hasGeometry = true;
			}
		}
        if (m_BoundHairGeometry)
        {
            const auto& low=m_BoundHairGeometry->BoundsMinimum; const auto& high=m_BoundHairGeometry->BoundsMaximum;
            minimum={std::min(minimum.X,low.X),std::min(minimum.Y,low.Y),std::min(minimum.Z,low.Z)};
            maximum={std::max(maximum.X,high.X),std::max(maximum.Y,high.Y),std::max(maximum.Z,high.Z)}; hasGeometry=true;
        }
		if (!hasGeometry)
		{
			if (m_AutomaticViewProjection) { m_ViewProjection = Matrix4::Identity(); m_CascadeCameraValid = false; }
			if (m_AutomaticLightViewProjection) m_LightViewProjection = Matrix4::Identity();
			return;
		}

		const Float3 center = Multiply(Add(minimum, maximum), 0.5f);
		const Float3 halfSize = Multiply(Subtract(maximum, minimum), 0.5f);
		const float radius = std::max(0.5f, std::sqrt(Dot(halfSize, halfSize)));
		if (m_AutomaticViewProjection)
		{
			const Float3 cameraDirection = Normalize({ 0.72f, 0.48f, 1.0f });
			const float cameraDistance = radius * 3.2f;
			const Float3 eye = Add(center, Multiply(cameraDirection, cameraDistance));
			const Matrix4 view = LookAt(eye, center, { 0.0f, 1.0f, 0.0f });
			m_Lighting.CameraPosition = eye;
			const float aspect = static_cast<float>(std::max(1, m_CreateInfo.Width))
				/ static_cast<float>(std::max(1, m_CreateInfo.Height));
			const Matrix4 projection = Perspective(1.0471975512f, aspect,
				std::max(0.001f, radius * 0.02f), cameraDistance + radius * 2.5f);
			m_CascadeCamera = {view, projection};
			m_CascadeCameraValid = true;
			m_ViewProjection = Multiply(projection, view);
		}
		if (m_AutomaticLightViewProjection)
		{
			Float3 lightEye = Add(
				center,
				Multiply(Normalize({ 0.45f, 1.0f, 0.35f }), radius * 3.0f));
			Float3 lightTarget = center;
			bool finiteShadowLight = false;
			if (m_Lighting.ShadowLightIndex >= 0 &&
				static_cast<std::uint32_t>(m_Lighting.ShadowLightIndex) <
					m_Lighting.LightCount)
			{
				const SceneLight& shadowLight = m_Lighting.Lights[
					static_cast<std::size_t>(m_Lighting.ShadowLightIndex)];
				finiteShadowLight =
					shadowLight.Type != LightType::Directional;
				if (shadowLight.Type == LightType::Directional)
				{
					Float3 directionToLight = shadowLight.Direction;
					if (Dot(directionToLight, directionToLight) <= 1.0e-12f)
					{
						directionToLight = { 0.45f, 1.0f, 0.35f };
					}
					lightEye = Add(
						center,
						Multiply(Normalize(directionToLight), radius * 3.0f));
				}
				else
				{
					lightEye = shadowLight.Position;
					if (shadowLight.Type == LightType::RectArea)
					{
						Float3 emissionDirection = shadowLight.Direction;
						if (Dot(emissionDirection, emissionDirection) <= 1.0e-12f)
						{
							emissionDirection = Subtract(center, lightEye);
						}
						if (Dot(emissionDirection, emissionDirection) <= 1.0e-12f)
						{
							emissionDirection = { 0.0f, -1.0f, 0.0f };
						}
						lightTarget = Add(
							lightEye,
							Normalize(emissionDirection));
					}
					else if (Dot(
						Subtract(lightTarget, lightEye),
						Subtract(lightTarget, lightEye)) <= 1.0e-12f)
					{
						lightEye = Add(
							center,
							Multiply(
								Normalize({ 0.45f, 1.0f, 0.35f }),
								radius * 3.0f));
					}
				}
			}
			const Float3 lightForward = Normalize(
				Subtract(lightTarget, lightEye));
			const Float3 lightUp = std::fabs(Dot(
				lightForward,
				{ 0.0f, 1.0f, 0.0f })) > 0.95f
				? Float3 { 1.0f, 0.0f, 0.0f }
				: Float3 { 0.0f, 1.0f, 0.0f };
			const Matrix4 lightView = LookAt(
				lightEye,
				lightTarget,
				lightUp);
			const auto worldCorner = [&minimum, &maximum](int corner)
			{
				return Float3 {
					(corner & 1) != 0 ? maximum.X : minimum.X,
					(corner & 2) != 0 ? maximum.Y : minimum.Y,
					(corner & 4) != 0 ? maximum.Z : minimum.Z };
			};
			if (finiteShadowLight)
			{
				constexpr float minimumNearPlane = 0.01f;
				float minimumSlopeX = std::numeric_limits<float>::infinity();
				float maximumSlopeX = -std::numeric_limits<float>::infinity();
				float minimumSlopeY = std::numeric_limits<float>::infinity();
				float maximumSlopeY = -std::numeric_limits<float>::infinity();
				float minimumDepth = std::numeric_limits<float>::infinity();
				float maximumDepth = 0.0f;
				std::size_t frontCornerCount = 0;
				bool hasCornerBehindEmitter = false;
				for (int corner = 0; corner < 8; ++corner)
				{
					const Float3 light = TransformPoint(
						lightView,
						worldCorner(corner));
					const float depth = -light.Z;
					if (depth <= minimumNearPlane)
					{
						hasCornerBehindEmitter = true;
						continue;
					}
					minimumSlopeX = std::min(
						minimumSlopeX,
						light.X / depth);
					maximumSlopeX = std::max(
						maximumSlopeX,
						light.X / depth);
					minimumSlopeY = std::min(
						minimumSlopeY,
						light.Y / depth);
					maximumSlopeY = std::max(
						maximumSlopeY,
						light.Y / depth);
					minimumDepth = std::min(minimumDepth, depth);
					maximumDepth = std::max(maximumDepth, depth);
					++frontCornerCount;
				}
				if (frontCornerCount == 0)
				{
					m_LightViewProjection = Matrix4::Identity();
					return;
				}
				const float slopeWidth = maximumSlopeX - minimumSlopeX;
				const float slopeHeight = maximumSlopeY - minimumSlopeY;
				const float paddingSlopeX = std::max(
					slopeWidth * 0.02f,
					1.0e-4f);
				const float paddingSlopeY = std::max(
					slopeHeight * 0.02f,
					1.0e-4f);
				const float nearPlane = hasCornerBehindEmitter
					? minimumNearPlane
					: std::max(minimumNearPlane, minimumDepth * 0.95f);
				const float farPlane = std::max(
					nearPlane + 0.1f,
					maximumDepth * 1.05f);
				const Matrix4 lightProjection = PerspectiveOffCenter(
					(minimumSlopeX - paddingSlopeX) * nearPlane,
					(maximumSlopeX + paddingSlopeX) * nearPlane,
					(minimumSlopeY - paddingSlopeY) * nearPlane,
					(maximumSlopeY + paddingSlopeY) * nearPlane,
					nearPlane,
					farPlane);
				m_LightViewProjection = Multiply(
					lightProjection,
					lightView);
			}
			else
			{
				Float3 lightMinimum {
					std::numeric_limits<float>::infinity(),
					std::numeric_limits<float>::infinity(),
					std::numeric_limits<float>::infinity() };
				Float3 lightMaximum {
					-std::numeric_limits<float>::infinity(),
					-std::numeric_limits<float>::infinity(),
					-std::numeric_limits<float>::infinity() };
				for (int corner = 0; corner < 8; ++corner)
				{
					const Float3 light = TransformPoint(
						lightView,
						worldCorner(corner));
					lightMinimum.X = std::min(lightMinimum.X, light.X);
					lightMinimum.Y = std::min(lightMinimum.Y, light.Y);
					lightMinimum.Z = std::min(lightMinimum.Z, light.Z);
					lightMaximum.X = std::max(lightMaximum.X, light.X);
					lightMaximum.Y = std::max(lightMaximum.Y, light.Y);
					lightMaximum.Z = std::max(lightMaximum.Z, light.Z);
				}
				const float padding = radius * 0.05f + 0.001f;
				lightMinimum = Subtract(
					lightMinimum,
					{ padding, padding, padding });
				lightMaximum = Add(
					lightMaximum,
					{ padding, padding, padding });
				m_LightViewProjection = Multiply(
					OrthographicBounds(lightMinimum, lightMaximum),
					lightView);
			}
		}
	}

	bool Renderer::Fail(const char* error)
	{
		m_LastError = error != nullptr ? error : "Unknown renderer error";
		return false;
	}
}
