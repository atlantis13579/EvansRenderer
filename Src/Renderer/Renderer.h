#pragma once

#include "IRHI.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace Evans::Renderer
{
	class Renderer
	{
	public:
		Renderer();
		explicit Renderer(std::unique_ptr<IRHI> rhi);
		~Renderer();

		Renderer(const Renderer&) = delete;
		Renderer& operator=(const Renderer&) = delete;

		bool Initialize(const RendererCreateInfo& createInfo);
		bool Initialize(const RendererCreateInfo& createInfo, std::unique_ptr<IRHI> rhi);
		void Shutdown();
		bool IsInitialized() const;
		RHIBackend GetBackend() const;
		const char* GetBackendName() const;
		RHICapabilities GetCapabilities() const;

		bool Resize(int width, int height);
		bool Resize(int width, int height, int shadowMapSize);
		bool AddMesh(const MeshData& mesh);
		bool UpdateMesh(const MeshData& mesh);
		bool SetWorldTransform(const std::string& id, const Matrix4& transform);
		bool SetOutline(const std::string& id, const OutlineSettings& outline);
		bool RemoveMesh(const std::string& id);
		void ClearScene();

		void SetRenderMode(RenderMode mode);
		RenderMode GetRenderMode() const;
		void SetShadowsEnabled(bool enabled);
		bool AreShadowsEnabled() const;
		void SetLighting(const LightingState& lighting);
		const LightingState& GetLighting() const;
		void SetViewProjection(const Matrix4& viewProjection);
		void SetLightViewProjection(const Matrix4& lightViewProjection);
		void SetAutomaticViewProjection(bool enabled);
		bool IsAutomaticViewProjection() const;
		bool RenderFrame();

		FrameCapture GetLastFrameCapture() const;
		const FrameCapture& GetLastFrameCaptureView() const noexcept;
		std::string GetLastError() const;
		std::size_t GetMeshCount() const;
		IRHI* GetRHI();
		const IRHI* GetRHI() const;

		static MeshData CreateTriangle(const std::string& id = "Triangle");

	private:
		struct MeshResource
		{
			MeshData Mesh;
			OutlineSettings Outline;
			BufferHandle VertexBuffer;
			BufferHandle IndexBuffer;
			TextureHandle DiffuseTexture;
			TextureHandle NormalTexture;
			TextureHandle OpacityTexture;
			Float3 BoundsCenter;
			bool Translucent = false;
		};

		struct TextureCacheKey
		{
			const ImageData* Image = nullptr;
			TextureColorSpace ColorSpace = TextureColorSpace::Linear;

			friend bool operator==(
				const TextureCacheKey& left,
				const TextureCacheKey& right)
			{
				return left.Image == right.Image &&
					left.ColorSpace == right.ColorSpace;
			}
		};

		struct TextureCacheKeyHash
		{
			std::size_t operator()(const TextureCacheKey& key) const noexcept
			{
				return std::hash<const ImageData*> {}(key.Image) ^
					(static_cast<std::size_t>(key.ColorSpace) << 1);
			}
		};

		struct TextureCacheEntry
		{
			std::shared_ptr<const ImageData> Image;
			TextureHandle Texture;
			std::size_t ReferenceCount = 0;
		};

		bool CreateMeshResources(const MeshData& mesh, MeshResource& resource);
		void DestroyMeshResources(MeshResource& resource);
		bool AcquireTexture(
			const std::shared_ptr<const ImageData>& image,
			TextureColorSpace colorSpace,
			const std::string& debugName,
			TextureHandle& texture);
		void ReleaseTexture(
			const std::shared_ptr<const ImageData>& image,
			TextureColorSpace colorSpace,
			TextureHandle& texture);
		void ConfigureMaterialPipeline(
			const MeshResource& mesh,
			PipelineState& pipeline) const;
		bool RenderShadowPass();
		bool RenderColorPass();
		bool RenderOutlinePass();
		bool DrawMesh(const MeshResource& mesh, const PipelineState& pipeline);
		void ApplyOutlines(FrameCapture& capture) const;
		void UpdateAutomaticViewProjections();
		bool Fail(const char* error);

		std::unique_ptr<IRHI> m_RHI;
		std::unordered_map<std::string, MeshResource> m_Meshes;
		std::unordered_map<
			TextureCacheKey,
			TextureCacheEntry,
			TextureCacheKeyHash> m_TextureCache;
		RendererCreateInfo m_CreateInfo;
		TextureHandle m_ShadowMap;
		Matrix4 m_ViewProjection;
		Matrix4 m_LightViewProjection;
		LightingState m_Lighting;
		FrameCapture m_LastFrameCapture;
		PassType m_CurrentPass = PassType::Color;
		RenderMode m_RenderMode = RenderMode::Solid;
		bool m_ShadowsEnabled = true;
		bool m_AutomaticViewProjection = true;
		bool m_AutomaticLightViewProjection = true;
		bool m_Initialized = false;
		std::string m_LastError;
	};
}
