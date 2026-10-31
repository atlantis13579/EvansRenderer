#include "Renderer/Renderer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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

		bool IsMeshTranslucent(const MeshData& mesh)
		{
			// Explicit alpha thresholds remain binary cutouts. Without a
			// threshold, both diffuse alpha and opacity-map red are blended.
			if (mesh.Material.AlphaCutoff > 0.0f)
			{
				return false;
			}

			float minimumAlpha = mesh.Color.W * mesh.Material.BaseColor.W;
			float minimumVertexAlpha = 1.0f;
			for (const Vertex& vertex : mesh.Vertices)
			{
				minimumVertexAlpha = std::min(
					minimumVertexAlpha,
					vertex.Color.W);
			}
			minimumAlpha *= minimumVertexAlpha;
			if (mesh.Material.DiffuseImage &&
				mesh.Material.DiffuseImage->IsValid())
			{
				const std::vector<std::uint8_t>& pixels =
					mesh.Material.DiffuseImage->PixelsRGBA8;
				std::uint8_t minimumTextureAlpha = 255;
				for (std::size_t offset = 3; offset < pixels.size(); offset += 4)
				{
					minimumTextureAlpha = std::min(
						minimumTextureAlpha,
						pixels[offset]);
				}
				minimumAlpha *=
					static_cast<float>(minimumTextureAlpha) / 255.0f;
			}
			if (mesh.Material.OpacityImage &&
				mesh.Material.OpacityImage->IsValid())
			{
				const std::vector<std::uint8_t>& pixels =
					mesh.Material.OpacityImage->PixelsRGBA8;
				std::uint8_t minimumOpacity = 255;
				for (std::size_t offset = 0; offset < pixels.size(); offset += 4)
				{
					minimumOpacity = std::min(minimumOpacity, pixels[offset]);
				}
				minimumAlpha *=
					static_cast<float>(minimumOpacity) / 255.0f;
			}
			return !std::isfinite(minimumAlpha) ||
				minimumAlpha < 1.0f - 1.0e-6f;
		}
	}

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
		rhiCreateInfo.EnableValidation = createInfo.EnableValidation;
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
		m_AutomaticViewProjection = createInfo.AutomaticViewProjection;
		m_AutomaticLightViewProjection = createInfo.AutomaticViewProjection;
		m_Initialized = true;
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
		if (m_RHI && m_RHI->IsInitialized())
		{
			ClearScene();
			if (m_ShadowMap)
			{
				m_RHI->DestroyTexture(m_ShadowMap);
			}
			m_RHI->Shutdown();
		}
		else
		{
			m_Meshes.clear();
			m_TextureCache.clear();
		}
		m_ShadowMap = {};
		m_LastFrameCapture = {};
		m_Initialized = false;
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
		if (viewportChanged && !m_RHI->Resize(width, height))
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
			const TextureHandle previousShadowMap = m_ShadowMap;
			m_ShadowMap = replacementShadowMap;
			if (previousShadowMap)
			{
				m_RHI->DestroyTexture(previousShadowMap);
			}
		}
		m_CreateInfo.Width = width;
		m_CreateInfo.Height = height;
		m_CreateInfo.ShadowMapSize = shadowMapSize;
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
		return true;
	}

	bool Renderer::UpdateMesh(const MeshData& mesh)
	{
		auto iterator = m_Meshes.find(mesh.Id);
		if (iterator == m_Meshes.end())
		{
			return Fail("Mesh identifier does not exist");
		}

		MeshResource replacement;
		if (!CreateMeshResources(mesh, replacement))
		{
			return false;
		}
		replacement.Outline = iterator->second.Outline;
		DestroyMeshResources(iterator->second);
		iterator->second = std::move(replacement);
		return true;
	}

	bool Renderer::SetWorldTransform(const std::string& id, const Matrix4& transform)
	{
		auto iterator = m_Meshes.find(id);
		if (iterator == m_Meshes.end())
		{
			return Fail("Mesh identifier does not exist");
		}
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
		return true;
	}

	void Renderer::ClearScene()
	{
		if (m_RHI && m_RHI->IsInitialized())
		{
			for (auto& pair : m_Meshes)
			{
				DestroyMeshResources(pair.second);
			}
		}
		m_Meshes.clear();
		m_TextureCache.clear();
	}

	void Renderer::SetRenderMode(RenderMode mode)
	{
		m_RenderMode = mode;
	}

	RenderMode Renderer::GetRenderMode() const
	{
		return m_RenderMode;
	}

	void Renderer::SetShadowsEnabled(bool enabled)
	{
		m_ShadowsEnabled = enabled;
	}

	bool Renderer::AreShadowsEnabled() const
	{
		return m_ShadowsEnabled;
	}

	void Renderer::SetLighting(const LightingState& lighting)
	{
		m_Lighting = lighting;
		m_Lighting.LightCount = std::min(
			m_Lighting.LightCount,
			MaximumSceneLights);
		if (m_Lighting.ShadowLightIndex < 0 ||
			static_cast<std::uint32_t>(m_Lighting.ShadowLightIndex) >=
				m_Lighting.LightCount ||
			!m_Lighting.Lights[
				static_cast<std::size_t>(m_Lighting.ShadowLightIndex)]
					.CastShadows)
		{
			m_Lighting.ShadowLightIndex = -1;
		}
	}

	const LightingState& Renderer::GetLighting() const
	{
		return m_Lighting;
	}

	void Renderer::SetViewProjection(const Matrix4& viewProjection)
	{
		m_ViewProjection = viewProjection;
		m_AutomaticViewProjection = false;
	}

	void Renderer::SetLightViewProjection(const Matrix4& lightViewProjection)
	{
		m_LightViewProjection = lightViewProjection;
		m_AutomaticLightViewProjection = false;
	}

	void Renderer::SetAutomaticViewProjection(bool enabled)
	{
		m_AutomaticViewProjection = enabled;
		m_AutomaticLightViewProjection = enabled;
	}

	bool Renderer::IsAutomaticViewProjection() const
	{
		return m_AutomaticViewProjection;
	}

	bool Renderer::RenderFrame()
	{
		if (!m_Initialized)
		{
			return Fail("Renderer is not initialized");
		}
		UpdateAutomaticViewProjections();
		m_RHI->SetDepthCaptureRequired(std::any_of(m_Meshes.begin(), m_Meshes.end(),
			[](const auto& entry) { return entry.second.Mesh.Visible && entry.second.Outline.Enabled; }));
		if (!m_RHI->BeginFrame())
		{
			m_LastError = m_RHI->GetLastError();
			return false;
		}
		m_LastError.clear();
		const auto abortFrame = [this]()
		{
			const std::string error = m_LastError.empty()
				? m_RHI->GetLastError()
				: m_LastError;
			m_RHI->AbortFrame();
			m_LastError = error;
			return false;
		};
		try
		{
			if (m_ShadowsEnabled && !RenderShadowPass())
			{
				return abortFrame();
			}
			if (!RenderColorPass())
			{
				return abortFrame();
			}
			if (!m_RHI->Present())
			{
				m_LastError = m_RHI->GetLastError();
				return abortFrame();
			}
			m_LastFrameCapture = m_RHI->GetLastFrameCapture();
			ApplyOutlines(m_LastFrameCapture);
		}
		catch (...)
		{
			try
			{
				m_RHI->AbortFrame();
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

	FrameCapture Renderer::GetLastFrameCapture() const
	{
		return m_LastFrameCapture;
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
			candidate.Mesh = mesh;
			Float3 minimum = mesh.Vertices.front().Position;
			Float3 maximum = minimum;
			for (const Vertex& vertex : mesh.Vertices)
			{
				minimum.X = std::min(minimum.X, vertex.Position.X);
				minimum.Y = std::min(minimum.Y, vertex.Position.Y);
				minimum.Z = std::min(minimum.Z, vertex.Position.Z);
				maximum.X = std::max(maximum.X, vertex.Position.X);
				maximum.Y = std::max(maximum.Y, vertex.Position.Y);
				maximum.Z = std::max(maximum.Z, vertex.Position.Z);
			}
			candidate.BoundsCenter = Multiply(Add(minimum, maximum), 0.5f);
			candidate.Translucent = IsMeshTranslucent(mesh);
			const auto rollback = [this, &candidate](std::string error)
			{
				DestroyMeshResources(candidate);
				m_LastError = std::move(error);
				return false;
			};

			BufferDesc vertexBufferDesc;
			vertexBufferDesc.Usage = BufferUsage::Vertex;
			vertexBufferDesc.SizeInBytes = mesh.Vertices.size() * sizeof(Vertex);
			vertexBufferDesc.DebugName = mesh.Id + ".Vertices";
			candidate.VertexBuffer = m_RHI->CreateBuffer(
				vertexBufferDesc,
				mesh.Vertices.data());
			if (!candidate.VertexBuffer)
			{
				return rollback(m_RHI->GetLastError());
			}

			BufferDesc indexBufferDesc;
			indexBufferDesc.Usage = BufferUsage::Index;
			indexBufferDesc.SizeInBytes = mesh.Indices.size() * sizeof(std::uint32_t);
			indexBufferDesc.DebugName = mesh.Id + ".Indices";
			candidate.IndexBuffer = m_RHI->CreateBuffer(
				indexBufferDesc,
				mesh.Indices.data());
			if (!candidate.IndexBuffer)
			{
				return rollback(m_RHI->GetLastError());
			}

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
					mesh.Material.OpacityImage,
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
		ReleaseTexture(
			resource.Mesh.Material.DiffuseImage,
			TextureColorSpace::SRGB,
			resource.DiffuseTexture);
		ReleaseTexture(
			resource.Mesh.Material.NormalImage,
			TextureColorSpace::Linear,
			resource.NormalTexture);
		ReleaseTexture(
			resource.Mesh.Material.OpacityImage,
			TextureColorSpace::Linear,
			resource.OpacityTexture);
		if (resource.VertexBuffer)
		{
			m_RHI->DestroyBuffer(resource.VertexBuffer);
		}
		if (resource.IndexBuffer)
		{
			m_RHI->DestroyBuffer(resource.IndexBuffer);
		}
		resource.VertexBuffer = {};
		resource.IndexBuffer = {};
		resource.Mesh = {};
		resource.Outline = {};
	}

	bool Renderer::AcquireTexture(
		const std::shared_ptr<const ImageData>& image,
		TextureColorSpace colorSpace,
		const std::string& debugName,
		TextureHandle& texture)
	{
		texture = {};
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
			texture = cached->second.Texture;
			return true;
		}

		TextureDesc description;
		description.Width = static_cast<int>(image->Width);
		description.Height = static_cast<int>(image->Height);
		description.Format = TextureFormat::RGBA8;
		description.ColorSpace = colorSpace;
		description.RenderTarget = false;
		description.DebugName = debugName;
		const TextureHandle created = m_RHI->CreateTexture(
			description,
			image->PixelsRGBA8.data());
		if (!created)
		{
			m_LastError = m_RHI->GetLastError();
			return false;
		}

		TextureCacheEntry entry;
		entry.Image = image;
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
			texture = created;
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

	void Renderer::ReleaseTexture(
		const std::shared_ptr<const ImageData>& image,
		TextureColorSpace colorSpace,
		TextureHandle& texture)
	{
		if (!texture)
		{
			return;
		}

		const auto cached = image
			? m_TextureCache.find({ image.get(), colorSpace })
			: m_TextureCache.end();
		if (cached != m_TextureCache.end() &&
			cached->second.Texture == texture)
		{
			if (cached->second.ReferenceCount > 1)
			{
				--cached->second.ReferenceCount;
			}
			else
			{
				if (m_RHI && m_RHI->IsInitialized())
				{
					m_RHI->DestroyTexture(cached->second.Texture);
				}
				m_TextureCache.erase(cached);
			}
		}
		else if (m_RHI && m_RHI->IsInitialized())
		{
			// This path should only be reached for a partially constructed
			// resource whose cache bookkeeping could not be completed.
			m_RHI->DestroyTexture(texture);
		}
		texture = {};
	}

	void Renderer::ConfigureMaterialPipeline(
		const MeshResource& mesh,
		PipelineState& pipeline) const
	{
		const MaterialData& material = mesh.Mesh.Material;
		pipeline.MaterialDiffuseColor = material.BaseColor;
		pipeline.MaterialAmbientColor = material.AmbientColor;
		pipeline.MaterialSpecularColor = material.SpecularColor;
		pipeline.MaterialEmissiveColor = material.EmissiveColor;
		pipeline.MaterialShininess = material.Shininess;
		pipeline.MaterialOpacity =
			mesh.Mesh.Color.W * material.BaseColor.W;
		pipeline.AlphaCutoff = material.AlphaCutoff;
		pipeline.DiffuseTexture = mesh.DiffuseTexture;
		pipeline.NormalTexture = mesh.NormalTexture;
		pipeline.OpacityTexture = mesh.OpacityTexture;
		pipeline.Lighting = m_Lighting;
	}

	bool Renderer::RenderShadowPass()
	{
		m_CurrentPass = PassType::Shadow;
		PassDesc pass;
		pass.Type = PassType::Shadow;
		pass.Target = m_ShadowMap;
		pass.ClearColorEnabled = false;
		if (!m_RHI->BeginPass(pass))
		{
			return Fail(m_RHI->GetLastError().c_str());
		}

		for (const auto& pair : m_Meshes)
		{
			const MeshResource& mesh = pair.second;
			if (!mesh.Mesh.Visible || !mesh.Mesh.CastShadow)
			{
				continue;
			}
			PipelineState pipeline;
			pipeline.Fill = FillMode::Solid;
			pipeline.ColorWrite = false;
			pipeline.BlendEnabled = mesh.Translucent;
			pipeline.Color = mesh.Mesh.Color;
			ConfigureMaterialPipeline(mesh, pipeline);
			if (!DrawMesh(mesh, pipeline))
			{
				return false;
			}
		}
		if (!m_RHI->EndPass())
		{
			return Fail(m_RHI->GetLastError().c_str());
		}
		return true;
	}

	bool Renderer::RenderColorPass()
	{
		m_CurrentPass = PassType::Color;
		PassDesc pass;
		pass.Type = PassType::Color;
		if (!m_RHI->BeginPass(pass))
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
			(mesh.Translucent ? translucentMeshes : opaqueMeshes)
				.push_back(&mesh);
		}

		const Float3 cameraPosition = m_Lighting.CameraPosition;
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
		std::sort(
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
			solid.DepthWrite = !translucent;
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

		if (!m_RHI->EndPass())
		{
			return Fail(m_RHI->GetLastError().c_str());
		}
		return true;
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
		if (!m_RHI->BeginPass(pass))
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
			if (!DrawMesh(mesh, outline))
			{
				return false;
			}
		}

		if (!m_RHI->EndPass())
		{
			return Fail(m_RHI->GetLastError().c_str());
		}
		return true;
	}

	void Renderer::ApplyOutlines(FrameCapture& capture) const
	{
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
			const MeshData& mesh = resource.Mesh;
			if (!mesh.Visible || !resource.Outline.Enabled ||
				mesh.Vertices.empty() || mesh.Indices.size() < 3)
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
				m_ViewProjection,
				mesh.WorldTransform);
			std::vector<ProjectedVertex> vertices(mesh.Vertices.size());
			for (std::size_t vertexIndex = 0;
				vertexIndex < mesh.Vertices.size();
				++vertexIndex)
			{
				const Vertex& source = mesh.Vertices[vertexIndex];
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
				triangle + 2 < mesh.Indices.size();
				triangle += 3)
			{
				const std::uint32_t index0 = mesh.Indices[triangle];
				const std::uint32_t index1 = mesh.Indices[triangle + 1];
				const std::uint32_t index2 = mesh.Indices[triangle + 2];
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
							sampleImage(
								mesh.Material.DiffuseImage,
								u,
								v,
								3U) *
							sampleImage(
								mesh.Material.OpacityImage,
								u,
								v,
								0U);
						if (mesh.Material.AlphaCutoff > 0.0f &&
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
		const Matrix4& viewProjection = m_CurrentPass == PassType::Shadow
			? m_LightViewProjection : m_ViewProjection;
		if (!m_RHI->BindPipeline(pipeline)
			|| !m_RHI->BindWorldTransform(mesh.Mesh.WorldTransform)
			|| !m_RHI->BindViewProjection(viewProjection)
			|| !m_RHI->BindLightViewProjection(m_LightViewProjection)
			|| !m_RHI->BindVertexBuffer(mesh.VertexBuffer)
			|| !m_RHI->BindIndexBuffer(mesh.IndexBuffer)
			|| !m_RHI->DrawIndexed(0, static_cast<std::uint32_t>(mesh.Mesh.Indices.size()), 0))
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
			for (const Vertex& vertex : resource.Mesh.Vertices)
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
		if (!hasGeometry)
		{
			if (m_AutomaticViewProjection) m_ViewProjection = Matrix4::Identity();
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
			m_ViewProjection = Multiply(Perspective(1.0471975512f, aspect,
				std::max(0.001f, radius * 0.02f), cameraDistance + radius * 2.5f), view);
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
