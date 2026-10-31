#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Evans::Renderer
{
	constexpr std::uint32_t MaximumSceneLights = 4;

	struct Float2
	{
		float X = 0.0f;
		float Y = 0.0f;
	};

	struct Float3
	{
		float X = 0.0f;
		float Y = 0.0f;
		float Z = 0.0f;
	};

	struct Float4
	{
		float X = 0.0f;
		float Y = 0.0f;
		float Z = 0.0f;
		float W = 1.0f;
	};

	struct Matrix4
	{
		std::array<float, 16> Values = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};

		static Matrix4 Identity() { return {}; }
	};

	struct Vertex
	{
		Float3 Position;
		Float3 Normal = { 0.0f, 0.0f, 1.0f };
		Float4 Color = { 1.0f, 1.0f, 1.0f, 1.0f };
		Float2 TexCoord;
	};

	struct ImageData
	{
		std::uint32_t Width = 0;
		std::uint32_t Height = 0;
		// Top-left RGBA8. RGB encoding is selected by TextureColorSpace.
		std::vector<std::uint8_t> PixelsRGBA8;

		bool IsValid() const
		{
			return Width > 0 && Height > 0 &&
				PixelsRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4;
		}
	};

	enum class LightType : std::uint8_t
	{
		Directional,
		Point,
		RectArea
	};

	struct SceneLight
	{
		LightType Type = LightType::Directional;
		Float3 Position;
		// Directional lights store the direction from the surface to the light.
		// Rect-area lights store the direction emitted by the front face.
		Float3 Direction = { 0.45f, 1.0f, 0.35f };
		Float3 Color = { 1.0f, 1.0f, 1.0f };
		float Intensity = 1.0f;
		Float2 AreaSize = { 1.0f, 1.0f };
		bool CastShadows = true;
	};

	struct LightingState
	{
		std::array<SceneLight, MaximumSceneLights> Lights;
		std::uint32_t LightCount = 1;
		std::int32_t ShadowLightIndex = 0;
		Float3 CameraPosition = { 0.0f, 0.0f, 5.0f };
		Float3 AmbientColor = { 1.0f, 1.0f, 1.0f };
		float AmbientStrength = 0.08f;
	};

	struct MaterialData
	{
		// Linear Phong diffuse reflectance (Kd); alpha is scalar opacity.
		Float4 BaseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		// Linear Phong ambient reflectance (Ka).
		Float4 AmbientColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		Float4 SpecularColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		Float4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		float Shininess = 32.0f;
		float AlphaCutoff = 0.0f;
		std::shared_ptr<const ImageData> DiffuseImage;
		std::shared_ptr<const ImageData> NormalImage;
		std::shared_ptr<const ImageData> OpacityImage;
	};

	enum class RHIBackend : std::uint8_t
	{
		DX11,
		DX12,
		Metal,
		OpenGLES,
		Vulkan
	};

	enum class RenderMode : std::uint8_t
	{
		Solid,
		Wireframe,
		SolidWireframe
	};

	enum class FillMode : std::uint8_t
	{
		Solid,
		Wireframe
	};

	enum class PassType : std::uint8_t
	{
		Shadow,
		Color,
		Outline
	};

	enum class BufferUsage : std::uint8_t
	{
		Vertex,
		Index,
		Uniform
	};

	enum class TextureFormat : std::uint8_t
	{
		RGBA8,
		Depth32
	};

	enum class TextureColorSpace : std::uint8_t
	{
		Linear,
		// Hardware sampling must decode RGB to linear before filtering.
		SRGB
	};

	struct BufferHandle
	{
		std::uint32_t Value = 0;

		explicit operator bool() const { return Value != 0; }
		friend bool operator==(BufferHandle left, BufferHandle right) { return left.Value == right.Value; }
		friend bool operator!=(BufferHandle left, BufferHandle right) { return !(left == right); }
	};

	struct TextureHandle
	{
		std::uint32_t Value = 0;

		explicit operator bool() const { return Value != 0; }
		friend bool operator==(TextureHandle left, TextureHandle right) { return left.Value == right.Value; }
		friend bool operator!=(TextureHandle left, TextureHandle right) { return !(left == right); }
	};

	struct BufferDesc
	{
		BufferUsage Usage = BufferUsage::Vertex;
		std::size_t SizeInBytes = 0;
		bool Dynamic = false;
		std::string DebugName;
	};

	struct TextureDesc
	{
		int Width = 1;
		int Height = 1;
		TextureFormat Format = TextureFormat::RGBA8;
		TextureColorSpace ColorSpace = TextureColorSpace::Linear;
		bool RenderTarget = false;
		std::string DebugName;
	};

	struct PipelineState
	{
		FillMode Fill = FillMode::Solid;
		bool DepthTest = true;
		bool DepthWrite = true;
		bool ColorWrite = true;
		// Straight-alpha source-over blending in the color pass.
		bool BlendEnabled = false;
		bool SamplesShadowMap = false;
		bool OutlineExtrusion = false;
		// Linear object tint for material passes. Wireframe and outline passes
		// interpret RGB as their display-sRGB override color.
		Float4 Color = { 1.0f, 1.0f, 1.0f, 1.0f };
		float OutlineThickness = 0.0f;
		// Linear Phong diffuse reflectance (Kd), separate from object tint.
		Float4 MaterialDiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		Float4 MaterialAmbientColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		Float4 MaterialSpecularColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		Float4 MaterialEmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		float MaterialShininess = 32.0f;
		float MaterialOpacity = 1.0f;
		float AlphaCutoff = 0.0f;
		TextureHandle DiffuseTexture;
		TextureHandle NormalTexture;
		TextureHandle OpacityTexture;
		LightingState Lighting;
	};

	struct PassDesc
	{
		PassType Type = PassType::Color;
		TextureHandle Target;
		Float4 ClearColor = { 0.02f, 0.02f, 0.025f, 1.0f };
		bool ClearColorEnabled = true;
		bool ClearDepthEnabled = true;
	};

	struct RHICreateInfo
	{
		void* NativeWindow = nullptr;
		int Width = 1280;
		int Height = 720;
		bool EnableValidation = true;
		// Live editor presentation can skip diagnostic depth captures. Main depth
		// is still captured on demand when the outline compositor needs it.
		bool CaptureDiagnosticBuffers = true;
	};

	struct RHICapabilities
	{
		bool NativeAPI = false;
		bool Wireframe = true;
		bool ShadowMaps = true;
		bool Outline = true;
		bool AsyncCommandRecording = false;
		std::uint32_t MaximumTextureSize = 16384;
	};

	struct DrawCallRecord
	{
		BufferHandle VertexBuffer;
		BufferHandle IndexBuffer;
		PipelineState Pipeline;
		Matrix4 WorldTransform;
		std::uint32_t FirstIndex = 0;
		std::uint32_t IndexCount = 0;
		std::int32_t BaseVertex = 0;
	};

	struct PassRecord
	{
		PassDesc Description;
		std::vector<DrawCallRecord> DrawCalls;
	};

	struct FrameStatistics
	{
		std::uint64_t FrameNumber = 0;
		std::uint32_t PassCount = 0;
		std::uint32_t DrawCallCount = 0;
		std::uint32_t ShadowDrawCallCount = 0;
		std::uint32_t OutlineDrawCallCount = 0;
		std::uint64_t SubmittedTriangleCount = 0;
	};

	struct FrameCapture
	{
		RHIBackend Backend = RHIBackend::Vulkan;
		std::vector<PassRecord> Passes;
		FrameStatistics Statistics;
		std::uint32_t Width = 0;
		std::uint32_t Height = 0;
		// Top-left RGBA8 with display-sRGB RGB and straight linear alpha.
		std::vector<std::uint8_t> ColorRGBA8;
		std::vector<float> Depth32;
		std::uint32_t ShadowWidth = 0;
		std::uint32_t ShadowHeight = 0;
		std::vector<float> ShadowDepth32;

		bool HasColorOutput() const
		{
			return Width > 0 && Height > 0
				&& ColorRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4;
		}
	};

	struct MeshData
	{
		std::string Id;
		std::vector<Vertex> Vertices;
		std::vector<std::uint32_t> Indices;
		Matrix4 WorldTransform;
		Float4 Color = { 0.72f, 0.76f, 0.82f, 1.0f };
		MaterialData Material;
		std::uint64_t SelectionGroup = 0;
		bool CastShadow = true;
		bool Visible = true;
	};

	struct OutlineSettings
	{
		bool Enabled = false;
		Float4 Color = { 1.0f, 0.55f, 0.05f, 1.0f };
		float Thickness = 3.0f;
	};

	struct RendererCreateInfo
	{
		RHIBackend Backend = RHIBackend::Vulkan;
		void* NativeWindow = nullptr;
		int Width = 1280;
		int Height = 720;
		int ShadowMapSize = 2048;
		bool EnableValidation = true;
		bool AutomaticViewProjection = true;
		bool CaptureDiagnosticBuffers = true;
	};
}
