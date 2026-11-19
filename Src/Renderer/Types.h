#pragma once
#include "Techniques/RenderBuffer.h"

#include "Techniques/TechniqueCatalog.h"
#include "Techniques/VoxelGlobalIllumination.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Evans::Renderer
{
	// Capacity of the existing inline shader ABI, not the complete scene list.
	constexpr std::uint32_t MaximumSceneLights = 4;
	class SceneLightCollection;

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
		// World-space radius per emitter sample; zero means unbounded.
		float Range = 0.0f;
		// Prepared rectangle transport: zero preserves the normalized four-sample
		// legacy model; otherwise Intensity is radiance and this is the grid width.
		std::uint32_t AreaSampleAxisCount = 0;
	};

	struct LightingState
	{
		std::array<SceneLight, MaximumSceneLights> Lights;
		std::uint32_t LightCount = 1;
		std::int32_t ShadowLightIndex = 0;
		Float3 CameraPosition = { 0.0f, 0.0f, 5.0f };
		Float3 AmbientColor = { 1.0f, 1.0f, 1.0f };
		float AmbientStrength = 0.08f;
		// Optional authoritative scene-order snapshot. SetLighting projects its
		// compatibility subset into Lights; copying a pass never copies the list.
		std::shared_ptr<const SceneLightCollection> SceneLights;
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
		float SubsurfaceWeight = 0.0f;
		bool Refractive = false;
		std::optional<float> Roughness, Metallic;
        float DielectricReflectance=0.04f;
		std::shared_ptr<const ImageData> DiffuseImage;
		std::shared_ptr<const ImageData> NormalImage;
		std::shared_ptr<const ImageData> OpacityImage;
        // Linear RGB stores coverage, glTF roughness (G) and metallic (B).
        std::shared_ptr<const ImageData> MetallicRoughnessImage;
        const std::shared_ptr<const ImageData>& SurfaceImage() const noexcept
        { return MetallicRoughnessImage ? MetallicRoughnessImage : OpacityImage; }
	};

	enum class RHIBackend : std::uint8_t
	{
		DX11,
		DX12,
		Metal,
		OpenGLES,
		Vulkan
	};

	inline Techniques::Backend TechniqueBackend(RHIBackend backend)
	{
		switch (backend)
		{
		case RHIBackend::DX11: return Techniques::Backend::DX11;
		case RHIBackend::DX12: return Techniques::Backend::DX12;
		case RHIBackend::Metal: return Techniques::Backend::Metal;
		case RHIBackend::OpenGLES: return Techniques::Backend::OpenGLES;
		case RHIBackend::Vulkan: return Techniques::Backend::Vulkan;
		}
		return Techniques::Backend::Unknown;
	}

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
		Outline,
		GBuffer,
		DeferredLighting,
		Velocity,
		EnvironmentCapture,
        PlanarCapture,
        ContactDepth,
        ForwardPlusDepth
	};

    // Captures and main-camera assignment share one immutable scene index space.
    // The raster camera includes the current temporal jitter, when enabled.
    enum class LightListLayout : std::uint32_t { Tiles = 0, Clusters = 1 };
    struct LightListFrameDesc
    {
        Matrix4 ViewProjection;
        std::shared_ptr<const SceneLightCollection> Lights;
        LightListLayout Layout = LightListLayout::Tiles;
    };
    // Main-camera opaque depth supplies short visibility rays toward the selected light.
    struct ContactShadowDesc
    {
        Matrix4 ViewProjection;
        LightingState Lighting;
    };
	// Geometry fills a material G-buffer. A separate fullscreen pass reconstructs
	// world position from depth and consumes the same scene lights as forward.
    // DirectDiffuse separates the same shadowed light integral for diffusion.
    // Its alpha stores sample coverage times the nonmetallic fraction.
    enum class DeferredOutputMode : std::uint32_t { Scene, DirectDiffuse };
	struct DeferredLightingDesc
	{
		Matrix4 ViewProjection;
		Matrix4 LightViewProjection;
		LightingState Lighting;
		Techniques::ShadingModel Shading = Techniques::ShadingModel::Phong;
		float ToonBands = 4.0f;
		bool SamplesShadowMap = true;
        DeferredOutputMode Output = DeferredOutputMode::Scene;
	};

	struct AmbientOcclusionDesc
	{
		Matrix4 ViewProjection;
	};

	struct DepthOfFieldDesc
	{
		Matrix4 ViewProjection;
	};

	// Motion is current minus previous screen position in top-left image space.
	// Previous clip coordinates must be interpolated before the perspective divide.
	// These inputs describe rigid geometry; changed vertex data invalidates history.
	struct MotionTransform
	{
		Matrix4 PreviousWorldViewProjection;
		float DeltaSeconds = 0.0f;
		bool Valid = false;
		Float2 JitterDeltaPixels;
	};

	struct MotionFrame
	{
		Matrix4 PreviousViewProjection;
		float DeltaSeconds = 0.0f;
		bool Valid = false;
		bool MotionDetected = false;
		bool CorrespondenceValid = false; // repeated scene time is valid for TAA, not shutter speed
	};
	struct TemporalFrame
	{
		Float2 JitterPixels;
		Float2 PreviousJitterPixels;
		std::uint32_t SampleIndex = 0;
		bool HistoryValid = false;
	};
    struct ReflectionDesc
    {
        bool SkyboxOnly=false,DrawSkybox=false;
        Matrix4 ViewProjection;
        Matrix4 PreviousViewProjection;
        TemporalFrame Sampling;
    };
	struct TemporalAADesc
	{
		Matrix4 PreviousViewProjection;
		TemporalFrame Sampling;
	};

    struct GlobalIlluminationDesc
    {
        Matrix4 ViewProjection;
        Matrix4 PreviousViewProjection;
        TemporalFrame Sampling;
        bool Translucency=false; // The scene has visible transparent lighting consumers.
    };

	struct MotionBlurDesc
	{
		Matrix4 ViewProjection;
		MotionFrame Motion;
	};
    struct DensityVolumeBinding;
	struct VolumetricFogDesc
	{
        const DensityVolumeBinding* Density = nullptr;
		Matrix4 ViewProjection;
		Matrix4 LightViewProjection;
		LightingState Lighting;
		bool SamplesShadowMap = false;
	};

	struct LightShaftDesc
	{
		Matrix4 ViewProjection;
		LightingState Lighting;
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
		// RGBA8 uploads store tightly packed mip levels from largest to smallest.
		int MipLevels = 1;
	};

	struct PipelineState
	{
		FillMode Fill = FillMode::Solid;
		// Triangle coverage with barycentric edge rejection preserves solid depth.
		bool WireframeOverlay = false;
		Techniques::ShadingModel Shading = Techniques::ShadingModel::Phong;
		float Roughness = 0.5f;
		float Metallic = 0.0f;
		float ToonBands = 4.0f;
		bool BackFaceCulling = false;
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
		float SubsurfaceWeight = 0.0f;
		float AlphaCutoff = 0.0f;
		TextureHandle DiffuseTexture;
		TextureHandle NormalTexture;
		TextureHandle OpacityTexture;
        bool PackedMetallicRoughness = false;
        Techniques::ParallaxOcclusionSettings Parallax;
        Float3 ParallaxBoundsMinimum{-0.5f,-0.5f,-0.5f},ParallaxBoundsMaximum{0.5f,0.5f,0.5f};
		LightingState Lighting;
	};

	struct PassDesc
	{
		PassType Type = PassType::Color;
		TextureHandle Target;
		Float4 ClearColor = { 0.02f, 0.02f, 0.025f, 1.0f };
		bool ClearColorEnabled = true;
		bool ClearDepthEnabled = true;
		std::uint32_t EnvironmentFaceIndex = 0;
        // Private OIT layer color and captured depth; the ordinary material
        // shader replays with depth EQUAL and source-over RGBA blending.
        bool DepthPeelingLayer = false;
	};

	struct RHICreateInfo
	{
		void* NativeWindow = nullptr;
		int Width = 1280;
		int Height = 720;
		// Live editor presentation can skip diagnostic depth captures. Main depth
		// is still captured on demand when the outline compositor needs it.
		bool CaptureDiagnosticBuffers = true;
	};

	struct RHICapabilities
	{
		bool InstancedRendering = false;
        bool GPUDrivenRendering = false;
        bool VisibilityBufferRendering = false;
        // Zero means no per-block scene-storage limit is exposed by the backend.
        std::uint64_t SceneStorageBlockBytes = 0;
        bool HybridRendering = false;
        bool MaterialAnisotropicFiltering = false;
		bool NativeAPI = false;
		bool Wireframe = true;
		bool ShadowMaps = true;
        bool VarianceShadowMaps = false;
        bool ContactShadows = false;
        bool ForwardPlusRendering = false;
        bool ClusteredForwardRendering = false;
        bool TiledDeferredRendering = false;
        bool ScreenSpaceGlobalIllumination = false;
        bool BakedLightProbes = false;
        bool BakedLightmaps = false;
        bool Refraction = false;
        bool DepthPeeling = false;
        bool RayTracedAmbientOcclusion = false;
        bool RayTracedReflections = false;
        bool RayTracedGI = false;
        bool VoxelGI = false;
        bool LumenGI = false;
        bool LumenHardwareRayTracing = false;
        bool PathTracing = false;
		bool Outline = true;
		bool AsyncCommandRecording = false;
		bool DeferredShading = false;
		bool ScalableAmbientObscurance = false;
		bool TemporalAntiAliasing = false;
        bool TemporalSuperResolution=false;
        bool TemporalEffects = false;
        bool ParallaxOcclusion = false;
        bool GeometryDisplacement = false;
        bool GeometryTessellation = false;
        bool GeometryVirtualization = false;
        bool NaniteRendering = false;
        bool MeshShaders = false;
        bool NaniteHardwareScheduling = false;
        bool VirtualShadowMaps = false;
		bool VolumetricFog = false;
		bool VolumetricLighting = false;
		bool LightShafts = false;
		std::uint32_t WeightedTransparencySampleCounts = 0;
		std::uint32_t MaximumTextureSize = 16384;
		std::uint32_t SupportedSampleCounts = 1;
	};

	struct DrawInstance
	{
		Matrix4 WorldTransform;
		MotionTransform Motion;
        Float4 Wind{0,0,0,0}, Gust{0,0,0,0}, VegetationHistory{0,0,0,0};
	};

	struct DrawCallRecord
	{
		BufferHandle VertexBuffer;
		BufferHandle IndexBuffer;
		PipelineState Pipeline;
		Matrix4 WorldTransform;
		MotionTransform Motion;
		std::uint32_t FirstIndex = 0;
		std::uint32_t IndexCount = 0;
		std::int32_t BaseVertex = 0;
		// Empty for a conventional draw; otherwise one actual native submission.
		std::vector<DrawInstance> Instances;
        bool GPUDriven = false; // Instances records candidates; visibility remains on the GPU.
        bool NaniteIndirect = false;
        std::uint32_t NaniteQueryIndex=0;
        bool VisibilityBuffer = false;
        bool MeshShader = false;
        std::uint32_t VisibilityFirstTriangle = 0;
	};

	struct PassRecord
	{
		PassDesc Description;
		std::vector<DrawCallRecord> DrawCalls;
	};

    struct PathTracingDesc
    {
        Matrix4 ViewProjection;
        Float4 DisplayBackground={0.02f,0.02f,0.025f,1};
        std::uint64_t Epoch=0,ResetSerial=0;
        bool DirectShadows=true;
    };
	struct FrameStatistics
	{
        std::uint32_t ContactDepthDrawCallCount = 0;
        std::uint32_t ForwardPlusDepthDrawCallCount = 0;
        std::uint32_t ForwardPlusAssignmentCount = 0;
        std::uint32_t ClusteredAssignmentCount = 0;
        std::uint32_t TiledDeferredAssignmentCount = 0;
        std::uint32_t ContactShadowPassCount = 0;
		std::uint64_t FrameNumber = 0;
		std::uint32_t PassCount = 0;
		std::uint32_t DrawCallCount = 0;
		std::uint32_t InstancedDrawCallCount = 0;
        std::uint32_t NaniteDrawDispatchCount=0,NaniteIndirectDrawCount=0;
        std::uint32_t NaniteRasterDispatchCount=0,NaniteRasterExportCount=0,NaniteGeometryAssemblyCount=0;
        std::uint32_t NaniteHardwareBinCount=0,NaniteHardwareDispatchCount=0,NaniteHardwareIndirectCount=0;
        std::uint32_t MeshShaderDispatchCount=0,MeshShaderIndirectCount=0,MeshShaderObjectDispatchCount=0,MeshShaderArgumentDispatchCount=0;
        std::uint32_t NaniteInstanceDispatchCount=0; // Count, scan and indirect assembly commands; no GPU readback.
        std::uint64_t NaniteCandidateTriangles=0;
        std::uint32_t NaniteOcclusionQueries=0,NaniteOcclusionPhases=0;
        std::uint32_t NaniteMaterialBinCount=0,NaniteMaterialDispatchCount=0;
        std::uint32_t NaniteMeshCount=0,NaniteSelectedClusters=0,NaniteResidentPages=0,NanitePendingPages=0,NaniteUploadedPages=0;
        std::uint64_t NaniteGeometryGeneration=0;
        std::uint64_t NaniteReadbackBytes=0;
        std::uint32_t NaniteHostRebuiltVertices=0;
        std::uint32_t NaniteScenePageBudget=0,NaniteFamilyCount=0,NaniteViewCount=0,NaniteBudgetBlockedGroups=0,NaniteCookRequests=0;
        std::uint64_t NaniteRootPages=0,NaniteAllocatedPages=0,NaniteCookSourceBytes=0,NanitePreparationMicroseconds=0;
        bool NaniteBudgetLimited=false;
        std::uint32_t GpuCullDispatchCount = 0, IndirectDrawCallCount = 0;
        std::uint32_t VisibilityRasterDrawCount = 0, VisibilityMaterialDrawCount = 0;
        std::uint64_t GpuCandidateInstanceCount = 0, GpuCandidateTriangleCount = 0;
		std::uint32_t ShadowDrawCallCount = 0;
        std::uint32_t ShadowMomentFilterPassCount = 0;
		std::uint32_t OutlineDrawCallCount = 0;
		std::uint64_t SubmittedTriangleCount = 0;
		std::uint32_t FrustumCulledMeshCount = 0;
		std::uint32_t LodReducedMeshCount = 0;
		// Native technique draws have dedicated counters outside ordinary mesh records.
		std::uint32_t PostProcessPassCount = 0;
		std::uint32_t DeferredLightingPassCount = 0;
        std::uint32_t SubsurfacePassCount = 0;
        std::uint32_t HairFurPassCount = 0;
        std::uint32_t HairFurDepthDrawCount=0,HairFurShadowDrawCount=0,HairFurCaptureDrawCount=0;
        std::uint32_t HairFurVelocityDrawCount=0;
		std::uint32_t AmbientOcclusionPassCount = 0;
		std::uint32_t ReflectionPassCount = 0;
        std::uint32_t SkyboxPassCount=0;
        std::uint32_t HybridRenderingPassCount = 0;
        std::uint32_t GlobalIlluminationPassCount = 0;
        std::uint32_t LumenTranslucencyProbeCount = 0;
        std::uint32_t LumenHairReceiverPassCount = 0;
        std::uint32_t LumenHardwareRayCount = 0; // Native trace slots; screen/cache answers can bypass individual intersections.
        std::uint32_t PathTracingPassCount = 0, PathTracingSampleCount = 0;
        std::uint32_t LightmapReceiverDrawCount = 0;
        std::uint32_t RefractionDrawCount = 0, RefractionPassCount = 0;
        std::uint32_t DepthPeelingDrawCount = 0, DepthPeelingPassCount = 0;
        std::uint64_t LightmapReceiverBuildCount = 0, LightmapBakeCount = 0;
        std::uint64_t ProbeBakeCount = 0;
        bool BakedLightingFromAsset = false;
		std::uint32_t EnvironmentCaptureFaceCount = 0;
		std::uint32_t EnvironmentMipPassCount = 0;
		std::uint32_t EnvironmentConversionPassCount = 0;
		std::uint32_t PlanarCapturePassCount = 0;
		std::uint32_t PlanarMipPassCount = 0;
		std::uint32_t DepthOfFieldPassCount = 0;
		std::uint32_t MotionBlurPassCount = 0;
		std::uint32_t TemporalAAPassCount = 0;
        std::uint32_t TemporalSuperResolutionPassCount=0;
        bool TemporalSuperResolutionHistoryValid=false;
        std::uint64_t TemporalSuperResolutionSerial=0;
        std::uint32_t TemporalSuperResolutionForegroundPixels=0;
        std::uint32_t TemporalSuperResolutionValidMotionPixels=0;
        std::uint32_t TemporalSuperResolutionAccumulatedPixels=0;
        std::uint32_t TemporalEffectsPassCount = 0;
        std::uint32_t ParallaxMaterialDrawCount = 0;
        std::uint32_t DisplacedMeshCount=0,DisplacedVertexCount=0,DisplacementPreparationDispatches=0;
        std::uint64_t DisplacementGeneration=0;
        std::uint32_t TessellatedMeshCount=0,TessellatedVertexCount=0,TessellationPreparationDispatches=0,GeometryPreparationDispatches=0;
        std::uint64_t PreparedGeometryGeneration=0;
        std::uint32_t VirtualMeshCount=0,VirtualNodeCount=0;
        std::uint64_t VirtualResidentPages=0,VirtualPendingPages=0,VirtualUploadedPages=0,VirtualCopiedPages=0,VirtualBootstrapPages=0;
        std::uint64_t VirtualUploadedBytes=0,VirtualCopiedBytes=0,VirtualPoolBytes=0,VirtualGeometryGeneration=0;
        std::uint64_t VirtualShadowSerial=0;
        std::uint32_t VirtualShadowRequestedPages=0,VirtualShadowResidentPages=0,VirtualShadowRenderedPages=0,VirtualShadowCopiedPages=0,VirtualShadowDrawCalls=0;
		std::uint32_t VolumetricFogPassCount = 0;
		std::uint32_t VolumetricLightingPassCount = 0;
        std::uint32_t ParticipatingMediaPassCount = 0;
        std::uint32_t VolumetricCloudPassCount = 0;
		std::uint32_t LightShaftPassCount = 0;
        // Completed GPU frame command-buffer interval, including transfers.
        // CPU preparation, fence waits and presentation are excluded; unavailable is explicit.
        double GpuFrameMilliseconds=0;
        bool GpuFrameTimingAvailable=false;
		std::uint32_t WeightedTransparencyPassCount = 0;
		std::uint32_t MultisamplePassCount = 0;
		std::uint32_t RasterSampleCount = 1;
	};

    // Immutable shared GPU surface. Owner keeps the native resource and handle alive.
    struct GpuImage
    {
        RHIBackend Backend = RHIBackend::DX11;
        void* SharedHandle = nullptr;
        void* ReadyHandle = nullptr;
        void* NativeTexture = nullptr;
        void* NativeDisplay = nullptr;
        void* NativeContext = nullptr;
        std::shared_ptr<void> ContextOwner;
        std::uint64_t AllocationBytes = 0;
        std::uint32_t MemoryType = 0, Format = 0;
        std::shared_ptr<void> Owner;
    };

	struct FrameCapture
	{
        std::uint32_t RenderWidth=0,RenderHeight=0;
		RHIBackend Backend = RHIBackend::Vulkan;
		std::vector<PassRecord> Passes;
		std::vector<Techniques::PostProcessPassKind> PostProcessingPasses;
		std::vector<Techniques::HdrPassKind> HDRPasses;
		std::vector<Techniques::AmbientOcclusionPassKind> AmbientOcclusionPasses;
		std::vector<Techniques::ReflectionPassKind> ReflectionPasses;
        std::vector<Techniques::RayGIPass> RayGIPasses;
        std::vector<Techniques::VoxelGIPass> VoxelGIPasses;
		std::vector<Techniques::DepthOfFieldPassKind> DepthOfFieldPasses;
		std::vector<Techniques::MotionBlurPassKind> MotionBlurPasses;
		FrameStatistics Statistics;
		MotionFrame Motion;
		TemporalFrame Temporal;
		std::uint32_t Width = 0;
		bool NeedsTemporalFrame = false;
		std::uint32_t Height = 0;
		// Top-left RGBA8 with display-sRGB RGB and straight linear alpha.
		std::vector<std::uint8_t> ColorRGBA8;
        std::shared_ptr<const GpuImage> GpuColor;
        std::shared_ptr<const GpuImage> GpuContext;
		std::vector<float> Depth32;
        std::vector<Float4> DiagnosticPixels;
		// Diagnostic top-left velocity: XY pixels, Z previous hardware depth, W validity.
		std::vector<Float4> MotionVectors;
		std::uint32_t ShadowWidth = 0;
		std::uint32_t ShadowHeight = 0;
		std::vector<float> ShadowDepth32;
		// Optional filtered RG32 VSM diagnostic data; empty when the backend does not capture it.
		std::vector<Float2> ShadowMoments;
		std::vector<std::vector<float>> CascadeDepth32;
        // Human-readable background status survives editor readback. Pending
        // work requests another frame; failures remain visible without retrying.
        std::string BakedLightingStatus;
        std::string VoxelGIStatus;
        std::string LumenGIStatus;
        std::string HairFurStatus;
        bool HairGeometryPending=false;

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
        // Small periodic deformations keep resident buffers and a stable Lumen
        // distance-field proxy. Rasterization and scene rays use live vertices.
        bool DynamicVertices = false;
        bool PlanarReceiver = false;
        // Opt in only for opaque instances without order-dependent depth ties.
        bool ReorderForInstancing = false;
        unsigned Vegetation = 0; // 1: root-anchored blades, 2: woody plants.
        bool ManagedLod = false;
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
		bool AutomaticViewProjection = true;
		bool CaptureDiagnosticBuffers = true;
	};
}
