#pragma once

#include "IRHI.h"
#include "Renderer/Volumetrics/DensityVolume.h"
#include "FrameHistory.h"
#include "Renderer/PostProcessing/TemporalSampling.h"
#include "Renderer/Geometry/LevelOfDetail.h"
#include "Renderer/BakedLighting/BakedProbeCache.h"
#include "Renderer/BakedLighting/BakedLightmapCache.h"
#include "Renderer/VoxelGI/VoxelRadianceCache.h"
#include "Renderer/Lumen/LumenRenderCache.h"
#include "Renderer/Lumen/LumenWorldRadianceCache.h"
#include "Renderer/Lighting/AreaLightSampling.h"
#include "Renderer/Hair/HairGeometryCache.h"
#include "Renderer/VirtualGeometry/VirtualGeometryStreaming.h"
#include "Renderer/Nanite/NaniteStream.h"
#include "Renderer/Nanite/NaniteAsset.h"
#include "Renderer/Nanite/NaniteSceneBudget.h"
#include "Renderer/VirtualShadowMaps/VirtualShadowScene.h"

#include <functional>
#include <memory>
#include <optional>
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
		void SetTechniqueSettings(const Techniques::Settings& settings);
        void SetDensityVolume(std::shared_ptr<const DensityVolume> source);
        void SetEnvironmentRadiance(std::shared_ptr<const EnvironmentRadiance> source);
        void SetBakedLightingDirectory(const std::filesystem::path& directory);
        void SetBakedLightingShadingOverride(std::optional<Techniques::ShadingModel> shading);
        void SetLightmapDirectory(const std::filesystem::path& directory);
        void SetHairSelectionGroup(std::uint64_t selectionGroup) { m_HairSelectionGroup=selectionGroup; }
		const Techniques::Settings& GetTechniqueSettings() const;
		void SetLighting(const LightingState& lighting);
		const LightingState& GetLighting() const;
		void SetViewProjection(const Matrix4& viewProjection);
		// CSM requires the original camera matrices; a combined matrix cannot
		// reliably recover stable camera-space split distances after float rounding.
		void SetCameraMatrices(const Matrix4& view, const Matrix4& projection);
		void SetLightViewProjection(const Matrix4& lightViewProjection);
		void SetAutomaticViewProjection(bool enabled);
		bool IsAutomaticViewProjection() const;
		// Supply scene sample time before RenderFrame; otherwise a monotonic clock
		// is used. Camera teleports require ResetFrameHistory, ordinary motion does not.
		void SetFrameTime(double sampleTimeSeconds, std::uint64_t epoch = 0);
        void SetVegetation(double time, Float4 wind, Float4 gust);
        void SetRenderBuffer(Techniques::RenderBuffer buffer)
        { m_RenderBuffer=buffer;if(m_RHI)m_RHI->SetDiagnosticBuffer(buffer); }
		void ResetFrameHistory() noexcept;
		bool RenderFrame();

		void SetGpuPresentationEnabled(bool enabled);
        void ReleasePresentationContext() noexcept;
		FrameCapture GetLastFrameCapture() const;
		const FrameCapture& GetLastFrameCaptureView() const noexcept;
		// Transfers the capture; the view is empty until the next completed frame.
		FrameCapture TakeLastFrameCapture() noexcept;
        LumenRenderStatus GetLumenGIStatus() const { return m_LumenGI?m_LumenGI->Status():LumenRenderStatus{}; }
        LumenWorldCacheStatistics GetLumenWorldCacheStatistics() const { return m_LumenWorldCache?m_LumenWorldCache->Statistics():LumenWorldCacheStatistics{}; }
        LumenWorldCacheState GetLumenWorldCacheState() const { return m_LumenWorldCache?m_LumenWorldCache->GetState():LumenWorldCacheState::Empty; }
        LumenGlobalDistanceStatistics GetLumenGlobalDistanceStatistics() const { return m_LumenGlobalDistance?m_LumenGlobalDistance->Statistics():LumenGlobalDistanceStatistics{}; }
        LumenGlobalDistanceState GetLumenGlobalDistanceState() const { return m_LumenGlobalDistance?m_LumenGlobalDistance->GetState():LumenGlobalDistanceState::Empty; }
        const VoxelRadianceStatus& GetVoxelGIStatus() const noexcept { return m_VoxelStatus; }
        const HairGeometryStatus& GetHairFurStatus() const noexcept { return m_HairStatus; }
		std::string GetLastError() const;
		std::size_t GetMeshCount() const;
		IRHI* GetRHI();
		const IRHI* GetRHI() const;
        // Immutable world-space geometry used by the native scene-ray consumers.
        [[nodiscard]] std::shared_ptr<const RaySceneUpload> GetRaySceneSnapshot() const noexcept { return m_RayScene; }
        // Frozen camera shared by planar rasterization and receiver lookup.
        [[nodiscard]] const PlanarCaptureFrame& GetPlanarCaptureFrame() const noexcept { return m_PlanarFrame; }

		static MeshData CreateTriangle(const std::string& id = "Triangle");

	private:
		struct MeshGeometryResource
		{
			std::vector<Vertex> Vertices;
			std::vector<std::uint32_t> Indices;
			BufferHandle VertexBuffer;
			BufferHandle IndexBuffer;
			Float3 BoundsMinimum, BoundsMaximum;
			MeshLodChain LodChain;
			std::vector<BufferHandle> LodIndexBuffers;
			struct WireOverlayBuffers { BufferHandle Vertices, Indices; };
			std::vector<WireOverlayBuffers> WireOverlays;
			bool LodPrepared = false;
            bool DynamicVertices = false;
            std::uint64_t ShadowRevision=0;
            bool OwnsVertexBuffer = true,OwnsIndexBuffer = true;
            std::uint32_t FirstIndex=0;
            std::uint32_t NativeIndexCount=0;
            bool HostGeometryReady=true;
            std::shared_ptr<const RefractionSourceGeometry> RefractionSource;
            std::shared_ptr<const HairSourceGeometry> HairSource;
            std::shared_ptr<const LumenSourceGeometry> LumenSource;
			std::uint64_t Hash = 0;
			std::size_t ReferenceCount = 1;
		};

		// Object state stays separate even when all geometry allocations are shared.
		struct MeshInstance
		{
			std::string Id;
			Matrix4 WorldTransform;
			Float4 Color;
			MaterialData Material;
			std::uint64_t SelectionGroup = 0;
			bool CastShadow = true, Visible = true;
            bool DynamicVertices = false;
			MeshInstance() = default;
			explicit MeshInstance(const MeshData& mesh)
				: Id(mesh.Id), WorldTransform(mesh.WorldTransform), Color(mesh.Color), Material(mesh.Material),
				  SelectionGroup(mesh.SelectionGroup), CastShadow(mesh.CastShadow), Visible(mesh.Visible), DynamicVertices(mesh.DynamicVertices),
                  Vegetation(mesh.Vegetation), ManagedLod(mesh.ManagedLod), PlanarReceiver(mesh.PlanarReceiver), ReorderForInstancing(mesh.ReorderForInstancing) {}
            unsigned Vegetation=0;
            bool ManagedLod=false;
            bool PlanarReceiver=false, ReorderForInstancing=false;
		};

		struct TextureCacheEntry;

		struct MeshResource
		{
			MeshInstance Mesh;
			MeshGeometryResource* Geometry = nullptr;
            std::uint64_t LumenKey=0;
            MeshGeometryResource* BaseGeometry = nullptr; // owns the original cache reference; Geometry is the current view
            MeshGeometryResource* SourceGeometry = nullptr; // stable PN/displaced surface for transport and grooming
			OutlineSettings Outline;
			TextureCacheEntry* DiffuseTexture = nullptr;
			TextureCacheEntry* NormalTexture = nullptr;
			TextureCacheEntry* OpacityTexture = nullptr;
			Float3 BoundsCenter;
			Float3 BoundsMinimum;
			Float3 BoundsMaximum;
			bool Translucent = false;
			Matrix4 PreviousWorldTransform;
			bool PreviousTransformValid = false;
			std::uint32_t SelectedLod = 0;
			std::uint32_t PreviousLod = 0;
            std::uint32_t SelectedFirstIndex() const { return SelectedLod?0:Geometry->FirstIndex; }
            std::uint32_t SelectedIndexCount() const
            { return !SelectedLod && Geometry->NativeIndexCount?Geometry->NativeIndexCount:static_cast<std::uint32_t>(SelectedIndices().size()); }
			const std::vector<std::uint32_t>& SelectedIndices() const
			{
				return SelectedLod ? Geometry->LodChain.Levels[SelectedLod].Indices : Geometry->Indices;
			}
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
			TextureDesc Description;
			TextureHandle Texture;
			std::size_t ReferenceCount = 0;

			static TextureHandle Handle(const TextureCacheEntry* entry)
			{
				return entry ? entry->Texture : TextureHandle{};
			}
		};

		bool CreateMeshResources(const MeshData& mesh, MeshResource& resource);
		void DestroyMeshResources(MeshResource& resource);
		bool AcquireGeometry(const MeshData& mesh, MeshGeometryResource*& output);
		void ReleaseGeometry(MeshGeometryResource*& geometry);
		void DestroyGeometryResources(MeshGeometryResource& geometry);
		bool PrepareWireOverlay(const MeshResource& mesh);
		bool EnsureMeshLods(MeshResource& resource);
        bool EnsureGeometryLods(MeshGeometryResource& geometry,const std::string& name);
		bool PrepareMeshLods();
        bool PrepareGeometry();
        bool PrepareVirtualGeometry();
        bool PrepareNaniteGeometry();
        bool PrepareNaniteHostGeometry();
        bool PrepareStreamedGeometry();
        bool PrepareNaniteFrameVisibility(VisibilityBufferDesc&);
        void CommitNaniteFrameVisibility();
        void ReleaseNaniteFrameVisibility();
        struct NaniteGeometryFamily
        {
            std::vector<MeshGeometryResource*> Sources;
            std::vector<std::unique_ptr<MeshGeometryResource>> Variants;
            std::vector<std::size_t> Bindings;
            std::vector<std::shared_ptr<const LightmapSourceGeometry>> LightmapSources;
            NaniteHierarchy Hierarchy;
            NanitePages Pages;
            std::unique_ptr<NaniteAssetRequest> Asset;
            std::unique_ptr<NaniteStream> Stream;
            NaniteStreamSnapshot Published;
            std::vector<std::uint32_t> SelectedClusters;
            std::uint32_t Capacity=0;
            bool AssetReady=false;
        };
        bool PrepareNaniteLightmapSources(NaniteGeometryFamily&);
        NaniteGeometryFamily* NaniteFamilyFor(const MeshResource&,std::size_t& part);
        struct VirtualGeometryFamily
        {
            std::vector<MeshGeometryResource*> Sources;
            std::vector<std::unique_ptr<MeshGeometryResource>> Variants;
            std::vector<Vertex> SourceVertices;
            std::vector<std::vector<std::uint32_t>> SourceIndices;
            std::vector<std::shared_ptr<const LightmapSourceGeometry>> LightmapSources;
            VirtualGeometryHierarchy Hierarchy;
            std::unique_ptr<VirtualGeometryStream> Stream;
            VirtualGeometryStreamSnapshot Published;
            std::vector<std::uint32_t> SelectedClusters;
        };
        bool PrepareVirtualLightmapSources(VirtualGeometryFamily& family);
        struct GeometryFamily
        {
            std::vector<MeshGeometryResource*> Sources;
            std::vector<std::unique_ptr<MeshGeometryResource>> Variants;
            BufferHandle VertexBuffer,IndexBuffer;
        };
        struct GeometryBinding
        {
            std::string Id;
            MeshGeometryResource* Source=nullptr;
            std::size_t Family=0;
            bool operator==(const GeometryBinding& other) const
            { return Id==other.Id && Source==other.Source && Family==other.Family; }
        };
        struct GeometryGeneration
        {
            explicit GeometryGeneration(Renderer* owner):Owner(owner) {}
            ~GeometryGeneration();
            Renderer* Owner;
            Techniques::DisplacementMappingSettings Displacement;
            Techniques::TessellationSettings Tessellation;
            std::vector<GeometryFamily> Families;
            std::vector<GeometryBinding> Bindings;
            std::vector<MeshGeometryResource*> SourceReferences;
            std::uint32_t VertexCount=0,DisplacementDispatches=0,TessellationDispatches=0;
            std::uint64_t Serial=0;
        };
        struct VirtualGeometryGeneration
        {
            explicit VirtualGeometryGeneration(Renderer* owner):Owner(owner) {}
            ~VirtualGeometryGeneration();
            Renderer* Owner;
            std::vector<VirtualGeometryFamily> Families;
            std::vector<GeometryBinding> Bindings;
            std::vector<MeshGeometryResource*> SourceReferences;
            std::uint32_t PageBudget=0;
            std::uint64_t SourceSerial=0,Serial=0;
        };
        struct NaniteGeometryGeneration
        {
            explicit NaniteGeometryGeneration(Renderer* owner):Owner(owner) {}
            ~NaniteGeometryGeneration();
            Renderer* Owner;
            std::vector<NaniteGeometryFamily> Families;
            std::vector<GeometryBinding> Bindings;
            std::vector<MeshGeometryResource*> SourceReferences;
            std::uint32_t PageBudget=0;
            std::uint32_t ScenePageBudget=0;
            NaniteSceneBudget Budget;
            bool BudgetReady=false;
            std::uint64_t SourceSerial=0,Serial=0;
        };
        VirtualGeometryFamily* VirtualFamilyFor(const MeshResource& mesh,std::size_t& part);
		bool AcquireTexture(
			const std::shared_ptr<const ImageData>& image,
			TextureColorSpace colorSpace,
			const std::string& debugName,
			TextureCacheEntry*& texture);
		void ReleaseTexture(TextureCacheEntry*& texture);
		bool WantsMaterialMips() const;
		bool PrepareMaterialTextures();
		TextureHandle UploadMaterialTexture(const TextureDesc& desc, const ImageData& image);
		void ConfigureMaterialPipeline(
			const MeshResource& mesh,
			PipelineState& pipeline) const;
		bool RenderShadowPass();
		bool PrepareCascadedShadows();
        bool PrepareVirtualShadows();
        void DiscardVirtualShadows();
        void CommitVirtualShadows();
        void DestroyVirtualShadows();
        void CaptureVirtualShadowStatistics(FrameStatistics& statistics) const;
        bool PrepareDensityVolume();
		bool PrepareEnvironmentCapture();
        bool PrepareBakedLightProbes();
        bool PrepareLightmaps();
        bool PrepareVoxelGI();
        bool PrepareLumenGI();
        bool PrepareRayScene();
        bool PrepareRefraction();
        bool IsRefractive(const MeshResource& mesh) const;
        bool IsDepthPeeled(const MeshResource& mesh) const;
        bool UsesPrivateTransparency(const MeshResource& mesh) const;
        bool PrepareDepthPeeling();
        bool RenderDepthPeeling();
		bool RenderEnvironmentCapture();
		bool PreparePlanarCapture();
		bool RenderPlanarCapture();
        bool RenderCameraDepthPass(PassType type);
        bool PrepareHairFur();
        HairFurDesc HairView(const Matrix4& projection) const;
		bool RenderColorPass(bool transparentPhase, bool materialInputs = false);
        bool RenderVisibilityGeometry();
		bool RenderVelocityPass();
		bool RenderOutlinePass();
		bool DrawMesh(const MeshResource& mesh, const PipelineState& pipeline);
		bool FlushDrawBatch();
        bool BeginMeshPass(const PassDesc& pass);
        bool NeedsCpuFrustumTest(const MeshResource& mesh) const;
		bool EndMeshPass();
		bool SubmitMesh(const MeshResource& mesh, const PipelineState& pipeline, const Matrix4& viewProjection,
			const MotionTransform& motion, const std::vector<DrawInstance>* instances, const GpuDrivenBatchHandle* gpu = nullptr);
		bool PrepareRenderResolution();
		void ApplyOutlines(FrameCapture& capture) const;
		void UpdateAutomaticViewProjections();
		bool Fail(const char* error);

		std::unique_ptr<IRHI> m_RHI;
		const MeshResource* m_BatchMesh = nullptr;
        struct InstanceDraw { const MeshResource* Mesh; PipelineState Pipeline; };
        std::vector<InstanceDraw> m_InstanceDraws;
        bool m_CollectInstanceDraws=false;
		PipelineState m_BatchPipeline;
		Matrix4 m_BatchViewProjection;
		std::vector<DrawInstance> m_BatchInstances;
        struct VisibilitySourceKey
        {
            const MeshResource* Mesh = nullptr;
            const MeshGeometryResource* Geometry = nullptr;
            Matrix4 World;
            std::uint32_t Lod = 0;
            bool operator==(const VisibilitySourceKey& other) const
            { return Mesh == other.Mesh && Geometry == other.Geometry && World.Values == other.World.Values && Lod == other.Lod; }
        };
        std::vector<VisibilitySourceKey> m_VisibilitySources;
        std::shared_ptr<const VisibilityScene> m_VisibilityScene;
        std::uint64_t m_VisibilitySceneRevision = 0, m_VisibilityGeometryViewSerial = 0;
        struct PreparedMeshDraw
        {
            const MeshResource* Mesh = nullptr;
            PipelineState Pipeline;
            Matrix4 ViewProjection;
            MotionTransform Motion;
            std::vector<DrawInstance> Instances;
            bool GPU = false;
            GpuDrivenBatchHandle Handle;
        };
        std::vector<PreparedMeshDraw> m_PreparedMeshDraws;
        PassDesc m_PreparedMeshPass;
        bool m_CapturingGpuPass = false;
        Techniques::RenderBuffer m_RenderBuffer=Techniques::RenderBuffer::Color;
        double m_VegetationTime=0, m_PreviousVegetationTime=0;
        std::uint64_t m_VegetationRevision=0, m_RayVegetationRevision=0;
        Float4 m_VegetationWind{0,0,0,0}, m_VegetationGust{0,0,0,0};
		std::unordered_map<std::string, MeshResource> m_Meshes;
		std::unordered_multimap<std::uint64_t, std::unique_ptr<MeshGeometryResource>> m_GeometryCache;
        std::unique_ptr<GeometryGeneration> m_PreparedGeometry;
        std::unique_ptr<VirtualGeometryGeneration> m_VirtualGeometry;
        std::unique_ptr<NaniteGeometryGeneration> m_NaniteGeometry,m_PendingNaniteGeometry;
        FrameStatistics m_NaniteStatistics;
        bool m_NanitePending=false,m_NaniteDepthHistoryValid=false;
        NaniteDepthHandle m_NaniteDepthHistory,m_NaniteDepthCandidate;
        NaniteVisibilityHandle m_NaniteMainVisibility;
        Matrix4 m_NaniteHistoryProjection;
        std::uint64_t m_NaniteSerial=0,m_NaniteFrame=0,m_LightmapNaniteSerial=0;
        FrameStatistics m_VirtualStatistics;
        bool m_VirtualPending=false;
        std::uint64_t m_VirtualSerial=0,m_VirtualFrame=0,m_GeometryViewSerial=0,m_RayGeometryViewSerial=0;
        std::uint64_t m_LightmapVirtualSerial=0,m_LightmapGeometryRevision=0;
        std::uint64_t m_GeometrySerial=0;
		std::unordered_map<
			TextureCacheKey,
			TextureCacheEntry,
			TextureCacheKeyHash> m_TextureCache;
        // CreateInfo retains presentation dimensions for the camera and LOD.
        // Native scene targets and raster jitter use the separate input extent.
		RendererCreateInfo m_CreateInfo;
        int m_RenderWidth=0,m_RenderHeight=0;
        int m_PreviousPresentedWidth=0,m_PreviousPresentedHeight=0;
        Float2 m_PreviousPresentedJitter;
        std::uint64_t m_PresentedSerial=0,m_TSRHistoryEpoch=1;
		TextureHandle m_ShadowMap;
        VirtualShadowSceneState m_VirtualShadowScene;
        VirtualShadowSceneUpdate m_VirtualShadowCandidateScene;
        VirtualShadowAtlasResult m_VirtualShadowPublished,m_VirtualShadowCandidate;
        std::uint32_t m_VirtualShadowRequestedPages=0;
        bool m_VirtualShadowFrameActive=false;
		std::array<TextureHandle, MaximumShadowCascades> m_CascadeMaps {};
		CascadedShadowFrame m_CascadeFrame;
		CascadedShadowCamera m_CascadeCamera;
		bool m_CascadeCameraValid = false;
		Matrix4 m_ShadowDrawProjection;
        std::shared_ptr<const DensityVolume> m_DensityVolume;
        DensityVolumeBinding m_DensityBinding;
        std::shared_ptr<const EnvironmentRadiance> m_EnvironmentSource;
		EnvironmentCaptureFrame m_EnvironmentFrame;
		Matrix4 m_EnvironmentDrawProjection;
		PlanarCaptureFrame m_PlanarFrame;
		Matrix4 m_ViewProjection;
		Matrix4 m_RasterViewProjection;
		Matrix4 m_CullingViewProjection;
		TemporalSampling m_TemporalSampling;
		TemporalFrame m_TemporalFrame;
		Matrix4 m_LightViewProjection;
		LightingState m_Lighting;
        PreparedAreaLighting m_AreaLighting;
        LightingState m_RenderLighting;
		Techniques::Settings m_Techniques;
        std::vector<const MeshResource*> m_PeelingMeshes;
        PreparedRefractionGeometry m_RefractionGeometry;
        BakedProbeCache m_BakedProbes;
        std::filesystem::path m_BakedLightingDirectory, m_LightmapDirectory;
        std::optional<Techniques::ShadingModel> m_BakedLightingShadingOverride;
        BakedLightmapCache m_BakedLightmaps;
        VoxelRadianceCache m_VoxelGI;
        std::unique_ptr<LumenRenderCache> m_LumenGI;
        std::unique_ptr<LumenWorldRadianceCache> m_LumenWorldCache;
        std::unique_ptr<LumenGlobalDistanceCache> m_LumenGlobalDistance;
        std::uint64_t m_NextLumenKey=0,m_LumenRevision=0,m_LumenSceneRevision=0,m_LumenGeometrySerial=0;
        HairGeometryCache m_HairGeometry;
        HairGeometryStatus m_HairStatus;
        PreparedHairLighting m_HairLighting;
        std::shared_ptr<const HairStrandGeometry> m_BoundHairGeometry;
        std::vector<HairSourceInstance> m_HairInstances;
        std::vector<std::string> m_HairInstanceIds;
        std::vector<Matrix4> m_HairLastPose;
        std::shared_ptr<const HairStrandGeometry> m_HairPoseBase,m_HairPosedGeometry;
        std::uint64_t m_HairPoseRevision=0,m_HairGroomSourceId=0;
        std::uint64_t m_HairSelectionGroup=0,m_HairSourceRevision=0,m_HairSourceSelection=0;
        bool m_HairSourceSelectedOnly=true,m_HairSourceAlphaBlending=true,m_HairCameraDepthWritten=false;
        VoxelRadianceStatus m_VoxelStatus;
        std::vector<VoxelSourceInstance> m_VoxelInstances;
        std::uint64_t m_VoxelSourceRevision=0;
        std::shared_ptr<const VoxelRadianceVolume> m_BoundVoxelVolume;
        LightmapBakeStatus m_LightmapStatus;
        std::vector<LightmapSourceInstance> m_LightmapInstances;
        std::uint64_t m_LightmapSourceRevision = 0;
        float m_LightmapTraceDistance = 1;
        std::uint64_t m_ProbeSceneRevision = 0;
        std::shared_ptr<const RaySceneUpload> m_RayScene;
        std::shared_ptr<const RaySceneUpload> m_RayShadowImageSource;
        std::shared_ptr<const RayShadowImage> m_RayShadowImage;
        RayShadowFrame m_RayShadowFrame;
        std::shared_ptr<const RaySceneShadingUpload> m_RayShading;
        std::shared_ptr<const SceneLightCollection> m_RayLights;
        LightingState m_RayLighting;
        Techniques::ShadingModel m_RayShadingModel=Techniques::ShadingModel::Phong;
        bool m_RayNormalMapping=false;
        std::uint64_t m_RaySceneRevision = 0;
        std::array<bool,3> m_RaySceneCoverage{};
        std::vector<std::pair<std::string,std::uint32_t>> m_RaySceneLods;
		std::uint32_t m_FrustumCulledMeshCount = 0;
        bool m_GpuPresentation=false;
		FrameCapture m_LastFrameCapture;
		FrameHistory m_FrameHistory;
        std::uint64_t m_PathResetSerial=0;
		MotionFrame m_CurrentMotionFrame;
		std::optional<double> m_NextSampleTimeSeconds;
		std::uint64_t m_NextHistoryEpoch = 0;
		PassType m_CurrentPass = PassType::Color;
		RenderMode m_RenderMode = RenderMode::Solid;
		bool m_ShadowsEnabled = true;
		bool m_AutomaticViewProjection = true;
		bool m_AutomaticLightViewProjection = true;
		bool m_Initialized = false;
		std::string m_LastError;
	};
}
