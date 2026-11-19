#pragma once

#include "Types.h"
#include "Renderer/Shadows/CascadedShadowMap.h"
#include "Renderer/Reflections/EnvironmentCapture.h"
#include "Renderer/Reflections/PlanarReflection.h"
#include "Renderer/BakedLighting/LightmapContract.h"
#include "Renderer/Transparency/RefractionContract.h"
#include "Renderer/Transparency/DepthPeelingContract.h"
#include "Renderer/RayTracing/SceneRayUpload.h"
#include "Renderer/RayTracing/RayTracedShadowScene.h"
#include "Renderer/RayTracing/RayTracedReflectionContract.h"
#include "Renderer/RayTracing/HybridRenderingContract.h"
#include "Renderer/Lighting/SubsurfaceScatteringContract.h"
#include "Renderer/Hair/HairFurContract.h"
#include "Renderer/PostProcessing/TemporalEffectsContract.h"
#include "Renderer/RayTracing/RayTracedGIContract.h"
#include "Renderer/VoxelGI/VoxelGlobalIlluminationContract.h"
#include "Renderer/RayTracing/PathTracingContract.h"
#include "Renderer/Geometry/GPUDrivenRendering.h"
#include "Renderer/Pipelines/VisibilityBuffer.h"
#include "Renderer/Geometry/DisplacementGeometry.h"
#include "Renderer/Geometry/PNTessellation.h"
#include "Renderer/VirtualGeometry/VirtualGeometryGpu.h"
#include "Renderer/Nanite/NaniteCache.h"
#include "Renderer/Nanite/NaniteFrameVisibility.h"
#include "Renderer/Nanite/NaniteDraws.h"
#include "Renderer/Nanite/NaniteRaster.h"
#include "Renderer/VirtualShadowMaps/VirtualShadowMaps.h"
#include "Renderer/VirtualShadowMaps/VirtualShadowSampling.h"
#include "Renderer/Lumen/LumenSurfaceCache.h"
#include "Renderer/Lumen/LumenSurfaceLookup.h"
#include "Renderer/Lumen/LumenSurfaceLighting.h"
#include "Renderer/Lumen/LumenProbeGather.h"
#include "Renderer/Lumen/LumenTemporal.h"
#include "Renderer/Lumen/LumenRadianceTrace.h"
#include "Renderer/Lumen/LumenProbeIntegration.h"
#include "Renderer/Lumen/LumenFrameTransaction.h"
#include "Renderer/Lumen/LumenTranslucency.h"
#include "Renderer/Lumen/LumenDistanceScene.h"
#include "Renderer/Lumen/LumenGlobalDistance.h"
#include "Renderer/VirtualGeometry/VirtualGeometryCache.h"
#include "Renderer/PostProcessing/TemporalSuperResolution.h"

#include <memory>
#include <string>
#include <vector>

namespace Evans::Renderer
{
    struct BakedProbeSamples;
	class IRHI
	{
	public:
		virtual ~IRHI() = default;

		virtual bool Initialize(const RHICreateInfo& createInfo) = 0;
		virtual void Shutdown() = 0;
		virtual bool IsInitialized() const = 0;
		virtual RHIBackend GetBackend() const = 0;
		virtual const char* GetName() const = 0;
		virtual RHICapabilities GetCapabilities() const = 0;

		virtual bool Resize(int width, int height) = 0;
		virtual BufferHandle CreateBuffer(const BufferDesc& desc, const void* initialData) = 0;
        // Geometry production is synchronous and only allowed between frames.
        // The caller owns drawable buffers and authoritative GPU readback mirrors.
        // Failure preserves output; no CPU substitute is used for native work.
        virtual bool CreateTessellatedGeometry(const PNTessellationGeometry&,
            const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
        virtual bool CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
        // GPU-only frame operations run between passes. Pending handles can
        // feed same-frame work; Present commits them and AbortFrame invalidates
        // them. Diagnostic reads and explicit release require a closed frame.
        // Prepare between passes, consume once per cluster inside a primary
        // opaque Color/GBuffer pass. Draw handles expire at Present/AbortFrame.
        // Post preparation must supply the exact main draw descriptors.
        virtual bool PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
        // Stable compatible bins compact GPU-visible indices and issue one
        // native indexed command per bin; no visible count is read on the CPU.
        virtual bool PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
        virtual bool DrawNaniteBins(NaniteDrawHandle) { return false; }
        virtual bool DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
        virtual bool CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
        virtual bool BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
        virtual bool RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
        virtual bool ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
        virtual bool ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
        virtual bool DestroyNaniteDepth(NaniteDepthHandle) { return false; }
        virtual bool DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }

        // Closed-frame, owning-thread operations. A successful pending poll leaves
        // output unchanged. A ready poll consumes the task and transfers buffers.
        // Cancellation invalidates the handle immediately; native resources retire
        // only after completion. Shutdown preserves submitted native lifetimes.
        virtual bool SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
        virtual bool PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
        virtual bool CancelNaniteGeometry(NaniteGeometryTask) { return false; }
        virtual bool CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
        // Page copies use a separate bounded task queue. Statistics charge the
        // submission frame; poll publishes a usable cache handle only on completion.
        // The same thread/frame, consumption and cancellation rules apply. Pending
        // polls and failed calls preserve output handles/statistics; cache release
        // drops public ownership while submitted tasks retain the native pool.
        virtual bool SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
        virtual bool PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
        virtual bool CancelNaniteCache(NaniteCacheTask) { return false; }
        virtual bool PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
        virtual bool DestroyNaniteCache(NaniteCacheHandle) { return false; }
        virtual bool CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
        virtual bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,
            VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
        virtual bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
        virtual bool CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }

        virtual bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
        virtual bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
        virtual bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
        virtual bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
        virtual bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
        // Independent native material atlases, prepared between scene frames.
        // Completion precedes handle publication; diagnostic readback is optional.
        virtual bool CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
        virtual bool BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
        virtual bool AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
        virtual bool CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
        virtual bool ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
        virtual bool DestroyLumenSurface(LumenSurfaceHandle) { return false; }
        virtual bool RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
        virtual bool BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
        virtual bool AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
        virtual bool CancelLumenLightingUpdate(LumenLightingTask) { return false; }
        virtual bool ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
        virtual bool DestroyLumenLighting(LumenLightingHandle) { return false; }
        // Trace and integration may append GPU-only work between frame passes.
        // Pending handles can feed later work in that frame; abort invalidates
        // them, successful presentation publishes them, and readback is closed-frame.
        virtual bool TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
        virtual bool AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
        virtual bool ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
        virtual bool ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
        virtual bool DestroyLumenRadiance(LumenRadianceHandle) { return false; }
        virtual bool IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
        virtual bool ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
        virtual bool ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
        virtual bool DestroyLumenProbes(LumenProbeHandle) { return false; }
        virtual bool LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
        virtual bool CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
        virtual bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle={}) { return false; }
        virtual bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
        virtual bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
        virtual bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
        virtual bool TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
        virtual bool DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }

        virtual bool DestroyTSRHistory(TSRHistoryHandle) { return false; }
        // Closed-frame image presentation, without publishing scene history or
        // changing the window/last scene capture. Output owns display-size bytes.
        virtual bool RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
        void SetTSRSceneFrame(const TSRSceneFrame& frame) { m_TSRSceneFrame=frame; }
        virtual bool SnapshotTSROpaque() { return false; }
        virtual bool CreateDisplacedVertexBuffer(const DisplacementGeometry&,
            const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
		virtual bool UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size) = 0;
		virtual bool DestroyBuffer(BufferHandle buffer) = 0;
		virtual TextureHandle CreateTexture(const TextureDesc& desc, const void* initialData) = 0;
		virtual bool DestroyTexture(TextureHandle texture) = 0;

		virtual bool BeginFrame() = 0;
		// Resolved frame settings are copied before command recording. Native
		// post passes execute before readback and never mutate editor requests.
		void SetTechniqueSettings(const Techniques::Settings& settings)
		{
			m_TechniqueSettings = ResolveTechniqueSettings(settings).Effective;
		}
        void SetBakedProbeSamples(std::shared_ptr<const BakedProbeSamples> samples) { m_BakedProbeSamples = std::move(samples); }
        void SetLightmapFrame(LightmapFrame frame) { m_LightmapFrame = std::move(frame); }
        void SetRefractionFrame(RefractionFrame frame) { m_RefractionFrame = std::move(frame); }
        void SetDepthPeelingFrame(DepthPeelingFrame frame) { m_DepthPeelingFrame = std::move(frame); }
        void SetRayScene(std::shared_ptr<const RaySceneUpload> scene) { m_RayScene = std::move(scene); }
        void SetRayShadowImage(std::shared_ptr<const RayShadowImage> image) { m_RayShadowImage=std::move(image); }
        const std::shared_ptr<const RayShadowImage>& GetRayShadowImage() const noexcept { return m_RayShadowImage; }
        void SetRayRadianceFrame(RayRadianceFrame frame) { m_RayRadianceFrame = std::move(frame); }
        void SetLumenGlobalDistance(LumenDistanceSceneHandle scene) { m_LumenGlobalDistance=scene; }
        LumenDistanceSceneHandle GetLumenGlobalDistance() const { return m_LumenGlobalDistance; }
        void SetLumenWorldCache(LumenWorldCacheBinding cache) { m_LumenWorldCache=std::move(cache); }
        const LumenWorldCacheBinding& GetLumenWorldCache() const { return m_LumenWorldCache; }
        void SetLumenLighting(LumenLightingHandle lighting) { m_LumenLighting=lighting; }
        LumenLightingHandle GetLumenLighting() const { return m_LumenLighting; }
        void SetVoxelGIFrame(VoxelGIFrame frame) { m_VoxelGIFrame=std::move(frame); }
        void SetHairFurFrame(HairFurFrame frame) { m_HairFurFrame=std::move(frame); }
		[[nodiscard]] Techniques::Resolution ResolveTechniqueSettings(const Techniques::Settings& requested) const
		{
			const bool initialized=IsInitialized();
            const auto capabilities=initialized?GetCapabilities():RHICapabilities{};
            return Techniques::ResolveSettingsForDevice(requested, TechniqueBackend(GetBackend()),
				!initialized || capabilities.DeferredShading,
				!initialized || capabilities.ScalableAmbientObscurance,
				!initialized ? Techniques::AllMultisampleCounts : capabilities.SupportedSampleCounts,
				!initialized || capabilities.InstancedRendering,
				!initialized || capabilities.TemporalAntiAliasing,
				!initialized || capabilities.VolumetricFog,
				!initialized ? Techniques::AllMultisampleCounts : capabilities.WeightedTransparencySampleCounts,
				!initialized || capabilities.VolumetricLighting,
				!initialized || capabilities.LightShafts,
                !initialized || capabilities.VarianceShadowMaps,
                !initialized || capabilities.ContactShadows,
                !initialized || capabilities.ForwardPlusRendering,
                !initialized || capabilities.ClusteredForwardRendering,
                !initialized || capabilities.TiledDeferredRendering,
                !initialized || capabilities.ScreenSpaceGlobalIllumination,
                !initialized || capabilities.BakedLightProbes,
                !initialized || capabilities.BakedLightmaps,
                !initialized || capabilities.Refraction,
                !initialized || capabilities.DepthPeeling,
                !initialized || capabilities.RayTracedAmbientOcclusion,
                !initialized || capabilities.RayTracedReflections,
                !initialized || capabilities.RayTracedGI,
                !initialized || capabilities.PathTracing,
                !initialized || capabilities.GPUDrivenRendering,
                !initialized || capabilities.VisibilityBufferRendering,
                !initialized || capabilities.HybridRendering,
                !initialized || capabilities.VoxelGI,
                !initialized || capabilities.TemporalEffects,
                !initialized || capabilities.ParallaxOcclusion,
                !initialized || capabilities.GeometryDisplacement,
                !initialized || capabilities.GeometryTessellation,
                !initialized || capabilities.GeometryVirtualization,
                !initialized || capabilities.TemporalSuperResolution,
                !initialized || capabilities.VirtualShadowMaps,
                !initialized || capabilities.LumenGI,
                !initialized || capabilities.NaniteRendering,
                !initialized || capabilities.MeshShaders,
                !initialized || capabilities.NaniteHardwareScheduling,
                !initialized || capabilities.LumenHardwareRayTracing);
		}
		// The renderer requests depth when a CPU presentation effect needs visibility.
		void SetDepthCaptureRequired(bool required) noexcept { m_DepthCaptureRequired = required; }
        void SetGpuPresentationEnabled(bool enabled) noexcept { m_GpuPresentationEnabled = enabled; }
		// Per-draw rigid-motion input, supplied independently of pipeline state.
		void SetMotionTransform(const MotionTransform& transform) noexcept { m_MotionTransform = transform; }
		void SetCascadedShadowState(const CascadedShadowState& state) { m_CascadedShadows = state; }
        void SetVirtualShadowSamplingFrame(const VirtualShadowSamplingFrame& frame) { m_VirtualShadowSampling=frame; }
		// Captures use a frozen probe and six private depth-tested HDR faces.
		void SetEnvironmentCaptureFrame(const EnvironmentCaptureFrame& frame) { m_EnvironmentCapture = frame; }
		virtual bool CompleteEnvironmentCapture() { return false; }
		void SetPlanarCaptureFrame(const PlanarCaptureFrame& frame) { m_PlanarCapture = frame; }
		virtual bool CompletePlanarCapture() { return false; }
		virtual void AbortFrame() {}
		virtual bool BeginPass(const PassDesc& desc) = 0;
		virtual bool BindPipeline(const PipelineState& pipeline) = 0;
		virtual bool BindWorldTransform(const Matrix4& worldTransform) = 0;
		virtual bool BindViewProjection(const Matrix4& viewProjection) = 0;
		virtual bool BindLightViewProjection(const Matrix4& lightViewProjection) = 0;
		virtual bool BindVertexBuffer(BufferHandle vertexBuffer) = 0;
		virtual bool BindIndexBuffer(BufferHandle indexBuffer) = 0;
		virtual bool DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex) = 0;
		virtual bool DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t,
			const std::vector<DrawInstance>&) { return false; }
		virtual bool EndPass() = 0;
        // Prepare outside a graphics pass; consume immutable GPU-written instance
        // and indexed-command buffers inside the matching pass. No count readback.
        virtual bool PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
        virtual bool DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
        // One opaque geometry operation: depth-tested integer IDs/barycentrics,
        // followed by material reconstruction into the ordinary G-buffer.
        virtual bool RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
		// Call only when DeferredShading is advertised. Implementations bind their
		// own G-buffer and depth inputs; no mesh is replayed during this light pass.
        // DirectDiffuse selects only ordinary deferred light integrals; tiled
        // lighting rejects that output until its own producer is implemented.
		virtual bool DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
        virtual bool ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
        // Depth-tested fiber color and receiver marker, after opaque lighting
        // and before the opaque MSAA resolve or any screen-space effect.
        virtual bool DrawHairFur(const HairFurDesc&) { return false; }
        // Last geometry draw inside a camera-depth, shadow or reflection-capture
        // pass. Camera depth and the later primary surface must share one view.
        virtual bool DrawHairFurInPass(const HairFurDesc&) { return false; }
        // Tail of the ordinary read-only velocity pass, using presented strand
        // correspondence and both current/previous camera-facing ribbon poses.
        virtual bool DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
        virtual bool GenerateContactShadows(const ContactShadowDesc&) { return false; }
        // Prepare complete light storage before captures. Tiled assignment reads
        // original opaque depth; clustered assignment depends only on the view.
        // Both precede material list reads and share immutable scene-light order.
        virtual bool PrepareLightListFrame(const LightListFrameDesc&) { return false; }
        virtual bool GenerateLightLists() { return false; }
		// Opaque material inputs precede this stage in either pipeline. Only ambient
		// radiance is attenuated, before transparent forward draws and tone mapping.
		virtual bool ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
        // Diffuse indirect lighting reads pre-GI HDR and original material data.
        // SSGI uses velocity and private history; probes use cached SH/moments.
        // History/projection publication is deferred until successful present.
        virtual bool ApplyPathTracing(const PathTracingDesc&) { return false; }
        virtual bool ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
		// Reflect opaque HDR/material inputs after AO/GI and before blur.
		virtual bool ApplyReflections(const ReflectionDesc&) { return false; }
		// Opaque HDR color and depth precede the lens gather. Sorted transparent
		// objects remain sharp and are composited afterward, before the HDR resolve.
		virtual bool ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
		virtual bool ApplyMotionBlur(const MotionBlurDesc&) { return false; }
		virtual bool ApplyTemporalAA(const TemporalAADesc&) { return false; }
        // Meter final scene HDR after transparency, before bloom/grading/tone resolve.
        virtual bool ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
		virtual bool ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
		virtual bool ApplyLightShafts(const LightShaftDesc&) { return false; }
		virtual bool ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
		virtual bool ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
        virtual bool ProcessRefraction(const RefractionDesc&) { return false; }
        virtual bool ProcessDepthPeeling(DepthPeelingPass) { return false; }
		virtual bool Present() = 0;
        virtual void ReleasePresentationContext() noexcept {}

		virtual FrameCapture GetLastFrameCapture() const = 0;
		// Transfers a completed capture when supported. The RHI capture may be
		// empty until the next successful presentation.
		virtual FrameCapture TakeLastFrameCapture() { return GetLastFrameCapture(); }
		virtual std::string GetLastError() const = 0;

        void SetDiagnosticBuffer(Techniques::RenderBuffer value) noexcept { m_DiagnosticBuffer=value; }
	protected:
        const TSRSceneFrame& GetTSRSceneFrame() const noexcept { return m_TSRSceneFrame; }
        const std::shared_ptr<const BakedProbeSamples>& GetBakedProbeSamples() const noexcept { return m_BakedProbeSamples; }
        const LightmapFrame& GetLightmapFrame() const noexcept { return m_LightmapFrame; }
        const RefractionFrame& GetRefractionFrame() const noexcept { return m_RefractionFrame; }
        const DepthPeelingFrame& GetDepthPeelingFrame() const noexcept { return m_DepthPeelingFrame; }
        const std::shared_ptr<const RaySceneUpload>& GetRayScene() const noexcept { return m_RayScene; }
        const RayRadianceFrame& GetRayRadianceFrame() const noexcept { return m_RayRadianceFrame; }
        const VoxelGIFrame& GetVoxelGIFrame() const noexcept { return m_VoxelGIFrame; }
        const HairFurFrame& GetHairFurFrame() const noexcept { return m_HairFurFrame; }

        Techniques::RenderBuffer GetDiagnosticBuffer() const noexcept { return m_DiagnosticBuffer; }
        bool IsDepthCaptureRequired() const noexcept { return m_DepthCaptureRequired; }
        bool IsGpuPresentationEnabled() const noexcept { return m_GpuPresentationEnabled; }
		const Techniques::Settings& GetTechniqueSettings() const noexcept { return m_TechniqueSettings; }
		const MotionTransform& GetMotionTransform() const noexcept { return m_MotionTransform; }
		void ResetMotionTransform() noexcept { m_MotionTransform = {}; }
		const CascadedShadowState& GetCascadedShadowState() const noexcept { return m_CascadedShadows; }
        const VirtualShadowSamplingFrame& GetVirtualShadowSamplingFrame() const noexcept { return m_VirtualShadowSampling; }

		const EnvironmentCaptureFrame& GetEnvironmentCaptureFrame() const noexcept { return m_EnvironmentCapture; }
		const PlanarCaptureFrame& GetPlanarCaptureFrame() const noexcept { return m_PlanarCapture; }

	private:
        TSRSceneFrame m_TSRSceneFrame;
        std::shared_ptr<const BakedProbeSamples> m_BakedProbeSamples;
        LightmapFrame m_LightmapFrame;
        RefractionFrame m_RefractionFrame;
        DepthPeelingFrame m_DepthPeelingFrame;
        std::shared_ptr<const RaySceneUpload> m_RayScene;
        std::shared_ptr<const RayShadowImage> m_RayShadowImage;
        RayRadianceFrame m_RayRadianceFrame;
        VoxelGIFrame m_VoxelGIFrame;
        LumenLightingHandle m_LumenLighting;
        LumenWorldCacheBinding m_LumenWorldCache;
        LumenDistanceSceneHandle m_LumenGlobalDistance;
        HairFurFrame m_HairFurFrame;
		bool m_DepthCaptureRequired = false;
        bool m_GpuPresentationEnabled = false;
        Techniques::RenderBuffer m_DiagnosticBuffer=Techniques::RenderBuffer::Color;
		Techniques::Settings m_TechniqueSettings;
		MotionTransform m_MotionTransform;
		CascadedShadowState m_CascadedShadows;
        VirtualShadowSamplingFrame m_VirtualShadowSampling;
		EnvironmentCaptureFrame m_EnvironmentCapture;
		PlanarCaptureFrame m_PlanarCapture;
	};

	std::unique_ptr<IRHI> CreateRHI(RHIBackend backend);
	std::vector<RHIBackend> EnumerateRHIBackends();
	const char* ToString(RHIBackend backend);
	bool TryParseRHIBackend(const std::string& name, RHIBackend& backend);
}
