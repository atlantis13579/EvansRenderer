#pragma once

#include "Renderer/IRHI.h"

#include <memory>

namespace Evans::Renderer
{
	class DX12RHI final : public IRHI
	{
	public:
		DX12RHI();
		~DX12RHI() override;
		DX12RHI(const DX12RHI&) = delete;
		DX12RHI& operator=(const DX12RHI&) = delete;

		bool Initialize(const RHICreateInfo& createInfo) override;
		void Shutdown() override;
		bool IsInitialized() const override;
		RHIBackend GetBackend() const override;
		const char* GetName() const override;
		RHICapabilities GetCapabilities() const override;
		bool Resize(int width, int height) override;
		BufferHandle CreateBuffer(const BufferDesc& desc, const void* initialData) override;
		bool CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) override;
        bool PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) override;
        bool PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) override;
        bool DrawNaniteBins(NaniteDrawHandle) override;
        bool DrawNaniteCluster(NaniteDrawHandle,unsigned) override;
        bool CaptureNaniteDepth(NaniteDepthHandle&) override;
        bool BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) override;
        bool RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) override;
        bool ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) override;
        bool ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) override;
        bool DestroyNaniteDepth(NaniteDepthHandle) override;
        bool DestroyNaniteVisibility(NaniteVisibilityHandle) override;
        bool SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) override;
        bool PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) override;
        bool CancelNaniteGeometry(NaniteGeometryTask) override;
        bool CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) override;
        bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) override;
        bool SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) override;
        bool PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) override;
        bool CancelNaniteCache(NaniteCacheTask) override;
        bool PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) override;
        bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) override;
        bool DestroyNaniteCache(NaniteCacheHandle) override;
        bool CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) override;
        bool CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) override;
        bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) override;
        bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) override;
        bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) override;
        bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) override;
        bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) override;
        bool CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) override;
        bool BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) override;
        bool AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool&,LumenSurfaceResult&,LumenCaptureStatistics&) override;
        bool CancelLumenCaptureUpdate(LumenCaptureTask) override;
        bool ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) override;
        bool DestroyLumenSurface(LumenSurfaceHandle) override;
        bool RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) override;
        bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task) override;
        bool AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics) override;
        bool CancelLumenLightingUpdate(LumenLightingTask task) override;
        bool ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) override;
        bool DestroyLumenLighting(LumenLightingHandle) override;
        bool TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) override;
        bool AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) override;
        bool ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) override;
        bool ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) override;
        bool DestroyLumenRadiance(LumenRadianceHandle) override;
        bool IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) override;
        bool ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) override;
        bool ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) override;
        bool DestroyLumenProbes(LumenProbeHandle) override;
        bool LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) override;
        bool CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) override;
        bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle={}) override;
        bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task) override;
        bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics) override;
        bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task) override;
        bool TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) override;
        bool DestroyLumenDistanceScene(LumenDistanceSceneHandle) override;
        bool DestroyTSRHistory(TSRHistoryHandle) override;
        bool RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) override;
        bool SnapshotTSROpaque() override;
		bool CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) override;
		bool CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) override;
		bool UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size) override;
		bool DestroyBuffer(BufferHandle buffer) override;
		TextureHandle CreateTexture(const TextureDesc& desc, const void* initialData) override;
		bool DestroyTexture(TextureHandle texture) override;
		bool BeginFrame() override;
		void AbortFrame() override;
		bool BeginPass(const PassDesc& desc) override;
		bool BindPipeline(const PipelineState& pipeline) override;
		bool BindWorldTransform(const Matrix4& worldTransform) override;
		bool BindViewProjection(const Matrix4& viewProjection) override;
		bool BindLightViewProjection(const Matrix4& lightViewProjection) override;
		bool BindVertexBuffer(BufferHandle vertexBuffer) override;
		bool BindIndexBuffer(BufferHandle indexBuffer) override;
		bool DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex) override;
		bool DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
			const std::vector<DrawInstance>& instances) override;
        bool PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) override;
        bool DrawGpuDrivenBatch(GpuDrivenBatchHandle) override;
        bool RenderVisibilityBuffer(const VisibilityBufferDesc&) override;
		bool EndPass() override;
		bool DrawDeferredLighting(const DeferredLightingDesc& desc) override;
        bool ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc) override;
        bool DrawHairFur(const HairFurDesc&) override;
        bool DrawHairFurInPass(const HairFurDesc&) override;
        bool DrawHairFurVelocity(const HairFurVelocityDesc&) override;
        bool GenerateContactShadows(const ContactShadowDesc& desc) override;
        bool PrepareLightListFrame(const LightListFrameDesc& desc) override;
        bool GenerateLightLists() override;
		bool ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc) override;
		bool CompleteEnvironmentCapture() override;
		bool CompletePlanarCapture() override;
		bool ApplyPathTracing(const PathTracingDesc& desc) override;
        bool ApplyGlobalIllumination(const GlobalIlluminationDesc& desc) override;
        bool ApplyReflections(const ReflectionDesc& desc) override;
		bool ApplyDepthOfField(const DepthOfFieldDesc& desc) override;
		bool ApplyMotionBlur(const MotionBlurDesc& desc) override;
		bool ApplyTemporalAA(const TemporalAADesc& desc) override;
        bool ApplyTemporalEffects(const TemporalEffectsDesc& desc) override;
		bool ApplyVolumetricFog(const VolumetricFogDesc& desc) override;
		bool ApplyLightShafts(const LightShaftDesc& desc) override;
		bool ProcessMultisampling(Techniques::MultisamplePassKind pass) override;
		bool ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass) override;
        bool ProcessRefraction(const RefractionDesc& desc) override;
        bool ProcessDepthPeeling(DepthPeelingPass pass) override;
		bool Present() override;
		FrameCapture GetLastFrameCapture() const override;
		FrameCapture TakeLastFrameCapture() override;
		std::string GetLastError() const override;

	private:
		bool DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
			const std::vector<DrawInstance>* instances);
		RHICapabilities QueryDeviceCapabilities() const;
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
