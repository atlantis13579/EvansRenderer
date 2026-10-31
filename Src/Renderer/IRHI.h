#pragma once

#include "Types.h"

#include <memory>
#include <string>
#include <vector>

namespace Evans::Renderer
{
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
		virtual bool UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size) = 0;
		virtual bool DestroyBuffer(BufferHandle buffer) = 0;
		virtual TextureHandle CreateTexture(const TextureDesc& desc, const void* initialData) = 0;
		virtual bool DestroyTexture(TextureHandle texture) = 0;

		virtual bool BeginFrame() = 0;
		// The renderer requests depth when a CPU presentation effect needs visibility.
		void SetDepthCaptureRequired(bool required) noexcept { m_DepthCaptureRequired = required; }
		virtual void AbortFrame() {}
		virtual bool BeginPass(const PassDesc& desc) = 0;
		virtual bool BindPipeline(const PipelineState& pipeline) = 0;
		virtual bool BindWorldTransform(const Matrix4& worldTransform) = 0;
		virtual bool BindViewProjection(const Matrix4& viewProjection) = 0;
		virtual bool BindLightViewProjection(const Matrix4& lightViewProjection) = 0;
		virtual bool BindVertexBuffer(BufferHandle vertexBuffer) = 0;
		virtual bool BindIndexBuffer(BufferHandle indexBuffer) = 0;
		virtual bool DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex) = 0;
		virtual bool EndPass() = 0;
		virtual bool Present() = 0;

		virtual FrameCapture GetLastFrameCapture() const = 0;
		virtual std::string GetLastError() const = 0;

	protected:
		bool IsDepthCaptureRequired() const noexcept { return m_DepthCaptureRequired; }

	private:
		bool m_DepthCaptureRequired = false;
	};

	std::unique_ptr<IRHI> CreateRHI(RHIBackend backend);
	std::vector<RHIBackend> EnumerateRHIBackends();
	const char* ToString(RHIBackend backend);
	bool TryParseRHIBackend(const std::string& name, RHIBackend& backend);
}
