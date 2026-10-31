#pragma once

#include "IRHI.h"

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
		bool EndPass() override;
		bool Present() override;
		FrameCapture GetLastFrameCapture() const override;
		std::string GetLastError() const override;

	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
