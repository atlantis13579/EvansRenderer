#include "Renderer/IRHI.h"

#include "Renderer/DX11/DX11RHI.h"
#include "Renderer/DX12/DX12RHI.h"
#include "Renderer/Metal/MetalRHI.h"
#include "Renderer/OpenGLES/OpenGLESRHI.h"
#include "Renderer/Vulkan/VulkanRHI.h"

#include <algorithm>
#include <cctype>

namespace Evans::Renderer
{
	std::unique_ptr<IRHI> CreateRHI(RHIBackend backend)
	{
		switch (backend)
		{
		case RHIBackend::DX11: return std::make_unique<DX11RHI>();
		case RHIBackend::DX12: return std::make_unique<DX12RHI>();
		case RHIBackend::Metal: return std::make_unique<MetalRHI>();
		case RHIBackend::OpenGLES: return std::make_unique<OpenGLESRHI>();
		case RHIBackend::Vulkan: return std::make_unique<VulkanRHI>();
		default: return nullptr;
		}
	}

	std::vector<RHIBackend> EnumerateRHIBackends()
	{
#if defined(_WIN32)
		return { RHIBackend::DX11, RHIBackend::DX12,
			RHIBackend::OpenGLES, RHIBackend::Vulkan };
#elif defined(__APPLE__)
		return { RHIBackend::Metal };
#else
		return {};
#endif
	}

	const char* ToString(RHIBackend backend)
	{
		switch (backend)
		{
		case RHIBackend::DX11: return "DX11";
		case RHIBackend::DX12: return "DX12";
		case RHIBackend::Metal: return "Metal";
		case RHIBackend::OpenGLES: return "OpenGLES";
		case RHIBackend::Vulkan: return "Vulkan";
		default: return "Unknown";
		}
	}

	bool TryParseRHIBackend(const std::string& name, RHIBackend& backend)
	{
		std::string lower = name;
		std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char value) {
			return static_cast<char>(std::tolower(value));
		});
		if (lower == "dx11" || lower == "directx11" || lower == "directx 11") backend = RHIBackend::DX11;
		else if (lower == "dx12" || lower == "directx12" || lower == "directx 12") backend = RHIBackend::DX12;
		else if (lower == "metal") backend = RHIBackend::Metal;
		else if (lower == "opengles" || lower == "open gl es" || lower == "gles") backend = RHIBackend::OpenGLES;
		else if (lower == "vulkan") backend = RHIBackend::Vulkan;
		else return false;
		return true;
	}
}
