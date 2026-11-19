#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/D3DShaderCache.h"
#include "Renderer/Lumen/LumenHairReceiver.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Nanite/NaniteAsync.h"
#include "Renderer/MaterialTechnique.h"
#include "Renderer/Shadows/PCFContract.h"
#include "Renderer/Lumen/LumenGlobalDistanceAsync.h"
#include "Renderer/Lumen/LumenLightingAsync.h"
#include "Renderer/Lumen/LumenCaptureAsync.h"
#include "Renderer/Nanite/NaniteMaterials.h"
#include "Renderer/Nanite/NaniteInstances.h"
#include "Renderer/Nanite/NaniteHardware.h"
#include "Renderer/DX11/DX11RHI.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/Geometry/Instancing.h"
#include "Renderer/DX11/DX11Multisampling.h"
#include "Renderer/DX11/DX11TemporalAA.h"
#include "Renderer/DX11/DX11GlobalIllumination.h"
#include "Renderer/DX11/DX11LightProbes.h"
#include "Renderer/DX11/DX11Lightmaps.h"
#include "Renderer/DX11/DX11Refraction.h"
#include "Renderer/DX11/DX11DepthPeeling.h"
#include "Renderer/DX11/DX11VolumetricFog.h"
#include "Renderer/DX11/DX11VolumetricLighting.h"
#include "Renderer/DX11/DX11LightShafts.h"
#include "Renderer/DX11/DX11WeightedTransparency.h"
#include "Renderer/DX11/DX11PCSS.h"
#include "Renderer/DX11/DX11VSM.h"
#include "Renderer/DX11/DX11ContactShadows.h"
#include "Renderer/DX11/DX11ForwardPlus.h"
#include "Renderer/DX11/DX11ClusteredForward.h"
#include "Renderer/Lighting/LightListFrame.h"
#include "Renderer/Pipelines/TiledDeferredContract.h"
#include "Renderer/TechniqueShader.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/PostProcessing/PostProcessContract.h"
#include "Renderer/PostProcessing/HdrContract.h"
#include "Renderer/Pipelines/DeferredContract.h"
#include "Renderer/RayTracing/RayTracedAOContract.h"
#include "Renderer/PostProcessing/DepthOfFieldContract.h"
#include "Renderer/Reflections/ReflectionContract.h"
#include "Renderer/Reflections/CubemapReflectionContract.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/Reflections/EnvironmentMapReflectionContract.h"
#include "Renderer/Reflections/PlanarReflectionContract.h"
#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#endif

namespace Evans::Renderer
{
#if defined(_WIN32)
	using Microsoft::WRL::ComPtr;

	namespace
	{
		constexpr float MaximumHalfFloatRadiance = 65504.0f;
		// Shader algorithms are independently maintained in Src/Shaders/DX11.

		std::string HResultMessage(const char* operation, HRESULT result)
		{
			std::ostringstream stream;
			stream << operation << " failed with HRESULT 0x"
				<< std::hex << std::uppercase << static_cast<unsigned long>(result);
			return stream.str();
		}

		float LinearToSRGBChannel(float value)
		{
			value = std::max(value, 0.0f);
			return value <= 0.0031308f
				? value * 12.92f
				: 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
		}

		float SRGBToLinearChannel(float value)
		{
			value = std::clamp(value, 0.0f, 1.0f);
			return value <= 0.04045f
				? value / 12.92f
				: std::pow((value + 0.055f) / 1.055f, 2.4f);
		}

		using DrawConstants = RealtimeDrawConstants;

	}

	struct DX11RHI::Impl
	{
		struct HdrSurface
		{
			ComPtr<ID3D11Texture2D> Texture;
			ComPtr<ID3D11RenderTargetView> Target;
			ComPtr<ID3D11ShaderResourceView> View;
			int Width = 0;
			int Height = 0;
		};
		struct DepthHierarchy
		{
			ComPtr<ID3D11Texture2D> Texture, Scratch;
			ComPtr<ID3D11ShaderResourceView> View;
			ComPtr<ID3D11RenderTargetView> ScratchTarget;
			int Width = 0, Height = 0;
			UINT MipCount = 0;
		};
		static constexpr std::size_t HdrScene = 0;
		static constexpr std::size_t HdrScratch = 1;
		static constexpr std::size_t BloomFirst = 2;
		static constexpr std::size_t BloomSecond = 3;
		static constexpr std::size_t HdrSurfaceCount = 4;
		static constexpr std::size_t HdrProgramCount = HdrTechniqueCount;
		struct BufferResource
		{
			BufferDesc Description;
			bool NaniteGeometry=false;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D11Buffer> Buffer;
		};

		struct TextureResource
		{
			TextureDesc Description;
			ComPtr<ID3D11Texture2D> Texture;
			ComPtr<ID3D11RenderTargetView> RenderTargetView;
			ComPtr<ID3D11DepthStencilView> DepthStencilView;
			ComPtr<ID3D11ShaderResourceView> ShaderResourceView;
			ComPtr<ID3D11ShaderResourceView> SRGBShaderResourceView;
			ComPtr<ID3D11Texture2D> StagingTexture;
		};

		bool Fail(const std::string& error)
		{
			LastError = error;
			return false;
		}

		bool CheckThread(const char* operation)
		{
			if (!Initialized) return Fail("DX11 RHI is not initialized");
			if (OwnerThread != std::this_thread::get_id())
			{
				return Fail(std::string(operation) + " must execute on the owning DX11 RHI thread");
			}
			return true;
		}

		bool EnsureInstancedVertex(std::size_t program)
		{
			if (!InstanceVertices[program])
			{
				if (!LoadTechniqueShaderSource("DX11", "InstancedRendering.hlsl", ShaderCode, LastError)) return false;
				constexpr const char* entries[] {"VSForward", "VSGouraud", "VSDeferred", "VSShadow", "VSVelocity"};
				ComPtr<ID3DBlob> bytecode;
				if (!CompileShader(entries[program], "vs_5_0", bytecode)) return false;
				const HRESULT result = Device->CreateVertexShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
					nullptr, InstanceVertices[program].ReleaseAndGetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(instancing)", result));
			}
			if (!InstanceBuffer)
			{
				D3D11_BUFFER_DESC desc {};
				desc.ByteWidth = sizeof(InstanceConstantBlock); desc.Usage = D3D11_USAGE_DYNAMIC;
				desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				const HRESULT result = Device->CreateBuffer(&desc, nullptr, InstanceBuffer.ReleaseAndGetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(instancing)", result));
			}
			return true;
		}

		bool CompileShader(const char* entryPoint, const char* target, ComPtr<ID3DBlob>& bytecode, const D3D_SHADER_MACRO* defines = nullptr, bool strictArithmetic = false)
		{
			UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
            if (strictArithmetic) flags |= D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_WARNINGS_ARE_ERRORS;
#if defined(_DEBUG)
			flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
			ComPtr<ID3DBlob> errors;
			const HRESULT result = Evans::Renderer::CompileCachedD3DShader(ShaderCode.data(), ShaderCode.size(), "EvansDX11.hlsl",
				defines, nullptr, entryPoint, target, flags, 0, bytecode.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors)
				{
					LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				}
				else
				{
					LastError = HResultMessage("D3DCompile", result);
				}
				return false;
			}
			return true;
		}

		bool CreateDepthTexture(int width, int height, bool shaderResource,
			ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11DepthStencilView>& depthView,
			ComPtr<ID3D11ShaderResourceView>& shaderView, ComPtr<ID3D11Texture2D>& staging)
		{
			D3D11_TEXTURE2D_DESC desc = {};
			desc.Width = static_cast<UINT>(width);
			desc.Height = static_cast<UINT>(height);
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R32_TYPELESS;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | (shaderResource ? D3D11_BIND_SHADER_RESOURCE : 0);
			HRESULT result = Device->CreateTexture2D(&desc, nullptr, texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(depth)", result));

			D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
			dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			result = Device->CreateDepthStencilView(texture.Get(), &dsvDesc, depthView.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilView", result));
			if (shaderResource)
			{
				D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
				srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				srvDesc.Texture2D.MipLevels = 1;
				result = Device->CreateShaderResourceView(texture.Get(), &srvDesc, shaderView.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateShaderResourceView(depth)", result));
			}

			desc.Usage = D3D11_USAGE_STAGING;
			desc.BindFlags = 0;
			desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(&desc, nullptr, staging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(depth staging)", result));
			return true;
		}

		bool CreateOffscreenTargets(int width, int height)
		{
			D3D11_TEXTURE2D_DESC colorDesc = {};
			colorDesc.Width = static_cast<UINT>(width);
			colorDesc.Height = static_cast<UINT>(height);
			colorDesc.MipLevels = 1;
			colorDesc.ArraySize = 1;
			colorDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			colorDesc.SampleDesc.Count = 1;
			colorDesc.Usage = D3D11_USAGE_DEFAULT;
			colorDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			ComPtr<ID3D11Texture2D> colorTexture;
			ComPtr<ID3D11RenderTargetView> colorTarget;
			ComPtr<ID3D11Texture2D> colorStaging;
			HRESULT result = Device->CreateTexture2D(&colorDesc, nullptr, colorTexture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(color)", result));
			D3D11_RENDER_TARGET_VIEW_DESC colorTargetDesc = {};
			colorTargetDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			colorTargetDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			result = Device->CreateRenderTargetView(
				colorTexture.Get(), &colorTargetDesc, colorTarget.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRenderTargetView", result));
			colorDesc.Usage = D3D11_USAGE_STAGING;
			colorDesc.BindFlags = 0;
			colorDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(&colorDesc, nullptr, colorStaging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(color staging)", result));

			colorDesc.Usage = D3D11_USAGE_DEFAULT;
			colorDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			colorDesc.CPUAccessFlags = 0;
			colorDesc.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
			ComPtr<ID3D11Texture2D> outlineMaskTexture;
			ComPtr<ID3D11RenderTargetView> outlineMaskTarget;
			ComPtr<ID3D11Texture2D> outlineMaskStaging;
			result = Device->CreateTexture2D(&colorDesc, nullptr, outlineMaskTexture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(outline mask)", result));
			result = Device->CreateRenderTargetView(
				outlineMaskTexture.Get(), nullptr, outlineMaskTarget.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateRenderTargetView(outline mask)", result));
			colorDesc.Usage = D3D11_USAGE_STAGING;
			colorDesc.BindFlags = 0;
			colorDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(
				&colorDesc, nullptr, outlineMaskStaging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateTexture2D(outline mask staging)", result));

			ComPtr<ID3D11Texture2D> depthTexture;
			ComPtr<ID3D11DepthStencilView> depthTarget;
			ComPtr<ID3D11Texture2D> depthStaging;
			ComPtr<ID3D11ShaderResourceView> depthShaderView;
			if (!CreateDepthTexture(width, height, true, depthTexture, depthTarget,
				depthShaderView, depthStaging)) return false;

			ColorTexture = std::move(colorTexture);
			ColorTarget = std::move(colorTarget);
			ColorStaging = std::move(colorStaging);
			SceneColorView.Reset();
			PostTexture.Reset();
			PostTarget.Reset();
			PostView.Reset();
			for (auto& surface : HdrSurfaces) surface = {};
			for (auto& surface : GBufferSurfaces) surface = {};
			for (auto& surface : AmbientOcclusionSurfaces) surface = {};
			ReflectionSurface = {};
			for (auto& surface : MotionSurfaces) surface = {};
			TemporalAA = {};
            GI = {}; Probes = {};
        Lightmaps = {}; FrameLightmaps = {}; LightmapActive = false;
        Peeling = {}; FramePeeling = {};
        Refraction = {}; FrameRefraction = {}; RefractionActive=RefractionResolved=false;
			// Prepare resizes these targets without discarding compiled programs.
			// TSR changes the internal extent, not the volume shader variants.
			VolumetricFog.Begin();
			LightShafts.Begin();
			MotionStaging.Reset();
			SaoDepthHierarchy = {};
			OutlineMaskTexture = std::move(outlineMaskTexture);
			OutlineMaskTarget = std::move(outlineMaskTarget);
			OutlineMaskStaging = std::move(outlineMaskStaging);
			DepthTexture = std::move(depthTexture);
			DepthTarget = std::move(depthTarget);
			DepthStaging = std::move(depthStaging);
			DepthShaderView = std::move(depthShaderView);
			return true;
		}

		bool EnsureTechnique(Techniques::ShadingModel model)
		{
			// Compile each standalone native source once, when first selected.
			// A shading switch preserves meshes, textures, and frame dimensions.
			const std::size_t index = ShadingTechniqueIndex(model, FrameAreaLights);
			const char* name = ShadingTechniqueFile(model, FrameAreaLights);
			if (index >= StandaloneShadingTechniqueCount || !name) return Fail("Unknown DX11 shading technique");
			if (VertexShaders[index] && PixelShaders[index]) return true;
			const std::string file = std::string(name) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX11", file.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex)
				|| !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			ComPtr<ID3D11VertexShader> vertexShader;
			ComPtr<ID3D11PixelShader> pixelShader;
			HRESULT result = Device->CreateVertexShader(vertex->GetBufferPointer(),
				vertex->GetBufferSize(), nullptr, vertexShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(technique)", result));
			result = Device->CreatePixelShader(pixel->GetBufferPointer(),
				pixel->GetBufferSize(), nullptr, pixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(technique)", result));
			VertexShaders[index] = std::move(vertexShader);
			PixelShaders[index] = std::move(pixelShader);
			return true;
		}

		bool EnsureWeightedTechnique(Techniques::ShadingModel model)
		{
			const auto index = ShadingTechniqueIndex(model, FrameAreaLights);
			const char* name = ShadingTechniqueFile(model, FrameAreaLights);
			if (index >= StandaloneShadingTechniqueCount || !name) return Fail("Unknown weighted transparency shading model");
			if (WeightedPixelShaders[index]) return true;
			const std::string file = std::string(name) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX11", file.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("PSWeighted", "ps_5_0", pixel)) return false;
			const auto result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr,
				WeightedPixelShaders[index].ReleaseAndGetAddressOf());
			return SUCCEEDED(result) || Fail(HResultMessage("CreatePixelShader(weighted transparency)", result));
		}

		ID3D11RasterizerState* GetOrientedRasterizer()
		{
			const bool shadow = CurrentPass.Type == PassType::Shadow;
			const bool capture = shadow || CurrentPass.Type == PassType::EnvironmentCapture || CurrentPass.Type == PassType::PlanarCapture;
			const bool multisample = FrameSampleCount > 1 && !capture;
			const bool mirrored = RealtimeWorldIsMirrored(BoundWorld) != (CurrentPass.Type == PassType::PlanarCapture);
			const std::uint32_t key = static_cast<std::uint32_t>(BoundPipeline.Fill)
				| (shadow ? 1u << 2 : 0u) | (mirrored ? 1u << 3 : 0u)
				| (BoundPipeline.BackFaceCulling ? 1u << 4 : 0u) | (multisample ? 1u << 5 : 0u);
			auto found = CulledRasterStates.find(key);
			if (found != CulledRasterStates.end()) return found->second.Get();
			D3D11_RASTERIZER_DESC desc {};
			desc.FillMode = BoundPipeline.Fill == FillMode::Wireframe ? D3D11_FILL_WIREFRAME : D3D11_FILL_SOLID;
			desc.CullMode = BoundPipeline.BackFaceCulling ? D3D11_CULL_BACK : D3D11_CULL_NONE;
			desc.FrontCounterClockwise = mirrored ? FALSE : TRUE;
			desc.DepthClipEnable = TRUE;
			desc.MultisampleEnable = multisample ? TRUE : FALSE;
			desc.DepthBias = shadow ? ShadowDepthBiasUnits : 0;
			desc.DepthBiasClamp = shadow ? ShadowDepthBiasClamp : 0.0f;
			desc.SlopeScaledDepthBias = shadow ? ShadowSlopeScaledDepthBias : 0.0f;
			ComPtr<ID3D11RasterizerState> state;
			const HRESULT result = Device->CreateRasterizerState(&desc, state.GetAddressOf());
			if (FAILED(result))
			{
				Fail(HResultMessage("CreateRasterizerState(back-face culling)", result));
				return nullptr;
			}
			ID3D11RasterizerState* output = state.Get();
			CulledRasterStates.emplace(key, std::move(state));
			return output;
		}

		bool PrepareCascadedShadowFrame(const CascadedShadowState& state)
		{
			FrameCascades = state;
			if (state.Frame.CascadeCount > MaximumShadowCascades || (state.Frame.CascadeCount && state.MapResolution <= 0))
				return Fail("Invalid DX11 cascaded shadow frame");
			for (std::uint32_t index = 0; index < state.Frame.CascadeCount; ++index)
			{
				const auto map = Textures.find(state.Maps[index].Value);
				if (map == Textures.end() || map->second.Description.Format != TextureFormat::Depth32 ||
					map->second.Description.Width != state.MapResolution || map->second.Description.Height != state.MapResolution)
					return Fail("DX11 CSM requires a complete set of equal-sized depth maps");
			}
			if (state.Frame.CascadeCount && (!CascadeVertexShader || !CascadePixelShader))
			{
				if (!LoadTechniqueShaderSource("DX11", "CascadedShadowMap.hlsl", ShaderCode, LastError)) return false;
				ComPtr<ID3DBlob> vertex, pixel;
				if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
				HRESULT result = Device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, CascadeVertexShader.ReleaseAndGetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(CSM)", result));
				result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, CascadePixelShader.ReleaseAndGetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(CSM)", result));
			}
            // Keep both bindings populated: an inactive cascade block still
            // carries single-map filtering, and never inherits active map selection.
            for (std::size_t active = 0; active < CascadeConstantBuffers.size(); ++active)
            {
                auto& buffer = CascadeConstantBuffers[active];
                if (!buffer)
                {
                    D3D11_BUFFER_DESC desc {};
                    desc.ByteWidth = sizeof(CascadedShadowConstants); desc.Usage = D3D11_USAGE_DYNAMIC;
                    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                    const HRESULT result = Device->CreateBuffer(&desc, nullptr, buffer.GetAddressOf());
                    if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(shadow receiver)", result));
                }
                const auto constants = active && FrameVirtualShadowAtlas ? BuildVirtualShadowSamplingConstants(FrameVirtualShadowAtlas->Info,FrameVirtualShadowSampling,FrameShadowFilter) :
                    BuildCascadedShadowConstants(active ? state : CascadedShadowState{}, FrameShadowFilter);
                D3D11_MAPPED_SUBRESOURCE mapped {};
                const HRESULT result = Context->Map(buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
                if (FAILED(result)) return Fail(HResultMessage("Map(shadow receiver)", result));
                std::memcpy(mapped.pData, &constants, sizeof(constants));
                Context->Unmap(buffer.Get(), 0);
            }
			return true;
		}

        bool CheckVSMMomentViews()
        {
            if (FrameCascades.Frame.CascadeCount)
            {
                for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                    if (!VSM.View(FrameCascades.Maps[index])) return Fail("VSM receiver requires every cascade's completed moment filter");
            }
            else if (!VSM.View(LastShadowTexture)) return Fail("VSM receiver requires a completed moment filter");
            return true;
        }

		void BindCascades(bool sample)
		{
			sample = sample && (FrameCascades.Frame.CascadeCount != 0 || FrameVirtualShadowAtlas);
			ID3D11Buffer* buffer = CascadeConstantBuffers[sample ? 1 : 0].Get();
			Context->PSSetConstantBuffers(2, 1, &buffer);
			ID3D11ShaderResourceView* views[MaximumShadowCascades] {};
			if (sample && FrameVirtualShadowAtlas)
            { views[0]=FrameVirtualShadowAtlas->View.Get(); views[1]=FrameVirtualShadowAtlas->DirectoryView.Get(); }
            else if (sample)
				for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
					views[index] = FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM ? VSM.View(FrameCascades.Maps[index]) :
                        Textures.at(FrameCascades.Maps[index].Value).ShaderResourceView.Get();
			Context->PSSetShaderResources(10, MaximumShadowCascades, views);
		}

		bool CreatePipelineObjects()
		{
			if (!EnsureTechnique(Techniques::ShadingModel::Phong)) return false;
			ComPtr<ID3DBlob> vertexBytecode;
			ComPtr<ID3DBlob> shadowPixelBytecode;
			ComPtr<ID3DBlob> shadowVertexBytecode;
			ComPtr<ID3DBlob> outlinePixelBytecode;
			if (!CompileShader("VSMain", "vs_5_0", vertexBytecode)
				|| !CompileShader("PSOutline", "ps_5_0", outlinePixelBytecode)) return false;
			if (!LoadTechniqueShaderSource("DX11", "ShadowMapping.hlsl", ShaderCode, LastError) ||
				!CompileShader("VSMain", "vs_5_0", shadowVertexBytecode) ||
				!CompileShader("PSMain", "ps_5_0", shadowPixelBytecode)) return false;
			HRESULT result = Device->CreateVertexShader(shadowVertexBytecode->GetBufferPointer(),
				shadowVertexBytecode->GetBufferSize(), nullptr, ShadowVertexShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateVertexShader(shadow)", result));
			result = Device->CreatePixelShader(shadowPixelBytecode->GetBufferPointer(),
				shadowPixelBytecode->GetBufferSize(), nullptr, ShadowPixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreatePixelShader(shadow)", result));
			result = Device->CreatePixelShader(outlinePixelBytecode->GetBufferPointer(),
				outlinePixelBytecode->GetBufferSize(), nullptr, OutlinePixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreatePixelShader(outline)", result));

			const D3D11_INPUT_ELEMENT_DESC inputElements[] = {
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};
			result = Device->CreateInputLayout(inputElements, 4, vertexBytecode->GetBufferPointer(),
				vertexBytecode->GetBufferSize(), InputLayout.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateInputLayout", result));

			D3D11_BUFFER_DESC constantDesc = {};
			constantDesc.ByteWidth = sizeof(DrawConstants);
			constantDesc.Usage = D3D11_USAGE_DYNAMIC;
			constantDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constantDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&constantDesc, nullptr, ConstantBuffer.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBuffer(constants)", result));

			D3D11_RASTERIZER_DESC rasterDesc = {};
			rasterDesc.FillMode = D3D11_FILL_SOLID;
			rasterDesc.CullMode = D3D11_CULL_NONE;
			rasterDesc.FrontCounterClockwise = TRUE;
			rasterDesc.DepthClipEnable = TRUE;
			rasterDesc.MultisampleEnable = TRUE;
			result = Device->CreateRasterizerState(&rasterDesc, SolidRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(solid)", result));
			D3D11_RASTERIZER_DESC shadowRasterDesc = rasterDesc;
			shadowRasterDesc.DepthBias = ShadowDepthBiasUnits;
			shadowRasterDesc.DepthBiasClamp = ShadowDepthBiasClamp;
			shadowRasterDesc.SlopeScaledDepthBias = ShadowSlopeScaledDepthBias;
			result = Device->CreateRasterizerState(
				&shadowRasterDesc, ShadowRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateRasterizerState(shadow)", result));
			rasterDesc.FillMode = D3D11_FILL_WIREFRAME;
			// Single-sample lines use the same rasterization rule as DX12. MSAA
			// wireframes use the oriented cache, keyed by their actual target samples.
			rasterDesc.MultisampleEnable = FALSE;
			result = Device->CreateRasterizerState(&rasterDesc, WireRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(wire)", result));
			rasterDesc.MultisampleEnable = TRUE;
			// The outline pass writes the visible selected surface into a mask.
			// Pixel-sized dilation is applied to that mask during presentation.
			rasterDesc.FillMode = D3D11_FILL_SOLID;
			rasterDesc.CullMode = D3D11_CULL_NONE;
			rasterDesc.FrontCounterClockwise = FALSE;
			rasterDesc.DepthBias = 0;
			rasterDesc.SlopeScaledDepthBias = 0.0f;
			result = Device->CreateRasterizerState(&rasterDesc, OutlineRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(outline)", result));

			D3D11_DEPTH_STENCIL_DESC depthDesc = {};
			depthDesc.DepthEnable = TRUE;
			depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			depthDesc.DepthFunc = D3D11_COMPARISON_LESS;
			result = Device->CreateDepthStencilState(&depthDesc, DepthWriteState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(write)", result));
			depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			depthDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
			result = Device->CreateDepthStencilState(&depthDesc, DepthReadState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(read)", result));
			depthDesc.DepthEnable = FALSE;
			result = Device->CreateDepthStencilState(&depthDesc, DepthDisabledState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(disabled)", result));

			D3D11_BLEND_DESC blendDesc = {};
			blendDesc.RenderTarget[0].RenderTargetWriteMask = 0;
			result = Device->CreateBlendState(&blendDesc, ColorWriteDisabledState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBlendState", result));
			D3D11_BLEND_DESC alphaBlendDesc = {};
			D3D11_RENDER_TARGET_BLEND_DESC& alphaBlend =
				alphaBlendDesc.RenderTarget[0];
			alphaBlend.BlendEnable = TRUE;
			alphaBlend.SrcBlend = D3D11_BLEND_SRC_ALPHA;
			alphaBlend.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
			alphaBlend.BlendOp = D3D11_BLEND_OP_ADD;
			alphaBlend.SrcBlendAlpha = D3D11_BLEND_ONE;
			alphaBlend.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
			alphaBlend.BlendOpAlpha = D3D11_BLEND_OP_ADD;
			alphaBlend.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
			result = Device->CreateBlendState(
				&alphaBlendDesc, AlphaBlendState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateBlendState(alpha)", result));

			D3D11_SAMPLER_DESC samplerDesc = {};
			samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.MaxLOD = 0;
			result = Device->CreateSamplerState(&samplerDesc, SurfaceSampler.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateSamplerState(surface)", result));
			samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
			samplerDesc.MaxAnisotropy = MaterialAnisotropy;
			samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
			result = Device->CreateSamplerState(&samplerDesc, AnisotropicSampler.GetAddressOf());
			return SUCCEEDED(result) || Fail(HResultMessage("ID3D11Device::CreateSamplerState(anisotropic)", result));
		}

		bool CreateHdrSurface(HdrSurface& surface, int width, int height)
		{
			if (surface.Texture && surface.Width == width && surface.Height == height) return true;
			HdrSurface candidate;
			D3D11_TEXTURE2D_DESC description {};
			description.Width = static_cast<UINT>(width);
			description.Height = static_cast<UINT>(height);
			description.MipLevels = description.ArraySize = 1;
			description.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
			description.SampleDesc.Count = 1;
			description.Usage = D3D11_USAGE_DEFAULT;
			description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			HRESULT result = Device->CreateTexture2D(&description, nullptr, candidate.Texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(HDR)", result));
			result = Device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(HDR)", result));
			result = Device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(HDR)", result));
			candidate.Width = width;
			candidate.Height = height;
			surface = std::move(candidate);
			return true;
		}

		static std::size_t HdrProgramIndex(Techniques::HdrPassKind pass)
		{
			return HdrShaderIndex(pass);
		}

		bool EnsureHdrProgram(Techniques::HdrPassKind pass)
		{
			const auto index = HdrProgramIndex(pass);
			if (HdrVertexShaders[index] && HdrPixelShaders[index]) return true;
			const std::string filename = std::string(HdrShaderFilename(pass)) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX11", filename.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			ComPtr<ID3D11VertexShader> vertexShader;
			ComPtr<ID3D11PixelShader> pixelShader;
			HRESULT result = Device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, vertexShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(HDR)", result));
			result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, pixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(HDR)", result));
			HdrVertexShaders[index] = std::move(vertexShader);
			HdrPixelShaders[index] = std::move(pixelShader);
			return true;
		}

		bool PrepareHdrFrame(const Techniques::HdrSettings& settings)
		{
			HdrSceneActive = Techniques::RequiresHDR(settings);
			if (!HdrSceneActive) return true;
			UINT support = 0;
			const UINT required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_SHADER_SAMPLE
				| D3D11_FORMAT_SUPPORT_RENDER_TARGET | D3D11_FORMAT_SUPPORT_BLENDABLE;
			if (FAILED(Device->CheckFormatSupport(DXGI_FORMAT_R16G16B16A16_FLOAT, &support)) || (support & required) != required)
				return Fail("DX11 requires sampleable, blendable RGBA16F render targets for HDR");
			if (!CreateHdrSurface(HdrSurfaces[HdrScene], CreateInfo.Width, CreateInfo.Height)
				|| !CreateHdrSurface(HdrSurfaces[HdrScratch], CreateInfo.Width, CreateInfo.Height)) return false;
			if (settings.Bloom || Techniques::UsesLensFlare(settings))
			{
				const int width = std::max(1, (CreateInfo.Width + 1) / 2);
				const int height = std::max(1, (CreateInfo.Height + 1) / 2);
				if (!CreateHdrSurface(HdrSurfaces[BloomFirst], width, height)
					|| !CreateHdrSurface(HdrSurfaces[BloomSecond], width, height)) return false;
			}
			for (auto pass : Techniques::BuildHdrPlan(settings)) if (!EnsureHdrProgram(pass)) return false;
			if (!HdrConstantBuffer)
			{
				D3D11_BUFFER_DESC buffer {};
				buffer.ByteWidth = sizeof(HdrConstants);
				buffer.Usage = D3D11_USAGE_DYNAMIC;
				buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
				buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				HRESULT result = Device->CreateBuffer(&buffer, nullptr, HdrConstantBuffer.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(HDR constants)", result));
			}
			if (!HdrSampler)
			{
				D3D11_SAMPLER_DESC sampler {};
				sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
				sampler.MaxLOD = D3D11_FLOAT32_MAX;
				HRESULT result = Device->CreateSamplerState(&sampler, HdrSampler.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateSamplerState(HDR)", result));
			}
			return true;
		}

		bool ApplyHdr(const Techniques::HdrSettings& settings)
		{
			if (!HdrSceneActive) return true;
			const auto plan = Techniques::BuildHdrPlan(settings);
			ID3D11Buffer* constantsBuffer = HdrConstantBuffer.Get();
			ID3D11SamplerState* sampler = HdrSampler.Get();
			Context->PSSetConstantBuffers(0, 1, &constantsBuffer);
			Context->PSSetSamplers(0, 1, &sampler);
			Context->IASetInputLayout(nullptr);
			Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			Context->RSSetState(SolidRasterState.Get());
			Context->OMSetDepthStencilState(DepthDisabledState.Get(), 0);
			Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
			std::size_t scene = HdrScene;
			ID3D11ShaderResourceView* nullViews[4] {};
			for (auto pass : plan)
			{
				std::size_t sourceIndex = scene;
				std::size_t targetIndex = scene == HdrScene ? HdrScratch : HdrScene;
				const bool tone = pass == Techniques::HdrPassKind::ToneMapping;
				const auto resources = DescribeHdrPass(pass);
				if (resources.SourceHalf >= 0) sourceIndex = BloomFirst + resources.SourceHalf;
				if (resources.DestinationHalf >= 0) targetIndex = BloomFirst + resources.DestinationHalf;
				const auto& source = HdrSurfaces[sourceIndex];
				const HdrConstants constants = BuildHdrConstants(settings, source.Width, source.Height, pass, CreateInfo.Width, CreateInfo.Height);
				D3D11_MAPPED_SUBRESOURCE mapped {};
				const HRESULT result = Context->Map(HdrConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
				if (FAILED(result)) return Fail(HResultMessage("Map(HDR constants)", result));
				std::memcpy(mapped.pData, &constants, sizeof(constants));
				Context->Unmap(HdrConstantBuffer.Get(), 0);
				Context->PSSetShaderResources(0, 4, nullViews);
				Context->OMSetRenderTargets(0, nullptr, nullptr);
				ID3D11RenderTargetView* target = tone ? ColorTarget.Get() : HdrSurfaces[targetIndex].Target.Get();
				Context->OMSetRenderTargets(1, &target, nullptr);
				ID3D11ShaderResourceView* inputs[2] = {source.View.Get(),
					resources.CompositeHalf >= 0 ? HdrSurfaces[BloomFirst + resources.CompositeHalf].View.Get() : nullptr};
				Context->PSSetShaderResources(0, 2, inputs);
				D3D11_VIEWPORT viewport {};
				viewport.Width = static_cast<float>(tone ? CreateInfo.Width : HdrSurfaces[targetIndex].Width);
				viewport.Height = static_cast<float>(tone ? CreateInfo.Height : HdrSurfaces[targetIndex].Height);
				viewport.MaxDepth = 1.0f;
				Context->RSSetViewports(1, &viewport);
				const auto program = HdrProgramIndex(pass);
				Context->VSSetShader(HdrVertexShaders[program].Get(), nullptr, 0);
				Context->PSSetShader(HdrPixelShaders[program].Get(), nullptr, 0);
				Context->Draw(3, 0);
				if (resources.UpdatesScene) scene = targetIndex;
				PendingFrame.HDRPasses.push_back(pass);
			}
			Context->PSSetShaderResources(0, 4, nullViews);
			Context->OMSetRenderTargets(0, nullptr, nullptr);
			PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
			return true;
		}

		struct EnvironmentTargets
		{
			ComPtr<ID3D11Texture2D> Color, Depth;
			ComPtr<ID3D11ShaderResourceView> View;
			std::array<ComPtr<ID3D11RenderTargetView>, EnvironmentFaceCount> Faces;
			ComPtr<ID3D11DepthStencilView> DepthTarget;
			UINT Resolution = 0;
		};
		bool PrepareEnvironmentCapture(const EnvironmentCaptureFrame& frame)
		{
			EnvironmentActive = Techniques::UsesEnvironmentTexture(FrameReflections);
			EnvironmentFacesWritten = 0; EnvironmentReady = false;
			FrameEnvironment = {};
			if (!EnvironmentActive) return true;
			if (!ValidEnvironmentSource(frame) || (!frame.Source && frame.Resolution != FrameReflections.CaptureResolution) || frame.MipCount == 0)
				return Fail("Environment reflections require a matching six-face capture frame before BeginFrame");
			UINT expectedMips = 0;
			for (auto extent = frame.Resolution; extent != 0; extent >>= 1) ++expectedMips;
			if (frame.MipCount != expectedMips) return Fail("Environment capture requires a complete mip chain");
			for (const auto& face : frame.Faces)
				for (float v : face.ViewProjection.Values)
					if (!std::isfinite(v)) return Fail("Environment capture projections must be finite");
			if (Environment.Resolution != frame.Resolution)
			{
				UINT support = 0;
				constexpr UINT Required = D3D11_FORMAT_SUPPORT_TEXTURECUBE | D3D11_FORMAT_SUPPORT_RENDER_TARGET |
					D3D11_FORMAT_SUPPORT_SHADER_SAMPLE | D3D11_FORMAT_SUPPORT_MIP_AUTOGEN;
				if (FAILED(Device->CheckFormatSupport(DXGI_FORMAT_R16G16B16A16_FLOAT, &support)) || (support & Required) != Required)
					return Fail("The device cannot render/sample RGBA16F cubemaps with generated mips");
				EnvironmentTargets candidate;
				D3D11_TEXTURE2D_DESC texture {};
				texture.Width = texture.Height = frame.Resolution;
				texture.MipLevels = frame.MipCount; texture.ArraySize = static_cast<UINT>(EnvironmentFaceCount);
				texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
				texture.SampleDesc.Count = 1; texture.Usage = D3D11_USAGE_DEFAULT;
				texture.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
				texture.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE | D3D11_RESOURCE_MISC_GENERATE_MIPS;
				HRESULT result = Device->CreateTexture2D(&texture, nullptr, candidate.Color.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(environment cube)", result));
				D3D11_SHADER_RESOURCE_VIEW_DESC view {};
				view.Format = texture.Format; view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
				view.TextureCube.MipLevels = frame.MipCount;
				result = Device->CreateShaderResourceView(candidate.Color.Get(), &view, candidate.View.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(environment cube)", result));
				for (UINT face = 0; face < EnvironmentFaceCount; ++face)
				{
					D3D11_RENDER_TARGET_VIEW_DESC target {};
					target.Format = texture.Format; target.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
					target.Texture2DArray.FirstArraySlice = face; target.Texture2DArray.ArraySize = 1;
					result = Device->CreateRenderTargetView(candidate.Color.Get(), &target, candidate.Faces[face].GetAddressOf());
					if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(environment face)", result));
				}
				texture.MipLevels = texture.ArraySize = 1; texture.Format = DXGI_FORMAT_D32_FLOAT;
				texture.BindFlags = D3D11_BIND_DEPTH_STENCIL; texture.MiscFlags = 0;
				result = Device->CreateTexture2D(&texture, nullptr, candidate.Depth.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(environment depth)", result));
				result = Device->CreateDepthStencilView(candidate.Depth.Get(), nullptr, candidate.DepthTarget.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateDepthStencilView(environment)", result));
				candidate.Resolution = frame.Resolution;
				Environment = std::move(candidate);
			}
			FrameEnvironment = frame;
			return true;
		}

		struct PlanarTargets
		{
			HdrSurface Color;
			ComPtr<ID3D11Texture2D> Depth;
			ComPtr<ID3D11DepthStencilView> DepthTarget;
			std::vector<ComPtr<ID3D11ShaderResourceView>> MipViews;
			std::vector<ComPtr<ID3D11RenderTargetView>> MipTargets;
		};
		bool PreparePlanarCapture(const PlanarCaptureFrame& frame)
		{
			PlanarActive = Techniques::UsesPlanarCapture(FrameReflections);
			PlanarWritten = PlanarReady = false;
			FramePlanar = {};
			if (!PlanarActive) return true;
			PlanarReflectionConstants constants;
			ReflectionDesc desc; desc.ViewProjection = frame.MainViewProjection;
			if (!BuildPlanarReflectionConstants(desc, frame, FrameReflections, FrameReflectionShading, CreateInfo.Width, CreateInfo.Height, constants))
				return Fail("Planar reflection requires a finite capture frame matching the main viewport before BeginFrame");
			const auto divisor = FrameReflections.Planar.ResolutionDivisor;
			UINT expectedMips = 0;
			for (auto extent = std::max(frame.Width, frame.Height); extent; extent >>= 1) ++expectedMips;
			if (frame.Width != (frame.MainWidth + divisor - 1) / divisor || frame.Height != (frame.MainHeight + divisor - 1) / divisor ||
				frame.MipCount != expectedMips) return Fail("Planar capture extent or mip chain disagrees with frame settings");
			for (float value : frame.ClippedViewProjection.Values) if (!std::isfinite(value)) return Fail("Planar clipping projection must be finite");
			if (Planar.Color.Width != static_cast<int>(frame.Width) || Planar.Color.Height != static_cast<int>(frame.Height))
			{
				UINT support = 0;
				constexpr UINT Required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_RENDER_TARGET |
					D3D11_FORMAT_SUPPORT_SHADER_SAMPLE;
				if (FAILED(Device->CheckFormatSupport(DXGI_FORMAT_R16G16B16A16_FLOAT, &support)) || (support & Required) != Required)
					return Fail("The device cannot render/sample RGBA16F planar captures");
				PlanarTargets candidate;
				D3D11_TEXTURE2D_DESC texture {};
				texture.Width = frame.Width; texture.Height = frame.Height;
				texture.MipLevels = frame.MipCount; texture.ArraySize = 1;
				texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
				texture.SampleDesc.Count = 1; texture.Usage = D3D11_USAGE_DEFAULT;
				texture.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
				HRESULT result = Device->CreateTexture2D(&texture, nullptr, candidate.Color.Texture.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(planar radiance)", result));
				result = Device->CreateShaderResourceView(candidate.Color.Texture.Get(), nullptr, candidate.Color.View.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(planar radiance)", result));
				result = Device->CreateRenderTargetView(candidate.Color.Texture.Get(), nullptr, candidate.Color.Target.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(planar radiance)", result));
				candidate.MipViews.resize(frame.MipCount); candidate.MipTargets.resize(frame.MipCount);
				for (UINT mip = 0; mip < frame.MipCount; ++mip)
				{
					D3D11_SHADER_RESOURCE_VIEW_DESC view {};
					view.Format = texture.Format; view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
					view.Texture2D.MostDetailedMip = mip; view.Texture2D.MipLevels = 1;
					result = Device->CreateShaderResourceView(candidate.Color.Texture.Get(), &view, candidate.MipViews[mip].GetAddressOf());
					if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(planar mip)", result));
					D3D11_RENDER_TARGET_VIEW_DESC target {};
					target.Format = texture.Format; target.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
					target.Texture2D.MipSlice = mip;
					result = Device->CreateRenderTargetView(candidate.Color.Texture.Get(), &target, candidate.MipTargets[mip].GetAddressOf());
					if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(planar mip)", result));
				}
				texture.MipLevels = 1; texture.Format = DXGI_FORMAT_D32_FLOAT;
				texture.BindFlags = D3D11_BIND_DEPTH_STENCIL; texture.MiscFlags = 0;
				result = Device->CreateTexture2D(&texture, nullptr, candidate.Depth.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(planar depth)", result));
				result = Device->CreateDepthStencilView(candidate.Depth.Get(), nullptr, candidate.DepthTarget.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateDepthStencilView(planar)", result));
				candidate.Color.Width = frame.Width; candidate.Color.Height = frame.Height;
				Planar = std::move(candidate);
			}
			FramePlanar = frame;
			return true;
		}

		enum ReflectionProgram : std::size_t { ScreenReflection, CubeReflection, MapReflection, PlanarReflection, ReflectionProgramCount };
		ReflectionProgram FrameReflectionProgram(bool planarPass) const
		{
			return planarPass ? PlanarReflection :
				FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude ? MapReflection :
				FrameReflections.Mode == Techniques::ReflectionMode::Cubemap ? CubeReflection : ScreenReflection;
		}
		bool EnsureEnvironmentMapResources()
		{
			const auto layout = BuildEnvironmentMapLayout(FrameEnvironment.Resolution);
			if (!layout.MipCount) return Fail("Environment map requires valid power-of-two capture dimensions");
			if (!EnvironmentMapSampler)
			{
				D3D11_SAMPLER_DESC sampler{};
				sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
				sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
				sampler.MaxLOD = D3D11_FLOAT32_MAX;
				const HRESULT result = Device->CreateSamplerState(&sampler, EnvironmentMapSampler.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateSamplerState(latitude-longitude)", result));
			}
			if (EnvironmentMap.Width == static_cast<int>(layout.Width)) return true;
			HdrSurface candidate;
			D3D11_TEXTURE2D_DESC texture{};
			texture.Width = layout.Width; texture.Height = layout.Height;
			texture.MipLevels = layout.MipCount; texture.ArraySize = 1;
			texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
			texture.SampleDesc.Count = 1; texture.Usage = D3D11_USAGE_DEFAULT;
			texture.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			texture.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
			HRESULT result = Device->CreateTexture2D(&texture, nullptr, candidate.Texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(latitude-longitude)", result));
			result = Device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(latitude-longitude)", result));
			result = Device->CreateRenderTargetView(candidate.Texture.Get(), nullptr, candidate.Target.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(latitude-longitude)", result));
			candidate.Width = layout.Width; candidate.Height = layout.Height;
			EnvironmentMap = std::move(candidate); EnvironmentMapMips = layout.MipCount;
			return true;
		}

		bool EnsureReflectionResources()
		{
            // Sky lookup and authored cube upload also need the cube mip
            // pipeline when the selected surface-reflection method is None.
            if (FrameReflections.Skybox)
            {
                const auto mode=FrameReflections.Mode;
                FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
                const bool ready=EnsureReflectionResources(false);
                FrameReflections.Mode=mode;
                if (!ready) return false;
            }

		    if (PlanarActive && !EnsureReflectionResources(true)) return false;
		    return FrameReflections.Mode==Techniques::ReflectionMode::None || EnsureReflectionResources(false);
		}

		bool EnsureReflectionResources(bool planarPass)
		{
			if (Techniques::UsesLumenReflections(FrameReflections)) return true; // Native resources are sized after the bounded query.
            if (FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
            return Techniques::UsesHybridRendering(FrameRayRadianceSettings.Pipeline) ? EnsureHybridRenderingResources() : EnsureRayReflectionResources();
			if (!HdrSceneActive || !DepthShaderView) return Fail("Reflections require HDR color and sampleable depth");
			if (!CreateHdrSurface(ReflectionSurface, CreateInfo.Width, CreateInfo.Height)) return false;
			const auto program = FrameReflectionProgram(planarPass);
			if (program == MapReflection && !EnsureEnvironmentMapResources()) return false;
			if (ReflectionVertex[program] && ReflectionPixel[program] && ReflectionConstantBuffer && ReflectionBlend) return true;
			const char* filename = program == PlanarReflection ? "PlanarReflections.hlsl" : program == MapReflection ? "EnvironmentMapReflections.hlsl" :
				program == CubeReflection ? "CubemapReflections.hlsl" : "SSR.hlsl";
			if (!LoadTechniqueShaderSource("DX11", filename, ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertexCode, pixelCode;
			if (!CompileShader("VSMain", "vs_5_0", vertexCode) || !CompileShader("PSMain", "ps_5_0", pixelCode)) return false;
			ComPtr<ID3D11VertexShader> vertex;
			ComPtr<ID3D11PixelShader> pixel;
			ComPtr<ID3D11Buffer> constants;
			ComPtr<ID3D11BlendState> blendState;
			ComPtr<ID3D11PixelShader> convert;
			ComPtr<ID3D11PixelShader> planarMip;
			if (program == PlanarReflection)
			{
				ComPtr<ID3DBlob> mipCode;
				if (!CompileShader("PSMip", "ps_5_0", mipCode)) return false;
				const HRESULT result = Device->CreatePixelShader(mipCode->GetBufferPointer(), mipCode->GetBufferSize(), nullptr, planarMip.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(planar area mip)", result));
			}
			if (program == MapReflection)
			{
				ComPtr<ID3DBlob> conversionCode;
				if (!CompileShader("PSConvert", "ps_5_0", conversionCode)) return false;
				const HRESULT result = Device->CreatePixelShader(conversionCode->GetBufferPointer(), conversionCode->GetBufferSize(), nullptr, convert.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(environment conversion)", result));
			}
			HRESULT result = Device->CreateVertexShader(vertexCode->GetBufferPointer(), vertexCode->GetBufferSize(), nullptr, vertex.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(reflections)", result));
			result = Device->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, pixel.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(reflections)", result));
			D3D11_BUFFER_DESC buffer {};
			buffer.ByteWidth = static_cast<UINT>(std::max({sizeof(ReflectionConstants), sizeof(PlanarReflectionConstants), sizeof(CubemapReflectionConstants)}));
			buffer.Usage = D3D11_USAGE_DYNAMIC;
			buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&buffer, nullptr, constants.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(reflections)", result));
			D3D11_BLEND_DESC blend {};
			auto& color = blend.RenderTarget[0];
			color.BlendEnable = TRUE;
			color.SrcBlend = color.DestBlend = D3D11_BLEND_ONE;
			color.BlendOp = color.BlendOpAlpha = D3D11_BLEND_OP_ADD;
			color.SrcBlendAlpha = D3D11_BLEND_ZERO; color.DestBlendAlpha = D3D11_BLEND_ONE;
			color.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE;
			result = Device->CreateBlendState(&blend, blendState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateBlendState(reflection additive RGB)", result));
			ReflectionVertex[program] = std::move(vertex); ReflectionPixel[program] = std::move(pixel);
			if (program == MapReflection) EnvironmentMapConvert = std::move(convert);
			if (program == PlanarReflection) PlanarMipPixel = std::move(planarMip);
			ReflectionConstantBuffer = std::move(constants); ReflectionBlend = std::move(blendState);
			return true;
		}

		bool ConvertEnvironmentMap()
		{
			EnvironmentMapReflectionConstants constants{};
			constants.Viewport = {float(EnvironmentMap.Width), float(EnvironmentMap.Height),
				1.0f / EnvironmentMap.Width, 1.0f / EnvironmentMap.Height};
			D3D11_MAPPED_SUBRESOURCE mapped{};
			const HRESULT result = Context->Map(ReflectionConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("Map(environment conversion)", result));
			std::memcpy(mapped.pData, &constants, sizeof(constants));
			Context->Unmap(ReflectionConstantBuffer.Get(), 0);
			// Cube input and the lat-long render target are distinct resources.
			// Unbind the output before native mip generation and later reflection reads.
			constexpr UINT InputCount = 9;
			ID3D11ShaderResourceView* nullViews[InputCount]{};
			Context->PSSetShaderResources(0, InputCount, nullViews);
			Context->OMSetRenderTargets(0, nullptr, nullptr);
			ID3D11RenderTargetView* target = EnvironmentMap.Target.Get();
			Context->OMSetRenderTargets(1, &target, nullptr);
			Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
			Context->OMSetDepthStencilState(DepthDisabledState.Get(), 0);
			Context->RSSetState(SolidRasterState.Get());
			D3D11_VIEWPORT viewport{};
			viewport.Width = float(EnvironmentMap.Width); viewport.Height = float(EnvironmentMap.Height); viewport.MaxDepth = 1;
			Context->RSSetViewports(1, &viewport);
			Context->IASetInputLayout(nullptr);
			Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			Context->VSSetShader(ReflectionVertex[MapReflection].Get(), nullptr, 0);
			Context->PSSetShader(EnvironmentMapConvert.Get(), nullptr, 0);
			ID3D11Buffer* buffer = ReflectionConstantBuffer.Get();
			Context->PSSetConstantBuffers(0, 1, &buffer);
			ID3D11ShaderResourceView* capture = Environment.View.Get();
			Context->PSSetShaderResources(7, 1, &capture);
			ID3D11SamplerState* sampler = HdrSampler.Get();
			Context->PSSetSamplers(0, 1, &sampler);
			Context->Draw(3, 0);
			Context->PSSetShaderResources(0, InputCount, nullViews);
			Context->OMSetRenderTargets(0, nullptr, nullptr);
			Context->GenerateMips(EnvironmentMap.View.Get());
			++PendingFrame.Statistics.EnvironmentConversionPassCount;
			return true;
		}

		bool EnsureDepthOfFieldResources()
		{
			if (!HdrSceneActive || !DepthShaderView) return Fail("Depth of field requires HDR color and sampleable scene depth");
			if (DepthOfFieldVertex && DepthOfFieldPixel && DepthOfFieldConstantBuffer && DepthOfFieldCompositeWriteMask) return true;
			if (!LoadTechniqueShaderSource("DX11", "DepthOfField.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertexCode, pixelCode;
			if (!CompileShader("VSMain", "vs_5_0", vertexCode) || !CompileShader("PSMain", "ps_5_0", pixelCode)) return false;
			ComPtr<ID3D11VertexShader> vertex;
			ComPtr<ID3D11PixelShader> pixel;
			ComPtr<ID3D11Buffer> constants;
			ComPtr<ID3D11BlendState> writeMask;
			HRESULT result = Device->CreateVertexShader(vertexCode->GetBufferPointer(), vertexCode->GetBufferSize(), nullptr, vertex.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(depth of field)", result));
			result = Device->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, pixel.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(depth of field)", result));
			D3D11_BUFFER_DESC buffer {};
			buffer.ByteWidth = sizeof(DepthOfFieldConstants);
			buffer.Usage = D3D11_USAGE_DYNAMIC;
			buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&buffer, nullptr, constants.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(depth of field)", result));
			D3D11_BLEND_DESC blend {};
			blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE;
			result = Device->CreateBlendState(&blend, writeMask.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateBlendState(depth of field RGB write)", result));
			DepthOfFieldVertex = std::move(vertex);
			DepthOfFieldPixel = std::move(pixel);
			DepthOfFieldConstantBuffer = std::move(constants);
			DepthOfFieldCompositeWriteMask = std::move(writeMask);
			return true;
		}

		bool EnsureVelocityResources()
		{
			if (!HdrSceneActive || !DepthShaderView) return Fail("Velocity requires HDR color and sampleable depth");
			if (!CreateHdrSurface(MotionSurfaces[MotionVelocity], CreateInfo.Width, CreateInfo.Height)) return false;
			if (MotionGeometryVertex && MotionGeometryPixel && MotionConstantBuffer) return true;
			if (!LoadTechniqueShaderSource("DX11", "MotionVectors.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> geometryVertex, geometryPixel;
			if (!CompileShader("VSVelocity", "vs_5_0", geometryVertex) || !CompileShader("PSVelocity", "ps_5_0", geometryPixel)) return false;
			HRESULT result = Device->CreateVertexShader(geometryVertex->GetBufferPointer(), geometryVertex->GetBufferSize(),
				nullptr, MotionGeometryVertex.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(motion geometry)", result));
			result = Device->CreatePixelShader(geometryPixel->GetBufferPointer(), geometryPixel->GetBufferSize(),
				nullptr, MotionGeometryPixel.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(motion geometry)", result));
			D3D11_BUFFER_DESC buffer {};
			buffer.ByteWidth = sizeof(MotionBlurConstants);
			buffer.Usage = D3D11_USAGE_DYNAMIC;
			buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&buffer, nullptr, MotionConstantBuffer.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(motion)", result));
			return true;
		}
		bool EnsureMotionResources()
		{
			if (!EnsureVelocityResources()) return false;
			const int tile = Techniques::MotionBlurTileSize;
			const int width = (CreateInfo.Width + tile - 1) / tile, height = (CreateInfo.Height + tile - 1) / tile;
			if (!CreateHdrSurface(MotionSurfaces[MotionTileMax], width, height) ||
				!CreateHdrSurface(MotionSurfaces[MotionNeighborMax], width, height)) return false;
			if (MotionScreenVertex && MotionScreenPixel && MotionCompositeMask) return true;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!LoadTechniqueShaderSource("DX11", "MotionBlur.hlsl", ShaderCode, LastError) ||
				!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			HRESULT result = Device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(),
				nullptr, MotionScreenVertex.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(motion fullscreen)", result));
			result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(),
				nullptr, MotionScreenPixel.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(motion fullscreen)", result));
			D3D11_BLEND_DESC blend {};
			blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE;
			result = Device->CreateBlendState(&blend, MotionCompositeMask.ReleaseAndGetAddressOf());
			return SUCCEEDED(result) || Fail(HResultMessage("CreateBlendState(motion)", result));
		}

		bool UploadMotionConstants(const MotionBlurConstants& constants)
		{
			D3D11_MAPPED_SUBRESOURCE mapped {};
			const HRESULT result = Context->Map(MotionConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("Map(motion constants)", result));
			std::memcpy(mapped.pData, &constants, sizeof(constants));
			Context->Unmap(MotionConstantBuffer.Get(), 0);
			return true;
		}

		bool ReadMotionVectors()
		{
			if (!MotionStaging)
			{
				D3D11_TEXTURE2D_DESC description {};
				MotionSurfaces[MotionVelocity].Texture->GetDesc(&description);
				description.BindFlags = 0;
				description.Usage = D3D11_USAGE_STAGING;
				description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
				const HRESULT result = Device->CreateTexture2D(&description, nullptr, MotionStaging.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(motion readback)", result));
			}
			Context->CopyResource(MotionStaging.Get(), MotionSurfaces[MotionVelocity].Texture.Get());
			D3D11_MAPPED_SUBRESOURCE mapped {};
			const HRESULT result = Context->Map(MotionStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("Map(motion readback)", result));
			const auto decodeHalf = [](std::uint16_t bits)
			{
				const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
				if (exponent == 31)
					return fraction ? std::numeric_limits<float>::quiet_NaN() :
						(bits & 32768 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
				const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
					std::ldexp(static_cast<float>(fraction + 1024), exponent - 25);
				return bits & 32768 ? -magnitude : magnitude;
			};
			PendingFrame.MotionVectors.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height);
			for (int y = 0; y < CreateInfo.Height; ++y)
			{
				const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(mapped.pData) + y * mapped.RowPitch);
				for (int x = 0; x < CreateInfo.Width; ++x)
					PendingFrame.MotionVectors[static_cast<std::size_t>(y) * CreateInfo.Width + x] =
                        DecodeMotionVector({decodeHalf(row[x * 4]), decodeHalf(row[x * 4 + 1]), decodeHalf(row[x * 4 + 2]), decodeHalf(row[x * 4 + 3])});
			}
			Context->Unmap(MotionStaging.Get(), 0);
			return true;
		}

		bool EnsureDeferredResources()
		{
			if (FrameSampleCount > 1 && !DeferredMultisamplePixel)
			{
				if (!LoadTechniqueShaderSource("DX11", "Deferred.hlsl", ShaderCode, LastError)) return false;
				const D3D_SHADER_MACRO defines[] = {{"EVANS_MULTISAMPLE", "1"}, {nullptr, nullptr}};
				ComPtr<ID3DBlob> pixel;
				if (!CompileShader("PSLighting", "ps_5_0", pixel, defines)) return false;
				const auto result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(),
					nullptr, DeferredMultisamplePixel.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(MSAA deferred)", result));
			}
			if (!HdrSceneActive || !MaterialCaptureActive) return Fail("Material capture requires an HDR scene and an active consumer");
			for (auto& target : GBufferSurfaces)
				if (!CreateHdrSurface(target, CreateInfo.Width, CreateInfo.Height)) return false;
			if (DeferredGeometryVertex && DeferredGeometryPixel && DeferredLightVertex && DeferredLightPixel && DeferredConstantBuffer)
				return true;
			if (!LoadTechniqueShaderSource("DX11", "Deferred.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> geometryVertex, geometryPixel, lightVertex, lightPixel;
			if (!CompileShader("VSMain", "vs_5_0", geometryVertex) || !CompileShader("PSMain", "ps_5_0", geometryPixel) ||
				!CompileShader("VSLighting", "vs_5_0", lightVertex) || !CompileShader("PSLighting", "ps_5_0", lightPixel)) return false;
			HRESULT result = Device->CreateVertexShader(geometryVertex->GetBufferPointer(), geometryVertex->GetBufferSize(),
				nullptr, DeferredGeometryVertex.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(G-buffer)", result));
			result = Device->CreatePixelShader(geometryPixel->GetBufferPointer(), geometryPixel->GetBufferSize(),
				nullptr, DeferredGeometryPixel.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(G-buffer)", result));
			result = Device->CreateVertexShader(lightVertex->GetBufferPointer(), lightVertex->GetBufferSize(),
				nullptr, DeferredLightVertex.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(deferred lighting)", result));
			result = Device->CreatePixelShader(lightPixel->GetBufferPointer(), lightPixel->GetBufferSize(),
				nullptr, DeferredLightPixel.ReleaseAndGetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(deferred lighting)", result));
			D3D11_BUFFER_DESC description {};
			description.ByteWidth = sizeof(DeferredConstants);
			description.Usage = D3D11_USAGE_DYNAMIC;
			description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&description, nullptr, DeferredConstantBuffer.ReleaseAndGetAddressOf());
			return SUCCEEDED(result) || Fail(HResultMessage("CreateBuffer(deferred lighting)", result));
		}

		bool SupportsFloatDepthTargets(bool requireMipChain) const
		{
			if (!Device) return false;
			UINT support = 0;
			const UINT required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_SHADER_SAMPLE
				| D3D11_FORMAT_SUPPORT_RENDER_TARGET | (requireMipChain ? D3D11_FORMAT_SUPPORT_MIP : 0);
			return SUCCEEDED(Device->CheckFormatSupport(DXGI_FORMAT_R32_FLOAT, &support)) && (support & required) == required;
		}

		bool EnsureSaoDepthHierarchy()
		{
			if (!SupportsFloatDepthTargets(true)) return Fail("DX11 SAO requires sampled and renderable R32_FLOAT textures");
			if (SaoDepthHierarchy.Texture && SaoDepthHierarchy.Width == CreateInfo.Width && SaoDepthHierarchy.Height == CreateInfo.Height)
				return true;
			DepthHierarchy candidate;
			candidate.Width = CreateInfo.Width;
			candidate.Height = CreateInfo.Height;
			candidate.MipCount = AmbientOcclusionDepthMipCount(CreateInfo.Width, CreateInfo.Height);
			D3D11_TEXTURE2D_DESC description {};
			description.Width = static_cast<UINT>(candidate.Width);
			description.Height = static_cast<UINT>(candidate.Height);
			description.MipLevels = candidate.MipCount;
			description.ArraySize = description.SampleDesc.Count = 1;
			description.Format = DXGI_FORMAT_R32_FLOAT;
			description.Usage = D3D11_USAGE_DEFAULT;
			description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			HRESULT result = Device->CreateTexture2D(&description, nullptr, candidate.Texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(SAO hierarchy)", result));
			result = Device->CreateShaderResourceView(candidate.Texture.Get(), nullptr, candidate.View.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(SAO hierarchy)", result));
			description.MipLevels = 1;
			description.BindFlags = D3D11_BIND_RENDER_TARGET;
			result = Device->CreateTexture2D(&description, nullptr, candidate.Scratch.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(SAO scratch)", result));
			result = Device->CreateRenderTargetView(candidate.Scratch.Get(), nullptr, candidate.ScratchTarget.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(SAO scratch)", result));
			SaoDepthHierarchy = std::move(candidate);
			return true;
		}

		#include "Renderer/DX11/DX11RayScene.inl"
		#include "Renderer/DX11/DX11RayRadiance.inl"
    #include "Renderer/DX11/DX11RayReflections.inl"
    #include "Renderer/DX11/DX11HybridRendering.inl"
    #include "Renderer/DX11/DX11SubsurfaceScattering.inl"
    #include "Renderer/DX11/DX11HairFur.inl"
    #include "Renderer/DX11/DX11TemporalEffects.inl"
    #include "Renderer/DX11/DX11ParallaxOcclusionMapping.inl"
    #include "Renderer/DX11/DX11DisplacementMapping.inl"
    #include "Renderer/DX11/DX11Tessellation.inl"
    #include "Renderer/VirtualGeometry/DX11/DX11VirtualGeometryCache.inl"
    #include "Renderer/VirtualGeometry/DX11/DX11VirtualGeometry.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteCache.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteGeometry.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteAsync.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteCacheAsync.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteFrameVisibility.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteDraws.inl"
#include "Renderer/DX11/DX11TemporalSuperResolution.inl"
#include "Renderer/VirtualShadowMaps/DX11/DX11VirtualShadowMaps.inl"
#include "Renderer/VirtualShadowMaps/DX11/DX11VirtualShadowAtlas.inl"
#include "Renderer/Lumen/DX11/DX11LumenSurface.inl"
#include "Renderer/Lumen/DX11/DX11LumenCaptureAsync.inl"
#include "Renderer/Lumen/DX11/DX11LumenSurfaceLookup.inl"
#include "Renderer/Lumen/DX11/DX11LumenDistanceTrace.inl"
#include "Renderer/Lumen/DX11/DX11LumenGlobalDistance.inl"
#include "Renderer/Lumen/DX11/DX11LumenGlobalDistanceAsync.inl"
#include "Renderer/Lumen/DX11/DX11LumenSurfaceLighting.inl"
#include "Renderer/Lumen/DX11/DX11LumenLightingAsync.inl"
#include "Renderer/Lumen/DX11/DX11LumenRadianceTrace.inl"
#include "Renderer/Lumen/DX11/DX11LumenProbeIntegration.inl"
#include "Renderer/Lumen/DX11/DX11LumenProbeGather.inl"
#include "Renderer/Lumen/DX11/DX11LumenTranslucency.inl"
#include "Renderer/Lumen/DX11/DX11LumenHairReceiver.inl"
#include "Renderer/DX11/DX11RayTracedShadows.inl"
#include "Renderer/DX11/DX11PCF.inl"
#include "Renderer/DX11/DX11Materials.inl"

    bool UsesMaterialSurface(const PipelineState& pipeline,PassType pass) const
    {
        return pass==PassType::Color && !pipeline.OutlineExtrusion &&
            FrameShadowFilter.Mode==Techniques::ShadowFilterMode::Hard &&
            !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive && !FrameRayShadowImage;
    }
    bool UsesMaterialStage(const PipelineState& pipeline,PassType pass) const
    { return (pass==PassType::GBuffer && !UsesParallaxMaterial(pipeline,pass)) || UsesMaterialSurface(pipeline,pass); }

    bool UsesDedicatedPCF() const
    { return FrameShadowFilter.Mode==Techniques::ShadowFilterMode::PCF && !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive; }
    bool UsesSurfacePCF(const PipelineState& pipeline,PassType pass) const
    { return UsesDedicatedPCF() && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap; }

#include "Renderer/Lumen/DX11/DX11LumenTemporal.inl"
#include "Renderer/Lumen/DX11/DX11LumenReflections.inl"
#include "Renderer/VirtualShadowMaps/DX11/DX11VirtualShadowReceivers.inl"
#include "Renderer/VirtualShadowMaps/DX11/DX11VirtualShadowLookup.inl"
    #include "Renderer/DX11/DX11RayGI.inl"
    #include "Renderer/DX11/DX11VoxelGI.inl"
    #include "Renderer/DX11/DX11PathTracing.inl"
    #include "Renderer/DX11/DX11GPUDriven.inl"
    #include "Renderer/DX11/DX11VisibilityBuffer.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteMaterials.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteFrameRaster.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteVisibilityGeometry.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteInstances.inl"
    #include "Renderer/Nanite/DX11/DX11NaniteHardware.inl"

		bool EnsureAmbientOcclusionResources()
		{
            const bool rayTraced = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::RayTraced;
            if (rayTraced && !EnsureRayScene()) return false;
			for (auto& surface : AmbientOcclusionSurfaces)
				if (!CreateHdrSurface(surface, CreateInfo.Width, CreateInfo.Height)) return false;
			const bool scalable = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO;
			if (scalable && !EnsureSaoDepthHierarchy()) return false;
			const std::size_t technique = AmbientOcclusionProgramIndex();
			if (AmbientOcclusionVertices[technique] && AmbientOcclusionPixels[technique] && AmbientOcclusionConstantBuffer && AmbientOcclusionBlend)
				return true;
			static constexpr std::array<const char*, AmbientOcclusionTechniqueCount> filenames {
				"SSAO.hlsl", "HBAO.hlsl", "SAO.hlsl", "GTAO.hlsl", "RayTracedAO.hlsl"};
			if (!LoadTechniqueShaderSource("DX11", filenames[technique], ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertexCode, pixelCode;
			if (!CompileShader("VSMain", "vs_5_0", vertexCode, nullptr, rayTraced) || !CompileShader("PSMain", "ps_5_0", pixelCode, nullptr, rayTraced)) return false;
			ComPtr<ID3D11VertexShader> vertex;
			ComPtr<ID3D11PixelShader> pixel;
			ComPtr<ID3D11Buffer> constants = AmbientOcclusionConstantBuffer;
			ComPtr<ID3D11BlendState> blend = AmbientOcclusionBlend;
			HRESULT result = Device->CreateVertexShader(vertexCode->GetBufferPointer(), vertexCode->GetBufferSize(), nullptr, vertex.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(ambient occlusion)", result));
			result = Device->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, pixel.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(ambient occlusion)", result));
			if (!constants)
			{
				D3D11_BUFFER_DESC buffer {};
				buffer.ByteWidth = sizeof(RayTracedAOConstants);
				buffer.Usage = D3D11_USAGE_DYNAMIC;
				buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
				buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				result = Device->CreateBuffer(&buffer, nullptr, constants.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(ambient occlusion)", result));
			}
			if (!blend)
			{
				D3D11_BLEND_DESC descriptor {};
				auto& target = descriptor.RenderTarget[0];
				target.BlendEnable = TRUE;
				target.SrcBlend = target.DestBlend = D3D11_BLEND_ONE;
				target.BlendOp = D3D11_BLEND_OP_REV_SUBTRACT;
				target.SrcBlendAlpha = D3D11_BLEND_ZERO;
				target.DestBlendAlpha = D3D11_BLEND_ONE;
				target.BlendOpAlpha = D3D11_BLEND_OP_ADD;
				target.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE;
				result = Device->CreateBlendState(&descriptor, blend.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateBlendState(ambient subtraction)", result));
			}
			AmbientOcclusionVertices[technique] = std::move(vertex);
			AmbientOcclusionPixels[technique] = std::move(pixel);
			AmbientOcclusionConstantBuffer = std::move(constants);
			AmbientOcclusionBlend = std::move(blend);
			return true;
		}

		std::size_t AmbientOcclusionProgramIndex() const
		{
			switch (FrameAmbientOcclusion.Mode)
			{
			case Techniques::AmbientOcclusionMode::HBAO: return 1;
			case Techniques::AmbientOcclusionMode::SAO: return 2;
			case Techniques::AmbientOcclusionMode::GTAO: return 3;
            case Techniques::AmbientOcclusionMode::RayTraced: return 4;
			default: return 0;
			}
		}

		bool EnsurePostResources()
		{
			if (PostTexture && SceneColorView) return true;
			D3D11_TEXTURE2D_DESC desc {};
			ColorTexture->GetDesc(&desc);
			ComPtr<ID3D11Texture2D> texture;
			ComPtr<ID3D11RenderTargetView> target;
			ComPtr<ID3D11ShaderResourceView> postView;
			ComPtr<ID3D11ShaderResourceView> sceneView;
			HRESULT result = Device->CreateTexture2D(&desc, nullptr, texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(post pingpong)", result));
			D3D11_RENDER_TARGET_VIEW_DESC rtv {};
			rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			rtv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			result = Device->CreateRenderTargetView(texture.Get(), &rtv, target.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(post)", result));
			D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
			srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			srv.Texture2D.MipLevels = 1;
			result = Device->CreateShaderResourceView(texture.Get(), &srv, postView.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(post)", result));
			result = Device->CreateShaderResourceView(ColorTexture.Get(), &srv, sceneView.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(scene color)", result));
			if (!PostConstantBuffer)
			{
				D3D11_BUFFER_DESC buffer {};
				buffer.ByteWidth = sizeof(PostProcessConstants);
				buffer.Usage = D3D11_USAGE_DYNAMIC;
				buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
				buffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				result = Device->CreateBuffer(&buffer, nullptr, PostConstantBuffer.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(post constants)", result));
			}
			if (!PostSampler)
			{
				D3D11_SAMPLER_DESC sampler {};
				sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
				sampler.MaxLOD = D3D11_FLOAT32_MAX;
				result = Device->CreateSamplerState(&sampler, PostSampler.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("CreateSamplerState(post)", result));
			}
			SceneColorView = std::move(sceneView);
			PostTexture = std::move(texture);
			PostTarget = std::move(target);
			PostView = std::move(postView);
			return true;
		}

		bool EnsurePostTechnique(Techniques::PostProcessPassKind pass)
		{
			const std::size_t index = static_cast<std::size_t>(pass);
			const char* name = PostProcessTechniqueFile(pass);
			if (index >= PostProcessTechniqueCount || !name) return Fail("Unknown DX11 post technique");
			if (PostVertexShaders[index] && PostPixelShaders[index]) return true;
			const std::string file = std::string(name) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX11", file.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			ComPtr<ID3D11VertexShader> vertexShader;
			ComPtr<ID3D11PixelShader> pixelShader;
			HRESULT result = Device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, vertexShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(post)", result));
			result = Device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, pixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(post)", result));
			PostVertexShaders[index] = std::move(vertexShader);
			PostPixelShaders[index] = std::move(pixelShader);
			return true;
		}

		bool ApplyPostProcessing(const Techniques::PostProcessSettings& settings, ID3D11Texture2D*& output)
		{
			output = ColorTexture.Get();
			const auto plan = Techniques::BuildPostProcessPlan(settings);
			if (plan.empty()) return true;
			if (!EnsurePostResources()) return false;
			for (auto pass : plan) if (!EnsurePostTechnique(pass)) return false;
			const PostProcessConstants constants = BuildPostProcessConstants(settings, CreateInfo.Width, CreateInfo.Height);
			D3D11_MAPPED_SUBRESOURCE mapped {};
			HRESULT result = Context->Map(PostConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("Map(post constants)", result));
			std::memcpy(mapped.pData, &constants, sizeof(constants));
			Context->Unmap(PostConstantBuffer.Get(), 0);
			ID3D11Buffer* constantBuffer = PostConstantBuffer.Get();
			ID3D11SamplerState* sampler = PostSampler.Get();
			Context->PSSetConstantBuffers(0, 1, &constantBuffer);
			Context->PSSetSamplers(0, 1, &sampler);
			Context->IASetInputLayout(nullptr);
			Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			Context->RSSetState(SolidRasterState.Get());
			Context->OMSetDepthStencilState(DepthDisabledState.Get(), 0);
			Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
			D3D11_VIEWPORT viewport {};
			viewport.Width = static_cast<float>(CreateInfo.Width);
			viewport.Height = static_cast<float>(CreateInfo.Height);
			viewport.MaxDepth = 1.0f;
			Context->RSSetViewports(1, &viewport);
			bool sourceIsMain = true;
			ID3D11ShaderResourceView* nullViews[4] {};
			for (auto pass : plan)
			{
				// A resource cannot be a sampled input and writable target at once.
				Context->PSSetShaderResources(0, 4, nullViews);
				Context->OMSetRenderTargets(0, nullptr, nullptr);
				ID3D11ShaderResourceView* source = sourceIsMain ? SceneColorView.Get() : PostView.Get();
				ID3D11RenderTargetView* target = sourceIsMain ? PostTarget.Get() : ColorTarget.Get();
				Context->OMSetRenderTargets(1, &target, nullptr);
				Context->PSSetShaderResources(0, 1, &source);
				const auto index = static_cast<std::size_t>(pass);
				Context->VSSetShader(PostVertexShaders[index].Get(), nullptr, 0);
				Context->PSSetShader(PostPixelShaders[index].Get(), nullptr, 0);
				Context->Draw(3, 0);
				sourceIsMain = !sourceIsMain;
				PendingFrame.PostProcessingPasses.push_back(pass);
			}
			Context->PSSetShaderResources(0, 4, nullViews);
			Context->OMSetRenderTargets(0, nullptr, nullptr);
			PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
			output = sourceIsMain ? ColorTexture.Get() : PostTexture.Get();
			return true;
		}

        struct SharedSurface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<IDXGIKeyedMutex> Mutex;
            HANDLE Handle = nullptr; // Legacy DXGI handle; owned by Texture.
            UINT Width = 0, Height = 0;
        };
        std::vector<std::shared_ptr<SharedSurface>> SharedSurfaces;

        bool PublishGpuColor(ID3D11Texture2D* source)
        {
            D3D11_TEXTURE2D_DESC desc{}; source->GetDesc(&desc);
            std::shared_ptr<SharedSurface> surface;
            for (const auto& candidate : SharedSurfaces)
                if (candidate.use_count() == 1 && candidate->Width == desc.Width && candidate->Height == desc.Height)
                { surface = candidate; break; }
            if (!surface)
            {
                // Retired frames keep their surface alive across resize and backend switches.
                SharedSurfaces.erase(std::remove_if(SharedSurfaces.begin(), SharedSurfaces.end(),
                    [](const auto& item) { return item.use_count() == 1; }), SharedSurfaces.end());
                surface = std::make_shared<SharedSurface>();
                desc.Usage = D3D11_USAGE_DEFAULT; desc.CPUAccessFlags = 0;
                desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
                HRESULT status = Device->CreateTexture2D(&desc, nullptr, surface->Texture.GetAddressOf());
                if (FAILED(status)) return Fail(HResultMessage("Create GPU viewport surface", status));
                ComPtr<IDXGIResource> resource;
                if (FAILED(surface->Texture.As(&resource)) || FAILED(surface->Texture.As(&surface->Mutex)) ||
                    FAILED(resource->GetSharedHandle(&surface->Handle))) return Fail("Cannot share GPU viewport surface");
                surface->Width = desc.Width; surface->Height = desc.Height;
                SharedSurfaces.push_back(surface);
            }
            auto status = surface->Mutex->AcquireSync(0, 5000);
            if (status != S_OK) return Fail(HResultMessage("Acquire GPU viewport producer", status));
            Context->OMSetRenderTargets(0, nullptr, nullptr);
            Context->CopyResource(surface->Texture.Get(), source);
            status = surface->Mutex->ReleaseSync(0);
            Context->Flush(); // Submit the copy before handing the surface to another device.
            if (FAILED(status)) return Fail(HResultMessage("Release GPU viewport producer", status));
            auto image = std::make_shared<GpuImage>();
            image->SharedHandle = surface->Handle; image->Owner = surface;
            PendingFrame.GpuColor = std::move(image);
            return true;
        }

		bool ReadColor(ID3D11Texture2D* source, std::vector<std::uint8_t>& output)
		{
			Context->CopyResource(ColorStaging.Get(), source);
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(ColorStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result))
            {
                auto error=HResultMessage("ID3D11DeviceContext::Map(color)",result);
                const auto removed=Device->GetDeviceRemovedReason();
                if (FAILED(removed)) error+="; "+HResultMessage("device removal",removed);
                return Fail(error);
            }
			output.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height * 4);
			const std::size_t rowSize = static_cast<std::size_t>(CreateInfo.Width) * 4;
			for (int row = 0; row < CreateInfo.Height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * rowSize,
					static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(ColorStaging.Get(), 0);
			return true;
		}

		bool ReadOutlineMask(std::vector<std::uint16_t>& output)
		{
			Context->CopyResource(OutlineMaskStaging.Get(), OutlineMaskTexture.Get());
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(
				OutlineMaskStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11DeviceContext::Map(outline mask)", result));
			output.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height * 4);
			const std::size_t rowSize =
				static_cast<std::size_t>(CreateInfo.Width) * 4 * sizeof(std::uint16_t);
			for (int row = 0; row < CreateInfo.Height; ++row)
			{
				std::memcpy(
					reinterpret_cast<std::uint8_t*>(output.data())
						+ static_cast<std::size_t>(row) * rowSize,
					static_cast<const std::uint8_t*>(mapped.pData)
						+ static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(OutlineMaskStaging.Get(), 0);
			return true;
		}

		void ApplyOutline(
			const std::vector<std::uint16_t>& mask,
			std::vector<std::uint8_t>& color) const
		{
			const int width = CreateInfo.Width;
			const int height = CreateInfo.Height;
			const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
			if (width <= 0 || height <= 0
				|| mask.size() != pixelCount * 4
				|| color.size() != pixelCount * 4)
			{
				return;
			}

			std::vector<std::uint32_t> bestDistance(
				pixelCount, std::numeric_limits<std::uint32_t>::max());
			std::vector<std::uint32_t> bestSource(
				pixelCount, std::numeric_limits<std::uint32_t>::max());
			const auto radiusAt = [&mask](std::size_t pixel)
			{
				return static_cast<int>(mask[pixel * 4 + 3] >> 8);
			};
			for (int sourceY = 0; sourceY < height; ++sourceY)
			{
				for (int sourceX = 0; sourceX < width; ++sourceX)
				{
					const std::size_t sourcePixel =
						static_cast<std::size_t>(sourceY) * width + sourceX;
					const int radius = radiusAt(sourcePixel);
					if (radius <= 0)
					{
						continue;
					}

					bool boundary = false;
					for (int neighborY = -1; neighborY <= 1 && !boundary; ++neighborY)
					{
						for (int neighborX = -1; neighborX <= 1; ++neighborX)
						{
							if (neighborX == 0 && neighborY == 0)
							{
								continue;
							}
							const int x = sourceX + neighborX;
							const int y = sourceY + neighborY;
							if (x < 0 || x >= width || y < 0 || y >= height
								|| radiusAt(static_cast<std::size_t>(y) * width + x) == 0)
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
					const int maximumX = std::min(width - 1, sourceX + radius);
					const int minimumY = std::max(0, sourceY - radius);
					const int maximumY = std::min(height - 1, sourceY + radius);
					const std::uint32_t radiusSquared =
						static_cast<std::uint32_t>(radius * radius);
					for (int targetY = minimumY; targetY <= maximumY; ++targetY)
					{
						for (int targetX = minimumX; targetX <= maximumX; ++targetX)
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
								static_cast<std::size_t>(targetY) * width + targetX;
							if (radiusAt(targetPixel) != 0
								|| distance >= bestDistance[targetPixel])
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

			for (std::size_t targetPixel = 0;
				targetPixel < pixelCount;
				++targetPixel)
			{
				if (bestSource[targetPixel] ==
					std::numeric_limits<std::uint32_t>::max())
				{
					continue;
				}
				const std::size_t sourcePixel = bestSource[targetPixel];
				const std::uint16_t packedOutline =
					mask[sourcePixel * 4 + 3];
				const float opacity =
					static_cast<float>(packedOutline & 0xffu) / 255.0f;
				if (opacity <= 0.0f)
				{
					continue;
				}
				for (std::size_t channel = 0; channel < 3; ++channel)
				{
					const float sourceSRGB =
						static_cast<float>(mask[sourcePixel * 4 + channel])
						/ 65535.0f;
					const float destinationSRGB =
						static_cast<float>(color[targetPixel * 4 + channel])
						/ 255.0f;
					const float blendedLinear =
						SRGBToLinearChannel(sourceSRGB) * opacity +
						SRGBToLinearChannel(destinationSRGB) *
							(1.0f - opacity);
					color[targetPixel * 4 + channel] =
						static_cast<std::uint8_t>(std::lround(
							std::clamp(
								LinearToSRGBChannel(blendedLinear),
								0.0f,
								1.0f) * 255.0f));
				}
				const float destinationAlpha =
					static_cast<float>(color[targetPixel * 4 + 3]) / 255.0f;
				color[targetPixel * 4 + 3] =
					static_cast<std::uint8_t>(std::lround(
						std::clamp(
							opacity + destinationAlpha * (1.0f - opacity),
							0.0f,
							1.0f) * 255.0f));
			}
		}

		bool ReadDepth(ID3D11Texture2D* source, ID3D11Texture2D* staging,
			int width, int height, std::vector<float>& output)
		{
			Context->CopyResource(staging, source);
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("ID3D11DeviceContext::Map(depth)", result));
			output.resize(static_cast<std::size_t>(width) * height);
			const std::size_t rowSize = static_cast<std::size_t>(width) * sizeof(float);
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * width,
					static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(staging, 0);
			return true;
		}

		void PresentToWindow(const FrameCapture& capture) const
		{
			if (CreateInfo.NativeWindow == nullptr || !capture.HasColorOutput()) return;
			HWND window = static_cast<HWND>(CreateInfo.NativeWindow);
			if (!IsWindow(window)) return;
			std::vector<std::uint8_t> bgra(capture.ColorRGBA8.size());
			for (std::size_t pixel = 0; pixel < capture.ColorRGBA8.size() / 4; ++pixel)
			{
				bgra[pixel * 4] = capture.ColorRGBA8[pixel * 4 + 2];
				bgra[pixel * 4 + 1] = capture.ColorRGBA8[pixel * 4 + 1];
				bgra[pixel * 4 + 2] = capture.ColorRGBA8[pixel * 4];
				bgra[pixel * 4 + 3] = 255;
			}
			BITMAPINFO bitmap = {};
			bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			bitmap.bmiHeader.biWidth = static_cast<LONG>(capture.Width);
			bitmap.bmiHeader.biHeight = -static_cast<LONG>(capture.Height);
			bitmap.bmiHeader.biPlanes = 1;
			bitmap.bmiHeader.biBitCount = 32;
			bitmap.bmiHeader.biCompression = BI_RGB;
			RECT client = {};
			GetClientRect(window, &client);
			HDC dc = GetDC(window);
			if (dc != nullptr)
			{
				StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0,
					static_cast<int>(capture.Width), static_cast<int>(capture.Height), bgra.data(),
					&bitmap, DIB_RGB_COLORS, SRCCOPY);
				ReleaseDC(window, dc);
			}
		}

		bool NeedsSampleColorRestore() const
		{
			return SubsurfaceRequested || SubsurfaceResolved || GIActive || AmbientOcclusionActive || ReflectionActive || Techniques::UsesVolumetricFog(FrameFog) || Techniques::UsesLightShafts(FrameLightShafts) ||
				(FramePostProcessing.MotionBlur.Enabled || Techniques::UsesTemporalAA(FramePostProcessing)) || FramePostProcessing.DepthOfField.Enabled;
		}
		bool OpaqueEffectsComplete() const
		{
			return SubsurfaceComplete() && ((!GIActive || GI.Resolved) && (!AmbientOcclusionActive || AmbientOcclusionApplied) && (!ReflectionActive || ReflectionApplied) &&
				(!Techniques::UsesVolumetricFog(FrameFog) || VolumetricFog.Resolved) &&
				(!Techniques::UsesLightShafts(FrameLightShafts) || LightShafts.Resolved) &&
				(!FramePostProcessing.MotionBlur.Enabled || MotionBlurResolved) &&
				(!Techniques::UsesTemporalAA(FramePostProcessing) || TemporalAA.Resolved) &&
				(!FramePostProcessing.DepthOfField.Enabled || DepthOfFieldResolved));
		}
		DX11Multisampling Multisampling;
		DX11TemporalAA TemporalAA;
        DX11GlobalIllumination GI;
        DX11LightProbes Probes;
        DX11DepthPeeling Peeling;
        DepthPeelingFrame FramePeeling;
        DX11Refraction Refraction;
        RefractionFrame FrameRefraction;
        Techniques::RefractionSettings FrameRefractionSettings;
        bool RefractionActive=false,RefractionResolved=false;
        DX11Lightmaps Lightmaps;
        LightmapFrame FrameLightmaps;
        Techniques::Settings FrameLightmapSettings;
        bool LightmapActive = false;
        Techniques::GlobalIlluminationSettings FrameGI;
        bool GIActive = false, EarlyTemporalActive = false, ProbeActive = false;
		DX11VolumetricFog VolumetricFog;
		DX11VolumetricLighting VolumetricLighting;
		DX11LightShafts LightShafts;
		Techniques::LightShaftSettings FrameLightShafts;
		Techniques::VolumetricLightingSettings FrameVolumeLighting;
		DX11WeightedTransparency WeightedTransparency;
		Techniques::TransparencySettings FrameTransparency;
		bool WeightedActive = false, WeightedPass = false;
		Techniques::VolumetricFogSettings FrameFog;
		UINT FrameSampleCount = 1;
		bool SamplesOpaqueResolved = false, SamplesVelocityResolved = false;
		bool SamplesColorRestored = false, SamplesColorResolved = false, SamplesTransparentWritten = false;
		ComPtr<ID3D11PixelShader> DeferredMultisamplePixel;
		RHICreateInfo CreateInfo;
		ComPtr<ID3D11Device> Device;
		ComPtr<ID3D11DeviceContext> Context;
		D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_11_0;
		D3D_DRIVER_TYPE DriverType = D3D_DRIVER_TYPE_UNKNOWN;
		ComPtr<ID3D11Texture2D> ColorTexture;
		ComPtr<ID3D11RenderTargetView> ColorTarget;
		ComPtr<ID3D11Texture2D> ColorStaging;
		ComPtr<ID3D11ShaderResourceView> SceneColorView;
		ComPtr<ID3D11Texture2D> PostTexture;
		ComPtr<ID3D11RenderTargetView> PostTarget;
		ComPtr<ID3D11ShaderResourceView> PostView;
		ComPtr<ID3D11Buffer> PostConstantBuffer;
		ComPtr<ID3D11SamplerState> PostSampler;
		std::array<HdrSurface, HdrSurfaceCount> HdrSurfaces;
		std::array<ComPtr<ID3D11VertexShader>, HdrProgramCount> HdrVertexShaders;
		std::array<ComPtr<ID3D11PixelShader>, HdrProgramCount> HdrPixelShaders;
		ComPtr<ID3D11Buffer> HdrConstantBuffer;
		ComPtr<ID3D11SamplerState> HdrSampler;
		Techniques::PostProcessSettings FramePostProcessing;
		bool DepthOfFieldResolved = false;
		bool VelocityCaptured = false, MotionBlurResolved = false;
		static constexpr std::size_t MotionVelocity = 0, MotionTileMax = 1, MotionNeighborMax = 2;
		std::array<HdrSurface, 3> MotionSurfaces;
		ComPtr<ID3D11Texture2D> MotionStaging;
		ComPtr<ID3D11VertexShader> MotionGeometryVertex, MotionScreenVertex;
		ComPtr<ID3D11PixelShader> MotionGeometryPixel, MotionScreenPixel;
		ComPtr<ID3D11Buffer> MotionConstantBuffer;
		ComPtr<ID3D11BlendState> MotionCompositeMask;
		ComPtr<ID3D11VertexShader> DepthOfFieldVertex;
		ComPtr<ID3D11PixelShader> DepthOfFieldPixel;
		ComPtr<ID3D11Buffer> DepthOfFieldConstantBuffer;
		ComPtr<ID3D11BlendState> DepthOfFieldCompositeWriteMask;
		bool HdrSceneActive = false;
		bool CurrentColorIsHdr = false;
		std::array<ComPtr<ID3D11VertexShader>, PostProcessTechniqueCount> PostVertexShaders;
		std::array<ComPtr<ID3D11PixelShader>, PostProcessTechniqueCount> PostPixelShaders;
		ComPtr<ID3D11Texture2D> OutlineMaskTexture;
		ComPtr<ID3D11RenderTargetView> OutlineMaskTarget;
		ComPtr<ID3D11Texture2D> OutlineMaskStaging;
		ComPtr<ID3D11Texture2D> DepthTexture;
		ComPtr<ID3D11DepthStencilView> DepthTarget;
		ComPtr<ID3D11Texture2D> DepthStaging;
		ComPtr<ID3D11ShaderResourceView> DepthShaderView;
		std::array<HdrSurface, GBufferAttachmentCount> GBufferSurfaces;
		ComPtr<ID3D11VertexShader> DeferredGeometryVertex, DeferredLightVertex;
		ComPtr<ID3D11PixelShader> DeferredGeometryPixel, DeferredLightPixel;
		ComPtr<ID3D11Buffer> DeferredConstantBuffer;
		bool DeferredActive = false;
		bool MaterialCaptureActive = false;
		bool ReflectionActive = false, ReflectionApplied = false;
		Techniques::ReflectionSettings FrameReflections;
		Techniques::ShadingModel FrameReflectionShading = Techniques::ShadingModel::Phong;
		HdrSurface ReflectionSurface;
		std::array<ComPtr<ID3D11VertexShader>, ReflectionProgramCount> ReflectionVertex;
		std::array<ComPtr<ID3D11PixelShader>, ReflectionProgramCount> ReflectionPixel;
		HdrSurface EnvironmentMap;
		UINT EnvironmentMapMips = 0;
		ComPtr<ID3D11PixelShader> EnvironmentMapConvert;
		ComPtr<ID3D11SamplerState> EnvironmentMapSampler;
		EnvironmentTargets Environment;
		EnvironmentCaptureFrame FrameEnvironment;
		UINT EnvironmentFacesWritten = 0;
		bool EnvironmentActive = false, EnvironmentReady = false;
		PlanarTargets Planar;
		PlanarCaptureFrame FramePlanar;
		bool PlanarActive = false, PlanarWritten = false, PlanarReady = false;
		ComPtr<ID3D11PixelShader> PlanarMipPixel;
		ComPtr<ID3D11Buffer> ReflectionConstantBuffer;
		ComPtr<ID3D11BlendState> ReflectionBlend;
		bool AmbientOcclusionActive = false;
		bool AmbientOcclusionApplied = false;
		bool SceneColorWritten = false;
		std::array<HdrSurface, 2> AmbientOcclusionSurfaces;
		DepthHierarchy SaoDepthHierarchy;
		static constexpr std::size_t AmbientOcclusionTechniqueCount = 5;
		Techniques::AmbientOcclusionSettings FrameAmbientOcclusion;
		std::array<ComPtr<ID3D11VertexShader>, AmbientOcclusionTechniqueCount> AmbientOcclusionVertices;
		std::array<ComPtr<ID3D11PixelShader>, AmbientOcclusionTechniqueCount> AmbientOcclusionPixels;
		ComPtr<ID3D11Buffer> AmbientOcclusionConstantBuffer;
		ComPtr<ID3D11BlendState> AmbientOcclusionBlend;
		bool GBufferWritten = false;
		bool DeferredResolved = false;
		Float4 DeferredBackground;
		std::array<ComPtr<ID3D11VertexShader>, StandaloneShadingTechniqueCount> VertexShaders;
		std::array<ComPtr<ID3D11PixelShader>, StandaloneShadingTechniqueCount> PixelShaders;
		std::array<ComPtr<ID3D11PixelShader>, StandaloneShadingTechniqueCount> WeightedPixelShaders;
		ComPtr<ID3D11VertexShader> ShadowVertexShader;
		ComPtr<ID3D11PixelShader> ShadowPixelShader;
		ComPtr<ID3D11VertexShader> CascadeVertexShader;
		ComPtr<ID3D11PixelShader> CascadePixelShader;
		std::array<ComPtr<ID3D11Buffer>, 2> CascadeConstantBuffers;
        Techniques::ShadowFilterSettings FrameShadowFilter;
        DX11PCSS PCSS;
        DX11VSM VSM;
        DX11ContactShadows ContactShadows;
        DX11ForwardPlus ForwardPlus;
        DX11ForwardPlus TiledDeferred {TiledLightShader::TiledDeferred};
        bool TiledActive = false;
        DX11ForwardPlus& TileProducer() { return TiledActive ? TiledDeferred : ForwardPlus; }
        DX11ClusteredForward ClusteredForward;
        bool ClusteredActive = false;
        LightListFrameState LightListFrame;
        bool LightListsActive = false, ForwardPlusDepthWritten = false;
        Techniques::ContactShadowSettings FrameContact;
        bool ContactDepthWritten = false, ContactGenerated = false, ContactReceiverActive = false;
		CascadedShadowState FrameCascades;
		ComPtr<ID3D11PixelShader> OutlinePixelShader;
		ComPtr<ID3D11InputLayout> InputLayout;
		std::array<ComPtr<ID3D11VertexShader>, InstanceVertexProgramCount> InstanceVertices;
		ComPtr<ID3D11Buffer> InstanceBuffer;
		ComPtr<ID3D11Buffer> ConstantBuffer;
		std::unordered_map<std::uint32_t, ComPtr<ID3D11RasterizerState>> CulledRasterStates;
		ComPtr<ID3D11RasterizerState> SolidRasterState;
		ComPtr<ID3D11RasterizerState> ShadowRasterState;
		ComPtr<ID3D11RasterizerState> WireRasterState;
		ComPtr<ID3D11RasterizerState> OutlineRasterState;
		ComPtr<ID3D11DepthStencilState> DepthWriteState;
		ComPtr<ID3D11DepthStencilState> DepthReadState;
		ComPtr<ID3D11DepthStencilState> DepthDisabledState;
		ComPtr<ID3D11BlendState> ColorWriteDisabledState;
		ComPtr<ID3D11BlendState> AlphaBlendState;
		ComPtr<ID3D11SamplerState> SurfaceSampler, AnisotropicSampler;
		bool AnisotropicFiltering = false;
		ID3D11SamplerState* MaterialSampler() const
		{
			return AnisotropicFiltering ? AnisotropicSampler.Get() : SurfaceSampler.Get();
		}
		std::unordered_map<std::uint32_t, BufferResource> Buffers;
		std::unordered_map<std::uint32_t, TextureResource> Textures;
		FrameCapture PendingFrame;
		FrameCapture LastFrame;
		PassDesc CurrentPass;
		PipelineState BoundPipeline;
		Matrix4 BoundWorld;
		Matrix4 BoundViewProjection;
		Matrix4 BoundLightViewProjection;
		BufferHandle BoundVertexBuffer;
		BufferHandle BoundIndexBuffer;
		TextureHandle LastShadowTexture;
		std::string ShaderCode;
		std::string LastError;
		std::thread::id OwnerThread;
		std::uint32_t NextBuffer = 0x11000001u;
		std::uint32_t NextTexture = 0x11800001u;
		std::uint64_t FrameNumber = 0;
		bool Initialized = false;
		bool FrameAreaLights = false;
        bool FrameOpen = false;
		bool PassOpen = false;
		bool PipelineBound = false;
		bool WorldBound = false;
		bool ViewProjectionBound = false;
		bool LightViewProjectionBound = false;
		bool HasOutlineMaskThisFrame = false;
	};

	DX11RHI::DX11RHI() : m_Impl(std::make_unique<Impl>()) {}
	DX11RHI::~DX11RHI() { Shutdown(); }

	bool DX11RHI::Initialize(const RHICreateInfo& createInfo)
	{
		if (m_Impl->Initialized) return m_Impl->Fail("DX11 RHI is already initialized");
		if (createInfo.Width <= 0 || createInfo.Height <= 0) return m_Impl->Fail("DX11 dimensions must be positive");
		const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
		const D3D_DRIVER_TYPE drivers[] = { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP };
		auto createDevice = [&](D3D_DRIVER_TYPE driver)
		{
			m_Impl->Device.Reset();
			m_Impl->Context.Reset();
			HRESULT result = D3D11CreateDevice(nullptr, driver, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
				levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
				m_Impl->Device.GetAddressOf(), &m_Impl->FeatureLevel,
				m_Impl->Context.GetAddressOf());
			if (result == E_INVALIDARG)
			{
				m_Impl->Device.Reset();
				m_Impl->Context.Reset();
				result = D3D11CreateDevice(nullptr, driver, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
					&levels[1], 1, D3D11_SDK_VERSION, m_Impl->Device.GetAddressOf(),
					&m_Impl->FeatureLevel, m_Impl->Context.GetAddressOf());
			}
			return result;
		};
		HRESULT lastResult = E_FAIL;
		for (D3D_DRIVER_TYPE driver : drivers)
		{
			lastResult = createDevice(driver);
			if (SUCCEEDED(lastResult))
			{
				m_Impl->DriverType = driver;
				break;
			}
		}
		if (FAILED(lastResult)) return m_Impl->Fail(HResultMessage("D3D11CreateDevice", lastResult));
		m_Impl->CreateInfo = createInfo;
		m_Impl->OwnerThread = std::this_thread::get_id();
		m_Impl->Initialized = true;
		if (!m_Impl->CreatePipelineObjects() || !m_Impl->CreateOffscreenTargets(createInfo.Width, createInfo.Height))
		{
			const std::string error = m_Impl->LastError;
			Shutdown();
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->LastError.clear();
		return true;
	}

	void DX11RHI::Shutdown()
	{
		if (!m_Impl) return;
		if (m_Impl->Context) m_Impl->Context->ClearState();
		m_Impl->Buffers.clear();
		m_Impl->Textures.clear();
		m_Impl->Multisampling = {};
		m_Impl->VolumetricFog = {}; m_Impl->FrameFog = {};
		m_Impl->VolumetricLighting = {}; m_Impl->FrameVolumeLighting = {};
        m_Impl->PCSS = {};
        m_Impl->VSM = {};
        m_Impl->ContactShadows = {};
        m_Impl->ForwardPlus.Shutdown(); m_Impl->TiledDeferred.Shutdown(); m_Impl->ClusteredForward.Shutdown(); m_Impl->LightListFrame.Reset();
        m_Impl->LightListsActive = m_Impl->ForwardPlusDepthWritten = false;
		m_Impl->LightShafts = {}; m_Impl->FrameLightShafts = {};
		m_Impl->WeightedTransparency = {}; m_Impl->WeightedActive = m_Impl->WeightedPass = false;
		m_Impl->DeferredMultisamplePixel.Reset();
		m_Impl->FrameSampleCount = 1;
		m_Impl->SharedSurfaces.clear();
		m_Impl->ColorTexture.Reset();
		m_Impl->ColorTarget.Reset();
		m_Impl->ColorStaging.Reset();
		m_Impl->SceneColorView.Reset();
		m_Impl->PostTexture.Reset();
		m_Impl->PostTarget.Reset();
		m_Impl->PostView.Reset();
		m_Impl->PostConstantBuffer.Reset();
		m_Impl->PostSampler.Reset();
		for (auto& surface : m_Impl->HdrSurfaces) surface = {};
		for (auto& shader : m_Impl->HdrVertexShaders) shader.Reset();
		for (auto& shader : m_Impl->HdrPixelShaders) shader.Reset();
		m_Impl->HdrConstantBuffer.Reset();
		m_Impl->HdrSampler.Reset();
		m_Impl->ReflectionSurface = {};
		m_Impl->ReflectionVertex = {}; m_Impl->ReflectionPixel = {};
		m_Impl->Environment = {}; m_Impl->FrameEnvironment = {};
		m_Impl->Planar = {}; m_Impl->FramePlanar = {};
		m_Impl->PlanarMipPixel.Reset();
		m_Impl->PlanarActive = m_Impl->PlanarWritten = m_Impl->PlanarReady = false;
		SetPlanarCaptureFrame({});
		m_Impl->EnvironmentMap = {}; m_Impl->EnvironmentMapMips = 0;
		m_Impl->EnvironmentMapConvert.Reset(); m_Impl->EnvironmentMapSampler.Reset();
		m_Impl->EnvironmentActive = m_Impl->EnvironmentReady = false;
		m_Impl->EnvironmentFacesWritten = 0;
		SetEnvironmentCaptureFrame({});
		m_Impl->ReflectionConstantBuffer.Reset(); m_Impl->ReflectionBlend.Reset();
		m_Impl->FrameReflections = {};
		m_Impl->DepthOfFieldVertex.Reset();
		m_Impl->MotionGeometryVertex.Reset();
		m_Impl->MotionGeometryPixel.Reset();
		m_Impl->MotionScreenVertex.Reset();
		m_Impl->MotionScreenPixel.Reset();
		m_Impl->MotionConstantBuffer.Reset();
		m_Impl->MotionCompositeMask.Reset();
		m_Impl->MotionStaging.Reset();
		for (auto& surface : m_Impl->MotionSurfaces) surface = {};
		m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
		m_Impl->DepthOfFieldPixel.Reset();
		m_Impl->DepthOfFieldConstantBuffer.Reset();
		m_Impl->DepthOfFieldCompositeWriteMask.Reset();
		m_Impl->FramePostProcessing = {};
        m_Impl->FrameGI = {}; m_Impl->GIActive = false; m_Impl->GI = {}; m_Impl->Probes = {};
        m_Impl->Lightmaps = {}; m_Impl->FrameLightmaps = {}; m_Impl->LightmapActive = false;
        m_Impl->Peeling = {}; m_Impl->FramePeeling = {};
        m_Impl->Refraction = {}; m_Impl->FrameRefraction = {}; m_Impl->RefractionActive=m_Impl->RefractionResolved=false;
		m_Impl->DepthOfFieldResolved = false;
		for (auto& shader : m_Impl->PostVertexShaders) shader.Reset();
		for (auto& shader : m_Impl->PostPixelShaders) shader.Reset();
		m_Impl->OutlineMaskTexture.Reset();
		m_Impl->OutlineMaskTarget.Reset();
		m_Impl->OutlineMaskStaging.Reset();
		m_Impl->DepthTexture.Reset();
		m_Impl->DepthTarget.Reset();
		m_Impl->DepthStaging.Reset();
		m_Impl->DepthShaderView.Reset();
		for (auto& surface : m_Impl->GBufferSurfaces) surface = {};
		m_Impl->DeferredGeometryVertex.Reset();
		m_Impl->DeferredGeometryPixel.Reset();
		m_Impl->DeferredLightVertex.Reset();
		m_Impl->DeferredLightPixel.Reset();
		m_Impl->DeferredConstantBuffer.Reset();
		for (auto& surface : m_Impl->AmbientOcclusionSurfaces) surface = {};
		m_Impl->SaoDepthHierarchy = {};
		for (auto& shader : m_Impl->AmbientOcclusionVertices) shader.Reset();
		for (auto& shader : m_Impl->AmbientOcclusionPixels) shader.Reset();
		m_Impl->AmbientOcclusionConstantBuffer.Reset();
        m_Impl->DestroyPathTracing();
        m_Impl->DestroyRayGI(); m_Impl->DestroyVoxelGI();
    m_Impl->DestroyRayReflections(); m_Impl->DestroyHybridRendering(); m_Impl->DestroySubsurface(); m_Impl->DestroyHairFur(); m_Impl->DestroyExposure(); m_Impl->DestroyParallax(); m_Impl->DestroyDisplacement(); m_Impl->DestroyTessellation(); m_Impl->DestroyVirtualGeometry(); m_Impl->DestroyNaniteFrameVisibility(); m_Impl->DestroyNaniteGeometry(); m_Impl->DestroyTSR(); m_Impl->DestroyVirtualShadowRequests(); m_Impl->DestroyVirtualShadowAtlases(); m_Impl->DestroyLumenProbes(); m_Impl->DestroyLumenRadianceTraces(); m_Impl->DestroyLumenLightingCaches(); m_Impl->DestroyLumenSurfaces(); m_Impl->DestroyLumenSurfaceLookup(); m_Impl->DestroyLumenDistanceScenes(); m_Impl->DestroyRayScene(); m_Impl->DestroyRayShadowImage(); m_Impl->DestroyPCF(); m_Impl->DestroyMaterials();
		m_Impl->AmbientOcclusionBlend.Reset();
		m_Impl->ReflectionActive = m_Impl->ReflectionApplied = false;
		m_Impl->MaterialCaptureActive = m_Impl->AmbientOcclusionActive = m_Impl->AmbientOcclusionApplied = m_Impl->SceneColorWritten = false;
		m_Impl->DeferredActive = m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
		for (auto& shader : m_Impl->VertexShaders) shader.Reset();
		for (auto& shader : m_Impl->PixelShaders) shader.Reset();
		for (auto& shader : m_Impl->WeightedPixelShaders) shader.Reset();
		m_Impl->CulledRasterStates.clear();
		m_Impl->ShadowVertexShader.Reset();
		m_Impl->ShadowPixelShader.Reset();
		m_Impl->CascadeVertexShader.Reset();
		m_Impl->CascadePixelShader.Reset();
		for (auto& buffer : m_Impl->CascadeConstantBuffers) buffer.Reset();
		m_Impl->FrameCascades = {};
		SetCascadedShadowState({});
        m_Impl->ContactShadows.BeginFrame();
        m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
		m_Impl->OutlinePixelShader.Reset();
		m_Impl->InputLayout.Reset();
		for (auto& shader : m_Impl->InstanceVertices) shader.Reset();
		m_Impl->InstanceBuffer.Reset();
        m_Impl->DestroyGpuDriven();
        m_Impl->DestroyVisibilityBuffer();
		m_Impl->ConstantBuffer.Reset();
		m_Impl->SolidRasterState.Reset();
		m_Impl->ShadowRasterState.Reset();
		m_Impl->WireRasterState.Reset();
		m_Impl->OutlineRasterState.Reset();
		m_Impl->DepthWriteState.Reset();
		m_Impl->DepthReadState.Reset();
		m_Impl->DepthDisabledState.Reset();
		m_Impl->ColorWriteDisabledState.Reset();
		m_Impl->AlphaBlendState.Reset();
		m_Impl->SurfaceSampler.Reset();
		m_Impl->AnisotropicSampler.Reset();
		m_Impl->Context.Reset();
		m_Impl->Device.Reset();
		m_Impl->Initialized = false;
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->HasOutlineMaskThisFrame = false;
		m_Impl->LastShadowTexture = {};
		m_Impl->OwnerThread = {};
	}

	bool DX11RHI::IsInitialized() const { return m_Impl->Initialized; }
	RHIBackend DX11RHI::GetBackend() const { return RHIBackend::DX11; }
	const char* DX11RHI::GetName() const { return "DirectX 11 Native RHI"; }
	RHICapabilities DX11RHI::GetCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = true;
		capabilities.MaterialAnisotropicFiltering = m_Impl->Initialized && bool(m_Impl->AnisotropicSampler);
    capabilities.TemporalEffects=m_Impl->Initialized && m_Impl->SupportsTemporalExposure();
		capabilities.InstancedRendering = true;
        capabilities.GPUDrivenRendering = m_Impl->Initialized;
        capabilities.VisibilityBufferRendering = m_Impl->Initialized && m_Impl->SupportsVisibilityBuffer();
		capabilities.DeferredShading = true;
    capabilities.ParallaxOcclusion=m_Impl->Initialized && capabilities.DeferredShading;
        capabilities.GeometryDisplacement=m_Impl->Initialized;
        capabilities.GeometryTessellation=m_Impl->Initialized;
        capabilities.GeometryVirtualization=m_Impl->Initialized;
        capabilities.NaniteRendering=capabilities.VisibilityBufferRendering && capabilities.GeometryVirtualization && m_Impl->Device->GetFeatureLevel()>=D3D_FEATURE_LEVEL_11_0;
        capabilities.NaniteHardwareScheduling=capabilities.NaniteRendering;
        capabilities.VirtualShadowMaps=m_Impl->Initialized && m_Impl->Device->GetFeatureLevel()>=D3D_FEATURE_LEVEL_11_0;
		capabilities.ScalableAmbientObscurance = m_Impl->SupportsFloatDepthTargets(true);
		capabilities.TemporalAntiAliasing = m_Impl->SupportsFloatDepthTargets(false);
        capabilities.TemporalSuperResolution=m_Impl->Initialized && capabilities.TemporalAntiAliasing && m_Impl->Device->GetFeatureLevel()>=D3D_FEATURE_LEVEL_11_0;
        capabilities.ScreenSpaceGlobalIllumination = capabilities.DeferredShading && capabilities.TemporalAntiAliasing;
        capabilities.BakedLightProbes = capabilities.DeferredShading;
        capabilities.BakedLightmaps = capabilities.DeferredShading;
        capabilities.VoxelGI = capabilities.DeferredShading && m_Impl->Initialized && m_Impl->SupportsVoxelGI();
        capabilities.LumenGI=capabilities.DeferredShading && m_Impl->Initialized;
        capabilities.Refraction = capabilities.DeferredShading;
        capabilities.RayTracedAmbientOcclusion = capabilities.DeferredShading && IsInitialized();
        capabilities.RayTracedReflections = capabilities.RayTracedAmbientOcclusion;
        capabilities.HybridRendering = capabilities.DeferredShading && capabilities.RayTracedReflections;
        capabilities.RayTracedGI = capabilities.RayTracedReflections;
        capabilities.PathTracing = capabilities.RayTracedReflections && m_Impl->SupportsPathTracing();
		capabilities.VolumetricFog = IsInitialized();
		capabilities.VolumetricLighting = IsInitialized();
        capabilities.DepthPeeling = capabilities.Refraction && capabilities.VolumetricLighting;
		capabilities.LightShafts = IsInitialized();
		capabilities.WeightedTransparencySampleCounts = DX11WeightedTransparency::SupportedCounts(m_Impl->Device.Get());
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
        capabilities.VarianceShadowMaps = DX11VSM::Supported(m_Impl->Device.Get());
        capabilities.ContactShadows = DX11ContactShadows::Supported(m_Impl->Device.Get());
        capabilities.ForwardPlusRendering = m_Impl->Device && m_Impl->Device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0;
        capabilities.ClusteredForwardRendering = capabilities.ForwardPlusRendering;
        capabilities.TiledDeferredRendering = capabilities.DeferredShading && capabilities.ForwardPlusRendering;
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = 16384;
		capabilities.SupportedSampleCounts = DX11Multisampling::SupportedCounts(m_Impl->Device.Get());
		return capabilities;
	}

	bool DX11RHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0)
			return m_Impl->Fail("DX11 cannot resize in its current state");
		if (width == m_Impl->CreateInfo.Width && height == m_Impl->CreateInfo.Height
			&& m_Impl->ColorTexture && m_Impl->ColorTarget && m_Impl->ColorStaging
			&& m_Impl->OutlineMaskTexture && m_Impl->OutlineMaskTarget
			&& m_Impl->OutlineMaskStaging && m_Impl->DepthTexture
			&& m_Impl->DepthTarget && m_Impl->DepthStaging)
		{
			m_Impl->LastError.clear();
			return true;
		}
		if (!m_Impl->CreateOffscreenTargets(width, height)) return false;
		m_Impl->Hair.ResetHistory();
    m_Impl->ExposureHistory.Invalidate();
    m_Impl->CreateInfo.Width = width;
		m_Impl->CreateInfo.Height = height;
		return true;
	}

	BufferHandle DX11RHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateBuffer") || desc.SizeInBytes == 0
			|| desc.SizeInBytes > std::numeric_limits<UINT>::max())
		{
			m_Impl->Fail("DX11 buffer description is invalid");
			return {};
		}
		Impl::BufferResource resource;
		resource.Description = desc;
		resource.Bytes.resize(desc.SizeInBytes);
		if (initialData) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
		D3D11_BUFFER_DESC nativeDesc = {};
		nativeDesc.ByteWidth = static_cast<UINT>(desc.SizeInBytes);
		nativeDesc.Usage = desc.Dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
		nativeDesc.CPUAccessFlags = desc.Dynamic ? D3D11_CPU_ACCESS_WRITE : 0;
		switch (desc.Usage)
		{
		case BufferUsage::Vertex: nativeDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER; break;
		case BufferUsage::Index: nativeDesc.BindFlags = D3D11_BIND_INDEX_BUFFER; break;
		case BufferUsage::Uniform: nativeDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; break;
		}
		D3D11_SUBRESOURCE_DATA data = {};
		data.pSysMem = initialData;
		const HRESULT result = m_Impl->Device->CreateBuffer(&nativeDesc,
			initialData ? &data : nullptr, resource.Buffer.GetAddressOf());
		if (FAILED(result))
		{
			m_Impl->Fail(HResultMessage("ID3D11Device::CreateBuffer", result));
			return {};
		}
		const BufferHandle handle { m_Impl->NextBuffer++ };
		m_Impl->Buffers.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX11RHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& candidate,VirtualGeometryCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache updates require a closed frame");
    return m_Impl->PrepareVirtualGeometryCache(upload,previous,candidate,statistics);
}
bool DX11RHI::SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& task,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Submit Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->SubmitNaniteCache(upload,previous,task,statistics);
}
bool DX11RHI::PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    if (!m_Impl->CheckThread("Poll Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->PollNaniteCache(task,ready,output);
}
bool DX11RHI::CancelNaniteCache(NaniteCacheTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->CancelNaniteCache(task);
}
bool DX11RHI::PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& candidate,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache updates require a closed frame");
    return m_Impl->PrepareNaniteCache(upload,previous,candidate,statistics);
}
bool DX11RHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache release requires a closed frame");
    return m_Impl->DestroyVirtualGeometryCache(handle);
}
bool DX11RHI::DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache release requires a closed frame");
    return m_Impl->DestroyNaniteCache(handle);
}
bool DX11RHI::SnapshotTSROpaque()
{
    if (!m_Impl->CheckThread("Snapshot TSR opaque")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || m_Impl->PassOpen) return m_Impl->Fail("TSR opaque snapshot requires a closed pass in an initialized frame");
    return m_Impl->SnapshotTSROpaque();
}
bool DX11RHI::RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& settings,FrameCapture& output)
{
    if (!m_Impl->CheckThread("Render TSR display")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR display work requires an initialized RHI and closed frame");
    return m_Impl->RenderTSRDisplay(handle,settings,output);
}
bool DX11RHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    if (!m_Impl->CheckThread("BuildVirtualShadowReceiverRequests")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow receivers require an initialized RHI and closed frame");
    return m_Impl->BuildVirtualShadowReceiverRequests(upload,output);
}
bool DX11RHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    if (!m_Impl->CheckThread("SampleVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow lookup requires an initialized RHI and closed frame");
    return m_Impl->SampleVirtualShadowAtlas(handle,upload,output);
}
bool DX11RHI::CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!m_Impl->CheckThread("CreateLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->CreateLumenDistanceScene(data,output);
}
bool DX11RHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    if (!m_Impl->CheckThread("CreateLumenGlobalDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CreateLumenGlobalDistanceScene(source,settings,output,previous);
}

bool DX11RHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenGlobalDistanceUpdate(source,settings,previous,task);
}

bool DX11RHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenGlobalDistanceUpdate(task,budget,ready,output,statistics);
}

bool DX11RHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{
    if (!m_Impl->CheckThread("CancelLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenGlobalDistanceUpdate(task);
}
bool DX11RHI::TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->TraceLumenDistanceScene(handle,upload,output);
}
bool DX11RHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenDistanceScene(handle);
}
bool DX11RHI::RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    if (!m_Impl->CheckThread("RelightLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->RelightLumenSurface(surface,distance,upload,output);
}

bool DX11RHI::BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenLightingUpdate(surface,distance,upload,task);
}

bool DX11RHI::AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenLightingUpdate(task,budget,ready,output,statistics);
}

bool DX11RHI::CancelLumenLightingUpdate(LumenLightingTask task)
{
    if (!m_Impl->CheckThread("CancelLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenLightingUpdate(task);
}
bool DX11RHI::ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenLighting(handle,output);
}
bool DX11RHI::DestroyLumenLighting(LumenLightingHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenLighting(handle);
}
bool DX11RHI::TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenRadiance")) return false;
    if (upload.Probes.DirectionalSide && m_Impl->FrameOpen) return m_Impl->Fail("Directional world cache generation requires a closed frame");
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->TraceLumenRadiance(lighting,upload,output);
}
bool DX11RHI::AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("AssembleLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("World radiance assembly requires an initialized RHI and closed frame");
    if (!m_Impl->ValidateLumenRecording(false)) return false;
    return m_Impl->AssembleLumenRadiance(assembly,output);
}
bool DX11RHI::ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbeOrigins")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbeOrigins(handle,output);
}
bool DX11RHI::ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    if (!m_Impl->CheckThread("ReadLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenRadiance(handle,output);
}
bool DX11RHI::DestroyLumenRadiance(LumenRadianceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenRadiance(handle);
}
bool DX11RHI::IntegrateLumenProbes(LumenRadianceHandle source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    if (!m_Impl->CheckThread("IntegrateLumenProbes")) return false;
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->IntegrateLumenProbes(source,upload,output);
}
bool DX11RHI::ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if (!m_Impl->CheckThread("ResolveLumenProbes")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || !m_Impl->ValidateLumenRecording(false)) return m_Impl->Fail("Lumen gather requires an open frame outside a graphics pass");
    return m_Impl->ResolveLumenProbes(handle,settings);
}
bool DX11RHI::ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbes(handle,output);
}
bool DX11RHI::DestroyLumenProbes(LumenProbeHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenProbes(handle);
}
bool DX11RHI::LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    if (!m_Impl->CheckThread("LookupLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface lookup requires an initialized RHI and closed frame");
    return m_Impl->LookupLumenSurface(handle,upload,output);
}
bool DX11RHI::CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!m_Impl->CheckThread("CaptureLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->CaptureLumenSurface(upload,output);
}
bool DX11RHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenCaptureUpdate(upload,task);
}
bool DX11RHI::AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenCaptureUpdate(task,budget,ready,output,statistics);
}
bool DX11RHI::CancelLumenCaptureUpdate(LumenCaptureTask task)
{
    if (!m_Impl->CheckThread("CancelLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenCaptureUpdate(task);
}
bool DX11RHI::ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->ReadLumenSurface(handle,output);
}
bool DX11RHI::DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenSurface(handle);
}
bool DX11RHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!m_Impl->CheckThread("RenderVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->RenderVirtualShadowAtlas(upload,previous,output);
}
bool DX11RHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->ReadVirtualShadowAtlas(handle,output);
}
bool DX11RHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!m_Impl->CheckThread("DestroyVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->DestroyVirtualShadowAtlas(handle);
}
bool DX11RHI::DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!m_Impl->CheckThread("Destroy TSR history")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR history release requires an initialized RHI and closed frame");
    return m_Impl->DestroyTSRHistory(handle);
}
bool DX11RHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle handle,const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached virtual geometry")) return false;
    if (m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateVirtualGeometry(upload,output,handle);
}
bool DX11RHI::SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& task)
{
    if (!m_Impl->CheckThread("Submit Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->SubmitNaniteGeometry(cache,upload,task);
}
bool DX11RHI::PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Poll Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->PollNaniteGeometry(task,ready,output);
}
bool DX11RHI::CancelNaniteGeometry(NaniteGeometryTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->CancelNaniteGeometry(task);
}
bool DX11RHI::CreateCachedNaniteGeometry(NaniteCacheHandle handle,const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateNaniteGeometry(upload,output,handle);
}

bool DX11RHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create virtual geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual geometry must be produced between frames");
    return m_Impl->CreateVirtualGeometry(upload,output);
}
bool DX11RHI::PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDraws") && m_Impl->PrepareNaniteDraws(visibility,draws,output);
}
bool DX11RHI::PrepareNaniteDrawBins(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDrawBins") && m_Impl->PrepareNaniteDraws(visibility,draws,output,true);
}
bool DX11RHI::DrawNaniteBins(NaniteDrawHandle handle)
{
    if (!m_Impl->CheckThread("DrawNaniteBins") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite bins require an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || !found->second->Binning ||
        std::any_of(found->second->Submitted.begin(),found->second->Submitted.end(),[](bool submitted){return submitted;}))
        return m_Impl->Fail("Nanite bin handle is invalid or already consumed");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled) return m_Impl->Fail("Nanite post bins must preserve main depth");
    for (unsigned id=0;id<record.Binning->Bins.size();++id)
    {
        const auto& bin=record.Binning->Bins[id]; const auto& draw=(*record.Draws)[bin.FirstDraw];
        if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
            !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
        m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=bin.FirstDraw; m_Impl->ActiveNaniteBin=id;
        struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
        if (!DrawIndexedGeometry(bin.FirstIndex,bin.IndexCount,0,nullptr)) return false;
        std::fill(record.Submitted.begin()+bin.FirstDraw,record.Submitted.begin()+bin.FirstDraw+bin.DrawCount,true);
    }
    return true;
}
bool DX11RHI::DrawNaniteCluster(NaniteDrawHandle handle,unsigned query)
{
    if (!m_Impl->CheckThread("DrawNaniteCluster") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite draw requires an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || found->second->Binning || query>=found->second->Draws->size() || found->second->Submitted[query])
        return m_Impl->Fail("Nanite draw handle, query index or duplicate submission is invalid");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled)
        return m_Impl->Fail("Nanite post pass must preserve main depth");
    const auto& draw=(*record.Draws)[query];
    if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
        !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
    m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=query;
    struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
    if (!DrawIndexedGeometry(draw.FirstIndex,draw.IndexCount,0,nullptr)) return false;
    record.Submitted[query]=true; return true;
}
bool DX11RHI::CaptureNaniteDepth(NaniteDepthHandle& output)
{
    if (!m_Impl->CheckThread("CaptureNaniteDepth") || !IsInitialized()) return false;
    return m_Impl->CaptureNaniteDepth(output);
}
bool DX11RHI::BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("BeginNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->BeginNaniteVisibility(history,queries,output);
}
bool DX11RHI::RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle depth,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("RetestNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->RetestNaniteVisibility(main,depth,output);
}
bool DX11RHI::ReadNaniteDepth(NaniteDepthHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDepthWords(found->second->Info,words,output,m_Impl->LastError);
}
bool DX11RHI::ReadNaniteVisibility(NaniteVisibilityHandle handle,std::vector<NaniteVisibilityState>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDecisionWords(found->second->Info,words,output,m_Impl->LastError);
}
bool DX11RHI::DestroyNaniteDepth(NaniteDepthHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDepths.erase(found); return true;
}
bool DX11RHI::DestroyNaniteVisibility(NaniteVisibilityHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDecisions.erase(found); return true;
}
bool DX11RHI::CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite geometry must be produced between frames");
    return m_Impl->CreateNaniteGeometry(upload,output);
}

bool DX11RHI::CreateTessellatedGeometry(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create tessellated geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Tessellated geometry must be prepared between frames");
    return m_Impl->CreateTessellated(geometry,settings,output);
}

bool DX11RHI::CreateDisplacedVertexBuffer(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    if (!m_Impl->CheckThread("Create displaced vertex buffer")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Displaced geometry must be prepared between frames");
    return m_Impl->CreateDisplaced(geometry,settings,output);
}

	bool DX11RHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer")) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || data == nullptr || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("DX11 buffer update is invalid");
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		if (iterator->second.Description.Dynamic)
		{
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = m_Impl->Context->Map(iterator->second.Buffer.Get(), 0,
				D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(buffer)", result));
			std::memcpy(mapped.pData, iterator->second.Bytes.data(), iterator->second.Bytes.size());
			m_Impl->Context->Unmap(iterator->second.Buffer.Get(), 0);
		}
		else
		{
			D3D11_BOX box = { static_cast<UINT>(offset), 0, 0,
				static_cast<UINT>(offset + size), 1, 1 };
			m_Impl->Context->UpdateSubresource(iterator->second.Buffer.Get(), 0, &box, data, 0, 0);
		}
		return true;
	}

	bool DX11RHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer")) return false;
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		if (!buffer || m_Impl->Buffers.erase(buffer.Value) == 0)
			return m_Impl->Fail("DX11 buffer handle is invalid");
		return true;
	}

	TextureHandle DX11RHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0 ||
			!ValidTextureMipLevels(desc, initialData))
		{
			m_Impl->Fail("DX11 texture description is invalid");
			return {};
		}
		Impl::TextureResource resource;
		resource.Description = desc;
		if (desc.Format == TextureFormat::Depth32)
		{
			if (!m_Impl->CreateDepthTexture(desc.Width, desc.Height, true, resource.Texture,
				resource.DepthStencilView, resource.ShaderResourceView, resource.StagingTexture)) return {};
			if (initialData != nullptr)
			{
				m_Impl->Context->UpdateSubresource(resource.Texture.Get(), 0, nullptr, initialData,
					static_cast<UINT>(desc.Width * sizeof(float)), 0);
			}
		}
		else
		{
			D3D11_TEXTURE2D_DESC nativeDesc = {};
			nativeDesc.Width = static_cast<UINT>(desc.Width);
			nativeDesc.Height = static_cast<UINT>(desc.Height);
			nativeDesc.MipLevels = desc.MipLevels;
			nativeDesc.ArraySize = 1;
			nativeDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			nativeDesc.SampleDesc.Count = 1;
			nativeDesc.Usage = D3D11_USAGE_DEFAULT;
			nativeDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE
				| (desc.RenderTarget ? D3D11_BIND_RENDER_TARGET : 0u);
			std::vector<D3D11_SUBRESOURCE_DATA> data;
			if (initialData)
			{
				data.resize(desc.MipLevels);
				VisitTextureMips(desc, [&](int level, int width, int, std::size_t offset) {
					data[level].pSysMem = static_cast<const std::uint8_t*>(initialData) + offset;
					data[level].SysMemPitch = static_cast<UINT>(width * 4);
				});
			}
			HRESULT result = m_Impl->Device->CreateTexture2D(&nativeDesc, initialData ? data.data() : nullptr,
				resource.Texture.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage("ID3D11Device::CreateTexture2D", result));
				return {};
			}
			if (desc.RenderTarget)
			{
				D3D11_RENDER_TARGET_VIEW_DESC renderTargetDesc = {};
				renderTargetDesc.Format = desc.ColorSpace == TextureColorSpace::SRGB
					? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
					: DXGI_FORMAT_R8G8B8A8_UNORM;
				renderTargetDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
				result = m_Impl->Device->CreateRenderTargetView(
					resource.Texture.Get(), &renderTargetDesc,
					resource.RenderTargetView.GetAddressOf());
				if (FAILED(result))
				{
					m_Impl->Fail(HResultMessage("ID3D11Device::CreateRenderTargetView(texture)", result));
					return {};
				}
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC linearViewDesc = {};
			linearViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			linearViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			linearViewDesc.Texture2D.MipLevels = desc.MipLevels;
			result = m_Impl->Device->CreateShaderResourceView(
				resource.Texture.Get(), &linearViewDesc,
				resource.ShaderResourceView.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage("ID3D11Device::CreateShaderResourceView(texture)", result));
				return {};
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC srgbViewDesc = linearViewDesc;
			srgbViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			result = m_Impl->Device->CreateShaderResourceView(
				resource.Texture.Get(), &srgbViewDesc,
				resource.SRGBShaderResourceView.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage(
					"ID3D11Device::CreateShaderResourceView(texture sRGB)",
					result));
				return {};
			}
		}
		const TextureHandle handle { m_Impl->NextTexture++ };
		m_Impl->Textures.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX11RHI::DestroyTexture(TextureHandle texture)
	{
		if (!m_Impl->CheckThread("DestroyTexture")) return false;
		if (!m_Impl->RetireLumenCaptureInput(texture)) return false;
		if (!texture || m_Impl->Textures.erase(texture.Value) == 0)
			return m_Impl->Fail("DX11 texture handle is invalid");
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
        m_Impl->VSM.Release(texture);
		return true;
	}

	bool DX11RHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX11 frame cannot begin in the current state");
		if (!m_Impl->CollectLumenGlobalDistance() || !m_Impl->CollectLumenLighting() || !m_Impl->CollectLumenCapture()) return false;
		ResetMotionTransform();
    m_Impl->ExposureHistory.Begin();
        m_Impl->GpuCursor = 0; m_Impl->ActiveGpuBatch = nullptr; ++m_Impl->GpuEpoch;
        m_Impl->FrameShadowFilter = GetTechniqueSettings().ShadowFilter;
        if (GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced) m_Impl->FrameShadowFilter.Mode=Techniques::ShadowFilterMode::Hard;
        if (!m_Impl->PrepareRayShadowImage(GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr)) return false;
        m_Impl->VSM.BeginFrame();
        m_Impl->ContactShadows.BeginFrame();
        m_Impl->FrameContact = GetTechniqueSettings().ContactShadows;
        m_Impl->ForwardPlus.BeginFrame(); m_Impl->TiledDeferred.BeginFrame(); m_Impl->ClusteredForward.BeginFrame(); m_Impl->LightListFrame.Reset(); m_Impl->ForwardPlusDepthWritten = false;
        m_Impl->TiledActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::TiledDeferred;
        m_Impl->ClusteredActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::ClusteredForward;
        m_Impl->LightListsActive = Techniques::UsesLightLists(GetTechniqueSettings().Pipeline);
        m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
		if (!m_Impl->PrepareVirtualShadowSampling(GetVirtualShadowSamplingFrame(),GetCascadedShadowState()) ||
            !m_Impl->PrepareCascadedShadowFrame(GetCascadedShadowState())) return false;
		m_Impl->SubsurfaceProducing=m_Impl->SubsurfaceResolved=m_Impl->SubsurfaceHasScene=false;
    m_Impl->FrameSubsurface=GetTechniqueSettings().Subsurface;
    m_Impl->SubsurfaceRequested=Techniques::UsesSubsurfaceScattering(m_Impl->FrameSubsurface);
    m_Impl->FramePostProcessing = GetTechniqueSettings().PostProcessing;
    if (!m_Impl->PrepareTSRScene(GetTSRSceneFrame(),m_Impl->FramePostProcessing)) return false;
        m_Impl->FrameGI = GetTechniqueSettings().GlobalIllumination;
        m_Impl->FrameLumenLighting=GetLumenLighting(); m_Impl->FrameLumenWorldCache=GetLumenWorldCache(); m_Impl->FrameLumenGlobalDistance=GetLumenGlobalDistance();
        m_Impl->FrameVoxelGI=GetVoxelGIFrame();
    if (!m_Impl->Hair.Freeze(GetHairFurFrame(),GetTechniqueSettings(),m_Impl->LastError)) return false;
        if (Techniques::UsesVoxelGI(m_Impl->FrameGI))
        {
            if (!m_Impl->FrameVoxelGI.Volume) return m_Impl->Fail("Voxel GI requires a current immutable radiance publication");
            m_Impl->FrameVoxelGI.Settings=m_Impl->FrameGI.Voxels; m_Impl->FrameVoxelGI.Intensity=m_Impl->FrameGI.Intensity;
            m_Impl->FrameVoxelGI.MaximumDistance=m_Impl->FrameGI.MaximumDistance; m_Impl->FrameVoxelGI.Shading=GetTechniqueSettings().Shading;
        }
        if (m_Impl->FrameVoxelGI.Volume && ((Techniques::UsesGlobalIllumination(m_Impl->FrameGI) && !Techniques::UsesVoxelGI(m_Impl->FrameGI)) ||
            Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline) ||
            !Techniques::RequiresHDR(m_Impl->FramePostProcessing.HDR) || m_Impl->FrameVoxelGI.Shading!=GetTechniqueSettings().Shading))
            return m_Impl->Fail("Voxel GI requires an exclusive HDR raster frame and matching receiver shading");
        m_Impl->GIActive = Techniques::UsesGlobalIllumination(m_Impl->FrameGI) || bool(m_Impl->FrameVoxelGI.Volume);
        m_Impl->EarlyTemporalActive = Techniques::UsesTemporalGI(m_Impl->FrameGI) || Techniques::UsesTemporalReflections(GetTechniqueSettings().Reflections);
        m_Impl->ProbeActive = Techniques::UsesBakedProbes(m_Impl->FrameGI);
        m_Impl->LightmapActive = Techniques::UsesLightmaps(m_Impl->FrameGI);
        m_Impl->FrameLightmaps = GetLightmapFrame();
        m_Impl->FramePeeling = GetDepthPeelingFrame();
        m_Impl->Peeling.Sequence.Abort();
        if (!ValidateDepthPeelingFrame(m_Impl->FramePeeling,m_Impl->LastError)) return false;
        m_Impl->FrameRefraction = GetRefractionFrame();
        m_Impl->FrameRefractionSettings = GetTechniqueSettings().Refraction;
        m_Impl->RefractionActive = Techniques::UsesRefraction(GetTechniqueSettings());
        m_Impl->RefractionResolved = false;
        m_Impl->FrameLightmapSettings = GetTechniqueSettings();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI) && !m_Impl->GI.Prepare(m_Impl->Device.Get(),
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError)) return false;
        if (m_Impl->ProbeActive && !m_Impl->Probes.Prepare(m_Impl->Device.Get(),GetBakedProbeSamples(),
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->LastError)) return false;
		m_Impl->FrameFog = GetTechniqueSettings().VolumetricFog;
		m_Impl->FrameVolumeLighting = GetTechniqueSettings().VolumetricLighting;
		m_Impl->FrameLightShafts = GetTechniqueSettings().LightShafts;
		m_Impl->LightShafts.Begin();
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShafts.Prepare(m_Impl->Device.Get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError)) return false;
		m_Impl->VolumetricFog.Begin();
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Prepare(m_Impl->Device.Get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameFog, m_Impl->LastError)) return false;
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)) &&
			!m_Impl->VolumetricLighting.Prepare(m_Impl->Device.Get(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				m_Impl->FrameFog, m_Impl->LastError)) return false;
		m_Impl->DepthOfFieldResolved = false;
		m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
		m_Impl->TemporalAA.Begin();
        m_Impl->GI.Begin(); m_Impl->Probes.Begin();
		if (!m_Impl->PrepareHdrFrame(m_Impl->FramePostProcessing.HDR)) return false;
		if (((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive) && !m_Impl->EnsureVelocityResources()) return false;
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalAA.Prepare(m_Impl->Device.Get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError)) return false;
		if (m_Impl->FramePostProcessing.DepthOfField.Enabled && !m_Impl->EnsureDepthOfFieldResources()) return false;
		if (m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->EnsureMotionResources()) return false;
		m_Impl->FrameAreaLights = GetTechniqueSettings().AreaLights.Enabled;
        m_Impl->FrameSampleCount = GetTechniqueSettings().SampleCount;
		m_Impl->FrameTransparency = GetTechniqueSettings().Transparency;
		m_Impl->WeightedActive = Techniques::UsesWeightedTransparency(GetTechniqueSettings());
		m_Impl->WeightedPass = false;
		m_Impl->WeightedTransparency.Begin();
		if (m_Impl->WeightedActive && !m_Impl->WeightedTransparency.Ensure(m_Impl->Device.Get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameSampleCount, m_Impl->LastError)) return false;
		m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
		m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
		m_Impl->DeferredActive = Techniques::UsesDeferredLighting(GetTechniqueSettings().Pipeline);
		m_Impl->FrameAmbientOcclusion = GetTechniqueSettings().AmbientOcclusion;
        m_Impl->FrameRayScene = GetRayScene();
        m_Impl->FrameRayRadiance = GetRayRadianceFrame(); m_Impl->FrameRayRadianceSettings = GetTechniqueSettings();
        m_Impl->PathFrame.BeginFrame(Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline));
		m_Impl->AmbientOcclusionActive = Techniques::UsesAmbientOcclusion(m_Impl->FrameAmbientOcclusion);
		m_Impl->FrameReflections = GetTechniqueSettings().Reflections;
		m_Impl->FrameReflectionShading = GetTechniqueSettings().Shading;
		m_Impl->AnisotropicFiltering = GetTechniqueSettings().AnisotropicFiltering;
		if (!m_Impl->PrepareEnvironmentCapture(GetEnvironmentCaptureFrame())) return false;
		if (!m_Impl->PreparePlanarCapture(GetPlanarCaptureFrame())) return false;
		m_Impl->ReflectionActive = Techniques::UsesReflections(m_Impl->FrameReflections);
		m_Impl->ReflectionApplied = false;
		m_Impl->MaterialCaptureActive = m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive;
		m_Impl->AmbientOcclusionApplied = m_Impl->SceneColorWritten = false;
		m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
		m_Impl->DeferredBackground = {};
		if (m_Impl->MaterialCaptureActive && !m_Impl->EnsureDeferredResources()) return false;
		if (m_Impl->AmbientOcclusionActive && !m_Impl->EnsureAmbientOcclusionResources()) return false;
		if (m_Impl->ReflectionActive && !m_Impl->EnsureReflectionResources()) return false;
    if (Techniques::UsesRayTracedGI(m_Impl->FrameGI) && !m_Impl->EnsureRayGIResources()) return false;
    if (m_Impl->FrameVoxelGI.Volume && !m_Impl->EnsureVoxelGIResources()) return false;
    if (m_Impl->PathFrame.Active && !m_Impl->EnsurePathTracing()) return false;
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->Multisampling.Prepare(m_Impl->Device.Get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameSampleCount,
			m_Impl->HdrSceneActive, m_Impl->MaterialCaptureActive, ((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive),
			m_Impl->LastError)) return false;
        if (m_Impl->LightmapActive && !m_Impl->Lightmaps.Prepare(m_Impl->Device.Get(),m_Impl->FrameLightmaps,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->FrameSampleCount,m_Impl->LastError)) return false;
        if (m_Impl->RefractionActive && !m_Impl->Refraction.Prepare(m_Impl->Device.Get(),m_Impl->FrameRefraction,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->LastError)) return false;
        if (m_Impl->FramePeeling.Enabled)
        {
            if (!m_Impl->HdrSceneActive || m_Impl->FrameSampleCount!=1 || m_Impl->WeightedActive)
                return m_Impl->Fail("Depth peeling requires single-sample HDR without weighted accumulation");
            if (!m_Impl->Peeling.Prepare(m_Impl->Device.Get(),m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->LastError) ||
                !m_Impl->Peeling.Sequence.Begin(m_Impl->FramePeeling.Settings.LayerCount,m_Impl->LastError)) return false;
        }
		if (m_Impl->Hair.Active && !m_Impl->EnsureHairFur()) return false;
    m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Statistics.RasterSampleCount = m_Impl->FrameSampleCount;
		m_Impl->PendingFrame.Backend = RHIBackend::DX11;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
		m_Impl->FrameOpen = true;
		m_Impl->HasOutlineMaskThisFrame = false;
		m_Impl->LastShadowTexture = {};
		m_Impl->LastError.clear();
		if (m_Impl->FramePostProcessing.TemporalEffects.Enabled && !m_Impl->EnsureExposure()) return false;
    return true;
	}

	void DX11RHI::AbortFrame()
	{
        m_Impl->Peeling.Sequence.Abort(); m_Impl->FramePeeling = {};
		if (!m_Impl || !m_Impl->Initialized)
		{
			return;
		}
		if (m_Impl->OwnerThread != std::this_thread::get_id())
		{
			m_Impl->LastError =
				"AbortFrame must execute on the owning DX11 RHI thread";
			return;
		}
        m_Impl->PathFrame.Abort();
		constexpr UINT DeferredInputCount = 19; // material inputs through complete lights and tile lists
		ID3D11ShaderResourceView* nullViews[DeferredInputCount] = {};
		m_Impl->Context->PSSetShaderResources(0, DeferredInputCount, nullViews);
        m_Impl->ForwardPlus.BeginFrame(); m_Impl->TiledDeferred.BeginFrame(); m_Impl->ClusteredForward.BeginFrame(); m_Impl->LightListFrame.Reset();
        m_Impl->LightListsActive = m_Impl->ForwardPlusDepthWritten = false;
        m_Impl->VSM.BeginFrame();
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
        m_Impl->ContactShadows.BeginFrame();
        m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
		m_Impl->PendingFrame = {};
		m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
		m_Impl->TemporalAA.Begin();
        m_Impl->GI.Begin(); m_Impl->Probes.Invalidate();
		m_Impl->VolumetricFog.Begin();
		m_Impl->LightShafts.Begin();
		m_Impl->WeightedTransparency.Begin();
		m_Impl->WeightedActive = m_Impl->WeightedPass = false;
		m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
		m_Impl->FrameCascades = {};
		m_Impl->EnvironmentActive = m_Impl->EnvironmentReady = false;
		m_Impl->EnvironmentFacesWritten = 0; m_Impl->FrameEnvironment = {};
		m_Impl->PlanarActive = m_Impl->PlanarWritten = m_Impl->PlanarReady = false;
		m_Impl->FramePlanar = {}; SetPlanarCaptureFrame({});
		SetEnvironmentCaptureFrame({});
		SetCascadedShadowState({});
		m_Impl->DepthOfFieldResolved = false;
		m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
		m_Impl->CurrentPass = {};
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		m_Impl->AbortTSRScene();
    m_Impl->FrameOpen = false;
    m_Impl->FinishLumenFrame(false); m_Impl->FinishNaniteFrame(false);
		m_Impl->PassOpen = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
		m_Impl->HasOutlineMaskThisFrame = false;
		m_Impl->ReflectionActive = m_Impl->ReflectionApplied = false;
		m_Impl->MaterialCaptureActive = m_Impl->AmbientOcclusionActive = m_Impl->AmbientOcclusionApplied = m_Impl->SceneColorWritten = false;
		m_Impl->DeferredActive = m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
	}

	bool DX11RHI::BeginPass(const PassDesc& desc)
	{
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !desc.Target && (desc.Type==PassType::GBuffer || (desc.Type==PassType::Color && !desc.DepthPeelingLayer && desc.ClearColorEnabled))) return m_Impl->Fail("Opaque scene writes cannot restart after the TSR snapshot");

    if (m_Impl->FrameOpen && m_Impl->ExposureHistory.Resolved && desc.Type!=PassType::Outline)
        return m_Impl->Fail("Temporal exposure is the last scene-HDR stage before display resolve");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved &&
        ((desc.Type==PassType::Color && !desc.Target && desc.ClearColorEnabled) ||
         (desc.ClearDepthEnabled && desc.Type!=PassType::Shadow && desc.Type!=PassType::EnvironmentCapture && desc.Type!=PassType::PlanarCapture)))
        return m_Impl->Fail("Completed hair color and primary depth cannot be cleared by a later pass");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete() && !desc.Target && (desc.Type==PassType::Velocity || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete hair rasterization before velocity or outline draws");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved && !desc.Target && desc.Type==PassType::GBuffer)
        return m_Impl->Fail("Opaque geometry cannot replace completed hair depth");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete() && !desc.Target &&
        (desc.Type==PassType::Color || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete subsurface diffusion before transparent or outline composition");
		if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX11 pass cannot begin in the current state");
        if (m_Impl->PathFrame.Active) return m_Impl->Fail("Path tracing owns primary visibility; raster passes are inactive");
        if (!ValidateDepthPeelingPass(desc,m_Impl->FramePeeling,m_Impl->Peeling.Sequence,m_Impl->LastError)) return false;
		const bool weighted = m_Impl->WeightedActive && m_Impl->WeightedTransparency.Prepared && !m_Impl->WeightedTransparency.Resolved;
		if (weighted && (desc.Type != PassType::Color || desc.Target || desc.ClearColorEnabled || desc.ClearDepthEnabled ||
			m_Impl->WeightedTransparency.Accumulated)) return m_Impl->Fail("Weighted transparency accepts one depth-preserving color pass before resolve");
		if (m_Impl->WeightedActive && m_Impl->SceneColorWritten && desc.Type == PassType::Color && !desc.Target && !weighted)
			return m_Impl->Fail("Prepare weighted transparency before transparent draws; resolve closes the transparent phase");
		constexpr UINT DeferredInputCount = 16; // material inputs, cascades, fog and contact visibility
		ID3D11ShaderResourceView* nullViews[DeferredInputCount] = {};
		m_Impl->Context->PSSetShaderResources(0, DeferredInputCount, nullViews);
		if (desc.Type == PassType::DeferredLighting)
			return m_Impl->Fail("Use DrawDeferredLighting for the fullscreen lighting phase");
		const bool environmentCapture = desc.Type == PassType::EnvironmentCapture;
		const bool planarCapture = desc.Type == PassType::PlanarCapture;
		const bool capture = environmentCapture || planarCapture;
		if (environmentCapture && (!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || desc.Target ||
			desc.EnvironmentFaceIndex >= EnvironmentFaceCount || !desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
			m_Impl->SceneColorWritten || m_Impl->GBufferWritten ||
			(m_Impl->EnvironmentFacesWritten & (1u << desc.EnvironmentFaceIndex)) != 0))
			return m_Impl->Fail("Capture each environment face once, clearing private color/depth before main geometry");
		if (!environmentCapture && m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady && desc.Type != PassType::Shadow)
			return m_Impl->Fail("Complete all environment faces and mips before main geometry");
		if (planarCapture && (!m_Impl->PlanarActive || m_Impl->PlanarWritten || m_Impl->PlanarReady || desc.Target ||
			!desc.ClearColorEnabled || !desc.ClearDepthEnabled || m_Impl->SceneColorWritten || m_Impl->GBufferWritten))
			return m_Impl->Fail("Capture the planar view once, clearing private color/depth before main geometry");
		if (!planarCapture && m_Impl->PlanarActive && !m_Impl->PlanarReady && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture)
			return m_Impl->Fail("Complete the planar capture and mips before main geometry");
		const bool velocity = desc.Type == PassType::Velocity;
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Light shafts must precede opaque velocity and temporal effects");
		if (m_Impl->LightShafts.Resolved && desc.Type != PassType::Shadow &&
			(desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
			return m_Impl->Fail("Scattered scene color and depth cannot be cleared");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Fog composition must precede opaque velocity and temporal effects");
		if (m_Impl->VolumetricFog.Resolved && desc.Type != PassType::Shadow &&
			(desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
			return m_Impl->Fail("Fogged scene color and depth cannot be cleared");
		const bool multisample = m_Impl->FrameSampleCount > 1 && !capture;
		if (multisample && desc.Target && desc.Type != PassType::Shadow)
			return m_Impl->Fail("MSAA uses internal sample-count-matched color attachments");
		if (multisample && (velocity || desc.Type == PassType::Outline) && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("Resolve multisample opaque inputs before velocity or outlines");
		if (multisample && m_Impl->SamplesOpaqueResolved && desc.Type == PassType::Color &&
			(desc.ClearColorEnabled || desc.ClearDepthEnabled ||
				(m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored)))
			return m_Impl->Fail("Restore processed opaque color before MSAA transparency; keep sample depth");
		if (velocity && (!((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive) || desc.Target || desc.ClearDepthEnabled ||
			!desc.ClearColorEnabled || !m_Impl->SceneColorWritten || m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved ||
			m_Impl->DepthOfFieldResolved || m_Impl->TemporalAA.Resolved || (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(!m_Impl->EarlyTemporalActive && m_Impl->ReflectionActive && !m_Impl->ReflectionApplied)))
			return m_Impl->Fail("Velocity requires completed opaque lighting/AO and a cleared internal target, once before blur");
		if ((m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved) && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("Opaque depth cannot change after velocity capture");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->FramePostProcessing.MotionBlur.Enabled)
		{
			if (m_Impl->SceneColorWritten && !m_Impl->MotionBlurResolved)
				return m_Impl->Fail("Forward transparency must follow motion blur");
			if (m_Impl->MotionBlurResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("Scene color/depth cannot be cleared after motion blur");
		}
		if (m_Impl->DepthOfFieldResolved && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("Depth cannot be cleared after depth of field, including through a custom color target");
		if (m_Impl->ReflectionApplied && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("Depth cannot be cleared after reflections");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->ReflectionActive)
		{
			if (!m_Impl->GBufferWritten) return m_Impl->Fail("Reflections require material capture before forward lighting");
			if (m_Impl->SceneColorWritten && !m_Impl->ReflectionApplied) return m_Impl->Fail("Transparency must follow reflections");
			if (m_Impl->ReflectionApplied && desc.ClearColorEnabled) return m_Impl->Fail("Scene color cannot be cleared after reflections");
		}
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->GIActive)
        {
            if (!m_Impl->GBufferWritten) return m_Impl->Fail("SSGI requires material capture before lighting");
            if (m_Impl->SceneColorWritten && !m_Impl->GI.Resolved) return m_Impl->Fail("Transparency must follow SSGI");
            if (m_Impl->GI.Resolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
                return m_Impl->Fail("Scene attachments cannot be cleared after SSGI");
        }
		const bool gbuffer = desc.Type == PassType::GBuffer;
        const bool contactDepth = desc.Type == PassType::ContactDepth;
        const bool forwardPlusDepth = desc.Type == PassType::ForwardPlusDepth;
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Prepared())
            return m_Impl->Fail("Prepare complete Forward+ lights before recording scene passes");
        if (forwardPlusDepth && (m_Impl->ClusteredActive || m_Impl->TiledActive || !m_Impl->LightListsActive || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ForwardPlusDepthWritten || m_Impl->GBufferWritten || m_Impl->SceneColorWritten || m_Impl->LightListFrame.Assigned()))
            return m_Impl->Fail("Forward+ depth must precede assignment and main-view lighting");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !desc.Target && desc.Type != PassType::Shadow && !capture &&
            (desc.Type == PassType::GBuffer || desc.ClearDepthEnabled))
            return m_Impl->Fail("Forward+ light lists require immutable opaque depth through lighting");
        if (m_Impl->LightListsActive && desc.Type == PassType::Color && !desc.Target && !m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("Generate Forward+ lists before main-view material lighting");
        if (contactDepth && (!m_Impl->FrameContact.Enabled || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ContactDepthWritten || m_Impl->GBufferWritten || m_Impl->SceneColorWritten || m_Impl->ContactGenerated))
            return m_Impl->Fail("Contact depth must clear opaque main-view depth once, before visibility and lighting");
        if (m_Impl->ContactGenerated && desc.Type != PassType::Shadow && !capture && desc.ClearDepthEnabled)
            return m_Impl->Fail("Contact visibility requires immutable opaque depth through lighting");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("Generate contact visibility before scene lighting");
		if (gbuffer && (!m_Impl->MaterialCaptureActive || desc.Target || m_Impl->SceneColorWritten || m_Impl->AmbientOcclusionApplied || m_Impl->DepthOfFieldResolved))
			return m_Impl->Fail("Material capture must precede scene lighting and use internal MRT attachments");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->DeferredActive && !m_Impl->DeferredResolved)
			return m_Impl->Fail("Resolve deferred lighting before forward transparent color draws");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionActive)
		{
			if (!m_Impl->GBufferWritten)
				return m_Impl->Fail("Ambient occlusion requires material capture before opaque forward color");
			if (m_Impl->SceneColorWritten && !m_Impl->AmbientOcclusionApplied)
				return m_Impl->Fail("Forward transparency must follow ambient occlusion");
			if (m_Impl->AmbientOcclusionApplied && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("Scene color and depth cannot be cleared after ambient occlusion");
		}
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->FramePostProcessing.DepthOfField.Enabled)
		{
			if (m_Impl->SceneColorWritten && !m_Impl->DepthOfFieldResolved)
				return m_Impl->Fail("Forward transparency must follow depth of field");
			if (m_Impl->DepthOfFieldResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("Scene color and depth cannot be cleared after depth of field");
		}
		m_Impl->CurrentColorIsHdr = capture || (m_Impl->HdrSceneActive && (desc.Type == PassType::Color || (IsGpuPresentationEnabled() && desc.Type == PassType::Outline)) && !desc.Target);
		ID3D11RenderTargetView* renderTarget = m_Impl->CurrentColorIsHdr
			? m_Impl->HdrSurfaces[Impl::HdrScene].Target.Get() : m_Impl->ColorTarget.Get();
		if (velocity) renderTarget = m_Impl->MotionSurfaces[Impl::MotionVelocity].Target.Get();
		ID3D11DepthStencilView* depthTarget = m_Impl->DepthTarget.Get();
		if (multisample && !desc.Target && desc.Type != PassType::Outline)
		{
			depthTarget = m_Impl->Multisampling.DepthTarget.Get();
			renderTarget = velocity ? m_Impl->Multisampling.Velocity.Target.Get() : m_Impl->Multisampling.Color.Target.Get();
		}
		int width = m_Impl->CreateInfo.Width;
		int height = m_Impl->CreateInfo.Height;
		if (environmentCapture)
		{
			renderTarget = m_Impl->Environment.Faces[desc.EnvironmentFaceIndex].Get();
			depthTarget = m_Impl->Environment.DepthTarget.Get();
			width = height = static_cast<int>(m_Impl->FrameEnvironment.Resolution);
		}
		if (planarCapture)
		{
			renderTarget = m_Impl->Planar.Color.Target.Get();
			depthTarget = m_Impl->Planar.DepthTarget.Get();
			width = static_cast<int>(m_Impl->FramePlanar.Width); height = static_cast<int>(m_Impl->FramePlanar.Height);
		}
		if (desc.Type == PassType::Outline && !IsGpuPresentationEnabled())
		{
			if (desc.Target)
			{
				return m_Impl->Fail("DX11 outline passes use the internal visibility mask");
			}
			renderTarget = m_Impl->OutlineMaskTarget.Get();
			if (!m_Impl->HasOutlineMaskThisFrame)
			{
				const float clearMask[] = { 0.0f, 0.0f, 0.0f, 0.0f };
				m_Impl->Context->ClearRenderTargetView(renderTarget, clearMask);
			}
			m_Impl->HasOutlineMaskThisFrame = true;
		}
		else if (desc.Target)
		{
			auto iterator = m_Impl->Textures.find(desc.Target.Value);
			if (iterator == m_Impl->Textures.end()) return m_Impl->Fail("DX11 pass target is invalid");
			width = iterator->second.Description.Width;
			height = iterator->second.Description.Height;
			if (desc.Type == PassType::Shadow)
			{
				if (!iterator->second.DepthStencilView) return m_Impl->Fail("DX11 shadow target is not a depth texture");
				renderTarget = nullptr;
				depthTarget = iterator->second.DepthStencilView.Get();
                m_Impl->LastShadowTexture = desc.Target;
                if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM &&
                    !m_Impl->VSM.BeginMap(m_Impl->Device.Get(), m_Impl->Context.Get(), desc.Target,
                        width, height, desc.ClearDepthEnabled, renderTarget, m_Impl->LastError)) return false;
			}
			else
			{
				if (!iterator->second.RenderTargetView) return m_Impl->Fail("DX11 color target is not renderable");
				if (width != m_Impl->CreateInfo.Width || height != m_Impl->CreateInfo.Height)
				{
					return m_Impl->Fail(
						"DX11 custom color target dimensions must match the default depth target");
				}
				renderTarget = iterator->second.RenderTargetView.Get();
			}
		}
		else if (desc.Type == PassType::Shadow)
		{
			return m_Impl->Fail("DX11 shadow pass requires a depth texture");
		}
        if (desc.DepthPeelingLayer && !m_Impl->Peeling.BeginLayer(m_Impl->Context.Get(),renderTarget,depthTarget,m_Impl->LastError)) return false;
		if (contactDepth || forwardPlusDepth) renderTarget = nullptr;
		if (gbuffer)
		{
			std::array<ID3D11RenderTargetView*, GBufferAttachmentCount> targets {};
			for (std::size_t index = 0; index < targets.size(); ++index)
			{
				targets[index] = multisample ? m_Impl->Multisampling.Materials[index].Target.Get() : m_Impl->GBufferSurfaces[index].Target.Get();
				const float emptyMaterial[4] {};
				if (desc.ClearColorEnabled) m_Impl->Context->ClearRenderTargetView(targets[index], emptyMaterial);
			}
			m_Impl->Context->OMSetRenderTargets(static_cast<UINT>(targets.size()), targets.data(), depthTarget);
			m_Impl->DeferredBackground = desc.ClearColor;
			m_Impl->DeferredBackground.X = std::clamp(desc.ClearColor.X, 0.0f, MaximumHalfFloatRadiance);
			m_Impl->DeferredBackground.Y = std::clamp(desc.ClearColor.Y, 0.0f, MaximumHalfFloatRadiance);
			m_Impl->DeferredBackground.Z = std::clamp(desc.ClearColor.Z, 0.0f, MaximumHalfFloatRadiance);
		}
		else if (weighted) m_Impl->WeightedTransparency.BindTargets(m_Impl->Context.Get(), depthTarget);
		else m_Impl->Context->OMSetRenderTargets(renderTarget ? 1u : 0u,
			renderTarget ? &renderTarget : nullptr, depthTarget);
		if (!gbuffer && desc.Type != PassType::Outline && desc.Type != PassType::Shadow && desc.ClearColorEnabled && renderTarget)
		{
			float color[] = {
				desc.ClearColor.X,
				desc.ClearColor.Y,
				desc.ClearColor.Z,
				desc.ClearColor.W };
			if (velocity) std::fill(std::begin(color), std::end(color), 0.0f);
			if (m_Impl->CurrentColorIsHdr)
				for (float& channel : color) channel = std::clamp(channel, 0.0f, MaximumHalfFloatRadiance);
			m_Impl->Context->ClearRenderTargetView(renderTarget, color);
		}
		if (desc.Type != PassType::Outline && desc.ClearDepthEnabled && depthTarget)
			m_Impl->Context->ClearDepthStencilView(depthTarget, D3D11_CLEAR_DEPTH, 1.0f, 0);
		D3D11_VIEWPORT viewport = {};
		viewport.Width = static_cast<float>(width);
		viewport.Height = static_cast<float>(height);
		viewport.MaxDepth = 1.0f;
		m_Impl->Context->RSSetViewports(1, &viewport);
		m_Impl->CurrentPass = desc;
		m_Impl->WeightedPass = weighted;
		m_Impl->PendingFrame.Passes.push_back({ desc, {} });
		m_Impl->PassOpen = true;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		return true;
	}

	bool DX11RHI::BindPipeline(const PipelineState& pipeline)
	{
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.DepthPeelingLayer && !pipeline.BlendEnabled) return m_Impl->Fail("Only transparent color draws follow the TSR opaque snapshot");

    if (m_Impl->FrameOpen && m_Impl->Hair.Active && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
        ((!m_Impl->Hair.Resolved && pipeline.BlendEnabled) || (m_Impl->Hair.Resolved && pipeline.DepthWrite)))
        return m_Impl->Fail("Opaque draws precede hair; transparent draws follow it without replacing depth");
		if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX11 pipeline must be bound inside a pass");
        if (m_Impl->CurrentPass.DepthPeelingLayer && !ValidateDepthPeelingPipeline(pipeline,m_Impl->LastError)) return false;
        if ((m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) &&
            (pipeline.BlendEnabled || !pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.OutlineExtrusion))
            return m_Impl->Fail("Contact prepass requires opaque depth writes without color or blending");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !m_Impl->CurrentPass.Target && pipeline.DepthWrite &&
            m_Impl->CurrentPass.Type != PassType::Shadow && m_Impl->CurrentPass.Type != PassType::EnvironmentCapture &&
            m_Impl->CurrentPass.Type != PassType::PlanarCapture)
            return m_Impl->Fail("Forward+ receivers cannot overwrite the depth used for assignment");
        if (m_Impl->LightListsActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && !pipeline.DepthTest)
            return m_Impl->Fail("Forward+ tile receivers require camera depth testing");
        if (m_Impl->ContactGenerated && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && pipeline.DepthWrite)
            return m_Impl->Fail("Contact receivers cannot overwrite the traced opaque depth");
		if (m_Impl->WeightedPass && (!pipeline.BlendEnabled || pipeline.DepthWrite || !pipeline.DepthTest))
			return m_Impl->Fail("Weighted transparency requires blended coverage and read-only opaque depth testing");
		if (m_Impl->WeightedActive && !m_Impl->WeightedPass && m_Impl->CurrentPass.Type == PassType::Color &&
			!m_Impl->CurrentPass.Target && pipeline.BlendEnabled) return m_Impl->Fail("Weighted transparency must be prepared before blended geometry");
		if (m_Impl->VelocityCaptured && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("Shared opaque depth is read-only after velocity capture");
		if (m_Impl->VolumetricFog.Resolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("Shared opaque depth is read-only after fog composition");
		if (m_Impl->LightShafts.Resolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("Shared opaque depth is read-only after light shafts");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && pipeline.BlendEnabled != m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Opaque color precedes light shafts; transparency follows them");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			Techniques::UsesVolumetricFog(m_Impl->FrameFog) && pipeline.BlendEnabled != m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Opaque draws precede volumetric fog; transparent draws follow it");
		if (m_Impl->ReflectionApplied && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("Shared opaque depth is read-only after reflections");
		if (m_Impl->CurrentPass.Type == PassType::Velocity &&
			(!pipeline.DepthTest || pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
			return m_Impl->Fail("Velocity geometry requires opaque coverage and read-only depth testing");
		if (m_Impl->CurrentPass.Type == PassType::Color && m_Impl->FramePostProcessing.MotionBlur.Enabled)
		{
			if (!m_Impl->CurrentPass.Target && pipeline.BlendEnabled != m_Impl->MotionBlurResolved)
				return m_Impl->Fail("Opaque draws precede motion blur; transparent draws follow it");
		}
		if (m_Impl->CurrentPass.Type == PassType::Color && Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) &&
			!m_Impl->CurrentPass.Target && pipeline.BlendEnabled != m_Impl->TemporalAA.Resolved)
			return m_Impl->Fail("Opaque draws precede TAA; transparent draws follow its resolve");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->GIActive &&
            pipeline.BlendEnabled != m_Impl->GI.Resolved) return m_Impl->Fail("SSGI separates opaque and transparent material draws");
		if (m_Impl->CurrentPass.Type == PassType::Color && m_Impl->ReflectionActive)
		{
			if (!m_Impl->CurrentPass.Target && pipeline.BlendEnabled != m_Impl->ReflectionApplied)
				return m_Impl->Fail("Opaque draws precede reflections; transparent draws follow them");
		}
		if (m_Impl->CurrentPass.Type == PassType::Color && m_Impl->FramePostProcessing.DepthOfField.Enabled)
		{
			if (m_Impl->DepthOfFieldResolved && pipeline.DepthWrite)
				return m_Impl->Fail("Color draws cannot modify shared scene depth after depth of field");
			if (!m_Impl->CurrentPass.Target && m_Impl->DepthOfFieldResolved && !pipeline.BlendEnabled)
				return m_Impl->Fail("Only transparent scene draws may follow depth of field");
			if (!m_Impl->CurrentPass.Target && !m_Impl->DepthOfFieldResolved && pipeline.BlendEnabled)
				return m_Impl->Fail("Transparent pipelines must follow depth of field");
		}
		if ((m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture) &&
			(!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite ||
			 pipeline.SamplesShadowMap || pipeline.OutlineExtrusion || pipeline.Shading != m_Impl->FrameReflectionShading))
			return m_Impl->Fail("Reflection capture requires the frame shading model, opaque depth and no recursive effects/shadows");
		if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && !m_Impl->FramePlanar.Active())
			return m_Impl->Fail("An inactive planar capture must clear without drawing geometry");
		if ((m_Impl->CurrentPass.Type == PassType::Color || m_Impl->CurrentPass.Type == PassType::EnvironmentCapture ||
			m_Impl->CurrentPass.Type == PassType::PlanarCapture) && !m_Impl->EnsureTechnique(pipeline.Shading))
			return false;
		if (m_Impl->CurrentPass.Type == PassType::GBuffer &&
			(pipeline.BlendEnabled || pipeline.Shading != m_Impl->FrameReflectionShading ||
			 (m_Impl->DeferredActive && pipeline.Shading == Techniques::ShadingModel::Gouraud)))
			return m_Impl->Fail("Deferred geometry requires an opaque material and the frame's shading model");
		m_Impl->BoundPipeline = pipeline;
		m_Impl->PipelineBound = true;
		return true;
	}

	bool DX11RHI::BindWorldTransform(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen) return false;
		m_Impl->BoundWorld = value;
		m_Impl->WorldBound = true;
		return true;
	}

	bool DX11RHI::BindViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen) return false;
        if (m_Impl->CurrentPass.DepthPeelingLayer && value.Values!=m_Impl->FramePeeling.ViewProjection.Values)
            return m_Impl->Fail("OIT capture and replay require the same raster projection");
		if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && value.Values != m_Impl->FramePlanar.ClippedViewProjection.Values)
			return m_Impl->Fail("Planar geometry must use the frozen oblique capture projection");
		m_Impl->BoundViewProjection = value;
		m_Impl->ViewProjectionBound = true;
		return true;
	}

	bool DX11RHI::BindLightViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen) return false;
		m_Impl->BoundLightViewProjection = value;
		m_Impl->LightViewProjectionBound = true;
		return true;
	}

	bool DX11RHI::BindVertexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Vertex)
			return m_Impl->Fail("DX11 vertex buffer is invalid");
		m_Impl->BoundVertexBuffer = buffer;
		return true;
	}

	bool DX11RHI::BindIndexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Index)
			return m_Impl->Fail("DX11 index buffer is invalid");
		m_Impl->BoundIndexBuffer = buffer;
		return true;
	}

	bool DX11RHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchHandle& handle)
    {
        handle = {};
        if (!m_Impl->CheckThread("PrepareGpuDrivenBatch") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Instances.empty() || !desc.IndexCount)
            return m_Impl->Fail("DX11 GPU batch preparation requires an open frame outside a graphics pass");
        std::uint32_t index = 0;
        if (!m_Impl->PrepareGpuBatch(desc, index)) return false;
        handle = {m_Impl->GpuEpoch, index};
        ++m_Impl->PendingFrame.Statistics.GpuCullDispatchCount;
        return true;
    }

    bool DX11RHI::RenderVisibilityBuffer(const VisibilityBufferDesc& desc)
    {
        if (desc.MeshShaders) return m_Impl->Fail("DX11 does not expose native mesh shader stages");
        if (!m_Impl->CheckThread("RenderVisibilityBuffer") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->DeferredActive || m_Impl->GBufferWritten || m_Impl->FrameSampleCount != 1)
            return m_Impl->Fail("DX11 visibility geometry requires an unresolved single-sample deferred frame");
        if (!ValidateVisibilityFrame(desc, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError) ||
            !m_Impl->PrepareVisibilityBuffer(desc.Scene)) return false;
        if (desc.NaniteRaster && !m_Impl->PrepareNaniteFrameRaster(desc)) return false;
        if (!m_Impl->PrepareNaniteHardware(desc)) return false;
        if (!m_Impl->PrepareNaniteInstances(desc)) return false;
        if (!m_Impl->PrepareNaniteMaterials(desc)) return false;
        PassDesc pass; pass.Type = PassType::GBuffer; pass.ClearColor = desc.Background;
        if (!BeginPass(pass) || !m_Impl->DrawVisibilityBuffer(desc)) return false;
        RecordVisibilityDraws(m_Impl->PendingFrame, desc);
        return EndPass();
    }

    bool DX11RHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle handle)
    {
        if (!m_Impl->CheckThread("DrawGpuDrivenBatch") || !m_Impl->PassOpen || !handle.Frame ||
            handle.Frame != m_Impl->GpuEpoch || handle.Index >= m_Impl->GpuCursor || m_Impl->ActiveGpuBatch)
            return m_Impl->Fail("DX11 indirect draw has an invalid frame token or pass state");
        auto& batch = m_Impl->GpuBatches[handle.Index];
        const auto& desc = batch.Description;
        if (!GpuDrivenDrawMatches(desc, m_Impl->BoundWorld, m_Impl->BoundViewProjection, m_Impl->BoundPipeline, m_Impl->CurrentPass))
            return m_Impl->Fail("DX11 indirect draw does not match its prepared instance data");
        m_Impl->ActiveGpuBatch = &batch;
        struct Reset { Impl* State; ~Reset() { State->ActiveGpuBatch = nullptr; } } reset {m_Impl.get()};
        return DrawIndexedGeometry(desc.FirstIndex, desc.IndexCount, desc.BaseVertex, &desc.Instances);
    }

    bool DX11RHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
		return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, nullptr);
	}

	bool DX11RHI::DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
		const std::vector<DrawInstance>& instances)
	{
		return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, &instances);
	}

	bool DX11RHI::DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
		const std::vector<DrawInstance>* instances)
	{
    if (m_Impl->PassOpen && m_Impl->Hair.Active && HairFurReplayPass(m_Impl->CurrentPass.Type) &&
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Hair replay ends the pass geometry stream");
		if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("DX11 DrawIndexed is missing required state");
        if (instances && m_Impl->CurrentPass.DepthPeelingLayer) return m_Impl->Fail("OIT replay requires ordinary invariant draws");
		if (instances && UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        return m_Impl->Fail("POM material capture currently requires individual mesh draws");
    std::optional<InstanceConstantBlock> instanceConstants;
		if (instances)
		{
			instanceConstants.emplace();
			if (m_Impl->CurrentPass.Type == PassType::Outline || m_Impl->BoundPipeline.OutlineExtrusion ||
				!BuildInstanceConstants(*instances, m_Impl->BoundWorld, *instanceConstants))
				return m_Impl->Fail("DX11 instancing requires a bounded batch with finite transforms and common winding");
			instanceConstants->ViewProjection = m_Impl->BoundViewProjection;
			instanceConstants->Material.X = m_Impl->BoundPipeline.Color.W * m_Impl->BoundPipeline.MaterialOpacity;
		}
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("DX11 DrawIndexed buffers are invalid");
		if (!ValidateIndexedGeometryDraw(vertex->second.Description,index->second.Description,index->second.Bytes,
		    vertex->second.NaniteGeometry && index->second.NaniteGeometry,firstIndex,indexCount,baseVertex,m_Impl->LastError)) return false;

		ID3D11ShaderResourceView* diffuseView = nullptr;
		ID3D11ShaderResourceView* normalView = nullptr;
		ID3D11ShaderResourceView* opacityView = nullptr;
		auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
			ID3D11ShaderResourceView*& view)
		{
			if (!handle)
			{
				return true;
			}
			auto texture = m_Impl->Textures.find(handle.Value);
			if (texture == m_Impl->Textures.end()
				|| texture->second.Description.Format != TextureFormat::RGBA8
				|| !texture->second.ShaderResourceView
				|| !texture->second.SRGBShaderResourceView)
			{
				return m_Impl->Fail(std::string("DX11 ") + usage
					+ " texture is invalid");
			}
			view = texture->second.Description.ColorSpace ==
				TextureColorSpace::SRGB
				? texture->second.SRGBShaderResourceView.Get()
				: texture->second.ShaderResourceView.Get();
			return true;
		};
		if (!resolveSurfaceTexture(
				m_Impl->BoundPipeline.DiffuseTexture, "diffuse", diffuseView)
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.NormalTexture, "normal", normalView)
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.OpacityTexture, "opacity", opacityView))
		{
			return false;
		}

		ID3D11ShaderResourceView* shadowView = nullptr;
		if (m_Impl->CurrentPass.Type != PassType::Shadow
			&& m_Impl->BoundPipeline.SamplesShadowMap
			&& m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end()
				&& shadow->second.Description.Format == TextureFormat::Depth32)
			{
				shadowView = shadow->second.ShaderResourceView.Get();
			}
		}

		const DrawConstants constants = BuildRealtimeDrawConstants(
			m_Impl->BoundPipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
			m_Impl->BoundLightViewProjection, m_Impl->CurrentPass.Type,
			(bool(m_Impl->FrameRayShadowImage) || shadowView != nullptr || (m_Impl->FrameVirtualShadowAtlas && m_Impl->BoundPipeline.SamplesShadowMap && m_Impl->CurrentPass.Type==PassType::Color)), diffuseView != nullptr, normalView != nullptr, opacityView != nullptr,
			m_Impl->CurrentColorIsHdr);

		D3D11_MAPPED_SUBRESOURCE mapped = {};
		HRESULT result = m_Impl->Context->Map(m_Impl->ConstantBuffer.Get(), 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mapped);
		if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(constants)", result));
		std::memcpy(mapped.pData, &constants, sizeof(constants));
		m_Impl->Context->Unmap(m_Impl->ConstantBuffer.Get(), 0);

		const UINT stride = sizeof(Vertex);
		const UINT offset = 0;
		ID3D11Buffer* vertexBuffer = vertex->second.Buffer.Get();
		m_Impl->Context->IASetInputLayout(m_Impl->InputLayout.Get());
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->Context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
		m_Impl->Context->IASetIndexBuffer(m_Impl->ActiveNaniteDraw && m_Impl->ActiveNaniteDraw->Binning?m_Impl->ActiveNaniteDraw->Arguments->Buffer.Get():index->second.Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);
		const std::size_t techniqueIndex = (m_Impl->CurrentPass.Type == PassType::Color || m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture)
			? static_cast<std::size_t>(m_Impl->BoundPipeline.Shading) : 0;
		const auto surfaceIndex = techniqueIndex == static_cast<std::size_t>(Techniques::ShadingModel::PBR) && m_Impl->FrameAreaLights ? AreaLightTechniqueIndex : techniqueIndex;
		const bool gbuffer = m_Impl->CurrentPass.Type == PassType::GBuffer;
		m_Impl->Context->VSSetShader(gbuffer ? m_Impl->DeferredGeometryVertex.Get() : m_Impl->VertexShaders[surfaceIndex].Get(), nullptr, 0);
		ID3D11Buffer* constantBuffer = m_Impl->ConstantBuffer.Get();
		m_Impl->Context->VSSetConstantBuffers(0, 1, &constantBuffer);
		m_Impl->Context->PSSetConstantBuffers(0, 1, &constantBuffer);
		ID3D11PixelShader* pixelShader = gbuffer ? m_Impl->DeferredGeometryPixel.Get() : m_Impl->PixelShaders[surfaceIndex].Get();
		if (m_Impl->WeightedPass)
		{
			if (!m_Impl->EnsureWeightedTechnique(m_Impl->BoundPipeline.Shading)) return false;
			pixelShader = m_Impl->WeightedPixelShaders[ShadingTechniqueIndex(m_Impl->BoundPipeline.Shading, m_Impl->FrameAreaLights)].Get();
		}
		if (m_Impl->CurrentPass.Type == PassType::Shadow)
		{
			m_Impl->Context->VSSetShader(m_Impl->ShadowVertexShader.Get(), nullptr, 0);
			pixelShader = m_Impl->ShadowPixelShader.Get();
		}
		else if (m_Impl->CurrentPass.Type == PassType::Outline)
		{
			pixelShader = IsGpuPresentationEnabled() ? m_Impl->PixelShaders[ShadingTechniqueIndex(Techniques::ShadingModel::Phong,false)].Get() : m_Impl->OutlinePixelShader.Get();
		}
        if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS &&
            m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap && shadowView)
        {
            if (!m_Impl->PCSS.Prepare(m_Impl->Device.Get(), techniqueIndex, m_Impl->WeightedPass, m_Impl->LastError) ||
                !m_Impl->PCSS.Bind(m_Impl->Context.Get(), m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting,
                    m_Impl->FrameCascades, m_Impl->FrameShadowFilter.PCSS, true, m_Impl->LastError)) return false;
            pixelShader = m_Impl->PCSS.Pixel(techniqueIndex, m_Impl->WeightedPass);
        }
        const bool vsmCaster = m_Impl->CurrentPass.Type == PassType::Shadow &&
            m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM;
        if (vsmCaster)
        {
            if (!m_Impl->VSM.BindCaster(m_Impl->Context.Get(), m_Impl->BoundViewProjection,
                m_Impl->FrameShadowFilter.VSM, m_Impl->LastError)) return false;
            m_Impl->Context->VSSetShader(m_Impl->VSM.CasterVertex(false), nullptr, 0);
            pixelShader = m_Impl->VSM.Pixel(DX11VSM::Caster);
        }
        if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM &&
            m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap && shadowView)
        {
            if (!m_Impl->CheckVSMMomentViews() ||
                !m_Impl->VSM.Prepare(m_Impl->Device.Get(), techniqueIndex, m_Impl->WeightedPass, m_Impl->LastError) ||
                !m_Impl->VSM.BindReceiver(m_Impl->Context.Get(), m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting,
                    m_Impl->FrameCascades, m_Impl->FrameShadowFilter.VSM, true, m_Impl->LastError)) return false;
            pixelShader = m_Impl->VSM.Pixel(techniqueIndex, m_Impl->WeightedPass);
            shadowView = m_Impl->VSM.View(m_Impl->LastShadowTexture);
        }

        if (m_Impl->CurrentPass.Type == PassType::ContactDepth)
        {
            if (!m_Impl->ContactShadows.PrepareDepth(m_Impl->Device.Get(), m_Impl->LastError)) return false;
            m_Impl->Context->VSSetShader(m_Impl->ContactShadows.DepthVS(), nullptr, 0);
            pixelShader = m_Impl->ContactShadows.DepthPS();
        }
        const bool contactReceiver = m_Impl->ContactReceiverActive && m_Impl->CurrentPass.Type == PassType::Color &&
            !m_Impl->CurrentPass.Target && !m_Impl->BoundPipeline.BlendEnabled;
        if (contactReceiver && !m_Impl->LightListsActive)
        {
            const auto filter = m_Impl->BoundPipeline.SamplesShadowMap && shadowView ? m_Impl->FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard;
            pixelShader = m_Impl->ContactShadows.Receiver(m_Impl->Device.Get(), m_Impl->BoundPipeline.Shading,
                filter, false, m_Impl->FrameSampleCount > 1, m_Impl->LastError);
            if (!pixelShader || !m_Impl->ContactShadows.BindReceiver(m_Impl->Device.Get(), m_Impl->Context.Get(), true, m_Impl->LastError)) return false;
        }
        const bool forwardPlusSurface = m_Impl->LightListsActive && (m_Impl->CurrentPass.Type == PassType::Color ||
            m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture);
        if (m_Impl->LightListsActive && (gbuffer || m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) &&
            !m_Impl->LightListFrame.ValidateDraw(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, true, m_Impl->LastError)) return false;
        if (forwardPlusSurface || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth)
        {
            const bool depthOnly = m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth;
            if (!depthOnly && m_Impl->BoundPipeline.Shading == Techniques::ShadingModel::Gouraud)
                return m_Impl->Fail("Forward+ material lists cannot supply Gouraud vertex lighting");
            const auto filter = !depthOnly && m_Impl->CurrentPass.Type == PassType::Color &&
                m_Impl->BoundPipeline.SamplesShadowMap && shadowView ? m_Impl->FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard;
            const auto program = depthOnly ? DX11ForwardPlus::DepthSurface : m_Impl->WeightedPass ? DX11ForwardPlus::WeightedSurface :
                contactReceiver && m_Impl->FrameSampleCount > 1 ? DX11ForwardPlus::ContactSurface : DX11ForwardPlus::OpaqueSurface;
            const auto clusterProgram = static_cast<DX11ClusteredForward::SurfaceProgram>(program);
            if (!(m_Impl->ClusteredActive ? m_Impl->ClusteredForward.PrepareMaterial(m_Impl->Device.Get(), filter, clusterProgram, m_Impl->LastError) :
                m_Impl->TileProducer().PrepareMaterial(m_Impl->Device.Get(), filter, program, m_Impl->LastError))) return false;
            if (!depthOnly)
            {
                const bool mainCamera = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target;
                const auto selection = !mainCamera ? LightListSelection::FullScene :
                    m_Impl->BoundPipeline.BlendEnabled ? LightListSelection::Transparent : LightListSelection::Opaque;
                LightListSurfaceConstants surface;
                if (!m_Impl->LightListFrame.Surface(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, selection, surface, m_Impl->LastError) ||
                    !(m_Impl->ClusteredActive ? m_Impl->ClusteredForward.BindSurface(m_Impl->Context.Get(), surface, m_Impl->LastError) :
                        m_Impl->TileProducer().BindSurface(m_Impl->Context.Get(), surface, m_Impl->LastError)) ||
                    !m_Impl->ContactShadows.BindReceiver(m_Impl->Device.Get(), m_Impl->Context.Get(), contactReceiver, m_Impl->LastError)) return false;
            }
            m_Impl->Context->VSSetShader(m_Impl->ClusteredActive ? m_Impl->ClusteredForward.MaterialVertex() : m_Impl->TileProducer().MaterialVertex(), nullptr, 0);
            pixelShader = m_Impl->ClusteredActive ? m_Impl->ClusteredForward.MaterialPixel(filter, clusterProgram) : m_Impl->TileProducer().MaterialPixel(filter, program);
        }
        if (m_Impl->UsesMaterialStage(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        {
            const auto material=SelectMaterialTechnique(m_Impl->BoundPipeline,m_Impl->WeightedPass);
            pixelShader=m_Impl->MaterialPixel(material,unsigned(m_Impl->BoundPipeline.Shading),m_Impl->WeightedPass,gbuffer);
            if (!pixelShader || (!gbuffer && !m_Impl->BindPCF(shadowView!=nullptr))) return false;
        }
        if (m_Impl->UsesSurfacePCF(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        {
            pixelShader=m_Impl->PCFPixel(unsigned(m_Impl->BoundPipeline.Shading),m_Impl->WeightedPass);
            if (!pixelShader || !m_Impl->BindPCF(true)) return false;
        }
        if (m_Impl->FrameRayShadowImage && !m_Impl->FrameAreaLights && m_Impl->CurrentPass.Type==PassType::Color &&
            m_Impl->BoundPipeline.SamplesShadowMap && !m_Impl->LightListsActive)
        {
            pixelShader=m_Impl->RayShadowPixel(unsigned(m_Impl->BoundPipeline.Shading),m_Impl->WeightedPass);
            if (!pixelShader) return false;
        }
        m_Impl->BindRayShadowImage();
        m_Impl->Context->PSSetShader(pixelShader, nullptr, 0);
        const bool translucentGI=m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
            m_Impl->BoundPipeline.BlendEnabled && m_Impl->FrameGI.Lumen.Translucency.Enabled && m_Impl->LumenVolumeFrame.Grid.W!=0;
        auto* volumeUniform=translucentGI?m_Impl->LumenVolumeUniform.Get():nullptr;
        auto* volumeImage=translucentGI?m_Impl->LumenVolumeAtlas.View.Get():nullptr;
        m_Impl->Context->PSSetConstantBuffers(Impl::LumenVolumeConstantSlot,1,&volumeUniform);
        m_Impl->Context->PSSetShaderResources(Impl::LumenVolumeTextureSlot,1,&volumeImage);
        auto* volumeSampler=m_Impl->SurfaceSampler.Get();
        m_Impl->Context->PSSetSamplers(Impl::LumenVolumeSamplerSlot,1,&volumeSampler);
		ID3D11RasterizerState* rasterizer = m_Impl->CurrentPass.Type == PassType::Shadow
			? m_Impl->ShadowRasterState.Get()
			: (m_Impl->BoundPipeline.OutlineExtrusion
				? m_Impl->OutlineRasterState.Get()
				: (m_Impl->BoundPipeline.Fill == FillMode::Wireframe
					? m_Impl->WireRasterState.Get() : m_Impl->SolidRasterState.Get()));
		if ((m_Impl->BoundPipeline.BackFaceCulling || RealtimeWorldIsMirrored(m_Impl->BoundWorld) || m_Impl->CurrentPass.Type == PassType::PlanarCapture ||
			(m_Impl->BoundPipeline.Fill == FillMode::Wireframe && m_Impl->FrameSampleCount > 1))
			&& !m_Impl->BoundPipeline.OutlineExtrusion)
		{
			rasterizer = m_Impl->GetOrientedRasterizer();
			if (!rasterizer) return false;
		}
		m_Impl->Context->RSSetState(rasterizer);
		m_Impl->Context->OMSetDepthStencilState(m_Impl->CurrentPass.DepthPeelingLayer ? m_Impl->Peeling.EqualDepthState() : !m_Impl->BoundPipeline.DepthTest
			? m_Impl->DepthDisabledState.Get()
			: (m_Impl->BoundPipeline.DepthWrite ? m_Impl->DepthWriteState.Get() : m_Impl->DepthReadState.Get()), 0);
		const float blendFactor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		ID3D11BlendState* blendState = nullptr;
		if (!m_Impl->BoundPipeline.ColorWrite && !vsmCaster)
		{
			blendState = m_Impl->ColorWriteDisabledState.Get();
		}
		else if (m_Impl->CurrentPass.Type == PassType::Color
			&& m_Impl->BoundPipeline.BlendEnabled)
		{
			blendState = m_Impl->WeightedPass ? m_Impl->WeightedTransparency.AccumulationBlend() : m_Impl->AlphaBlendState.Get();
		}
		m_Impl->Context->OMSetBlendState(
			blendState, blendFactor, 0xFFFFFFFFu);
		ID3D11ShaderResourceView* shaderViews[] = {
			shadowView,
			diffuseView,
			normalView,
			opacityView
		};
		m_Impl->Context->PSSetShaderResources(0, 4, shaderViews);
		ID3D11SamplerState* sampler = m_Impl->MaterialSampler();
		m_Impl->Context->PSSetSamplers(0, 1, &sampler);
		if (!m_Impl->VolumetricFog.BindSurface(m_Impl->Device.Get(), m_Impl->Context.Get(), m_Impl->CurrentPass.Type == PassType::Color &&
			!m_Impl->CurrentPass.Target && m_Impl->BoundPipeline.BlendEnabled && Techniques::UsesVolumetricFog(m_Impl->FrameFog),
            sampler, m_Impl->LastError)) return false;
		m_Impl->WeightedTransparency.BindSurface(m_Impl->Context.Get(), m_Impl->WeightedPass);
		if (m_Impl->CurrentPass.Type == PassType::Velocity)
		{
			MotionBlurConstants motion {};
			if (!BuildMotionGeometryConstants(m_Impl->BoundPipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
				GetMotionTransform(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				diffuseView != nullptr, opacityView != nullptr, motion))
				return m_Impl->Fail("Velocity requires finite current and previous transforms");
			if (instances) motion.Options.W = 1; // Per-instance previous matrices encode validity.
			if (!m_Impl->UploadMotionConstants(motion)) return false;
			m_Impl->Context->VSSetShader(m_Impl->MotionGeometryVertex.Get(), nullptr, 0);
			m_Impl->Context->PSSetShader(m_Impl->MotionGeometryPixel.Get(), nullptr, 0);
			ID3D11Buffer* motionBuffer = m_Impl->MotionConstantBuffer.Get();
			m_Impl->Context->VSSetConstantBuffers(0, 1, &motionBuffer);
			m_Impl->Context->PSSetConstantBuffers(0, 1, &motionBuffer);
			ID3D11ShaderResourceView* motionInputs[5] {nullptr, nullptr, nullptr, diffuseView, opacityView};
			m_Impl->Context->PSSetShaderResources(0, 5, motionInputs);
		}
		m_Impl->BindCascades(m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap);
		if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->FrameCascades.Frame.CascadeCount != 0 && !vsmCaster)
		{
			m_Impl->Context->VSSetShader(m_Impl->CascadeVertexShader.Get(), nullptr, 0);
			m_Impl->Context->PSSetShader(m_Impl->CascadePixelShader.Get(), nullptr, 0);
		}
		if (instances)
		{
			const auto stage = m_Impl->CurrentPass.Type == PassType::Velocity ? InstanceVertexProgram::Velocity :
				m_Impl->CurrentPass.Type == PassType::Shadow ? InstanceVertexProgram::Shadow :
				gbuffer ? InstanceVertexProgram::Deferred : (m_Impl->CurrentPass.Type == PassType::ContactDepth ||
                    m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) ? InstanceVertexProgram::Forward :
                m_Impl->BoundPipeline.Shading == Techniques::ShadingModel::Gouraud ?
				InstanceVertexProgram::Gouraud : InstanceVertexProgram::Forward;
			const auto program = static_cast<std::size_t>(stage);
			if (!m_Impl->EnsureInstancedVertex(program)) return false;
            if (!m_Impl->ActiveGpuBatch)
            {
                D3D11_MAPPED_SUBRESOURCE instanceMapping {};
                result = m_Impl->Context->Map(m_Impl->InstanceBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &instanceMapping);
                if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(instance constants)", result));
                std::memcpy(instanceMapping.pData, &*instanceConstants, sizeof(InstanceConstantBlock));
                m_Impl->Context->Unmap(m_Impl->InstanceBuffer.Get(), 0);
            }
			ID3D11Buffer* buffer = m_Impl->ActiveGpuBatch ? m_Impl->ActiveGpuBatch->VertexConstants.Get() : m_Impl->InstanceBuffer.Get();
			m_Impl->Context->VSSetConstantBuffers(3, 1, &buffer);
			m_Impl->Context->VSSetShader(vsmCaster ? m_Impl->VSM.CasterVertex(true) : m_Impl->InstanceVertices[program].Get(), nullptr, 0);
            if (m_Impl->ActiveGpuBatch) m_Impl->Context->DrawIndexedInstancedIndirect(m_Impl->ActiveGpuBatch->Arguments.Get(), 0);
            else m_Impl->Context->DrawIndexedInstanced(indexCount, static_cast<UINT>(instances->size()), firstIndex, baseVertex, 0);
		}
        else
        {
            if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type) && !m_Impl->BindParallax()) return false;
            if (m_Impl->ActiveNaniteDraw) m_Impl->Context->DrawIndexedInstancedIndirect(m_Impl->ActiveNaniteDraw->Arguments->Buffer.Get(),m_Impl->NaniteArgumentOffset());
            else m_Impl->Context->DrawIndexed(indexCount, firstIndex, baseVertex);
        }

		if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type)) ++m_Impl->PendingFrame.Statistics.ParallaxMaterialDrawCount;
    DrawCallRecord record;
		record.VertexBuffer = m_Impl->BoundVertexBuffer;
		record.IndexBuffer = m_Impl->BoundIndexBuffer;
		record.Pipeline = m_Impl->BoundPipeline;
		record.WorldTransform = m_Impl->BoundWorld;
		record.Motion = GetMotionTransform();
		record.FirstIndex = firstIndex;
		record.IndexCount = indexCount;
		record.BaseVertex = baseVertex;
		if (instances) record.Instances = *instances;
		record.GPUDriven = m_Impl->ActiveGpuBatch != nullptr;
        record.NaniteIndirect=m_Impl->ActiveNaniteDraw!=nullptr; record.NaniteQueryIndex=record.NaniteIndirect?m_Impl->ActiveNaniteQuery:0;
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
		FrameStatistics& stats = m_Impl->PendingFrame.Statistics;
		++stats.DrawCallCount;
		stats.InstancedDrawCallCount += instances ? 1u : 0u;
		if (m_Impl->ActiveGpuBatch)
        {
            ++stats.IndirectDrawCallCount;
            stats.GpuCandidateInstanceCount += instances->size();
            stats.GpuCandidateTriangleCount += std::uint64_t(indexCount / 3) * instances->size();
        }
        else if (m_Impl->ActiveNaniteDraw)
        {
            ++stats.IndirectDrawCallCount; ++stats.NaniteIndirectDrawCount;
            stats.NaniteCandidateTriangles+=indexCount/3;
        }
        else stats.SubmittedTriangleCount += std::uint64_t(indexCount / 3) * (instances ? instances->size() : 1);
		stats.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1u : 0u;
        stats.ContactDepthDrawCallCount += m_Impl->CurrentPass.Type == PassType::ContactDepth ? 1u : 0u;
        stats.ForwardPlusDepthDrawCallCount += m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth ? 1u : 0u;
		stats.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1u : 0u;
		return true;
	}

	bool DX11RHI::EndPass()
	{
    if (m_Impl->FrameOpen && m_Impl->PassOpen && m_Impl->Hair.Active && !m_Impl->Hair.Frame.Geometry->Indices.empty() &&
        HairFurReplayPass(m_Impl->CurrentPass.Type) && (m_Impl->CurrentPass.Type!=PassType::PlanarCapture || m_Impl->FramePlanar.Active()) &&
        m_Impl->Hair.ReplayedPass!=m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Complete requested hair replay before ending its native geometry pass");
		if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX11 has no open pass");
		constexpr UINT SurfaceInputCount = 19;
		ID3D11ShaderResourceView* nullViews[SurfaceInputCount] = {};
		m_Impl->Context->PSSetShaderResources(0, SurfaceInputCount, nullViews);
        if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM)
        {
            if (!m_Impl->VSM.FilterMap(m_Impl->Context.Get(), m_Impl->CurrentPass.Target,
                m_Impl->FrameShadowFilter.VSM, m_Impl->LastError)) return false;
            m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount += 2;
        }

		if (m_Impl->CurrentPass.Type == PassType::EnvironmentCapture)
		{
			m_Impl->EnvironmentFacesWritten |= 1u << m_Impl->CurrentPass.EnvironmentFaceIndex;
			++m_Impl->PendingFrame.Statistics.EnvironmentCaptureFaceCount;
		}
		if (m_Impl->CurrentPass.Type == PassType::GBuffer) m_Impl->GBufferWritten = true;
        if (m_Impl->CurrentPass.Type == PassType::ContactDepth) m_Impl->ContactDepthWritten = true;
        if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) m_Impl->ForwardPlusDepthWritten = true;
		if (m_Impl->CurrentPass.Type == PassType::PlanarCapture)
		{
			m_Impl->PlanarWritten = true;
			++m_Impl->PendingFrame.Statistics.PlanarCapturePassCount;
		}
		if (m_Impl->CurrentPass.Type == PassType::Velocity) m_Impl->VelocityCaptured = true;
		if (m_Impl->WeightedPass) m_Impl->WeightedTransparency.Accumulated = true;
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target)
		{
			m_Impl->SceneColorWritten = true;
			if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved)
			{
				m_Impl->SamplesTransparentWritten = true;
				m_Impl->SamplesColorResolved = false;
			}
		}
        if (m_Impl->CurrentPass.DepthPeelingLayer && !m_Impl->Peeling.Sequence.Advance(DepthPeelingEvent::EndShading,m_Impl->LastError)) return false;
		m_Impl->PassOpen = false;
		return true;
	}

    bool DX11RHI::PrepareLightListFrame(const LightListFrameDesc& desc)
    {
        if (!m_Impl->CheckThread("PrepareLightListFrame") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Layout != (m_Impl->ClusteredActive ? LightListLayout::Clusters : LightListLayout::Tiles) || !m_Impl->LightListsActive || !m_Impl->PendingFrame.Passes.empty())
            return m_Impl->Fail("Forward+ light upload must precede all scene passes");
        return m_Impl->LightListFrame.Prepare(desc, m_Impl->LastError) &&
            (m_Impl->ClusteredActive ? m_Impl->ClusteredForward.UploadScene(m_Impl->Device.Get(), m_Impl->Context.Get(), desc.Lights, m_Impl->LastError) :
                m_Impl->TileProducer().UploadScene(m_Impl->Device.Get(), m_Impl->Context.Get(), desc.Lights, m_Impl->LastError));
    }

    bool DX11RHI::GenerateLightLists()
    {
        if (!m_Impl->CheckThread("GenerateLightLists") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->LightListsActive || !m_Impl->LightListFrame.Prepared() || m_Impl->LightListFrame.Assigned() ||
            m_Impl->SceneColorWritten || (!m_Impl->ClusteredActive && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten))
            return m_Impl->Fail("Forward+ assignment requires completed opaque depth before main-view lighting");
        m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
        constexpr UINT FirstMaterialStorage = 16, MaterialStorageCount = 3;
        ID3D11ShaderResourceView* empty[MaterialStorageCount] {};
        m_Impl->Context->PSSetShaderResources(FirstMaterialStorage, MaterialStorageCount, empty);
        auto* depth = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisampling.DepthView.Get() : m_Impl->DepthShaderView.Get();
        const auto& frame = m_Impl->LightListFrame.Description();
        if (m_Impl->ClusteredActive)
        {
            if (!m_Impl->ClusteredForward.Generate(m_Impl->Device.Get(), m_Impl->Context.Get(),
                    m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
                !m_Impl->LightListFrame.CompleteAssignment(m_Impl->ClusteredForward.Grid(), m_Impl->LastError)) return false;
            ++m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount;
            return true;
        }
        if (!m_Impl->TileProducer().Generate(m_Impl->Device.Get(), m_Impl->Context.Get(), depth,
                frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
            !m_Impl->LightListFrame.CompleteAssignment(m_Impl->TileProducer().Grid(), m_Impl->LastError)) return false;
        ++(m_Impl->TiledActive ? m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount : m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount);
        return true;
    }

    bool DX11RHI::GenerateContactShadows(const ContactShadowDesc& desc)
    {
        if (!m_Impl->CheckThread("GenerateContactShadows") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->FrameContact.Enabled || m_Impl->ContactGenerated || m_Impl->SceneColorWritten ||
            (!m_Impl->ContactDepthWritten && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten))
            return m_Impl->Fail("Contact visibility requires completed opaque depth before lighting, once per frame");
        if (m_Impl->LightListsActive)
        {
            if (!m_Impl->LightListFrame.Assigned())
                return m_Impl->Fail("DX11 Forward+ contact visibility must follow light assignment");
            if (!m_Impl->LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
        }

        m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
        auto lighting = desc.Lighting;
        if (lighting.SceneLights) lighting.SceneLights->WriteLegacySubset(lighting);
        if (!m_Impl->Hair.ValidateContactProducer(desc.ViewProjection,lighting,m_Impl->LastError)) return false;
        const auto light = lighting.ShadowLightIndex;
        m_Impl->ContactReceiverActive = light >= 0 && light < int(std::min(lighting.LightCount, MaximumSceneLights)) &&
            lighting.Lights[light].CastShadows;
        if (m_Impl->ContactReceiverActive)
        {
            ContactShadowDispatch dispatch; dispatch.ViewProjection = desc.ViewProjection; dispatch.Light = lighting.Lights[light];
            dispatch.MaximumDistance = m_Impl->FrameContact.MaximumDistance;
            dispatch.Parameters.StartBias = m_Impl->FrameContact.StartBias; dispatch.Parameters.Thickness = m_Impl->FrameContact.Thickness;
            dispatch.Parameters.MaximumSteps = m_Impl->FrameContact.MaximumSteps;
            auto* depth = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisampling.DepthView.Get() : m_Impl->DepthShaderView.Get();
            if (!m_Impl->ContactShadows.Generate(m_Impl->Device.Get(), m_Impl->Context.Get(), depth, dispatch, m_Impl->LastError)) return false;
            ++m_Impl->PendingFrame.Statistics.ContactShadowPassCount;
        }
        m_Impl->Hair.RecordContact(desc.ViewProjection,lighting,m_Impl->ContactReceiverActive);
        m_Impl->ContactGenerated = true;
        return true;
    }

	bool DX11RHI::DrawHairFurVelocity(const HairFurVelocityDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurVelocity") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->CurrentPass.Type!=PassType::Velocity ||
        !m_Impl->Hair.Active || !m_Impl->Hair.Resolved || !m_Impl->Hair.VelocityRequested || m_Impl->Hair.VelocityResolved ||
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size() || !m_Impl->Hair.MatchesPrimaryView(desc.View))
        return m_Impl->Fail("Fiber velocity requires matching completed primary geometry, once at the read-only velocity pass tail");
    if (m_Impl->ViewProjectionBound && desc.View.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Fiber velocity projection must match the current mesh velocity pass");
    HairFurConstants constants;
    if (!BuildHairFurVelocityConstants(desc,m_Impl->Hair,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,constants,m_Impl->LastError) ||
        !m_Impl->ReplayHairFur(constants,HairFurRasterPhase::Velocity)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size(); m_Impl->Hair.VelocityResolved=true;
    ++m_Impl->PendingFrame.Statistics.HairFurVelocityDrawCount; return true;
}

bool DX11RHI::DrawHairFurInPass(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurInPass") || !m_Impl->FrameOpen || !m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved ||
        !HairFurReplayPass(m_Impl->CurrentPass.Type) || m_Impl->CurrentPass.Type==PassType::Velocity || m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size())
        return m_Impl->Fail("Hair replay requires an open depth, shadow or capture pass, once at its geometry tail");
    const auto type=m_Impl->CurrentPass.Type;
    if (type==PassType::PlanarCapture && !m_Impl->FramePlanar.Active()) return m_Impl->Fail("Inactive planar captures must remain empty");
    if (HairFurCameraDepthPass(type) && !m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth views must match");
    if ((HairFurCameraDepthPass(type) || type==PassType::Shadow) && m_Impl->ViewProjectionBound && desc.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Hair replay projection must match the current mesh pass");
    if (type==PassType::EnvironmentCapture || type==PassType::PlanarCapture)
    {
        auto expected=desc;
        if (type==PassType::EnvironmentCapture)
        {
            const auto& face=m_Impl->FrameEnvironment.Faces[m_Impl->CurrentPass.EnvironmentFaceIndex];
            expected.ViewProjection=EnvironmentRasterProjection(face,GetBackend()); expected.Lighting.CameraPosition=m_Impl->FrameEnvironment.Position;
            expected.Forward=face.Forward; expected.Orthographic=false; expected.ClipPlane={0,0,0,0};
        }
        else
        {
            expected.ViewProjection=m_Impl->FramePlanar.ClippedViewProjection; expected.Lighting.CameraPosition=m_Impl->FramePlanar.Position;
            expected.ClipPlane=m_Impl->FramePlanar.ClipPlane;
        }
        if (!SameHairFurRasterView(desc,expected)) return m_Impl->Fail("Hair captures require the frozen reflection camera and clip plane");
    }
    const auto phase=HairFurReplayPhase(type,m_Impl->FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM); HairFurConstants constants;
    if (!BuildHairFurReplayConstants(desc,m_Impl->Hair.Frame,phase,m_Impl->FrameShadowFilter.VSM,constants,m_Impl->LastError) || !m_Impl->ReplayHairFur(constants,phase)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size();
    if (type==PassType::Shadow) m_Impl->Hair.RecordShadowView(m_Impl->CurrentPass.Target,desc.ViewProjection);
    if (HairFurCameraDepthPass(type)) m_Impl->Hair.RecordPrimaryView(desc);
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty())
    {
        if (type==PassType::Shadow) ++m_Impl->PendingFrame.Statistics.HairFurShadowDrawCount;
        else if (HairFurCameraDepthPass(type)) ++m_Impl->PendingFrame.Statistics.HairFurDepthDrawCount;
        else ++m_Impl->PendingFrame.Statistics.HairFurCaptureDrawCount;
    }
    return true;
}

bool DX11RHI::DrawHairFur(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFur") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved || !m_Impl->GBufferWritten ||
        !m_Impl->SceneColorWritten || (m_Impl->DeferredActive && !m_Impl->DeferredResolved) || m_Impl->SamplesOpaqueResolved || m_Impl->VelocityCaptured ||
        !HairFurPrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Hair requires completed opaque lighting, once before MSAA resolution and screen-space effects");
    if (!m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth and primary surface require identical frozen raster inputs");
    HairFurConstants constants;
    if (!BuildHairFurConstants(desc.ViewProjection,desc.Lighting,m_Impl->Hair.Frame.Settings,*m_Impl->Hair.Frame.Lights,
        desc.Orthographic,desc.Forward,desc.ClipPlane,constants)) return m_Impl->Fail("Hair camera and lighting constants must be finite");
    if (!m_Impl->PrepareHairShadowReceiver(desc,constants) || !m_Impl->RenderHairFur(constants)) return false;
    m_Impl->Hair.RecordPrimaryView(desc); m_Impl->Hair.Resolved=true;
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty()) ++m_Impl->PendingFrame.Statistics.HairFurPassCount;
    return true;
}

bool DX11RHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc)
{
    if (!m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before subsurface diffusion");
    if (!m_Impl->CheckThread("ApplySubsurfaceScattering") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || !m_Impl->DeferredResolved || !m_Impl->SubsurfaceHasScene || m_Impl->SubsurfaceResolved || m_Impl->TiledActive ||
        m_Impl->PendingFrame.Statistics.SubsurfacePassCount ||
        (m_Impl->FrameSampleCount>1 && (!m_Impl->SamplesOpaqueResolved || m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)))
        return m_Impl->Fail("Subsurface diffusion requires resolved opaque scene lighting, once before transparency");
    if (!SubsurfacePrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Subsurface diffusion must precede AO, GI, reflections and lens/temporal effects");
    if (m_Impl->SubsurfaceRequested && !(Techniques::SanitizeSubsurfaceScatteringSettings(desc.Settings)==m_Impl->FrameSubsurface))
        return m_Impl->Fail("Subsurface controls must match the immutable frame settings");
    SubsurfaceScatteringConstants constants;
    if (!Techniques::UsesSubsurfaceScattering(desc.Settings) ||
        !BuildSubsurfaceScatteringConstants(m_Impl->SubsurfaceLighting.ViewProjection,desc.Settings,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,Techniques::SubsurfacePass::Horizontal,constants))
        return m_Impl->Fail("Subsurface diffusion requires active finite profile and camera parameters");
    if (!m_Impl->SnapshotSubsurfaceScene()) return false;
    auto source=m_Impl->SubsurfaceLighting; source.Output=DeferredOutputMode::DirectDiffuse;
    m_Impl->SubsurfaceProducing=true;
    struct Reset { bool& Value; ~Reset() { Value=false; } } reset{m_Impl->SubsurfaceProducing};
    const bool produced=DrawDeferredLighting(source); m_Impl->SubsurfaceProducing=false;
    return produced && m_Impl->CaptureSubsurfaceSource() && m_Impl->FilterSubsurface(constants);
}

bool DX11RHI::DrawDeferredLighting(const DeferredLightingDesc& desc)
	{
        if (desc.Output == DeferredOutputMode::DirectDiffuse && m_Impl->TiledActive)
            return m_Impl->Fail("Direct-diffuse output requires ordinary deferred lighting");
		if (!m_Impl->CheckThread("DrawDeferredLighting") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->DeferredActive || !m_Impl->GBufferWritten || (m_Impl->DeferredResolved && !m_Impl->SubsurfaceProducing))
			return m_Impl->Fail("Deferred lighting requires a completed G-buffer and executes once per frame");
		if (desc.Shading != GetTechniqueSettings().Shading)
			return m_Impl->Fail("Deferred lighting must use the G-buffer's shading model");
        if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("Deferred lighting requires current contact visibility");
		ID3D11ShaderResourceView* shadowView = nullptr;
		const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
		if (shadow != m_Impl->Textures.end()) shadowView = shadow->second.ShaderResourceView.Get();
        if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM && desc.SamplesShadowMap && shadowView)
        {
            if (!m_Impl->CheckVSMMomentViews()) return false;
            shadowView = m_Impl->VSM.View(m_Impl->LastShadowTexture);
        }

		DeferredConstants constants {};
		if (!BuildDeferredConstants(desc, (shadowView != nullptr || bool(m_Impl->FrameVirtualShadowAtlas) || bool(m_Impl->FrameRayShadowImage)), m_Impl->DeferredBackground, constants))
			return m_Impl->Fail("Deferred position reconstruction requires an invertible finite view-projection matrix");
		D3D11_MAPPED_SUBRESOURCE mapped {};
		const HRESULT result = m_Impl->Context->Map(m_Impl->DeferredConstantBuffer.Get(), 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mapped);
		if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(deferred constants)", result));
        TiledDeferredReconstruction reconstruction;
        RealtimeDrawConstants surface;
        if (m_Impl->TiledActive)
        {
            if (!BuildTiledDeferredConstants(desc, (shadowView != nullptr || bool(m_Impl->FrameVirtualShadowAtlas) || bool(m_Impl->FrameRayShadowImage)), m_Impl->DeferredBackground, surface, reconstruction))
            { m_Impl->Context->Unmap(m_Impl->DeferredConstantBuffer.Get(), 0); return m_Impl->Fail("Invalid tiled camera"); }
            std::memcpy(mapped.pData, &reconstruction, sizeof(reconstruction));
        }
        else std::memcpy(mapped.pData, &constants, sizeof(constants));
		m_Impl->Context->Unmap(m_Impl->DeferredConstantBuffer.Get(), 0);
        if (m_Impl->TiledActive)
        {
            LightListSurfaceConstants selection;
            if (!m_Impl->LightListFrame.Surface(desc.Lighting, desc.ViewProjection, LightListSelection::Opaque, selection, m_Impl->LastError) ||
                !m_Impl->TiledDeferred.BindSurface(m_Impl->Context.Get(), selection, m_Impl->LastError)) return false;
            if (FAILED(m_Impl->Context->Map(m_Impl->ConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                return m_Impl->Fail("DX11 tiled material constants map failed");
            std::memcpy(mapped.pData, &surface, sizeof(surface));
            m_Impl->Context->Unmap(m_Impl->ConstantBuffer.Get(), 0);
            ID3D11Buffer* material = m_Impl->ConstantBuffer.Get();
            m_Impl->Context->PSSetConstantBuffers(0, 1, &material);
        }
		constexpr UINT InputCount = 4 + GBufferAttachmentCount + 1;
		ID3D11ShaderResourceView* inputs[InputCount] {};
		m_Impl->Context->PSSetShaderResources(0, InputCount, inputs);
		// DSV must be detached before the same opaque depth is sampled as an SRV.
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		const bool multisample = m_Impl->FrameSampleCount > 1;
		ID3D11RenderTargetView* target = multisample ? m_Impl->Multisampling.Color.Target.Get() : m_Impl->HdrSurfaces[Impl::HdrScene].Target.Get();
		m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
		inputs[0] = shadowView;
		for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
			inputs[4 + index] = multisample ? m_Impl->Multisampling.Materials[index].View.Get() : m_Impl->GBufferSurfaces[index].View.Get();
		inputs[InputCount - 1] = multisample ? m_Impl->Multisampling.DepthView.Get() : m_Impl->DepthShaderView.Get();
		m_Impl->Context->PSSetShaderResources(0, InputCount, inputs);
		ID3D11Buffer* constantBuffer = m_Impl->DeferredConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(1, 1, &constantBuffer);
		m_Impl->Context->IASetInputLayout(nullptr);
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->BindCascades(desc.SamplesShadowMap);
		m_Impl->Context->VSSetShader(m_Impl->DeferredLightVertex.Get(), nullptr, 0);
        ID3D11PixelShader* lightingPixel = multisample ? m_Impl->DeferredMultisamplePixel.Get() : m_Impl->DeferredLightPixel.Get();
        if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS && desc.SamplesShadowMap && shadowView)
        {
            const auto program = multisample ? DX11PCSS::DeferredMultisample : DX11PCSS::Deferred;
            if (!m_Impl->PCSS.Prepare(m_Impl->Device.Get(), program, false, m_Impl->LastError) ||
                !m_Impl->PCSS.Bind(m_Impl->Context.Get(), desc.LightViewProjection, desc.Lighting, m_Impl->FrameCascades,
                    m_Impl->FrameShadowFilter.PCSS, true, m_Impl->LastError)) return false;
            lightingPixel = m_Impl->PCSS.Pixel(program);
        }
        if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM && desc.SamplesShadowMap && shadowView)
        {
            const auto program = multisample ? DX11VSM::DeferredMultisample : DX11VSM::Deferred;
            if (!m_Impl->VSM.Prepare(m_Impl->Device.Get(), program, false, m_Impl->LastError) ||
                !m_Impl->VSM.BindReceiver(m_Impl->Context.Get(), desc.LightViewProjection, desc.Lighting, m_Impl->FrameCascades,
                    m_Impl->FrameShadowFilter.VSM, true, m_Impl->LastError)) return false;
            lightingPixel = m_Impl->VSM.Pixel(program);
        }
        if (m_Impl->ContactReceiverActive && !m_Impl->TiledActive)
        {
            const auto filter = desc.SamplesShadowMap && shadowView ? m_Impl->FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard;
            lightingPixel = m_Impl->ContactShadows.Receiver(m_Impl->Device.Get(), desc.Shading, filter, true, multisample, m_Impl->LastError);
            if (!lightingPixel || !m_Impl->ContactShadows.BindReceiver(m_Impl->Device.Get(), m_Impl->Context.Get(), true, m_Impl->LastError)) return false;
        }
        if (m_Impl->TiledActive)
        {
            const auto filter = desc.SamplesShadowMap && shadowView ? m_Impl->FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard;
            if (!m_Impl->TiledDeferred.PrepareDeferred(m_Impl->Device.Get(), filter, multisample, m_Impl->LastError) ||
                !m_Impl->ContactShadows.BindReceiver(m_Impl->Device.Get(), m_Impl->Context.Get(), m_Impl->ContactReceiverActive, m_Impl->LastError)) return false;
            m_Impl->Context->VSSetShader(m_Impl->TiledDeferred.DeferredVertex(), nullptr, 0);
            lightingPixel = m_Impl->TiledDeferred.DeferredPixel(filter, multisample);
        }
        if (m_Impl->UsesDedicatedPCF())
        {
            lightingPixel=m_Impl->PCFPixel(multisample?6:5);
            if (!lightingPixel || !m_Impl->BindPCF(desc.SamplesShadowMap && shadowView)) return false;
        }
        if (m_Impl->FrameRayShadowImage && desc.SamplesShadowMap && !m_Impl->TiledActive)
        {
            lightingPixel=m_Impl->RayShadowPixel(multisample?DX11PCSS::DeferredMultisample:DX11PCSS::Deferred);
            if (!lightingPixel) return false;
        }
        m_Impl->BindRayShadowImage();
        m_Impl->Context->PSSetShader(lightingPixel, nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
		m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
		m_Impl->Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
		const D3D11_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(m_Impl->CreateInfo.Width),
			static_cast<float>(m_Impl->CreateInfo.Height), 0.0f, 1.0f};
		m_Impl->Context->RSSetViewports(1, &viewport);
		m_Impl->Context->Draw(3, 0);
		std::fill(std::begin(inputs), std::end(inputs), nullptr);
		m_Impl->Context->PSSetShaderResources(0, InputCount, inputs);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
        ID3D11ShaderResourceView* noContact = nullptr;
        m_Impl->Context->PSSetShaderResources(15, 1, &noContact);
		m_Impl->BindCascades(false);
        if (m_Impl->SubsurfaceProducing)
        { ++m_Impl->PendingFrame.Statistics.SubsurfacePassCount; return true; }
        m_Impl->SubsurfaceLighting=desc;
        m_Impl->SubsurfaceHasScene=desc.Output==DeferredOutputMode::Scene;
		PassDesc record;
		record.Type = PassType::DeferredLighting;
		record.ClearColorEnabled = record.ClearDepthEnabled = false;
		m_Impl->PendingFrame.Passes.push_back({record, {}});
		++m_Impl->PendingFrame.Statistics.DeferredLightingPassCount;
		m_Impl->DeferredResolved = true;
		m_Impl->SceneColorWritten = true;
		return true;
	}

	bool DX11RHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyAmbientOcclusion")) return false;
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("AO requires resolved multisample depth and material inputs");
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->AmbientOcclusionActive ||
			!m_Impl->GBufferWritten || !m_Impl->SceneColorWritten || m_Impl->AmbientOcclusionApplied ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) || m_Impl->DepthOfFieldResolved || m_Impl->VelocityCaptured || m_Impl->ReflectionApplied)
			return m_Impl->Fail("Ambient occlusion requires material capture and completed opaque lighting, once per frame");
		using Pass = Techniques::AmbientOcclusionPassKind;
		const auto passes = Techniques::BuildAmbientOcclusionPlan(m_Impl->FrameAmbientOcclusion,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
		const bool scalable = m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO;
		RayTracedAOConstants packet {};
        auto& constants = packet.Surface;
        const bool rayTraced = m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::RayTraced;
		if (!BuildAmbientOcclusionFrameConstants(desc, m_Impl->FrameAmbientOcclusion, m_Impl->FrameRayScene.get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Pass::Estimate, packet))
			return m_Impl->Fail("Ambient occlusion requires a finite, invertible camera projection");
		if (scalable) constants.Options.Z = m_Impl->SaoDepthHierarchy.MipCount - 1;
		constexpr UINT InputCount = 4 + static_cast<UINT>(RaySceneBufferCount);
		ID3D11ShaderResourceView* nullViews[InputCount] {};
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->Context->IASetInputLayout(nullptr);
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		const std::size_t technique = m_Impl->AmbientOcclusionProgramIndex();
		m_Impl->Context->VSSetShader(m_Impl->AmbientOcclusionVertices[technique].Get(), nullptr, 0);
		m_Impl->Context->PSSetShader(m_Impl->AmbientOcclusionPixels[technique].Get(), nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
		m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
		const D3D11_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(m_Impl->CreateInfo.Width),
			static_cast<float>(m_Impl->CreateInfo.Height), 0.0f, 1.0f};
		m_Impl->Context->RSSetViewports(1, &viewport);
		ID3D11Buffer* buffer = m_Impl->AmbientOcclusionConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(0, 1, &buffer);
		UINT destinationMip = 0;
		for (Pass pass : passes)
		{
			const bool hierarchy = pass == Pass::DepthPrepare || pass == Pass::DepthReduce;
			if (pass == Pass::DepthReduce) ++destinationMip;
			constants.Options.X = static_cast<std::uint32_t>(pass);
			constants.Options.Y = hierarchy ? destinationMip : 0;
			D3D11_MAPPED_SUBRESOURCE mapped {};
			const HRESULT result = m_Impl->Context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(ambient occlusion constants)", result));
			std::memcpy(mapped.pData, &packet, sizeof(packet));
			m_Impl->Context->Unmap(buffer, 0);
			m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			const bool composite = pass == Pass::Composite;
			const std::size_t targetIndex = pass == Pass::BilateralHorizontal ? 1 : 0;
			const std::size_t sourceIndex = pass == Pass::BilateralVertical ? 1 : 0;
			ID3D11RenderTargetView* target = hierarchy ? m_Impl->SaoDepthHierarchy.ScratchTarget.Get() :
				(composite ? m_Impl->HdrSurfaces[Impl::HdrScene].Target.Get() : m_Impl->AmbientOcclusionSurfaces[targetIndex].Target.Get());
			ID3D11ShaderResourceView* depthInput = scalable && pass != Pass::DepthPrepare
				? m_Impl->SaoDepthHierarchy.View.Get() : m_Impl->DepthShaderView.Get();
			ID3D11ShaderResourceView* inputs[InputCount] {depthInput,
				m_Impl->GBufferSurfaces[1].View.Get(), m_Impl->GBufferSurfaces[3].View.Get(),
				hierarchy || pass == Pass::Estimate ? nullptr : m_Impl->AmbientOcclusionSurfaces[sourceIndex].View.Get()};
            if (rayTraced) for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
                inputs[4+slot]=m_Impl->RaySceneViews[slot].Get();
			const UINT passWidth = hierarchy ? std::max(1, m_Impl->CreateInfo.Width >> destinationMip) : m_Impl->CreateInfo.Width;
			const UINT passHeight = hierarchy ? std::max(1, m_Impl->CreateInfo.Height >> destinationMip) : m_Impl->CreateInfo.Height;
			const D3D11_VIEWPORT passViewport {0.0f, 0.0f, static_cast<float>(passWidth), static_cast<float>(passHeight), 0.0f, 1.0f};
			m_Impl->Context->RSSetViewports(1, &passViewport);
			m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
			m_Impl->Context->PSSetShaderResources(0, InputCount, inputs);
			// Reverse subtraction uses destination scene - lost ambient. Color-only
			// writes preserve opacity, without a scene-color feedback texture.
			m_Impl->Context->OMSetBlendState(composite ? m_Impl->AmbientOcclusionBlend.Get() : nullptr, nullptr, UINT_MAX);
			m_Impl->Context->Draw(3, 0);
			if (hierarchy)
			{
				// Scratch separates rendering from the mip chain's read/copy phases.
				m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
				m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
				const D3D11_BOX rectangle {0, 0, 0, passWidth, passHeight, 1};
				m_Impl->Context->CopySubresourceRegion(m_Impl->SaoDepthHierarchy.Texture.Get(), destinationMip,
					0, 0, 0, m_Impl->SaoDepthHierarchy.Scratch.Get(), 0, &rectangle);
			}
			m_Impl->PendingFrame.AmbientOcclusionPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount;
		}
		m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
		m_Impl->AmbientOcclusionApplied = true;
		return true;
	}

	bool DX11RHI::CompleteEnvironmentCapture()
	{
		constexpr UINT AllFaces = (1u << EnvironmentFaceCount) - 1u;
		if (!m_Impl->CheckThread("CompleteEnvironmentCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || (!m_Impl->FrameEnvironment.Source && m_Impl->EnvironmentFacesWritten != AllFaces))
			return m_Impl->Fail("Environment mip generation requires six completed faces exactly once per frame");
        if (const auto& source=m_Impl->FrameEnvironment.Source)
        {
            m_Impl->Context->OMSetRenderTargets(0,nullptr,nullptr);
            ID3D11ShaderResourceView* empty[11]{};
            m_Impl->Context->PSSetShaderResources(0,11,empty);
            const UINT rowBytes=source->Resolution*4*sizeof(std::uint16_t);
            for (UINT face=0;face<EnvironmentFaceCount;++face)
                m_Impl->Context->UpdateSubresource(m_Impl->Environment.Color.Get(),face*m_Impl->FrameEnvironment.MipCount,
                    nullptr,source->FacesRGBA16[face].data(),rowBytes,0);
        }
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		// Generate only after all base faces are written. No incomplete cube is sampled.
		if (m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude)
		{
			if (!m_Impl->ConvertEnvironmentMap()) return false;
		}
		else m_Impl->Context->GenerateMips(m_Impl->Environment.View.Get());
		m_Impl->EnvironmentReady = true;
		return true;
	}

	bool DX11RHI::CompletePlanarCapture()
	{
		if (!m_Impl->CheckThread("CompletePlanarCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->PlanarActive || !m_Impl->PlanarWritten || m_Impl->PlanarReady)
			return m_Impl->Fail("Planar mip generation requires one completed capture exactly once per frame");
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->Context->IASetInputLayout(nullptr);
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->Context->VSSetShader(m_Impl->ReflectionVertex[Impl::PlanarReflection].Get(), nullptr, 0);
		m_Impl->Context->PSSetShader(m_Impl->PlanarMipPixel.Get(), nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
		m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
		m_Impl->Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
		ID3D11Buffer* buffer = m_Impl->ReflectionConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(0, 1, &buffer);
		ID3D11ShaderResourceView* nullViews[11] {};
		for (UINT mip = 1; mip < m_Impl->FramePlanar.MipCount; ++mip)
		{
			const auto width = std::max(1u, m_Impl->FramePlanar.Width >> mip), height = std::max(1u, m_Impl->FramePlanar.Height >> mip);
			PlanarReflectionConstants constants {};
			constants.Viewport = {float(width), float(height), float(std::max(1u, m_Impl->FramePlanar.Width >> (mip - 1))),
				float(std::max(1u, m_Impl->FramePlanar.Height >> (mip - 1)))};
			D3D11_MAPPED_SUBRESOURCE mapped {};
			const HRESULT result = m_Impl->Context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(planar mip constants)", result));
			std::memcpy(mapped.pData, &constants, sizeof(constants)); m_Impl->Context->Unmap(buffer, 0);
			m_Impl->Context->PSSetShaderResources(0, 11, nullViews);
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			ID3D11RenderTargetView* target = m_Impl->Planar.MipTargets[mip].Get();
			m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
			ID3D11ShaderResourceView* source = m_Impl->Planar.MipViews[mip - 1].Get();
			m_Impl->Context->PSSetShaderResources(10, 1, &source);
			D3D11_VIEWPORT viewport {}; viewport.Width = float(width); viewport.Height = float(height); viewport.MaxDepth = 1;
			m_Impl->Context->RSSetViewports(1, &viewport);
			m_Impl->Context->Draw(3, 0);
			++m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
		}
		m_Impl->Context->PSSetShaderResources(0, 11, nullViews);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->PlanarReady = true;
		return true;
	}

    bool DX11RHI::ApplyPathTracing(const PathTracingDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyPathTracing") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PathFrame.Active || m_Impl->PathFrame.Resolved || !m_Impl->HdrSceneActive || m_Impl->SceneColorWritten)
        return m_Impl->Fail("Path tracing requires an untouched HDR frame with primary-ray visibility");
    return m_Impl->TracePaths(desc);
}

bool DX11RHI::ApplyGlobalIllumination(const GlobalIlluminationDesc& desc)
    {
    if (m_Impl->FrameOpen && m_Impl->EarlyTemporalActive && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyGlobalIllumination") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->GIActive || m_Impl->GI.Resolved || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten ||
            !m_Impl->GBufferWritten || (m_Impl->EarlyTemporalActive && !m_Impl->VelocityCaptured) ||
            (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || (m_Impl->EarlyTemporalActive && !m_Impl->SamplesVelocityResolved))) ||
            (m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) || m_Impl->ReflectionApplied ||
            m_Impl->VolumetricFog.Resolved || m_Impl->LightShafts.Resolved || m_Impl->TemporalAA.Resolved ||
            m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
            return m_Impl->Fail("SSGI requires opaque material, depth and velocity, before reflections and volume effects");
        if (Techniques::UsesLumen(m_Impl->FrameGI)) return m_Impl->ApplyLumenGI(desc);
        if (m_Impl->FrameVoxelGI.Volume) return m_Impl->ApplyVoxelGI(desc);
        if (Techniques::UsesRayTracedGI(m_Impl->FrameGI)) return m_Impl->ApplyRayGI(desc);
        if (m_Impl->LightmapActive)
        {
            auto texture = [&](TextureHandle handle) -> ID3D11ShaderResourceView* {
                const auto found=m_Impl->Textures.find(handle.Value);
                if (!handle || found==m_Impl->Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return nullptr;
                return found->second.Description.ColorSpace==TextureColorSpace::SRGB ?
                    found->second.SRGBShaderResourceView.Get() : found->second.ShaderResourceView.Get();
            };
            if (!m_Impl->Lightmaps.Resolve(m_Impl->Context.Get(),m_Impl->FrameLightmaps,desc,m_Impl->FrameLightmapSettings,
                m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(),m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
                m_Impl->FrameSampleCount>1 ? m_Impl->Multisampling.DepthTarget.Get() : m_Impl->DepthTarget.Get(),
                m_Impl->AmbientOcclusionApplied ? m_Impl->AmbientOcclusionSurfaces[0].View.Get() : nullptr,
                m_Impl->MaterialSampler(),texture,m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount,m_Impl->LastError)) return false;
            if (m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount) RecordLightmapDraws(m_Impl->FrameLightmaps,m_Impl->PendingFrame.Statistics);
            m_Impl->GI.Resolved=true; return true;
        }
        if (m_Impl->ProbeActive)
        {
            std::uint32_t projectionPasses=0;
            if (!m_Impl->Probes.Resolve(m_Impl->Context.Get(),desc,m_Impl->FrameGI,m_Impl->FrameReflectionShading,
                m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(),m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
                m_Impl->DepthShaderView.Get(),m_Impl->GBufferSurfaces[1].View.Get(),m_Impl->GBufferSurfaces[0].View.Get(),
                m_Impl->GBufferSurfaces[3].View.Get(),projectionPasses,m_Impl->LastError)) return false;
            m_Impl->GI.Resolved=true;
            m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount=projectionPasses+1;
            return true;
        }
        if (!m_Impl->GI.Resolve(m_Impl->Context.Get(), desc, m_Impl->FrameGI, m_Impl->FrameReflectionShading,
            m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(), m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
            m_Impl->DepthShaderView.Get(), m_Impl->GBufferSurfaces[1].View.Get(),
            m_Impl->GBufferSurfaces[0].View.Get(), m_Impl->GBufferSurfaces[3].View.Get(),
            m_Impl->MotionSurfaces[Impl::MotionVelocity].View.Get(), m_Impl->LastError)) return false;
        m_Impl->PendingFrame.Temporal = desc.Sampling;
        m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount = 4;
        return true;
    }

	bool DX11RHI::ApplyReflections(const ReflectionDesc& desc)
	{
        if (desc.SkyboxOnly)
        {
            const auto saved=m_Impl->FrameReflections;
            const bool active=m_Impl->ReflectionActive,done=m_Impl->ReflectionApplied,planar=m_Impl->PlanarActive;
            m_Impl->FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
            m_Impl->FrameReflections.EnvironmentIntensity=0;
            m_Impl->FrameReflections.Planar.Enabled=false;
            m_Impl->ReflectionActive=true; m_Impl->ReflectionApplied=false; m_Impl->PlanarActive=false;
            auto request=desc; request.SkyboxOnly=false; request.DrawSkybox=true;
            const bool success=m_Impl->EnsureReflectionResources(false) && ApplyReflections(request);
            m_Impl->FrameReflections=saved; m_Impl->ReflectionActive=active; m_Impl->ReflectionApplied=done; m_Impl->PlanarActive=planar;
            if (success) m_Impl->PendingFrame.Statistics.SkyboxPassCount+=2;
            return success;
        }

    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyReflections")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->ReflectionActive || m_Impl->ReflectionApplied ||
			!m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || !m_Impl->GBufferWritten ||
			(m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady) ||
			(m_Impl->PlanarActive && !m_Impl->PlanarReady) ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved) ||
			(m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("Reflections require resolved opaque materials, lighting and AO, once before blur/transparency");
		if (Techniques::UsesLumenReflections(m_Impl->FrameReflections)) return m_Impl->ApplyLumenReflections(desc);
        if (m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
        return Techniques::UsesHybridRendering(m_Impl->FrameRayRadianceSettings.Pipeline) ? m_Impl->ApplyHybridRendering(desc) : m_Impl->ApplyRayReflections(desc);
		for (const bool planarPass : {false,true})
		{
		    if (planarPass ? !m_Impl->PlanarActive : m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::None) continue;
			ReflectionConstants constants{};
			CubemapReflectionConstants cubeConstants{};
			EnvironmentMapReflectionConstants mapConstants{};
			PlanarReflectionConstants planarConstants{};
			const auto program = m_Impl->FrameReflectionProgram(planarPass);
			const bool cube = program == Impl::CubeReflection, environmentMap = program == Impl::MapReflection;
			const bool planar = program == Impl::PlanarReflection;
			if (program == Impl::ScreenReflection && !BuildReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
				return m_Impl->Fail("SSR requires a finite invertible camera projection");
			if (cube && !BuildCubemapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameEnvironment.MipCount, cubeConstants))
				return m_Impl->Fail("Cubemap reflections require a finite invertible camera projection");
			if (environmentMap && !BuildEnvironmentMapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->EnvironmentMapMips, mapConstants))
				return m_Impl->Fail("Environment map reflections require a finite invertible camera projection");
			if (planar && !BuildPlanarReflectionConstants(desc, m_Impl->FramePlanar, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, planarConstants))
				return m_Impl->Fail("Planar lookup requires the frozen main camera and viewport used for capture");
			constexpr UINT InputCount = 10;
			ID3D11ShaderResourceView* nullViews[InputCount] {};
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
			m_Impl->Context->IASetInputLayout(nullptr);
			m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			m_Impl->Context->VSSetShader(m_Impl->ReflectionVertex[program].Get(), nullptr, 0);
			m_Impl->Context->PSSetShader(m_Impl->ReflectionPixel[program].Get(), nullptr, 0);
			m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
			m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
			ID3D11Buffer* buffer = m_Impl->ReflectionConstantBuffer.Get();
			m_Impl->Context->PSSetConstantBuffers(0, 1, &buffer);
			ID3D11SamplerState* samplers[] = {m_Impl->HdrSampler.Get(), environmentMap ? m_Impl->EnvironmentMapSampler.Get() : nullptr};
			m_Impl->Context->PSSetSamplers(0, 2, samplers);
			D3D11_VIEWPORT viewport {};
			viewport.Width = static_cast<float>(m_Impl->CreateInfo.Width);
			viewport.Height = static_cast<float>(m_Impl->CreateInfo.Height);
			viewport.MaxDepth = 1.0f;
			m_Impl->Context->RSSetViewports(1, &viewport);
			using Pass = Techniques::ReflectionPassKind;
			for (const auto pass : {planar ? Pass::PlanarLookup : Techniques::UsesEnvironmentCapture(m_Impl->FrameReflections) ? Pass::EnvironmentLookup : Pass::Trace, Pass::Composite})
			{
				const bool composite = pass == Pass::Composite;
				constants.Options.X = static_cast<std::uint32_t>(pass);
				cubeConstants.Options.X = static_cast<std::uint32_t>(pass);
				mapConstants.Options.X = static_cast<std::uint32_t>(pass);
				planarConstants.Options.X = static_cast<std::uint32_t>(pass);
				D3D11_MAPPED_SUBRESOURCE mapped {};
				const HRESULT result = m_Impl->Context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
				if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(reflection constants)", result));
				if (planar) std::memcpy(mapped.pData, &planarConstants, sizeof(planarConstants));
				else if (environmentMap) std::memcpy(mapped.pData, &mapConstants, sizeof(mapConstants));
				else if (cube) std::memcpy(mapped.pData, &cubeConstants, sizeof(cubeConstants));
				else std::memcpy(mapped.pData, &constants, sizeof(constants));
				m_Impl->Context->Unmap(buffer, 0);
				m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
				m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
				ID3D11RenderTargetView* target = composite ? m_Impl->HdrSurfaces[Impl::HdrScene].Target.Get() : m_Impl->ReflectionSurface.Target.Get();
				m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
				m_Impl->Context->OMSetBlendState(composite ? m_Impl->ReflectionBlend.Get() : nullptr, nullptr, UINT_MAX);
				ID3D11ShaderResourceView* inputs[InputCount] {};
				if (composite) inputs[6] = m_Impl->ReflectionSurface.View.Get();
				else
				{
					inputs[0] = m_Impl->HdrSurfaces[Impl::HdrScene].View.Get();
					inputs[1] = m_Impl->DepthShaderView.Get();
					inputs[2] = m_Impl->GBufferSurfaces[1].View.Get();
					inputs[3] = m_Impl->GBufferSurfaces[2].View.Get();
					inputs[4] = m_Impl->GBufferSurfaces[0].View.Get();
					inputs[5] = m_Impl->GBufferSurfaces[3].View.Get();
					if (cube) inputs[7] = m_Impl->Environment.View.Get();
					if (environmentMap) inputs[8] = m_Impl->EnvironmentMap.View.Get();
					if (planar) inputs[9] = m_Impl->Planar.Color.View.Get();
				}
				m_Impl->Context->PSSetShaderResources(0, InputCount, inputs);
				m_Impl->Context->Draw(3, 0);
				m_Impl->PendingFrame.ReflectionPasses.push_back(pass);
				++m_Impl->PendingFrame.Statistics.ReflectionPassCount;
			}
			m_Impl->Context->PSSetShaderResources(0, InputCount, nullViews);
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			m_Impl->Context->OMSetBlendState(nullptr, nullptr, UINT_MAX);
		}
		m_Impl->ReflectionApplied = true;
		return true;
	}

	bool DX11RHI::ApplyDepthOfField(const DepthOfFieldDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Light shafts must complete before depth of field");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Fog must complete before depth of field");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalAA.Resolved)
			return m_Impl->Fail("TAA must complete before depth of field");
		if (!m_Impl->CheckThread("ApplyDepthOfField")) return false;
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("Depth of field requires resolved multisample depth and color");
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePostProcessing.DepthOfField.Enabled ||
			!m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || m_Impl->DepthOfFieldResolved ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->MotionBlurResolved))
			return m_Impl->Fail("Depth of field requires completed opaque lighting and ambient occlusion, once before transparency");
		using Pass = Techniques::DepthOfFieldPassKind;
		DepthOfFieldConstants constants {};
		if (!BuildDepthOfFieldConstants(desc, m_Impl->FramePostProcessing.DepthOfField,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Pass::Gather, constants))
			return m_Impl->Fail("Depth of field requires a finite camera with a supported projection");
		const auto passes = Techniques::BuildDepthOfFieldPlan(m_Impl->FramePostProcessing.DepthOfField);
		ID3D11ShaderResourceView* nullViews[4] {};
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
		m_Impl->Context->IASetInputLayout(nullptr);
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->Context->VSSetShader(m_Impl->DepthOfFieldVertex.Get(), nullptr, 0);
		m_Impl->Context->PSSetShader(m_Impl->DepthOfFieldPixel.Get(), nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
		m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
		ID3D11Buffer* buffer = m_Impl->DepthOfFieldConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(0, 1, &buffer);
		D3D11_VIEWPORT viewport {};
		viewport.Width = static_cast<float>(m_Impl->CreateInfo.Width);
		viewport.Height = static_cast<float>(m_Impl->CreateInfo.Height);
		viewport.MaxDepth = 1.0f;
		m_Impl->Context->RSSetViewports(1, &viewport);
		for (const auto pass : passes)
		{
			const bool composite = pass == Pass::Composite;
			constants.Options.X = static_cast<std::uint32_t>(pass);
			D3D11_MAPPED_SUBRESOURCE mapped {};
			const HRESULT result = m_Impl->Context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return m_Impl->Fail(HResultMessage("Map(depth of field constants)", result));
			std::memcpy(mapped.pData, &constants, sizeof(constants));
			m_Impl->Context->Unmap(buffer, 0);
			m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			const auto& source = m_Impl->HdrSurfaces[composite ? Impl::HdrScratch : Impl::HdrScene];
			auto& destination = m_Impl->HdrSurfaces[composite ? Impl::HdrScene : Impl::HdrScratch];
			ID3D11RenderTargetView* target = destination.Target.Get();
			m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
			m_Impl->Context->OMSetBlendState(composite ? m_Impl->DepthOfFieldCompositeWriteMask.Get() : nullptr, nullptr, UINT_MAX);
			ID3D11ShaderResourceView* inputs[2] {source.View.Get(), m_Impl->DepthShaderView.Get()};
			m_Impl->Context->PSSetShaderResources(0, 2, inputs);
			m_Impl->Context->Draw(3, 0);
			m_Impl->PendingFrame.DepthOfFieldPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount;
		}
		m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->DepthOfFieldResolved = true;
		return true;
	}

	bool DX11RHI::ApplyLightShafts(const LightShaftDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyLightShafts") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!Techniques::UsesLightShafts(m_Impl->FrameLightShafts) || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten ||
			m_Impl->LightShafts.Resolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalAA.Resolved ||
			m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
			(Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Resolved) ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
			return m_Impl->Fail("Light shafts require completed opaque lighting, AO, reflections and fog, once before temporal effects and transparency");
		if (!m_Impl->LightShafts.Resolve(m_Impl->Context.Get(), desc, m_Impl->FrameLightShafts,
			m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(), m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
			m_Impl->DepthShaderView.Get(), m_Impl->LastError)) return false;
		m_Impl->PendingFrame.Statistics.LightShaftPassCount += 2;
		return true;
	}

	bool DX11RHI::ApplyVolumetricFog(const VolumetricFogDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyVolumetricFog") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!Techniques::UsesVolumetricFog(m_Impl->FrameFog) || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten ||
			m_Impl->VolumetricFog.Resolved || m_Impl->LightShafts.Resolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalAA.Resolved ||
			m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
			return m_Impl->Fail("Fog requires completed opaque lighting, AO and reflections, once before temporal effects and transparency");
		ID3D11ShaderResourceView* integratedAtlas = nullptr;
		if ((Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)))
		{
			if (m_Impl->LightListsActive && !m_Impl->LightListFrame.ValidateDraw(
                desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
			auto volume = desc;
            if (m_Impl->LightListsActive) volume.Lighting.SceneLights = m_Impl->LightListFrame.Description().Lights;
			std::array<ID3D11ShaderResourceView*, 5> shadows {};
			if (volume.SamplesShadowMap)
			{
				if (m_Impl->FrameVirtualShadowAtlas)
                {
                    shadows[1]=m_Impl->FrameVirtualShadowAtlas->View.Get();
                    shadows[2]=m_Impl->FrameVirtualShadowAtlas->DirectoryView.Get();
                }
                else if (m_Impl->FrameCascades.Frame.CascadeCount)
				{
					for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
						shadows[index + 1] = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value).ShaderResourceView.Get();
				}
				else
				{
					const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
					if (shadow != m_Impl->Textures.end() && shadow->second.Description.Format == TextureFormat::Depth32)
						shadows[0] = shadow->second.ShaderResourceView.Get();
					volume.SamplesShadowMap = shadows[0] != nullptr || bool(m_Impl->FrameRayShadowImage);
				}
			}
            m_Impl->BindRayShadowImage();
            ID3D11PixelShader* pcssInjection = nullptr;
            if (!Techniques::UsesBoundedMedium(m_Impl->FrameFog) && m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS && volume.SamplesShadowMap)
            {
                if (!m_Impl->PCSS.Prepare(m_Impl->Device.Get(), DX11PCSS::Volume, false, m_Impl->LastError) ||
                    !m_Impl->PCSS.Bind(m_Impl->Context.Get(), volume.LightViewProjection, volume.Lighting, m_Impl->FrameCascades,
                        m_Impl->FrameShadowFilter.PCSS, true, m_Impl->LastError)) return false;
                pcssInjection = m_Impl->PCSS.Pixel(DX11PCSS::Volume);
            }
            if (m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM && volume.SamplesShadowMap)
            {
                if (!m_Impl->CheckVSMMomentViews() || !m_Impl->VSM.Prepare(m_Impl->Device.Get(), DX11VSM::Volume, false, m_Impl->LastError) ||
                    !m_Impl->VSM.BindReceiver(m_Impl->Context.Get(), volume.LightViewProjection, volume.Lighting, m_Impl->FrameCascades,
                        m_Impl->FrameShadowFilter.VSM, true, m_Impl->LastError)) return false;
                if (m_Impl->FrameCascades.Frame.CascadeCount)
                    for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
                        shadows[index + 1] = m_Impl->VSM.View(m_Impl->FrameCascades.Maps[index]);
                else shadows[0] = m_Impl->VSM.View(m_Impl->LastShadowTexture);
                pcssInjection = m_Impl->VSM.Pixel(DX11VSM::Volume);
            }
            ID3D11ShaderResourceView* density=nullptr;
            if (volume.Density)
            {
                const auto found=m_Impl->Textures.find(volume.Density->Texture.Value);
                if (found==m_Impl->Textures.end() || !ValidDensityBinding(*volume.Density,found->second.Description))
                    return m_Impl->Fail("DX11 volume density requires a matching linear byte atlas");
                density=found->second.ShaderResourceView.Get();
            }
            std::uint32_t passes = 0;
			if (!m_Impl->VolumetricLighting.Generate(m_Impl->Context.Get(), volume, m_Impl->FrameFog,
				m_Impl->FrameVolumeLighting, m_Impl->FrameCascades, m_Impl->FrameShadowFilter,
                shadows, passes, m_Impl->LastError, Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? nullptr : pcssInjection,
                m_Impl->LightListsActive ? (m_Impl->ClusteredActive ? m_Impl->ClusteredForward.SceneLights() : m_Impl->TileProducer().SceneLights()) : nullptr,
                m_Impl->FrameVirtualShadowAtlas ? &m_Impl->FrameVirtualShadowAtlas->Info : nullptr,
                &m_Impl->FrameVirtualShadowSampling,density)) return false;
			integratedAtlas = m_Impl->VolumetricLighting.View();
			(Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount) += passes;
		}
		if (!m_Impl->VolumetricFog.Resolve(m_Impl->Context.Get(), desc, m_Impl->FrameFog,
			m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(), m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
			m_Impl->DepthShaderView.Get(), m_Impl->LastError, integratedAtlas)) return false;
		(Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricFogPassCount) += integratedAtlas ? 1 : 2;
		return true;
	}

	bool DX11RHI::ApplyTemporalEffects(const TemporalEffectsDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyTemporalEffects") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FramePostProcessing.TemporalEffects.Enabled || !m_Impl->HdrSceneActive || m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Temporal exposure requires completed scene HDR, once before presentation");
    if (!m_Impl->OpaqueEffectsComplete() || !m_Impl->Hair.Complete() || !m_Impl->Hair.VelocityComplete() ||
        !(m_Impl->PathFrame.Active?m_Impl->PathFrame.Resolved:m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->SceneColorWritten) ||
        (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("Complete opaque effects and final multisample color before exposure metering");
    return m_Impl->ResolveExposure(desc);
}

bool DX11RHI::ApplyTemporalAA(const TemporalAADesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Light shafts must complete before TAA");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Fog must complete before TAA");
		if (!m_Impl->CheckThread("ApplyTemporalAA") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) || !m_Impl->VelocityCaptured ||
			m_Impl->TemporalAA.Resolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved))
			return m_Impl->Fail("TAA requires completed opaque color, depth and velocity, once before blur and transparency");
		if (!m_Impl->TemporalAA.Resolve(m_Impl->Context.Get(), desc, m_Impl->FramePostProcessing.TemporalAA,
			m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(), m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),
			m_Impl->DepthShaderView.Get(), m_Impl->MotionSurfaces[Impl::MotionVelocity].View.Get(), m_Impl->LastError)) return false;
		m_Impl->PendingFrame.Temporal = desc.Sampling;
		++m_Impl->PendingFrame.Statistics.TemporalAAPassCount;
		return true;
	}

	bool DX11RHI::ApplyMotionBlur(const MotionBlurDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Light shafts must complete before motion blur");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Fog must complete before motion blur");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalAA.Resolved)
			return m_Impl->Fail("TAA must complete before motion blur");
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved)
			return m_Impl->Fail("Motion blur requires resolved multisample velocity");
		if (!m_Impl->CheckThread("ApplyMotionBlur") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->FramePostProcessing.MotionBlur.Enabled || !m_Impl->HdrSceneActive || !m_Impl->VelocityCaptured ||
			m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("Motion blur requires completed velocity and opaque lighting, once before depth of field/transparency");
		MotionBlurConstants constants {};
		if (!BuildMotionBlurConstants(desc, m_Impl->FramePostProcessing.MotionBlur,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
			return m_Impl->Fail("Motion blur requires a finite camera and valid scene sample interval");
		using Pass = Techniques::MotionBlurPassKind;
		ID3D11ShaderResourceView* nullViews[5] {};
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->Context->PSSetShaderResources(0, 5, nullViews);
		m_Impl->Context->IASetInputLayout(nullptr);
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->Context->VSSetShader(m_Impl->MotionScreenVertex.Get(), nullptr, 0);
		m_Impl->Context->PSSetShader(m_Impl->MotionScreenPixel.Get(), nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->SolidRasterState.Get());
		m_Impl->Context->OMSetDepthStencilState(m_Impl->DepthDisabledState.Get(), 0);
		ID3D11Buffer* buffer = m_Impl->MotionConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(0, 1, &buffer);
		for (const auto pass : Techniques::BuildMotionBlurPlan(m_Impl->FramePostProcessing.MotionBlur))
		{
			constants.Options.X = static_cast<std::uint32_t>(pass);
			if (!m_Impl->UploadMotionConstants(constants)) return false;
			m_Impl->Context->PSSetShaderResources(0, 5, nullViews);
			m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
			const bool composite = pass == Pass::Composite;
			auto& destination = pass == Pass::TileMax ? m_Impl->MotionSurfaces[Impl::MotionTileMax] :
				(pass == Pass::NeighborMax ? m_Impl->MotionSurfaces[Impl::MotionNeighborMax] :
				m_Impl->HdrSurfaces[composite ? Impl::HdrScene : Impl::HdrScratch]);
			ID3D11RenderTargetView* target = destination.Target.Get();
			m_Impl->Context->OMSetRenderTargets(1, &target, nullptr);
			m_Impl->Context->OMSetBlendState(composite ? m_Impl->MotionCompositeMask.Get() : nullptr, nullptr, UINT_MAX);
			ID3D11ShaderResourceView* inputs[5] {
				(pass == Pass::Gather || composite) ? m_Impl->HdrSurfaces[composite ? Impl::HdrScratch : Impl::HdrScene].View.Get() : nullptr,
				pass == Pass::Gather ? m_Impl->DepthShaderView.Get() : nullptr,
				m_Impl->MotionSurfaces[Impl::MotionVelocity].View.Get(),
				pass == Pass::NeighborMax ? m_Impl->MotionSurfaces[Impl::MotionTileMax].View.Get() :
				(pass == Pass::Gather ? m_Impl->MotionSurfaces[Impl::MotionNeighborMax].View.Get() : nullptr), nullptr};
			m_Impl->Context->PSSetShaderResources(0, 5, inputs);
			D3D11_VIEWPORT viewport {};
			viewport.Width = static_cast<float>(destination.Width);
			viewport.Height = static_cast<float>(destination.Height);
			viewport.MaxDepth = 1.0f;
			m_Impl->Context->RSSetViewports(1, &viewport);
			m_Impl->Context->Draw(3, 0);
			m_Impl->PendingFrame.MotionBlurPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.MotionBlurPassCount;
		}
		m_Impl->Context->PSSetShaderResources(0, 5, nullViews);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->MotionBlurResolved = true;
		return true;
	}

    bool DX11RHI::ProcessDepthPeeling(DepthPeelingPass pass)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessDepthPeeling") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->FramePeeling.Enabled || !m_Impl->OpaqueEffectsComplete() || !m_Impl->SceneColorWritten ||
            (m_Impl->DeferredActive && !m_Impl->DeferredResolved)) return m_Impl->Fail("OIT requires completed opaque HDR processing outside a raster pass");
        auto geometry=[&](const DepthPeelingDraw& draw) {
            const auto vertex=m_Impl->Buffers.find(draw.VertexBuffer.Value),index=m_Impl->Buffers.find(draw.IndexBuffer.Value);
            if (vertex==m_Impl->Buffers.end() || index==m_Impl->Buffers.end()) return m_Impl->Fail("Invalid DX11 OIT buffer handle");
            if (!ValidateDepthPeelingBuffers(draw,vertex->second.Description,index->second.Description,vertex->second.Bytes.size(),index->second.Bytes,m_Impl->LastError,vertex->second.NaniteGeometry && index->second.NaniteGeometry)) return false;
            auto* buffer=vertex->second.Buffer.Get(); const UINT stride=sizeof(Vertex),offset=0;
            m_Impl->Context->IASetVertexBuffers(0,1,&buffer,&stride,&offset);
            m_Impl->Context->IASetIndexBuffer(index->second.Buffer.Get(),DXGI_FORMAT_R32_UINT,0); return true;
        };
        auto texture=[&](TextureHandle handle)->ID3D11ShaderResourceView* {
            const auto found=m_Impl->Textures.find(handle.Value);
            if (!handle || found==m_Impl->Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return nullptr;
            return found->second.Description.ColorSpace==TextureColorSpace::SRGB ? found->second.SRGBShaderResourceView.Get() : found->second.ShaderResourceView.Get();
        };
        if (!m_Impl->Peeling.Process(m_Impl->Context.Get(),pass,m_Impl->FramePeeling,m_Impl->MaterialSampler(),
            m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(),m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),m_Impl->DepthShaderView.Get(),
            geometry,texture,m_Impl->LastError)) return false;
        RecordDepthPeelingPass(m_Impl->FramePeeling,pass,m_Impl->PendingFrame.Statistics); return true;
    }

	bool DX11RHI::ProcessRefraction(const RefractionDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessRefraction") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->RefractionActive || m_Impl->RefractionResolved || m_Impl->FrameSampleCount!=1 ||
            !m_Impl->HdrSceneActive || !m_Impl->OpaqueEffectsComplete() || (!m_Impl->SceneColorWritten || (m_Impl->DeferredActive && !m_Impl->DeferredResolved)))
            return m_Impl->Fail("Refraction requires completed single-sample opaque HDR processing and executes once per frame");
        RefractionConstants constants;
        if (!BuildRefractionConstants(desc.ViewProjection,desc.Environment,m_Impl->FrameRefractionSettings,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,constants)) return m_Impl->Fail("DX11 refraction projection is invalid");
        auto texture=[&](TextureHandle handle)->ID3D11ShaderResourceView* {
            const auto found=m_Impl->Textures.find(handle.Value);
            if (!handle || found==m_Impl->Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return nullptr;
            return found->second.Description.ColorSpace==TextureColorSpace::SRGB ? found->second.SRGBShaderResourceView.Get() : found->second.ShaderResourceView.Get();
        };
        const bool volumeActive=m_Impl->FrameGI.Lumen.Translucency.Enabled && m_Impl->LumenVolumeFrame.Grid.W!=0;
        auto* volumeUniform=volumeActive?m_Impl->LumenVolumeUniform.Get():nullptr;
        auto* volumeImage=volumeActive?m_Impl->LumenVolumeAtlas.View.Get():nullptr;
        auto* volumeSampler=m_Impl->SurfaceSampler.Get();
        m_Impl->Context->PSSetConstantBuffers(Impl::LumenVolumeConstantSlot,1,&volumeUniform);
        m_Impl->Context->PSSetShaderResources(Impl::LumenVolumeTextureSlot,1,&volumeImage);
        m_Impl->Context->PSSetSamplers(Impl::LumenVolumeSamplerSlot,1,&volumeSampler);
        if (!m_Impl->Refraction.Resolve(m_Impl->Context.Get(),m_Impl->FrameRefraction,constants,
            m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get(),m_Impl->HdrSurfaces[Impl::HdrScene].View.Get(),m_Impl->DepthShaderView.Get(),texture,m_Impl->LastError)) return false;
        m_Impl->RefractionResolved=true;
        if (m_Impl->FrameRefraction.Geometry && !m_Impl->FrameRefraction.Draws.empty()) RecordRefractionDraws(m_Impl->FrameRefraction,m_Impl->PendingFrame.Statistics);
        return true;
    }

    bool DX11RHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		using Pass = Techniques::WeightedTransparencyPassKind;
		if (!m_Impl->CheckThread("ProcessWeightedTransparency") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->WeightedActive || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || !m_Impl->OpaqueEffectsComplete() ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved ||
				(m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored))))
			return m_Impl->Fail("Weighted transparency requires completed opaque processing and restored sample color");
		auto& weighted = m_Impl->WeightedTransparency;
		if (pass == Pass::Prepare)
		{
			if (weighted.Prepared || m_Impl->SamplesTransparentWritten)
				return m_Impl->Fail("Prepare weighted transparency once before transparent geometry");
			if (!weighted.Prepare(m_Impl->Context.Get(), m_Impl->FrameTransparency, m_Impl->LastError)) return false;
		}
		else if (pass == Pass::Resolve)
		{
			if (!weighted.Prepared || !weighted.Accumulated || weighted.Resolved)
				return m_Impl->Fail("Resolve weighted transparency once after its geometry pass");
			weighted.Resolve(m_Impl->Context.Get(), m_Impl->FrameSampleCount > 1 ? m_Impl->Multisampling.Color.Target.Get() :
				m_Impl->HdrSurfaces[Impl::HdrScene].Target.Get());
		}
		else return m_Impl->Fail("Unknown weighted transparency stage");
		++m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount;
		return true;
	}

	bool DX11RHI::ProcessMultisampling(Techniques::MultisamplePassKind pass)
	{
    if (pass==Techniques::MultisamplePassKind::OpaqueResolve && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before opaque MSAA resolution");
		using Pass = Techniques::MultisamplePassKind;
		if (!m_Impl->CheckThread("ProcessMultisampling") || !m_Impl->FrameOpen || m_Impl->PassOpen || m_Impl->FrameSampleCount <= 1)
			return m_Impl->Fail("MSAA processing requires a multisample frame outside a raster pass");
		switch (pass)
		{
		case Pass::OpaqueResolve:
			if (!m_Impl->SceneColorWritten || m_Impl->SamplesOpaqueResolved ||
				(m_Impl->MaterialCaptureActive && !m_Impl->GBufferWritten) ||
				(m_Impl->DeferredActive && !m_Impl->DeferredResolved))
				return m_Impl->Fail("Resolve MSAA opaque inputs once after opaque lighting");
			break;
		case Pass::VelocityResolve:
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->VelocityCaptured || m_Impl->SamplesVelocityResolved || m_Impl->MotionBlurResolved || m_Impl->TemporalAA.Resolved)
				return m_Impl->Fail("Resolve MSAA velocity once after velocity capture and before blur");
			break;
		case Pass::ColorRestore:
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->NeedsSampleColorRestore() || !m_Impl->OpaqueEffectsComplete() ||
				m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)
				return m_Impl->Fail("Restore processed opaque color once before MSAA transparency");
			break;
		case Pass::ColorResolve:
			if (m_Impl->WeightedActive && !m_Impl->WeightedTransparency.Resolved)
				return m_Impl->Fail("Resolve weighted transparency before the final MSAA color resolve");
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesTransparentWritten || m_Impl->SamplesColorResolved)
				return m_Impl->Fail("Resolve final MSAA color once after transparent composition");
			break;
		default: return m_Impl->Fail("Unknown MSAA pass");
		}
		DX11Multisampling::ResolveTargets targets;
		targets.Color = m_Impl->HdrSceneActive ? m_Impl->HdrSurfaces[Impl::HdrScene].Texture.Get() : m_Impl->ColorTexture.Get();
		// Opaque screen effects require HDR, so ColorRestore always samples the
		// HDR scene. Without these effects, preserve original sample colors.
		targets.ColorView = m_Impl->HdrSurfaces[Impl::HdrScene].View.Get();
		targets.Depth = m_Impl->DepthTarget.Get();
		targets.Velocity = m_Impl->MotionSurfaces[Impl::MotionVelocity].Target.Get();
		for (std::size_t i = 0; i < targets.Materials.size(); ++i) targets.Materials[i] = m_Impl->GBufferSurfaces[i].Target.Get();
		m_Impl->Multisampling.Process(m_Impl->Context.Get(), pass, targets);
		if (pass == Pass::OpaqueResolve) m_Impl->SamplesOpaqueResolved = m_Impl->SamplesColorResolved = true;
		if (pass == Pass::VelocityResolve) m_Impl->SamplesVelocityResolved = true;
		if (pass == Pass::ColorRestore) m_Impl->SamplesColorRestored = true;
		if (pass == Pass::ColorResolve) m_Impl->SamplesColorResolved = true;
		++m_Impl->PendingFrame.Statistics.MultisamplePassCount;
		return true;
	}
	bool DX11RHI::Present()
	{
    if (!m_Impl->CheckThread("Lumen frame publication")) return false;
    Impl::LumenPresentPublication lumenPublication{*m_Impl};
    Impl::NanitePresentPublication nanitePublication{*m_Impl};
    if (!m_Impl->CheckThread("TSR present boundary")) return false;
    if (m_Impl->FrameOpen && m_Impl->TSRScene.Active && (!m_Impl->TSRScene.OpaqueReady || !m_Impl->VelocityCaptured || (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesVelocityResolved))) return m_Impl->Fail("TSR presentation requires completed opaque snapshot and physical motion");

    if (m_Impl->FrameOpen && m_Impl->FramePostProcessing.TemporalEffects.Enabled && !m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Complete temporal exposure before presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->FrameOpen && m_Impl->PendingFrame.Statistics.SubsurfacePassCount && !m_Impl->SubsurfaceResolved)
        return m_Impl->Fail("Incomplete subsurface diffusion must be aborted before presentation");
        if (m_Impl->PathFrame.Active && !m_Impl->PathFrame.Resolved) return m_Impl->Fail("Path tracing must resolve before presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX11 frame cannot be presented in the current state");
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("Forward+ frame requires completed light assignment before presentation");
		if (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesColorResolved))
			return m_Impl->Fail("Resolve MSAA opaque inputs and final color before presentation");
		if (m_Impl->DeferredActive && !m_Impl->DeferredResolved)
			return m_Impl->Fail("Deferred frames must resolve lighting before presentation");
		if (m_Impl->ReflectionActive && !m_Impl->ReflectionApplied)
			return m_Impl->Fail("SSR must complete before presentation");
		if (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied)
			return m_Impl->Fail("Ambient-occlusion frames must apply ambient occlusion before presentation");
		if (m_Impl->FramePostProcessing.DepthOfField.Enabled && !m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("Depth-of-field frames must resolve depth of field before presentation");
		if (m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->MotionBlurResolved)
			return m_Impl->Fail("Motion-blur frames must resolve velocity reconstruction before presentation");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalAA.Resolved)
			return m_Impl->Fail("TAA must resolve before presentation");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->VolumetricFog.Resolved)
			return m_Impl->Fail("Fog must resolve before presentation");
        if (m_Impl->FramePeeling.Enabled && m_Impl->Peeling.Sequence.Stage()!=DepthPeelingStage::Complete)
            return m_Impl->Fail("Present requires completed OIT composition");
		if (m_Impl->RefractionActive && !m_Impl->RefractionResolved)
            return m_Impl->Fail("Present requires completed refraction composition");
        if (m_Impl->WeightedActive && !m_Impl->WeightedTransparency.Resolved)
			return m_Impl->Fail("Weighted transparency must resolve before presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShafts.Resolved)
			return m_Impl->Fail("Light shafts must resolve before presentation");
		ID3D11Texture2D* finalColor = nullptr;
		if (!m_Impl->TSRScene.Active && (!m_Impl->ApplyHdr(m_Impl->FramePostProcessing.HDR) || !m_Impl->ApplyPostProcessing(m_Impl->FramePostProcessing, finalColor))) return false;
		m_Impl->PendingFrame.Statistics.PassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size())
			+ m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount
            + m_Impl->PendingFrame.Statistics.ContactShadowPassCount
            + m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount
            + m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount
            + m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount
            + m_Impl->PendingFrame.Statistics.PostProcessPassCount + m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount
			+ m_Impl->PendingFrame.Statistics.HairFurPassCount + m_Impl->PendingFrame.Statistics.SubsurfacePassCount + m_Impl->PendingFrame.Statistics.PathTracingPassCount + m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount + m_Impl->PendingFrame.Statistics.ReflectionPassCount + m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount + m_Impl->PendingFrame.Statistics.MotionBlurPassCount
			+ m_Impl->PendingFrame.Statistics.MultisamplePassCount + m_Impl->PendingFrame.Statistics.EnvironmentConversionPassCount
			+ m_Impl->PendingFrame.Statistics.PlanarMipPassCount + m_Impl->PendingFrame.Statistics.TemporalAAPassCount + m_Impl->PendingFrame.Statistics.TemporalEffectsPassCount
			+ m_Impl->PendingFrame.Statistics.VolumetricFogPassCount + m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount + m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount + m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount
			+ m_Impl->PendingFrame.Statistics.LightShaftPassCount
            + m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount + m_Impl->PendingFrame.Statistics.DepthPeelingPassCount;
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) ||
			(!IsGpuPresentationEnabled() && m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0);
        const bool gpuPresentation = IsGpuPresentationEnabled() && !captureDepth && !m_Impl->TSRScene.Active &&
            !m_Impl->HasOutlineMaskThisFrame && GetDiagnosticBuffer() == Techniques::RenderBuffer::Color;
        if (gpuPresentation)
        {
            if (!m_Impl->PublishGpuColor(finalColor)) return false;
        }
        else if (!m_Impl->TSRScene.Active && !m_Impl->ReadColor(finalColor, m_Impl->PendingFrame.ColorRGBA8)) return false;
        const int diagnosticIndex=Techniques::MaterialBufferIndex(GetDiagnosticBuffer());
        ID3D11Texture2D* diagnostic=diagnosticIndex>=0?m_Impl->GBufferSurfaces[diagnosticIndex].Texture.Get():GetDiagnosticBuffer()==Techniques::RenderBuffer::AmbientOcclusion?m_Impl->AmbientOcclusionSurfaces[0].Texture.Get():nullptr;
        if (diagnostic)
        {
            D3D11_TEXTURE2D_DESC desc{};diagnostic->GetDesc(&desc);desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
            ComPtr<ID3D11Texture2D> staging;
            if (FAILED(m_Impl->Device->CreateTexture2D(&desc,nullptr,staging.GetAddressOf()))) return m_Impl->Fail("Cannot allocate buffer inspection readback");
            m_Impl->Context->CopyResource(staging.Get(),diagnostic);D3D11_MAPPED_SUBRESOURCE mapped{};
            if (FAILED(m_Impl->Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped))) return m_Impl->Fail("Cannot map buffer inspection readback");
            DecodeDiagnosticRows(mapped.pData,mapped.RowPitch,desc.Width,desc.Height,m_Impl->PendingFrame.DiagnosticPixels);
            m_Impl->Context->Unmap(staging.Get(),0);
        }
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && ((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive) &&
			!m_Impl->ReadMotionVectors()) return false;
		if (captureDepth &&
			!m_Impl->ReadDepth(m_Impl->DepthTexture.Get(), m_Impl->DepthStaging.Get(),
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				m_Impl->PendingFrame.Depth32)) return false;
		if (m_Impl->HasOutlineMaskThisFrame && !m_Impl->TSRScene.Active)
		{
			std::vector<std::uint16_t> outlineMask;
			if (!m_Impl->ReadOutlineMask(outlineMask)) return false;
			m_Impl->ApplyOutline(outlineMask, m_Impl->PendingFrame.ColorRGBA8);
		}
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
		{
			m_Impl->PendingFrame.CascadeDepth32.resize(m_Impl->FrameCascades.Frame.CascadeCount);
			for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
			{
				const auto& map = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value);
				if (!m_Impl->ReadDepth(map.Texture.Get(), map.StagingTexture.Get(),
					m_Impl->FrameCascades.MapResolution, m_Impl->FrameCascades.MapResolution,
					m_Impl->PendingFrame.CascadeDepth32[index])) return false;
			}
		}
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end())
			{
				m_Impl->PendingFrame.ShadowWidth = static_cast<std::uint32_t>(shadow->second.Description.Width);
				m_Impl->PendingFrame.ShadowHeight = static_cast<std::uint32_t>(shadow->second.Description.Height);
				if (!m_Impl->ReadDepth(shadow->second.Texture.Get(), shadow->second.StagingTexture.Get(),
					shadow->second.Description.Width, shadow->second.Description.Height,
					m_Impl->PendingFrame.ShadowDepth32)) return false;
			}
		}
        if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("Contact frame has no completed visibility phase");
		if (!m_Impl->FinishTSRScene()) return false;
    m_Impl->CommitTSRScene();
    m_Impl->Hair.CommitPresent(m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height);
    m_Impl->ExposureHistory.Commit();
    ApplyDiagnosticCapture(m_Impl->PendingFrame,GetDiagnosticBuffer());
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameOpen = false;
		m_Impl->PresentToWindow(m_Impl->LastFrame);
		m_Impl->PathFrame.Commit();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI)) m_Impl->GI.Commit();
        else m_Impl->GI.Invalidate();
        if (m_Impl->ProbeActive) m_Impl->Probes.Commit();
        if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing)) m_Impl->TemporalAA.Commit();
		else m_Impl->TemporalAA.Invalidate();
		lumenPublication.Published=true; nanitePublication.Published=true;
    return true;
	}

	FrameCapture DX11RHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
	FrameCapture DX11RHI::TakeLastFrameCapture() { return std::move(m_Impl->LastFrame); }
	std::string DX11RHI::GetLastError() const { return m_Impl->LastError; }
#else
	struct DX11RHI::Impl
	{
		std::string LastError = "DirectX 11 is only available on Windows";
	};

	DX11RHI::DX11RHI() : m_Impl(std::make_unique<Impl>()) {}
	DX11RHI::~DX11RHI() = default;
	bool DX11RHI::Initialize(const RHICreateInfo&) { return false; }
	void DX11RHI::Shutdown() {}
	bool DX11RHI::IsInitialized() const { return false; }
	RHIBackend DX11RHI::GetBackend() const { return RHIBackend::DX11; }
	const char* DX11RHI::GetName() const { return "DirectX 11 Unavailable"; }
	RHICapabilities DX11RHI::GetCapabilities() const { return {}; }
	bool DX11RHI::Resize(int, int) { return false; }
	BufferHandle DX11RHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
    bool DX11RHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
bool DX11RHI::PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
bool DX11RHI::SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
bool DX11RHI::PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
bool DX11RHI::CancelNaniteCache(NaniteCacheTask) { return false; }
    bool DX11RHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
bool DX11RHI::DestroyNaniteCache(NaniteCacheHandle) { return false; }
    bool DX11RHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
    bool DX11RHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
    bool DX11RHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
    bool DX11RHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
    bool DX11RHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
    bool DX11RHI::CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
    bool DX11RHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle) { return false; }
    bool DX11RHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
    bool DX11RHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
    bool DX11RHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
    bool DX11RHI::TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
    bool DX11RHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }
    bool DX11RHI::RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
    bool DX11RHI::BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
    bool DX11RHI::AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
    bool DX11RHI::CancelLumenLightingUpdate(LumenLightingTask) { return false; }
    bool DX11RHI::ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
    bool DX11RHI::DestroyLumenLighting(LumenLightingHandle) { return false; }
    bool DX11RHI::TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
    bool DX11RHI::ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
    bool DX11RHI::AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
    bool DX11RHI::ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
    bool DX11RHI::DestroyLumenRadiance(LumenRadianceHandle) { return false; }
    bool DX11RHI::IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
    bool DX11RHI::ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
    bool DX11RHI::ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
    bool DX11RHI::DestroyLumenProbes(LumenProbeHandle) { return false; }
    bool DX11RHI::LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
    bool DX11RHI::CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
    bool DX11RHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
    bool DX11RHI::AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
    bool DX11RHI::CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
    bool DX11RHI::ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
    bool DX11RHI::DestroyLumenSurface(LumenSurfaceHandle) { return false; }
    bool DX11RHI::DestroyTSRHistory(TSRHistoryHandle) { return false; }
    bool DX11RHI::RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
    bool DX11RHI::SnapshotTSROpaque() { return false; }
    bool DX11RHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool DX11RHI::CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
bool DX11RHI::SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
bool DX11RHI::PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
bool DX11RHI::CancelNaniteGeometry(NaniteGeometryTask) { return false; }
    bool DX11RHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool DX11RHI::PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool DX11RHI::PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool DX11RHI::DrawNaniteBins(NaniteDrawHandle) { return false; }
bool DX11RHI::DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
bool DX11RHI::CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
bool DX11RHI::BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
bool DX11RHI::RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
bool DX11RHI::ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
bool DX11RHI::ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
bool DX11RHI::DestroyNaniteDepth(NaniteDepthHandle) { return false; }
bool DX11RHI::DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }
bool DX11RHI::CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
    bool DX11RHI::CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
    bool DX11RHI::CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
	bool DX11RHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
	bool DX11RHI::DestroyBuffer(BufferHandle) { return false; }
	TextureHandle DX11RHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
	bool DX11RHI::DestroyTexture(TextureHandle) { return false; }
	bool DX11RHI::BeginFrame() { return false; }
	void DX11RHI::AbortFrame() {}
	bool DX11RHI::BeginPass(const PassDesc&) { return false; }
	bool DX11RHI::BindPipeline(const PipelineState&) { return false; }
	bool DX11RHI::BindWorldTransform(const Matrix4&) { return false; }
	bool DX11RHI::BindViewProjection(const Matrix4&) { return false; }
	bool DX11RHI::BindLightViewProjection(const Matrix4&) { return false; }
	bool DX11RHI::BindVertexBuffer(BufferHandle) { return false; }
	bool DX11RHI::BindIndexBuffer(BufferHandle) { return false; }
	bool DX11RHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
    bool DX11RHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
    bool DX11RHI::RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
    bool DX11RHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
	bool DX11RHI::DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>&) { return false; }
	bool DX11RHI::EndPass() { return false; }
	bool DX11RHI::DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
    bool DX11RHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
    bool DX11RHI::DrawHairFur(const HairFurDesc&) { return false; }
    bool DX11RHI::DrawHairFurInPass(const HairFurDesc&) { return false; }
    bool DX11RHI::DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
    bool DX11RHI::GenerateContactShadows(const ContactShadowDesc&) { return false; }
    bool DX11RHI::PrepareLightListFrame(const LightListFrameDesc&) { return false; }
    bool DX11RHI::GenerateLightLists() { return false; }
	bool DX11RHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
	bool DX11RHI::CompleteEnvironmentCapture() { return false; }
	bool DX11RHI::CompletePlanarCapture() { return false; }
	bool DX11RHI::ApplyPathTracing(const PathTracingDesc&) { return false; }
    bool DX11RHI::ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
    bool DX11RHI::ApplyReflections(const ReflectionDesc&) { return false; }
	bool DX11RHI::ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
	bool DX11RHI::ApplyMotionBlur(const MotionBlurDesc&) { return false; }
	bool DX11RHI::ApplyTemporalAA(const TemporalAADesc&) { return false; }
    bool DX11RHI::ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
	bool DX11RHI::ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
	bool DX11RHI::ApplyLightShafts(const LightShaftDesc&) { return false; }
	bool DX11RHI::ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
	bool DX11RHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
    bool DX11RHI::ProcessDepthPeeling(DepthPeelingPass) { return false; }
    bool DX11RHI::ProcessRefraction(const RefractionDesc&) { return false; }
	bool DX11RHI::Present() { return false; }
	FrameCapture DX11RHI::GetLastFrameCapture() const { return {}; }
	FrameCapture DX11RHI::TakeLastFrameCapture() { return {}; }
	std::string DX11RHI::GetLastError() const { return m_Impl->LastError; }
#endif
}
