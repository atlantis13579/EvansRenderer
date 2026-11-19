#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/DX12/DXILShaderCache.h"
#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/D3DShaderCache.h"
#include "Renderer/Lumen/LumenHairReceiver.h"
#include "Renderer/Lumen/LumenHardwareRayTracing.h"
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
#include "Renderer/MeshShaders/MeshShaders.h"
#include "Renderer/DX12/DX12RHI.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/Geometry/Instancing.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/TechniqueShader.h"
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
#include "Renderer/PostProcessing/TemporalAAContract.h"
#include "Renderer/Lighting/GlobalIlluminationContract.h"
#include "Renderer/BakedLighting/LightProbeContract.h"
#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Volumetrics/VolumetricLightingContract.h"
#include "Renderer/Volumetrics/LightShaftContract.h"
#include "Renderer/Transparency/WeightedTransparencyContract.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/DX12/DX12ContactShadows.h"
#include "Renderer/DX12/DX12ForwardPlus.h"
#include "Renderer/DX12/DX12ClusteredForward.h"
#include "Renderer/Lighting/LightListFrame.h"
#include "Renderer/Pipelines/TiledDeferredContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace Evans::Renderer
{
#if defined(_WIN32)
	using Microsoft::WRL::ComPtr;

	namespace
	{
		constexpr std::size_t DefaultConstantBufferCapacity = 4u * 1024u * 1024u;
		constexpr std::size_t InstanceUploadPageCapacity = 64u * 1024u;
		constexpr UINT DescriptorCapacity = 256;
        // Terrain materials have independent diffuse/normal tiles; keep RTV/DSV
        // pools small while allowing resident material and compute SRVs to scale.
        constexpr UINT ShaderDescriptorCapacity = 8192;
		constexpr UINT MaterialTextureCount = 4;
		constexpr UINT CascadeFirstTextureRegister = 10;
		constexpr UINT ForwardCascadeFirstRoot = MaterialTextureCount + 1;
		constexpr UINT ForwardCascadeConstantsRoot = ForwardCascadeFirstRoot + MaximumShadowCascades;
		constexpr UINT ForwardInstanceRoot = ForwardCascadeConstantsRoot + 1;
		constexpr UINT ForwardFogConstantsRoot = ForwardInstanceRoot + 1;
		constexpr UINT ForwardFogTextureRoot = ForwardFogConstantsRoot + 1;
		constexpr UINT ForwardTransparencyRoot = ForwardFogTextureRoot + 1;
		constexpr UINT ForwardShadowFilterRoot = ForwardTransparencyRoot + 1;
        constexpr UINT ForwardContactControlRoot = ForwardShadowFilterRoot + 1, ForwardContactImageRoot = ForwardContactControlRoot + 1;
        constexpr UINT ContactControlRegister = 7, ContactImageRegister = 15;
        constexpr UINT ForwardPlusControlRoot = ForwardContactImageRoot + 1;
        constexpr UINT ForwardPlusStorageFirstRoot = ForwardPlusControlRoot + 1;
        constexpr UINT ForwardPlusStorageCount = 3, ForwardPlusFirstRegister = 16, ForwardPlusControlRegister = 8;
        constexpr UINT ForwardLumenVolumeRoot=ForwardPlusStorageFirstRoot+ForwardPlusStorageCount;
        constexpr UINT ForwardLumenVolumeImageRoot=ForwardLumenVolumeRoot+1;
        constexpr UINT ForwardRayShadowImageRoot=ForwardLumenVolumeImageRoot+1;
        constexpr UINT ForwardMaterialSamplerRoot = ForwardRayShadowImageRoot + 1;
        constexpr UINT RayShadowImageRegister=20;
        constexpr UINT LumenVolumeConstantRegister=9,LumenVolumeImageRegister=19,LumenVolumeSamplerRegister=2;
        constexpr std::size_t ForwardPlusFilterCount = static_cast<std::size_t>(Techniques::ShadowFilterMode::VSM) + 1;
        enum ForwardPlusProgram : std::size_t { ForwardPlusSurface, ForwardPlusContact, ForwardPlusWeighted, ForwardPlusDepth, ForwardPlusProgramCount };
		constexpr UINT ShadowFilterConstantRegister = 6, VolumeShadowFilterRoot = 8;
        constexpr UINT VolumeSceneLightRoot = VolumeShadowFilterRoot + 1;
        constexpr UINT VolumeRayShadowRoot=VolumeSceneLightRoot+1;
        constexpr UINT VolumeDensityRoot=VolumeRayShadowRoot+1,VolumeDensityRegister=6;
        constexpr UINT VolumeSceneLightRegister = 16;
		constexpr std::size_t PCSSDeferredProgram = 5, PCSSMultisampleProgram = 6, PCSSVolumeProgram = 7;
		constexpr std::size_t PCSSProgramCount = 8;
        constexpr std::size_t VSMCasterProgram = 8, VSMFilterProgram = 9, VSMProgramCount = 10;
		constexpr std::size_t VSMDeferredProgram = 5, VSMDeferredMultisampleProgram = 6, VSMVolumeProgram = 7;
        constexpr UINT VSMFilterConstantRoot = 0, VSMFilterImageRoot = 1;
        constexpr DXGI_FORMAT VSMFormat = DXGI_FORMAT_R32G32_FLOAT;
		constexpr UINT TransparencyConstantRegister = 5;
		constexpr std::array<DXGI_FORMAT, 2> TransparencyFormats {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_FLOAT};
		constexpr UINT FogConstantRegister = 4, FogTextureRegister = 14, FogSamplerRegister = 1;
		constexpr UINT MotionInstanceRoot = 6; // Five texture tables follow the motion CBV.
		constexpr UINT InstanceConstantRegister = 3;
		// DX12's CSM pixel input uses TEXCOORD1 for alpha; the single-map input
		// uses COLOR0. They need separate vertex output signatures within this API.
		constexpr std::size_t CascadeInstanceProgram = InstanceVertexProgramCount;
		constexpr std::size_t DX12InstanceProgramCount = InstanceVertexProgramCount + 1;
		constexpr UINT DeferredBaseTextureCount = static_cast<UINT>(GBufferAttachmentCount) + 2;
		constexpr UINT DeferredCascadeFirstRoot = DeferredBaseTextureCount + 1;
		constexpr UINT DeferredCascadeConstantsRoot = DeferredCascadeFirstRoot + MaximumShadowCascades;
		constexpr UINT DeferredShadowFilterRoot = DeferredCascadeConstantsRoot + 1;
        constexpr UINT DeferredContactControlRoot = DeferredShadowFilterRoot + 1, DeferredContactImageRoot = DeferredContactControlRoot + 1;
        constexpr UINT DeferredSurfaceRoot = DeferredContactImageRoot + 1, DeferredListRoot = DeferredSurfaceRoot + 1;
        constexpr UINT DeferredStorageFirstRoot = DeferredListRoot + 1;
        constexpr UINT DeferredRayShadowImageRoot=DeferredStorageFirstRoot+ForwardPlusStorageCount;
		constexpr UINT InvalidDescriptor = std::numeric_limits<UINT>::max();
		constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;
		constexpr float MaximumHalfFloatRadiance = 65504.0f;

		// Shader algorithms are independently maintained in Src/Shaders/DX12.

		using D3D12CreateDeviceFunction = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
		using D3D12SerializeRootSignatureFunction = HRESULT(WINAPI*)(
			const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

		std::size_t AlignUp(std::size_t value, std::size_t alignment)
		{
			return (value + alignment - 1) & ~(alignment - 1);
		}

		std::string HResultMessage(const char* operation, HRESULT result)
		{
			std::ostringstream stream;
			stream << operation << " failed with HRESULT 0x"
				<< std::hex << std::uppercase << static_cast<unsigned long>(result);
			return stream.str();
		}

		D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
		{
			D3D12_HEAP_PROPERTIES properties = {};
			properties.Type = type;
			properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
			properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			properties.CreationNodeMask = 1;
			properties.VisibleNodeMask = 1;
			return properties;
		}

		D3D12_RESOURCE_DESC BufferResourceDesc(std::size_t size)
		{
			D3D12_RESOURCE_DESC desc = {};
			desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			desc.Width = static_cast<UINT64>(size);
			desc.Height = 1;
			desc.DepthOrArraySize = 1;
			desc.MipLevels = 1;
			desc.Format = DXGI_FORMAT_UNKNOWN;
			desc.SampleDesc.Count = 1;
			desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			return desc;
		}
	}

	struct DX12RHI::Impl
	{
		struct DescriptorArena
		{
			ComPtr<ID3D12DescriptorHeap> Heap;
			D3D12_DESCRIPTOR_HEAP_TYPE Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			UINT Capacity = 0;
			UINT Increment = 0;
			UINT Next = 0;
			bool ShaderVisible = false;
			std::vector<UINT> FreeList;

			UINT Allocate()
			{
				if (!FreeList.empty())
				{
					const UINT value = FreeList.back();
					FreeList.pop_back();
					return value;
				}
				return Next < Capacity ? Next++ : InvalidDescriptor;
			}

			void Release(UINT index)
			{
				if (index != InvalidDescriptor && index < Next) FreeList.push_back(index);
			}

			D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT index) const
			{
				D3D12_CPU_DESCRIPTOR_HANDLE handle = Heap->GetCPUDescriptorHandleForHeapStart();
				handle.ptr += static_cast<SIZE_T>(index) * Increment;
				return handle;
			}

			D3D12_GPU_DESCRIPTOR_HANDLE Gpu(UINT index) const
			{
				D3D12_GPU_DESCRIPTOR_HANDLE handle = Heap->GetGPUDescriptorHandleForHeapStart();
				handle.ptr += static_cast<UINT64>(index) * Increment;
				return handle;
			}
		};

		struct ReadbackResource
		{
			ComPtr<ID3D12Resource> Buffer;
			D3D12_PLACED_SUBRESOURCE_FOOTPRINT Footprint = {};
			UINT RowCount = 0;
			UINT64 RowSize = 0;
			UINT64 TotalSize = 0;
		};

		struct HdrSurface
		{
			ComPtr<ID3D12Resource> Resource;
			D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_RENDER_TARGET;
			D3D12_RESOURCE_STATES FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			UINT RTV = InvalidDescriptor;
			UINT SRV = InvalidDescriptor;
			int Width = 0;
			int Height = 0;
			UINT Samples = 1;
			DXGI_FORMAT Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		};
        struct VSMMomentMap
        {
            HdrSurface Raw, Horizontal, Filtered;
            ReadbackResource Diagnostic;
            bool Ready = false, RawWritten = false;
        };

		enum class SampleProgram : UINT { Depth, Material, Velocity, Restore, Count };
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
			IndexedDrawRange LastValidRange;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D12Resource> Resource;
			D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON;
			D3D12_RESOURCE_STATES FrameStartState = D3D12_RESOURCE_STATE_COMMON;
			D3D12_VERTEX_BUFFER_VIEW VertexView = {};
			D3D12_INDEX_BUFFER_VIEW IndexView = {};
			bool Dirty = false;
			bool FrameStartDirty = false;
		};

		struct TextureResource
		{
			TextureDesc Description;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D12Resource> Resource;
			D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON;
			D3D12_RESOURCE_STATES FrameStartState = D3D12_RESOURCE_STATE_COMMON;
			ReadbackResource Readback;
			UINT RTV = InvalidDescriptor;
			UINT DSV = InvalidDescriptor;
			UINT SRV = InvalidDescriptor;
			bool Dirty = false;
			bool FrameStartDirty = false;
		};

		bool Fail(const std::string& error)
		{
			LastError = error;
			return false;
		}

		bool FailHRESULT(const char* operation, HRESULT result)
		{
			std::string error = HResultMessage(operation, result);
			if (Device)
			{
				const HRESULT removed = Device->GetDeviceRemovedReason();
				if (FAILED(removed))
					error += "; device removed reason " + HResultMessage("", removed);
			}
			return Fail(error);
		}

		bool CheckThread(const char* operation)
		{
			if (!Initialized) return Fail("DX12 RHI is not initialized");
			if (OwnerThread != std::this_thread::get_id())
				return Fail(std::string(operation) + " must execute on the owning DX12 RHI thread");
			return RecoverSubmittedWork(operation);
		}

		bool LoadRuntime()
		{
			D3D12Module = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
			if (!D3D12Module) return Fail("Could not load the system d3d12.dll");
			CreateDeviceFunction = reinterpret_cast<D3D12CreateDeviceFunction>(
				GetProcAddress(D3D12Module, "D3D12CreateDevice"));
			SerializeRootSignatureFunction = reinterpret_cast<D3D12SerializeRootSignatureFunction>(
				GetProcAddress(D3D12Module, "D3D12SerializeRootSignature"));
			if (!CreateDeviceFunction || !SerializeRootSignatureFunction)
				return Fail("The system d3d12.dll does not expose required DirectX 12 entry points");
			return true;
		}

		bool CreateDescriptorArena(DescriptorArena& arena, D3D12_DESCRIPTOR_HEAP_TYPE type,
			UINT capacity, bool shaderVisible)
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type = type;
			desc.NumDescriptors = capacity;
			desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
				: D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			const HRESULT result = Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(arena.Heap.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateDescriptorHeap", result);
			arena.Type = type;
			arena.Capacity = capacity;
			arena.Increment = Device->GetDescriptorHandleIncrementSize(type);
			arena.ShaderVisible = shaderVisible;
			return true;
		}

		bool CreateBufferResource(std::size_t size, D3D12_HEAP_TYPE heapType,
			D3D12_RESOURCE_STATES initialState, ComPtr<ID3D12Resource>& output)
		{
			const D3D12_HEAP_PROPERTIES heap = HeapProperties(heapType);
			const D3D12_RESOURCE_DESC desc = BufferResourceDesc(size);
			const HRESULT result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
				initialState, nullptr, IID_PPV_ARGS(output.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("ID3D12Device::CreateCommittedResource(buffer)", result);
		}

		bool CreateReadback(ID3D12Resource* texture, ReadbackResource& output)
		{
			const D3D12_RESOURCE_DESC desc = texture->GetDesc();
			Device->GetCopyableFootprints(&desc, 0, 1, 0, &output.Footprint,
				&output.RowCount, &output.RowSize, &output.TotalSize);
			return CreateBufferResource(static_cast<std::size_t>(output.TotalSize), D3D12_HEAP_TYPE_READBACK,
				D3D12_RESOURCE_STATE_COPY_DEST, output.Buffer);
		}

		D3D12_RESOURCE_STATES FinalBufferState(BufferUsage usage) const
		{
			return usage == BufferUsage::Index ? D3D12_RESOURCE_STATE_INDEX_BUFFER
				: D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
		}

		D3D12_RESOURCE_STATES FinalTextureState(const TextureDesc& desc) const
		{
			if (desc.Format == TextureFormat::Depth32) return D3D12_RESOURCE_STATE_DEPTH_WRITE;
			return desc.RenderTarget ? D3D12_RESOURCE_STATE_RENDER_TARGET
				: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
		}

		void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES& current,
			D3D12_RESOURCE_STATES next)
		{
			if (current == next) return;
			D3D12_RESOURCE_BARRIER barrier = {};
			barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barrier.Transition.pResource = resource;
			barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			barrier.Transition.StateBefore = current;
			barrier.Transition.StateAfter = next;
			CommandList->ResourceBarrier(1, &barrier);
			current = next;
		}

		bool CreateMainTargets(int width, int height)
		{
			ComPtr<ID3D12Resource> color;
			ComPtr<ID3D12Resource> depth;
			ReadbackResource colorReadback;
			ReadbackResource depthReadback;

			D3D12_RESOURCE_DESC colorDesc = {};
			colorDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			colorDesc.Width = static_cast<UINT64>(width);
			colorDesc.Height = static_cast<UINT>(height);
			colorDesc.DepthOrArraySize = 1;
			colorDesc.MipLevels = 1;
			colorDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			colorDesc.SampleDesc.Count = 1;
			colorDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			colorDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
			D3D12_CLEAR_VALUE colorClear = {};
			colorClear.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			colorClear.Color[3] = 1.0f;
			D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
			HRESULT result = Device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &colorDesc,
				D3D12_RESOURCE_STATE_RENDER_TARGET, &colorClear, IID_PPV_ARGS(color.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateCommittedResource(main color)", result);
			if (!CreateReadback(color.Get(), colorReadback)) return false;

			D3D12_RESOURCE_DESC depthDesc = colorDesc;
			depthDesc.Format = DXGI_FORMAT_R32_TYPELESS;
			depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
			D3D12_CLEAR_VALUE depthClear = {};
			depthClear.Format = DXGI_FORMAT_D32_FLOAT;
			depthClear.DepthStencil.Depth = 1.0f;
			result = Device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &depthDesc,
				D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthClear, IID_PPV_ARGS(depth.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateCommittedResource(main depth)", result);
			if (!CreateReadback(depth.Get(), depthReadback)) return false;

			D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
			rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
			Device->CreateRenderTargetView(color.Get(), &rtv, RTVArena.Cpu(MainColorRTV));
			D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
			dsv.Format = DXGI_FORMAT_D32_FLOAT;
			dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			Device->CreateDepthStencilView(depth.Get(), &dsv, DSVArena.Cpu(MainDepthDSV));
			if (MainDepthReadOnlyDSV == InvalidDescriptor) MainDepthReadOnlyDSV = DSVArena.Allocate();
			if (MainDepthReadOnlyDSV == InvalidDescriptor) return Fail("DX12 read-only depth descriptor allocation failed");
			dsv.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
			Device->CreateDepthStencilView(depth.Get(), &dsv, DSVArena.Cpu(MainDepthReadOnlyDSV));
			if (MainDepthSRV == InvalidDescriptor) MainDepthSRV = SRVArena.Allocate();
			if (MainDepthSRV == InvalidDescriptor) return Fail("DX12 main depth SRV allocation failed");
			D3D12_SHADER_RESOURCE_VIEW_DESC depthView {};
			depthView.Format = DXGI_FORMAT_R32_FLOAT;
			depthView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			depthView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			depthView.Texture2D.MipLevels = 1;
			Device->CreateShaderResourceView(depth.Get(), &depthView, SRVArena.Cpu(MainDepthSRV));

			MainColor = std::move(color);
			MainDepth = std::move(depth);
			PostColor.Reset();
			PostColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			for (auto& surface : HdrSurfaces)
			{
				surface.Resource.Reset();
				surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			for (auto& surface : GBufferSurfaces)
			{
				surface.Resource.Reset();
				surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			for (auto& surface : AmbientOcclusionSurfaces)
			{
				surface.Resource.Reset();
				surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			DestroySaoDepthHierarchy();
			ReflectionSurface.Resource.Reset();
			ReflectionSurface.State = ReflectionSurface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			for (auto& surface : MotionSurfaces)
			{
				surface.Resource.Reset();
				surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			MotionReadback = {};
			MainColorReadback = std::move(colorReadback);
			for (auto* history : {&TemporalColorHistory, &TemporalDepthHistory})
				for (auto& surface : *history)
				{
					surface.Resource.Reset();
					surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
				}
			TemporalHistoryReadIndex = 0;
            TemporalHistoryValid = TemporalResolved = false;
            GI.Invalidate(); Probes.Invalidate();
            ResetLightmaps(); ResetRefraction(); ResetPeeling();
            for (auto& surface : ProbeSurfaces) { surface.Resource.Reset(); surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET; }
            for (auto& surface : GISurfaces)
            {
                surface.Resource.Reset();
                surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
            }
			FogAtlas.Resource.Reset();
			FogAtlas.State = FogAtlas.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			LightShaftMask.Resource.Reset();
			LightShaftMask.State = LightShaftMask.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			LightShaftResolved = false;
			for (auto& surface : VolumeLightScratch)
			{
				surface.Resource.Reset();
				surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			FogResolved = false;
			MainDepthReadback = std::move(depthReadback);
			MainColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			MainDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
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
			const HRESULT result = Evans::Renderer::CompileCachedD3DShader(ShaderCode.data(), ShaderCode.size(), "EvansDX12.hlsl",
				defines, nullptr, entryPoint, target, flags, 0,
				bytecode.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				else LastError = HResultMessage("D3DCompile", result);
				return false;
			}
			return true;
		}

        bool UploadForwardPlusScene()
        {
            for (UINT* descriptor : {&ForwardPlusSceneView, &ForwardPlusHeaderView, &ForwardPlusIndexView,
                &ForwardPlusEmptyHeaderView, &ForwardPlusEmptyIndexView, &ForwardPlusEmptyContactView})
            {
                if (*descriptor == InvalidDescriptor) *descriptor = SRVArena.Allocate();
                if (*descriptor == InvalidDescriptor) return Fail("DX12 Forward+ descriptor storage is exhausted");
            }
            const auto& lights = LightListFrame.Description().Lights->GetShaderLights();
            const auto lightOffset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            const auto lightBytes = std::max<std::size_t>(lights.size(), 1) * sizeof(ShaderLight);
            const auto emptyOffset = lightOffset + lightBytes;
            const auto required = lightBytes + sizeof(ForwardPlusTile) + sizeof(std::uint32_t);
            if (lightOffset > ConstantBufferCapacity || required > ConstantBufferCapacity - lightOffset)
                return Fail("DX12 Forward+ complete-light upload exceeds the frame arena");
            std::memset(ConstantData + lightOffset, 0, required);
            if (!lights.empty()) std::memcpy(ConstantData + lightOffset, lights.data(), lights.size() * sizeof(ShaderLight));
            const auto view = [&](UINT descriptor, std::size_t offset, UINT stride, UINT count)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv {}; srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Buffer.FirstElement = offset / stride; srv.Buffer.NumElements = count; srv.Buffer.StructureByteStride = stride;
                Device->CreateShaderResourceView(ConstantBuffer.Get(), &srv, SRVArena.Cpu(descriptor));
            };
            view(ForwardPlusSceneView, lightOffset, sizeof(ShaderLight), UINT(std::max<std::size_t>(lights.size(), 1)));
            ForwardPlusSceneAddress = ConstantBuffer->GetGPUVirtualAddress() + lightOffset;
            view(ForwardPlusEmptyHeaderView, emptyOffset, sizeof(ForwardPlusTile), 1);
            view(ForwardPlusEmptyIndexView, emptyOffset + sizeof(ForwardPlusTile), sizeof(std::uint32_t), 1);
            D3D12_SHADER_RESOURCE_VIEW_DESC contact {}; contact.Format = DXGI_FORMAT_R32_FLOAT;
            contact.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            contact.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            contact.Texture2DArray.ArraySize = contact.Texture2DArray.MipLevels = 1;
            Device->CreateShaderResourceView(nullptr, &contact, SRVArena.Cpu(ForwardPlusEmptyContactView));
            // The frame arena and descriptors are reused only after the RHI fence.
            // Capture descriptors never change when main-view assignment finishes.
            ConstantOffset = lightOffset + required;
            return true;
        }
        bool BindForwardPlusSurface()
        {
            const bool mainView = CurrentPass.Type == PassType::Color && !CurrentPass.Target;
            const auto selection = !mainView ? LightListSelection::FullScene :
                BoundPipeline.BlendEnabled ? LightListSelection::Transparent : LightListSelection::Opaque;
            LightListSurfaceConstants constants;
            if (!LightListFrame.Surface(BoundPipeline.Lighting, BoundViewProjection, selection, constants, LastError)) return false;
            const auto offset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (offset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - offset)
                return Fail("DX12 Forward+ material constants exceed the frame arena");
            std::memcpy(ConstantData + offset, &constants, sizeof(constants));
            CommandList->SetGraphicsRootConstantBufferView(ForwardPlusControlRoot, ConstantBuffer->GetGPUVirtualAddress() + offset);
            ConstantOffset = offset + sizeof(constants);
            const UINT views[] {ForwardPlusSceneView, mainView ? ForwardPlusHeaderView : ForwardPlusEmptyHeaderView,
                mainView ? ForwardPlusIndexView : ForwardPlusEmptyIndexView};
            for (UINT index = 0; index < ForwardPlusStorageCount; ++index)
                CommandList->SetGraphicsRootDescriptorTable(ForwardPlusStorageFirstRoot + index, SRVArena.Gpu(views[index]));
            const bool contact = mainView && !BoundPipeline.BlendEnabled && ContactReceiverActive;
            CommandList->SetGraphicsRoot32BitConstant(ForwardContactControlRoot, contact ? 1u : 0u, 0);
            CommandList->SetGraphicsRootDescriptorTable(ForwardContactImageRoot,
                SRVArena.Gpu(contact ? ContactDescriptor : ForwardPlusEmptyContactView));
            return true;
        }
        bool UsesForwardPlusMaterial(PassType pass) const
        { return LightListsActive && (pass == PassType::Color || pass == PassType::EnvironmentCapture || pass == PassType::PlanarCapture); }
        Techniques::ShadowFilterMode ForwardPlusFilter(const PipelineState& pipeline, PassType pass) const
        { return pass == PassType::Color && pipeline.SamplesShadowMap ? FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard; }
        ForwardPlusProgram ForwardPlusVariant(const PipelineState& pipeline, PassType pass) const
        {
            return pass == PassType::ForwardPlusDepth ? ForwardPlusDepth : WeightedPass ? ForwardPlusWeighted :
                UsesContactReceiver(pipeline, pass) && ActiveSampleCount > 1 ? ForwardPlusContact : ForwardPlusSurface;
        }
        bool EnsureForwardPlusProgram(Techniques::ShadowFilterMode filter, ForwardPlusProgram program)
        {
            auto& vertex = ClusteredActive ? ClusteredVertex : TiledActive ? TiledVertex : ForwardPlusVertex;
            auto& pixel = (ClusteredActive ? ClusteredPixels : TiledActive ? TiledPixels : ForwardPlusPixels)[static_cast<std::size_t>(filter)][program];
            if (vertex && pixel) return true;
            if (!LoadTechniqueShaderSource("DX12", ClusteredActive ? "ClusteredForward.hlsl" : TiledActive ? "TiledDeferred.hlsl" : "ForwardPlus.hlsl", ShaderCode, LastError)) return false;
            const auto filterName = std::to_string(static_cast<std::uint32_t>(filter));
            const D3D_SHADER_MACRO defines[] {{ClusteredActive ? "EVANS_CLUSTERED_FORWARD_SURFACE" : TiledActive ? "EVANS_TILED_DEFERRED_SURFACE" : "EVANS_FORWARD_PLUS_SURFACE", "1"},
                {ClusteredActive ? "EVANS_CLUSTERED_FORWARD_FILTER" : TiledActive ? "EVANS_TILED_DEFERRED_FILTER" : "EVANS_FORWARD_PLUS_FILTER", filterName.c_str()}, {nullptr, nullptr}};
            if (!vertex && !CompileShader("VSMain", "vs_5_0", vertex, defines, true)) return false;
            constexpr const char* entries[] {"PSMain", "PSContact", "PSWeighted", "PSDepth"};
            return pixel || CompileShader(entries[program], "ps_5_0", pixel, defines, true);
        }

		bool UsesPCSS() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS; }
		bool UsesSurfacePCSS(const PipelineState& pipeline, PassType pass) const
		{ return UsesPCSS() && pass == PassType::Color && pipeline.SamplesShadowMap; }
		bool UsesVSM() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM; }
        bool UsesSurfaceVSM(const PipelineState& pipeline, PassType pass) const
        { return UsesVSM() && pass == PassType::Color && pipeline.SamplesShadowMap; }
        bool UsesContactReceiver(const PipelineState& pipeline, PassType pass) const
        { return FrameContact.Enabled && pass == PassType::Color && !CurrentPassTarget && !pipeline.BlendEnabled; }
        UINT DeferredPipelineKey() const
        { return FrameSampleCount | (UsesPCSS() ? 1u << 16 : UsesVSM() ? 1u << 17 : 0) | (FrameContact.Enabled ? 1u << 18 : 0) | (TiledActive ? 1u << 19 : 0) | (FrameRayShadowImage ? 1u << 20 : 0) | (UsesDedicatedPCF()?1u<<21:0); }
        void BindContactReceiver(UINT controlRoot, UINT imageRoot)
        {
            CommandList->SetGraphicsRoot32BitConstant(controlRoot, ContactReceiverActive ? 1u : 0u, 0);
            CommandList->SetGraphicsRootDescriptorTable(imageRoot, SRVArena.Gpu(ContactDescriptor));
        }

		bool EnsurePCSSProgram(std::size_t program, bool weighted = false)
		{
			if (program >= PCSSProgramCount || (weighted && program >= ShadingTechniqueCount))
				return Fail("Invalid DX12 PCSS receiver program");
			auto& binary = PCSSPixelBytecodes[weighted ? PCSSProgramCount + program : program];
			if (binary) return true;
			if (!LoadTechniqueShaderSource("DX12", "PCSS.hlsl", ShaderCode, LastError)) return false;
			const auto index = std::to_string(program);
			const D3D_SHADER_MACRO defines[] {{"EVANS_PCSS_PROGRAM", index.c_str()}, {nullptr, nullptr}};
			return CompileShader(weighted ? "PSWeighted" : "PSMain", "ps_5_0", binary, defines);
		}

		bool BindPCSS(UINT root, const Matrix4& lightProjection, const LightingState& lighting, bool shadows)
		{
			PCSSConstants constants;
			if (!BuildPCSSConstants(lightProjection, lighting, FrameCascades, FrameShadowFilter.PCSS,
				shadows, constants, LastError)) return false;
			const auto offset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			if (offset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - offset)
				return Fail("DX12 PCSS exhausted the frame constant upload buffer");
			// A root CBV costs two DWORDs. Each aligned upload remains immutable until
			// the frame fence completes; later transparent draws cannot overwrite it.
			std::memcpy(ConstantData + offset, &constants, sizeof(constants));
			CommandList->SetGraphicsRootConstantBufferView(root, ConstantBuffer->GetGPUVirtualAddress() + offset);
			ConstantOffset = offset + sizeof(constants);
			return true;
		}

        bool SupportsVSM() const
        {
            if (!Device) return false;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT format {VSMFormat};
            constexpr UINT Required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
            return SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format))) &&
                (format.Support1 & Required) == Required;
        }

        bool EnsureVSMProgram(std::size_t program, bool weighted = false)
        {
            if (program >= VSMProgramCount || (weighted && program >= ShadingTechniqueCount))
                return Fail("Invalid DX12 variance-shadow program");
            auto& pixel = VSMPixelBytecodes[weighted ? VSMProgramCount + program : program];
            if (pixel && (program != VSMCasterProgram || (VSMCasterVertex && VSMInstanceVertex)) &&
                (program != VSMFilterProgram || VSMFilterVertex)) return true;
            if (!LoadTechniqueShaderSource("DX12", "VSM.hlsl", ShaderCode, LastError)) return false;
            const auto index = std::to_string(program);
            const D3D_SHADER_MACRO defines[] {{"EVANS_VSM_PROGRAM", index.c_str()}, {nullptr, nullptr}};
            if (!pixel && !CompileShader(weighted ? "PSWeighted" : "PSMain", "ps_5_0", pixel, defines)) return false;
            if (program == VSMCasterProgram)
            {
                if (!VSMCasterVertex && !CompileShader("VSMain", "vs_5_0", VSMCasterVertex, defines)) return false;
                if (!VSMInstanceVertex && !CompileShader("VSInstanced", "vs_5_0", VSMInstanceVertex, defines)) return false;
            }
            if (program == VSMFilterProgram && !VSMFilterVertex && !CompileShader("VSMain", "vs_5_0", VSMFilterVertex, defines)) return false;
            return true;
        }

        bool UploadVSM(UINT root, const VSMConstants& constants)
        {
            const auto offset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (offset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - offset)
                return Fail("DX12 VSM exhausted the frame constant upload buffer");
            std::memcpy(ConstantData + offset, &constants, sizeof(constants));
            CommandList->SetGraphicsRootConstantBufferView(root, ConstantBuffer->GetGPUVirtualAddress() + offset);
            ConstantOffset = offset + sizeof(constants);
            return true;
        }

        bool VSMMomentDescriptor(TextureHandle handle, UINT& descriptor)
        {
            auto found = VSMMoments.find(handle.Value);
            if (found == VSMMoments.end() || !found->second.Ready)
                return Fail("DX12 VSM receiver requires completed moment filters in this frame");
            auto& image = found->second.Filtered;
            Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            descriptor = image.SRV;
            return true;
        }

        bool BindVSM(UINT root, const Matrix4& projection, const LightingState& lighting, bool shadows)
        {
            if (shadows)
            {
                UINT descriptor;
                if (!FrameCascades.Frame.CascadeCount)
                { if (!VSMMomentDescriptor(LastShadowTexture, descriptor)) return false; }
                else for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                    if (!VSMMomentDescriptor(FrameCascades.Maps[index], descriptor)) return false;
            }
            VSMConstants constants;
            return BuildVSMConstants(projection, lighting, FrameCascades, FrameShadowFilter.VSM, shadows, constants, LastError) &&
                UploadVSM(root, constants);
        }

        bool EnsureVSMFilter()
        {
            if (VSMFilterPipeline) return true;
            if (!EnsureVSMProgram(VSMFilterProgram)) return false;
            if (!VSMFilterRootSignature)
            {
                D3D12_DESCRIPTOR_RANGE range {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                D3D12_ROOT_PARAMETER parameters[2] {};
                parameters[VSMFilterConstantRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                parameters[VSMFilterConstantRoot].Descriptor.ShaderRegister = ShadowFilterConstantRegister;
                parameters[VSMFilterConstantRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                parameters[VSMFilterImageRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[VSMFilterImageRoot].DescriptorTable = {1, &range};
                parameters[VSMFilterImageRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                D3D12_ROOT_SIGNATURE_DESC signature {}; signature.NumParameters = 2; signature.pParameters = parameters;
                signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
                ComPtr<ID3DBlob> serialized, errors;
                HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.GetAddressOf());
                if (FAILED(result)) return FailHRESULT("SerializeRootSignature(VSM filter)", result);
                result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(VSMFilterRootSignature.GetAddressOf()));
                if (FAILED(result)) return FailHRESULT("CreateRootSignature(VSM filter)", result);
            }
            const auto& pixel = VSMPixelBytecodes[VSMFilterProgram];
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
            pipeline.pRootSignature = VSMFilterRootSignature.Get();
            pipeline.VS = {VSMFilterVertex->GetBufferPointer(), VSMFilterVertex->GetBufferSize()};
            pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
            auto& color = pipeline.BlendState.RenderTarget[0];
            color.SrcBlend = color.SrcBlendAlpha = D3D12_BLEND_ONE; color.DestBlend = color.DestBlendAlpha = D3D12_BLEND_ZERO;
            color.BlendOp = color.BlendOpAlpha = D3D12_BLEND_OP_ADD; color.LogicOp = D3D12_LOGIC_OP_NOOP;
            color.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            pipeline.SampleMask = UINT_MAX; pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; pipeline.RasterizerState.DepthClipEnable = TRUE;
            pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            auto& stencil = pipeline.DepthStencilState.FrontFace;
            stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
            stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS; pipeline.DepthStencilState.BackFace = stencil;
            pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pipeline.NumRenderTargets = 1; pipeline.RTVFormats[0] = VSMFormat; pipeline.SampleDesc.Count = 1;
            const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(VSMFilterPipeline.GetAddressOf()));
            return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(VSM filter)", result);
        }

        bool BeginVSMMoments(TextureHandle handle, int width, int height, bool clear, D3D12_CPU_DESCRIPTOR_HANDLE& target)
        {
            if (!SupportsVSM()) return Fail("DX12 VSM requires renderable, shader-loadable RG32_FLOAT moments");
            if (!EnsureVSMFilter()) return false;
            auto& map = VSMMoments[handle.Value];
            if (!clear && !map.RawWritten) return Fail("DX12 VSM requires an initial caster clear in each frame");
            constexpr float Unoccluded[] {1, 1, 0, 0};
            // Keep descriptors and resources stable until submission completes.
            // Per-map scratch also supports differently sized maps in one list.
            for (auto* image : {&map.Raw, &map.Horizontal, &map.Filtered})
                if (!CreateHdrSurface(*image, width, height, VSMFormat, 1, Unoccluded)) return false;
            map.Ready = false;
            Transition(map.Raw.Resource.Get(), map.Raw.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
            target = RTVArena.Cpu(map.Raw.RTV);
            if (clear) CommandList->ClearRenderTargetView(target, Unoccluded, 0, nullptr);
            return true;
        }

        bool FilterVSMMoments(TextureHandle handle)
        {
            auto found = VSMMoments.find(handle.Value);
            if (found == VSMMoments.end()) return Fail("DX12 VSM filter has no caster map");
            auto& map = found->second;
            auto constants = BuildVSMParameters(FrameShadowFilter.VSM);
            CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
            CommandList->SetGraphicsRootSignature(VSMFilterRootSignature.Get());
            CommandList->SetPipelineState(VSMFilterPipeline.Get());
            CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const D3D12_VIEWPORT viewport {0, 0, float(map.Raw.Width), float(map.Raw.Height), 0, 1};
            const D3D12_RECT scissor {0, 0, map.Raw.Width, map.Raw.Height};
            CommandList->RSSetViewports(1, &viewport); CommandList->RSSetScissorRects(1, &scissor);
            for (UINT axis = 0; axis < 2; ++axis)
            {
                constants.Control.W = axis;
                if (!UploadVSM(VSMFilterConstantRoot, constants)) return false;
                auto& source = axis == 0 ? map.Raw : map.Horizontal;
                auto& destination = axis == 0 ? map.Horizontal : map.Filtered;
                Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
                CommandList->SetGraphicsRootDescriptorTable(VSMFilterImageRoot, SRVArena.Gpu(source.SRV));
                const auto target = RTVArena.Cpu(destination.RTV);
                CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
                constexpr UINT TriangleVertexCount = 3;
                CommandList->DrawInstanced(TriangleVertexCount, 1, 0, 0);
                CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
            }
            Transition(map.Filtered.Resource.Get(), map.Filtered.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            map.Ready = map.RawWritten = true;
            PendingFrame.Statistics.ShadowMomentFilterPassCount += 2;
            return true;
        }

        void DestroyVSMMoments(TextureHandle handle)
        {
            const auto found = VSMMoments.find(handle.Value);
            if (found == VSMMoments.end()) return;
            for (auto* image : {&found->second.Raw, &found->second.Horizontal, &found->second.Filtered})
            { RTVArena.Release(image->RTV); SRVArena.Release(image->SRV); }
            VSMMoments.erase(found);
        }

		bool EnsureTechnique(Techniques::ShadingModel model)
		{
			// Shader selection is a pipeline dimension. Sources are independent
			// native programs; the PSO cache includes the selected technique.
			const std::size_t index = ShadingTechniqueIndex(model, FrameAreaLights);
			const char* name = ShadingTechniqueFile(model, FrameAreaLights);
			if (index >= StandaloneShadingTechniqueCount || !name) return Fail("Unknown DX12 shading technique");
			if (VertexBytecodes[index] && PixelBytecodes[index]) return true;
			const std::string file = std::string(name) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX12", file.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex)
				|| !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			VertexBytecodes[index] = std::move(vertex);
			PixelBytecodes[index] = std::move(pixel);
			return true;
		}

		bool EnsureInstanceProgram(std::size_t index)
		{
			if (InstanceBytecodes[index]) return true;
			if (!LoadTechniqueShaderSource("DX12", "InstancedRendering.hlsl", ShaderCode, LastError)) return false;
			constexpr const char* entries[] {"VSForward", "VSGouraud", "VSDeferred", "VSShadow", "VSVelocity", "VSCascade"};
			return CompileShader(entries[index], "vs_5_0", InstanceBytecodes[index]);
		}

		bool UploadInstances(const InstanceConstantBlock& constants, bool velocity)
		{
            if (ActiveGpuBatch)
            {
                CommandList->SetGraphicsRootConstantBufferView(velocity ? MotionInstanceRoot : ForwardInstanceRoot,
                    ActiveGpuBatch->Output->GetGPUVirtualAddress());
                return true;
            }
			// Each command references an immutable, aligned slice. Pages survive until
			// the frame fence completes; BeginFrame waits before resetting the cursor.
			// Root CBVs permit indexing *within* this constant array, without an SRV
			// descriptor per batch. The page pool grows independently of draw constants.
			// https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-descriptors-directly-in-the-root-signature
			InstancePageOffset = AlignUp(InstancePageOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			if (InstancePageOffset + sizeof(constants) > InstanceUploadPageCapacity)
			{
				++InstancePageIndex; InstancePageOffset = 0;
			}
			if (InstancePageIndex == InstancePages.size())
			{
				InstanceUploadPage page;
				if (!CreateBufferResource(InstanceUploadPageCapacity, D3D12_HEAP_TYPE_UPLOAD,
					D3D12_RESOURCE_STATE_GENERIC_READ, page.Buffer)) return false;
				D3D12_RANGE noRead {0, 0};
				const HRESULT result = page.Buffer->Map(0, &noRead, reinterpret_cast<void**>(&page.Data));
				if (FAILED(result)) return FailHRESULT("Map(instance upload page)", result);
				InstancePages.push_back(std::move(page));
			}
			auto& page = InstancePages[InstancePageIndex];
			std::memcpy(page.Data + InstancePageOffset, &constants, sizeof(constants));
			CommandList->SetGraphicsRootConstantBufferView(velocity ? MotionInstanceRoot : ForwardInstanceRoot,
				page.Buffer->GetGPUVirtualAddress() + InstancePageOffset);
			InstancePageOffset += sizeof(constants);
			return true;
		}

		bool PrepareCascadedShadows(const CascadedShadowState& state)
		{
			FrameCascades = state;
			if (state.Frame.CascadeCount == 0) return true;
			if (state.Frame.CascadeCount > MaximumShadowCascades || state.MapResolution <= 0)
				return Fail("DX12 cascaded shadow frame is invalid");
			for (std::uint32_t index = 0; index < state.Frame.CascadeCount; ++index)
			{
				const auto found = Textures.find(state.Maps[index].Value);
				if (found == Textures.end() || found->second.Description.Format != TextureFormat::Depth32 ||
					found->second.Description.Width != state.MapResolution || found->second.Description.Height != state.MapResolution ||
					found->second.DSV == InvalidDescriptor || found->second.SRV == InvalidDescriptor)
					return Fail("DX12 CSM requires complete depth targets with equal dimensions");
			}
			if (CascadeVertexBytecode && CascadePixelBytecode) return true;
			if (!LoadTechniqueShaderSource("DX12", "CascadedShadowMap.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			CascadeVertexBytecode = std::move(vertex);
			CascadePixelBytecode = std::move(pixel);
			return true;
		}

		void UploadCascadedShadowConstants()
		{
			// Root CBVs point into the frame's immutable upload allocation. Reserve an
			// inactive block too: depth/outline draws must not inherit previous bindings.
			// https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html
			for (unsigned active = 0; active < CascadeConstantAddresses.size(); ++active)
			{
				ConstantOffset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
				const auto constants = active && FrameVirtualShadowAtlas ? BuildVirtualShadowSamplingConstants(FrameVirtualShadowAtlas->Info,FrameVirtualShadowSampling,FrameShadowFilter) :
                    BuildCascadedShadowConstants(active ? FrameCascades : CascadedShadowState{}, FrameShadowFilter);
				std::memcpy(ConstantData + ConstantOffset, &constants, sizeof(constants));
				CascadeConstantAddresses[active] = ConstantBuffer->GetGPUVirtualAddress() + ConstantOffset;
				ConstantOffset += sizeof(constants);
			}
		}

		bool BindCascadedShadows(bool enabled, bool deferred)
		{
			const bool active = enabled && (FrameCascades.Frame.CascadeCount != 0 || FrameVirtualShadowAtlas);
			const UINT firstRoot = deferred ? DeferredCascadeFirstRoot : ForwardCascadeFirstRoot;
			const UINT constantRoot = deferred ? DeferredCascadeConstantsRoot : ForwardCascadeConstantsRoot;
			CommandList->SetGraphicsRootConstantBufferView(constantRoot, CascadeConstantAddresses[active ? 1 : 0]);
			for (std::uint32_t index = 0; index < MaximumShadowCascades; ++index)
			{
				UINT descriptor = NullShadowSRV;
				if (active && FrameVirtualShadowAtlas && index<FrameVirtualShadowViews.size()) descriptor=FrameVirtualShadowViews[index];
                else if (active && index < FrameCascades.Frame.CascadeCount)
				{
					auto& texture = Textures.at(FrameCascades.Maps[index].Value);
					Transition(texture.Resource.Get(), texture.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					descriptor = texture.SRV;
					if (UsesVSM() && !VSMMomentDescriptor(FrameCascades.Maps[index], descriptor)) return false;
				}
				CommandList->SetGraphicsRootDescriptorTable(firstRoot + index, SRVArena.Gpu(descriptor));
			}
			return true;
		}

		ComPtr<ID3D12DescriptorHeap> MaterialSamplers;
		bool AnisotropicFiltering = false;

		bool CreateMaterialSamplers()
		{
			D3D12_DESCRIPTOR_HEAP_DESC layout{};
			layout.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
			layout.NumDescriptors = 2;
			layout.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			const HRESULT result = Device->CreateDescriptorHeap(&layout, IID_PPV_ARGS(MaterialSamplers.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateDescriptorHeap(material samplers)", result);
			auto descriptor = MaterialSamplers->GetCPUDescriptorHandleForHeapStart();
			D3D12_SAMPLER_DESC sampler{};
			sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
			sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
			sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			sampler.MaxAnisotropy = 1;
			Device->CreateSampler(&sampler, descriptor);
			descriptor.ptr += Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
			sampler.Filter = D3D12_FILTER_ANISOTROPIC;
			sampler.MaxAnisotropy = MaterialAnisotropy;
			sampler.MaxLOD = D3D12_FLOAT32_MAX;
			Device->CreateSampler(&sampler, descriptor);
			return true;
		}

		static void DescribeMaterialSampler(D3D12_ROOT_PARAMETER& parameter, D3D12_DESCRIPTOR_RANGE& range)
		{
			range = {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
			parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			parameter.DescriptorTable = {1, &range};
			parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		}

		void BindMaterialSampler(UINT root)
		{
			auto descriptor = MaterialSamplers->GetGPUDescriptorHandleForHeapStart();
			if (AnisotropicFiltering)
				descriptor.ptr += Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
			CommandList->SetGraphicsRootDescriptorTable(root, descriptor);
		}

		void BindSceneDescriptorHeaps()
		{
			ID3D12DescriptorHeap* heaps[] = {SRVArena.Heap.Get(), MaterialSamplers.Get()};
			CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
		}

		bool CreatePipelineObjects()
		{
			if (!CreateMaterialSamplers()) return false;
			if (!EnsureTechnique(Techniques::ShadingModel::Phong)) return false;
			if (!CompileShader("PSOutline", "ps_5_0", OutlinePixelBytecode)) return false;
			if (!LoadTechniqueShaderSource("DX12", "ShadowMapping.hlsl", ShaderCode, LastError) ||
				!CompileShader("VSMain", "vs_5_0", ShadowVertexBytecode) ||
				!CompileShader("PSMain", "ps_5_0", ShadowPixelBytecode)) return false;

			std::array<D3D12_DESCRIPTOR_RANGE, MaterialTextureCount + MaximumShadowCascades> textureRanges {};
			std::array<D3D12_ROOT_PARAMETER, ForwardMaterialSamplerRoot + 1> parameters {};
			parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[0].Descriptor.ShaderRegister = 0;
			parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
			for (UINT textureIndex = 0; textureIndex < textureRanges.size(); ++textureIndex)
			{
				D3D12_DESCRIPTOR_RANGE& range = textureRanges[textureIndex];
				range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
				range.NumDescriptors = 1;
				range.BaseShaderRegister = textureIndex < MaterialTextureCount ? textureIndex :
					CascadeFirstTextureRegister + textureIndex - MaterialTextureCount;
				range.OffsetInDescriptorsFromTableStart =
					D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
				D3D12_ROOT_PARAMETER& parameter = parameters[textureIndex + 1];
				parameter.ParameterType =
					D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
				parameter.DescriptorTable.NumDescriptorRanges = 1;
				parameter.DescriptorTable.pDescriptorRanges = &range;
				parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			}
			parameters[ForwardCascadeConstantsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[ForwardCascadeConstantsRoot].Descriptor.ShaderRegister = 2;
			parameters[ForwardCascadeConstantsRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			parameters[ForwardInstanceRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[ForwardInstanceRoot].Descriptor.ShaderRegister = InstanceConstantRegister;
			parameters[ForwardInstanceRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
			// Root snapshots keep per-draw fog parameters immutable while recording.
			parameters[ForwardFogConstantsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[ForwardFogConstantsRoot].Descriptor.ShaderRegister = FogConstantRegister;
			parameters[ForwardFogConstantsRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_DESCRIPTOR_RANGE fogRange {};
			fogRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; fogRange.NumDescriptors = 1;
			fogRange.BaseShaderRegister = FogTextureRegister;
			fogRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
			parameters[ForwardFogTextureRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			parameters[ForwardFogTextureRoot].DescriptorTable = {1, &fogRange};
			parameters[ForwardFogTextureRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			// Fog and Lumen use immutable CBV snapshots so the complete surface
            // interface stays below the 64-DWORD root-signature limit.
			parameters[ForwardTransparencyRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			parameters[ForwardTransparencyRoot].Constants = {TransparencyConstantRegister, 0, sizeof(WeightedTransparencyConstants) / sizeof(std::uint32_t)};
			parameters[ForwardTransparencyRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			parameters[ForwardShadowFilterRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[ForwardShadowFilterRoot].Descriptor.ShaderRegister = ShadowFilterConstantRegister;
			parameters[ForwardShadowFilterRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
			// One control word and one image table keep contact independent of the primary filter.
			parameters[ForwardContactControlRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			parameters[ForwardContactControlRoot].Constants = {ContactControlRegister, 0, 1};
			parameters[ForwardContactControlRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_DESCRIPTOR_RANGE contactRange {};
			contactRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; contactRange.NumDescriptors = 1;
			contactRange.BaseShaderRegister = ContactImageRegister;
			contactRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
			parameters[ForwardContactImageRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			parameters[ForwardContactImageRoot].DescriptorTable = {1, &contactRange};
			parameters[ForwardContactImageRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            parameters[ForwardLumenVolumeRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[ForwardLumenVolumeRoot].Descriptor.ShaderRegister=LumenVolumeConstantRegister;
            parameters[ForwardLumenVolumeRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
            D3D12_DESCRIPTOR_RANGE lumenRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,LumenVolumeImageRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[ForwardLumenVolumeImageRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[ForwardLumenVolumeImageRoot].DescriptorTable={1,&lumenRange};
            parameters[ForwardLumenVolumeImageRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
            D3D12_DESCRIPTOR_RANGE rayRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,RayShadowImageRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[ForwardRayShadowImageRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[ForwardRayShadowImageRoot].DescriptorTable={1,&rayRange};
            parameters[ForwardRayShadowImageRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
            parameters[ForwardPlusControlRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[ForwardPlusControlRoot].Descriptor.ShaderRegister = ForwardPlusControlRegister;
            parameters[ForwardPlusControlRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            std::array<D3D12_DESCRIPTOR_RANGE, ForwardPlusStorageCount> lightRanges {};
            for (UINT index = 0; index < ForwardPlusStorageCount; ++index)
            {
                lightRanges[index] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, ForwardPlusFirstRegister + index, 0,
                    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                auto& parameter = parameters[ForwardPlusStorageFirstRoot + index];
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameter.DescriptorTable = {1, &lightRanges[index]};
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            }
			D3D12_STATIC_SAMPLER_DESC sampler {};
			sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
			sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
			sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			sampler.MaxLOD = D3D12_FLOAT32_MAX;
			sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			std::array<D3D12_STATIC_SAMPLER_DESC, 2> samplers {sampler, sampler};
			samplers[0].ShaderRegister = FogSamplerRegister;
			samplers[1].ShaderRegister = LumenVolumeSamplerRegister;
			D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
			rootDesc.NumParameters = static_cast<UINT>(parameters.size());
			rootDesc.pParameters = parameters.data();
			D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
			DescribeMaterialSampler(parameters[ForwardMaterialSamplerRoot], materialSamplerRange);
			rootDesc.NumStaticSamplers = static_cast<UINT>(samplers.size());
			rootDesc.pStaticSamplers = samplers.data();
			rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
				| D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
				| D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
				| D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
			ComPtr<ID3DBlob> serialized;
			ComPtr<ID3DBlob> errors;
			HRESULT result = SerializeRootSignatureFunction(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
				serialized.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				else LastError = HResultMessage("D3D12SerializeRootSignature", result);
				return false;
			}
			result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
				IID_PPV_ARGS(RootSignature.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateRootSignature", result);

			if (!CreateBufferResource(ConstantBufferCapacity, D3D12_HEAP_TYPE_UPLOAD,
				D3D12_RESOURCE_STATE_GENERIC_READ, ConstantBuffer)) return false;
			D3D12_RANGE noRead = { 0, 0 };
			result = ConstantBuffer->Map(0, &noRead, reinterpret_cast<void**>(&ConstantData));
			if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(constants)", result);
			return true;
		}

		std::uint64_t PipelineKey(const PipelineState& pipeline, PassType pass, bool instanced) const
		{
            constexpr unsigned ForwardPlusFilterKeyShift = 38, PeelingLayerKeyShift = 42;
			// PlanarCapture is the ninth pass type: reserve four low bits before
			// packing fill mode, otherwise a solid mirror collides with wire color.
			static_assert(static_cast<std::uint32_t>(PassType::ForwardPlusDepth) < (1u << 4), "Extend the native pass key field");
			return static_cast<std::uint32_t>(pass)
				| (static_cast<std::uint32_t>(pipeline.Fill) << 4)
				| (pipeline.DepthTest ? 1u << 6 : 0u)
				| (pipeline.DepthWrite ? 1u << 7 : 0u)
				| (pipeline.ColorWrite ? 1u << 8 : 0u)
				| (pipeline.OutlineExtrusion ? 1u << 9 : 0u)
				| (pipeline.BlendEnabled ? 1u << 10 : 0u)
				| (static_cast<std::uint32_t>(ShadingTechniqueIndex(pipeline.Shading, FrameAreaLights)) << 11)
				| (pipeline.BackFaceCulling ? 1u << 14 : 0u)
				| (RealtimeWorldIsMirrored(BoundWorld) ? 1u << 15 : 0u)
				| (ActiveColorFormat == DXGI_FORMAT_R16G16B16A16_FLOAT ? 1u << 16 : 0u)
				| (pass == PassType::Shadow && FrameCascades.Frame.CascadeCount != 0 ? 1u << 17 : 0u)
				| (ActiveSampleCount << 18)
				| (instanced ? std::uint64_t{1} << 32 : 0) // Separate from the packed pass/sample word.
				| (WeightedPass ? std::uint64_t{1} << 33 : 0)
                | (CurrentPass.DepthPeelingLayer ? std::uint64_t{1} << PeelingLayerKeyShift : 0)
                | (UsesParallaxMaterial(pipeline,pass) ? std::uint64_t{1} << 43 : 0)
                | (UsesSurfacePCF(pipeline,pass)?std::uint64_t{1}<<45:0)
                | (UsesMaterialStage(pipeline,pass)?std::uint64_t(unsigned(SelectMaterialTechnique(pipeline,WeightedPass))+1)<<MaterialPipelineKeyShift:0)
                | (FrameRayShadowImage && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap && !UsesForwardPlusMaterial(pass) ? std::uint64_t{1} << 44 : 0)
				| (UsesSurfacePCSS(pipeline, pass) ? std::uint64_t{1} << 34 : 0)
                | (UsesVSM() && (pass == PassType::Shadow || UsesSurfaceVSM(pipeline, pass)) ? std::uint64_t{1} << 35 : 0)
                | (UsesContactReceiver(pipeline, pass) ? std::uint64_t{1} << 36 : 0)
                | (UsesForwardPlusMaterial(pass) ? std::uint64_t{1} << 37 : 0)
                | (UsesForwardPlusMaterial(pass) && ClusteredActive ? std::uint64_t{1} << 40 : 0)
                | (UsesForwardPlusMaterial(pass) && TiledActive ? std::uint64_t{1} << 41 : 0)
                // Forward+ specializes Hard and PCF in distinct fragment programs.
                | (UsesForwardPlusMaterial(pass) ? std::uint64_t(ForwardPlusFilter(pipeline, pass)) << ForwardPlusFilterKeyShift : 0);
		}

		ID3D12PipelineState* GetPipeline(const PipelineState& pipeline, PassType pass, bool instanced = false)
		{
			const auto key = PipelineKey(pipeline, pass, instanced);
			auto existing = Pipelines.find(key);
			if (existing != Pipelines.end()) return existing->second.Get();

			const D3D12_INPUT_ELEMENT_DESC inputElements[] = {
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
			};
			D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
			const bool velocity = pass == PassType::Velocity;
			const bool parallax=UsesParallaxMaterial(pipeline,pass);
            if (parallax && !EnsureParallax()) return nullptr;
            desc.pRootSignature = parallax ? ParallaxRoot.Get() : velocity ? MotionRootSignature.Get() : RootSignature.Get();
			const auto model = pass == PassType::Color || pass == PassType::EnvironmentCapture || pass == PassType::PlanarCapture
				? pipeline.Shading : Techniques::ShadingModel::Phong;
			const bool geometry = pass == PassType::GBuffer;
            const bool forwardPlus = UsesForwardPlusMaterial(pass) || pass == PassType::ForwardPlusDepth;
            if (forwardPlus && model == Techniques::ShadingModel::Gouraud)
            { Fail("DX12 Forward+ cannot consume pixel-tile lists during Gouraud vertex lighting"); return nullptr; }
			if (geometry ? !EnsureDeferredResources() : (!forwardPlus && !EnsureTechnique(model))) return nullptr;
			const std::size_t techniqueIndex = static_cast<std::size_t>(model);
            const auto surfaceIndex = ShadingTechniqueIndex(model, FrameAreaLights);
			ID3DBlob* vertexShader = geometry ? DeferredVertexBytecode.Get() : VertexBytecodes[surfaceIndex].Get();
            if (forwardPlus)
            {
                if (!EnsureForwardPlusProgram(ForwardPlusFilter(pipeline, pass), ForwardPlusVariant(pipeline, pass))) return nullptr;
                vertexShader = ClusteredActive ? ClusteredVertex.Get() : TiledActive ? TiledVertex.Get() : ForwardPlusVertex.Get();
            }
			if (velocity) vertexShader = MotionVertexBytecode.Get();
			if (pass == PassType::Shadow) vertexShader = ShadowVertexBytecode.Get();
			if (pass == PassType::Shadow && FrameCascades.Frame.CascadeCount != 0) vertexShader = CascadeVertexBytecode.Get();
			if (instanced)
			{
				const auto program = velocity ? InstanceVertexProgram::Velocity :
					pass == PassType::Shadow ? InstanceVertexProgram::Shadow : geometry ? InstanceVertexProgram::Deferred :
					model == Techniques::ShadingModel::Gouraud ? InstanceVertexProgram::Gouraud : InstanceVertexProgram::Forward;
				const auto programIndex = pass == PassType::Shadow && FrameCascades.Frame.CascadeCount != 0 ?
					CascadeInstanceProgram : static_cast<std::size_t>(program);
				if (!EnsureInstanceProgram(programIndex)) return nullptr;
				vertexShader = InstanceBytecodes[programIndex].Get();
			}
            const bool vsmCaster = UsesVSM() && pass == PassType::Shadow;
            if (vsmCaster)
            {
                if (!EnsureVSMProgram(VSMCasterProgram)) return nullptr;
                vertexShader = instanced ? VSMInstanceVertex.Get() : VSMCasterVertex.Get();
            }

            if (pass == PassType::ContactDepth)
            {
                if (!ContactShadows.PrepareDepth(LastError)) return nullptr;
                if (!instanced) vertexShader = ContactShadows.DepthVS();
            }
            if (parallax) vertexShader=ParallaxVertex.Get();
			desc.VS = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
			ID3DBlob* pixelShader = pass == PassType::Shadow
				? ShadowPixelBytecode.Get()
				: (geometry ? DeferredPixelBytecode.Get()
					: (pass == PassType::Outline ? OutlinePixelBytecode.Get() : PixelBytecodes[surfaceIndex].Get()));
			if (velocity) pixelShader = MotionPixelBytecode.Get();
			if (WeightedPass && !forwardPlus)
			{
				if (!EnsureWeightedTechnique(model)) return nullptr;
				pixelShader = WeightedPixelBytecodes[surfaceIndex].Get();
			}
			if (pass == PassType::Shadow && FrameCascades.Frame.CascadeCount != 0) pixelShader = CascadePixelBytecode.Get();
			if (UsesSurfacePCSS(pipeline, pass) && !forwardPlus)
			{
				if (!EnsurePCSSProgram(techniqueIndex, WeightedPass)) return nullptr;
				pixelShader = PCSSPixelBytecodes[WeightedPass ? PCSSProgramCount + techniqueIndex : techniqueIndex].Get();
			}
            if (vsmCaster) pixelShader = VSMPixelBytecodes[VSMCasterProgram].Get();
            else if (UsesSurfaceVSM(pipeline, pass) && !forwardPlus)
            {
                if (!EnsureVSMProgram(techniqueIndex, WeightedPass)) return nullptr;
                pixelShader = VSMPixelBytecodes[WeightedPass ? VSMProgramCount + techniqueIndex : techniqueIndex].Get();
            }

            if (pass == PassType::ContactDepth) pixelShader = ContactShadows.DepthPS();
            if (UsesContactReceiver(pipeline, pass) && !forwardPlus)
            {
                const auto filter = pipeline.SamplesShadowMap ? FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard;
                pixelShader = ContactShadows.Receiver(model, filter, false, ActiveSampleCount > 1, LastError);
                if (!pixelShader) return nullptr;
            }
            if (forwardPlus) pixelShader = (ClusteredActive ? ClusteredPixels : TiledActive ? TiledPixels : ForwardPlusPixels)[static_cast<std::size_t>(ForwardPlusFilter(pipeline, pass))][ForwardPlusVariant(pipeline, pass)].Get();
            if (UsesMaterialStage(pipeline,pass))
            {
                const auto material=SelectMaterialTechnique(pipeline,WeightedPass);
                if (!EnsureMaterialPixel(material,unsigned(model),WeightedPass,geometry)) return nullptr;
                pixelShader=MaterialPixels[MaterialProgramIndex(material,unsigned(model),WeightedPass,geometry)].Get();
            }
            if (UsesSurfacePCF(pipeline,pass))
            {
                if (!EnsurePCFProgram(unsigned(model),WeightedPass)) return nullptr;
                pixelShader=PCFPixelBytecodes[WeightedPass?8+techniqueIndex:techniqueIndex].Get();
            }
			if (FrameRayShadowImage && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap && !forwardPlus)
            {
                if (!EnsureRayShadowProgram(unsigned(model),WeightedPass)) return nullptr;
                pixelShader=RayShadowPixelBytecodes[WeightedPass?8+techniqueIndex:techniqueIndex].Get();
            }
            if (parallax) pixelShader=ParallaxPixel.Get();
            desc.PS = { pixelShader->GetBufferPointer(), pixelShader->GetBufferSize() };
			desc.BlendState.AlphaToCoverageEnable = FALSE;
			desc.BlendState.IndependentBlendEnable = FALSE;
			D3D12_RENDER_TARGET_BLEND_DESC& blend = desc.BlendState.RenderTarget[0];
			blend.BlendEnable = pass == PassType::Color && pipeline.BlendEnabled
				? TRUE : FALSE;
			blend.LogicOpEnable = FALSE;
			blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
			blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
			blend.BlendOp = D3D12_BLEND_OP_ADD;
			blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
			blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = pipeline.ColorWrite || vsmCaster ? D3D12_COLOR_WRITE_ENABLE_ALL : 0;
			desc.SampleMask = UINT_MAX;
			desc.RasterizerState.FillMode = pipeline.Fill == FillMode::Wireframe
				? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
			desc.RasterizerState.CullMode = pipeline.OutlineExtrusion ? D3D12_CULL_MODE_FRONT
				: (pipeline.BackFaceCulling ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE);
			desc.RasterizerState.FrontCounterClockwise = (RealtimeWorldIsMirrored(BoundWorld) != (pass == PassType::PlanarCapture)) ? FALSE : TRUE;
			desc.RasterizerState.DepthBias = pass == PassType::Shadow ? ShadowDepthBiasUnits : D3D12_DEFAULT_DEPTH_BIAS;
			desc.RasterizerState.DepthBiasClamp = pass == PassType::Shadow
				? ShadowDepthBiasClamp : D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
			desc.RasterizerState.SlopeScaledDepthBias = pass == PassType::Shadow
				? ShadowSlopeScaledDepthBias : D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
			desc.RasterizerState.DepthClipEnable = TRUE;
			desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
			desc.DepthStencilState.DepthEnable = pipeline.DepthTest ? TRUE : FALSE;
			desc.DepthStencilState.DepthWriteMask = pipeline.DepthWrite
				? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
			desc.DepthStencilState.DepthFunc = CurrentPass.DepthPeelingLayer ? D3D12_COMPARISON_FUNC_EQUAL : pipeline.DepthWrite
				? D3D12_COMPARISON_FUNC_LESS : D3D12_COMPARISON_FUNC_LESS_EQUAL;
			desc.DepthStencilState.StencilEnable = FALSE;
			desc.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			desc.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			desc.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			desc.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			desc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			desc.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			desc.DepthStencilState.BackFace = desc.DepthStencilState.FrontFace;
			desc.InputLayout = { inputElements, static_cast<UINT>(std::size(inputElements)) };
			desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			desc.NumRenderTargets = (pass == PassType::Shadow && !vsmCaster) || pass == PassType::ContactDepth || pass == PassType::ForwardPlusDepth ? 0u : (geometry ? static_cast<UINT>(GBufferAttachmentCount) : 1u);
			for (UINT attachment = 0; attachment < desc.NumRenderTargets; ++attachment)
				desc.RTVFormats[attachment] = ActiveColorFormat;
			desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
			desc.SampleDesc.Count = ActiveSampleCount;
			desc.RasterizerState.MultisampleEnable = ActiveSampleCount > 1;
			if (WeightedPass)
			{
				desc.NumRenderTargets = static_cast<UINT>(TransparencyFormats.size());
				desc.BlendState.IndependentBlendEnable = TRUE;
				for (UINT index = 0; index < desc.NumRenderTargets; ++index)
				{
					desc.RTVFormats[index] = TransparencyFormats[index];
					auto& target = desc.BlendState.RenderTarget[index];
					target = blend;
					target.SrcBlend = target.DestBlend = target.SrcBlendAlpha = target.DestBlendAlpha = D3D12_BLEND_ONE;
				}
				auto& revealage = desc.BlendState.RenderTarget[1];
				revealage.SrcBlend = revealage.SrcBlendAlpha = D3D12_BLEND_ZERO;
				revealage.DestBlend = D3D12_BLEND_INV_SRC_COLOR;
				revealage.RenderTargetWriteMask = pipeline.ColorWrite ? D3D12_COLOR_WRITE_ENABLE_RED : 0;
			}
			ComPtr<ID3D12PipelineState> nativePipeline;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,
				IID_PPV_ARGS(nativePipeline.GetAddressOf()));
			if (FAILED(result))
			{
				FailHRESULT("ID3D12Device::CreateGraphicsPipelineState", result);
				return nullptr;
			}
			ID3D12PipelineState* output = nativePipeline.Get();
			Pipelines.emplace(key, std::move(nativePipeline));
			return output;
		}

		bool CreateNativeDevice()
		{
			HRESULT result = CreateDXGIFactory2(0, IID_PPV_ARGS(Factory.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateDXGIFactory2", result);
			ComPtr<IDXGIFactory6> factory6;
			Factory.As(&factory6);
			for (UINT index = 0; ; ++index)
			{
				ComPtr<IDXGIAdapter1> adapter;
				result = factory6
					? factory6->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
						IID_PPV_ARGS(adapter.GetAddressOf()))
					: Factory->EnumAdapters1(index, adapter.GetAddressOf());
				if (result == DXGI_ERROR_NOT_FOUND) break;
				if (FAILED(result)) continue;
				DXGI_ADAPTER_DESC1 adapterDesc = {};
				if (FAILED(adapter->GetDesc1(&adapterDesc)) || (adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
					continue;
				result = CreateDeviceFunction(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
					IID_PPV_ARGS(Device.GetAddressOf()));
				if (SUCCEEDED(result))
				{
					Adapter = std::move(adapter);
					break;
				}
			}
			if (!Device) return Fail("No hardware adapter could create a DirectX 12 device");

			D3D12_COMMAND_QUEUE_DESC queueDesc = {};
			queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
			queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
			result = Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(CommandQueue.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateCommandQueue", result);
			result = Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
				IID_PPV_ARGS(CommandAllocator.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateCommandAllocator", result);
			result = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, CommandAllocator.Get(),
				nullptr, IID_PPV_ARGS(CommandList.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateCommandList", result);
			result = CommandList->Close();
			if (FAILED(result)) return FailHRESULT("ID3D12GraphicsCommandList::Close(initial)", result);
			result = Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(Fence.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateFence", result);
			FenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
			if (!FenceEvent) return Fail("CreateEventW failed for the DX12 fence");

			if (!CreateDescriptorArena(RTVArena, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, DescriptorCapacity, false)
				|| !CreateDescriptorArena(DSVArena, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, DescriptorCapacity, false)
				|| !CreateDescriptorArena(SRVArena, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ShaderDescriptorCapacity, true))
				return false;
			MainColorRTV = RTVArena.Allocate();
			MainDepthDSV = DSVArena.Allocate();
			NullShadowSRV = SRVArena.Allocate();
			NullColorSRV = SRVArena.Allocate();
			NullArraySRV = SRVArena.Allocate();
			if (MainColorRTV == InvalidDescriptor || MainDepthDSV == InvalidDescriptor
				|| NullShadowSRV == InvalidDescriptor
				|| NullColorSRV == InvalidDescriptor || NullArraySRV == InvalidDescriptor)
				return Fail("DX12 descriptor heaps could not reserve main target descriptors");
			D3D12_SHADER_RESOURCE_VIEW_DESC nullView = {};
			nullView.Format = DXGI_FORMAT_R32_FLOAT;
			nullView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			nullView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			nullView.Texture2D.MipLevels = 1;
			Device->CreateShaderResourceView(nullptr, &nullView, SRVArena.Cpu(NullShadowSRV));
			nullView.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			Device->CreateShaderResourceView(nullptr, &nullView, SRVArena.Cpu(NullColorSRV));
            nullView.Format=DXGI_FORMAT_R32_FLOAT; nullView.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            nullView.Texture2DArray={0,1,0,1,0}; Device->CreateShaderResourceView(nullptr,&nullView,SRVArena.Cpu(NullArraySRV));

			return CreatePipelineObjects()
				&& CreateMainTargets(CreateInfo.Width, CreateInfo.Height);
		}

		bool SignalAndWait()
		{
			const UINT64 value = ++FenceValue;
			HRESULT result = CommandQueue->Signal(Fence.Get(), value);
			if (FAILED(result)) return FailHRESULT("ID3D12CommandQueue::Signal", result);
            if (ForwardPlusRecordOpen && ForwardPlusListsRecorded && FrameWasSubmitted)
            {
                if (!(ClusteredActive ? ClusteredForward.MarkSubmitted(Fence.Get(), value, LastError) : TileProducer().MarkSubmitted(Fence.Get(), value, LastError))) return false;
                ForwardPlusRecordOpen = false;
            }
			const ULONGLONG deadline = GetTickCount64() + GpuWaitTimeoutMilliseconds;
			while (Fence->GetCompletedValue() < value)
			{
				result = Fence->SetEventOnCompletion(value, FenceEvent);
				if (FAILED(result)) return FailHRESULT("ID3D12Fence::SetEventOnCompletion", result);
				const ULONGLONG now = GetTickCount64();
				const DWORD remaining = now < deadline ? static_cast<DWORD>(deadline - now) : 0;
				const DWORD waitResult =
					WaitForSingleObject(FenceEvent, remaining);
				if (waitResult != WAIT_OBJECT_0)
				{
					const HRESULT removedReason = Device->GetDeviceRemovedReason();
					if (FAILED(removedReason))
						return FailHRESULT("DX12 fence device removal", removedReason);
					return Fail(waitResult == WAIT_TIMEOUT
						? "Waiting for the DX12 renderer fence timed out"
						: "Waiting for the DX12 renderer fence failed");
				}
			}
			if (Fence->GetCompletedValue() == UINT64_MAX)
				return FailHRESULT("DX12 fence device removal", Device->GetDeviceRemovedReason());
			SubmittedWorkPending = false;
			return true;
		}

		bool RecoverSubmittedWork(const char* operation)
		{
			if (!SubmittedWorkPending) return true;
			if (!SignalAndWait())
				return Fail(std::string(operation) + " blocked until submitted DX12 work completes: " + LastError);
			FrameUploads.clear();
            if (!FrameOpen) { LumenFrameProbeWork.clear(); LumenFrameRadianceWork.clear(); NaniteFrameWork.clear(); NaniteFrameReadback.reset(); }
			return true;
		}

		bool WaitForGPU()
		{
			return !CommandQueue || !Fence || SignalAndWait();
		}

		bool UploadDirtyResources()
		{
			for (auto& pair : Buffers)
			{
				BufferResource& resource = pair.second;
				if (!resource.Dirty) continue;
				ComPtr<ID3D12Resource> upload;
				if (!CreateBufferResource(resource.Bytes.size(), D3D12_HEAP_TYPE_UPLOAD,
					D3D12_RESOURCE_STATE_GENERIC_READ, upload)) return false;
				void* mapped = nullptr;
				D3D12_RANGE noRead = { 0, 0 };
				HRESULT result = upload->Map(0, &noRead, &mapped);
				if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(buffer upload)", result);
				std::memcpy(mapped, resource.Bytes.data(), resource.Bytes.size());
				upload->Unmap(0, nullptr);
				Transition(resource.Resource.Get(), resource.State, D3D12_RESOURCE_STATE_COPY_DEST);
				CommandList->CopyBufferRegion(resource.Resource.Get(), 0, upload.Get(), 0, resource.Bytes.size());
				Transition(resource.Resource.Get(), resource.State, FinalBufferState(resource.Description.Usage));
				FrameUploads.push_back(std::move(upload));
				resource.Dirty = false;
			}

			for (auto& pair : Textures)
			{
				TextureResource& resource = pair.second;
				if (!resource.Dirty) continue;
				const auto layout = resource.Resource->GetDesc();
				std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(layout.MipLevels);
				UINT64 bytes = 0;
				Device->GetCopyableFootprints(&layout, 0, layout.MipLevels, 0, footprints.data(), nullptr, nullptr, &bytes);
				ComPtr<ID3D12Resource> upload;
				if (!CreateBufferResource(static_cast<std::size_t>(bytes),
					D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload)) return false;
				void* mapped = nullptr;
				D3D12_RANGE noRead = { 0, 0 };
				HRESULT result = upload->Map(0, &noRead, &mapped);
				if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(texture upload)", result);
				VisitTextureMips(resource.Description, [&](int level, int width, int height, std::size_t offset) {
					const auto& footprint = footprints[level];
					const std::size_t rowSize = std::size_t(width) * 4;
					for (int row = 0; row < height; ++row)
						std::memcpy(static_cast<std::uint8_t*>(mapped) + footprint.Offset +
							std::size_t(row) * footprint.Footprint.RowPitch,
							resource.Bytes.data() + offset + std::size_t(row) * rowSize, rowSize);
				});
				upload->Unmap(0, nullptr);
				Transition(resource.Resource.Get(), resource.State, D3D12_RESOURCE_STATE_COPY_DEST);
				D3D12_TEXTURE_COPY_LOCATION destination = {};
				destination.pResource = resource.Resource.Get();
				destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				D3D12_TEXTURE_COPY_LOCATION source = {};
				source.pResource = upload.Get();
				source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
				for (UINT level = 0; level < layout.MipLevels; ++level)
				{
					destination.SubresourceIndex = level;
					source.PlacedFootprint = footprints[level];
					CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
				}
				Transition(resource.Resource.Get(), resource.State, FinalTextureState(resource.Description));
				FrameUploads.push_back(std::move(upload));
				resource.Dirty = false;
			}
			return true;
		}

		void CopyToReadback(ID3D12Resource* source, ReadbackResource& destination)
		{
			D3D12_TEXTURE_COPY_LOCATION sourceLocation = {};
			sourceLocation.pResource = source;
			sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			D3D12_TEXTURE_COPY_LOCATION destinationLocation = {};
			destinationLocation.pResource = destination.Buffer.Get();
			destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			destinationLocation.PlacedFootprint = destination.Footprint;
			CommandList->CopyTextureRegion(&destinationLocation, 0, 0, 0, &sourceLocation, nullptr);
		}

		bool CreateHdrSurface(HdrSurface& surface, int width, int height,
			DXGI_FORMAT format = DXGI_FORMAT_R16G16B16A16_FLOAT, UINT samples = 1, const float* clearColor = nullptr)
		{
			if (surface.Resource && surface.Width == width && surface.Height == height && surface.Samples == samples && surface.Format == format) return true;
			if (surface.RTV == InvalidDescriptor) surface.RTV = RTVArena.Allocate();
			if (surface.SRV == InvalidDescriptor) surface.SRV = SRVArena.Allocate();
			if (surface.RTV == InvalidDescriptor || surface.SRV == InvalidDescriptor)
				return Fail("DX12 HDR descriptor allocation failed");
			D3D12_RESOURCE_DESC description {};
			description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			description.Width = static_cast<UINT64>(width);
			description.Height = static_cast<UINT>(height);
			description.DepthOrArraySize = description.MipLevels = 1;
			description.Format = format;
			description.SampleDesc.Count = samples;
			description.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
			const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
			D3D12_CLEAR_VALUE clear {};
			clear.Format = description.Format;
			clear.Color[3] = 1.0f;
			if (clearColor) std::memcpy(clear.Color, clearColor, sizeof(clear.Color));
			ComPtr<ID3D12Resource> resource;
			const HRESULT result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
				D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(resource.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateCommittedResource(HDR)", result);
			D3D12_RENDER_TARGET_VIEW_DESC target {};
			target.Format = description.Format;
			target.ViewDimension = samples > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMS : D3D12_RTV_DIMENSION_TEXTURE2D;
			Device->CreateRenderTargetView(resource.Get(), &target, RTVArena.Cpu(surface.RTV));
			D3D12_SHADER_RESOURCE_VIEW_DESC view {};
			view.Format = description.Format;
			view.ViewDimension = samples > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
			view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			if (samples == 1) view.Texture2D.MipLevels = 1;
			Device->CreateShaderResourceView(resource.Get(), &view, SRVArena.Cpu(surface.SRV));
			surface.Resource = std::move(resource);
			surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			surface.Width = width;
			surface.Height = height;
			surface.Samples = samples; surface.Format = format;
			return true;
		}

		bool EnsureDeferredResources()
		{
			if (!HdrSceneActive || !MaterialCaptureActive) return Fail("DX12 material capture requires HDR and an active consumer");
			for (auto& surface : GBufferSurfaces)
				if (!CreateHdrSurface(surface, CreateInfo.Width, CreateInfo.Height)) return false;
			if (DeferredLightingPipelines.count(DeferredPipelineKey())) return true;
			// Mesh geometry uses the existing b0/material root signature. Lighting has
			// its own b1 CBV and seven independent image tables, so descriptor arena
			// allocation need not be contiguous and resizing can reuse each slot.
			if (!DeferredRootSignature)
			{
				constexpr UINT LightingImageCount = DeferredBaseTextureCount + MaximumShadowCascades;
				std::array<D3D12_DESCRIPTOR_RANGE, LightingImageCount> images {};
				std::array<D3D12_ROOT_PARAMETER, DeferredRayShadowImageRoot + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[0].Descriptor.ShaderRegister = 1;
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < LightingImageCount; ++index)
				{
					images[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					images[index].NumDescriptors = 1;
					images[index].BaseShaderRegister = index == 0 ? 0 : index + 3; // t0 shadow, t4..t9 material/depth, t10..t13 cascades
					images[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index + 1].DescriptorTable.NumDescriptorRanges = 1;
					parameters[index + 1].DescriptorTable.pDescriptorRanges = &images[index];
					parameters[index + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				parameters[DeferredCascadeConstantsRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[DeferredCascadeConstantsRoot].Descriptor.ShaderRegister = 2;
				parameters[DeferredCascadeConstantsRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				parameters[DeferredShadowFilterRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[DeferredShadowFilterRoot].Descriptor.ShaderRegister = ShadowFilterConstantRegister;
				parameters[DeferredShadowFilterRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				// One control word and one image table keep contact independent of the primary filter.
				parameters[DeferredContactControlRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
				parameters[DeferredContactControlRoot].Constants = {ContactControlRegister, 0, 1};
				parameters[DeferredContactControlRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_DESCRIPTOR_RANGE contactRange {};
				contactRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; contactRange.NumDescriptors = 1;
				contactRange.BaseShaderRegister = ContactImageRegister;
				contactRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
				parameters[DeferredContactImageRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
				parameters[DeferredContactImageRoot].DescriptorTable = {1, &contactRange};
				parameters[DeferredContactImageRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                parameters[DeferredSurfaceRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                parameters[DeferredSurfaceRoot].Descriptor.ShaderRegister = 0;
                parameters[DeferredSurfaceRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                parameters[DeferredListRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                parameters[DeferredListRoot].Descriptor.ShaderRegister = ForwardPlusControlRegister;
                parameters[DeferredListRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                std::array<D3D12_DESCRIPTOR_RANGE, ForwardPlusStorageCount> storage {};
                for (UINT index = 0; index < ForwardPlusStorageCount; ++index)
                {
                    storage[index] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, ForwardPlusFirstRegister + index, 0,
                        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                    auto& parameter = parameters[DeferredStorageFirstRoot + index];
                    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    parameter.DescriptorTable = {1, &storage[index]};
                    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                }
                D3D12_DESCRIPTOR_RANGE rayRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,RayShadowImageRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                parameters[DeferredRayShadowImageRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[DeferredRayShadowImageRoot].DescriptorTable={1,&rayRange};
                parameters[DeferredRayShadowImageRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size());
				signature.pParameters = parameters.data();
				ComPtr<ID3DBlob> serialized;
				ComPtr<ID3DBlob> errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(deferred)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(DeferredRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(deferred)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "Deferred.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> geometryVertex, geometryPixel, lightingVertex, lightingPixel;
			const D3D_SHADER_MACRO multisample[] = {{"EVANS_MULTISAMPLE", "1"}, {nullptr, nullptr}};
			if (!CompileShader("VSMain", "vs_5_0", geometryVertex)
				|| !CompileShader("PSMain", "ps_5_0", geometryPixel)
				|| !CompileShader("VSLighting", "vs_5_0", lightingVertex)
				|| !CompileShader("PSLighting", "ps_5_0", lightingPixel, FrameSampleCount > 1 ? multisample : nullptr)) return false;
			if (!TiledActive && UsesPCSS())
			{
				const auto program = FrameSampleCount > 1 ? PCSSMultisampleProgram : PCSSDeferredProgram;
				if (!EnsurePCSSProgram(program)) return false;
				lightingPixel = PCSSPixelBytecodes[program];
			}
			if (!TiledActive && UsesVSM())
			{
				const auto program = FrameSampleCount > 1 ? VSMDeferredMultisampleProgram : VSMDeferredProgram;
				if (!EnsureVSMProgram(program)) return false;
				lightingPixel = VSMPixelBytecodes[program];
			}
            if (!TiledActive && FrameContact.Enabled)
            {
                auto* receiver = ContactShadows.Receiver(Techniques::ShadingModel::Phong, FrameShadowFilter.Mode,
                    true, FrameSampleCount > 1, LastError);
                if (!receiver) return false;
                lightingPixel = receiver;
            }
            if (TiledActive)
            {
                if (!LoadTechniqueShaderSource("DX12", "TiledDeferred.hlsl", ShaderCode, LastError)) return false;
                const auto filter = std::to_string(static_cast<std::uint32_t>(FrameShadowFilter.Mode));
                const D3D_SHADER_MACRO definitions[] {{"EVANS_TILED_DEFERRED_SURFACE", "1"}, {"EVANS_TILED_DEFERRED_LIGHTING", "1"},
                    {"EVANS_TILED_DEFERRED_FILTER", filter.c_str()}, {FrameSampleCount > 1 ? "EVANS_MULTISAMPLE" : nullptr, "1"}, {nullptr, nullptr}};
                if (!CompileShader("VSLighting", "vs_5_0", lightingVertex, definitions) ||
                    !CompileShader("PSLighting", "ps_5_0", lightingPixel, definitions)) return false;
            }
            if (UsesDedicatedPCF())
            {
                const unsigned program=FrameSampleCount>1?6:5;
                if (!EnsurePCFProgram(program)) return false;
                lightingPixel=PCFPixelBytecodes[program];
            }
            if (FrameRayShadowImage && !TiledActive)
            {
                const unsigned program=FrameSampleCount>1?6:5;
                if (!EnsureRayShadowProgram(program)) return false;
                lightingPixel=RayShadowPixelBytecodes[program];
            }
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = DeferredRootSignature.Get();
			pipeline.VS = {lightingVertex->GetBufferPointer(), lightingVertex->GetBufferSize()};
			pipeline.PS = {lightingPixel->GetBufferPointer(), lightingPixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1;
			pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.SampleDesc.Count = FrameSampleCount;
			pipeline.RasterizerState.MultisampleEnable = FrameSampleCount > 1;
			ComPtr<ID3D12PipelineState> native;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(native.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(deferred lighting)", result);
			DeferredLightingPipelines.emplace(DeferredPipelineKey(), std::move(native));
			DeferredVertexBytecode = std::move(geometryVertex);
			DeferredPixelBytecode = std::move(geometryPixel);
			return true;
		}

		UINT SupportedSampleCounts() const
		{
			UINT counts = 1;
			if (!Device) return counts;
			for (auto format : {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT})
			{
				D3D12_FEATURE_DATA_FORMAT_SUPPORT support {format};
				if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
					!(support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD)) return counts;
			}
			for (auto format : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT})
			{
				D3D12_FEATURE_DATA_FORMAT_SUPPORT support {format};
				constexpr UINT Required = D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
				if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
					(support.Support1 & Required) != Required) return counts;
			}
			for (UINT samples : {2u, 4u, 8u})
			{
				bool supported = true;
				for (auto format : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_D32_FLOAT})
				{
					D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality {format, samples, D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 0};
					supported &= SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))) &&
						quality.NumQualityLevels > 0;
				}
				if (supported) counts |= samples;
			}
			return counts;
		}

		bool EnsureMultisamplePrograms()
		{
			if (!MultisampleRootSignature)
			{
				// Independent tables keep every descriptor stable throughout
				// recording. Resolve stages bind only their statically used inputs.
				constexpr UINT TextureCount = 8;
				std::array<D3D12_DESCRIPTOR_RANGE, TextureCount> ranges {};
				std::array<D3D12_ROOT_PARAMETER, TextureCount> parameters {};
				for (UINT index = 0; index < TextureCount; ++index)
				{
					ranges[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[index].NumDescriptors = 1;
					ranges[index].BaseShaderRegister = index;
					parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index].DescriptorTable = {1, &ranges[index]};
					parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_ROOT_SIGNATURE_DESC desc {};
				desc.NumParameters = TextureCount; desc.pParameters = parameters.data();
				ComPtr<ID3DBlob> serialized, errors;
				auto result = SerializeRootSignatureFunction(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result)) return FailHRESULT("SerializeRootSignature(MSAA)", result);
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(MultisampleRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(MSAA)", result);
			}
			if (MultisampleVertex) return true;
			if (!LoadTechniqueShaderSource("DX12", "MSAA.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			std::array<ComPtr<ID3DBlob>, static_cast<std::size_t>(SampleProgram::Count)> pixels;
			const char* entries[] = {"PSDepth", "PSMaterial", "PSVelocity", "PSRestoreColor"};
			if (!CompileShader("VSMain", "vs_5_0", vertex)) return false;
			for (std::size_t index = 0; index < pixels.size(); ++index)
				if (!CompileShader(entries[index], "ps_5_0", pixels[index])) return false;
			MultisampleVertex = std::move(vertex); MultisamplePixels = std::move(pixels);
			return true;
		}

		ID3D12PipelineState* GetMultisamplePipeline(SampleProgram program)
		{
			const bool restore = program == SampleProgram::Restore;
			const bool depth = program == SampleProgram::Depth || program == SampleProgram::Material;
			const auto format = restore ? MultisampleSurfaces[MultisampleColor].Format : DXGI_FORMAT_R16G16B16A16_FLOAT;
			const UINT samples = restore ? FrameSampleCount : 1;
			const UINT key = static_cast<UINT>(program) | (samples << 4) | (format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 1u << 8 : 0);
			const auto found = MultisamplePipelines.find(key);
			if (found != MultisamplePipelines.end()) return found->second.Get();
			D3D12_GRAPHICS_PIPELINE_STATE_DESC desc {};
			desc.pRootSignature = MultisampleRootSignature.Get();
			desc.VS = {MultisampleVertex->GetBufferPointer(), MultisampleVertex->GetBufferSize()};
			const auto& pixel = MultisamplePixels[static_cast<UINT>(program)];
			desc.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = desc.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			desc.RasterizerState.DepthClipEnable = TRUE;
			desc.RasterizerState.MultisampleEnable = samples > 1;
			desc.DepthStencilState.DepthEnable = depth;
			desc.DepthStencilState.DepthWriteMask = depth ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
			desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			auto& stencil = desc.DepthStencilState.FrontFace;
			stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			desc.DepthStencilState.BackFace = stencil;
			desc.SampleMask = UINT_MAX; desc.SampleDesc.Count = samples;
			desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			desc.NumRenderTargets = program == SampleProgram::Material ? GBufferAttachmentCount : (program == SampleProgram::Depth ? 0 : 1);
			for (UINT index = 0; index < desc.NumRenderTargets; ++index) desc.RTVFormats[index] = format;
			desc.DSVFormat = depth ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
			ComPtr<ID3D12PipelineState> pipeline;
			const auto result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc, IID_PPV_ARGS(pipeline.GetAddressOf()));
			if (FAILED(result)) { FailHRESULT("CreateGraphicsPipelineState(MSAA)", result); return nullptr; }
			const auto native = pipeline.Get();
			MultisamplePipelines.emplace(key, std::move(pipeline));
			return native;
		}

		bool EnsureMultisampleResources()
		{
			const auto format = HdrSceneActive ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			if (!CreateHdrSurface(MultisampleSurfaces[MultisampleColor], CreateInfo.Width, CreateInfo.Height, format, FrameSampleCount)) return false;
			if (MaterialCaptureActive) for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
				if (!CreateHdrSurface(MultisampleSurfaces[MultisampleMaterialFirst + index], CreateInfo.Width,
					CreateInfo.Height, DXGI_FORMAT_R16G16B16A16_FLOAT, FrameSampleCount)) return false;
			if (((Techniques::RequiresMotionVectors(FramePostProcessing) || TSRScene.Active) || EarlyTemporalActive) && !CreateHdrSurface(MultisampleSurfaces[MultisampleVelocity],
				CreateInfo.Width, CreateInfo.Height, DXGI_FORMAT_R16G16B16A16_FLOAT, FrameSampleCount)) return false;
			auto& depth = MultisampleDepth;
			if (!depth.Resource || depth.Width != CreateInfo.Width || depth.Height != CreateInfo.Height || depth.Samples != FrameSampleCount)
			{
				if (depth.SRV == InvalidDescriptor) depth.SRV = SRVArena.Allocate();
				if (MultisampleDepthDSV == InvalidDescriptor) MultisampleDepthDSV = DSVArena.Allocate();
				if (MultisampleDepthReadDSV == InvalidDescriptor) MultisampleDepthReadDSV = DSVArena.Allocate();
				if (depth.SRV == InvalidDescriptor || MultisampleDepthDSV == InvalidDescriptor || MultisampleDepthReadDSV == InvalidDescriptor)
					return Fail("DX12 MSAA depth descriptors exhausted");
				D3D12_RESOURCE_DESC desc {};
				desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
				desc.Width = CreateInfo.Width; desc.Height = CreateInfo.Height;
				desc.DepthOrArraySize = desc.MipLevels = 1;
				desc.Format = DXGI_FORMAT_R32_TYPELESS; desc.SampleDesc.Count = FrameSampleCount;
				desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
				D3D12_CLEAR_VALUE clear {}; clear.Format = DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth = 1;
				const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
				ComPtr<ID3D12Resource> resource;
				const auto result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
					D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(resource.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateCommittedResource(MSAA depth)", result);
				D3D12_DEPTH_STENCIL_VIEW_DESC dsv {};
				dsv.Format = DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
				Device->CreateDepthStencilView(resource.Get(), &dsv, DSVArena.Cpu(MultisampleDepthDSV));
				dsv.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
				Device->CreateDepthStencilView(resource.Get(), &dsv, DSVArena.Cpu(MultisampleDepthReadDSV));
				D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
				srv.Format = DXGI_FORMAT_R32_FLOAT; srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
				srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
				Device->CreateShaderResourceView(resource.Get(), &srv, SRVArena.Cpu(depth.SRV));
				depth.Resource = std::move(resource); depth.Samples = FrameSampleCount;
				depth.Width = CreateInfo.Width; depth.Height = CreateInfo.Height;
				depth.State = depth.FrameStartState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
			}
			if (!EnsureMultisamplePrograms()) return false;
			return GetMultisamplePipeline(MaterialCaptureActive ? SampleProgram::Material : SampleProgram::Depth) &&
				(!((Techniques::RequiresMotionVectors(FramePostProcessing) || TSRScene.Active) || EarlyTemporalActive) || GetMultisamplePipeline(SampleProgram::Velocity)) &&
				(!NeedsSampleColorRestore() || GetMultisamplePipeline(SampleProgram::Restore));
		}

		bool NeedsSampleColorRestore() const
		{
			return SubsurfaceRequested || SubsurfaceResolved || GIActive || AmbientOcclusionActive || ReflectionActive || Techniques::UsesVolumetricFog(FrameFog) || Techniques::UsesLightShafts(FrameLightShafts) ||
				((FramePostProcessing.MotionBlur.Enabled || Techniques::UsesTemporalAA(FramePostProcessing)) || EarlyTemporalActive) || FramePostProcessing.DepthOfField.Enabled;
		}
		bool OpaqueEffectsComplete() const
		{
			return SubsurfaceComplete() && ((!GIActive || GI.Resolved) && (!AmbientOcclusionActive || AmbientOcclusionApplied) && (!ReflectionActive || ReflectionApplied) &&
				(!Techniques::UsesVolumetricFog(FrameFog) || FogResolved) &&
				(!Techniques::UsesLightShafts(FrameLightShafts) || LightShaftResolved) &&
				(!FramePostProcessing.MotionBlur.Enabled || MotionBlurResolved) &&
				(!Techniques::UsesTemporalAA(FramePostProcessing) || TemporalResolved) &&
				(!FramePostProcessing.DepthOfField.Enabled || DepthOfFieldResolved));
		}

		bool SupportsFloatDepthTargets(bool requireMipChain) const
		{
			if (!Device) return false;
			D3D12_FEATURE_DATA_FORMAT_SUPPORT format {DXGI_FORMAT_R32_FLOAT};
			const UINT required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE
				| D3D12_FORMAT_SUPPORT1_RENDER_TARGET | (requireMipChain ? D3D12_FORMAT_SUPPORT1_MIP : 0);
			return SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format)))
				&& (format.Support1 & required) == required;
		}

		void DestroySaoDepthHierarchy()
		{
			SRVArena.Release(SaoDepthHierarchy.SRV);
			RTVArena.Release(SaoDepthScratch.RTV);
			SaoDepthHierarchy = {};
			SaoDepthScratch = {};
			SaoDepthMipCount = 0;
		}

		bool EnsureSaoDepthHierarchy()
		{
			if (!SupportsFloatDepthTargets(true)) return Fail("DX12 SAO requires sampled and renderable R32_FLOAT textures");
			if (SaoDepthHierarchy.Resource && SaoDepthHierarchy.Width == CreateInfo.Width && SaoDepthHierarchy.Height == CreateInfo.Height)
				return true;
			const UINT mipCount = AmbientOcclusionDepthMipCount(CreateInfo.Width, CreateInfo.Height);
			D3D12_RESOURCE_DESC description {};
			description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			description.Width = static_cast<UINT64>(CreateInfo.Width);
			description.Height = static_cast<UINT>(CreateInfo.Height);
			description.DepthOrArraySize = 1;
			description.SampleDesc.Count = 1;
			description.MipLevels = static_cast<UINT16>(mipCount);
			description.Format = DXGI_FORMAT_R32_FLOAT;
			const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
			ComPtr<ID3D12Resource> hierarchy, scratch;
			HRESULT result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(hierarchy.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateCommittedResource(SAO hierarchy)", result);
			description.MipLevels = 1;
			description.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
			result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
				D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(scratch.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateCommittedResource(SAO scratch)", result);
			const UINT shaderView = SRVArena.Allocate();
			const UINT targetView = RTVArena.Allocate();
			if (shaderView == InvalidDescriptor || targetView == InvalidDescriptor)
			{
				SRVArena.Release(shaderView);
				RTVArena.Release(targetView);
				return Fail("DX12 SAO descriptor allocation failed");
			}
			D3D12_SHADER_RESOURCE_VIEW_DESC view {};
			view.Format = DXGI_FORMAT_R32_FLOAT;
			view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			view.Texture2D.MipLevels = mipCount;
			Device->CreateShaderResourceView(hierarchy.Get(), &view, SRVArena.Cpu(shaderView));
			D3D12_RENDER_TARGET_VIEW_DESC target {};
			target.Format = DXGI_FORMAT_R32_FLOAT;
			target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
			Device->CreateRenderTargetView(scratch.Get(), &target, RTVArena.Cpu(targetView));
			DestroySaoDepthHierarchy();
			SaoDepthHierarchy.Resource = std::move(hierarchy);
			SaoDepthHierarchy.SRV = shaderView;
			SaoDepthHierarchy.State = SaoDepthHierarchy.FrameStartState = D3D12_RESOURCE_STATE_COPY_DEST;
			SaoDepthHierarchy.Width = SaoDepthScratch.Width = CreateInfo.Width;
			SaoDepthHierarchy.Height = SaoDepthScratch.Height = CreateInfo.Height;
			SaoDepthScratch.Resource = std::move(scratch);
			SaoDepthScratch.RTV = targetView;
			SaoDepthScratch.State = SaoDepthScratch.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			SaoDepthMipCount = mipCount;
			return true;
		}

#include "Renderer/DX12/DX12ShaderCompiler.inl"
		#include "Renderer/DX12/DX12RayScene.inl"
		#include "Renderer/DX12/DX12RayRadiance.inl"
    #include "Renderer/DX12/DX12RayReflections.inl"
    #include "Renderer/DX12/DX12HybridRendering.inl"
    #include "Renderer/DX12/DX12SubsurfaceScattering.inl"
    #include "Renderer/DX12/DX12HairFur.inl"
    #include "Renderer/DX12/DX12TemporalEffects.inl"
    #include "Renderer/DX12/DX12ParallaxOcclusionMapping.inl"
    #include "Renderer/DX12/DX12DisplacementMapping.inl"
    #include "Renderer/DX12/DX12Tessellation.inl"
    #include "Renderer/VirtualGeometry/DX12/DX12VirtualGeometryCache.inl"
    #include "Renderer/VirtualGeometry/DX12/DX12VirtualGeometry.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteCache.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteGeometry.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteAsync.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteCacheAsync.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteFrameVisibility.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteDraws.inl"
#include "Renderer/DX12/DX12TemporalSuperResolution.inl"
#include "Renderer/VirtualShadowMaps/DX12/DX12VirtualShadowMaps.inl"
#include "Renderer/VirtualShadowMaps/DX12/DX12VirtualShadowAtlas.inl"
#include "Renderer/Lumen/DX12/DX12LumenSurface.inl"
#include "Renderer/Lumen/DX12/DX12LumenCaptureAsync.inl"
#include "Renderer/Lumen/DX12/DX12LumenSurfaceLookup.inl"
#include "Renderer/Lumen/DX12/DX12LumenDistanceTrace.inl"
#include "Renderer/Lumen/DX12/DX12LumenGlobalDistance.inl"
#include "Renderer/Lumen/DX12/DX12LumenGlobalDistanceAsync.inl"
#include "Renderer/Lumen/DX12/DX12LumenSurfaceLighting.inl"
#include "Renderer/Lumen/DX12/DX12LumenLightingAsync.inl"
#include "Renderer/Lumen/DX12/DX12LumenRadianceTrace.inl"
#include "Renderer/Lumen/DX12/DX12LumenProbeIntegration.inl"
#include "Renderer/Lumen/DX12/DX12LumenProbeGather.inl"
#include "Renderer/Lumen/DX12/DX12LumenTranslucency.inl"
#include "Renderer/Lumen/DX12/DX12LumenHairReceiver.inl"
#include "Renderer/DX12/DX12RayTracedShadows.inl"
#include "Renderer/DX12/DX12PCF.inl"
#include "Renderer/DX12/DX12Materials.inl"

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

#include "Renderer/Lumen/DX12/DX12LumenTemporal.inl"
#include "Renderer/Lumen/DX12/DX12LumenReflections.inl"
#include "Renderer/VirtualShadowMaps/DX12/DX12VirtualShadowReceivers.inl"
#include "Renderer/VirtualShadowMaps/DX12/DX12VirtualShadowLookup.inl"
    #include "Renderer/DX12/DX12RayGI.inl"
    #include "Renderer/DX12/DX12VoxelGI.inl"
    #include "Renderer/DX12/DX12PathTracing.inl"
    #include "Renderer/DX12/DX12GPUDriven.inl"
    #include "Renderer/DX12/DX12VisibilityBuffer.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteMaterials.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteFrameRaster.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteVisibilityGeometry.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteInstances.inl"
    #include "Renderer/Nanite/DX12/DX12NaniteHardware.inl"
    #include "Renderer/MeshShaders/DX12/DX12MeshShaders.inl"
    #include "Renderer/Profiling/DX12/DX12FrameTiming.inl"

		bool EnsureAmbientOcclusionResources()
		{
            const bool rayTraced = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::RayTraced;
            if (rayTraced && !EnsureRayScene()) return false;
			for (auto& surface : AmbientOcclusionSurfaces)
				if (!CreateHdrSurface(surface, CreateInfo.Width, CreateInfo.Height)) return false;
			const bool scalable = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO;
			if (scalable && !EnsureSaoDepthHierarchy()) return false;
			const std::size_t technique = AmbientOcclusionProgramIndex();
			if (AmbientOcclusionPipelines[technique][0] && AmbientOcclusionPipelines[technique][1]
				&& (!scalable || AmbientOcclusionPipelines[technique][HierarchyPipelineIndex])) return true;
			if (!AmbientOcclusionRootSignature)
			{
				constexpr UINT ImageCount = 4 + static_cast<UINT>(RaySceneBufferCount);
				std::array<D3D12_DESCRIPTOR_RANGE, ImageCount> images {};
				std::array<D3D12_ROOT_PARAMETER, ImageCount + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[0].Descriptor.ShaderRegister = 0;
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < ImageCount; ++index)
				{
					images[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					images[index].NumDescriptors = 1;
					images[index].BaseShaderRegister = index;
					images[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index + 1].DescriptorTable.NumDescriptorRanges = 1;
					parameters[index + 1].DescriptorTable.pDescriptorRanges = &images[index];
					parameters[index + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size());
				signature.pParameters = parameters.data();
				ComPtr<ID3DBlob> serialized, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(ambient occlusion)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(AmbientOcclusionRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(ambient occlusion)", result);
			}
			static constexpr std::array<const char*, AmbientOcclusionTechniqueCount> filenames {
				"SSAO.hlsl", "HBAO.hlsl", "SAO.hlsl", "GTAO.hlsl", "RayTracedAO.hlsl"};
			if (!LoadTechniqueShaderSource("DX12", filenames[technique], ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex, nullptr, rayTraced) || !CompileShader("PSMain", "ps_5_0", pixel, nullptr, rayTraced)) return false;
			std::array<ComPtr<ID3D12PipelineState>, 3> pipelines;
			const std::size_t pipelineCount = scalable ? pipelines.size() : HierarchyPipelineIndex;
			for (std::size_t index = 0; index < pipelineCount; ++index)
			{
				const bool composite = index == 1;
				D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
				pipeline.pRootSignature = AmbientOcclusionRootSignature.Get();
				pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				auto& blend = pipeline.BlendState.RenderTarget[0];
				blend.BlendEnable = composite;
				blend.SrcBlend = D3D12_BLEND_ONE;
				blend.DestBlend = composite ? D3D12_BLEND_ONE : D3D12_BLEND_ZERO;
				blend.BlendOp = composite ? D3D12_BLEND_OP_REV_SUBTRACT : D3D12_BLEND_OP_ADD;
				blend.SrcBlendAlpha = D3D12_BLEND_ZERO;
				blend.DestBlendAlpha = D3D12_BLEND_ONE;
				blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
				blend.LogicOp = D3D12_LOGIC_OP_NOOP;
				blend.RenderTargetWriteMask = composite
					? D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_BLUE
					: D3D12_COLOR_WRITE_ENABLE_ALL;
				pipeline.SampleMask = UINT_MAX;
				pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
				pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
				pipeline.RasterizerState.DepthClipEnable = TRUE;
				pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
				pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
				pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
				pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
				pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
				pipeline.NumRenderTargets = 1;
				pipeline.RTVFormats[0] = index == HierarchyPipelineIndex ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
				pipeline.SampleDesc.Count = 1;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(pipelines[index].GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(ambient occlusion)", result);
			}
			AmbientOcclusionPipelines[technique] = std::move(pipelines);
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

		static std::size_t HdrProgramIndex(Techniques::HdrPassKind pass)
		{
			return HdrShaderIndex(pass);
		}

		bool EnsureHdrRootSignature()
		{
			if (HdrRootSignature) return true;
			D3D12_DESCRIPTOR_RANGE images[2] {};
			D3D12_ROOT_PARAMETER parameters[3] {};
			parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			parameters[0].Constants.ShaderRegister = 0;
			parameters[0].Constants.Num32BitValues = sizeof(HdrConstants) / sizeof(std::uint32_t);
			parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			for (UINT index = 0; index < std::size(images); ++index)
			{
				images[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
				images[index].NumDescriptors = 1;
				images[index].BaseShaderRegister = index;
				images[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
				parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
				parameters[index + 1].DescriptorTable.NumDescriptorRanges = 1;
				parameters[index + 1].DescriptorTable.pDescriptorRanges = &images[index];
				parameters[index + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			}
			D3D12_STATIC_SAMPLER_DESC sampler {};
			sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
			sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
			sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			sampler.MaxLOD = D3D12_FLOAT32_MAX;
			sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_ROOT_SIGNATURE_DESC signature {};
			signature.NumParameters = static_cast<UINT>(std::size(parameters));
			signature.pParameters = parameters;
			signature.NumStaticSamplers = 1;
			signature.pStaticSamplers = &sampler;
			signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
			ComPtr<ID3DBlob> serialized;
			ComPtr<ID3DBlob> errors;
			HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				else FailHRESULT("SerializeRootSignature(HDR)", result);
				return false;
			}
			result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(HdrRootSignature.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateRootSignature(HDR)", result);
		}

		bool EnsureHdrPipeline(Techniques::HdrPassKind pass)
		{
			const auto index = HdrProgramIndex(pass);
			if (HdrPipelines[index]) return true;
			if (!EnsureHdrRootSignature()) return false;
			const std::string filename = std::string(HdrShaderFilename(pass)) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX12", filename.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = HdrRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1;
			pipeline.RTVFormats[0] = pass == Techniques::HdrPassKind::ToneMapping
				? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.SampleDesc.Count = 1;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(HdrPipelines[index].GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(HDR)", result);
		}

		struct EnvironmentTargets
		{
			HdrSurface Cube, Scratch, Depth;
			std::array<UINT, EnvironmentFaceCount> FaceRTVs;
			std::vector<UINT> ParentMipSRVs;
			UINT DepthDSV = InvalidDescriptor, Resolution = 0, MipCount = 0;
			EnvironmentTargets() { FaceRTVs.fill(InvalidDescriptor); }
		};
		void DestroyEnvironmentTargets(EnvironmentTargets& targets)
		{
			SRVArena.Release(targets.Cube.SRV);
			SRVArena.Release(targets.Scratch.SRV);
			RTVArena.Release(targets.Scratch.RTV);
			DSVArena.Release(targets.DepthDSV);
			for (auto descriptor : targets.FaceRTVs) RTVArena.Release(descriptor);
			for (auto descriptor : targets.ParentMipSRVs) SRVArena.Release(descriptor);
			targets = {};
		}
		bool PrepareEnvironmentCapture(const EnvironmentCaptureFrame& frame)
		{
			EnvironmentActive = Techniques::UsesEnvironmentTexture(FrameReflections);
			EnvironmentReady = false; EnvironmentFacesWritten = 0; FrameEnvironment = {};
			if (!EnvironmentActive) return true;
			if (!ValidEnvironmentSource(frame) || (!frame.Source && frame.Resolution != FrameReflections.CaptureResolution) || frame.MipCount == 0)
				return Fail("DX12 environment reflections require a matching capture frame before BeginFrame");
			UINT expected = 0;
			for (auto extent = frame.Resolution; extent != 0; extent >>= 1) ++expected;
			if (frame.MipCount != expected) return Fail("DX12 capture requires the complete cube mip chain");
			for (const auto& face : frame.Faces)
				for (float value : face.ViewProjection.Values)
					if (!std::isfinite(value)) return Fail("DX12 capture projections must be finite");
			if (Environment.Resolution != frame.Resolution)
			{
				EnvironmentTargets candidate;
				const auto discard = [&]() { DestroyEnvironmentTargets(candidate); return false; };
				D3D12_RESOURCE_DESC texture {};
				texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
				texture.Width = texture.Height = frame.Resolution;
				texture.DepthOrArraySize = static_cast<UINT16>(EnvironmentFaceCount);
				texture.MipLevels = static_cast<UINT16>(frame.MipCount);
				texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
				texture.SampleDesc.Count = 1; texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
				const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
				D3D12_CLEAR_VALUE clear {}; clear.Format = texture.Format; clear.Color[3] = 1;
				auto result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
					D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(candidate.Cube.Resource.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateCommittedResource(environment cube)", result);
				candidate.Cube.SRV = SRVArena.Allocate();
				candidate.DepthDSV = DSVArena.Allocate();
				for (auto& target : candidate.FaceRTVs) target = RTVArena.Allocate();
				for (UINT mip = 0; mip < frame.MipCount - 1; ++mip) candidate.ParentMipSRVs.push_back(SRVArena.Allocate());
				if (candidate.Cube.SRV == InvalidDescriptor || candidate.DepthDSV == InvalidDescriptor ||
					std::find(candidate.FaceRTVs.begin(), candidate.FaceRTVs.end(), InvalidDescriptor) != candidate.FaceRTVs.end() ||
					std::find(candidate.ParentMipSRVs.begin(), candidate.ParentMipSRVs.end(), InvalidDescriptor) != candidate.ParentMipSRVs.end())
				{
					Fail("DX12 environment descriptor allocation failed"); return discard();
				}
				D3D12_SHADER_RESOURCE_VIEW_DESC view {};
				view.Format = texture.Format; view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
				view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE; view.TextureCube.MipLevels = frame.MipCount;
				Device->CreateShaderResourceView(candidate.Cube.Resource.Get(), &view, SRVArena.Cpu(candidate.Cube.SRV));
				view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
				view.Texture2DArray = {}; view.Texture2DArray.MipLevels = 1;
				view.Texture2DArray.ArraySize = static_cast<UINT>(EnvironmentFaceCount);
				for (UINT mip = 0; mip < candidate.ParentMipSRVs.size(); ++mip)
				{
					view.Texture2DArray.MostDetailedMip = mip;
					Device->CreateShaderResourceView(candidate.Cube.Resource.Get(), &view, SRVArena.Cpu(candidate.ParentMipSRVs[mip]));
				}
				D3D12_RENDER_TARGET_VIEW_DESC target {}; target.Format = texture.Format;
				target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY; target.Texture2DArray.ArraySize = 1;
				for (UINT face = 0; face < EnvironmentFaceCount; ++face)
				{
					target.Texture2DArray.FirstArraySlice = face;
					Device->CreateRenderTargetView(candidate.Cube.Resource.Get(), &target, RTVArena.Cpu(candidate.FaceRTVs[face]));
				}
				texture.DepthOrArraySize = texture.MipLevels = 1;
				texture.Format = DXGI_FORMAT_D32_FLOAT; texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
				clear = {}; clear.Format = texture.Format; clear.DepthStencil.Depth = 1;
				result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
					D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(candidate.Depth.Resource.GetAddressOf()));
				if (FAILED(result)) { FailHRESULT("CreateCommittedResource(environment depth)", result); return discard(); }
				D3D12_DEPTH_STENCIL_VIEW_DESC depth {}; depth.Format = texture.Format; depth.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
				Device->CreateDepthStencilView(candidate.Depth.Resource.Get(), &depth, DSVArena.Cpu(candidate.DepthDSV));
				candidate.Depth.State = candidate.Depth.FrameStartState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
				if (!CreateHdrSurface(candidate.Scratch, frame.Resolution / 2, frame.Resolution / 2)) return discard();
				candidate.Resolution = frame.Resolution; candidate.MipCount = frame.MipCount;
				// Allocation is transactional, before any commands reference these slots.
				DestroyEnvironmentTargets(Environment);
				Environment = std::move(candidate);
			}
			FrameEnvironment = frame;
			return true;
		}
		template<class Constants> bool UploadReflectionConstants(const Constants& constants)
		{
			const auto offset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			if (offset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - offset)
				return Fail("DX12 reflection constants exhausted the frame upload buffer");
			std::memcpy(ConstantData + offset, &constants, sizeof(constants));
			CommandList->SetGraphicsRootConstantBufferView(0, ConstantBuffer->GetGPUVirtualAddress() + offset);
			ConstantOffset = offset + sizeof(constants);
			return true;
		}

		struct EnvironmentMapTargets
		{
			HdrSurface Image, Scratch;
			std::vector<UINT> ParentMipSRVs;
			UINT MipCount = 0;
		};
		void DestroyEnvironmentMapTargets(EnvironmentMapTargets& targets)
		{
			for (auto* surface : {&targets.Image, &targets.Scratch})
			{
				SRVArena.Release(surface->SRV); RTVArena.Release(surface->RTV);
			}
			for (auto descriptor : targets.ParentMipSRVs) SRVArena.Release(descriptor);
			targets = {};
		}
		bool EnsureEnvironmentMapResources()
		{
			const auto layout = BuildEnvironmentMapLayout(FrameEnvironment.Resolution);
			if (!layout.MipCount) return Fail("DX12 environment map requires valid capture dimensions");
			if (EnvironmentMap.Image.Resource && EnvironmentMap.Image.Width == static_cast<int>(layout.Width)) return true;
			EnvironmentMapTargets candidate;
			const auto discard = [&]() { DestroyEnvironmentMapTargets(candidate); return false; };
			D3D12_RESOURCE_DESC texture{};
			texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			texture.Width = layout.Width; texture.Height = layout.Height;
			texture.DepthOrArraySize = 1; texture.MipLevels = static_cast<UINT16>(layout.MipCount);
			texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
			texture.SampleDesc.Count = 1; texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
			const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
			D3D12_CLEAR_VALUE clear{}; clear.Format = texture.Format; clear.Color[3] = 1;
			const auto result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
				D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(candidate.Image.Resource.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateCommittedResource(latitude-longitude)", result);
			candidate.Image.SRV = SRVArena.Allocate(); candidate.Image.RTV = RTVArena.Allocate();
			for (UINT mip = 0; mip + 1 < layout.MipCount; ++mip) candidate.ParentMipSRVs.push_back(SRVArena.Allocate());
			if (candidate.Image.SRV == InvalidDescriptor || candidate.Image.RTV == InvalidDescriptor ||
				std::find(candidate.ParentMipSRVs.begin(), candidate.ParentMipSRVs.end(), InvalidDescriptor) != candidate.ParentMipSRVs.end())
			{
				Fail("DX12 environment map descriptor allocation failed"); return discard();
			}
			D3D12_SHADER_RESOURCE_VIEW_DESC view{};
			view.Format = texture.Format; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			view.Texture2D.MipLevels = layout.MipCount;
			Device->CreateShaderResourceView(candidate.Image.Resource.Get(), &view, SRVArena.Cpu(candidate.Image.SRV));
			// Each draw keeps its own persistent parent view. No descriptor is rewritten
			// while an earlier command still refers to it.
			view.Texture2D.MipLevels = 1;
			for (UINT mip = 0; mip < candidate.ParentMipSRVs.size(); ++mip)
			{
				view.Texture2D.MostDetailedMip = mip;
				Device->CreateShaderResourceView(candidate.Image.Resource.Get(), &view, SRVArena.Cpu(candidate.ParentMipSRVs[mip]));
			}
			D3D12_RENDER_TARGET_VIEW_DESC target{};
			target.Format = texture.Format; target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
			Device->CreateRenderTargetView(candidate.Image.Resource.Get(), &target, RTVArena.Cpu(candidate.Image.RTV));
			candidate.Image.Width = layout.Width; candidate.Image.Height = layout.Height;
			candidate.MipCount = layout.MipCount;
			if (!CreateHdrSurface(candidate.Scratch, layout.Width / 2, layout.Height / 2)) return discard();
			DestroyEnvironmentMapTargets(EnvironmentMap);
			EnvironmentMap = std::move(candidate);
			return true;
		}
		struct PlanarTargets
		{
			HdrSurface Image, Scratch, Depth;
			std::vector<UINT> ParentMipSRVs;
			UINT DepthDSV = InvalidDescriptor;
		};
		void DestroyPlanarTargets(PlanarTargets& targets)
		{
			for (auto* surface : {&targets.Image, &targets.Scratch})
			{
				SRVArena.Release(surface->SRV); RTVArena.Release(surface->RTV);
			}
			DSVArena.Release(targets.DepthDSV);
			for (auto descriptor : targets.ParentMipSRVs) SRVArena.Release(descriptor);
			targets = {};
		}
		bool PreparePlanarCapture(const PlanarCaptureFrame& frame)
		{
			PlanarActive = Techniques::UsesPlanarCapture(FrameReflections);
			PlanarWritten = PlanarReady = false; FramePlanar = {};
			if (!PlanarActive) return true;
			ReflectionDesc desc; desc.ViewProjection = frame.MainViewProjection;
			PlanarReflectionConstants constants;
			if (!BuildPlanarReflectionConstants(desc, frame, FrameReflections, FrameReflectionShading, CreateInfo.Width, CreateInfo.Height, constants))
				return Fail("DX12 planar capture requires a finite camera and matching main viewport before BeginFrame");
			UINT expectedMips = 0;
			for (auto extent = std::max(frame.Width, frame.Height); extent; extent >>= 1) ++expectedMips;
			const auto divisor = FrameReflections.Planar.ResolutionDivisor;
			if (frame.MipCount != expectedMips || frame.Width != (frame.MainWidth + divisor - 1) / divisor ||
				frame.Height != (frame.MainHeight + divisor - 1) / divisor) return Fail("DX12 planar capture extent or mip chain is inconsistent");
			for (float value : frame.ClippedViewProjection.Values) if (!std::isfinite(value)) return Fail("DX12 planar clipping projection must be finite");
			if (Planar.Image.Width != static_cast<int>(frame.Width) || Planar.Image.Height != static_cast<int>(frame.Height))
			{
				PlanarTargets candidate;
				const auto discard = [&]() { DestroyPlanarTargets(candidate); return false; };
				D3D12_RESOURCE_DESC texture {};
				texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; texture.Width = frame.Width; texture.Height = frame.Height;
				texture.DepthOrArraySize = 1; texture.MipLevels = static_cast<UINT16>(frame.MipCount);
				texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; texture.SampleDesc.Count = 1;
				texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
				const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
				D3D12_CLEAR_VALUE clear {}; clear.Format = texture.Format; clear.Color[3] = 1;
				auto result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
					D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(candidate.Image.Resource.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateCommittedResource(planar radiance)", result);
				candidate.Image.SRV = SRVArena.Allocate(); candidate.Image.RTV = RTVArena.Allocate();
				candidate.DepthDSV = DSVArena.Allocate();
				for (UINT mip = 0; mip + 1 < frame.MipCount; ++mip) candidate.ParentMipSRVs.push_back(SRVArena.Allocate());
				if (candidate.Image.SRV == InvalidDescriptor || candidate.Image.RTV == InvalidDescriptor || candidate.DepthDSV == InvalidDescriptor ||
					std::find(candidate.ParentMipSRVs.begin(), candidate.ParentMipSRVs.end(), InvalidDescriptor) != candidate.ParentMipSRVs.end())
				{
					Fail("DX12 planar descriptor allocation failed"); return discard();
				}
				D3D12_SHADER_RESOURCE_VIEW_DESC view {};
				view.Format = texture.Format; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
				view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
				view.Texture2D.MipLevels = frame.MipCount;
				Device->CreateShaderResourceView(candidate.Image.Resource.Get(), &view, SRVArena.Cpu(candidate.Image.SRV));
				view.Texture2D.MipLevels = 1;
				for (UINT mip = 0; mip < candidate.ParentMipSRVs.size(); ++mip)
				{
					view.Texture2D.MostDetailedMip = mip;
					Device->CreateShaderResourceView(candidate.Image.Resource.Get(), &view, SRVArena.Cpu(candidate.ParentMipSRVs[mip]));
				}
				D3D12_RENDER_TARGET_VIEW_DESC target {}; target.Format = texture.Format; target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
				Device->CreateRenderTargetView(candidate.Image.Resource.Get(), &target, RTVArena.Cpu(candidate.Image.RTV));
				candidate.Image.Width = frame.Width; candidate.Image.Height = frame.Height;
				texture.MipLevels = 1; texture.Format = DXGI_FORMAT_D32_FLOAT; texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
				clear = {}; clear.Format = texture.Format; clear.DepthStencil.Depth = 1;
				result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
					D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(candidate.Depth.Resource.GetAddressOf()));
				if (FAILED(result)) { FailHRESULT("CreateCommittedResource(planar depth)", result); return discard(); }
				D3D12_DEPTH_STENCIL_VIEW_DESC depth {}; depth.Format = texture.Format; depth.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
				Device->CreateDepthStencilView(candidate.Depth.Resource.Get(), &depth, DSVArena.Cpu(candidate.DepthDSV));
				candidate.Depth.State = candidate.Depth.FrameStartState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
				if (!CreateHdrSurface(candidate.Scratch, std::max(1u, frame.Width >> 1), std::max(1u, frame.Height >> 1))) return discard();
				DestroyPlanarTargets(Planar); Planar = std::move(candidate);
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
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 reflections require HDR color and sampleable depth");
			if (!CreateHdrSurface(ReflectionSurface, CreateInfo.Width, CreateInfo.Height)) return false;
			const auto program = FrameReflectionProgram(planarPass);
			if (program == MapReflection && !EnsureEnvironmentMapResources()) return false;
			if (ReflectionPipelines[program][0] && ReflectionPipelines[program][1] &&
				(program != CubeReflection || CubeMipPipeline) && (program != MapReflection || (MapConversionPipeline && MapMipPipeline)) &&
				(program != PlanarReflection || PlanarMipPipeline)) return true;
			if (!ReflectionRootSignature)
			{
				// Each draw owns an aligned upload slice and stable SRV slots. A CBV
				// keeps extended planar parameters within the root-signature budget.
				std::array<D3D12_DESCRIPTOR_RANGE, ReflectionTextureCount> ranges {};
				std::array<D3D12_ROOT_PARAMETER, ReflectionTextureCount + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[0].Descriptor.ShaderRegister = 0;
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < ReflectionTextureCount; ++index)
				{
					ranges[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[index].NumDescriptors = 1;
					ranges[index].BaseShaderRegister = index;
					ranges[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					auto& parameter = parameters[index + 1];
					parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameter.DescriptorTable.NumDescriptorRanges = 1;
					parameter.DescriptorTable.pDescriptorRanges = &ranges[index];
					parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				std::array<D3D12_STATIC_SAMPLER_DESC, 2> samplers{};
				auto& sampler = samplers[0];
				sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
				sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				sampler.MaxLOD = D3D12_FLOAT32_MAX;
				sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				samplers[1] = sampler;
				samplers[1].ShaderRegister = 1; samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size());
				signature.pParameters = parameters.data();
				signature.NumStaticSamplers = static_cast<UINT>(samplers.size()); signature.pStaticSamplers = samplers.data();
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> serialized, errors;
				auto result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(SSR)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(ReflectionRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(SSR)", result);
			}
			const char* filename = program == PlanarReflection ? "PlanarReflections.hlsl" : program == MapReflection ? "EnvironmentMapReflections.hlsl" :
				program == CubeReflection ? "CubemapReflections.hlsl" : "SSR.hlsl";
			if (!LoadTechniqueShaderSource("DX12", filename, ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel, mipPixel, conversionPixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			if ((program != ScreenReflection) && !CompileShader("PSMip", "ps_5_0", mipPixel)) return false;
			if (program == MapReflection && !CompileShader("PSConvert", "ps_5_0", conversionPixel)) return false;
			std::array<ComPtr<ID3D12PipelineState>, 2> pipelines;
			ComPtr<ID3D12PipelineState> mipPipeline, conversionPipeline;
			for (std::size_t phase = 0; phase < pipelines.size(); ++phase)
			{
				D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
				pipeline.pRootSignature = ReflectionRootSignature.Get();
				pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				auto& blend = pipeline.BlendState.RenderTarget[0];
				const bool composite = phase == static_cast<std::size_t>(Techniques::ReflectionPassKind::Composite);
				blend.BlendEnable = composite;
				blend.SrcBlend = D3D12_BLEND_ONE; blend.DestBlend = composite ? D3D12_BLEND_ONE : D3D12_BLEND_ZERO;
				blend.SrcBlendAlpha = D3D12_BLEND_ZERO; blend.DestBlendAlpha = D3D12_BLEND_ONE;
				blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
				blend.LogicOp = D3D12_LOGIC_OP_NOOP;
				blend.RenderTargetWriteMask = composite ?
					D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_BLUE : D3D12_COLOR_WRITE_ENABLE_ALL;
				pipeline.SampleMask = UINT_MAX;
				pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
				pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
				pipeline.RasterizerState.DepthClipEnable = TRUE;
				pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
				pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
				pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
				pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
				pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
				pipeline.NumRenderTargets = 1;
				pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
				pipeline.SampleDesc.Count = 1;
				const auto result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(pipelines[phase].GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(SSR)", result);
				if ((program != ScreenReflection) && !composite)
				{
					pipeline.PS = {mipPixel->GetBufferPointer(), mipPixel->GetBufferSize()};
					const auto mipResult = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(mipPipeline.GetAddressOf()));
					if (FAILED(mipResult)) return FailHRESULT("CreateGraphicsPipelineState(environment mip)", mipResult);
					if (program == MapReflection)
					{
						pipeline.PS = {conversionPixel->GetBufferPointer(), conversionPixel->GetBufferSize()};
						const auto conversionResult = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(conversionPipeline.GetAddressOf()));
						if (FAILED(conversionResult)) return FailHRESULT("CreateGraphicsPipelineState(environment conversion)", conversionResult);
					}
				}
			}
			ReflectionPipelines[program] = std::move(pipelines);
			if (program == CubeReflection) CubeMipPipeline = std::move(mipPipeline);
			if (program == PlanarReflection) PlanarMipPipeline = std::move(mipPipeline);
			if (program == MapReflection)
			{
				MapMipPipeline = std::move(mipPipeline); MapConversionPipeline = std::move(conversionPipeline);
			}
			return true;
		}

		bool ConvertEnvironmentMap()
		{
			auto& image = EnvironmentMap.Image;
			auto& scratch = EnvironmentMap.Scratch;
			auto& cube = Environment.Cube;
			CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			CommandList->SetGraphicsRootSignature(ReflectionRootSignature.Get());
			CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			Transition(cube.Resource.Get(), cube.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			std::array<UINT, ReflectionTextureCount> inputs;
			inputs.fill(NullColorSRV); inputs[1] = NullShadowSRV; inputs[7] = cube.SRV;
			const auto draw = [&](HdrSurface& destination, UINT width, UINT height, ID3D12PipelineState* pipeline)
			{
				Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
				const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
				CommandList->RSSetViewports(1, &viewport); CommandList->RSSetScissorRects(1, &scissor);
				for (UINT slot = 0; slot < inputs.size(); ++slot)
					CommandList->SetGraphicsRootDescriptorTable(slot + 1, SRVArena.Gpu(inputs[slot]));
				EnvironmentMapReflectionConstants constants{};
				constants.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
				if (!UploadReflectionConstants(constants)) return false;
				const auto target = RTVArena.Cpu(destination.RTV);
				CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				CommandList->SetPipelineState(pipeline);
				CommandList->DrawInstanced(3, 1, 0, 0);
				CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
				return true;
			};
			if (!draw(image, image.Width, image.Height, MapConversionPipeline.Get())) return false;
			++PendingFrame.Statistics.EnvironmentConversionPassCount;
			for (UINT mip = 1; mip < EnvironmentMap.MipCount; ++mip)
			{
				const UINT width = std::max(1u, static_cast<UINT>(image.Width) >> mip);
				const UINT height = std::max(1u, static_cast<UINT>(image.Height) >> mip);
				Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				inputs[10] = EnvironmentMap.ParentMipSRVs[mip - 1];
				if (!draw(scratch, width, height, MapMipPipeline.Get())) return false;
				// Read and write different resources, then copy the reduced rectangle.
				// Whole-image transitions keep rollback independent of subresource count.
				// See Microsoft, "Using Resource Barriers to Synchronize Resource States".
				Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
				Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_COPY_DEST);
				D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource = scratch.Resource.Get();
				source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource = image.Resource.Get();
				destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; destination.SubresourceIndex = mip;
				const D3D12_BOX region{0, 0, 0, width, height, 1};
				CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, &region);
				++PendingFrame.Statistics.EnvironmentMipPassCount;
			}
			Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			return true;
		}

		bool EnsureDepthOfFieldResources()
		{
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 depth of field requires HDR color and sampleable depth");
			if (DepthOfFieldPipelines[0] && DepthOfFieldPipelines[1]) return true;
			static_assert(sizeof(DepthOfFieldConstants) <= sizeof(HdrConstants), "Depth of field must fit the fullscreen root constant range");
			if (!EnsureHdrRootSignature()) return false;
			if (!LoadTechniqueShaderSource("DX12", "DepthOfField.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			std::array<ComPtr<ID3D12PipelineState>, 2> pipelines;
			for (std::size_t index = 0; index < pipelines.size(); ++index)
			{
				D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
				pipeline.pRootSignature = HdrRootSignature.Get();
				pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				auto& blend = pipeline.BlendState.RenderTarget[0];
				blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
				blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
				blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
				blend.LogicOp = D3D12_LOGIC_OP_NOOP;
				blend.RenderTargetWriteMask = index == 0 ? D3D12_COLOR_WRITE_ENABLE_ALL :
					D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_BLUE;
				pipeline.SampleMask = UINT_MAX;
				pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
				pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
				pipeline.RasterizerState.DepthClipEnable = TRUE;
				pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
				pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
				pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
				pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
				pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
				pipeline.NumRenderTargets = 1;
				pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
				pipeline.SampleDesc.Count = 1;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(pipelines[index].GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(depth of field)", result);
			}
			DepthOfFieldPipelines = std::move(pipelines);
			return true;
		}

		bool EnsureVelocityResources()
		{
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 velocity requires HDR color and sampleable opaque depth");
			if (!CreateHdrSurface(MotionSurfaces[MotionVelocity], CreateInfo.Width, CreateInfo.Height)) return false;
			if (CreateInfo.CaptureDiagnosticBuffers && !MotionReadback.Buffer &&
				!CreateReadback(MotionSurfaces[MotionVelocity].Resource.Get(), MotionReadback)) return false;
			if (MotionRootSignature && MotionVertexBytecode && MotionPixelBytecode) return true;
			if (!MotionRootSignature)
			{
				// A CBV keeps the 320-byte matrix/optics contract out of the 64-DWORD
				// root budget. Each draw/pass owns an aligned upload slice until its fence.
				// https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
				std::array<D3D12_DESCRIPTOR_RANGE, 5> ranges {};
				std::array<D3D12_ROOT_PARAMETER, MotionInstanceRoot + 2> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[0].Descriptor.ShaderRegister = 0;
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
				for (UINT slot = 0; slot < ranges.size(); ++slot)
				{
					ranges[slot].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[slot].NumDescriptors = 1;
					ranges[slot].BaseShaderRegister = slot;
					ranges[slot].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[slot + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[slot + 1].DescriptorTable = {1, &ranges[slot]};
					parameters[slot + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				parameters[MotionInstanceRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[MotionInstanceRoot].Descriptor.ShaderRegister = InstanceConstantRegister;
				parameters[MotionInstanceRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
				D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
				DescribeMaterialSampler(parameters.back(), materialSamplerRange);
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size());
				signature.pParameters = parameters.data();
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> serialized, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(motion blur)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(MotionRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(motion blur)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "MotionVectors.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> velocityVertex, velocityPixel;
			if (!CompileShader("VSVelocity", "vs_5_0", velocityVertex) || !CompileShader("PSVelocity", "ps_5_0", velocityPixel)) return false;
			MotionVertexBytecode = std::move(velocityVertex);
			MotionPixelBytecode = std::move(velocityPixel);
			return true;
		}

		bool EnsureMotionResources()
		{
			if (!EnsureVelocityResources()) return false;
			const int tile = Techniques::MotionBlurTileSize;
			if (!CreateHdrSurface(MotionSurfaces[MotionTileMax], (CreateInfo.Width + tile - 1) / tile, (CreateInfo.Height + tile - 1) / tile) ||
				!CreateHdrSurface(MotionSurfaces[MotionNeighborMax], (CreateInfo.Width + tile - 1) / tile, (CreateInfo.Height + tile - 1) / tile)) return false;
			if (MotionScreenPipelines[0] && MotionScreenPipelines[1]) return true;
			ComPtr<ID3DBlob> screenVertex, screenPixel;
			if (!LoadTechniqueShaderSource("DX12", "MotionBlur.hlsl", ShaderCode, LastError) ||
				!CompileShader("VSMain", "vs_5_0", screenVertex) || !CompileShader("PSMain", "ps_5_0", screenPixel)) return false;
			std::array<ComPtr<ID3D12PipelineState>, 2> pipelines;
			for (std::size_t index = 0; index < pipelines.size(); ++index)
			{
				D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
				pipeline.pRootSignature = MotionRootSignature.Get();
				pipeline.VS = {screenVertex->GetBufferPointer(), screenVertex->GetBufferSize()};
				pipeline.PS = {screenPixel->GetBufferPointer(), screenPixel->GetBufferSize()};
				auto& blend = pipeline.BlendState.RenderTarget[0];
				blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
				blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
				blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
				blend.LogicOp = D3D12_LOGIC_OP_NOOP;
				blend.RenderTargetWriteMask = index == 0 ? D3D12_COLOR_WRITE_ENABLE_ALL :
					D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_BLUE;
				pipeline.SampleMask = UINT_MAX;
				pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
				pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
				pipeline.RasterizerState.DepthClipEnable = TRUE;
				pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
				pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
				pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
				pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
				pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
				pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
				pipeline.NumRenderTargets = 1;
				pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
				pipeline.SampleDesc.Count = 1;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(pipelines[index].GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(motion blur)", result);
			}
			MotionScreenPipelines = std::move(pipelines);
			return true;
		}

		UINT WeightedSampleCounts() const
		{
			if (!Device) return 0;
			UINT counts = Techniques::AllMultisampleCounts;
			for (const auto format : TransparencyFormats)
			{
				D3D12_FEATURE_DATA_FORMAT_SUPPORT support {format};
				constexpr UINT Required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_RENDER_TARGET |
					D3D12_FORMAT_SUPPORT1_BLENDABLE | D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
				if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
					(support.Support1 & Required) != Required) return 0;
				constexpr UINT Multisample = D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD;
				for (UINT samples : {2u, 4u, 8u})
				{
					D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality {format, samples, D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 0};
					if ((support.Support1 & Multisample) != Multisample ||
						FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))) ||
						quality.NumQualityLevels == 0) counts &= ~samples;
				}
			}
			return counts;
		}

		bool EnsureWeightedTechnique(Techniques::ShadingModel model)
		{
			const auto index = ShadingTechniqueIndex(model, FrameAreaLights);
			const char* name = ShadingTechniqueFile(model, FrameAreaLights);
			if (index >= WeightedPixelBytecodes.size() || !name) return Fail("Unknown DX12 transparency shading model");
			if (WeightedPixelBytecodes[index]) return true;
			const std::string file = std::string(name) + ".hlsl";
			return LoadTechniqueShaderSource("DX12", file.c_str(), ShaderCode, LastError) &&
				CompileShader("PSWeighted", "ps_5_0", WeightedPixelBytecodes[index]);
		}

		bool EnsureWeightedResources()
		{
			if (!HdrSceneActive) return Fail("DX12 weighted transparency requires scene-linear HDR");
			const float emptySum[4] {}, fullVisibility[4] {1, 1, 1, 1};
			for (std::size_t index = 0; index < WeightedSurfaces.size(); ++index)
				if (!CreateHdrSurface(WeightedSurfaces[index], CreateInfo.Width, CreateInfo.Height, TransparencyFormats[index],
					FrameSampleCount, index == 0 ? emptySum : fullVisibility)) return false;
			if (WeightedResolvePipelines.count(FrameSampleCount)) return true;
			if (!WeightedRootSignature)
			{
				// Single-sample and multisample loads occupy separate register pairs.
				// Per-surface descriptors are allocated before recording and remain
				// immutable until the submitted frame has completed.
				std::array<D3D12_DESCRIPTOR_RANGE, 4> ranges {};
				std::array<D3D12_ROOT_PARAMETER, 4> parameters {};
				for (UINT index = 0; index < ranges.size(); ++index)
				{
					ranges[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[index].NumDescriptors = 1; ranges[index].BaseShaderRegister = index;
					ranges[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index].DescriptorTable = {1, &ranges[index]};
					parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> binary, errors;
				auto result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					binary.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(weighted transparency)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, binary->GetBufferPointer(), binary->GetBufferSize(), IID_PPV_ARGS(WeightedRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(weighted transparency)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "WeightedBlendedOIT.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) ||
				!CompileShader(FrameSampleCount > 1 ? "PSResolveSamples" : "PSMain", "ps_5_0", pixel)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = WeightedRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& color = pipeline.BlendState.RenderTarget[0];
			color.BlendEnable = TRUE;
			color.SrcBlend = D3D12_BLEND_SRC_ALPHA; color.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
			color.SrcBlendAlpha = D3D12_BLEND_ONE; color.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
			color.BlendOp = color.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			color.LogicOp = D3D12_LOGIC_OP_NOOP; color.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE; pipeline.RasterizerState.MultisampleEnable = FrameSampleCount > 1;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			auto& stencil = pipeline.DepthStencilState.FrontFace;
			stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS; pipeline.DepthStencilState.BackFace = stencil;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1; pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.SampleDesc.Count = FrameSampleCount;
			ComPtr<ID3D12PipelineState> native;
			const auto result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(native.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(weighted transparency)", result);
			WeightedResolvePipelines.emplace(FrameSampleCount, std::move(native));
			return true;
		}

		bool EnsureVolumeLightResources()
		{
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& programs = cloud ? CloudVolumePipelines : media ? MediaVolumePipelines : VolumeLightPipelines;
			const auto grid = BuildFogGrid(CreateInfo.Width,CreateInfo.Height,FrameFog,D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
			if (grid.Width == 0) return Fail("DX12 volume light grid is invalid");
			for (auto& surface : VolumeLightScratch)
				if (!CreateHdrSurface(surface, grid.AtlasWidth(), grid.AtlasHeight(), DXGI_FORMAT_R32G32B32A32_FLOAT)) return false;
			if (programs[0] && programs[1] && programs[2] &&
				(media || ((!UsesPCSS() || PCSSVolumePipeline) && (!UsesVSM() || VSMVolumePipeline)))) return true;
			if (!VolumeLightRootSignature)
			{
				// 32 medium DWORDs + two root CBVs + six SRV tables + one root SRV = 44 DWORDs.
				// Lighting/cascade data lives in immutable, aligned frame uploads.
				// https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
				constexpr UINT TextureCount = 6;
				std::array<D3D12_DESCRIPTOR_RANGE, TextureCount> ranges {};
				std::array<D3D12_ROOT_PARAMETER, VolumeDensityRoot + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
				parameters[0].Constants = {0, 0, FogGridConstantBytes / sizeof(std::uint32_t)};
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[1].Descriptor.ShaderRegister = 1;
				parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				parameters[VolumeShadowFilterRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[VolumeShadowFilterRoot].Descriptor.ShaderRegister = ShadowFilterConstantRegister;
				parameters[VolumeShadowFilterRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                parameters[VolumeSceneLightRoot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                parameters[VolumeSceneLightRoot].Descriptor.ShaderRegister = VolumeSceneLightRegister;
                parameters[VolumeSceneLightRoot].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < TextureCount; ++index)
				{
					ranges[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[index].NumDescriptors = 1; ranges[index].BaseShaderRegister = index;
					ranges[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index + 2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index + 2].DescriptorTable = {1, &ranges[index]};
					parameters[index + 2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
                D3D12_DESCRIPTOR_RANGE rayRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,RayShadowImageRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                parameters[VolumeRayShadowRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[VolumeRayShadowRoot].DescriptorTable={1,&rayRange};
                parameters[VolumeRayShadowRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
                D3D12_DESCRIPTOR_RANGE densityRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,VolumeDensityRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                parameters[VolumeDensityRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[VolumeDensityRoot].DescriptorTable={1,&densityRange};
                parameters[VolumeDensityRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> binary, diagnostics;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					binary.GetAddressOf(), diagnostics.GetAddressOf());
				if (FAILED(result))
				{
					if (diagnostics) LastError.assign(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(volume lighting)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, binary->GetBufferPointer(), binary->GetBufferSize(),
					IID_PPV_ARGS(VolumeLightRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(volume lighting)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", cloud ? "VolumetricClouds.hlsl" : media ? "ParticipatingMedia.hlsl" : "VolumetricLighting.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, inject, prefix;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSInject", "ps_5_0", inject) ||
				!CompileShader("PSPrefix", "ps_5_0", prefix)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = VolumeLightRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			auto& color = pipeline.BlendState.RenderTarget[0];
			color.SrcBlend = color.SrcBlendAlpha = D3D12_BLEND_ONE;
			color.DestBlend = color.DestBlendAlpha = D3D12_BLEND_ZERO;
			color.BlendOp = color.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			color.LogicOp = D3D12_LOGIC_OP_NOOP; color.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			auto& stencil = pipeline.DepthStencilState.FrontFace;
			stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = stencil;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1; pipeline.SampleDesc.Count = 1;
			for (std::size_t index = 0; index < programs.size(); ++index)
			{
				const auto& pixel = index == 0 ? inject : prefix;
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				pipeline.RTVFormats[0] = index == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(programs[index].ReleaseAndGetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(volume lighting)", result);
			}
			if (!media && UsesPCSS())
			{
				if (!EnsurePCSSProgram(PCSSVolumeProgram)) return false;
				const auto& pixel = PCSSPixelBytecodes[PCSSVolumeProgram];
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				pipeline.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(PCSSVolumePipeline.ReleaseAndGetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(PCSS volume lighting)", result);
			}
			if (!media && UsesVSM())
			{
				if (!EnsureVSMProgram(VSMVolumeProgram)) return false;
				const auto& pixel = VSMPixelBytecodes[VSMVolumeProgram];
				pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
				pipeline.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
				const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(VSMVolumePipeline.ReleaseAndGetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(VSM volume lighting)", result);
			}
			return true;
		}

		bool GenerateVolumeLighting(const VolumetricFogDesc& requested)
		{
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& programs = cloud ? CloudVolumePipelines : media ? MediaVolumePipelines : VolumeLightPipelines;
			auto desc = requested;
            if (LightListsActive && !LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, LastError)) return false;
            if (LightListsActive) desc.Lighting.SceneLights = LightListFrame.Description().Lights;
			if (!FrameRayShadowImage && !FrameVirtualShadowAtlas && !FrameCascades.Frame.CascadeCount)
			{
				const auto shadow = Textures.find(LastShadowTexture.Value);
				desc.SamplesShadowMap = desc.SamplesShadowMap && shadow != Textures.end() &&
					shadow->second.Description.Format == TextureFormat::Depth32;
			}
			VolumetricLightingConstants lights;
			if (!BuildVolumetricLightingConstants(desc, FrameVolumeLighting, FrameCascades, FrameShadowFilter, lights, LightListsActive, &FrameFog,
                FrameVirtualShadowAtlas ? &FrameVirtualShadowAtlas->Info : nullptr, &FrameVirtualShadowSampling,
                {FogConstants.ViewportGrid.Z, FogConstants.ViewportGrid.W}))
				return Fail("DX12 volume lighting requires finite lights and shadow transforms");
			CommandList->SetGraphicsRootSignature(VolumeLightRootSignature.Get());
            BindRayShadowImage(VolumeRayShadowRoot);
            UINT density=NullColorSRV;
            if (desc.Density)
            {
                const auto found=Textures.find(desc.Density->Texture.Value);
                if (found==Textures.end() || !ValidDensityBinding(*desc.Density,found->second.Description))
                    return Fail("DX12 volume density requires a matching linear byte atlas");
                Transition(found->second.Resource.Get(),found->second.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                density=found->second.SRV;
            }
            CommandList->SetGraphicsRootDescriptorTable(VolumeDensityRoot,SRVArena.Gpu(density));
            // A root SRV shares the immutable scene upload without another heap
            // descriptor. Legacy volume variants do not dereference this buffer.
            CommandList->SetGraphicsRootShaderResourceView(VolumeSceneLightRoot,
                LightListsActive ? ForwardPlusSceneAddress : ConstantBuffer->GetGPUVirtualAddress());
			if (UsesPCSS() && !BindPCSS(VolumeShadowFilterRoot, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap)) return false;
			if (UsesVSM() && !BindVSM(VolumeShadowFilterRoot, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap)) return false;
			const auto passes = BuildVolumetricLightingPasses(FogConstants.Options.Y);
			const auto firstOffset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			const auto stride = AlignUp(sizeof(lights), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			const auto required = stride * passes.size();
			if (passes.empty() || firstOffset > ConstantBufferCapacity || required > ConstantBufferCapacity - firstOffset)
				return Fail("DX12 volume light scan exhausted the frame upload buffer");
			std::array<UINT, 6> inputs {NullColorSRV, NullShadowSRV, NullShadowSRV, NullShadowSRV, NullShadowSRV, NullShadowSRV};
			CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			if (FrameRayShadowImage) {}
        else if (lights.Control.Y && FrameVirtualShadowAtlas)
            {
                inputs[2]=FrameVirtualShadowViews[0]; inputs[3]=FrameVirtualShadowViews[1];
            }
            else if (lights.Control.Y)
			{
				const bool cascaded = lights.Control.Y == 2;
				const auto count = cascaded ? FrameCascades.Frame.CascadeCount : 1;
				for (std::uint32_t index = 0; index < count; ++index)
				{
					auto& shadow = Textures.at((cascaded ? FrameCascades.Maps[index] : LastShadowTexture).Value);
					Transition(shadow.Resource.Get(), shadow.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					inputs[cascaded ? index + 2 : 1] = UsesVSM() ?
						VSMMoments.at((cascaded ? FrameCascades.Maps[index] : LastShadowTexture).Value).Filtered.SRV : shadow.SRV;
				}
			}
			CommandList->SetGraphicsRoot32BitConstants(0, FogGridConstantBytes / sizeof(std::uint32_t), &FogConstants, 0);
			CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			const D3D12_VIEWPORT viewport {0, 0, float(FogAtlas.Width), float(FogAtlas.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, FogAtlas.Width, FogAtlas.Height};
			CommandList->RSSetViewports(1, &viewport); CommandList->RSSetScissorRects(1, &scissor);
			for (std::size_t index = 0; index < passes.size(); ++index)
			{
				const bool inject = index == 0, final = index + 1 == passes.size();
				lights.Control.W = passes[index].PrefixStride;
				const auto offset = firstOffset + index * stride;
				std::memcpy(ConstantData + offset, &lights, sizeof(lights));
				CommandList->SetGraphicsRootConstantBufferView(1, ConstantBuffer->GetGPUVirtualAddress() + offset);
				if (!inject)
				{
					auto& previous = VolumeLightScratch[(index - 1) % VolumeLightScratch.size()];
					Transition(previous.Resource.Get(), previous.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					inputs = {previous.SRV, NullShadowSRV, NullShadowSRV, NullShadowSRV, NullShadowSRV, NullShadowSRV};
				}
				auto& destination = final ? FogAtlas : VolumeLightScratch[index % VolumeLightScratch.size()];
				Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				for (UINT slot = 0; slot < inputs.size(); ++slot)
					CommandList->SetGraphicsRootDescriptorTable(slot + 2, SRVArena.Gpu(inputs[slot]));
				CommandList->SetPipelineState(inject && !media && UsesPCSS() ? PCSSVolumePipeline.Get() :
					inject && !media && UsesVSM() ? VSMVolumePipeline.Get() : programs[inject ? 0 : final ? 2 : 1].Get());
				const auto target = RTVArena.Cpu(destination.RTV);
				CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				constexpr UINT TriangleVertexCount = 3;
				CommandList->DrawInstanced(TriangleVertexCount, 1, 0, 0);
				CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			}
			ConstantOffset = firstOffset + required;
			(Techniques::UsesVolumetricClouds(FrameFog) ? PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(FrameFog) ? PendingFrame.Statistics.ParticipatingMediaPassCount : PendingFrame.Statistics.VolumetricLightingPassCount) += static_cast<std::uint32_t>(passes.size());
			return true;
		}

		bool EnsureLightShaftResources()
		{
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 light shafts require HDR color and sampleable opaque depth");
			if (!CreateHdrSurface(LightShaftMask, CreateInfo.Width, CreateInfo.Height)) return false;
			if (LightShaftPipeline) return true;
			if (!LightShaftRootSignature)
			{
				constexpr UINT InputCount = 3;
				std::array<D3D12_DESCRIPTOR_RANGE, InputCount> ranges {};
				std::array<D3D12_ROOT_PARAMETER, InputCount + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
				parameters[0].Constants = {0, 0, sizeof(LightShaftConstants) / sizeof(std::uint32_t)};
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < ranges.size(); ++index)
				{
					ranges[index].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[index].NumDescriptors = 1; ranges[index].BaseShaderRegister = index;
					ranges[index].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index + 1].DescriptorTable = {1, &ranges[index]};
					parameters[index + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> binary, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					binary.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(light shafts)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, binary->GetBufferPointer(), binary->GetBufferSize(), IID_PPV_ARGS(LightShaftRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(light shafts)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "LightShafts.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = LightShaftRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			auto& stencil = pipeline.DepthStencilState.FrontFace;
			stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS; pipeline.DepthStencilState.BackFace = stencil;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1; pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.SampleDesc.Count = 1;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(LightShaftPipeline.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(light shafts)", result);
		}

		bool EnsureFogResources()
		{
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& selectedPipeline = cloud ? CloudFogPipeline : media ? MediaFogPipeline : FogPipeline;
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 fog requires floating scene color and sampled depth");
			const auto grid = BuildFogGrid(CreateInfo.Width,CreateInfo.Height,FrameFog,D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
			if (grid.Width == 0 || !CreateHdrSurface(FogAtlas, grid.AtlasWidth(), grid.AtlasHeight())) return false;
			if (selectedPipeline) return true;
			if (!FogRootSignature)
			{
				constexpr std::array<UINT, 3> textureRegisters {0, 1, FogTextureRegister};
				std::array<D3D12_DESCRIPTOR_RANGE, textureRegisters.size()> ranges {};
				std::array<D3D12_ROOT_PARAMETER, textureRegisters.size() + 1> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
				parameters[0].Constants = {FogConstantRegister, 0, sizeof(VolumetricFogConstants) / sizeof(std::uint32_t)};
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT index = 0; index < ranges.size(); ++index)
				{
					auto& range = ranges[index];
					range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors = 1;
					range.BaseShaderRegister = textureRegisters[index];
					range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[index + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[index + 1].DescriptorTable = {1, &range};
					parameters[index + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_STATIC_SAMPLER_DESC sampler {};
				sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
				sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				sampler.MaxAnisotropy = 1; sampler.MaxLOD = D3D12_FLOAT32_MAX;
				sampler.ShaderRegister = FogSamplerRegister; sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.NumStaticSamplers = 1; signature.pStaticSamplers = &sampler;
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> binary, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					binary.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(fog)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, binary->GetBufferPointer(), binary->GetBufferSize(), IID_PPV_ARGS(FogRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(fog)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", cloud ? "VolumetricClouds.hlsl" : media ? "ParticipatingMedia.hlsl" : "VolumetricFog.hlsl", ShaderCode, LastError)) return false;
			const D3D_SHADER_MACRO definitions[] {{cloud ? "EVANS_CLOUD_COMPOSITE" : "EVANS_MEDIA_COMPOSITE", "1"}, {nullptr, nullptr}};
            ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex, media ? definitions : nullptr) || !CompileShader("PSMain", "ps_5_0", pixel, media ? definitions : nullptr)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = FogRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& color = pipeline.BlendState.RenderTarget[0];
			color.SrcBlend = color.SrcBlendAlpha = D3D12_BLEND_ONE;
			color.DestBlend = color.DestBlendAlpha = D3D12_BLEND_ZERO;
			color.BlendOp = color.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			color.LogicOp = D3D12_LOGIC_OP_NOOP; color.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			auto& stencil = pipeline.DepthStencilState.FrontFace;
			stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = stencil;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1; pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.SampleDesc.Count = 1;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(selectedPipeline.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(fog)", result);
		}

        #include "Renderer/DX12/DX12Lightmaps.inl"
    #include "Renderer/DX12/DX12Refraction.inl"
    #include "Renderer/DX12/DX12DepthPeeling.inl"

        bool EnsureProbeResources()
        {
            if (!FrameProbeSamples || FrameProbeSamples->Texels.empty()) return Fail("DX12 probes require baked samples");
            const auto count=FrameProbeSamples->ProbeCount();
            if (!ProbeSurfaces[ProbeSH].Resource || ProbeSurfaces[ProbeSH].Height != static_cast<int>(count)) Probes.Invalidate();
            if (!CreateHdrSurface(ProbeSurfaces[ProbeRaw],FrameProbeSamples->Width(),FrameProbeSamples->Height(),DXGI_FORMAT_R32G32B32A32_FLOAT) ||
                !CreateHdrSurface(ProbeSurfaces[ProbeSH],ProbeSHCoefficientCount,count) ||
                !CreateHdrSurface(ProbeSurfaces[ProbeMoments],ProbeMomentAtlasColumns*ProbeMomentTileSize,
                    ((count+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize) ||
                !CreateHdrSurface(ProbeSurfaces[ProbeComposite],CreateInfo.Width,CreateInfo.Height)) return false;
            if (!ProbeRootSignature)
            {
                constexpr UINT TextureCount=8;
                std::array<D3D12_DESCRIPTOR_RANGE,TextureCount> ranges {};
                std::array<D3D12_ROOT_PARAMETER,TextureCount+1> parameters {};
                parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
                parameters[0].Descriptor={0,0}; parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
                for (UINT slot=0;slot<TextureCount;++slot)
                {
                    ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1;
                    ranges[slot].BaseShaderRegister=slot; ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                    parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    parameters[slot+1].DescriptorTable={1,&ranges[slot]}; parameters[slot+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
                }
                D3D12_ROOT_SIGNATURE_DESC signature {}; signature.NumParameters=static_cast<UINT>(parameters.size());
                signature.pParameters=parameters.data(); signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
                ComPtr<ID3DBlob> bytes,errors;
                auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,bytes.GetAddressOf(),errors.GetAddressOf());
                if (FAILED(result))
                {
                    if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize());
                    else FailHRESULT("SerializeRootSignature(probes)",result);
                    return false;
                }
                result=Evans::Renderer::CreateCachedRootSignature(Device,0,bytes->GetBufferPointer(),bytes->GetBufferSize(),IID_PPV_ARGS(ProbeRootSignature.GetAddressOf()));
                if (FAILED(result)) return FailHRESULT("CreateRootSignature(probes)",result);
            }
            const char* sources[] {"BakedGI.hlsl","LightProbes.hlsl"};
            for (std::size_t index=0;index<ProbePipelines.size();++index)
            {
                if (ProbePipelines[index]) continue;
                if (!LoadTechniqueShaderSource("DX12",sources[index],ShaderCode,LastError)) return false;
                ComPtr<ID3DBlob> vertex,pixel;
                if (!CompileShader("VSMain","vs_5_0",vertex) || !CompileShader("PSMain","ps_5_0",pixel)) return false;
                D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
                pipeline.pRootSignature=ProbeRootSignature.Get();
                pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
                auto& blend=pipeline.BlendState.RenderTarget[0];
                blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
                blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
                blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL; pipeline.SampleMask=UINT_MAX;
                pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
                pipeline.RasterizerState.DepthClipEnable=TRUE;
                pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
                pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
                pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK;
                pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
                pipeline.DepthStencilState.FrontFace.StencilFailOp=D3D12_STENCIL_OP_KEEP;
                pipeline.DepthStencilState.FrontFace.StencilDepthFailOp=D3D12_STENCIL_OP_KEEP;
                pipeline.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
                pipeline.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
                pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
                pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; pipeline.SampleDesc.Count=1;
                const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(ProbePipelines[index].GetAddressOf()));
                if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(probes)",result);
            }
            return true;
        }

        bool ResolveProbes(const GlobalIlluminationDesc& desc)
        {
            LightProbeConstants constants;
            if (!FrameProbeSamples || !BuildLightProbeConstants(desc,*FrameProbeSamples,FrameGI,FrameReflectionShading,
                CreateInfo.Width,CreateInfo.Height,constants)) return Fail("DX12 probes require finite geometry and sample inputs");
            auto& scene=HdrSurfaces[HdrScene]; auto& raw=ProbeSurfaces[ProbeRaw];
            constexpr auto Sampled=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
            Transition(scene.Resource.Get(),scene.State,Sampled); Transition(MainDepth.Get(),MainDepthState,Sampled);
            for (auto& material:GBufferSurfaces) Transition(material.Resource.Get(),material.State,Sampled);
            const bool projection=Probes.NeedsProjection(FrameProbeSamples,constants.Visibility.Y);
            if (projection)
            {
                Probes.Invalidate();
                const auto texture=raw.Resource->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {}; UINT64 bytes=0;
                Device->GetCopyableFootprints(&texture,0,1,0,&footprint,nullptr,nullptr,&bytes);
                ComPtr<ID3D12Resource> upload;
                if (!CreateBufferResource(static_cast<std::size_t>(bytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,upload)) return false;
                void* mapping=nullptr; const D3D12_RANGE noRead {0,0};
                const auto result=upload->Map(0,&noRead,&mapping);
                if (FAILED(result)) return FailHRESULT("Map(probe samples)",result);
                const auto rowBytes=static_cast<std::size_t>(FrameProbeSamples->Width())*sizeof(Float4);
                for (std::uint32_t row=0;row<FrameProbeSamples->Height();++row)
                    std::memcpy(static_cast<std::uint8_t*>(mapping)+footprint.Offset+static_cast<std::size_t>(row)*footprint.Footprint.RowPitch,
                        FrameProbeSamples->Texels.data()+static_cast<std::size_t>(row)*FrameProbeSamples->Width(),rowBytes);
                upload->Unmap(0,nullptr); Transition(raw.Resource.Get(),raw.State,D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION source {}; source.pResource=upload.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                source.PlacedFootprint=footprint;
                D3D12_TEXTURE_COPY_LOCATION destination {}; destination.pResource=raw.Resource.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                CommandList->CopyTextureRegion(&destination,0,0,0,&source,nullptr); FrameUploads.push_back(std::move(upload));
                Transition(raw.Resource.Get(),raw.State,Sampled);
            }
            CommandList->SetGraphicsRootSignature(ProbeRootSignature.Get());
            CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            auto draw=[&](ProbePass phase,std::size_t targetIndex) {
                constants.Options.X=static_cast<unsigned>(phase);
                const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
                if (offset>ConstantBufferCapacity || sizeof(constants)>ConstantBufferCapacity-offset) return Fail("DX12 probes exhausted the frame constant upload");
                std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
                CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
                const bool gather=phase==ProbePass::Gather;
                const UINT inputs[] {raw.SRV,gather?ProbeSurfaces[ProbeSH].SRV:NullColorSRV,gather?ProbeSurfaces[ProbeMoments].SRV:NullColorSRV,
                    gather?scene.SRV:NullColorSRV,gather?MainDepthSRV:NullColorSRV,gather?GBufferSurfaces[1].SRV:NullColorSRV,
                    gather?GBufferSurfaces[0].SRV:NullColorSRV,gather?GBufferSurfaces[3].SRV:NullColorSRV};
                for (UINT slot=0;slot<std::size(inputs);++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot]));
                auto& output=ProbeSurfaces[targetIndex]; Transition(output.Resource.Get(),output.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
                const D3D12_VIEWPORT viewport {0,0,float(output.Width),float(output.Height),0,1};
                const D3D12_RECT scissor {0,0,output.Width,output.Height};
                CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
                const auto target=RTVArena.Cpu(output.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
                CommandList->SetPipelineState(ProbePipelines[gather?1:0].Get()); CommandList->DrawInstanced(3,1,0,0);
                CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); Transition(output.Resource.Get(),output.State,Sampled); return true;
            };
            if (projection)
            {
                if (!draw(ProbePass::ProjectIrradiance,ProbeSH) || !draw(ProbePass::ProjectDistance,ProbeMoments)) return false;
                Probes.Projected(FrameProbeSamples,constants.Visibility.Y);
            }
            for (auto index:{ProbeSH,ProbeMoments}) Transition(ProbeSurfaces[index].Resource.Get(),ProbeSurfaces[index].State,Sampled);
            if (!draw(ProbePass::Gather,ProbeComposite)) return false;
            auto& composite=ProbeSurfaces[ProbeComposite]; Transition(composite.Resource.Get(),composite.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(scene.Resource.Get(),composite.Resource.Get());
            Probes.Resolved=GI.Resolved=true;
            PendingFrame.Statistics.GlobalIlluminationPassCount=projection?3:1;
            return true;
        }

		bool EnsureGIResources()
		{
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 SSGI requires HDR color and sampleable depth");
            for (std::size_t index = 0; index < GISurfaces.size(); ++index)
                if (!CreateHdrSurface(GISurfaces[index], CreateInfo.Width, CreateInfo.Height,
                    index >= GIDepthFirst && index < GINormalFirst ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT)) return false;
            if (GIPipeline && GITemporalPipeline) return true;
			if (!GIRootSignature)
			{
                // Upload allocations snapshot each 320-byte phase block. A CBV
                // keeps the root signature below the 64-DWORD hardware limit.
				std::array<D3D12_DESCRIPTOR_RANGE, 10> ranges {};
				std::array<D3D12_ROOT_PARAMETER, 11> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
				parameters[0].Descriptor = {0, 0};
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT slot = 0; slot < ranges.size(); ++slot)
				{
					ranges[slot].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[slot].NumDescriptors = 1;
					ranges[slot].BaseShaderRegister = slot;
					ranges[slot].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[slot + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[slot + 1].DescriptorTable = {1, &ranges[slot]};
					parameters[slot + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_STATIC_SAMPLER_DESC sampler {};
				sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
				sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				sampler.MaxAnisotropy = 1; sampler.MaxLOD = D3D12_FLOAT32_MAX;
				sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.NumStaticSamplers = 1; signature.pStaticSamplers = &sampler;
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> serialized, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(SSGI)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(GIRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(SSGI)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "SSGI.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel, temporal;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel) || !CompileShader("PSTemporal", "ps_5_0", temporal)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = GIRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1;
			pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			// Inactive RTV slots remain DXGI_FORMAT_UNKNOWN in the single-output pipeline.
			pipeline.SampleDesc.Count = 1;
            HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(GIPipeline.ReleaseAndGetAddressOf()));
            if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(SSGI)", result);
            pipeline.PS = {temporal->GetBufferPointer(), temporal->GetBufferSize()};
            pipeline.NumRenderTargets = 3;
            pipeline.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT;
            pipeline.RTVFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(GITemporalPipeline.ReleaseAndGetAddressOf()));
            return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(SSGI history)", result);
        }

		bool EnsureTemporalResources()
		{
			if (!HdrSceneActive || MainDepthSRV == InvalidDescriptor)
				return Fail("DX12 TAA requires HDR color and sampleable depth");
			for (std::size_t index = 0; index < TemporalColorHistory.size(); ++index)
			{
				if (!CreateHdrSurface(TemporalColorHistory[index], CreateInfo.Width, CreateInfo.Height) ||
					!CreateHdrSurface(TemporalDepthHistory[index], CreateInfo.Width, CreateInfo.Height, DXGI_FORMAT_R32_FLOAT)) return false;
			}
			if (TemporalPipeline) return true;
			if (!TemporalRootSignature)
			{
				// Root constants snapshot all 128 bytes in the recorded draw. Each
				// history texture retains its own SRV; no live descriptor is rewritten
				// between recording and fence completion. Root cost: 32 + 5 DWORDs.
				std::array<D3D12_DESCRIPTOR_RANGE, 5> ranges {};
				std::array<D3D12_ROOT_PARAMETER, 6> parameters {};
				parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
				parameters[0].Constants = {0, 0, sizeof(TemporalAAConstants) / sizeof(std::uint32_t)};
				parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				for (UINT slot = 0; slot < ranges.size(); ++slot)
				{
					ranges[slot].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					ranges[slot].NumDescriptors = 1;
					ranges[slot].BaseShaderRegister = slot;
					ranges[slot].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
					parameters[slot + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
					parameters[slot + 1].DescriptorTable = {1, &ranges[slot]};
					parameters[slot + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				}
				D3D12_STATIC_SAMPLER_DESC sampler {};
				sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
				sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
				sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
				sampler.MaxAnisotropy = 1; sampler.MaxLOD = D3D12_FLOAT32_MAX;
				sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
				D3D12_ROOT_SIGNATURE_DESC signature {};
				signature.NumParameters = static_cast<UINT>(parameters.size()); signature.pParameters = parameters.data();
				signature.NumStaticSamplers = 1; signature.pStaticSamplers = &sampler;
				signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
				ComPtr<ID3DBlob> serialized, errors;
				HRESULT result = SerializeRootSignatureFunction(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
					serialized.GetAddressOf(), errors.GetAddressOf());
				if (FAILED(result))
				{
					if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
					else FailHRESULT("SerializeRootSignature(TAA)", result);
					return false;
				}
				result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
					IID_PPV_ARGS(TemporalRootSignature.GetAddressOf()));
				if (FAILED(result)) return FailHRESULT("CreateRootSignature(TAA)", result);
			}
			if (!LoadTechniqueShaderSource("DX12", "TAA.hlsl", ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex, pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = TemporalRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 2;
			pipeline.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
			pipeline.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT;
			pipeline.SampleDesc.Count = 1;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline, IID_PPV_ARGS(TemporalPipeline.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(TAA)", result);
		}

		bool UploadMotionConstants(const MotionBlurConstants& constants)
		{
			const auto offset = AlignUp(ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
			if (offset > ConstantBufferCapacity || sizeof(constants) > ConstantBufferCapacity - offset)
				return Fail("DX12 motion constants exhausted the frame upload buffer");
			std::memcpy(ConstantData + offset, &constants, sizeof(constants));
			CommandList->SetGraphicsRootConstantBufferView(0, ConstantBuffer->GetGPUVirtualAddress() + offset);
			ConstantOffset = offset + sizeof(constants);
			return true;
		}

		bool ReadMotionVectors()
		{
			D3D12_RANGE readable {0, static_cast<SIZE_T>(MotionReadback.TotalSize)};
			void* mapped = nullptr;
			const HRESULT result = MotionReadback.Buffer->Map(0, &readable, &mapped);
			if (FAILED(result)) return FailHRESULT("Map(motion readback)", result);
			const auto decode = [](std::uint16_t packed)
			{
				const unsigned exponent = (packed >> 10) & 31u, mantissa = packed & 1023u;
				float value = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24) :
					std::ldexp(static_cast<float>(mantissa + 1024u), static_cast<int>(exponent) - 25);
				if (exponent == 31) value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
				return packed & 32768u ? -value : value;
			};
			PendingFrame.MotionVectors.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height);
			for (int y = 0; y < CreateInfo.Height; ++y)
			{
				const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(mapped) +
					MotionReadback.Footprint.Offset + static_cast<std::size_t>(y) * MotionReadback.Footprint.Footprint.RowPitch);
				for (int x = 0; x < CreateInfo.Width; ++x)
					PendingFrame.MotionVectors[static_cast<std::size_t>(y) * CreateInfo.Width + x] =
                        DecodeMotionVector({decode(row[x * 4]), decode(row[x * 4 + 1]), decode(row[x * 4 + 2]), decode(row[x * 4 + 3])});
			}
			const D3D12_RANGE noWrite {0, 0};
			MotionReadback.Buffer->Unmap(0, &noWrite);
			return true;
		}

		bool PrepareHdrFrame(const Techniques::HdrSettings& settings)
		{
			HdrSceneActive = Techniques::RequiresHDR(settings);
			if (!HdrSceneActive) return true;
			D3D12_FEATURE_DATA_FORMAT_SUPPORT format {DXGI_FORMAT_R16G16B16A16_FLOAT};
			const UINT required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE
				| D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE;
			if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format)))
				|| (format.Support1 & required) != required)
				return Fail("DX12 requires sampleable, blendable RGBA16F render targets for HDR");
			if (!CreateHdrSurface(HdrSurfaces[HdrScene], CreateInfo.Width, CreateInfo.Height)
				|| !CreateHdrSurface(HdrSurfaces[HdrScratch], CreateInfo.Width, CreateInfo.Height)) return false;
			if (settings.Bloom || Techniques::UsesLensFlare(settings))
			{
				const int width = std::max(1, (CreateInfo.Width + 1) / 2);
				const int height = std::max(1, (CreateInfo.Height + 1) / 2);
				if (!CreateHdrSurface(HdrSurfaces[BloomFirst], width, height)
					|| !CreateHdrSurface(HdrSurfaces[BloomSecond], width, height)) return false;
			}
			for (auto pass : Techniques::BuildHdrPlan(settings)) if (!EnsureHdrPipeline(pass)) return false;
			return true;
		}

		bool ApplyHdr(const Techniques::HdrSettings& settings)
		{
			if (!HdrSceneActive) return true;
			const auto plan = Techniques::BuildHdrPlan(settings);
			CommandList->SetGraphicsRootSignature(HdrRootSignature.Get());
			CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			std::size_t scene = HdrScene;
			for (auto pass : plan)
			{
				std::size_t sourceIndex = scene;
				std::size_t targetIndex = scene == HdrScene ? HdrScratch : HdrScene;
				const bool tone = pass == Techniques::HdrPassKind::ToneMapping;
				const auto resources = DescribeHdrPass(pass);
				if (resources.SourceHalf >= 0) sourceIndex = BloomFirst + resources.SourceHalf;
				if (resources.DestinationHalf >= 0) targetIndex = BloomFirst + resources.DestinationHalf;
				auto& source = HdrSurfaces[sourceIndex];
				const HdrConstants constants = BuildHdrConstants(settings, source.Width, source.Height, pass, CreateInfo.Width, CreateInfo.Height);
				CommandList->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
				Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				if (tone) Transition(MainColor.Get(), MainColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
				else Transition(HdrSurfaces[targetIndex].Resource.Get(), HdrSurfaces[targetIndex].State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				UINT bloomDescriptor = NullColorSRV;
				if (resources.CompositeHalf >= 0)
				{
					auto& bloom = HdrSurfaces[BloomFirst + resources.CompositeHalf];
					Transition(bloom.Resource.Get(), bloom.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					bloomDescriptor = bloom.SRV;
				}
				CommandList->SetGraphicsRootDescriptorTable(1, SRVArena.Gpu(source.SRV));
				CommandList->SetGraphicsRootDescriptorTable(2, SRVArena.Gpu(bloomDescriptor));
				const auto target = RTVArena.Cpu(tone ? MainColorRTV : HdrSurfaces[targetIndex].RTV);
				CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				const int width = tone ? CreateInfo.Width : HdrSurfaces[targetIndex].Width;
				const int height = tone ? CreateInfo.Height : HdrSurfaces[targetIndex].Height;
				const D3D12_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
				const D3D12_RECT scissor {0, 0, width, height};
				CommandList->RSSetViewports(1, &viewport);
				CommandList->RSSetScissorRects(1, &scissor);
				CommandList->SetPipelineState(HdrPipelines[HdrProgramIndex(pass)].Get());
				CommandList->DrawInstanced(3, 1, 0, 0);
				if (resources.UpdatesScene) scene = targetIndex;
				PendingFrame.HDRPasses.push_back(pass);
			}
			PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
			return true;
		}

		bool EnsurePostResources()
		{
			if (PostColor) return true;
			if (PostColorRTV == InvalidDescriptor) PostColorRTV = RTVArena.Allocate();
			if (PostColorSRV == InvalidDescriptor) PostColorSRV = SRVArena.Allocate();
			if (MainColorSRV == InvalidDescriptor) MainColorSRV = SRVArena.Allocate();
			if (PostColorRTV == InvalidDescriptor || PostColorSRV == InvalidDescriptor || MainColorSRV == InvalidDescriptor)
				return Fail("DX12 post-processing descriptor allocation failed");
			const D3D12_RESOURCE_DESC description = MainColor->GetDesc();
			const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
			D3D12_CLEAR_VALUE clear {};
			clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			clear.Color[3] = 1.0f;
			ComPtr<ID3D12Resource> color;
			HRESULT result = Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
				D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(color.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("CreateCommittedResource(post pingpong)", result);
			D3D12_RENDER_TARGET_VIEW_DESC rtv {};
			rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
			Device->CreateRenderTargetView(color.Get(), &rtv, RTVArena.Cpu(PostColorRTV));
			D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
			srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srv.Texture2D.MipLevels = 1;
			Device->CreateShaderResourceView(color.Get(), &srv, SRVArena.Cpu(PostColorSRV));
			Device->CreateShaderResourceView(MainColor.Get(), &srv, SRVArena.Cpu(MainColorSRV));
			PostColor = std::move(color);
			PostColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			return true;
		}

		bool EnsurePostRootSignature()
		{
			if (PostRootSignature) return true;
			D3D12_DESCRIPTOR_RANGE imageRange {};
			imageRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
			imageRange.NumDescriptors = 1;
			imageRange.BaseShaderRegister = 0;
			imageRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
			D3D12_ROOT_PARAMETER parameters[2] {};
			parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			parameters[0].Constants.ShaderRegister = 0;
			parameters[0].Constants.Num32BitValues = sizeof(PostProcessConstants) / sizeof(std::uint32_t);
			parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			parameters[1].DescriptorTable.NumDescriptorRanges = 1;
			parameters[1].DescriptorTable.pDescriptorRanges = &imageRange;
			parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_STATIC_SAMPLER_DESC sampler {};
			sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
			sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
			sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			sampler.MaxLOD = D3D12_FLOAT32_MAX;
			sampler.ShaderRegister = 0;
			sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_ROOT_SIGNATURE_DESC description {};
			description.NumParameters = static_cast<UINT>(std::size(parameters));
			description.pParameters = parameters;
			description.NumStaticSamplers = 1;
			description.pStaticSamplers = &sampler;
			description.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
			ComPtr<ID3DBlob> serialized;
			ComPtr<ID3DBlob> errors;
			HRESULT result = SerializeRootSignatureFunction(&description, D3D_ROOT_SIGNATURE_VERSION_1,
				serialized.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				else FailHRESULT("SerializeRootSignature(post)", result);
				return false;
			}
			result = Evans::Renderer::CreateCachedRootSignature(Device,0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
				IID_PPV_ARGS(PostRootSignature.GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateRootSignature(post)", result);
		}

		bool EnsurePostPipeline(Techniques::PostProcessPassKind pass)
		{
			const std::size_t index = static_cast<std::size_t>(pass);
			const char* name = PostProcessTechniqueFile(pass);
			if (index >= PostProcessTechniqueCount || !name) return Fail("Unknown DX12 post technique");
			if (PostPipelines[index]) return true;
			if (!EnsurePostRootSignature()) return false;
			const std::string file = std::string(name) + ".hlsl";
			if (!LoadTechniqueShaderSource("DX12", file.c_str(), ShaderCode, LastError)) return false;
			ComPtr<ID3DBlob> vertex;
			ComPtr<ID3DBlob> pixel;
			if (!CompileShader("VSMain", "vs_5_0", vertex) || !CompileShader("PSMain", "ps_5_0", pixel)) return false;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
			pipeline.pRootSignature = PostRootSignature.Get();
			pipeline.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
			pipeline.PS = {pixel->GetBufferPointer(), pixel->GetBufferSize()};
			auto& blend = pipeline.BlendState.RenderTarget[0];
			blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
			blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
			blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blend.LogicOp = D3D12_LOGIC_OP_NOOP;
			blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			pipeline.SampleMask = UINT_MAX;
			pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
			pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
			pipeline.RasterizerState.DepthClipEnable = TRUE;
			pipeline.DepthStencilState.DepthEnable = FALSE;
			pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
			pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
			pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
			pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
			pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			pipeline.NumRenderTargets = 1;
			pipeline.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			pipeline.SampleDesc.Count = 1;
			const HRESULT result = Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,
				IID_PPV_ARGS(PostPipelines[index].GetAddressOf()));
			return SUCCEEDED(result) || FailHRESULT("CreateGraphicsPipelineState(post)", result);
		}

		bool ApplyPostProcessing(const Techniques::PostProcessSettings& settings, bool& outputIsMain)
		{
			outputIsMain = true;
			const auto plan = Techniques::BuildPostProcessPlan(settings);
			if (plan.empty()) return true;
			if (!EnsurePostResources()) return false;
			for (auto pass : plan) if (!EnsurePostPipeline(pass)) return false;
			const PostProcessConstants constants = BuildPostProcessConstants(settings, CreateInfo.Width, CreateInfo.Height);
			CommandList->SetGraphicsRootSignature(PostRootSignature.Get());
			CommandList->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
			const D3D12_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(CreateInfo.Width), static_cast<float>(CreateInfo.Height), 0.0f, 1.0f};
			const D3D12_RECT scissor {0, 0, CreateInfo.Width, CreateInfo.Height};
			CommandList->RSSetViewports(1, &viewport);
			CommandList->RSSetScissorRects(1, &scissor);
			CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			for (auto pass : plan)
			{
				// The separate pingpong image prevents read/write feedback. Both
				// native views are sRGB, so filtering matches the other APIs.
				if (outputIsMain)
				{
					Transition(MainColor.Get(), MainColorState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					Transition(PostColor.Get(), PostColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
				}
				else
				{
					Transition(PostColor.Get(), PostColorState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					Transition(MainColor.Get(), MainColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
				}
				const auto target = RTVArena.Cpu(outputIsMain ? PostColorRTV : MainColorRTV);
				CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				CommandList->SetGraphicsRootDescriptorTable(1, SRVArena.Gpu(outputIsMain ? MainColorSRV : PostColorSRV));
				CommandList->SetPipelineState(PostPipelines[static_cast<std::size_t>(pass)].Get());
				CommandList->DrawInstanced(3, 1, 0, 0);
				outputIsMain = !outputIsMain;
				PendingFrame.PostProcessingPasses.push_back(pass);
			}
			PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
			return true;
		}

        struct GpuSurface
        {
            ComPtr<ID3D12Resource> Resource;
            HANDLE Handle = nullptr;
            ~GpuSurface() { if (Handle) CloseHandle(Handle); }
        };
        std::vector<std::shared_ptr<GpuSurface>> GpuSurfaces;
        bool PublishGpuColor(ID3D12Resource* source)
        {
            const auto description = source->GetDesc();
            std::shared_ptr<GpuSurface> surface;
            for (const auto& item : GpuSurfaces)
                if (item.use_count() == 1 && item->Resource->GetDesc().Width == description.Width &&
                    item->Resource->GetDesc().Height == description.Height) { surface=item; break; }
            if (!surface)
            {
                GpuSurfaces.erase(std::remove_if(GpuSurfaces.begin(),GpuSurfaces.end(),
                    [](const auto& item) { return item.use_count()==1; }),GpuSurfaces.end());
                surface=std::make_shared<GpuSurface>();
                auto desc=description; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
                const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
                auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,
                    D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(surface->Resource.GetAddressOf()));
                if (FAILED(status)) return FailHRESULT("Create shared DX12 viewport",status);
                status=Device->CreateSharedHandle(surface->Resource.Get(),nullptr,GENERIC_ALL,nullptr,&surface->Handle);
                if (FAILED(status)) return FailHRESULT("Export DX12 viewport",status);
                GpuSurfaces.push_back(surface);
            }
            auto state=D3D12_RESOURCE_STATE_COMMON;
            Transition(surface->Resource.Get(),state,D3D12_RESOURCE_STATE_COPY_DEST);
            CommandList->CopyResource(surface->Resource.Get(),source);
            Transition(surface->Resource.Get(),state,D3D12_RESOURCE_STATE_COMMON);
            auto image=std::make_shared<GpuImage>(); image->Backend=RHIBackend::DX12;
            image->SharedHandle=surface->Handle; image->Owner=surface; PendingFrame.GpuColor=std::move(image);
            return true;
        }

		bool ReadColor(ReadbackResource& readback, int width, int height,
			std::vector<std::uint8_t>& output)
		{
			D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(readback.TotalSize) };
			void* mapped = nullptr;
			const HRESULT result = readback.Buffer->Map(0, &readRange, &mapped);
			if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(color readback)", result);
			const std::size_t rowSize = static_cast<std::size_t>(width) * 4;
			output.resize(rowSize * static_cast<std::size_t>(height));
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * rowSize,
					static_cast<const std::uint8_t*>(mapped) + readback.Footprint.Offset
					+ static_cast<std::size_t>(row) * readback.Footprint.Footprint.RowPitch, rowSize);
			}
			D3D12_RANGE noWrite = { 0, 0 };
			readback.Buffer->Unmap(0, &noWrite);
			return true;
		}

		bool ReadDepth(ReadbackResource& readback, int width, int height, std::vector<float>& output)
		{
			D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(readback.TotalSize) };
			void* mapped = nullptr;
			const HRESULT result = readback.Buffer->Map(0, &readRange, &mapped);
			if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(depth readback)", result);
			const std::size_t rowSize = static_cast<std::size_t>(width) * sizeof(float);
			output.resize(static_cast<std::size_t>(width) * height);
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * width,
					static_cast<const std::uint8_t*>(mapped) + readback.Footprint.Offset
					+ static_cast<std::size_t>(row) * readback.Footprint.Footprint.RowPitch, rowSize);
			}
			D3D12_RANGE noWrite = { 0, 0 };
			readback.Buffer->Unmap(0, &noWrite);
			return true;
		}

		bool ReadVSMMoments(VSMMomentMap& map)
		{
			D3D12_RANGE range {0, static_cast<SIZE_T>(map.Diagnostic.TotalSize)};
			void* data = nullptr;
			const auto result = map.Diagnostic.Buffer->Map(0, &range, &data);
			if (FAILED(result)) return FailHRESULT("Map(VSM moment readback)", result);
			auto& output = PendingFrame.ShadowMoments;
			output.resize(static_cast<std::size_t>(map.Filtered.Width) * map.Filtered.Height);
			for (int row = 0; row < map.Filtered.Height; ++row)
				std::memcpy(output.data() + static_cast<std::size_t>(row) * map.Filtered.Width,
					static_cast<const std::uint8_t*>(data) + map.Diagnostic.Footprint.Offset +
					static_cast<std::size_t>(row) * map.Diagnostic.Footprint.Footprint.RowPitch,
					static_cast<std::size_t>(map.Filtered.Width) * sizeof(Float2));
			D3D12_RANGE noWrite {0, 0}; map.Diagnostic.Buffer->Unmap(0, &noWrite);
			return true;
		}

		void ReleaseAll()
		{
            // Shutdown has already drained submitted work, or confirmed device
            // removal. Discard any remaining list before rolling back producers.
            if (CommandList && FrameOpen) (void)CommandList->Close();
            CommandList.Reset();
            std::string forwardError;
            if (ForwardPlusRecordOpen && !FrameWasSubmitted) (void)(ClusteredActive ? ClusteredForward.AbortFrame(forwardError) : TileProducer().AbortFrame(forwardError));
            (void)ForwardPlus.Shutdown(forwardError); (void)TiledDeferred.Shutdown(forwardError); (void)ClusteredForward.Shutdown(forwardError);
            LightListFrame.Reset(); ForwardPlusVertex.Reset(); ClusteredVertex.Reset(); TiledVertex.Reset();
            for (auto& filter : ForwardPlusPixels) for (auto& pixel : filter) pixel.Reset();
            for (auto& filter : ClusteredPixels) for (auto& pixel : filter) pixel.Reset();
            for (auto& filter : TiledPixels) for (auto& pixel : filter) pixel.Reset();
            ForwardPlusSceneView = ForwardPlusHeaderView = ForwardPlusIndexView = InvalidDescriptor;
            ForwardPlusEmptyHeaderView = ForwardPlusEmptyIndexView = ForwardPlusEmptyContactView = InvalidDescriptor;
            LightListsActive = ForwardPlusDepthWritten = ForwardPlusRecordOpen = ForwardPlusListsRecorded = false;
			for (auto& page : InstancePages) if (page.Buffer && page.Data) page.Buffer->Unmap(0, nullptr);
			InstancePages.clear(); InstancePageIndex = InstancePageOffset = 0;
            DestroyGpuDriven();
            DestroyVisibilityBuffer();
			for (auto& shader : InstanceBytecodes) shader.Reset();
			if (ConstantBuffer && ConstantData) ConstantBuffer->Unmap(0, nullptr);
			ConstantData = nullptr; ConstantBufferCapacity=DefaultConstantBufferCapacity;
			Buffers.clear();
			Textures.clear();
			Pipelines.clear();
			FrameUploads.clear();
			LastFrame = {};
			PendingFrame = {};
			MainColorReadback = {};
			MainDepthReadback = {};
            GpuSurfaces.clear();
			MainColor.Reset();
			PostColor.Reset();
			PostRootSignature.Reset();
			for (auto& surface : HdrSurfaces) surface = {};
			for (auto& surface : GBufferSurfaces) surface = {};
			for (auto& surface : AmbientOcclusionSurfaces) surface = {};
			DestroySaoDepthHierarchy();
			for (auto& technique : AmbientOcclusionPipelines)
				for (auto& pipeline : technique) pipeline.Reset();
			AmbientOcclusionRootSignature.Reset();
            DestroyPathTracing();
        DestroyRayGI(); DestroyVoxelGI();
            DestroyRayReflections(); DestroyHybridRendering(); DestroySubsurface(); DestroyHairFur(); DestroyExposure(); DestroyParallax(); DestroyDisplacement(); DestroyTessellation(); DestroyVirtualGeometry(); DestroyNaniteFrameVisibility(); DestroyNaniteGeometry(); DestroyTSR(); DestroyVirtualShadowRequests(); DestroyVirtualShadowAtlases(); DestroyLumenProbes(); DestroyLumenRadianceTraces(); DestroyLumenLightingCaches(); DestroyLumenSurfaces(); DestroyLumenSurfaceLookup(); DestroyLumenDistanceScenes(); DestroyRayScene(); DestroyRayShadowImage(); DestroyPCF(); DestroyMaterials();
			ReflectionSurface = {};
			ReflectionRootSignature.Reset();
			for (auto& program : ReflectionPipelines) for (auto& pipeline : program) pipeline.Reset();
			CubeMipPipeline.Reset(); DestroyEnvironmentTargets(Environment);
			MapConversionPipeline.Reset(); MapMipPipeline.Reset(); DestroyEnvironmentMapTargets(EnvironmentMap);
			PlanarMipPipeline.Reset(); DestroyPlanarTargets(Planar); FramePlanar = {};
			PlanarActive = PlanarWritten = PlanarReady = false;
			FrameEnvironment = {}; EnvironmentFacesWritten = 0;
			EnvironmentActive = EnvironmentReady = false;
			FrameReflections = {};
			ReflectionActive = ReflectionApplied = false;
			MaterialCaptureActive = AmbientOcclusionActive = AmbientOcclusionApplied = SceneColorWritten = false;
			DeferredVertexBytecode.Reset();
			DeferredPixelBytecode.Reset();
			DeferredLightingPipelines.clear();
			MultisamplePipelines.clear(); MultisampleRootSignature.Reset(); MultisampleVertex.Reset();
			for (auto& pixel : MultisamplePixels) pixel.Reset();
			for (auto& surface : MultisampleSurfaces) surface = {};
			MultisampleDepth = {};
			MultisampleDepthDSV = MultisampleDepthReadDSV = InvalidDescriptor;
			FrameSampleCount = ActiveSampleCount = 1;
			DeferredRootSignature.Reset();
			for (auto& pipeline : HdrPipelines) pipeline.Reset();
			for (auto& pipeline : DepthOfFieldPipelines) pipeline.Reset();
			for (auto& pipeline : MotionScreenPipelines) pipeline.Reset();
			for (auto& surface : MotionSurfaces) surface = {};
			MotionRootSignature.Reset();
			MotionVertexBytecode.Reset();
			MotionPixelBytecode.Reset();
			MotionReadback = {};
			VelocityCaptured = MotionBlurResolved = false;
			TemporalColorHistory = {}; TemporalDepthHistory = {};
            GISurfaces = {}; GIRootSignature.Reset(); GIPipeline.Reset(); GITemporalPipeline.Reset();
            ProbeSurfaces = {}; Probes = {}; FrameProbeSamples.reset(); ProbeRootSignature.Reset(); ProbePipelines = {};
            ResetLightmaps(true); ResetRefraction(true); ResetPeeling(true);
            GI = {}; FrameGI = {}; GIActive = false;
			TemporalRootSignature.Reset(); TemporalPipeline.Reset();
			FogAtlas = {}; FogRootSignature.Reset(); FogPipeline.Reset(); MediaFogPipeline.Reset(); CloudFogPipeline.Reset();
			LightShaftMask = {}; LightShaftRootSignature.Reset(); LightShaftPipeline.Reset();
			FrameLightShafts = {}; LightShaftResolved = false;
			VolumeLightScratch = {}; VolumeLightRootSignature.Reset(); VolumeLightPipelines = {}; MediaVolumePipelines = {}; CloudVolumePipelines = {};
			PCSSVolumePipeline.Reset(); PCSSPixelBytecodes = {};
            VSMVolumePipeline.Reset(); VSMPixelBytecodes = {}; VSMCasterVertex.Reset(); VSMInstanceVertex.Reset();
            VSMFilterVertex.Reset(); VSMFilterPipeline.Reset(); VSMFilterRootSignature.Reset(); VSMMoments.clear();
            ContactShadows = {}; ContactDescriptor = InvalidDescriptor; FrameContact = {};
            ContactDepthWritten = ContactGenerated = ContactReceiverActive = false;
			FrameVolumeLighting = {};
			WeightedSurfaces = {}; WeightedRootSignature.Reset(); WeightedResolvePipelines.clear();
			for (auto& shader : WeightedPixelBytecodes) shader.Reset();
			FrameTransparency = {}; WeightedActive = WeightedPass = WeightedPrepared = WeightedAccumulated = WeightedResolved = false;
			FrameFog = {}; FogConstants = {}; FogResolved = false;
			TemporalHistoryReadIndex = 0;
            TemporalHistoryValid = TemporalResolved = false;
            GI.Invalidate(); Probes.Invalidate();
            for (auto& surface : ProbeSurfaces) { surface.Resource.Reset(); surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET; }
            for (auto& surface : GISurfaces)
            {
                surface.Resource.Reset();
                surface.State = surface.FrameStartState = D3D12_RESOURCE_STATE_RENDER_TARGET;
            }
			FramePostProcessing = {};
			DepthOfFieldResolved = false;
			HdrRootSignature.Reset();
			for (auto& pipeline : PostPipelines) pipeline.Reset();
			PostColorRTV = PostColorSRV = MainColorSRV = InvalidDescriptor;
			MainDepthSRV = InvalidDescriptor;
			MainDepthReadOnlyDSV = InvalidDescriptor;
			MainDepth.Reset();
			ConstantBuffer.Reset();
			RootSignature.Reset();
			MaterialSamplers.Reset();
			for (auto& shader : VertexBytecodes) shader.Reset();
			for (auto& shader : PixelBytecodes) shader.Reset();
			ShadowVertexBytecode.Reset(); ShadowPixelBytecode.Reset();
			CascadeVertexBytecode.Reset(); CascadePixelBytecode.Reset();
			FrameCascades = {}; CascadeConstantAddresses = {};
			OutlinePixelBytecode.Reset();
			RTVArena = {};
			DSVArena = {};
			SRVArena = {};
            DestroyFrameTiming();
			Fence.Reset();
			CommandList.Reset();
			CommandAllocator.Reset();
			CommandQueue.Reset();
            LumenRayDevice.Reset(); DestroyDXILCompiler();
			Device.Reset();
			Adapter.Reset();
			Factory.Reset();
			if (FenceEvent)
			{
				CloseHandle(FenceEvent);
				FenceEvent = nullptr;
			}
			CreateDeviceFunction = nullptr;
			SerializeRootSignatureFunction = nullptr;
			if (D3D12Module)
			{
				FreeLibrary(D3D12Module);
				D3D12Module = nullptr;
			}
			Initialized = false;
			FrameOpen = false;
			PassOpen = false;
			FrameSnapshotValid = false;
			FrameWasSubmitted = false;
			SubmittedWorkPending = false;
		}

		HMODULE D3D12Module = nullptr;
		D3D12CreateDeviceFunction CreateDeviceFunction = nullptr;
		D3D12SerializeRootSignatureFunction SerializeRootSignatureFunction = nullptr;
		ComPtr<IDXGIFactory4> Factory;
		ComPtr<IDXGIAdapter1> Adapter;
		ComPtr<ID3D12Device> Device;
		ComPtr<ID3D12CommandQueue> CommandQueue;
		ComPtr<ID3D12CommandAllocator> CommandAllocator;
		ComPtr<ID3D12GraphicsCommandList> CommandList;
		ComPtr<ID3D12Fence> Fence;
		HANDLE FenceEvent = nullptr;
		UINT64 FenceValue = 0;
		DescriptorArena RTVArena;
		DescriptorArena DSVArena;
		DescriptorArena SRVArena;
		UINT MainColorRTV = InvalidDescriptor;
		UINT MainDepthDSV = InvalidDescriptor;
		UINT MainDepthReadOnlyDSV = InvalidDescriptor;
		UINT MainDepthSRV = InvalidDescriptor;
		UINT NullShadowSRV = InvalidDescriptor;
		UINT NullColorSRV = InvalidDescriptor;
		UINT NullArraySRV = InvalidDescriptor;
		ComPtr<ID3D12Resource> MainColor;
		ComPtr<ID3D12Resource> PostColor;
		D3D12_RESOURCE_STATES PostColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
		D3D12_RESOURCE_STATES FrameStartMainColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
		D3D12_RESOURCE_STATES FrameStartPostColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
		D3D12_RESOURCE_STATES FrameStartMainDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		bool FrameSnapshotValid = false;
		bool FrameWasSubmitted = false;
		bool SubmittedWorkPending = false;
		UINT PostColorRTV = InvalidDescriptor;
		UINT PostColorSRV = InvalidDescriptor;
		UINT MainColorSRV = InvalidDescriptor;
		ComPtr<ID3D12RootSignature> PostRootSignature;
		std::array<HdrSurface, HdrSurfaceCount> HdrSurfaces;
		std::array<HdrSurface, GBufferAttachmentCount> GBufferSurfaces;
		ComPtr<ID3DBlob> DeferredVertexBytecode;
		ComPtr<ID3DBlob> DeferredPixelBytecode;
		ComPtr<ID3D12RootSignature> DeferredRootSignature;
		std::unordered_map<UINT, ComPtr<ID3D12PipelineState>> DeferredLightingPipelines;
		static constexpr std::size_t MultisampleColor = 0, MultisampleVelocity = 1, MultisampleMaterialFirst = 2;
		std::array<HdrSurface, MultisampleMaterialFirst + GBufferAttachmentCount> MultisampleSurfaces;
		HdrSurface MultisampleDepth;
		UINT MultisampleDepthDSV = InvalidDescriptor, MultisampleDepthReadDSV = InvalidDescriptor;
		UINT FrameSampleCount = 1, ActiveSampleCount = 1;
		bool SamplesOpaqueResolved = false, SamplesVelocityResolved = false;
		bool SamplesColorRestored = false, SamplesColorResolved = false, SamplesTransparentWritten = false;
		std::array<ComPtr<ID3DBlob>, static_cast<std::size_t>(SampleProgram::Count)> MultisamplePixels;
		ComPtr<ID3DBlob> MultisampleVertex;
		ComPtr<ID3D12RootSignature> MultisampleRootSignature;
		std::unordered_map<UINT, ComPtr<ID3D12PipelineState>> MultisamplePipelines;
		bool DeferredActive = false;
		bool MaterialCaptureActive = false;
		static constexpr UINT ReflectionTextureCount = 11;
		HdrSurface ReflectionSurface;
		ComPtr<ID3D12RootSignature> ReflectionRootSignature;
		std::array<std::array<ComPtr<ID3D12PipelineState>, 2>, ReflectionProgramCount> ReflectionPipelines;
		ComPtr<ID3D12PipelineState> CubeMipPipeline;
		ComPtr<ID3D12PipelineState> MapConversionPipeline, MapMipPipeline;
		EnvironmentMapTargets EnvironmentMap;
		EnvironmentTargets Environment;
		EnvironmentCaptureFrame FrameEnvironment;
		PlanarTargets Planar;
		PlanarCaptureFrame FramePlanar;
		ComPtr<ID3D12PipelineState> PlanarMipPipeline;
		bool PlanarActive = false, PlanarWritten = false, PlanarReady = false;
		UINT EnvironmentFacesWritten = 0;
		bool EnvironmentActive = false, EnvironmentReady = false;
		Techniques::ReflectionSettings FrameReflections;
		Techniques::ShadingModel FrameReflectionShading = Techniques::ShadingModel::Phong;
		bool ReflectionActive = false, ReflectionApplied = false;
		bool AmbientOcclusionActive = false;
		bool AmbientOcclusionApplied = false;
		bool SceneColorWritten = false;
		std::array<HdrSurface, 2> AmbientOcclusionSurfaces;
		HdrSurface SaoDepthHierarchy, SaoDepthScratch;
		UINT SaoDepthMipCount = 0;
		static constexpr std::size_t HierarchyPipelineIndex = 2;
		static constexpr std::size_t AmbientOcclusionTechniqueCount = 5;
		Techniques::AmbientOcclusionSettings FrameAmbientOcclusion;
		std::array<std::array<ComPtr<ID3D12PipelineState>, 3>, AmbientOcclusionTechniqueCount> AmbientOcclusionPipelines;
		ComPtr<ID3D12RootSignature> AmbientOcclusionRootSignature;
		bool GBufferWritten = false;
		bool DeferredResolved = false;
		Float4 DeferredBackground;
		std::array<ComPtr<ID3D12PipelineState>, HdrProgramCount> HdrPipelines;
		ComPtr<ID3D12RootSignature> HdrRootSignature;
		std::array<ComPtr<ID3D12PipelineState>, 2> DepthOfFieldPipelines;
		static constexpr std::size_t MotionVelocity = 0, MotionTileMax = 1, MotionNeighborMax = 2;
		std::array<HdrSurface, 3> MotionSurfaces;
		std::array<ComPtr<ID3D12PipelineState>, 2> MotionScreenPipelines;
		ComPtr<ID3D12RootSignature> MotionRootSignature;
		ComPtr<ID3DBlob> MotionVertexBytecode, MotionPixelBytecode;
		ReadbackResource MotionReadback;
		bool VelocityCaptured = false, MotionBlurResolved = false;
		std::array<HdrSurface, 2> TemporalColorHistory, TemporalDepthHistory;
        static constexpr std::size_t GIRaw = 0, GIFiltered = 1, GIComposite = 2, GIColorFirst = 3,
            GIDepthFirst = 5, GINormalFirst = 7, GITargetCount = 9;
        std::array<HdrSurface, GITargetCount> GISurfaces;
        static constexpr std::size_t ProbeRaw=0,ProbeSH=1,ProbeMoments=2,ProbeComposite=3;
        std::array<HdrSurface,4> ProbeSurfaces;
        ProbeProjectionState Probes;
        std::shared_ptr<const BakedProbeSamples> FrameProbeSamples;
        ComPtr<ID3D12RootSignature> ProbeRootSignature;
        std::array<ComPtr<ID3D12PipelineState>,2> ProbePipelines;
        GlobalIlluminationHistory GI;
        Techniques::GlobalIlluminationSettings FrameGI;
        bool GIActive = false, EarlyTemporalActive = false, ProbeActive = false;
        ComPtr<ID3D12RootSignature> GIRootSignature;
        ComPtr<ID3D12PipelineState> GIPipeline, GITemporalPipeline;
		HdrSurface FogAtlas;
		HdrSurface LightShaftMask;
		ComPtr<ID3D12RootSignature> LightShaftRootSignature;
		ComPtr<ID3D12PipelineState> LightShaftPipeline;
		Techniques::LightShaftSettings FrameLightShafts;
		bool LightShaftResolved = false;
		std::array<HdrSurface, 2> VolumeLightScratch;
		ComPtr<ID3D12RootSignature> VolumeLightRootSignature;
		std::array<ComPtr<ID3D12PipelineState>, 3> VolumeLightPipelines, MediaVolumePipelines, CloudVolumePipelines;
		ComPtr<ID3D12PipelineState> PCSSVolumePipeline;
        ComPtr<ID3D12PipelineState> VSMVolumePipeline, VSMFilterPipeline;
        ComPtr<ID3D12RootSignature> VSMFilterRootSignature;
        ComPtr<ID3DBlob> VSMCasterVertex, VSMInstanceVertex, VSMFilterVertex;
        std::array<ComPtr<ID3DBlob>, VSMProgramCount + ShadingTechniqueCount> VSMPixelBytecodes;
        std::unordered_map<std::uint32_t, VSMMomentMap> VSMMoments;
		std::array<ComPtr<ID3DBlob>, PCSSProgramCount + ShadingTechniqueCount> PCSSPixelBytecodes;
		Techniques::VolumetricLightingSettings FrameVolumeLighting;
		ComPtr<ID3D12RootSignature> FogRootSignature;
		ComPtr<ID3D12PipelineState> FogPipeline, MediaFogPipeline, CloudFogPipeline;
		Techniques::VolumetricFogSettings FrameFog;
		VolumetricFogConstants FogConstants;
		bool FogResolved = false;
		std::array<HdrSurface, 2> WeightedSurfaces;
		ComPtr<ID3D12RootSignature> WeightedRootSignature;
		std::unordered_map<UINT, ComPtr<ID3D12PipelineState>> WeightedResolvePipelines;
		std::array<ComPtr<ID3DBlob>, StandaloneShadingTechniqueCount> WeightedPixelBytecodes;
		Techniques::TransparencySettings FrameTransparency;
		bool WeightedActive = false, WeightedPass = false, WeightedPrepared = false, WeightedAccumulated = false, WeightedResolved = false;
		ComPtr<ID3D12RootSignature> TemporalRootSignature;
		ComPtr<ID3D12PipelineState> TemporalPipeline;
		std::size_t TemporalHistoryReadIndex = 0;
		bool TemporalHistoryValid = false, TemporalResolved = false;
		Techniques::PostProcessSettings FramePostProcessing;
		bool DepthOfFieldResolved = false;
		bool HdrSceneActive = false;
		DXGI_FORMAT ActiveColorFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		std::array<ComPtr<ID3D12PipelineState>, PostProcessTechniqueCount> PostPipelines;
		ComPtr<ID3D12Resource> MainDepth;
		D3D12_RESOURCE_STATES MainColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
		D3D12_RESOURCE_STATES MainDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		ReadbackResource MainColorReadback;
		ReadbackResource MainDepthReadback;
		ComPtr<ID3D12RootSignature> RootSignature;
		std::array<ComPtr<ID3DBlob>, StandaloneShadingTechniqueCount> VertexBytecodes;
		std::array<ComPtr<ID3DBlob>, StandaloneShadingTechniqueCount> PixelBytecodes;
		ComPtr<ID3DBlob> ShadowVertexBytecode, ShadowPixelBytecode;
		ComPtr<ID3DBlob> CascadeVertexBytecode, CascadePixelBytecode;
		CascadedShadowState FrameCascades;
        Techniques::ShadowFilterSettings FrameShadowFilter;
        DX12ContactShadows ContactShadows;
        DX12ForwardPlus ForwardPlus;
        DX12ForwardPlus TiledDeferred {TiledLightShader::TiledDeferred};
        bool TiledActive = false;
        DX12ForwardPlus& TileProducer() { return TiledActive ? TiledDeferred : ForwardPlus; }
        DX12ClusteredForward ClusteredForward;
        bool ClusteredActive = false;
        LightListFrameState LightListFrame;
        bool LightListsActive = false, ForwardPlusDepthWritten = false;
        // These flags survive an unsuccessful queue signal or wait. Recovery
        // registers the eventual fence before any producer buffer can be reused.
        bool ForwardPlusRecordOpen = false, ForwardPlusListsRecorded = false;
        ComPtr<ID3DBlob> ForwardPlusVertex, ClusteredVertex, TiledVertex;
        std::array<std::array<ComPtr<ID3DBlob>, ForwardPlusProgramCount>, ForwardPlusFilterCount> ForwardPlusPixels, ClusteredPixels, TiledPixels;
        UINT ForwardPlusSceneView = InvalidDescriptor, ForwardPlusHeaderView = InvalidDescriptor, ForwardPlusIndexView = InvalidDescriptor;
        D3D12_GPU_VIRTUAL_ADDRESS ForwardPlusSceneAddress = 0;
        UINT ForwardPlusEmptyHeaderView = InvalidDescriptor, ForwardPlusEmptyIndexView = InvalidDescriptor, ForwardPlusEmptyContactView = InvalidDescriptor;
        Techniques::ContactShadowSettings FrameContact;
        UINT ContactDescriptor = InvalidDescriptor;
        bool ContactDepthWritten = false, ContactGenerated = false, ContactReceiverActive = false;
		std::array<D3D12_GPU_VIRTUAL_ADDRESS, 2> CascadeConstantAddresses {};
		ComPtr<ID3DBlob> OutlinePixelBytecode;
		std::unordered_map<std::uint64_t, ComPtr<ID3D12PipelineState>> Pipelines;
		std::array<ComPtr<ID3DBlob>, DX12InstanceProgramCount> InstanceBytecodes;
		struct InstanceUploadPage
		{
			ComPtr<ID3D12Resource> Buffer;
			std::uint8_t* Data = nullptr;
		};
		std::vector<InstanceUploadPage> InstancePages;
		std::size_t InstancePageIndex = 0, InstancePageOffset = 0;
		ComPtr<ID3D12Resource> ConstantBuffer;
        std::size_t ConstantBufferCapacity=DefaultConstantBufferCapacity;
		std::uint8_t* ConstantData = nullptr;
		std::size_t ConstantOffset = 0;
		std::vector<ComPtr<ID3D12Resource>> FrameUploads;
		std::unordered_map<std::uint32_t, BufferResource> Buffers;
		std::unordered_map<std::uint32_t, TextureResource> Textures;
		std::uint32_t NextBuffer = 0x12000001u;
		std::uint32_t NextTexture = 0x12800001u;
		RHICreateInfo CreateInfo;
		std::thread::id OwnerThread;
		PassDesc CurrentPass;
		TextureHandle CurrentPassTarget;
		PipelineState BoundPipeline;
		Matrix4 BoundWorld;
		Matrix4 BoundViewProjection;
		Matrix4 BoundLightViewProjection;
		BufferHandle BoundVertexBuffer;
		BufferHandle BoundIndexBuffer;
		TextureHandle LastShadowTexture;
		FrameCapture PendingFrame;
		FrameCapture LastFrame;
		std::uint64_t FrameNumber = 0;
		bool Initialized = false;
		RHICapabilities DeviceCapabilities;
		bool FrameAreaLights = false;
        bool FrameOpen = false;
		bool PassOpen = false;
		bool PipelineBound = false;
		bool WorldBound = false;
		bool ViewProjectionBound = false;
		bool LightViewProjectionBound = false;
		std::string ShaderCode;
		std::string LastError;
	};

	DX12RHI::DX12RHI() : m_Impl(std::make_unique<Impl>()) {}

	DX12RHI::~DX12RHI()
	{
		Shutdown();
		// A destructor cannot report an unresolved GPU wait. Retain the failed
		// device and its heaps until process exit instead of freeing in-flight data.
		if (m_Impl && m_Impl->SubmittedWorkPending) (void)m_Impl.release();
	}

	bool DX12RHI::Initialize(const RHICreateInfo& createInfo)
	{
		if (m_Impl->Initialized) return m_Impl->Fail("DX12 RHI is already initialized");
		if (createInfo.Width <= 0 || createInfo.Height <= 0)
			return m_Impl->Fail("DX12 dimensions must be positive");
		m_Impl->OwnerThread = std::this_thread::get_id();
		m_Impl->CreateInfo = createInfo;
		if (!m_Impl->LoadRuntime() || !m_Impl->CreateNativeDevice())
		{
			const std::string error = m_Impl->LastError;
			m_Impl->ReleaseAll();
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->Initialized = true;
		m_Impl->DeviceCapabilities = QueryDeviceCapabilities();
		m_Impl->LastError.clear();
		return true;
	}

	void DX12RHI::Shutdown()
	{
		if (!m_Impl) return;
		if (m_Impl->Initialized && m_Impl->OwnerThread != std::this_thread::get_id())
		{
			m_Impl->Fail("Shutdown must execute on the owning DX12 RHI thread");
			return;
		}
		if (!m_Impl->NaniteAsyncTasks.Empty() || !m_Impl->NaniteCacheTasks.Empty() || !m_Impl->LumenGlobalTasks.Empty() || !m_Impl->LumenLightingTasks.Empty() || !m_Impl->LumenCaptureTasks.Empty()) m_Impl->SubmittedWorkPending=true;
		if (m_Impl->SubmittedWorkPending && !m_Impl->RecoverSubmittedWork("Shutdown")
			&& (!m_Impl->Device || SUCCEEDED(m_Impl->Device->GetDeviceRemovedReason()))) return;
		m_Impl->ReleaseAll();
		SetCascadedShadowState({});
		SetEnvironmentCaptureFrame({});
		SetPlanarCaptureFrame({});
	}

	bool DX12RHI::IsInitialized() const { return m_Impl->Initialized; }
	RHIBackend DX12RHI::GetBackend() const { return RHIBackend::DX12; }
	const char* DX12RHI::GetName() const { return "DirectX 12 Native RHI"; }

	RHICapabilities DX12RHI::GetCapabilities() const
	{
		return m_Impl->Initialized ? m_Impl->DeviceCapabilities : QueryDeviceCapabilities();
	}

	RHICapabilities DX12RHI::QueryDeviceCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = true;
		capabilities.MaterialAnisotropicFiltering = m_Impl->Initialized && bool(m_Impl->MaterialSamplers);
    capabilities.TemporalEffects=m_Impl->Initialized && m_Impl->SupportsTemporalExposure();
		capabilities.InstancedRendering = true;
        capabilities.GPUDrivenRendering = m_Impl->Initialized;
        capabilities.VisibilityBufferRendering = m_Impl->Initialized && m_Impl->SupportsVisibilityBuffer();
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
		capabilities.VarianceShadowMaps = m_Impl->SupportsVSM();
        capabilities.ContactShadows = DX12ContactShadows::Supported(m_Impl->Device.Get());
        capabilities.ForwardPlusRendering = IsInitialized();
        capabilities.ClusteredForwardRendering = capabilities.ForwardPlusRendering;
		capabilities.DeferredShading = true;
    capabilities.ParallaxOcclusion=m_Impl->Initialized && capabilities.DeferredShading;
        capabilities.GeometryDisplacement=m_Impl->Initialized;
        capabilities.GeometryTessellation=m_Impl->Initialized;
        capabilities.GeometryVirtualization=m_Impl->Initialized;
        capabilities.NaniteRendering=capabilities.VisibilityBufferRendering && capabilities.GeometryVirtualization;
        capabilities.NaniteHardwareScheduling=capabilities.NaniteRendering;
        capabilities.MeshShaders=capabilities.VisibilityBufferRendering && m_Impl->SupportsMeshShaders();
        capabilities.VirtualShadowMaps=m_Impl->Initialized;
        capabilities.TiledDeferredRendering = capabilities.ForwardPlusRendering;
		capabilities.ScalableAmbientObscurance = m_Impl->SupportsFloatDepthTargets(true);
		capabilities.TemporalAntiAliasing = m_Impl->SupportsFloatDepthTargets(false);
        capabilities.TemporalSuperResolution=m_Impl->Initialized && capabilities.TemporalAntiAliasing;
        capabilities.ScreenSpaceGlobalIllumination = capabilities.DeferredShading && capabilities.TemporalAntiAliasing;
        capabilities.BakedLightProbes = capabilities.DeferredShading;
        capabilities.BakedLightmaps = capabilities.DeferredShading;
        capabilities.VoxelGI = capabilities.DeferredShading && m_Impl->Initialized && m_Impl->SupportsVoxelGI();
        capabilities.LumenGI=capabilities.DeferredShading && m_Impl->Initialized;
        capabilities.LumenHardwareRayTracing=capabilities.LumenGI && m_Impl->SupportsLumenHardwareRayTracing();
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
		capabilities.WeightedTransparencySampleCounts = m_Impl->WeightedSampleCounts();
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
		capabilities.SupportedSampleCounts = m_Impl->SupportedSampleCounts();
		return capabilities;
	}

	bool DX12RHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize")) return false;
		if (m_Impl->FrameOpen || width <= 0 || height <= 0)
			return m_Impl->Fail("DX12 resize is invalid while a frame is open or dimensions are non-positive");
		if (width == m_Impl->CreateInfo.Width && height == m_Impl->CreateInfo.Height) return true;
		if (!m_Impl->WaitForGPU() || !m_Impl->CreateMainTargets(width, height)) return false;
		m_Impl->Hair.ResetHistory();
    m_Impl->ExposureHistory.Invalidate();
    m_Impl->CreateInfo.Width = width;
		m_Impl->CreateInfo.Height = height;
		return true;
	}

	BufferHandle DX12RHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateBuffer")) return {};
		if (m_Impl->FrameOpen || desc.SizeInBytes == 0)
		{
			m_Impl->Fail("DX12 buffer creation is invalid while a frame is open or size is zero");
			return {};
		}
		Impl::BufferResource resource;
		resource.Description = desc;
		resource.Bytes.resize(desc.SizeInBytes);
		if (initialData) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
		// Default-heap buffers start in COMMON. UploadDirtyResources records the
		// copy transition; read-only buffers can promote on their first GPU use.
		resource.State = D3D12_RESOURCE_STATE_COMMON;
		if (!m_Impl->CreateBufferResource(desc.SizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
			resource.State, resource.Resource)) return {};
		resource.Dirty = initialData != nullptr;
		if (desc.Usage == BufferUsage::Vertex)
		{
			resource.VertexView.BufferLocation = resource.Resource->GetGPUVirtualAddress();
			resource.VertexView.SizeInBytes = static_cast<UINT>(desc.SizeInBytes);
			resource.VertexView.StrideInBytes = sizeof(Vertex);
		}
		else if (desc.Usage == BufferUsage::Index)
		{
			resource.IndexView.BufferLocation = resource.Resource->GetGPUVirtualAddress();
			resource.IndexView.SizeInBytes = static_cast<UINT>(desc.SizeInBytes);
			resource.IndexView.Format = DXGI_FORMAT_R32_UINT;
		}
		const BufferHandle handle { m_Impl->NextBuffer++ };
		m_Impl->Buffers.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX12RHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& candidate,VirtualGeometryCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache updates require a closed frame");
    return m_Impl->PrepareVirtualGeometryCache(upload,previous,candidate,statistics);
}
bool DX12RHI::SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& task,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Submit Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->SubmitNaniteCache(upload,previous,task,statistics);
}
bool DX12RHI::PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    if (!m_Impl->CheckThread("Poll Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->PollNaniteCache(task,ready,output);
}
bool DX12RHI::CancelNaniteCache(NaniteCacheTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->CancelNaniteCache(task);
}
bool DX12RHI::PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& candidate,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache updates require a closed frame");
    return m_Impl->PrepareNaniteCache(upload,previous,candidate,statistics);
}
bool DX12RHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache release requires a closed frame");
    return m_Impl->DestroyVirtualGeometryCache(handle);
}
bool DX12RHI::DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache release requires a closed frame");
    return m_Impl->DestroyNaniteCache(handle);
}
bool DX12RHI::SnapshotTSROpaque()
{
    if (!m_Impl->CheckThread("Snapshot TSR opaque")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || m_Impl->PassOpen) return m_Impl->Fail("TSR opaque snapshot requires a closed pass in an initialized frame");
    return m_Impl->SnapshotTSROpaque();
}
bool DX12RHI::RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& settings,FrameCapture& output)
{
    if (!m_Impl->CheckThread("Render TSR display")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR display work requires an initialized RHI and closed frame");
    return m_Impl->RenderTSRDisplay(handle,settings,output);
}
bool DX12RHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    if (!m_Impl->CheckThread("BuildVirtualShadowReceiverRequests")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow receivers require an initialized RHI and closed frame");
    return m_Impl->BuildVirtualShadowReceiverRequests(upload,output);
}
bool DX12RHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    if (!m_Impl->CheckThread("SampleVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow lookup requires an initialized RHI and closed frame");
    return m_Impl->SampleVirtualShadowAtlas(handle,upload,output);
}
bool DX12RHI::CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!m_Impl->CheckThread("CreateLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->CreateLumenDistanceScene(data,output);
}
bool DX12RHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    if (!m_Impl->CheckThread("CreateLumenGlobalDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CreateLumenGlobalDistanceScene(source,settings,output,previous);
}

bool DX12RHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenGlobalDistanceUpdate(source,settings,previous,task);
}

bool DX12RHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenGlobalDistanceUpdate(task,budget,ready,output,statistics);
}

bool DX12RHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{
    if (!m_Impl->CheckThread("CancelLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenGlobalDistanceUpdate(task);
}
bool DX12RHI::TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->TraceLumenDistanceScene(handle,upload,output);
}
bool DX12RHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenDistanceScene(handle);
}
bool DX12RHI::RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    if (!m_Impl->CheckThread("RelightLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->RelightLumenSurface(surface,distance,upload,output);
}

bool DX12RHI::BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenLightingUpdate(surface,distance,upload,task);
}

bool DX12RHI::AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenLightingUpdate(task,budget,ready,output,statistics);
}

bool DX12RHI::CancelLumenLightingUpdate(LumenLightingTask task)
{
    if (!m_Impl->CheckThread("CancelLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenLightingUpdate(task);
}
bool DX12RHI::ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenLighting(handle,output);
}
bool DX12RHI::DestroyLumenLighting(LumenLightingHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenLighting(handle);
}
bool DX12RHI::TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenRadiance")) return false;
    if (upload.Probes.DirectionalSide && m_Impl->FrameOpen) return m_Impl->Fail("Directional world cache generation requires a closed frame");
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->TraceLumenRadiance(lighting,upload,output);
}
bool DX12RHI::AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("AssembleLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("World radiance assembly requires an initialized RHI and closed frame");
    if (!m_Impl->ValidateLumenRecording(false)) return false;
    return m_Impl->AssembleLumenRadiance(assembly,output);
}
bool DX12RHI::ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbeOrigins")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbeOrigins(handle,output);
}
bool DX12RHI::ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    if (!m_Impl->CheckThread("ReadLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenRadiance(handle,output);
}
bool DX12RHI::DestroyLumenRadiance(LumenRadianceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenRadiance(handle);
}
bool DX12RHI::IntegrateLumenProbes(LumenRadianceHandle source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    if (!m_Impl->CheckThread("IntegrateLumenProbes")) return false;
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->IntegrateLumenProbes(source,upload,output);
}
bool DX12RHI::ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if (!m_Impl->CheckThread("ResolveLumenProbes")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || !m_Impl->ValidateLumenRecording(false)) return m_Impl->Fail("Lumen gather requires an open frame outside a graphics pass");
    return m_Impl->ResolveLumenProbes(handle,settings);
}
bool DX12RHI::ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbes(handle,output);
}
bool DX12RHI::DestroyLumenProbes(LumenProbeHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenProbes(handle);
}
bool DX12RHI::LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    if (!m_Impl->CheckThread("LookupLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface lookup requires an initialized RHI and closed frame");
    return m_Impl->LookupLumenSurface(handle,upload,output);
}
bool DX12RHI::CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!m_Impl->CheckThread("CaptureLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->CaptureLumenSurface(upload,output);
}
bool DX12RHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenCaptureUpdate(upload,task);
}
bool DX12RHI::AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenCaptureUpdate(task,budget,ready,output,statistics);
}
bool DX12RHI::CancelLumenCaptureUpdate(LumenCaptureTask task)
{
    if (!m_Impl->CheckThread("CancelLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenCaptureUpdate(task);
}
bool DX12RHI::ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->ReadLumenSurface(handle,output);
}
bool DX12RHI::DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenSurface(handle);
}
bool DX12RHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!m_Impl->CheckThread("RenderVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->RenderVirtualShadowAtlas(upload,previous,output);
}
bool DX12RHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->ReadVirtualShadowAtlas(handle,output);
}
bool DX12RHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!m_Impl->CheckThread("DestroyVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->DestroyVirtualShadowAtlas(handle);
}
bool DX12RHI::DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!m_Impl->CheckThread("Destroy TSR history")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR history release requires an initialized RHI and closed frame");
    return m_Impl->DestroyTSRHistory(handle);
}
bool DX12RHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle handle,const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached virtual geometry")) return false;
    if (m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateVirtualGeometry(upload,output,handle);
}
bool DX12RHI::SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& task)
{
    if (!m_Impl->CheckThread("Submit Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->SubmitNaniteGeometry(cache,upload,task);
}
bool DX12RHI::PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Poll Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->PollNaniteGeometry(task,ready,output);
}
bool DX12RHI::CancelNaniteGeometry(NaniteGeometryTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->CancelNaniteGeometry(task);
}
bool DX12RHI::CreateCachedNaniteGeometry(NaniteCacheHandle handle,const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateNaniteGeometry(upload,output,handle);
}

bool DX12RHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create virtual geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual geometry must be produced between frames");
    return m_Impl->CreateVirtualGeometry(upload,output);
}
bool DX12RHI::PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDraws") && m_Impl->PrepareNaniteDraws(visibility,draws,output);
}
bool DX12RHI::PrepareNaniteDrawBins(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDrawBins") && m_Impl->PrepareNaniteDraws(visibility,draws,output,true);
}
bool DX12RHI::DrawNaniteBins(NaniteDrawHandle handle)
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
bool DX12RHI::DrawNaniteCluster(NaniteDrawHandle handle,unsigned query)
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
bool DX12RHI::CaptureNaniteDepth(NaniteDepthHandle& output)
{
    if (!m_Impl->CheckThread("CaptureNaniteDepth") || !IsInitialized()) return false;
    return m_Impl->CaptureNaniteDepth(output);
}
bool DX12RHI::BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("BeginNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->BeginNaniteVisibility(history,queries,output);
}
bool DX12RHI::RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle depth,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("RetestNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->RetestNaniteVisibility(main,depth,output);
}
bool DX12RHI::ReadNaniteDepth(NaniteDepthHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDepthWords(found->second->Info,words,output,m_Impl->LastError);
}
bool DX12RHI::ReadNaniteVisibility(NaniteVisibilityHandle handle,std::vector<NaniteVisibilityState>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDecisionWords(found->second->Info,words,output,m_Impl->LastError);
}
bool DX12RHI::DestroyNaniteDepth(NaniteDepthHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDepths.erase(found); return true;
}
bool DX12RHI::DestroyNaniteVisibility(NaniteVisibilityHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDecisions.erase(found); return true;
}
bool DX12RHI::CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite geometry must be produced between frames");
    return m_Impl->CreateNaniteGeometry(upload,output);
}

bool DX12RHI::CreateTessellatedGeometry(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create tessellated geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Tessellated geometry must be prepared between frames");
    return m_Impl->CreateTessellated(geometry,settings,output);
}

bool DX12RHI::CreateDisplacedVertexBuffer(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    if (!m_Impl->CheckThread("Create displaced vertex buffer")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Displaced geometry must be prepared between frames");
    return m_Impl->CreateDisplaced(geometry,settings,output);
}

	bool DX12RHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer")) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		if (m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 buffers can only be updated between frames");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || !data || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("DX12 buffer update is invalid");
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
		iterator->second.LastValidRange = {};
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		iterator->second.Dirty = true;
		return true;
	}

	bool DX12RHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer")) return false;
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		if (m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 buffers can only be destroyed between frames");
		if (!buffer || m_Impl->Buffers.erase(buffer.Value) == 0)
			return m_Impl->Fail("DX12 buffer handle is invalid");
		return true;
	}

	TextureHandle DX12RHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture")) return {};
		if (m_Impl->FrameOpen || desc.Width <= 0 || desc.Height <= 0 || !ValidTextureMipLevels(desc, initialData))
		{
			m_Impl->Fail("DX12 texture creation is invalid");
			return {};
		}
		Impl::TextureResource resource;
		resource.Description = desc;
		resource.State = initialData ? D3D12_RESOURCE_STATE_COPY_DEST : m_Impl->FinalTextureState(desc);
		D3D12_RESOURCE_DESC nativeDesc = {};
		nativeDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		nativeDesc.Width = static_cast<UINT64>(desc.Width);
		nativeDesc.Height = static_cast<UINT>(desc.Height);
		nativeDesc.DepthOrArraySize = 1;
		nativeDesc.MipLevels = static_cast<UINT16>(desc.MipLevels);
		nativeDesc.SampleDesc.Count = 1;
		nativeDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		D3D12_CLEAR_VALUE clearValue = {};
		const D3D12_CLEAR_VALUE* clear = nullptr;
		if (desc.Format == TextureFormat::Depth32)
		{
			nativeDesc.Format = DXGI_FORMAT_R32_TYPELESS;
			nativeDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
			clearValue.Format = DXGI_FORMAT_D32_FLOAT;
			clearValue.DepthStencil.Depth = 1.0f;
			clear = &clearValue;
			resource.DSV = m_Impl->DSVArena.Allocate();
		}
		else
		{
			nativeDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			nativeDesc.Flags = desc.RenderTarget ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
				: D3D12_RESOURCE_FLAG_NONE;
			if (desc.RenderTarget)
			{
				clearValue.Format = desc.ColorSpace == TextureColorSpace::SRGB
					? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
					: DXGI_FORMAT_R8G8B8A8_UNORM;
				clearValue.Color[3] = 1.0f;
				clear = &clearValue;
				resource.RTV = m_Impl->RTVArena.Allocate();
			}
		}
		resource.SRV = m_Impl->SRVArena.Allocate();
		if (resource.SRV == InvalidDescriptor
			|| (desc.Format == TextureFormat::Depth32 && resource.DSV == InvalidDescriptor)
			|| (desc.Format == TextureFormat::RGBA8 && desc.RenderTarget && resource.RTV == InvalidDescriptor))
		{
			m_Impl->RTVArena.Release(resource.RTV);
			m_Impl->DSVArena.Release(resource.DSV);
			m_Impl->SRVArena.Release(resource.SRV);
			m_Impl->Fail("DX12 texture descriptor heap is exhausted");
			return {};
		}
		const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
		const HRESULT result = m_Impl->Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
			&nativeDesc, resource.State, clear, IID_PPV_ARGS(resource.Resource.GetAddressOf()));
		if (FAILED(result))
		{
			m_Impl->RTVArena.Release(resource.RTV);
			m_Impl->DSVArena.Release(resource.DSV);
			m_Impl->SRVArena.Release(resource.SRV);
			m_Impl->FailHRESULT("ID3D12Device::CreateCommittedResource(texture)", result);
			return {};
		}
		if (!m_Impl->CreateReadback(resource.Resource.Get(), resource.Readback))
		{
			m_Impl->RTVArena.Release(resource.RTV);
			m_Impl->DSVArena.Release(resource.DSV);
			m_Impl->SRVArena.Release(resource.SRV);
			return {};
		}
		if (desc.Format == TextureFormat::Depth32)
		{
			D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
			dsv.Format = DXGI_FORMAT_D32_FLOAT;
			dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			m_Impl->Device->CreateDepthStencilView(resource.Resource.Get(), &dsv,
				m_Impl->DSVArena.Cpu(resource.DSV));
			D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
			srv.Format = DXGI_FORMAT_R32_FLOAT;
			srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srv.Texture2D.MipLevels = 1;
			m_Impl->Device->CreateShaderResourceView(resource.Resource.Get(), &srv,
				m_Impl->SRVArena.Cpu(resource.SRV));
		}
		else
		{
			if (desc.RenderTarget)
			{
				D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
				rtv.Format = desc.ColorSpace == TextureColorSpace::SRGB
					? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
					: DXGI_FORMAT_R8G8B8A8_UNORM;
				rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
				m_Impl->Device->CreateRenderTargetView(resource.Resource.Get(), &rtv,
					m_Impl->RTVArena.Cpu(resource.RTV));
			}
			D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
			srv.Format = desc.ColorSpace == TextureColorSpace::SRGB
				? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
				: DXGI_FORMAT_R8G8B8A8_UNORM;
			srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srv.Texture2D.MipLevels = desc.MipLevels;
			m_Impl->Device->CreateShaderResourceView(resource.Resource.Get(), &srv,
				m_Impl->SRVArena.Cpu(resource.SRV));
		}
		if (initialData)
		{
			resource.Bytes.resize(TextureMipBytes(desc));
			std::memcpy(resource.Bytes.data(), initialData, resource.Bytes.size());
			resource.Dirty = true;
		}
		const TextureHandle handle { m_Impl->NextTexture++ };
		m_Impl->Textures.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX12RHI::DestroyTexture(TextureHandle texture)
	{
		if (!m_Impl->CheckThread("DestroyTexture")) return false;
		if (!m_Impl->RetireLumenCaptureInput(texture)) return false;
		if (m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 textures can only be destroyed between frames");
		auto iterator = m_Impl->Textures.find(texture.Value);
		if (!texture || iterator == m_Impl->Textures.end())
			return m_Impl->Fail("DX12 texture handle is invalid");
		m_Impl->DestroyVSMMoments(texture);
		m_Impl->RTVArena.Release(iterator->second.RTV);
		m_Impl->DSVArena.Release(iterator->second.DSV);
		m_Impl->SRVArena.Release(iterator->second.SRV);
		m_Impl->Textures.erase(iterator);
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
		return true;
	}

	bool DX12RHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame")) return false;
		if (m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 frame cannot begin in the current state");
		if (!m_Impl->CollectLumenGlobalDistance() || !m_Impl->CollectLumenLighting() || !m_Impl->CollectLumenCapture()) return false;
		ResetMotionTransform();
    m_Impl->ExposureHistory.Begin();
        // Deferred PSO selection precedes copy recording. Establish its frame
        // permutation now; the upload still publishes only after submission.
        m_Impl->FrameRayShadowImage=GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr;
        m_Impl->FrameShadowFilter = GetTechniqueSettings().ShadowFilter;
        if (GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced) m_Impl->FrameShadowFilter.Mode=Techniques::ShadowFilterMode::Hard;
        m_Impl->FrameContact = GetTechniqueSettings().ContactShadows;
        m_Impl->LightListFrame.Reset(); m_Impl->ForwardPlusDepthWritten = m_Impl->ForwardPlusListsRecorded = false;
        m_Impl->TiledActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::TiledDeferred;
        m_Impl->ClusteredActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::ClusteredForward;
        m_Impl->LightListsActive = Techniques::UsesLightLists(GetTechniqueSettings().Pipeline);
        m_Impl->ContactShadows.BeginFrame();
        m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
		if (!m_Impl->PrepareVirtualShadowSampling(GetVirtualShadowSamplingFrame(),GetCascadedShadowState()) ||
            !m_Impl->PrepareCascadedShadows(GetCascadedShadowState())) return false;
		m_Impl->SubsurfaceProducing=m_Impl->SubsurfaceResolved=m_Impl->SubsurfaceHasScene=false;
    m_Impl->FrameSubsurface=GetTechniqueSettings().Subsurface;
    m_Impl->SubsurfaceRequested=Techniques::UsesSubsurfaceScattering(m_Impl->FrameSubsurface);
    m_Impl->FramePostProcessing = GetTechniqueSettings().PostProcessing;
    if (!m_Impl->PrepareTSRScene(GetTSRSceneFrame(),m_Impl->FramePostProcessing)) return false;
        m_Impl->FrameGI = GetTechniqueSettings().GlobalIllumination;
        m_Impl->FrameLumenLighting=GetLumenLighting(); m_Impl->FrameLumenWorldCache=GetLumenWorldCache(); m_Impl->FrameLumenGlobalDistance=GetLumenGlobalDistance();
        m_Impl->PendingVoxelVolume.reset();
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
        m_Impl->FrameProbeSamples = GetBakedProbeSamples();
        m_Impl->LightmapActive = Techniques::UsesLightmaps(m_Impl->FrameGI);
        m_Impl->FrameLightmaps = GetLightmapFrame();
        m_Impl->FramePeeling=GetDepthPeelingFrame(); m_Impl->PeelingSequence.Abort();
        m_Impl->FrameRefraction = GetRefractionFrame();
        m_Impl->FrameRefractionSettings = GetTechniqueSettings().Refraction;
        m_Impl->RefractionActive = Techniques::UsesRefraction(GetTechniqueSettings());
        m_Impl->RefractionResolved = false;
        m_Impl->FrameLightmapSettings = GetTechniqueSettings();
        m_Impl->PendingUploadedLightmap.reset();
		m_Impl->FrameFog = GetTechniqueSettings().VolumetricFog;
		m_Impl->FrameVolumeLighting = GetTechniqueSettings().VolumetricLighting;
		m_Impl->FrameLightShafts = GetTechniqueSettings().LightShafts;
		m_Impl->LightShaftResolved = false;
		m_Impl->FogResolved = false;
		m_Impl->FrameAreaLights = GetTechniqueSettings().AreaLights.Enabled;
        m_Impl->FrameSampleCount = GetTechniqueSettings().SampleCount;
		m_Impl->FrameTransparency = GetTechniqueSettings().Transparency;
		m_Impl->WeightedActive = Techniques::UsesWeightedTransparency(GetTechniqueSettings());
		m_Impl->WeightedPass = m_Impl->WeightedPrepared = m_Impl->WeightedAccumulated = m_Impl->WeightedResolved = false;
		m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
		m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
		m_Impl->DepthOfFieldResolved = false;
		m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
		if (!m_Impl->PrepareHdrFrame(m_Impl->FramePostProcessing.HDR)) return false;
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->EnsureLightShaftResources()) return false;
		if (m_Impl->WeightedActive && !m_Impl->EnsureWeightedResources()) return false;
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->EnsureFogResources()) return false;
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)) &&
			!m_Impl->EnsureVolumeLightResources()) return false;
		m_Impl->TemporalResolved = false;
        m_Impl->GI.Begin(); m_Impl->Probes.Begin();
		if (((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive) && !m_Impl->EnsureVelocityResources()) return false;
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->EnsureTemporalResources()) return false;
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI) && !m_Impl->EnsureGIResources()) return false;
        if (m_Impl->ProbeActive && !m_Impl->EnsureProbeResources()) return false;
		if (m_Impl->FramePostProcessing.DepthOfField.Enabled && !m_Impl->EnsureDepthOfFieldResources()) return false;
		if (m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->EnsureMotionResources()) return false;
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
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->EnsureMultisampleResources()) return false;
		if (m_Impl->LightmapActive && !m_Impl->EnsureLightmapResources()) return false;
        if (m_Impl->FramePeeling.Enabled && !m_Impl->EnsurePeelingResources()) return false;
        if (m_Impl->RefractionActive && !m_Impl->EnsureRefractionResources()) return false;
		if (m_Impl->Hair.Active && !m_Impl->EnsureHairFur()) return false;
        m_Impl->FrameUploads.clear();
		HRESULT result = m_Impl->CommandAllocator->Reset();
		if (FAILED(result)) return m_Impl->FailHRESULT("ID3D12CommandAllocator::Reset", result);
		result = m_Impl->CommandList->Reset(m_Impl->CommandAllocator.Get(), nullptr);
		if (FAILED(result)) return m_Impl->FailHRESULT("ID3D12GraphicsCommandList::Reset", result);
        m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Backend = RHIBackend::DX12;
		m_Impl->PendingFrame.Statistics.RasterSampleCount = m_Impl->FrameSampleCount;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = true;
        m_Impl->BeginFrameTiming();
		m_Impl->FrameStartMainColorState = m_Impl->MainColorState;
		m_Impl->FrameStartPostColorState = m_Impl->PostColorState;
		m_Impl->FrameStartMainDepthState = m_Impl->MainDepthState;
		for (auto& surface : m_Impl->HdrSurfaces) surface.FrameStartState = surface.State;
		for (auto& entry : m_Impl->VSMMoments)
		{
			auto& map = entry.second;
			map.Ready = map.RawWritten = false;
			for (auto* surface : {&map.Raw, &map.Horizontal, &map.Filtered}) surface->FrameStartState = surface->State;
		}
		for (auto& surface : m_Impl->GBufferSurfaces) surface.FrameStartState = surface.State;
        for (auto& surface : m_Impl->VisibilityImages) surface.FrameStartState = surface.State;
		for (auto& surface : m_Impl->AmbientOcclusionSurfaces) surface.FrameStartState = surface.State;
		m_Impl->ReflectionSurface.FrameStartState = m_Impl->ReflectionSurface.State;
        for (auto& image:m_Impl->RayReflectionImages) image.FrameStartState=image.State;
        for (auto& image:m_Impl->HybridRenderingImages) image.FrameStartState=image.State;
        for (auto& image:m_Impl->SubsurfaceImages) image.FrameStartState=image.State;
        for (auto& image:m_Impl->RayGIImages) image.FrameStartState=image.State;
        for (auto& image:m_Impl->VoxelGIImages) image.FrameStartState=image.State;
        m_Impl->VoxelVolumeFrameState=m_Impl->VoxelVolumeState;
        for (auto& image:m_Impl->PathMeans) image.FrameStartState=image.State;
        m_Impl->PathDepth.FrameStartState=m_Impl->PathDepth.State;
		m_Impl->FogAtlas.FrameStartState = m_Impl->FogAtlas.State;
		m_Impl->LightShaftMask.FrameStartState = m_Impl->LightShaftMask.State;
		for (auto& surface : m_Impl->VolumeLightScratch) surface.FrameStartState = surface.State;
		for (auto& surface : m_Impl->WeightedSurfaces) surface.FrameStartState = surface.State;
		for (auto* surface : {&m_Impl->Environment.Cube, &m_Impl->Environment.Scratch, &m_Impl->Environment.Depth,
			&m_Impl->EnvironmentMap.Image, &m_Impl->EnvironmentMap.Scratch, &m_Impl->Planar.Image, &m_Impl->Planar.Scratch, &m_Impl->Planar.Depth})
			surface->FrameStartState = surface->State;
		for (auto& surface : m_Impl->MotionSurfaces) surface.FrameStartState = surface.State;
        for (auto& surface : m_Impl->GISurfaces) surface.FrameStartState = surface.State;
        for (auto& surface : m_Impl->ProbeSurfaces) surface.FrameStartState = surface.State;
        m_Impl->LumenGatherTarget.FrameStartState=m_Impl->LumenGatherTarget.State;
        for (auto* image:{&m_Impl->LumenHairIndirect,&m_Impl->LumenHairSamples}) image->FrameStartState=image->State;
        m_Impl->LumenVolumeAtlas.FrameStartState=m_Impl->LumenVolumeAtlas.State;
        m_Impl->LumenVisibilityFrameState=m_Impl->LumenVisibilityState;
        m_Impl->LumenTemporalFrameStates=m_Impl->LumenTemporalStates;
        m_Impl->LumenSpecularFrameStates=m_Impl->LumenSpecularStates; m_Impl->LumenReflectionEstimateFrameState=m_Impl->LumenReflectionEstimateState;
        for (auto& surface : m_Impl->LightmapSurfaces) surface.FrameStartState = surface.State;
        for (auto& surface : m_Impl->RefractionImages) surface.FrameStartState = surface.State;
        for (auto& surface : m_Impl->PeelingImages) surface.FrameStartState = surface.State;
		for (auto* history : {&m_Impl->TemporalColorHistory, &m_Impl->TemporalDepthHistory})
			for (auto& surface : *history) surface.FrameStartState = surface.State;
		for (auto& surface : m_Impl->MultisampleSurfaces) surface.FrameStartState = surface.State;
		m_Impl->MultisampleDepth.FrameStartState = m_Impl->MultisampleDepth.State;
		m_Impl->SaoDepthHierarchy.FrameStartState = m_Impl->SaoDepthHierarchy.State;
		m_Impl->SaoDepthScratch.FrameStartState = m_Impl->SaoDepthScratch.State;
		for (auto& pair : m_Impl->Buffers)
		{
			pair.second.FrameStartState = pair.second.State;
			pair.second.FrameStartDirty = pair.second.Dirty;
		}
		for (auto& pair : m_Impl->Textures)
		{
			pair.second.FrameStartState = pair.second.State;
			pair.second.FrameStartDirty = pair.second.Dirty;
		}
		m_Impl->FrameSnapshotValid = true;
		m_Impl->FrameWasSubmitted = false;
		m_Impl->ConstantOffset = 0;
		m_Impl->InstancePageIndex = m_Impl->InstancePageOffset = 0;
        m_Impl->GpuCursor = 0; m_Impl->ActiveGpuBatch = nullptr; ++m_Impl->GpuEpoch;
		m_Impl->UploadCascadedShadowConstants();
		m_Impl->LastError.clear();
		m_Impl->BindSceneDescriptorHeaps();
		if (!m_Impl->UploadDirtyResources())
		{
			AbortFrame();
			return false;
		}
		if (!m_Impl->PrepareRayShadowImage(GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr)) { AbortFrame(); return false; }
		if (m_Impl->FramePostProcessing.TemporalEffects.Enabled && !m_Impl->EnsureExposure()) return false;
    m_Impl->SnapshotExposureStates(false);
    return true;
	}

	void DX12RHI::AbortFrame()
	{
        m_Impl->PeelingSequence.Abort(); m_Impl->FramePeeling={};
		if (!m_Impl || !m_Impl->Initialized)
		{
			return;
		}
		if (m_Impl->OwnerThread != std::this_thread::get_id())
		{
			m_Impl->LastError =
				"AbortFrame must execute on the owning DX12 RHI thread";
			return;
		}
        m_Impl->PathFrame.Abort();
        // Present may fail after submission. Conservatively invalidate content
        // identity whenever an upload was recorded in an aborted transaction.
        if (m_Impl->PendingVoxelVolume) m_Impl->UploadedVoxelVolume.reset();
        m_Impl->PendingVoxelVolume.reset();
		if (m_Impl->FrameOpen)
		{
			(void)m_Impl->CommandList->Close();
		}
        if (m_Impl->ForwardPlusRecordOpen && !m_Impl->FrameWasSubmitted)
        {
            std::string error;
            if (!(m_Impl->ClusteredActive ? m_Impl->ClusteredForward.AbortFrame(error) : m_Impl->TileProducer().AbortFrame(error))) { m_Impl->LastError = error; return; }
            m_Impl->ForwardPlusRecordOpen = m_Impl->ForwardPlusListsRecorded = false;
        }
        m_Impl->LightListFrame.Reset(); m_Impl->LightListsActive = m_Impl->ForwardPlusDepthWritten = false;
		if (m_Impl->FrameSnapshotValid && !m_Impl->FrameWasSubmitted)
        {
            m_Impl->AbortGpuDriven();
            m_Impl->ContactShadows.AbortFrame();
			// Discarded commands changed neither GPU state nor uploaded contents.
			// Resource membership is stable while a frame is open. Preserve any
			// submitted state even if its subsequent fence wait fails.
			m_Impl->MainColorState = m_Impl->FrameStartMainColorState;
			m_Impl->PostColorState = m_Impl->FrameStartPostColorState;
			m_Impl->MainDepthState = m_Impl->FrameStartMainDepthState;
			for (auto& surface : m_Impl->HdrSurfaces) surface.State = surface.FrameStartState;
			for (auto& entry : m_Impl->VSMMoments)
				for (auto* surface : {&entry.second.Raw, &entry.second.Horizontal, &entry.second.Filtered})
					surface->State = surface->FrameStartState;
			for (auto& surface : m_Impl->GBufferSurfaces) surface.State = surface.FrameStartState;
            for (auto& surface : m_Impl->VisibilityImages) surface.State = surface.FrameStartState;
			for (auto& surface : m_Impl->AmbientOcclusionSurfaces) surface.State = surface.FrameStartState;
			m_Impl->ReflectionSurface.State = m_Impl->ReflectionSurface.FrameStartState;
            for (auto& image:m_Impl->RayReflectionImages) image.State=image.FrameStartState;
            for (auto& image:m_Impl->HybridRenderingImages) image.State=image.FrameStartState;
            for (auto& image:m_Impl->SubsurfaceImages) image.State=image.FrameStartState;
            m_Impl->SnapshotExposureStates(true);
        for (auto& image:m_Impl->RayGIImages) image.State=image.FrameStartState;
        for (auto& image:m_Impl->VoxelGIImages) image.State=image.FrameStartState;
        m_Impl->VoxelVolumeState=m_Impl->VoxelVolumeFrameState;
        for (auto& image:m_Impl->PathMeans) image.State=image.FrameStartState;
        m_Impl->PathDepth.State=m_Impl->PathDepth.FrameStartState;
			m_Impl->FogAtlas.State = m_Impl->FogAtlas.FrameStartState;
			m_Impl->LightShaftMask.State = m_Impl->LightShaftMask.FrameStartState;
			for (auto& surface : m_Impl->VolumeLightScratch) surface.State = surface.FrameStartState;
			for (auto& surface : m_Impl->WeightedSurfaces) surface.State = surface.FrameStartState;
			for (auto* surface : {&m_Impl->Environment.Cube, &m_Impl->Environment.Scratch, &m_Impl->Environment.Depth,
				&m_Impl->EnvironmentMap.Image, &m_Impl->EnvironmentMap.Scratch, &m_Impl->Planar.Image, &m_Impl->Planar.Scratch, &m_Impl->Planar.Depth})
				surface->State = surface->FrameStartState;
			for (auto& surface : m_Impl->MotionSurfaces) surface.State = surface.FrameStartState;
        for (auto& surface : m_Impl->GISurfaces) surface.State = surface.FrameStartState;
        for (auto& surface : m_Impl->ProbeSurfaces) surface.State = surface.FrameStartState;
        m_Impl->LumenGatherTarget.State=m_Impl->LumenGatherTarget.FrameStartState;
        for (auto* image:{&m_Impl->LumenHairIndirect,&m_Impl->LumenHairSamples}) image->State=image->FrameStartState;
        m_Impl->LumenVolumeAtlas.State=m_Impl->LumenVolumeAtlas.FrameStartState;
        m_Impl->LumenVisibilityState=m_Impl->LumenVisibilityFrameState;
        m_Impl->LumenTemporalStates=m_Impl->LumenTemporalFrameStates;
        m_Impl->LumenSpecularStates=m_Impl->LumenSpecularFrameStates; m_Impl->LumenReflectionEstimateState=m_Impl->LumenReflectionEstimateFrameState;
        for (auto& surface : m_Impl->LightmapSurfaces) surface.State = surface.FrameStartState;
        for (auto& surface : m_Impl->RefractionImages) surface.State = surface.FrameStartState;
        for (auto& surface : m_Impl->PeelingImages) surface.State = surface.FrameStartState;
			for (auto* history : {&m_Impl->TemporalColorHistory, &m_Impl->TemporalDepthHistory})
				for (auto& surface : *history) surface.State = surface.FrameStartState;
			for (auto& surface : m_Impl->MultisampleSurfaces) surface.State = surface.FrameStartState;
			m_Impl->MultisampleDepth.State = m_Impl->MultisampleDepth.FrameStartState;
			m_Impl->SaoDepthHierarchy.State = m_Impl->SaoDepthHierarchy.FrameStartState;
			m_Impl->SaoDepthScratch.State = m_Impl->SaoDepthScratch.FrameStartState;
			for (auto& pair : m_Impl->Buffers)
			{
				pair.second.State = pair.second.FrameStartState;
				pair.second.Dirty = pair.second.FrameStartDirty;
			}
			for (auto& pair : m_Impl->Textures)
			{
				pair.second.State = pair.second.FrameStartState;
				pair.second.Dirty = pair.second.FrameStartDirty;
			}
		}
        m_Impl->ContactShadows.BeginFrame();
        m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
		m_Impl->FrameSnapshotValid = false;
		m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
		for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
		m_Impl->WeightedActive = m_Impl->WeightedPass = m_Impl->WeightedPrepared = m_Impl->WeightedAccumulated = m_Impl->WeightedResolved = false;
		m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
		m_Impl->FrameCascades = {};
		SetCascadedShadowState({});
		SetEnvironmentCaptureFrame({});
		m_Impl->PendingFrame = {};
		m_Impl->FrameEnvironment = {}; m_Impl->EnvironmentFacesWritten = 0;
		SetPlanarCaptureFrame({}); m_Impl->FramePlanar = {};
		m_Impl->PlanarActive = m_Impl->PlanarWritten = m_Impl->PlanarReady = false;
		m_Impl->EnvironmentActive = m_Impl->EnvironmentReady = false;
		m_Impl->ReflectionActive = m_Impl->ReflectionApplied = false;
		m_Impl->MaterialCaptureActive = m_Impl->AmbientOcclusionActive = m_Impl->AmbientOcclusionApplied = m_Impl->SceneColorWritten = false;
		m_Impl->DeferredActive = m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
		m_Impl->CurrentPass = {};
		m_Impl->CurrentPassTarget = {};
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		if (!m_Impl->SubmittedWorkPending) m_Impl->FrameUploads.clear();
		m_Impl->ConstantOffset = 0;
		m_Impl->AbortTSRScene();
    m_Impl->FrameOpen = false;
    m_Impl->FinishLumenFrame(false); m_Impl->FinishNaniteFrame(false); m_Impl->FinishRayShadowImage(false);
		m_Impl->DepthOfFieldResolved = false;
		m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
		m_Impl->PassOpen = false;
		m_Impl->TemporalResolved = false;
        m_Impl->GI.Begin(); m_Impl->Probes.Invalidate();
        if (m_Impl->PendingUploadedLightmap) m_Impl->UploadedLightmap.reset();
        m_Impl->PendingUploadedLightmap.reset();
		m_Impl->FogResolved = false;
		m_Impl->LightShaftResolved = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
	}

	bool DX12RHI::BeginPass(const PassDesc& desc)
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
		if (!m_Impl->CheckThread("BeginPass")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX12 pass cannot begin in the current state");
        if (m_Impl->PathFrame.Active) return m_Impl->Fail("Path tracing owns primary visibility; raster passes are inactive");
        if (!ValidateDepthPeelingPass(desc,m_Impl->FramePeeling,m_Impl->PeelingSequence,m_Impl->LastError)) return false;
		const bool weighted = m_Impl->WeightedActive && m_Impl->WeightedPrepared && !m_Impl->WeightedResolved;
		if (weighted && (desc.Type != PassType::Color || desc.Target || desc.ClearColorEnabled || desc.ClearDepthEnabled || m_Impl->WeightedAccumulated))
			return m_Impl->Fail("DX12 weighted transparency accepts one depth-preserving color pass before resolve");
		if (m_Impl->WeightedActive && m_Impl->SceneColorWritten && desc.Type == PassType::Color && !desc.Target && !weighted)
			return m_Impl->Fail("DX12 prepare weighted transparency before transparent geometry; resolve closes that phase");
		if (desc.Type == PassType::DeferredLighting)
			return m_Impl->Fail("Use DrawDeferredLighting for DX12 fullscreen lighting");
		const bool environmentCapture = desc.Type == PassType::EnvironmentCapture;
		const bool planarCapture = desc.Type == PassType::PlanarCapture;
		const bool capture = environmentCapture || planarCapture;
		if (environmentCapture && (!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || desc.Target ||
			desc.EnvironmentFaceIndex >= EnvironmentFaceCount || !desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
			m_Impl->SceneColorWritten || m_Impl->GBufferWritten || (m_Impl->EnvironmentFacesWritten & (1u << desc.EnvironmentFaceIndex))))
			return m_Impl->Fail("DX12 capture requires six distinct cleared faces before main geometry");
		if (!environmentCapture && m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady && desc.Type != PassType::Shadow)
			return m_Impl->Fail("DX12 complete the environment capture and mips before main geometry");
		if (planarCapture && (!m_Impl->PlanarActive || m_Impl->PlanarWritten || m_Impl->PlanarReady || desc.Target ||
			!desc.ClearColorEnabled || !desc.ClearDepthEnabled || m_Impl->SceneColorWritten || m_Impl->GBufferWritten))
			return m_Impl->Fail("DX12 capture the planar view once, clearing private color/depth before main geometry");
		if (!planarCapture && m_Impl->PlanarActive && !m_Impl->PlanarReady && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture)
			return m_Impl->Fail("DX12 complete the planar capture and mips before main geometry");
		const bool velocity = desc.Type == PassType::Velocity;
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 light shafts must precede opaque velocity and temporal effects");
		if (m_Impl->LightShaftResolved && desc.Type != PassType::Shadow &&
			(desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
			return m_Impl->Fail("DX12 scattered color and opaque depth cannot be cleared");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->FogResolved)
			return m_Impl->Fail("DX12 fog composition must precede velocity and temporal effects");
		if (m_Impl->FogResolved && desc.Type != PassType::Shadow &&
			(desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
			return m_Impl->Fail("DX12 fogged scene color and depth cannot be cleared");
		const bool multisample = m_Impl->FrameSampleCount > 1 && !capture;
		if (multisample && desc.Target && desc.Type != PassType::Shadow)
			return m_Impl->Fail("DX12 MSAA requires internal sample-count-matched attachments");
		if (multisample && (velocity || desc.Type == PassType::Outline) && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("DX12 resolve opaque samples before velocity or outlines");
		if (multisample && m_Impl->SamplesOpaqueResolved && desc.Type == PassType::Color &&
			(desc.ClearColorEnabled || desc.ClearDepthEnabled || (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored)))
			return m_Impl->Fail("DX12 transparency must load restored color and retain multisampled depth");
		if (velocity && (!((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive) || desc.Target || desc.ClearDepthEnabled ||
			!desc.ClearColorEnabled || !m_Impl->SceneColorWritten || m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved ||
			m_Impl->DepthOfFieldResolved || m_Impl->TemporalResolved || (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(!m_Impl->EarlyTemporalActive && m_Impl->ReflectionActive && !m_Impl->ReflectionApplied)))
			return m_Impl->Fail("DX12 velocity requires opaque lighting/AO/SSR and a fresh internal target, once before blur");
		if (m_Impl->VelocityCaptured && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("DX12 opaque depth cannot be cleared after velocity capture");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->FramePostProcessing.MotionBlur.Enabled)
		{
			if (m_Impl->SceneColorWritten && !m_Impl->MotionBlurResolved)
				return m_Impl->Fail("DX12 transparent color must follow motion blur");
			if (m_Impl->MotionBlurResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("DX12 transparent color must load the completed scene and opaque depth");
		}
		if (desc.Type == PassType::Color && !desc.Target && Techniques::UsesTemporalAA(m_Impl->FramePostProcessing))
		{
			if (m_Impl->SceneColorWritten && !m_Impl->TemporalResolved)
				return m_Impl->Fail("DX12 transparent composition must follow TAA");
			if (m_Impl->TemporalResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("DX12 TAA color and opaque depth must be retained for transparency");
		}
		if (m_Impl->DepthOfFieldResolved && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("DX12 depth cannot be cleared after depth of field, including through custom color targets");
		if (m_Impl->ReflectionApplied && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
			return m_Impl->Fail("DX12 shared depth cannot be cleared after SSR");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->GIActive)
        {
            if (!m_Impl->GBufferWritten) return m_Impl->Fail("SSGI requires material capture before lighting");
            if (m_Impl->SceneColorWritten && !m_Impl->GI.Resolved) return m_Impl->Fail("Transparency must follow SSGI");
            if (m_Impl->GI.Resolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
                return m_Impl->Fail("Scene attachments cannot be cleared after SSGI");
        }
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->ReflectionActive)
		{
			if (!m_Impl->GBufferWritten) return m_Impl->Fail("DX12 SSR requires material capture before forward lighting");
			if (m_Impl->SceneColorWritten && !m_Impl->ReflectionApplied) return m_Impl->Fail("DX12 transparency must follow SSR");
			if (m_Impl->ReflectionApplied && desc.ClearColorEnabled) return m_Impl->Fail("DX12 scene color cannot be cleared after SSR");
		}
		const bool geometry = desc.Type == PassType::GBuffer;
        const bool contactDepth = desc.Type == PassType::ContactDepth;
        const bool contactReadOnly = m_Impl->FrameContact.Enabled && desc.Type == PassType::Color && !desc.Target;
        const bool forwardPlusDepth = desc.Type == PassType::ForwardPlusDepth;
        const bool forwardPlusReadOnly = m_Impl->LightListsActive && !m_Impl->ClusteredActive && desc.Type == PassType::Color && !desc.Target;
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Prepared())
            return m_Impl->Fail("DX12 Forward+ complete lights must be prepared before scene passes");
        if (forwardPlusDepth && (m_Impl->ClusteredActive || m_Impl->TiledActive || !m_Impl->LightListsActive || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ForwardPlusDepthWritten || m_Impl->GBufferWritten || m_Impl->SceneColorWritten || m_Impl->LightListFrame.Assigned()))
            return m_Impl->Fail("DX12 Forward+ depth clears the main opaque depth once before assignment");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !desc.Target && desc.Type != PassType::Shadow && !capture &&
            (desc.Type == PassType::GBuffer || desc.ClearDepthEnabled))
            return m_Impl->Fail("DX12 Forward+ cannot clear the depth used by light assignment");
        if (m_Impl->LightListsActive && desc.Type == PassType::Color && !desc.Target && !m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("DX12 Forward+ scene lighting requires completed light assignment");
        if (contactDepth && (!m_Impl->FrameContact.Enabled || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ContactDepthWritten || m_Impl->GBufferWritten || m_Impl->SceneColorWritten || m_Impl->ContactGenerated))
            return m_Impl->Fail("DX12 contact depth clears the main opaque camera depth once before visibility");
        if (m_Impl->ContactGenerated && desc.Type != PassType::Shadow && !capture && desc.ClearDepthEnabled)
            return m_Impl->Fail("DX12 contact lighting requires immutable traced depth");
        if (contactReadOnly && !m_Impl->ContactGenerated)
            return m_Impl->Fail("DX12 scene lighting requires completed contact visibility");
		if (geometry && (!m_Impl->MaterialCaptureActive || desc.Target || m_Impl->SceneColorWritten || m_Impl->AmbientOcclusionApplied || m_Impl->DepthOfFieldResolved))
			return m_Impl->Fail("DX12 material capture must precede opaque lighting and use the main target");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->DeferredActive && !m_Impl->DeferredResolved)
			return m_Impl->Fail("DX12 forward transparency must follow deferred lighting");
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionActive)
		{
			if (!m_Impl->GBufferWritten)
				return m_Impl->Fail("DX12 Ambient occlusion requires material capture before opaque forward color");
			if (m_Impl->SceneColorWritten && !m_Impl->AmbientOcclusionApplied)
				return m_Impl->Fail("DX12 forward transparency must follow ambient occlusion");
			if (m_Impl->AmbientOcclusionApplied && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("DX12 scene color and depth cannot be cleared after ambient occlusion");
		}
		if (desc.Type == PassType::Color && !desc.Target && m_Impl->FramePostProcessing.DepthOfField.Enabled)
		{
			if (m_Impl->SceneColorWritten && !m_Impl->DepthOfFieldResolved)
				return m_Impl->Fail("DX12 forward transparency must follow depth of field");
			if (m_Impl->DepthOfFieldResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
				return m_Impl->Fail("DX12 scene color and depth cannot be cleared after depth of field");
		}
		D3D12_CPU_DESCRIPTOR_HANDLE rtv = {};
		std::array<D3D12_CPU_DESCRIPTOR_HANDLE, GBufferAttachmentCount> materialTargets {};
		D3D12_CPU_DESCRIPTOR_HANDLE dsv = m_Impl->DSVArena.Cpu(m_Impl->MainDepthDSV);
		bool hasRTV = true;
		int width = m_Impl->CreateInfo.Width;
		int height = m_Impl->CreateInfo.Height;
		m_Impl->CurrentPassTarget = desc.Target;
		m_Impl->ActiveColorFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		m_Impl->ActiveSampleCount = !capture && !desc.Target && desc.Type != PassType::Outline ? m_Impl->FrameSampleCount : 1;
		if (desc.Type == PassType::Shadow)
		{
			auto target = m_Impl->Textures.find(desc.Target.Value);
			if (!desc.Target || target == m_Impl->Textures.end()
				|| target->second.Description.Format != TextureFormat::Depth32)
				return m_Impl->Fail("DX12 shadow pass requires a valid depth texture target");
			m_Impl->Transition(target->second.Resource.Get(), target->second.State,
				D3D12_RESOURCE_STATE_DEPTH_WRITE);
			dsv = m_Impl->DSVArena.Cpu(target->second.DSV);
			hasRTV = false;
			width = target->second.Description.Width;
			height = target->second.Description.Height;
			m_Impl->LastShadowTexture = desc.Target;
			if (m_Impl->UsesVSM())
			{
				if (!m_Impl->BeginVSMMoments(desc.Target, width, height, desc.ClearDepthEnabled, rtv)) return false;
				hasRTV = true;
				m_Impl->ActiveColorFormat = VSMFormat;
			}
		}
		else if (environmentCapture)
		{
			auto& cube = m_Impl->Environment.Cube;
			auto& depth = m_Impl->Environment.Depth;
			m_Impl->Transition(cube.Resource.Get(), cube.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			m_Impl->Transition(depth.Resource.Get(), depth.State, D3D12_RESOURCE_STATE_DEPTH_WRITE);
			rtv = m_Impl->RTVArena.Cpu(m_Impl->Environment.FaceRTVs[desc.EnvironmentFaceIndex]);
			dsv = m_Impl->DSVArena.Cpu(m_Impl->Environment.DepthDSV);
			width = height = static_cast<int>(m_Impl->FrameEnvironment.Resolution);
			m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
		}
		else if (planarCapture)
		{
			auto& image = m_Impl->Planar.Image; auto& depth = m_Impl->Planar.Depth;
			m_Impl->Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			m_Impl->Transition(depth.Resource.Get(), depth.State, D3D12_RESOURCE_STATE_DEPTH_WRITE);
			rtv = m_Impl->RTVArena.Cpu(image.RTV); dsv = m_Impl->DSVArena.Cpu(m_Impl->Planar.DepthDSV);
			width = static_cast<int>(m_Impl->FramePlanar.Width); height = static_cast<int>(m_Impl->FramePlanar.Height);
			m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
		}
        else if (contactDepth || forwardPlusDepth)
        {
            auto* depth = multisample ? m_Impl->MultisampleDepth.Resource.Get() : m_Impl->MainDepth.Get();
            auto& state = multisample ? m_Impl->MultisampleDepth.State : m_Impl->MainDepthState;
            m_Impl->Transition(depth, state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            dsv = m_Impl->DSVArena.Cpu(multisample ? m_Impl->MultisampleDepthDSV : m_Impl->MainDepthDSV);
            hasRTV = false;
        }
		else if (multisample && desc.Type != PassType::Outline)
		{
			auto& depth = m_Impl->MultisampleDepth;
			m_Impl->Transition(depth.Resource.Get(), depth.State, (velocity || contactReadOnly || forwardPlusReadOnly) ? D3D12_RESOURCE_STATE_DEPTH_READ : D3D12_RESOURCE_STATE_DEPTH_WRITE);
			dsv = m_Impl->DSVArena.Cpu((velocity || contactReadOnly || forwardPlusReadOnly) ? m_Impl->MultisampleDepthReadDSV : m_Impl->MultisampleDepthDSV);
			if (geometry)
			{
				m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
				for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
				{
					auto& target = m_Impl->MultisampleSurfaces[Impl::MultisampleMaterialFirst + index];
					m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
					materialTargets[index] = m_Impl->RTVArena.Cpu(target.RTV);
				}
			}
			else
			{
				auto& target = m_Impl->MultisampleSurfaces[velocity ? Impl::MultisampleVelocity : Impl::MultisampleColor];
				m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(target.RTV);
				m_Impl->ActiveColorFormat = target.Format;
			}
		}
		else
		{
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				(velocity || contactReadOnly || forwardPlusReadOnly) ? D3D12_RESOURCE_STATE_DEPTH_READ : D3D12_RESOURCE_STATE_DEPTH_WRITE);
			if (velocity)
			{
				auto& target = m_Impl->MotionSurfaces[Impl::MotionVelocity];
				m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(target.RTV);
				dsv = m_Impl->DSVArena.Cpu(m_Impl->MainDepthReadOnlyDSV);
				m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
			}
			else if (geometry)
			{
				m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
				for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
				{
					auto& target = m_Impl->GBufferSurfaces[index];
					m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
					materialTargets[index] = m_Impl->RTVArena.Cpu(target.RTV);
				}
			}
			else if (desc.Target)
			{
				auto target = m_Impl->Textures.find(desc.Target.Value);
				if (target == m_Impl->Textures.end() || target->second.RTV == InvalidDescriptor
					|| target->second.Description.Width != m_Impl->CreateInfo.Width
					|| target->second.Description.Height != m_Impl->CreateInfo.Height)
					return m_Impl->Fail("DX12 color pass target is invalid or does not match main depth dimensions");
				m_Impl->Transition(target->second.Resource.Get(), target->second.State,
					D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(target->second.RTV);
			}
			else if (m_Impl->HdrSceneActive)
			{
				auto& target = m_Impl->HdrSurfaces[Impl::HdrScene];
				m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(target.RTV);
				m_Impl->ActiveColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
			}
			else
			{
				m_Impl->Transition(m_Impl->MainColor.Get(), m_Impl->MainColorState,
					D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(m_Impl->MainColorRTV);
			}
		}
        if (desc.DepthPeelingLayer && !m_Impl->BeginPeelingLayer(rtv,dsv)) return false;
        if (!desc.DepthPeelingLayer && (contactReadOnly || forwardPlusReadOnly) && !multisample) dsv = m_Impl->DSVArena.Cpu(m_Impl->MainDepthReadOnlyDSV);
		if (weighted)
		{
			auto* depth = multisample ? m_Impl->MultisampleDepth.Resource.Get() : m_Impl->MainDepth.Get();
			auto& state = multisample ? m_Impl->MultisampleDepth.State : m_Impl->MainDepthState;
			m_Impl->Transition(depth, state, D3D12_RESOURCE_STATE_DEPTH_READ);
			dsv = m_Impl->DSVArena.Cpu(multisample ? m_Impl->MultisampleDepthReadDSV : m_Impl->MainDepthReadOnlyDSV);
			const D3D12_CPU_DESCRIPTOR_HANDLE targets[] {m_Impl->RTVArena.Cpu(m_Impl->WeightedSurfaces[0].RTV),
				m_Impl->RTVArena.Cpu(m_Impl->WeightedSurfaces[1].RTV)};
			m_Impl->CommandList->OMSetRenderTargets(2, targets, FALSE, &dsv);
		}
		else if (geometry) m_Impl->CommandList->OMSetRenderTargets(static_cast<UINT>(materialTargets.size()), materialTargets.data(), FALSE, &dsv);
		else if (hasRTV) m_Impl->CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
		else m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
		if (hasRTV && desc.ClearColorEnabled && desc.Type != PassType::Shadow)
		{
			float color[] = { desc.ClearColor.X, desc.ClearColor.Y, desc.ClearColor.Z, desc.ClearColor.W };
			if (velocity) std::fill(std::begin(color), std::end(color), 0.0f);
			if (m_Impl->ActiveColorFormat == DXGI_FORMAT_R16G16B16A16_FLOAT)
				for (float& channel : color) channel = std::clamp(channel, 0.0f, MaximumHalfFloatRadiance);
			if (geometry)
			{
				m_Impl->DeferredBackground = {color[0], color[1], color[2], color[3]};
				const float emptyMaterial[4] {};
				for (auto target : materialTargets) m_Impl->CommandList->ClearRenderTargetView(target, emptyMaterial, 0, nullptr);
			}
			else m_Impl->CommandList->ClearRenderTargetView(rtv, color, 0, nullptr);
		}
		if (desc.ClearDepthEnabled)
			m_Impl->CommandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
		D3D12_VIEWPORT viewport = {};
		viewport.Width = static_cast<float>(width);
		viewport.Height = static_cast<float>(height);
		viewport.MaxDepth = 1.0f;
		D3D12_RECT scissor = { 0, 0, width, height };
		m_Impl->CommandList->RSSetViewports(1, &viewport);
		m_Impl->CommandList->RSSetScissorRects(1, &scissor);
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

	bool DX12RHI::BindPipeline(const PipelineState& pipeline)
	{
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.DepthPeelingLayer && !pipeline.BlendEnabled) return m_Impl->Fail("Only transparent color draws follow the TSR opaque snapshot");

    if (m_Impl->FrameOpen && m_Impl->Hair.Active && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
        ((!m_Impl->Hair.Resolved && pipeline.BlendEnabled) || (m_Impl->Hair.Resolved && pipeline.DepthWrite)))
        return m_Impl->Fail("Opaque draws precede hair; transparent draws follow it without replacing depth");
		if (!m_Impl->CheckThread("BindPipeline")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 pipeline must be bound inside a pass");
        if ((m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) &&
            (pipeline.BlendEnabled || !pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.OutlineExtrusion))
            return m_Impl->Fail("DX12 contact depth requires opaque depth writes without color");
        if (m_Impl->ContactGenerated && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && pipeline.DepthWrite)
            return m_Impl->Fail("DX12 contact receivers must preserve the traced depth");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !m_Impl->CurrentPass.Target && pipeline.DepthWrite &&
            m_Impl->CurrentPass.Type != PassType::Shadow && m_Impl->CurrentPass.Type != PassType::EnvironmentCapture &&
            m_Impl->CurrentPass.Type != PassType::PlanarCapture)
            return m_Impl->Fail("DX12 Forward+ receivers cannot overwrite assignment depth");
        if (m_Impl->LightListsActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && !pipeline.DepthTest)
            return m_Impl->Fail("DX12 Forward+ tile receivers require camera depth testing");

		if (m_Impl->WeightedPass && (!pipeline.BlendEnabled || pipeline.DepthWrite || !pipeline.DepthTest))
			return m_Impl->Fail("DX12 weighted transparency requires blended coverage and read-only opaque depth testing");
		if (m_Impl->WeightedActive && !m_Impl->WeightedPass && m_Impl->CurrentPass.Type == PassType::Color &&
			!m_Impl->CurrentPass.Target && pipeline.BlendEnabled) return m_Impl->Fail("DX12 prepare weighted transparency before blended geometry");
		if (m_Impl->VelocityCaptured && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("DX12 shared opaque depth is read-only after velocity capture");
		if (m_Impl->FogResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("DX12 shared opaque depth is read-only after fog composition");
		if (m_Impl->LightShaftResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("Shared opaque depth is read-only after light shafts");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && pipeline.BlendEnabled != m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 opaque color precedes light shafts; transparency follows them");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			Techniques::UsesVolumetricFog(m_Impl->FrameFog) && pipeline.BlendEnabled != m_Impl->FogResolved)
			return m_Impl->Fail("DX12 opaque draws precede fog; transparent draws follow its composition");
		if (m_Impl->CurrentPass.Type == PassType::Velocity &&
			(!pipeline.DepthTest || pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
			return m_Impl->Fail("DX12 velocity requires opaque coverage with depth testing and no depth writes");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			m_Impl->FramePostProcessing.MotionBlur.Enabled && pipeline.BlendEnabled != m_Impl->MotionBlurResolved)
			return m_Impl->Fail("DX12 opaque draws precede motion blur; transparent draws follow it");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
			Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && pipeline.BlendEnabled != m_Impl->TemporalResolved)
			return m_Impl->Fail("DX12 opaque draws precede TAA; transparent draws follow it");
		if (m_Impl->ReflectionApplied && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
			return m_Impl->Fail("DX12 opaque depth is read-only after SSR");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->GIActive &&
            pipeline.BlendEnabled != m_Impl->GI.Resolved) return m_Impl->Fail("SSGI separates opaque and transparent material draws");
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->ReflectionActive &&
			pipeline.BlendEnabled != m_Impl->ReflectionApplied)
			return m_Impl->Fail("DX12 opaque draws precede SSR; transparency follows it");
		if (m_Impl->CurrentPass.Type == PassType::Color && m_Impl->FramePostProcessing.DepthOfField.Enabled)
		{
			if (m_Impl->DepthOfFieldResolved && pipeline.DepthWrite)
				return m_Impl->Fail("DX12 color draws cannot modify shared scene depth after depth of field");
			if (!m_Impl->CurrentPass.Target && m_Impl->DepthOfFieldResolved && !pipeline.BlendEnabled)
				return m_Impl->Fail("DX12 only transparent scene draws may follow depth of field");
			if (!m_Impl->CurrentPass.Target && !m_Impl->DepthOfFieldResolved && pipeline.BlendEnabled)
				return m_Impl->Fail("DX12 transparent pipelines must follow depth of field");
		}
		if ((m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture) &&
			(!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite ||
			 pipeline.SamplesShadowMap || pipeline.OutlineExtrusion || pipeline.Shading != m_Impl->FrameReflectionShading))
			return m_Impl->Fail("DX12 reflection capture requires opaque depth and the frame shading model without shadow sampling");
		if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && !m_Impl->FramePlanar.Active())
			return m_Impl->Fail("DX12 inactive planar capture must clear without drawing geometry");
		if (m_Impl->CurrentPass.Type == PassType::GBuffer
			&& (pipeline.BlendEnabled || pipeline.Shading != m_Impl->FrameReflectionShading ||
				(m_Impl->DeferredActive && pipeline.Shading == Techniques::ShadingModel::Gouraud)))
			return m_Impl->Fail("DX12 G-buffer requires opaque materials and the frame shading model");
		ID3D12PipelineState* native = m_Impl->GetPipeline(pipeline, m_Impl->CurrentPass.Type);
		if (!native) return false;
		m_Impl->CommandList->SetPipelineState(native);
		const bool parallax=UsesParallaxMaterial(pipeline,m_Impl->CurrentPass.Type);
        m_Impl->CommandList->SetGraphicsRootSignature(parallax?m_Impl->ParallaxRoot.Get():m_Impl->CurrentPass.Type == PassType::Velocity ?
            m_Impl->MotionRootSignature.Get() : m_Impl->RootSignature.Get());
        if (!parallax) m_Impl->BindMaterialSampler(m_Impl->CurrentPass.Type == PassType::Velocity ?
            MotionInstanceRoot + 1 : ForwardMaterialSamplerRoot);
		// Conventional vertex programs do not read b3. Bind a valid dummy address
		// anyway; an instanced submission replaces it after its upload succeeds.
		if (!parallax) m_Impl->CommandList->SetGraphicsRootConstantBufferView(m_Impl->CurrentPass.Type == PassType::Velocity ?
            MotionInstanceRoot : ForwardInstanceRoot, m_Impl->ConstantBuffer->GetGPUVirtualAddress());
        if (m_Impl->CurrentPass.DepthPeelingLayer && !ValidateDepthPeelingPipeline(pipeline,m_Impl->LastError)) return false;
		m_Impl->BoundPipeline = pipeline;
		m_Impl->PipelineBound = true;
		return true;
	}

	bool DX12RHI::BindWorldTransform(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindWorldTransform")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 world transform must be bound inside a pass");
		m_Impl->BoundWorld = value;
		m_Impl->WorldBound = true;
		return true;
	}

	bool DX12RHI::BindViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindViewProjection")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 view projection must be bound inside a pass");
		if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && value.Values != m_Impl->FramePlanar.ClippedViewProjection.Values)
			return m_Impl->Fail("DX12 planar geometry requires the frozen oblique capture projection");
        if (m_Impl->CurrentPass.DepthPeelingLayer && value.Values!=m_Impl->FramePeeling.ViewProjection.Values)
            return m_Impl->Fail("OIT capture and replay require the same raster projection");
		m_Impl->BoundViewProjection = value;
		m_Impl->ViewProjectionBound = true;
		return true;
	}

	bool DX12RHI::BindLightViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindLightViewProjection")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 light view projection must be bound inside a pass");
		m_Impl->BoundLightViewProjection = value;
		m_Impl->LightViewProjectionBound = true;
		return true;
	}

	bool DX12RHI::BindVertexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindVertexBuffer")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 vertex buffer must be bound inside a pass");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Vertex)
			return m_Impl->Fail("DX12 vertex buffer is invalid");
		m_Impl->BoundVertexBuffer = buffer;
		return true;
	}

	bool DX12RHI::BindIndexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindIndexBuffer")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 index buffer must be bound inside a pass");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Index)
			return m_Impl->Fail("DX12 index buffer is invalid");
		m_Impl->BoundIndexBuffer = buffer;
		return true;
	}

	bool DX12RHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchHandle& handle)
    {
        handle = {};
        if (!m_Impl->CheckThread("PrepareGpuDrivenBatch") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Instances.empty() || !desc.IndexCount)
            return m_Impl->Fail("DX12 GPU batch preparation requires an open frame outside a graphics pass");
        std::uint32_t index = 0;
        if (!m_Impl->PrepareGpuBatch(desc, index)) return false;
        handle = {m_Impl->GpuEpoch, index};
        ++m_Impl->PendingFrame.Statistics.GpuCullDispatchCount;
        return true;
    }

    bool DX12RHI::RenderVisibilityBuffer(const VisibilityBufferDesc& desc)
    {
        if (!m_Impl->CheckThread("RenderVisibilityBuffer") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->DeferredActive || m_Impl->GBufferWritten || m_Impl->FrameSampleCount != 1)
            return m_Impl->Fail("DX12 visibility geometry requires an unresolved single-sample deferred frame");
        if (!ValidateVisibilityFrame(desc, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError) ||
            !m_Impl->PrepareVisibilityBuffer(desc.Scene)) return false;
        if (desc.NaniteRaster && !m_Impl->PrepareNaniteFrameRaster(desc)) return false;
        if (!m_Impl->PrepareNaniteHardware(desc)) return false;
        if (!m_Impl->PrepareMeshShaders(desc)) return false;
        if (!m_Impl->PrepareNaniteInstances(desc)) return false;
        if (!m_Impl->PrepareNaniteMaterials(desc)) return false;
        PassDesc pass; pass.Type = PassType::GBuffer; pass.ClearColor = desc.Background;
        if (!BeginPass(pass) || !m_Impl->DrawVisibilityBuffer(desc)) return false;
        RecordVisibilityDraws(m_Impl->PendingFrame, desc);
        return EndPass();
    }

    bool DX12RHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle handle)
    {
        if (!m_Impl->CheckThread("DrawGpuDrivenBatch") || !m_Impl->PassOpen || !handle.Frame ||
            handle.Frame != m_Impl->GpuEpoch || handle.Index >= m_Impl->GpuCursor || m_Impl->ActiveGpuBatch)
            return m_Impl->Fail("DX12 indirect draw has an invalid frame token or pass state");
        auto& batch = m_Impl->GpuBatches[handle.Index];
        const auto& desc = batch.Description;
        if (!GpuDrivenDrawMatches(desc, m_Impl->BoundWorld, m_Impl->BoundViewProjection, m_Impl->BoundPipeline, m_Impl->CurrentPass))
            return m_Impl->Fail("DX12 indirect draw does not match its prepared instance data");
        m_Impl->ActiveGpuBatch = &batch;
        struct Reset { Impl* State; ~Reset() { State->ActiveGpuBatch = nullptr; } } reset {m_Impl.get()};
        return DrawIndexedGeometry(desc.FirstIndex, desc.IndexCount, desc.BaseVertex, &desc.Instances);
    }

    bool DX12RHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
		return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, nullptr);
	}

	bool DX12RHI::DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
		const std::vector<DrawInstance>& instances)
	{
		return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, &instances);
	}

	bool DX12RHI::DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
		const std::vector<DrawInstance>* instances)
	{
    if (m_Impl->PassOpen && m_Impl->Hair.Active && HairFurReplayPass(m_Impl->CurrentPass.Type) &&
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Hair replay ends the pass geometry stream");
		if (!m_Impl->CheckThread("DrawIndexed")) return false;
		if (!m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("DX12 DrawIndexed is missing required state");
        if (m_Impl->LightListsActive && (m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth || m_Impl->CurrentPass.Type == PassType::GBuffer) &&
            !m_Impl->LightListFrame.ValidateDraw(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, true, m_Impl->LastError)) return false;
        if (instances && m_Impl->CurrentPass.DepthPeelingLayer) return m_Impl->Fail("OIT replay requires ordinary invariant draws");
		if (instances && UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        return m_Impl->Fail("POM material capture currently requires individual mesh draws");
    std::optional<InstanceConstantBlock> instanceConstants;
		if (instances)
		{
			instanceConstants.emplace();
			if (m_Impl->CurrentPass.Type == PassType::Outline || m_Impl->BoundPipeline.OutlineExtrusion ||
				!BuildInstanceConstants(*instances, m_Impl->BoundWorld, *instanceConstants))
				return m_Impl->Fail("DX12 instancing requires a bounded batch with finite transforms and common winding");
			instanceConstants->ViewProjection = m_Impl->BoundViewProjection;
			instanceConstants->Material.X = m_Impl->BoundPipeline.Color.W * m_Impl->BoundPipeline.MaterialOpacity;
		}
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("DX12 DrawIndexed buffers are invalid");
		if (!ValidateIndexedGeometryDraw(vertex->second.Description,index->second.Description,index->second.Bytes,
		    vertex->second.NaniteGeometry && index->second.NaniteGeometry,firstIndex,indexCount,baseVertex,m_Impl->LastError,
		    &index->second.LastValidRange)) return false;

		m_Impl->ConstantOffset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		const bool velocity = m_Impl->CurrentPass.Type == PassType::Velocity;
		const std::size_t constantBytes = velocity ? sizeof(MotionBlurConstants) : sizeof(RealtimeDrawConstants);
		if (m_Impl->ConstantOffset > m_Impl->ConstantBufferCapacity
			|| constantBytes > m_Impl->ConstantBufferCapacity - m_Impl->ConstantOffset)
			return m_Impl->Fail("DX12 per-frame constant upload capacity is exhausted");

		UINT shadowDescriptor = m_Impl->NullShadowSRV;
		bool shadowAvailable = false;
		if (m_Impl->CurrentPass.Type != PassType::Shadow
			&& m_Impl->BoundPipeline.SamplesShadowMap
			&& m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow == m_Impl->Textures.end()
				|| shadow->second.Description.Format != TextureFormat::Depth32)
				return m_Impl->Fail("DX12 shadow map resource is unavailable");
			m_Impl->Transition(shadow->second.Resource.Get(), shadow->second.State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			shadowDescriptor = shadow->second.SRV;
			shadowAvailable = true;
		}

		std::array<UINT, 3> surfaceDescriptors {
			m_Impl->NullColorSRV,
			m_Impl->NullColorSRV,
			m_Impl->NullColorSRV };
		auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
			UINT& descriptor)
		{
			if (!handle) return true;
			auto texture = m_Impl->Textures.find(handle.Value);
			if (texture == m_Impl->Textures.end()
				|| texture->second.Description.Format != TextureFormat::RGBA8
				|| texture->second.SRV == InvalidDescriptor)
			{
				return m_Impl->Fail(std::string("DX12 ") + usage
					+ " texture is invalid");
			}
			m_Impl->Transition(texture->second.Resource.Get(), texture->second.State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			descriptor = texture->second.SRV;
			return true;
		};
		if (!resolveSurfaceTexture(
				m_Impl->BoundPipeline.DiffuseTexture, "diffuse",
				surfaceDescriptors[0])
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.NormalTexture, "normal",
				surfaceDescriptors[1])
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.OpacityTexture, "opacity",
				surfaceDescriptors[2]))
		{
			return false;
		}

		if (velocity)
		{
			MotionBlurConstants constants {};
			if (!BuildMotionGeometryConstants(m_Impl->BoundPipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
				GetMotionTransform(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				m_Impl->BoundPipeline.DiffuseTexture != TextureHandle{}, m_Impl->BoundPipeline.OpacityTexture != TextureHandle{}, constants))
				return m_Impl->Fail("DX12 velocity requires finite current and previous transforms");
			if (instances) constants.Options.W = 1; // Zero previous matrices reject invalid instances in the pixel stage.
			if (!m_Impl->UploadMotionConstants(constants)) return false;
			const std::array<UINT, 5> descriptors {m_Impl->NullColorSRV, m_Impl->NullShadowSRV,
				m_Impl->NullColorSRV, surfaceDescriptors[0], surfaceDescriptors[2]};
			for (UINT slot = 0; slot < descriptors.size(); ++slot)
				m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(descriptors[slot]));
		}
        else if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        {
            if (!m_Impl->BindParallaxDraw(surfaceDescriptors)) return false;
        }
		else
		{
			const auto constants = BuildRealtimeDrawConstants(m_Impl->BoundPipeline, m_Impl->BoundWorld,
				m_Impl->BoundViewProjection, m_Impl->BoundLightViewProjection, m_Impl->CurrentPass.Type, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || (m_Impl->FrameVirtualShadowAtlas && m_Impl->BoundPipeline.SamplesShadowMap && m_Impl->CurrentPass.Type==PassType::Color)),
				m_Impl->BoundPipeline.DiffuseTexture != TextureHandle{}, m_Impl->BoundPipeline.NormalTexture != TextureHandle{},
				m_Impl->BoundPipeline.OpacityTexture != TextureHandle{}, m_Impl->CurrentPass.Type != PassType::Shadow &&
				m_Impl->ActiveColorFormat == DXGI_FORMAT_R16G16B16A16_FLOAT);
			std::memcpy(m_Impl->ConstantData + m_Impl->ConstantOffset, &constants, sizeof(constants));
			m_Impl->CommandList->SetGraphicsRootConstantBufferView(0,
				m_Impl->ConstantBuffer->GetGPUVirtualAddress() + m_Impl->ConstantOffset);
			m_Impl->ConstantOffset += sizeof(constants);
			if (m_Impl->UsesSurfacePCSS(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type) &&
				!m_Impl->BindPCSS(ForwardShadowFilterRoot, m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting, shadowAvailable)) return false;
			if (m_Impl->UsesVSM() && m_Impl->CurrentPass.Type == PassType::Shadow)
			{
				auto moments = BuildVSMParameters(m_Impl->FrameShadowFilter.VSM);
				if (!BuildVSMEncoding(m_Impl->BoundViewProjection, moments.DepthEncoding[0], m_Impl->LastError) ||
					!m_Impl->UploadVSM(ForwardShadowFilterRoot, moments)) return false;
			}
			if (m_Impl->UsesSurfaceVSM(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type))
			{
				if (!m_Impl->BindVSM(ForwardShadowFilterRoot, m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting, shadowAvailable)) return false;
				if (shadowAvailable && !m_Impl->VSMMomentDescriptor(m_Impl->LastShadowTexture, shadowDescriptor)) return false;
			}
            if (m_Impl->UsesContactReceiver(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type))
                m_Impl->BindContactReceiver(ForwardContactControlRoot, ForwardContactImageRoot);
            if (m_Impl->UsesForwardPlusMaterial(m_Impl->CurrentPass.Type) && !m_Impl->BindForwardPlusSurface()) return false;
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(shadowDescriptor));
			for (UINT slot = 0; slot < surfaceDescriptors.size(); ++slot)
				m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 2, m_Impl->SRVArena.Gpu(surfaceDescriptors[slot]));
			if (!m_Impl->BindCascadedShadows(m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap, false)) return false;
			const bool fogActive = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
				m_Impl->BoundPipeline.BlendEnabled && m_Impl->FogResolved && Techniques::UsesVolumetricFog(m_Impl->FrameFog);
			const VolumetricFogConstants disabledFog {};
			const auto fogOffset=AlignUp(m_Impl->ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (fogOffset>m_Impl->ConstantBufferCapacity || sizeof(VolumetricFogConstants)>m_Impl->ConstantBufferCapacity-fogOffset) return m_Impl->Fail("Surface fog exhausted frame constants");
            std::memcpy(m_Impl->ConstantData+fogOffset,fogActive?&m_Impl->FogConstants:&disabledFog,sizeof(VolumetricFogConstants));
            m_Impl->ConstantOffset=fogOffset+sizeof(VolumetricFogConstants);
            m_Impl->CommandList->SetGraphicsRootConstantBufferView(ForwardFogConstantsRoot,m_Impl->ConstantBuffer->GetGPUVirtualAddress()+fogOffset);
            if ((m_Impl->UsesSurfacePCF(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type) ||
                m_Impl->UsesMaterialSurface(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type)) &&
                !m_Impl->BindPCF(ForwardShadowFilterRoot,shadowAvailable)) return false;
            if (!m_Impl->BindLumenTranslucency()) return false;
            m_Impl->BindRayShadowImage(ForwardRayShadowImageRoot);
			if (fogActive) m_Impl->Transition(m_Impl->FogAtlas.Resource.Get(), m_Impl->FogAtlas.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(ForwardFogTextureRoot,
				m_Impl->SRVArena.Gpu(fogActive ? m_Impl->FogAtlas.SRV : m_Impl->NullColorSRV));
			const auto transparency = BuildWeightedTransparencyConstants(m_Impl->FrameTransparency);
			m_Impl->CommandList->SetGraphicsRoot32BitConstants(ForwardTransparencyRoot,
				sizeof(transparency) / sizeof(std::uint32_t), &transparency, 0);
		}
        // Track IA reads explicitly: a later Nanite assembly pass can consume
        // these same buffers as SRVs within this command list. Implicit promotion
        // alone would leave the CPU state journal incorrectly at COMMON.
        m_Impl->Transition(vertex->second.Resource.Get(), vertex->second.State, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->CommandList->IASetVertexBuffers(0, 1, &vertex->second.VertexView);
		if (m_Impl->ActiveNaniteDraw && m_Impl->ActiveNaniteDraw->Binning)
        {
            const auto& packed=*m_Impl->ActiveNaniteDraw->Arguments;
            const D3D12_INDEX_BUFFER_VIEW view{packed.Buffer->GetGPUVirtualAddress(),UINT(packed.Words*sizeof(unsigned)),DXGI_FORMAT_R32_UINT};
            m_Impl->CommandList->IASetIndexBuffer(&view);
        }
        else
        {
            m_Impl->Transition(index->second.Resource.Get(), index->second.State, D3D12_RESOURCE_STATE_INDEX_BUFFER);
            m_Impl->CommandList->IASetIndexBuffer(&index->second.IndexView);
        }
		// The world transform may be bound after BindPipeline. Mirrored winding
		// belongs to this draw's PSO, not the previous mesh's transform.
		ID3D12PipelineState* native = m_Impl->GetPipeline(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type, instances != nullptr);
		if (!native) return false;
		if (instances && !m_Impl->UploadInstances(*instanceConstants, velocity)) return false;
		m_Impl->CommandList->SetPipelineState(native);
        if (m_Impl->ActiveGpuBatch)
            m_Impl->CommandList->ExecuteIndirect(m_Impl->GpuDrawSignature.Get(), 1, m_Impl->ActiveGpuBatch->Arguments.Get(), 0, nullptr, 0);
        else if (m_Impl->ActiveNaniteDraw) m_Impl->CommandList->ExecuteIndirect(m_Impl->GpuDrawSignature.Get(),1,m_Impl->ActiveNaniteDraw->Arguments->Buffer.Get(),m_Impl->NaniteArgumentOffset(),nullptr,0);
        else m_Impl->CommandList->DrawIndexedInstanced(indexCount, instances ? static_cast<UINT>(instances->size()) : 1, firstIndex, baseVertex, 0);

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
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(std::move(record));
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

	bool DX12RHI::EndPass()
	{
    if (m_Impl->FrameOpen && m_Impl->PassOpen && m_Impl->Hair.Active && !m_Impl->Hair.Frame.Geometry->Indices.empty() &&
        HairFurReplayPass(m_Impl->CurrentPass.Type) && (m_Impl->CurrentPass.Type!=PassType::PlanarCapture || m_Impl->FramePlanar.Active()) &&
        m_Impl->Hair.ReplayedPass!=m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Complete requested hair replay before ending its native geometry pass");
		if (!m_Impl->CheckThread("EndPass")) return false;
		if (!m_Impl->PassOpen)
			return m_Impl->Fail("DX12 has no open pass");
		if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->CurrentPassTarget)
		{
			auto target = m_Impl->Textures.find(m_Impl->CurrentPassTarget.Value);
			if (target == m_Impl->Textures.end())
				return m_Impl->Fail("DX12 shadow target disappeared during the pass");
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			if (m_Impl->UsesVSM() && !m_Impl->FilterVSMMoments(m_Impl->CurrentPassTarget)) return false;
			m_Impl->Transition(target->second.Resource.Get(), target->second.State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
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
			m_Impl->PlanarWritten = true; ++m_Impl->PendingFrame.Statistics.PlanarCapturePassCount;
		}
		if (m_Impl->CurrentPass.Type == PassType::Velocity) m_Impl->VelocityCaptured = true;
		if (m_Impl->WeightedPass) m_Impl->WeightedAccumulated = true;
		if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target)
		{
			m_Impl->SceneColorWritten = true;
			if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved)
			{
				m_Impl->SamplesTransparentWritten = true;
				m_Impl->SamplesColorResolved = false;
			}
		}
        if (m_Impl->CurrentPass.DepthPeelingLayer && !m_Impl->PeelingSequence.Advance(DepthPeelingEvent::EndShading,m_Impl->LastError)) return false;
		m_Impl->PassOpen = false;
		return true;
	}

    bool DX12RHI::PrepareLightListFrame(const LightListFrameDesc& desc)
    {
        if (!m_Impl->CheckThread("PrepareLightListFrame") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Layout != (m_Impl->ClusteredActive ? LightListLayout::Clusters : LightListLayout::Tiles) || !m_Impl->LightListsActive || !m_Impl->PendingFrame.Passes.empty())
            return m_Impl->Fail("DX12 Forward+ complete-light upload must precede scene passes");
        if (!m_Impl->LightListFrame.Prepare(desc, m_Impl->LastError) || !(m_Impl->ClusteredActive ? m_Impl->ClusteredForward.BeginFrame(m_Impl->LastError) : m_Impl->TileProducer().BeginFrame(m_Impl->LastError))) return false;
        m_Impl->ForwardPlusRecordOpen = true;
        return m_Impl->UploadForwardPlusScene();
    }

    bool DX12RHI::GenerateLightLists()
    {
        if (!m_Impl->CheckThread("GenerateLightLists") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->LightListsActive || !m_Impl->LightListFrame.Prepared() || m_Impl->LightListFrame.Assigned() ||
            m_Impl->SceneColorWritten || (!m_Impl->ClusteredActive && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten))
            return m_Impl->Fail("DX12 Forward+ assignment requires completed opaque depth before main-view lighting");
        m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        const bool multisample = m_Impl->FrameSampleCount > 1;
        auto* depth = multisample ? m_Impl->MultisampleDepth.Resource.Get() : m_Impl->MainDepth.Get();
        auto& depthState = multisample ? m_Impl->MultisampleDepth.State : m_Impl->MainDepthState;
        const auto& frame = m_Impl->LightListFrame.Description();
        if (m_Impl->ClusteredActive)
        {
            if (!m_Impl->ClusteredForward.Record(m_Impl->Device.Get(), m_Impl->CommandList.Get(),
                m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError)) return false;
            m_Impl->ForwardPlusListsRecorded = true;
            if (!m_Impl->LightListFrame.CompleteAssignment(m_Impl->ClusteredForward.Grid(), m_Impl->LastError)) return false;
        }
        else
        {
            if (!m_Impl->TileProducer().Record(m_Impl->Device.Get(), m_Impl->CommandList.Get(), depth, depthState,
                frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError)) return false;
            m_Impl->ForwardPlusListsRecorded = true;
            if (!m_Impl->LightListFrame.CompleteAssignment(m_Impl->TileProducer().Grid(), m_Impl->LastError)) return false;
        }
        m_Impl->Device->CopyDescriptorsSimple(1, m_Impl->SRVArena.Cpu(m_Impl->ForwardPlusHeaderView),
            m_Impl->ClusteredActive ? m_Impl->ClusteredForward.HeaderView() : m_Impl->TileProducer().HeaderView(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_Impl->Device->CopyDescriptorsSimple(1, m_Impl->SRVArena.Cpu(m_Impl->ForwardPlusIndexView),
            m_Impl->ClusteredActive ? m_Impl->ClusteredForward.IndexView() : m_Impl->TileProducer().IndexView(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_Impl->BindSceneDescriptorHeaps();
        ++(m_Impl->ClusteredActive ? m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount : m_Impl->TiledActive ? m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount : m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount);
        return true;
    }

    bool DX12RHI::GenerateContactShadows(const ContactShadowDesc& desc)
    {
        if (!m_Impl->CheckThread("GenerateContactShadows") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->FrameContact.Enabled || m_Impl->ContactGenerated || m_Impl->SceneColorWritten ||
            (!m_Impl->ContactDepthWritten && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten))
            return m_Impl->Fail("DX12 contact visibility requires completed opaque depth before lighting, once per frame");
        if (m_Impl->LightListsActive)
        {
            if (!m_Impl->LightListFrame.Assigned())
                return m_Impl->Fail("DX12 Forward+ contact visibility must follow light assignment");
            if (!m_Impl->LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
        }

        if (m_Impl->ContactDescriptor == InvalidDescriptor) m_Impl->ContactDescriptor = m_Impl->SRVArena.Allocate();
        if (m_Impl->ContactDescriptor == InvalidDescriptor) return m_Impl->Fail("DX12 contact receiver descriptor allocation failed");
        m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        auto lighting = desc.Lighting;
        if (lighting.SceneLights) lighting.SceneLights->WriteLegacySubset(lighting);
        if (!m_Impl->Hair.ValidateContactProducer(desc.ViewProjection,lighting,m_Impl->LastError)) return false;
        const auto light = lighting.ShadowLightIndex;
        m_Impl->ContactReceiverActive = light >= 0 && light < int(std::min(lighting.LightCount, MaximumSceneLights)) &&
            lighting.Lights[light].CastShadows;
        if (m_Impl->ContactReceiverActive)
        {
            ContactShadowDispatch trace; trace.ViewProjection = desc.ViewProjection; trace.Light = lighting.Lights[light];
            trace.MaximumDistance = m_Impl->FrameContact.MaximumDistance;
            trace.Parameters.StartBias = m_Impl->FrameContact.StartBias; trace.Parameters.Thickness = m_Impl->FrameContact.Thickness;
            trace.Parameters.MaximumSteps = m_Impl->FrameContact.MaximumSteps;
            const bool multisample = m_Impl->FrameSampleCount > 1;
            auto* depth = multisample ? m_Impl->MultisampleDepth.Resource.Get() : m_Impl->MainDepth.Get();
            auto& state = multisample ? m_Impl->MultisampleDepth.State : m_Impl->MainDepthState;
            if (!m_Impl->ContactShadows.Record(m_Impl->Device.Get(), m_Impl->CommandList.Get(), depth, state, trace, m_Impl->LastError)) return false;
            m_Impl->Device->CopyDescriptorsSimple(1, m_Impl->SRVArena.Cpu(m_Impl->ContactDescriptor),
                m_Impl->ContactShadows.ResultView(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            ++m_Impl->PendingFrame.Statistics.ContactShadowPassCount;
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC empty {}; empty.Format = DXGI_FORMAT_R32_FLOAT;
            empty.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            empty.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY; empty.Texture2DArray.ArraySize = empty.Texture2DArray.MipLevels = 1;
            m_Impl->Device->CreateShaderResourceView(nullptr, &empty, m_Impl->SRVArena.Cpu(m_Impl->ContactDescriptor));
        }
        m_Impl->BindSceneDescriptorHeaps();
        m_Impl->Hair.RecordContact(desc.ViewProjection,lighting,m_Impl->ContactReceiverActive);
        m_Impl->ContactGenerated = true;
        return true;
    }

	bool DX12RHI::DrawHairFurVelocity(const HairFurVelocityDesc& desc)
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

bool DX12RHI::DrawHairFurInPass(const HairFurDesc& desc)
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

bool DX12RHI::DrawHairFur(const HairFurDesc& desc)
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

bool DX12RHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc)
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

bool DX12RHI::DrawDeferredLighting(const DeferredLightingDesc& desc)
	{
        if (desc.Output == DeferredOutputMode::DirectDiffuse && m_Impl->TiledActive)
            return m_Impl->Fail("Direct-diffuse output requires ordinary deferred lighting");
		if (!m_Impl->CheckThread("DrawDeferredLighting")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->DeferredActive
			|| !m_Impl->GBufferWritten || (m_Impl->DeferredResolved && !m_Impl->SubsurfaceProducing) || desc.Shading != GetTechniqueSettings().Shading)
			return m_Impl->Fail("DX12 lighting requires one completed G-buffer with the frame shading model");
        if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("DX12 deferred lighting requires current contact visibility");
		auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
		const bool shadowAvailable = shadow != m_Impl->Textures.end()
			&& shadow->second.Description.Format == TextureFormat::Depth32;
		DeferredConstants constants;
		if (!BuildDeferredConstants(desc, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || bool(m_Impl->FrameVirtualShadowAtlas)), m_Impl->DeferredBackground, constants))
			return m_Impl->Fail("DX12 deferred lighting requires a finite, invertible camera projection");
		const std::size_t offset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		if (offset > m_Impl->ConstantBufferCapacity || sizeof(constants) > m_Impl->ConstantBufferCapacity - offset)
			return m_Impl->Fail("DX12 deferred constant upload capacity is exhausted");
		std::memcpy(m_Impl->ConstantData + offset, &constants, sizeof(constants));
		m_Impl->ConstantOffset = offset + sizeof(constants);

        std::size_t surfaceOffset = 0, selectionOffset = 0;
        if (m_Impl->TiledActive)
        {
            RealtimeDrawConstants surface;
            TiledDeferredReconstruction reconstruction;
            LightListSurfaceConstants selection;
            if (!BuildTiledDeferredConstants(desc, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || bool(m_Impl->FrameVirtualShadowAtlas)), m_Impl->DeferredBackground, surface, reconstruction) ||
                !m_Impl->LightListFrame.Surface(desc.Lighting, desc.ViewProjection, LightListSelection::Opaque, selection, m_Impl->LastError)) return false;
            // The tiled camera occupies the same b1 allocation; its material and
            // scene-order list selection use independent aligned root CBVs.
            std::memcpy(m_Impl->ConstantData + offset, &reconstruction, sizeof(reconstruction));
            surfaceOffset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            selectionOffset = AlignUp(surfaceOffset + sizeof(surface), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (selectionOffset > m_Impl->ConstantBufferCapacity || sizeof(selection) > m_Impl->ConstantBufferCapacity - selectionOffset)
                return m_Impl->Fail("DX12 tiled lighting constants exceed the frame arena");
            std::memcpy(m_Impl->ConstantData + surfaceOffset, &surface, sizeof(surface));
            std::memcpy(m_Impl->ConstantData + selectionOffset, &selection, sizeof(selection));
            m_Impl->ConstantOffset = selectionOffset + sizeof(selection);
        }

		// End the writable depth binding before sampling it. All state changes are
		// journaled with the frame; AbortFrame restores only unsubmitted transitions.
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		const bool multisample = m_Impl->FrameSampleCount > 1;
		auto* depth = multisample ? m_Impl->MultisampleDepth.Resource.Get() : m_Impl->MainDepth.Get();
		auto& depthState = multisample ? m_Impl->MultisampleDepth.State : m_Impl->MainDepthState;
		const UINT depthView = multisample ? m_Impl->MultisampleDepth.SRV : m_Impl->MainDepthSRV;
		m_Impl->Transition(depth, depthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
		{
			auto& surface = multisample ? m_Impl->MultisampleSurfaces[Impl::MultisampleMaterialFirst + index] : m_Impl->GBufferSurfaces[index];
			m_Impl->Transition(surface.Resource.Get(), surface.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}
		if (shadowAvailable && desc.SamplesShadowMap)
			m_Impl->Transition(shadow->second.Resource.Get(), shadow->second.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		auto& scene = multisample ? m_Impl->MultisampleSurfaces[Impl::MultisampleColor] : m_Impl->HdrSurfaces[Impl::HdrScene];
		m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
		const auto target = m_Impl->RTVArena.Cpu(scene.RTV);
		m_Impl->CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
		m_Impl->CommandList->SetPipelineState(m_Impl->DeferredLightingPipelines.at(m_Impl->DeferredPipelineKey()).Get());
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->DeferredRootSignature.Get());
        m_Impl->BindRayShadowImage(DeferredRayShadowImageRoot);
        if (m_Impl->TiledActive)
        {
            const auto base = m_Impl->ConstantBuffer->GetGPUVirtualAddress();
            m_Impl->CommandList->SetGraphicsRootConstantBufferView(DeferredSurfaceRoot, base + surfaceOffset);
            m_Impl->CommandList->SetGraphicsRootConstantBufferView(DeferredListRoot, base + selectionOffset);
            const UINT views[] {m_Impl->ForwardPlusSceneView, m_Impl->ForwardPlusHeaderView, m_Impl->ForwardPlusIndexView};
            for (UINT index = 0; index < ForwardPlusStorageCount; ++index)
                m_Impl->CommandList->SetGraphicsRootDescriptorTable(DeferredStorageFirstRoot + index, m_Impl->SRVArena.Gpu(views[index]));
            m_Impl->CommandList->SetGraphicsRoot32BitConstant(DeferredContactControlRoot, m_Impl->ContactReceiverActive ? 1u : 0u, 0);
            m_Impl->CommandList->SetGraphicsRootDescriptorTable(DeferredContactImageRoot,
                m_Impl->SRVArena.Gpu(m_Impl->ContactReceiverActive ? m_Impl->ContactDescriptor : m_Impl->ForwardPlusEmptyContactView));
        }
        if (m_Impl->FrameContact.Enabled) m_Impl->BindContactReceiver(DeferredContactControlRoot, DeferredContactImageRoot);
        if (m_Impl->UsesDedicatedPCF() && !m_Impl->BindPCF(DeferredShadowFilterRoot,shadowAvailable && desc.SamplesShadowMap)) return false;
		if (m_Impl->UsesPCSS() && !m_Impl->BindPCSS(DeferredShadowFilterRoot, desc.LightViewProjection,
			desc.Lighting, shadowAvailable && desc.SamplesShadowMap)) return false;
		if (m_Impl->UsesVSM() && !m_Impl->BindVSM(DeferredShadowFilterRoot, desc.LightViewProjection,
			desc.Lighting, shadowAvailable && desc.SamplesShadowMap)) return false;
		m_Impl->CommandList->SetGraphicsRootConstantBufferView(0, m_Impl->ConstantBuffer->GetGPUVirtualAddress() + offset);
		UINT shadowDescriptor = shadowAvailable && desc.SamplesShadowMap ? shadow->second.SRV : m_Impl->NullShadowSRV;
		if (m_Impl->UsesVSM() && shadowAvailable && desc.SamplesShadowMap &&
			!m_Impl->VSMMomentDescriptor(m_Impl->LastShadowTexture, shadowDescriptor)) return false;
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(shadowDescriptor));
		for (UINT index = 0; index < GBufferAttachmentCount; ++index)
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(index + 2, m_Impl->SRVArena.Gpu(multisample ? m_Impl->MultisampleSurfaces[Impl::MultisampleMaterialFirst + index].SRV : m_Impl->GBufferSurfaces[index].SRV));
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(static_cast<UINT>(GBufferAttachmentCount) + 2, m_Impl->SRVArena.Gpu(depthView));
		if (!m_Impl->BindCascadedShadows(desc.SamplesShadowMap, true)) return false;
		D3D12_VIEWPORT viewport {};
		viewport.Width = static_cast<float>(m_Impl->CreateInfo.Width);
		viewport.Height = static_cast<float>(m_Impl->CreateInfo.Height);
		viewport.MaxDepth = 1.0f;
		const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
		m_Impl->CommandList->RSSetViewports(1, &viewport);
		m_Impl->CommandList->RSSetScissorRects(1, &scissor);
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
        if (m_Impl->SubsurfaceProducing)
        { ++m_Impl->PendingFrame.Statistics.SubsurfacePassCount; return true; }
        m_Impl->SubsurfaceLighting=desc;
        m_Impl->SubsurfaceHasScene=desc.Output==DeferredOutputMode::Scene;
		PassDesc pass;
		pass.Type = PassType::DeferredLighting;
		pass.ClearColorEnabled = pass.ClearDepthEnabled = false;
		m_Impl->PendingFrame.Passes.push_back({pass, {}});
		++m_Impl->PendingFrame.Statistics.DeferredLightingPassCount;
		m_Impl->DeferredResolved = true;
		m_Impl->SceneColorWritten = true;
		return true;
	}

	bool DX12RHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyAmbientOcclusion")) return false;
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("DX12 AO requires resolved sample depth and materials");
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->AmbientOcclusionActive ||
			!m_Impl->GBufferWritten || !m_Impl->SceneColorWritten || m_Impl->AmbientOcclusionApplied ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) || m_Impl->DepthOfFieldResolved || m_Impl->VelocityCaptured || m_Impl->ReflectionApplied)
			return m_Impl->Fail("DX12 Ambient occlusion requires completed material capture and opaque lighting, once per frame");
		using Pass = Techniques::AmbientOcclusionPassKind;
		const auto passes = Techniques::BuildAmbientOcclusionPlan(m_Impl->FrameAmbientOcclusion,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
		const bool scalable = m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO;
		RayTracedAOConstants packet {};
        auto& constants = packet.Surface;
        const bool rayTraced = m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::RayTraced;
		if (!BuildAmbientOcclusionFrameConstants(desc, m_Impl->FrameAmbientOcclusion, m_Impl->FrameRayScene.get(),
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Pass::Estimate, packet))
			return m_Impl->Fail("DX12 Ambient occlusion requires a finite, invertible camera projection");
		if (scalable) constants.Options.Z = m_Impl->SaoDepthMipCount - 1;
		const std::size_t firstOffset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		const std::size_t stride = AlignUp(sizeof(packet), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		const std::size_t required = (passes.size() - 1) * stride + sizeof(packet);
		if (firstOffset > m_Impl->ConstantBufferCapacity || required > m_Impl->ConstantBufferCapacity - firstOffset)
			return m_Impl->Fail("DX12 ambient occlusion constant upload capacity is exhausted");
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		for (std::size_t index : {std::size_t(1), std::size_t(3)})
		{
			auto& surface = m_Impl->GBufferSurfaces[index];
			m_Impl->Transition(surface.Resource.Get(), surface.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->AmbientOcclusionRootSignature.Get());
        if (rayTraced) for (UINT slot=0;slot<RaySceneBufferCount;++slot)
            m_Impl->CommandList->SetGraphicsRootDescriptorTable(5+slot,m_Impl->SRVArena.Gpu(m_Impl->RaySceneViews[slot]));
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(m_Impl->MainDepthSRV));
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(2, m_Impl->SRVArena.Gpu(m_Impl->GBufferSurfaces[1].SRV));
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(3, m_Impl->SRVArena.Gpu(m_Impl->GBufferSurfaces[3].SRV));
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		const D3D12_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(m_Impl->CreateInfo.Width),
			static_cast<float>(m_Impl->CreateInfo.Height), 0.0f, 1.0f};
		const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
		m_Impl->CommandList->RSSetViewports(1, &viewport);
		m_Impl->CommandList->RSSetScissorRects(1, &scissor);
		UINT destinationMip = 0;
		for (std::size_t index = 0; index < passes.size(); ++index)
		{
			const Pass pass = passes[index];
			const bool composite = pass == Pass::Composite;
			const bool hierarchy = pass == Pass::DepthPrepare || pass == Pass::DepthReduce;
			if (pass == Pass::DepthReduce) ++destinationMip;
			constants.Options.X = static_cast<std::uint32_t>(pass);
			constants.Options.Y = hierarchy ? destinationMip : 0;
			const std::size_t offset = firstOffset + index * stride;
			std::memcpy(m_Impl->ConstantData + offset, &packet, sizeof(packet));
			m_Impl->ConstantOffset = offset + sizeof(packet);
			m_Impl->CommandList->SetGraphicsRootConstantBufferView(0, m_Impl->ConstantBuffer->GetGPUVirtualAddress() + offset);
			UINT previousVisibility = m_Impl->NullColorSRV;
			if (!hierarchy && pass != Pass::Estimate)
			{
				auto& source = m_Impl->AmbientOcclusionSurfaces[pass == Pass::BilateralVertical ? 1 : 0];
				m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				previousVisibility = source.SRV;
			}
			UINT depthInput = m_Impl->MainDepthSRV;
			if (scalable && pass != Pass::DepthPrepare)
			{
				m_Impl->Transition(m_Impl->SaoDepthHierarchy.Resource.Get(), m_Impl->SaoDepthHierarchy.State,
					D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				depthInput = m_Impl->SaoDepthHierarchy.SRV;
			}
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(depthInput));
			auto& target = hierarchy ? m_Impl->SaoDepthScratch : (composite ? m_Impl->HdrSurfaces[Impl::HdrScene]
				: m_Impl->AmbientOcclusionSurfaces[pass == Pass::BilateralHorizontal ? 1 : 0]);
			m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			const UINT passWidth = hierarchy ? std::max(1, m_Impl->CreateInfo.Width >> destinationMip) : m_Impl->CreateInfo.Width;
			const UINT passHeight = hierarchy ? std::max(1, m_Impl->CreateInfo.Height >> destinationMip) : m_Impl->CreateInfo.Height;
			const D3D12_VIEWPORT passViewport {0.0f, 0.0f, static_cast<float>(passWidth), static_cast<float>(passHeight), 0.0f, 1.0f};
			const D3D12_RECT passScissor {0, 0, static_cast<LONG>(passWidth), static_cast<LONG>(passHeight)};
			m_Impl->CommandList->RSSetViewports(1, &passViewport);
			m_Impl->CommandList->RSSetScissorRects(1, &passScissor);
			const auto targetView = m_Impl->RTVArena.Cpu(target.RTV);
			m_Impl->CommandList->OMSetRenderTargets(1, &targetView, FALSE, nullptr);
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(4, m_Impl->SRVArena.Gpu(previousVisibility));
			const std::size_t pipeline = hierarchy ? Impl::HierarchyPipelineIndex : (composite ? 1 : 0);
			m_Impl->CommandList->SetPipelineState(m_Impl->AmbientOcclusionPipelines[m_Impl->AmbientOcclusionProgramIndex()][pipeline].Get());
			m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
			if (hierarchy)
			{
				// Only whole-resource states are needed: the chain is read while a
				// distinct scratch target is written, then receives one mip rectangle.
				m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
				m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(m_Impl->NullShadowSRV));
				m_Impl->Transition(m_Impl->SaoDepthScratch.Resource.Get(), m_Impl->SaoDepthScratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
				m_Impl->Transition(m_Impl->SaoDepthHierarchy.Resource.Get(), m_Impl->SaoDepthHierarchy.State, D3D12_RESOURCE_STATE_COPY_DEST);
				D3D12_TEXTURE_COPY_LOCATION source {}, destination {};
				source.pResource = m_Impl->SaoDepthScratch.Resource.Get();
				source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				destination.pResource = m_Impl->SaoDepthHierarchy.Resource.Get();
				destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				destination.SubresourceIndex = destinationMip;
				const D3D12_BOX rectangle {0, 0, 0, passWidth, passHeight, 1};
				m_Impl->CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, &rectangle);
			}
			m_Impl->PendingFrame.AmbientOcclusionPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount;
		}
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->AmbientOcclusionApplied = true;
		return true;
	}

	bool DX12RHI::CompleteEnvironmentCapture()
	{
		constexpr UINT CompleteFaces = (1u << EnvironmentFaceCount) - 1u;
		if (!m_Impl->CheckThread("CompleteEnvironmentCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || (!m_Impl->FrameEnvironment.Source && m_Impl->EnvironmentFacesWritten != CompleteFaces))
			return m_Impl->Fail("DX12 environment filtering requires all six completed faces exactly once");
        if (const auto& source=m_Impl->FrameEnvironment.Source)
        {
            auto& cube=m_Impl->Environment.Cube;
            auto layout=cube.Resource->GetDesc(); layout.MipLevels=1;
            std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT,EnvironmentFaceCount> footprints{};
            UINT64 bytes=0;
            m_Impl->Device->GetCopyableFootprints(&layout,0,static_cast<UINT>(EnvironmentFaceCount),0,footprints.data(),nullptr,nullptr,&bytes);
            ComPtr<ID3D12Resource> upload;
            if (!m_Impl->CreateBufferResource(static_cast<std::size_t>(bytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,upload)) return false;
            void* mapped=nullptr; const D3D12_RANGE noRead{0,0};
            const auto result=upload->Map(0,&noRead,&mapped);
            if (FAILED(result)) return m_Impl->FailHRESULT("Map(HDR environment)",result);
            const auto rowBytes=static_cast<std::size_t>(source->Resolution)*4*sizeof(std::uint16_t);
            for (UINT face=0;face<EnvironmentFaceCount;++face)
                for (UINT row=0;row<source->Resolution;++row)
                    std::memcpy(static_cast<std::uint8_t*>(mapped)+footprints[face].Offset+row*footprints[face].Footprint.RowPitch,
                        source->FacesRGBA16[face].data()+static_cast<std::size_t>(row)*source->Resolution*4,rowBytes);
            upload->Unmap(0,nullptr);
            m_Impl->CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
            m_Impl->Transition(cube.Resource.Get(),cube.State,D3D12_RESOURCE_STATE_COPY_DEST);
            for (UINT face=0;face<EnvironmentFaceCount;++face)
            {
                D3D12_TEXTURE_COPY_LOCATION from{}; from.pResource=upload.Get();
                from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint=footprints[face];
                D3D12_TEXTURE_COPY_LOCATION to{}; to.pResource=cube.Resource.Get();
                to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; to.SubresourceIndex=face*m_Impl->Environment.MipCount;
                m_Impl->CommandList->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            }
            m_Impl->FrameUploads.push_back(std::move(upload));
        }
		if (m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude)
		{
			if (!m_Impl->ConvertEnvironmentMap()) return false;
			m_Impl->EnvironmentReady = true;
			return true;
		}
		auto& environment = m_Impl->Environment;
		auto& cube = environment.Cube; auto& scratch = environment.Scratch;
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->ReflectionRootSignature.Get());
		m_Impl->CommandList->SetPipelineState(m_Impl->CubeMipPipeline.Get());
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		for (UINT mip = 1; mip < environment.MipCount; ++mip)
		{
			const UINT extent = environment.Resolution >> mip;
			const D3D12_VIEWPORT viewport {0, 0, float(extent), float(extent), 0, 1};
			const D3D12_RECT scissor {0, 0, static_cast<LONG>(extent), static_cast<LONG>(extent)};
			m_Impl->CommandList->RSSetViewports(1, &viewport); m_Impl->CommandList->RSSetScissorRects(1, &scissor);
			for (UINT face = 0; face < EnvironmentFaceCount; ++face)
			{
				m_Impl->Transition(cube.Resource.Get(), cube.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				std::array<UINT, Impl::ReflectionTextureCount> inputs; inputs.fill(m_Impl->NullColorSRV);
				inputs[1] = m_Impl->NullShadowSRV;
				inputs[7] = cube.SRV; inputs[8] = environment.ParentMipSRVs[mip - 1];
				for (UINT slot = 0; slot < inputs.size(); ++slot)
					m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
				CubemapReflectionConstants constants {}; constants.Options.Z = face;
				if (!m_Impl->UploadReflectionConstants(constants)) return false;
				const auto target = m_Impl->RTVArena.Cpu(scratch.RTV);
				m_Impl->CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
				m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
				// Keep all cube subresources in one tracked state. Reduce to a separate
				// target before copying, so cancellation restores only three resource states.
				m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
				m_Impl->Transition(cube.Resource.Get(), cube.State, D3D12_RESOURCE_STATE_COPY_DEST);
				D3D12_TEXTURE_COPY_LOCATION source {}; source.pResource = scratch.Resource.Get();
				source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				D3D12_TEXTURE_COPY_LOCATION destination {}; destination.pResource = cube.Resource.Get();
				destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				destination.SubresourceIndex = mip + face * environment.MipCount;
				const D3D12_BOX region {0, 0, 0, extent, extent, 1};
				m_Impl->CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, &region);
				++m_Impl->PendingFrame.Statistics.EnvironmentMipPassCount;
			}
		}
		m_Impl->Transition(cube.Resource.Get(), cube.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->EnvironmentReady = true;
		return true;
	}

	bool DX12RHI::CompletePlanarCapture()
	{
		if (!m_Impl->CheckThread("CompletePlanarCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->PlanarActive || !m_Impl->PlanarWritten || m_Impl->PlanarReady)
			return m_Impl->Fail("DX12 planar mip generation requires one completed capture exactly once per frame");
		auto& image = m_Impl->Planar.Image; auto& scratch = m_Impl->Planar.Scratch;
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->ReflectionRootSignature.Get());
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->CommandList->SetPipelineState(m_Impl->PlanarMipPipeline.Get());
		std::array<UINT, Impl::ReflectionTextureCount> inputs;
		inputs.fill(m_Impl->NullColorSRV); inputs[1] = m_Impl->NullShadowSRV;
		for (UINT mip = 1; mip < m_Impl->FramePlanar.MipCount; ++mip)
		{
			const UINT width = std::max(1u, m_Impl->FramePlanar.Width >> mip), height = std::max(1u, m_Impl->FramePlanar.Height >> mip);
			m_Impl->Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			inputs[10] = m_Impl->Planar.ParentMipSRVs[mip - 1];
			for (UINT slot = 0; slot < inputs.size(); ++slot)
				m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
			PlanarReflectionConstants constants {};
			constants.Viewport = {float(width), float(height), float(std::max(1u, m_Impl->FramePlanar.Width >> (mip - 1))),
				float(std::max(1u, m_Impl->FramePlanar.Height >> (mip - 1)))};
			if (!m_Impl->UploadReflectionConstants(constants)) return false;
			const D3D12_VIEWPORT viewport {0, 0, float(width), float(height), 0, 1};
			const D3D12_RECT scissor {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
			m_Impl->CommandList->RSSetViewports(1, &viewport); m_Impl->CommandList->RSSetScissorRects(1, &scissor);
			const auto target = m_Impl->RTVArena.Cpu(scratch.RTV);
			m_Impl->CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
			m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			// Distinct source and target images permit whole-resource barriers and
			// one frame-start state per image, including rollback of unsubmitted work.
			m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_Impl->Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_COPY_DEST);
			D3D12_TEXTURE_COPY_LOCATION source {}; source.pResource = scratch.Resource.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			D3D12_TEXTURE_COPY_LOCATION destination {}; destination.pResource = image.Resource.Get();
			destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; destination.SubresourceIndex = mip;
			const D3D12_BOX rectangle {0, 0, 0, width, height, 1};
			m_Impl->CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, &rectangle);
			++m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
		}
		m_Impl->Transition(image.Resource.Get(), image.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->PlanarReady = true;
		return true;
	}

    bool DX12RHI::ApplyPathTracing(const PathTracingDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyPathTracing") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PathFrame.Active || m_Impl->PathFrame.Resolved || !m_Impl->HdrSceneActive || m_Impl->SceneColorWritten)
        return m_Impl->Fail("Path tracing requires an untouched HDR frame with primary-ray visibility");
    return m_Impl->TracePaths(desc);
}

bool DX12RHI::ApplyGlobalIllumination(const GlobalIlluminationDesc& desc)
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
            m_Impl->FogResolved || m_Impl->LightShaftResolved || m_Impl->TemporalResolved ||
            m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
            return m_Impl->Fail("DX12 SSGI requires opaque material, depth and velocity before reflections and volumes");
        if (Techniques::UsesLumen(m_Impl->FrameGI)) return m_Impl->ApplyLumenGI(desc);
        if (m_Impl->FrameVoxelGI.Volume) return m_Impl->ApplyVoxelGI(desc);
        if (Techniques::UsesRayTracedGI(m_Impl->FrameGI)) return m_Impl->ApplyRayGI(desc);
        if (m_Impl->LightmapActive) return m_Impl->ResolveLightmaps(desc);
        if (m_Impl->ProbeActive) return m_Impl->ResolveProbes(desc);
        GlobalIlluminationConstants constants;
        if (!BuildGlobalIlluminationConstants(desc, m_Impl->FrameGI, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->GI.Valid, constants))
            return m_Impl->Fail("DX12 SSGI requires invertible current and previous projections");
        auto& scene = m_Impl->HdrSurfaces[Impl::HdrScene];
        auto& velocity = m_Impl->MotionSurfaces[Impl::MotionVelocity];
        auto& images = m_Impl->GISurfaces;
        const unsigned read = m_Impl->GI.ReadIndex, write = m_Impl->GI.WriteIndex();
        m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        constexpr auto Sampled = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        m_Impl->Transition(scene.Resource.Get(), scene.State, Sampled);
        m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, Sampled);
        m_Impl->Transition(velocity.Resource.Get(), velocity.State, Sampled);
        for (auto& material : m_Impl->GBufferSurfaces) m_Impl->Transition(material.Resource.Get(), material.State, Sampled);
        for (auto& image : images) m_Impl->Transition(image.Resource.Get(), image.State, Sampled);
        m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->GIRootSignature.Get());
        const D3D12_VIEWPORT viewport {0, 0, float(scene.Width), float(scene.Height), 0, 1};
        const D3D12_RECT scissor {0, 0, scene.Width, scene.Height};
        m_Impl->CommandList->RSSetViewports(1, &viewport); m_Impl->CommandList->RSSetScissorRects(1, &scissor);
        m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        using Phase = Techniques::GlobalIlluminationPassKind;
        for (auto phase : {Phase::Trace, Phase::Spatial, Phase::Temporal, Phase::Composite})
        {
            constants.Options.X = static_cast<unsigned>(phase);
            const auto offset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (offset > m_Impl->ConstantBufferCapacity || sizeof(constants) > m_Impl->ConstantBufferCapacity - offset)
                return m_Impl->Fail("DX12 SSGI exhausted the frame upload buffer");
            std::memcpy(m_Impl->ConstantData + offset, &constants, sizeof(constants));
            m_Impl->CommandList->SetGraphicsRootConstantBufferView(0, m_Impl->ConstantBuffer->GetGPUVirtualAddress() + offset);
            m_Impl->ConstantOffset = offset + sizeof(constants);
            const auto target = phase == Phase::Trace ? Impl::GIRaw : phase == Phase::Spatial ? Impl::GIFiltered :
                phase == Phase::Temporal ? Impl::GIColorFirst + write : Impl::GIComposite;
            const auto estimate = phase == Phase::Spatial ? images[Impl::GIRaw].SRV : phase == Phase::Temporal ?
                images[Impl::GIFiltered].SRV : phase == Phase::Composite ? images[Impl::GIColorFirst + write].SRV : m_Impl->NullColorSRV;
            const UINT inputs[] {scene.SRV, m_Impl->MainDepthSRV, m_Impl->GBufferSurfaces[1].SRV,
                m_Impl->GBufferSurfaces[0].SRV, m_Impl->GBufferSurfaces[3].SRV, velocity.SRV, estimate,
                images[Impl::GIColorFirst + read].SRV, images[Impl::GIDepthFirst + read].SRV, images[Impl::GINormalFirst + read].SRV};
            for (UINT slot = 0; slot < std::size(inputs); ++slot)
                m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
            const std::size_t outputIndices[] {target, Impl::GIDepthFirst + write, Impl::GINormalFirst + write};
            const unsigned count = phase == Phase::Temporal ? 3 : 1;
            D3D12_CPU_DESCRIPTOR_HANDLE outputs[3] {};
            for (unsigned i = 0; i < count; ++i)
            {
                auto& output = images[outputIndices[i]];
                m_Impl->Transition(output.Resource.Get(), output.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
                outputs[i] = m_Impl->RTVArena.Cpu(output.RTV);
            }
            m_Impl->CommandList->OMSetRenderTargets(count, outputs, FALSE, nullptr);
            m_Impl->CommandList->SetPipelineState(phase == Phase::Temporal ? m_Impl->GITemporalPipeline.Get() : m_Impl->GIPipeline.Get());
            m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
            m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
            for (unsigned i = 0; i < count; ++i)
                m_Impl->Transition(images[outputIndices[i]].Resource.Get(), images[outputIndices[i]].State, Sampled);
        }
        auto& composite = images[Impl::GIComposite];
        m_Impl->Transition(composite.Resource.Get(), composite.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_COPY_DEST);
        m_Impl->CommandList->CopyResource(scene.Resource.Get(), composite.Resource.Get());
        m_Impl->GI.Resolved = true;
        m_Impl->PendingFrame.Temporal = desc.Sampling;
        m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount = 4;
        return true;
    }

	bool DX12RHI::ApplyReflections(const ReflectionDesc& desc)
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
			(m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady) ||
			(m_Impl->PlanarActive && !m_Impl->PlanarReady) ||
			!m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || !m_Impl->GBufferWritten ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved) ||
			(m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("DX12 reflections require resolved opaque lighting/materials and AO, once before blur/transparency");
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
				return m_Impl->Fail("DX12 SSR requires a finite invertible camera projection");
			if (cube && !BuildCubemapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameEnvironment.MipCount, cubeConstants))
				return m_Impl->Fail("DX12 cubemap reflections require a finite invertible camera projection");
			if (environmentMap && !BuildEnvironmentMapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->EnvironmentMap.MipCount, mapConstants))
				return m_Impl->Fail("DX12 environment map reflections require a finite invertible camera projection");
			if (planar && !BuildPlanarReflectionConstants(desc, m_Impl->FramePlanar, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, planarConstants))
				return m_Impl->Fail("DX12 planar lookup requires the frozen capture camera and main viewport");
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			for (std::size_t index = 0; index < GBufferAttachmentCount - 1; ++index)
			{
				auto& material = m_Impl->GBufferSurfaces[index];
				m_Impl->Transition(material.Resource.Get(), material.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			}
			m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->ReflectionRootSignature.Get());
			m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(m_Impl->CreateInfo.Width),
				static_cast<float>(m_Impl->CreateInfo.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
			m_Impl->CommandList->RSSetViewports(1, &viewport);
			m_Impl->CommandList->RSSetScissorRects(1, &scissor);
			auto& scene = m_Impl->HdrSurfaces[Impl::HdrScene];
			auto& reflection = m_Impl->ReflectionSurface;
			using Pass = Techniques::ReflectionPassKind;
			for (auto phase : {planar ? Pass::PlanarLookup : Techniques::UsesEnvironmentCapture(m_Impl->FrameReflections) ? Pass::EnvironmentLookup : Pass::Trace, Pass::Composite})
			{
				const bool composite = phase == Pass::Composite;
				auto& source = composite ? reflection : scene;
				auto& destination = composite ? scene : reflection;
				m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
				m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				std::array<UINT, Impl::ReflectionTextureCount> inputs;
				inputs.fill(m_Impl->NullColorSRV);
				inputs[1] = m_Impl->NullShadowSRV;
				if (composite) inputs[6] = reflection.SRV;
				else
				{
					inputs[0] = scene.SRV; inputs[1] = m_Impl->MainDepthSRV;
					inputs[2] = m_Impl->GBufferSurfaces[1].SRV; inputs[3] = m_Impl->GBufferSurfaces[2].SRV;
					inputs[4] = m_Impl->GBufferSurfaces[0].SRV; inputs[5] = m_Impl->GBufferSurfaces[3].SRV;
				}
				if (m_Impl->EnvironmentActive) inputs[7] = m_Impl->Environment.Cube.SRV;
				if (cube) inputs[8] = m_Impl->Environment.ParentMipSRVs[0];
				if (environmentMap) inputs[9] = m_Impl->EnvironmentMap.Image.SRV;
				if (planar) inputs[9] = m_Impl->Planar.Image.SRV;
				for (UINT index = 0; index < inputs.size(); ++index)
					m_Impl->CommandList->SetGraphicsRootDescriptorTable(index + 1, m_Impl->SRVArena.Gpu(inputs[index]));
				constants.Options.X = static_cast<std::uint32_t>(phase);
				if (planar)
				{
					planarConstants.Options.X = static_cast<std::uint32_t>(phase);
					if (!m_Impl->UploadReflectionConstants(planarConstants)) return false;
				}
				else if (environmentMap)
				{
					mapConstants.Options.X = static_cast<std::uint32_t>(phase);
					if (!m_Impl->UploadReflectionConstants(mapConstants)) return false;
				}
				else if (cube)
				{
					cubeConstants.Options.X = static_cast<std::uint32_t>(phase);
					if (!m_Impl->UploadReflectionConstants(cubeConstants)) return false;
				}
				else if (!m_Impl->UploadReflectionConstants(constants)) return false;
				const auto target = m_Impl->RTVArena.Cpu(destination.RTV);
				m_Impl->CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
				m_Impl->CommandList->SetPipelineState(m_Impl->ReflectionPipelines[program][composite ? 1 : 0].Get());
				m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
				m_Impl->PendingFrame.ReflectionPasses.push_back(phase);
				++m_Impl->PendingFrame.Statistics.ReflectionPassCount;
			}
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		}
		m_Impl->ReflectionApplied = true;
		return true;
	}

	bool DX12RHI::ApplyDepthOfField(const DepthOfFieldDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 light shafts must complete before depth of field");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->FogResolved)
			return m_Impl->Fail("DX12 fog must complete before depth of field");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalResolved)
			return m_Impl->Fail("DX12 TAA must complete before depth of field");
		if (!m_Impl->CheckThread("ApplyDepthOfField")) return false;
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
			return m_Impl->Fail("DX12 depth of field requires resolved sample color and depth");
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePostProcessing.DepthOfField.Enabled ||
			!m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || m_Impl->DepthOfFieldResolved ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->MotionBlurResolved))
			return m_Impl->Fail("DX12 depth of field requires completed opaque lighting and ambient occlusion, once before transparency");
		using Pass = Techniques::DepthOfFieldPassKind;
		DepthOfFieldConstants constants {};
		if (!BuildDepthOfFieldConstants(desc, m_Impl->FramePostProcessing.DepthOfField,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Pass::Gather, constants))
			return m_Impl->Fail("DX12 depth of field requires a finite camera with a supported projection");
		const auto passes = Techniques::BuildDepthOfFieldPlan(m_Impl->FramePostProcessing.DepthOfField);
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->HdrRootSignature.Get());
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		const D3D12_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(m_Impl->CreateInfo.Width),
			static_cast<float>(m_Impl->CreateInfo.Height), 0.0f, 1.0f};
		const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
		m_Impl->CommandList->RSSetViewports(1, &viewport);
		m_Impl->CommandList->RSSetScissorRects(1, &scissor);
		for (const auto pass : passes)
		{
			const bool composite = pass == Pass::Composite;
			auto& source = m_Impl->HdrSurfaces[composite ? Impl::HdrScratch : Impl::HdrScene];
			auto& target = m_Impl->HdrSurfaces[composite ? Impl::HdrScene : Impl::HdrScratch];
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			constants.Options.X = static_cast<std::uint32_t>(pass);
			m_Impl->CommandList->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(source.SRV));
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(2, m_Impl->SRVArena.Gpu(m_Impl->MainDepthSRV));
			const auto targetHandle = m_Impl->RTVArena.Cpu(target.RTV);
			m_Impl->CommandList->OMSetRenderTargets(1, &targetHandle, FALSE, nullptr);
			m_Impl->CommandList->SetPipelineState(m_Impl->DepthOfFieldPipelines[composite ? 1 : 0].Get());
			m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
			m_Impl->PendingFrame.DepthOfFieldPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount;
		}
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->DepthOfFieldResolved = true;
		return true;
	}

	bool DX12RHI::ApplyLightShafts(const LightShaftDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyLightShafts") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!Techniques::UsesLightShafts(m_Impl->FrameLightShafts) || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten ||
			m_Impl->LightShaftResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved ||
			m_Impl->DepthOfFieldResolved || (m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->FogResolved) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
			return m_Impl->Fail("DX12 light shafts require resolved opaque lighting, AO, reflections and fog, once before temporal effects and transparency");
		LightShaftConstants constants;
		if (!BuildLightShaftConstants(desc, m_Impl->FrameLightShafts, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
			return m_Impl->Fail("DX12 light shafts require finite camera/light inputs and valid dimensions");
		auto& mask = m_Impl->LightShaftMask;
		auto& scene = m_Impl->HdrSurfaces[Impl::HdrScene];
		auto& scratch = m_Impl->HdrSurfaces[Impl::HdrScratch];
		auto* commands = m_Impl->CommandList.Get();
		commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		commands->SetGraphicsRootSignature(m_Impl->LightShaftRootSignature.Get());
		commands->SetPipelineState(m_Impl->LightShaftPipeline.Get());
		commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		const D3D12_VIEWPORT viewport {0, 0, float(m_Impl->CreateInfo.Width), float(m_Impl->CreateInfo.Height), 0, 1};
		const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
		commands->RSSetViewports(1, &viewport); commands->RSSetScissorRects(1, &scissor);
		m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		for (UINT phase = 0; phase < 2; ++phase)
		{
			constants.Control.Y = phase;
			// Root arguments are versioned with the draw commands. Changing the
			// phase never rewrites a pending draw's upload memory or SRV descriptors.
			commands->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
			std::array<UINT, 3> inputs {m_Impl->NullColorSRV, m_Impl->MainDepthSRV, m_Impl->NullColorSRV};
			if (phase != 0)
			{
				m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(mask.Resource.Get(), mask.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				inputs = {scene.SRV, m_Impl->MainDepthSRV, mask.SRV};
			}
			auto& destination = phase == 0 ? mask : scratch;
			m_Impl->Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			for (UINT index = 0; index < inputs.size(); ++index)
				commands->SetGraphicsRootDescriptorTable(index + 1, m_Impl->SRVArena.Gpu(inputs[index]));
			const auto target = m_Impl->RTVArena.Cpu(destination.RTV);
			commands->OMSetRenderTargets(1, &target, FALSE, nullptr);
			constexpr UINT FullscreenVertexCount = 3;
			commands->DrawInstanced(FullscreenVertexCount, 1, 0, 0);
			commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		}
		m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
		m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_COPY_DEST);
		commands->CopyResource(scene.Resource.Get(), scratch.Resource.Get());
		m_Impl->LightShaftResolved = true;
		m_Impl->PendingFrame.Statistics.LightShaftPassCount += 2;
		return true;
	}

	bool DX12RHI::ApplyVolumetricFog(const VolumetricFogDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		if (!m_Impl->CheckThread("ApplyVolumetricFog") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!Techniques::UsesVolumetricFog(m_Impl->FrameFog) || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten ||
			m_Impl->FogResolved || m_Impl->LightShaftResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved ||
			m_Impl->DepthOfFieldResolved || (m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied) ||
			(m_Impl->ReflectionActive && !m_Impl->ReflectionApplied) ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
			return m_Impl->Fail("DX12 fog requires completed opaque lighting, AO and reflections, once before temporal effects and transparency");
		if (!BuildVolumetricFogConstants(desc, m_Impl->FrameFog, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
			D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION, m_Impl->FogConstants))
			return m_Impl->Fail("DX12 fog requires a finite invertible camera frustum");
		const bool lighting = (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog));
		if (lighting && !m_Impl->GenerateVolumeLighting(desc)) return false;
		auto& atlas = m_Impl->FogAtlas;
		auto& scene = m_Impl->HdrSurfaces[Impl::HdrScene];
		auto& scratch = m_Impl->HdrSurfaces[Impl::HdrScratch];
		auto* commands = m_Impl->CommandList.Get();
		commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		commands->SetGraphicsRootSignature(m_Impl->FogRootSignature.Get());
		commands->SetPipelineState((Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->CloudFogPipeline : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->MediaFogPipeline : m_Impl->FogPipeline).Get());
		commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		for (UINT phase = lighting ? 1U : 0U; phase < 2; ++phase)
		{
			auto constants = m_Impl->FogConstants;
			constants.Options.X = phase;
			commands->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
			std::array<UINT, 3> inputs {m_Impl->NullColorSRV, m_Impl->NullShadowSRV, m_Impl->NullColorSRV};
			if (phase != 0)
			{
				m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(atlas.Resource.Get(), atlas.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				inputs = {scene.SRV, m_Impl->MainDepthSRV, atlas.SRV};
			}
			auto& destination = phase == 0 ? atlas : scratch;
			m_Impl->Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			for (UINT slot = 0; slot < inputs.size(); ++slot)
				commands->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
			const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(destination.Width), static_cast<float>(destination.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, destination.Width, destination.Height};
			commands->RSSetViewports(1, &viewport); commands->RSSetScissorRects(1, &scissor);
			const auto target = m_Impl->RTVArena.Cpu(destination.RTV);
			commands->OMSetRenderTargets(1, &target, FALSE, nullptr);
			commands->DrawInstanced(3, 1, 0, 0);
			commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		}
		// The scene is never sampled while bound for writing. Atlas/Scratch/Scene
		// states participate in the frame snapshot: discarded recording rolls back
		// CPU state, while submitted commands retain their actual GPU transitions.
		m_Impl->Transition(scratch.Resource.Get(), scratch.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
		m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_COPY_DEST);
		commands->CopyResource(scene.Resource.Get(), scratch.Resource.Get());
		m_Impl->FogResolved = true;
		(Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricFogPassCount) += lighting ? 1 : 2;
		return true;
	}

	bool DX12RHI::ApplyTemporalEffects(const TemporalEffectsDesc& desc)
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

bool DX12RHI::ApplyTemporalAA(const TemporalAADesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 light shafts must complete before TAA");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->FogResolved)
			return m_Impl->Fail("DX12 fog must complete before TAA");
		if (!m_Impl->CheckThread("ApplyTemporalAA")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) ||
			!m_Impl->HdrSceneActive || !m_Impl->VelocityCaptured || m_Impl->TemporalResolved ||
			m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
			(m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved))
			return m_Impl->Fail("DX12 TAA requires completed opaque color/depth/velocity, once before blur and transparency");
		TemporalAAConstants constants;
		if (!BuildTemporalAAConstants(desc, m_Impl->FramePostProcessing.TemporalAA, m_Impl->CreateInfo.Width,
			m_Impl->CreateInfo.Height, m_Impl->TemporalHistoryValid, constants))
			return m_Impl->Fail("DX12 TAA requires finite invertible previous projection and valid dimensions");
		const auto writeIndex = 1 - m_Impl->TemporalHistoryReadIndex;
		auto& scene = m_Impl->HdrSurfaces[Impl::HdrScene];
		auto& velocity = m_Impl->MotionSurfaces[Impl::MotionVelocity];
		auto& outputColor = m_Impl->TemporalColorHistory[writeIndex];
		auto& outputDepth = m_Impl->TemporalDepthHistory[writeIndex];
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->Transition(velocity.Resource.Get(), velocity.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		std::array<UINT, 5> inputs {scene.SRV, m_Impl->MainDepthSRV, velocity.SRV,
			m_Impl->NullColorSRV, m_Impl->MainDepthSRV};
		if (m_Impl->TemporalHistoryValid)
		{
			auto& oldColor = m_Impl->TemporalColorHistory[m_Impl->TemporalHistoryReadIndex];
			auto& oldDepth = m_Impl->TemporalDepthHistory[m_Impl->TemporalHistoryReadIndex];
			m_Impl->Transition(oldColor.Resource.Get(), oldColor.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			m_Impl->Transition(oldDepth.Resource.Get(), oldDepth.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			inputs[3] = oldColor.SRV; inputs[4] = oldDepth.SRV;
		}
		m_Impl->Transition(outputColor.Resource.Get(), outputColor.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
		m_Impl->Transition(outputDepth.Resource.Get(), outputDepth.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->TemporalRootSignature.Get());
		m_Impl->CommandList->SetGraphicsRoot32BitConstants(0, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
		for (UINT slot = 0; slot < inputs.size(); ++slot)
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
		const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(outputColor.Width), static_cast<float>(outputColor.Height), 0, 1};
		const D3D12_RECT scissor {0, 0, outputColor.Width, outputColor.Height};
		m_Impl->CommandList->RSSetViewports(1, &viewport);
		m_Impl->CommandList->RSSetScissorRects(1, &scissor);
		const D3D12_CPU_DESCRIPTOR_HANDLE outputs[] {m_Impl->RTVArena.Cpu(outputColor.RTV), m_Impl->RTVArena.Cpu(outputDepth.RTV)};
		m_Impl->CommandList->OMSetRenderTargets(2, outputs, FALSE, nullptr);
		m_Impl->CommandList->SetPipelineState(m_Impl->TemporalPipeline.Get());
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		// Later effects write the scene copy, never the pending/committed history.
		// The read index changes only after submission, fence, diagnostics and
		// readback succeed. Discarded recording restores all four image states.
		m_Impl->Transition(outputColor.Resource.Get(), outputColor.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
		m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_COPY_DEST);
		m_Impl->CommandList->CopyResource(scene.Resource.Get(), outputColor.Resource.Get());
		m_Impl->TemporalResolved = true;
		m_Impl->PendingFrame.Temporal = desc.Sampling;
		++m_Impl->PendingFrame.Statistics.TemporalAAPassCount;
		return true;
	}

	bool DX12RHI::ApplyMotionBlur(const MotionBlurDesc& desc)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 light shafts must complete before motion blur");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->FogResolved)
			return m_Impl->Fail("DX12 fog must complete before motion blur");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalResolved)
			return m_Impl->Fail("DX12 TAA must complete before motion blur");
		if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved)
			return m_Impl->Fail("DX12 motion blur requires resolved sample velocity");
		if (!m_Impl->CheckThread("ApplyMotionBlur")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePostProcessing.MotionBlur.Enabled ||
			!m_Impl->HdrSceneActive || !m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("DX12 motion blur requires opaque velocity, once before depth of field/transparency");
		MotionBlurConstants constants {};
		if (!BuildMotionBlurConstants(desc, m_Impl->FramePostProcessing.MotionBlur,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
			return m_Impl->Fail("DX12 motion blur requires a finite camera and valid scene sample interval");
		const auto passes = Techniques::BuildMotionBlurPlan(m_Impl->FramePostProcessing.MotionBlur);
		const auto first = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		const auto stride = AlignUp(sizeof(constants), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		if (first > m_Impl->ConstantBufferCapacity || passes.size() * stride > m_Impl->ConstantBufferCapacity - first)
			return m_Impl->Fail("DX12 motion reconstruction exhausted the frame upload buffer");
		using Pass = Techniques::MotionBlurPassKind;
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		auto& velocity = m_Impl->MotionSurfaces[Impl::MotionVelocity];
		m_Impl->Transition(velocity.Resource.Get(), velocity.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->MotionRootSignature.Get());
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		for (const auto pass : passes)
		{
			const bool composite = pass == Pass::Composite;
			auto& destination = pass == Pass::TileMax ? m_Impl->MotionSurfaces[Impl::MotionTileMax] :
				(pass == Pass::NeighborMax ? m_Impl->MotionSurfaces[Impl::MotionNeighborMax] :
				m_Impl->HdrSurfaces[composite ? Impl::HdrScene : Impl::HdrScratch]);
			m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			std::array<UINT, 5> inputs {m_Impl->NullColorSRV, m_Impl->MainDepthSRV,
				velocity.SRV, m_Impl->NullColorSRV, m_Impl->NullColorSRV};
			if (pass == Pass::Gather || composite)
			{
				auto& source = m_Impl->HdrSurfaces[composite ? Impl::HdrScratch : Impl::HdrScene];
				m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				inputs[0] = source.SRV;
			}
			if (pass == Pass::NeighborMax || pass == Pass::Gather)
			{
				auto& maximum = m_Impl->MotionSurfaces[pass == Pass::NeighborMax ? Impl::MotionTileMax : Impl::MotionNeighborMax];
				m_Impl->Transition(maximum.Resource.Get(), maximum.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				inputs[3] = maximum.SRV;
			}
			m_Impl->Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			constants.Options.X = static_cast<std::uint32_t>(pass);
			if (!m_Impl->UploadMotionConstants(constants)) return false;
			for (UINT slot = 0; slot < inputs.size(); ++slot)
				m_Impl->CommandList->SetGraphicsRootDescriptorTable(slot + 1, m_Impl->SRVArena.Gpu(inputs[slot]));
			const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(destination.Width), static_cast<float>(destination.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, destination.Width, destination.Height};
			m_Impl->CommandList->RSSetViewports(1, &viewport);
			m_Impl->CommandList->RSSetScissorRects(1, &scissor);
			const auto target = m_Impl->RTVArena.Cpu(destination.RTV);
			m_Impl->CommandList->OMSetRenderTargets(1, &target, FALSE, nullptr);
			m_Impl->CommandList->SetPipelineState(m_Impl->MotionScreenPipelines[composite ? 1 : 0].Get());
			m_Impl->CommandList->DrawInstanced(3, 1, 0, 0);
			m_Impl->PendingFrame.MotionBlurPasses.push_back(pass);
			++m_Impl->PendingFrame.Statistics.MotionBlurPassCount;
		}
		m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		m_Impl->MotionBlurResolved = true;
		return true;
	}

    bool DX12RHI::ProcessDepthPeeling(DepthPeelingPass pass)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessDepthPeeling") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePeeling.Enabled ||
            !m_Impl->OpaqueEffectsComplete() || !m_Impl->SceneColorWritten || (m_Impl->DeferredActive && !m_Impl->DeferredResolved))
            return m_Impl->Fail("OIT requires completed opaque HDR processing outside a raster pass");
        if (!m_Impl->ProcessPeeling(pass)) return false;
        RecordDepthPeelingPass(m_Impl->FramePeeling,pass,m_Impl->PendingFrame.Statistics); return true;
    }

	bool DX12RHI::ProcessRefraction(const RefractionDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessRefraction") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->RefractionActive || m_Impl->RefractionResolved || m_Impl->FrameSampleCount!=1 ||
            !m_Impl->HdrSceneActive || !m_Impl->OpaqueEffectsComplete() || (!m_Impl->SceneColorWritten || (m_Impl->DeferredActive && !m_Impl->DeferredResolved)))
            return m_Impl->Fail("Refraction requires completed single-sample opaque HDR processing and executes once per frame");
        return m_Impl->ResolveRefraction(desc);
    }

    bool DX12RHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass)
	{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
		using Pass = Techniques::WeightedTransparencyPassKind;
		if (!m_Impl->CheckThread("ProcessWeightedTransparency") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
			!m_Impl->WeightedActive || !m_Impl->HdrSceneActive || !m_Impl->SceneColorWritten || !m_Impl->OpaqueEffectsComplete() ||
			(m_Impl->DeferredActive && !m_Impl->DeferredResolved) ||
			(m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved ||
				(m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored))))
			return m_Impl->Fail("DX12 weighted transparency requires completed opaque processing and restored sample color");
		if (pass == Pass::Prepare)
		{
			if (m_Impl->WeightedPrepared || m_Impl->SamplesTransparentWritten)
				return m_Impl->Fail("DX12 prepare transparency once before its geometry pass");
		}
		else if (pass == Pass::Resolve)
		{
			if (!m_Impl->WeightedPrepared || !m_Impl->WeightedAccumulated || m_Impl->WeightedResolved)
				return m_Impl->Fail("DX12 resolve transparency once after its geometry pass");
		}
		else return m_Impl->Fail("Unknown DX12 weighted transparency stage");

		auto* commands = m_Impl->CommandList.Get();
		commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		for (auto& surface : m_Impl->WeightedSurfaces)
			m_Impl->Transition(surface.Resource.Get(), surface.State, pass == Pass::Prepare ?
				D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		if (pass == Pass::Prepare)
		{
			const float zero[4] {}, one[4] {1, 1, 1, 1};
			commands->ClearRenderTargetView(m_Impl->RTVArena.Cpu(m_Impl->WeightedSurfaces[0].RTV), zero, 0, nullptr);
			commands->ClearRenderTargetView(m_Impl->RTVArena.Cpu(m_Impl->WeightedSurfaces[1].RTV), one, 0, nullptr);
			m_Impl->WeightedPrepared = true;
		}
		else
		{
			auto& scene = m_Impl->FrameSampleCount > 1 ? m_Impl->MultisampleSurfaces[Impl::MultisampleColor] : m_Impl->HdrSurfaces[Impl::HdrScene];
			m_Impl->Transition(scene.Resource.Get(), scene.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
			commands->SetGraphicsRootSignature(m_Impl->WeightedRootSignature.Get());
			ID3D12DescriptorHeap* heaps[] {m_Impl->SRVArena.Heap.Get()};
			commands->SetDescriptorHeaps(1, heaps);
			const UINT firstInput = m_Impl->FrameSampleCount > 1 ? 2 : 0;
			for (UINT index = 0; index < 4; ++index)
			{
				const bool used = index >= firstInput && index < firstInput + 2;
				commands->SetGraphicsRootDescriptorTable(index, m_Impl->SRVArena.Gpu(used ?
					m_Impl->WeightedSurfaces[index - firstInput].SRV : m_Impl->NullColorSRV));
			}
			commands->SetPipelineState(m_Impl->WeightedResolvePipelines.at(m_Impl->FrameSampleCount).Get());
			const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(scene.Width), static_cast<float>(scene.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, scene.Width, scene.Height};
			commands->RSSetViewports(1, &viewport); commands->RSSetScissorRects(1, &scissor);
			commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			const auto target = m_Impl->RTVArena.Cpu(scene.RTV);
			commands->OMSetRenderTargets(1, &target, FALSE, nullptr);
			commands->DrawInstanced(3, 1, 0, 0);
			commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
			m_Impl->WeightedResolved = true;
		}
		++m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount;
		return true;
	}

	bool DX12RHI::ProcessMultisampling(Techniques::MultisamplePassKind pass)
	{
    if (pass==Techniques::MultisamplePassKind::OpaqueResolve && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before opaque MSAA resolution");
		using Pass = Techniques::MultisamplePassKind;
		using Program = Impl::SampleProgram;
		if (!m_Impl->CheckThread("ProcessMultisampling") || !m_Impl->FrameOpen || m_Impl->PassOpen || m_Impl->FrameSampleCount <= 1)
			return m_Impl->Fail("DX12 sample processing requires a multisample frame outside a raster pass");
		switch (pass)
		{
		case Pass::OpaqueResolve:
			if (!m_Impl->SceneColorWritten || m_Impl->SamplesOpaqueResolved ||
				(m_Impl->MaterialCaptureActive && !m_Impl->GBufferWritten) || (m_Impl->DeferredActive && !m_Impl->DeferredResolved))
				return m_Impl->Fail("DX12 resolve opaque samples once after opaque lighting");
			break;
		case Pass::VelocityResolve:
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->VelocityCaptured || m_Impl->SamplesVelocityResolved || m_Impl->MotionBlurResolved || m_Impl->TemporalResolved)
				return m_Impl->Fail("DX12 resolve sample velocity once between velocity capture and motion blur");
			break;
		case Pass::ColorRestore:
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->NeedsSampleColorRestore() || !m_Impl->OpaqueEffectsComplete() ||
				m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)
				return m_Impl->Fail("DX12 restore processed opaque color once before transparent sample composition");
			break;
		case Pass::ColorResolve:
			if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
				return m_Impl->Fail("DX12 resolve weighted transparency before the final MSAA color resolve");
			if (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesTransparentWritten || m_Impl->SamplesColorResolved)
				return m_Impl->Fail("DX12 final sample resolve requires completed transparent composition");
			break;
		default: return m_Impl->Fail("Unknown DX12 multisample pass");
		}

		auto* commands = m_Impl->CommandList.Get();
		commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		auto& sampleColor = m_Impl->MultisampleSurfaces[Impl::MultisampleColor];
		if (pass == Pass::OpaqueResolve || pass == Pass::ColorResolve)
		{
			auto* output = m_Impl->HdrSceneActive ? m_Impl->HdrSurfaces[Impl::HdrScene].Resource.Get() : m_Impl->MainColor.Get();
			auto& outputState = m_Impl->HdrSceneActive ? m_Impl->HdrSurfaces[Impl::HdrScene].State : m_Impl->MainColorState;
			// ResolveSubresource requires explicit source/destination states.
			// Both transitions participate in the unsubmitted-frame journal.
			m_Impl->Transition(sampleColor.Resource.Get(), sampleColor.State, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
			m_Impl->Transition(output, outputState, D3D12_RESOURCE_STATE_RESOLVE_DEST);
			commands->ResolveSubresource(output, 0, sampleColor.Resource.Get(), 0, sampleColor.Format);
		}
		if (pass != Pass::ColorResolve)
		{
			const bool opaque = pass == Pass::OpaqueResolve;
			const bool material = opaque && m_Impl->MaterialCaptureActive;
			const auto program = opaque ? (material ? Program::Material : Program::Depth) :
				(pass == Pass::VelocityResolve ? Program::Velocity : Program::Restore);
			auto* pipeline = m_Impl->GetMultisamplePipeline(program);
			if (!pipeline) return false;
			commands->SetGraphicsRootSignature(m_Impl->MultisampleRootSignature.Get());
			commands->SetPipelineState(pipeline);
			auto& depth = m_Impl->MultisampleDepth;
			if (program != Program::Restore)
			{
				m_Impl->Transition(depth.Resource.Get(), depth.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				commands->SetGraphicsRootDescriptorTable(0, m_Impl->SRVArena.Gpu(depth.SRV));
			}
			if (opaque)
			{
				m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState, D3D12_RESOURCE_STATE_DEPTH_WRITE);
				const auto dsv = m_Impl->DSVArena.Cpu(m_Impl->MainDepthDSV);
				std::array<D3D12_CPU_DESCRIPTOR_HANDLE, GBufferAttachmentCount> targets {};
				if (material) for (UINT index = 0; index < GBufferAttachmentCount; ++index)
				{
					auto& source = m_Impl->MultisampleSurfaces[Impl::MultisampleMaterialFirst + index];
					auto& destination = m_Impl->GBufferSurfaces[index];
					m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					m_Impl->Transition(destination.Resource.Get(), destination.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
					targets[index] = m_Impl->RTVArena.Cpu(destination.RTV);
					commands->SetGraphicsRootDescriptorTable(index + 1, m_Impl->SRVArena.Gpu(source.SRV));
				}
				commands->OMSetRenderTargets(material ? GBufferAttachmentCount : 0, material ? targets.data() : nullptr, FALSE, &dsv);
			}
			else if (program == Program::Velocity)
			{
				auto& source = m_Impl->MultisampleSurfaces[Impl::MultisampleVelocity];
				auto& target = m_Impl->MotionSurfaces[Impl::MotionVelocity];
				m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(target.Resource.Get(), target.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				const auto rtv = m_Impl->RTVArena.Cpu(target.RTV);
				commands->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
				commands->SetGraphicsRootDescriptorTable(6, m_Impl->SRVArena.Gpu(source.SRV));
			}
			else
			{
				auto& source = m_Impl->HdrSurfaces[Impl::HdrScene];
				m_Impl->Transition(source.Resource.Get(), source.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
				m_Impl->Transition(sampleColor.Resource.Get(), sampleColor.State, D3D12_RESOURCE_STATE_RENDER_TARGET);
				const auto rtv = m_Impl->RTVArena.Cpu(sampleColor.RTV);
				commands->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
				commands->SetGraphicsRootDescriptorTable(7, m_Impl->SRVArena.Gpu(source.SRV));
			}
			const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(m_Impl->CreateInfo.Width), static_cast<float>(m_Impl->CreateInfo.Height), 0, 1};
			const D3D12_RECT scissor {0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height};
			commands->RSSetViewports(1, &viewport); commands->RSSetScissorRects(1, &scissor);
			commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			constexpr UINT FullscreenTriangleVertices = 3;
			commands->DrawInstanced(FullscreenTriangleVertices, 1, 0, 0);
			commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
		}
		if (pass == Pass::OpaqueResolve) m_Impl->SamplesOpaqueResolved = m_Impl->SamplesColorResolved = true;
		if (pass == Pass::VelocityResolve) m_Impl->SamplesVelocityResolved = true;
		if (pass == Pass::ColorRestore) m_Impl->SamplesColorRestored = true;
		if (pass == Pass::ColorResolve) m_Impl->SamplesColorResolved = true;
		++m_Impl->PendingFrame.Statistics.MultisamplePassCount;
		return true;
	}
	bool DX12RHI::Present()
	{
    if (!m_Impl->CheckThread("Lumen frame publication")) return false;
    Impl::LumenPresentPublication lumenPublication{*m_Impl};
    Impl::NanitePresentPublication nanitePublication{*m_Impl};
    Impl::RayShadowPresentPublication rayPublication{*m_Impl};
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
		if (!m_Impl->CheckThread("Present")) return false;
		if (!m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX12 frame cannot be presented in the current state");
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("DX12 Forward+ frame requires completed light assignment before presentation");
		if (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesColorResolved))
			return m_Impl->Fail("DX12 MSAA frame must resolve opaque inputs and final color before presentation");
		if (m_Impl->DeferredActive && !m_Impl->DeferredResolved)
			return m_Impl->Fail("DX12 deferred frame requires completed lighting before presentation");
		if (m_Impl->ReflectionActive && !m_Impl->ReflectionApplied)
			return m_Impl->Fail("DX12 SSR must complete before presentation");
		if (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionApplied)
			return m_Impl->Fail("DX12 Ambient-occlusion frame requires ambient occlusion before presentation");
		if (m_Impl->FramePostProcessing.DepthOfField.Enabled && !m_Impl->DepthOfFieldResolved)
			return m_Impl->Fail("DX12 depth-of-field frames must resolve depth of field before presentation");
		if (m_Impl->FramePostProcessing.MotionBlur.Enabled && !m_Impl->MotionBlurResolved)
			return m_Impl->Fail("DX12 motion-blur frames must resolve shutter reconstruction before presentation");
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing) && !m_Impl->TemporalResolved)
			return m_Impl->Fail("DX12 TAA must resolve before presentation");
		if (Techniques::UsesVolumetricFog(m_Impl->FrameFog) && !m_Impl->FogResolved)
			return m_Impl->Fail("DX12 fog must resolve before presentation");
		if (Techniques::UsesLightShafts(m_Impl->FrameLightShafts) && !m_Impl->LightShaftResolved)
			return m_Impl->Fail("DX12 light shafts must resolve before presentation");
        if (m_Impl->FramePeeling.Enabled && m_Impl->PeelingSequence.Stage()!=DepthPeelingStage::Complete)
            return m_Impl->Fail("Present requires completed OIT composition");
		if (m_Impl->RefractionActive && !m_Impl->RefractionResolved)
            return m_Impl->Fail("Present requires completed refraction composition");
        if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
			return m_Impl->Fail("DX12 weighted transparency must resolve before presentation");
        if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("DX12 contact visibility must complete before presentation");
		bool outputIsMain = true;
		if (!m_Impl->TSRScene.Active && (!m_Impl->ApplyHdr(m_Impl->FramePostProcessing.HDR) || !m_Impl->ApplyPostProcessing(m_Impl->FramePostProcessing, outputIsMain))) return false;
		m_Impl->PendingFrame.Statistics.PassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size())
			+ m_Impl->PendingFrame.Statistics.PostProcessPassCount + m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount
			+ m_Impl->PendingFrame.Statistics.HairFurPassCount + m_Impl->PendingFrame.Statistics.SubsurfacePassCount + m_Impl->PendingFrame.Statistics.PathTracingPassCount + m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount + m_Impl->PendingFrame.Statistics.ReflectionPassCount + m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount + m_Impl->PendingFrame.Statistics.MotionBlurPassCount
			+ m_Impl->PendingFrame.Statistics.MultisamplePassCount + m_Impl->PendingFrame.Statistics.EnvironmentMipPassCount
			+ m_Impl->PendingFrame.Statistics.EnvironmentConversionPassCount + m_Impl->PendingFrame.Statistics.PlanarMipPassCount
			+ m_Impl->PendingFrame.Statistics.TemporalAAPassCount + m_Impl->PendingFrame.Statistics.TemporalEffectsPassCount + m_Impl->PendingFrame.Statistics.VolumetricFogPassCount
			+ m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount + m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount + m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount
			+ m_Impl->PendingFrame.Statistics.LightShaftPassCount
			+ m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount
            + m_Impl->PendingFrame.Statistics.ContactShadowPassCount
            + m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount
            + m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount
            + m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount
			+ m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount + m_Impl->PendingFrame.Statistics.DepthPeelingPassCount;
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) ||
			(!IsGpuPresentationEnabled() && m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0);

		ID3D12Resource* finalColor = outputIsMain ? m_Impl->MainColor.Get() : m_Impl->PostColor.Get();
		auto& finalState = outputIsMain ? m_Impl->MainColorState : m_Impl->PostColorState;
        const bool gpuPresentation=IsGpuPresentationEnabled() && !captureDepth && !m_Impl->TSRScene.Active &&
            GetDiagnosticBuffer()==Techniques::RenderBuffer::Color;
        if (!m_Impl->TSRScene.Active)
        {
            m_Impl->Transition(finalColor,finalState,D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (gpuPresentation) { if (!m_Impl->PublishGpuColor(finalColor)) return false; }
            else m_Impl->CopyToReadback(finalColor,m_Impl->MainColorReadback);
        }
        Impl::ReadbackResource diagnosticReadback;
        const int diagnosticIndex=Techniques::MaterialBufferIndex(GetDiagnosticBuffer());
        auto* diagnostic=diagnosticIndex>=0?&m_Impl->GBufferSurfaces[diagnosticIndex]:GetDiagnosticBuffer()==Techniques::RenderBuffer::AmbientOcclusion?&m_Impl->AmbientOcclusionSurfaces[0]:nullptr;
        if (diagnostic && diagnostic->Resource)
        {
            if (!m_Impl->CreateReadback(diagnostic->Resource.Get(),diagnosticReadback)) return false;
            m_Impl->Transition(diagnostic->Resource.Get(),diagnostic->State,D3D12_RESOURCE_STATE_COPY_SOURCE);
            m_Impl->CopyToReadback(diagnostic->Resource.Get(),diagnosticReadback);
        }
		const bool captureMotion = (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && ((Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive);
		if (captureMotion)
		{
            auto& velocity = m_Impl->MotionSurfaces[Impl::MotionVelocity];
            if (!m_Impl->MotionReadback.Buffer && !m_Impl->CreateReadback(velocity.Resource.Get(),m_Impl->MotionReadback)) return false;
            m_Impl->Transition(velocity.Resource.Get(), velocity.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_Impl->CopyToReadback(velocity.Resource.Get(), m_Impl->MotionReadback);
		}
		if (captureDepth)
		{
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_Impl->CopyToReadback(m_Impl->MainDepth.Get(), m_Impl->MainDepthReadback);
		}
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
			for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
			{
				auto& map = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value);
				m_Impl->Transition(map.Resource.Get(), map.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
				m_Impl->CopyToReadback(map.Resource.Get(), map.Readback);
				m_Impl->Transition(map.Resource.Get(), map.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			}
		Impl::TextureResource* shadowResource = nullptr;
		Impl::VSMMomentMap* momentResource = nullptr;
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end())
			{
				shadowResource = &shadow->second;
				m_Impl->Transition(shadowResource->Resource.Get(), shadowResource->State,
					D3D12_RESOURCE_STATE_COPY_SOURCE);
				m_Impl->CopyToReadback(shadowResource->Resource.Get(), shadowResource->Readback);
				if (m_Impl->UsesVSM())
				{
					auto& map = m_Impl->VSMMoments.at(m_Impl->LastShadowTexture.Value);
					if (!map.Ready) return m_Impl->Fail("DX12 diagnostic capture requires filtered VSM moments");
					if (!map.Diagnostic.Buffer && !m_Impl->CreateReadback(map.Filtered.Resource.Get(), map.Diagnostic)) return false;
					m_Impl->Transition(map.Filtered.Resource.Get(), map.Filtered.State, D3D12_RESOURCE_STATE_COPY_SOURCE);
					m_Impl->CopyToReadback(map.Filtered.Resource.Get(), map.Diagnostic);
					m_Impl->Transition(map.Filtered.Resource.Get(), map.Filtered.State, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
					momentResource = &map;
				}
			}
		}
		m_Impl->Transition(m_Impl->MainColor.Get(), m_Impl->MainColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
		if (captureDepth)
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				D3D12_RESOURCE_STATE_DEPTH_WRITE);
		if (shadowResource)
			m_Impl->Transition(shadowResource->Resource.Get(), shadowResource->State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        m_Impl->EndFrameTiming();
		HRESULT result = m_Impl->CommandList->Close();
		if (FAILED(result))
		{
			m_Impl->FrameOpen = false;
			m_Impl->FailHRESULT("ID3D12GraphicsCommandList::Close", result);
			AbortFrame();
			return false;
		}
		ID3D12CommandList* lists[] = { m_Impl->CommandList.Get() };
		m_Impl->CommandQueue->ExecuteCommandLists(1, lists);
		// Buffer states decay at ExecuteCommandLists completion, including explicit
		// transitions. Keep the next frame's upload/abort journal consistent with it.
		// https://learn.microsoft.com/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
		for (auto& pair : m_Impl->Buffers) pair.second.State = D3D12_RESOURCE_STATE_COMMON;
		m_Impl->FrameWasSubmitted = true;
		m_Impl->SubmittedWorkPending = true;
		m_Impl->FrameSnapshotValid = false;
		if (!m_Impl->SignalAndWait())
		{
			m_Impl->FrameOpen = false;
			return false;
		}
        m_Impl->ReadFrameTiming();
		for (auto& pair : m_Impl->Textures)
		{
			auto& texture = pair.second;
			if (texture.FrameStartDirty && !texture.Dirty)
				std::vector<std::uint8_t>{}.swap(texture.Bytes);
		}
		if (!gpuPresentation && !m_Impl->TSRScene.Active && !m_Impl->ReadColor(m_Impl->MainColorReadback, m_Impl->CreateInfo.Width,
			m_Impl->CreateInfo.Height, m_Impl->PendingFrame.ColorRGBA8))
		{
			m_Impl->FrameOpen = false;
			return false;
		}
		if (captureDepth &&
			!m_Impl->ReadDepth(m_Impl->MainDepthReadback, m_Impl->CreateInfo.Width,
				m_Impl->CreateInfo.Height, m_Impl->PendingFrame.Depth32))
		{
			m_Impl->FrameOpen = false;
			return false;
		}
		if (momentResource && !m_Impl->ReadVSMMoments(*momentResource))
		{
			m_Impl->FrameOpen = false;
			return false;
		}
        if (diagnosticReadback.Buffer)
        {
            void* mapped=nullptr;const D3D12_RANGE range{0,SIZE_T(diagnosticReadback.TotalSize)};
            if (FAILED(diagnosticReadback.Buffer->Map(0,&range,&mapped))) return m_Impl->Fail("Cannot map diagnostic readback");
            DecodeDiagnosticRows(static_cast<std::uint8_t*>(mapped)+diagnosticReadback.Footprint.Offset,diagnosticReadback.Footprint.Footprint.RowPitch,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->PendingFrame.DiagnosticPixels);
            const D3D12_RANGE noWrite{0,0};diagnosticReadback.Buffer->Unmap(0,&noWrite);
        }
		if (captureMotion && !m_Impl->ReadMotionVectors())
		{
			m_Impl->FrameOpen = false;
			return false;
		}
		if (shadowResource)
		{
			m_Impl->PendingFrame.ShadowWidth = static_cast<std::uint32_t>(shadowResource->Description.Width);
			m_Impl->PendingFrame.ShadowHeight = static_cast<std::uint32_t>(shadowResource->Description.Height);
			if (!m_Impl->ReadDepth(shadowResource->Readback, shadowResource->Description.Width,
				shadowResource->Description.Height, m_Impl->PendingFrame.ShadowDepth32))
			{
				m_Impl->FrameOpen = false;
				return false;
			}
		}
		if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
		{
			m_Impl->PendingFrame.CascadeDepth32.resize(m_Impl->FrameCascades.Frame.CascadeCount);
			for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
			{
				auto& map = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value);
				if (!m_Impl->ReadDepth(map.Readback, m_Impl->FrameCascades.MapResolution,
					m_Impl->FrameCascades.MapResolution, m_Impl->PendingFrame.CascadeDepth32[index]))
				{
					m_Impl->FrameOpen = false;
					return false;
				}
			}
		}
		if (!m_Impl->FinishTSRScene()) return false;
    m_Impl->CommitTSRScene();
    m_Impl->Hair.CommitPresent(m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height);
    m_Impl->ExposureHistory.Commit();
    ApplyDiagnosticCapture(m_Impl->PendingFrame,GetDiagnosticBuffer());
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameUploads.clear();
		m_Impl->FrameOpen = false;
		if (Techniques::UsesTemporalAA(m_Impl->FramePostProcessing))
		{
			m_Impl->TemporalHistoryReadIndex = 1 - m_Impl->TemporalHistoryReadIndex;
			m_Impl->TemporalHistoryValid = true;
		}
		else m_Impl->TemporalHistoryValid = false;
        if (m_Impl->PendingVoxelVolume) m_Impl->UploadedVoxelVolume=std::move(m_Impl->PendingVoxelVolume);
        m_Impl->PathFrame.Commit();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI)) m_Impl->GI.Commit(); else m_Impl->GI.Invalidate();
        if (m_Impl->ProbeActive) m_Impl->Probes.Commit();
        if (m_Impl->PendingUploadedLightmap) m_Impl->UploadedLightmap=std::move(m_Impl->PendingUploadedLightmap);
		lumenPublication.Published=true; nanitePublication.Published=true; rayPublication.Published=true;
    return true;
	}

	FrameCapture DX12RHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
	FrameCapture DX12RHI::TakeLastFrameCapture() { return std::move(m_Impl->LastFrame); }
	std::string DX12RHI::GetLastError() const { return m_Impl->LastError; }
#else
	struct DX12RHI::Impl
	{
		std::string LastError = "DirectX 12 is only available on Windows";
	};

	DX12RHI::DX12RHI() : m_Impl(std::make_unique<Impl>()) {}
	DX12RHI::~DX12RHI() = default;
	bool DX12RHI::Initialize(const RHICreateInfo&) { return false; }
	void DX12RHI::Shutdown() {}
	bool DX12RHI::IsInitialized() const { return false; }
	RHIBackend DX12RHI::GetBackend() const { return RHIBackend::DX12; }
	const char* DX12RHI::GetName() const { return "DirectX 12 Unavailable"; }
	RHICapabilities DX12RHI::GetCapabilities() const { return {}; }
	bool DX12RHI::Resize(int, int) { return false; }
	BufferHandle DX12RHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
    bool DX12RHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
bool DX12RHI::PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
bool DX12RHI::SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
bool DX12RHI::PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
bool DX12RHI::CancelNaniteCache(NaniteCacheTask) { return false; }
    bool DX12RHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
bool DX12RHI::DestroyNaniteCache(NaniteCacheHandle) { return false; }
    bool DX12RHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
    bool DX12RHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
    bool DX12RHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
    bool DX12RHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
    bool DX12RHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
    bool DX12RHI::CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
    bool DX12RHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle) { return false; }
    bool DX12RHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
    bool DX12RHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
    bool DX12RHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
    bool DX12RHI::TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
    bool DX12RHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }
    bool DX12RHI::RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
    bool DX12RHI::BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
    bool DX12RHI::AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
    bool DX12RHI::CancelLumenLightingUpdate(LumenLightingTask) { return false; }
    bool DX12RHI::ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
    bool DX12RHI::DestroyLumenLighting(LumenLightingHandle) { return false; }
    bool DX12RHI::TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
    bool DX12RHI::ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
    bool DX12RHI::AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
    bool DX12RHI::ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
    bool DX12RHI::DestroyLumenRadiance(LumenRadianceHandle) { return false; }
    bool DX12RHI::IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
    bool DX12RHI::ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
    bool DX12RHI::ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
    bool DX12RHI::DestroyLumenProbes(LumenProbeHandle) { return false; }
    bool DX12RHI::LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
    bool DX12RHI::CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
    bool DX12RHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
    bool DX12RHI::AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
    bool DX12RHI::CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
    bool DX12RHI::ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
    bool DX12RHI::DestroyLumenSurface(LumenSurfaceHandle) { return false; }
    bool DX12RHI::DestroyTSRHistory(TSRHistoryHandle) { return false; }
    bool DX12RHI::RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
    bool DX12RHI::SnapshotTSROpaque() { return false; }
    bool DX12RHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool DX12RHI::CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
bool DX12RHI::SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
bool DX12RHI::PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
bool DX12RHI::CancelNaniteGeometry(NaniteGeometryTask) { return false; }
    bool DX12RHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool DX12RHI::PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool DX12RHI::PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool DX12RHI::DrawNaniteBins(NaniteDrawHandle) { return false; }
bool DX12RHI::DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
bool DX12RHI::CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
bool DX12RHI::BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
bool DX12RHI::RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
bool DX12RHI::ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
bool DX12RHI::ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
bool DX12RHI::DestroyNaniteDepth(NaniteDepthHandle) { return false; }
bool DX12RHI::DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }
bool DX12RHI::CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
    bool DX12RHI::CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
    bool DX12RHI::CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
	bool DX12RHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
	bool DX12RHI::DestroyBuffer(BufferHandle) { return false; }
	TextureHandle DX12RHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
	bool DX12RHI::DestroyTexture(TextureHandle) { return false; }
	bool DX12RHI::BeginFrame() { return false; }
	void DX12RHI::AbortFrame() {}
	bool DX12RHI::BeginPass(const PassDesc&) { return false; }
	bool DX12RHI::BindPipeline(const PipelineState&) { return false; }
	bool DX12RHI::BindWorldTransform(const Matrix4&) { return false; }
	bool DX12RHI::BindViewProjection(const Matrix4&) { return false; }
	bool DX12RHI::BindLightViewProjection(const Matrix4&) { return false; }
	bool DX12RHI::BindVertexBuffer(BufferHandle) { return false; }
	bool DX12RHI::BindIndexBuffer(BufferHandle) { return false; }
	bool DX12RHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
    bool DX12RHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
    bool DX12RHI::RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
    bool DX12RHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
	bool DX12RHI::DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>&) { return false; }
	bool DX12RHI::EndPass() { return false; }
	bool DX12RHI::GenerateContactShadows(const ContactShadowDesc&) { return false; }
    bool DX12RHI::PrepareLightListFrame(const LightListFrameDesc&) { return false; }
    bool DX12RHI::GenerateLightLists() { return false; }
	bool DX12RHI::DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
    bool DX12RHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
    bool DX12RHI::DrawHairFur(const HairFurDesc&) { return false; }
    bool DX12RHI::DrawHairFurInPass(const HairFurDesc&) { return false; }
    bool DX12RHI::DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
	bool DX12RHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
	bool DX12RHI::CompleteEnvironmentCapture() { return false; }
	bool DX12RHI::CompletePlanarCapture() { return false; }
	bool DX12RHI::ApplyPathTracing(const PathTracingDesc&) { return false; }
    bool DX12RHI::ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
    bool DX12RHI::ApplyReflections(const ReflectionDesc&) { return false; }
	bool DX12RHI::ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
	bool DX12RHI::ApplyMotionBlur(const MotionBlurDesc&) { return false; }
	bool DX12RHI::ApplyTemporalAA(const TemporalAADesc&) { return false; }
    bool DX12RHI::ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
	bool DX12RHI::ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
	bool DX12RHI::ApplyLightShafts(const LightShaftDesc&) { return false; }
	bool DX12RHI::ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
	bool DX12RHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
    bool DX12RHI::ProcessDepthPeeling(DepthPeelingPass) { return false; }
    bool DX12RHI::ProcessRefraction(const RefractionDesc&) { return false; }
	bool DX12RHI::Present() { return false; }
	FrameCapture DX12RHI::GetLastFrameCapture() const { return {}; }
	FrameCapture DX12RHI::TakeLastFrameCapture() { return {}; }
	std::string DX12RHI::GetLastError() const { return m_Impl->LastError; }
#endif
}
