#include "Renderer/DX12RHI.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
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
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace Evans::Renderer
{
#if defined(_WIN32)
	using Microsoft::WRL::ComPtr;

	namespace
	{
		constexpr std::size_t ConstantBufferCapacity = 4u * 1024u * 1024u;
		constexpr UINT DescriptorCapacity = 256;
		constexpr UINT InvalidDescriptor = std::numeric_limits<UINT>::max();
		constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;

		constexpr const char* ShaderSourceTemplate = R"(
struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

cbuffer DrawConstants : register(b{{DRAWCONSTANTS_BINDING}})
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 ViewProjection;
    row_major float4x4 LightViewProjection;
    float4 PipelineColor;
    float4 MaterialDiffuseColor;
    float4 MaterialAmbientColor;
    float4 MaterialSpecularColor;
    float4 MaterialEmissiveColor;
    float4 MaterialParameters;
    float4 AmbientColorStrength;
    float4 CameraLightCount;
    float4 RenderParameters;
    uint4 TextureFlags;
    LightConstants Lights[{{MAX_LIGHTS}}];
};

Texture2D<float> ShadowMap : register(t{{SHADOWMAP_BINDING}});
Texture2D<float4> DiffuseMap : register(t{{DIFFUSEMAP_BINDING}});
Texture2D<float4> NormalMap : register(t{{NORMALMAP_BINDING}});
Texture2D<float4> OpacityMap : register(t{{OPACITYMAP_BINDING}});
SamplerState SurfaceSampler : register(s{{SURFACESAMPLER_BINDING}});

float3 SRGBToLinear(float3 color)
{
    float3 low = color / 12.92f;
    float3 high = pow((color + 0.055f) / 1.055f, 2.4f);
    return float3(
        color.r <= 0.04045f ? low.r : high.r,
        color.g <= 0.04045f ? low.g : high.g,
        color.b <= 0.04045f ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0f);
    color /= max(1.0f, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f ? value * rsqrt(lengthSquared) : fallback;
}

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 TexCoord : TEXCOORD3;
};

VSOutput {{VERTEX_ENTRY}}(VSInput input)
{
    VSOutput output;
    float4 worldPosition = mul(World, float4(input.Position, 1.0f));
    float3 worldNormal = NormalizeSafe(
        mul((float3x3)NormalWorld, input.Normal),
        float3(0.0f, 1.0f, 0.0f));
    if (RenderParameters.y > 1.5f)
    {
        worldPosition.xyz += worldNormal * (MaterialParameters.w * 0.01f);
    }
    output.Position = mul(ViewProjection, worldPosition);
#if defined(EVANS_VULKAN)
    output.Position.y = -output.Position.y;
#endif
    output.WorldPosition = worldPosition.xyz / max(abs(worldPosition.w), 1.0e-7f);
    output.Normal = worldNormal;
    output.Color = float4(SRGBToLinear(saturate(input.Color.rgb)), input.Color.a);
    output.LightPosition = mul(LightViewProjection, worldPosition);
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    return output;
}

float4 SampleDiffuse(float2 texCoord)
{
    return TextureFlags.x != 0
        ? DiffuseMap.Sample(SurfaceSampler, texCoord)
        : float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float SampleOpacity(float2 texCoord)
{
    return TextureFlags.z != 0
        ? OpacityMap.Sample(SurfaceSampler, texCoord).r
        : 1.0f;
}

float SurfaceAlpha(VSOutput input, float4 diffuseSample)
{
    return saturate(
        input.Color.a *
        MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input.TexCoord));
}

void ApplyAlphaClip(float alpha)
{
    if (MaterialParameters.z > 0.0f)
    {
        clip(alpha - MaterialParameters.z);
    }
}

void ApplyShadowAlpha(float alpha, float2 pixelPosition)
{
    if (MaterialParameters.z > 0.0f)
    {
        clip(alpha - MaterialParameters.z);
        return;
    }
    if (RenderParameters.w > 0.5f)
    {
        if (alpha <= 0.0f) clip(-1.0f);
        if (alpha >= 1.0f) return;
        float threshold = frac(
            52.9829189f *
            frac(dot(floor(pixelPosition), float2(0.06711056f, 0.00583715f))));
        clip(alpha - threshold);
    }
}

float SampleShadow(
    float4 lightPosition,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (RenderParameters.x < 0.5f || abs(lightPosition.w) < 1.0e-7f)
    {
        return 1.0f;
    }
    float3 projected = lightPosition.xyz / lightPosition.w;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
    {
        return 1.0f;
    }
    uint width;
    uint height;
    ShadowMap.GetDimensions(width, height);
    if (width == 0 || height == 0) return 1.0f;

    float directionLengthSquared = dot(directionToLight, directionToLight);
    float3 direction = directionLengthSquared > 1.0e-12f
        ? directionToLight * rsqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(dot(surfaceNormal, direction));
    float sineToLight = sqrt(max(1.0f - normalToLight * normalToLight, 0.0f));
    float slope = sineToLight / max(normalToLight, 0.15f);
    float receiverBias = min(0.0045f, 0.00035f + 0.00060f * slope);
    float2 texelPosition = uv * float2(width, height) - 0.5f;
    float2 interpolation = frac(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0f;
    [unroll]
    for (int y = {{SHADOW_MIN_OFFSET}}; y <= {{SHADOW_MAX_OFFSET}}; ++y)
    {
        [unroll]
        for (int x = {{SHADOW_MIN_OFFSET}}; x <= {{SHADOW_MAX_OFFSET}}; ++x)
        {
            int2 pixel00 = clamp(basePixel + int2(x, y), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel10 = clamp(pixel00 + int2(1, 0), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel01 = clamp(pixel00 + int2(0, 1), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel11 = clamp(pixel00 + int2(1, 1), int2(0, 0), int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth > ShadowMap.Load(int3(pixel00, 0)) ? 1.0f : 0.0f;
            float comparison10 = comparisonDepth > ShadowMap.Load(int3(pixel10, 0)) ? 1.0f : 0.0f;
            float comparison01 = comparisonDepth > ShadowMap.Load(int3(pixel01, 0)) ? 1.0f : 0.0f;
            float comparison11 = comparisonDepth > ShadowMap.Load(int3(pixel11, 0)) ? 1.0f : 0.0f;
            float comparisonTop = lerp(comparison00, comparison10, interpolation.x);
            float comparisonBottom = lerp(comparison01, comparison11, interpolation.x);
            occlusion += lerp(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0f - {{SHADOW_STRENGTH}}f * occlusion / {{SHADOW_SAMPLE_COUNT}}.0f;
}

float3 ResolveSurfaceNormal(VSOutput input, bool isFrontFace)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0f, 1.0f, 0.0f));
    if (!isFrontFace) normal = -normal;
    if (TextureFlags.y == 0) return normal;

    float3 positionDx = ddx(input.WorldPosition);
    float3 positionDy = ddy(input.WorldPosition);
    float2 texCoordDx = ddx(input.TexCoord);
    float2 texCoordDy = ddy(input.TexCoord);
    float determinant = texCoordDx.x * texCoordDy.y - texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < 1.0e-10f) return normal;
    float inverseDeterminant = rcp(determinant);
    float3 tangent = (positionDx * texCoordDy.y - positionDy * texCoordDx.y) * inverseDeterminant;
    float3 sourceBitangent = (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) * inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    tangent = NormalizeSafe(tangent, NormalizeSafe(
        cross(helper, normal), float3(1.0f, 0.0f, 0.0f)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0f ? -1.0f : 1.0f;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness, float3(0.0f, 0.0f, 1.0f));
    float3 tangentNormal = NormalMap.Sample(SurfaceSampler, input.TexCoord).xyz * 2.0f - 1.0f;
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= 1.0e-12f) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0f);
    if (normalToLight <= 0.0f) return;
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0f),
        max(MaterialParameters.x, 1.0f));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateSceneLight(
    uint lightIndex,
    VSOutput input,
    float3 normal,
    float3 viewDirection,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    LightConstants light = Lights[lightIndex];
    uint type = (uint)(light.PositionType.w + 0.5f);
    float3 shadowDirection = type == 0
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5f;
    float visibility = isShadowLight
        ? SampleShadow(input.LightPosition, normal, shadowDirection)
        : 1.0f;
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0f) *
        max(light.DirectionIntensity.w, 0.0f);
    if (type == 0)
    {
        AccumulateLightSample(
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    if (type == 1)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
        AccumulateLightSample(
            toLight, radiance / distanceSquared, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }

    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(emissionDirection.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), float3(0.0f, 0.0f, 1.0f));
    [unroll]
    for (int sampleY = 0; sampleY < 2; ++sampleY)
    {
        [unroll]
        for (int sampleX = 0; sampleX < 2; ++sampleX)
        {
            float offsetX = sampleX == 0 ? -0.25f : 0.25f;
            float offsetY = sampleY == 0 ? -0.25f : 0.25f;
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine = max(dot(emissionDirection, -direction), 0.0f);
            AccumulateLightSample(
                direction,
                radiance * (0.25f * emitterCosine / distanceSquared),
                visibility, normal, viewDirection, diffuseLight, specularLight);
        }
    }
}

float4 {{FRAGMENT_ENTRY}}(
    VSOutput input,
    bool isFrontFace : SV_IsFrontFace) : SV_TARGET
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    float alpha = SurfaceAlpha(input, diffuseSample);
    ApplyAlphaClip(alpha);
    if (RenderParameters.y > 0.5f)
    {
        return float4(SRGBToLinear(saturate(PipelineColor.rgb)), alpha);
    }

    float3 surfaceBase =
        input.Color.rgb *
        max(PipelineColor.rgb, 0.0f) *
        max(diffuseSample.rgb, 0.0f);
    float3 normal = ResolveSurfaceNormal(input, isFrontFace);
    float3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - input.WorldPosition, normal);
    float3 diffuseLight = 0.0f;
    float3 specularLight = 0.0f;
    uint lightCount = min((uint)(CameraLightCount.w + 0.5f), 4u);
    [loop]
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(
            lightIndex, input, normal, viewDirection,
            diffuseLight, specularLight);
    }

    float3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, 0.0f) *
            max(AmbientColorStrength.rgb, 0.0f) *
            max(AmbientColorStrength.w, 0.0f) +
        surfaceBase * max(MaterialDiffuseColor.rgb, 0.0f) * diffuseLight +
        max(MaterialSpecularColor.rgb, 0.0f) * specularLight +
        max(MaterialEmissiveColor.rgb, 0.0f);
    return float4(PrepareDisplayLinear(color), alpha);
}

void {{SHADOW_ENTRY}}(VSOutput input)
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyShadowAlpha(SurfaceAlpha(input, diffuseSample), input.Position.xy);
}

float4 {{OUTLINE_ENTRY}}(VSOutput input) : SV_TARGET
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyAlphaClip(SurfaceAlpha(input, diffuseSample));
    return float4(SRGBToLinear(saturate(PipelineColor.rgb)), PipelineColor.a);
}
)";

		using D3D12CreateDeviceFunction = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
		using D3D12GetDebugInterfaceFunction = HRESULT(WINAPI*)(REFIID, void**);
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

		struct BufferResource
		{
			BufferDesc Description;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D12Resource> Resource;
			D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON;
			D3D12_VERTEX_BUFFER_VIEW VertexView = {};
			D3D12_INDEX_BUFFER_VIEW IndexView = {};
			bool Dirty = false;
		};

		struct TextureResource
		{
			TextureDesc Description;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D12Resource> Resource;
			D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON;
			ReadbackResource Readback;
			UINT RTV = InvalidDescriptor;
			UINT DSV = InvalidDescriptor;
			UINT SRV = InvalidDescriptor;
			bool Dirty = false;
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
				if (FAILED(removed)) error += "; device removed reason " + HResultMessage("", removed);
			}
			return Fail(error);
		}

		bool CheckThread(const char* operation)
		{
			if (!Initialized) return Fail("DX12 RHI is not initialized");
			if (OwnerThread != std::this_thread::get_id())
				return Fail(std::string(operation) + " must execute on the owning DX12 RHI thread");
			return true;
		}

		bool LoadRuntime()
		{
			D3D12Module = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
			if (!D3D12Module) return Fail("Could not load the system d3d12.dll");
			CreateDeviceFunction = reinterpret_cast<D3D12CreateDeviceFunction>(
				GetProcAddress(D3D12Module, "D3D12CreateDevice"));
			GetDebugInterfaceFunction = reinterpret_cast<D3D12GetDebugInterfaceFunction>(
				GetProcAddress(D3D12Module, "D3D12GetDebugInterface"));
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

			MainColor = std::move(color);
			MainDepth = std::move(depth);
			MainColorReadback = std::move(colorReadback);
			MainDepthReadback = std::move(depthReadback);
			MainColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
			MainDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
			return true;
		}

		bool CompileShader(const char* entryPoint, const char* target, ComPtr<ID3DBlob>& bytecode)
		{
			UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
			flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
			ComPtr<ID3DBlob> errors;
			const HRESULT result = D3DCompile(ShaderCode.data(), ShaderCode.size(), "EvansDX12.hlsl",
				nullptr, nullptr, entryPoint, target, flags, 0,
				bytecode.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(result))
			{
				if (errors) LastError.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
				else LastError = HResultMessage("D3DCompile", result);
				return false;
			}
			return true;
		}

		bool CreatePipelineObjects()
		{
			if (!PrepareBuiltinRealtimeShader(
					RealtimeShaderLanguage::HLSL,
					ShaderSourceTemplate,
					{
						RealtimeShaderStage::Vertex,
						RealtimeShaderStage::Fragment,
						RealtimeShaderStage::Shadow,
						RealtimeShaderStage::Outline },
					sizeof(RealtimeDrawConstants),
					MaximumSceneLights,
					ShaderCode,
					LastError))
			{
				return false;
			}
			const RealtimeShaderProgram& shader =
				BuiltinRealtimeShader().Program;
			if (!CompileShader(
					shader.FindStage(RealtimeShaderStage::Vertex)->HLSL.c_str(),
					"vs_5_0",
					VertexBytecode)
				|| !CompileShader(
					shader.FindStage(RealtimeShaderStage::Fragment)->HLSL.c_str(),
					"ps_5_0",
					PixelBytecode)
				|| !CompileShader(
					shader.FindStage(RealtimeShaderStage::Shadow)->HLSL.c_str(),
					"ps_5_0",
					ShadowPixelBytecode)
				|| !CompileShader(
					shader.FindStage(RealtimeShaderStage::Outline)->HLSL.c_str(),
					"ps_5_0",
					OutlinePixelBytecode)) return false;

			std::array<D3D12_DESCRIPTOR_RANGE, 4> textureRanges {};
			std::array<D3D12_ROOT_PARAMETER, 5> parameters {};
			parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			parameters[0].Descriptor.ShaderRegister = 0;
			parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
			for (UINT textureIndex = 0; textureIndex < textureRanges.size(); ++textureIndex)
			{
				D3D12_DESCRIPTOR_RANGE& range = textureRanges[textureIndex];
				range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
				range.NumDescriptors = 1;
				range.BaseShaderRegister = textureIndex;
				range.OffsetInDescriptorsFromTableStart =
					D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
				D3D12_ROOT_PARAMETER& parameter = parameters[textureIndex + 1];
				parameter.ParameterType =
					D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
				parameter.DescriptorTable.NumDescriptorRanges = 1;
				parameter.DescriptorTable.pDescriptorRanges = &range;
				parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			}
			D3D12_STATIC_SAMPLER_DESC sampler {};
			sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
			sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
			sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
			sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
			sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			sampler.MaxLOD = D3D12_FLOAT32_MAX;
			sampler.ShaderRegister = 0;
			sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
			D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
			rootDesc.NumParameters = static_cast<UINT>(parameters.size());
			rootDesc.pParameters = parameters.data();
			rootDesc.NumStaticSamplers = 1;
			rootDesc.pStaticSamplers = &sampler;
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
			result = Device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
				IID_PPV_ARGS(RootSignature.GetAddressOf()));
			if (FAILED(result)) return FailHRESULT("ID3D12Device::CreateRootSignature", result);

			if (!CreateBufferResource(ConstantBufferCapacity, D3D12_HEAP_TYPE_UPLOAD,
				D3D12_RESOURCE_STATE_GENERIC_READ, ConstantBuffer)) return false;
			D3D12_RANGE noRead = { 0, 0 };
			result = ConstantBuffer->Map(0, &noRead, reinterpret_cast<void**>(&ConstantData));
			if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(constants)", result);
			return true;
		}

		std::uint32_t PipelineKey(const PipelineState& pipeline, PassType pass) const
		{
			return static_cast<std::uint32_t>(pass)
				| (static_cast<std::uint32_t>(pipeline.Fill) << 2)
				| (pipeline.DepthTest ? 1u << 4 : 0u)
				| (pipeline.DepthWrite ? 1u << 5 : 0u)
				| (pipeline.ColorWrite ? 1u << 6 : 0u)
				| (pipeline.OutlineExtrusion ? 1u << 7 : 0u)
				| (pipeline.BlendEnabled ? 1u << 8 : 0u);
		}

		ID3D12PipelineState* GetPipeline(const PipelineState& pipeline, PassType pass)
		{
			const std::uint32_t key = PipelineKey(pipeline, pass);
			auto existing = Pipelines.find(key);
			if (existing != Pipelines.end()) return existing->second.Get();

			const D3D12_INPUT_ELEMENT_DESC inputElements[] = {
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
			};
			D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
			desc.pRootSignature = RootSignature.Get();
			desc.VS = { VertexBytecode->GetBufferPointer(), VertexBytecode->GetBufferSize() };
			ID3DBlob* pixelShader = pass == PassType::Shadow
				? ShadowPixelBytecode.Get()
				: (pass == PassType::Outline
					? OutlinePixelBytecode.Get()
					: PixelBytecode.Get());
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
			blend.RenderTargetWriteMask = pipeline.ColorWrite ? D3D12_COLOR_WRITE_ENABLE_ALL : 0;
			desc.SampleMask = UINT_MAX;
			desc.RasterizerState.FillMode = pipeline.Fill == FillMode::Wireframe
				? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
			desc.RasterizerState.CullMode = pipeline.OutlineExtrusion ? D3D12_CULL_MODE_FRONT : D3D12_CULL_MODE_NONE;
			desc.RasterizerState.FrontCounterClockwise = TRUE;
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
			desc.DepthStencilState.DepthFunc = pipeline.DepthWrite
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
			desc.NumRenderTargets = pass == PassType::Shadow ? 0u : 1u;
			desc.RTVFormats[0] = pass == PassType::Shadow
				? DXGI_FORMAT_UNKNOWN
				: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
			desc.SampleDesc.Count = 1;
			ComPtr<ID3D12PipelineState> nativePipeline;
			const HRESULT result = Device->CreateGraphicsPipelineState(&desc,
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
			UINT factoryFlags = 0;
			if (CreateInfo.EnableValidation && GetDebugInterfaceFunction)
			{
				ComPtr<ID3D12Debug> debug;
				if (SUCCEEDED(GetDebugInterfaceFunction(IID_PPV_ARGS(debug.GetAddressOf()))))
				{
					debug->EnableDebugLayer();
					factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
				}
			}

			HRESULT result = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(Factory.GetAddressOf()));
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
				|| !CreateDescriptorArena(SRVArena, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DescriptorCapacity, true))
				return false;
			MainColorRTV = RTVArena.Allocate();
			MainDepthDSV = DSVArena.Allocate();
			NullShadowSRV = SRVArena.Allocate();
			NullColorSRV = SRVArena.Allocate();
			if (MainColorRTV == InvalidDescriptor || MainDepthDSV == InvalidDescriptor
				|| NullShadowSRV == InvalidDescriptor
				|| NullColorSRV == InvalidDescriptor)
				return Fail("DX12 descriptor heaps could not reserve main target descriptors");
			D3D12_SHADER_RESOURCE_VIEW_DESC nullView = {};
			nullView.Format = DXGI_FORMAT_R32_FLOAT;
			nullView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			nullView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			nullView.Texture2D.MipLevels = 1;
			Device->CreateShaderResourceView(nullptr, &nullView, SRVArena.Cpu(NullShadowSRV));
			nullView.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			Device->CreateShaderResourceView(nullptr, &nullView, SRVArena.Cpu(NullColorSRV));

			return CreatePipelineObjects()
				&& CreateMainTargets(CreateInfo.Width, CreateInfo.Height);
		}

		bool SignalAndWait()
		{
			const UINT64 value = ++FenceValue;
			HRESULT result = CommandQueue->Signal(Fence.Get(), value);
			if (FAILED(result)) return FailHRESULT("ID3D12CommandQueue::Signal", result);
			if (Fence->GetCompletedValue() < value)
			{
				result = Fence->SetEventOnCompletion(value, FenceEvent);
				if (FAILED(result)) return FailHRESULT("ID3D12Fence::SetEventOnCompletion", result);
				const DWORD waitResult =
					WaitForSingleObject(FenceEvent, GpuWaitTimeoutMilliseconds);
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
				ComPtr<ID3D12Resource> upload;
				if (!CreateBufferResource(static_cast<std::size_t>(resource.Readback.TotalSize),
					D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload)) return false;
				void* mapped = nullptr;
				D3D12_RANGE noRead = { 0, 0 };
				HRESULT result = upload->Map(0, &noRead, &mapped);
				if (FAILED(result)) return FailHRESULT("ID3D12Resource::Map(texture upload)", result);
				const std::size_t rowSize = static_cast<std::size_t>(resource.Description.Width) * 4;
				for (int row = 0; row < resource.Description.Height; ++row)
				{
					std::memcpy(static_cast<std::uint8_t*>(mapped) + resource.Readback.Footprint.Offset
						+ static_cast<std::size_t>(row) * resource.Readback.Footprint.Footprint.RowPitch,
						resource.Bytes.data() + static_cast<std::size_t>(row) * rowSize, rowSize);
				}
				upload->Unmap(0, nullptr);
				Transition(resource.Resource.Get(), resource.State, D3D12_RESOURCE_STATE_COPY_DEST);
				D3D12_TEXTURE_COPY_LOCATION destination = {};
				destination.pResource = resource.Resource.Get();
				destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				D3D12_TEXTURE_COPY_LOCATION source = {};
				source.pResource = upload.Get();
				source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
				source.PlacedFootprint = resource.Readback.Footprint;
				CommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
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

		void ReleaseAll()
		{
			if (ConstantBuffer && ConstantData) ConstantBuffer->Unmap(0, nullptr);
			ConstantData = nullptr;
			Buffers.clear();
			Textures.clear();
			Pipelines.clear();
			FrameUploads.clear();
			LastFrame = {};
			PendingFrame = {};
			MainColorReadback = {};
			MainDepthReadback = {};
			MainColor.Reset();
			MainDepth.Reset();
			ConstantBuffer.Reset();
			RootSignature.Reset();
			VertexBytecode.Reset();
			PixelBytecode.Reset();
			ShadowPixelBytecode.Reset();
			OutlinePixelBytecode.Reset();
			RTVArena = {};
			DSVArena = {};
			SRVArena = {};
			Fence.Reset();
			CommandList.Reset();
			CommandAllocator.Reset();
			CommandQueue.Reset();
			Device.Reset();
			Adapter.Reset();
			Factory.Reset();
			if (FenceEvent)
			{
				CloseHandle(FenceEvent);
				FenceEvent = nullptr;
			}
			CreateDeviceFunction = nullptr;
			GetDebugInterfaceFunction = nullptr;
			SerializeRootSignatureFunction = nullptr;
			if (D3D12Module)
			{
				FreeLibrary(D3D12Module);
				D3D12Module = nullptr;
			}
			Initialized = false;
			FrameOpen = false;
			PassOpen = false;
		}

		HMODULE D3D12Module = nullptr;
		D3D12CreateDeviceFunction CreateDeviceFunction = nullptr;
		D3D12GetDebugInterfaceFunction GetDebugInterfaceFunction = nullptr;
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
		UINT NullShadowSRV = InvalidDescriptor;
		UINT NullColorSRV = InvalidDescriptor;
		ComPtr<ID3D12Resource> MainColor;
		ComPtr<ID3D12Resource> MainDepth;
		D3D12_RESOURCE_STATES MainColorState = D3D12_RESOURCE_STATE_RENDER_TARGET;
		D3D12_RESOURCE_STATES MainDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		ReadbackResource MainColorReadback;
		ReadbackResource MainDepthReadback;
		ComPtr<ID3D12RootSignature> RootSignature;
		ComPtr<ID3DBlob> VertexBytecode;
		ComPtr<ID3DBlob> PixelBytecode;
		ComPtr<ID3DBlob> ShadowPixelBytecode;
		ComPtr<ID3DBlob> OutlinePixelBytecode;
		std::unordered_map<std::uint32_t, ComPtr<ID3D12PipelineState>> Pipelines;
		ComPtr<ID3D12Resource> ConstantBuffer;
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
		m_Impl->LastError.clear();
		return true;
	}

	void DX12RHI::Shutdown()
	{
		if (!m_Impl) return;
		if (m_Impl->Initialized && m_Impl->OwnerThread == std::this_thread::get_id())
			m_Impl->WaitForGPU();
		m_Impl->ReleaseAll();
	}

	bool DX12RHI::IsInitialized() const { return m_Impl->Initialized; }
	RHIBackend DX12RHI::GetBackend() const { return RHIBackend::DX12; }
	const char* DX12RHI::GetName() const { return "DirectX 12 Native RHI"; }

	RHICapabilities DX12RHI::GetCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = true;
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
		return capabilities;
	}

	bool DX12RHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0)
			return m_Impl->Fail("DX12 resize is invalid while a frame is open or dimensions are non-positive");
		if (width == m_Impl->CreateInfo.Width && height == m_Impl->CreateInfo.Height) return true;
		if (!m_Impl->WaitForGPU() || !m_Impl->CreateMainTargets(width, height)) return false;
		m_Impl->CreateInfo.Width = width;
		m_Impl->CreateInfo.Height = height;
		return true;
	}

	BufferHandle DX12RHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateBuffer") || m_Impl->FrameOpen || desc.SizeInBytes == 0)
		{
			m_Impl->Fail("DX12 buffer creation is invalid while a frame is open or size is zero");
			return {};
		}
		Impl::BufferResource resource;
		resource.Description = desc;
		resource.Bytes.resize(desc.SizeInBytes);
		if (initialData) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
		resource.State = initialData ? D3D12_RESOURCE_STATE_COPY_DEST : m_Impl->FinalBufferState(desc.Usage);
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

	bool DX12RHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 buffers can only be updated between frames");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || !data || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("DX12 buffer update is invalid");
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		iterator->second.Dirty = true;
		return true;
	}

	bool DX12RHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 buffers can only be destroyed between frames");
		if (!buffer || m_Impl->Buffers.erase(buffer.Value) == 0)
			return m_Impl->Fail("DX12 buffer handle is invalid");
		return true;
	}

	TextureHandle DX12RHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture") || m_Impl->FrameOpen
			|| desc.Width <= 0 || desc.Height <= 0)
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
		nativeDesc.MipLevels = 1;
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
			srv.Texture2D.MipLevels = 1;
			m_Impl->Device->CreateShaderResourceView(resource.Resource.Get(), &srv,
				m_Impl->SRVArena.Cpu(resource.SRV));
		}
		if (initialData)
		{
			resource.Bytes.resize(static_cast<std::size_t>(desc.Width) * desc.Height * 4);
			std::memcpy(resource.Bytes.data(), initialData, resource.Bytes.size());
			resource.Dirty = true;
		}
		const TextureHandle handle { m_Impl->NextTexture++ };
		m_Impl->Textures.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX12RHI::DestroyTexture(TextureHandle texture)
	{
		if (!m_Impl->CheckThread("DestroyTexture") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 textures can only be destroyed between frames");
		auto iterator = m_Impl->Textures.find(texture.Value);
		if (!texture || iterator == m_Impl->Textures.end())
			return m_Impl->Fail("DX12 texture handle is invalid");
		m_Impl->RTVArena.Release(iterator->second.RTV);
		m_Impl->DSVArena.Release(iterator->second.DSV);
		m_Impl->SRVArena.Release(iterator->second.SRV);
		m_Impl->Textures.erase(iterator);
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
		return true;
	}

	bool DX12RHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX12 frame cannot begin in the current state");
		m_Impl->FrameUploads.clear();
		HRESULT result = m_Impl->CommandAllocator->Reset();
		if (FAILED(result)) return m_Impl->FailHRESULT("ID3D12CommandAllocator::Reset", result);
		result = m_Impl->CommandList->Reset(m_Impl->CommandAllocator.Get(), nullptr);
		if (FAILED(result)) return m_Impl->FailHRESULT("ID3D12GraphicsCommandList::Reset", result);
		m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Backend = RHIBackend::DX12;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = true;
		m_Impl->ConstantOffset = 0;
		m_Impl->LastError.clear();
		ID3D12DescriptorHeap* heaps[] = { m_Impl->SRVArena.Heap.Get() };
		m_Impl->CommandList->SetDescriptorHeaps(1, heaps);
		if (!m_Impl->UploadDirtyResources())
		{
			m_Impl->CommandList->Close();
			m_Impl->FrameOpen = false;
			return false;
		}
		return true;
	}

	void DX12RHI::AbortFrame()
	{
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
		if (m_Impl->FrameOpen)
		{
			(void)m_Impl->CommandList->Close();
		}
		m_Impl->PendingFrame = {};
		m_Impl->CurrentPass = {};
		m_Impl->CurrentPassTarget = {};
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameUploads.clear();
		m_Impl->ConstantOffset = 0;
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
	}

	bool DX12RHI::BeginPass(const PassDesc& desc)
	{
		if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX12 pass cannot begin in the current state");
		D3D12_CPU_DESCRIPTOR_HANDLE rtv = {};
		D3D12_CPU_DESCRIPTOR_HANDLE dsv = m_Impl->DSVArena.Cpu(m_Impl->MainDepthDSV);
		bool hasRTV = true;
		int width = m_Impl->CreateInfo.Width;
		int height = m_Impl->CreateInfo.Height;
		m_Impl->CurrentPassTarget = desc.Target;
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
		}
		else
		{
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				D3D12_RESOURCE_STATE_DEPTH_WRITE);
			if (desc.Target)
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
			else
			{
				m_Impl->Transition(m_Impl->MainColor.Get(), m_Impl->MainColorState,
					D3D12_RESOURCE_STATE_RENDER_TARGET);
				rtv = m_Impl->RTVArena.Cpu(m_Impl->MainColorRTV);
			}
		}
		if (hasRTV) m_Impl->CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
		else m_Impl->CommandList->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
		if (hasRTV && desc.ClearColorEnabled)
		{
			const float color[] = { desc.ClearColor.X, desc.ClearColor.Y, desc.ClearColor.Z, desc.ClearColor.W };
			m_Impl->CommandList->ClearRenderTargetView(rtv, color, 0, nullptr);
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
		if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 pipeline must be bound inside a pass");
		ID3D12PipelineState* native = m_Impl->GetPipeline(pipeline, m_Impl->CurrentPass.Type);
		if (!native) return false;
		m_Impl->CommandList->SetPipelineState(native);
		m_Impl->CommandList->SetGraphicsRootSignature(m_Impl->RootSignature.Get());
		m_Impl->BoundPipeline = pipeline;
		m_Impl->PipelineBound = true;
		return true;
	}

	bool DX12RHI::BindWorldTransform(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 world transform must be bound inside a pass");
		m_Impl->BoundWorld = value;
		m_Impl->WorldBound = true;
		return true;
	}

	bool DX12RHI::BindViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 view projection must be bound inside a pass");
		m_Impl->BoundViewProjection = value;
		m_Impl->ViewProjectionBound = true;
		return true;
	}

	bool DX12RHI::BindLightViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 light view projection must be bound inside a pass");
		m_Impl->BoundLightViewProjection = value;
		m_Impl->LightViewProjectionBound = true;
		return true;
	}

	bool DX12RHI::BindVertexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 vertex buffer must be bound inside a pass");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Vertex)
			return m_Impl->Fail("DX12 vertex buffer is invalid");
		m_Impl->BoundVertexBuffer = buffer;
		return true;
	}

	bool DX12RHI::BindIndexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 index buffer must be bound inside a pass");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Index)
			return m_Impl->Fail("DX12 index buffer is invalid");
		m_Impl->BoundIndexBuffer = buffer;
		return true;
	}

	bool DX12RHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
		if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("DX12 DrawIndexed is missing required state");
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("DX12 DrawIndexed buffers are invalid");
		const std::size_t availableIndices = index->second.Bytes.size() / sizeof(std::uint32_t);
		if (firstIndex > availableIndices || indexCount > availableIndices - firstIndex
			|| vertex->second.Bytes.size() % sizeof(Vertex) != 0
			|| index->second.Bytes.size() % sizeof(std::uint32_t) != 0 || indexCount % 3 != 0)
			return m_Impl->Fail("DX12 DrawIndexed buffer ranges are invalid");
		const std::size_t availableVertices = vertex->second.Bytes.size() / sizeof(Vertex);
		const auto* indices = reinterpret_cast<const std::uint32_t*>(index->second.Bytes.data());
		for (std::size_t offset = firstIndex;
			offset < static_cast<std::size_t>(firstIndex) + indexCount; ++offset)
		{
			const std::int64_t resolved = static_cast<std::int64_t>(indices[offset]) + baseVertex;
			if (resolved < 0 || static_cast<std::uint64_t>(resolved) >= availableVertices)
				return m_Impl->Fail("DX12 DrawIndexed references a vertex outside the vertex buffer");
		}

		m_Impl->ConstantOffset = AlignUp(m_Impl->ConstantOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
		if (m_Impl->ConstantOffset > ConstantBufferCapacity
			|| sizeof(RealtimeDrawConstants) > ConstantBufferCapacity - m_Impl->ConstantOffset)
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

		const RealtimeDrawConstants constants = BuildRealtimeDrawConstants(
			m_Impl->BoundPipeline,
			m_Impl->BoundWorld,
			m_Impl->BoundViewProjection,
			m_Impl->BoundLightViewProjection,
			m_Impl->CurrentPass.Type,
			shadowAvailable,
			m_Impl->BoundPipeline.DiffuseTexture != TextureHandle {},
			m_Impl->BoundPipeline.NormalTexture != TextureHandle {},
			m_Impl->BoundPipeline.OpacityTexture != TextureHandle {});
		std::memcpy(
			m_Impl->ConstantData + m_Impl->ConstantOffset,
			&constants,
			sizeof(constants));
		m_Impl->CommandList->SetGraphicsRootConstantBufferView(0,
			m_Impl->ConstantBuffer->GetGPUVirtualAddress() + m_Impl->ConstantOffset);
		m_Impl->CommandList->SetGraphicsRootDescriptorTable(1, m_Impl->SRVArena.Gpu(shadowDescriptor));
		for (UINT textureIndex = 0; textureIndex < surfaceDescriptors.size(); ++textureIndex)
		{
			m_Impl->CommandList->SetGraphicsRootDescriptorTable(
				textureIndex + 2,
				m_Impl->SRVArena.Gpu(surfaceDescriptors[textureIndex]));
		}
		m_Impl->CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->CommandList->IASetVertexBuffers(0, 1, &vertex->second.VertexView);
		m_Impl->CommandList->IASetIndexBuffer(&index->second.IndexView);
		m_Impl->CommandList->DrawIndexedInstanced(indexCount, 1, firstIndex, baseVertex, 0);
		m_Impl->ConstantOffset += sizeof(RealtimeDrawConstants);

		DrawCallRecord record;
		record.VertexBuffer = m_Impl->BoundVertexBuffer;
		record.IndexBuffer = m_Impl->BoundIndexBuffer;
		record.Pipeline = m_Impl->BoundPipeline;
		record.WorldTransform = m_Impl->BoundWorld;
		record.FirstIndex = firstIndex;
		record.IndexCount = indexCount;
		record.BaseVertex = baseVertex;
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
		FrameStatistics& stats = m_Impl->PendingFrame.Statistics;
		++stats.DrawCallCount;
		stats.SubmittedTriangleCount += indexCount / 3;
		stats.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1u : 0u;
		stats.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1u : 0u;
		return true;
	}

	bool DX12RHI::EndPass()
	{
		if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX12 has no open pass");
		if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->CurrentPassTarget)
		{
			auto target = m_Impl->Textures.find(m_Impl->CurrentPassTarget.Value);
			if (target == m_Impl->Textures.end())
				return m_Impl->Fail("DX12 shadow target disappeared during the pass");
			m_Impl->Transition(target->second.Resource.Get(), target->second.State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}
		m_Impl->PassOpen = false;
		return true;
	}

	bool DX12RHI::Present()
	{
		if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX12 frame cannot be presented in the current state");
		m_Impl->PendingFrame.Statistics.PassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size());
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || m_Impl->CreateInfo.CaptureDiagnosticBuffers ||
			m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0;

		m_Impl->Transition(m_Impl->MainColor.Get(), m_Impl->MainColorState, D3D12_RESOURCE_STATE_COPY_SOURCE);
		m_Impl->CopyToReadback(m_Impl->MainColor.Get(), m_Impl->MainColorReadback);
		if (captureDepth)
		{
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_Impl->CopyToReadback(m_Impl->MainDepth.Get(), m_Impl->MainDepthReadback);
		}
		Impl::TextureResource* shadowResource = nullptr;
		if (m_Impl->CreateInfo.CaptureDiagnosticBuffers && m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end())
			{
				shadowResource = &shadow->second;
				m_Impl->Transition(shadowResource->Resource.Get(), shadowResource->State,
					D3D12_RESOURCE_STATE_COPY_SOURCE);
				m_Impl->CopyToReadback(shadowResource->Resource.Get(), shadowResource->Readback);
			}
		}
		m_Impl->Transition(m_Impl->MainColor.Get(), m_Impl->MainColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
		if (captureDepth)
			m_Impl->Transition(m_Impl->MainDepth.Get(), m_Impl->MainDepthState,
				D3D12_RESOURCE_STATE_DEPTH_WRITE);
		if (shadowResource)
			m_Impl->Transition(shadowResource->Resource.Get(), shadowResource->State,
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

		HRESULT result = m_Impl->CommandList->Close();
		if (FAILED(result))
		{
			m_Impl->FrameOpen = false;
			return m_Impl->FailHRESULT("ID3D12GraphicsCommandList::Close", result);
		}
		ID3D12CommandList* lists[] = { m_Impl->CommandList.Get() };
		m_Impl->CommandQueue->ExecuteCommandLists(1, lists);
		if (!m_Impl->SignalAndWait())
		{
			m_Impl->FrameOpen = false;
			return false;
		}
		if (!m_Impl->ReadColor(m_Impl->MainColorReadback, m_Impl->CreateInfo.Width,
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
		m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameUploads.clear();
		m_Impl->FrameOpen = false;
		return true;
	}

	FrameCapture DX12RHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
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
	bool DX12RHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
	bool DX12RHI::EndPass() { return false; }
	bool DX12RHI::Present() { return false; }
	FrameCapture DX12RHI::GetLastFrameCapture() const { return {}; }
	std::string DX12RHI::GetLastError() const { return m_Impl->LastError; }
#endif
}
