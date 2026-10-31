#include "Renderer/MetalRHI.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"

#include <memory>

#if defined(__APPLE__)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Evans::Renderer
{
	namespace
	{
		using ObjCObject = void*;
		using ObjCClass = void*;
		using ObjCSelector = void*;
		using NSUInteger = std::size_t;
		using NSInteger = std::intptr_t;
		using ObjCBool = signed char;

		constexpr ObjCBool ObjCYes = 1;
		constexpr ObjCBool ObjCNo = 0;
		constexpr NSUInteger PixelFormatRGBA8Unorm = 70;
		constexpr NSUInteger PixelFormatRGBA8UnormSRGB = 71;
		constexpr NSUInteger PixelFormatDepth32Float = 252;
		constexpr NSUInteger TextureUsageShaderRead = 1;
		constexpr NSUInteger TextureUsageRenderTarget = 4;
		constexpr NSUInteger StorageModePrivate = 2;
		constexpr NSUInteger LoadActionLoad = 1;
		constexpr NSUInteger LoadActionClear = 2;
		constexpr NSUInteger StoreActionStore = 1;
		constexpr NSUInteger PrimitiveTypeTriangle = 3;
		constexpr NSUInteger IndexTypeUInt32 = 1;
		constexpr NSUInteger TriangleFillModeFill = 0;
		constexpr NSUInteger TriangleFillModeLines = 1;
		constexpr NSUInteger CullModeNone = 0;
		constexpr NSUInteger CullModeFront = 1;
		constexpr NSUInteger WindingCounterClockwise = 1;
		constexpr NSUInteger CompareFunctionLess = 1;
		constexpr NSUInteger CompareFunctionLessEqual = 3;
		constexpr NSUInteger CompareFunctionAlways = 7;
		constexpr NSUInteger ColorWriteMaskNone = 0;
		constexpr NSUInteger ColorWriteMaskAll = 0x0f;
		constexpr NSUInteger BlendFactorOne = 1;
		constexpr NSUInteger BlendFactorSourceAlpha = 4;
		constexpr NSUInteger BlendFactorOneMinusSourceAlpha = 5;
		constexpr NSUInteger BlendOperationAdd = 0;
		constexpr NSUInteger SamplerFilterLinear = 1;
		constexpr NSUInteger SamplerAddressModeRepeat = 2;
		constexpr NSUInteger CommandBufferStatusError = 5;

		struct MTLClearColor
		{
			double Red;
			double Green;
			double Blue;
			double Alpha;
		};

		struct MTLViewport
		{
			double OriginX;
			double OriginY;
			double Width;
			double Height;
			double NearDepth;
			double FarDepth;
		};

		struct MTLOrigin
		{
			NSUInteger X;
			NSUInteger Y;
			NSUInteger Z;
		};

		struct MTLSize
		{
			NSUInteger Width;
			NSUInteger Height;
			NSUInteger Depth;
		};

		struct MetalRuntime
		{
			using GetClassFunction = ObjCClass (*)(const char*);
			using RegisterSelectorFunction = ObjCSelector (*)(const char*);
			using CreateDeviceFunction = ObjCObject (*)();

			bool Load(std::string& error)
			{
				if (MessageSend != nullptr) return true;
				ObjCRuntime = dlopen("/usr/lib/libobjc.A.dylib", RTLD_LAZY | RTLD_LOCAL);
				Foundation = dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RTLD_LAZY | RTLD_LOCAL);
				Metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY | RTLD_LOCAL);
				if (ObjCRuntime == nullptr || Foundation == nullptr || Metal == nullptr)
				{
					const char* dynamicError = dlerror();
					error = dynamicError ? dynamicError : "Metal framework loading failed";
					return false;
				}
				GetClass = reinterpret_cast<GetClassFunction>(dlsym(ObjCRuntime, "objc_getClass"));
				RegisterSelector = reinterpret_cast<RegisterSelectorFunction>(dlsym(ObjCRuntime, "sel_registerName"));
				MessageSend = dlsym(ObjCRuntime, "objc_msgSend");
				CreateSystemDefaultDevice = reinterpret_cast<CreateDeviceFunction>(
					dlsym(Metal, "MTLCreateSystemDefaultDevice"));
				if (GetClass == nullptr || RegisterSelector == nullptr || MessageSend == nullptr
					|| CreateSystemDefaultDevice == nullptr)
				{
					error = "Metal or Objective-C runtime entry points are unavailable";
					return false;
				}
				return true;
			}

			void* ObjCRuntime = nullptr;
			void* Foundation = nullptr;
			void* Metal = nullptr;
			GetClassFunction GetClass = nullptr;
			RegisterSelectorFunction RegisterSelector = nullptr;
			void* MessageSend = nullptr;
			CreateDeviceFunction CreateSystemDefaultDevice = nullptr;
		};

		MetalRuntime& Runtime()
		{
			static MetalRuntime runtime;
			return runtime;
		}

		template<typename ReturnType, typename... ArgumentTypes>
		ReturnType Send(ObjCObject object, const char* selectorName, ArgumentTypes... arguments)
		{
			using Function = ReturnType (*)(ObjCObject, ObjCSelector, ArgumentTypes...);
			const auto function = reinterpret_cast<Function>(Runtime().MessageSend);
			return function(object, Runtime().RegisterSelector(selectorName), arguments...);
		}

		ObjCObject GetClass(const char* name)
		{
			return Runtime().GetClass(name);
		}

		ObjCObject MakeString(const char* text)
		{
			return Send<ObjCObject, const char*>(GetClass("NSString"), "stringWithUTF8String:", text);
		}

		void ReleaseObject(ObjCObject object)
		{
			if (object != nullptr) Send<void>(object, "release");
		}

		std::string ErrorDescription(ObjCObject error)
		{
			if (error == nullptr) return "unknown Metal error";
			ObjCObject description = Send<ObjCObject>(error, "localizedDescription");
			const char* utf8 = description ? Send<const char*>(description, "UTF8String") : nullptr;
			return utf8 ? utf8 : "unknown Metal error";
		}

		std::size_t AlignUp(std::size_t value, std::size_t alignment)
		{
			return (value + alignment - 1) / alignment * alignment;
		}

		constexpr const char* ShaderSourceTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct EvansVertex
{
    float Values[12];
};

struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

struct DrawConstants
{
    float4 World[4];
    float4 NormalWorld[4];
    float4 ViewProjection[4];
    float4 LightViewProjection[4];
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

struct VertexOutput
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
};

float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value),
        dot(rows[2], value), dot(rows[3], value));
}

float3 SRGBToLinear(float3 color)
{
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, 2.4);
    return float3(
        color.r <= 0.04045 ? low.r : high.r,
        color.g <= 0.04045 ? low.g : high.g,
        color.b <= 0.04045 ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0);
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12
        ? value * rsqrt(lengthSquared)
        : fallbackValue;
}

vertex VertexOutput {{VERTEX_ENTRY}}(device const EvansVertex* vertices [[buffer(0)]],
    constant DrawConstants& constants [[buffer(1)]], uint vertexId [[vertex_id]])
{
    const device EvansVertex& input = vertices[vertexId];
    const float3 position = float3(input.Values[0], input.Values[1], input.Values[2]);
    const float3 inputNormal = float3(input.Values[3], input.Values[4], input.Values[5]);
    float4 worldPosition = TransformRows(constants.World, float4(position, 1.0));
    float3 normal = NormalizeSafe(
        TransformRows(constants.NormalWorld, float4(inputNormal, 0.0)).xyz,
        float3(0.0, 1.0, 0.0));
    if (constants.RenderParameters.y > 1.5)
        worldPosition.xyz += normal * (constants.MaterialParameters.w * 0.01);

    VertexOutput output;
    output.Position = TransformRows(constants.ViewProjection, worldPosition);
    output.WorldPosition = worldPosition.xyz / max(abs(worldPosition.w), 1.0e-7);
    output.Normal = normal;
    output.Color = float4(
        SRGBToLinear(saturate(float3(input.Values[6], input.Values[7], input.Values[8]))),
        input.Values[9]);
    output.LightPosition = TransformRows(constants.LightViewProjection, worldPosition);
    output.TexCoord = float2(input.Values[10], 1.0 - input.Values[11]);
    return output;
}

float4 SampleDiffuse(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> diffuseMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.x != 0
        ? diffuseMap.sample(surfaceSampler, input.TexCoord)
        : float4(1.0);
}

float SampleOpacity(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.z != 0
        ? opacityMap.sample(surfaceSampler, input.TexCoord).r
        : 1.0;
}

float SurfaceAlpha(
    VertexOutput input,
    constant DrawConstants& constants,
    float4 diffuseSample,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return saturate(
        input.Color.a *
        constants.MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input, constants, opacityMap, surfaceSampler));
}

void ApplyAlphaClip(float alpha, constant DrawConstants& constants)
{
    if (constants.MaterialParameters.z > 0.0 &&
        alpha < constants.MaterialParameters.z)
        discard_fragment();
}

void ApplyShadowAlpha(
    float alpha,
    float2 pixelPosition,
    constant DrawConstants& constants)
{
    if (constants.MaterialParameters.z > 0.0)
    {
        if (alpha < constants.MaterialParameters.z) discard_fragment();
        return;
    }
    if (constants.RenderParameters.w > 0.5)
    {
        if (alpha <= 0.0) discard_fragment();
        if (alpha >= 1.0) return;
        float threshold = fract(
            52.9829189 *
            fract(dot(floor(pixelPosition), float2(0.06711056, 0.00583715))));
        if (alpha < threshold) discard_fragment();
    }
}

float SampleShadow(
    VertexOutput input,
    constant DrawConstants& constants,
    depth2d<float, access::read> shadowMap,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (constants.RenderParameters.x < 0.5 ||
        abs(input.LightPosition.w) < 1.0e-7)
        return 1.0;

    const float3 projected =
        input.LightPosition.xyz / input.LightPosition.w;
    const float2 uv =
        float2(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
    if (!all(uv >= 0.0) || !all(uv <= 1.0) ||
        projected.z < 0.0 || projected.z > 1.0)
        return 1.0;

    const uint width = shadowMap.get_width();
    const uint height = shadowMap.get_height();
    if (width == 0 || height == 0) return 1.0;
    float directionLengthSquared = dot(directionToLight, directionToLight);
    float3 direction = directionLengthSquared > 1.0e-12
        ? directionToLight * rsqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(dot(surfaceNormal, direction));
    float sineToLight = sqrt(max(1.0 - normalToLight * normalToLight, 0.0));
    float slope = sineToLight / max(normalToLight, 0.15);
    float receiverBias = min(0.0045, 0.00035 + 0.00060 * slope);
    float2 texelPosition = uv * float2(width, height) - 0.5;
    float2 interpolation = fract(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0;
    for (int y = {{SHADOW_MIN_OFFSET}}; y <= {{SHADOW_MAX_OFFSET}}; ++y)
    {
        for (int x = {{SHADOW_MIN_OFFSET}}; x <= {{SHADOW_MAX_OFFSET}}; ++x)
        {
            int2 pixel00 = clamp(
                basePixel + int2(x, y),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel10 = clamp(
                pixel00 + int2(1, 0),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel01 = clamp(
                pixel00 + int2(0, 1),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel11 = clamp(
                pixel00 + int2(1, 1),
                int2(0),
                int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth >
                shadowMap.read(uint2(pixel00)) ? 1.0 : 0.0;
            float comparison10 = comparisonDepth >
                shadowMap.read(uint2(pixel10)) ? 1.0 : 0.0;
            float comparison01 = comparisonDepth >
                shadowMap.read(uint2(pixel01)) ? 1.0 : 0.0;
            float comparison11 = comparisonDepth >
                shadowMap.read(uint2(pixel11)) ? 1.0 : 0.0;
            float comparisonTop =
                mix(comparison00, comparison10, interpolation.x);
            float comparisonBottom =
                mix(comparison01, comparison11, interpolation.x);
            occlusion +=
                mix(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0 - {{SHADOW_STRENGTH}} * occlusion / {{SHADOW_SAMPLE_COUNT}}.0;
}

float3 ResolveSurfaceNormal(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> normalMap,
    sampler surfaceSampler,
    bool frontFacing)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0, 1.0, 0.0));
    if (!frontFacing) normal = -normal;
    if (constants.TextureFlags.y == 0) return normal;
    float3 positionDx = dfdx(input.WorldPosition);
    float3 positionDy = dfdy(input.WorldPosition);
    float2 texCoordDx = dfdx(input.TexCoord);
    float2 texCoordDy = dfdy(input.TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < 1.0e-10) return normal;
    float inverseDeterminant = 1.0 / determinant;
    float3 tangent =
        (positionDx * texCoordDy.y - positionDy * texCoordDx.y) *
        inverseDeterminant;
    float3 sourceBitangent =
        (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) *
        inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < 0.999
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    tangent = NormalizeSafe(
        tangent,
        NormalizeSafe(cross(helper, normal), float3(1.0, 0.0, 0.0)));
    float handedness =
        dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        float3(0.0, 0.0, 1.0));
    float3 tangentNormal =
        normalMap.sample(surfaceSampler, input.TexCoord).xyz * 2.0 - 1.0;
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
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= 1.0e-12) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    if (normalToLight <= 0.0) return;
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(constants.MaterialParameters.x, 1.0));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateSceneLight(
    uint lightIndex,
    VertexOutput input,
    float3 normal,
    float3 viewDirection,
    constant DrawConstants& constants,
    depth2d<float, access::read> shadowMap,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    LightConstants light = constants.Lights[lightIndex];
    uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == 0
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(constants.RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? SampleShadow(
            input, constants, shadowMap, normal, shadowDirection)
        : 1.0;
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0) *
        max(light.DirectionIntensity.w, 0.0);
    if (type == 0)
    {
        AccumulateLightSample(
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, constants, diffuseLight, specularLight);
        return;
    }
    if (type == 1)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), 1.0e-6);
        AccumulateLightSample(
            toLight, radiance / distanceSquared, visibility,
            normal, viewDirection, constants, diffuseLight, specularLight);
        return;
    }
    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, float3(0.0, -1.0, 0.0));
    float3 helper = abs(emissionDirection.y) < 0.999
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), float3(1.0, 0.0, 0.0));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), float3(0.0, 0.0, 1.0));
    for (int sampleY = 0; sampleY < 2; ++sampleY)
    {
        for (int sampleX = 0; sampleX < 2; ++sampleX)
        {
            float offsetX = sampleX == 0 ? -0.25 : 0.25;
            float offsetY = sampleY == 0 ? -0.25 : 0.25;
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), 1.0e-6);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine =
                max(dot(emissionDirection, -direction), 0.0);
            AccumulateLightSample(
                direction,
                radiance * (0.25 * emitterCosine / distanceSquared),
                visibility, normal, viewDirection, constants,
                diffuseLight, specularLight);
        }
    }
}

fragment float4 {{FRAGMENT_ENTRY}}(
    VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]],
    depth2d<float, access::read> shadowMap [[texture({{SHADOWMAP_BINDING}})]],
    texture2d<float> diffuseMap [[texture({{DIFFUSEMAP_BINDING}})]],
    texture2d<float> normalMap [[texture({{NORMALMAP_BINDING}})]],
    texture2d<float> opacityMap [[texture({{OPACITYMAP_BINDING}})]],
    sampler surfaceSampler [[sampler({{SURFACESAMPLER_BINDING}})]],
    bool frontFacing [[front_facing]])
{
    float4 diffuseSample =
        SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
    float alpha = SurfaceAlpha(
        input, constants, diffuseSample, opacityMap, surfaceSampler);
    ApplyAlphaClip(alpha, constants);
    if (constants.RenderParameters.y > 0.5)
    {
        return float4(
            SRGBToLinear(saturate(constants.PipelineColor.rgb)),
            alpha);
    }
    float3 surfaceBase =
        input.Color.rgb *
        max(constants.PipelineColor.rgb, 0.0) *
        max(diffuseSample.rgb, 0.0);
    float3 normal = ResolveSurfaceNormal(
        input, constants, normalMap, surfaceSampler, frontFacing);
    float3 viewDirection = NormalizeSafe(
        constants.CameraLightCount.xyz - input.WorldPosition,
        normal);
    float3 diffuseLight = 0.0;
    float3 specularLight = 0.0;
    uint lightCount = min(uint(constants.CameraLightCount.w + 0.5), 4u);
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(
            lightIndex, input, normal, viewDirection, constants, shadowMap,
            diffuseLight, specularLight);
    }
    float3 color =
        surfaceBase *
            max(constants.MaterialAmbientColor.rgb, 0.0) *
            max(constants.AmbientColorStrength.rgb, 0.0) *
            max(constants.AmbientColorStrength.w, 0.0) +
        surfaceBase *
            max(constants.MaterialDiffuseColor.rgb, 0.0) *
            diffuseLight +
        max(constants.MaterialSpecularColor.rgb, 0.0) * specularLight +
        max(constants.MaterialEmissiveColor.rgb, 0.0);
    return float4(PrepareDisplayLinear(color), alpha);
}

fragment void {{SHADOW_ENTRY}}(
    VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]],
    texture2d<float> diffuseMap [[texture({{DIFFUSEMAP_BINDING}})]],
    texture2d<float> opacityMap [[texture({{OPACITYMAP_BINDING}})]],
    sampler surfaceSampler [[sampler({{SURFACESAMPLER_BINDING}})]])
{
    float4 diffuseSample =
        SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
    ApplyShadowAlpha(
        SurfaceAlpha(
            input, constants, diffuseSample, opacityMap, surfaceSampler),
        input.Position.xy,
        constants);
}
)MSL";
	}

	struct MetalRHI::Impl
	{
		struct BufferResource
		{
			BufferDesc Description;
			std::vector<std::uint8_t> Bytes;
			ObjCObject Buffer = nullptr;
		};

		struct TextureResource
		{
			TextureDesc Description;
			ObjCObject Texture = nullptr;
		};

		bool Fail(const std::string& error)
		{
			LastError = error;
			return false;
		}

		bool CheckThread(const char* operation)
		{
			if (!Initialized) return Fail("Metal RHI is not initialized");
			if (OwnerThread != std::this_thread::get_id())
				return Fail(std::string(operation) + " must execute on the owning Metal RHI thread");
			return true;
		}

		void SetLabel(ObjCObject object, const std::string& label)
		{
			if (object != nullptr && !label.empty())
				Send<void, ObjCObject>(object, "setLabel:", MakeString(label.c_str()));
		}

		ObjCObject CreateNativeTexture(int width, int height, NSUInteger format,
			NSUInteger usage, const std::string& label)
		{
			ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLTextureDescriptor"), "new");
			if (descriptor == nullptr)
			{
				Fail("Metal failed to create a texture descriptor");
				return nullptr;
			}
			Send<void, NSUInteger>(descriptor, "setTextureType:", static_cast<NSUInteger>(2));
			Send<void, NSUInteger>(descriptor, "setPixelFormat:", format);
			Send<void, NSUInteger>(descriptor, "setWidth:", static_cast<NSUInteger>(width));
			Send<void, NSUInteger>(descriptor, "setHeight:", static_cast<NSUInteger>(height));
			Send<void, NSUInteger>(descriptor, "setDepth:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setMipmapLevelCount:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setArrayLength:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setSampleCount:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setStorageMode:", StorageModePrivate);
			Send<void, NSUInteger>(descriptor, "setUsage:", usage);
			ObjCObject texture = Send<ObjCObject, ObjCObject>(Device, "newTextureWithDescriptor:", descriptor);
			ReleaseObject(descriptor);
			if (texture == nullptr)
			{
				Fail("Metal failed to allocate texture " + label);
				return nullptr;
			}
			SetLabel(texture, label);
			return texture;
		}

		bool CommandSucceeded(ObjCObject commandBuffer, const char* operation)
		{
			const NSUInteger status = Send<NSUInteger>(commandBuffer, "status");
			if (status != CommandBufferStatusError) return true;
			return Fail(std::string(operation) + ": "
				+ ErrorDescription(Send<ObjCObject>(commandBuffer, "error")));
		}

		bool UploadTexture(ObjCObject texture, int width, int height,
			std::size_t bytesPerPixel, const void* source)
		{
			if (source == nullptr) return true;
			const std::size_t sourcePitch = static_cast<std::size_t>(width) * bytesPerPixel;
			const std::size_t stagingPitch = AlignUp(sourcePitch, 256);
			const std::size_t stagingSize = stagingPitch * static_cast<std::size_t>(height);
			ObjCObject pool = Send<ObjCObject>(GetClass("NSAutoreleasePool"), "new");
			ObjCObject staging = Send<ObjCObject, NSUInteger, NSUInteger>(Device,
				"newBufferWithLength:options:", stagingSize, static_cast<NSUInteger>(0));
			if (staging == nullptr)
			{
				if (pool) Send<void>(pool, "drain");
				return Fail("Metal failed to allocate a texture upload buffer");
			}
			auto* destination = static_cast<std::uint8_t*>(Send<void*>(staging, "contents"));
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(destination + static_cast<std::size_t>(row) * stagingPitch,
					static_cast<const std::uint8_t*>(source) + static_cast<std::size_t>(row) * sourcePitch,
					sourcePitch);
			}
			ObjCObject commandBuffer = Send<ObjCObject>(CommandQueue, "commandBuffer");
			ObjCObject blit = commandBuffer ? Send<ObjCObject>(commandBuffer, "blitCommandEncoder") : nullptr;
			if (blit == nullptr)
			{
				ReleaseObject(staging);
				if (pool) Send<void>(pool, "drain");
				return Fail("Metal failed to create a texture upload command encoder");
			}
			const MTLSize size { static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1 };
			const MTLOrigin origin { 0, 0, 0 };
			Send<void, ObjCObject, NSUInteger, NSUInteger, NSUInteger, MTLSize, ObjCObject,
				NSUInteger, NSUInteger, MTLOrigin>(blit,
				"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
				"toTexture:destinationSlice:destinationLevel:destinationOrigin:",
				staging, 0, stagingPitch, 0, size, texture, 0, 0, origin);
			Send<void>(blit, "endEncoding");
			Send<void>(commandBuffer, "commit");
			Send<void>(commandBuffer, "waitUntilCompleted");
			const bool succeeded = CommandSucceeded(commandBuffer, "Metal texture upload failed");
			ReleaseObject(staging);
			if (pool) Send<void>(pool, "drain");
			return succeeded;
		}

		ObjCObject CreatePipeline(
			PassType pass,
			bool colorWrite,
			bool blendEnabled)
		{
			const bool shadow = pass == PassType::Shadow;
			const RealtimeShaderProgram& shader =
				BuiltinRealtimeShader().Program;
			ObjCObject vertexFunction = Send<ObjCObject, ObjCObject>(Library,
				"newFunctionWithName:", MakeString(
					shader.FindStage(
						RealtimeShaderStage::Vertex)->Metal.c_str()));
			ObjCObject fragmentFunction = Send<ObjCObject, ObjCObject>(
				Library,
				"newFunctionWithName:",
				MakeString(shadow
					? shader.FindStage(
						RealtimeShaderStage::Shadow)->Metal.c_str()
					: shader.FindStage(
						RealtimeShaderStage::Fragment)->Metal.c_str()));
			if (vertexFunction == nullptr || fragmentFunction == nullptr)
			{
				ReleaseObject(vertexFunction);
				ReleaseObject(fragmentFunction);
				Fail("Metal shader entry points are unavailable");
				return nullptr;
			}

			ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
			Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertexFunction);
			Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragmentFunction);
			Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
			if (!shadow)
			{
				ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
				ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments,
					"objectAtIndexedSubscript:", static_cast<NSUInteger>(0));
				Send<void, NSUInteger>(
					attachment,
					"setPixelFormat:",
					PixelFormatRGBA8UnormSRGB);
				Send<void, NSUInteger>(attachment, "setWriteMask:",
					colorWrite ? ColorWriteMaskAll : ColorWriteMaskNone);
				Send<void, ObjCBool>(
					attachment,
					"setBlendingEnabled:",
					blendEnabled ? ObjCYes : ObjCNo);
				if (blendEnabled)
				{
					Send<void, NSUInteger>(
						attachment,
						"setSourceRGBBlendFactor:",
						BlendFactorSourceAlpha);
					Send<void, NSUInteger>(
						attachment,
						"setDestinationRGBBlendFactor:",
						BlendFactorOneMinusSourceAlpha);
					Send<void, NSUInteger>(
						attachment,
						"setRgbBlendOperation:",
						BlendOperationAdd);
					Send<void, NSUInteger>(
						attachment,
						"setSourceAlphaBlendFactor:",
						BlendFactorOne);
					Send<void, NSUInteger>(
						attachment,
						"setDestinationAlphaBlendFactor:",
						BlendFactorOneMinusSourceAlpha);
					Send<void, NSUInteger>(
						attachment,
						"setAlphaBlendOperation:",
						BlendOperationAdd);
				}
			}
			ObjCObject error = nullptr;
			ObjCObject pipeline = Send<ObjCObject, ObjCObject, ObjCObject*>(Device,
				"newRenderPipelineStateWithDescriptor:error:", descriptor, &error);
			if (pipeline == nullptr)
				Fail("Metal pipeline creation failed: " + ErrorDescription(error));
			ReleaseObject(descriptor);
			ReleaseObject(vertexFunction);
			ReleaseObject(fragmentFunction);
			return pipeline;
		}

		ObjCObject CreateDepthState(NSUInteger compareFunction, bool writeEnabled)
		{
			ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLDepthStencilDescriptor"), "new");
			Send<void, NSUInteger>(descriptor, "setDepthCompareFunction:", compareFunction);
			Send<void, ObjCBool>(descriptor, "setDepthWriteEnabled:", writeEnabled ? ObjCYes : ObjCNo);
			ObjCObject state = Send<ObjCObject, ObjCObject>(Device,
				"newDepthStencilStateWithDescriptor:", descriptor);
			ReleaseObject(descriptor);
			if (state == nullptr) Fail("Metal failed to create a depth-stencil state");
			return state;
		}

		bool CreatePipelineObjects()
		{
			if (!PrepareBuiltinRealtimeShader(
					RealtimeShaderLanguage::Metal,
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
			ObjCObject error = nullptr;
			ObjCObject source = MakeString(ShaderCode.c_str());
			Library = Send<ObjCObject, ObjCObject, ObjCObject, ObjCObject*>(Device,
				"newLibraryWithSource:options:error:", source, nullptr, &error);
			if (Library == nullptr)
				return Fail("Metal shader compilation failed: " + ErrorDescription(error));
			ColorPipeline = CreatePipeline(PassType::Color, true, false);
			ColorBlendPipeline = CreatePipeline(PassType::Color, true, true);
			ColorWriteDisabledPipeline =
				CreatePipeline(PassType::Color, false, false);
			ShadowPipeline = CreatePipeline(PassType::Shadow, false, false);
			ObjCObject samplerDescriptor =
				Send<ObjCObject>(GetClass("MTLSamplerDescriptor"), "new");
			if (samplerDescriptor != nullptr)
			{
				Send<void, NSUInteger>(
					samplerDescriptor, "setMinFilter:", SamplerFilterLinear);
				Send<void, NSUInteger>(
					samplerDescriptor, "setMagFilter:", SamplerFilterLinear);
				Send<void, NSUInteger>(
					samplerDescriptor,
					"setSAddressMode:",
					SamplerAddressModeRepeat);
				Send<void, NSUInteger>(
					samplerDescriptor,
					"setTAddressMode:",
					SamplerAddressModeRepeat);
				Send<void, NSUInteger>(
					samplerDescriptor,
					"setRAddressMode:",
					SamplerAddressModeRepeat);
				SurfaceSampler = Send<ObjCObject, ObjCObject>(
					Device,
					"newSamplerStateWithDescriptor:",
					samplerDescriptor);
				ReleaseObject(samplerDescriptor);
			}
			DepthWriteState = CreateDepthState(CompareFunctionLess, true);
			DepthReadState = CreateDepthState(CompareFunctionLessEqual, false);
			DepthDisabledState = CreateDepthState(CompareFunctionAlways, false);
			return ColorPipeline != nullptr && ColorBlendPipeline != nullptr
				&& ColorWriteDisabledPipeline != nullptr
				&& ShadowPipeline != nullptr && DepthWriteState != nullptr
				&& DepthReadState != nullptr && DepthDisabledState != nullptr
				&& SurfaceSampler != nullptr;
		}

		bool CreateOffscreenTargets(int width, int height)
		{
			ObjCObject color = CreateNativeTexture(width, height, PixelFormatRGBA8UnormSRGB,
				TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal offscreen color");
			ObjCObject depth = CreateNativeTexture(width, height, PixelFormatDepth32Float,
				TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal offscreen depth");
			if (color == nullptr || depth == nullptr)
			{
				ReleaseObject(color);
				ReleaseObject(depth);
				return false;
			}
			ReleaseObject(ColorTexture);
			ReleaseObject(DepthTexture);
			ColorTexture = color;
			DepthTexture = depth;
			return true;
		}

		void ReleaseResources()
		{
			if (RenderEncoder != nullptr)
			{
				Send<void>(RenderEncoder, "endEncoding");
				RenderEncoder = nullptr;
			}
			ReleaseObject(CommandBuffer);
			CommandBuffer = nullptr;
			if (FramePool != nullptr)
			{
				Send<void>(FramePool, "drain");
				FramePool = nullptr;
			}
			for (auto& pair : Buffers) ReleaseObject(pair.second.Buffer);
			for (auto& pair : Textures) ReleaseObject(pair.second.Texture);
			Buffers.clear();
			Textures.clear();
			ReleaseObject(DummyShadowTexture);
			ReleaseObject(DummyColorTexture);
			ReleaseObject(SurfaceSampler);
			ReleaseObject(ColorTexture);
			ReleaseObject(DepthTexture);
			ReleaseObject(ColorPipeline);
			ReleaseObject(ColorBlendPipeline);
			ReleaseObject(ColorWriteDisabledPipeline);
			ReleaseObject(ShadowPipeline);
			ReleaseObject(DepthWriteState);
			ReleaseObject(DepthReadState);
			ReleaseObject(DepthDisabledState);
			ReleaseObject(Library);
			ReleaseObject(CommandQueue);
			ReleaseObject(Device);
			DummyShadowTexture = nullptr;
			DummyColorTexture = nullptr;
			SurfaceSampler = nullptr;
			ColorTexture = nullptr;
			DepthTexture = nullptr;
			ColorPipeline = nullptr;
			ColorBlendPipeline = nullptr;
			ColorWriteDisabledPipeline = nullptr;
			ShadowPipeline = nullptr;
			DepthWriteState = nullptr;
			DepthReadState = nullptr;
			DepthDisabledState = nullptr;
			Library = nullptr;
			CommandQueue = nullptr;
			Device = nullptr;
		}

		bool EnqueueReadback(ObjCObject blit, ObjCObject texture, int width, int height,
			std::size_t bytesPerPixel, ObjCObject& buffer, std::size_t& rowPitch)
		{
			rowPitch = AlignUp(static_cast<std::size_t>(width) * bytesPerPixel, 256);
			const std::size_t byteCount = rowPitch * static_cast<std::size_t>(height);
			buffer = Send<ObjCObject, NSUInteger, NSUInteger>(Device,
				"newBufferWithLength:options:", byteCount, static_cast<NSUInteger>(0));
			if (buffer == nullptr) return Fail("Metal failed to allocate a readback buffer");
			const MTLOrigin origin { 0, 0, 0 };
			const MTLSize size { static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1 };
			Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject,
				NSUInteger, NSUInteger, NSUInteger>(blit,
				"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:"
				"destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
				texture, 0, 0, origin, size, buffer, 0, rowPitch, 0);
			return true;
		}

		template<typename ElementType>
		void CopyReadback(ObjCObject buffer, std::size_t rowPitch, int width, int height,
			std::vector<ElementType>& output)
		{
			output.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
			const auto* source = static_cast<const std::uint8_t*>(Send<void*>(buffer, "contents"));
			const std::size_t outputPitch = static_cast<std::size_t>(width) * sizeof(ElementType);
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(reinterpret_cast<std::uint8_t*>(output.data())
					+ static_cast<std::size_t>(row) * outputPitch,
					source + static_cast<std::size_t>(row) * rowPitch, outputPitch);
			}
		}

		RHICreateInfo CreateInfo;
		ObjCObject Device = nullptr;
		ObjCObject CommandQueue = nullptr;
		ObjCObject Library = nullptr;
		ObjCObject ColorPipeline = nullptr;
		ObjCObject ColorBlendPipeline = nullptr;
		ObjCObject ColorWriteDisabledPipeline = nullptr;
		ObjCObject ShadowPipeline = nullptr;
		ObjCObject DepthWriteState = nullptr;
		ObjCObject DepthReadState = nullptr;
		ObjCObject DepthDisabledState = nullptr;
		ObjCObject ColorTexture = nullptr;
		ObjCObject DepthTexture = nullptr;
		ObjCObject DummyShadowTexture = nullptr;
		ObjCObject DummyColorTexture = nullptr;
		ObjCObject SurfaceSampler = nullptr;
		ObjCObject FramePool = nullptr;
		ObjCObject CommandBuffer = nullptr;
		ObjCObject RenderEncoder = nullptr;
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
		std::uint32_t NextBuffer = 0x4d000001u;
		std::uint32_t NextTexture = 0x4d800001u;
		std::uint64_t FrameNumber = 0;
		bool Initialized = false;
		bool FrameOpen = false;
		bool PassOpen = false;
		bool PipelineBound = false;
		bool WorldBound = false;
		bool ViewProjectionBound = false;
		bool LightViewProjectionBound = false;
	};

	MetalRHI::MetalRHI() : m_Impl(std::make_unique<Impl>()) {}
	MetalRHI::~MetalRHI() { Shutdown(); }

	bool MetalRHI::Initialize(const RHICreateInfo& createInfo)
	{
		if (m_Impl->Initialized) return m_Impl->Fail("Metal RHI is already initialized");
		if (createInfo.Width <= 0 || createInfo.Height <= 0
			|| createInfo.Width > 16384 || createInfo.Height > 16384)
			return m_Impl->Fail("Metal dimensions are invalid");
		if (!Runtime().Load(m_Impl->LastError)) return false;
		ObjCObject pool = Send<ObjCObject>(GetClass("NSAutoreleasePool"), "new");
		m_Impl->Device = Runtime().CreateSystemDefaultDevice();
		if (m_Impl->Device == nullptr)
		{
			if (pool) Send<void>(pool, "drain");
			return m_Impl->Fail("Metal has no available system device");
		}
		m_Impl->CommandQueue = Send<ObjCObject>(m_Impl->Device, "newCommandQueue");
		if (m_Impl->CommandQueue == nullptr)
		{
			m_Impl->Fail("Metal failed to create a command queue");
			m_Impl->ReleaseResources();
			if (pool) Send<void>(pool, "drain");
			return false;
		}
		m_Impl->CreateInfo = createInfo;
		if (!m_Impl->CreatePipelineObjects()
			|| !m_Impl->CreateOffscreenTargets(createInfo.Width, createInfo.Height))
		{
			const std::string error = m_Impl->LastError;
			m_Impl->ReleaseResources();
			if (pool) Send<void>(pool, "drain");
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->DummyShadowTexture = m_Impl->CreateNativeTexture(1, 1, PixelFormatDepth32Float,
			TextureUsageShaderRead, "Evans Metal dummy shadow");
		const float clearDepth = 1.0f;
		if (m_Impl->DummyShadowTexture == nullptr
			|| !m_Impl->UploadTexture(m_Impl->DummyShadowTexture, 1, 1, sizeof(float), &clearDepth))
		{
			const std::string error = m_Impl->LastError;
			m_Impl->ReleaseResources();
			if (pool) Send<void>(pool, "drain");
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->DummyColorTexture = m_Impl->CreateNativeTexture(
			1,
			1,
			PixelFormatRGBA8Unorm,
			TextureUsageShaderRead,
			"Evans Metal dummy surface");
		const std::uint32_t white = 0xffffffffu;
		if (m_Impl->DummyColorTexture == nullptr
			|| !m_Impl->UploadTexture(
				m_Impl->DummyColorTexture,
				1,
				1,
				4,
				&white))
		{
			const std::string error = m_Impl->LastError;
			m_Impl->ReleaseResources();
			if (pool) Send<void>(pool, "drain");
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->OwnerThread = std::this_thread::get_id();
		m_Impl->Initialized = true;
		m_Impl->LastError.clear();
		if (pool) Send<void>(pool, "drain");
		return true;
	}

	void MetalRHI::Shutdown()
	{
		if (!m_Impl) return;
		m_Impl->ReleaseResources();
		m_Impl->Initialized = false;
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->OwnerThread = {};
	}

	bool MetalRHI::IsInitialized() const { return m_Impl->Initialized; }
	RHIBackend MetalRHI::GetBackend() const { return RHIBackend::Metal; }
	const char* MetalRHI::GetName() const { return "Metal Native RHI"; }
	RHICapabilities MetalRHI::GetCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = true;
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = 16384;
		return capabilities;
	}

	bool MetalRHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0
			|| width > 16384 || height > 16384)
			return m_Impl->Fail("Metal cannot resize in its current state");
		if (!m_Impl->CreateOffscreenTargets(width, height)) return false;
		m_Impl->CreateInfo.Width = width;
		m_Impl->CreateInfo.Height = height;
		return true;
	}

	BufferHandle MetalRHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateBuffer") || desc.SizeInBytes == 0)
		{
			m_Impl->Fail("Metal buffer description is invalid");
			return {};
		}
		Impl::BufferResource resource;
		resource.Description = desc;
		resource.Bytes.resize(desc.SizeInBytes);
		if (initialData) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
		resource.Buffer = Send<ObjCObject, NSUInteger, NSUInteger>(m_Impl->Device,
			"newBufferWithLength:options:", desc.SizeInBytes, static_cast<NSUInteger>(0));
		if (resource.Buffer == nullptr)
		{
			m_Impl->Fail("Metal failed to allocate a buffer");
			return {};
		}
		if (initialData)
			std::memcpy(Send<void*>(resource.Buffer, "contents"), initialData, desc.SizeInBytes);
		m_Impl->SetLabel(resource.Buffer, desc.DebugName);
		const BufferHandle handle { m_Impl->NextBuffer++ };
		m_Impl->Buffers.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool MetalRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer")) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || data == nullptr || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("Metal buffer update is invalid");
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		std::memcpy(static_cast<std::uint8_t*>(Send<void*>(iterator->second.Buffer, "contents")) + offset,
			data, size);
		return true;
	}

	bool MetalRHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer")) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (!buffer || iterator == m_Impl->Buffers.end()) return m_Impl->Fail("Metal buffer handle is invalid");
		ReleaseObject(iterator->second.Buffer);
		m_Impl->Buffers.erase(iterator);
		return true;
	}

	TextureHandle MetalRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0
			|| desc.Width > 16384 || desc.Height > 16384)
		{
			m_Impl->Fail("Metal texture description is invalid");
			return {};
		}
		const NSUInteger format = desc.Format == TextureFormat::Depth32
			? PixelFormatDepth32Float
			: (desc.RenderTarget || desc.ColorSpace == TextureColorSpace::SRGB
				? PixelFormatRGBA8UnormSRGB
				: PixelFormatRGBA8Unorm);
		NSUInteger usage = TextureUsageShaderRead;
		if (desc.RenderTarget) usage |= TextureUsageRenderTarget;
		Impl::TextureResource resource;
		resource.Description = desc;
		resource.Texture = m_Impl->CreateNativeTexture(desc.Width, desc.Height, format, usage, desc.DebugName);
		const std::size_t bytesPerPixel = desc.Format == TextureFormat::Depth32 ? sizeof(float) : 4;
		if (resource.Texture == nullptr
			|| !m_Impl->UploadTexture(resource.Texture, desc.Width, desc.Height, bytesPerPixel, initialData))
		{
			ReleaseObject(resource.Texture);
			return {};
		}
		const TextureHandle handle { m_Impl->NextTexture++ };
		m_Impl->Textures.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool MetalRHI::DestroyTexture(TextureHandle texture)
	{
		if (!m_Impl->CheckThread("DestroyTexture")) return false;
		auto iterator = m_Impl->Textures.find(texture.Value);
		if (!texture || iterator == m_Impl->Textures.end()) return m_Impl->Fail("Metal texture handle is invalid");
		ReleaseObject(iterator->second.Texture);
		m_Impl->Textures.erase(iterator);
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
		return true;
	}

	bool MetalRHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
			return m_Impl->Fail("Metal frame cannot begin in the current state");
		m_Impl->FramePool = Send<ObjCObject>(GetClass("NSAutoreleasePool"), "new");
		ObjCObject commandBuffer = Send<ObjCObject>(m_Impl->CommandQueue, "commandBuffer");
		m_Impl->CommandBuffer = commandBuffer ? Send<ObjCObject>(commandBuffer, "retain") : nullptr;
		if (m_Impl->CommandBuffer == nullptr)
		{
			if (m_Impl->FramePool) Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
			return m_Impl->Fail("Metal failed to create a command buffer");
		}
		m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Backend = RHIBackend::Metal;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = true;
		m_Impl->LastError.clear();
		return true;
	}

	void MetalRHI::AbortFrame()
	{
		if (!m_Impl || !m_Impl->Initialized)
		{
			return;
		}
		if (m_Impl->OwnerThread != std::this_thread::get_id())
		{
			m_Impl->LastError =
				"AbortFrame must execute on the owning Metal RHI thread";
			return;
		}
		if (m_Impl->RenderEncoder != nullptr)
		{
			Send<void>(m_Impl->RenderEncoder, "endEncoding");
			m_Impl->RenderEncoder = nullptr;
		}
		ReleaseObject(m_Impl->CommandBuffer);
		m_Impl->CommandBuffer = nullptr;
		if (m_Impl->FramePool != nullptr)
		{
			Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
		}
		m_Impl->PendingFrame = {};
		m_Impl->CurrentPass = {};
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
	}

	bool MetalRHI::BeginPass(const PassDesc& desc)
	{
		if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("Metal pass cannot begin in the current state");
		ObjCObject colorTarget = m_Impl->ColorTexture;
		ObjCObject depthTarget = m_Impl->DepthTexture;
		int width = m_Impl->CreateInfo.Width;
		int height = m_Impl->CreateInfo.Height;
		if (desc.Target)
		{
			auto iterator = m_Impl->Textures.find(desc.Target.Value);
			if (iterator == m_Impl->Textures.end()) return m_Impl->Fail("Metal pass target is invalid");
			width = iterator->second.Description.Width;
			height = iterator->second.Description.Height;
			if (!iterator->second.Description.RenderTarget)
				return m_Impl->Fail("Metal pass target is not renderable");
			if (desc.Type == PassType::Shadow)
			{
				if (iterator->second.Description.Format != TextureFormat::Depth32)
					return m_Impl->Fail("Metal shadow target is not a depth texture");
				colorTarget = nullptr;
				depthTarget = iterator->second.Texture;
				m_Impl->LastShadowTexture = desc.Target;
			}
			else
			{
				if (iterator->second.Description.Format != TextureFormat::RGBA8)
					return m_Impl->Fail("Metal color target is not an RGBA8 texture");
				if (width != m_Impl->CreateInfo.Width || height != m_Impl->CreateInfo.Height)
					return m_Impl->Fail("Metal color targets must match the offscreen depth dimensions");
				colorTarget = iterator->second.Texture;
			}
		}
		else if (desc.Type == PassType::Shadow)
		{
			return m_Impl->Fail("Metal shadow pass requires a depth texture");
		}

		ObjCObject passDescriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),
			"renderPassDescriptor");
		if (colorTarget != nullptr)
		{
			ObjCObject attachments = Send<ObjCObject>(passDescriptor, "colorAttachments");
			ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments,
				"objectAtIndexedSubscript:", static_cast<NSUInteger>(0));
			Send<void, ObjCObject>(attachment, "setTexture:", colorTarget);
			Send<void, NSUInteger>(attachment, "setLoadAction:",
				desc.ClearColorEnabled ? LoadActionClear : LoadActionLoad);
			Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
			const MTLClearColor clearColor { desc.ClearColor.X, desc.ClearColor.Y,
				desc.ClearColor.Z, desc.ClearColor.W };
			Send<void, MTLClearColor>(attachment, "setClearColor:", clearColor);
		}
		ObjCObject depthAttachment = Send<ObjCObject>(passDescriptor, "depthAttachment");
		Send<void, ObjCObject>(depthAttachment, "setTexture:", depthTarget);
		Send<void, NSUInteger>(depthAttachment, "setLoadAction:",
			desc.ClearDepthEnabled ? LoadActionClear : LoadActionLoad);
		Send<void, NSUInteger>(depthAttachment, "setStoreAction:", StoreActionStore);
		Send<void, double>(depthAttachment, "setClearDepth:", 1.0);
		m_Impl->RenderEncoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer,
			"renderCommandEncoderWithDescriptor:", passDescriptor);
		if (m_Impl->RenderEncoder == nullptr)
			return m_Impl->Fail("Metal failed to create a render command encoder");
		const MTLViewport viewport { 0.0, 0.0, static_cast<double>(width),
			static_cast<double>(height), 0.0, 1.0 };
		Send<void, MTLViewport>(m_Impl->RenderEncoder, "setViewport:", viewport);
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

	bool MetalRHI::BindPipeline(const PipelineState& pipeline)
	{
		if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
			return m_Impl->Fail("Metal pipeline must be bound inside a pass");
		m_Impl->BoundPipeline = pipeline;
		m_Impl->PipelineBound = true;
		return true;
	}

	bool MetalRHI::BindWorldTransform(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen) return false;
		m_Impl->BoundWorld = value;
		m_Impl->WorldBound = true;
		return true;
	}

	bool MetalRHI::BindViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen) return false;
		m_Impl->BoundViewProjection = value;
		m_Impl->ViewProjectionBound = true;
		return true;
	}

	bool MetalRHI::BindLightViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen) return false;
		m_Impl->BoundLightViewProjection = value;
		m_Impl->LightViewProjectionBound = true;
		return true;
	}

	bool MetalRHI::BindVertexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Vertex)
			return m_Impl->Fail("Metal vertex buffer is invalid");
		m_Impl->BoundVertexBuffer = buffer;
		return true;
	}

	bool MetalRHI::BindIndexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Index)
			return m_Impl->Fail("Metal index buffer is invalid");
		m_Impl->BoundIndexBuffer = buffer;
		return true;
	}

	bool MetalRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
		if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("Metal DrawIndexed is missing required state");
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("Metal DrawIndexed buffers are invalid");
		const std::size_t availableIndices = index->second.Bytes.size() / sizeof(std::uint32_t);
		if (firstIndex > availableIndices || indexCount > availableIndices - firstIndex)
			return m_Impl->Fail("Metal DrawIndexed exceeds the index buffer");
		if (vertex->second.Bytes.size() % sizeof(Vertex) != 0
			|| index->second.Bytes.size() % sizeof(std::uint32_t) != 0 || indexCount % 3 != 0)
			return m_Impl->Fail("Metal DrawIndexed requires complete vertices and triangle-list indices");
		const std::size_t availableVertices = vertex->second.Bytes.size() / sizeof(Vertex);
		const auto* indices = reinterpret_cast<const std::uint32_t*>(index->second.Bytes.data());
		for (std::size_t offset = firstIndex; offset < static_cast<std::size_t>(firstIndex) + indexCount; ++offset)
		{
			const std::int64_t resolved = static_cast<std::int64_t>(indices[offset]) + baseVertex;
			if (resolved < 0 || static_cast<std::uint64_t>(resolved) >= availableVertices)
				return m_Impl->Fail("Metal DrawIndexed references a vertex outside the vertex buffer");
		}

		ObjCObject shadowTexture = m_Impl->DummyShadowTexture;
		bool shadowAvailable = false;
		if (m_Impl->CurrentPass.Type != PassType::Shadow
			&& m_Impl->BoundPipeline.SamplesShadowMap
			&& m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(
				m_Impl->LastShadowTexture.Value);
			if (shadow == m_Impl->Textures.end()
				|| shadow->second.Description.Format != TextureFormat::Depth32
				|| shadow->second.Texture == nullptr)
			{
				return m_Impl->Fail(
					"Metal shadow map resource is unavailable");
			}
			shadowTexture = shadow->second.Texture;
			shadowAvailable = true;
		}
		std::array<ObjCObject, 3> surfaceTextures {
			m_Impl->DummyColorTexture,
			m_Impl->DummyColorTexture,
			m_Impl->DummyColorTexture };
		auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
			ObjCObject& nativeTexture)
		{
			if (!handle) return true;
			auto texture = m_Impl->Textures.find(handle.Value);
			if (texture == m_Impl->Textures.end()
				|| texture->second.Description.Format != TextureFormat::RGBA8
				|| texture->second.Texture == nullptr)
			{
				return m_Impl->Fail(
					std::string("Metal ") + usage + " texture is invalid");
			}
			nativeTexture = texture->second.Texture;
			return true;
		};
		if (!resolveSurfaceTexture(
				m_Impl->BoundPipeline.DiffuseTexture,
				"diffuse",
				surfaceTextures[0])
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.NormalTexture,
				"normal",
				surfaceTextures[1])
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.OpacityTexture,
				"opacity",
				surfaceTextures[2]))
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
		ObjCObject pipeline = m_Impl->CurrentPass.Type == PassType::Shadow
			? m_Impl->ShadowPipeline
			: (m_Impl->BoundPipeline.ColorWrite ? m_Impl->ColorPipeline
				: m_Impl->ColorWriteDisabledPipeline);
		if (m_Impl->CurrentPass.Type == PassType::Color
			&& m_Impl->BoundPipeline.ColorWrite
			&& m_Impl->BoundPipeline.BlendEnabled)
		{
			pipeline = m_Impl->ColorBlendPipeline;
		}
		Send<void, ObjCObject>(m_Impl->RenderEncoder, "setRenderPipelineState:", pipeline);
		const bool wireframe = m_Impl->BoundPipeline.Fill == FillMode::Wireframe;
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setTriangleFillMode:",
			wireframe ? TriangleFillModeLines : TriangleFillModeFill);
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setCullMode:",
			m_Impl->BoundPipeline.OutlineExtrusion ? CullModeFront : CullModeNone);
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setFrontFacingWinding:", WindingCounterClockwise);
		if (m_Impl->CurrentPass.Type == PassType::Shadow)
			Send<void, float, float, float>(m_Impl->RenderEncoder, "setDepthBias:slopeScale:clamp:",
				0.0015f, 2.0f, 0.01f);
		else
			Send<void, float, float, float>(m_Impl->RenderEncoder, "setDepthBias:slopeScale:clamp:",
				0.0f, 0.0f, 0.0f);
		ObjCObject depthState = !m_Impl->BoundPipeline.DepthTest ? m_Impl->DepthDisabledState
			: (m_Impl->BoundPipeline.DepthWrite ? m_Impl->DepthWriteState : m_Impl->DepthReadState);
		Send<void, ObjCObject>(m_Impl->RenderEncoder, "setDepthStencilState:", depthState);
		Send<void, ObjCObject, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setVertexBuffer:offset:atIndex:", vertex->second.Buffer, 0, 0);
		Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setVertexBytes:length:atIndex:", &constants, sizeof(constants), 1);
		Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 1);
		Send<void, ObjCObject, NSUInteger>(m_Impl->RenderEncoder,
			"setFragmentTexture:atIndex:", shadowTexture, 0);
		for (NSUInteger textureIndex = 0;
			textureIndex < surfaceTextures.size();
			++textureIndex)
		{
			Send<void, ObjCObject, NSUInteger>(m_Impl->RenderEncoder,
				"setFragmentTexture:atIndex:",
				surfaceTextures[textureIndex],
				textureIndex + 1);
		}
		Send<void, ObjCObject, NSUInteger>(m_Impl->RenderEncoder,
			"setFragmentSamplerState:atIndex:", m_Impl->SurfaceSampler, 0);
		const NSUInteger indexOffset = static_cast<NSUInteger>(firstIndex) * sizeof(std::uint32_t);
		Send<void, NSUInteger, NSUInteger, NSUInteger, ObjCObject, NSUInteger, NSUInteger, NSInteger, NSUInteger>(
			m_Impl->RenderEncoder,
			"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
			"instanceCount:baseVertex:baseInstance:",
			PrimitiveTypeTriangle, static_cast<NSUInteger>(indexCount), IndexTypeUInt32,
			index->second.Buffer, indexOffset, 1, static_cast<NSInteger>(baseVertex), 0);

		DrawCallRecord record;
		record.VertexBuffer = m_Impl->BoundVertexBuffer;
		record.IndexBuffer = m_Impl->BoundIndexBuffer;
		record.Pipeline = m_Impl->BoundPipeline;
		record.WorldTransform = m_Impl->BoundWorld;
		record.FirstIndex = firstIndex;
		record.IndexCount = indexCount;
		record.BaseVertex = baseVertex;
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
		FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
		++statistics.DrawCallCount;
		statistics.SubmittedTriangleCount += indexCount / 3;
		statistics.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1u : 0u;
		statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1u : 0u;
		return true;
	}

	bool MetalRHI::EndPass()
	{
		if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
			return m_Impl->Fail("Metal has no open pass");
		Send<void>(m_Impl->RenderEncoder, "endEncoding");
		m_Impl->RenderEncoder = nullptr;
		m_Impl->PassOpen = false;
		return true;
	}

	bool MetalRHI::Present()
	{
		if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("Metal frame cannot be presented in the current state");
		m_Impl->PendingFrame.Statistics.PassCount =
			static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size());
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || m_Impl->CreateInfo.CaptureDiagnosticBuffers ||
			m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0;

		ObjCObject colorReadback = nullptr;
		ObjCObject depthReadback = nullptr;
		ObjCObject shadowReadback = nullptr;
		std::size_t colorPitch = 0;
		std::size_t depthPitch = 0;
		std::size_t shadowPitch = 0;
		ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
		if (blit == nullptr) return m_Impl->Fail("Metal failed to create a readback command encoder");
		bool enqueued = m_Impl->EnqueueReadback(blit, m_Impl->ColorTexture,
			m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, 4, colorReadback, colorPitch);
		if (enqueued && captureDepth)
		{
			enqueued = m_Impl->EnqueueReadback(blit, m_Impl->DepthTexture,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				sizeof(float), depthReadback, depthPitch);
		}
		Impl::TextureResource* shadow = nullptr;
		if (enqueued && m_Impl->CreateInfo.CaptureDiagnosticBuffers &&
			m_Impl->LastShadowTexture)
		{
			auto iterator = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (iterator != m_Impl->Textures.end())
			{
				shadow = &iterator->second;
				enqueued = m_Impl->EnqueueReadback(blit, shadow->Texture,
					shadow->Description.Width, shadow->Description.Height,
					sizeof(float), shadowReadback, shadowPitch);
			}
		}
		Send<void>(blit, "endEncoding");
		if (!enqueued)
		{
			ReleaseObject(colorReadback);
			ReleaseObject(depthReadback);
			ReleaseObject(shadowReadback);
			return false;
		}
		Send<void>(m_Impl->CommandBuffer, "commit");
		Send<void>(m_Impl->CommandBuffer, "waitUntilCompleted");
		if (!m_Impl->CommandSucceeded(m_Impl->CommandBuffer, "Metal frame execution failed"))
		{
			ReleaseObject(colorReadback);
			ReleaseObject(depthReadback);
			ReleaseObject(shadowReadback);
			return false;
		}
		std::vector<std::uint32_t> colorPixels;
		m_Impl->CopyReadback(colorReadback, colorPitch, m_Impl->CreateInfo.Width,
			m_Impl->CreateInfo.Height, colorPixels);
		m_Impl->PendingFrame.ColorRGBA8.resize(colorPixels.size() * sizeof(std::uint32_t));
		std::memcpy(m_Impl->PendingFrame.ColorRGBA8.data(), colorPixels.data(),
			m_Impl->PendingFrame.ColorRGBA8.size());
		if (depthReadback != nullptr)
		{
			m_Impl->CopyReadback(depthReadback, depthPitch, m_Impl->CreateInfo.Width,
				m_Impl->CreateInfo.Height, m_Impl->PendingFrame.Depth32);
		}
		if (shadow != nullptr && shadowReadback != nullptr)
		{
			m_Impl->PendingFrame.ShadowWidth = static_cast<std::uint32_t>(shadow->Description.Width);
			m_Impl->PendingFrame.ShadowHeight = static_cast<std::uint32_t>(shadow->Description.Height);
			m_Impl->CopyReadback(shadowReadback, shadowPitch, shadow->Description.Width,
				shadow->Description.Height, m_Impl->PendingFrame.ShadowDepth32);
		}
		ReleaseObject(colorReadback);
		ReleaseObject(depthReadback);
		ReleaseObject(shadowReadback);
		ReleaseObject(m_Impl->CommandBuffer);
		m_Impl->CommandBuffer = nullptr;
		if (m_Impl->FramePool != nullptr)
		{
			Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
		}
		m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameOpen = false;
		return true;
	}

	FrameCapture MetalRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
	std::string MetalRHI::GetLastError() const { return m_Impl->LastError; }
}

#else

#include <string>

namespace Evans::Renderer
{
	struct MetalRHI::Impl
	{
		std::string LastError = "Metal is only available on Apple platforms";
	};

	MetalRHI::MetalRHI() : m_Impl(std::make_unique<Impl>()) {}
	MetalRHI::~MetalRHI() = default;
	bool MetalRHI::Initialize(const RHICreateInfo&) { return false; }
	void MetalRHI::Shutdown() {}
	bool MetalRHI::IsInitialized() const { return false; }
	RHIBackend MetalRHI::GetBackend() const { return RHIBackend::Metal; }
	const char* MetalRHI::GetName() const { return "Metal Unavailable"; }
	RHICapabilities MetalRHI::GetCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = false;
		capabilities.Wireframe = false;
		capabilities.ShadowMaps = false;
		capabilities.Outline = false;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = 0;
		return capabilities;
	}
	bool MetalRHI::Resize(int, int) { return false; }
	BufferHandle MetalRHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
	bool MetalRHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
	bool MetalRHI::DestroyBuffer(BufferHandle) { return false; }
	TextureHandle MetalRHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
	bool MetalRHI::DestroyTexture(TextureHandle) { return false; }
	bool MetalRHI::BeginFrame() { return false; }
	void MetalRHI::AbortFrame() {}
	bool MetalRHI::BeginPass(const PassDesc&) { return false; }
	bool MetalRHI::BindPipeline(const PipelineState&) { return false; }
	bool MetalRHI::BindWorldTransform(const Matrix4&) { return false; }
	bool MetalRHI::BindViewProjection(const Matrix4&) { return false; }
	bool MetalRHI::BindLightViewProjection(const Matrix4&) { return false; }
	bool MetalRHI::BindVertexBuffer(BufferHandle) { return false; }
	bool MetalRHI::BindIndexBuffer(BufferHandle) { return false; }
	bool MetalRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
	bool MetalRHI::EndPass() { return false; }
	bool MetalRHI::Present() { return false; }
	FrameCapture MetalRHI::GetLastFrameCapture() const { return {}; }
	std::string MetalRHI::GetLastError() const { return m_Impl->LastError; }
}

#endif
