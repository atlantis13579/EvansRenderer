#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/Metal/MetalShaderCache.h"
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
#include "Renderer/Metal/MetalRHI.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"
#include "Renderer/TechniqueShader.h"
#include "Renderer/PostProcessing/PostProcessContract.h"
#include "Renderer/PostProcessing/HdrContract.h"
#include "Renderer/Pipelines/DeferredContract.h"
#include "Renderer/RayTracing/RayTracedAOContract.h"
#include "Renderer/Reflections/ReflectionContract.h"
#include "Renderer/Reflections/CubemapReflectionContract.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/Reflections/EnvironmentMapReflectionContract.h"
#include "Renderer/Reflections/PlanarReflectionContract.h"
#include "Renderer/PostProcessing/DepthOfFieldContract.h"
#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Renderer/PostProcessing/TemporalAAContract.h"
#include "Renderer/Lighting/GlobalIlluminationContract.h"
#include "Renderer/BakedLighting/LightProbeContract.h"
#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Volumetrics/VolumetricLightingContract.h"
#include "Renderer/Volumetrics/LightShaftContract.h"
#include "Renderer/Transparency/WeightedTransparencyContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/Geometry/Instancing.h"
#include "Renderer/Lighting/LightListFrame.h"
#include "Renderer/Pipelines/TiledDeferredContract.h"

#include <memory>

#if defined(__APPLE__)

#include "Renderer/Metal/MetalContactShadows.h"
#include "Renderer/Metal/MetalForwardPlus.h"
#include "Renderer/Metal/MetalClusteredForward.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <optional>
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
		constexpr NSUInteger PixelFormatRGBA16Float = 115;
        constexpr NSUInteger PixelFormatR32Float = 55;
        constexpr NSUInteger PixelFormatRG32Float = 105;
        constexpr NSUInteger PixelFormatRGBA32Float = 125;
        constexpr float MaximumHalfFloatRadiance = 65504.0f;
        constexpr std::uint32_t LightShaftPhaseCount = 2;
        constexpr NSUInteger CascadeConstantBufferIndex = 2;
        constexpr NSUInteger FirstCascadeTextureIndex = 10;
        constexpr NSUInteger InstanceConstantBufferIndex = 3;
        constexpr NSUInteger FogConstantBufferIndex = 4;
        constexpr NSUInteger TransparencyConstantBufferIndex = 5;
        constexpr NSUInteger PCSSConstantBufferIndex = 6;
        constexpr NSUInteger VSMConstantBufferIndex = PCSSConstantBufferIndex; // Exclusive receiver filters.
        constexpr NSUInteger ForwardPlusSelectionIndex = 7;
        constexpr NSUInteger ForwardPlusSceneIndex = 8;
        constexpr NSUInteger ForwardPlusHeaderIndex = 9;
        constexpr NSUInteger ForwardPlusLightIndex = 10;
        constexpr NSUInteger CommandStatusCommitted = 2;
        constexpr NSUInteger CommandStatusCompleted = 4;
        constexpr NSUInteger FogAtlasTextureIndex = FirstCascadeTextureIndex + MaximumShadowCascades;
        constexpr int MaximumNativeTextureDimension = 16384;
        constexpr std::size_t InstanceUploadPageBytes = 64 * 1024;
        // Conservative constant-buffer offset alignment for both Mac and Apple GPUs.
        constexpr std::size_t InstanceUploadAlignment = 256;
        constexpr std::size_t InstanceUploadStride =
            (sizeof(InstanceConstantBlock) + InstanceUploadAlignment - 1) / InstanceUploadAlignment * InstanceUploadAlignment;
        static_assert(InstanceUploadStride <= InstanceUploadPageBytes, "Instance block exceeds its upload page");
		constexpr NSUInteger PixelFormatDepth32Float = 252;
		constexpr NSUInteger TextureType2D = 2;
        constexpr NSUInteger TextureType2DMultisample = 4;
        constexpr NSUInteger TextureTypeCube = 5;
        constexpr NSUInteger TextureUsageShaderRead = 1;
		constexpr NSUInteger TextureUsageRenderTarget = 4;
		constexpr NSUInteger StorageModePrivate = 2;
        constexpr NSUInteger ResourceStorageModeShift = 4;
		constexpr NSUInteger LoadActionLoad = 1;
		constexpr NSUInteger LoadActionClear = 2;
		constexpr NSUInteger StoreActionStore = 1;
        constexpr NSUInteger StoreActionMultisampleResolve = 2;
		constexpr NSUInteger PrimitiveTypeTriangle = 3;
		constexpr NSUInteger IndexTypeUInt32 = 1;
		constexpr NSUInteger TriangleFillModeFill = 0;
		constexpr NSUInteger TriangleFillModeLines = 1;
		constexpr NSUInteger CullModeNone = 0;
		constexpr NSUInteger CullModeFront = 1;
		constexpr NSUInteger CullModeBack = 2;
		constexpr NSUInteger WindingCounterClockwise = 1;
		constexpr NSUInteger WindingClockwise = 0;
		constexpr NSUInteger CompareFunctionLess = 1;
        constexpr NSUInteger CompareFunctionEqual = 2;
		constexpr NSUInteger CompareFunctionLessEqual = 3;
		constexpr NSUInteger CompareFunctionAlways = 7;
		constexpr NSUInteger ColorWriteMaskNone = 0;
		constexpr NSUInteger ColorWriteMaskAll = 0x0f;
        constexpr NSUInteger ColorWriteMaskRGB = 0x0e;
        constexpr NSUInteger ColorWriteMaskRed = 0x08;
        constexpr NSUInteger BlendFactorZero = 0;
        constexpr NSUInteger BlendFactorOneMinusSourceColor = 3;
		constexpr NSUInteger BlendFactorOne = 1;
		constexpr NSUInteger BlendFactorSourceAlpha = 4;
		constexpr NSUInteger BlendFactorOneMinusSourceAlpha = 5;
        constexpr NSUInteger BlendFactorOneMinusDestinationAlpha = 9;
		constexpr NSUInteger BlendOperationAdd = 0;
        constexpr NSUInteger BlendOperationReverseSubtract = 2;
		constexpr NSUInteger SamplerFilterLinear = 1;
		constexpr NSUInteger SamplerAddressModeRepeat = 2;
		constexpr NSUInteger CommandBufferStatusError = 5;
        constexpr NSUInteger GPUFamilyApple4 = 1004;
        constexpr NSUInteger GPUFamilyApple7 = 1007;
        constexpr NSUInteger GPUFamilyMac1 = 2001;
        constexpr NSUInteger GPUFamilyMac2 = 2002;
        constexpr NSUInteger FeatureSetMacGPUFamily1Version1 = 10000;

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

        // Apple requires both the vertex [[invariant]] attribute and this option.
        // Keep older devices on their existing language path; OIT must check this
        // capability before relying on exact capture/replay depth equality.
        // https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance
        bool SupportsPositionInvariance(ObjCObject options)
        {
            return options && Send<ObjCBool, ObjCSelector>(options, "respondsToSelector:",
                Runtime().RegisterSelector("setPreserveInvariance:"));
        }
        ObjCObject PositionInvariantCompileOptions()
        {
            const auto options = Send<ObjCObject>(GetClass("MTLCompileOptions"), "new");
            if (SupportsPositionInvariance(options))
            {
                // MTLLanguageVersion2_1, encoded as major << 16 | minor.
                constexpr NSUInteger LanguageVersion2_1 = (2u << 16) | 1u;
                Send<void, NSUInteger>(options, "setLanguageVersion:", LanguageVersion2_1);
                Send<void, ObjCBool>(options, "setPreserveInvariance:", ObjCYes);
                Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
            }
            return options;
        }
        ObjCObject CompilePositionInvariantLibrary(ObjCObject device, const std::string& source, ObjCObject* error)
        {
            const auto options = PositionInvariantCompileOptions();
            const auto library = Evans::Renderer::CompileCachedMetalLibrary(device,MakeString(source.c_str()),options,error);
            ReleaseObject(options);
            return library;
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

	}

	struct MetalRHI::Impl
	{
		struct BufferResource
		{
			BufferDesc Description;
			bool NaniteGeometry=false;
			std::vector<std::uint8_t> Bytes;
			ObjCObject Buffer = nullptr;
		};

		struct TextureResource
		{
			TextureDesc Description;
			ObjCObject Texture = nullptr;
		};

        enum class SamplePhase : std::size_t { Color, Depth, Material, Velocity, Restore, Count };
        static constexpr std::size_t SampleColor = 0, SampleVelocity = 1, SampleMaterialFirst = 2;
        struct MultisampleTargets
        {
            std::array<ObjCObject, SampleMaterialFirst + GBufferAttachmentCount> Colors {};
            ObjCObject Depth = nullptr;
            int Width = 0, Height = 0;
            NSUInteger Samples = 1, Format = PixelFormatRGBA16Float;
        } Multisample;
        struct SamplePipelines
        {
            std::array<ObjCObject, static_cast<std::size_t>(SamplePhase::Count)> Resolve {};
            std::array<ObjCObject, 3> Deferred {};
            std::array<std::array<ObjCObject, 3>, StandaloneShadingTechniqueCount> Shading {};
            ObjCObject Velocity = nullptr;
        };
        std::unordered_map<std::uint32_t, SamplePipelines> SampleGroups;
        std::unordered_map<std::string, ObjCObject> NamedTechniqueLibraries;
        ObjCObject SampleDepthWriteState = nullptr;
        static constexpr NSUInteger ContactTextureIndex = 15;
        MetalContactShadows ContactProducer;
        bool ContactSupported = false, ContactDepthWritten = false, ContactGenerated = false, ContactReceiverActive = false;
        Techniques::ContactShadowSettings FrameContact {};
        std::unordered_map<std::string, ObjCObject> ContactLibraries;
        std::unordered_map<std::uint32_t, ObjCObject> ContactDepthPipelines, ContactSurfacePipelines, ContactLightingPipelines;

        MetalForwardPlus ForwardPlusProducer;
        MetalForwardPlus TiledProducer {TiledLightShader::TiledDeferred};
        bool TiledActive = false, TiledSupported = false;
        MetalForwardPlus& TileProducer() { return TiledActive ? TiledProducer : ForwardPlusProducer; }
        MetalClusteredForward ClusteredProducer;
        bool ClusteredActive = false, ClusteredSupported = false;
        LightListFrameState LightListFrame;
        bool ForwardPlusSupported = false, LightListsActive = false, ForwardPlusDepthWritten = false;
        ObjCObject ForwardPlusSceneBuffer = nullptr, ForwardPlusEmptyBuffer = nullptr;
        std::unordered_map<std::uint32_t, ObjCObject> ForwardPlusLibraries, ForwardPlusPipelines;
        std::unordered_map<std::uint32_t, ObjCObject> TiledLightingLibraries, TiledLightingPipelines;

        bool UsesForwardPlusMaterial() const
        {
            return LightListsActive && (CurrentPass.Type == PassType::Color ||
                CurrentPass.Type == PassType::EnvironmentCapture || CurrentPass.Type == PassType::PlanarCapture);
        }
        std::uint32_t ForwardPlusFilter() const
        {
            return static_cast<std::uint32_t>(CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap ?
                FrameShadowFilter.Mode : Techniques::ShadowFilterMode::Hard);
        }
        ObjCObject ForwardPlusLibrary(NSUInteger samples)
        {
            constexpr std::uint32_t ContactKeyBit = 1u << 4, SampleContactKeyBit = 1u << 5, ClusteredKeyBit = 1u << 6, TiledKeyBit = 1u << 7;
            const bool contact = UsesSurfaceContact();
            const auto key = ForwardPlusFilter() | (contact ? ContactKeyBit : 0u) |
                (contact && samples > 1 ? SampleContactKeyBit : 0u) | (ClusteredActive ? ClusteredKeyBit : 0u) | (TiledActive ? TiledKeyBit : 0u);
            auto& library = ForwardPlusLibraries[key];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", ClusteredActive ? "ClusteredForward.metal" : TiledActive ? "TiledDeferred.metal" : "ForwardPlus.metal", source, LastError)) return nullptr;
            const std::string prefix = ClusteredActive ? "EVANS_CLUSTERED_FORWARD" : TiledActive ? "EVANS_TILED_DEFERRED" : "EVANS_FORWARD_PLUS";
            source.insert(0, "#define " + prefix + "_SURFACE 1\n#define " + prefix + "_FILTER " +
                std::to_string(ForwardPlusFilter()) + "\n#define EVANS_CONTACT_SHADOWS " + std::to_string(contact ? 1 : 0) +
                "\n#define EVANS_CONTACT_MULTISAMPLE " + std::to_string(contact && samples > 1 ? 1 : 0) + "\n");
            ObjCObject options = PositionInvariantCompileOptions();
            Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
            ReleaseObject(options);
            if (!library) Fail("Metal Forward+ material compilation failed: " + ErrorDescription(error));
            return library;
        }
        ObjCObject ForwardPlusPipeline(bool hdr, NSUInteger samples, bool instanced, bool depth)
        {
            if (BoundPipeline.Shading == Techniques::ShadingModel::Gouraud)
            { Fail("Metal Forward+ pixel lists cannot supply Gouraud vertex lighting"); return nullptr; }
            constexpr std::uint32_t SampleKeyShift = 4;
            constexpr std::uint32_t HdrKeyBit = 1u << 12, ColorKeyBit = 1u << 13, BlendKeyBit = 1u << 14;
            constexpr std::uint32_t WeightedKeyBit = 1u << 15, InstanceKeyBit = 1u << 16, DepthKeyBit = 1u << 17,
                ContactKeyBit = 1u << 18, ClusteredKeyBit = 1u << 19, TiledKeyBit = 1u << 20;
            const auto key = ForwardPlusFilter() | (static_cast<std::uint32_t>(samples) << SampleKeyShift) |
                (hdr ? HdrKeyBit : 0u) | (BoundPipeline.ColorWrite ? ColorKeyBit : 0u) |
                (BoundPipeline.BlendEnabled ? BlendKeyBit : 0u) | (WeightedPass ? WeightedKeyBit : 0u) |
                (instanced ? InstanceKeyBit : 0u) | (depth ? DepthKeyBit : 0u) | (UsesSurfaceContact() ? ContactKeyBit : 0u) |
                (ClusteredActive ? ClusteredKeyBit : 0u) | (TiledActive ? TiledKeyBit : 0u);
            auto& pipeline = ForwardPlusPipelines[key];
            if (pipeline) return pipeline;
            auto library = ForwardPlusLibrary(samples);
            if (!library) return nullptr;
            if (instanced && !InstanceLibrary)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", "InstancedRendering.metal", source, LastError)) return nullptr;
                ObjCObject error = nullptr;
                InstanceLibrary = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
                if (!InstanceLibrary) { Fail("Metal instance shader compilation failed: " + ErrorDescription(error)); return nullptr; }
            }
            // Instance transforms reuse the independent instancing vertex stage;
            // both its depth and lighting draws link the Forward+ fragment ABI.
            pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, BoundPipeline.BlendEnabled,
                hdr, samples, WeightedPass, library, instanced ? InstanceLibrary : library,
                instanced ? "InstanceForwardVertex" : "EvansVertexMain",
                depth ? "ForwardDepthFragment" : WeightedPass ? "EvansWeightedFragment" : "EvansFragmentMain", depth);
            return pipeline;
        }
        bool UploadForwardPlusScene()
        {
            const auto& lights = LightListFrame.Description().Lights->GetShaderLights();
            if (!ForwardPlusSceneBuffer)
                ForwardPlusSceneBuffer = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:",
                    MaximumForwardPlusLights * sizeof(ShaderLight), 0);
            if (!ForwardPlusEmptyBuffer)
            {
                ForwardPlusEmptyBuffer = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", sizeof(ShaderLight), 0);
                if (ForwardPlusEmptyBuffer) std::memset(Send<void*>(ForwardPlusEmptyBuffer, "contents"), 0, sizeof(ShaderLight));
            }
            if (!ForwardPlusSceneBuffer || !ForwardPlusEmptyBuffer) return Fail("Metal Forward+ scene buffer allocation failed");
            auto* destination = Send<void*>(ForwardPlusSceneBuffer, "contents");
            // BeginFrame has retired the previous submission before this shared
            // CPU upload. Captures and tile lists retain the same index space.
            if (lights.empty()) std::memset(destination, 0, sizeof(ShaderLight));
            else std::memcpy(destination, lights.data(), lights.size() * sizeof(ShaderLight));
            return true;
        }
        bool BindForwardPlusSurface()
        {
            const bool mainCamera = CurrentPass.Type == PassType::Color && !CurrentPass.Target;
            const auto selection = !mainCamera ? LightListSelection::FullScene :
                BoundPipeline.BlendEnabled ? LightListSelection::Transparent : LightListSelection::Opaque;
            LightListSurfaceConstants constants;
            if (!LightListFrame.Surface(BoundPipeline.Lighting, BoundViewProjection, selection, constants, LastError)) return false;
            auto headers = mainCamera ? (ClusteredActive ? ClusteredProducer.Headers() : TileProducer().Headers()) : ForwardPlusEmptyBuffer;
            auto indices = mainCamera ? (ClusteredActive ? ClusteredProducer.Indices() : TileProducer().Indices()) : ForwardPlusEmptyBuffer;
            if (!ForwardPlusSceneBuffer || !headers || !indices) return Fail("Metal Forward+ light storage is unavailable");
            Send<void, const void*, NSUInteger, NSUInteger>(RenderEncoder,
                "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), ForwardPlusSelectionIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder,
                "setFragmentBuffer:offset:atIndex:", ForwardPlusSceneBuffer, 0, ForwardPlusSceneIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setFragmentBuffer:offset:atIndex:", headers, 0, ForwardPlusHeaderIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setFragmentBuffer:offset:atIndex:", indices, 0, ForwardPlusLightIndex);
            if (ForwardPlusFilter() <= static_cast<std::uint32_t>(Techniques::ShadowFilterMode::PCF))
            {
                PCSSConstants filtering {};
                filtering.EmitterAndLimits = BuildCascadedShadowConstants({}, FrameShadowFilter).Filtering;
                Send<void, const void*, NSUInteger, NSUInteger>(RenderEncoder,
                    "setFragmentBytes:length:atIndex:", &filtering, sizeof(filtering), PCSSConstantBufferIndex);
            }
            return true;
        }
        bool RetireForwardPlusAfterDiscard()
        {
            // The RHI released its unsubmitted command buffer. Retire both
            // producers before shared scene uploads, including a pipeline switch.
            const auto retire = [&](auto& producer, bool& supported)
            {
                if (!supported) return true;
                std::string error;
                if (producer.AbortFrame(error)) return true;
                if (producer.BeginFrame(error)) return producer.AbortFrame(LastError);
                if (!producer.Shutdown(error)) return Fail(error);
                supported = producer.Initialize(Device, error);
                return supported || Fail(error);
            };
            return retire(ForwardPlusProducer, ForwardPlusSupported) && retire(TiledProducer, TiledSupported) && retire(ClusteredProducer, ClusteredSupported);
        }

        bool UsesSurfaceContact() const
        {
            return ContactReceiverActive && CurrentPass.Type == PassType::Color && !CurrentPass.Target &&
                !BoundPipeline.BlendEnabled && !BoundPipeline.OutlineExtrusion && BoundPipeline.Fill == FillMode::Solid;
        }
        ObjCObject ContactLibrary(bool deferred, bool pcss, bool vsm, NSUInteger samples)
        {
            const auto model = static_cast<std::uint32_t>(BoundPipeline.Shading);
            const auto program = deferred ? (samples > 1 ? 6u : 5u) : model;
            const std::string filename = vsm ? "VSM.metal" : pcss ? "PCSS.metal" : deferred ? "Deferred.metal" :
                std::string(ShadingTechniqueFile(BoundPipeline.Shading, true)) + ".metal";
            const std::string key = filename + ":" + std::to_string(program) + (samples > 1 ? ":samples" : ":single");
            auto& library = ContactLibraries[key];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", filename.c_str(), source, LastError)) return nullptr;
            std::string definitions = "#define EVANS_CONTACT_SHADOWS 1\n#define EVANS_CONTACT_MULTISAMPLE " +
                std::to_string(samples > 1 ? 1 : 0) + "\n";
            if (vsm || pcss) definitions += std::string("#define ") + (vsm ? "EVANS_VSM_PROGRAM " : "EVANS_PCSS_PROGRAM ") +
                std::to_string(program) + "\n";
            source.insert(0, definitions);
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal contact receiver compilation failed: " + ErrorDescription(error));
            return library;
        }
        ObjCObject ContactDepthPipeline(bool instanced, NSUInteger samples)
        {
            constexpr std::uint32_t InstanceBit = 1u << 8;
            auto& pipeline = ContactDepthPipelines[std::uint32_t(samples) | (instanced ? InstanceBit : 0u)];
            if (!pipeline) pipeline = CreateTechniquePipeline("ContactShadows.metal",
                instanced ? "ContactDepthInstanceVertex" : "ContactDepthVertex", "ContactDepthFragment", samples, {}, true);
            // Match ordinary mesh compilation for bit-identical raster coverage.
            // The compute producer compiles this source separately with safe math.
            return pipeline;
        }
        ObjCObject ContactSurfacePipeline(bool hdr, NSUInteger samples)
        {
            const bool pcss = UsesSurfacePCSS(), vsm = UsesSurfaceVSM();
            constexpr std::uint32_t SampleShift = 4, HdrBit = 1u << 8, WriteBit = 1u << 9,
                PCSSBit = 1u << 10, VSMBit = 1u << 11;
            auto& pipeline = ContactSurfacePipelines[static_cast<std::uint32_t>(BoundPipeline.Shading) |
                (std::uint32_t(samples) << SampleShift) | (hdr ? HdrBit : 0u) |
                (BoundPipeline.ColorWrite ? WriteBit : 0u) | (pcss ? PCSSBit : 0u) | (vsm ? VSMBit : 0u)];
            if (!pipeline)
            {
                const auto library = ContactLibrary(false, pcss, vsm, samples);
                if (!library) return nullptr;
                pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, false, hdr, samples, false, library);
            }
            return pipeline;
        }
        ObjCObject TiledLightingPipeline()
        {
            constexpr std::uint32_t FilterShift = 8, ContactBit = 1u << 12;
            const auto filter = static_cast<std::uint32_t>(FrameShadowFilter.Mode);
            const auto key = FrameSampleCount | (filter << FilterShift) | (ContactReceiverActive ? ContactBit : 0u);
            auto& pipeline = TiledLightingPipelines[key];
            if (pipeline) return pipeline;
            auto& library = TiledLightingLibraries[key];
            if (!library)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", "TiledDeferred.metal", source, LastError)) return nullptr;
                source.insert(0, "#define EVANS_TILED_DEFERRED_SURFACE 1\n#define EVANS_TILED_DEFERRED_LIGHTING 1\n"
                    "#define EVANS_TILED_DEFERRED_FILTER " + std::to_string(filter) +
                    "\n#define EVANS_CONTACT_SHADOWS " + std::to_string(ContactReceiverActive ? 1 : 0) +
                    "\n#define EVANS_CONTACT_MULTISAMPLE " + std::to_string(ContactReceiverActive && FrameSampleCount > 1 ? 1 : 0) + "\n");
                ObjCObject options = Send<ObjCObject>(GetClass("MTLCompileOptions"), "new");
                Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
                ObjCObject error = nullptr;
                library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
                ReleaseObject(options);
                if (!library) { Fail("Metal tiled lighting compilation failed: " + ErrorDescription(error)); return nullptr; }
            }
            pipeline = CreateTechniquePipeline("TiledDeferred.metal", "TiledLightingVertex",
                FrameSampleCount > 1 ? "TiledLightingSampleFragment" : "TiledLightingFragment",
                FrameSampleCount, {PixelFormatRGBA16Float}, false, ColorWriteMaskAll, false, library);
            return pipeline;
        }

        ObjCObject ContactLightingPipeline(bool pcss, bool vsm, NSUInteger samples)
        {
            constexpr std::uint32_t PCSSBit = 1u << 8, VSMBit = 1u << 9;
            auto& pipeline = ContactLightingPipelines[std::uint32_t(samples) | (pcss ? PCSSBit : 0u) | (vsm ? VSMBit : 0u)];
            if (!pipeline)
            {
                const auto library = ContactLibrary(true, pcss, vsm, samples);
                if (!library) return nullptr;
                pipeline = CreateTechniquePipeline(vsm ? "VSM.metal" : pcss ? "PCSS.metal" : "Deferred.metal",
                    "DeferredLightingVertex", samples > 1 ? "DeferredLightingSampleFragment" : "DeferredLightingFragment",
                    samples, {PixelFormatRGBA16Float}, false, ColorWriteMaskAll, false, library);
            }
            return pipeline;
        }
        std::uint32_t SupportedSampleCounts = 1, FrameSampleCount = 1;
        struct WeightedPipelineGroup
        {
            std::array<ObjCObject, StandaloneShadingTechniqueCount> Surfaces {};
            ObjCObject Resolve = nullptr;
        };
        std::unordered_map<std::uint32_t, WeightedPipelineGroup> WeightedGroups;
        std::array<ObjCObject, 2> WeightedTextures {};
        std::uint32_t WeightedSampleCounts = 0, WeightedTargetSamples = 0;
        int WeightedWidth = 0, WeightedHeight = 0;
        Techniques::TransparencySettings FrameTransparency {};
        bool WeightedActive = false, WeightedPass = false, WeightedPrepared = false;
        bool WeightedAccumulated = false, WeightedResolved = false;
        bool SamplesOpaqueResolved = false, SamplesVelocityResolved = false;
        bool SamplesColorRestored = false, SamplesColorResolved = false, SamplesTransparentWritten = false;

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

        bool QueryDeferredSupport() const
        {
            // Five RGBA16F colors need 40 bytes/pixel, beyond Apple2/3's 32-byte
            // imageblock limit. Apple4+ and Mac families support this layout.
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            const auto familySelector = Runtime().RegisterSelector("supportsFamily:");
            if (Send<ObjCBool, ObjCSelector>(Device, "respondsToSelector:", familySelector))
                return Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyApple4) ||
                    Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyMac1) ||
                    Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyMac2);
            const auto featureSelector = Runtime().RegisterSelector("supportsFeatureSet:");
            return Send<ObjCBool, ObjCSelector>(Device, "respondsToSelector:", featureSelector) &&
                Send<ObjCBool, NSUInteger>(Device, "supportsFeatureSet:", FeatureSetMacGPUFamily1Version1);
        }

        void ResetDeferredFrame()
        {
            LightListFrame.Reset();
            LightListsActive = ForwardPlusDepthWritten = false;
            FrameContact = {};
            ContactDepthWritten = ContactGenerated = ContactReceiverActive = false;
            DeferredActive = GBufferWritten = DeferredResolved = false;
            AmbientOcclusionActive = AmbientOcclusionResolved = ForwardColorWritten = false;
            ReflectionActive = ReflectionResolved = false;
            FrameReflections = {};
            FrameEnvironment = {}; EnvironmentFacesWritten = 0;
            EnvironmentActive = EnvironmentReady = false;
            FramePlanar = {}; PlanarActive = PlanarWritten = PlanarReady = false;
            FrameReflectionShading = Techniques::ShadingModel::Phong;
            DepthOfFieldActive = DepthOfFieldResolved = false;
            MotionBlurActive = VelocityCaptured = MotionBlurResolved = false;
            VelocityActive = TemporalActive = TemporalResolved = false;
            GI.Begin(); Probes.Begin();
            GIActive = EarlyTemporalActive = ProbeActive = false;
            FogActive = FogResolved = false;
            LightShaftActive = LightShaftResolved = false;
            FrameLightShafts = {};
            FrameFog = {};
            FrameVolumeLighting = {};
            FogConstants = {};
            WeightedActive = WeightedPass = WeightedPrepared = WeightedAccumulated = WeightedResolved = false;
            FrameTransparency = {};
            SamplesOpaqueResolved = SamplesVelocityResolved = false;
            SamplesColorRestored = SamplesColorResolved = SamplesTransparentWritten = false;
            FrameSampleCount = 1;
            FrameAmbientOcclusionSettings = {};
            FramePostProcessSettings = {};
            DeferredBackground = {};
        }

        void DestroyDeferredTargets()
        {
            for (auto& texture : GBufferTextures) { ReleaseObject(texture); texture = nullptr; }
            ResetDeferredFrame();
        }

        bool EnsureDeferredPipelines()
        {
            if (DeferredLightingPipeline) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "Deferred.metal", source, LastError)) return false;
            ObjCObject error = nullptr;
            ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) return Fail("Metal deferred shader compilation failed: " + ErrorDescription(error));
            const std::array<const char*, 4> entryNames {
                "DeferredGeometryVertex", "DeferredGeometryFragment", "DeferredLightingVertex", "DeferredLightingFragment" };
            std::array<ObjCObject, 4> functions{};
            bool valid = true;
            for (std::size_t index = 0; index < functions.size(); ++index)
            {
                functions[index] = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString(entryNames[index]));
                valid = valid && functions[index] != nullptr;
            }
            std::array<ObjCObject, 3> pipelines{};
            if (!valid) Fail("Metal deferred shader entry point is missing");
            for (std::size_t index = 0; valid && index < pipelines.size(); ++index)
            {
                const bool lighting = index == 2;
                const bool colorWrite = index != 1;
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
                Send<void, ObjCObject>(descriptor, "setVertexFunction:", functions[lighting ? 2 : 0]);
                Send<void, ObjCObject>(descriptor, "setFragmentFunction:", functions[lighting ? 3 : 1]);
                if (!lighting) Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                const std::size_t attachmentCount = lighting ? 1 : GBufferAttachmentCount;
                for (std::size_t color = 0; color < attachmentCount; ++color)
                {
                    ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", color);
                    Send<void, NSUInteger>(attachment, "setPixelFormat:", PixelFormatRGBA16Float);
                    Send<void, NSUInteger>(attachment, "setWriteMask:", colorWrite ? ColorWriteMaskAll : ColorWriteMaskNone);
                    Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCNo);
                }
                error = nullptr;
                pipelines[index] = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
                ReleaseObject(descriptor);
                valid = pipelines[index] != nullptr;
                if (!valid) Fail("Metal deferred pipeline creation failed: " + ErrorDescription(error));
            }
            for (auto function : functions) ReleaseObject(function);
            ReleaseObject(library);
            if (!valid)
            {
                for (auto pipeline : pipelines) ReleaseObject(pipeline);
                return false;
            }
            DeferredGeometryPipelines = {pipelines[0], pipelines[1]};
            DeferredLightingPipeline = pipelines[2];
            return true;
        }

        bool EnsureDeferredResources()
        {
            if (!DeferredSupported || !HdrActive)
                return Fail("Metal deferred shading requires HDR and five RGBA16F color attachments");
            if (!EnsureDeferredPipelines()) return false;
            if (GBufferTextures[0]) return true;
            // Commit the new set only after every allocation succeeds.
            std::array<ObjCObject, GBufferAttachmentCount> textures{};
            for (auto& texture : textures)
            {
                texture = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal G-buffer");
                if (!texture)
                {
                    for (auto allocated : textures) ReleaseObject(allocated);
                    return false;
                }
            }
            GBufferTextures = textures;
            return true;
        }

        void DestroyAmbientOcclusionTargets()
        {
            for (auto& texture : AmbientOcclusionTextures) { ReleaseObject(texture); texture = nullptr; }
            ReleaseObject(SaoDepthHierarchy); SaoDepthHierarchy = nullptr;
            ReleaseObject(SaoDepthScratch); SaoDepthScratch = nullptr;
        }

        bool EnsureSaoDepthTargets()
        {
            if (SaoDepthHierarchy && SaoDepthScratch) return true;
            if (!FloatDepthTargetSupported) return Fail("Metal SAO requires an R32Float render target");
            const auto levels = AmbientOcclusionDepthMipCount(CreateInfo.Width, CreateInfo.Height);
            ObjCObject hierarchy = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatR32Float,
                TextureUsageShaderRead, "Evans Metal SAO depth hierarchy", levels);
            ObjCObject scratch = hierarchy ? CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatR32Float,
                TextureUsageRenderTarget, "Evans Metal SAO depth scratch") : nullptr;
            if (!hierarchy || !scratch)
            {
                ReleaseObject(hierarchy); ReleaseObject(scratch);
                return false;
            }
            ReleaseObject(SaoDepthHierarchy); ReleaseObject(SaoDepthScratch);
            SaoDepthHierarchy = hierarchy;
            SaoDepthScratch = scratch;
            return true;
        }

        #include "Renderer/Metal/MetalRayScene.inl"
        #include "Renderer/Metal/MetalRayRadiance.inl"
    #include "Renderer/Metal/MetalRayReflections.inl"
    #include "Renderer/Metal/MetalHybridRendering.inl"
    #include "Renderer/Metal/MetalSubsurfaceScattering.inl"
    #include "Renderer/Metal/MetalHairFur.inl"
    #include "Renderer/Metal/MetalTemporalEffects.inl"
    #include "Renderer/Metal/MetalParallaxOcclusionMapping.inl"
    #include "Renderer/Metal/MetalDisplacementMapping.inl"
    #include "Renderer/Metal/MetalTessellation.inl"
    #include "Renderer/VirtualGeometry/Metal/MetalVirtualGeometryCache.inl"
    #include "Renderer/VirtualGeometry/Metal/MetalVirtualGeometry.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteCache.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteGeometry.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteAsync.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteCacheAsync.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteFrameVisibility.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteDraws.inl"
#include "Renderer/Metal/MetalTemporalSuperResolution.inl"
#include "Renderer/VirtualShadowMaps/Metal/MetalVirtualShadowMaps.inl"
#include "Renderer/VirtualShadowMaps/Metal/MetalVirtualShadowAtlas.inl"
#include "Renderer/Lumen/Metal/MetalLumenSurface.inl"
#include "Renderer/Lumen/Metal/MetalLumenCaptureAsync.inl"
#include "Renderer/Lumen/Metal/MetalLumenSurfaceLookup.inl"
#include "Renderer/Lumen/Metal/MetalLumenDistanceTrace.inl"
#include "Renderer/Lumen/Metal/MetalLumenGlobalDistance.inl"
#include "Renderer/Lumen/Metal/MetalLumenGlobalDistanceAsync.inl"
#include "Renderer/Lumen/Metal/MetalLumenSurfaceLighting.inl"
#include "Renderer/Lumen/Metal/MetalLumenLightingAsync.inl"
#include "Renderer/Lumen/Metal/MetalLumenRadianceTrace.inl"
#include "Renderer/Lumen/Metal/MetalLumenProbeIntegration.inl"
#include "Renderer/Lumen/Metal/MetalLumenProbeGather.inl"
#include "Renderer/Lumen/Metal/MetalLumenTranslucency.inl"
#include "Renderer/Lumen/Metal/MetalLumenHairReceiver.inl"
#include "Renderer/Metal/MetalRayTracedShadows.inl"
#include "Renderer/Metal/MetalPCF.inl"
#include "Renderer/Metal/MetalMaterials.inl"

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
    bool UsesSurfacePCF() const
    { return UsesDedicatedPCF() && !FrameAreaLights && CurrentPass.Type==PassType::Color && BoundPipeline.SamplesShadowMap; }

#include "Renderer/Lumen/Metal/MetalLumenTemporal.inl"
#include "Renderer/Lumen/Metal/MetalLumenReflections.inl"
#include "Renderer/VirtualShadowMaps/Metal/MetalVirtualShadowReceivers.inl"
#include "Renderer/VirtualShadowMaps/Metal/MetalVirtualShadowLookup.inl"
    #include "Renderer/Metal/MetalRayGI.inl"
    #include "Renderer/Metal/MetalVoxelGI.inl"
    #include "Renderer/Metal/MetalPathTracing.inl"
    #include "Renderer/Metal/MetalGPUDriven.inl"
    #include "Renderer/Metal/MetalVisibilityBuffer.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteMaterials.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteFrameRaster.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteVisibilityGeometry.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteInstances.inl"
    #include "Renderer/Nanite/Metal/MetalNaniteHardware.inl"
    #include "Renderer/MeshShaders/Metal/MetalMeshShaders.inl"
    #include "Renderer/Profiling/Metal/MetalFrameTiming.inl"

        bool EnsureAmbientOcclusionResources()
        {
            const bool rayTraced = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::RayTraced;
            if (rayTraced && !EnsureRayScene()) return false;
            const bool horizonBased = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::HBAO;
            const bool scalable = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::SAO;
            const bool groundTruth = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::GTAO;
            auto& cachedPipelines = rayTraced ? RayAOPipelines : groundTruth ? GtaoPipelines : scalable ? SaoPipelines : horizonBased ? HbaoPipelines : SsaoPipelines;
            if (!cachedPipelines[0])
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", rayTraced ? "RayTracedAO.metal" : groundTruth ? "GTAO.metal" : scalable ? "SAO.metal" : horizonBased ? "HBAO.metal" : "SSAO.metal", source, LastError)) return false;
                ObjCObject error = nullptr;
                ObjCObject options = Send<ObjCObject>(GetClass("MTLCompileOptions"), "new");
                if (rayTraced) Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
                ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
                ReleaseObject(options);
                if (!library) return Fail("Metal AO shader compilation failed: " + ErrorDescription(error));
                ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString(rayTraced ? "VSMain" : groundTruth ? "GTAOVertexMain" : scalable ? "SAOVertexMain" : horizonBased ? "HBAOVertexMain" : "SSAOVertexMain"));
                ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString(rayTraced ? "PSMain" : groundTruth ? "GTAOFragmentMain" : scalable ? "SAOFragmentMain" : horizonBased ? "HBAOFragmentMain" : "SSAOFragmentMain"));
                ObjCObject depthFragment = scalable ? Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString("SAODepthFragmentMain")) : nullptr;
                std::array<ObjCObject, 4> pipelines{};
                bool valid = vertex && fragment && (!scalable || depthFragment);
                if (!valid) Fail("Metal AO shader entry point is missing");
                const std::size_t pipelineCount = scalable ? pipelines.size() : 2;
                for (std::size_t index = 0; valid && index < pipelineCount; ++index)
                {
                    const bool composite = index == 1;
                    const bool depthPrepare = index == 2;
                    const bool depthTarget = index >= 2;
                    ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
                    Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
                    Send<void, ObjCObject>(descriptor, "setFragmentFunction:", depthPrepare ? depthFragment : fragment);
                    ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                    ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                    Send<void, NSUInteger>(attachment, "setPixelFormat:", depthTarget ? PixelFormatR32Float : PixelFormatRGBA16Float);
                    Send<void, NSUInteger>(attachment, "setWriteMask:", composite ? ColorWriteMaskRGB : ColorWriteMaskAll);
                    Send<void, ObjCBool>(attachment, "setBlendingEnabled:", composite ? ObjCYes : ObjCNo);
                    if (composite)
                    {
                        Send<void, NSUInteger>(attachment, "setSourceRGBBlendFactor:", BlendFactorOne);
                        Send<void, NSUInteger>(attachment, "setDestinationRGBBlendFactor:", BlendFactorOne);
                        Send<void, NSUInteger>(attachment, "setRgbBlendOperation:", BlendOperationReverseSubtract);
                    }
                    error = nullptr;
                    pipelines[index] = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
                    ReleaseObject(descriptor);
                    valid = pipelines[index] != nullptr;
                    if (!valid) Fail("Metal AO pipeline creation failed: " + ErrorDescription(error));
                }
                ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(depthFragment); ReleaseObject(library);
                if (!valid)
                {
                    for (auto pipeline : pipelines) ReleaseObject(pipeline);
                    return false;
                }
                cachedPipelines = pipelines;
            }
            if (AmbientOcclusionTextures[0]) return !scalable || EnsureSaoDepthTargets();
            std::array<ObjCObject, 2> textures{};
            for (auto& texture : textures)
            {
                texture = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal AO visibility");
                if (!texture)
                {
                    for (auto allocated : textures) ReleaseObject(allocated);
                    return false;
                }
            }
            AmbientOcclusionTextures = textures;
            return !scalable || EnsureSaoDepthTargets();
        }

        std::uint32_t SamplePipelineKey() const
        {
            constexpr std::uint32_t HdrFormatBit = 1U << 8U;
            return FrameSampleCount | (HdrActive ? HdrFormatBit : 0U);
        }

        void DestroyWeightedTargets()
        {
            for (auto& texture : WeightedTextures) { ReleaseObject(texture); texture = nullptr; }
            WeightedWidth = WeightedHeight = 0;
            WeightedTargetSamples = 0;
        }

        bool SupportsFloat32Multisampling() const
        {
            // RGBA32F blending is available earlier than RGBA32F MSAA. Avoid
            // issuing an invalid texture allocation on Apple2 through Apple6.
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            const auto familySelector = Runtime().RegisterSelector("supportsFamily:");
            if (Send<ObjCBool, ObjCSelector>(Device, "respondsToSelector:", familySelector))
                return Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyApple7) ||
                    Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyMac1) ||
                    Send<ObjCBool, NSUInteger>(Device, "supportsFamily:", GPUFamilyMac2);
            const auto featureSelector = Runtime().RegisterSelector("supportsFeatureSet:");
            return Send<ObjCBool, ObjCSelector>(Device, "respondsToSelector:", featureSelector) &&
                Send<ObjCBool, NSUInteger>(Device, "supportsFeatureSet:", FeatureSetMacGPUFamily1Version1);
        }

        std::uint32_t ProbeWeightedSampleCounts()
        {
            const auto previousError = LastError;
            const bool multisample = SupportsFloat32Multisampling();
            std::uint32_t supported = 0;
            // Probe both the sampleable formats and the two different blend
            // equations. Texture allocation alone does not prove blend support.
            for (std::uint32_t samples : {1U, 2U, 4U, 8U})
            {
                if (!(SupportedSampleCounts & samples) || (samples > 1 && !multisample)) continue;
                bool valid = true;
                for (auto format : {PixelFormatRGBA32Float, PixelFormatR32Float})
                {
                    ObjCObject texture = CreateNativeTexture(1, 1, format,
                        TextureUsageShaderRead | TextureUsageRenderTarget, "Evans weighted format probe", 1, samples);
                    valid &= texture != nullptr;
                    ReleaseObject(texture);
                }
                if (!valid) continue;
                auto& group = WeightedGroups[samples];
                // Initialize calls this while the Phong library is selected.
                group.Surfaces[static_cast<std::size_t>(Techniques::ShadingModel::Phong)] =
                    CreatePipeline(PassType::Color, true, true, true, samples, true);
                group.Resolve = CreateTechniquePipeline("WeightedBlendedOIT.metal", "TransparencyResolveVertex",
                    samples > 1 ? "TransparencyResolveSamplesFragment" : "TransparencyResolveFragment",
                    samples, {PixelFormatRGBA16Float}, false, ColorWriteMaskAll, true);
                if (group.Surfaces[static_cast<std::size_t>(Techniques::ShadingModel::Phong)] && group.Resolve)
                    supported |= samples;
            }
            LastError = previousError;
            return supported;
        }

        bool EnsureWeightedResources()
        {
            if (!HdrActive || !HdrTextures[0] || !(WeightedSampleCounts & FrameSampleCount))
                return Fail("Metal weighted transparency requires supported float blending and HDR samples");
            if (WeightedTextures[0] && WeightedWidth == CreateInfo.Width && WeightedHeight == CreateInfo.Height &&
                WeightedTargetSamples == FrameSampleCount) return true;
            std::array<ObjCObject, 2> candidate {};
            const std::array<NSUInteger, 2> formats {PixelFormatRGBA32Float, PixelFormatR32Float};
            for (std::size_t index = 0; index < candidate.size(); ++index)
            {
                candidate[index] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, formats[index],
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans weighted transparency statistics", 1, FrameSampleCount);
                if (!candidate[index])
                {
                    for (auto texture : candidate) ReleaseObject(texture);
                    return false;
                }
            }
            // Present waits for completion; AbortFrame discards its unsubmitted
            // command buffer. Target replacement cannot affect in-flight work.
            DestroyWeightedTargets();
            WeightedTextures = candidate;
            WeightedWidth = CreateInfo.Width; WeightedHeight = CreateInfo.Height;
            WeightedTargetSamples = FrameSampleCount;
            return true;
        }

        ObjCObject WeightedSurfacePipeline(Techniques::ShadingModel model)
        {
            auto& pipeline = WeightedGroups[FrameSampleCount].Surfaces[ShadingTechniqueIndex(model, FrameAreaLights)];
            if (!pipeline) pipeline = CreatePipeline(PassType::Color, true, true, true, FrameSampleCount, true);
            return pipeline;
        }

        std::uint32_t ProbeSampleCounts()
        {
            const std::string previousError = LastError;
            std::uint32_t supported = 1;
            for (std::uint32_t samples : {2U, 4U, 8U})
            {
                if (!Send<ObjCBool, NSUInteger>(Device, "supportsTextureSampleCount:", samples)) continue;
                bool valid = true;
                // Probe the actual sampled, private color/depth formats instead
                // of assuming every format accepts the device's sample counts.
                for (NSUInteger format : {PixelFormatRGBA8UnormSRGB, PixelFormatRGBA16Float, PixelFormatDepth32Float})
                {
                    ObjCObject probe = CreateNativeTexture(1, 1, format, TextureUsageShaderRead | TextureUsageRenderTarget,
                        "Evans MSAA capability probe", 1, samples);
                    valid = valid && probe != nullptr;
                    ReleaseObject(probe);
                }
                if (valid) supported |= samples;
            }
            LastError = previousError;
            return supported;
        }

        void DestroyMultisampleTargets(MultisampleTargets& targets)
        {
            for (auto texture : targets.Colors) ReleaseObject(texture);
            ReleaseObject(targets.Depth);
            targets = {};
        }

        ObjCObject CreateTechniquePipeline(const char* filename, const char* vertexName, const char* fragmentName,
            NSUInteger samples, const std::vector<NSUInteger>& colorFormats, bool depth,
            NSUInteger writeMask = ColorWriteMaskAll, bool sourceOver = false, ObjCObject libraryOverride = nullptr, bool keepFirstSurface = false)
        {
            auto& library = libraryOverride ? libraryOverride : NamedTechniqueLibraries[filename];
            if (!library)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", filename, source, LastError)) return nullptr;
                ObjCObject error = nullptr;
                library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
                if (!library) { Fail("Metal technique shader compilation failed: " + ErrorDescription(error)); return nullptr; }
            }
            ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString(vertexName));
            ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString(fragmentName));
            if (!vertex || !fragment)
            {
                ReleaseObject(vertex); ReleaseObject(fragment);
                Fail("Metal technique shader entry point is missing"); return nullptr;
            }
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
            Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
            Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragment);
            Send<void, NSUInteger>(descriptor, "setSampleCount:", samples);
            if (depth) Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            for (std::size_t index = 0; index < colorFormats.size(); ++index)
            {
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
                Send<void, NSUInteger>(attachment, "setPixelFormat:", colorFormats[index]);
                Send<void, NSUInteger>(attachment, "setWriteMask:", writeMask);
                Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCNo);
                if (sourceOver)
                {
                    Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCYes);
                    Send<void, NSUInteger>(attachment, "setSourceRGBBlendFactor:", BlendFactorSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setDestinationRGBBlendFactor:", BlendFactorOneMinusSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setSourceAlphaBlendFactor:", BlendFactorOne);
                    Send<void, NSUInteger>(attachment, "setDestinationAlphaBlendFactor:", BlendFactorOneMinusSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setRgbBlendOperation:", BlendOperationAdd);
                    Send<void, NSUInteger>(attachment, "setAlphaBlendOperation:", BlendOperationAdd);
                }
                if (keepFirstSurface)
                {
                    Send<void,ObjCBool>(attachment,"setBlendingEnabled:",ObjCYes);
                    Send<void,NSUInteger>(attachment,"setSourceRGBBlendFactor:",BlendFactorOneMinusDestinationAlpha);
                    Send<void,NSUInteger>(attachment,"setDestinationRGBBlendFactor:",BlendFactorOne);
                    Send<void,NSUInteger>(attachment,"setSourceAlphaBlendFactor:",BlendFactorOne);
                    Send<void,NSUInteger>(attachment,"setDestinationAlphaBlendFactor:",BlendFactorZero);
                    Send<void,NSUInteger>(attachment,"setRgbBlendOperation:",BlendOperationAdd);
                    Send<void,NSUInteger>(attachment,"setAlphaBlendOperation:",BlendOperationAdd);
                }
            }
            ObjCObject error = nullptr;
            ObjCObject pipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
            if (!pipeline) Fail("Metal technique pipeline creation failed: " + ErrorDescription(error));
            ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment);
            return pipeline;
        }

        ObjCObject CreateSamplePipeline(const char* filename, const char* vertexName, const char* fragmentName,
            NSUInteger samples, NSUInteger format, std::size_t colorCount, bool depth, bool colorWrite = true)
        {
            return CreateTechniquePipeline(filename, vertexName, fragmentName, samples,
                std::vector<NSUInteger>(colorCount, format), depth, colorWrite ? ColorWriteMaskAll : ColorWriteMaskNone);
        }

        bool EnsureMultisampleResources()
        {
            const bool material = Hair.Active || GIActive || DeferredActive || AmbientOcclusionActive || ReflectionActive;
            const NSUInteger format = HdrActive ? PixelFormatRGBA16Float : PixelFormatRGBA8UnormSRGB;
            if (!Multisample.Depth || Multisample.Width != CreateInfo.Width || Multisample.Height != CreateInfo.Height ||
                Multisample.Samples != FrameSampleCount || Multisample.Format != format ||
                (material && !Multisample.Colors[SampleMaterialFirst]) || (VelocityActive && !Multisample.Colors[SampleVelocity]))
            {
                MultisampleTargets candidate;
                candidate.Width = CreateInfo.Width; candidate.Height = CreateInfo.Height;
                candidate.Samples = FrameSampleCount; candidate.Format = format;
                const auto allocate = [&](NSUInteger pixelFormat, const char* name)
                {
                    return CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, pixelFormat,
                        TextureUsageShaderRead | TextureUsageRenderTarget, name, 1, FrameSampleCount);
                };
                candidate.Depth = allocate(PixelFormatDepth32Float, "Evans MSAA opaque depth");
                candidate.Colors[SampleColor] = allocate(format, "Evans MSAA scene color");
                bool valid = candidate.Depth && candidate.Colors[SampleColor];
                if (VelocityActive)
                {
                    candidate.Colors[SampleVelocity] = allocate(PixelFormatRGBA16Float, "Evans MSAA rigid velocity");
                    valid = valid && candidate.Colors[SampleVelocity];
                }
                if (material) for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
                {
                    candidate.Colors[SampleMaterialFirst + index] = allocate(PixelFormatRGBA16Float, "Evans MSAA material");
                    valid = valid && candidate.Colors[SampleMaterialFirst + index];
                }
                if (!valid) { DestroyMultisampleTargets(candidate); return false; }
                DestroyMultisampleTargets(Multisample);
                Multisample = candidate;
            }
            auto& group = SampleGroups[SamplePipelineKey()];
            if (material)
            {
                for (std::size_t index = 0; index < 2; ++index)
                {
                    if (!group.Deferred[index]) group.Deferred[index] = CreateSamplePipeline("Deferred.metal",
                        "DeferredGeometryVertex", "DeferredGeometryFragment", FrameSampleCount,
                        PixelFormatRGBA16Float, GBufferAttachmentCount, true, index == 0);
                    if (!group.Deferred[index]) return false;
                }
            }
            if (DeferredActive)
            {
                if (!group.Deferred[2]) group.Deferred[2] = CreateSamplePipeline("Deferred.metal", "DeferredLightingVertex",
                    "DeferredLightingSampleFragment", FrameSampleCount, PixelFormatRGBA16Float, 1, false);
                if (!group.Deferred[2]) return false;
            }
            if (VelocityActive)
            {
                if (!group.Velocity) group.Velocity = CreateSamplePipeline("MotionVectors.metal", "MotionVelocityVertex",
                    "MotionVelocityFragment", FrameSampleCount, PixelFormatRGBA16Float, 1, true);
                if (!group.Velocity) return false;
            }
            if (!SampleDepthWriteState) SampleDepthWriteState = CreateDepthState(CompareFunctionAlways, true);
            return SampleDepthWriteState != nullptr;
        }

        bool DrawSampleResolve(SamplePhase phase)
        {
            const bool material = phase == SamplePhase::Material, depthOnly = phase == SamplePhase::Depth;
            const bool velocity = phase == SamplePhase::Velocity, restore = phase == SamplePhase::Restore;
            const std::size_t colorCount = depthOnly ? 0 : material ? GBufferAttachmentCount : 1;
            const NSUInteger format = material || velocity || HdrActive ? PixelFormatRGBA16Float : PixelFormatRGBA8UnormSRGB;
            const auto index = static_cast<std::size_t>(phase);
            auto& pipeline = SampleGroups[SamplePipelineKey()].Resolve[index];
            constexpr std::array<const char*, static_cast<std::size_t>(SamplePhase::Count)> entries {
                "MSAAColorFragment", "MSAADepthFragment", "MSAAMaterialFragment", "MSAAVelocityFragment", "MSAARestoreFragment"};
            if (!pipeline) pipeline = CreateSamplePipeline("MSAA.metal", "MSAAResolveVertex", entries[index],
                restore ? FrameSampleCount : 1, format, colorCount, material || depthOnly);
            if (!pipeline) return false;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            for (std::size_t color = 0; color < colorCount; ++color)
            {
                ObjCObject destination = material ? GBufferTextures[color] : velocity ? MotionTextures[MotionVelocity] :
                    restore ? Multisample.Colors[SampleColor] : HdrActive ? HdrTextures[0] : ColorTexture;
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", color);
                Send<void, ObjCObject>(attachment, "setTexture:", destination);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                Send<void, MTLClearColor>(attachment, "setClearColor:", MTLClearColor{});
            }
            if (material || depthOnly)
            {
                ObjCObject attachment = Send<ObjCObject>(descriptor, "depthAttachment");
                Send<void, ObjCObject>(attachment, "setTexture:", DepthTexture);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                Send<void, double>(attachment, "setClearDepth:", 1.0);
            }
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return Fail("Metal failed to create a multisample resolve encoder");
            // Tracked private textures persist between encoders. End each writer
            // before sampling it; the retained opaque sample depth is never resolved
            // in place or attached while read by a reconstruction shader.
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", pipeline);
            Send<void, ObjCObject>(encoder, "setDepthStencilState:", material || depthOnly ? SampleDepthWriteState : DepthDisabledState);
            const MTLViewport viewport {0, 0, static_cast<double>(CreateInfo.Width), static_cast<double>(CreateInfo.Height), 0, 1};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            const auto bind = [&](NSUInteger slot, ObjCObject texture)
            {
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", texture, slot);
            };
            if (phase == SamplePhase::Color) bind(0, Multisample.Colors[SampleColor]);
            if (depthOnly || material || velocity) bind(1, Multisample.Depth);
            if (material) for (NSUInteger color = 0; color < GBufferAttachmentCount; ++color)
                bind(2 + color, Multisample.Colors[SampleMaterialFirst + color]);
            if (velocity) bind(7, Multisample.Colors[SampleVelocity]);
            if (restore) bind(8, HdrTextures[0]);
            constexpr NSUInteger FullscreenTriangleVertices = 3;
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, FullscreenTriangleVertices);
            Send<void>(encoder, "endEncoding");
            return true;
        }

        bool NeedsSampleColorRestore() const { return SubsurfaceRequested || SubsurfaceResolved || GIActive || AmbientOcclusionActive || ReflectionActive || FogActive || LightShaftActive || TemporalActive || MotionBlurActive || DepthOfFieldActive; }
        bool OpaqueEffectsComplete() const
        {
            return SubsurfaceComplete() && ((!GIActive || GI.Resolved) && (!AmbientOcclusionActive || AmbientOcclusionResolved) && (!ReflectionActive || ReflectionResolved) &&
                (!FogActive || FogResolved) && (!LightShaftActive || LightShaftResolved) && (!TemporalActive || TemporalResolved) && (!MotionBlurActive || MotionBlurResolved) &&
                (!DepthOfFieldActive || DepthOfFieldResolved));
        }

        struct GpuSurface { std::shared_ptr<void> Texture; int Width=0,Height=0; };
        std::vector<std::shared_ptr<GpuSurface>> GpuSurfaces;
        bool PublishGpuColor(ObjCObject encoder,ObjCObject source)
        {
            std::shared_ptr<GpuSurface> surface;
            for (const auto& item:GpuSurfaces) if(item.use_count()==1 && item->Width==CreateInfo.Width && item->Height==CreateInfo.Height) { surface=item; break; }
            if(!surface)
            {
                GpuSurfaces.erase(std::remove_if(GpuSurfaces.begin(),GpuSurfaces.end(),[](const auto& item){return item.use_count()==1;}),GpuSurfaces.end());
                surface=std::make_shared<GpuSurface>();
                auto texture=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA8Unorm,TextureUsageShaderRead,"GPU viewport");
                if(!texture)return false;
                surface->Texture=std::shared_ptr<void>(texture,[](void* object){ReleaseObject(object);});
                surface->Width=CreateInfo.Width;surface->Height=CreateInfo.Height;GpuSurfaces.push_back(surface);
            }
            const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
            Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(encoder,
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                source,0,0,origin,size,surface->Texture.get(),0,0,origin);
            auto image=std::make_shared<GpuImage>();image->Backend=RHIBackend::Metal;
            image->NativeTexture=surface->Texture.get();image->Owner=surface;PendingFrame.GpuColor=std::move(image);return true;
        }

		ObjCObject CreateNativeTexture(int width, int height, NSUInteger format,
			NSUInteger usage, const std::string& label, NSUInteger mipLevels = 1, NSUInteger samples = 1, bool cube = false)
		{
			ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLTextureDescriptor"), "new");
			if (descriptor == nullptr)
			{
				Fail("Metal failed to create a texture descriptor");
				return nullptr;
			}
			Send<void, NSUInteger>(descriptor, "setTextureType:", cube ? TextureTypeCube : (samples > 1 ? TextureType2DMultisample : TextureType2D));
			Send<void, NSUInteger>(descriptor, "setPixelFormat:", format);
			Send<void, NSUInteger>(descriptor, "setWidth:", static_cast<NSUInteger>(width));
			Send<void, NSUInteger>(descriptor, "setHeight:", static_cast<NSUInteger>(height));
			Send<void, NSUInteger>(descriptor, "setDepth:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setMipmapLevelCount:", mipLevels);
			Send<void, NSUInteger>(descriptor, "setArrayLength:", static_cast<NSUInteger>(1));
			Send<void, NSUInteger>(descriptor, "setSampleCount:", samples);
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
			std::size_t bytesPerPixel, const void* source, int mipLevels = 1)
		{
			if (source == nullptr) return true;
			TextureDesc layout;
			layout.Width = width;
			layout.Height = height;
			layout.MipLevels = mipLevels;
			std::vector<std::size_t> stagingOffsets(mipLevels);
			std::size_t stagingSize = 0;
			VisitTextureMips(layout, [&](int level, int w, int h, std::size_t) {
				stagingOffsets[level] = stagingSize;
				stagingSize += AlignUp(std::size_t(w) * bytesPerPixel, 256) * h;
			});
			ObjCObject pool = Send<ObjCObject>(GetClass("NSAutoreleasePool"), "new");
			ObjCObject staging = Send<ObjCObject, NSUInteger, NSUInteger>(Device,
				"newBufferWithLength:options:", stagingSize, static_cast<NSUInteger>(0));
			if (staging == nullptr)
			{
				if (pool) Send<void>(pool, "drain");
				return Fail("Metal failed to allocate a texture upload buffer");
			}
			auto* destination = static_cast<std::uint8_t*>(Send<void*>(staging, "contents"));
			VisitTextureMips(layout, [&](int level, int w, int h, std::size_t offset) {
				const std::size_t sourcePitch = std::size_t(w) * bytesPerPixel;
				const std::size_t stagingPitch = AlignUp(sourcePitch, 256);
				const auto* texels = static_cast<const std::uint8_t*>(source) + (offset / 4) * bytesPerPixel;
				for (int row = 0; row < h; ++row)
					std::memcpy(destination + stagingOffsets[level] + std::size_t(row) * stagingPitch,
						texels + std::size_t(row) * sourcePitch, sourcePitch);
			});
			ObjCObject commandBuffer = Send<ObjCObject>(CommandQueue, "commandBuffer");
			ObjCObject blit = commandBuffer ? Send<ObjCObject>(commandBuffer, "blitCommandEncoder") : nullptr;
			if (blit == nullptr)
			{
				ReleaseObject(staging);
				if (pool) Send<void>(pool, "drain");
				return Fail("Metal failed to create a texture upload command encoder");
			}
			VisitTextureMips(layout, [&](int level, int w, int h, std::size_t) {
				const MTLSize size { static_cast<NSUInteger>(w), static_cast<NSUInteger>(h), 1 };
				const MTLOrigin origin { 0, 0, 0 };
				Send<void, ObjCObject, NSUInteger, NSUInteger, NSUInteger, MTLSize, ObjCObject,
					NSUInteger, NSUInteger, MTLOrigin>(blit,
					"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
					"toTexture:destinationSlice:destinationLevel:destinationOrigin:",
					staging, stagingOffsets[level], AlignUp(std::size_t(w) * bytesPerPixel, 256),
					0, size, texture, 0, static_cast<NSUInteger>(level), origin);
			});
			Send<void>(blit, "endEncoding");
			Send<void>(commandBuffer, "commit");
			Send<void>(commandBuffer, "waitUntilCompleted");
			const bool succeeded = CommandSucceeded(commandBuffer, "Metal texture upload failed");
			ReleaseObject(staging);
			if (pool) Send<void>(pool, "drain");
			return succeeded;
		}

        static constexpr std::size_t VSMRaw = 0, VSMHorizontal = 1, VSMFiltered = 2;
        static constexpr std::uint32_t VSMDeferred = 5, VSMMultisample = 6, VSMVolume = 7, VSMCaster = 8, VSMFilter = 9;
        struct VSMMomentMap
        {
            std::array<ObjCObject, 3> Images {};
            int Width = 0, Height = 0;
            bool Ready = false, RawWritten = false;
        };

        bool UsesVSM() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM; }
        bool UsesSurfaceVSM() const
        { return UsesVSM() && CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap; }

        ObjCObject VSMLibrary(std::size_t program)
        {
            auto& library = VSMLibraries[program];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "VSM.metal", source, LastError)) return nullptr;
            source.insert(0, "#define EVANS_VSM_PROGRAM " + std::to_string(program) + "\n");
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal VSM shader compilation failed: " + ErrorDescription(error));
            return library;
        }

        ObjCObject VSMSurfacePipeline(bool hdr, NSUInteger samples)
        {
            const auto model = static_cast<std::uint32_t>(BoundPipeline.Shading);
            constexpr std::uint32_t SampleShift = 4, HdrBit = 1u << 8, WriteBit = 1u << 9,
                BlendBit = 1u << 10, WeightedBit = 1u << 11;
            const auto key = model | (std::uint32_t(samples) << SampleShift) | (hdr ? HdrBit : 0u) |
                (BoundPipeline.ColorWrite ? WriteBit : 0u) | (BoundPipeline.BlendEnabled ? BlendBit : 0u) |
                (WeightedPass ? WeightedBit : 0u);
            auto& pipeline = VSMSurfacePipelines[key];
            if (pipeline) return pipeline;
            const auto library = VSMLibrary(model);
            if (!library) return nullptr;
            pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, BoundPipeline.BlendEnabled,
                hdr, samples, WeightedPass, library);
            return pipeline;
        }

        ObjCObject VSMScreenPipeline(std::uint32_t program, NSUInteger samples = 1)
        {
            constexpr std::uint32_t ProgramShift = 16;
            auto& pipeline = VSMScreenPipelines[(program << ProgramShift) | std::uint32_t(samples)];
            if (pipeline) return pipeline;
            const auto library = VSMLibrary(program);
            if (!library) return nullptr;
            const bool volume = program == VSMVolume, filter = program == VSMFilter;
            pipeline = CreateTechniquePipeline("VSM.metal", filter ? "MomentFilterVertex" : volume ? "VolumeVertexMain" : "DeferredLightingVertex",
                filter ? "MomentFilterFragment" : volume ? "VolumeFragmentMain" :
                    program == VSMMultisample ? "DeferredLightingSampleFragment" : "DeferredLightingFragment",
                samples, {filter ? PixelFormatRG32Float : volume ? PixelFormatRGBA32Float : PixelFormatRGBA16Float},
                false, ColorWriteMaskAll, false, library);
            return pipeline;
        }

        ObjCObject VSMCasterPipeline(bool instanced)
        {
            auto& pipeline = VSMCasterPipelines[instanced ? 1 : 0];
            if (pipeline) return pipeline;
            const auto library = VSMLibrary(VSMCaster);
            if (!library) return nullptr;
            pipeline = CreateTechniquePipeline("VSM.metal", instanced ? "MomentInstanceVertex" : "MomentCasterVertex",
                "MomentCasterFragment", 1, {PixelFormatRG32Float}, true, ColorWriteMaskAll, false, library);
            return pipeline;
        }

        void DestroyMomentMap(VSMMomentMap& map)
        {
            for (auto image : map.Images) ReleaseObject(image);
            map = {};
        }

        bool EnsureMomentMap(TextureHandle handle)
        {
            if (!VSMSupported) return Fail("Metal VSM requires renderable RG32F moment textures");
            const auto depth = Textures.find(handle.Value);
            if (depth == Textures.end() || depth->second.Description.Format != TextureFormat::Depth32)
                return Fail("Metal VSM requires a depth-map coverage target");
            auto& map = VSMMoments[handle.Value];
            if (map.Images[VSMRaw]) return true;
            VSMMomentMap candidate;
            candidate.Width = depth->second.Description.Width; candidate.Height = depth->second.Description.Height;
            for (auto& image : candidate.Images)
            {
                image = CreateNativeTexture(candidate.Width, candidate.Height, PixelFormatRG32Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans linear shadow moments");
                if (!image) { DestroyMomentMap(candidate); return false; }
            }
            map = candidate;
            return true;
        }

        bool MomentImage(TextureHandle handle, ObjCObject& image)
        {
            const auto map = VSMMoments.find(handle.Value);
            if (map == VSMMoments.end() || !map->second.Ready)
                return Fail("Metal VSM requires completed moment filters from this frame");
            image = map->second.Images[VSMFiltered];
            return true;
        }

        bool BindVSM(ObjCObject encoder, const Matrix4& projection, const LightingState& lights, bool enabled)
        {
            if (enabled)
            {
                ObjCObject image = nullptr;
                if (!FrameCascades.Frame.CascadeCount) { if (!MomentImage(LastShadowTexture, image)) return false; }
                else for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                    if (!MomentImage(FrameCascades.Maps[index], image)) return false;
            }
            VSMConstants constants;
            if (!BuildVSMConstants(projection, lights, FrameCascades, FrameShadowFilter.VSM, enabled, constants, LastError)) return false;
            // Inline bytes are snapshotted per command; later maps cannot mutate them.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:",
                &constants, sizeof(constants), VSMConstantBufferIndex);
            return true;
        }

        bool FilterMoments(TextureHandle handle)
        {
            const auto pipeline = VSMScreenPipeline(VSMFilter);
            if (!pipeline) return false;
            auto& map = VSMMoments.at(handle.Value);
            auto constants = BuildVSMParameters(FrameShadowFilter.VSM);
            for (std::size_t axis = 0; axis < 2; ++axis)
            {
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(attachment, "setTexture:", map.Images[axis + 1]);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                Send<void, MTLClearColor>(attachment, "setClearColor:", {1, 1, 0, 0});
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return Fail("Metal VSM filter encoder creation failed");
                const MTLViewport viewport {0, 0, double(map.Width), double(map.Height), 0, 1};
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", pipeline);
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                constants.Control.W = static_cast<std::uint32_t>(axis);
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", map.Images[axis], 0);
                constexpr NSUInteger TriangleVertices = 3;
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, TriangleVertices);
                Send<void>(encoder, "endEncoding");
            }
            // Tracked private resources order raster writes and subsequent reads
            // between encoders; raw moments are preserved for later load passes.
            map.Ready = map.RawWritten = true;
            PendingFrame.Statistics.ShadowMomentFilterPassCount += 2;
            return true;
        }

        bool UsesSurfacePCSS() const
        {
            return CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap &&
                FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
        }

        ObjCObject RayShadowLibrary(std::size_t program)
        {
            auto& library = RayShadowLibraries[program];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "RayTracedShadows.metal", source, LastError)) return nullptr;
            source.insert(0, "#define EVANS_RAY_SHADOW_PROGRAM " + std::to_string(program) + "\n");
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal RayShadow shader compilation failed: " + ErrorDescription(error));
            return library;
        }

        ObjCObject RayShadowSurfacePipeline(bool hdr, NSUInteger samples)
        {
            const auto model = static_cast<std::uint32_t>(BoundPipeline.Shading);
            constexpr std::uint32_t SampleShift = 4, HdrBit = 1u << 8, WriteBit = 1u << 9,
                BlendBit = 1u << 10, WeightedBit = 1u << 11;
            const auto key = model | (std::uint32_t(samples) << SampleShift) | (hdr ? HdrBit : 0u) |
                (BoundPipeline.ColorWrite ? WriteBit : 0u) | (BoundPipeline.BlendEnabled ? BlendBit : 0u) |
                (WeightedPass ? WeightedBit : 0u);
            auto& pipeline = RayShadowSurfacePipelines[key];
            if (pipeline) return pipeline;
            const auto library = RayShadowLibrary(model);
            if (!library) return nullptr;
            pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, BoundPipeline.BlendEnabled,
                hdr, samples, WeightedPass, library);
            return pipeline;
        }

        ObjCObject RayShadowScreenPipeline(std::uint32_t program, NSUInteger samples = 1)
        {
            constexpr std::uint32_t ProgramShift = 16;
            auto& pipeline = RayShadowScreenPipelines[(program << ProgramShift) | std::uint32_t(samples)];
            if (pipeline) return pipeline;
            const auto library = RayShadowLibrary(program);
            if (!library) return nullptr;
            const bool volume = program == 7;
            pipeline = CreateTechniquePipeline("RayTracedShadows.metal", volume ? "VolumeVertexMain" : "DeferredLightingVertex",
                volume ? "VolumeFragmentMain" : program == 6 ? "DeferredLightingSampleFragment" : "DeferredLightingFragment",
                samples, {volume ? PixelFormatRGBA32Float : PixelFormatRGBA16Float}, false,
                ColorWriteMaskAll, false, library);
            return pipeline;
        }

        ObjCObject PCSSLibrary(std::size_t program)
        {
            auto& library = PCSSLibraries[program];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "PCSS.metal", source, LastError)) return nullptr;
            source.insert(0, "#define EVANS_PCSS_PROGRAM " + std::to_string(program) + "\n");
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal PCSS shader compilation failed: " + ErrorDescription(error));
            return library;
        }

        bool BindPCSS(ObjCObject encoder, const Matrix4& lightProjection, const LightingState& lights, bool samplesShadow)
        {
            PCSSConstants constants;
            std::string error;
            if (!BuildPCSSConstants(lightProjection, lights, FrameCascades, FrameShadowFilter.PCSS,
                    samplesShadow, constants, error)) return Fail(error);
            // setFragmentBytes copies this small block for the encoded draw.
            // Later receivers and aborted frames never reuse a mutable pointer.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:",
                &constants, sizeof(constants), PCSSConstantBufferIndex);
            return true;
        }

        ObjCObject PCSSSurfacePipeline(bool hdr, NSUInteger samples)
        {
            const auto model = static_cast<std::uint32_t>(BoundPipeline.Shading);
            constexpr std::uint32_t SampleShift = 4, HdrBit = 1u << 8, WriteBit = 1u << 9,
                BlendBit = 1u << 10, WeightedBit = 1u << 11;
            const auto key = model | (std::uint32_t(samples) << SampleShift) | (hdr ? HdrBit : 0u) |
                (BoundPipeline.ColorWrite ? WriteBit : 0u) | (BoundPipeline.BlendEnabled ? BlendBit : 0u) |
                (WeightedPass ? WeightedBit : 0u);
            auto& pipeline = PCSSSurfacePipelines[key];
            if (pipeline) return pipeline;
            const auto library = PCSSLibrary(model);
            if (!library) return nullptr;
            pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, BoundPipeline.BlendEnabled,
                hdr, samples, WeightedPass, library);
            return pipeline;
        }

        ObjCObject PCSSScreenPipeline(std::uint32_t program, NSUInteger samples = 1)
        {
            constexpr std::uint32_t ProgramShift = 16;
            auto& pipeline = PCSSScreenPipelines[(program << ProgramShift) | std::uint32_t(samples)];
            if (pipeline) return pipeline;
            const auto library = PCSSLibrary(program);
            if (!library) return nullptr;
            const bool volume = program == 7;
            pipeline = CreateTechniquePipeline("PCSS.metal", volume ? "VolumeVertexMain" : "DeferredLightingVertex",
                volume ? "VolumeFragmentMain" : program == 6 ? "DeferredLightingSampleFragment" : "DeferredLightingFragment",
                samples, {volume ? PixelFormatRGBA32Float : PixelFormatRGBA16Float}, false,
                ColorWriteMaskAll, false, library);
            return pipeline;
        }

		ObjCObject CreatePipeline(
			PassType pass,
			bool colorWrite,
			bool blendEnabled,
            bool hdr = false, NSUInteger samples = 1, bool weighted = false, ObjCObject fragmentLibrary = nullptr,
            ObjCObject vertexLibrary = nullptr, const char* vertexName = nullptr, const char* fragmentName = nullptr,
            bool depthOnly = false)
		{
			const bool shadow = pass == PassType::Shadow;
			const RealtimeShaderProgram& shader =
				BuiltinRealtimeShader().Program;
			ObjCObject vertexFunction = Send<ObjCObject, ObjCObject>(vertexLibrary ? vertexLibrary : shadow ? ShadowLibrary : Library,
				"newFunctionWithName:", MakeString(
					vertexName ? vertexName : shadow ? "ShadowDepthVertex" : shader.FindStage(
						RealtimeShaderStage::Vertex)->Metal.c_str()));
			ObjCObject fragmentFunction = Send<ObjCObject, ObjCObject>(
				fragmentLibrary ? fragmentLibrary : shadow ? ShadowLibrary : Library,
				"newFunctionWithName:",
				MakeString(fragmentName ? fragmentName : shadow
                    ? "ShadowDepthFragment"
                    : weighted ? "EvansWeightedFragment" : shader.FindStage(
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
            Send<void, NSUInteger>(descriptor, "setSampleCount:", shadow ? 1 : samples);
			Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragmentFunction);
			Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
			if (!shadow && !depthOnly)
			{
				ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
				ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments,
					"objectAtIndexedSubscript:", static_cast<NSUInteger>(0));
				Send<void, NSUInteger>(
					attachment,
					"setPixelFormat:",
					hdr ? PixelFormatRGBA16Float : PixelFormatRGBA8UnormSRGB);
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
            if (weighted)
            {
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                const std::array<NSUInteger, 2> formats {PixelFormatRGBA32Float, PixelFormatR32Float};
                for (std::size_t index = 0; index < formats.size(); ++index)
                {
                    ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
                    const bool accumulation = index == 0;
                    Send<void, NSUInteger>(attachment, "setPixelFormat:", formats[index]);
                    Send<void, NSUInteger>(attachment, "setWriteMask:", accumulation ? ColorWriteMaskAll : ColorWriteMaskRed);
                    Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCYes);
                    Send<void, NSUInteger>(attachment, "setSourceRGBBlendFactor:", accumulation ? BlendFactorOne : BlendFactorZero);
                    Send<void, NSUInteger>(attachment, "setDestinationRGBBlendFactor:", accumulation ? BlendFactorOne : BlendFactorOneMinusSourceColor);
                    Send<void, NSUInteger>(attachment, "setSourceAlphaBlendFactor:", accumulation ? BlendFactorOne : BlendFactorZero);
                    Send<void, NSUInteger>(attachment, "setDestinationAlphaBlendFactor:", BlendFactorOne);
                    Send<void, NSUInteger>(attachment, "setRgbBlendOperation:", BlendOperationAdd);
                    Send<void, NSUInteger>(attachment, "setAlphaBlendOperation:", BlendOperationAdd);
                }
            }
			ObjCObject error = nullptr;
			ObjCObject pipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
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

        bool PrepareCascadedShadows(const CascadedShadowState& state)
        {
            FrameCascades = {};
            CascadeTextures.fill(nullptr);
            const auto count = state.Frame.CascadeCount;
            if (count > MaximumShadowCascades || (count && state.MapResolution <= 0))
                return Fail("Metal cascade count or map resolution is invalid");
            for (std::uint32_t index = 0; index < count; ++index)
            {
                const auto map = Textures.find(state.Maps[index].Value);
                if (map == Textures.end() || map->second.Description.Format != TextureFormat::Depth32 ||
                    map->second.Description.Width != state.MapResolution ||
                    map->second.Description.Height != state.MapResolution || !map->second.Texture)
                    return Fail("Metal cascade depth target is invalid");
                CascadeTextures[index] = map->second.Texture;
            }
            if (count && !CascadeDepthPipeline)
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", "CascadedShadowMap.metal", source, LastError)) return false;
                ObjCObject error = nullptr;
                ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
                if (!library) return Fail("Metal cascade shader compilation failed: " + ErrorDescription(error));
                ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString("CascadeDepthVertex"));
                ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString("CascadeDepthFragment"));
                if (vertex && fragment)
                {
                    ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
                    Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
                    Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragment);
                    Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
                    CascadeDepthPipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
                    ReleaseObject(descriptor);
                    if (!CascadeDepthPipeline) Fail("Metal cascade depth pipeline failed: " + ErrorDescription(error));
                }
                else Fail("Metal cascade depth entry point is missing");
                ReleaseObject(vertex);
                ReleaseObject(fragment);
                ReleaseObject(library);
                if (!CascadeDepthPipeline) return false;
            }
            FrameCascades = state;
            return true;
        }

        bool BindCascades(ObjCObject encoder, bool enabled, bool moments = false)
        {
            const bool active = enabled && (FrameCascades.Frame.CascadeCount > 0 || FrameVirtualShadowAtlas);
            const auto constants = active && FrameVirtualShadowAtlas ? BuildVirtualShadowSamplingConstants(FrameVirtualShadowAtlas->Info,FrameVirtualShadowSampling,FrameShadowFilter) :
                BuildCascadedShadowConstants(active ? FrameCascades : CascadedShadowState{}, FrameShadowFilter);
            // setFragmentBytes snapshots this small block into the command stream;
            // later draws can disable CSM without changing earlier encoded draws.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:",
                &constants, sizeof(constants), CascadeConstantBufferIndex);
            for (NSUInteger index = 0; index < MaximumShadowCascades; ++index)
            {
                ObjCObject texture = moments ? DummyColorTexture : DummyShadowTexture;
                if (active && FrameVirtualShadowAtlas && index<2)
                    texture=index?FrameVirtualShadowAtlas->Directory:FrameVirtualShadowAtlas->Depth;
                else if (active && index < FrameCascades.Frame.CascadeCount)
                {
                    texture = CascadeTextures[index];
                    if (moments && !MomentImage(FrameCascades.Maps[index], texture)) return false;
                }
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", texture, FirstCascadeTextureIndex + index);
            }
            return true;
        }

        ObjCObject InstancePipeline(bool hdr, NSUInteger samples)
        {
            enum class Program : std::uint32_t { Forward, Gouraud, Deferred, Shadow, Velocity, Cascade };
            const bool shadow = CurrentPass.Type == PassType::Shadow;
            const bool geometry = CurrentPass.Type == PassType::GBuffer;
            const bool velocity = CurrentPass.Type == PassType::Velocity;
            const bool cascade = shadow && FrameCascades.Frame.CascadeCount;
            const auto program = shadow ? (cascade ? Program::Cascade : Program::Shadow) :
                geometry ? Program::Deferred : velocity ? Program::Velocity :
                BoundPipeline.Shading == Techniques::ShadingModel::Gouraud ? Program::Gouraud : Program::Forward;
            const bool colorWrite = velocity || BoundPipeline.ColorWrite;
            const bool blended = !shadow && !geometry && !velocity && BoundPipeline.BlendEnabled;
            const bool ray=FrameRayShadowImage && !FrameAreaLights && CurrentPass.Type==PassType::Color && BoundPipeline.SamplesShadowMap;
            const bool pcf=UsesSurfacePCF();
            const bool material=UsesMaterialStage(BoundPipeline,CurrentPass.Type);
            const auto materialKind=SelectMaterialTechnique(BoundPipeline,WeightedPass);
            const bool pcss = UsesSurfacePCSS();
            const bool vsm = UsesSurfaceVSM();
            const bool contact = UsesSurfaceContact();
            constexpr std::uint32_t ShadingKeyShift = 4, SampleKeyShift = 8;
            constexpr std::uint32_t ColorWriteKeyBit = 1U << 16U, HdrKeyBit = 1U << 17U;
            constexpr std::uint32_t BlendKeyBit = 1U << 18U, WeightedKeyBit = 1U << 19U, PCSSKeyBit = 1U << 20U, VSMKeyBit = 1U << 21U,
                ContactKeyBit = 1u << 22;
            const auto key = static_cast<std::uint32_t>(program) |
                (static_cast<std::uint32_t>(ShadingTechniqueIndex(BoundPipeline.Shading, FrameAreaLights)) << ShadingKeyShift) |
                (static_cast<std::uint32_t>(samples) << SampleKeyShift) |
                (colorWrite ? ColorWriteKeyBit : 0U) | (hdr ? HdrKeyBit : 0U) |
                (blended ? BlendKeyBit : 0U) | (WeightedPass ? WeightedKeyBit : 0U) | (pcss ? PCSSKeyBit : 0U) | (vsm ? VSMKeyBit : 0U) |
                (contact ? ContactKeyBit : 0u) | (ray ? 1u<<23 : 0u) | (pcf?1u<<24:0u) |
                (material?(unsigned(materialKind)+1)<<25:0u);
            auto& pipeline = InstancePipelines[key];
            if (pipeline) return pipeline;

            const auto loadLibrary = [&](const char* filename, ObjCObject& library)
            {
                if (library) return true;
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", filename, source, LastError)) return false;
                ObjCObject error = nullptr;
                library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
                return library || Fail(std::string("Metal instance shader compilation failed (") + filename + "): " + ErrorDescription(error));
            };
            if (!loadLibrary("InstancedRendering.metal", InstanceLibrary)) return nullptr;
            ObjCObject fragmentLibrary = nullptr;
            const char* fragmentName = WeightedPass ? "EvansWeightedFragment" : "EvansFragmentMain";
            if (geometry && material)
            {
                fragmentLibrary=MaterialLibrary(materialKind,0,true);
                fragmentName="DeferredGeometryFragment";
            }
            else if (geometry || velocity || cascade)
            {
                const char* filename = geometry ? "Deferred.metal" : velocity ? "MotionVectors.metal" : "CascadedShadowMap.metal";
                // These are the existing consumer fragments, including their alpha
                // coverage rules; no shader source is included or translated.
                auto& library = NamedTechniqueLibraries[filename];
                if (!loadLibrary(filename, library)) return nullptr;
                fragmentLibrary = library;
                fragmentName = geometry ? "DeferredGeometryFragment" : velocity ? "MotionVelocityFragment" : "CascadeDepthFragment";
            }
            else if (shadow) { fragmentLibrary = ShadowLibrary; fragmentName = "ShadowDepthFragment"; }
            else fragmentLibrary = material?MaterialLibrary(materialKind,unsigned(BoundPipeline.Shading),false):pcf?PCFLibrary(unsigned(BoundPipeline.Shading)):ray ? RayShadowLibrary(static_cast<unsigned>(BoundPipeline.Shading)) : contact ? ContactLibrary(false, pcss, vsm, samples) :
                vsm ? VSMLibrary(static_cast<std::size_t>(BoundPipeline.Shading)) : pcss ? PCSSLibrary(static_cast<std::size_t>(BoundPipeline.Shading)) :
                TechniqueLibraries[ShadingTechniqueIndex(BoundPipeline.Shading, FrameAreaLights)];
            if (!fragmentLibrary) return nullptr;
            constexpr std::array<const char*, 6> vertexNames {
                "InstanceForwardVertex", "InstanceGouraudVertex", "InstanceDeferredVertex",
                "InstanceShadowVertex", "InstanceVelocityVertex", "InstanceCascadeVertex"};
            ObjCObject vertex = Send<ObjCObject, ObjCObject>(InstanceLibrary,
                "newFunctionWithName:", MakeString(vertexNames[static_cast<std::size_t>(program)]));
            ObjCObject fragment = Send<ObjCObject, ObjCObject>(fragmentLibrary, "newFunctionWithName:", MakeString(fragmentName));
            if (!vertex || !fragment)
            {
                ReleaseObject(vertex); ReleaseObject(fragment);
                Fail("Metal instance vertex or consumer fragment entry point is missing");
                return nullptr;
            }
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
            Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
            Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragment);
            Send<void, NSUInteger>(descriptor, "setSampleCount:", shadow ? 1 : samples);
            Send<void, NSUInteger>(descriptor, "setDepthAttachmentPixelFormat:", PixelFormatDepth32Float);
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            const std::size_t attachmentCount = shadow ? 0 : geometry ? GBufferAttachmentCount : WeightedPass ? 2 : 1;
            for (std::size_t index = 0; index < attachmentCount; ++index)
            {
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
                Send<void, NSUInteger>(attachment, "setPixelFormat:",
                    WeightedPass ? (index == 0 ? PixelFormatRGBA32Float : PixelFormatR32Float) :
                        geometry || velocity || hdr ? PixelFormatRGBA16Float : PixelFormatRGBA8UnormSRGB);
                Send<void, NSUInteger>(attachment, "setWriteMask:", WeightedPass && index == 1 ? ColorWriteMaskRed :
                    colorWrite ? ColorWriteMaskAll : ColorWriteMaskNone);
                Send<void, ObjCBool>(attachment, "setBlendingEnabled:", blended || WeightedPass ? ObjCYes : ObjCNo);
                if (blended || WeightedPass)
                {
                    const bool accumulation = index == 0;
                    Send<void, NSUInteger>(attachment, "setSourceRGBBlendFactor:", WeightedPass ?
                        (accumulation ? BlendFactorOne : BlendFactorZero) : BlendFactorSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setDestinationRGBBlendFactor:", WeightedPass ?
                        (accumulation ? BlendFactorOne : BlendFactorOneMinusSourceColor) : BlendFactorOneMinusSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setSourceAlphaBlendFactor:", WeightedPass && !accumulation ? BlendFactorZero : BlendFactorOne);
                    Send<void, NSUInteger>(attachment, "setDestinationAlphaBlendFactor:", WeightedPass ? BlendFactorOne : BlendFactorOneMinusSourceAlpha);
                    Send<void, NSUInteger>(attachment, "setRgbBlendOperation:", BlendOperationAdd);
                    Send<void, NSUInteger>(attachment, "setAlphaBlendOperation:", BlendOperationAdd);
                }
            }
            ObjCObject error = nullptr;
            pipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
            ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment);
            if (!pipeline) Fail("Metal instance pipeline creation failed: " + ErrorDescription(error));
            return pipeline;
        }

        struct InstanceUploadPage { ObjCObject Buffer = nullptr; std::uint8_t* Mapped = nullptr; std::size_t Cursor = 0; };
        bool UploadInstances(const InstanceConstantBlock& constants)
        {
            if (ActiveGpuBatch)
            {
                Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setVertexBuffer:offset:atIndex:",
                    ActiveGpuBatch->Output, 0, InstanceConstantBufferIndex);
                return true;
            }
            std::size_t index = 0;
            while (index < InstancePages.size() && InstancePages[index].Cursor + InstanceUploadStride > InstanceUploadPageBytes) ++index;
            if (index == InstancePages.size())
            {
                InstanceUploadPage page;
                // options=0 explicitly selects shared storage and the default CPU cache.
                page.Buffer = Send<ObjCObject, NSUInteger, NSUInteger>(Device,
                    "newBufferWithLength:options:", InstanceUploadPageBytes, 0);
                page.Mapped = page.Buffer ? Send<std::uint8_t*>(page.Buffer, "contents") : nullptr;
                if (!page.Mapped)
                {
                    ReleaseObject(page.Buffer);
                    return Fail("Metal failed to allocate shared instance upload memory");
                }
                try { InstancePages.push_back(page); }
                catch (...) { ReleaseObject(page.Buffer); throw; }
            }
            auto& page = InstancePages[index];
            // Each encoded draw owns a disjoint slice until Present has waited for
            // GPU completion, or AbortFrame discards the unsubmitted command buffer.
            std::memcpy(page.Mapped + page.Cursor, &constants, sizeof(constants));
            Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setVertexBuffer:offset:atIndex:",
                page.Buffer, page.Cursor, InstanceConstantBufferIndex);
            page.Cursor += InstanceUploadStride;
            return true;
        }

        bool SelectTechnique(Techniques::ShadingModel model, bool hdr = false, NSUInteger samples = 1)
        {
            const std::size_t index = ShadingTechniqueIndex(model, FrameAreaLights);
            if (index >= TechniqueLibraries.size()) return Fail("Invalid Metal shading model");
            if (TechniqueLibraries[index] == nullptr)
            {
                const std::string filename = std::string(ShadingTechniqueFile(model, FrameAreaLights)) + ".metal";
                if (!LoadTechniqueShaderSource("Metal", filename.c_str(), ShaderCode, LastError)) return false;
                ObjCObject error = nullptr;
                Library = CompilePositionInvariantLibrary(Device, ShaderCode, &error);
                if (Library == nullptr) return Fail("Metal shader compilation failed: " + ErrorDescription(error));
                TechniqueLibraries[index] = Library;
            }
            Library = TechniqueLibraries[index];
            auto& ordinary = hdr ? HdrTechniquePipelines[index] : TechniquePipelines[index];
            auto& pipelines = samples > 1 ? SampleGroups[SamplePipelineKey()].Shading[index] : ordinary;
            if (!pipelines[0]) pipelines[0] = CreatePipeline(PassType::Color, true, false, hdr, samples);
            if (!pipelines[1]) pipelines[1] = CreatePipeline(PassType::Color, true, true, hdr, samples);
            if (!pipelines[2]) pipelines[2] = CreatePipeline(PassType::Color, false, false, hdr, samples);
            ColorPipeline = pipelines[0];
            ColorBlendPipeline = pipelines[1];
            ColorWriteDisabledPipeline = pipelines[2];
            return ColorPipeline && ColorBlendPipeline && ColorWriteDisabledPipeline;
        }

		bool CreatePipelineObjects()
		{
            if (!SelectTechnique(Techniques::ShadingModel::Phong)) return false;
            std::string shadowSource;
            if (!LoadTechniqueShaderSource("Metal", "ShadowMapping.metal", shadowSource, LastError)) return false;
            ObjCObject shadowError = nullptr;
            ShadowLibrary = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(shadowSource.c_str()),nullptr,&shadowError);
            if (!ShadowLibrary) return Fail("Metal shadow shader compilation failed: " + ErrorDescription(shadowError));
            ShadowPipeline = CreatePipeline(PassType::Shadow, false, false);
            if (!ShadowPipeline) return false;
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
				Send<void, float>(samplerDescriptor, "setLodMaxClamp:", 0.0f);
				SurfaceSampler = Send<ObjCObject, ObjCObject>(Device, "newSamplerStateWithDescriptor:", samplerDescriptor);
				constexpr NSUInteger MipmapFilterLinear = 2;
				Send<void, NSUInteger>(samplerDescriptor, "setMipFilter:", MipmapFilterLinear);
				Send<void, NSUInteger>(samplerDescriptor, "setMaxAnisotropy:", MaterialAnisotropy);
				Send<void, float>(samplerDescriptor, "setLodMaxClamp:", std::numeric_limits<float>::max());
				AnisotropicSampler = Send<ObjCObject, ObjCObject>(Device, "newSamplerStateWithDescriptor:", samplerDescriptor);
				ReleaseObject(samplerDescriptor);
			}
			DepthWriteState = CreateDepthState(CompareFunctionLess, true);
			DepthReadState = CreateDepthState(CompareFunctionLessEqual, false);
			DepthDisabledState = CreateDepthState(CompareFunctionAlways, false);
			return ColorPipeline != nullptr && ColorBlendPipeline != nullptr
				&& ColorWriteDisabledPipeline != nullptr
				&& ShadowPipeline != nullptr && DepthWriteState != nullptr
				&& DepthReadState != nullptr && DepthDisabledState != nullptr
				&& SurfaceSampler != nullptr && AnisotropicSampler != nullptr;
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
            DestroyMultisampleTargets(Multisample);
            DestroyPostTargets();
            DestroyHdrTargets();
			DestroyDeferredTargets();
            DestroyAmbientOcclusionTargets();
			ColorTexture = color;
			DepthTexture = depth;
			return true;
		}

        void DestroyPostTargets()
        {
            for (auto& texture : PostTextures)
            {
                ReleaseObject(texture);
                texture = nullptr;
            }
        }

        bool EnsurePostTargets()
        {
            if (PostTextures[0] && PostTextures[1]) return true;
            DestroyPostTargets();
            for (auto& texture : PostTextures)
            {
                texture = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA8UnormSRGB,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal post-process color");
                if (!texture) { DestroyPostTargets(); return false; }
            }
            return true;
        }

        ObjCObject CreateFullscreenPipeline(const char* filename, bool hdrTechnique, bool hdrOutput)
        {
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", filename, source, LastError)) return nullptr;
            ObjCObject error = nullptr;
            ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) { Fail("Metal post-process compilation failed: " + ErrorDescription(error)); return nullptr; }
            ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString(hdrTechnique ? "HdrVertexMain" : "PostVertexMain"));
            ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString(hdrTechnique ? "HdrFragmentMain" : "PostFragmentMain"));
            if (!vertex || !fragment)
            {
                ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
                Fail("Metal post-process entry point is missing");
                return nullptr;
            }
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
            Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
            Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragment);
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, NSUInteger>(attachment, "setPixelFormat:", hdrOutput ? PixelFormatRGBA16Float : PixelFormatRGBA8UnormSRGB);
            Send<void, NSUInteger>(attachment, "setWriteMask:", ColorWriteMaskAll);
            Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCNo);
            ObjCObject pipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
            ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
            if (!pipeline) Fail("Metal post-process pipeline failed: " + ErrorDescription(error));
            return pipeline;
        }

        ObjCObject GetPostPipeline(Techniques::PostProcessPassKind pass)
        {
            const std::size_t index = static_cast<std::size_t>(pass);
            if (index >= PostPipelines.size()) { Fail("Invalid Metal post-process pass"); return nullptr; }
            constexpr const char* filenames[] = {"FXAA.metal", "Vignette.metal", "FilmGrain.metal", "ScreenSpaceDistortion.metal", "NPR.metal", "Pixelation.metal", "ColorEdgeOutline.metal", "DynamicDistortion.metal", "OilPaint.metal", "Watercolor.metal", "DepthNormalOutline.metal"};
            static_assert(std::size(filenames) == PostProcessTechniqueCount);
            if (!PostPipelines[index]) PostPipelines[index] = CreateFullscreenPipeline(filenames[index], false, false);
            return PostPipelines[index];
        }

        void DestroyHdrTargets()
        {
        DestroyPathTargets();
            DestroyLightShaftTargets();
            DestroyWeightedTargets();
            DestroyFogTargets();
            DestroyTemporalTargets();
            DestroyGITargets();
            DestroyProbeTargets();
            DestroyLightmapTargets(); DestroyRefractionTargets(); DestroyPeelingTargets();
            ReleaseObject(ReflectionTexture);
            ReflectionTexture = nullptr;
            DestroyMotionTargets();
            for (auto& texture : HdrTextures) { ReleaseObject(texture); texture = nullptr; }
        }

        bool EnsureHdrTargets()
        {
            if (HdrTextures[0]) return true;
            for (std::size_t index = 0; index < HdrTextures.size(); ++index)
            {
                const int width = index >= 2 ? (CreateInfo.Width + 1) / 2 : CreateInfo.Width;
                const int height = index >= 2 ? (CreateInfo.Height + 1) / 2 : CreateInfo.Height;
                HdrTextures[index] = CreateNativeTexture(width, height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans HDR radiance");
                if (!HdrTextures[index]) { DestroyHdrTargets(); return false; }
            }
            return true;
        }

        void DestroyMotionTargets()
        {
            for (auto& texture : MotionTextures) { ReleaseObject(texture); texture = nullptr; }
        }

        bool EnsureVelocityResources()
        {
            if (!HdrActive || !HdrTextures[0] || !DepthTexture)
                return Fail("Metal velocity requires HDR color and opaque depth");
            if (!MotionPipelines[MotionGeometryPipeline])
                MotionPipelines[MotionGeometryPipeline] = CreateTechniquePipeline("MotionVectors.metal",
                    "MotionVelocityVertex", "MotionVelocityFragment", 1, {PixelFormatRGBA16Float}, true);
            if (!MotionPipelines[MotionGeometryPipeline]) return false;
            if (MotionTextures[MotionVelocity]) return true;
            MotionTextures[MotionVelocity] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans rigid motion vectors");
            return MotionTextures[MotionVelocity] != nullptr;
        }

        bool EnsureMotionResources()
        {
            if (!EnsureVelocityResources()) return false;
            for (std::size_t index = MotionFilterPipeline; index < MotionPipelines.size(); ++index)
            {
                if (!MotionPipelines[index])
                    MotionPipelines[index] = CreateTechniquePipeline("MotionBlur.metal", "MotionFilterVertex", "MotionFilterFragment",
                        1, {PixelFormatRGBA16Float}, false, index == MotionCompositePipeline ? ColorWriteMaskRGB : ColorWriteMaskAll);
                if (!MotionPipelines[index]) return false;
            }
            if (MotionTextures[MotionNeighborMax]) return true;
            const int tile = Techniques::MotionBlurTileSize;
            for (std::size_t index = MotionTileMax; index < MotionTextures.size(); ++index)
            {
                const int width = (CreateInfo.Width + tile - 1) / tile;
                const int height = (CreateInfo.Height + tile - 1) / tile;
                MotionTextures[index] = CreateNativeTexture(width, height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans motion vectors/maxima");
                if (!MotionTextures[index]) { DestroyMotionTargets(); return false; }
            }
            return true;
        }

        void DestroyVolumeLightTargets()
        {
            for (auto& texture : VolumeLightTextures) { ReleaseObject(texture); texture = nullptr; }
        }

        ObjCObject MediaLibrary(bool composition, bool moments)
        {
            const std::size_t index = composition ? 0 : moments ? 2 : 1;
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& library = cloud ? CloudLibraries[index] : MediaLibraries[index];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", cloud ? "VolumetricClouds.metal" : "ParticipatingMedia.metal", source, LastError)) return nullptr;
            if (composition) source.insert(0, cloud ? "#define EVANS_CLOUD_COMPOSITE 1\n" : "#define EVANS_MEDIA_COMPOSITE 1\n");
            else if (moments) source.insert(0, cloud ? "#define EVANS_CLOUD_MOMENTS 1\n" : "#define EVANS_MEDIA_MOMENTS 1\n");
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal bounded-medium compilation failed: " + ErrorDescription(error));
            return library;
        }

        bool EnsureVolumeLightResources()
        {
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& programs = cloud ? CloudVolumePipelines[UsesVSM()] : media ? MediaVolumePipelines[UsesVSM()] : VolumeLightPipelines;
            const auto library = media ? MediaLibrary(false, UsesVSM()) : nullptr;
            if (media && !library) return false;
            if (!VolumeLightSupported || !FogAtlas) return Fail("Metal volume lighting requires FP32 transport targets and a cumulative fog atlas");
            const std::array<NSUInteger, 2> formats {PixelFormatRGBA32Float, PixelFormatRGBA16Float};
            for (std::size_t index = 0; index < programs.size(); ++index)
            {
                if (!programs[index])
                    programs[index] = CreateTechniquePipeline(cloud ? "VolumetricClouds.metal" : media ? "ParticipatingMedia.metal" : "VolumetricLighting.metal", "VolumeVertexMain", "VolumeFragmentMain",
                        1, {formats[index]}, false, ColorWriteMaskAll, false, library);
                if (!programs[index]) return false;
            }
            if (VolumeLightTextures[0] && VolumeLightTextures[1]) return true;
            std::array<ObjCObject, 2> candidate {};
            for (auto& texture : candidate)
            {
                texture = CreateNativeTexture(FogDimensions.AtlasWidth(), FogDimensions.AtlasHeight(), PixelFormatRGBA32Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans FP32 volume transport");
                if (!texture)
                {
                    for (auto allocated : candidate) ReleaseObject(allocated);
                    return false;
                }
            }
            DestroyVolumeLightTargets(); VolumeLightTextures = candidate;
            return true;
        }

        bool GenerateVolumeLighting(const VolumetricFogDesc& requested)
        {
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& programs = cloud ? CloudVolumePipelines[UsesVSM()] : media ? MediaVolumePipelines[UsesVSM()] : VolumeLightPipelines;
            auto desc = requested;
            ObjCObject density=DummyColorTexture;
            if (desc.Density)
            {
                const auto found=Textures.find(desc.Density->Texture.Value);
                if (found==Textures.end() || !ValidDensityBinding(*desc.Density,found->second.Description))
                    return Fail("Metal volume density requires a matching linear byte atlas");
                density=found->second.Texture;
            }
            if (LightListsActive && !LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, LastError)) return false;
            if (LightListsActive) desc.Lighting.SceneLights = LightListFrame.Description().Lights;
            const auto shadow = Textures.find(LastShadowTexture.Value);
            if (!FrameRayShadowImage && !FrameVirtualShadowAtlas && !FrameCascades.Frame.CascadeCount)
                desc.SamplesShadowMap = desc.SamplesShadowMap && shadow != Textures.end() &&
                    shadow->second.Description.Format == TextureFormat::Depth32 && shadow->second.Texture;
            VolumetricLightingConstants lighting;
            if (!BuildVolumetricLightingConstants(desc, FrameVolumeLighting, FrameCascades, FrameShadowFilter, lighting, LightListsActive, &FrameFog,
                FrameVirtualShadowAtlas ? &FrameVirtualShadowAtlas->Info : nullptr, &FrameVirtualShadowSampling,
                {FogConstants.ViewportGrid.Z, FogConstants.ViewportGrid.W}))
                return Fail("Metal volume lighting requires finite lights and shadow transforms");
            // The inactive branch still needs a valid device buffer argument.
            if (!ForwardPlusEmptyBuffer)
            {
                ForwardPlusEmptyBuffer = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", sizeof(ShaderLight), 0);
                if (!ForwardPlusEmptyBuffer) return Fail("Metal volume dummy light allocation failed");
                std::memset(Send<void*>(ForwardPlusEmptyBuffer, "contents"), 0, sizeof(ShaderLight));
            }
            const bool pcss = FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
            const bool vsm = UsesVSM();
            const auto injection = media ? programs[0] : vsm ? VSMScreenPipeline(VSMVolume) : pcss ? PCSSScreenPipeline(7) : programs[0];
            if (!injection) return false;
            const auto passes = BuildVolumetricLightingPasses(FogDimensions.Slices);
            if (passes.empty()) return Fail("Metal volume scan requires a valid depth grid");
            std::array<ObjCObject, 6> inputs {DummyColorTexture, DummyShadowTexture, DummyShadowTexture,
                DummyShadowTexture, DummyShadowTexture, DummyShadowTexture};
            if (vsm) inputs.fill(DummyColorTexture);
            if (FrameRayShadowImage) {}
            else if (lighting.Control.Y && FrameVirtualShadowAtlas)
            { inputs[2]=FrameVirtualShadowAtlas->Depth; inputs[3]=FrameVirtualShadowAtlas->Directory; }
            else if (lighting.Control.Y == 1)
            {
                inputs[1] = shadow->second.Texture;
                if (vsm && !MomentImage(LastShadowTexture, inputs[1])) return false;
            }
            else if (lighting.Control.Y == 2)
                for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                {
                    inputs[index + 2] = CascadeTextures[index];
                    if (vsm && !MomentImage(FrameCascades.Maps[index], inputs[index + 2])) return false;
                }
            for (std::size_t index = 0; index < passes.size(); ++index)
            {
                const bool inject = index == 0, final = index + 1 == passes.size();
                if (!inject) inputs = {VolumeLightTextures[(index - 1) % VolumeLightTextures.size()], DummyShadowTexture,
                    DummyShadowTexture, DummyShadowTexture, DummyShadowTexture, DummyShadowTexture};
                if (!inject && media && vsm)
                    for (std::size_t unit = 1; unit < inputs.size(); ++unit) inputs[unit] = DummyColorTexture;
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(Send<ObjCObject>(descriptor, "colorAttachments"), "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(attachment, "setTexture:", final ? FogAtlas : VolumeLightTextures[index % VolumeLightTextures.size()]);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return Fail("Metal failed to create volume-light transport encoder");
                const MTLViewport viewport {0.0, 0.0, double(FogDimensions.AtlasWidth()), double(FogDimensions.AtlasHeight()), 0.0, 1.0};
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", inject ? injection : programs[final ? 1 : 0]);
                BindRayShadowImage(encoder);
                if (inject && pcss && !BindPCSS(encoder, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap))
                { Send<void>(encoder, "endEncoding"); return false; }
                if (inject && vsm && !BindVSM(encoder, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap))
                { Send<void>(encoder, "endEncoding"); return false; }
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                for (NSUInteger unit = 0; unit < inputs.size(); ++unit)
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[unit], unit);
                Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",density,6);
                lighting.Control.W = passes[index].PrefixStride;
                // Both blocks are below Metal's 4 KB inline-data limit and are
                // copied per encoder; later scan strides cannot mutate earlier work.
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &FogConstants, sizeof(FogConstants), 0);
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &lighting, sizeof(lighting), 1);
                Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setFragmentBuffer:offset:atIndex:",
                    LightListsActive ? ForwardPlusSceneBuffer : ForwardPlusEmptyBuffer, 0, ForwardPlusSceneIndex);
                constexpr NSUInteger FullscreenVertexCount = 3;
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, FullscreenVertexCount);
                Send<void>(encoder, "endEncoding");
            }
            // Device-created private textures use tracked hazards on this
            // MTLCommandQueue; encoder boundaries order the ping-pong dependencies.
            (Techniques::UsesVolumetricClouds(FrameFog) ? PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(FrameFog) ? PendingFrame.Statistics.ParticipatingMediaPassCount : PendingFrame.Statistics.VolumetricLightingPassCount) += static_cast<std::uint32_t>(passes.size());
            return true;
        }

        void DestroyLightShaftTargets()
        {
            ReleaseObject(LightShaftMask); LightShaftMask = nullptr;
            LightShaftWidth = LightShaftHeight = 0;
            LightShaftResolved = false;
        }

        bool EnsureLightShaftResources()
        {
            if (!LightShaftSupported || !HdrActive || !HdrTextures[0])
                return Fail("Metal light shafts require floating HDR color and sampled scene depth");
            if (!LightShaftPipeline)
                LightShaftPipeline = CreateTechniquePipeline("LightShafts.metal", "ShaftVertexMain", "ShaftFragmentMain",
                    1, {PixelFormatRGBA16Float}, false);
            if (!LightShaftPipeline) return false;
            if (LightShaftMask && LightShaftWidth == CreateInfo.Width && LightShaftHeight == CreateInfo.Height) return true;
            ObjCObject candidate = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans light-shaft occlusion mask");
            if (!candidate) return false;
            // No in-flight frame can reference the old texture here: Present
            // waits for completion, and AbortFrame releases unsubmitted commands.
            DestroyLightShaftTargets(); LightShaftMask = candidate;
            LightShaftWidth = CreateInfo.Width; LightShaftHeight = CreateInfo.Height;
            return true;
        }

        void DestroyFogTargets()
        {
            DestroyVolumeLightTargets();
            ReleaseObject(FogAtlas); FogAtlas = nullptr;
            FogDimensions = {};
            FogResolved = false;
        }

        bool EnsureFogResources()
        {
            if (!FogSupported || !HdrActive || !HdrTextures[0])
                return Fail("Metal fog requires a floating HDR scene and sampled depth");
            const bool media = Techniques::UsesBoundedMedium(FrameFog);
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& pipeline = cloud ? CloudFogPipeline : media ? MediaFogPipeline : FogPipeline;
            if (!pipeline)
            {
                const auto library = media ? MediaLibrary(true, false) : nullptr;
                if (media && !library) return false;
                pipeline = CreateTechniquePipeline(cloud ? "VolumetricClouds.metal" : media ? "ParticipatingMedia.metal" : "VolumetricFog.metal", "FogVertexMain", "FogFragmentMain",
                    1, {PixelFormatRGBA16Float}, false, ColorWriteMaskAll, false, library);
            }
            if (!pipeline) return false;
            const auto grid = BuildFogGrid(CreateInfo.Width,CreateInfo.Height,FrameFog,MaximumNativeTextureDimension);
            if (grid.Width == 0) return Fail("Metal fog atlas dimensions exceed the texture limit");
            if (FogAtlas && FogDimensions.Width == grid.Width && FogDimensions.Height == grid.Height &&
                FogDimensions.Slices == grid.Slices && FogDimensions.Planes == grid.Planes) return true;
            ObjCObject candidate = CreateNativeTexture(grid.AtlasWidth(), grid.AtlasHeight(), PixelFormatRGBA16Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans cumulative height medium");
            if (!candidate) return false;
            // BeginFrame follows completion or cancellation of the previous
            // command buffer. Replace the old atlas only after allocation succeeds.
            DestroyFogTargets(); FogAtlas = candidate; FogDimensions = grid;
            return true;
        }

        void DestroyTemporalTargets()
        {
            for (auto& texture : TemporalColors) { ReleaseObject(texture); texture = nullptr; }
            for (auto& texture : TemporalDepths) { ReleaseObject(texture); texture = nullptr; }
            TemporalHistoryReadIndex = 0;
            TemporalHistoryValid = false;
        }

        #include "Renderer/Metal/MetalLightmaps.inl"
    #include "Renderer/Metal/MetalRefraction.inl"
    #include "Renderer/Metal/MetalDepthPeeling.inl"

        void DestroyProbeTargets()
        {
            for (auto& texture : ProbeTextures) { ReleaseObject(texture); texture = nullptr; }
            ProbeCount = ProbeRayCount = ProbeWidth = ProbeHeight = 0;
            UploadedProbeSamples.reset(); Probes.Invalidate();
        }

        bool EnsureProbeResources()
        {
            if (!FrameProbeSamples || !HdrActive || !HdrTextures[0])
                return Fail("Metal light probes require baked ray samples and HDR color");
            const char* sources[] {"BakedGI.metal", "LightProbes.metal"};
            for (std::size_t i = 0; i < ProbePipelines.size(); ++i)
            {
                if (!ProbePipelines[i]) ProbePipelines[i] = CreateTechniquePipeline(sources[i], "VSMain", "PSMain",
                    1, {PixelFormatRGBA16Float}, false);
                if (!ProbePipelines[i]) return false;
            }
            const auto count = FrameProbeSamples->ProbeCount(), rays = FrameProbeSamples->Settings.RayCount;
            if (ProbeCount != count || ProbeRayCount != rays || ProbeWidth != CreateInfo.Width || ProbeHeight != CreateInfo.Height)
                DestroyProbeTargets();
            if (!ProbeTextures[ProbeRaw])
            {
                const int widths[] {int(rays + 1), int(ProbeSHCoefficientCount),
                    int(ProbeMomentTileSize * ProbeMomentAtlasColumns), CreateInfo.Width};
                const int heights[] {int(count + 1), int(count),
                    int(ProbeMomentTileSize * ((count + ProbeMomentAtlasColumns - 1) / ProbeMomentAtlasColumns)), CreateInfo.Height};
                for (std::size_t i = 0; i < ProbeTextures.size(); ++i)
                {
                    ProbeTextures[i] = CreateNativeTexture(widths[i], heights[i],
                        i == ProbeRaw ? PixelFormatRGBA32Float : PixelFormatRGBA16Float,
                        TextureUsageShaderRead | (i == ProbeRaw ? 0 : TextureUsageRenderTarget), "Evans baked light probes");
                    if (!ProbeTextures[i]) { DestroyProbeTargets(); return false; }
                }
                ProbeCount = count; ProbeRayCount = rays; ProbeWidth = CreateInfo.Width; ProbeHeight = CreateInfo.Height;
            }
            if (UploadedProbeSamples != FrameProbeSamples)
            {
                if (!UploadTexture(ProbeTextures[ProbeRaw], rays + 1, count + 1, sizeof(Float4), FrameProbeSamples->Texels.data()))
                    return false;
                UploadedProbeSamples = FrameProbeSamples; Probes.Invalidate();
            }
            return true;
        }

        bool ResolveProbes(const GlobalIlluminationDesc& desc)
        {
            LightProbeConstants constants;
            if (!FrameProbeSamples || !BuildLightProbeConstants(desc, *FrameProbeSamples, FrameGI, FrameReflectionShading,
                CreateInfo.Width, CreateInfo.Height, constants)) return Fail("Metal light probes require a finite invertible camera");
            const auto draw = [&](ProbePass phase)
            {
                const bool gather = phase == ProbePass::Gather;
                const auto target = gather ? ProbeComposite : phase == ProbePass::ProjectIrradiance ? ProbeSH : ProbeMoments;
                const unsigned width = gather ? CreateInfo.Width : target == ProbeSH ? ProbeSHCoefficientCount :
                    ProbeMomentTileSize * ProbeMomentAtlasColumns;
                const unsigned height = gather ? CreateInfo.Height : target == ProbeSH ? ProbeCount :
                    ProbeMomentTileSize * ((ProbeCount + ProbeMomentAtlasColumns - 1) / ProbeMomentAtlasColumns);
                constants.Options.X = static_cast<unsigned>(phase);
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(Send<ObjCObject>(descriptor, "colorAttachments"),
                    "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(attachment, "setTexture:", ProbeTextures[target]);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return Fail("Metal failed to create light probe encoder");
                const MTLViewport viewport {0, 0, double(width), double(height), 0, 1};
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", ProbePipelines[gather ? 1 : 0]);
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                const ObjCObject inputs[] {ProbeTextures[ProbeRaw], gather ? ProbeTextures[ProbeSH] : nullptr,
                    gather ? ProbeTextures[ProbeMoments] : nullptr, gather ? HdrTextures[0] : nullptr,
                    gather ? DepthTexture : nullptr, gather ? GBufferTextures[1] : nullptr,
                    gather ? GBufferTextures[0] : nullptr, gather ? GBufferTextures[3] : nullptr};
                for (NSUInteger i = 0; i < std::size(inputs); ++i)
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[i], i);
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
                Send<void>(encoder, "endEncoding");
                return true;
            };
            const bool project = Probes.NeedsProjection(FrameProbeSamples, FrameGI.Probes.MomentSharpness);
            if (project)
            {
                Probes.Invalidate();
                if (!draw(ProbePass::ProjectIrradiance) || !draw(ProbePass::ProjectDistance)) return false;
                Probes.Projected(FrameProbeSamples, FrameGI.Probes.MomentSharpness);
            }
            if (!draw(ProbePass::Gather)) return false;
            ObjCObject blit = Send<ObjCObject>(CommandBuffer, "blitCommandEncoder");
            if (!blit) return Fail("Metal failed to create light probe composite copy encoder");
            const MTLOrigin origin {0, 0, 0};
            const MTLSize size {static_cast<NSUInteger>(CreateInfo.Width), static_cast<NSUInteger>(CreateInfo.Height), 1};
            Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject, NSUInteger, NSUInteger, MTLOrigin>(blit,
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                ProbeTextures[ProbeComposite], 0, 0, origin, size, HdrTextures[0], 0, 0, origin);
            Send<void>(blit, "endEncoding");
            Probes.Resolved = GI.Resolved = true;
            PendingFrame.Statistics.GlobalIlluminationPassCount = project ? 3 : 1;
            return true;
        }

        void DestroyGITargets()
        {
            for (auto& texture : GITextures) { ReleaseObject(texture); texture = nullptr; }
            GI.Invalidate();
        }
        bool EnsureGIResources()
        {
            if (!TemporalSupported || !HdrActive || !HdrTextures[0])
                return Fail("Metal SSGI requires floating material and history targets");
            if (!GIPipeline) GIPipeline = CreateTechniquePipeline("SSGI.metal", "GIVertexMain", "GIFragmentMain",
                1, {PixelFormatRGBA16Float}, false);
            if (!GITemporalPipeline) GITemporalPipeline = CreateTechniquePipeline("SSGI.metal", "GIVertexMain", "GIHistoryFragment",
                1, {PixelFormatRGBA16Float, PixelFormatR32Float, PixelFormatRGBA16Float}, false);
            if (!GIPipeline || !GITemporalPipeline) return false;
            if (GITextures[0]) return true;
            for (std::size_t i = 0; i < GITextures.size(); ++i)
            {
                const bool depth = i >= GIDepthFirst && i < GINormalFirst;
                GITextures[i] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height,
                    depth ? PixelFormatR32Float : PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans screen-space diffuse transport");
                if (!GITextures[i]) { DestroyGITargets(); return false; }
            }
            GI.Invalidate(); return true;
        }

        bool EnsureTemporalResources()
        {
            if (!TemporalSupported || !HdrActive || !HdrTextures[0])
                return Fail("Metal TAA requires renderable color/depth history targets and HDR color");
            if (!TemporalPipeline)
                TemporalPipeline = CreateTechniquePipeline("TAA.metal", "TemporalVertexMain", "TemporalFragmentMain",
                    1, {PixelFormatRGBA16Float, PixelFormatR32Float}, false);
            if (!TemporalPipeline) return false;
            if (TemporalColors[0]) return true;
            for (std::size_t index = 0; index < TemporalColors.size(); ++index)
            {
                TemporalColors[index] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans temporal color history");
                TemporalDepths[index] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatR32Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans temporal depth history");
                if (!TemporalColors[index] || !TemporalDepths[index]) { DestroyTemporalTargets(); return false; }
            }
            return true;
        }

        bool PrepareEnvironmentCapture(const EnvironmentCaptureFrame& frame)
        {
            EnvironmentActive = Techniques::UsesEnvironmentTexture(FrameReflections);
            EnvironmentReady = false; EnvironmentFacesWritten = 0; FrameEnvironment = {};
            if (!EnvironmentActive) return true;
            if (!HdrActive || !ValidEnvironmentSource(frame) || (!frame.Source && frame.Resolution != FrameReflections.CaptureResolution) || !frame.Resolution)
                return Fail("Metal capture requires HDR and the frame's requested face resolution");
            std::uint32_t expectedMips = 0;
            for (auto extent = frame.Resolution; extent; extent >>= 1) ++expectedMips;
            if (frame.MipCount != expectedMips) return Fail("Metal capture requires the full mip chain");
            for (const auto& face : frame.Faces)
                for (float value : face.ViewProjection.Values)
                    if (!std::isfinite(value)) return Fail("Metal capture requires finite face cameras");
            if (!EnvironmentTexture || !EnvironmentDepth || EnvironmentResolution != frame.Resolution)
            {
                // A cube has arrayLength one; render-pass slice selects one of its
                // six faces. Keep depth single-sample and private to the capture.
                ObjCObject cube = CreateNativeTexture(frame.Resolution, frame.Resolution, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal scene cube", frame.MipCount, 1, true);
                ObjCObject depth = cube ? CreateNativeTexture(frame.Resolution, frame.Resolution, PixelFormatDepth32Float,
                    TextureUsageRenderTarget, "Evans Metal cube depth") : nullptr;
                if (!cube || !depth) { ReleaseObject(cube); ReleaseObject(depth); return false; }
                ReleaseObject(EnvironmentTexture); ReleaseObject(EnvironmentDepth);
                EnvironmentTexture = cube; EnvironmentDepth = depth; EnvironmentResolution = frame.Resolution;
            }
            FrameEnvironment = frame;
            return true;
        }

        bool EnsureEnvironmentMapTexture()
        {
            const auto layout = BuildEnvironmentMapLayout(FrameEnvironment.Resolution);
            if (!layout.MipCount) return Fail("Metal environment map requires valid capture dimensions");
            if (EnvironmentMapTexture && EnvironmentMapDimensions.Width == layout.Width) return true;
            ObjCObject candidate = CreateNativeTexture(layout.Width, layout.Height, PixelFormatRGBA16Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal latitude-longitude radiance", layout.MipCount);
            if (!candidate) return false;
            ReleaseObject(EnvironmentMapTexture);
            EnvironmentMapTexture = candidate;
            EnvironmentMapDimensions = layout;
            return true;
        }

        bool ConvertEnvironmentMap()
        {
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject color = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(color, "setTexture:", EnvironmentMapTexture);
            Send<void, NSUInteger>(color, "setLevel:", 0);
            Send<void, NSUInteger>(color, "setLoadAction:", LoadActionClear);
            Send<void, NSUInteger>(color, "setStoreAction:", StoreActionStore);
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return Fail("Metal failed to create latitude-longitude conversion encoder");
            const auto width = EnvironmentMapDimensions.Width, height = EnvironmentMapDimensions.Height;
            const MTLViewport viewport {0, 0, double(width), double(height), 0, 1};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", EnvironmentConversionPipeline);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", EnvironmentTexture, 0);
            EnvironmentMapReflectionConstants constants {};
            constants.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
            // setFragmentBytes snapshots this draw. The color-only destination is
            // separate from the sampled cube; no previous main-view state leaks in.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
            ++PendingFrame.Statistics.EnvironmentConversionPassCount;
            return true;
        }

        bool PreparePlanarCapture(const PlanarCaptureFrame& frame)
        {
            PlanarActive = Techniques::UsesPlanarCapture(FrameReflections);
            PlanarWritten = PlanarReady = false;
            FramePlanar = {};
            if (!PlanarActive) return true;
            PlanarReflectionConstants constants{};
            ReflectionDesc desc; desc.ViewProjection = frame.MainViewProjection;
            if (!HdrActive || !BuildPlanarReflectionConstants(desc, frame, FrameReflections, FrameReflectionShading,
                CreateInfo.Width, CreateInfo.Height, constants))
                return Fail("Metal planar capture requires a finite camera matching the main viewport before BeginFrame");
            const auto divisor = FrameReflections.Planar.ResolutionDivisor;
            std::uint32_t expectedMips = 0;
            for (auto extent = std::max(frame.Width, frame.Height); extent; extent >>= 1) ++expectedMips;
            if (frame.Width != (frame.MainWidth + divisor - 1) / divisor ||
                frame.Height != (frame.MainHeight + divisor - 1) / divisor || frame.MipCount != expectedMips)
                return Fail("Metal planar extent or mip chain disagrees with frame settings");
            for (float value : frame.ClippedViewProjection.Values)
                if (!std::isfinite(value)) return Fail("Metal planar clipping projection must be finite");
            if (PlanarWidth != frame.Width || PlanarHeight != frame.Height || !PlanarTexture || !PlanarDepth || !PlanarScratch)
            {
                ObjCObject image = CreateNativeTexture(frame.Width, frame.Height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal planar capture", frame.MipCount);
                ObjCObject depth = image ? CreateNativeTexture(frame.Width, frame.Height, PixelFormatDepth32Float,
                    TextureUsageRenderTarget, "Evans Metal planar depth") : nullptr;
                ObjCObject scratch = depth ? CreateNativeTexture(std::max(1u, frame.Width >> 1), std::max(1u, frame.Height >> 1),
                    PixelFormatRGBA16Float, TextureUsageRenderTarget, "Evans Metal planar mip scratch") : nullptr;
                if (!image || !depth || !scratch)
                {
                    ReleaseObject(image); ReleaseObject(depth); ReleaseObject(scratch);
                    return false;
                }
                ReleaseObject(PlanarTexture); ReleaseObject(PlanarDepth); ReleaseObject(PlanarScratch);
                PlanarTexture = image; PlanarDepth = depth; PlanarScratch = scratch;
                PlanarWidth = frame.Width; PlanarHeight = frame.Height;
            }
            FramePlanar = frame;
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
            if (!HdrActive || !GBufferTextures[0]) return Fail("Metal reflections require HDR and material targets");
            const bool latitudeLongitude = !planarPass && FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
            auto& selected = planarPass ? PlanarPipelines : latitudeLongitude ? EnvironmentMapPipelines : FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? CubeReflectionPipelines : ReflectionPipelines;
            const bool auxiliary = latitudeLongitude || planarPass;
            if (latitudeLongitude && !EnsureEnvironmentMapTexture()) return false;
            if (!selected[0] || !selected[1] || (latitudeLongitude && !EnvironmentConversionPipeline) || (planarPass && !PlanarMipPipeline))
            {
                std::string source;
                if (!LoadTechniqueShaderSource("Metal", planarPass ? "PlanarReflections.metal" : latitudeLongitude ? "EnvironmentMapReflections.metal" :
                    FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? "CubemapReflections.metal" : "SSR.metal", source, LastError)) return false;
                ObjCObject error = nullptr;
                ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
                if (!library) return Fail("Metal reflection compilation failed: " + ErrorDescription(error));
                ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString(planarPass ? "PlanarVertexMain" : latitudeLongitude ? "EnvironmentVertexMain" : FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? "CubeVertexMain" : "SSRVertexMain"));
                ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString(planarPass ? "PlanarFragmentMain" : latitudeLongitude ? "EnvironmentFragmentMain" : FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? "CubeFragmentMain" : "SSRFragmentMain"));
                ObjCObject conversion = auxiliary ? Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:",
                    MakeString(planarPass ? "PlanarMipMain" : "EnvironmentConvertMain")) : nullptr;
                if (!vertex || !fragment || (auxiliary && !conversion))
                {
                    ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(conversion); ReleaseObject(library);
                    return Fail("Metal reflection entry point is missing");
                }
                constexpr std::size_t AuxiliaryPipelineIndex = 2;
                std::array<ObjCObject, AuxiliaryPipelineIndex + 1> pipelines{};
                const auto count = auxiliary ? pipelines.size() : selected.size();
                for (std::size_t index = 0; index < count; ++index)
                {
                    const bool composite = index == static_cast<std::size_t>(Techniques::ReflectionPassKind::Composite);
                    ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
                    Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
                    Send<void, ObjCObject>(descriptor, "setFragmentFunction:", index == AuxiliaryPipelineIndex ? conversion : fragment);
                    ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                    ObjCObject color = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                    Send<void, NSUInteger>(color, "setPixelFormat:", PixelFormatRGBA16Float);
                    Send<void, NSUInteger>(color, "setWriteMask:", composite ? ColorWriteMaskRGB : ColorWriteMaskAll);
                    Send<void, ObjCBool>(color, "setBlendingEnabled:", composite ? ObjCYes : ObjCNo);
                    if (composite)
                    {
                        Send<void, NSUInteger>(color, "setSourceRGBBlendFactor:", BlendFactorOne);
                        Send<void, NSUInteger>(color, "setDestinationRGBBlendFactor:", BlendFactorOne);
                        Send<void, NSUInteger>(color, "setRgbBlendOperation:", BlendOperationAdd);
                    }
                    error = nullptr;
                    pipelines[index] = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
                    ReleaseObject(descriptor);
                    if (!pipelines[index]) break;
                }
                ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(conversion); ReleaseObject(library);
                if (!pipelines[0] || !pipelines[1] || (auxiliary && !pipelines[AuxiliaryPipelineIndex]))
                {
                    for (auto pipeline : pipelines) ReleaseObject(pipeline);
                    return Fail("Metal reflection pipeline creation failed: " + ErrorDescription(error));
                }
                for (auto pipeline : selected) ReleaseObject(pipeline);
                selected = {pipelines[0], pipelines[1]};
                if (planarPass)
                {
                    ReleaseObject(PlanarMipPipeline);
                    PlanarMipPipeline = pipelines[AuxiliaryPipelineIndex];
                }
                else if (latitudeLongitude)
                {
                    ReleaseObject(EnvironmentConversionPipeline);
                    EnvironmentConversionPipeline = pipelines[AuxiliaryPipelineIndex];
                }
            }
            if (!ReflectionTexture)
                ReflectionTexture = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRGBA16Float,
                    TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal reflection radiance");
            return ReflectionTexture != nullptr;
        }

        bool EnsureDepthOfFieldPipelines()
        {
            if (DepthOfFieldPipelines[0] && DepthOfFieldPipelines[1]) return true;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "DepthOfField.metal", source, LastError)) return false;
            ObjCObject error = nullptr;
            ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) return Fail("Metal depth-of-field compilation failed: " + ErrorDescription(error));
            ObjCObject vertex = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString("DepthOfFieldVertexMain"));
            ObjCObject fragment = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString("DepthOfFieldFragmentMain"));
            if (!vertex || !fragment)
            {
                ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
                return Fail("Metal depth-of-field entry point is missing");
            }
            std::array<ObjCObject, 2> pipelines{};
            for (std::size_t index = 0; index < pipelines.size(); ++index)
            {
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"), "new");
                Send<void, ObjCObject>(descriptor, "setVertexFunction:", vertex);
                Send<void, ObjCObject>(descriptor, "setFragmentFunction:", fragment);
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                Send<void, NSUInteger>(attachment, "setPixelFormat:", PixelFormatRGBA16Float);
                Send<void, NSUInteger>(attachment, "setWriteMask:", index == 1 ? ColorWriteMaskRGB : ColorWriteMaskAll);
                Send<void, ObjCBool>(attachment, "setBlendingEnabled:", ObjCNo);
                pipelines[index] = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
                ReleaseObject(descriptor);
                if (!pipelines[index]) break;
            }
            ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
            if (!pipelines[0] || !pipelines[1])
            {
                for (auto pipeline : pipelines) ReleaseObject(pipeline);
                return Fail("Metal depth-of-field pipeline failed: " + ErrorDescription(error));
            }
            for (auto pipeline : DepthOfFieldPipelines) ReleaseObject(pipeline);
            DepthOfFieldPipelines = pipelines;
            return true;
        }

        bool ApplyHdr(const Techniques::HdrSettings& settings)
        {
            const auto plan = Techniques::BuildHdrPlan(settings);
            if (plan.empty()) return true;
            ObjCObject scene = HdrTextures[0];
            using Pass = Techniques::HdrPassKind;
            for (const auto pass : plan)
            {
                const auto index = HdrShaderIndex(pass);
                if (!HdrPipelines[index])
                {
                    const std::string filename = std::string(HdrShaderFilename(pass)) + ".metal";
                    HdrPipelines[index] = CreateFullscreenPipeline(filename.c_str(), true, pass != Pass::ToneMapping);
                }
                if (!HdrPipelines[index]) return false;
                ObjCObject source = scene;
                ObjCObject destination = scene == HdrTextures[0] ? HdrTextures[1] : HdrTextures[0];
                int width = CreateInfo.Width, height = CreateInfo.Height;
                int sourceWidth = width, sourceHeight = height;
                const auto resources = DescribeHdrPass(pass);
                constexpr std::size_t FirstHalfTarget = 2;
                if (resources.DestinationHalf >= 0)
                {
                    width = (width + 1) / 2; height = (height + 1) / 2;
                    destination = HdrTextures[FirstHalfTarget + resources.DestinationHalf];
                    if (resources.SourceHalf >= 0)
                    {
                        source = HdrTextures[FirstHalfTarget + resources.SourceHalf];
                        sourceWidth = width; sourceHeight = height;
                    }
                }
                else if (pass == Pass::ToneMapping) destination = ColorTexture;
                const HdrConstants constants = BuildHdrConstants(settings, sourceWidth, sourceHeight, pass, CreateInfo.Width, CreateInfo.Height);
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(attachment, "setTexture:", destination);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return Fail("Metal failed to create HDR encoder");
                const MTLViewport viewport{0.0, 0.0, static_cast<double>(width), static_cast<double>(height), 0.0, 1.0};
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", HdrPipelines[index]);
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", source, 0);
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:",
                    resources.CompositeHalf >= 0 ? HdrTextures[FirstHalfTarget + resources.CompositeHalf] : source, 1);
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
                Send<void>(encoder, "endEncoding");
                if (resources.UpdatesScene) scene = destination;
                PendingFrame.HDRPasses.push_back(pass);
            }
            PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
            PendingFrame.Statistics.PassCount += static_cast<std::uint32_t>(plan.size());
            return true;
        }

        bool ApplyPostProcess(const Techniques::PostProcessSettings& settings, ObjCObject& resultTexture)
        {
            resultTexture = ColorTexture;
            const auto plan = Techniques::BuildPostProcessPlan(settings);
            if (plan.empty()) return true;
            if (!EnsurePostTargets()) return false;
            const PostProcessConstants constants = BuildPostProcessConstants(settings, CreateInfo.Width, CreateInfo.Height);
            std::size_t destination = 0;
            for (const auto pass : plan)
            {
                ObjCObject pipeline = GetPostPipeline(pass);
                if (!pipeline) return false;
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(attachment, "setTexture:", PostTextures[destination]);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return Fail("Metal failed to create post-process encoder");
                const MTLViewport viewport{0.0, 0.0, static_cast<double>(CreateInfo.Width), static_cast<double>(CreateInfo.Height), 0.0, 1.0};
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", pipeline);
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", resultTexture, 0);
                if (pass == Techniques::PostProcessPassKind::DepthNormalOutline)
                {
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", DepthTexture, 1);
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", GBufferTextures[1], 2);
                }
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
                Send<void>(encoder, "endEncoding");
                resultTexture = PostTextures[destination];
                destination ^= 1U;
                PendingFrame.PostProcessingPasses.push_back(pass);
            }
            PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
            PendingFrame.Statistics.PassCount += static_cast<std::uint32_t>(plan.size());
            return true;
        }

		void ReleaseResources()
		{
			if (RenderEncoder != nullptr)
			{
				Send<void>(RenderEncoder, "endEncoding");
				RenderEncoder = nullptr;
			}
            ContactProducer.Shutdown();
            ContactSupported = false;
            for (auto& entry : ContactLibraries) ReleaseObject(entry.second);
            for (auto& entry : ContactDepthPipelines) ReleaseObject(entry.second);
            for (auto& entry : ContactSurfacePipelines) ReleaseObject(entry.second);
            for (auto& entry : ContactLightingPipelines) ReleaseObject(entry.second);
            ContactLibraries.clear(); ContactDepthPipelines.clear(); ContactSurfacePipelines.clear(); ContactLightingPipelines.clear();
			if (CommandBuffer)
            {
                const auto status = Send<NSUInteger>(CommandBuffer, "status");
                if (status >= CommandStatusCommitted && status < CommandStatusCompleted)
                    Send<void>(CommandBuffer, "waitUntilCompleted");
            }
			ReleaseObject(CommandBuffer);
			CommandBuffer = nullptr;
            std::string forwardPlusError;
            ForwardPlusProducer.AbortFrame(forwardPlusError); TiledProducer.AbortFrame(forwardPlusError);
            ForwardPlusProducer.Shutdown(forwardPlusError); TiledProducer.Shutdown(forwardPlusError);
            ClusteredProducer.AbortFrame(forwardPlusError); ClusteredProducer.Shutdown(forwardPlusError);
            ForwardPlusSupported = ClusteredSupported = TiledSupported = false;
            ReleaseObject(ForwardPlusSceneBuffer); ReleaseObject(ForwardPlusEmptyBuffer);
            ForwardPlusSceneBuffer = ForwardPlusEmptyBuffer = nullptr;
            for (auto& entry : ForwardPlusPipelines) ReleaseObject(entry.second);
            for (auto& entry : ForwardPlusLibraries) ReleaseObject(entry.second);
            ForwardPlusPipelines.clear(); ForwardPlusLibraries.clear();
            for (auto& entry : TiledLightingPipelines) ReleaseObject(entry.second);
            for (auto& entry : TiledLightingLibraries) ReleaseObject(entry.second);
            TiledLightingPipelines.clear(); TiledLightingLibraries.clear();
			if (FramePool != nullptr)
			{
				Send<void>(FramePool, "drain");
				FramePool = nullptr;
			}
			for (auto& pair : Buffers) ReleaseObject(pair.second.Buffer);
			for (auto& pair : Textures) ReleaseObject(pair.second.Texture);
			Buffers.clear();
			Textures.clear();
            for (auto& page : InstancePages) ReleaseObject(page.Buffer);
            InstancePages.clear();
            DestroyGpuDriven();
            DestroyVisibilityBuffer();
            for (auto& entry : InstancePipelines) ReleaseObject(entry.second);
            InstancePipelines.clear();
            ReleaseObject(InstanceLibrary); InstanceLibrary = nullptr;
			ReleaseObject(DummyShadowTexture);
			ReleaseObject(DummyColorTexture);
			ReleaseObject(SurfaceSampler);
			ReleaseObject(AnisotropicSampler);
			ReleaseObject(ColorTexture);
			ReleaseObject(DepthTexture);
            DestroyMultisampleTargets(Multisample);
            DestroyPostTargets();
            DestroyHdrTargets();
            DestroyDeferredTargets();
            DestroyAmbientOcclusionTargets();
            DestroyPathTracing();
        DestroyRayGI(); DestroyVoxelGI();
            DestroyRayReflections(); DestroyHybridRendering(); DestroySubsurface(); DestroyHairFur(); DestroyExposure(); DestroyParallax(); DestroyDisplacement(); DestroyTessellation(); DestroyVirtualGeometry(); DestroyNaniteFrameVisibility(); DestroyNaniteGeometry(); DestroyTSR(); DestroyVirtualShadowRequests(); DestroyVirtualShadowAtlases(); DestroyLumenProbes(); DestroyLumenRadianceTraces(); DestroyLumenLightingCaches(); DestroyLumenSurfaces(); DestroyLumenSurfaceLookup(); DestroyLumenDistanceScenes();
            DestroyRayScene(); DestroyRayShadowImage(); DestroyPCF(); DestroyMaterials();
            for (auto& pipeline : SsaoPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : HbaoPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : SaoPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : GtaoPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : DeferredGeometryPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            ReleaseObject(DeferredLightingPipeline);
            DeferredLightingPipeline = nullptr;
            ReleaseObject(CascadeDepthPipeline);
            CascadeDepthPipeline = nullptr;
            ReleaseObject(ShadowPipeline);
            ReleaseObject(ShadowLibrary);
            ShadowPipeline = ShadowLibrary = nullptr;
            FrameCascades = {};
            CascadeTextures.fill(nullptr);
            for (auto& entry : SampleGroups)
            {
                for (auto pipeline : entry.second.Resolve) ReleaseObject(pipeline);
                for (auto pipeline : entry.second.Deferred) ReleaseObject(pipeline);
                for (auto& row : entry.second.Shading) for (auto pipeline : row) ReleaseObject(pipeline);
                ReleaseObject(entry.second.Velocity);
            }
            SampleGroups.clear();
            DestroyWeightedTargets();
            for (const auto& entry : WeightedGroups)
            {
                for (auto pipeline : entry.second.Surfaces) ReleaseObject(pipeline);
                ReleaseObject(entry.second.Resolve);
            }
            WeightedGroups.clear();
            WeightedSampleCounts = 0;
            for (auto& entry : NamedTechniqueLibraries) ReleaseObject(entry.second);
            NamedTechniqueLibraries.clear();
            for (auto& pair:RayShadowSurfacePipelines) ReleaseObject(pair.second);
            for (auto& pair:RayShadowScreenPipelines) ReleaseObject(pair.second);
            RayShadowSurfacePipelines.clear(); RayShadowScreenPipelines.clear();
            for (auto& library:RayShadowLibraries) { ReleaseObject(library); library=nullptr; }
            for (auto& pair : PCSSSurfacePipelines) ReleaseObject(pair.second);
            for (auto& pair : PCSSScreenPipelines) ReleaseObject(pair.second);
            PCSSSurfacePipelines.clear(); PCSSScreenPipelines.clear();
            for (auto& library : PCSSLibraries) { ReleaseObject(library); library = nullptr; }
            for (auto& pair : VSMSurfacePipelines) ReleaseObject(pair.second);
            for (auto& pair : VSMScreenPipelines) ReleaseObject(pair.second);
            VSMSurfacePipelines.clear(); VSMScreenPipelines.clear();
            for (auto& pipeline : VSMCasterPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& library : VSMLibraries) { ReleaseObject(library); library = nullptr; }
            for (auto& entry : VSMMoments) DestroyMomentMap(entry.second);
            VSMMoments.clear(); VSMSupported = false;
            ReleaseObject(SampleDepthWriteState); SampleDepthWriteState = nullptr;
            SupportedSampleCounts = 1;
            DeferredSupported = false;
            FloatDepthTargetSupported = TemporalSupported = FogSupported = false;
            VolumeLightSupported = false;
            LightShaftSupported = false;
            for (auto& pipeline : HdrPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : ReflectionPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : CubeReflectionPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : EnvironmentMapPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            ReleaseObject(EnvironmentConversionPipeline); EnvironmentConversionPipeline = nullptr;
            ReleaseObject(EnvironmentMapTexture); EnvironmentMapTexture = nullptr;
            EnvironmentMapDimensions = {};
            ReleaseObject(EnvironmentTexture); ReleaseObject(EnvironmentDepth);
            EnvironmentTexture = EnvironmentDepth = nullptr; EnvironmentResolution = 0;
            for (auto& pipeline : PlanarPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            ReleaseObject(PlanarMipPipeline); PlanarMipPipeline = nullptr;
            ReleaseObject(PlanarTexture); ReleaseObject(PlanarDepth); ReleaseObject(PlanarScratch);
            PlanarTexture = PlanarDepth = PlanarScratch = nullptr; PlanarWidth = PlanarHeight = 0;
            for (auto& pipeline : DepthOfFieldPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : MotionPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            ReleaseObject(TemporalPipeline); TemporalPipeline = nullptr;
            ReleaseObject(GIPipeline); GIPipeline = nullptr;
            ReleaseObject(GITemporalPipeline); GITemporalPipeline = nullptr;
            for (auto& pipeline : ProbePipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            DestroyLightmapTargets(true); DestroyRefractionTargets(true); DestroyPeelingTargets(true);
            FrameProbeSamples.reset();
            ReleaseObject(FogPipeline); FogPipeline = nullptr;
            ReleaseObject(CloudFogPipeline); CloudFogPipeline = nullptr;
            for (auto& group : CloudVolumePipelines)
                for (auto& pipeline : group) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& library : CloudLibraries) { ReleaseObject(library); library = nullptr; }
            ReleaseObject(MediaFogPipeline); MediaFogPipeline = nullptr;
            for (auto& group : MediaVolumePipelines)
                for (auto& pipeline : group) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& library : MediaLibraries) { ReleaseObject(library); library = nullptr; }
            ReleaseObject(LightShaftPipeline); LightShaftPipeline = nullptr;
            for (auto& pipeline : VolumeLightPipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipelines : HdrTechniquePipelines)
                for (auto& pipeline : pipelines) { ReleaseObject(pipeline); pipeline = nullptr; }
            for (auto& pipeline : PostPipelines)
            {
                ReleaseObject(pipeline);
                pipeline = nullptr;
            }
            for (auto& pipelines : TechniquePipelines)
                for (auto& pipeline : pipelines)
                {
                    ReleaseObject(pipeline);
                    pipeline = nullptr;
                }
            for (auto& library : TechniqueLibraries)
            {
                ReleaseObject(library);
                library = nullptr;
            }
			ReleaseObject(DepthWriteState);
			ReleaseObject(DepthReadState);
			ReleaseObject(DepthDisabledState);
            GpuSurfaces.clear();
			ReleaseObject(CommandQueue);
			ReleaseObject(Device);
			DummyShadowTexture = nullptr;
			DummyColorTexture = nullptr;
			SurfaceSampler = AnisotropicSampler = nullptr;
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

        void ReadMotionVectors(ObjCObject buffer, std::size_t rowPitch)
        {
            std::vector<std::array<std::uint16_t, 4>> packed;
            CopyReadback(buffer, rowPitch, CreateInfo.Width, CreateInfo.Height, packed);
            const auto decode = [](std::uint16_t bits)
            {
                constexpr unsigned FractionBits = 10, ExponentBias = 15;
                constexpr unsigned FractionMask = 0x3ff, ExponentMask = 0x1f, SignMask = 0x8000;
                const unsigned exponent = (bits >> FractionBits) & ExponentMask;
                const unsigned fraction = bits & FractionMask;
                float value = exponent == 0 ? std::ldexp(float(fraction), 1 - int(ExponentBias + FractionBits)) :
                    exponent == ExponentMask ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
                    std::ldexp(1.0f + float(fraction) / float(1U << FractionBits), int(exponent) - int(ExponentBias));
                return (bits & SignMask) ? -value : value;
            };
            PendingFrame.MotionVectors.resize(packed.size());
            for (std::size_t pixel = 0; pixel < packed.size(); ++pixel)
                PendingFrame.MotionVectors[pixel] = DecodeMotionVector({decode(packed[pixel][0]), decode(packed[pixel][1]),
                    decode(packed[pixel][2]), decode(packed[pixel][3])});
        }

		RHICreateInfo CreateInfo;
		ObjCObject Device = nullptr;
		ObjCObject CommandQueue = nullptr;
        std::array<ObjCObject, StandaloneShadingTechniqueCount> TechniqueLibraries{};
        std::array<std::array<ObjCObject, 3>, StandaloneShadingTechniqueCount> TechniquePipelines{};
        std::array<std::array<ObjCObject, 3>, StandaloneShadingTechniqueCount> HdrTechniquePipelines{};
        std::array<ObjCObject, 4> HdrTextures{};
        std::array<ObjCObject, HdrTechniqueCount> HdrPipelines{};
        bool HdrActive = false;
        static constexpr std::size_t MotionVelocity = 0, MotionTileMax = 1, MotionNeighborMax = 2;
        static constexpr std::size_t MotionGeometryPipeline = 0, MotionFilterPipeline = 1, MotionCompositePipeline = 2;
        std::array<ObjCObject, 3> MotionTextures {}, MotionPipelines {};
        bool MotionBlurActive = false, VelocityCaptured = false, MotionBlurResolved = false;
        bool VelocityActive = false, TemporalActive = false, TemporalResolved = false;
        bool TemporalHistoryValid = false, TemporalSupported = false;
        static constexpr std::size_t GIRaw = 0, GIFiltered = 1, GIComposite = 2, GIColorFirst = 3,
            GIDepthFirst = 5, GINormalFirst = 7, GITargetCount = 9;
        std::array<ObjCObject, GITargetCount> GITextures {};
        ObjCObject GIPipeline = nullptr, GITemporalPipeline = nullptr;
        GlobalIlluminationHistory GI;
        Techniques::GlobalIlluminationSettings FrameGI;
        bool GIActive = false, EarlyTemporalActive = false, ProbeActive = false;
        static constexpr std::size_t ProbeRaw = 0, ProbeSH = 1, ProbeMoments = 2, ProbeComposite = 3;
        std::array<ObjCObject, 4> ProbeTextures {};
        std::array<ObjCObject, 2> ProbePipelines {};
        ProbeProjectionState Probes;
        std::shared_ptr<const BakedProbeSamples> FrameProbeSamples, UploadedProbeSamples;
        unsigned ProbeCount = 0, ProbeRayCount = 0;
        int ProbeWidth = 0, ProbeHeight = 0;
        std::size_t TemporalHistoryReadIndex = 0;
        std::array<ObjCObject, 2> TemporalColors {}, TemporalDepths {};
        ObjCObject TemporalPipeline = nullptr;
        bool LightShaftActive = false, LightShaftResolved = false, LightShaftSupported = false;
        Techniques::LightShaftSettings FrameLightShafts;
        ObjCObject LightShaftMask = nullptr, LightShaftPipeline = nullptr;
        int LightShaftWidth = 0, LightShaftHeight = 0;
        bool FogActive = false, FogResolved = false, FogSupported = false;
        Techniques::VolumetricFogSettings FrameFog;
        Techniques::VolumetricLightingSettings FrameVolumeLighting;
        bool VolumeLightSupported = false;
        std::array<ObjCObject, 2> VolumeLightTextures {}, VolumeLightPipelines {};
        std::array<ObjCObject,8> RayShadowLibraries {};
        std::unordered_map<std::uint32_t,ObjCObject> RayShadowSurfacePipelines,RayShadowScreenPipelines;
        std::array<ObjCObject, 8> PCSSLibraries {};
        std::array<ObjCObject, 10> VSMLibraries {};
        std::array<ObjCObject, 2> VSMCasterPipelines {};
        std::unordered_map<std::uint32_t, ObjCObject> VSMSurfacePipelines, VSMScreenPipelines;
        std::unordered_map<std::uint32_t, VSMMomentMap> VSMMoments;
        bool VSMSupported = false;
        std::unordered_map<std::uint32_t, ObjCObject> PCSSSurfacePipelines, PCSSScreenPipelines;
        VolumetricFogConstants FogConstants;
        FogGrid FogDimensions;
        ObjCObject FogAtlas = nullptr, FogPipeline = nullptr, MediaFogPipeline = nullptr, CloudFogPipeline = nullptr;
        std::array<ObjCObject, 3> MediaLibraries {}, CloudLibraries {};
        std::array<std::array<ObjCObject, 2>, 2> MediaVolumePipelines {}, CloudVolumePipelines {};
        std::array<ObjCObject, 2> DepthOfFieldPipelines{};
        bool DepthOfFieldActive = false;
        bool DepthOfFieldResolved = false;
        Techniques::PostProcessSettings FramePostProcessSettings{};
        std::array<ObjCObject, GBufferAttachmentCount> GBufferTextures{};
        std::array<ObjCObject, 2> DeferredGeometryPipelines{};
        ObjCObject DeferredLightingPipeline = nullptr;
        bool DeferredSupported = false;
        bool DeferredActive = false;
        std::array<ObjCObject, 2> AmbientOcclusionTextures{};
        std::array<ObjCObject, 4> SsaoPipelines{};
        std::array<ObjCObject, 4> HbaoPipelines{};
        std::array<ObjCObject, 4> SaoPipelines{};
        std::array<ObjCObject, 4> GtaoPipelines{};
        ObjCObject SaoDepthHierarchy = nullptr;
        ObjCObject SaoDepthScratch = nullptr;
        bool FloatDepthTargetSupported = false;
        Techniques::AmbientOcclusionSettings FrameAmbientOcclusionSettings{};
        bool AmbientOcclusionActive = false;
        bool AmbientOcclusionResolved = false;
        ObjCObject ReflectionTexture = nullptr;
        std::array<ObjCObject, 2> ReflectionPipelines{}, CubeReflectionPipelines{};
        std::array<ObjCObject, 2> EnvironmentMapPipelines{};
        ObjCObject EnvironmentConversionPipeline = nullptr, EnvironmentMapTexture = nullptr;
        EnvironmentMapLayout EnvironmentMapDimensions{};
        ObjCObject EnvironmentTexture = nullptr, EnvironmentDepth = nullptr;
        std::uint32_t EnvironmentResolution = 0, EnvironmentFacesWritten = 0;
        EnvironmentCaptureFrame FrameEnvironment {};
        bool EnvironmentActive = false, EnvironmentReady = false;
        std::array<ObjCObject, 2> PlanarPipelines{};
        ObjCObject PlanarMipPipeline = nullptr;
        ObjCObject PlanarTexture = nullptr, PlanarDepth = nullptr, PlanarScratch = nullptr;
        std::uint32_t PlanarWidth = 0, PlanarHeight = 0;
        PlanarCaptureFrame FramePlanar{};
        bool PlanarActive = false, PlanarWritten = false, PlanarReady = false;
        bool ReflectionActive = false, ReflectionResolved = false;
        Techniques::ReflectionSettings FrameReflections{};
        Techniques::ShadingModel FrameReflectionShading = Techniques::ShadingModel::Phong;
        bool ForwardColorWritten = false;
        bool GBufferWritten = false;
        bool DeferredResolved = false;
        Float4 DeferredBackground;
		ObjCObject Library = nullptr;
        ObjCObject InstanceLibrary = nullptr;
        std::unordered_map<std::uint32_t, ObjCObject> InstancePipelines;
        std::vector<InstanceUploadPage> InstancePages;
		ObjCObject ColorPipeline = nullptr;
		ObjCObject ColorBlendPipeline = nullptr;
		ObjCObject ColorWriteDisabledPipeline = nullptr;
		ObjCObject ShadowPipeline = nullptr;
        ObjCObject ShadowLibrary = nullptr;
        ObjCObject CascadeDepthPipeline = nullptr;
        CascadedShadowState FrameCascades {};
        Techniques::ShadowFilterSettings FrameShadowFilter;
        std::array<ObjCObject, MaximumShadowCascades> CascadeTextures {};
		ObjCObject DepthWriteState = nullptr;
		ObjCObject DepthReadState = nullptr;
		ObjCObject DepthDisabledState = nullptr;
		ObjCObject ColorTexture = nullptr;
        std::array<ObjCObject, 2> PostTextures{};
        std::array<ObjCObject, PostProcessTechniqueCount> PostPipelines{};
		ObjCObject DepthTexture = nullptr;
		ObjCObject DummyShadowTexture = nullptr;
		ObjCObject DummyColorTexture = nullptr;
		ObjCObject SurfaceSampler = nullptr, AnisotropicSampler = nullptr;
		bool AnisotropicFiltering = false;
		ObjCObject MaterialSampler() const { return AnisotropicFiltering ? AnisotropicSampler : SurfaceSampler; }
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
		bool FrameAreaLights = false;
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
			|| createInfo.Width > MaximumNativeTextureDimension || createInfo.Height > MaximumNativeTextureDimension)
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
        m_Impl->DeferredSupported = m_Impl->QueryDeferredSupport();
        m_Impl->SupportedSampleCounts = m_Impl->ProbeSampleCounts();
        std::string contactError;
        const auto samplePositionSelector = Runtime().RegisterSelector("getDefaultSamplePositions:count:");
        m_Impl->ContactSupported = Send<ObjCBool, ObjCSelector>(m_Impl->Device, "respondsToSelector:", samplePositionSelector) &&
            m_Impl->ContactProducer.Initialize(m_Impl->Device, contactError);
        std::string forwardPlusError;
        const auto bufferLengthSelector = Runtime().RegisterSelector("maxBufferLength");
        m_Impl->ForwardPlusSupported = Send<ObjCBool, ObjCSelector>(m_Impl->Device, "respondsToSelector:", bufferLengthSelector) &&
            Send<NSUInteger>(m_Impl->Device, "maxBufferLength") >= MaximumForwardPlusLights * sizeof(ShaderLight) &&
            m_Impl->ForwardPlusProducer.Initialize(m_Impl->Device, forwardPlusError);
        m_Impl->TiledSupported = m_Impl->ForwardPlusSupported && m_Impl->TiledProducer.Initialize(m_Impl->Device, forwardPlusError);
        m_Impl->ClusteredSupported = m_Impl->ForwardPlusSupported && m_Impl->ClusteredProducer.Initialize(m_Impl->Device, forwardPlusError);
        // TAA's two attachments occupy 12 bytes/pixel and do not depend on the
        // larger deferred G-buffer. Probe its formats independently of that gate.
        ObjCObject depthProbe = m_Impl->CreateNativeTexture(1, 1, PixelFormatR32Float,
            TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal float depth format probe");
        ObjCObject colorProbe = m_Impl->CreateNativeTexture(1, 1, PixelFormatRGBA16Float,
            TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal temporal color format probe");
        m_Impl->FloatDepthTargetSupported = depthProbe != nullptr;
        m_Impl->TemporalSupported = depthProbe != nullptr && colorProbe != nullptr;
        // Fog uses RGBA16F and the main sampled depth texture; R32F history and
        // the deferred attachment budget are independent capabilities.
        m_Impl->FogSupported = colorProbe != nullptr;
        // Shafts share the RGBA16F requirement, without fog-atlas dimensions,
        // FP32 volume transport or the temporal R32F history attachment.
        m_Impl->LightShaftSupported = colorProbe != nullptr;
        ReleaseObject(depthProbe); ReleaseObject(colorProbe);
        {
            const auto previousError = m_Impl->LastError;
            ObjCObject momentProbe = m_Impl->CreateNativeTexture(1, 1, PixelFormatRG32Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal RG32F moment probe");
            m_Impl->VSMSupported = momentProbe != nullptr;
            ReleaseObject(momentProbe);
            m_Impl->LastError = previousError;
        }
        if (m_Impl->FogSupported)
        {
            const auto previousError = m_Impl->LastError;
            ObjCObject transportProbe = m_Impl->CreateNativeTexture(1, 1, PixelFormatRGBA32Float,
                TextureUsageShaderRead | TextureUsageRenderTarget, "Evans Metal FP32 volume transport probe");
            m_Impl->VolumeLightSupported = transportProbe != nullptr;
            ReleaseObject(transportProbe);
            m_Impl->LastError = previousError;
        }
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
		m_Impl->WeightedSampleCounts = m_Impl->ProbeWeightedSampleCounts();
		m_Impl->OwnerThread = std::this_thread::get_id();
		m_Impl->VisibilitySupported = m_Impl->DeferredSupported && m_Impl->ProbeVisibilityBuffer();
        const auto voxelProbeError=m_Impl->LastError;
        m_Impl->VoxelSamplingSupported=m_Impl->ProbeVoxelSamplingSupport();
        m_Impl->LastError=voxelProbeError;
        m_Impl->Initialized = true;
		m_Impl->LastError.clear();
		if (pool) Send<void>(pool, "drain");
		return true;
	}

	void MetalRHI::Shutdown()
	{
        SetEnvironmentCaptureFrame({});
        SetPlanarCaptureFrame({});
        SetCascadedShadowState({});
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
		capabilities.MaterialAnisotropicFiltering = m_Impl->Initialized && m_Impl->AnisotropicSampler != nullptr;
    capabilities.TemporalEffects=m_Impl->Initialized && true;
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
        capabilities.InstancedRendering = m_Impl->Initialized;
        capabilities.GPUDrivenRendering = m_Impl->Initialized;
        capabilities.VisibilityBufferRendering = m_Impl->Initialized && m_Impl->DeferredSupported && m_Impl->VisibilitySupported;
        capabilities.SupportedSampleCounts = m_Impl->Initialized ? m_Impl->SupportedSampleCounts : 1;
        capabilities.DeferredShading = m_Impl->Initialized && m_Impl->DeferredSupported;
        capabilities.ScalableAmbientObscurance = capabilities.DeferredShading && m_Impl->FloatDepthTargetSupported;
        capabilities.TemporalAntiAliasing = m_Impl->Initialized && m_Impl->TemporalSupported;
        capabilities.TemporalSuperResolution=m_Impl->Initialized && m_Impl->Device && m_Impl->CommandQueue;
        capabilities.ScreenSpaceGlobalIllumination = capabilities.DeferredShading && capabilities.TemporalAntiAliasing;
        capabilities.BakedLightProbes = capabilities.DeferredShading;
        capabilities.BakedLightmaps = capabilities.DeferredShading;
        capabilities.VoxelGI = capabilities.DeferredShading && m_Impl->Initialized && m_Impl->VoxelSamplingSupported;
        capabilities.LumenGI=capabilities.DeferredShading && m_Impl->Initialized;
        capabilities.LumenHardwareRayTracing=capabilities.LumenGI && m_Impl->SupportsLumenHardwareRayTracing();
        capabilities.Refraction = capabilities.DeferredShading;
        capabilities.RayTracedAmbientOcclusion = capabilities.DeferredShading;
        capabilities.RayTracedReflections = capabilities.DeferredShading;
        capabilities.HybridRendering = capabilities.DeferredShading && capabilities.RayTracedReflections;
        capabilities.RayTracedGI = capabilities.RayTracedReflections;
        capabilities.PathTracing = capabilities.RayTracedReflections && (m_Impl->WeightedSampleCounts & 1U) != 0;
        capabilities.VolumetricFog = m_Impl->Initialized && m_Impl->FogSupported;
        capabilities.VolumetricLighting = m_Impl->Initialized && m_Impl->VolumeLightSupported;
        const auto invariantOptions = PositionInvariantCompileOptions();
        capabilities.DepthPeeling = capabilities.Refraction && capabilities.VolumetricLighting && SupportsPositionInvariance(invariantOptions);
        capabilities.ParallaxOcclusion=capabilities.DeferredShading && SupportsPositionInvariance(invariantOptions);
        capabilities.GeometryDisplacement=m_Impl->Initialized;
        capabilities.GeometryTessellation=m_Impl->Initialized;
        capabilities.GeometryVirtualization=m_Impl->Initialized;
        capabilities.NaniteRendering=capabilities.VisibilityBufferRendering && capabilities.GeometryVirtualization;
        capabilities.NaniteHardwareScheduling=capabilities.NaniteRendering;
        capabilities.MeshShaders=capabilities.VisibilityBufferRendering && m_Impl->SupportsMeshShaders();
        capabilities.VirtualShadowMaps=m_Impl->Initialized && m_Impl->Device && m_Impl->CommandQueue;
        ReleaseObject(invariantOptions);
        capabilities.LightShafts = m_Impl->Initialized && m_Impl->LightShaftSupported;
        capabilities.VarianceShadowMaps = m_Impl->Initialized && m_Impl->VSMSupported;
        capabilities.ContactShadows = m_Impl->Initialized && m_Impl->ContactSupported;
        capabilities.ForwardPlusRendering = m_Impl->Initialized && m_Impl->ForwardPlusSupported;
        capabilities.ClusteredForwardRendering = m_Impl->Initialized && m_Impl->ClusteredSupported;
        capabilities.TiledDeferredRendering = capabilities.DeferredShading && m_Impl->TiledSupported;
        capabilities.WeightedTransparencySampleCounts = m_Impl->Initialized ? m_Impl->WeightedSampleCounts : 0;
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = MaximumNativeTextureDimension;
		return capabilities;
	}

	bool MetalRHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0
			|| width > MaximumNativeTextureDimension || height > MaximumNativeTextureDimension)
			return m_Impl->Fail("Metal cannot resize in its current state");
		if (!m_Impl->CreateOffscreenTargets(width, height)) return false;
		m_Impl->Hair.ResetHistory();
    m_Impl->ExposureHistory.Invalidate();
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

	bool MetalRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& candidate,VirtualGeometryCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache updates require a closed frame");
    return m_Impl->PrepareVirtualGeometryCache(upload,previous,candidate,statistics);
}
bool MetalRHI::SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& task,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Submit Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->SubmitNaniteCache(upload,previous,task,statistics);
}
bool MetalRHI::PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    if (!m_Impl->CheckThread("Poll Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->PollNaniteCache(task,ready,output);
}
bool MetalRHI::CancelNaniteCache(NaniteCacheTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->CancelNaniteCache(task);
}
bool MetalRHI::PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& candidate,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache updates require a closed frame");
    return m_Impl->PrepareNaniteCache(upload,previous,candidate,statistics);
}
bool MetalRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache release requires a closed frame");
    return m_Impl->DestroyVirtualGeometryCache(handle);
}
bool MetalRHI::DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache release requires a closed frame");
    return m_Impl->DestroyNaniteCache(handle);
}
bool MetalRHI::SnapshotTSROpaque()
{
    if (!m_Impl->CheckThread("Snapshot TSR opaque")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || m_Impl->PassOpen) return m_Impl->Fail("TSR opaque snapshot requires a closed pass in an initialized frame");
    return m_Impl->SnapshotTSROpaque();
}
bool MetalRHI::RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& settings,FrameCapture& output)
{
    if (!m_Impl->CheckThread("Render TSR display")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR display work requires an initialized RHI and closed frame");
    return m_Impl->RenderTSRDisplay(handle,settings,output);
}
bool MetalRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    if (!m_Impl->CheckThread("BuildVirtualShadowReceiverRequests")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow receivers require an initialized RHI and closed frame");
    return m_Impl->BuildVirtualShadowReceiverRequests(upload,output);
}
bool MetalRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    if (!m_Impl->CheckThread("SampleVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow lookup requires an initialized RHI and closed frame");
    return m_Impl->SampleVirtualShadowAtlas(handle,upload,output);
}
bool MetalRHI::CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!m_Impl->CheckThread("CreateLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->CreateLumenDistanceScene(data,output);
}
bool MetalRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    if (!m_Impl->CheckThread("CreateLumenGlobalDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CreateLumenGlobalDistanceScene(source,settings,output,previous);
}

bool MetalRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenGlobalDistanceUpdate(source,settings,previous,task);
}

bool MetalRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenGlobalDistanceUpdate(task,budget,ready,output,statistics);
}

bool MetalRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{
    if (!m_Impl->CheckThread("CancelLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenGlobalDistanceUpdate(task);
}
bool MetalRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->TraceLumenDistanceScene(handle,upload,output);
}
bool MetalRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenDistanceScene(handle);
}
bool MetalRHI::RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    if (!m_Impl->CheckThread("RelightLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->RelightLumenSurface(surface,distance,upload,output);
}

bool MetalRHI::BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenLightingUpdate(surface,distance,upload,task);
}

bool MetalRHI::AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenLightingUpdate(task,budget,ready,output,statistics);
}

bool MetalRHI::CancelLumenLightingUpdate(LumenLightingTask task)
{
    if (!m_Impl->CheckThread("CancelLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenLightingUpdate(task);
}
bool MetalRHI::ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenLighting(handle,output);
}
bool MetalRHI::DestroyLumenLighting(LumenLightingHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenLighting(handle);
}
bool MetalRHI::TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenRadiance")) return false;
    if (upload.Probes.DirectionalSide && m_Impl->FrameOpen) return m_Impl->Fail("Directional world cache generation requires a closed frame");
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->TraceLumenRadiance(lighting,upload,output);
}
bool MetalRHI::AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("AssembleLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("World radiance assembly requires an initialized RHI and closed frame");
    if (!m_Impl->ValidateLumenRecording(false)) return false;
    return m_Impl->AssembleLumenRadiance(assembly,output);
}
bool MetalRHI::ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbeOrigins")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbeOrigins(handle,output);
}
bool MetalRHI::ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    if (!m_Impl->CheckThread("ReadLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenRadiance(handle,output);
}
bool MetalRHI::DestroyLumenRadiance(LumenRadianceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenRadiance(handle);
}
bool MetalRHI::IntegrateLumenProbes(LumenRadianceHandle source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    if (!m_Impl->CheckThread("IntegrateLumenProbes")) return false;
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->IntegrateLumenProbes(source,upload,output);
}
bool MetalRHI::ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if (!m_Impl->CheckThread("ResolveLumenProbes")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || !m_Impl->ValidateLumenRecording(false)) return m_Impl->Fail("Lumen gather requires an open frame outside a graphics pass");
    return m_Impl->ResolveLumenProbes(handle,settings);
}
bool MetalRHI::ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbes(handle,output);
}
bool MetalRHI::DestroyLumenProbes(LumenProbeHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenProbes(handle);
}
bool MetalRHI::LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    if (!m_Impl->CheckThread("LookupLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface lookup requires an initialized RHI and closed frame");
    return m_Impl->LookupLumenSurface(handle,upload,output);
}
bool MetalRHI::CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!m_Impl->CheckThread("CaptureLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->CaptureLumenSurface(upload,output);
}
bool MetalRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenCaptureUpdate(upload,task);
}
bool MetalRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenCaptureUpdate(task,budget,ready,output,statistics);
}
bool MetalRHI::CancelLumenCaptureUpdate(LumenCaptureTask task)
{
    if (!m_Impl->CheckThread("CancelLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenCaptureUpdate(task);
}
bool MetalRHI::ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->ReadLumenSurface(handle,output);
}
bool MetalRHI::DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenSurface(handle);
}
bool MetalRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!m_Impl->CheckThread("RenderVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->RenderVirtualShadowAtlas(upload,previous,output);
}
bool MetalRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->ReadVirtualShadowAtlas(handle,output);
}
bool MetalRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!m_Impl->CheckThread("DestroyVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->DestroyVirtualShadowAtlas(handle);
}
bool MetalRHI::DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!m_Impl->CheckThread("Destroy TSR history")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR history release requires an initialized RHI and closed frame");
    return m_Impl->DestroyTSRHistory(handle);
}
bool MetalRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle handle,const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached virtual geometry")) return false;
    if (m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateVirtualGeometry(upload,output,handle);
}
bool MetalRHI::SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& task)
{
    if (!m_Impl->CheckThread("Submit Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->SubmitNaniteGeometry(cache,upload,task);
}
bool MetalRHI::PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Poll Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->PollNaniteGeometry(task,ready,output);
}
bool MetalRHI::CancelNaniteGeometry(NaniteGeometryTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->CancelNaniteGeometry(task);
}
bool MetalRHI::CreateCachedNaniteGeometry(NaniteCacheHandle handle,const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateNaniteGeometry(upload,output,handle);
}

bool MetalRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create virtual geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual geometry must be produced between frames");
    return m_Impl->CreateVirtualGeometry(upload,output);
}
bool MetalRHI::PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDraws") && m_Impl->PrepareNaniteDraws(visibility,draws,output);
}
bool MetalRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDrawBins") && m_Impl->PrepareNaniteDraws(visibility,draws,output,true);
}
bool MetalRHI::DrawNaniteBins(NaniteDrawHandle handle)
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
bool MetalRHI::DrawNaniteCluster(NaniteDrawHandle handle,unsigned query)
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
bool MetalRHI::CaptureNaniteDepth(NaniteDepthHandle& output)
{
    if (!m_Impl->CheckThread("CaptureNaniteDepth") || !IsInitialized()) return false;
    return m_Impl->CaptureNaniteDepth(output);
}
bool MetalRHI::BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("BeginNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->BeginNaniteVisibility(history,queries,output);
}
bool MetalRHI::RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle depth,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("RetestNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->RetestNaniteVisibility(main,depth,output);
}
bool MetalRHI::ReadNaniteDepth(NaniteDepthHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDepthWords(found->second->Info,words,output,m_Impl->LastError);
}
bool MetalRHI::ReadNaniteVisibility(NaniteVisibilityHandle handle,std::vector<NaniteVisibilityState>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDecisionWords(found->second->Info,words,output,m_Impl->LastError);
}
bool MetalRHI::DestroyNaniteDepth(NaniteDepthHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDepths.erase(found); return true;
}
bool MetalRHI::DestroyNaniteVisibility(NaniteVisibilityHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDecisions.erase(found); return true;
}
bool MetalRHI::CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite geometry must be produced between frames");
    return m_Impl->CreateNaniteGeometry(upload,output);
}

bool MetalRHI::CreateTessellatedGeometry(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create tessellated geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Tessellated geometry must be prepared between frames");
    return m_Impl->CreateTessellated(geometry,settings,output);
}

bool MetalRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    if (!m_Impl->CheckThread("Create displaced vertex buffer")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Displaced geometry must be prepared between frames");
    return m_Impl->CreateDisplaced(geometry,settings,output);
}

	bool MetalRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer")) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || data == nullptr || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("Metal buffer update is invalid");
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		std::memcpy(static_cast<std::uint8_t*>(Send<void*>(iterator->second.Buffer, "contents")) + offset,
			data, size);
		return true;
	}

	bool MetalRHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer")) return false;
		if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (!buffer || iterator == m_Impl->Buffers.end()) return m_Impl->Fail("Metal buffer handle is invalid");
		ReleaseObject(iterator->second.Buffer);
		m_Impl->Buffers.erase(iterator);
		return true;
	}

	TextureHandle MetalRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0
			|| desc.Width > MaximumNativeTextureDimension || desc.Height > MaximumNativeTextureDimension
			|| !ValidTextureMipLevels(desc, initialData))
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
		resource.Texture = m_Impl->CreateNativeTexture(desc.Width, desc.Height, format, usage, desc.DebugName, desc.MipLevels);
		const std::size_t bytesPerPixel = desc.Format == TextureFormat::Depth32 ? sizeof(float) : 4;
		if (resource.Texture == nullptr
			|| !m_Impl->UploadTexture(resource.Texture, desc.Width, desc.Height, bytesPerPixel, initialData, desc.MipLevels))
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
		if (!m_Impl->RetireLumenCaptureInput(texture)) return false;
		auto iterator = m_Impl->Textures.find(texture.Value);
		if (!texture || iterator == m_Impl->Textures.end()) return m_Impl->Fail("Metal texture handle is invalid");
        const auto moments = m_Impl->VSMMoments.find(texture.Value);
        if (moments != m_Impl->VSMMoments.end())
        { m_Impl->DestroyMomentMap(moments->second); m_Impl->VSMMoments.erase(moments); }
		ReleaseObject(iterator->second.Texture);
		m_Impl->Textures.erase(iterator);
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
		return true;
	}

	bool MetalRHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
			return m_Impl->Fail("Metal frame cannot begin in the current state");
    if (!m_Impl->PrepareRayShadowImage(GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr)) return false;
		if (!m_Impl->CollectLumenGlobalDistance() || !m_Impl->CollectLumenLighting() || !m_Impl->CollectLumenCapture()) return false;
		ResetMotionTransform();
    m_Impl->ExposureHistory.Begin();
        if (m_Impl->ForwardPlusSupported && !m_Impl->ForwardPlusProducer.BeginFrame(m_Impl->LastError)) return false;
        if (m_Impl->ClusteredSupported && !m_Impl->ClusteredProducer.BeginFrame(m_Impl->LastError)) return false;
        if (m_Impl->TiledSupported && !m_Impl->TiledProducer.BeginFrame(m_Impl->LastError)) return false;
        // This backend's Present waits for completion before closing the frame.
        // An aborted frame releases its command buffer before the next BeginFrame.
        for (auto& page : m_Impl->InstancePages) page.Cursor = 0;
        m_Impl->GpuCursor = 0; m_Impl->ActiveGpuBatch = nullptr; ++m_Impl->GpuEpoch;
        m_Impl->ResetDeferredFrame();
		m_Impl->TiledActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::TiledDeferred;
        m_Impl->ClusteredActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::ClusteredForward;
        m_Impl->LightListsActive = Techniques::UsesLightLists(GetTechniqueSettings().Pipeline);
        if (m_Impl->LightListsActive && !(m_Impl->ClusteredActive ? m_Impl->ClusteredSupported : m_Impl->TiledActive ? m_Impl->TiledSupported : m_Impl->ForwardPlusSupported))
            return m_Impl->Fail("Metal Forward+ requires native compute light-list support");
		m_Impl->FramePool = Send<ObjCObject>(GetClass("NSAutoreleasePool"), "new");
        m_Impl->FrameAreaLights = GetTechniqueSettings().AreaLights.Enabled;
        m_Impl->FrameSampleCount = GetTechniqueSettings().SampleCount;
        m_Impl->FrameContact = GetTechniqueSettings().ContactShadows;
        if (m_Impl->FrameContact.Enabled && (!m_Impl->ContactSupported || !m_Impl->ContactProducer.BeginFrame(m_Impl->LastError)))
        {
            m_Impl->ResetDeferredFrame();
            if (m_Impl->FramePool) Send<void>(m_Impl->FramePool, "drain");
            m_Impl->FramePool = nullptr;
            return m_Impl->Fail("Metal contact frame cannot begin: " + m_Impl->LastError);
        }
        m_Impl->WeightedActive = Techniques::UsesWeightedTransparency(GetTechniqueSettings());
        m_Impl->FrameTransparency = GetTechniqueSettings().Transparency;
        m_Impl->FrameShadowFilter = GetTechniqueSettings().ShadowFilter;
    if (GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced) m_Impl->FrameShadowFilter.Mode=Techniques::ShadowFilterMode::Hard;
        for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
        m_Impl->SubsurfaceProducing=m_Impl->SubsurfaceResolved=m_Impl->SubsurfaceHasScene=false;
    m_Impl->FrameSubsurface=GetTechniqueSettings().Subsurface;
    m_Impl->SubsurfaceRequested=Techniques::UsesSubsurfaceScattering(m_Impl->FrameSubsurface);
    m_Impl->FramePostProcessSettings = GetTechniqueSettings().PostProcessing;
    if (!m_Impl->PrepareTSRScene(GetTSRSceneFrame(),m_Impl->FramePostProcessSettings)) return false;
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
            !Techniques::RequiresHDR(m_Impl->FramePostProcessSettings.HDR) || m_Impl->FrameVoxelGI.Shading!=GetTechniqueSettings().Shading))
            return m_Impl->Fail("Voxel GI requires an exclusive HDR raster frame and matching receiver shading");
        m_Impl->GIActive = Techniques::UsesGlobalIllumination(m_Impl->FrameGI) || bool(m_Impl->FrameVoxelGI.Volume);
        m_Impl->EarlyTemporalActive = Techniques::UsesTemporalGI(m_Impl->FrameGI) || Techniques::UsesTemporalReflections(GetTechniqueSettings().Reflections);
        m_Impl->ProbeActive = Techniques::UsesBakedProbes(m_Impl->FrameGI);
        m_Impl->LightmapActive = Techniques::UsesLightmaps(m_Impl->FrameGI);
        m_Impl->FrameLightmaps = GetLightmapFrame();
        m_Impl->FramePeeling=GetDepthPeelingFrame(); m_Impl->PeelingSequence.Abort();
        m_Impl->FrameRefraction = GetRefractionFrame();
        m_Impl->FrameRefractionSettings = GetTechniqueSettings().Refraction;
        m_Impl->RefractionActive = Techniques::UsesRefraction(GetTechniqueSettings());
        m_Impl->RefractionResolved = false;
        m_Impl->FrameLightmapSettings = GetTechniqueSettings();
        m_Impl->FrameProbeSamples = GetBakedProbeSamples();
        m_Impl->FrameFog = GetTechniqueSettings().VolumetricFog;
        m_Impl->FrameVolumeLighting = GetTechniqueSettings().VolumetricLighting;
        m_Impl->FogActive = Techniques::UsesVolumetricFog(m_Impl->FrameFog);
        m_Impl->FrameLightShafts = GetTechniqueSettings().LightShafts;
        m_Impl->LightShaftActive = Techniques::UsesLightShafts(m_Impl->FrameLightShafts);
        m_Impl->DepthOfFieldActive = m_Impl->FramePostProcessSettings.DepthOfField.Enabled;
        m_Impl->MotionBlurActive = m_Impl->FramePostProcessSettings.MotionBlur.Enabled;
        m_Impl->TemporalActive = Techniques::UsesTemporalAA(m_Impl->FramePostProcessSettings);
        m_Impl->VelocityActive = (Techniques::RequiresMotionVectors(m_Impl->FramePostProcessSettings) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive;
        m_Impl->HdrActive = Techniques::RequiresHDR(m_Impl->FramePostProcessSettings.HDR);
        if (m_Impl->HdrActive && !m_Impl->EnsureHdrTargets())
        {
            if (m_Impl->FramePool) Send<void>(m_Impl->FramePool, "drain");
            m_Impl->FramePool = nullptr;
            return false;
        }
		m_Impl->DeferredActive = Techniques::UsesDeferredLighting(GetTechniqueSettings().Pipeline);
        m_Impl->AmbientOcclusionActive = Techniques::UsesAmbientOcclusion(GetTechniqueSettings().AmbientOcclusion);
        m_Impl->FrameAmbientOcclusionSettings = GetTechniqueSettings().AmbientOcclusion;
        m_Impl->FrameRayScene = GetRayScene();
        m_Impl->FrameRayRadiance = GetRayRadianceFrame();
        m_Impl->FrameRayRadianceSettings = GetTechniqueSettings();
        m_Impl->PathFrame.BeginFrame(Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline));
        m_Impl->FrameReflections = GetTechniqueSettings().Reflections;
        m_Impl->FrameReflectionShading = GetTechniqueSettings().Shading;
        m_Impl->AnisotropicFiltering = GetTechniqueSettings().AnisotropicFiltering;
        m_Impl->ReflectionActive = Techniques::UsesReflections(m_Impl->FrameReflections);
        if (!m_Impl->PrepareEnvironmentCapture(GetEnvironmentCaptureFrame()) ||
            !m_Impl->PreparePlanarCapture(GetPlanarCaptureFrame()) ||
            ((m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) && !m_Impl->EnsureDeferredResources()) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->EnsureAmbientOcclusionResources()) ||
            (m_Impl->ReflectionActive && !m_Impl->EnsureReflectionResources()) ||
            (Techniques::UsesRayTracedGI(m_Impl->FrameGI) && !m_Impl->EnsureRayGIResources()) ||
            (m_Impl->FrameVoxelGI.Volume && !m_Impl->EnsureVoxelGIResources()) ||
            (m_Impl->PathFrame.Active && !m_Impl->EnsurePathTracing()) ||
            (m_Impl->FogActive && !m_Impl->EnsureFogResources()) ||
            (m_Impl->LightShaftActive && !m_Impl->EnsureLightShaftResources()) ||
            (m_Impl->FogActive && (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)) && !m_Impl->EnsureVolumeLightResources()) ||
            (m_Impl->DepthOfFieldActive && !m_Impl->EnsureDepthOfFieldPipelines()) ||
            (m_Impl->VelocityActive && !m_Impl->EnsureVelocityResources()) ||
            (m_Impl->TemporalActive && !m_Impl->EnsureTemporalResources()) ||
            (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI) && !m_Impl->EnsureGIResources()) ||
            (m_Impl->ProbeActive && !m_Impl->EnsureProbeResources()) ||
            (m_Impl->MotionBlurActive && !m_Impl->EnsureMotionResources()) ||
            (m_Impl->FrameSampleCount > 1 && !m_Impl->EnsureMultisampleResources()) ||
            (m_Impl->Hair.Active && !m_Impl->EnsureHairFur()) ||
            (m_Impl->LightmapActive && !m_Impl->EnsureLightmapResources()) ||
            (m_Impl->RefractionActive && !m_Impl->EnsureRefractionResources()) ||
            (m_Impl->FramePeeling.Enabled && !m_Impl->EnsurePeelingResources()) ||
            (m_Impl->WeightedActive && !m_Impl->EnsureWeightedResources()) ||
            !m_Impl->PrepareVirtualShadowSampling(GetVirtualShadowSamplingFrame(),GetCascadedShadowState()) ||
            !m_Impl->PrepareCascadedShadows(GetCascadedShadowState()))
        {
            m_Impl->ResetDeferredFrame();
            if (m_Impl->FramePool) Send<void>(m_Impl->FramePool, "drain");
            m_Impl->FramePool = nullptr;
            return false;
        }
		ObjCObject commandBuffer = Send<ObjCObject>(m_Impl->CommandQueue, "commandBuffer");
		m_Impl->CommandBuffer = commandBuffer ? Send<ObjCObject>(commandBuffer, "retain") : nullptr;
		if (m_Impl->CommandBuffer == nullptr)
		{
            m_Impl->ResetDeferredFrame();
			if (m_Impl->FramePool) Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
			return m_Impl->Fail("Metal failed to create a command buffer");
		}
		m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Backend = RHIBackend::Metal;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
        m_Impl->PendingFrame.Statistics.RasterSampleCount = m_Impl->FrameSampleCount;
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = true;
		m_Impl->LastError.clear();
		if (m_Impl->FramePostProcessSettings.TemporalEffects.Enabled && !m_Impl->EnsureExposure()) return false;
    return true;
	}

	void MetalRHI::AbortFrame()
	{
        m_Impl->PeelingSequence.Abort(); m_Impl->FramePeeling={};
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
        m_Impl->PathFrame.Abort();
		m_Impl->Probes.Invalidate();
        if (m_Impl->RenderEncoder != nullptr)
		{
			Send<void>(m_Impl->RenderEncoder, "endEncoding");
			m_Impl->RenderEncoder = nullptr;
		}
        if (m_Impl->ContactSupported)
        {
            std::string ignored;
            m_Impl->ContactProducer.AbortFrame(ignored);
        }
		ReleaseObject(m_Impl->CommandBuffer);
		m_Impl->CommandBuffer = nullptr;
        m_Impl->RetireForwardPlusAfterDiscard();
		if (m_Impl->FramePool != nullptr)
		{
			Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
		}
		m_Impl->PendingFrame = {};
		m_Impl->CurrentPass = {};
        for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		m_Impl->AbortTSRScene();
    m_Impl->FrameOpen = false;
    m_Impl->FinishLumenFrame(false); m_Impl->FinishNaniteFrame(false);
        m_Impl->FrameCascades = {};
        m_Impl->CascadeTextures.fill(nullptr);
        SetCascadedShadowState({});
        SetEnvironmentCaptureFrame({});
        SetPlanarCaptureFrame({});
        m_Impl->ResetDeferredFrame();
		m_Impl->PassOpen = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
	}

	bool MetalRHI::BeginPass(const PassDesc& desc)
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
			return m_Impl->Fail("Metal pass cannot begin in the current state");
        if (m_Impl->PathFrame.Active) return m_Impl->Fail("Path tracing owns primary visibility; raster passes are inactive");
        const bool mainColor = desc.Type == PassType::Color && !desc.Target;
        const bool contactDepth = desc.Type == PassType::ContactDepth;
        const bool forwardPlusDepth = desc.Type == PassType::ForwardPlusDepth;
        const bool privateCapture = desc.Type == PassType::EnvironmentCapture || desc.Type == PassType::PlanarCapture;
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Prepared())
            return m_Impl->Fail("Metal Forward+ requires its complete frame lights before any pass");
        if (forwardPlusDepth && (m_Impl->ClusteredActive || m_Impl->TiledActive || !m_Impl->LightListsActive || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ForwardPlusDepthWritten || m_Impl->GBufferWritten || m_Impl->LightListFrame.Assigned() ||
            m_Impl->ForwardColorWritten || m_Impl->ContactGenerated))
            return m_Impl->Fail("Metal Forward+ depth requires one cleared opaque camera prepass before assignment");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !privateCapture && desc.Type != PassType::Shadow &&
            (desc.ClearDepthEnabled || forwardPlusDepth || contactDepth || desc.Type == PassType::GBuffer))
            return m_Impl->Fail("Metal Forward+ assignment depth cannot be cleared or regenerated");
        if (m_Impl->LightListsActive && mainColor && (!m_Impl->LightListFrame.Assigned() || (!m_Impl->ClusteredActive && desc.ClearDepthEnabled)))
            return m_Impl->Fail("Metal Forward+ lighting requires completed lists and preserved assignment depth");
        if (contactDepth && (!m_Impl->FrameContact.Enabled || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ContactDepthWritten || m_Impl->GBufferWritten || m_Impl->ContactGenerated || m_Impl->ForwardColorWritten || m_Impl->DeferredResolved))
            return m_Impl->Fail("Metal contact depth requires one cleared opaque camera prepass before visibility");
        if (m_Impl->FrameContact.Enabled && mainColor && (!m_Impl->ContactGenerated || desc.ClearDepthEnabled))
            return m_Impl->Fail("Metal main lighting requires completed contact visibility and preserved camera depth");
        if (m_Impl->ContactGenerated && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture &&
            desc.Type != PassType::PlanarCapture && (desc.ClearDepthEnabled || contactDepth || desc.Type == PassType::GBuffer))
            return m_Impl->Fail("Metal traced opaque depth cannot be cleared or regenerated after contact visibility");
        if (!ValidateDepthPeelingPass(desc,m_Impl->FramePeeling,m_Impl->PeelingSequence,m_Impl->LastError)) return false;
        const bool weightedPass = mainColor && m_Impl->WeightedActive && m_Impl->WeightedPrepared && !m_Impl->WeightedResolved;
        if (m_Impl->WeightedActive && mainColor && m_Impl->WeightedResolved)
            return m_Impl->Fail("Metal main color cannot be redrawn after weighted composition");
        if (m_Impl->WeightedPrepared && !m_Impl->WeightedResolved &&
            (!weightedPass || m_Impl->WeightedAccumulated || desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Metal prepared transparency requires one depth-preserving main color pass");
        if (m_Impl->WeightedActive && mainColor && !m_Impl->WeightedPrepared &&
            (m_Impl->DeferredResolved || m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Metal prepare weighted statistics before transparent geometry");
        if (desc.Type == PassType::DeferredLighting)
            return m_Impl->Fail("Use DrawDeferredLighting for Metal fullscreen lighting");
        const bool planar = desc.Type == PassType::PlanarCapture;
        if (planar && (!m_Impl->PlanarActive || m_Impl->PlanarWritten || m_Impl->PlanarReady || desc.Target ||
            !desc.ClearColorEnabled || !desc.ClearDepthEnabled || m_Impl->ForwardColorWritten || m_Impl->GBufferWritten))
            return m_Impl->Fail("Metal planar capture requires one cleared private target before main geometry");
        if (!planar && m_Impl->PlanarActive && !m_Impl->PlanarReady && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture)
            return m_Impl->Fail("Metal complete planar capture and mips before main geometry");
        const bool capture = desc.Type == PassType::EnvironmentCapture;
        if (capture && (!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || desc.Target ||
            desc.EnvironmentFaceIndex >= EnvironmentFaceCount || !desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
            m_Impl->ForwardColorWritten || m_Impl->GBufferWritten || (m_Impl->EnvironmentFacesWritten & (1u << desc.EnvironmentFaceIndex))))
            return m_Impl->Fail("Metal capture requires six distinct cleared faces before main geometry");
        if (!capture && m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady && desc.Type != PassType::Shadow)
            return m_Impl->Fail("Metal complete capture and mip generation before main geometry");
        const bool velocity = desc.Type == PassType::Velocity;
        if (velocity && !m_Impl->EarlyTemporalActive && m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
            return m_Impl->Fail("Metal light shafts must precede velocity replay");
        if (m_Impl->LightShaftResolved && desc.Type != PassType::Shadow &&
            (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
            return m_Impl->Fail("Metal shaft color and its opaque depth must be preserved");
        if (velocity && !m_Impl->EarlyTemporalActive && m_Impl->FogActive && !m_Impl->FogResolved)
            return m_Impl->Fail("Metal fog must precede opaque velocity replay");
        if (m_Impl->FogResolved && desc.Type != PassType::Shadow &&
            (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
            return m_Impl->Fail("Metal fogged scene color and opaque depth must be preserved");
        const bool multisample = m_Impl->FrameSampleCount > 1 && desc.Type != PassType::Outline && !capture && !planar;
        if (multisample && desc.Target && desc.Type != PassType::Shadow)
            return m_Impl->Fail("Metal MSAA requires internal sample-count-matched attachments");
        if (m_Impl->FrameSampleCount > 1 && (velocity || desc.Type == PassType::Outline) && !m_Impl->SamplesOpaqueResolved)
            return m_Impl->Fail("Metal resolve opaque sample depth before velocity or outline draws");
        if (multisample && m_Impl->SamplesOpaqueResolved && desc.Type == PassType::Color &&
            (desc.ClearColorEnabled || desc.ClearDepthEnabled || (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored)))
            return m_Impl->Fail("Metal transparency must load processed sample color and preserve opaque depth");
        const bool opaqueComplete = m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten;
        if (velocity && (!m_Impl->VelocityActive || desc.Target || !desc.ClearColorEnabled || desc.ClearDepthEnabled ||
            !opaqueComplete || m_Impl->VelocityCaptured || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
            (!m_Impl->EarlyTemporalActive && m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)))
            return m_Impl->Fail("Metal velocity requires opaque lighting/AO and a fresh internal target, once before blur");
        if (m_Impl->VelocityCaptured && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
            return m_Impl->Fail("Metal shared opaque depth cannot be cleared after velocity capture");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->MotionBlurActive)
        {
            if (opaqueComplete && !m_Impl->MotionBlurResolved)
                return m_Impl->Fail("Metal transparent color must follow motion blur");
            if (m_Impl->MotionBlurResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
                return m_Impl->Fail("Metal transparent color must load completed scene color and depth");
        }
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->TemporalActive)
        {
            if (opaqueComplete && !m_Impl->TemporalResolved)
                return m_Impl->Fail("Metal transparent color must follow temporal reconstruction");
            if (m_Impl->TemporalResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
                return m_Impl->Fail("Metal transparent color must load reconstructed scene color and depth");
        }
        if (m_Impl->ReflectionResolved && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
            return m_Impl->Fail("Metal opaque depth cannot be cleared after reflections");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->GIActive)
        {
            if (!m_Impl->GBufferWritten) return m_Impl->Fail("SSGI requires material capture before lighting");
            if (opaqueComplete && !m_Impl->GI.Resolved) return m_Impl->Fail("Transparency must follow SSGI");
            if (m_Impl->GI.Resolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
                return m_Impl->Fail("Scene attachments cannot be cleared after SSGI");
        }
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->ReflectionActive)
        {
            if (!m_Impl->GBufferWritten) return m_Impl->Fail("Metal reflections require material capture before opaque lighting");
            if (opaqueComplete && !m_Impl->ReflectionResolved) return m_Impl->Fail("Metal transparency must follow reflections");
            if (m_Impl->ReflectionResolved && desc.ClearColorEnabled) return m_Impl->Fail("Metal scene color cannot be cleared after reflections");
        }
        const bool geometry = desc.Type == PassType::GBuffer;
        if (m_Impl->DepthOfFieldResolved && (geometry ||
            (desc.Type == PassType::Color && !desc.Target && (desc.ClearColorEnabled || desc.ClearDepthEnabled))))
            return m_Impl->Fail("Metal post-DoF transparency must load the completed scene and opaque depth");
        if (m_Impl->DepthOfFieldResolved && desc.Target && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
            return m_Impl->Fail("Metal custom color passes must preserve shared opaque depth after depth of field");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->DepthOfFieldActive &&
            !m_Impl->DepthOfFieldResolved && (m_Impl->DeferredActive || m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Metal depth of field must precede transparent forward draws");
        if (geometry && (!(m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) || desc.Target ||
            m_Impl->DeferredResolved || m_Impl->AmbientOcclusionResolved || m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Metal material capture requires an unresolved deferred/AO frame and the main target");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->DeferredActive && !m_Impl->DeferredResolved)
            return m_Impl->Fail("Metal forward transparency must follow deferred lighting");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved &&
            (!m_Impl->GBufferWritten || m_Impl->DeferredActive || m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Metal ambient occlusion must follow captured opaque color and precede transparency");
        if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionResolved &&
            (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Metal post-AO transparency must load opaque color and depth");
		ObjCObject colorTarget = m_Impl->HdrActive ? m_Impl->HdrTextures[0] : m_Impl->ColorTexture;
        if (velocity) colorTarget = m_Impl->MotionTextures[Impl::MotionVelocity];
        ObjCObject depthTarget = m_Impl->DepthTexture;
        if (multisample && !desc.Target)
        {
            colorTarget = m_Impl->Multisample.Colors[velocity ? Impl::SampleVelocity : Impl::SampleColor];
            depthTarget = m_Impl->Multisample.Depth;
        }
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

        if (capture)
        {
            colorTarget = m_Impl->EnvironmentTexture; depthTarget = m_Impl->EnvironmentDepth;
            width = height = static_cast<int>(m_Impl->FrameEnvironment.Resolution);
        }
        else if (planar)
        {
            colorTarget = m_Impl->PlanarTexture; depthTarget = m_Impl->PlanarDepth;
            width = static_cast<int>(m_Impl->FramePlanar.Width); height = static_cast<int>(m_Impl->FramePlanar.Height);
        }
        const bool vsmCaster = desc.Type == PassType::Shadow && m_Impl->UsesVSM();
        if (vsmCaster)
        {
            if (!m_Impl->EnsureMomentMap(desc.Target)) return false;
            auto& map = m_Impl->VSMMoments.at(desc.Target.Value);
            if (!desc.ClearDepthEnabled && !map.RawWritten) return m_Impl->Fail("Metal VSM requires an initial caster clear each frame");
            map.Ready = false;
            colorTarget = map.Images[Impl::VSMRaw];
        }
        if (desc.DepthPeelingLayer)
        {
            if (!m_Impl->PeelingSequence.Advance(DepthPeelingEvent::BeginShading,m_Impl->LastError)) return false;
            colorTarget=m_Impl->PeelingLayer; depthTarget=m_Impl->PeelingDepth[m_Impl->PeelingSequence.Step().DepthWrite];
        }
        if (contactDepth || forwardPlusDepth) colorTarget = nullptr;
		ObjCObject passDescriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),
			"renderPassDescriptor");
		if (colorTarget != nullptr)
		{
			ObjCObject attachments = Send<ObjCObject>(passDescriptor, "colorAttachments");
            const bool hdr = m_Impl->HdrActive && !desc.Target;
			const MTLClearColor clearColor {
                hdr ? std::clamp(desc.ClearColor.X, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.X,
                hdr ? std::clamp(desc.ClearColor.Y, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.Y,
                hdr ? std::clamp(desc.ClearColor.Z, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.Z, desc.ClearColor.W };
            const std::size_t count = weightedPass ? m_Impl->WeightedTextures.size() : (geometry ? GBufferAttachmentCount : 1);
            for (std::size_t index = 0; index < count; ++index)
            {
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
                Send<void, ObjCObject>(attachment, "setTexture:", geometry ? (multisample ? m_Impl->Multisample.Colors[Impl::SampleMaterialFirst + index] : m_Impl->GBufferTextures[index]) : colorTarget);
                if (weightedPass) Send<void, ObjCObject>(attachment, "setTexture:", m_Impl->WeightedTextures[index]);
                if (capture) Send<void, NSUInteger>(attachment, "setSlice:", desc.EnvironmentFaceIndex);
                Send<void, NSUInteger>(attachment, "setLoadAction:", (desc.DepthPeelingLayer || (vsmCaster ? desc.ClearDepthEnabled : desc.ClearColorEnabled)) ? LoadActionClear : LoadActionLoad);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
                Send<void, MTLClearColor>(attachment, "setClearColor:", vsmCaster ? MTLClearColor{1, 1, 0, 0} : (geometry || velocity || desc.DepthPeelingLayer) ? MTLClearColor{} : clearColor);
            }
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
        if (geometry && desc.ClearColorEnabled)
            m_Impl->DeferredBackground = {std::clamp(desc.ClearColor.X, 0.0f, MaximumHalfFloatRadiance),
                std::clamp(desc.ClearColor.Y, 0.0f, MaximumHalfFloatRadiance),
                std::clamp(desc.ClearColor.Z, 0.0f, MaximumHalfFloatRadiance), desc.ClearColor.W};
		const MTLViewport viewport { 0.0, 0.0, static_cast<double>(width),
			static_cast<double>(height), 0.0, 1.0 };
		Send<void, MTLViewport>(m_Impl->RenderEncoder, "setViewport:", viewport);
		m_Impl->CurrentPass = desc;
        m_Impl->WeightedPass = weightedPass;
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
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.DepthPeelingLayer && !pipeline.BlendEnabled) return m_Impl->Fail("Only transparent color draws follow the TSR opaque snapshot");

    if (m_Impl->FrameOpen && m_Impl->Hair.Active && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
        ((!m_Impl->Hair.Resolved && pipeline.BlendEnabled) || (m_Impl->Hair.Resolved && pipeline.DepthWrite)))
        return m_Impl->Fail("Opaque draws precede hair; transparent draws follow it without replacing depth");
        if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth &&
            (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.BlendEnabled ||
             pipeline.SamplesShadowMap || pipeline.OutlineExtrusion))
            return m_Impl->Fail("Metal Forward+ depth requires opaque/cutout coverage and depth-only writes");
        if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && pipeline.DepthWrite && m_Impl->CurrentPass.Type != PassType::Shadow &&
            m_Impl->CurrentPass.Type != PassType::EnvironmentCapture && m_Impl->CurrentPass.Type != PassType::PlanarCapture)
            return m_Impl->Fail("Metal Forward+ receivers must preserve the assignment depth");
        if (m_Impl->LightListsActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && !pipeline.DepthTest)
            return m_Impl->Fail("Metal Forward+ main receivers require depth testing");
        if (m_Impl->CurrentPass.Type == PassType::ContactDepth &&
            (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.BlendEnabled ||
             pipeline.SamplesShadowMap || pipeline.OutlineExtrusion))
            return m_Impl->Fail("Metal contact depth requires opaque/cutout coverage and depth-only writes");
        if (m_Impl->ContactGenerated && m_Impl->CurrentPass.Type != PassType::Shadow &&
            m_Impl->CurrentPass.Type != PassType::EnvironmentCapture && m_Impl->CurrentPass.Type != PassType::PlanarCapture && pipeline.DepthWrite)
            return m_Impl->Fail("Metal lighting must preserve the depth used for contact visibility");
        if (m_Impl->WeightedPass && (!pipeline.BlendEnabled || pipeline.DepthWrite || !pipeline.DepthTest || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
            return m_Impl->Fail("Metal weighted geometry requires color accumulation and read-only depth testing");
        if (m_Impl->WeightedActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
            pipeline.BlendEnabled != m_Impl->WeightedPass)
            return m_Impl->Fail("Metal opaque color precedes weighted preparation; transparent color follows it");
		if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
			return m_Impl->Fail("Metal pipeline must be bound inside a pass");
        if (m_Impl->LightShaftResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
            return m_Impl->Fail("Metal preserve opaque depth after light-shaft composition");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->LightShaftActive &&
            pipeline.BlendEnabled != m_Impl->LightShaftResolved)
            return m_Impl->Fail("Metal opaque draws precede light shafts; transparent draws follow them");
        if (m_Impl->FogResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
            return m_Impl->Fail("Metal opaque depth is read-only after fog composition");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->FogActive &&
            pipeline.BlendEnabled != m_Impl->FogResolved)
            return m_Impl->Fail("Metal opaque draws precede fog; transparent draws follow it");
        if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved && !m_Impl->CurrentPass.Target && pipeline.DepthWrite)
            return m_Impl->Fail("Metal opaque depth is read-only after sample resolve");
        if (m_Impl->VelocityCaptured && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
            return m_Impl->Fail("Metal shared opaque depth is read-only after velocity capture");
        if (m_Impl->CurrentPass.Type == PassType::Velocity &&
            (!pipeline.DepthTest || pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
            return m_Impl->Fail("Metal velocity requires opaque coverage and read-only depth testing");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->MotionBlurActive &&
            pipeline.BlendEnabled != m_Impl->MotionBlurResolved)
            return m_Impl->Fail("Metal opaque draws precede motion blur; transparent draws follow it");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->TemporalActive &&
            pipeline.BlendEnabled != m_Impl->TemporalResolved)
            return m_Impl->Fail("Metal opaque draws precede TAA; transparent draws follow it");
        if (m_Impl->ReflectionResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
            return m_Impl->Fail("Metal opaque depth is read-only after reflections");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->GIActive &&
            pipeline.BlendEnabled != m_Impl->GI.Resolved) return m_Impl->Fail("SSGI separates opaque and transparent material draws");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->ReflectionActive &&
            pipeline.BlendEnabled != m_Impl->ReflectionResolved)
            return m_Impl->Fail("Metal opaque draws precede reflections; transparent draws follow them");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->DepthOfFieldActive)
        {
            if (m_Impl->DepthOfFieldResolved && (!pipeline.BlendEnabled || pipeline.DepthWrite))
                return m_Impl->Fail("Metal only transparent, depth-read-only draws may follow depth of field");
            if (!m_Impl->DepthOfFieldResolved && pipeline.BlendEnabled)
                return m_Impl->Fail("Metal transparent draws require depth of field first");
        }
        if (m_Impl->DepthOfFieldResolved && m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type != PassType::Shadow &&
            pipeline.DepthTest && pipeline.DepthWrite)
            return m_Impl->Fail("Metal custom draws cannot overwrite shared opaque depth after depth of field");
        if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && !m_Impl->FramePlanar.Active())
            return m_Impl->Fail("Metal inactive planar captures must only clear their target");
        if ((m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture) &&
            (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite ||
             pipeline.SamplesShadowMap || pipeline.OutlineExtrusion || pipeline.Shading != m_Impl->FrameReflectionShading))
            return m_Impl->Fail("Metal capture requires opaque depth and the frame shading model without shadow sampling");
        if (m_Impl->CurrentPass.Type == PassType::GBuffer &&
            (pipeline.BlendEnabled || pipeline.Shading != m_Impl->FrameReflectionShading ||
             (m_Impl->DeferredActive && pipeline.Shading == Techniques::ShadingModel::Gouraud)))
            return m_Impl->Fail("Metal G-buffer requires opaque materials and a compatible frame shading model");
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
            m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved && pipeline.BlendEnabled)
            return m_Impl->Fail("Metal transparency must follow ambient occlusion");
        if (m_Impl->CurrentPass.DepthPeelingLayer && !ValidateDepthPeelingPipeline(pipeline,m_Impl->LastError)) return false;
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
        if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && value.Values != m_Impl->FramePlanar.ClippedViewProjection.Values)
            return m_Impl->Fail("Metal planar capture requires the frozen oblique mirror projection");
        if (m_Impl->CurrentPass.DepthPeelingLayer && value.Values!=m_Impl->FramePeeling.ViewProjection.Values)
            return m_Impl->Fail("OIT capture and replay require the same raster projection");
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

	bool MetalRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchHandle& handle)
    {
        handle = {};
        if (!m_Impl->CheckThread("PrepareGpuDrivenBatch") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Instances.empty() || !desc.IndexCount)
            return m_Impl->Fail("Metal GPU batch preparation requires an open frame outside a graphics pass");
        std::uint32_t index = 0;
        if (!m_Impl->PrepareGpuBatch(desc, index)) return false;
        handle = {m_Impl->GpuEpoch, index};
        ++m_Impl->PendingFrame.Statistics.GpuCullDispatchCount;
        return true;
    }

    bool MetalRHI::RenderVisibilityBuffer(const VisibilityBufferDesc& desc)
    {
        if (!m_Impl->CheckThread("RenderVisibilityBuffer") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->DeferredActive || m_Impl->GBufferWritten || m_Impl->FrameSampleCount != 1)
            return m_Impl->Fail("Metal visibility geometry requires an unresolved single-sample deferred frame");
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

    bool MetalRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle handle)
    {
        if (!m_Impl->CheckThread("DrawGpuDrivenBatch") || !m_Impl->PassOpen || !handle.Frame ||
            handle.Frame != m_Impl->GpuEpoch || handle.Index >= m_Impl->GpuCursor || m_Impl->ActiveGpuBatch)
            return m_Impl->Fail("Metal indirect draw has an invalid frame token or pass state");
        auto& batch = m_Impl->GpuBatches[handle.Index];
        const auto& desc = batch.Description;
        if (!GpuDrivenDrawMatches(desc, m_Impl->BoundWorld, m_Impl->BoundViewProjection, m_Impl->BoundPipeline, m_Impl->CurrentPass))
            return m_Impl->Fail("Metal indirect draw does not match its prepared instance data");
        m_Impl->ActiveGpuBatch = &batch;
        struct Reset { Impl* State; ~Reset() { State->ActiveGpuBatch = nullptr; } } reset {m_Impl.get()};
        return DrawIndexedGeometry(desc.FirstIndex, desc.IndexCount, desc.BaseVertex, &desc.Instances);
    }

    bool MetalRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
        return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, nullptr);
    }

    bool MetalRHI::DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
        const std::vector<DrawInstance>& instances)
    {
        return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, &instances);
    }

    bool MetalRHI::DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
        const std::vector<DrawInstance>* instances)
	{
    if (m_Impl->PassOpen && m_Impl->Hair.Active && HairFurReplayPass(m_Impl->CurrentPass.Type) &&
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Hair replay ends the pass geometry stream");
		if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("Metal DrawIndexed is missing required state");
        if (m_Impl->LightListsActive && (m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::GBuffer ||
            m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) &&
            !m_Impl->LightListFrame.ValidateDraw(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, true, m_Impl->LastError)) return false;
        if (instances && m_Impl->CurrentPass.DepthPeelingLayer) return m_Impl->Fail("OIT replay requires ordinary invariant draws");
        if (instances && UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        return m_Impl->Fail("POM material capture currently requires individual mesh draws");
    std::optional<InstanceConstantBlock> instanceConstants;
        if (instances)
        {
            if (m_Impl->BoundPipeline.OutlineExtrusion ||
                m_Impl->CurrentPass.Type == PassType::Outline)
                return m_Impl->Fail("Metal instances do not support outline extrusion");
            instanceConstants.emplace();
            if (!BuildInstanceConstants(*instances, m_Impl->BoundWorld, *instanceConstants))
                return m_Impl->Fail("Metal instance transforms, winding or batch size are invalid");
            instanceConstants->ViewProjection = m_Impl->BoundViewProjection;
            instanceConstants->Material.X = m_Impl->BoundPipeline.Color.W * m_Impl->BoundPipeline.MaterialOpacity;
        }
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("Metal DrawIndexed buffers are invalid");
		if (!ValidateIndexedGeometryDraw(vertex->second.Description,index->second.Description,index->second.Bytes,
		    vertex->second.NaniteGeometry && index->second.NaniteGeometry,firstIndex,indexCount,baseVertex,m_Impl->LastError)) return false;

		ObjCObject shadowTexture = m_Impl->DummyShadowTexture;
        const bool vsm = m_Impl->UsesSurfaceVSM();
        const bool vsmCaster = m_Impl->UsesVSM() && m_Impl->CurrentPass.Type == PassType::Shadow;
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
        if (vsm)
        {
            if (shadowAvailable) { if (!m_Impl->MomentImage(m_Impl->LastShadowTexture, shadowTexture)) return false; }
            else shadowTexture = m_Impl->DummyColorTexture;
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
			(bool(m_Impl->FrameRayShadowImage) || shadowAvailable || (m_Impl->FrameVirtualShadowAtlas && m_Impl->BoundPipeline.SamplesShadowMap && m_Impl->CurrentPass.Type==PassType::Color)),
			m_Impl->BoundPipeline.DiffuseTexture != TextureHandle {},
			m_Impl->BoundPipeline.NormalTexture != TextureHandle {},
			m_Impl->BoundPipeline.OpacityTexture != TextureHandle {},
            m_Impl->HdrActive && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type != PassType::Shadow);
        const bool geometry = m_Impl->CurrentPass.Type == PassType::GBuffer;
        const bool velocity = m_Impl->CurrentPass.Type == PassType::Velocity;
        const bool contactDepth = m_Impl->CurrentPass.Type == PassType::ContactDepth;
        const bool forwardPlusDepth = m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth;
        const bool forwardPlus = forwardPlusDepth || m_Impl->UsesForwardPlusMaterial();
        const bool multisample = m_Impl->FrameSampleCount > 1 && !m_Impl->CurrentPass.Target &&
            m_Impl->CurrentPass.Type != PassType::Shadow && m_Impl->CurrentPass.Type != PassType::Outline &&
            m_Impl->CurrentPass.Type != PassType::EnvironmentCapture && m_Impl->CurrentPass.Type != PassType::PlanarCapture;
        ObjCObject pipeline = nullptr;
        if (m_Impl->CurrentPass.Type == PassType::Shadow)
            pipeline = m_Impl->FrameCascades.Frame.CascadeCount ? m_Impl->CascadeDepthPipeline : m_Impl->ShadowPipeline;
        else if (contactDepth)
        {
            pipeline = m_Impl->ContactDepthPipeline(instances != nullptr, m_Impl->FrameSampleCount);
            if (!pipeline || (instances && !m_Impl->UploadInstances(*instanceConstants))) return false;
        }
        else if (forwardPlus)
        {
            pipeline = m_Impl->ForwardPlusPipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,
                multisample ? m_Impl->FrameSampleCount : 1, instances != nullptr, forwardPlusDepth);
            if (!pipeline || (instances && !m_Impl->UploadInstances(*instanceConstants))) return false;
        }
        else if (velocity) pipeline = multisample ? m_Impl->SampleGroups[m_Impl->SamplePipelineKey()].Velocity : m_Impl->MotionPipelines[Impl::MotionGeometryPipeline];
        else if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        {
            pipeline=m_Impl->ParallaxMaterialPipeline(multisample?m_Impl->FrameSampleCount:1,m_Impl->BoundPipeline.ColorWrite);
            if (!pipeline) return false;
        }
        else if (geometry)
            pipeline = multisample ? m_Impl->SampleGroups[m_Impl->SamplePipelineKey()].Deferred[m_Impl->BoundPipeline.ColorWrite ? 0 : 1] :
                m_Impl->DeferredGeometryPipelines[m_Impl->BoundPipeline.ColorWrite ? 0 : 1];
        else
        {
            if (!m_Impl->SelectTechnique(m_Impl->BoundPipeline.Shading,
                m_Impl->HdrActive && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type != PassType::Shadow,
                multisample ? m_Impl->FrameSampleCount : 1)) return false;
            pipeline = m_Impl->BoundPipeline.ColorWrite ? m_Impl->ColorPipeline : m_Impl->ColorWriteDisabledPipeline;
        }
		if (!forwardPlus && m_Impl->CurrentPass.Type == PassType::Color
			&& m_Impl->BoundPipeline.ColorWrite
			&& m_Impl->BoundPipeline.BlendEnabled)
		{
			pipeline = m_Impl->ColorBlendPipeline;
		}
        if (m_Impl->WeightedPass)
        {
            if (!forwardPlus) pipeline = m_Impl->WeightedSurfacePipeline(m_Impl->BoundPipeline.Shading);
            if (!pipeline) return false;
            const auto transparency = BuildWeightedTransparencyConstants(m_Impl->FrameTransparency);
            Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
                "setFragmentBytes:length:atIndex:", &transparency, sizeof(transparency), TransparencyConstantBufferIndex);
        }
        if (vsmCaster)
        {
            pipeline = m_Impl->VSMCasterPipeline(instances != nullptr);
            if (!pipeline || (instances && !m_Impl->UploadInstances(*instanceConstants))) return false;
            auto moments = BuildVSMParameters(m_Impl->FrameShadowFilter.VSM);
            if (!BuildVSMEncoding(m_Impl->BoundViewProjection, moments.DepthEncoding[0], m_Impl->LastError)) return false;
            Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
                "setVertexBytes:length:atIndex:", &moments, sizeof(moments), VSMConstantBufferIndex);
            Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
                "setFragmentBytes:length:atIndex:", &moments, sizeof(moments), VSMConstantBufferIndex);
        }
        else if (instances && !contactDepth && !forwardPlus)
        {
            pipeline = m_Impl->InstancePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target &&
                m_Impl->CurrentPass.Type != PassType::Shadow, multisample ? m_Impl->FrameSampleCount : 1);
            if (!pipeline || !m_Impl->UploadInstances(*instanceConstants)) return false;
        }
        else if (!forwardPlus && m_Impl->UsesSurfaceContact())
        {
            pipeline = m_Impl->ContactSurfacePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,
                multisample ? m_Impl->FrameSampleCount : 1);
            if (!pipeline) return false;
        }
        else if (!forwardPlus && vsm)
        {
            pipeline = m_Impl->VSMSurfacePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,
                multisample ? m_Impl->FrameSampleCount : 1);
            if (!pipeline) return false;
        }
        else if (!forwardPlus && m_Impl->UsesSurfacePCSS())
        {
            pipeline = m_Impl->PCSSSurfacePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,
                multisample ? m_Impl->FrameSampleCount : 1);
            if (!pipeline) return false;
        }
        if (!instances && m_Impl->UsesMaterialStage(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        {
            const auto material=SelectMaterialTechnique(m_Impl->BoundPipeline,m_Impl->WeightedPass);
            pipeline=geometry?m_Impl->MaterialGeometryPipeline(material,multisample?m_Impl->FrameSampleCount:1):
                m_Impl->MaterialSurfacePipeline(material,m_Impl->HdrActive && !m_Impl->CurrentPass.Target,multisample?m_Impl->FrameSampleCount:1);
            if (!pipeline) return false;
        }
        if (!instances && m_Impl->UsesSurfacePCF())
        {
            pipeline=m_Impl->PCFSurfacePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,multisample?m_Impl->FrameSampleCount:1);
            if (!pipeline) return false;
        }
        if (m_Impl->UsesSurfacePCF() || m_Impl->UsesMaterialSurface(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
            m_Impl->BindPCF(m_Impl->RenderEncoder,shadowAvailable);
        if (!instances && !forwardPlus && m_Impl->FrameRayShadowImage && !m_Impl->FrameAreaLights && m_Impl->CurrentPass.Type==PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap)
        {
            pipeline=m_Impl->RayShadowSurfacePipeline(m_Impl->HdrActive && !m_Impl->CurrentPass.Target,multisample?m_Impl->FrameSampleCount:1);
            if (!pipeline) return false;
        }
        m_Impl->BindRayShadowImage(m_Impl->RenderEncoder);
        if (m_Impl->UsesSurfacePCSS() && !m_Impl->BindPCSS(m_Impl->RenderEncoder,
                m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting, shadowAvailable)) return false;
        if (vsm && !m_Impl->BindVSM(m_Impl->RenderEncoder,
                m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting, shadowAvailable)) return false;
        if (m_Impl->UsesForwardPlusMaterial() && !m_Impl->BindForwardPlusSurface()) return false;
        if (m_Impl->UsesSurfaceContact())
            Send<void, ObjCObject, NSUInteger>(m_Impl->RenderEncoder, "setFragmentTexture:atIndex:",
                m_Impl->ContactProducer.Result(), Impl::ContactTextureIndex);
		Send<void, ObjCObject>(m_Impl->RenderEncoder, "setRenderPipelineState:", pipeline);
		const bool wireframe = m_Impl->BoundPipeline.Fill == FillMode::Wireframe && !vsmCaster;
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setTriangleFillMode:",
			wireframe ? TriangleFillModeLines : TriangleFillModeFill);
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setCullMode:",
			m_Impl->BoundPipeline.OutlineExtrusion ? CullModeFront :
                (m_Impl->BoundPipeline.BackFaceCulling ? CullModeBack : CullModeNone));
		Send<void, NSUInteger>(m_Impl->RenderEncoder, "setFrontFacingWinding:",
            (RealtimeWorldIsMirrored(m_Impl->BoundWorld) != (m_Impl->CurrentPass.Type == PassType::PlanarCapture)) ?
                WindingClockwise : WindingCounterClockwise);
		if (m_Impl->CurrentPass.Type == PassType::Shadow)
			Send<void, float, float, float>(m_Impl->RenderEncoder, "setDepthBias:slopeScale:clamp:",
				0.0015f, 2.0f, 0.01f);
		else
			Send<void, float, float, float>(m_Impl->RenderEncoder, "setDepthBias:slopeScale:clamp:",
				0.0f, 0.0f, 0.0f);
		ObjCObject depthState = m_Impl->CurrentPass.DepthPeelingLayer ? m_Impl->PeelingEqualState : !m_Impl->BoundPipeline.DepthTest ? m_Impl->DepthDisabledState
			: (m_Impl->BoundPipeline.DepthWrite ? m_Impl->DepthWriteState : m_Impl->DepthReadState);
		Send<void, ObjCObject>(m_Impl->RenderEncoder, "setDepthStencilState:", depthState);
		Send<void, ObjCObject, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setVertexBuffer:offset:atIndex:", vertex->second.Buffer, 0, 0);
		Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setVertexBytes:length:atIndex:", &constants, sizeof(constants), 1);
		Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
			"setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 1);
        if (velocity)
        {
            MotionBlurConstants motion {};
            if (!BuildMotionGeometryConstants(m_Impl->BoundPipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
                    GetMotionTransform(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
                    bool(m_Impl->BoundPipeline.DiffuseTexture), bool(m_Impl->BoundPipeline.OpacityTexture), motion))
                return m_Impl->Fail("Metal velocity requires finite current and previous transforms");
            // Per-instance previous clip W, rather than the representative object,
            // determines whether this fragment has usable transform history.
            if (instances) motion.Options.W = 1;
            Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
                "setVertexBytes:length:atIndex:", &motion, sizeof(motion), 1);
            Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
                "setFragmentBytes:length:atIndex:", &motion, sizeof(motion), 1);
        }
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
			"setFragmentSamplerState:atIndex:", m_Impl->MaterialSampler(), 0);
        if (!m_Impl->BindCascades(m_Impl->RenderEncoder,
            m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap, vsm)) return false;
        const bool foggedSurface = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
            m_Impl->BoundPipeline.BlendEnabled && m_Impl->FogActive && m_Impl->FogResolved;
        const VolumetricFogConstants disabledFog {};
        const auto& surfaceFog = foggedSurface ? m_Impl->FogConstants : disabledFog;
        auto volume=m_Impl->LumenVolumeFrame;
        const bool translucentGI=m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->BoundPipeline.BlendEnabled && m_Impl->FrameGI.Lumen.Translucency.Enabled && volume.Grid.W!=0;
        if (!translucentGI) volume.Grid.W=0;
        constexpr NSUInteger LumenVolumeConstantIndex=9,LumenVolumeTextureIndex=19;
        Send<void,const void*,NSUInteger,NSUInteger>(m_Impl->RenderEncoder,"setFragmentBytes:length:atIndex:",&volume,sizeof(volume),LumenVolumeConstantIndex);
        Send<void,ObjCObject,NSUInteger>(m_Impl->RenderEncoder,"setFragmentTexture:atIndex:",translucentGI?m_Impl->LumenVolumeAtlas:m_Impl->DummyColorTexture,LumenVolumeTextureIndex);
        Send<void, const void*, NSUInteger, NSUInteger>(m_Impl->RenderEncoder,
            "setFragmentBytes:length:atIndex:", &surfaceFog, sizeof(surfaceFog), FogConstantBufferIndex);
        Send<void, ObjCObject, NSUInteger>(m_Impl->RenderEncoder, "setFragmentTexture:atIndex:",
            foggedSurface ? m_Impl->FogAtlas : m_Impl->DummyColorTexture, FogAtlasTextureIndex);
		if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type) && !m_Impl->BindParallaxDraw()) return false;
        const NSUInteger indexOffset = static_cast<NSUInteger>(firstIndex) * sizeof(std::uint32_t);
        if (m_Impl->ActiveGpuBatch)
            Send<void, NSUInteger, NSUInteger, ObjCObject, NSUInteger, ObjCObject, NSUInteger>(m_Impl->RenderEncoder,
                "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
                PrimitiveTypeTriangle, IndexTypeUInt32, index->second.Buffer, 0, m_Impl->ActiveGpuBatch->Arguments, 0);
        else if (m_Impl->ActiveNaniteDraw)
            Send<void, NSUInteger, NSUInteger, ObjCObject, NSUInteger, ObjCObject, NSUInteger>(m_Impl->RenderEncoder,
                "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
                PrimitiveTypeTriangle,IndexTypeUInt32,m_Impl->ActiveNaniteDraw->Binning?m_Impl->ActiveNaniteDraw->Arguments->Buffer:index->second.Buffer,0,m_Impl->ActiveNaniteDraw->Arguments->Buffer,
                m_Impl->NaniteArgumentOffset());
        else Send<void, NSUInteger, NSUInteger, NSUInteger, ObjCObject, NSUInteger, NSUInteger, NSInteger, NSUInteger>(
			m_Impl->RenderEncoder,
			"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
			"instanceCount:baseVertex:baseInstance:",
			PrimitiveTypeTriangle, static_cast<NSUInteger>(indexCount), IndexTypeUInt32,
			index->second.Buffer, indexOffset, instances ? instances->size() : 1, static_cast<NSInteger>(baseVertex), 0);

		if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type)) ++m_Impl->PendingFrame.Statistics.ParallaxMaterialDrawCount;
    DrawCallRecord record;
		record.VertexBuffer = m_Impl->BoundVertexBuffer;
		record.IndexBuffer = m_Impl->BoundIndexBuffer;
		record.Pipeline = m_Impl->BoundPipeline;
		record.WorldTransform = m_Impl->BoundWorld;
		record.Motion = GetMotionTransform();
        if (instances) record.Instances = *instances;
		record.GPUDriven = m_Impl->ActiveGpuBatch != nullptr;
        record.NaniteIndirect=m_Impl->ActiveNaniteDraw!=nullptr; record.NaniteQueryIndex=record.NaniteIndirect?m_Impl->ActiveNaniteQuery:0;
		record.FirstIndex = firstIndex;
		record.IndexCount = indexCount;
		record.BaseVertex = baseVertex;
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
		FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
		++statistics.DrawCallCount;
        statistics.InstancedDrawCallCount += instances ? 1u : 0u;
		if (m_Impl->ActiveGpuBatch)
        {
            ++statistics.IndirectDrawCallCount;
            statistics.GpuCandidateInstanceCount += instances->size();
            statistics.GpuCandidateTriangleCount += std::uint64_t(indexCount / 3) * instances->size();
        }
        else if (m_Impl->ActiveNaniteDraw)
        {
            ++statistics.IndirectDrawCallCount; ++statistics.NaniteIndirectDrawCount;
            statistics.NaniteCandidateTriangles+=indexCount/3;
        }
        else statistics.SubmittedTriangleCount += (indexCount / 3) * (instances ? instances->size() : 1);
		statistics.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1u : 0u;
		statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1u : 0u;
        statistics.ContactDepthDrawCallCount += contactDepth ? 1u : 0u;
        statistics.ForwardPlusDepthDrawCallCount += forwardPlusDepth ? 1u : 0u;
		return true;
	}

	bool MetalRHI::EndPass()
	{
    if (m_Impl->FrameOpen && m_Impl->PassOpen && m_Impl->Hair.Active && !m_Impl->Hair.Frame.Geometry->Indices.empty() &&
        HairFurReplayPass(m_Impl->CurrentPass.Type) && (m_Impl->CurrentPass.Type!=PassType::PlanarCapture || m_Impl->FramePlanar.Active()) &&
        m_Impl->Hair.ReplayedPass!=m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Complete requested hair replay before ending its native geometry pass");
		if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
			return m_Impl->Fail("Metal has no open pass");
		Send<void>(m_Impl->RenderEncoder, "endEncoding");
		m_Impl->RenderEncoder = nullptr;
        if (m_Impl->CurrentPass.DepthPeelingLayer && !m_Impl->PeelingSequence.Advance(DepthPeelingEvent::EndShading,m_Impl->LastError)) return false;
        m_Impl->PassOpen = false;
        if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->UsesVSM() &&
            !m_Impl->FilterMoments(m_Impl->CurrentPass.Target)) return false;
        if (m_Impl->CurrentPass.Type == PassType::EnvironmentCapture)
        {
            m_Impl->EnvironmentFacesWritten |= 1u << m_Impl->CurrentPass.EnvironmentFaceIndex;
            ++m_Impl->PendingFrame.Statistics.EnvironmentCaptureFaceCount;
        }
        if (m_Impl->CurrentPass.Type == PassType::PlanarCapture)
        {
            m_Impl->PlanarWritten = true;
            ++m_Impl->PendingFrame.Statistics.PlanarCapturePassCount;
        }
        if (m_Impl->CurrentPass.Type == PassType::GBuffer) m_Impl->GBufferWritten = true;
        if (m_Impl->CurrentPass.Type == PassType::ContactDepth) m_Impl->ContactDepthWritten = true;
        if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) m_Impl->ForwardPlusDepthWritten = true;
        if (m_Impl->CurrentPass.Type == PassType::Velocity) m_Impl->VelocityCaptured = true;
        if (m_Impl->WeightedPass) m_Impl->WeightedAccumulated = true;
        if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target)
        {
            if (!m_Impl->DeferredActive) m_Impl->ForwardColorWritten = true;
            if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved)
            {
                m_Impl->SamplesTransparentWritten = true;
                m_Impl->SamplesColorResolved = false;
            }
        }
        m_Impl->WeightedPass = false;
		m_Impl->PassOpen = false;
		return true;
	}

    bool MetalRHI::PrepareLightListFrame(const LightListFrameDesc& desc)
    {
        if (!m_Impl->CheckThread("PrepareLightListFrame") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            desc.Layout != (m_Impl->ClusteredActive ? LightListLayout::Clusters : LightListLayout::Tiles) || !m_Impl->LightListsActive || !m_Impl->PendingFrame.Passes.empty())
            return m_Impl->Fail("Metal Forward+ light storage must precede every frame pass");
        return m_Impl->LightListFrame.Prepare(desc, m_Impl->LastError) && m_Impl->UploadForwardPlusScene();
    }

    bool MetalRHI::GenerateLightLists()
    {
        if (!m_Impl->CheckThread("GenerateLightLists") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->LightListsActive || !m_Impl->LightListFrame.Prepared() || m_Impl->LightListFrame.Assigned() ||
            (!m_Impl->ClusteredActive && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten) || m_Impl->ForwardColorWritten || m_Impl->ContactGenerated)
            return m_Impl->Fail("Metal Forward+ assignment requires completed opaque depth, once before visibility and lighting");
        const auto& frame = m_Impl->LightListFrame.Description();
        if (m_Impl->ClusteredActive)
        {
            if (!m_Impl->ClusteredProducer.Record(m_Impl->CommandBuffer, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
                frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
                !m_Impl->LightListFrame.CompleteAssignment(m_Impl->ClusteredProducer.Grid(), m_Impl->LastError)) return false;
            ++m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount;
            return true;
        }
        // Record ends its compute encoder. Metal tracked hazards order the
        // original per-sample depth, list writes and subsequent fragment reads.
        if (!m_Impl->TileProducer().Record(m_Impl->CommandBuffer,
                m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Depth : m_Impl->DepthTexture,
                frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
            !m_Impl->LightListFrame.CompleteAssignment(m_Impl->TileProducer().Grid(), m_Impl->LastError)) return false;
        ++(m_Impl->TiledActive ? m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount : m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount);
        return true;
    }

    bool MetalRHI::GenerateContactShadows(const ContactShadowDesc& desc)
    {
        if (!m_Impl->CheckThread("GenerateContactShadows") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->FrameContact.Enabled || m_Impl->ContactGenerated || m_Impl->ForwardColorWritten || m_Impl->DeferredResolved ||
            m_Impl->SamplesOpaqueResolved || (!m_Impl->ContactDepthWritten && !m_Impl->GBufferWritten && !m_Impl->ForwardPlusDepthWritten))
            return m_Impl->Fail("Metal contact visibility requires completed opaque depth, once before lighting and resolve");
        if (m_Impl->LightListsActive)
        {
            if (!m_Impl->LightListFrame.Assigned()) return m_Impl->Fail("Metal contact visibility must follow Forward+ assignment");
            if (!m_Impl->LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
        }
        auto lighting = desc.Lighting;
        if (lighting.SceneLights) lighting.SceneLights->WriteLegacySubset(lighting);
        if (!m_Impl->Hair.ValidateContactProducer(desc.ViewProjection,lighting,m_Impl->LastError)) return false;
        const auto light = lighting.ShadowLightIndex;
        const bool active = light >= 0 && light < int(std::min(lighting.LightCount, MaximumSceneLights)) &&
            lighting.Lights[light].CastShadows;
        if (active)
        {
            ContactShadowDispatch trace; trace.ViewProjection = desc.ViewProjection; trace.Light = lighting.Lights[light];
            trace.MaximumDistance = m_Impl->FrameContact.MaximumDistance;
            trace.Parameters.StartBias = m_Impl->FrameContact.StartBias;
            trace.Parameters.Thickness = m_Impl->FrameContact.Thickness;
            trace.Parameters.MaximumSteps = m_Impl->FrameContact.MaximumSteps;
            if (!m_Impl->ContactProducer.Record(m_Impl->CommandBuffer,
                    m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Depth : m_Impl->DepthTexture, trace, m_Impl->LastError)) return false;
            ++m_Impl->PendingFrame.Statistics.ContactShadowPassCount;
        }
        m_Impl->ContactReceiverActive = active;
        m_Impl->Hair.RecordContact(desc.ViewProjection,lighting,m_Impl->ContactReceiverActive);
        m_Impl->ContactGenerated = true;
        return true;
    }

    bool MetalRHI::DrawHairFurVelocity(const HairFurVelocityDesc& desc)
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

bool MetalRHI::DrawHairFurInPass(const HairFurDesc& desc)
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

bool MetalRHI::DrawHairFur(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFur") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved || !m_Impl->GBufferWritten ||
        !(m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardColorWritten) || m_Impl->SamplesOpaqueResolved || m_Impl->VelocityCaptured ||
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

bool MetalRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc)
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

bool MetalRHI::DrawDeferredLighting(const DeferredLightingDesc& desc)
    {
        if (desc.Output == DeferredOutputMode::DirectDiffuse && m_Impl->TiledActive)
            return m_Impl->Fail("Direct-diffuse output requires ordinary deferred lighting");
        if (!m_Impl->CheckThread("DrawDeferredLighting")) return false;
        if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
            return m_Impl->Fail("Metal deferred lighting requires completed contact visibility");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->DeferredActive || !m_Impl->GBufferWritten ||
            (m_Impl->DeferredResolved && !m_Impl->SubsurfaceProducing) || desc.Shading != GetTechniqueSettings().Shading ||
            desc.Shading == Techniques::ShadingModel::Gouraud)
            return m_Impl->Fail("Metal lighting requires one completed G-buffer with the frame shading model");
        const bool multisample = m_Impl->FrameSampleCount > 1;
        const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        const bool shadowAvailable = shadow != m_Impl->Textures.end() &&
            shadow->second.Description.Format == TextureFormat::Depth32 && shadow->second.Texture;
        DeferredConstants constants{};
        if (!BuildDeferredConstants(desc, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || bool(m_Impl->FrameVirtualShadowAtlas)), m_Impl->DeferredBackground, constants))
            return m_Impl->Fail("Metal deferred lighting requires a finite, invertible camera projection");
        const bool pcss = m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
        const bool vsm = m_Impl->UsesVSM();
        const auto lightingPipeline = m_Impl->TiledActive ? m_Impl->TiledLightingPipeline() : m_Impl->UsesDedicatedPCF()?m_Impl->PCFScreenPipeline(multisample?6:5,m_Impl->FrameSampleCount): m_Impl->FrameRayShadowImage ? m_Impl->RayShadowScreenPipeline(multisample?6:5,m_Impl->FrameSampleCount) : m_Impl->ContactReceiverActive ? m_Impl->ContactLightingPipeline(pcss, vsm, m_Impl->FrameSampleCount) :
            vsm ? m_Impl->VSMScreenPipeline(multisample ? Impl::VSMMultisample : Impl::VSMDeferred, m_Impl->FrameSampleCount) :
            pcss ? m_Impl->PCSSScreenPipeline(multisample ? 6 : 5, m_Impl->FrameSampleCount) :
            multisample ? m_Impl->SampleGroups[m_Impl->SamplePipelineKey()].Deferred[2] : m_Impl->DeferredLightingPipeline;
        if (!lightingPipeline) return false;
        ObjCObject shadowTexture = vsm ? m_Impl->DummyColorTexture : m_Impl->DummyShadowTexture;
        if (shadowAvailable && desc.SamplesShadowMap)
        {
            shadowTexture = shadow->second.Texture;
            if (vsm && !m_Impl->MomentImage(m_Impl->LastShadowTexture, shadowTexture)) return false;
        }
        // A color-only encoder avoids binding opaque depth as an attachment while
        // the fragment stage reads it. The next forward pass loads the same depth.
        ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
        ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
        ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
        Send<void, ObjCObject>(attachment, "setTexture:", multisample ? m_Impl->Multisample.Colors[Impl::SampleColor] : m_Impl->HdrTextures[0]);
        Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
        Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
        ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
        if (!encoder) return m_Impl->Fail("Metal failed to create deferred lighting encoder");
        const MTLViewport viewport{0.0, 0.0, static_cast<double>(m_Impl->CreateInfo.Width),
            static_cast<double>(m_Impl->CreateInfo.Height), 0.0, 1.0};
        Send<void, MTLViewport>(encoder, "setViewport:", viewport);
        Send<void, ObjCObject>(encoder, "setRenderPipelineState:", lightingPipeline);
        if (m_Impl->ContactReceiverActive)
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:",
                m_Impl->ContactProducer.Result(), Impl::ContactTextureIndex);
        m_Impl->BindRayShadowImage(encoder);
        if (m_Impl->UsesDedicatedPCF()) m_Impl->BindPCF(encoder,shadowAvailable && desc.SamplesShadowMap);
        if (pcss && !m_Impl->BindPCSS(encoder, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap && shadowAvailable))
        { Send<void>(encoder, "endEncoding"); return false; }
        if (vsm && !m_Impl->BindVSM(encoder, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap && shadowAvailable))
        { Send<void>(encoder, "endEncoding"); return false; }
        Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
        Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
        Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
        if (m_Impl->TiledActive)
        {
            RealtimeDrawConstants surface;
            TiledDeferredReconstruction reconstruction;
            LightListSurfaceConstants selection;
            if (!BuildTiledDeferredConstants(desc, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || bool(m_Impl->FrameVirtualShadowAtlas)), m_Impl->DeferredBackground, surface, reconstruction) ||
                !m_Impl->LightListFrame.Surface(desc.Lighting, desc.ViewProjection, LightListSelection::Opaque, selection, m_Impl->LastError))
            { Send<void>(encoder, "endEncoding"); return false; }
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &reconstruction, sizeof(reconstruction), 0);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &surface, sizeof(surface), 1);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &selection, sizeof(selection), ForwardPlusSelectionIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setFragmentBuffer:offset:atIndex:", m_Impl->ForwardPlusSceneBuffer, 0, ForwardPlusSceneIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setFragmentBuffer:offset:atIndex:", m_Impl->TiledProducer.Headers(), 0, ForwardPlusHeaderIndex);
            Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setFragmentBuffer:offset:atIndex:", m_Impl->TiledProducer.Indices(), 0, ForwardPlusLightIndex);
            if (!pcss && !vsm)
            {
                PCSSConstants filter {};
                filter.EmitterAndLimits = BuildCascadedShadowConstants({}, m_Impl->FrameShadowFilter).Filtering;
                Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &filter, sizeof(filter), PCSSConstantBufferIndex);
            }
        }
        for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", multisample ? m_Impl->Multisample.Colors[Impl::SampleMaterialFirst + index] : m_Impl->GBufferTextures[index], index);
        Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", multisample ? m_Impl->Multisample.Depth : m_Impl->DepthTexture, GBufferAttachmentCount);
        Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:",
            shadowTexture, GBufferAttachmentCount + 1);
        if (!m_Impl->BindCascades(encoder, desc.SamplesShadowMap, vsm))
        { Send<void>(encoder, "endEncoding"); return false; }
        Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
        Send<void>(encoder, "endEncoding");
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
        return true;
    }

    bool MetalRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyAmbientOcclusion")) return false;
        if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
            return m_Impl->Fail("Metal ApplyAmbientOcclusion requires resolved multisample inputs");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->AmbientOcclusionActive || m_Impl->AmbientOcclusionResolved ||
            m_Impl->ReflectionResolved || m_Impl->DepthOfFieldResolved || m_Impl->VelocityCaptured ||
            !m_Impl->GBufferWritten || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Metal ambient occlusion requires completed opaque color and material inputs once per frame");
        RayTracedAOConstants packet{};
        auto& constants = packet.Surface;
        const bool rayTraced = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::RayTraced;
        if (!BuildAmbientOcclusionFrameConstants(desc, m_Impl->FrameAmbientOcclusionSettings, m_Impl->FrameRayScene.get(), m_Impl->CreateInfo.Width,
            m_Impl->CreateInfo.Height, Techniques::AmbientOcclusionPassKind::Estimate, packet))
            return m_Impl->Fail("Metal ambient occlusion requires a finite, invertible camera projection");
        const bool scalable = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::SAO;
        const bool groundTruth = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::GTAO;
        const auto& pipelines = rayTraced ? m_Impl->RayAOPipelines : groundTruth ? m_Impl->GtaoPipelines : scalable ? m_Impl->SaoPipelines :
            m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::HBAO
                ? m_Impl->HbaoPipelines : m_Impl->SsaoPipelines;
        const auto plan = Techniques::BuildAmbientOcclusionPlan(m_Impl->FrameAmbientOcclusionSettings,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
        const auto maxMip = scalable ? AmbientOcclusionDepthMipCount(m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height) - 1 : 0;
        std::uint32_t destinationMip = 0;
        for (const auto pass : plan)
        {
            const auto stage = static_cast<std::uint32_t>(pass);
            const bool prepare = pass == Techniques::AmbientOcclusionPassKind::DepthPrepare;
            const bool reduce = pass == Techniques::AmbientOcclusionPassKind::DepthReduce;
            const bool hierarchy = prepare || reduce;
            if (reduce) ++destinationMip;
            const int width = hierarchy ? std::max(1, m_Impl->CreateInfo.Width >> destinationMip) : m_Impl->CreateInfo.Width;
            const int height = hierarchy ? std::max(1, m_Impl->CreateInfo.Height >> destinationMip) : m_Impl->CreateInfo.Height;
            const MTLViewport viewport{0.0, 0.0, static_cast<double>(width), static_cast<double>(height), 0.0, 1.0};
            const bool composite = pass == Techniques::AmbientOcclusionPassKind::Composite;
            const std::size_t destination = stage == 1 ? 1 : 0;
            // The raw stage does not sample previous AO; bind a distinct valid
            // material image so no unused slot aliases its destination.
            ObjCObject previous = hierarchy || stage == 0 ? m_Impl->GBufferTextures[3] : m_Impl->AmbientOcclusionTextures[stage == 2 ? 1 : 0];
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(attachment, "setTexture:", hierarchy ? m_Impl->SaoDepthScratch :
                composite ? m_Impl->HdrTextures[0] : m_Impl->AmbientOcclusionTextures[destination]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", composite ? LoadActionLoad : LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            // Depth is sampled, so the fullscreen pass has no depth attachment.
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create AO encoder");
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            const std::size_t pipelineIndex = prepare ? 2 : reduce ? 3 : composite ? 1 : 0;
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", pipelines[pipelineIndex]);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            constants.Options.X = stage;
            constants.Options.Y = hierarchy ? destinationMip : 0;
            constants.Options.Z = maxMip;
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &packet, sizeof(packet), 0);
            if (rayTraced) for (NSUInteger slot=0;slot<RaySceneBufferCount;++slot)
                Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setFragmentBuffer:offset:atIndex:",
                    m_Impl->RaySceneBuffers[slot],0,slot+1);
            ObjCObject depthInput = scalable && !prepare ? m_Impl->SaoDepthHierarchy : m_Impl->DepthTexture;
            const std::array<ObjCObject, 4> inputs{depthInput, m_Impl->GBufferTextures[1], m_Impl->GBufferTextures[3], previous};
            for (std::size_t index = 0; index < inputs.size(); ++index)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[index], index);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
            if (hierarchy)
            {
                // A distinct scratch texture eliminates sampled/rendered mip aliasing.
                ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
                if (!blit) return m_Impl->Fail("Metal failed to create SAO depth copy encoder");
                const MTLOrigin origin{0, 0, 0};
                const MTLSize size{static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1};
                Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject,
                    NSUInteger, NSUInteger, MTLOrigin>(blit,
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:"
                    "toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                    m_Impl->SaoDepthScratch, 0, 0, origin, size, m_Impl->SaoDepthHierarchy, 0, destinationMip, origin);
                Send<void>(blit, "endEncoding");
            }
            m_Impl->PendingFrame.AmbientOcclusionPasses.push_back(pass);
        }
        m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount = static_cast<std::uint32_t>(plan.size());
        m_Impl->AmbientOcclusionResolved = true;
        return true;
    }

    bool MetalRHI::CompleteEnvironmentCapture()
    {
        constexpr std::uint32_t AllFaces = (1u << EnvironmentFaceCount) - 1u;
        if (!m_Impl->CheckThread("CompleteEnvironmentCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || (!m_Impl->FrameEnvironment.Source && m_Impl->EnvironmentFacesWritten != AllFaces))
            return m_Impl->Fail("Metal complete all six distinct cube faces before generating mips");
        if (const auto& source=m_Impl->FrameEnvironment.Source)
        {
            constexpr std::size_t UploadRowAlignment=256;
            const auto rowBytes=static_cast<std::size_t>(source->Resolution)*4*sizeof(std::uint16_t);
            const auto pitch=AlignUp(rowBytes,UploadRowAlignment),faceBytes=pitch*source->Resolution;
            ObjCObject staging=Send<ObjCObject,NSUInteger,NSUInteger>(m_Impl->Device,"newBufferWithLength:options:",faceBytes*EnvironmentFaceCount,0);
            if (!staging) return m_Impl->Fail("Metal failed to allocate HDR environment upload");
            auto* mapped=static_cast<std::uint8_t*>(Send<void*>(staging,"contents"));
            for (std::size_t face=0;face<EnvironmentFaceCount;++face)
                for (std::size_t row=0;row<source->Resolution;++row)
                    std::memcpy(mapped+face*faceBytes+row*pitch,source->FacesRGBA16[face].data()+row*source->Resolution*4,rowBytes);
            ObjCObject blit=Send<ObjCObject>(m_Impl->CommandBuffer,"blitCommandEncoder");
            if (!blit) { ReleaseObject(staging); return m_Impl->Fail("Metal failed to encode HDR environment upload"); }
            const MTLSize size{source->Resolution,source->Resolution,1}; const MTLOrigin origin{0,0,0};
            for (NSUInteger face=0;face<EnvironmentFaceCount;++face)
                Send<void,ObjCObject,NSUInteger,NSUInteger,NSUInteger,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
                    "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
                    "toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                    staging,face*faceBytes,pitch,faceBytes,size,m_Impl->EnvironmentTexture,face,0,origin);
            Send<void>(blit,"endEncoding");
            // The frame command buffer retains referenced resources through completion.
            ReleaseObject(staging);
        }
        const bool latitudeLongitude = m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
        if (latitudeLongitude && !m_Impl->ConvertEnvironmentMap()) return false;
        // Sequential encoders and tracked private textures order capture, mip
        // generation and sampling. The unsubmitted command buffer is discarded
        // on abort; the next frame clears all six faces before sampling again.
        // https://developer.apple.com/documentation/metal/mtlblitcommandencoder/generatemipmaps(for:)
        ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
        if (!blit) return m_Impl->Fail("Metal failed to create environment mip encoder");
        // The image chain includes the final 2x1 -> 1x1 reduction. No cube mips
        // are needed when conversion samples only the six base faces.
        Send<void, ObjCObject>(blit, "generateMipmapsForTexture:",
            latitudeLongitude ? m_Impl->EnvironmentMapTexture : m_Impl->EnvironmentTexture);
        Send<void>(blit, "endEncoding");
        m_Impl->EnvironmentReady = true;
        return true;
    }

    bool MetalRHI::CompletePlanarCapture()
    {
        if (!m_Impl->CheckThread("CompletePlanarCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->PlanarActive || !m_Impl->PlanarWritten || m_Impl->PlanarReady)
            return m_Impl->Fail("Metal planar mips require one completed capture exactly once per frame");
        // Device-created textures use tracked hazards on this MTLCommandQueue.
        // Sequential render/blit encoders order each reduction and copy without
        // sampling an attached image. Aborting discards the unsubmitted buffer;
        // the next frame clears its capture before lookup, including inactive planes.
        // https://developer.apple.com/documentation/metal/resource-synchronization
        for (std::uint32_t mip = 1; mip < m_Impl->FramePlanar.MipCount; ++mip)
        {
            const auto width = std::max(1u, m_Impl->FramePlanar.Width >> mip);
            const auto height = std::max(1u, m_Impl->FramePlanar.Height >> mip);
            PlanarReflectionConstants constants{};
            constants.Viewport = {float(width), float(height), 0, 0};
            constants.Options.X = mip - 1;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject color = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(color, "setTexture:", m_Impl->PlanarScratch);
            Send<void, NSUInteger>(color, "setLoadAction:", LoadActionClear);
            Send<void, NSUInteger>(color, "setStoreAction:", StoreActionStore);
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create planar mip encoder");
            const MTLViewport viewport{0, 0, static_cast<double>(width), static_cast<double>(height), 0, 1};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", m_Impl->PlanarMipPipeline);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->PlanarTexture, 0);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
            ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
            if (!blit) return m_Impl->Fail("Metal failed to create planar mip copy encoder");
            const MTLOrigin origin{0, 0, 0};
            const MTLSize size{width, height, 1};
            Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject,
                NSUInteger, NSUInteger, MTLOrigin>(blit,
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:"
                "toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                m_Impl->PlanarScratch, 0, 0, origin, size, m_Impl->PlanarTexture, 0, mip, origin);
            Send<void>(blit, "endEncoding");
            ++m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
        }
        m_Impl->PlanarReady = true;
        return true;
    }

    bool MetalRHI::ApplyPathTracing(const PathTracingDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyPathTracing") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PathFrame.Active || m_Impl->PathFrame.Resolved || !m_Impl->HdrActive || m_Impl->ForwardColorWritten)
        return m_Impl->Fail("Path tracing requires an untouched HDR frame with primary-ray visibility");
    return m_Impl->TracePaths(desc);
}

bool MetalRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc& desc)
    {
    if (m_Impl->FrameOpen && m_Impl->EarlyTemporalActive && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyGlobalIllumination") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->GIActive || m_Impl->GI.Resolved || !m_Impl->HdrActive || !m_Impl->GBufferWritten ||
            (m_Impl->EarlyTemporalActive && !m_Impl->VelocityCaptured) ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || (m_Impl->EarlyTemporalActive && !m_Impl->SamplesVelocityResolved))) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) || m_Impl->ReflectionResolved ||
            m_Impl->FogResolved || m_Impl->LightShaftResolved || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
            return m_Impl->Fail("Metal GI requires opaque materials/depth before reflections and volumes, with velocity for SSGI");
        if (Techniques::UsesLumen(m_Impl->FrameGI)) return m_Impl->ApplyLumenGI(desc);
        if (m_Impl->FrameVoxelGI.Volume) return m_Impl->ApplyVoxelGI(desc);
        if (Techniques::UsesRayTracedGI(m_Impl->FrameGI)) return m_Impl->ApplyRayGI(desc);
        if (m_Impl->LightmapActive) return m_Impl->ResolveLightmaps(desc);
        if (m_Impl->ProbeActive) return m_Impl->ResolveProbes(desc);
        GlobalIlluminationConstants constants;
        if (!BuildGlobalIlluminationConstants(desc, m_Impl->FrameGI, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->GI.Valid, constants))
            return m_Impl->Fail("Metal SSGI requires invertible current and previous cameras");
        const unsigned read = m_Impl->GI.ReadIndex, write = m_Impl->GI.WriteIndex();
        const MTLViewport viewport {0, 0, double(m_Impl->CreateInfo.Width), double(m_Impl->CreateInfo.Height), 0, 1};
        using Phase = Techniques::GlobalIlluminationPassKind;
        for (auto phase : {Phase::Trace, Phase::Spatial, Phase::Temporal, Phase::Composite})
        {
            constants.Options.X = static_cast<unsigned>(phase);
            const auto target = phase == Phase::Trace ? Impl::GIRaw : phase == Phase::Spatial ? Impl::GIFiltered :
                phase == Phase::Temporal ? Impl::GIColorFirst + write : Impl::GIComposite;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            const std::size_t outputIndices[] {target, Impl::GIDepthFirst + write, Impl::GINormalFirst + write};
            const NSUInteger outputCount = phase == Phase::Temporal ? 3 : 1;
            for (NSUInteger i = 0; i < outputCount; ++i)
            {
                ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", i);
                Send<void, ObjCObject>(attachment, "setTexture:", m_Impl->GITextures[outputIndices[i]]);
                Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
                Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            }
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create SSGI encoder");
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", phase == Phase::Temporal ? m_Impl->GITemporalPipeline : m_Impl->GIPipeline);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            const auto estimate = phase == Phase::Spatial ? m_Impl->GITextures[Impl::GIRaw] : phase == Phase::Temporal ?
                m_Impl->GITextures[Impl::GIFiltered] : phase == Phase::Composite ? m_Impl->GITextures[Impl::GIColorFirst + write] : m_Impl->HdrTextures[0];
            const ObjCObject inputs[] {m_Impl->HdrTextures[0], m_Impl->DepthTexture, m_Impl->GBufferTextures[1],
                m_Impl->GBufferTextures[0], m_Impl->GBufferTextures[3], m_Impl->MotionTextures[Impl::MotionVelocity], estimate,
                m_Impl->GITextures[Impl::GIColorFirst + read], m_Impl->GITextures[Impl::GIDepthFirst + read], m_Impl->GITextures[Impl::GINormalFirst + read]};
            for (NSUInteger i = 0; i < std::size(inputs); ++i)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[i], i);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
        }
        // Tracked private textures serialize render and copy encoders. Publishing
        // the pending history slot is deferred until command completion/present.
        ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
        if (!blit) return m_Impl->Fail("Metal failed to create SSGI composite copy encoder");
        const MTLOrigin origin {0, 0, 0};
        const MTLSize size {static_cast<NSUInteger>(m_Impl->CreateInfo.Width), static_cast<NSUInteger>(m_Impl->CreateInfo.Height), 1};
        Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject, NSUInteger, NSUInteger, MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            m_Impl->GITextures[Impl::GIComposite], 0, 0, origin, size, m_Impl->HdrTextures[0], 0, 0, origin);
        Send<void>(blit, "endEncoding");
        m_Impl->GI.Resolved = true;
        m_Impl->PendingFrame.Temporal = desc.Sampling;
        m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount = 4;
        return true;
    }

    bool MetalRHI::ApplyReflections(const ReflectionDesc& desc)
    {
        if (desc.SkyboxOnly)
        {
            const auto saved=m_Impl->FrameReflections;
            const bool active=m_Impl->ReflectionActive,done=m_Impl->ReflectionResolved,planar=m_Impl->PlanarActive;
            m_Impl->FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
            m_Impl->FrameReflections.EnvironmentIntensity=0;
            m_Impl->FrameReflections.Planar.Enabled=false;
            m_Impl->ReflectionActive=true; m_Impl->ReflectionResolved=false; m_Impl->PlanarActive=false;
            auto request=desc; request.SkyboxOnly=false; request.DrawSkybox=true;
            const bool success=m_Impl->EnsureReflectionResources(false) && ApplyReflections(request);
            m_Impl->FrameReflections=saved; m_Impl->ReflectionActive=active; m_Impl->ReflectionResolved=done; m_Impl->PlanarActive=planar;
            if (success) m_Impl->PendingFrame.Statistics.SkyboxPassCount+=2;
            return success;
        }

    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyReflections")) return false;
        if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
            return m_Impl->Fail("Metal reflections require resolved sample radiance, depth and materials");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->ReflectionActive || m_Impl->ReflectionResolved ||
            !m_Impl->PendingFrame.ReflectionPasses.empty() || !m_Impl->HdrActive || !m_Impl->GBufferWritten ||
            (m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady) ||
            (m_Impl->PlanarActive && !m_Impl->PlanarReady) ||
            (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->DepthOfFieldResolved ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved))
            return m_Impl->Fail("Metal reflections require opaque lighting/materials and AO, once before velocity and transparency");
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
            const bool cube = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::Cubemap;
            const bool latitudeLongitude = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
            const bool valid = planarPass ? BuildPlanarReflectionConstants(desc, m_Impl->FramePlanar,
                m_Impl->FrameReflections, m_Impl->FrameReflectionShading, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, planarConstants) :
                latitudeLongitude ? BuildEnvironmentMapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
                m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->EnvironmentMapDimensions.MipCount, mapConstants) :
                cube ? BuildCubemapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
                m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameEnvironment.MipCount, cubeConstants) :
                BuildReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
                    m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants);
            if (!valid) return m_Impl->Fail("Metal reflections require a finite invertible projection");
            const auto& pipelines = planarPass ? m_Impl->PlanarPipelines : latitudeLongitude ? m_Impl->EnvironmentMapPipelines : cube ? m_Impl->CubeReflectionPipelines : m_Impl->ReflectionPipelines;
            const MTLViewport viewport{0, 0, static_cast<double>(m_Impl->CreateInfo.Width),
                static_cast<double>(m_Impl->CreateInfo.Height), 0, 1};
            for (const auto pass : {planarPass ? Techniques::ReflectionPassKind::PlanarLookup : Techniques::UsesEnvironmentCapture(m_Impl->FrameReflections) ? Techniques::ReflectionPassKind::EnvironmentLookup : Techniques::ReflectionPassKind::Trace,
                Techniques::ReflectionPassKind::Composite})
            {
                const bool composite = pass == Techniques::ReflectionPassKind::Composite;
                ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
                ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
                ObjCObject color = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
                Send<void, ObjCObject>(color, "setTexture:", composite ? m_Impl->HdrTextures[0] : m_Impl->ReflectionTexture);
                Send<void, NSUInteger>(color, "setLoadAction:", composite ? LoadActionLoad : LoadActionClear);
                Send<void, NSUInteger>(color, "setStoreAction:", StoreActionStore);
                // Sequential encoders own disjoint inputs/outputs. No depth attachment
                // aliases sampled D32, and inactive slots use correctly typed dummies.
                // setFragmentBytes snapshots the rows and phase for each encoded draw.
                ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
                if (!encoder) return m_Impl->Fail("Metal failed to create reflection encoder");
                Send<void, MTLViewport>(encoder, "setViewport:", viewport);
                Send<void, ObjCObject>(encoder, "setRenderPipelineState:", pipelines[composite ? 1 : 0]);
                Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
                Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
                const std::array<ObjCObject, 7> inputs = composite ?
                    std::array<ObjCObject, 7>{m_Impl->DummyColorTexture, m_Impl->DummyShadowTexture,
                        m_Impl->DummyColorTexture, m_Impl->DummyColorTexture, m_Impl->DummyColorTexture,
                        m_Impl->DummyColorTexture, m_Impl->ReflectionTexture} :
                    std::array<ObjCObject, 7>{m_Impl->HdrTextures[0], m_Impl->DepthTexture,
                        m_Impl->GBufferTextures[1], m_Impl->GBufferTextures[2], m_Impl->GBufferTextures[0],
                        m_Impl->GBufferTextures[3], m_Impl->DummyColorTexture};
                for (std::size_t index = 0; index < inputs.size(); ++index)
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[index], index);
                if (planarPass)
                {
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->PlanarTexture, 7);
                    planarConstants.Options.X = static_cast<std::uint32_t>(pass);
                    Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &planarConstants, sizeof(planarConstants), 0);
                }
                else if (latitudeLongitude)
                {
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->EnvironmentMapTexture, 7);
                    mapConstants.Options.X = static_cast<std::uint32_t>(pass);
                    Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &mapConstants, sizeof(mapConstants), 0);
                }
                else if (cube)
                {
                    Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->EnvironmentTexture, 7);
                    cubeConstants.Options.X = static_cast<std::uint32_t>(pass);
                    Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &cubeConstants, sizeof(cubeConstants), 0);
                }
                else
                {
                    constants.Options.X = static_cast<std::uint32_t>(pass);
                    Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
                }
                Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
                Send<void>(encoder, "endEncoding");
                m_Impl->PendingFrame.ReflectionPasses.push_back(pass);
            }
        }
        m_Impl->ReflectionResolved = true;
        m_Impl->PendingFrame.Statistics.ReflectionPassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.ReflectionPasses.size());
        return true;
    }

    bool MetalRHI::ApplyLightShafts(const LightShaftDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyLightShafts")) return false;
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->LightShaftActive || m_Impl->LightShaftResolved ||
            (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
            (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) || (m_Impl->FogActive && !m_Impl->FogResolved) ||
            (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
            return m_Impl->Fail("Metal light shafts require resolved opaque lighting, AO, reflections and fog before temporal effects and transparency");
        LightShaftConstants constants {};
        if (!BuildLightShaftConstants(desc, m_Impl->FrameLightShafts, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
            return m_Impl->Fail("Metal light shafts require a finite camera and selected light");
        const MTLViewport viewport {0.0, 0.0, double(m_Impl->CreateInfo.Width), double(m_Impl->CreateInfo.Height), 0.0, 1.0};
        for (std::uint32_t phase = 0; phase < LightShaftPhaseCount; ++phase)
        {
            const bool mask = phase == 0;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(attachment, "setTexture:", mask ? m_Impl->LightShaftMask : m_Impl->HdrTextures[1]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create light-shaft encoder");
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", m_Impl->LightShaftPipeline);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            const std::array<ObjCObject, 3> inputs = mask ?
                std::array<ObjCObject, 3>{m_Impl->DummyColorTexture, m_Impl->DepthTexture, m_Impl->DummyColorTexture} :
                std::array<ObjCObject, 3>{m_Impl->HdrTextures[0], m_Impl->DepthTexture, m_Impl->LightShaftMask};
            // Every shader input is valid and distinct from the color attachment.
            // Depth is sampled only; it is not attached to either encoder.
            for (NSUInteger index = 0; index < inputs.size(); ++index)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[index], index);
            constants.Control.Y = phase;
            // setFragmentBytes copies this small block, preserving each draw's
            // phase even though the stack variable changes on the next iteration.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
        }
        // Tracked private resources on one command queue order mask writes,
        // sampled reads and the final HDR copy at encoder boundaries.
        // https://developer.apple.com/documentation/metal/resource-synchronization
        ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
        if (!blit) return m_Impl->Fail("Metal failed to create light-shaft color copy encoder");
        const MTLOrigin origin {0, 0, 0};
        const MTLSize size {static_cast<NSUInteger>(m_Impl->CreateInfo.Width), static_cast<NSUInteger>(m_Impl->CreateInfo.Height), 1};
        Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject, NSUInteger, NSUInteger, MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            m_Impl->HdrTextures[1], 0, 0, origin, size, m_Impl->HdrTextures[0], 0, 0, origin);
        Send<void>(blit, "endEncoding");
        m_Impl->LightShaftResolved = true;
        m_Impl->PendingFrame.Statistics.LightShaftPassCount += LightShaftPhaseCount;
        return true;
    }

    bool MetalRHI::ApplyVolumetricFog(const VolumetricFogDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ApplyVolumetricFog")) return false;
        if (m_Impl->LightShaftResolved) return m_Impl->Fail("Metal fog must precede light shafts");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->FogActive || m_Impl->FogResolved ||
            (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
            (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
            (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
            (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
            return m_Impl->Fail("Metal fog requires resolved opaque lighting, AO and reflections, once before temporal effects and transparency");
        if (!BuildVolumetricFogConstants(desc, m_Impl->FrameFog, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
                MaximumNativeTextureDimension, m_Impl->FogConstants)) return m_Impl->Fail("Metal fog requires a finite invertible frustum");
        const bool lighting = (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog));
        if (lighting && !m_Impl->GenerateVolumeLighting(desc)) return false;
        for (std::uint32_t phase = lighting ? 1U : 0U; phase < 2; ++phase)
        {
            const bool integrate = phase == 0;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(attachment, "setTexture:", integrate ? m_Impl->FogAtlas : m_Impl->HdrTextures[1]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create volumetric fog encoder");
            const MTLViewport viewport {0.0, 0.0,
                static_cast<double>(integrate ? m_Impl->FogDimensions.AtlasWidth() : m_Impl->CreateInfo.Width),
                static_cast<double>(integrate ? m_Impl->FogDimensions.Height : m_Impl->CreateInfo.Height), 0.0, 1.0};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:",
                Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->CloudFogPipeline : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->MediaFogPipeline : m_Impl->FogPipeline);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            // Even the integration branch has valid, non-aliasing bindings.
            // Neither encoder attaches the depth texture that composition reads.
            const std::array<ObjCObject, 3> inputs = integrate ?
                std::array<ObjCObject, 3>{m_Impl->DummyColorTexture, m_Impl->DummyShadowTexture, m_Impl->DummyColorTexture} :
                std::array<ObjCObject, 3>{m_Impl->HdrTextures[0], m_Impl->DepthTexture, m_Impl->FogAtlas};
            for (NSUInteger index = 0; index < inputs.size(); ++index)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[index], index);
            auto constants = m_Impl->FogConstants; constants.Options.X = phase;
            // Metal copies this 128-byte block into command-buffer-owned storage.
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
        }
        // Device-created private textures use default tracked hazards on this
        // MTLCommandQueue. Encoder boundaries order atlas writes, reads and blit.
        // https://developer.apple.com/documentation/metal/resource-synchronization
        ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
        if (!blit) return m_Impl->Fail("Metal failed to create fog color copy encoder");
        const MTLOrigin origin {0, 0, 0};
        const MTLSize size {static_cast<NSUInteger>(m_Impl->CreateInfo.Width), static_cast<NSUInteger>(m_Impl->CreateInfo.Height), 1};
        Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject, NSUInteger, NSUInteger, MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            m_Impl->HdrTextures[1], 0, 0, origin, size, m_Impl->HdrTextures[0], 0, 0, origin);
        Send<void>(blit, "endEncoding");
        m_Impl->FogResolved = true;
        (Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricFogPassCount) += lighting ? 1 : 2;
        return true;
    }

    bool MetalRHI::ApplyDepthOfField(const DepthOfFieldDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
        if (!m_Impl->CheckThread("ApplyDepthOfField")) return false;
        if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Metal light shafts must precede depth of field");
        if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Metal fog must precede depth of field");
        if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
            return m_Impl->Fail("Metal ApplyDepthOfField requires resolved multisample inputs");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->DepthOfFieldActive ||
            m_Impl->DepthOfFieldResolved || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
            (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
            (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved) ||
            (m_Impl->TemporalActive && !m_Impl->TemporalResolved))
            return m_Impl->Fail("Metal depth of field requires completed opaque lighting and AO once per frame");
        DepthOfFieldConstants constants{};
        if (!BuildDepthOfFieldConstants(desc, m_Impl->FramePostProcessSettings.DepthOfField,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Techniques::DepthOfFieldPassKind::Gather, constants))
            return m_Impl->Fail("Metal depth of field requires a finite conventional perspective or affine camera");
        const auto plan = Techniques::BuildDepthOfFieldPlan(m_Impl->FramePostProcessSettings.DepthOfField);
        const MTLViewport viewport{0.0, 0.0, static_cast<double>(m_Impl->CreateInfo.Width),
            static_cast<double>(m_Impl->CreateInfo.Height), 0.0, 1.0};
        for (const auto pass : plan)
        {
            const bool composite = pass == Techniques::DepthOfFieldPassKind::Composite;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(attachment, "setTexture:", m_Impl->HdrTextures[composite ? 0 : 1]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", composite ? LoadActionLoad : LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            // D32 is sampled by the shader, so neither encoder attaches depth.
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create depth-of-field encoder");
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", m_Impl->DepthOfFieldPipelines[composite ? 1 : 0]);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->HdrTextures[composite ? 1 : 0], 0);
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->DepthTexture, 1);
            constants.Options.X = static_cast<std::uint32_t>(pass);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
            m_Impl->PendingFrame.DepthOfFieldPasses.push_back(pass);
        }
        m_Impl->DepthOfFieldResolved = true;
        m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount = static_cast<std::uint32_t>(plan.size());
        return true;
    }

    bool MetalRHI::ApplyTemporalEffects(const TemporalEffectsDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyTemporalEffects") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FramePostProcessSettings.TemporalEffects.Enabled || !m_Impl->HdrActive || m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Temporal exposure requires completed scene HDR, once before presentation");
    if (!m_Impl->OpaqueEffectsComplete() || !m_Impl->Hair.Complete() || !m_Impl->Hair.VelocityComplete() ||
        !(m_Impl->PathFrame.Active?m_Impl->PathFrame.Resolved:m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardColorWritten) ||
        (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("Complete opaque effects and final multisample color before exposure metering");
    return m_Impl->ResolveExposure(desc);
}

bool MetalRHI::ApplyTemporalAA(const TemporalAADesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
        if (!m_Impl->CheckThread("ApplyTemporalAA")) return false;
        if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Metal light shafts must precede temporal reconstruction");
        if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Metal fog must precede temporal reconstruction");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->TemporalActive ||
            !m_Impl->VelocityCaptured || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
            (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved))
            return m_Impl->Fail("Metal TAA requires completed opaque velocity once before blur/transparency");
        TemporalAAConstants constants {};
        if (!BuildTemporalAAConstants(desc, m_Impl->FramePostProcessSettings.TemporalAA, m_Impl->CreateInfo.Width,
                m_Impl->CreateInfo.Height, m_Impl->TemporalHistoryValid, constants))
            return m_Impl->Fail("Metal TAA requires a finite previous camera and jitter");
        const auto read = m_Impl->TemporalHistoryReadIndex;
        const auto write = 1 - read;
        ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
        ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
        const std::array<ObjCObject, 2> outputs {m_Impl->TemporalColors[write], m_Impl->TemporalDepths[write]};
        for (NSUInteger index = 0; index < outputs.size(); ++index)
        {
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
            Send<void, ObjCObject>(attachment, "setTexture:", outputs[index]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
        }
        ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
        if (!encoder) return m_Impl->Fail("Metal failed to create temporal reconstruction encoder");
        const MTLViewport viewport {0.0, 0.0, static_cast<double>(m_Impl->CreateInfo.Width),
            static_cast<double>(m_Impl->CreateInfo.Height), 0.0, 1.0};
        Send<void, MTLViewport>(encoder, "setViewport:", viewport);
        Send<void, ObjCObject>(encoder, "setRenderPipelineState:", m_Impl->TemporalPipeline);
        Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
        Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
        const std::array<ObjCObject, 5> inputs {m_Impl->HdrTextures[0], m_Impl->DepthTexture,
            m_Impl->MotionTextures[Impl::MotionVelocity], m_Impl->TemporalColors[read], m_Impl->TemporalDepths[read]};
        for (NSUInteger index = 0; index < inputs.size(); ++index)
            Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[index], index);
        Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
        Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
        Send<void>(encoder, "endEncoding");
        // Default tracked private textures order the render, blit and later
        // encoders on this command buffer. Keep the pending pair immutable by
        // copying radiance into the scene before any later blur/transparency.
        ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
        if (!blit) return m_Impl->Fail("Metal failed to create temporal color copy encoder");
        const MTLOrigin origin {0, 0, 0};
        const MTLSize size {static_cast<NSUInteger>(m_Impl->CreateInfo.Width), static_cast<NSUInteger>(m_Impl->CreateInfo.Height), 1};
        Send<void, ObjCObject, NSUInteger, NSUInteger, MTLOrigin, MTLSize, ObjCObject, NSUInteger, NSUInteger, MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            m_Impl->TemporalColors[write], 0, 0, origin, size, m_Impl->HdrTextures[0], 0, 0, origin);
        Send<void>(blit, "endEncoding");
        m_Impl->TemporalResolved = true;
        m_Impl->PendingFrame.Temporal = desc.Sampling;
        ++m_Impl->PendingFrame.Statistics.TemporalAAPassCount;
        return true;
    }

    bool MetalRHI::ApplyMotionBlur(const MotionBlurDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
        if (!m_Impl->CheckThread("ApplyMotionBlur")) return false;
        if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Metal light shafts must precede motion blur");
        if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Metal fog must precede motion blur");
        if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved)
            return m_Impl->Fail("Metal ApplyMotionBlur requires resolved multisample inputs");
        if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->MotionBlurActive ||
            !m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
            (m_Impl->TemporalActive && !m_Impl->TemporalResolved) ||
            (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved))
            return m_Impl->Fail("Metal motion blur requires completed velocity, once before lens blur/transparency");
        MotionBlurConstants constants {};
        if (!BuildMotionBlurConstants(desc, m_Impl->FramePostProcessSettings.MotionBlur,
                m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
            return m_Impl->Fail("Metal motion blur requires a finite camera and valid sample interval");
        using Pass = Techniques::MotionBlurPassKind;
        const auto plan = Techniques::BuildMotionBlurPlan(m_Impl->FramePostProcessSettings.MotionBlur);
        const int tileSize = Techniques::MotionBlurTileSize;
        for (const auto pass : plan)
        {
            const bool composite = pass == Pass::Composite;
            const bool tiles = pass == Pass::TileMax || pass == Pass::NeighborMax;
            ObjCObject destination = tiles ? m_Impl->MotionTextures[pass == Pass::TileMax ? Impl::MotionTileMax : Impl::MotionNeighborMax] :
                m_Impl->HdrTextures[composite ? 0 : 1];
            const int width = tiles ? (m_Impl->CreateInfo.Width + tileSize - 1) / tileSize : m_Impl->CreateInfo.Width;
            const int height = tiles ? (m_Impl->CreateInfo.Height + tileSize - 1) / tileSize : m_Impl->CreateInfo.Height;
            ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
            ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", 0);
            Send<void, ObjCObject>(attachment, "setTexture:", destination);
            Send<void, NSUInteger>(attachment, "setLoadAction:", composite ? LoadActionLoad : LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            // Each encoder reads textures produced by a completed preceding
            // encoder on this command buffer. Default tracked resources provide
            // ordering; disjoint attachments avoid texture feedback. No depth
            // attachment is used while the reconstruction samples opaque D32.
            ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
            if (!encoder) return m_Impl->Fail("Metal failed to create motion reconstruction encoder");
            const MTLViewport viewport {0.0, 0.0, static_cast<double>(width), static_cast<double>(height), 0.0, 1.0};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:",
                m_Impl->MotionPipelines[composite ? Impl::MotionCompositePipeline : Impl::MotionFilterPipeline]);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            const std::array<ObjCObject, 4> inputs {m_Impl->HdrTextures[composite ? 1 : 0], m_Impl->DepthTexture,
                m_Impl->MotionTextures[Impl::MotionVelocity],
                m_Impl->MotionTextures[pass == Pass::NeighborMax ? Impl::MotionTileMax :
                    (pass == Pass::Gather ? Impl::MotionNeighborMax : Impl::MotionVelocity)]};
            for (NSUInteger unit = 0; unit < inputs.size(); ++unit)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", inputs[unit], unit);
            constants.Options.X = static_cast<std::uint32_t>(pass);
            Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:", PrimitiveTypeTriangle, 0, 3);
            Send<void>(encoder, "endEncoding");
            m_Impl->PendingFrame.MotionBlurPasses.push_back(pass);
        }
        m_Impl->MotionBlurResolved = true;
        m_Impl->PendingFrame.Statistics.MotionBlurPassCount = static_cast<std::uint32_t>(plan.size());
        return true;
    }

    bool MetalRHI::ProcessDepthPeeling(DepthPeelingPass pass)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessDepthPeeling") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePeeling.Enabled ||
            !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
            return m_Impl->Fail("OIT requires completed opaque HDR processing outside a raster pass");
        if (!m_Impl->ProcessPeeling(pass)) return false;
        RecordDepthPeelingPass(m_Impl->FramePeeling,pass,m_Impl->PendingFrame.Statistics); return true;
    }

    bool MetalRHI::ProcessRefraction(const RefractionDesc& desc)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        if (!m_Impl->CheckThread("ProcessRefraction") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->RefractionActive || m_Impl->RefractionResolved || m_Impl->FrameSampleCount!=1 ||
            !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
            return m_Impl->Fail("Refraction requires completed single-sample opaque HDR processing and executes once per frame");
        return m_Impl->ResolveRefraction(desc);
    }

    bool MetalRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass)
    {
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
        using Pass = Techniques::WeightedTransparencyPassKind;
        if (!m_Impl->CheckThread("ProcessWeightedTransparency") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
            !m_Impl->WeightedActive || !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
            (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved ||
                (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored))))
            return m_Impl->Fail("Metal weighted transparency requires completed opaque processing and restored sample color");
        if (pass == Pass::Prepare)
        {
            if (m_Impl->WeightedPrepared || m_Impl->SamplesTransparentWritten)
                return m_Impl->Fail("Metal prepare weighted statistics once before geometry");
        }
        else if (pass == Pass::Resolve)
        {
            if (!m_Impl->WeightedPrepared || !m_Impl->WeightedAccumulated || m_Impl->WeightedResolved)
                return m_Impl->Fail("Metal resolve weighted statistics once after geometry");
        }
        else return m_Impl->Fail("Unknown Metal weighted transparency stage");

        // Ordinary tracked textures and separate encoders on the same command
        // buffer order attachment writes before shader reads. No heap aliases or
        // untracked resources are used by these statistics.
        ObjCObject descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
        ObjCObject attachments = Send<ObjCObject>(descriptor, "colorAttachments");
        const bool prepare = pass == Pass::Prepare;
        const std::size_t attachmentCount = prepare ? m_Impl->WeightedTextures.size() : 1;
        ObjCObject scene = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Colors[Impl::SampleColor] : m_Impl->HdrTextures[0];
        for (std::size_t index = 0; index < attachmentCount; ++index)
        {
            ObjCObject attachment = Send<ObjCObject, NSUInteger>(attachments, "objectAtIndexedSubscript:", index);
            Send<void, ObjCObject>(attachment, "setTexture:", prepare ? m_Impl->WeightedTextures[index] : scene);
            Send<void, NSUInteger>(attachment, "setLoadAction:", prepare ? LoadActionClear : LoadActionLoad);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            Send<void, MTLClearColor>(attachment, "setClearColor:", index == 0 ? MTLClearColor{} : MTLClearColor{1, 1, 1, 1});
        }
        ObjCObject encoder = Send<ObjCObject, ObjCObject>(m_Impl->CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
        if (!encoder) return m_Impl->Fail("Metal failed to encode weighted transparency");
        if (!prepare)
        {
            const MTLViewport viewport {0, 0, static_cast<double>(m_Impl->CreateInfo.Width),
                static_cast<double>(m_Impl->CreateInfo.Height), 0, 1};
            Send<void, MTLViewport>(encoder, "setViewport:", viewport);
            Send<void, ObjCObject>(encoder, "setRenderPipelineState:", m_Impl->WeightedGroups.at(m_Impl->FrameSampleCount).Resolve);
            Send<void, NSUInteger>(encoder, "setCullMode:", CullModeNone);
            Send<void, NSUInteger>(encoder, "setTriangleFillMode:", TriangleFillModeFill);
            for (NSUInteger index = 0; index < m_Impl->WeightedTextures.size(); ++index)
                Send<void, ObjCObject, NSUInteger>(encoder, "setFragmentTexture:atIndex:", m_Impl->WeightedTextures[index], index);
            constexpr NSUInteger FullscreenVertexCount = 3;
            Send<void, NSUInteger, NSUInteger, NSUInteger>(encoder, "drawPrimitives:vertexStart:vertexCount:",
                PrimitiveTypeTriangle, 0, FullscreenVertexCount);
        }
        Send<void>(encoder, "endEncoding");
        if (prepare) m_Impl->WeightedPrepared = true;
        else m_Impl->WeightedResolved = true;
        ++m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount;
        return true;
    }

    bool MetalRHI::ProcessMultisampling(Techniques::MultisamplePassKind pass)
    {
    if (pass==Techniques::MultisamplePassKind::OpaqueResolve && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before opaque MSAA resolution");
        using Pass = Techniques::MultisamplePassKind;
        using Phase = Impl::SamplePhase;
        if (!m_Impl->CheckThread("ProcessMultisampling") || !m_Impl->FrameOpen || m_Impl->PassOpen || m_Impl->FrameSampleCount <= 1)
            return m_Impl->Fail("Metal sample processing requires a multisample frame outside a raster pass");
        const bool material = m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive;
        switch (pass)
        {
        case Pass::OpaqueResolve:
            if (m_Impl->SamplesOpaqueResolved || (material && !m_Impl->GBufferWritten) ||
                !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
                return m_Impl->Fail("Metal resolve opaque samples once after opaque lighting");
            if (!m_Impl->DrawSampleResolve(Phase::Color) || !m_Impl->DrawSampleResolve(material ? Phase::Material : Phase::Depth)) return false;
            m_Impl->SamplesOpaqueResolved = m_Impl->SamplesColorResolved = true;
            break;
        case Pass::VelocityResolve:
            if (!m_Impl->SamplesOpaqueResolved || !m_Impl->VelocityCaptured || m_Impl->SamplesVelocityResolved ||
                m_Impl->TemporalResolved || m_Impl->MotionBlurResolved)
                return m_Impl->Fail("Metal resolve sample velocity once before motion blur");
            if (!m_Impl->DrawSampleResolve(Phase::Velocity)) return false;
            m_Impl->SamplesVelocityResolved = true;
            break;
        case Pass::ColorRestore:
            if (!m_Impl->SamplesOpaqueResolved || !m_Impl->NeedsSampleColorRestore() || !m_Impl->OpaqueEffectsComplete() ||
                m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)
                return m_Impl->Fail("Metal restore processed opaque radiance once before sample transparency");
            if (!m_Impl->DrawSampleResolve(Phase::Restore)) return false;
            m_Impl->SamplesColorRestored = true;
            break;
        case Pass::ColorResolve:
            if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
                return m_Impl->Fail("Metal composite weighted transparency before the final sample average");
            if (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesTransparentWritten || m_Impl->SamplesColorResolved)
                return m_Impl->Fail("Metal final sample resolve requires transparent composition");
            if (!m_Impl->DrawSampleResolve(Phase::Color)) return false;
            m_Impl->SamplesColorResolved = true;
            break;
        default: return m_Impl->Fail("Unknown Metal sample resolve phase");
        }
        ++m_Impl->PendingFrame.Statistics.MultisamplePassCount;
        return true;
    }

	bool MetalRHI::Present()
	{
    if (!m_Impl->CheckThread("Lumen frame publication")) return false;
    Impl::LumenPresentPublication lumenPublication{*m_Impl};
    Impl::NanitePresentPublication nanitePublication{*m_Impl};
    if (!m_Impl->CheckThread("TSR present boundary")) return false;
    if (m_Impl->FrameOpen && m_Impl->TSRScene.Active && (!m_Impl->TSRScene.OpaqueReady || !m_Impl->VelocityCaptured || (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesVelocityResolved))) return m_Impl->Fail("TSR presentation requires completed opaque snapshot and physical motion");

    if (m_Impl->FrameOpen && m_Impl->FramePostProcessSettings.TemporalEffects.Enabled && !m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Complete temporal exposure before presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->FrameOpen && m_Impl->PendingFrame.Statistics.SubsurfacePassCount && !m_Impl->SubsurfaceResolved)
        return m_Impl->Fail("Incomplete subsurface diffusion must be aborted before presentation");
        if (m_Impl->PathFrame.Active && !m_Impl->PathFrame.Resolved) return m_Impl->Fail("Path tracing must resolve before presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
		if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("Metal frame cannot be presented in the current state");
        if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("Metal Forward+ frame requires completed light assignment before presentation");
        if (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesColorResolved))
            return m_Impl->Fail("Metal multisample color/depth must resolve before presentation");
        if (m_Impl->DeferredActive && !m_Impl->DeferredResolved)
            return m_Impl->Fail("Metal deferred frame must resolve its G-buffer before presentation");
        if (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved)
            return m_Impl->Fail("Metal ambient occlusion must complete before presentation");
        if (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)
            return m_Impl->Fail("Metal reflections must complete before presentation");
        if (m_Impl->DepthOfFieldActive && !m_Impl->DepthOfFieldResolved)
            return m_Impl->Fail("Metal depth of field must complete before presentation");
        if (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved)
            return m_Impl->Fail("Metal motion blur must complete before presentation");
        if (m_Impl->TemporalActive && !m_Impl->TemporalResolved)
            return m_Impl->Fail("Metal temporal reconstruction must complete before presentation");
        if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
            return m_Impl->Fail("Metal light shafts must resolve before presentation");
        if (m_Impl->FogActive && !m_Impl->FogResolved)
            return m_Impl->Fail("Metal fog must complete before presentation");
        if (m_Impl->FramePeeling.Enabled && m_Impl->PeelingSequence.Stage()!=DepthPeelingStage::Complete)
            return m_Impl->Fail("Present requires completed OIT composition");
        if (m_Impl->RefractionActive && !m_Impl->RefractionResolved)
            return m_Impl->Fail("Present requires completed refraction composition");
        if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
            return m_Impl->Fail("Metal weighted transparency must complete before presentation");
		m_Impl->PendingFrame.Statistics.PassCount =
            static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size()) + m_Impl->PendingFrame.Statistics.HairFurPassCount + m_Impl->PendingFrame.Statistics.SubsurfacePassCount + m_Impl->PendingFrame.Statistics.PathTracingPassCount + m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount + m_Impl->PendingFrame.Statistics.ReflectionPassCount +
            m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount +
            m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount + m_Impl->PendingFrame.Statistics.MotionBlurPassCount +
            m_Impl->PendingFrame.Statistics.TemporalAAPassCount + m_Impl->PendingFrame.Statistics.TemporalEffectsPassCount + m_Impl->PendingFrame.Statistics.VolumetricFogPassCount +
            m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount + m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount + m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount + m_Impl->PendingFrame.Statistics.LightShaftPassCount +
            m_Impl->PendingFrame.Statistics.MultisamplePassCount + m_Impl->PendingFrame.Statistics.EnvironmentConversionPassCount +
            m_Impl->PendingFrame.Statistics.PlanarMipPassCount + m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount + m_Impl->PendingFrame.Statistics.DepthPeelingPassCount +
            m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount + m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount
            + m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount + m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount;
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) ||
			(!IsGpuPresentationEnabled() && m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0);

		ObjCObject colorReadback = nullptr;
		ObjCObject depthReadback = nullptr;
		ObjCObject shadowReadback = nullptr;
        ObjCObject momentReadback = nullptr;
        std::size_t momentPitch = 0;
        ObjCObject motionReadback = nullptr;
        std::array<ObjCObject, MaximumShadowCascades> cascadeReadbacks {};
        std::array<std::size_t, MaximumShadowCascades> cascadePitches {};
		std::size_t colorPitch = 0;
		std::size_t depthPitch = 0;
		std::size_t shadowPitch = 0;
        std::size_t motionPitch = 0;
        ObjCObject postColor = m_Impl->ColorTexture;
        if (!m_Impl->TSRScene.Active && (!m_Impl->ApplyHdr(m_Impl->FramePostProcessSettings.HDR) || !m_Impl->ApplyPostProcess(m_Impl->FramePostProcessSettings, postColor))) return false;
		ObjCObject blit = Send<ObjCObject>(m_Impl->CommandBuffer, "blitCommandEncoder");
		if (blit == nullptr) return m_Impl->Fail("Metal failed to create a readback command encoder");
        ObjCObject diagnosticReadback=nullptr;std::size_t diagnosticPitch=0;
        const int diagnosticIndex=Techniques::MaterialBufferIndex(GetDiagnosticBuffer());
        ObjCObject diagnostic=diagnosticIndex>=0?m_Impl->GBufferTextures[diagnosticIndex]:GetDiagnosticBuffer()==Techniques::RenderBuffer::AmbientOcclusion?m_Impl->AmbientOcclusionTextures[0]:nullptr;
        if (diagnostic && !m_Impl->EnqueueReadback(blit,diagnostic,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,8,diagnosticReadback,diagnosticPitch)) return false;
        const bool gpuPresentation=IsGpuPresentationEnabled() && !captureDepth && !m_Impl->TSRScene.Active && GetDiagnosticBuffer()==Techniques::RenderBuffer::Color;
        bool enqueued = gpuPresentation ? m_Impl->PublishGpuColor(blit,postColor) : m_Impl->TSRScene.Active || m_Impl->EnqueueReadback(blit, postColor,m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, 4, colorReadback, colorPitch);
        if (enqueued && m_Impl->VelocityActive && (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
            enqueued = m_Impl->EnqueueReadback(blit, m_Impl->MotionTextures[Impl::MotionVelocity],
                m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, 4 * sizeof(std::uint16_t), motionReadback, motionPitch);
		if (enqueued && captureDepth)
		{
			enqueued = m_Impl->EnqueueReadback(blit, m_Impl->DepthTexture,
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				sizeof(float), depthReadback, depthPitch);
		}
		Impl::TextureResource* shadow = nullptr;
		if (enqueued && (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) &&
			m_Impl->LastShadowTexture)
		{
			auto iterator = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (iterator != m_Impl->Textures.end())
			{
				shadow = &iterator->second;
				enqueued = m_Impl->EnqueueReadback(blit, shadow->Texture,
					shadow->Description.Width, shadow->Description.Height,
					sizeof(float), shadowReadback, shadowPitch);
                if (enqueued && m_Impl->UsesVSM())
                {
                    ObjCObject moments = nullptr;
                    enqueued = m_Impl->MomentImage(m_Impl->LastShadowTexture, moments) &&
                        m_Impl->EnqueueReadback(blit, moments, shadow->Description.Width, shadow->Description.Height,
                            sizeof(Float2), momentReadback, momentPitch);
                }
			}
		}
        if (enqueued && (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
            for (std::uint32_t index = 0; enqueued && index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
                enqueued = m_Impl->EnqueueReadback(blit, m_Impl->CascadeTextures[index],
                    m_Impl->FrameCascades.MapResolution, m_Impl->FrameCascades.MapResolution,
                    sizeof(float), cascadeReadbacks[index], cascadePitches[index]);
		Send<void>(blit, "endEncoding");
		if (!enqueued)
		{
            ReleaseObject(momentReadback);
            for (auto buffer : cascadeReadbacks) ReleaseObject(buffer);
			ReleaseObject(motionReadback);
			ReleaseObject(colorReadback);
			ReleaseObject(depthReadback);
			ReleaseObject(shadowReadback);
			return false;
		}
		Send<void>(m_Impl->CommandBuffer, "commit");
		Send<void>(m_Impl->CommandBuffer, "waitUntilCompleted");
		if (!m_Impl->CommandSucceeded(m_Impl->CommandBuffer, "Metal frame execution failed"))
		{
            ReleaseObject(momentReadback);
            for (auto buffer : cascadeReadbacks) ReleaseObject(buffer);
			ReleaseObject(motionReadback);
			ReleaseObject(colorReadback);
			ReleaseObject(depthReadback);
			ReleaseObject(shadowReadback);
			return false;
		}
        if (!gpuPresentation && !m_Impl->TSRScene.Active)
        { std::vector<std::uint32_t> colorPixels; m_Impl->CopyReadback(colorReadback,colorPitch,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,colorPixels); m_Impl->PendingFrame.ColorRGBA8.resize(colorPixels.size()*sizeof(std::uint32_t)); std::memcpy(m_Impl->PendingFrame.ColorRGBA8.data(),colorPixels.data(),m_Impl->PendingFrame.ColorRGBA8.size()); }
        if (diagnosticReadback)
        {
            std::vector<std::uint64_t> pixels;m_Impl->CopyReadback(diagnosticReadback,diagnosticPitch,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,pixels);
            DecodeDiagnosticRows(pixels.data(),std::size_t(m_Impl->CreateInfo.Width)*8,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->PendingFrame.DiagnosticPixels);ReleaseObject(diagnosticReadback);
        }
        if (motionReadback) m_Impl->ReadMotionVectors(motionReadback, motionPitch);
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
        if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
        {
            m_Impl->PendingFrame.CascadeDepth32.resize(m_Impl->FrameCascades.Frame.CascadeCount);
            for (std::size_t index = 0; index < m_Impl->PendingFrame.CascadeDepth32.size(); ++index)
                m_Impl->CopyReadback(cascadeReadbacks[index], cascadePitches[index],
                    m_Impl->FrameCascades.MapResolution, m_Impl->FrameCascades.MapResolution,
                    m_Impl->PendingFrame.CascadeDepth32[index]);
        }
        if (momentReadback && shadow)
            m_Impl->CopyReadback(momentReadback, momentPitch, shadow->Description.Width,
                shadow->Description.Height, m_Impl->PendingFrame.ShadowMoments);
        ReleaseObject(momentReadback);
        for (auto buffer : cascadeReadbacks) ReleaseObject(buffer);
		ReleaseObject(colorReadback);
		ReleaseObject(depthReadback);
		ReleaseObject(shadowReadback);
        ReleaseObject(motionReadback);
        m_Impl->ReadFrameTiming();
		ReleaseObject(m_Impl->CommandBuffer);
		m_Impl->CommandBuffer = nullptr;
		if (m_Impl->FramePool != nullptr)
		{
			Send<void>(m_Impl->FramePool, "drain");
			m_Impl->FramePool = nullptr;
		}
		if (!m_Impl->FinishTSRScene()) return false;
    m_Impl->CommitTSRScene();
    m_Impl->Hair.CommitPresent(m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height);
    m_Impl->ExposureHistory.Commit();
    ApplyDiagnosticCapture(m_Impl->PendingFrame,GetDiagnosticBuffer());
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameOpen = false;
        // Encoders may have produced a pending pair, but only successful GPU
        // completion and readback can publish it. Abort never changes the index.
        if (m_Impl->TemporalResolved)
        {
            m_Impl->TemporalHistoryReadIndex = 1 - m_Impl->TemporalHistoryReadIndex;
            m_Impl->TemporalHistoryValid = true;
        }
        else m_Impl->TemporalHistoryValid = false;
        m_Impl->PathFrame.Commit();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI)) m_Impl->GI.Commit(); else m_Impl->GI.Invalidate();
        if (m_Impl->ProbeActive) m_Impl->Probes.Commit();
		lumenPublication.Published=true; nanitePublication.Published=true;
    return true;
	}

	FrameCapture MetalRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
	FrameCapture MetalRHI::TakeLastFrameCapture() { return std::move(m_Impl->LastFrame); }
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
    bool MetalRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
bool MetalRHI::PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
bool MetalRHI::SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
bool MetalRHI::PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
bool MetalRHI::CancelNaniteCache(NaniteCacheTask) { return false; }
    bool MetalRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
bool MetalRHI::DestroyNaniteCache(NaniteCacheHandle) { return false; }
    bool MetalRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
    bool MetalRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
    bool MetalRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
    bool MetalRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
    bool MetalRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
    bool MetalRHI::CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
    bool MetalRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle) { return false; }
    bool MetalRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
    bool MetalRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
    bool MetalRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
    bool MetalRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
    bool MetalRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }
    bool MetalRHI::RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
    bool MetalRHI::BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
    bool MetalRHI::AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
    bool MetalRHI::CancelLumenLightingUpdate(LumenLightingTask) { return false; }
    bool MetalRHI::ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
    bool MetalRHI::DestroyLumenLighting(LumenLightingHandle) { return false; }
    bool MetalRHI::TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
    bool MetalRHI::ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
    bool MetalRHI::AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
    bool MetalRHI::ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
    bool MetalRHI::DestroyLumenRadiance(LumenRadianceHandle) { return false; }
    bool MetalRHI::IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
    bool MetalRHI::ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
    bool MetalRHI::ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
    bool MetalRHI::DestroyLumenProbes(LumenProbeHandle) { return false; }
    bool MetalRHI::LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
    bool MetalRHI::CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
    bool MetalRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
    bool MetalRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
    bool MetalRHI::CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
    bool MetalRHI::ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
    bool MetalRHI::DestroyLumenSurface(LumenSurfaceHandle) { return false; }
    bool MetalRHI::DestroyTSRHistory(TSRHistoryHandle) { return false; }
    bool MetalRHI::RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
    bool MetalRHI::SnapshotTSROpaque() { return false; }
    bool MetalRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool MetalRHI::CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
bool MetalRHI::SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
bool MetalRHI::PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
bool MetalRHI::CancelNaniteGeometry(NaniteGeometryTask) { return false; }
    bool MetalRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool MetalRHI::PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool MetalRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool MetalRHI::DrawNaniteBins(NaniteDrawHandle) { return false; }
bool MetalRHI::DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
bool MetalRHI::CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
bool MetalRHI::BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
bool MetalRHI::RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
bool MetalRHI::ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
bool MetalRHI::ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
bool MetalRHI::DestroyNaniteDepth(NaniteDepthHandle) { return false; }
bool MetalRHI::DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }
bool MetalRHI::CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
    bool MetalRHI::CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
    bool MetalRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
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
	bool MetalRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
    bool MetalRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
    bool MetalRHI::RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
    bool MetalRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
    bool MetalRHI::DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>&) { return false; }
    bool MetalRHI::DrawIndexedGeometry(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>*) { return false; }
	bool MetalRHI::EndPass() { return false; }
    bool MetalRHI::PrepareLightListFrame(const LightListFrameDesc&) { return false; }
    bool MetalRHI::GenerateLightLists() { return false; }
    bool MetalRHI::GenerateContactShadows(const ContactShadowDesc&) { return false; }
	bool MetalRHI::DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
    bool MetalRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
    bool MetalRHI::DrawHairFur(const HairFurDesc&) { return false; }
    bool MetalRHI::DrawHairFurInPass(const HairFurDesc&) { return false; }
    bool MetalRHI::DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
	bool MetalRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
    bool MetalRHI::CompleteEnvironmentCapture() { return false; }
    bool MetalRHI::CompletePlanarCapture() { return false; }
    bool MetalRHI::ApplyPathTracing(const PathTracingDesc&) { return false; }
    bool MetalRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
    bool MetalRHI::ApplyReflections(const ReflectionDesc&) { return false; }
    bool MetalRHI::ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
    bool MetalRHI::ApplyMotionBlur(const MotionBlurDesc&) { return false; }
    bool MetalRHI::ApplyTemporalAA(const TemporalAADesc&) { return false; }
    bool MetalRHI::ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
    bool MetalRHI::ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
    bool MetalRHI::ApplyLightShafts(const LightShaftDesc&) { return false; }
    bool MetalRHI::ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
    bool MetalRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
    bool MetalRHI::ProcessDepthPeeling(DepthPeelingPass) { return false; }
    bool MetalRHI::ProcessRefraction(const RefractionDesc&) { return false; }
	bool MetalRHI::Present() { return false; }
	FrameCapture MetalRHI::GetLastFrameCapture() const { return {}; }
	FrameCapture MetalRHI::TakeLastFrameCapture() { return {}; }
	std::string MetalRHI::GetLastError() const { return m_Impl->LastError; }
}

#endif
