#include "Renderer/Metal/MetalShaderCache.h"
#include "Renderer/Metal/MetalContactShadows.h"
#include "Renderer/TechniqueShader.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <thread>

namespace Evans::Renderer
{
    namespace
    {
        constexpr NSUInteger TileWidth = 8, TileHeight = 8;
        constexpr NSUInteger MaximumSamples = 8;
        constexpr float PixelCenter = 0.5f;
        static_assert(sizeof(MTLSamplePosition) == sizeof(Float2), "Metal raster sample ABI changed");
        std::string Diagnostic(NSError* error)
        {
            const char* text = error ? [[error localizedDescription] UTF8String] : nullptr;
            return text ? text : "no native diagnostic";
        }
        bool Fail(std::string& error, const char* message) { error = message; return false; }
    }
    struct MetalContactShadows::Impl
    {
        id<MTLDevice> Device = nil;
        id<MTLComputePipelineState> Single = nil, Multisample = nil;
        id<MTLTexture> Visibility = nil;
        id<MTLCommandBuffer> Recorded = nil;
        std::thread::id Owner;
        bool FrameOpen = false, Ready = false;

        bool CheckOwner(std::string& error) const
        {
            return (Device && Owner == std::this_thread::get_id()) ||
                Fail(error, "Metal contact visibility requires its initialized owning thread");
        }
        bool EnsureTarget(NSUInteger width, NSUInteger height, NSUInteger samples, std::string& error)
        {
            if (Visibility && Visibility.width == width && Visibility.height == height && Visibility.arrayLength == samples)
                return true;
            MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
            descriptor.textureType = MTLTextureType2DArray;
            descriptor.pixelFormat = MTLPixelFormatR32Float;
            descriptor.width = width; descriptor.height = height;
            descriptor.arrayLength = samples;
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> candidate = [Device newTextureWithDescriptor:descriptor];
            [descriptor release];
            if (!candidate) return Fail(error, "Metal contact visibility array allocation failed");
            candidate.label = @"Evans contact visibility, one layer per depth sample";
            [Visibility release]; Visibility = candidate;
            return true;
        }
    };
    MetalContactShadows::MetalContactShadows() : m_Impl(std::make_unique<Impl>()) {}
    MetalContactShadows::~MetalContactShadows() { Shutdown(); }
    bool MetalContactShadows::Initialize(void* device, std::string& error)
    {
        error.clear();
        if (m_Impl->Device) return Fail(error, "Metal contact visibility is already initialized");
        if (!device) return Fail(error, "Metal contact visibility requires a native device");
        @autoreleasepool
        {
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "ContactShadows.metal", source, error)) return false;
            auto native = (id<MTLDevice>)device;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // ContactVertex packs its Color/UV words for the host ABI, and MSL 2.0
            // rejects component access on packed vectors; every other technique
            // library in this renderer compiles at 2.1.
            options.languageVersion = MTLLanguageVersion2_1;
            // Apple's default enables algebraic reassociation and assumes finite
            // values. DDA corner bounds and compensated arithmetic require IEEE.
            options.fastMathEnabled = NO;
            NSError* diagnostic = nil;
            id<MTLLibrary> library = Evans::Renderer::CompileCachedMetalLibrary(native,[NSString stringWithUTF8String:source.c_str()],options,&diagnostic);
            [options release];
            if (!library) { error = "Metal contact shader compilation failed: " + Diagnostic(diagnostic); return false; }
            std::array<id<MTLComputePipelineState>, 2> pipelines {};
            const std::array<NSString*, 2> names {@"ContactVisibilitySingle", @"ContactVisibilityMultisample"};
            for (std::size_t index = 0; index < names.size(); ++index)
            {
                id<MTLFunction> function = [library newFunctionWithName:names[index]];
                if (function) pipelines[index] = Evans::Renderer::CreateCachedMetalComputePipeline(native,function,&diagnostic);
                [function release];
                if (!pipelines[index] || pipelines[index].maxTotalThreadsPerThreadgroup < TileWidth * TileHeight)
                {
                    error = "Metal contact compute pipeline is unavailable: " + Diagnostic(diagnostic);
                    for (auto pipeline : pipelines) [pipeline release];
                    [library release];
                    return false;
                }
            }
            [library release];
            m_Impl->Device = [native retain];
            m_Impl->Owner = std::this_thread::get_id();
            m_Impl->Single = pipelines[0]; m_Impl->Multisample = pipelines[1];
            if (!m_Impl->EnsureTarget(1, 1, 1, error)) { Shutdown(); return false; }
            return true;
        }
    }
    void MetalContactShadows::Shutdown()
    {
        // Destruction may also follow a failed submission. Never release an
        // allocation still used by committed work; unsubmitted work is discarded.
        if (m_Impl->Recorded && m_Impl->Recorded.status >= MTLCommandBufferStatusCommitted &&
            m_Impl->Recorded.status < MTLCommandBufferStatusCompleted) [m_Impl->Recorded waitUntilCompleted];
        [m_Impl->Recorded release]; [m_Impl->Visibility release];
        [m_Impl->Single release]; [m_Impl->Multisample release]; [m_Impl->Device release];
        *m_Impl = {};
    }
    bool MetalContactShadows::BeginFrame(std::string& error)
    {
        error.clear();
        if (!m_Impl->CheckOwner(error)) return false;
        if (m_Impl->Recorded && m_Impl->Recorded.status < MTLCommandBufferStatusCompleted)
            return Fail(error, "Metal contact visibility must complete or abort before target reuse");
        [m_Impl->Recorded release]; m_Impl->Recorded = nil;
        m_Impl->Ready = false; m_Impl->FrameOpen = true;
        return true;
    }
    bool MetalContactShadows::AbortFrame(std::string& error)
    {
        error.clear();
        if (!m_Impl->CheckOwner(error)) return false;
        if (m_Impl->Recorded && m_Impl->Recorded.status >= MTLCommandBufferStatusCommitted)
            return Fail(error, "Metal contact abort cannot discard submitted work");
        [m_Impl->Recorded release]; m_Impl->Recorded = nil;
        m_Impl->Ready = m_Impl->FrameOpen = false;
        return true;
    }
    bool MetalContactShadows::Record(void* commandBuffer, void* depthTexture,
        const ContactShadowDispatch& trace, std::string& error)
    {
        error.clear();
        if (!m_Impl->CheckOwner(error)) return false;
        m_Impl->Ready = false;
        if (!m_Impl->FrameOpen || m_Impl->Recorded)
            return Fail(error, "Metal contact visibility records once per begun frame");
        auto commands = (id<MTLCommandBuffer>)commandBuffer;
        auto depth = (id<MTLTexture>)depthTexture;
        if (!commands || !depth || commands.device != m_Impl->Device || depth.device != m_Impl->Device ||
            commands.status >= MTLCommandBufferStatusCommitted)
            return Fail(error, "Metal contact depth and unsubmitted commands must belong to the initialized device");
        const auto samples = depth.sampleCount;
        // Heap defaults differ from device-created resources. Inspect the resolved
        // mode where available; older heaps require synchronization we do not own.
        // https://developer.apple.com/documentation/metal/resource-synchronization
        bool untracked = (depth.resourceOptions & MTLResourceHazardTrackingModeUntracked) != 0;
        if (@available(macOS 10.15, iOS 13.0, *)) untracked = depth.hazardTrackingMode == MTLHazardTrackingModeUntracked;
        else untracked = untracked || depth.heap != nil;
        if (depth.pixelFormat != MTLPixelFormatDepth32Float || !(depth.usage & MTLTextureUsageShaderRead) ||
            untracked ||
            depth.storageMode == MTLStorageModeMemoryless ||
            depth.textureType != (samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D) ||
            depth.width > UINT32_MAX || depth.height > UINT32_MAX ||
            (samples != 1 && samples != 2 && samples != 4 && samples != MaximumSamples))
            return Fail(error, "Metal contact input must be tracked, stored, sampled D32 with 1/2/4/8 default raster samples");
        ContactShadowConstants constants;
        if (!BuildContactShadowConstants(trace, std::uint32_t(depth.width), std::uint32_t(depth.height),
                std::uint32_t(samples), constants, error)) return false;
        std::array<MTLSamplePosition, MaximumSamples> positions {};
        if (samples == 1) positions[0] = MTLSamplePositionMake(PixelCenter, PixelCenter);
        else
        {
            if (![m_Impl->Device respondsToSelector:@selector(getDefaultSamplePositions:count:)])
                return Fail(error, "Metal contact MSAA requires device-reported raster sample positions");
            [m_Impl->Device getDefaultSamplePositions:positions.data() count:samples];
        }
        for (NSUInteger sample = 0; sample < samples; ++sample)
            if (!std::isfinite(positions[sample].x) || !std::isfinite(positions[sample].y) ||
                positions[sample].x < 0 || positions[sample].x >= 1 || positions[sample].y < 0 || positions[sample].y >= 1)
                return Fail(error, "Metal returned invalid default sample positions");
        if (!m_Impl->EnsureTarget(depth.width, depth.height, samples, error)) return false;
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        if (!encoder) return Fail(error, "Metal contact compute encoder creation failed");
        encoder.label = @"Evans contact visibility";
        [encoder setComputePipelineState:samples > 1 ? m_Impl->Multisample : m_Impl->Single];
        [encoder setTexture:depth atIndex:0];
        [encoder setTexture:m_Impl->Visibility atIndex:1];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:0];
        [encoder setBytes:positions.data() length:sizeof(positions) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((depth.width + TileWidth - 1) / TileWidth,
            (depth.height + TileHeight - 1) / TileHeight, samples)
            threadsPerThreadgroup:MTLSizeMake(TileWidth, TileHeight, 1)];
        [encoder endEncoding];
        m_Impl->Recorded = [commands retain];
        m_Impl->Ready = true;
        return true;
    }
    void* MetalContactShadows::Result() const { return m_Impl->Ready ? (void*)m_Impl->Visibility : nullptr; }
}
