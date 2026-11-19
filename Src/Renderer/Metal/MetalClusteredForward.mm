#include "Renderer/Metal/MetalShaderCache.h"
#include "Renderer/Metal/MetalClusteredForward.h"
#include "Renderer/TechniqueShader.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <thread>

namespace Evans::Renderer
{
    namespace
    {
        constexpr NSUInteger ThreadsPerCluster = 64;
        constexpr NSUInteger LegacyBufferLimit = 256u * 1024u * 1024u;
        std::string Diagnostic(NSError* error)
        {
            const char* description = error ? error.localizedDescription.UTF8String : nullptr;
            return description ? description : "no native diagnostic";
        }
        bool Fail(std::string& error, const char* reason) { error = reason; return false; }
    }
    struct MetalClusteredForward::Impl
    {
        id<MTLDevice> Device = nil;
        std::array<id<MTLComputePipelineState>, 1> Pipelines {};
        id<MTLBuffer> Bounds = nil, Headers = nil, Indices = nil;
        id<MTLCommandBuffer> Commands = nil;
        std::thread::id Owner;
        NSUInteger BufferLimit = 0;
        ClusteredForwardGrid CurrentGrid;
        bool Open = false, Ready = false;

        bool CheckOwner(std::string& error) const
        { return (Device && Owner == std::this_thread::get_id()) || Fail(error, "Metal Clustered Forward requires its initialized owning thread"); }
        bool Completed(std::string& error) const
        {
            if (!Commands) return true;
            if (Commands.status == MTLCommandBufferStatusError)
            { error = "Metal Clustered Forward command buffer failed: " + Diagnostic(Commands.error); return false; }
            return Commands.status == MTLCommandBufferStatusCompleted || Fail(error, "Metal Clustered Forward commands must complete or be discarded before reuse");
        }
        bool Usable() const { return Ready && Commands && Commands.status != MTLCommandBufferStatusError; }
        void ReleaseFrame()
        {
            [Commands release]; Commands = nil;
            Ready = Open = false; CurrentGrid = {};
        }
        void Release()
        {
            ReleaseFrame(); [Bounds release]; [Headers release]; [Indices release];
            Bounds = Headers = Indices = nil;
            for (auto pipeline : Pipelines) [pipeline release]; Pipelines = {};
            [Device release]; Device = nil; Owner = {}; BufferLimit = 0;
        }
        bool EnsureLists(const ClusteredForwardGrid& grid, std::string& error)
        {
            const auto headerBytes = std::uint64_t(grid.ClusterCount + ClusteredBoundarySlots) * sizeof(ClusteredForwardHeader);
            const auto indexBytes = std::uint64_t(grid.IndexCount) * sizeof(std::uint32_t);
            if (headerBytes > BufferLimit || indexBytes > BufferLimit)
                return Fail(error, "Metal Clustered Forward lists exceed the device buffer-length limit");
            if (Headers && Headers.length == headerBytes && Indices && Indices.length == indexBytes) return true;
            // Device-created resources use tracked hazards; private outputs stay
            // on the GPU. Allocate both before retiring the previous allocation.
            auto headers = [Device newBufferWithLength:NSUInteger(headerBytes) options:MTLResourceStorageModePrivate];
            auto indices = [Device newBufferWithLength:NSUInteger(indexBytes) options:MTLResourceStorageModePrivate];
            if (!headers || !indices)
            { [headers release]; [indices release]; return Fail(error, "Metal Clustered Forward list allocation failed"); }
            headers.label = @"Evans Clustered Forward tile headers"; indices.label = @"Evans Clustered Forward light indices";
            [Headers release]; [Indices release]; Headers = headers; Indices = indices; return true;
        }
    };
    MetalClusteredForward::MetalClusteredForward() : m_Impl(std::make_unique<Impl>()) {}
    MetalClusteredForward::~MetalClusteredForward()
    {
        auto commands = m_Impl->Commands;
        if (commands && commands.status >= MTLCommandBufferStatusCommitted && commands.status < MTLCommandBufferStatusCompleted)
            [commands waitUntilCompleted];
        m_Impl->Release();
    }
    bool MetalClusteredForward::Initialize(void* device, std::string& error)
    {
        error.clear(); auto& state = *m_Impl;
        if (state.Device) return Fail(error, "Metal Clustered Forward is already initialized");
        if (!device) return Fail(error, "Metal Clustered Forward requires an owning native device");
        @autoreleasepool
        {
            auto native = (id<MTLDevice>)device;
            const auto groupLimit = native.maxThreadsPerThreadgroup;
            if (groupLimit.width < ThreadsPerCluster || groupLimit.height < 1 || groupLimit.depth < 1)
                return Fail(error, "Metal Clustered Forward requires a 64-thread compute group");
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "ClusteredForward.metal", source, error)) return false;
            auto options = [[MTLCompileOptions alloc] init]; options.languageVersion = MTLLanguageVersion2_0;
            // Reassociation/finite-only assumptions would invalidate error bounds.
            options.fastMathEnabled = NO;
            NSError* diagnostic = nil;
            auto library = Evans::Renderer::CompileCachedMetalLibrary(native,[NSString stringWithUTF8String:source.c_str()],options,&diagnostic);
            [options release];
            if (!library) { error = "Metal Clustered Forward compilation failed: " + Diagnostic(diagnostic); return false; }
            std::array<id<MTLComputePipelineState>, 1> pipelines {};
            const std::array<NSString*, 1> names {@"ClusterAssign"};
            for (unsigned variant = 0; variant < pipelines.size(); ++variant)
            {
                auto function = [library newFunctionWithName:names[variant]];
                if (function) pipelines[variant] = Evans::Renderer::CreateCachedMetalComputePipeline(native,function,&diagnostic);
                [function release];
                if (!pipelines[variant] || pipelines[variant].maxTotalThreadsPerThreadgroup < ThreadsPerCluster ||
                    pipelines[variant].staticThreadgroupMemoryLength > native.maxThreadgroupMemoryLength)
                {
                    error = "Metal Clustered Forward pipeline or threadgroup memory unavailable: " + Diagnostic(diagnostic);
                    for (auto pipeline : pipelines) [pipeline release]; [library release]; return false;
                }
            }
            [library release];
            state.Device = [native retain]; state.Owner = std::this_thread::get_id(); state.Pipelines = pipelines;
            state.BufferLimit = LegacyBufferLimit;
            if (@available(macOS 10.14, iOS 12.0, *)) state.BufferLimit = native.maxBufferLength;
            const auto boundBytes = MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds);
            if (boundBytes <= state.BufferLimit) state.Bounds = [native newBufferWithLength:boundBytes options:MTLResourceStorageModeShared];
            if (!state.Bounds || !state.Bounds.contents) { state.Release(); return Fail(error, "Metal Clustered Forward support upload allocation failed"); }
            state.Bounds.label = @"Evans Clustered Forward support bounds";
            return true;
        }
    }
    bool MetalClusteredForward::BeginFrame(std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (!state.CheckOwner(error)) return false;
        state.Ready = false;
        if (!state.Completed(error)) return false;
        state.ReleaseFrame(); state.Open = true; return true;
    }
    bool MetalClusteredForward::AbortFrame(std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (!state.CheckOwner(error)) return false;
        if (state.Commands && state.Commands.status != MTLCommandBufferStatusNotEnqueued)
            return Fail(error, "Metal Clustered Forward cannot abort enqueued or submitted commands");
        state.ReleaseFrame(); return true;
    }
    bool MetalClusteredForward::Shutdown(std::string& error)
    {
        auto& state = *m_Impl; error.clear(); if (!state.Device) return true;
        if (!state.CheckOwner(error)) return false;
        // An errored submission is terminal and can be released for reinitialization.
        if (state.Commands && state.Commands.status != MTLCommandBufferStatusError && !state.Completed(error)) return false;
        state.Release(); return true;
    }
    bool MetalClusteredForward::Record(void* commandBuffer, std::uint32_t width, std::uint32_t height, const Matrix4& camera,
        const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (!state.CheckOwner(error)) return false;
        state.Ready = false;
        if (!state.Open || state.Commands) return Fail(error, "Metal Clustered Forward requires one recording per open frame");
        auto commands = (id<MTLCommandBuffer>)commandBuffer;
        if (!commands || commands.device != state.Device || commands.status != MTLCommandBufferStatusNotEnqueued)
            return Fail(error, "Metal clustered assignment requires a fresh command buffer on its initialized device");
        ClusteredForwardGrid grid; ClusteredForwardConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildClusteredForwardGrid(width, height, lights.size(), parameters, grid, error) ||
            !BuildClusteredForwardConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error)) return false;
        if (!state.EnsureLists(grid, error)) return false;
        if (!bounds.empty()) std::memcpy(state.Bounds.contents, bounds.data(), bounds.size() * sizeof(ForwardPlusGpuBounds));
        // Tracked resources directly bound to serial encoders on one classic
        // MTLCommandBuffer order lists -> lighting/blit. This contract
        // does not use Metal 4 queues, heaps with disabled tracking, or argument buffers.
        // https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/tracked
        auto encoder = [commands computeCommandEncoder];
        if (!encoder) return Fail(error, "Metal Clustered Forward compute encoder creation failed");
        encoder.label = @"Evans Clustered Forward conservative tile assignment";
        [encoder setComputePipelineState:state.Pipelines[0]];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:0];
        [encoder setBuffer:state.Bounds offset:0 atIndex:1]; [encoder setBuffer:state.Headers offset:0 atIndex:2];
        [encoder setBuffer:state.Indices offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(grid.TilesX, grid.TilesY, grid.DepthSlices) threadsPerThreadgroup:MTLSizeMake(ThreadsPerCluster, 1, 1)];
        [encoder endEncoding];
        state.Commands = [commands retain];
        state.CurrentGrid = grid; state.Ready = true; return true;
    }
    void* MetalClusteredForward::Headers() const { return m_Impl->Usable() ? (void*)m_Impl->Headers : nullptr; }
    void* MetalClusteredForward::Indices() const { return m_Impl->Usable() ? (void*)m_Impl->Indices : nullptr; }
    ClusteredForwardGrid MetalClusteredForward::Grid() const { return m_Impl->Usable() ? m_Impl->CurrentGrid : ClusteredForwardGrid {}; }
}
