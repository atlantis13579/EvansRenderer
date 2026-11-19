#pragma once
#if defined(__APPLE__)
#if defined(__OBJC__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif
namespace Evans::Renderer
{
    // Opaque entry points also serve the C++ Objective-C runtime backend.
    void* CompileCachedMetalLibrary(void* device,void* source,void* options,void** error);
    void* CreateCachedMetalRenderPipeline(void* device,void* descriptor,void** error);
    void* CreateCachedMetalComputePipeline(void* device,void* function,void** error);
    void* CreateCachedMetalMeshPipeline(void* device,void* descriptor,void** error);
#if defined(__OBJC__)
    inline id<MTLLibrary> CompileCachedMetalLibrary(id<MTLDevice> device,NSString* source,MTLCompileOptions* options,NSError** error)
    { return (id<MTLLibrary>)CompileCachedMetalLibrary((void*)device,(void*)source,(void*)options,(void**)error); }
    inline id<MTLRenderPipelineState> CreateCachedMetalRenderPipeline(id<MTLDevice> device,MTLRenderPipelineDescriptor* descriptor,NSError** error)
    { return (id<MTLRenderPipelineState>)CreateCachedMetalRenderPipeline((void*)device,(void*)descriptor,(void**)error); }
    inline id<MTLComputePipelineState> CreateCachedMetalComputePipeline(id<MTLDevice> device,id<MTLFunction> function,NSError** error)
    { return (id<MTLComputePipelineState>)CreateCachedMetalComputePipeline((void*)device,(void*)function,(void**)error); }
#endif
}
#endif
