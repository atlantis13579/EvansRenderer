#include "Renderer/Metal/MetalShaderCache.h"
#include "Renderer/ShaderCache.h"
#include <dispatch/dispatch.h>
#include <objc/message.h>
#include <fstream>
#include <mutex>
#include <cstring>
#include <cstdlib>

namespace Evans::Renderer
{
    namespace
    {
        std::mutex MetalCacheMutex;
        NSString* Text(const std::string& value) { return [NSString stringWithUTF8String:value.c_str()]; }
        NSURL* Url(const std::filesystem::path& path) { return [NSURL fileURLWithPath:Text(path.string())]; }
        std::string Utf8(NSString* value) { const auto* text=[value UTF8String];return text?text:""; }
        struct WorkDirectory
        {
            std::filesystem::path Path=ShaderCache::Directory()/"metal-work"/Utf8([[NSUUID UUID] UUIDString]);
            WorkDirectory() { std::filesystem::create_directories(Path); }
            ~WorkDirectory() { std::error_code error;std::filesystem::remove_all(Path,error); }
        };
        bool Run(NSArray<NSString*>* arguments,std::string& output)
        {
            NSTask* task=[[NSTask alloc] init];NSPipe* pipe=[NSPipe pipe];
            task.executableURL=[NSURL fileURLWithPath:@"/usr/bin/xcrun"];task.arguments=arguments;
            task.standardOutput=pipe;task.standardError=pipe;
            NSError* error=nil;
            if(![task launchAndReturnError:&error]){[task release];return false;}
            NSData* data=[[pipe fileHandleForReading] readDataToEndOfFile];[task waitUntilExit];
            output.assign(static_cast<const char*>(data.bytes),data.length);
            const bool success=task.terminationStatus==0;[task release];return success;
        }
        std::string CompilerIdentity()
        {
            std::string path,version,sdk;
            if(!Run(@[@"--sdk",@"macosx",@"--find",@"metal"],path) ||
                !Run(@[@"--sdk",@"macosx",@"metal",@"--version"],version) ||
                !Run(@[@"--sdk",@"macosx",@"--show-sdk-version"],sdk))return {};
            while(!path.empty() && (path.back()=='\n' || path.back()=='\r'))path.pop_back();
            const auto binary=ShaderCache::FileIdentity(path);
            return binary.empty()?std::string{}:binary+version+sdk;
        }
        id<MTLLibrary> LibraryFromBytes(id<MTLDevice> device,const std::vector<std::uint8_t>& data,NSError** error)
        {
            void* copy=std::malloc(data.size());if(!copy)return nil;std::memcpy(copy,data.data(),data.size());
            dispatch_data_t bytes=dispatch_data_create(copy,data.size(),nullptr,DISPATCH_DATA_DESTRUCTOR_FREE);
            id<MTLLibrary> library=[device newLibraryWithData:bytes error:error];dispatch_release(bytes);return library;
        }
        bool ReadFile(const std::filesystem::path& path,std::vector<std::uint8_t>& bytes)
        {
            std::ifstream input(path,std::ios::binary|std::ios::ate);if(!input)return false;
            const auto size=input.tellg();if(size<=0 || size>256*1024*1024)return false;
            input.seekg(0);bytes.resize(std::size_t(size));return bool(input.read(reinterpret_cast<char*>(bytes.data()),size));
        }
        bool WriteFile(const std::filesystem::path& path,const void* bytes,std::size_t size)
        { std::ofstream out(path,std::ios::binary);out.write(static_cast<const char*>(bytes),std::streamsize(size));out.close();return bool(out); }
        struct Archive
        {
            WorkDirectory Work;ShaderCache::Digest Key{};id<MTLBinaryArchive> Object=nil;
            explicit Archive(id<MTLDevice> device)
            {
                if(@available(macOS 11.0,*))
                {
                    ShaderCache::Key key("metal-pipeline-v1");key.Value(device.registryID).Add(Utf8([[NSProcessInfo processInfo] operatingSystemVersionString]));Key=key.Finish();
                    MTLBinaryArchiveDescriptor* description=[[MTLBinaryArchiveDescriptor alloc] init];
                    std::vector<std::uint8_t> data;
                    if(ShaderCache::Read("metal-pipeline",Key,data) && WriteFile(Work.Path/"input.metallib",data.data(),data.size()))
                        description.url=Url(Work.Path/"input.metallib");
                    Object=[device newBinaryArchiveWithDescriptor:description error:nullptr];
                    if(!Object && description.url)
                    { description.url=nil;Object=[device newBinaryArchiveWithDescriptor:description error:nullptr]; }
                    [description release];
                }
            }
            ~Archive() { [Object release]; }
            void Save() noexcept
            {
                try
                {
                    if(@available(macOS 11.0,*))
                    {
                        if(Object && [Object serializeToURL:Url(Work.Path/"output.metallib") error:nullptr])
                        {
                            std::vector<std::uint8_t> data;
                            if(ReadFile(Work.Path/"output.metallib",data))ShaderCache::Write("metal-pipeline",Key,data.data(),data.size());
                        }
                    }
                }
                catch(...) {}
            }
        };
    }
    void* CompileCachedMetalLibrary(void* native,void* text,void* optionsValue,void** error)
    {
        auto device=(id<MTLDevice>)native;auto source=(NSString*)text;auto options=(MTLCompileOptions*)optionsValue;
        if(@available(macOS 11.0,*)) {}
        else { return [device newLibraryWithSource:source options:options error:(NSError**)error]; }
        if(!ShaderCache::Enabled())
        { return [device newLibraryWithSource:source options:options error:(NSError**)error]; }
        std::lock_guard<std::mutex> lock(MetalCacheMutex);
        try
        {
            static const auto compiler=CompilerIdentity();
            // Runtime MTLLibrary objects cannot be serialized. The native Metal
            // toolchain produces reusable IR; machines without it retain the
            // ordinary runtime compiler path and still benefit from PSO archives.
            if(!compiler.empty())
            {
                if(!options)options=[[[MTLCompileOptions alloc] init] autorelease];
                NSMutableArray<NSString*>* arguments=[NSMutableArray arrayWithObjects:@"--sdk",@"macosx",@"metal",@"-c",@"-O3",nil];
                const auto language=static_cast<unsigned long>(options.languageVersion);
                // Metal 1.x-2.x spell the target as macos-metalM.m; 3.0 and later drop the prefix.
                if(language)
                {
                    const auto major=language>>16, minor=language&65535;
                    [arguments addObject:Text((major>=3?"-std=metal":"-std=macos-metal")+std::to_string(major)+"."+std::to_string(minor))];
                }
                [arguments addObject:options.fastMathEnabled?@"-ffast-math":@"-fno-fast-math"];
                if(@available(macOS 11.0,*))if(options.preserveInvariance)[arguments addObject:@"-fpreserve-invariance"];
                for(NSString* name in [[options.preprocessorMacros allKeys] sortedArrayUsingSelector:@selector(compare:)])
                { [arguments addObject:@"-D"];[arguments addObject:[NSString stringWithFormat:@"%@=%@",name,options.preprocessorMacros[name]]]; }
                ShaderCache::Key key("metal-library-v1");key.Add(compiler).Add(Utf8(source));
                for(NSString* argument in arguments)key.Add(Utf8(argument));
                const auto digest=key.Finish();std::vector<std::uint8_t> data;
                NSError* cacheError=nil;
                if(ShaderCache::Read("metal-shader",digest,data))
                {
                    auto library=LibraryFromBytes(device,data,&cacheError);if(library)return library;
                }
                WorkDirectory work;const auto sourcePath=work.Path/"source.metal",air=work.Path/"source.air",libraryPath=work.Path/"source.metallib";
                NSData* utf8=[source dataUsingEncoding:NSUTF8StringEncoding];
                if(WriteFile(sourcePath,utf8.bytes,utf8.length))
                {
                    [arguments addObjectsFromArray:@[Text(sourcePath.string()),@"-o",Text(air.string())]];
                    std::string diagnostics;
                    if(Run(arguments,diagnostics) && Run(@[@"--sdk",@"macosx",@"metallib",Text(air.string()),@"-o",Text(libraryPath.string())],diagnostics) && ReadFile(libraryPath,data))
                    {
                        auto library=LibraryFromBytes(device,data,&cacheError);
                        if(library){ShaderCache::Write("metal-shader",digest,data.data(),data.size());return library;}
                    }
                }
            }
        }
        catch(...) {}
        return [device newLibraryWithSource:source options:(MTLCompileOptions*)optionsValue error:(NSError**)error];
    }
    void* CreateCachedMetalRenderPipeline(void* native,void* value,void** error)
    {
        auto device=(id<MTLDevice>)native;auto description=(MTLRenderPipelineDescriptor*)value;
        if(@available(macOS 11.0,*))if(ShaderCache::Enabled())
        {
            std::lock_guard<std::mutex> lock(MetalCacheMutex);
            try
            {
                Archive archive(device);
                if(archive.Object)
                {
                    MTLRenderPipelineDescriptor* descriptor=[description copy];descriptor.binaryArchives=@[archive.Object];NSError* miss=nil;
                    auto pipeline=[device newRenderPipelineStateWithDescriptor:descriptor options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nullptr error:&miss];
                    if(!pipeline)
                    {
                        const bool recorded=[archive.Object addRenderPipelineFunctionsWithDescriptor:descriptor error:nullptr];
                        pipeline=[device newRenderPipelineStateWithDescriptor:descriptor error:(NSError**)error];
                        if(pipeline && recorded)archive.Save();
                    }
                    [descriptor release];return pipeline;
                }
            }
            catch(...) {}
        }
        return [device newRenderPipelineStateWithDescriptor:description error:(NSError**)error];
    }
    void* CreateCachedMetalComputePipeline(void* native,void* function,void** error)
    {
        auto device=(id<MTLDevice>)native;
        if(@available(macOS 11.0,*))if(ShaderCache::Enabled())
        {
            std::lock_guard<std::mutex> lock(MetalCacheMutex);
            try
            {
                Archive archive(device);
                if(archive.Object)
                {
                    MTLComputePipelineDescriptor* descriptor=[[MTLComputePipelineDescriptor alloc] init];
                    descriptor.computeFunction=(id<MTLFunction>)function;descriptor.binaryArchives=@[archive.Object];NSError* miss=nil;
                    auto pipeline=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nullptr error:&miss];
                    if(!pipeline)
                    {
                        const bool recorded=[archive.Object addComputePipelineFunctionsWithDescriptor:descriptor error:nullptr];
                        pipeline=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nullptr error:(NSError**)error];
                        if(pipeline && recorded)archive.Save();
                    }
                    [descriptor release];return pipeline;
                }
            }
            catch(...) {}
        }
        return [device newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError**)error];
    }
    void* CreateCachedMetalMeshPipeline(void* native,void* value,void** error)
    {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
        if(@available(macOS 13.0,*))if(ShaderCache::Enabled())
        {
            auto device=(id<MTLDevice>)native;
            std::lock_guard<std::mutex> lock(MetalCacheMutex);
            try
            {
                Archive archive(device);
                if(archive.Object)
                {
                    MTLMeshRenderPipelineDescriptor* descriptor=[(MTLMeshRenderPipelineDescriptor*)value copy];
                    descriptor.binaryArchives=@[archive.Object];NSError* miss=nil;
                    auto pipeline=[device newRenderPipelineStateWithMeshDescriptor:descriptor options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nullptr error:&miss];
                    if(!pipeline)
                    {
                        const bool recorded=[archive.Object addMeshRenderPipelineFunctionsWithDescriptor:descriptor error:nullptr];
                        pipeline=[device newRenderPipelineStateWithMeshDescriptor:descriptor options:MTLPipelineOptionNone reflection:nullptr error:(NSError**)error];
                        if(pipeline && recorded)archive.Save();
                    }
                    [descriptor release];return pipeline;
                }
            }
            catch(...) {}
        }
#endif
        // Preserve the runtime backend's feature negotiation with older SDKs.
        using Create=void*(*)(void*,SEL,void*,NSUInteger,void*,void**);
        return reinterpret_cast<Create>(objc_msgSend)(native,sel_registerName("newRenderPipelineStateWithMeshDescriptor:options:reflection:error:"),value,0,nullptr,error);
    }
}
