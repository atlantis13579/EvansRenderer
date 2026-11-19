ObjCObject LumenGlobalCompose=nullptr;
void DestroyLumenGlobalDistance() { DestroyLumenGlobalDistanceAsync(); ReleaseObject(LumenGlobalCompose); LumenGlobalCompose=nullptr; }
bool EnsureLumenGlobalDistance()
{
    if (LumenGlobalCompose) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","LumenGlobalDistance.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal global distance compilation failed: "+ErrorDescription(error));
    compilation.Function=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString("lumenComposeGlobalDistance"));
    if (!compilation.Function) return Fail("Metal global distance kernel is unavailable");
    LumenGlobalCompose=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Function,&error);
    if (!LumenGlobalCompose) return Fail("Metal global distance pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenGlobalCompose,"maxTotalThreadsPerThreadgroup")<LumenDistanceTraceThreads)
    { DestroyLumenGlobalDistance(); return Fail("Metal global distance threadgroup exceeds device limits"); }
    return true;
}
bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source scene");
    std::shared_ptr<LumenDistanceRecord> retained;
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second)
            return Fail("Global distance reuse requires the exact immutable source scene");
        retained=old->second;
    }
    auto plan=std::make_shared<LumenGlobalDistancePlan>();
    if (!BuildLumenGlobalDistancePlan(found->second->Info,settings,*plan,LastError) || !EnsureLumenGlobalDistance()) return false;
    LumenGlobalDistanceUpdate update;
    if (!BuildLumenGlobalDistanceUpdate(found->second->Info,*plan,retained?retained->Info.Global.get():nullptr,update,LastError)) return false;
    auto frame=plan->Frame; frame.Dispatch={0,unsigned(update.Compose.size()),0,0};
    auto candidate=std::make_shared<LumenDistanceRecord>(); candidate->Base=found->second; candidate->Info=candidate->Base->Info; candidate->Info.Global=plan; candidate->Hardware=candidate->Base->Hardware;
    candidate->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(plan->WordCount)*sizeof(std::uint32_t),0);
    if (!candidate->Buffer) return Fail("Metal global distance allocation failed");
    auto* mapped=Send<std::uint32_t*>(candidate->Buffer,"contents"); if (!mapped) return Fail("Metal global distance header upload is unavailable");
    std::memcpy(mapped+plan->Frame.Header.Layout.W,&plan->Frame.Header,sizeof(LumenGlobalDistanceHeader));
    struct DirtyBuffer { ObjCObject Buffer=nullptr; ~DirtyBuffer() { ReleaseObject(Buffer); } } dirty;
    dirty.Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::max<std::size_t>(1,update.Compose.size())*sizeof(std::uint32_t),0);
    if (!dirty.Buffer) return Fail("Metal global distance dirty-list allocation failed");
    auto* indices=Send<void*>(dirty.Buffer,"contents"); if (!indices) return Fail("Metal global distance dirty-list upload is unavailable");
    if (!update.Compose.empty()) std::memcpy(indices,update.Compose.data(),update.Compose.size()*sizeof(std::uint32_t));
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal global distance command buffer is unavailable");
    const auto blit=Send<ObjCObject>(command,"blitCommandEncoder"); if (!blit) return Fail("Metal global distance blit is unavailable");
    Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(blit,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
        candidate->Base->Buffer,0,candidate->Buffer,0,std::max(1u,candidate->Info.Counts.W)*sizeof(std::uint32_t));
    for (const auto& copy:update.Copies)
        Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(blit,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            retained->Buffer,std::size_t(copy.SourceWord)*sizeof(std::uint32_t),candidate->Buffer,
            std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t));
    Send<void>(blit,"endEncoding");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal global distance encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenGlobalCompose);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",candidate->Base->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",dirty.Buffer,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",candidate->Buffer,0,3);
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const MTLSize grid{std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1},threads{LumenDistanceTraceThreads,1,1};
    if (groups) Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal global distance composition failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
