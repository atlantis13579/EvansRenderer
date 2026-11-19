struct LumenHardwareRecord;
struct LumenDistanceRecord
{
    LumenDistanceSceneInfo Info; std::shared_ptr<LumenDistanceRecord> Base; ObjCObject Buffer=nullptr;
    std::shared_ptr<LumenHardwareRecord> Hardware;
    ~LumenDistanceRecord() { ReleaseObject(Buffer); }
};
#include "Renderer/Lumen/Metal/MetalLumenHardwareRayTracing.inl"
std::unordered_map<std::uint64_t,std::shared_ptr<LumenDistanceRecord>> LumenDistanceScenes;
ObjCObject LumenDistanceTrace=nullptr;
void DestroyLumenDistanceScenes()
{ LumenDistanceScenes.clear(); DestroyLumenGlobalDistance(); DestroyLumenHardwarePrograms(); ReleaseObject(LumenDistanceTrace); LumenDistanceTrace=nullptr; }
bool EnsureLumenDistanceTrace(bool hardware=false)
{
    if (hardware) return EnsureLumenHardwarePipeline(LumenHardwareProgram::Distance);
    if (LumenDistanceTrace) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
    Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
    compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal Lumen distance compilation failed: "+ErrorDescription(error));
    compilation.Function=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString("lumenTraceDistance"));
    if (!compilation.Function) return Fail("Metal Lumen distance kernel is unavailable");
    LumenDistanceTrace=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Function,&error);
    if (!LumenDistanceTrace) return Fail("Metal Lumen distance pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenDistanceTrace,"maxTotalThreadsPerThreadgroup")<LumenDistanceTraceThreads)
    { ReleaseObject(LumenDistanceTrace); LumenDistanceTrace=nullptr; return Fail("Metal Lumen distance threadgroup exceeds device limits"); }
    return true;
}
bool CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!ValidateLumenDistanceScene(data,LastError) || !EnsureLumenDistanceTrace()) return false;
    const std::uint32_t empty=0; const auto bytes=std::max<std::size_t>(1,data.Words.size())*sizeof(std::uint32_t);
    auto candidate=std::make_shared<LumenDistanceRecord>(); candidate->Info=data.Info;
    candidate->Buffer=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",data.Words.empty()?&empty:data.Words.data(),bytes,0);
    if (!candidate->Buffer) return Fail("Metal immutable Lumen distance upload failed");
    if (!BuildLumenHardwareAcceleration(candidate)) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
bool TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    const auto found=LumenDistanceScenes.find(handle.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown Lumen distance scene");
    const auto record=found->second; LumenDistanceTraceFrame frame;
    if (!BuildLumenDistanceTraceFrame(record->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace(bool(record->Hardware))) return false;
    struct Resources
    {
        ObjCObject Queries=nullptr,Feedback=nullptr;
        ~Resources() { ReleaseObject(Queries); ReleaseObject(Feedback); }
    } resources;
    resources.Queries=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",upload.Rays.data(),upload.Rays.size()*sizeof(LumenDistanceRay),0);
    const auto bytes=upload.Rays.size()*sizeof(LumenDistanceHit);
    resources.Feedback=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    if (!resources.Queries || !resources.Feedback || !Send<void*>(resources.Feedback,"contents")) return Fail("Metal Lumen distance query allocation failed");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen distance command buffer is unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen distance encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",record->Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Distance)]:LumenDistanceTrace);
    BindLumenHardware(encoder,*record);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Queries,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Feedback,0,3);
    const auto groups=unsigned((upload.Rays.size()+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads);
    const MTLSize grid{std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1},threads{LumenDistanceTraceThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen distance submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    LumenDistanceTraceResult result; result.Identity=record->Info.Identity; result.Hits.resize(upload.Rays.size()); result.HardwareTracing=bool(record->Hardware);
    std::memcpy(result.Hits.data(),Send<void*>(resources.Feedback,"contents"),bytes);
    if (!ValidateLumenDistanceTraceResult(record->Info,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
bool DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{ return LumenDistanceScenes.erase(handle.Value)!=0 || Fail("Unknown Lumen distance scene"); }
