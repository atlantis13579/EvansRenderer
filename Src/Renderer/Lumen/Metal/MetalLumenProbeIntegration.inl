struct LumenProbeRecord
{
    LumenProbeInfo Info; LumenRadianceHandle SourceHandle; ObjCObject Buffer=nullptr;
    ~LumenProbeRecord() { ReleaseObject(Buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenProbeRecord>> LumenProbes;
ObjCObject LumenProbeIntegrate=nullptr;
void DestroyLumenProbes()
{ DestroyLumenGather(); FrameLumen.Finish(false,LumenRadianceTraces,LumenProbes); LumenProbes.clear(); ReleaseObject(LumenProbeIntegrate); LumenProbeIntegrate=nullptr; }
bool EnsureLumenProbeIntegration()
{
    if (LumenProbeIntegrate) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr; compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal Lumen probe compilation failed: "+ErrorDescription(error));
    compilation.Function=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString("lumenIntegrateProbes"));
    if (!compilation.Function) return Fail("Metal Lumen probe integration kernel is unavailable");
    LumenProbeIntegrate=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Function,&error);
    if (!LumenProbeIntegrate) return Fail("Metal Lumen probe pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenProbeIntegrate,"maxTotalThreadsPerThreadgroup")<LumenProbeThreads)
    { ReleaseObject(LumenProbeIntegrate); LumenProbeIntegrate=nullptr; return Fail("Metal Lumen probe integration exceeds threadgroup capacity"); }
    return true;
}
bool ReadLumenProbeRecord(const LumenProbeRecord& record,std::vector<LumenProbeIrradiance>& output)
{
    const auto* contents=Send<void*>(record.Buffer,"contents"); if (!contents) return Fail("Metal Lumen probe buffer is unavailable");
    std::vector<LumenProbeIrradiance> probes(LumenRadianceProbeCount(record.Info.Source)); std::memcpy(probes.data(),contents,probes.size()*sizeof(LumenProbeIrradiance));
    if (!ValidateLumenProbeIrradiance(record.Info,probes,LastError)) return false;
    output=std::move(probes); return true;
}
bool IntegrateLumenProbes(LumenRadianceHandle sourceHandle,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    const auto found=LumenRadianceTraces.find(sourceHandle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen probe radiance source");
    const auto source=found->second; LumenProbeIntegrationFrame frame;
    if (!BuildLumenProbeIntegrationFrame(source->Info,upload,frame,LastError) || (FrameOpen && !FrameLumen.CanIntegrate(frame.Counts.X,LastError)) || !EnsureLumenProbeIntegration()) return false;
    auto record=std::make_shared<LumenProbeRecord>(); record->Info={source->Info,upload.SkyRadiance,upload.Environment}; record->SourceHandle=sourceHandle;
    record->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(frame.Counts.X)*sizeof(LumenProbeIrradiance),0);
    if (!record->Buffer || !Send<void*>(record->Buffer,"contents")) return Fail("Metal Lumen probe allocation failed");
    const auto command=FrameOpen?CommandBuffer:Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen probe command buffer is unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen probe encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenProbeIntegrate); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->Origins,0,2); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Buffer,0,3);
    const auto groups=(frame.Counts.X+LumenProbeThreads-1)/LumenProbeThreads;
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{groups,1,1},MTLSize{LumenProbeThreads,1,1}); Send<void>(encoder,"endEncoding");
    if (!FrameOpen)
    {
        Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
        if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen probe submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    }
    LumenProbeIntegrationResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenProbeRecord(*record,result.Probes)) return false;
    result.Handle=NewLumenProbeHandle(); if (!result.Handle) return Fail("Lumen probe handle space exhausted");
    LumenProbes.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,frame.Counts.X);
    output=std::move(result); return true;
}
bool ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen probe cache");
    return ReadLumenProbeRecord(*found->second,output);
}
bool DestroyLumenProbes(LumenProbeHandle handle) { return LumenProbes.erase(handle.Value)!=0 || Fail("Unknown Lumen probe cache"); }

// Handles are usable by later GPU work in this frame, then either published or
// cancelled together. Native command ownership is independent of publication.
bool LumenCommandsEnded() const { constexpr NSUInteger CommittedStatus=2; return !CommandBuffer || Send<NSUInteger>(CommandBuffer,"status")>=CommittedStatus; }
bool ValidateLumenRecording(bool readback)
{
    if (PassOpen || (FrameOpen && (readback || LumenCommandsEnded())))
        return Fail("Lumen frame work requires an open command stream, no graphics pass and no CPU readback");
    return true;
}
void FinishLumenFrame(bool publish)
{
    LumenHistory.Finish(publish); FinishLumenReflections(publish);
    FrameLumen.Finish(publish,LumenRadianceTraces,LumenProbes);
    ReleaseLumenGIResults();

}
struct LumenPresentPublication
{
    Impl& Owner; bool Published=false;
    ~LumenPresentPublication()
    { if (Published || !Owner.FrameOpen || Owner.LumenCommandsEnded()) Owner.FinishLumenFrame(Published); }
};
