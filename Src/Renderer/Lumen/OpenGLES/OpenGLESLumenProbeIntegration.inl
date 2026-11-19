struct LumenProbeRecord
{
    Impl& Owner; LumenProbeInfo Info; LumenRadianceHandle SourceHandle; GLuint Buffer=0;
    explicit LumenProbeRecord(Impl& owner):Owner(owner) {}
    ~LumenProbeRecord() { if (Buffer) Owner.GlDeleteBuffers(1,&Buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenProbeRecord>> LumenProbes;
GLuint LumenProbeIntegrate=0;
void DestroyLumenProbes()
{ DestroyLumenGather(); FrameLumen.Finish(false,LumenRadianceTraces,LumenProbes); LumenProbes.clear(); if (LumenProbeIntegrate) GlDeleteProgram(LumenProbeIntegrate); LumenProbeIntegrate=0; }
bool EnsureLumenProbeIntegration()
{
    if (!EnsureLumenSurface() || !EnsureLumenDistanceTrace()) return false;
    if (LumenProbeIntegrate) return true;
    GLint bindings=0; GlGetIntegerv(DistanceMaxComputeStorage,&bindings); if (bindings<3) return Fail("Lumen probe integration needs three compute storage bindings");
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen probe shader has no version directive");
    source.insert(version+1,"#define EVANS_LUMEN_PROBE_INTEGRATE_STAGE 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenProbeIntegrate=program; return true;
}
bool ReadLumenProbeRecord(const LumenProbeRecord& record,std::vector<LumenProbeIrradiance>& output)
{
    struct Binding
    {
        Impl& Owner; GLint Previous=0;
        explicit Binding(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(GpuStorageBinding,&Previous); }
        ~Binding() { Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,Previous); }
    } binding(*this);
    const auto count=std::size_t(LumenRadianceProbeCount(record.Info.Source)),bytes=count*sizeof(LumenProbeIrradiance); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record.Buffer);
    std::vector<LumenProbeIrradiance> probes(count); const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead);
    if (!mapped) return Fail("Map Lumen probe readback failed");
    std::memcpy(probes.data(),mapped,bytes); const auto valid=LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen probe irradiance")) return Fail("Lumen probe readback became invalid");
    if (!ValidateLumenProbeIrradiance(record.Info,probes,LastError)) return false;
    output=std::move(probes); return true;
}
bool IntegrateLumenProbes(LumenRadianceHandle sourceHandle,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    const auto found=LumenRadianceTraces.find(sourceHandle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen probe radiance source");
    const auto source=found->second; LumenProbeIntegrationFrame frame;
    if (!BuildLumenProbeIntegrationFrame(source->Info,upload,frame,LastError) || (FrameOpen && !FrameLumen.CanIntegrate(frame.Counts.X,LastError)) || !EnsureLumenProbeIntegration()) return false;
    const auto bytes=std::size_t(frame.Counts.X)*sizeof(LumenProbeIrradiance);
    std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (maximum<=0 || bytes>std::uint64_t(maximum)) return Fail("Lumen probes exceed ES storage size");
    const auto groups=(frame.Counts.X+LumenProbeThreads-1)/LumenProbeThreads;
    GLint maxX=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); if (groups>unsigned(maxX)) return Fail("Lumen probes exceed ES dispatch limit");
    auto record=std::make_shared<LumenProbeRecord>(*this); record->Info={source->Info,upload.SkyRadiance,upload.Environment}; record->SourceHandle=sourceHandle;
    struct Temporary { Impl& Owner; GLuint Buffers[1]{}; ~Temporary() { Owner.GlDeleteBuffers(1,Buffers); } } work{*this};
    LumenCaptureState state(*this); LumenRadianceStorageState storageState(*this); GlGenBuffers(1,work.Buffers);
    GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    GlGenBuffers(1,&record->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW);
    if (!CheckGL("Allocate Lumen probe integration")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,source->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,record->Buffer); GlUseProgram(LumenProbeIntegrate);
    GpuDispatch(groups,1,1); GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier); if (!CheckGL("Integrate Lumen probe irradiance")) return false;
    LumenProbeIntegrationResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenProbeRecord(*record,result.Probes)) return false; }
    else if (!FrameOpen) { LumenFinish(); if (!CheckGL("Complete Lumen probe irradiance")) return false; }
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
bool LumenCommandsEnded() const { return false; }
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
