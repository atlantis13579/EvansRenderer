void*(EVANS_GL_CALL* LightingFenceSync)(GLenum,GLbitfield)=nullptr;
GLenum(EVANS_GL_CALL* LightingClientWaitSync)(void*,GLbitfield,std::uint64_t)=nullptr;
void(EVANS_GL_CALL* LightingDeleteSync)(void*)=nullptr;
void(EVANS_GL_CALL* LightingFlush)()=nullptr;
static constexpr GLenum LightingSyncComplete=0x9117,LightingAlreadySignaled=0x911A,LightingTimeoutExpired=0x911B,LightingConditionSatisfied=0x911C;
struct LumenLightingSliceWork : LumenLightingJob
{
    Impl& Owner; std::shared_ptr<LumenLightingRecord> Record,Previous;
    GLuint Uniform=0,Data=0; void* Completion=nullptr; bool Unfenced=false;
    explicit LumenLightingSliceWork(Impl& owner):Owner(owner) {}
    ~LumenLightingSliceWork()
    {
        if (Completion) Owner.LightingDeleteSync(Completion);
        if (Uniform) Owner.GlDeleteBuffers(1,&Uniform); if (Data) Owner.GlDeleteBuffers(1,&Data);
    }
};
LumenLightingQueue<LumenLightingSliceWork> LumenLightingTasks;
bool CollectLumenLighting()
{ return LumenLightingTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenLighting(work,ready); }); }
bool CancelLumenLightingUpdate(LumenLightingTask task)
{ return (LumenLightingTasks.Cancel(task) || Fail("Invalid Lumen lighting task")) && CollectLumenLighting(); }
void DestroyLumenLightingAsync() { LumenLightingTasks.Clear(); }
bool ProbeLumenLighting(LumenLightingSliceWork& work,bool& ready)
{
    if (work.Unfenced) return Fail("Lumen lighting submission has no ES completion fence");
    const auto status=LightingClientWaitSync(work.Completion,0,0);
    ready=status==LightingAlreadySignaled || status==LightingConditionSatisfied;
    return ready || status==LightingTimeoutExpired || Fail("ES lighting slice fence query failed");
}
bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& output)
{
    if (!LoadGL(LightingFenceSync,"glFenceSync") || !LoadGL(LightingClientWaitSync,"glClientWaitSync") ||
        !LoadGL(LightingDeleteSync,"glDeleteSync") || !LoadGL(LightingFlush,"glFlush") || !CollectLumenLighting() || !EnsureLumenSurfaceLighting(bool(upload.PreviousLighting))) return false;
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    auto job=std::make_shared<LumenLightingSliceWork>(*this);
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,job->Previous)) return false;
    if (!job->Prepare(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,LastError) || !LumenLightingTasks.CanSubmit(*job,LastError)) return false;
    const auto& input=job->Input;
    const auto bytes=std::size_t(input.Frame.Trace.Counts.X)*sizeof(LumenLightingTexel),inputBytes=input.Words.size()*sizeof(std::uint32_t);
    std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (maximum<=0 || std::max(bytes,inputBytes)>std::uint64_t(maximum)) return Fail("Lumen lighting exceeds ES storage limits");
    auto record=std::make_shared<LumenLightingRecord>(*this); record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{record->Surface->Identity,record->Distance->Info.Identity,upload.LightingRevision},record->Surface->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (job->Previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=job->Previous->Info.Bounce+1; }
    job->Record=record;
    LumenCaptureState state(*this); LumenLookupStorageState storageState(*this);
    GlGenBuffers(1,&job->Uniform); GlBindBuffer(GL_UNIFORM_BUFFER,job->Uniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(input.Frame),nullptr,GL_DYNAMIC_DRAW);
    GlGenBuffers(1,&job->Data); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,job->Data); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(inputBytes),input.Words.data(),GL_STATIC_DRAW);
    GlGenBuffers(1,&record->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW);
    if (!CheckGL("Allocate lighting slice resources")) return false;
    const auto task=LumenLightingTasks.Insert(job); if (!task) return Fail("Lumen lighting task space exhausted"); output=task; return true;
}
bool SubmitLumenLightingSlice(LumenLightingSliceWork& work,const LumenSurfaceLightingFrame& frame)
{
    const auto groups=LumenLightingGroups(frame),x=std::min(groups,LumenLightingColumns),y=(groups+LumenLightingColumns-1)/LumenLightingColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Lighting slice exceeds ES dispatch limits");
    if (work.Completion) { LightingDeleteSync(work.Completion); work.Completion=nullptr; }
    LumenCaptureState state(*this); LumenLookupStorageState storageState(*this); const auto& record=work.Record;
    GlBindBuffer(GL_UNIFORM_BUFFER,work.Uniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(frame),&frame); GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Uniform);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record->Distance->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,work.Data);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,record->Buffer);
    if (work.Previous) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,work.Previous->Buffer);
    const auto pipeline=work.Previous?LumenSurfaceRadiosity:LumenSurfaceLighting;
    GlUseProgram(pipeline); const char* names[]{"LightingAlbedo","LightingNormal","LightingEmission"};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        GlActiveTexture(GL_TEXTURE0+plane); LumenBindSampler(plane,0); GlBindTexture(GL_TEXTURE_2D,record->Surface->Images[plane]);
        GlUniform1i(GlGetUniformLocation(pipeline,names[plane]),GLint(plane));
    }
    if (!CheckGL("Bind lighting slice resources")) return false;
    work.InFlight=true; GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier);
    work.Completion=LightingFenceSync(LightingSyncComplete,0); LightingFlush();
    if (!work.Completion) { work.Unfenced=true; return Fail("Create lighting slice fence failed"); }
    return CheckGL("Submit lighting slice");
}
bool AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    if (!CollectLumenLighting()) return false;
    return LumenLightingTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenLighting(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenLightingSlice(work,frame); },
        [&](auto& work,LumenSurfaceLightingResult& result)
        {
            result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
            result.Info=work.Record->Info; LumenLightingCaches.emplace(result.Handle.Value,work.Record); return true;
        });
}
