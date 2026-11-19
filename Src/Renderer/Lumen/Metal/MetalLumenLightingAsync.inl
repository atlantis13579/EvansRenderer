struct LumenLightingSliceWork : LumenLightingJob
{
    std::shared_ptr<LumenLightingRecord> Record,Previous;
    ObjCObject Data=nullptr,Completion=nullptr;
    ~LumenLightingSliceWork() { ReleaseObject(Data); ReleaseObject(Completion); }
};
LumenLightingQueue<LumenLightingSliceWork> LumenLightingTasks;
bool CollectLumenLighting()
{ return LumenLightingTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenLighting(work,ready); }); }
bool CancelLumenLightingUpdate(LumenLightingTask task)
{ return (LumenLightingTasks.Cancel(task) || Fail("Invalid Lumen lighting task")) && CollectLumenLighting(); }
void DestroyLumenLightingAsync()
{
    LumenLightingTasks.Visit([](auto& work) { if (work.InFlight) Send<void>(work.Completion,"waitUntilCompleted"); return true; });
    LumenLightingTasks.Clear();
}
bool ProbeLumenLighting(LumenLightingSliceWork& work,bool& ready)
{
    const auto status=Send<NSUInteger>(work.Completion,"status");
    if (status==CommandBufferStatusError) return Fail("Metal lighting slice failed: "+ErrorDescription(Send<ObjCObject>(work.Completion,"error")));
    constexpr NSUInteger CompletedStatus=4; ready=status==CompletedStatus; return true;
}
bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& output)
{
    if (!CollectLumenLighting()) return false;
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    if (!EnsureLumenSurfaceLighting(bool(upload.PreviousLighting),bool(occluders->second->Hardware))) return false;
    auto job=std::make_shared<LumenLightingSliceWork>();
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,job->Previous)) return false;
    if (!job->Prepare(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,LastError) || !LumenLightingTasks.CanSubmit(*job,LastError)) return false;
    const auto& input=job->Input;
    auto record=std::make_shared<LumenLightingRecord>(); record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{record->Surface->Identity,record->Distance->Info.Identity,upload.LightingRevision},record->Surface->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (job->Previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=job->Previous->Info.Bounce+1; }
    job->Record=record;
    job->Data=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",input.Words.data(),input.Words.size()*sizeof(std::uint32_t),0);
    record->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",NSUInteger(input.Frame.Trace.Counts.X)*sizeof(LumenLightingTexel),0);
    if (!job->Data || !record->Buffer || !Send<void*>(record->Buffer,"contents")) return Fail("Metal Lumen lighting allocation failed");
    const auto task=LumenLightingTasks.Insert(job); if (!task) return Fail("Lumen lighting task space exhausted"); output=task; return true;
}
bool SubmitLumenLightingSlice(LumenLightingSliceWork& work,const LumenSurfaceLightingFrame& frame)
{
    const auto& record=work.Record;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen lighting command buffer is unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen lighting encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",work.Record->Distance->Hardware?LumenHardwarePipelines[unsigned(work.Previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:work.Previous?LumenSurfaceRadiosity:LumenSurfaceLighting);
    BindLumenHardware(encoder,*work.Record->Distance); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Distance->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Data,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Buffer,0,3);
    if (work.Previous) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Previous->Buffer,0,4);
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record->Surface->Images[plane],plane);
    const auto groups=LumenLightingGroups(frame);
    const MTLSize grid{std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1},threads{LumenLightingThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    ReleaseObject(work.Completion); work.Completion=Send<ObjCObject>(command,"retain");
    work.InFlight=true; Send<void>(command,"commit"); return true;
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
