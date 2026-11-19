struct LumenGlobalWork : LumenGlobalDistanceJob
{
    std::shared_ptr<LumenDistanceRecord> Record,Previous;
    ObjCObject Dirty=nullptr,Completion=nullptr;
    ~LumenGlobalWork() { ReleaseObject(Dirty); ReleaseObject(Completion); }
};
LumenGlobalDistanceQueue<LumenGlobalWork> LumenGlobalTasks;
void DestroyLumenGlobalDistanceAsync()
{
    LumenGlobalTasks.Visit([](auto& work) { if (work.InFlight) Send<void>(work.Completion,"waitUntilCompleted"); return true; });
    LumenGlobalTasks.Clear();
}
bool ProbeLumenGlobalDistance(LumenGlobalWork& work,bool& ready)
{
    const auto status=Send<NSUInteger>(work.Completion,"status");
    if (status==CommandBufferStatusError) return Fail("Metal global distance update failed: "+ErrorDescription(Send<ObjCObject>(work.Completion,"error")));
    constexpr NSUInteger CompletedStatus=4;
    ready=status==CompletedStatus; return true;
}
bool CollectLumenGlobalDistance()
{ return LumenGlobalTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenGlobalDistance(work,ready); }); }
bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{ return (LumenGlobalTasks.Cancel(task) || Fail("Invalid global distance task")) && CollectLumenGlobalDistance(); }
bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& output)
{
    if (!CollectLumenGlobalDistance() || !EnsureLumenGlobalDistance()) return false;
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source");
    auto work=std::make_shared<LumenGlobalWork>();
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second) return Fail("Global distance reuse crosses source generations");
        work->Previous=old->second;
    }
    if (!work->Prepare(found->second->Info,settings,work->Previous?work->Previous->Info.Global.get():nullptr,LastError) || !LumenGlobalTasks.CanSubmit(*work,LastError)) return false;
    work->Record=std::make_shared<LumenDistanceRecord>(); auto& record=*work->Record;
    record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=work->Plan;
    record.Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(work->Plan->WordCount)*sizeof(std::uint32_t),0);
    if (!record.Buffer) return Fail("Allocate asynchronous Metal global field failed");
    auto* mapped=Send<std::uint32_t*>(record.Buffer,"contents"); if (!mapped) return Fail("Metal global header is not writable");
    std::memcpy(mapped+work->Plan->Frame.Header.Layout.W,&work->Plan->Frame.Header,sizeof(LumenGlobalDistanceHeader));
    work->Dirty=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::max<std::size_t>(1,work->Update.Compose.size())*sizeof(std::uint32_t),0);
    if (!work->Dirty) return Fail("Allocate Metal global dirty list failed");
    mapped=Send<std::uint32_t*>(work->Dirty,"contents"); if (!mapped) return Fail("Metal global dirty list is not writable");
    if (!work->Update.Compose.empty()) std::memcpy(mapped,work->Update.Compose.data(),work->Update.Compose.size()*sizeof(std::uint32_t));
    const auto task=LumenGlobalTasks.Insert(work); if (!task) return Fail("Global task space exhausted"); output=task; return true;
}
bool SubmitLumenGlobalDistanceSlice(LumenGlobalWork& work,const LumenGlobalDistanceFrame& frame)
{
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Create Metal global slice commands failed");
    auto& record=*work.Record;
    if (!work.Initialized)
    {
        const auto blit=Send<ObjCObject>(command,"blitCommandEncoder"); if (!blit) return Fail("Create Metal global slice blit failed");
        Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(blit,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            record.Base->Buffer,0,record.Buffer,0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t));
        for (const auto& copy:work.Update.Copies)
            Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(blit,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                work.Previous->Buffer,std::size_t(copy.SourceWord)*sizeof(std::uint32_t),record.Buffer,
                std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t));
        Send<void>(blit,"endEncoding");
    }
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal global slice encoder failed");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenGlobalCompose);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record.Base->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Dirty,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record.Buffer,0,3);
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const MTLSize grid{std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1},threads{LumenDistanceTraceThreads,1,1};
    if (groups) Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads);
    Send<void>(encoder,"endEncoding"); ReleaseObject(work.Completion); work.Completion=Send<ObjCObject>(command,"retain");
    work.InFlight=true; Send<void>(command,"commit"); return true;
}
bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    if (!CollectLumenGlobalDistance()) return false;
    return LumenGlobalTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenGlobalDistance(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenGlobalDistanceSlice(work,frame); },
        [&](auto& work,LumenDistanceSceneHandle& handle)
        {
            handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Global scene handle space exhausted");
            LumenDistanceScenes.emplace(handle.Value,work.Record); return true;
        });
}
