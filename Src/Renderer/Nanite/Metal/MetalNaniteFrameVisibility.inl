struct NaniteFrameBuffer
{
    ObjCObject Buffer=nullptr; unsigned Words=0;
    ~NaniteFrameBuffer() { ReleaseObject(Buffer); }
};
struct NaniteDepthRecord { NaniteDepthInfo Info; std::shared_ptr<NaniteFrameBuffer> Data; };
struct NaniteVisibilityRecord
{
    NaniteFrameVisibilityInfo Info; std::uint64_t ConstantAlignmentPadding=0; NaniteVisibilityConstants Constants;
    std::shared_ptr<NaniteFrameBuffer> Queries,Data;
    std::shared_ptr<const std::vector<NaniteVisibilityQuery>> QueryRecords;
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDepthRecord>> NaniteDepths;
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteVisibilityRecord>> NaniteDecisions;
NaniteVisibilityTransaction FrameNanite;
std::array<ObjCObject,7> NaniteFrameShaders{};
std::shared_ptr<NaniteFrameBuffer> NaniteFrameDummy;
bool NaniteCommandsEnded() const
{ constexpr NSUInteger CommittedStatus=2; return !CommandBuffer || Send<NSUInteger>(CommandBuffer,"status")>=CommittedStatus; }
bool ValidateNaniteRecording(bool rasterInternal=false)
{ return (FrameOpen && (!PassOpen || (rasterInternal && CurrentPass.Type==PassType::GBuffer)) && !NaniteCommandsEnded()) || Fail("Nanite GPU work requires an open command stream between graphics passes"); }
void FinishNaniteFrame(bool publish) { FrameNanite.Finish(publish,NaniteDepths,NaniteDecisions); FinishNaniteDraws(); }
struct NanitePresentPublication
{
    Impl& Owner; bool Published=false;
    ~NanitePresentPublication() { if (Published || !Owner.FrameOpen || Owner.NaniteCommandsEnded()) Owner.FinishNaniteFrame(Published); }
};
void DestroyNaniteFrameVisibility()
{
    FinishNaniteFrame(false); NaniteDepths.clear(); NaniteDecisions.clear(); NaniteFrameDummy.reset();
    for (auto& shader:NaniteFrameShaders) { ReleaseObject(shader); shader=nullptr; }
}
bool CreateNaniteFrameBuffer(unsigned words,const void* initial,bool,std::shared_ptr<NaniteFrameBuffer>& output)
{
    auto buffer=std::make_shared<NaniteFrameBuffer>(); buffer->Words=words;
    buffer->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(words)*sizeof(unsigned),0);
    if (!buffer->Buffer) return Fail("Metal Nanite frame buffer allocation failed");
    if (initial)
    { void* mapped=Send<void*>(buffer->Buffer,"contents"); if (!mapped) return Fail("Metal Nanite input is not host visible"); std::memcpy(mapped,initial,std::size_t(words)*sizeof(unsigned)); }
    output=std::move(buffer); return true;
}
bool EnsureNaniteFrameVisibility()
{
    if (!NaniteFrameDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameDummy)) return false; }
    if (std::all_of(NaniteFrameShaders.begin(),NaniteFrameShaders.end(),[](auto p){return p!=nullptr;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteVisibility.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr; Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Compile Metal Nanite frame visibility: "+ErrorDescription(error));
    constexpr const char* entries[]{"EvansNaniteReduce","EvansNaniteCapture","EvansNaniteFrameCull","EvansNaniteDraws","EvansNaniteBinCount","EvansNaniteBinScan","EvansNaniteBinScatter"};
    bool drawLibrary=false;
    for (unsigned pass=0;pass<NaniteFrameShaders.size();++pass) if (!NaniteFrameShaders[pass])
    {
        if (pass>=3 && !drawLibrary)
        {
            ReleaseObject(library); library=nullptr; drawLibrary=true;
            if (!LoadTechniqueShaderSource("Metal","NaniteDraws.metal",source,LastError)) return false;
            options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
            library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
            ReleaseObject(options); if (!library) return Fail("Compile Metal Nanite draws: "+ErrorDescription(error));
        }
        ObjCObject function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[pass]));
        if (function) NaniteFrameShaders[pass]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!NaniteFrameShaders[pass]) { ReleaseObject(library); return Fail("Create Metal Nanite frame pipeline: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(NaniteFrameShaders[pass],"maxTotalThreadsPerThreadgroup")<NaniteVisibilityThreads)
        { ReleaseObject(NaniteFrameShaders[pass]); NaniteFrameShaders[pass]=nullptr; ReleaseObject(library); return Fail("Metal Nanite frame workgroup exceeds device capacity"); }
    }
    ReleaseObject(library); return true;
}
bool DispatchNaniteFrame(unsigned pass,const NaniteVisibilityConstants& constants,unsigned groupsX,unsigned groupsY,
    ObjCObject source,ObjCObject depth,ObjCObject main,ObjCObject output,ObjCObject camera)
{
    ObjCObject encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Metal Nanite frame encoder creation failed");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",NaniteFrameShaders[pass]);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
    const ObjCObject buffers[]{source,output,depth?depth:NaniteFrameDummy->Buffer,main?main:NaniteFrameDummy->Buffer};
    for (NSUInteger slot=0;slot<4;++slot) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",buffers[slot],0,slot+1);
    if (camera) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",camera,0);
    const MTLSize groups{groupsX,groupsY,1},threads{NaniteVisibilityThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding"); return true;
}
bool CaptureNaniteDepth(NaniteDepthHandle& output,bool rasterInternal=false)
{
    NaniteVisibilityWork work;
    if (!ValidateNaniteRecording(rasterInternal) || FrameSampleCount!=1 || !DepthTexture || !NanitePrimaryDepthInitialized(PendingFrame.Passes)) return Fail("Nanite depth capture requires initialized single-sample camera depth");
    if (!BuildNaniteDepthWork(CreateInfo.Width,CreateInfo.Height,work,LastError) || !FrameNanite.CanAllocate(std::size_t(work.OutputWords)*sizeof(unsigned),true,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteDepthRecord>();
    if (!CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    record->Info={FrameNanite.Frame(),unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),work.OutputWords,false};
    for (const auto& job:work.Dispatches)
        if (!DispatchNaniteFrame(job.Pass==2?1:0,job.Constants,job.GroupsX,job.GroupsY,NaniteFrameDummy->Buffer,nullptr,nullptr,record->Data->Buffer,job.Pass==2?DepthTexture:nullptr)) return false;
    NaniteDepthHandle handle{NewNaniteVisibilityIdentity()}; NaniteDepths.emplace(handle.Value,record); FrameNanite.Record(handle,std::size_t(work.OutputWords)*sizeof(unsigned)); output=handle; return true;
}
bool BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!ValidateNaniteRecording()) return false;
    std::shared_ptr<NaniteDepthRecord> depth;
    if (history) { const auto found=NaniteDepths.find(history.Value); if (found==NaniteDepths.end() || !found->second->Info.Published) return Fail("Nanite history must be a committed depth snapshot"); depth=found->second; }
    NaniteVisibilityWork work; if (!BuildNaniteFrameCull(queries,depth?depth->Info.Width:CreateInfo.Width,depth?depth->Info.Height:CreateInfo.Height,NaniteVisibilityPhase::Main,work,LastError)) return false;
    const auto bytes=(work.Input.size()+work.OutputWords)*sizeof(unsigned)+queries.size()*sizeof(NaniteVisibilityQuery);
    if (!FrameNanite.CanAllocate(bytes,false,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>();
    if (!CreateNaniteFrameBuffer(unsigned(work.Input.size()),work.Input.data(),false,record->Queries) || !CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    const auto& job=work.Dispatches.front(); record->Constants=job.Constants; record->Constants.Layout.W=depth?1:0;
    record->Info={FrameNanite.Frame(),work.OutputWords,NaniteVisibilityPhase::Main,false};
    record->QueryRecords=std::make_shared<const std::vector<NaniteVisibilityQuery>>(queries);
    if (!DispatchNaniteFrame(2,record->Constants,job.GroupsX,job.GroupsY,record->Queries->Buffer,depth?depth->Data->Buffer:NaniteFrameDummy->Buffer,NaniteFrameDummy->Buffer,record->Data->Buffer,nullptr)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}
bool RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle camera,NaniteVisibilityHandle& output,bool rasterInternal=false)
{
    if (!ValidateNaniteRecording(rasterInternal)) return false;
    const auto m=NaniteDecisions.find(main.Value); const auto d=NaniteDepths.find(camera.Value);
    if (m==NaniteDecisions.end() || d==NaniteDepths.end() || m->second->Info.Frame!=FrameNanite.CurrentFrame() || d->second->Info.Frame!=FrameNanite.CurrentFrame() ||
        m->second->Info.Phase!=NaniteVisibilityPhase::Main || m->second->Info.Published || d->second->Info.Published) return Fail("Nanite retest requires this frame's main decisions and camera depth");
    if (m->second->Draws && !NaniteMainDrawsComplete(m->second->Draws)) return Fail("Submit all main Nanite candidates before retesting");
    const auto bytes=std::size_t(m->second->Info.QueryCount)*sizeof(unsigned); if (!FrameNanite.CanAllocate(bytes,false,LastError)) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>(); record->Info=m->second->Info; record->Info.Phase=NaniteVisibilityPhase::Post; record->Queries=m->second->Queries; record->QueryRecords=m->second->QueryRecords; record->Draws=m->second->Draws;
    if (!CreateNaniteFrameBuffer(record->Info.QueryCount,nullptr,true,record->Data)) return false;
    record->Constants=m->second->Constants; record->Constants.Counts.Y=d->second->Info.Width; record->Constants.Counts.Z=d->second->Info.Height; record->Constants.Counts.W=1; record->Constants.Layout.W=1;
    const unsigned groups=(record->Info.QueryCount+NaniteVisibilityThreads-1)/NaniteVisibilityThreads,gx=record->Constants.Target.W;
    if (!DispatchNaniteFrame(2,record->Constants,gx,(groups+gx-1)/gx,record->Queries->Buffer,d->second->Data->Buffer,m->second->Data->Buffer,record->Data->Buffer,nullptr)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}

bool ReadNaniteFrameBuffer(const NaniteFrameBuffer& buffer,std::vector<unsigned>& words)
{
    // Present waits for this frame's command buffer before publishing handles.
    const void* mapped=Send<void*>(buffer.Buffer,"contents"); if (!mapped) return Fail("Metal Nanite feedback is not host visible");
    words.resize(buffer.Words); std::memcpy(words.data(),mapped,std::size_t(buffer.Words)*sizeof(unsigned)); return true;
}
