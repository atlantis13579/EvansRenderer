// Native argument storage is retained by frame command ownership. Logical draw
// batches expire together at presentation/abort; geometry is locked meanwhile.
struct NaniteDrawRecord
{
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
    std::shared_ptr<NaniteFrameBuffer> Ranges,Arguments,SourceIndices;
    std::unique_ptr<NaniteBinLayout> Binning;
    std::vector<bool> Submitted;
    std::uint64_t Visibility=0;
    NaniteVisibilityPhase Phase=NaniteVisibilityPhase::Main;
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDrawRecord>> NaniteDrawBatches;
NaniteDrawRecord* ActiveNaniteDraw=nullptr;
unsigned ActiveNaniteQuery=0,ActiveNaniteBin=0;
std::vector<BufferHandle> NaniteCopyLocks;
unsigned NaniteArgumentOffset() const
{ return unsigned((ActiveNaniteDraw->Binning?ActiveNaniteDraw->Binning->Constants.Source.W+ActiveNaniteBin*NaniteDrawArgumentWords:ActiveNaniteQuery*NaniteDrawArgumentWords)*sizeof(unsigned)); }
std::size_t NaniteDrawBytes=0;
void FinishNaniteDraws()
{
    ActiveNaniteDraw=nullptr; NaniteDrawBatches.clear(); NaniteDrawBytes=0; NaniteCopyLocks.clear();
    // Published decisions support diagnostic reads, never another frame's draw
    // preparation. Release CPU descriptors and queries after their last consumer.
    for (auto& item:NaniteDecisions) if (item.second->Info.Published)
    { item.second->QueryRecords.reset(); item.second->Draws.reset(); item.second->Queries.reset(); }
}
bool NaniteBufferLocked(BufferHandle buffer) const
{
    if (std::find(NaniteCopyLocks.begin(),NaniteCopyLocks.end(),buffer)!=NaniteCopyLocks.end()) return true;
    for (const auto& pair:NaniteDrawBatches) for (const auto& draw:*pair.second->Draws)
        if (draw.VertexBuffer==buffer || draw.IndexBuffer==buffer) return true;
    return false;
}
bool NaniteMainDrawsComplete(const std::shared_ptr<const std::vector<NaniteClusterDraw>>& draws) const
{
    for (const auto& pair:NaniteDrawBatches) if (pair.second->Draws==draws && pair.second->Phase==NaniteVisibilityPhase::Main)
        return std::all_of(pair.second->Submitted.begin(),pair.second->Submitted.end(),[](bool value){return value;});
    return false;
}
bool PrepareNaniteBinCommands(NaniteDrawRecord& record,const NaniteVisibilityRecord& visibility)
{
    const auto& layout=*record.Binning;
    if (!CreateNaniteFrameBuffer(unsigned(layout.Metadata.size()),layout.Metadata.data(),false,record.Ranges) ||
        !CreateNaniteFrameBuffer(layout.SourceIndexWords,nullptr,true,record.SourceIndices) ||
        !CreateNaniteFrameBuffer(layout.OutputWords,nullptr,true,record.Arguments)) return false;
    NaniteCopyLocks.reserve(NaniteCopyLocks.size()+layout.Bins.size());
    for (const auto& bin:layout.Bins) NaniteCopyLocks.push_back((*record.Draws)[bin.FirstDraw].IndexBuffer);
    ObjCObject encoder=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!encoder) return Fail("Create Nanite bin copy encoder failed");
    for (const auto& bin:layout.Bins)
    {
        const auto buffer=Buffers.at((*record.Draws)[bin.FirstDraw].IndexBuffer.Value).Buffer;
        Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(encoder,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",buffer,NSUInteger(bin.FirstIndex)*sizeof(unsigned),record.SourceIndices->Buffer,NSUInteger(bin.OutputIndex)*sizeof(unsigned),NSUInteger(bin.IndexCount)*sizeof(unsigned));
    }
    Send<void>(encoder,"endEncoding");
    auto constants=layout.Constants; constants.Counts.W=unsigned(record.Phase);
    const unsigned jobs[]{constants.Counts.Y,constants.Counts.Z,constants.Counts.X};
    for (unsigned pass=0;pass<3;++pass)
    {
        constants.Target.Z=jobs[pass]; constants.Target.W=std::min(jobs[pass],NaniteDrawDispatchColumns);
        const unsigned rows=(jobs[pass]+constants.Target.W-1)/constants.Target.W;
        if (!DispatchNaniteFrame(4+pass,constants,constants.Target.W,rows,record.Ranges->Buffer,visibility.Data->Buffer,record.SourceIndices->Buffer,record.Arguments->Buffer,nullptr)) return false;
    }

    return true;
}
bool PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output,bool binned=false)
{
    if (!ValidateNaniteRecording() || FrameSampleCount!=1) return Fail("Nanite draws require a single-sample frame between passes");
    const auto found=NaniteDecisions.find(visibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Frame!=FrameNanite.CurrentFrame() ||
        found->second->Info.QueryCount!=draws.size()) return Fail("Nanite draws require matching current-frame visibility");
    for (const auto& batch:NaniteDrawBatches) if (batch.second->Visibility==visibility.Value) return Fail("Nanite visibility already has draw commands");
    const auto& visibilityRecord=*found->second;
    const bool post=visibilityRecord.Info.Phase==NaniteVisibilityPhase::Post;
    if ((!post && visibilityRecord.Draws) || (post && !visibilityRecord.Draws)) return Fail("Nanite main descriptors must be prepared once before retesting");
    auto bytes=draws.size()*(sizeof(NaniteClusterDraw)+sizeof(unsigned)*(NaniteDrawRangeWords+NaniteDrawArgumentWords)+sizeof(bool));
    if (NaniteDrawBatches.size()>=NaniteMaximumDrawBatches || bytes>NaniteMaximumDrawBytes-NaniteDrawBytes) return Fail("Nanite draw frame budget exceeded");
    std::unique_ptr<NaniteBinLayout> binning;
    if (binned)
    {
        binning=std::make_unique<NaniteBinLayout>(); if (!BuildNaniteDrawBins(draws,*binning,LastError)) return false;
        bytes+=(binning->Metadata.size()+binning->OutputWords+binning->SourceIndexWords)*sizeof(unsigned);
        if (bytes>NaniteMaximumDrawBytes-NaniteDrawBytes) return Fail("Nanite compacted index stream exceeds the frame budget");
    }
    std::vector<unsigned> ranges; ranges.reserve(draws.size()*NaniteDrawRangeWords);
    for (unsigned i=0;i<draws.size();++i)
    {
        const auto& draw=draws[i];
        if (post && !SameNaniteDraw(draw,(*visibilityRecord.Draws)[i])) return Fail("Nanite post draws must preserve main geometry, transforms and materials");
        const auto vertices=Buffers.find(draw.VertexBuffer.Value),indices=Buffers.find(draw.IndexBuffer.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Nanite draw geometry handle is invalid");
        if (!post && !ValidateNaniteClusterDraw(draw,(*visibilityRecord.QueryRecords)[i],vertices->second.Description,vertices->second.Bytes,
            indices->second.Description,indices->second.Bytes,LastError)) return false;
        ranges.push_back(draw.FirstIndex); ranges.push_back(draw.IndexCount);
    }
    if (!EnsureNaniteFrameVisibility()) return false;

    auto record=std::make_shared<NaniteDrawRecord>(); record->Phase=visibilityRecord.Info.Phase; record->Visibility=visibility.Value;
    record->Draws=post?visibilityRecord.Draws:std::make_shared<const std::vector<NaniteClusterDraw>>(draws);
    record->Submitted.resize(draws.size(),false); record->Binning=std::move(binning);
    if (binned) { if (!PrepareNaniteBinCommands(*record,visibilityRecord)) return false; }
    else
    {
        if (!CreateNaniteFrameBuffer(unsigned(ranges.size()),ranges.data(),false,record->Ranges) ||
            !CreateNaniteFrameBuffer(unsigned(draws.size())*NaniteDrawArgumentWords,nullptr,true,record->Arguments)) return false;
        NaniteVisibilityConstants constants{}; constants.Counts.X=unsigned(draws.size()); constants.Counts.W=unsigned(record->Phase);
        const unsigned groups=(constants.Counts.X+NaniteVisibilityThreads-1)/NaniteVisibilityThreads;
        constants.Target.W=std::min(groups,NaniteDrawDispatchColumns); const unsigned rows=(groups+constants.Target.W-1)/constants.Target.W;
        if (!DispatchNaniteFrame(3,constants,constants.Target.W,rows,record->Ranges->Buffer,visibilityRecord.Data->Buffer,nullptr,record->Arguments->Buffer,nullptr)) return false;
    }
    const NaniteDrawHandle handle{NewNaniteVisibilityIdentity()}; NaniteDrawBatches.emplace(handle.Value,record);
    found->second->Draws=record->Draws; NaniteDrawBytes+=bytes; PendingFrame.Statistics.NaniteDrawDispatchCount+=binned?3:1; output=handle; return true;
}
