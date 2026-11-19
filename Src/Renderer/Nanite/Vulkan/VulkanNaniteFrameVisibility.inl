struct NaniteFrameBuffer
{
    Impl& Owner; RawBuffer Allocation; unsigned Words=0;
    explicit NaniteFrameBuffer(Impl& owner):Owner(owner) {}
    ~NaniteFrameBuffer() { Owner.DestroyRawBuffer(Allocation); }
};
struct NaniteDepthRecord { NaniteDepthInfo Info; std::shared_ptr<NaniteFrameBuffer> Data; };
struct NaniteVisibilityRecord
{
    NaniteFrameVisibilityInfo Info; std::uint64_t ConstantAlignmentPadding=0; NaniteVisibilityConstants Constants;
    std::shared_ptr<NaniteFrameBuffer> Queries,Data;
    std::shared_ptr<const std::vector<NaniteVisibilityQuery>> QueryRecords;
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
};
struct NaniteFrameCommandResources
{
    Impl& Owner; RawBuffer Uniform; VkDescriptorPool Pool=0;
    std::array<std::shared_ptr<NaniteFrameBuffer>,4> Buffers;
    explicit NaniteFrameCommandResources(Impl& owner):Owner(owner) {}
    ~NaniteFrameCommandResources()
    { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDepthRecord>> NaniteDepths;
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteVisibilityRecord>> NaniteDecisions;
NaniteVisibilityTransaction FrameNanite;
std::array<VkDescriptorSetLayout,7> NaniteFrameSets{};
std::array<VkPipelineLayout,7> NaniteFrameLayouts{};
std::array<VkPipeline,7> NaniteFrameShaders{};
VkSampler NaniteFrameSampler=0;
std::vector<std::shared_ptr<NaniteFrameCommandResources>> NaniteFrameWork;
std::shared_ptr<NaniteFrameCommandResources> NaniteFrameReadback;
std::shared_ptr<NaniteFrameBuffer> NaniteFrameDummy;
bool NaniteCommandsEnded() const { return FrameSubmitted || FrameCommandEnded; }
bool ValidateNaniteRecording(bool rasterInternal=false)
{ return (FrameOpen && (!PassOpen || (rasterInternal && CurrentPass.Type==PassType::GBuffer)) && !NaniteCommandsEnded()) || Fail("Nanite GPU work requires an open command stream between graphics passes"); }
void FinishNaniteFrame(bool publish)
{ FrameNanite.Finish(publish,NaniteDepths,NaniteDecisions); FinishNaniteDraws(); if (!FrameSubmitted || publish) NaniteFrameWork.clear(); }
bool RecoverNaniteFrameWork()
{
    if (FrameOpen || (NaniteFrameWork.empty() && !NaniteFrameReadback)) return true;
    if ((FrameSubmitted || NaniteFrameReadback) && !Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover Nanite frame commands")) return false;
    NaniteFrameWork.clear(); NaniteFrameReadback.reset(); return true;
}
struct NanitePresentPublication
{
    Impl& Owner; bool Published=false;
    ~NanitePresentPublication() { if (Published || !Owner.FrameOpen || Owner.NaniteCommandsEnded()) Owner.FinishNaniteFrame(Published); }
};
void DestroyNaniteFrameVisibility()
{
    FinishNaniteFrame(false); NaniteFrameReadback.reset(); NaniteFrameWork.clear(); NaniteDepths.clear(); NaniteDecisions.clear(); NaniteFrameDummy.reset();
    for (auto& shader:NaniteFrameShaders) { if (shader) DestroyPipelineFunction(Device,shader,nullptr); shader=0; }
    for (auto& layout:NaniteFrameLayouts) { if (layout) DestroyPipelineLayoutFunction(Device,layout,nullptr); layout=0; }
    for (auto& set:NaniteFrameSets) { if (set) DestroyDescriptorSetLayoutFunction(Device,set,nullptr); set=0; }
    if (NaniteFrameSampler) DestroySamplerFunction(Device,NaniteFrameSampler,nullptr); NaniteFrameSampler=0;
}
bool CreateNaniteFrameBuffer(unsigned words,const void* initial,bool writable,std::shared_ptr<NaniteFrameBuffer>& output,bool indirect=false,bool indexed=false)
{
    const auto bytes=std::size_t(words)*sizeof(unsigned); if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Nanite frame buffer exceeds Vulkan storage range");
    auto buffer=std::make_shared<NaniteFrameBuffer>(*this); buffer->Words=words;
    const VkFlags memory=writable?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkFlags usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|(writable?VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT:0)|(indirect?VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:0)|(indexed?VK_BUFFER_USAGE_INDEX_BUFFER_BIT:0);
    if (!CreateRawBuffer(bytes,usage,memory,buffer->Allocation)) return false;
    if (initial) std::memcpy(buffer->Allocation.Mapped,initial,bytes); output=std::move(buffer); return true;
}
bool EnsureNaniteFrameVisibility()
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<4 || DeviceLimits.maxDescriptorSetStorageBuffers<4 ||
        DeviceLimits.maxComputeWorkGroupInvocations<NaniteVisibilityThreads || DeviceLimits.maxComputeWorkGroupSize[0]<NaniteVisibilityThreads)
        return Fail("Vulkan Nanite frame visibility requires compute and four storage buffers");
    if (!LoadDevice(NaniteGeometryCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(NaniteGeometryDispatch,"vkCmdDispatch")) return false;
    if (!NaniteFrameDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameDummy)) return false; }
    if (!NaniteFrameSampler)
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; info.minFilter=info.magFilter=VK_FILTER_NEAREST;
        info.addressModeU=info.addressModeV=info.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; info.maxAnisotropy=1;
        if (!Check(CreateSamplerFunction(Device,&info,nullptr,&NaniteFrameSampler),"Create Nanite frame depth sampler")) return false;
    }
    constexpr unsigned stages[]{0,2,3,0};
    for (unsigned pass=0;pass<NaniteFrameShaders.size();++pass)
    {
        if (!NaniteFrameSets[pass])
        {
            VkDescriptorSetLayoutBinding bindings[5]{}; const unsigned count=pass>=2?5:3;
            for (unsigned slot=0;slot<count;++slot)
                bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(pass==1 && slot==1)?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; info.bindingCount=count; info.pBindings=bindings;
            if (!Check(CreateDescriptorSetLayoutFunction(Device,&info,nullptr,&NaniteFrameSets[pass]),"Create Nanite frame descriptor layout")) return false;
        }
        if (!NaniteFrameLayouts[pass])
        {
            VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; info.setLayoutCount=1; info.pSetLayouts=&NaniteFrameSets[pass];
            if (!Check(CreatePipelineLayoutFunction(Device,&info,nullptr,&NaniteFrameLayouts[pass]),"Create Nanite frame pipeline layout")) return false;
        }
        if (!NaniteFrameShaders[pass])
        {
            VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,pass>=4?VulkanNaniteBinShaders[pass-4]:pass==3?VulkanNaniteDraws:VulkanNaniteVisibilityShaders[stages[pass]],module,LastError)) return false;
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; info.layout=NaniteFrameLayouts[pass]; info.basePipelineIndex=-1;
            info.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
            const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&info,nullptr,&NaniteFrameShaders[pass]); DestroyShaderModuleFunction(Device,module,nullptr);
            if (!Check(status,"Create Nanite frame compute pipeline")) return false;
        }
    }
    return true;
}
bool DispatchNaniteFrame(unsigned pass,const NaniteVisibilityConstants& constants,unsigned groupsX,unsigned groupsY,
    const std::shared_ptr<NaniteFrameBuffer>& source,const std::shared_ptr<NaniteFrameBuffer>& depth,
    const std::shared_ptr<NaniteFrameBuffer>& main,const std::shared_ptr<NaniteFrameBuffer>& output,bool camera)
{
    if (groupsX>DeviceLimits.maxComputeWorkGroupCount[0] || groupsY>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Nanite frame dispatch exceeds Vulkan dimensions");
    auto work=std::make_shared<NaniteFrameCommandResources>(*this); work->Buffers={source,output,depth,main};
    if (!CreateRawBuffer(sizeof(constants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform)) return false;
    std::memcpy(work->Uniform.Mapped,&constants,sizeof(constants));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Nanite frame descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&NaniteFrameSets[pass];
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate Nanite frame descriptors")) return false;
    const unsigned count=pass>=2?5:3; VkDescriptorBufferInfo buffers[5]{}; VkWriteDescriptorSet writes[5]{};
    buffers[0]={work->Uniform.Buffer,0,sizeof(constants)};
    for (unsigned slot=1;slot<count;++slot)
    { const auto& buffer=work->Buffers[slot-1]; if (buffer) buffers[slot]={buffer->Allocation.Buffer,0,buffer->Allocation.Size}; }
    const VkDescriptorImageInfo image{NaniteFrameSampler,MainDepth.View,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    for (unsigned slot=0;slot<count;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=set; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(camera && slot==1)?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (camera && slot==1) write.pImageInfo=&image; else write.pBufferInfo=&buffers[slot];
    }
    UpdateDescriptorSets(Device,count,writes,0,nullptr); NaniteFrameWork.push_back(work);
    const auto original=MainDepth.Layout; if (camera) TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; host.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; host.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
    host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; host.buffer=work->Uniform.Buffer; host.size=sizeof(constants);
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&host,0,nullptr);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteFrameLayouts[pass],0,1,&set,0,nullptr); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteFrameShaders[pass]);
    NaniteGeometryDispatch(CommandBuffer,groupsX,groupsY,1);
    VkBufferMemoryBarrier ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; ready.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; ready.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    ready.srcQueueFamilyIndex=ready.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; ready.buffer=output->Allocation.Buffer; ready.size=output->Allocation.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&ready,0,nullptr);
    if (camera) TransitionImage(CommandBuffer,MainDepth,original); return true;
}
bool CaptureNaniteDepth(NaniteDepthHandle& output,bool rasterInternal=false)
{
    NaniteVisibilityWork work;
    if (!ValidateNaniteRecording(rasterInternal) || FrameSampleCount!=1 || !MainDepth.View || !NanitePrimaryDepthInitialized(PendingFrame.Passes)) return Fail("Nanite depth capture requires initialized single-sample camera depth");
    if (!BuildNaniteDepthWork(CreateInfo.Width,CreateInfo.Height,work,LastError) || !FrameNanite.CanAllocate(std::size_t(work.OutputWords)*sizeof(unsigned),true,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteDepthRecord>();
    if (!CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    record->Info={FrameNanite.Frame(),unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),work.OutputWords,false};
    for (const auto& job:work.Dispatches)
        if (!DispatchNaniteFrame(job.Pass==2?1:0,job.Constants,job.GroupsX,job.GroupsY,NaniteFrameDummy,nullptr,nullptr,record->Data,job.Pass==2)) return false;
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
    if (!DispatchNaniteFrame(2,record->Constants,job.GroupsX,job.GroupsY,record->Queries,depth?depth->Data:NaniteFrameDummy,NaniteFrameDummy,record->Data,false)) return false;
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
    if (!DispatchNaniteFrame(2,record->Constants,gx,(groups+gx-1)/gx,record->Queries,d->second->Data,m->second->Data,record->Data,false)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}

bool ReadNaniteFrameBuffer(const std::shared_ptr<NaniteFrameBuffer>& buffer,std::vector<unsigned>& words)
{
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Read Nanite frame feedback") || !RecoverNaniteFrameWork()) return false;
    if (!LoadDevice(NaniteCopyBuffer,"vkCmdCopyBuffer")) return false;
    auto work=std::make_shared<NaniteFrameCommandResources>(*this); work->Buffers[0]=buffer;
    const auto bytes=std::size_t(buffer->Words)*sizeof(unsigned);
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform) || !BeginImmediate()) return false;
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=buffer->Allocation.Buffer; barrier.size=bytes;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    const VkBufferCopy copy{0,0,bytes}; NaniteCopyBuffer(CommandBuffer,buffer->Allocation.Buffer,work->Uniform.Buffer,1,&copy);
    barrier.buffer=work->Uniform.Buffer; barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    NaniteFrameReadback=work; if (!EndImmediate()) return false; NaniteFrameReadback.reset();
    words.resize(buffer->Words); std::memcpy(words.data(),work->Uniform.Mapped,bytes); return true;
}
