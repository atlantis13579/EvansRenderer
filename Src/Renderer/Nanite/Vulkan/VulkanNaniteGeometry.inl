// Each dispatch has an aligned immutable uniform slice. Prefix and decode use
// separate submissions; scratch/descriptors survive unsuccessful fence waits.
// Coherent native VB/IB allocations become drawable only after host/status checks.
VkDescriptorSetLayout NaniteGeometrySetLayout=0;
VkPipelineLayout NaniteGeometryLayout=0;
std::array<VkPipeline,5> NaniteGeometryPipelines{};
std::array<RawBuffer,7> NaniteGeometryScratch{};
std::shared_ptr<NaniteCachePool> NaniteBorrowedPool;
VkDescriptorPool NaniteGeometryPool=0;
PFN_vkCreateComputePipelines NaniteGeometryCreatePipelines=nullptr;
PFN_vkCmdDispatch NaniteGeometryDispatch=nullptr;
static constexpr std::size_t NaniteState=4,NaniteVertices=5,NaniteIndices=6,NaniteStorageCount=6;
void DestroyNaniteGeometryScratch()
{
    if (NaniteGeometryPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,NaniteGeometryPool,nullptr);
    NaniteGeometryPool=0;
    for (std::size_t slot=0;slot<NaniteGeometryScratch.size();++slot)
        if (slot==3 && NaniteBorrowedPool) NaniteGeometryScratch[slot]={}; else DestroyRawBuffer(NaniteGeometryScratch[slot]);
    NaniteBorrowedPool.reset();
}
void DestroyNaniteGeometry()
{
    DestroyNaniteGeometryScratch();
    DestroyNaniteAsync(); DestroyNaniteCacheAsync(); DestroyNaniteCaches();
    for (auto& pipeline:NaniteGeometryPipelines) { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (NaniteGeometryLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,NaniteGeometryLayout,nullptr);
    if (NaniteGeometrySetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,NaniteGeometrySetLayout,nullptr);
    NaniteGeometryLayout=0; NaniteGeometrySetLayout=0;
}
bool SupportsNaniteGeometry() const
{
    return GraphicsQueueHasCompute && DeviceLimits.maxPerStageDescriptorStorageBuffers>=NaniteStorageCount && DeviceLimits.maxDescriptorSetStorageBuffers>=NaniteStorageCount &&
        DeviceLimits.maxPerStageResources>=NaniteStorageCount+1 && DeviceLimits.maxUniformBufferRange>=sizeof(NaniteGpuConstants) &&
        DeviceLimits.maxComputeWorkGroupInvocations>=NaniteGpuWidth && DeviceLimits.maxComputeWorkGroupSize[0]>=NaniteGpuWidth &&
        DeviceLimits.maxComputeSharedMemorySize>=(NaniteGpuWidth+1)*sizeof(ShaderUInt4);
}
bool EnsureNaniteGeometry()
{
    if (!SupportsNaniteGeometry()) return Fail("Vulkan Nanite geometry requires compute, shared scan memory and six storage descriptors");
    if (!LoadDevice(NaniteGeometryCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(NaniteGeometryDispatch,"vkCmdDispatch")) return false;
    if (!NaniteGeometrySetLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,7> bindings{};
        for (std::uint32_t slot=0;slot<bindings.size();++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; info.bindingCount=static_cast<unsigned>(bindings.size()); info.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&info,nullptr,&NaniteGeometrySetLayout),"Create Nanite geometry descriptor layout")) return false;
    }
    if (!NaniteGeometryLayout)
    {
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; info.setLayoutCount=1; info.pSetLayouts=&NaniteGeometrySetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&info,nullptr,&NaniteGeometryLayout),"Create Nanite geometry pipeline layout")) return false;
    }
    for (std::size_t stage=0;stage<NaniteGeometryPipelines.size();++stage) if (!NaniteGeometryPipelines[stage])
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteGeometryShaders[stage],module,LastError)) return false;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; info.layout=NaniteGeometryLayout; info.basePipelineIndex=-1;
        info.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto result=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&info,nullptr,&NaniteGeometryPipelines[stage]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(result,"Create Nanite geometry compute pipeline")) return false;
    }
    return true;
}
bool CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output,NaniteCacheHandle cache={})
{
    std::vector<NaniteGpuDispatch> jobs;
    if (!BuildNaniteGpuSchedule(upload,jobs,LastError) || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { const auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Vulkan Nanite cache handle is invalid"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    for (const auto& job:jobs) if (job.GroupsX>DeviceLimits.maxComputeWorkGroupCount[0] || job.GroupsY>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Nanite geometry exceeds Vulkan workgroup dimensions");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover Nanite geometry scratch")) return false;
    DestroyNaniteGeometryScratch();
    const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const VkDeviceSize uniformStride=(sizeof(NaniteGpuConstants)+alignment-1)/alignment*alignment;
    const std::size_t stateBytes=upload.InitialState.size()*sizeof(std::uint32_t);
    const void* data[]{nullptr,upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{static_cast<std::size_t>(uniformStride*jobs.size()),upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),stateBytes,sizeof(std::uint32_t),sizeof(std::uint32_t)};
    const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::size_t slot=0;slot<NaniteGeometryScratch.size();++slot)
    {
        if (slot==3 && cached) { NaniteBorrowedPool=cached->Pool; NaniteGeometryScratch[slot]=*cached->Pool; continue; }
        if (slot && sizes[slot]>DeviceLimits.maxStorageBufferRange) return Fail("Nanite geometry storage exceeds Vulkan range");
        VkFlags usage=slot?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (slot==NaniteVertices) usage|=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; if (slot==NaniteIndices) usage|=VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (!CreateRawBuffer(sizes[slot],usage,host,NaniteGeometryScratch[slot])) return false;
        if (data[slot]) std::memcpy(NaniteGeometryScratch[slot].Mapped,data[slot],sizes[slot]);
    }
    for (std::size_t stage=0;stage<jobs.size();++stage) std::memcpy(static_cast<std::uint8_t*>(NaniteGeometryScratch[0].Mapped)+stage*uniformStride,&jobs[stage].Constants,sizeof(NaniteGpuConstants));
    const auto count=static_cast<unsigned>(jobs.size());
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,count},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count*static_cast<unsigned>(NaniteStorageCount)}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=count; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&NaniteGeometryPool),"Create Nanite descriptor pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(count,NaniteGeometrySetLayout); std::vector<VkDescriptorSet> sets(count);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=NaniteGeometryPool; allocation.descriptorSetCount=count; allocation.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocation,sets.data()),"Allocate Nanite descriptor sets")) return false;
    auto update=[&](unsigned stage)
    {
        std::array<VkDescriptorBufferInfo,7> buffers{}; std::array<VkWriteDescriptorSet,7> writes{};
        for (unsigned slot=0;slot<writes.size();++slot)
        {
            buffers[slot]={NaniteGeometryScratch[slot].Buffer,slot?0:stage*uniformStride,slot?NaniteGeometryScratch[slot].Size:sizeof(NaniteGpuConstants)};
            auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=sets[stage]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[slot];
        }
        UpdateDescriptorSets(Device,static_cast<unsigned>(writes.size()),writes.data(),0,nullptr);
    };
    for (unsigned stage=0;stage<count;++stage) update(stage);
    auto dispatch=[&](unsigned stage)
    {
        const auto& job=jobs[stage]; CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteGeometryLayout,0,1,&sets[stage],0,nullptr);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteGeometryPipelines[static_cast<unsigned>(job.Pass)]); NaniteGeometryDispatch(CommandBuffer,job.GroupsX,job.GroupsY,1);
    };
    auto stateBarrier=[&](VkFlags stage,VkFlags access)
    {
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=access;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=NaniteGeometryScratch[NaniteState].Buffer; barrier.size=stateBytes;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,stage,0,0,nullptr,1,&barrier,0,nullptr);
    };
    if (!BeginImmediate()) return false;
    for (unsigned stage=0;stage+1<count;++stage) { dispatch(stage); stateBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT); }
    stateBarrier(VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_HOST_READ_BIT|VK_ACCESS_SHADER_READ_BIT);
    if (!EndImmediate()) return false;
    NaniteGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size()); std::memcpy(candidate.Readback.State.data(),NaniteGeometryScratch[NaniteState].Mapped,stateBytes);
    if (!ValidateNaniteGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (vertexBytes>DeviceLimits.maxStorageBufferRange || indexBytes>DeviceLimits.maxStorageBufferRange) return Fail("Nanite drawable streams exceed Vulkan storage range");
    DestroyRawBuffer(NaniteGeometryScratch[NaniteVertices]); DestroyRawBuffer(NaniteGeometryScratch[NaniteIndices]);
    if (!CreateRawBuffer(vertexBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,host,NaniteGeometryScratch[NaniteVertices]) ||
        !CreateRawBuffer(indexBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT,host,NaniteGeometryScratch[NaniteIndices])) return false;
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount;
    std::memcpy(static_cast<std::uint8_t*>(NaniteGeometryScratch[0].Mapped)+(count-1)*uniformStride,&constants,sizeof(constants)); update(count-1);
    if (!BeginImmediate()) return false; dispatch(count-1);
    std::array<VkBufferMemoryBarrier,3> visible{};
    for (std::size_t stream=0;stream<visible.size();++stream)
    {
        auto& barrier=visible[stream]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT|(stream==1?VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:stream==2?VK_ACCESS_INDEX_READ_BIT:0);
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=NaniteGeometryScratch[NaniteState+stream].Buffer; barrier.size=NaniteGeometryScratch[NaniteState+stream].Size;
    }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,static_cast<unsigned>(visible.size()),visible.data(),0,nullptr);
    if (!EndImmediate()) return false;
    candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    std::memcpy(candidate.Readback.State.data(),NaniteGeometryScratch[NaniteState].Mapped,stateBytes); std::memcpy(candidate.Readback.Vertices.data(),NaniteGeometryScratch[NaniteVertices].Mapped,vertexBytes); std::memcpy(candidate.Readback.Indices.data(),NaniteGeometryScratch[NaniteIndices].Mapped,indexBytes);
    if (!ValidateNaniteGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    { auto& resource=resources[stream]; resource.NaniteGeometry=true; resource.Allocation=NaniteGeometryScratch[NaniteVertices+stream]; resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size()); }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=count;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    NaniteGeometryScratch[NaniteVertices]={}; NaniteGeometryScratch[NaniteIndices]={}; output=std::move(candidate); DestroyNaniteGeometryScratch(); return true;
}
