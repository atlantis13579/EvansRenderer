// Host-coherent storage supplies immutable inputs and the authoritative CPU
// mirror. Explicit compute and vertex/host barriers precede publication. Scratch
// remains owned after a failed wait until the frame fence can be recovered.
VkDescriptorSetLayout DisplacementSetLayout=0;
VkPipelineLayout DisplacementLayout=0;
std::array<VkPipeline,2> DisplacementPipelines{};
std::array<RawBuffer,7> DisplacementScratch{};
VkDescriptorPool DisplacementPool=0;
PFN_vkCreateComputePipelines DisplacementCreatePipelines=nullptr;
PFN_vkCmdDispatch DisplacementDispatch=nullptr;
static constexpr std::size_t DisplacementOutput=6,DisplacementStorageCount=6;
void DestroyDisplacementScratch()
{
    if (DisplacementPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,DisplacementPool,nullptr);
    DisplacementPool=0; for (auto& buffer:DisplacementScratch) DestroyRawBuffer(buffer);
}
void DestroyDisplacement()
{
    DestroyDisplacementScratch();
    for (auto& pipeline:DisplacementPipelines) { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (DisplacementLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,DisplacementLayout,nullptr);
    if (DisplacementSetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,DisplacementSetLayout,nullptr);
    DisplacementLayout=0; DisplacementSetLayout=0;
}
bool SupportsDisplacement() const
{
    return GraphicsQueueHasCompute && DeviceLimits.maxPerStageDescriptorStorageBuffers>=DisplacementStorageCount &&
        DeviceLimits.maxDescriptorSetStorageBuffers>=DisplacementStorageCount && DeviceLimits.maxPerStageResources>=DisplacementStorageCount+1 &&
        DeviceLimits.maxComputeWorkGroupInvocations>=DisplacementThreadCount && DeviceLimits.maxComputeWorkGroupSize[0]>=DisplacementThreadCount;
}
bool EnsureDisplacement()
{
    if (!SupportsDisplacement()) return Fail("Vulkan displacement requires compute and six storage-buffer descriptors");
    if (!LoadDevice(DisplacementCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(DisplacementDispatch,"vkCmdDispatch")) return false;
    if (!DisplacementSetLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,7> bindings{};
        for (std::uint32_t slot=0;slot<bindings.size();++slot)
            bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount=static_cast<std::uint32_t>(bindings.size()); info.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&info,nullptr,&DisplacementSetLayout),"Create displacement descriptor layout")) return false;
    }
    if (!DisplacementLayout)
    {
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; info.setLayoutCount=1; info.pSetLayouts=&DisplacementSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&info,nullptr,&DisplacementLayout),"Create displacement pipeline layout")) return false;
    }
    for (std::size_t phase=0;phase<DisplacementPipelines.size();++phase) if (!DisplacementPipelines[phase])
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanDisplacementShaders[phase],module,LastError)) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pipeline.layout=DisplacementLayout; pipeline.basePipelineIndex=-1;
        pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto result=Evans::Renderer::CreateCachedVulkanComputePipelines(DisplacementCreatePipelines,Device,0,1,&pipeline,nullptr,&DisplacementPipelines[phase]);
        DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(result,"Create displacement compute pipeline")) return false;
    }
    return true;
}
bool CreateDisplaced(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    DisplacementConstants constants;
    if (!ValidateDisplacementUpload(geometry,settings,constants,LastError) || !EnsureDisplacement()) return false;
    const std::size_t vertexBytes=geometry.Vertices.size()*sizeof(Vertex);
    const void* data[]{&constants,geometry.Vertices.data(),geometry.Directions.data(),geometry.Indices.data(),geometry.Neighborhoods.data(),geometry.AdjacentTriangles.data(),nullptr};
    const std::size_t sizes[]{sizeof(constants),vertexBytes,geometry.Directions.size()*sizeof(Float4),geometry.Indices.size()*sizeof(std::uint32_t),
        geometry.Neighborhoods.size()*sizeof(ShaderUInt4),geometry.AdjacentTriangles.size()*sizeof(std::uint32_t),vertexBytes};
    const auto groups=static_cast<std::uint32_t>((geometry.Vertices.size()+DisplacementThreadCount-1)/DisplacementThreadCount);
    if (groups>DeviceLimits.maxComputeWorkGroupCount[0]) return Fail("Displacement dispatch exceeds the Vulkan group limit");
    for (std::size_t slot=1;slot<DisplacementScratch.size();++slot)
        if (sizes[slot]>DeviceLimits.maxStorageBufferRange) return Fail("Displacement allocation exceeds maxStorageBufferRange");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover displacement scratch fence")) return false;
    DestroyDisplacementScratch();
    const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::size_t slot=0;slot<DisplacementScratch.size();++slot)
    {
        VkFlags usage=slot?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (slot==DisplacementOutput) usage|=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (!CreateRawBuffer(sizes[slot],usage,host,DisplacementScratch[slot])) return false;
        if (data[slot]) std::memcpy(DisplacementScratch[slot].Mapped,data[slot],sizes[slot]);
    }
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,DisplacementStorageCount}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&DisplacementPool),"Create displacement descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool=DisplacementPool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&DisplacementSetLayout;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate displacement descriptors")) return false;
    std::array<VkDescriptorBufferInfo,7> buffers{}; std::array<VkWriteDescriptorSet,7> writes{};
    for (std::uint32_t slot=0;slot<writes.size();++slot)
    {
        buffers[slot]={DisplacementScratch[slot].Buffer,0,sizes[slot]}; writes[slot].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[slot].dstSet=set; writes[slot].dstBinding=slot; writes[slot].descriptorCount=1;
        writes[slot].descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[slot].pBufferInfo=&buffers[slot];
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
    BufferResource resource; resource.Description={BufferUsage::Vertex,vertexBytes,false,"Displaced vertices"}; resource.Bytes.resize(vertexBytes);
    DisplacedVertexBuffer candidate; candidate.Vertices.resize(geometry.Vertices.size());
    if (!BeginImmediate()) return false;
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,DisplacementLayout,0,1,&set,0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,DisplacementPipelines[0]); DisplacementDispatch(CommandBuffer,groups,1,1);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer=DisplacementScratch[DisplacementOutput].Buffer; barrier.size=vertexBytes;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,DisplacementPipelines[1]); DisplacementDispatch(CommandBuffer,groups,1,1);
    barrier.dstAccessMask=VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|VK_ACCESS_HOST_READ_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    if (!EndImmediate()) return false;
    const void* mapped=DisplacementScratch[DisplacementOutput].Mapped;
    std::memcpy(candidate.Vertices.data(),mapped,vertexBytes); std::memcpy(resource.Bytes.data(),mapped,vertexBytes);
    if (!ValidateTriangleGeometry(candidate.Vertices,geometry.Indices,LastError)) return false;
    resource.Allocation=DisplacementScratch[DisplacementOutput]; DisplacementScratch[DisplacementOutput]={};
    candidate.Buffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(DisplacementPipelines.size());
    try { Buffers.emplace(candidate.Buffer.Value,std::move(resource)); }
    catch (...) { DestroyRawBuffer(resource.Allocation); throw; }
    output=std::move(candidate); DestroyDisplacementScratch(); return true;
}
