// Coherent storage is both the native drawable output and its CPU mirror.
// A control-point dependency precedes evaluation; vertex/index and host access
// are made visible together. Scratch survives unsuccessful fence waits.
VkDescriptorSetLayout TessellationSetLayout=0;
VkPipelineLayout TessellationLayout=0;
std::array<VkPipeline,3> TessellationPipelines{};
std::array<RawBuffer,9> TessellationScratch{};
VkDescriptorPool TessellationPool=0;
PFN_vkCreateComputePipelines TessellationCreatePipelines=nullptr;
PFN_vkCmdDispatch TessellationDispatch=nullptr;
static constexpr std::size_t TessellationControls=6,TessellationVertices=7,TessellationIndices=8,TessellationStorageCount=8;
void DestroyTessellationScratch()
{
    if (TessellationPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,TessellationPool,nullptr);
    TessellationPool=0; for (auto& buffer:TessellationScratch) DestroyRawBuffer(buffer);
}
void DestroyTessellation()
{
    DestroyTessellationScratch();
    for (auto& pipeline:TessellationPipelines) { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (TessellationLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,TessellationLayout,nullptr);
    if (TessellationSetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,TessellationSetLayout,nullptr);
    TessellationLayout=0; TessellationSetLayout=0;
}
bool SupportsTessellation() const
{
    return GraphicsQueueHasCompute && DeviceLimits.maxPerStageDescriptorStorageBuffers>=TessellationStorageCount &&
        DeviceLimits.maxDescriptorSetStorageBuffers>=TessellationStorageCount && DeviceLimits.maxPerStageResources>=TessellationStorageCount+1 &&
        DeviceLimits.maxUniformBufferRange>=sizeof(TessellationConstants) && DeviceLimits.maxComputeWorkGroupInvocations>=TessellationThreadCount &&
        DeviceLimits.maxComputeWorkGroupSize[0]>=TessellationThreadCount;
}
bool EnsureTessellation()
{
    if (!SupportsTessellation()) return Fail("Vulkan PN tessellation requires compute and eight storage descriptors");
    if (!LoadDevice(TessellationCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(TessellationDispatch,"vkCmdDispatch")) return false;
    if (!TessellationSetLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,9> bindings{};
        for (std::uint32_t slot=0;slot<bindings.size();++slot)
            bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; info.bindingCount=static_cast<std::uint32_t>(bindings.size()); info.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&info,nullptr,&TessellationSetLayout),"Create PN descriptor layout")) return false;
    }
    if (!TessellationLayout)
    {
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; info.setLayoutCount=1; info.pSetLayouts=&TessellationSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&info,nullptr,&TessellationLayout),"Create PN pipeline layout")) return false;
    }
    for (std::size_t phase=0;phase<TessellationPipelines.size();++phase) if (!TessellationPipelines[phase])
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanTessellationShaders[phase],module,LastError)) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pipeline.layout=TessellationLayout; pipeline.basePipelineIndex=-1;
        pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto result=Evans::Renderer::CreateCachedVulkanComputePipelines(TessellationCreatePipelines,Device,0,1,&pipeline,nullptr,&TessellationPipelines[phase]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(result,"Create PN compute pipeline")) return false;
    }
    return true;
}
bool CreateTessellated(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    TessellationConstants constants;
    if (!ValidateTessellationUpload(geometry,settings,constants,LastError) || !EnsureTessellation()) return false;
    const std::size_t vertexBytes=std::size_t(constants.Outputs.X)*sizeof(Vertex),indexBytes=std::size_t(constants.Outputs.Y)*sizeof(std::uint32_t);
    const void* data[]{&constants,geometry.Vertices.data(),geometry.Patches.data(),geometry.PatchEdges.data(),geometry.AttributeEdges.data(),geometry.GeometricEdges.data(),nullptr,nullptr,nullptr};
    const std::size_t sizes[]{sizeof(constants),geometry.Vertices.size()*sizeof(Vertex),geometry.Patches.size()*sizeof(ShaderUInt4),geometry.PatchEdges.size()*sizeof(ShaderUInt4),
        geometry.AttributeEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*2*sizeof(Float4),vertexBytes,indexBytes};
    const unsigned counts[]{constants.Inputs.W,constants.Outputs.X,constants.Outputs.Y/3};
    for (auto count:counts) if ((count+TessellationThreadCount-1)/TessellationThreadCount>DeviceLimits.maxComputeWorkGroupCount[0]) return Fail("PN dispatch exceeds the Vulkan group limit");
    for (std::size_t slot=1;slot<TessellationScratch.size();++slot) if (sizes[slot]>DeviceLimits.maxStorageBufferRange) return Fail("PN storage exceeds maxStorageBufferRange");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover PN scratch fence")) return false;
    DestroyTessellationScratch(); const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::size_t slot=0;slot<TessellationScratch.size();++slot)
    {
        VkFlags usage=slot?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (slot==TessellationVertices) usage|=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; if (slot==TessellationIndices) usage|=VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (!CreateRawBuffer(sizes[slot],usage,host,TessellationScratch[slot])) return false;
        if (data[slot]) std::memcpy(TessellationScratch[slot].Mapped,data[slot],sizes[slot]);
    }
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,TessellationStorageCount}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&TessellationPool),"Create PN descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool=TessellationPool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&TessellationSetLayout;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate PN descriptors")) return false;
    std::array<VkDescriptorBufferInfo,9> buffers{}; std::array<VkWriteDescriptorSet,9> writes{};
    for (std::uint32_t slot=0;slot<writes.size();++slot)
    {
        buffers[slot]={TessellationScratch[slot].Buffer,0,sizes[slot]}; writes[slot].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[slot].dstSet=set; writes[slot].dstBinding=slot; writes[slot].descriptorCount=1;
        writes[slot].descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[slot].pBufferInfo=&buffers[slot];
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
    if (!BeginImmediate()) return false;
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TessellationLayout,0,1,&set,0,nullptr);
    for (std::size_t phase=0;phase<TessellationPipelines.size();++phase)
    {
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TessellationPipelines[phase]);
        TessellationDispatch(CommandBuffer,(counts[phase]+TessellationThreadCount-1)/TessellationThreadCount,1,1);
        if (phase==0)
        {
            VkBufferMemoryBarrier dependency{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; dependency.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            dependency.dstAccessMask=VK_ACCESS_SHADER_READ_BIT; dependency.srcQueueFamilyIndex=dependency.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            dependency.buffer=TessellationScratch[TessellationControls].Buffer; dependency.size=sizes[TessellationControls];
            CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&dependency,0,nullptr);
        }
    }
    std::array<VkBufferMemoryBarrier,2> visible{};
    for (std::size_t stream=0;stream<visible.size();++stream)
    {
        auto& barrier=visible[stream]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask=(stream?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)|VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=TessellationScratch[TessellationVertices+stream].Buffer; barrier.size=sizes[TessellationVertices+stream];
    }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,static_cast<std::uint32_t>(visible.size()),visible.data(),0,nullptr);
    if (!EndImmediate()) return false;
    TessellatedGeometryBuffers candidate; candidate.Vertices.resize(constants.Outputs.X); candidate.Indices.resize(constants.Outputs.Y);
    void* mirrors[]{candidate.Vertices.data(),candidate.Indices.data()}; std::array<BufferResource,2> resources;
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"PN tessellated vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"PN tessellated indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; const auto bytes=resource.Description.SizeInBytes; resource.Bytes.resize(bytes);
        std::memcpy(mirrors[stream],TessellationScratch[TessellationVertices+stream].Mapped,bytes); std::memcpy(resource.Bytes.data(),mirrors[stream],bytes);
        resource.Allocation=TessellationScratch[TessellationVertices+stream];
    }
    if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,LastError)) return false;
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=3;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    TessellationScratch[TessellationVertices]={}; TessellationScratch[TessellationIndices]={};
    output=std::move(candidate); DestroyTessellationScratch(); return true;
}
