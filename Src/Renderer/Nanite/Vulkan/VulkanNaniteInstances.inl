NaniteInstanceWork InstancePlan;
struct InstanceResources
{
    Impl& Owner; RawBuffer Uniform; VkDescriptorPool Pool=0;
    std::vector<std::shared_ptr<NaniteFrameBuffer>> Buffers;
    std::vector<std::array<VkDescriptorSet,2>> Sets;
    explicit InstanceResources(Impl& owner):Owner(owner) {}
    ~InstanceResources() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); }
};
std::shared_ptr<InstanceResources> InstanceWork;
VkDescriptorSetLayout InstanceSetLayout=0;
VkPipelineLayout InstanceLayout=0;
std::array<VkPipeline,3> InstanceKernels{};
void(EVANS_VK_CALL* InstanceDispatchIndirect)(VkCommandBuffer,VkBuffer,VkDeviceSize)=nullptr;
void DestroyNaniteInstances()
{
    InstancePlan={}; InstanceWork.reset();
    for (auto pipeline:InstanceKernels) if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); InstanceKernels={};
    if (InstanceLayout) DestroyPipelineLayoutFunction(Device,InstanceLayout,nullptr); InstanceLayout=0;
    if (InstanceSetLayout) DestroyDescriptorSetLayoutFunction(Device,InstanceSetLayout,nullptr); InstanceSetLayout=0;
}
bool PrepareNaniteInstances(const VisibilityBufferDesc& frame)
{
    std::vector<NaniteVisibilityGeometryConstants> constants;
    if (frame.GpuInstances && NaniteAssemblyWork) for (const auto& source:NaniteAssemblyWork->Sources) constants.push_back(source.Constants);
    if (!BuildNaniteInstanceWork(frame,constants,InstancePlan,LastError)) return false;
    InstanceWork.reset(); if (InstancePlan.Batches.empty()) return true;
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<4 || DeviceLimits.maxDescriptorSetStorageBuffers<4)
        return Fail("Vulkan instance assembly requires four storage bindings");
    if (!LoadDevice(InstanceDispatchIndirect,"vkCmdDispatchIndirect")) return false;
    auto work=std::make_shared<InstanceResources>(*this);
    for (const auto& batch:InstancePlan.Batches)
    {
        const auto bytes=batch.Upload.size()*sizeof(unsigned);
        if (bytes>DeviceLimits.maxStorageBufferRange || batch.MaximumGroupsX>DeviceLimits.maxComputeWorkGroupCount[0] ||
            batch.MaximumGroupsY>DeviceLimits.maxComputeWorkGroupCount[1] || (batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads>DeviceLimits.maxComputeWorkGroupCount[0])
            return Fail("Vulkan instance schedule exceeds storage/dispatch limits");
        auto buffer=std::make_shared<NaniteFrameBuffer>(*this); buffer->Words=unsigned(batch.Upload.size());
        if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buffer->Allocation)) return false;
        std::memcpy(buffer->Allocation.Mapped,batch.Upload.data(),bytes); work->Buffers.push_back(std::move(buffer));
    }
    if (!InstanceSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (unsigned i=0;i<5;++i) bindings[i]={i,i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=5; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&InstanceSetLayout),"Create instance schedule descriptors")) return false;
    }
    if (!InstanceLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&InstanceSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&InstanceLayout),"Create instance schedule layout")) return false;
    }
    for (unsigned pass=0;pass<InstanceKernels.size();++pass) if (!InstanceKernels[pass])
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteInstanceShaders[pass],module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=InstanceLayout; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&desc,nullptr,&InstanceKernels[pass]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create instance schedule pipeline")) return false;
    }
    const unsigned families=unsigned(InstancePlan.Batches.size()),setCount=families*2;
    const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const auto stride=(sizeof(ShaderUInt4)+alignment-1)/alignment*alignment;
    if (!CreateRawBuffer(stride*families,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,setCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,setCount*4}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=setCount; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create instance schedule pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(setCount,InstanceSetLayout); std::vector<VkDescriptorSet> sets(setCount);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=setCount; allocation.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocation,sets.data()),"Allocate instance schedule sets")) return false;
    for (unsigned family=0;family<families;++family)
    {
        const auto& batch=InstancePlan.Batches[family]; const auto& vertex=Buffers.at(batch.Vertex.Value).Allocation;
        const auto& index=Buffers.at(batch.Index.Value).Allocation; const auto& buffer=work->Buffers[family]->Allocation;
        std::memcpy(static_cast<std::uint8_t*>(work->Uniform.Mapped)+family*stride,&batch.Parameters,sizeof(batch.Parameters));
        work->Sets.push_back({sets[family*2],sets[family*2+1]});
        for (unsigned mode=0;mode<2;++mode)
        {
            const auto& first=mode?vertex:NaniteRasterDecisions->Allocation;
            VkDescriptorBufferInfo buffers[]{{work->Uniform.Buffer,family*stride,sizeof(ShaderUInt4)},{first.Buffer,0,first.Size},
                {index.Buffer,0,index.Size},{VisibilityGeometry.Buffer,0,VisibilityGeometry.Size},{buffer.Buffer,0,buffer.Size}};
            VkWriteDescriptorSet writes[5]{};
            for (unsigned binding=0;binding<5;++binding)
            {
                auto& write=writes[binding]; write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=work->Sets.back()[mode]; write.dstBinding=binding;
                write.descriptorCount=1; write.descriptorType=binding?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[binding];
            }
            UpdateDescriptorSets(Device,5,writes,0,nullptr);
        }
    }
    InstanceWork=std::move(work); return true;
}
bool DrawNaniteInstances()
{
    for (unsigned family=0;family<InstancePlan.Batches.size();++family)
    {
        const auto& batch=InstancePlan.Batches[family]; const auto& work=InstanceWork->Buffers[family]->Allocation;
        NaniteCopyLocks.push_back(batch.Vertex); NaniteCopyLocks.push_back(batch.Index);
        const RawBuffer* inputs[]{&Buffers.at(batch.Vertex.Value).Allocation,&Buffers.at(batch.Index.Value).Allocation,&NaniteRasterDecisions->Allocation,&work};
        VkBufferMemoryBarrier barriers[4]{};
        for (unsigned i=0;i<4;++i)
        {
            auto& barrier=barriers[i]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|(i==3?VK_ACCESS_SHADER_WRITE_BIT:0);
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=inputs[i]->Buffer; barrier.size=inputs[i]->Size;
        }
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,4,barriers,0,nullptr);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,InstanceLayout,0,1,&InstanceWork->Sets[family][0],0,nullptr);
        for (unsigned pass=0;pass<2;++pass)
        {
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,InstanceKernels[pass]); NaniteGeometryDispatch(CommandBuffer,pass?1:(batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads,1,1);
            auto barrier=barriers[3]; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|(pass?VK_ACCESS_INDIRECT_COMMAND_READ_BIT:VK_ACCESS_SHADER_WRITE_BIT);
            CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|(pass?VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT:0),0,0,nullptr,1,&barrier,0,nullptr);
        }
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,InstanceLayout,0,1,&InstanceWork->Sets[family][1],0,nullptr);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,InstanceKernels[2]); InstanceDispatchIndirect(CommandBuffer,work.Buffer,0);
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount; PendingFrame.Statistics.NaniteInstanceDispatchCount+=unsigned(InstanceKernels.size());
    }
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=VisibilityGeometry.Buffer; barrier.size=VisibilityGeometry.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|(MeshStagesEnabled?VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT:0))|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr); return true;
}
