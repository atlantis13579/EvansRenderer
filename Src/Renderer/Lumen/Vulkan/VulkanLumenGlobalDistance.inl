VkDescriptorSetLayout LumenGlobalSet=0;
VkPipelineLayout LumenGlobalLayout=0;
VkPipeline LumenGlobalCompose=0;
void DestroyLumenGlobalDistance()
{
    DestroyLumenGlobalDistanceAsync();
    if (LumenGlobalCompose) DestroyPipelineFunction(Device,LumenGlobalCompose,nullptr); LumenGlobalCompose=0;
    if (LumenGlobalLayout) DestroyPipelineLayoutFunction(Device,LumenGlobalLayout,nullptr); LumenGlobalLayout=0;
    if (LumenGlobalSet) DestroyDescriptorSetLayoutFunction(Device,LumenGlobalSet,nullptr); LumenGlobalSet=0;
}
bool EnsureLumenGlobalDistance()
{
    if (!EnsureLumenDistanceTrace()) return false;
    if (DeviceLimits.maxUniformBufferRange<sizeof(LumenGlobalDistanceFrame)) return Fail("Global distance constants exceed the Vulkan uniform limit");
    if (!LumenGlobalSet)
    {
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned slot=0;slot<4;++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=4; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenGlobalSet),"Create global distance descriptors")) return false;
    }
    if (!LumenGlobalLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenGlobalSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenGlobalLayout),"Create global distance layout")) return false;
    }
    if (LumenGlobalCompose) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenGlobalDistanceShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenGlobalLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenGlobalCompose); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create global distance composition");
}
bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source scene");
    std::shared_ptr<LumenDistanceRecord> retained;
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second)
            return Fail("Global distance reuse requires the exact immutable source scene");
        retained=old->second;
    }
    auto plan=std::make_shared<LumenGlobalDistancePlan>();
    if (!BuildLumenGlobalDistancePlan(found->second->Info,settings,*plan,LastError) || !EnsureLumenGlobalDistance()) return false;
    LumenGlobalDistanceUpdate update;
    if (!BuildLumenGlobalDistanceUpdate(found->second->Info,*plan,retained?retained->Info.Global.get():nullptr,update,LastError)) return false;
    auto frame=plan->Frame; frame.Dispatch={0,unsigned(update.Compose.size()),0,0};
    const auto bytes=std::size_t(plan->WordCount)*sizeof(std::uint32_t);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Global distance scene exceeds Vulkan storage range");
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Global distance dispatch exceeds Vulkan workgroup limits");
    auto work=std::make_shared<LumenDistanceWork>(*this); work->Record=std::make_shared<LumenDistanceRecord>(*this);
    auto& record=*work->Record; record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=plan; record.Hardware=record.Base->Hardware;
    work->CopySource=retained;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(plan->Frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host,work->Uniform) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record.Scene)) return false;
    const auto dirtyBytes=std::max<std::size_t>(1,update.Compose.size())*sizeof(std::uint32_t);
    if (!CreateRawBuffer(dirtyBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Upload)) return false;
    if (!update.Compose.empty()) std::memcpy(work->Upload.Mapped,update.Compose.data(),update.Compose.size()*sizeof(std::uint32_t));
    std::memcpy(work->Uniform.Mapped,&frame,sizeof(frame));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create global distance descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenGlobalSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate global distance descriptors")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&record.Base->Scene,&work->Upload,&record.Scene}; VkDescriptorBufferInfo infos[4]{}; VkWriteDescriptorSet writes[4]{};
    for (unsigned slot=0;slot<4;++slot)
    {
        infos[slot]={buffers[slot]->Buffer,0,buffers[slot]->Size}; auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1; write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&infos[slot];
    }
    UpdateDescriptorSets(Device,4,writes,0,nullptr); if (!BeginImmediate()) return false;
    LumenDistanceBarrier(record.Base->Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    LumenDistanceBarrier(work->Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Upload,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!update.Copies.empty())
    {
        LumenDistanceBarrier(retained->Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
        std::vector<VkBufferCopy> regions; regions.reserve(update.Copies.size());
        for (const auto& copy:update.Copies) regions.push_back({std::size_t(copy.SourceWord)*sizeof(std::uint32_t),
            std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t)});
        DistanceCopyBuffer(CommandBuffer,retained->Scene.Buffer,record.Scene.Buffer,unsigned(regions.size()),regions.data());
        LumenDistanceBarrier(retained->Scene,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    const VkBufferCopy prefix{0,0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t)};
    DistanceCopyBuffer(CommandBuffer,record.Base->Scene.Buffer,record.Scene.Buffer,1,&prefix);
    const VkBufferCopy header{offsetof(LumenGlobalDistanceFrame,Header),plan->Frame.Header.Layout.W*sizeof(std::uint32_t),sizeof(LumenGlobalDistanceHeader)};
    DistanceCopyBuffer(CommandBuffer,work->Uniform.Buffer,record.Scene.Buffer,1,&header);
    LumenDistanceBarrier(record.Base->Scene,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(record.Scene,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenGlobalCompose);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenGlobalLayout,0,1,&work->Set,0,nullptr); if (groups) DistanceDispatch(CommandBuffer,x,y,1);
    LumenDistanceBarrier(record.Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitLumenDistanceWork(work)) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,work->Record); output=handle; return true;
}
