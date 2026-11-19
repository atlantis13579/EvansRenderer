struct LumenHardwareRecord;
struct LumenDistanceRecord
{
    Impl& Owner; LumenDistanceSceneInfo Info; std::shared_ptr<LumenDistanceRecord> Base; RawBuffer Scene;
    std::shared_ptr<LumenHardwareRecord> Hardware;
    explicit LumenDistanceRecord(Impl& owner):Owner(owner) {}
    ~LumenDistanceRecord() { Owner.DestroyRawBuffer(Scene); }
};
struct LumenDistanceWork
{
    Impl& Owner; RawBuffer Upload,Uniform,Feedback; VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    std::shared_ptr<LumenDistanceRecord> Record,CopySource;
    std::vector<RawBuffer> HardwareInputs;
    explicit LumenDistanceWork(Impl& owner):Owner(owner) {}
    ~LumenDistanceWork()
    { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Upload); Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Feedback); for (auto& buffer:HardwareInputs) Owner.DestroyRawBuffer(buffer); }
};
#include "Renderer/Lumen/Vulkan/VulkanLumenHardwareRayTracing.inl"
std::unordered_map<std::uint64_t,std::shared_ptr<LumenDistanceRecord>> LumenDistanceScenes;
std::shared_ptr<LumenDistanceWork> LumenDistancePending;
VkDescriptorSetLayout LumenDistanceSet=0; VkPipelineLayout LumenDistanceLayout=0; VkPipeline LumenDistanceTrace=0;
PFN_vkCreateComputePipelines DistanceCreateCompute=nullptr; PFN_vkCmdDispatch DistanceDispatch=nullptr; PFN_vkCmdCopyBuffer DistanceCopyBuffer=nullptr;
bool RecoverLumenDistanceScene()
{
    if (!LumenDistancePending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen distance work")) return false;
    LumenDistancePending.reset(); return true;
}
void DestroyLumenDistanceScenes()
{
    LumenDistancePending.reset(); LumenDistanceScenes.clear(); DestroyLumenGlobalDistance(); DestroyLumenHardwarePrograms();
    if (LumenDistanceTrace) DestroyPipelineFunction(Device,LumenDistanceTrace,nullptr); LumenDistanceTrace=0;
    if (LumenDistanceLayout) DestroyPipelineLayoutFunction(Device,LumenDistanceLayout,nullptr); LumenDistanceLayout=0;
    if (LumenDistanceSet) DestroyDescriptorSetLayoutFunction(Device,LumenDistanceSet,nullptr); LumenDistanceSet=0;
    if (LumenHardwareSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenHardwareSetLayout,nullptr); LumenHardwareSetLayout=0;
}
bool EnsureLumenDistanceTrace(bool hardware=false)
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 ||
        DeviceLimits.maxUniformBufferRange<sizeof(LumenDistanceTraceFrame) || DeviceLimits.maxComputeWorkGroupInvocations<LumenDistanceTraceThreads ||
        DeviceLimits.maxComputeWorkGroupSize[0]<LumenDistanceTraceThreads) return Fail("Vulkan Lumen distance tracing exceeds descriptor/compute limits");
    if (!LoadDevice(DistanceCreateCompute,"vkCreateComputePipelines") || !LoadDevice(DistanceDispatch,"vkCmdDispatch") || !LoadDevice(DistanceCopyBuffer,"vkCmdCopyBuffer")) return false;
    if (!LumenDistanceSet)
    {
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned slot=0;slot<4;++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=4; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenDistanceSet),"Create Lumen distance descriptor layout")) return false;
    }
    if (!LumenDistanceLayout)
    {
        if (!CreateLumenQueryLayout(LumenDistanceSet,LumenDistanceLayout)) return false;
    }
    if (hardware) return EnsureLumenHardwarePipeline(LumenHardwareProgram::Distance,LumenDistanceLayout);
    if (LumenDistanceTrace) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenDistanceTraceShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenDistanceLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenDistanceTrace); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen distance compute pipeline");
}
void LumenDistanceBarrier(const RawBuffer& buffer,VkFlags before,VkFlags after,VkFlags sourceStage,VkFlags targetStage)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=before; barrier.dstAccessMask=after;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=buffer.Buffer; barrier.size=buffer.Size;
    CmdPipelineBarrier(CommandBuffer,sourceStage,targetStage,0,0,nullptr,1,&barrier,0,nullptr);
}
bool SubmitLumenDistanceWork(const std::shared_ptr<LumenDistanceWork>& work)
{
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen distance commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen distance fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen distance work"); }
    LumenDistancePending=work; return RecoverLumenDistanceScene();
}
bool CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!ValidateLumenDistanceScene(data,LastError) || !EnsureLumenDistanceTrace()) return false;
    const std::uint32_t empty=0; const auto bytes=std::max<std::size_t>(1,data.Words.size())*sizeof(std::uint32_t);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen distance scene exceeds storage range");
    auto work=std::make_shared<LumenDistanceWork>(*this); work->Record=std::make_shared<LumenDistanceRecord>(*this); work->Record->Info=data.Info;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host,work->Upload) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,work->Record->Scene)) return false;
    std::memcpy(work->Upload.Mapped,data.Words.empty()?&empty:data.Words.data(),bytes);
    if (!BeginImmediate()) return false;
    LumenDistanceBarrier(work->Upload,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,bytes}; DistanceCopyBuffer(CommandBuffer,work->Upload.Buffer,work->Record->Scene.Buffer,1,&copy);
    LumenDistanceBarrier(work->Record->Scene,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!BuildLumenHardwareAcceleration(*work) || !SubmitLumenDistanceWork(work)) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,work->Record); output=handle; return true;
}
bool TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    const auto found=LumenDistanceScenes.find(handle.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown Lumen distance scene");
    LumenDistanceTraceFrame frame; if (!BuildLumenDistanceTraceFrame(found->second->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace(bool(found->second->Hardware))) return false;
    const auto bytes=upload.Rays.size()*sizeof(LumenDistanceHit),queryBytes=upload.Rays.size()*sizeof(LumenDistanceRay);
    if (bytes>DeviceLimits.maxStorageBufferRange || queryBytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen distance queries exceed storage range");
    const auto groups=unsigned((upload.Rays.size()+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads);
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen distance dispatch exceeds native groups");
    auto work=std::make_shared<LumenDistanceWork>(*this); work->Record=found->second;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) ||
        !CreateRawBuffer(queryBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Upload) || !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Feedback)) return false;
    std::memcpy(work->Uniform.Mapped,&frame,sizeof(frame)); std::memcpy(work->Upload.Mapped,upload.Rays.data(),queryBytes);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen distance descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenDistanceSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate Lumen distance descriptor set")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&work->Record->Scene,&work->Upload,&work->Feedback}; VkDescriptorBufferInfo infos[4]{}; VkWriteDescriptorSet writes[4]{};
    for (unsigned slot=0;slot<4;++slot)
    {
        infos[slot]={buffers[slot]->Buffer,0,buffers[slot]->Size}; auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1; write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&infos[slot];
    }
    UpdateDescriptorSets(Device,4,writes,0,nullptr);
    LumenDistanceTraceResult result; result.Identity=work->Record->Info.Identity; result.Hits.resize(upload.Rays.size()); result.HardwareTracing=bool(work->Record->Hardware);
    if (!BeginImmediate()) return false;
    LumenDistanceBarrier(work->Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Upload,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,work->Record->Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Distance)]:LumenDistanceTrace);
    BindLumenHardware(CommandBuffer,LumenDistanceLayout,*work->Record);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenDistanceLayout,0,1,&work->Set,0,nullptr); DistanceDispatch(CommandBuffer,x,y,1);
    LumenDistanceBarrier(work->Feedback,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!SubmitLumenDistanceWork(work)) return false;
    std::memcpy(result.Hits.data(),work->Feedback.Mapped,bytes);
    if (!ValidateLumenDistanceTraceResult(work->Record->Info,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
bool DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{ return LumenDistanceScenes.erase(handle.Value)!=0 || Fail("Unknown Lumen distance scene"); }
