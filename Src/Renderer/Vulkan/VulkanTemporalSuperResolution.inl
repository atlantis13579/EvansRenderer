// Coherent storage buffers and explicit host/compute dependencies preserve each
// immutable history version. Pending submissions retain descriptors and sources
// until fence recovery; GLSL is compiled in memory by the Vulkan compiler.
struct TSRNativeHistory
{
    Impl& Owner; TSRHistoryInfo Info; std::array<RawBuffer,2> Buffers{};
    RawImage Color{}; VkFramebuffer Framebuffer=0;
    explicit TSRNativeHistory(Impl& owner):Owner(owner) {}
    ~TSRNativeHistory() { if (Framebuffer) Owner.DestroyFramebufferFunction(Owner.Device,Framebuffer,nullptr); Owner.DestroyRawImage(Color); for (auto& buffer:Buffers) Owner.DestroyRawBuffer(buffer); }
};
struct TSRNativeImages { std::array<RawImage*,TSRSceneImageCount> Images{}; std::uint32_t Flags=0; };
struct TSRWork
{
    Impl& Owner; std::array<RawBuffer,4> Sources{}; VkDescriptorPool Pool=0,PackPool=0;
    std::shared_ptr<TSRNativeHistory> Previous,Candidate;
    explicit TSRWork(Impl& owner):Owner(owner) {}
    ~TSRWork() { if (PackPool) Owner.DestroyDescriptorPoolFunction(Owner.Device,PackPool,nullptr); if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); for (auto& buffer:Sources) Owner.DestroyRawBuffer(buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<TSRNativeHistory>> TSRHistories;
TSRSceneState TSRScene;
RawImage TSROpaque{};
std::shared_ptr<TSRWork> TSRPending;
struct TSRDisplayWork;
std::shared_ptr<TSRDisplayWork> TSRPendingDisplay;
VkDescriptorSetLayout TSRSet=0; VkPipelineLayout TSRLayout=0; VkPipeline TSRPipeline=0;
VkDescriptorSetLayout TSRPackSet=0; VkPipelineLayout TSRPackLayout=0; VkPipeline TSRPackPipeline=0; VkSampler TSRPackSampler=0;
VkDescriptorSetLayout TSRExportSet=0; VkPipelineLayout TSRExportLayout=0; VkPipeline TSRExportPipeline=0; VkRenderPass TSRExportPass=0;
std::array<VkShaderModule,2> TSRExportModules{};
PFN_vkCreateComputePipelines TSRCreatePipeline=nullptr; PFN_vkCmdDispatch TSRDispatch=nullptr;
void DestroyTSR()
{
    TSRScene.Reset(); DestroyRawImage(TSROpaque); TSRPendingDisplay.reset(); TSRPending.reset(); TSRHistories.clear();
    if (TSRPackPipeline) DestroyPipelineFunction(Device,TSRPackPipeline,nullptr); TSRPackPipeline=0;
    if (TSRPackLayout) DestroyPipelineLayoutFunction(Device,TSRPackLayout,nullptr); TSRPackLayout=0;
    if (TSRPackSet) DestroyDescriptorSetLayoutFunction(Device,TSRPackSet,nullptr); TSRPackSet=0;
    if (TSRPackSampler) DestroySamplerFunction(Device,TSRPackSampler,nullptr); TSRPackSampler=0;
    if (TSRPipeline) DestroyPipelineFunction(Device,TSRPipeline,nullptr); TSRPipeline=0;
    if (TSRLayout) DestroyPipelineLayoutFunction(Device,TSRLayout,nullptr); TSRLayout=0;
    if (TSRSet) DestroyDescriptorSetLayoutFunction(Device,TSRSet,nullptr); TSRSet=0;
    if (TSRExportPipeline) DestroyPipelineFunction(Device,TSRExportPipeline,nullptr); TSRExportPipeline=0;
    for (auto& module:TSRExportModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (TSRExportPass) DestroyRenderPassFunction(Device,TSRExportPass,nullptr); TSRExportPass=0;
    if (TSRExportLayout) DestroyPipelineLayoutFunction(Device,TSRExportLayout,nullptr); TSRExportLayout=0;
    if (TSRExportSet) DestroyDescriptorSetLayoutFunction(Device,TSRExportSet,nullptr); TSRExportSet=0;
}
bool DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Release TSR history fence")) return false;
    return (handle && TSRHistories.erase(handle.Value)) || Fail("Vulkan TSR history handle is invalid or belongs to another RHI");
}
bool EnsureTSR()
{
    constexpr unsigned StorageBindings=4;
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<StorageBindings || DeviceLimits.maxDescriptorSetStorageBuffers<StorageBindings ||
        DeviceLimits.maxPerStageResources<StorageBindings+1 || DeviceLimits.maxUniformBufferRange<sizeof(TSRConstants) ||
        DeviceLimits.maxComputeWorkGroupInvocations<TSRGroupSide*TSRGroupSide || DeviceLimits.maxComputeWorkGroupSize[0]<TSRGroupSide || DeviceLimits.maxComputeWorkGroupSize[1]<TSRGroupSide)
        return Fail("Vulkan TSR requires compute, four storage buffers and its frame block");
    if (!LoadDevice(TSRCreatePipeline,"vkCreateComputePipelines") || !LoadDevice(TSRDispatch,"vkCmdDispatch")) return false;
    if (!TSRSet)
    {
        std::array<VkDescriptorSetLayoutBinding,StorageBindings+1> bindings{};
        for (unsigned slot=0;slot<bindings.size();++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=static_cast<unsigned>(bindings.size()); desc.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&TSRSet),"Create TSR descriptor layout")) return false;
    }
    if (!TSRLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&TSRSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&TSRLayout),"Create TSR pipeline layout")) return false;
    }
    if (!TSRPipeline)
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanTSRCompute,module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=TSRLayout; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(TSRCreatePipeline,Device,0,1,&desc,nullptr,&TSRPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create TSR compute pipeline")) return false;
    }
    return true;
}
bool ExportTSR(const std::shared_ptr<TSRNativeHistory>& record,TSRReadback& result,bool readback=true)
{
    if (!TSRExportSet)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=2; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&TSRExportSet),"Create TSR HDR export set layout")) return false;
    }
    if (!TSRExportLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&TSRExportSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&TSRExportLayout),"Create TSR HDR export pipeline layout")) return false;
    }
    if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,TSRExportPass) ||
        !CreateFullscreenPipeline(VulkanTSRExport,TSRExportModules,TSRExportPass,TSRExportPipeline,TSRExportLayout)) return false;
    if (!CreateRawImage(record->Info.Width,record->Info.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,record->Color) ||
        !CreateFramebuffer(TSRExportPass,&record->Color.View,1,record->Info.Width,record->Info.Height,record->Framebuffer)) return false;
    auto work=std::make_shared<TSRWork>(*this); work->Candidate=record;
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(record->Info.Width,record->Info.Height,0,constants,LastError)) return false;
    const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto bytes=std::size_t(record->Info.Width)*record->Info.Height*4*sizeof(std::uint16_t);
    if (!CreateRawBuffer(sizeof(constants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Sources[0]) ||
        (readback && !CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host,work->Sources[1]))) return false;
    std::memcpy(work->Sources[0].Mapped,&constants,sizeof(constants));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create TSR HDR export descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&TSRExportSet;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate TSR HDR export set")) return false;
    const VkDescriptorBufferInfo buffers[]{{work->Sources[0].Buffer,0,sizeof(constants)},{record->Buffers[0].Buffer,0,record->Buffers[0].Size}};
    std::array<VkWriteDescriptorSet,2> writes{};
    for (unsigned slot=0;slot<writes.size();++slot) { writes[slot]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[slot].dstSet=set; writes[slot].dstBinding=slot; writes[slot].descriptorCount=1; writes[slot].descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[slot].pBufferInfo=&buffers[slot]; }
    UpdateDescriptorSets(Device,static_cast<unsigned>(writes.size()),writes.data(),0,nullptr); TSRPending=work; if (!BeginImmediate()) return false;
    std::array<VkBufferMemoryBarrier,2> ready{};
    for (unsigned slot=0;slot<ready.size();++slot) { auto& b=ready[slot]; b={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; b.srcAccessMask=slot?VK_ACCESS_SHADER_WRITE_BIT:VK_ACCESS_HOST_WRITE_BIT; b.dstAccessMask=slot?VK_ACCESS_SHADER_READ_BIT:VK_ACCESS_UNIFORM_READ_BIT; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=buffers[slot].buffer; b.size=buffers[slot].range; }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,static_cast<unsigned>(ready.size()),ready.data(),0,nullptr);
    TransitionImage(CommandBuffer,record->Color,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=TSRExportPass; begin.framebuffer=record->Framebuffer; begin.renderArea={{0,0},{record->Info.Width,record->Info.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const VkViewport viewport{0,0,float(record->Info.Width),float(record->Info.Height),0,1}; const VkRect2D scissor{{0,0},{record->Info.Width,record->Info.Height}};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,TSRExportPipeline); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,TSRExportLayout,0,1,&set,0,nullptr); CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,record->Color,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if(readback){
    VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={record->Info.Width,record->Info.Height,1}; CmdCopyImageToBuffer(CommandBuffer,record->Color.Image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,work->Sources[1].Buffer,1,&copy);
    VkBufferMemoryBarrier visible{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; visible.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; visible.dstAccessMask=VK_ACCESS_HOST_READ_BIT; visible.srcQueueFamilyIndex=visible.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; visible.buffer=work->Sources[1].Buffer; visible.size=bytes;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&visible,0,nullptr);
    }
    TransitionImage(CommandBuffer,record->Color,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); if (!EndImmediate()) return false; TSRPending.reset();
    if(!readback)return true;
    result.PresentationHDR.resize(std::size_t(record->Info.Width)*record->Info.Height); const auto* pixels=static_cast<const std::uint16_t*>(work->Sources[1].Mapped);
    for (std::size_t i=0;i<result.PresentationHDR.size();++i) result.PresentationHDR[i]=DecodeTSRHalfPixel(pixels+i*4);
    return true;
}
bool PackTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRWork& work)
{
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(frame.InputWidth,frame.InputHeight,images.Flags,constants,LastError)) return false;
    for (unsigned slot=0;slot<TSRSceneImageCount;++slot)
        if (TSRImageRequired(slot,images.Flags) && (!images.Images[slot] || !images.Images[slot]->Image || images.Images[slot]->Width!=frame.InputWidth || images.Images[slot]->Height!=frame.InputHeight || images.Images[slot]->Samples!=1 || images.Images[slot]->ArrayLayers!=1))
            return Fail("Vulkan TSR requires resolved render-sized scene images");
    if (DeviceLimits.maxPerStageDescriptorSampledImages<TSRSceneImageCount || DeviceLimits.maxDescriptorSetSampledImages<TSRSceneImageCount ||
        DeviceLimits.maxPerStageDescriptorSamplers<TSRSceneImageCount || DeviceLimits.maxDescriptorSetSamplers<TSRSceneImageCount || DeviceLimits.maxPerStageResources<TSRSceneImageCount+2)
        return Fail("Vulkan TSR packing requires six image descriptors and two buffer descriptors");
    if (!TSRPackSet)
    {
        std::array<VkDescriptorSetLayoutBinding,TSRSceneImageCount+2> bindings{};
        for (unsigned slot=0;slot<bindings.size();++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==bindings.size()-1?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=unsigned(bindings.size()); desc.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&TSRPackSet),"Create TSR pack set layout")) return false;
    }
    if (!TSRPackLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&TSRPackSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&TSRPackLayout),"Create TSR pack pipeline layout")) return false;
    }
    if (!TSRPackPipeline)
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanTSRPack,module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=TSRPackLayout; desc.basePipelineIndex=-1; desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(TSRCreatePipeline,Device,0,1,&desc,nullptr,&TSRPackPipeline); DestroyShaderModuleFunction(Device,module,nullptr); if (!Check(status,"Create TSR packing pipeline")) return false;
    }
    if (!TSRPackSampler)
    {
        VkSamplerCreateInfo desc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; desc.minFilter=desc.magFilter=VK_FILTER_NEAREST; desc.addressModeU=desc.addressModeV=desc.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&desc,nullptr,&TSRPackSampler),"Create TSR image pack sampler")) return false;
    }
    if (!CreateRawBuffer(sizeof(constants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work.Sources[3])) return false;
    std::memcpy(work.Sources[3].Mapped,&constants,sizeof(constants));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,TSRSceneImageCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work.PackPool),"Create TSR packing descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work.PackPool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&TSRPackSet;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate TSR packing descriptor set")) return false;
    const VkDescriptorBufferInfo buffers[]{{work.Sources[3].Buffer,0,sizeof(constants)},{work.Sources[1].Buffer,0,work.Sources[1].Size}};
    std::array<VkDescriptorImageInfo,TSRSceneImageCount> imageInfo{}; std::array<VkWriteDescriptorSet,TSRSceneImageCount+2> writes{};
    for (unsigned slot=0;slot<writes.size();++slot)
    {
        auto& write=writes[slot]; write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=set; write.dstBinding=slot; write.descriptorCount=1;
        if (slot==0 || slot==writes.size()-1) { write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[slot?1:0]; }
        else { const auto image=slot-1; auto* source=TSRImageRequired(image,images.Flags)?images.Images[image]:images.Images[TSRSceneFinal]; imageInfo[image]={TSRPackSampler,source->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&imageInfo[image]; }
    }
    UpdateDescriptorSets(Device,unsigned(writes.size()),writes.data(),0,nullptr);
    std::array<VkImageMemoryBarrier,TSRSceneImageCount> ready{}; unsigned count=0;
    for (unsigned image=0;image<TSRSceneImageCount;++image) if (TSRImageRequired(image,images.Flags))
    {
        bool duplicate=false;
        for (unsigned prior=0;prior<image;++prior) if (TSRImageRequired(prior,images.Flags) && images.Images[prior]->Image==images.Images[image]->Image) { duplicate=true; break; }
        if (duplicate) continue;
        auto& source=*images.Images[image]; TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        auto& b=ready[count++]; b={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT; b.oldLayout=b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=source.Image; b.subresourceRange={source.Aspect,0,source.MipLevels,0,1};
    }
    VkBufferMemoryBarrier uniform{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; uniform.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; uniform.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT; uniform.srcQueueFamilyIndex=uniform.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; uniform.buffer=work.Sources[3].Buffer; uniform.size=sizeof(constants);
    constexpr VkFlags SceneWriters=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    CmdPipelineBarrier(CommandBuffer,SceneWriters|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&uniform,count,ready.data());
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TSRPackPipeline); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TSRPackLayout,0,1,&set,0,nullptr);
    TSRDispatch(CommandBuffer,(frame.InputWidth+TSRGroupSide-1)/TSRGroupSide,(frame.InputHeight+TSRGroupSide-1)/TSRGroupSide,1); return true;
}
bool ReconstructTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRHistoryHandle previous,TSRResult& output)
{
    std::shared_ptr<TSRNativeHistory> old;
    if (previous) { const auto found=TSRHistories.find(previous.Value); if (found==TSRHistories.end()) return Fail("Vulkan previous TSR history is invalid"); old=found->second; }
    TSRConstants constants; if (!BuildTSRConstants(frame,old?&old->Info:nullptr,constants,LastError) || !EnsureTSR()) return false;
    const unsigned groupsX=(constants.Extent.Z+TSRGroupSide-1)/TSRGroupSide,groupsY=(constants.Extent.W+TSRGroupSide-1)/TSRGroupSide;
    if (groupsX>DeviceLimits.maxComputeWorkGroupCount[0] || groupsY>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("TSR dispatch exceeds Vulkan group counts");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover TSR submission")) return false;
    TSRPending.reset(); auto work=std::make_shared<TSRWork>(*this); work->Previous=old; work->Candidate=std::make_shared<TSRNativeHistory>(*this);
    auto& record=*work->Candidate; record.Info=TSRHistoryIdentity(frame);
    const auto bytes=std::size_t(record.Info.Width)*record.Info.Height*TSRHistoryVectors*sizeof(Float4);
    const std::array<Float4,TSRHistoryVectors> empty{}; const void* data[]{&constants,nullptr,empty.data()}; const std::size_t sizes[]{sizeof(constants),std::size_t(frame.InputWidth)*frame.InputHeight*(TSRInputVectors*sizeof(Float4)),sizeof(empty)};
    const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (unsigned slot=0;slot<3;++slot)
    {
        if (slot && sizes[slot]>DeviceLimits.maxStorageBufferRange) return Fail("TSR source exceeds Vulkan storage range");
        if (!CreateRawBuffer(sizes[slot],slot?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,(!frame.CaptureDiagnostics && slot==1)?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:host,work->Sources[slot])) return false;
        if (data[slot]) std::memcpy(work->Sources[slot].Mapped,data[slot],sizes[slot]);
    }
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("TSR output exceeds Vulkan storage range");
    for (auto& buffer:record.Buffers) if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,frame.CaptureDiagnostics?host:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buffer)) return false;
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create TSR descriptor pool")) return false;
    VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&TSRSet;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&set),"Allocate TSR descriptor set")) return false;
    const RawBuffer* bindings[]{&work->Sources[0],&work->Sources[1],constants.Options.X?&old->Buffers[0]:&work->Sources[2],&record.Buffers[0],&record.Buffers[1]};
    std::array<VkDescriptorBufferInfo,5> buffers{}; std::array<VkWriteDescriptorSet,5> writes{};
    for (unsigned slot=0;slot<writes.size();++slot)
    { buffers[slot]={bindings[slot]->Buffer,0,bindings[slot]->Size}; writes[slot]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[slot].dstSet=set; writes[slot].dstBinding=slot; writes[slot].descriptorCount=1; writes[slot].descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[slot].pBufferInfo=&buffers[slot]; }
    UpdateDescriptorSets(Device,static_cast<unsigned>(writes.size()),writes.data(),0,nullptr); TSRPending=work;
    if (!BeginImmediate()) return false;
    if (!PackTSRImages(frame,images,*work)) return false;
    std::array<VkBufferMemoryBarrier,3> visible{};
    for (unsigned slot=0;slot<visible.size();++slot)
    { auto& barrier=visible[slot]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=slot?VK_ACCESS_SHADER_READ_BIT:VK_ACCESS_UNIFORM_READ_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=bindings[slot]->Buffer; barrier.size=bindings[slot]->Size; }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,static_cast<unsigned>(visible.size()),visible.data(),0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TSRPipeline); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,TSRLayout,0,1,&set,0,nullptr); TSRDispatch(CommandBuffer,groupsX,groupsY,1);
    std::array<VkBufferMemoryBarrier,2> completed{};
    for (unsigned slot=0;slot<completed.size();++slot)
    { auto& barrier=completed[slot]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT|VK_ACCESS_SHADER_READ_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=record.Buffers[slot].Buffer; barrier.size=bytes; }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,static_cast<unsigned>(completed.size()),completed.data(),0,nullptr);
    if (!EndImmediate()) return false;
    TSRPending.reset(); TSRResult candidate; candidate.HistoryUsed=constants.Options.X!=0; candidate.Info=record.Info; if(frame.CaptureDiagnostics){candidate.Readback.History.resize(bytes/sizeof(Float4)); candidate.Readback.Auxiliary.resize(bytes/sizeof(Float4));
    std::memcpy(candidate.Readback.History.data(),record.Buffers[0].Mapped,bytes); std::memcpy(candidate.Readback.Auxiliary.data(),record.Buffers[1].Mapped,bytes);
}
    if (!ExportTSR(work->Candidate,candidate.Readback,frame.CaptureDiagnostics) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,candidate.Readback,LastError))) return false;
    candidate.Handle=NewTSRHistoryHandle(); if (!candidate.Handle) return Fail("TSR history handle space is exhausted");
    TSRHistories.emplace(candidate.Handle.Value,work->Candidate); output=std::move(candidate); return true;
}
struct TSRDisplayWork
{
    Impl& Owner; int Width=0,Height=0; bool Submitted=false; FrameCapture Capture;
    std::shared_ptr<TSRNativeHistory> Source; VkImageLayout SourceLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    RawImage Color{}; RawBuffer Readback{}; std::array<RawImage,4> HDR{}; std::array<RawImage,2> Post{};
    std::array<VkFramebuffer,4> HdrFBO{}; std::array<VkFramebuffer,2> PostFBO{}; VkFramebuffer Resolve=0; VkSampler Sampler=0;
    std::size_t Uniform=0; std::uint32_t Descriptor=0;
    explicit TSRDisplayWork(Impl& owner):Owner(owner) {}
    ~TSRDisplayWork()
    {
        for (auto frame:HdrFBO) if (frame) Owner.DestroyFramebufferFunction(Owner.Device,frame,nullptr);
        for (auto frame:PostFBO) if (frame) Owner.DestroyFramebufferFunction(Owner.Device,frame,nullptr);
        if (Resolve) Owner.DestroyFramebufferFunction(Owner.Device,Resolve,nullptr); if (Sampler) Owner.DestroySamplerFunction(Owner.Device,Sampler,nullptr);
        for (auto& image:HDR) Owner.DestroyRawImage(image); for (auto& image:Post) Owner.DestroyRawImage(image); Owner.DestroyRawImage(Color); Owner.DestroyRawBuffer(Readback);
    }
    void Swap()
    {
        std::swap(Width,Owner.CreateInfo.Width); std::swap(Height,Owner.CreateInfo.Height); std::swap(Capture,Owner.PendingFrame); std::swap(Color,Owner.MainColor);
        HDR.swap(Owner.HdrImages); Post.swap(Owner.PostImages); HdrFBO.swap(Owner.HdrFramebuffers); PostFBO.swap(Owner.PostFramebuffers); std::swap(Resolve,Owner.HdrResolveFramebuffer); std::swap(Sampler,Owner.PostSampler);
        std::swap(Uniform,Owner.UniformOffset); std::swap(Descriptor,Owner.DescriptorSetIndex);
    }
};
bool PrepareTSRScene(const TSRSceneFrame& requested,const Techniques::PostProcessSettings& post)
{
    const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value);
    const auto found=TSRHistories.find(TSRScene.Committed.Value);
    return TSRScene.Begin(requested,CreateInfo.Width,CreateInfo.Height,post,found==TSRHistories.end()?nullptr:&found->second->Info,LastError);
}
void AbortTSRScene() { const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value); }
void CommitTSRScene() { const auto old=TSRScene.Commit(); if (old) TSRHistories.erase(old.Value); }
bool RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& requested,FrameCapture& output)
{
    const auto found=TSRHistories.find(handle.Value); if (found==TSRHistories.end()) return Fail("Vulkan TSR display history is invalid");
    Techniques::PostProcessSettings settings; if (!BuildTSRDisplaySettings(requested,settings,LastError) || !Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover TSR display")) return false;
    TSRPendingDisplay.reset(); auto work=std::make_shared<TSRDisplayWork>(*this); work->Source=found->second; work->SourceLayout=work->Source->Color.Layout; work->Width=work->Source->Info.Width; work->Height=work->Source->Info.Height;
    if (!EnsurePostRenderPass() || !CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,HdrRenderPass) ||
        !CreateRawImage(work->Width,work->Height,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,work->Color) ||
        !CreateRawBuffer(VkDeviceSize(work->Width)*work->Height*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Readback) ||
        !CreateFramebuffer(PostRenderPass,&work->Color.View,1,work->Width,work->Height,work->Resolve)) return false;
    for (unsigned image=0;image<work->HDR.size();++image)
    {
        const auto width=image>=2?(work->Width+1)/2:work->Width,height=image>=2?(work->Height+1)/2:work->Height;
        if (!CreateRawImage(width,height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,work->HDR[image]) ||
            !CreateFramebuffer(HdrRenderPass,&work->HDR[image].View,1,width,height,work->HdrFBO[image])) return false;
    }
    {
        struct Restore { TSRDisplayWork& Work; explicit Restore(TSRDisplayWork& work):Work(work){Work.Swap();} ~Restore(){Work.Swap(); if (!Work.Submitted) Work.Source->Color.Layout=Work.SourceLayout;} } restore(*work);
        if (!BeginImmediate()) return false;
        auto& source=work->Source->Color; TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={source.Width,source.Height,1}; CmdCopyImage(CommandBuffer,source.Image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,HdrImages[0].Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        RawImage* color=&MainColor; if (!ApplyHdr(settings.HDR) || !ApplyPostProcess(settings,color)) return false; if(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics){if(!PublishGpuColor(*color))return false;}else CopyImageToReadback(*color,work->Readback);
        if (!Check(EndCommandBuffer(CommandBuffer),"Close TSR display") || !Check(ResetFences(Device,1,&FrameFence),"Reset TSR display fence")) return false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer; TSRPendingDisplay=work;
        AttachGpuSignal(submit);
        const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence); if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit TSR display"); } work->Submitted=true;
        if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Complete TSR display")) return false;
        CompleteGpuSignal();
        PendingFrame.Width=work->Source->Info.Width; PendingFrame.Height=work->Source->Info.Height; if(!(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics)){PendingFrame.ColorRGBA8.resize(std::size_t(PendingFrame.Width)*PendingFrame.Height*4); std::memcpy(PendingFrame.ColorRGBA8.data(),work->Readback.Mapped,PendingFrame.ColorRGBA8.size());} PendingFrame.Statistics.PassCount=PendingFrame.Statistics.PostProcessPassCount;
    }
    TSRPendingDisplay.reset(); output=std::move(work->Capture); return true;
}
bool SnapshotTSROpaque()
{
    if (!TSRScene.Active || TSRScene.OpaqueReady || !OpaqueEffectsComplete() || !(DeferredActive?DeferredResolved:ForwardOpaqueComplete) || (FrameSampleCount>1 && !SamplesOpaqueResolved)) return Fail("TSR snapshot requires completed opaque effects exactly once");
    if (TSROpaque.Width!=MainColor.Width || TSROpaque.Height!=MainColor.Height) { DestroyRawImage(TSROpaque); if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,TSROpaque)) return false; }
    auto& source=HdrImages[0]; const auto original=source.Layout; TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,TSROpaque,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={source.Width,source.Height,1}; CmdCopyImage(CommandBuffer,source.Image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,TSROpaque.Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy); TransitionImage(CommandBuffer,source,original); TSRScene.OpaqueReady=true; return true;
}
bool FinishTSRScene()
{
    if (!TSRScene.Active) return true;
    TSRScene.MotionReady=VelocityCaptured && (FrameSampleCount==1 || SamplesVelocityResolved);
    if (!TSRScene.OpaqueReady || !TSRScene.MotionReady) return Fail("TSR presentation requires opaque snapshot and resolved physical motion");
    struct Restore { bool& Value; explicit Restore(bool& value):Value(value){Value=false;} ~Restore(){Value=true;} } restore(FrameOpen);
    TSRNativeImages images; images.Flags=TSRTextureMotion; images.Images[TSRSceneFinal]=&HdrImages[0]; images.Images[TSRSceneOpaque]=&TSROpaque; images.Images[TSRSceneMotion]=&MotionImages[MotionVelocity]; images.Images[TSRSceneDepth]=&MainDepth;
    TSRResult candidate; if (!ReconstructTSRImages(TSRScene.Frame,images,TSRScene.Committed,candidate)) return false; TSRScene.Pending=candidate.Handle;
    FrameCapture display; return RenderTSRDisplay(candidate.Handle,FramePostProcessing,display) && ComposeTSRSceneCapture(TSRScene.Frame,candidate,std::move(display),PendingFrame,LastError);
}
