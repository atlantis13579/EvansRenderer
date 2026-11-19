// The frame retains the immutable atlas until subsequent native work can safely
// replace its bindings. Releasing a public handle cannot invalidate encoded draws.
std::shared_ptr<VirtualShadowAtlasRecord> FrameVirtualShadowAtlas;
VirtualShadowSamplingFrame FrameVirtualShadowSampling;
bool PrepareVirtualShadowSampling(const VirtualShadowSamplingFrame& frame,const CascadedShadowState& cascades)
{
    if (!ValidateVirtualShadowSamplingFrame(frame,LastError)) return false;
    std::shared_ptr<VirtualShadowAtlasRecord> candidate;
    if (frame.Atlas)
    {
        if (cascades.Frame.CascadeCount) return Fail("Virtual and cascaded shadow representations are mutually exclusive");
        if (FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::Hard && FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::PCF)
            return Fail("Virtual scene sampling currently requires hard or PCF filtering");
        const auto found=VirtualShadowAtlases.find(frame.Atlas.Value);
        if (found==VirtualShadowAtlases.end()) return Fail("Virtual scene sampling requires a current atlas handle");
        candidate=found->second;
    }
    FrameVirtualShadowAtlas=std::move(candidate); FrameVirtualShadowSampling=frame; return true;
}
VkDescriptorSetLayout VirtualLookupSet=0; VkPipelineLayout VirtualLookupLayout=0; VkPipeline VirtualLookupPipeline=0;
void DestroyVirtualShadowLookup()
{ FrameVirtualShadowAtlas.reset(); FrameVirtualShadowSampling={};
    if (VirtualLookupPipeline) DestroyPipelineFunction(Device,VirtualLookupPipeline,nullptr); VirtualLookupPipeline=0;
    if (VirtualLookupLayout) DestroyPipelineLayoutFunction(Device,VirtualLookupLayout,nullptr); VirtualLookupLayout=0;
    if (VirtualLookupSet) DestroyDescriptorSetLayoutFunction(Device,VirtualLookupSet,nullptr); VirtualLookupSet=0;
}
bool CreateVirtualShadowDirectory(const std::shared_ptr<VirtualShadowAtlasRecord>& record)
{
    VirtualShadowDirectory directory; if (!BuildVirtualShadowDirectory(record->Info,directory,LastError)) return false;
    auto work=std::make_shared<VirtualShadowAtlasWork>(*this); work->Candidate=record;
    const auto bytes=directory.Depth.size()*sizeof(float);
    if (!CreateRawImage(directory.Width,directory.Height,VK_FORMAT_D32_SFLOAT,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,record->Directory) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->DirectoryUpload)) return false;
    std::memcpy(work->DirectoryUpload.Mapped,directory.Depth.data(),bytes);
    if (!BeginImmediate()) return false;
    TransitionImage(CommandBuffer,record->Directory,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_DEPTH_BIT,0,0,1}; copy.imageExtent={directory.Width,directory.Height,1};
    CmdCopyBufferToImage(CommandBuffer,work->DirectoryUpload.Buffer,record->Directory.Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
    TransitionImage(CommandBuffer,record->Directory,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return CompleteVirtualAtlasWork(work,nullptr);
}
bool EnsureVirtualShadowLookup()
{
    if (!EnsureVirtualShadowAtlas()) return false;
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<2) return Fail("Vulkan virtual lookup needs two storage buffers");
    if (!VirtualLookupSet)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=5; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&VirtualLookupSet),"Create virtual lookup descriptor layout")) return false;
    }
    if (!VirtualLookupLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&VirtualLookupSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&VirtualLookupLayout),"Create virtual lookup pipeline layout")) return false;
    }
    if (!VirtualLookupPipeline)
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanVirtualShadowLookup,module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=VirtualLookupLayout; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(VirtualShadowCreatePipelines,Device,0,1,&create,nullptr,&VirtualLookupPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create virtual lookup pipeline")) return false;
    }
    return true;
}
bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Virtual lookup atlas handle is invalid");
    VirtualShadowLookupConstants constants;
    if (!BuildVirtualShadowLookupConstants(found->second->Info,upload,constants,LastError) || !EnsureVirtualShadowLookup()) return false;
    auto work=std::make_shared<VirtualShadowAtlasWork>(*this); work->Candidate=found->second;
    const auto bytes=upload.Samples.size()*sizeof(Float4); const auto groups=(constants.Filtering.Z+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    const auto x=std::min(groups,VirtualShadowGroupColumns),y=(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns;
    if (bytes>DeviceLimits.maxStorageBufferRange || x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Virtual lookup exceeds Vulkan storage or dispatch limits");
    constexpr auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(constants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Queries) || !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Feedback)) return false;
    std::memcpy(work->Uniform.Mapped,&constants,sizeof(constants)); std::memcpy(work->Queries.Mapped,upload.Samples.data(),bytes);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create virtual lookup descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&VirtualLookupSet;
    VkDescriptorSet set=0; if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"Allocate virtual lookup descriptors")) return false;
    const VkDescriptorBufferInfo uniform{work->Uniform.Buffer,0,sizeof(constants)},query{work->Queries.Buffer,0,bytes},feedback{work->Feedback.Buffer,0,bytes};
    const VkDescriptorImageInfo atlas{VirtualAtlasSamplers[1],work->Candidate->Depth.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        directory{VirtualAtlasSamplers[1],work->Candidate->Directory.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[5]{};
    for (unsigned binding=0;binding<5;++binding) { writes[binding].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[binding].dstSet=set; writes[binding].dstBinding=binding; writes[binding].descriptorCount=1; }
    writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].pBufferInfo=&uniform;
    writes[1].descriptorType=writes[2].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[1].pImageInfo=&atlas; writes[2].pImageInfo=&directory;
    writes[3].descriptorType=writes[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].pBufferInfo=&query; writes[4].pBufferInfo=&feedback;
    UpdateDescriptorSets(Device,5,writes,0,nullptr);
    if (!BeginImmediate()) return false;
    VkBufferMemoryBarrier visible[2]{};
    for (unsigned index=0;index<2;++index)
    {
        auto& barrier=visible[index]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask=index?VK_ACCESS_SHADER_READ_BIT:VK_ACCESS_UNIFORM_READ_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=index?work->Queries.Buffer:work->Uniform.Buffer; barrier.size=index?bytes:sizeof(constants);
    }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,2,visible,0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualLookupPipeline); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualLookupLayout,0,1,&set,0,nullptr); VirtualShadowDispatch(CommandBuffer,x,y,1);
    VkBufferMemoryBarrier read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; read.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    read.srcQueueFamilyIndex=read.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; read.buffer=work->Feedback.Buffer; read.size=bytes;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&read,0,nullptr);
    if (!CompleteVirtualAtlasWork(work,nullptr)) return false;
    VirtualShadowLookupResult result; result.Samples.resize(upload.Samples.size()); std::memcpy(result.Samples.data(),work->Feedback.Mapped,bytes);
    if (!ValidateVirtualShadowLookupResult(work->Candidate->Info,upload,result,LastError)) return false; output=std::move(result); return true;
}
