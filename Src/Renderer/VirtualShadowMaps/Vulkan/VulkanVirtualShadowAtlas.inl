struct VirtualShadowAtlasRecord
{
    Impl& Owner; VirtualShadowSnapshot Info; RawImage Depth,Directory; VkFramebuffer Framebuffer=0;
    explicit VirtualShadowAtlasRecord(Impl& owner):Owner(owner) {}
    ~VirtualShadowAtlasRecord() { if (Framebuffer) Owner.DestroyFramebufferFunction(Owner.Device,Framebuffer,nullptr); Owner.DestroyRawImage(Depth); Owner.DestroyRawImage(Directory); }
};
struct VirtualShadowAtlasWork
{
    Impl& Owner; RawBuffer Uniform,Feedback,Queries,DirectoryUpload; VkDescriptorPool Pool=0; VkDeviceSize UniformStride=0;
    std::vector<VkDescriptorSet> Sets;
    std::vector<RawImage*> CoverageImages;
    std::shared_ptr<VirtualShadowAtlasRecord> Candidate,Parent;
    explicit VirtualShadowAtlasWork(Impl& owner):Owner(owner) {}
    ~VirtualShadowAtlasWork() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Feedback); Owner.DestroyRawBuffer(Queries); Owner.DestroyRawBuffer(DirectoryUpload); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<VirtualShadowAtlasRecord>> VirtualShadowAtlases;
std::shared_ptr<VirtualShadowAtlasWork> VirtualShadowAtlasPending;
VkDescriptorSetLayout VirtualAtlasSet=0; VkPipelineLayout VirtualAtlasLayout=0; VkRenderPass VirtualAtlasPass=0;
std::array<VkPipeline,3> VirtualAtlasRaster{};
VkPipeline VirtualAtlasCopy=0,VirtualAtlasRead=0; std::array<VkSampler,2> VirtualAtlasSamplers{};
RawImage VirtualAtlasWhite;
bool RecoverVirtualShadowAtlas()
{
    if (!VirtualShadowAtlasPending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained virtual atlas work")) return false;
    VirtualShadowAtlasPending.reset(); return true;
}
void DestroyVirtualShadowAtlases()
{
    DestroyVirtualShadowLookup();
    VirtualShadowAtlasPending.reset(); VirtualShadowAtlases.clear(); DestroyRawImage(VirtualAtlasWhite);
    for (auto& pipeline:VirtualAtlasRaster) { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto* pipeline:{&VirtualAtlasCopy,&VirtualAtlasRead}) { if (*pipeline) DestroyPipelineFunction(Device,*pipeline,nullptr); *pipeline=0; }
    for (auto& sampler:VirtualAtlasSamplers) { if (sampler) DestroySamplerFunction(Device,sampler,nullptr); sampler=0; }
    if (VirtualAtlasPass) DestroyRenderPassFunction(Device,VirtualAtlasPass,nullptr); VirtualAtlasPass=0;
    if (VirtualAtlasLayout) DestroyPipelineLayoutFunction(Device,VirtualAtlasLayout,nullptr); VirtualAtlasLayout=0;
    if (VirtualAtlasSet) DestroyDescriptorSetLayoutFunction(Device,VirtualAtlasSet,nullptr); VirtualAtlasSet=0;
}
bool EnsureVirtualShadowAtlas()
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorSamplers<3 || DeviceLimits.maxPerStageDescriptorSampledImages<3 ||
        DeviceLimits.maxPerStageDescriptorStorageBuffers<1 || DeviceLimits.maxUniformBufferRange<sizeof(VirtualShadowRasterConstants) ||
        DeviceLimits.maxComputeWorkGroupInvocations<VirtualShadowGroupWidth || DeviceLimits.maxComputeWorkGroupSize[0]<VirtualShadowGroupWidth)
        return Fail("Vulkan virtual atlas exceeds device compute/descriptor support");
    if (!LoadDevice(VirtualShadowCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(VirtualShadowDispatch,"vkCmdDispatch")) return false;
    if (!VirtualAtlasSet)
    {
        const VkFlags all=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT;
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,all,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=5; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&VirtualAtlasSet),"Create virtual atlas descriptor layout")) return false;
    }
    if (!VirtualAtlasLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&VirtualAtlasSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&VirtualAtlasLayout),"Create virtual atlas pipeline layout")) return false;
    }
    if (!VirtualAtlasPass && !CreateShadowRenderPass(true,VirtualAtlasPass)) return false;
    for (unsigned index=0;index<VirtualAtlasSamplers.size();++index) if (!VirtualAtlasSamplers[index])
    {
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; sampler.minFilter=sampler.magFilter=index?VK_FILTER_NEAREST:VK_FILTER_LINEAR;
        sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=index?VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.maxLod=0; sampler.maxAnisotropy=1;
        if (!Check(CreateSamplerFunction(Device,&sampler,nullptr,&VirtualAtlasSamplers[index]),"Create virtual atlas samplers")) return false;
    }
    if (!VirtualAtlasWhite.Image)
    {
        const std::uint8_t white[]{255,255,255,255};
        if (!CreateRawImage(1,1,VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,VirtualAtlasWhite) ||
            !InitializeImage(VirtualAtlasWhite,white,sizeof(white))) { DestroyRawImage(VirtualAtlasWhite); return false; }
    }
    if (VirtualAtlasRaster[0] && VirtualAtlasRaster[1] && VirtualAtlasRaster[2] && VirtualAtlasCopy && VirtualAtlasRead) return true;
    struct Modules
    {
        Impl& Owner; std::array<VkShaderModule,5> Values{};
        explicit Modules(Impl& owner):Owner(owner) {}
        ~Modules(){for(auto module:Values) if(module) Owner.DestroyShaderModuleFunction(Owner.Device,module,nullptr);}
    } modules(*this);
    for (unsigned stage=0;stage<modules.Values.size();++stage)
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanVirtualAtlasShaders[stage],modules.Values[stage],LastError)) return false;
    const VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{{0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Position)},
        {2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex,Color)},{3,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,TexCoord)}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding; input.vertexAttributeDescriptionCount=3; input.pVertexAttributeDescriptions=attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=depth.depthWriteEnable=VK_TRUE_VALUE;
    depth.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL; depth.maxDepthBounds=1;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
    VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,modules.Values[0],"main",nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,modules.Values[1],"main",nullptr}};
    VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; create.stageCount=2; create.pStages=stages;
    create.pVertexInputState=&input; create.pInputAssemblyState=&assembly; create.pViewportState=&viewport; create.pRasterizationState=&raster;
    create.pMultisampleState=&samples; create.pDepthStencilState=&depth; create.pColorBlendState=&blend; create.pDynamicState=&dynamic;
    create.layout=VirtualAtlasLayout; create.renderPass=VirtualAtlasPass; create.basePipelineIndex=-1;
    for (unsigned cull=0;cull<VirtualAtlasRaster.size();++cull) if (!VirtualAtlasRaster[cull])
    {
        raster.cullMode=cull?CullBackFaces:VK_CULL_MODE_NONE; raster.frontFace=cull==2?VK_FRONT_FACE_CLOCKWISE:VK_FRONT_FACE_COUNTER_CLOCKWISE;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&VirtualAtlasRaster[cull]),"Create virtual atlas raster pipeline")) return false;
    }
    if (!VirtualAtlasCopy)
    {
        stages[0].module=modules.Values[2]; stages[1].module=modules.Values[3]; input.vertexBindingDescriptionCount=input.vertexAttributeDescriptionCount=0;
        raster.cullMode=VK_CULL_MODE_NONE; depth.depthCompareOp=VK_COMPARE_OP_ALWAYS;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&VirtualAtlasCopy),"Create virtual atlas page copy pipeline")) return false;
    }
    if (!VirtualAtlasRead)
    {
        VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; compute.layout=VirtualAtlasLayout; compute.basePipelineIndex=-1;
        compute.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,modules.Values[4],"main",nullptr};
        if (!Check(Evans::Renderer::CreateCachedVulkanComputePipelines(VirtualShadowCreatePipelines,Device,0,1,&compute,nullptr,&VirtualAtlasRead),"Create virtual atlas feedback pipeline")) return false;
    }
    return true;
}
bool CreateVirtualAtlasWork(VirtualShadowAtlasWork& work,unsigned draws,std::size_t feedbackCount)
{
    work.UniformStride=AlignUp(sizeof(VirtualShadowRasterConstants),std::max<VkDeviceSize>(16,DeviceLimits.minUniformBufferOffsetAlignment));
    constexpr auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto feedbackBytes=std::max<std::size_t>(1,feedbackCount)*sizeof(float);
    if (feedbackBytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan virtual atlas feedback exceeds the storage range");
    if (!CreateRawBuffer(work.UniformStride*draws,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work.Uniform) ||
        !CreateRawBuffer(feedbackBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work.Feedback)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,draws},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,draws*3},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,draws}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=draws; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work.Pool),"Create virtual atlas descriptor pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(draws,VirtualAtlasSet); work.Sets.resize(draws);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work.Pool; allocate.descriptorSetCount=draws; allocate.pSetLayouts=layouts.data();
    return Check(AllocateDescriptorSets(Device,&allocate,work.Sets.data()),"Allocate virtual atlas draw sets");
}
void WriteVirtualAtlasDraw(VirtualShadowAtlasWork& work,unsigned index,const VirtualShadowRasterConstants& constants,RawImage* diffuse,RawImage* opacity,RawImage* source)
{
    const auto offset=work.UniformStride*index; std::memcpy(static_cast<std::uint8_t*>(work.Uniform.Mapped)+offset,&constants,sizeof(constants));
    const VkDescriptorBufferInfo uniform{work.Uniform.Buffer,offset,sizeof(constants)},feedback{work.Feedback.Buffer,0,work.Feedback.Size};
    RawImage* images[]{diffuse?diffuse:&VirtualAtlasWhite,opacity?opacity:&VirtualAtlasWhite,source?source:&VirtualAtlasWhite};
    VkDescriptorImageInfo imageInfo[3]{};
    for (unsigned slot=0;slot<3;++slot) imageInfo[slot]={VirtualAtlasSamplers[slot==2?1:0],images[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[5]{};
    for (unsigned slot=0;slot<5;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work.Sets[index]; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==4?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (slot==0) write.pBufferInfo=&uniform; else if (slot==4) write.pBufferInfo=&feedback; else write.pImageInfo=&imageInfo[slot-1];
    }
    UpdateDescriptorSets(Device,5,writes,0,nullptr);
}
bool CompleteVirtualAtlasWork(const std::shared_ptr<VirtualShadowAtlasWork>& work,std::vector<float>* output)
{
    if (!Check(EndCommandBuffer(CommandBuffer),"End virtual atlas commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset virtual atlas fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit virtual atlas commands"); }
    // Gate all later RHI resource mutations until this submission completes.
    // Host metadata follows the commands only after a successful queue submit.
    VirtualShadowAtlasPending=work;
    for (auto* image:work->CoverageImages) image->Layout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (!RecoverVirtualShadowAtlas()) return false;
    if (output)
    {
        std::vector<float> result(static_cast<std::size_t>(work->Feedback.Size/sizeof(float)));
        std::memcpy(result.data(),work->Feedback.Mapped,result.size()*sizeof(float));
        if (!ValidateVirtualShadowAtlasDepth(work->Candidate->Info,result,LastError)) return false; *output=std::move(result);
    }
    return true;
}
void VirtualAtlasUniformBarrier(const VirtualShadowAtlasWork& work)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=work.Uniform.Buffer; barrier.size=work.Uniform.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
}
bool EncodeVirtualDepthPack(const VirtualShadowAtlasWork& work,unsigned width,unsigned height,unsigned set)
{
    const auto count=std::size_t(width)*height;
    const auto groups=static_cast<unsigned>((count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth);
    const auto x=std::min(groups,VirtualShadowGroupColumns),y=(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan virtual atlas read dispatch exceeds native groups");
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualAtlasRead);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualAtlasLayout,0,1,&work.Sets[set],0,nullptr); VirtualShadowDispatch(CommandBuffer,x,y,1);
    VkBufferMemoryBarrier read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; read.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    read.srcQueueFamilyIndex=read.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; read.buffer=work.Feedback.Buffer; read.size=work.Feedback.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&read,0,nullptr); return true;
}
bool EncodeVirtualAtlasRead(const VirtualShadowAtlasWork& work,unsigned set)
{ return EncodeVirtualDepthPack(work,work.Candidate->Info.Layout.AtlasWidth,work.Candidate->Info.Layout.AtlasHeight,set); }
bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Vulkan virtual atlas handle is invalid");
    if (!EnsureVirtualShadowAtlas()) return false;
    auto work=std::make_shared<VirtualShadowAtlasWork>(*this); work->Candidate=found->second; const auto& layout=work->Candidate->Info.Layout;
    if (!CreateVirtualAtlasWork(*work,1,std::size_t(layout.AtlasWidth)*layout.AtlasHeight)) return false;
    VirtualShadowRasterConstants constants; constants.Extent={layout.AtlasWidth,layout.AtlasHeight,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    WriteVirtualAtlasDraw(*work,0,constants,nullptr,nullptr,&work->Candidate->Depth);
    if (!BeginImmediate()) return false; VirtualAtlasUniformBarrier(*work);
    if (!EncodeVirtualAtlasRead(*work,0)) return false; return CompleteVirtualAtlasWork(work,&output);
}
bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{ return (handle && VirtualShadowAtlases.erase(handle.Value)) || Fail("Vulkan virtual atlas handle is invalid or already released"); }
bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!EnsureVirtualShadowAtlas()) return false; auto work=std::make_shared<VirtualShadowAtlasWork>(*this);
    if (previous) { const auto found=VirtualShadowAtlases.find(previous.Value); if (found==VirtualShadowAtlases.end()) return Fail("Vulkan virtual atlas parent is invalid"); work->Parent=found->second; }
    VirtualShadowPlan plan;
    if (!BuildVirtualShadowPlan(upload.Identity,upload.Settings,upload.Requests,upload.Invalidations,work->Parent?&work->Parent->Info:nullptr,plan,LastError)) return false;
    for (const auto& caster:upload.Casters)
    {
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Virtual atlas caster buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        {
            const auto image=Textures.find(texture.Value); if (image==Textures.end() || image->second.Description.Format!=TextureFormat::RGBA8) return Fail("Virtual atlas coverage requires existing color textures");
            if (std::find(work->CoverageImages.begin(),work->CoverageImages.end(),&image->second.Image)==work->CoverageImages.end()) work->CoverageImages.push_back(&image->second.Image);
        }
    }
    const auto count=std::uint64_t(plan.Dirty.size())*upload.Casters.size()+plan.Copies.size()+1;
    if (count>UINT32_MAX/3) return Fail("Virtual atlas descriptor count exceeds native capacity");
    const auto drawCount=static_cast<unsigned>(count); work->Candidate=std::make_shared<VirtualShadowAtlasRecord>(*this);
    auto& record=*work->Candidate; record.Info=plan.Candidate; const auto& layout=record.Info.Layout;
    if (layout.AtlasWidth>DeviceLimits.maxImageDimension2D || layout.AtlasHeight>DeviceLimits.maxImageDimension2D) return Fail("Vulkan virtual atlas exceeds image limits");
    if (!CreateRawImage(layout.AtlasWidth,layout.AtlasHeight,VK_FORMAT_D32_SFLOAT,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,record.Depth) ||
        !CreateFramebuffer(VirtualAtlasPass,&record.Depth.View,1,layout.AtlasWidth,layout.AtlasHeight,record.Framebuffer) ||
        !CreateVirtualAtlasWork(*work,drawCount,upload.Readback?std::size_t(layout.AtlasWidth)*layout.AtlasHeight:0)) return false;
    std::vector<VirtualShadowRasterConstants> frames(drawCount); unsigned next=0;
    for (const auto& copy:plan.Copies)
    { frames[next]=BuildVirtualShadowCopyConstants(work->Parent->Info.Layout,layout,copy); WriteVirtualAtlasDraw(*work,next,frames[next],nullptr,nullptr,&work->Parent->Depth); ++next; }
    for (const auto& tile:plan.Dirty) for (const auto& caster:upload.Casters)
    {
        if (!BuildVirtualShadowRasterConstants(record.Info,tile,caster,frames[next],LastError)) return false;
        WriteVirtualAtlasDraw(*work,next,frames[next],caster.Diffuse?&Textures.at(caster.Diffuse.Value).Image:nullptr,caster.Opacity?&Textures.at(caster.Opacity.Value).Image:nullptr,nullptr); ++next;
    }
    frames[next].Extent={layout.AtlasWidth,layout.AtlasHeight,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0}; WriteVirtualAtlasDraw(*work,next,frames[next],nullptr,nullptr,&record.Depth);
    if (!BeginImmediate()) return false; VirtualAtlasUniformBarrier(*work);
    for (auto* image:work->CoverageImages) { auto view=*image; TransitionImage(CommandBuffer,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); }
    for (const auto& caster:upload.Casters)
    {
        VkBufferMemoryBarrier visible[2]{}; const BufferHandle handles[]{caster.Vertices,caster.Indices};
        for (unsigned slot=0;slot<2;++slot)
        {
            auto& barrier=visible[slot]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask=slot?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=Buffers.at(handles[slot].Value).Allocation.Buffer; barrier.size=Buffers.at(handles[slot].Value).Allocation.Size;
        }
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,2,visible,0,nullptr);
    }
    TransitionImage(CommandBuffer,record.Depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    VkClearValue clear{}; clear.depthStencil={1,0}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass=VirtualAtlasPass; begin.framebuffer=record.Framebuffer; begin.renderArea={{0,0},{layout.AtlasWidth,layout.AtlasHeight}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
    const auto drawFrame=[&](unsigned index)
    {
        const auto& frame=frames[index]; const VkViewport viewport{frame.Origins.X,frame.Origins.Y,float(stride),float(stride),0,1};
        const VkRect2D scissor{{static_cast<std::int32_t>(frame.Origins.X),static_cast<std::int32_t>(frame.Origins.Y)},{stride,stride}};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VirtualAtlasLayout,0,1,&work->Sets[index],0,nullptr);
    };
    next=0; CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VirtualAtlasCopy);
    for (std::size_t copy=0;copy<plan.Copies.size();++copy) { drawFrame(next++); CmdDraw(CommandBuffer,3,1,0,0); }
    VirtualShadowAtlasResult result; const VkDeviceSize offset=0;
    for (const auto& tile:plan.Dirty) for (const auto& caster:upload.Casters)
    {
        (void)tile; drawFrame(next++); const auto raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0;
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VirtualAtlasRaster[raster]);
        CmdBindVertexBuffers(CommandBuffer,0,1,&Buffers.at(caster.Vertices.Value).Allocation.Buffer,&offset);
        CmdBindIndexBuffer(CommandBuffer,Buffers.at(caster.Indices.Value).Allocation.Buffer,0,VK_INDEX_TYPE_UINT32); CmdDrawIndexed(CommandBuffer,caster.IndexCount,1,caster.FirstIndex,0,0); ++result.DrawCalls;
    }
    CmdEndRenderPass(CommandBuffer); TransitionImage(CommandBuffer,record.Depth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (upload.Readback && !EncodeVirtualAtlasRead(*work,next)) return false;
    if (!CompleteVirtualAtlasWork(work,upload.Readback?&result.Depth:nullptr)) return false;
    std::vector<unsigned> rendered; for (const auto& tile:plan.Dirty) rendered.push_back(tile.Page);
    if (!CompleteVirtualShadowPlan(plan,rendered,record.Info,LastError)) return false;
    if (!CreateVirtualShadowDirectory(work->Candidate)) return false;
    result.Handle=NewVirtualShadowAtlasHandle(); if (!result.Handle) return Fail("Virtual atlas identity space is exhausted"); result.Info=record.Info;
    result.CopiedPages=static_cast<unsigned>(plan.Copies.size()); result.RenderedPages=static_cast<unsigned>(plan.Dirty.size());
    VirtualShadowAtlases.emplace(result.Handle.Value,work->Candidate); output=std::move(result); return true;
}
