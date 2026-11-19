struct LumenSurfaceRecord
{
    Impl& Owner; LumenSurfaceIdentity Identity; LumenSurfaceLayout Layout;
    std::vector<LumenSurfaceCapture> Captures;
    std::array<RawImage,LumenSurfacePlanes> Images{}; RawImage Depth; VkFramebuffer Framebuffer=0;
    explicit LumenSurfaceRecord(Impl& owner):Owner(owner) {}
    ~LumenSurfaceRecord()
    { if (Framebuffer) Owner.DestroyFramebufferFunction(Owner.Device,Framebuffer,nullptr); for (auto& image:Images) Owner.DestroyRawImage(image); Owner.DestroyRawImage(Depth); }
};
struct LumenSurfaceWork
{
    Impl& Owner; RawBuffer Uniform,Feedback; VkDescriptorPool Pool=0; VkDeviceSize UniformStride=0;
    std::vector<VkDescriptorSet> Sets; std::vector<RawImage*> MaterialImages;
    std::shared_ptr<LumenSurfaceRecord> Candidate;
    explicit LumenSurfaceWork(Impl& owner):Owner(owner) {}
    ~LumenSurfaceWork() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Feedback); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenSurfaceRecord>> LumenSurfaces;
std::shared_ptr<LumenSurfaceWork> LumenSurfacePending;
VkDescriptorSetLayout LumenSurfaceSet=0; VkPipelineLayout LumenPipelineLayout=0; VkRenderPass LumenSurfacePass=0;
std::array<VkPipeline,4> LumenCapturePipelines{}; VkPipeline LumenMaterialRead=0;
std::array<VkSampler,2> LumenSamplers{}; RawImage LumenWhite;
PFN_vkCreateComputePipelines LumenCreateCompute=nullptr; PFN_vkCmdDispatch LumenDispatch=nullptr;
bool RecoverLumenSurface()
{
    if (!LumenSurfacePending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen material capture")) return false;
    LumenSurfacePending.reset(); return true;
}
void DestroyLumenSurfaces()
{
    DestroyLumenCaptureAsync();
    LumenSurfacePending.reset(); LumenSurfaces.clear(); DestroyRawImage(LumenWhite);
    for (auto& pipeline:LumenCapturePipelines) { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (LumenMaterialRead) DestroyPipelineFunction(Device,LumenMaterialRead,nullptr); LumenMaterialRead=0;
    for (auto& sampler:LumenSamplers) { if (sampler) DestroySamplerFunction(Device,sampler,nullptr); sampler=0; }
    if (LumenSurfacePass) DestroyRenderPassFunction(Device,LumenSurfacePass,nullptr); LumenSurfacePass=0;
    if (LumenPipelineLayout) DestroyPipelineLayoutFunction(Device,LumenPipelineLayout,nullptr); LumenPipelineLayout=0;
    if (LumenSurfaceSet) DestroyDescriptorSetLayoutFunction(Device,LumenSurfaceSet,nullptr); LumenSurfaceSet=0;
}
bool EnsureLumenSurface()
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxColorAttachments<LumenSurfacePlanes || DeviceLimits.maxPerStageDescriptorSamplers<LumenSurfacePlanes ||
        DeviceLimits.maxPerStageDescriptorSampledImages<LumenSurfacePlanes || DeviceLimits.maxPerStageDescriptorStorageBuffers<1 ||
        DeviceLimits.maxUniformBufferRange<sizeof(LumenSurfaceConstants) || DeviceLimits.maxComputeWorkGroupInvocations<LumenSurfaceReadThreads ||
        DeviceLimits.maxComputeWorkGroupSize[0]<LumenSurfaceReadThreads) return Fail("Vulkan Lumen material capture exceeds native descriptor/compute limits");
    if (!LoadDevice(LumenCreateCompute,"vkCreateComputePipelines") || !LoadDevice(LumenDispatch,"vkCmdDispatch")) return false;
    if (!LumenSurfaceSet)
    {
        const VkFlags all=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT,images=VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT;
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,all,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,images,nullptr},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,images,nullptr},
            {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,images,nullptr},{4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=5; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenSurfaceSet),"Create Lumen material descriptor layout")) return false;
    }
    if (!LumenPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenSurfaceSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenPipelineLayout),"Create Lumen material pipeline layout")) return false;
    }
    if (!LumenSurfacePass)
    {
        std::array<VkAttachmentDescription,LumenSurfacePlanes+1> attachments{}; std::array<VkAttachmentReference,LumenSurfacePlanes> colors{};
        for (unsigned slot=0;slot<attachments.size();++slot)
        {
            const bool depth=slot==LumenSurfacePlanes; auto& attachment=attachments[slot];
            attachment.format=depth?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R32G32B32A32_SFLOAT; attachment.samples=VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout=attachment.finalLayout=depth?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            if (!depth) colors[slot]={slot,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const VkAttachmentReference depth{LumenSurfacePlanes,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=LumenSurfacePlanes; subpass.pColorAttachments=colors.data(); subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; create.attachmentCount=unsigned(attachments.size()); create.pAttachments=attachments.data();
        create.subpassCount=1; create.pSubpasses=&subpass; create.dependencyCount=1; create.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&create,nullptr,&LumenSurfacePass),"Create Lumen material render pass")) return false;
    }
    for (unsigned slot=0;slot<LumenSamplers.size();++slot) if (!LumenSamplers[slot])
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.minFilter=create.magFilter=slot?VK_FILTER_NEAREST:VK_FILTER_LINEAR;
        create.addressModeU=create.addressModeV=create.addressModeW=slot?VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:VK_SAMPLER_ADDRESS_MODE_REPEAT;
        create.maxLod=0; create.maxAnisotropy=1;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&LumenSamplers[slot]),"Create Lumen material sampler")) return false;
    }
    if (!LumenWhite.Image)
    {
        const std::uint8_t white[]{255,255,255,255};
        if (!CreateRawImage(1,1,VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,LumenWhite) ||
            !InitializeImage(LumenWhite,white,sizeof(white))) { DestroyRawImage(LumenWhite); return false; }
    }
    if (LumenMaterialRead && std::all_of(LumenCapturePipelines.begin(),LumenCapturePipelines.end(),[](auto pipeline){return pipeline!=0;})) return true;
    struct Modules
    {
        Impl& Owner; std::array<VkShaderModule,3> Values{};
        explicit Modules(Impl& owner):Owner(owner) {}
        ~Modules(){for(auto module:Values) if(module) Owner.DestroyShaderModuleFunction(Owner.Device,module,nullptr);}
    } modules(*this);
    for (unsigned stage=0;stage<modules.Values.size();++stage)
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenSurfaceShaders[stage],modules.Values[stage],LastError)) return false;
    const VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{{0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Position)},
        {1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Normal)},{2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex,Color)},{3,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,TexCoord)}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding;
    input.vertexAttributeDescriptionCount=4; input.pVertexAttributeDescriptions=attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=depth.depthWriteEnable=VK_TRUE_VALUE; depth.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
    std::array<VkPipelineColorBlendAttachmentState,LumenSurfacePlanes> attachments{}; for (auto& attachment:attachments) attachment.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=LumenSurfacePlanes; blend.pAttachments=attachments.data();
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
    VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,modules.Values[0],"main",nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,modules.Values[1],"main",nullptr}};
    VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; create.stageCount=2; create.pStages=stages;
    create.pVertexInputState=&input; create.pInputAssemblyState=&assembly; create.pViewportState=&viewport; create.pRasterizationState=&raster;
    create.pMultisampleState=&samples; create.pDepthStencilState=&depth; create.pColorBlendState=&blend; create.pDynamicState=&dynamic;
    create.layout=LumenPipelineLayout; create.renderPass=LumenSurfacePass; create.basePipelineIndex=-1;
    for (unsigned mode=0;mode<LumenCapturePipelines.size();++mode) if (!LumenCapturePipelines[mode])
    {
        raster.cullMode=mode>=2?CullBackFaces:VK_CULL_MODE_NONE; raster.frontFace=(mode&1)?VK_FRONT_FACE_CLOCKWISE:VK_FRONT_FACE_COUNTER_CLOCKWISE;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&LumenCapturePipelines[mode]),"Create Lumen material capture pipeline")) return false;
    }
    if (!LumenMaterialRead)
    {
        VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; compute.layout=LumenPipelineLayout; compute.basePipelineIndex=-1;
        compute.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,modules.Values[2],"main",nullptr};
        if (!Check(Evans::Renderer::CreateCachedVulkanComputePipelines(LumenCreateCompute,Device,0,1,&compute,nullptr,&LumenMaterialRead),"Create Lumen material export pipeline")) return false;
    }
    return true;
}
bool CreateLumenWork(LumenSurfaceWork& work,unsigned sets,bool readback)
{
    work.UniformStride=AlignUp(sizeof(LumenSurfaceConstants),std::max<VkDeviceSize>(16,DeviceLimits.minUniformBufferOffsetAlignment));
    constexpr auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto& layout=work.Candidate->Layout; const auto bytes=readback?std::size_t(layout.Width)*layout.Height*sizeof(LumenSurfaceTexel):sizeof(Float4);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen material export exceeds its storage range");
    if (!CreateRawBuffer(work.UniformStride*sets,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work.Uniform) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work.Feedback)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,sets},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,sets*LumenSurfacePlanes},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,sets}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=sets; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work.Pool),"Create Lumen material descriptor pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(sets,LumenSurfaceSet); work.Sets.resize(sets);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work.Pool; allocate.descriptorSetCount=sets; allocate.pSetLayouts=layouts.data();
    return Check(AllocateDescriptorSets(Device,&allocate,work.Sets.data()),"Allocate Lumen material descriptor sets");
}
void WriteLumenSet(LumenSurfaceWork& work,unsigned index,const void* constants,std::size_t bytes,const std::array<RawImage*,LumenSurfacePlanes>& images,bool read)
{
    const auto offset=work.UniformStride*index; auto* data=static_cast<std::uint8_t*>(work.Uniform.Mapped)+offset;
    std::memset(data,0,sizeof(LumenSurfaceConstants)); std::memcpy(data,constants,bytes);
    const VkDescriptorBufferInfo uniform{work.Uniform.Buffer,offset,sizeof(LumenSurfaceConstants)},feedback{work.Feedback.Buffer,0,work.Feedback.Size};
    VkDescriptorImageInfo sources[LumenSurfacePlanes];
    for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) sources[slot]={LumenSamplers[read?1:0],images[slot]?images[slot]->View:LumenWhite.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[5]{};
    for (unsigned slot=0;slot<5;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work.Sets[index]; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==4?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (slot==0) write.pBufferInfo=&uniform; else if (slot==4) write.pBufferInfo=&feedback; else write.pImageInfo=&sources[slot-1];
    }
    UpdateDescriptorSets(Device,5,writes,0,nullptr);
}
void LumenUniformBarrier(const LumenSurfaceWork& work)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=work.Uniform.Buffer; barrier.size=work.Uniform.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
}
bool EncodeLumenRead(const LumenSurfaceWork& work,unsigned set)
{
    const auto count=std::size_t(work.Candidate->Layout.Width)*work.Candidate->Layout.Height;
    const auto groups=unsigned((count+LumenSurfaceReadThreads-1)/LumenSurfaceReadThreads),x=std::min(groups,LumenSurfaceReadColumns),y=(groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen export dispatch exceeds native groups");
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenMaterialRead); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenPipelineLayout,0,1,&work.Sets[set],0,nullptr);
    LumenDispatch(CommandBuffer,x,y,1);
    VkBufferMemoryBarrier read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; read.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    read.srcQueueFamilyIndex=read.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; read.buffer=work.Feedback.Buffer; read.size=work.Feedback.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&read,0,nullptr); return true;
}
bool CompleteLumenWork(const std::shared_ptr<LumenSurfaceWork>& work,std::vector<LumenSurfaceTexel>* output)
{
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen material commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen material fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen material capture"); }
    LumenSurfacePending=work; for (auto* image:work->MaterialImages) image->Layout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (!RecoverLumenSurface()) return false;
    if (output)
    {
        std::vector<LumenSurfaceTexel> result(std::size_t(work->Feedback.Size/sizeof(LumenSurfaceTexel)));
        std::memcpy(result.data(),work->Feedback.Mapped,result.size()*sizeof(LumenSurfaceTexel));
        if (!ValidateLumenSurfaceTexels(work->Candidate->Layout,result,LastError)) return false; *output=std::move(result);
    }
    return true;
}
bool CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    LumenSurfaceLayout layout; if (!ValidateLumenSurfaceUpload(upload,layout,LastError) || !EnsureLumenSurface()) return false;
    if (layout.Width>DeviceLimits.maxImageDimension2D || layout.Height>DeviceLimits.maxImageDimension2D) return Fail("Vulkan Lumen material atlas exceeds image dimensions");
    auto work=std::make_shared<LumenSurfaceWork>(*this); unsigned commands=1;
    for (const auto& card:upload.Cards) commands+=card.DrawCount;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        {
            const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid");
            if (std::find(work->MaterialImages.begin(),work->MaterialImages.end(),&found->second.Image)==work->MaterialImages.end()) work->MaterialImages.push_back(&found->second.Image);
        }
    }
    work->Candidate=std::make_shared<LumenSurfaceRecord>(*this); auto& record=*work->Candidate; record.Layout=layout; record.Identity=upload.Identity; record.Captures=upload.Cards;
    std::array<VkImageView,LumenSurfacePlanes+1> views;
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        if (!CreateRawImage(layout.Width,layout.Height,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,record.Images[plane])) return false;
        views[plane]=record.Images[plane].View;
    }
    if (!CreateRawImage(layout.Width,layout.Height,VK_FORMAT_D32_SFLOAT,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,record.Depth)) return false;
    views[LumenSurfacePlanes]=record.Depth.View;
    if (!CreateFramebuffer(LumenSurfacePass,views.data(),unsigned(views.size()),layout.Width,layout.Height,record.Framebuffer) || !CreateLumenWork(*work,commands,upload.Readback)) return false;
    unsigned next=0;
    for (const auto& card:upload.Cards) for (unsigned index=card.FirstDraw;index<std::uint64_t(card.FirstDraw)+card.DrawCount;++index)
    {
        const auto& draw=upload.Draws[index]; LumenSurfaceConstants constants; if (!BuildLumenSurfaceConstants(layout,card,draw,constants,LastError)) return false;
        WriteLumenSet(*work,next++,&constants,sizeof(constants),{draw.Diffuse?&Textures.at(draw.Diffuse.Value).Image:nullptr,
            draw.Normal?&Textures.at(draw.Normal.Value).Image:nullptr,draw.Opacity?&Textures.at(draw.Opacity.Value).Image:nullptr},false);
    }
    const ShaderUInt4 extent{layout.Width,layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    WriteLumenSet(*work,next,&extent,sizeof(extent),{&record.Images[0],&record.Images[1],&record.Images[2]},true);
    if (!BeginImmediate()) return false; LumenUniformBarrier(*work);
    for (auto* image:work->MaterialImages) { auto temporary=*image; TransitionImage(CommandBuffer,temporary,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); }
    for (const auto& draw:upload.Draws)
    {
        VkBufferMemoryBarrier barriers[2]{}; const BufferHandle handles[]{draw.Vertices,draw.Indices};
        for (unsigned slot=0;slot<2;++slot)
        {
            auto& barrier=barriers[slot]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=slot?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=Buffers.at(handles[slot].Value).Allocation.Buffer; barrier.size=Buffers.at(handles[slot].Value).Allocation.Size;
        }
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,2,barriers,0,nullptr);
    }
    for (auto& image:record.Images) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(CommandBuffer,record.Depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    std::array<VkClearValue,LumenSurfacePlanes+1> clear{}; clear[1].color.float32[3]=clear[2].color.float32[3]=1; clear[LumenSurfacePlanes].depthStencil={1,0};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=LumenSurfacePass; begin.framebuffer=record.Framebuffer;
    begin.renderArea={{0,0},{layout.Width,layout.Height}}; begin.clearValueCount=unsigned(clear.size()); begin.pClearValues=clear.data();
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
    next=0; LumenSurfaceResult result; result.Layout=layout; result.Identity=upload.Identity; const VkDeviceSize offset=0;
    for (unsigned card=0;card<upload.Cards.size();++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const VkViewport viewport{float(x),float(y),float(stride),float(stride),0,1}; const VkRect2D scissor{{std::int32_t(x),std::int32_t(y)},{stride,stride}};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const unsigned raster=(draw.BackFaceCulling?2u:0u)+(RealtimeWorldIsMirrored(draw.World)?1u:0u);
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenCapturePipelines[raster]); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenPipelineLayout,0,1,&work->Sets[next++],0,nullptr);
            CmdBindVertexBuffers(CommandBuffer,0,1,&Buffers.at(draw.Vertices.Value).Allocation.Buffer,&offset); CmdBindIndexBuffer(CommandBuffer,Buffers.at(draw.Indices.Value).Allocation.Buffer,0,VK_INDEX_TYPE_UINT32);
            CmdDrawIndexed(CommandBuffer,draw.IndexCount,1,draw.FirstIndex,0,0); ++result.DrawCalls;
        }
    }
    CmdEndRenderPass(CommandBuffer); for (auto& image:record.Images) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (upload.Readback && !EncodeLumenRead(*work,next)) return false;
    if (!CompleteLumenWork(work,upload.Readback?&result.Texels:nullptr)) return false;
    result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space is exhausted");
    LumenSurfaces.emplace(result.Handle.Value,work->Candidate); output=std::move(result); return true;
}
bool ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Lumen surface handle is invalid or expired");
    if (!EnsureLumenSurface()) return false;
    auto work=std::make_shared<LumenSurfaceWork>(*this); work->Candidate=found->second; auto& record=*work->Candidate;
    if (!CreateLumenWork(*work,1,true)) return false;
    const ShaderUInt4 extent{record.Layout.Width,record.Layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    WriteLumenSet(*work,0,&extent,sizeof(extent),{&record.Images[0],&record.Images[1],&record.Images[2]},true);
    if (!BeginImmediate()) return false; LumenUniformBarrier(*work); if (!EncodeLumenRead(*work,0)) return false; return CompleteLumenWork(work,&output);
}
bool DestroyLumenSurface(LumenSurfaceHandle handle)
{ return (handle && LumenSurfaces.erase(handle.Value)) || Fail("Lumen surface handle is invalid or already released"); }
