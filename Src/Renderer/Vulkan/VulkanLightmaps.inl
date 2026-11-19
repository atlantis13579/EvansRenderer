// VulkanRHI::Impl members. GLSL is compiled at runtime; only CPU irradiance
// and original receiver geometry are uploaded. Each recorded draw owns a set.
static constexpr std::size_t LightmapFront=0,LightmapBack=1,LightmapCovered=2,LightmapResolved=3,LightmapComposite=4;
std::array<RawImage,5> LightmapImages{};
std::array<RawBuffer,2> LightmapUploads{};
RawBuffer LightmapVertices;
VkRenderPass LightmapSurfacePass=0;
VkFramebuffer LightmapSurfaceFB=0,LightmapCompositeFB=0;
VkImageView LightmapDepthView=0;
VkDescriptorSetLayout LightmapDescriptorLayout=0;
VkPipelineLayout LightmapLayout=0;
VkDescriptorPool LightmapPool=0;
std::vector<VkDescriptorSet> LightmapSets;
std::array<VkShaderModule,2> LightmapModules{};
std::array<VkPipeline,8> LightmapSurfacePipelines{};
VkPipeline LightmapCompositePipeline=0;
std::shared_ptr<const LightmapBakeResult> StagedLightmap,UploadedLightmap,PendingUploadedLightmap;
LightmapFrame FrameLightmaps;
Techniques::Settings FrameLightmapSettings;
bool LightmapActive=false;

void DestroyLightmapTargets()
{
    for (auto& framebuffer:{&LightmapSurfaceFB,&LightmapCompositeFB})
    { if (*framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,*framebuffer,nullptr); *framebuffer=0; }
    for (auto& pipeline:LightmapSurfacePipelines)
    { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (LightmapSurfacePass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,LightmapSurfacePass,nullptr);
    LightmapSurfacePass=LightmapDepthView=0;
    for (auto& image:LightmapImages) DestroyRawImage(image);
    for (auto& upload:LightmapUploads) DestroyRawBuffer(upload);
    DestroyRawBuffer(LightmapVertices);
    StagedLightmap.reset(); UploadedLightmap.reset(); PendingUploadedLightmap.reset();
}

void DestroyLightmapState()
{
    DestroyLightmapTargets();
    if (LightmapCompositePipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,LightmapCompositePipeline,nullptr);
    LightmapCompositePipeline=0;
    for (auto& module:LightmapModules)
    { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (LightmapPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,LightmapPool,nullptr);
    if (LightmapLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,LightmapLayout,nullptr);
    if (LightmapDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,LightmapDescriptorLayout,nullptr);
    LightmapPool=LightmapLayout=LightmapDescriptorLayout=0; LightmapSets.clear(); FrameLightmaps={}; LightmapActive=false;
}

bool CreateLightmapSurfacePass()
{
    const bool multisample=FrameSampleCount>1;
    std::array<VkAttachmentDescription,3> attachments{};
    auto& color=attachments[0]; color.format=VK_FORMAT_R16G16B16A16_SFLOAT; color.samples=FrameSampleCount;
    color.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; color.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout=color.finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    auto& depth=attachments[1]; depth.format=VK_FORMAT_D32_SFLOAT; depth.samples=FrameSampleCount;
    depth.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; depth.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; depth.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout=depth.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    attachments[2]=color; attachments[2].samples=1; attachments[2].loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    const VkAttachmentReference colorRef{0,color.initialLayout},depthRef{1,depth.initialLayout},resolveRef{2,color.initialLayout};
    VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount=1; subpass.pColorAttachments=&colorRef; subpass.pDepthStencilAttachment=&depthRef;
    subpass.pResolveAttachments=multisample ? &resolveRef : nullptr;
    VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
    dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
    VkRenderPassCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    create.attachmentCount=multisample ? 3 : 2; create.pAttachments=attachments.data();
    create.subpassCount=1; create.pSubpasses=&subpass; create.dependencyCount=1; create.pDependencies=&dependency;
    return Check(CreateRenderPassFunction(Device,&create,nullptr,&LightmapSurfacePass),"vkCreateRenderPass(lightmap receivers)");
}

bool CreateLightmapPipeline(std::uint32_t variant,bool composite,VkPipeline& output)
{
    if (output) return true;
    for (std::size_t stage=0;stage<LightmapModules.size();++stage)
    {
        if (LightmapModules[stage]) continue;
        std::vector<std::uint32_t> words;
        if (!CompileVulkanShader(VulkanLightmapShaders[stage],words,LastError)) return false;
        VkShaderModuleCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create.codeSize=words.size()*sizeof(std::uint32_t); create.pCode=words.data();
        if (!Check(CreateShaderModuleFunction(Device,&create,nullptr,&LightmapModules[stage]),"vkCreateShaderModule(lightmaps)")) return false;
    }
    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    for (std::size_t stage=0;stage<stages.size();++stage)
    {
        stages[stage].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage].stage=stage==0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[stage].module=LightmapModules[stage]; stages[stage].pName="main";
    }
    const VkVertexInputBindingDescription binding{0,sizeof(LightmapReceiverVertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] {
        {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(LightmapReceiverVertex,Position)},
        {1,0,VK_FORMAT_R32G32_SFLOAT,offsetof(LightmapReceiverVertex,MaterialUV)},
        {2,0,VK_FORMAT_R32G32_SFLOAT,offsetof(LightmapReceiverVertex,AtlasUV)},
        {3,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(LightmapReceiverVertex,Color)}};
    VkPipelineVertexInputStateCreateInfo input{}; input.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding;
    input.vertexAttributeDescriptionCount=static_cast<std::uint32_t>(std::size(attributes)); input.pVertexAttributeDescriptions=attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{}; assembly.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{}; viewport.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{}; raster.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1;
    raster.cullMode=!composite && (variant&1) ? CullBackFaces : VK_CULL_MODE_NONE;
    // Match opaque rasterization, including the shader's clip-Y convention.
    raster.frontFace=(variant&2) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisample{}; multisample.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples=composite ? 1 : FrameSampleCount;
    VkPipelineDepthStencilStateCreateInfo depth{}; depth.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable=composite ? VK_FALSE_VALUE : VK_TRUE_VALUE; depth.depthCompareOp=VK_COMPARE_OP_EQUAL; depth.maxDepthBounds=1;
    VkPipelineColorBlendAttachmentState color{};
    color.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    if (!composite && (variant&4))
    {
        color.blendEnable=VK_TRUE_VALUE; color.srcColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        color.dstColorBlendFactor=VK_BLEND_FACTOR_ONE; color.colorBlendOp=VK_BLEND_OP_ADD;
        color.srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE; color.dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO; color.alphaBlendOp=VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo blend{}; blend.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount=1; blend.pAttachments=&color;
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{}; dynamic.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount=static_cast<std::uint32_t>(std::size(dynamicStates)); dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    create.stageCount=static_cast<std::uint32_t>(stages.size()); create.pStages=stages.data();
    create.pVertexInputState=&input; create.pInputAssemblyState=&assembly; create.pViewportState=&viewport;
    create.pRasterizationState=&raster; create.pMultisampleState=&multisample; create.pDepthStencilState=&depth;
    create.pColorBlendState=&blend; create.pDynamicState=&dynamic; create.layout=LightmapLayout;
    create.renderPass=composite ? HdrRenderPass : LightmapSurfacePass;
    return Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&output),"vkCreateGraphicsPipelines(lightmaps)");
}

bool EnsureLightmapResources()
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) return true;
    const auto& source=*FrameLightmaps.Source;
    if (!HdrActive || !source.Receivers || !source.Irradiance || !source.Irradiance->Atlas || source.Receivers->Vertices.size()<3)
        return Fail("Vulkan lightmaps require HDR and a complete bake snapshot");
    const auto side=source.Irradiance->Atlas->Settings.Resolution;
    const auto& depth=FrameSampleCount>1 ? Multisample.Depth : MainDepth;
    if (LightmapImages[LightmapFront].Width!=side || LightmapImages[LightmapCovered].Width!=MainColor.Width ||
        LightmapImages[LightmapCovered].Height!=MainColor.Height || LightmapImages[LightmapCovered].Samples!=FrameSampleCount ||
        LightmapDepthView!=depth.View) DestroyLightmapTargets();
    if (!LightmapDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,8> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t index=2;index<bindings.size();++index) bindings[index]={index,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LightmapDescriptorLayout),"vkCreateDescriptorSetLayout(lightmaps)")) return false;
    }
    if (!LightmapLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&LightmapDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LightmapLayout),"vkCreatePipelineLayout(lightmaps)")) return false;
    }
    const auto setCount=FrameLightmaps.Draws.size()+1;
    if (setCount>std::numeric_limits<std::uint32_t>::max()/6) return Fail("Vulkan lightmap draw count exceeds descriptor limits");
    if (LightmapSets.size()<setCount)
    {
        if (LightmapPool) DestroyDescriptorPoolFunction(Device,LightmapPool,nullptr);
        LightmapPool=0; LightmapSets.clear(); const auto count=static_cast<std::uint32_t>(setCount);
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,count},{VK_DESCRIPTOR_TYPE_SAMPLER,count},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,count*6}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=count; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LightmapPool),"vkCreateDescriptorPool(lightmaps)")) return false;
        std::vector<VkDescriptorSetLayout> layouts(count,LightmapDescriptorLayout); LightmapSets.resize(count);
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=LightmapPool; allocate.descriptorSetCount=count; allocate.pSetLayouts=layouts.data();
        if (!Check(AllocateDescriptorSets(Device,&allocate,LightmapSets.data()),"vkAllocateDescriptorSets(lightmaps)")) { LightmapSets.clear(); return false; }
    }
    if (!LightmapSurfaceFB)
    {
        constexpr VkFlags TargetUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        for (std::size_t index=0;index<LightmapImages.size();++index)
        {
            const bool atlas=index<=LightmapBack;
            const auto usage=atlas ? VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT :
                TargetUsage|(index==LightmapComposite ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
            if (!CreateRawImage(atlas ? side : MainColor.Width,atlas ? side : MainColor.Height,
                atlas ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,usage,
                LightmapImages[index],1,index==LightmapCovered ? FrameSampleCount : 1)) { DestroyLightmapTargets(); return false; }
        }
        LightmapDepthView=depth.View;
        const VkImageView views[]{LightmapImages[LightmapCovered].View,depth.View,LightmapImages[LightmapResolved].View};
        if (!CreateLightmapSurfacePass() || !CreateFramebuffer(LightmapSurfacePass,views,FrameSampleCount>1 ? 3 : 2,
            MainColor.Width,MainColor.Height,LightmapSurfaceFB) || !CreateFramebuffer(HdrRenderPass,&LightmapImages[LightmapComposite].View,1,
            MainColor.Width,MainColor.Height,LightmapCompositeFB)) { DestroyLightmapTargets(); return false; }
    }
    if (!CreateLightmapPipeline(0,true,LightmapCompositePipeline)) return false;
    for (std::uint32_t variant=0;variant<LightmapSurfacePipelines.size();++variant)
        if (!CreateLightmapPipeline(variant,false,LightmapSurfacePipelines[variant])) return false;
    if (StagedLightmap!=FrameLightmaps.Source)
    {
        const auto& bake=*source.Irradiance;
        if (bake.FrontIrradiance.size()!=static_cast<std::size_t>(side)*side ||
            (!bake.BackIrradiance.empty() && bake.BackIrradiance.size()!=bake.FrontIrradiance.size())) return Fail("Vulkan lightmap upload size is invalid");
        StagedLightmap.reset(); UploadedLightmap.reset();
        for (auto index:{LightmapFront,LightmapBack})
        {
            DestroyRawBuffer(LightmapUploads[index]);
            const auto& texels=index==LightmapFront || bake.BackIrradiance.empty() ? bake.FrontIrradiance : bake.BackIrradiance;
            const auto bytes=texels.size()*sizeof(Float4);
            if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,LightmapUploads[index])) return false;
            std::memcpy(LightmapUploads[index].Mapped,texels.data(),bytes);
        }
        DestroyRawBuffer(LightmapVertices);
        const auto bytes=source.Receivers->Vertices.size()*sizeof(LightmapReceiverVertex);
        if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,LightmapVertices)) return false;
        std::memcpy(LightmapVertices.Mapped,source.Receivers->Vertices.data(),bytes); StagedLightmap=FrameLightmaps.Source;
    }
    return true;
}

bool ResolveLightmaps(const GlobalIlluminationDesc& desc)
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) { GI.Resolved=true; return true; }
    if (UploadedLightmap!=FrameLightmaps.Source)
    {
        for (auto index:{LightmapFront,LightmapBack})
        {
            auto& image=LightmapImages[index]; TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={image.Width,image.Height,1};
            CmdCopyBufferToImage(CommandBuffer,LightmapUploads[index].Buffer,image.Image,image.Layout,1,&copy);
        }
        UploadedLightmap.reset(); PendingUploadedLightmap=FrameLightmaps.Source;
    }
    auto writeSet=[&](std::size_t index,const LightmapConstants& constants,const std::array<RawImage*,6>& inputs,std::uint32_t& dynamicOffset) {
        const auto offset=AlignUp(UniformOffset,UniformAlignment);
        if (offset>UniformBufferSize || sizeof(constants)>UniformBufferSize-offset) return Fail("Vulkan lightmap uniform arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants)); UniformOffset=offset+sizeof(constants);
        dynamicOffset=static_cast<std::uint32_t>(offset);
        const VkDescriptorBufferInfo buffer{UniformBuffer.Buffer,0,sizeof(constants)};
        std::array<VkDescriptorImageInfo,7> images{}; images[0]={MaterialSampler(),0,VK_IMAGE_LAYOUT_UNDEFINED};
        for (std::size_t slot=0;slot<inputs.size();++slot)
        {
            TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            images[slot+1]={0,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        std::array<VkWriteDescriptorSet,8> writes{};
        for (std::uint32_t slot=0;slot<writes.size();++slot)
        {
            auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet=LightmapSets[index]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : slot==1 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (slot==0) write.pBufferInfo=&buffer; else write.pImageInfo=&images[slot-1];
        }
        UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr); return true;
    };
    auto texture=[&](TextureHandle handle)->RawImage* {
        if (!handle) return &DummyColor;
        const auto found=Textures.find(handle.Value);
        return found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? &found->second.Image : nullptr;
    };
    // All transitions and descriptor updates precede the render pass. A separate
    // set per draw prevents a later material from overwriting earlier bindings.
    std::vector<std::uint32_t> offsets(FrameLightmaps.Draws.size());
    for (std::size_t index=0;index<FrameLightmaps.Draws.size();++index)
    {
        const auto& draw=FrameLightmaps.Draws[index]; LightmapConstants constants;
        if (!BuildLightmapConstants(FrameLightmaps,draw,desc.ViewProjection,FrameGI.Intensity,FrameLightmapSettings.Shading,FrameLightmapSettings.Metallic,constants))
            return Fail("Vulkan lightmap receiver uniforms are invalid");
        auto* diffuse=texture(draw.DiffuseTexture); auto* opacity=texture(draw.OpacityTexture);
        if (!diffuse || !opacity) return Fail("Vulkan lightmap receiver references an invalid material texture");
        if (!writeSet(index,constants,{&LightmapImages[LightmapFront],&LightmapImages[LightmapBack],diffuse,opacity,&DummyColor,&DummyColor},offsets[index])) return false;
    }
    auto& covered=LightmapImages[LightmapCovered]; auto& resolved=LightmapImages[LightmapResolved];
    auto& depth=FrameSampleCount>1 ? Multisample.Depth : MainDepth;
    TransitionImage(CommandBuffer,covered,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(CommandBuffer,depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    if (FrameSampleCount>1) TransitionImage(CommandBuffer,resolved,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass=LightmapSurfacePass; begin.framebuffer=LightmapSurfaceFB; begin.renderArea={{0,0},{covered.Width,covered.Height}};
    begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(covered.Width),float(covered.Height),0,1};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    const VkDeviceSize vertexOffset=0; CmdBindVertexBuffers(CommandBuffer,0,1,&LightmapVertices.Buffer,&vertexOffset);
    for (std::size_t index=0;index<FrameLightmaps.Draws.size();++index)
    {
        const auto& draw=FrameLightmaps.Draws[index]; if (!draw.VertexCount) continue;
        const auto variant=(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)|(FrameLightmaps.KeepFirstSurface ? 4u : 0u);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LightmapSurfacePipelines[variant]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LightmapLayout,0,1,&LightmapSets[index],1,&offsets[index]);
        CmdDraw(CommandBuffer,draw.VertexCount,1,draw.FirstVertex,0);
    }
    CmdEndRenderPass(CommandBuffer);
    auto& scene=HdrImages[0]; auto& composite=LightmapImages[LightmapComposite];
    LightmapConstants constants{}; constants.Options.W=static_cast<unsigned>(LightmapPass::Composite); std::uint32_t offset=0;
    constants.Options.X=AmbientOcclusionResolved ? 1u : 0u;
    const auto index=FrameLightmaps.Draws.size();
    if (!writeSet(index,constants,{AmbientOcclusionResolved ? &AmbientOcclusionImages[0] : &DummyColor,
        &LightmapImages[LightmapBack],&DummyColor,&DummyColor,&scene,
        FrameSampleCount>1 ? &resolved : &covered},offset)) return false;
    TransitionImage(CommandBuffer,composite,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin.renderPass=HdrRenderPass; begin.framebuffer=LightmapCompositeFB;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LightmapCompositePipeline);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LightmapLayout,0,1,&LightmapSets[index],1,&offset);
    CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,composite,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,scene,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={scene.Width,scene.Height,1};
    CmdCopyImage(CommandBuffer,composite.Image,composite.Layout,scene.Image,scene.Layout,1,&copy);
    GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=2;
    RecordLightmapDraws(FrameLightmaps,PendingFrame.Statistics); return true;
}
