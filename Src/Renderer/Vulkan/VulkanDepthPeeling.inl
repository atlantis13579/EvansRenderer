// VulkanRHI::Impl members. Descriptor sets are written before recording and
// remain immutable for the frame. Parity selects prior depth and accumulation;
// each encoded draw uses a distinct dynamic uniform slice, never a rewritten UBO.
DepthPeelingFrame FramePeeling;
DepthPeelingSequence PeelingSequence;
std::array<RawImage,5> PeelingImages{}; // D32 pair, RGBA32F pair, RGBA16F layer.
std::array<VkFramebuffer,2> PeelingCaptureFB{},PeelingLayerFB{},PeelingAccumulationFB{};
VkFramebuffer PeelingCompositionFB=0;
VkRenderPass PeelingCapturePass=0,PeelingAccumulationPass=0;
VkDescriptorSetLayout PeelingDescriptorLayout=0;
VkPipelineLayout PeelingLayout=0;
VkDescriptorPool PeelingPool=0;
std::vector<VkDescriptorSet> PeelingSets;
std::array<VkShaderModule,6> PeelingModules{};
std::array<VkPipeline,6> PeelingPipelines{};
RawBuffer PeelingUniforms;
std::size_t PeelingUniformOffset=0;
// Repeated material replays reuse identical immutable bindings and constants;
// the layer budget must not multiply the main frame's descriptor/UBO demand.
std::unordered_map<std::string,std::pair<std::uint32_t,VkDescriptorSet>> PeelingDrawBindings;
void DestroyPeelingTargets()
{
    for (auto* array:{&PeelingCaptureFB,&PeelingLayerFB,&PeelingAccumulationFB})
        for (auto& framebuffer:*array) { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    if (PeelingCompositionFB && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,PeelingCompositionFB,nullptr);
    PeelingCompositionFB=0;
    for (auto& image:PeelingImages) DestroyRawImage(image);
}
void DestroyPeelingState()
{
    DestroyPeelingTargets(); DestroyRawBuffer(PeelingUniforms);
    for (auto& pipeline:PeelingPipelines) { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto& module:PeelingModules) { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    for (auto* pass:{&PeelingCapturePass,&PeelingAccumulationPass}) { if (*pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,*pass,nullptr); *pass=0; }
    if (PeelingPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,PeelingPool,nullptr);
    if (PeelingLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,PeelingLayout,nullptr);
    if (PeelingDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,PeelingDescriptorLayout,nullptr);
    PeelingPool=PeelingLayout=PeelingDescriptorLayout=0;
    PeelingSets.clear(); PeelingDrawBindings.clear(); PeelingSequence.Abort(); FramePeeling={}; PeelingUniformOffset=0;
}

bool CreatePeelingPipeline(std::uint32_t variant)
{
    auto& output=PeelingPipelines[variant]; if (output) return true;
    const bool composite=variant>=4;
    for (std::size_t index=0;index<PeelingModules.size();++index)
        if (!PeelingModules[index] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanOITShaders[index],PeelingModules[index],LastError)) return false;
    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    for (std::size_t index=0;index<stages.size();++index)
    {
        stages[index].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[index].stage=index==0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[index].module=PeelingModules[index+(composite ? (variant-3)*2 : 0)]; stages[index].pName="main";
    }
    const VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{
        {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Position)}, {1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Normal)},
        {2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex,Color)}, {3,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,TexCoord)}};
    VkPipelineVertexInputStateCreateInfo input{}; input.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (!composite) { input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding; input.vertexAttributeDescriptionCount=std::uint32_t(std::size(attributes)); input.pVertexAttributeDescriptions=attributes; }
    VkPipelineInputAssemblyStateCreateInfo assembly{}; assembly.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{}; viewport.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{}; raster.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1; raster.cullMode=!composite && (variant&1) ? CullBackFaces : VK_CULL_MODE_NONE;
    raster.frontFace=(variant&2) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo samples{}; samples.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; samples.rasterizationSamples=1;
    VkPipelineDepthStencilStateCreateInfo depth{}; depth.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable=depth.depthWriteEnable=composite ? VK_FALSE_VALUE : VK_TRUE_VALUE; depth.depthCompareOp=VK_COMPARE_OP_LESS; depth.maxDepthBounds=1;
    std::array<VkPipelineColorBlendAttachmentState,2> attachments{};
    for (auto& attachment:attachments) attachment.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{}; blend.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; blend.attachmentCount=composite ? 1 : 0; blend.pAttachments=attachments.data();
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{}; dynamic.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; create.stageCount=2; create.pStages=stages.data();
    create.pVertexInputState=&input; create.pInputAssemblyState=&assembly; create.pViewportState=&viewport; create.pRasterizationState=&raster;
    create.pMultisampleState=&samples; create.pDepthStencilState=&depth; create.pColorBlendState=&blend; create.pDynamicState=&dynamic;
    create.layout=PeelingLayout; create.renderPass=composite ? (variant==4 ? PeelingAccumulationPass : HdrRenderPass) : PeelingCapturePass;
    return Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&output),"vkCreateGraphicsPipelines(OIT)");
}

bool EnsurePeelingResources()
{
    if (!ValidateDepthPeelingFrame(FramePeeling,LastError)) return false;
    if (!HdrActive || FrameSampleCount!=1 || WeightedActive)
        return Fail("Depth peeling requires single-sample HDR without weighted accumulation");
    if (!PeelingDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,10> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t index=2;index<bindings.size();++index) bindings[index]={index,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; create.bindingCount=std::uint32_t(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&PeelingDescriptorLayout),"vkCreateDescriptorSetLayout(OIT)")) return false;
    }
    if (!PeelingLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; create.setLayoutCount=1; create.pSetLayouts=&PeelingDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&PeelingLayout),"vkCreatePipelineLayout(OIT)")) return false;
    }
    if (!PeelingCapturePass && !CreateShadowRenderPass(true,PeelingCapturePass)) return false;
    if (!PeelingAccumulationPass && !CreateFullscreenRenderPass(VK_FORMAT_R32G32B32A32_SFLOAT,PeelingAccumulationPass)) return false;
    if (PeelingImages[0].Width!=MainColor.Width || PeelingImages[0].Height!=MainColor.Height) DestroyPeelingTargets();
    if (!PeelingCompositionFB)
    {
        for (unsigned index=0;index<PeelingImages.size();++index)
        {
            const bool depth=index<2;
            const auto format=depth ? VK_FORMAT_D32_SFLOAT : index==4 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
            const auto usage=VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|
                (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            if (!CreateRawImage(MainColor.Width,MainColor.Height,format,depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,usage,PeelingImages[index]))
            { DestroyPeelingTargets(); return false; }
        }
        for (unsigned parity=0;parity<2;++parity)
        {
            const VkImageView layer[]{PeelingImages[4].View,PeelingImages[parity].View};
            if (!CreateFramebuffer(PeelingCapturePass,&PeelingImages[parity].View,1,MainColor.Width,MainColor.Height,PeelingCaptureFB[parity]) ||
                !CreateFramebuffer(HdrColorRenderPasses[1],layer,2,MainColor.Width,MainColor.Height,PeelingLayerFB[parity]) ||
                !CreateFramebuffer(PeelingAccumulationPass,&PeelingImages[parity+2].View,1,MainColor.Width,MainColor.Height,PeelingAccumulationFB[parity]))
            { DestroyPeelingTargets(); return false; }
        }
        if (!CreateFramebuffer(HdrRenderPass,&PeelingImages[4].View,1,MainColor.Width,MainColor.Height,PeelingCompositionFB))
        { DestroyPeelingTargets(); return false; }
    }
    for (unsigned variant=0;variant<PeelingPipelines.size();++variant) if (!CreatePeelingPipeline(variant)) return false;
    const auto draws=FramePeeling.Draws.size();
    const auto stride=AlignUp(sizeof(DepthPeelingConstants),UniformAlignment);
    if (draws>(UINT32_MAX/stride)/(FramePeeling.Settings.LayerCount+1)-1 || draws>(UINT32_MAX/8-3)/2)
        return Fail("Vulkan OIT uniform or descriptor capacity exceeded");
    const auto bytes=(draws+1)*(FramePeeling.Settings.LayerCount+1)*stride;
    if (PeelingUniforms.Size<bytes)
    {
        DestroyRawBuffer(PeelingUniforms);
        if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,PeelingUniforms)) return false;
    }
    const auto required=draws*2+3;
    if (PeelingSets.size()<required)
    {
        if (PeelingPool) DestroyDescriptorPoolFunction(Device,PeelingPool,nullptr);
        PeelingPool=0; PeelingSets.clear(); const auto count=std::uint32_t(required);
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,count},{VK_DESCRIPTOR_TYPE_SAMPLER,count},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,count*8}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; create.maxSets=count; create.poolSizeCount=3; create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&PeelingPool),"vkCreateDescriptorPool(OIT)")) return false;
        std::vector<VkDescriptorSetLayout> layouts(count,PeelingDescriptorLayout); PeelingSets.resize(count);
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=PeelingPool; allocate.descriptorSetCount=count; allocate.pSetLayouts=layouts.data();
        if (!Check(AllocateDescriptorSets(Device,&allocate,PeelingSets.data()),"vkAllocateDescriptorSets(OIT)")) { PeelingSets.clear(); return false; }
    }
    auto writeSet=[&](std::size_t index,const std::array<RawImage*,8>& inputs) {
        const VkDescriptorBufferInfo buffer{PeelingUniforms.Buffer,0,sizeof(DepthPeelingConstants)};
        std::array<VkDescriptorImageInfo,9> images{}; images[0]={MaterialSampler(),0,VK_IMAGE_LAYOUT_UNDEFINED};
        for (std::size_t slot=0;slot<inputs.size();++slot) images[slot+1]={0,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkWriteDescriptorSet,10> writes{};
        for (std::uint32_t slot=0;slot<writes.size();++slot)
        {
            auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=PeelingSets[index]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : slot==1 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (slot==0) write.pBufferInfo=&buffer; else write.pImageInfo=&images[slot-1];
        }
        UpdateDescriptorSets(Device,std::uint32_t(writes.size()),writes.data(),0,nullptr);
    };
    auto texture=[&](TextureHandle handle)->RawImage* {
        if (!handle) return &DummyColor; const auto found=Textures.find(handle.Value);
        return found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? &found->second.Image : nullptr;
    };
    for (unsigned parity=0;parity<2;++parity)
    {
        for (std::size_t index=0;index<draws;++index)
        {
            const auto& draw=FramePeeling.Draws[index]; auto* diffuse=texture(draw.DiffuseTexture); auto* opacity=texture(draw.OpacityTexture);
            if (!diffuse || !opacity) return Fail("Invalid Vulkan OIT material texture");
            writeSet(parity*draws+index,{&MainDepth,&PeelingImages[1-parity],diffuse,opacity,&DummyColor,&DummyColor,&DummyColor,&MainDepth});
        }
        writeSet(draws*2+parity,{&DummyColor,&DummyColor,&DummyColor,&DummyColor,&PeelingImages[4],&PeelingImages[parity+2],&HdrImages[0],&PeelingImages[parity]});
    }
    const auto parity=FramePeeling.Settings.LayerCount%2;
    writeSet(draws*2+2,{&DummyColor,&DummyColor,&DummyColor,&DummyColor,&DummyColor,&PeelingImages[parity+2],&HdrImages[0],&PeelingImages[parity]});
    PeelingUniformOffset=0; PeelingDrawBindings.clear();
    return PeelingSequence.Begin(FramePeeling.Settings.LayerCount,LastError);
}

bool UploadPeelingUniforms(const DepthPeelingConstants& value,std::uint32_t& offset)
{
    const auto aligned=AlignUp(PeelingUniformOffset,UniformAlignment);
    if (aligned>PeelingUniforms.Size || sizeof(value)>PeelingUniforms.Size-aligned) return Fail("Vulkan OIT uniform reservation exhausted");
    std::memcpy(static_cast<std::uint8_t*>(PeelingUniforms.Mapped)+aligned,&value,sizeof(value));
    PeelingUniformOffset=aligned+sizeof(value); offset=std::uint32_t(aligned); return true;
}
void BeginPeelingRenderPass(VkRenderPass pass,VkFramebuffer framebuffer,const VkClearValue& clear)
{
    VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass=pass; begin.framebuffer=framebuffer; begin.renderArea={{0,0},{MainColor.Width,MainColor.Height}};
    begin.clearValueCount=1; begin.pClearValues=&clear; CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
}
bool ProcessPeeling(DepthPeelingPass pass)
{
    const auto step=PeelingSequence.Step(); DepthPeelingConstants base;
    if (!BuildDepthPeelingConstants(FramePeeling.ViewProjection,MainColor.Width,MainColor.Height,step.FirstLayer,FramePeeling.Settings.ShowOverflow,base))
        return Fail("Invalid Vulkan OIT frame constants");
    const auto draws=FramePeeling.Draws.size();
    if (pass==DepthPeelingPass::Capture)
    {
        if (!PeelingSequence.Advance(DepthPeelingEvent::BeginCapture,LastError)) return false;
        if (step.FirstLayer)
        {
            VkClearValue initial{}; initial.color.float32[3]=1;
            for (unsigned parity=0;parity<2;++parity)
            {
                TransitionImage(CommandBuffer,PeelingImages[parity+2],VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                BeginPeelingRenderPass(PeelingAccumulationPass,PeelingAccumulationFB[parity],initial); CmdEndRenderPass(CommandBuffer);
            }
            VkClearValue clear{}; clear.depthStencil={PeelingClearDepth,0};
            TransitionImage(CommandBuffer,PeelingImages[step.DepthRead],VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            BeginPeelingRenderPass(PeelingCapturePass,PeelingCaptureFB[step.DepthRead],clear); CmdEndRenderPass(CommandBuffer);
        }
        TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,PeelingImages[step.DepthRead],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        std::vector<std::uint32_t> offsets(draws);
        for (std::size_t index=0;index<draws;++index)
        {
            const auto& draw=FramePeeling.Draws[index]; if (!draw.IndexCount) continue;
            const auto vertex=Buffers.find(draw.VertexBuffer.Value),indices=Buffers.find(draw.IndexBuffer.Value);
            if (vertex==Buffers.end() || indices==Buffers.end()) return Fail("Invalid Vulkan OIT buffers");
            if (!ValidateDepthPeelingBuffers(draw,vertex->second.Description,indices->second.Description,vertex->second.Bytes.size(),indices->second.Bytes,LastError,vertex->second.NaniteGeometry && indices->second.NaniteGeometry)) return false;
            for (auto texture:{draw.DiffuseTexture,draw.OpacityTexture}) if (texture)
            {
                const auto found=Textures.find(texture.Value);
                if (found==Textures.end()) return Fail("Vulkan OIT material texture was removed during a frame");
                TransitionImage(CommandBuffer,found->second.Image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            DepthPeelingConstants value;
            if (!BuildDepthPeelingDrawConstants(base,draw.World,draw.Opacity,draw.AlphaCutoff,bool(draw.DiffuseTexture),bool(draw.OpacityTexture),value) ||
                !UploadPeelingUniforms(value,offsets[index])) return false;
        }
        TransitionImage(CommandBuffer,PeelingImages[step.DepthWrite],VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        VkClearValue clear{}; clear.depthStencil={PeelingClearDepth,0}; BeginPeelingRenderPass(PeelingCapturePass,PeelingCaptureFB[step.DepthWrite],clear);
        const VkDeviceSize zero=0;
        for (std::size_t index=0;index<draws;++index)
        {
            const auto& draw=FramePeeling.Draws[index]; if (!draw.IndexCount) continue;
            const auto& vertices=Buffers.at(draw.VertexBuffer.Value).Allocation; const auto& indices=Buffers.at(draw.IndexBuffer.Value).Allocation;
            CmdBindVertexBuffers(CommandBuffer,0,1,&vertices.Buffer,&zero); CmdBindIndexBuffer(CommandBuffer,indices.Buffer,0,VK_INDEX_TYPE_UINT32);
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PeelingPipelines[(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)]);
            CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PeelingLayout,0,1,&PeelingSets[step.DepthWrite*draws+index],1,&offsets[index]);
            CmdDrawIndexed(CommandBuffer,draw.IndexCount,1,draw.FirstIndex,draw.BaseVertex,0);
        }
        CmdEndRenderPass(CommandBuffer); return PeelingSequence.Advance(DepthPeelingEvent::EndCapture,LastError);
    }
    const bool compose=pass==DepthPeelingPass::Compose;
    if ((!compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedAccumulation) ||
        (compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedComposition)) return Fail("Vulkan OIT resolve is out of order");
    std::uint32_t offset; if (!UploadPeelingUniforms(base,offset)) return false;
    TransitionImage(CommandBuffer,PeelingImages[step.AccumulationRead+2],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,PeelingImages[step.DepthWrite],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!compose) TransitionImage(CommandBuffer,PeelingImages[4],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    auto& output=PeelingImages[compose ? 4 : step.AccumulationWrite+2];
    TransitionImage(CommandBuffer,output,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    BeginPeelingRenderPass(compose ? HdrRenderPass : PeelingAccumulationPass,compose ? PeelingCompositionFB : PeelingAccumulationFB[step.AccumulationWrite],VkClearValue{});
    const auto descriptor=PeelingSets[draws*2+(compose ? 2 : step.AccumulationRead)];
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PeelingPipelines[compose ? 5 : 4]);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PeelingLayout,0,1,&descriptor,1,&offset);
    CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    if (compose)
    {
        auto& scene=HdrImages[0]; TransitionImage(CommandBuffer,output,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,scene,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={scene.Width,scene.Height,1};
        CmdCopyImage(CommandBuffer,output.Image,output.Layout,scene.Image,scene.Layout,1,&copy);
    }
    return PeelingSequence.Advance(compose ? DepthPeelingEvent::Compose : DepthPeelingEvent::Accumulate,LastError);
}
