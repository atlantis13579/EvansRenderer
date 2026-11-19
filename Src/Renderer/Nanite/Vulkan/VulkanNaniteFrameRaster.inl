// Immutable descriptor sets retain GPU inputs through frame submission. The
// visibility render pass is split around compute without any geometry readback.
NaniteFrameRasterWork NaniteRasterPlan;
std::shared_ptr<NaniteFrameBuffer> NaniteRasterDecisions;
NaniteDepthHandle NaniteRasterMainDepth;
NaniteVisibilityHandle NaniteRasterPost;
bool PrepareNaniteRasterVisibility(const VisibilityBufferDesc& frame)
{
    // Previous frame resources are retired only after the frame fence/abort.
    NaniteDepths.erase(NaniteRasterMainDepth.Value); NaniteDecisions.erase(NaniteRasterPost.Value);
    NaniteRasterMainDepth={}; NaniteRasterPost={}; NaniteRasterDecisions.reset();
    if (!NaniteFrameDummy) { const unsigned zero=0; if (!CreateNaniteFrameBuffer(1,&zero,false,NaniteFrameDummy)) return false; }
    NaniteRasterDecisions=NaniteFrameDummy;
    if (!frame.NaniteVisibility) return true;
    const auto found=NaniteDecisions.find(frame.NaniteVisibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Phase!=NaniteVisibilityPhase::Main ||
        found->second->Info.Frame!=FrameNanite.CurrentFrame() || found->second->Info.QueryCount!=frame.NaniteClusters.size() || found->second->Draws)
        return Fail("Nanite raster requires matching unconsumed main-phase cluster decisions");
    NaniteRasterDecisions=found->second->Data; return true;
}
bool RetestNaniteRaster(const VisibilityBufferDesc& frame)
{
    if (!CaptureNaniteDepth(NaniteRasterMainDepth,true) || !RetestNaniteVisibility(frame.NaniteVisibility,NaniteRasterMainDepth,NaniteRasterPost,true)) return false;
    NaniteRasterDecisions=NaniteDecisions.at(NaniteRasterPost.Value)->Data;
    for (auto& job:NaniteRasterPlan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    return true;
}
std::array<VkDescriptorSetLayout,2> NaniteRasterFrameSets{};
std::array<VkPipelineLayout,2> NaniteRasterFrameLayouts{};
std::array<VkPipeline,6> NaniteRasterFramePipelines{};
std::array<VkShaderModule,4> NaniteRasterGraphicsModules{};
std::unordered_map<unsigned,VkPipeline> NaniteRasterGraphics;
struct NaniteRasterFrameResources
{
    Impl& Owner; RawBuffer Uniform; VkDescriptorPool Pool=0;
    std::shared_ptr<NaniteFrameBuffer> Input,Output;
    std::vector<VkDescriptorSet> ComputeSets,GraphicsSets;
    explicit NaniteRasterFrameResources(Impl& owner):Owner(owner) {}
    ~NaniteRasterFrameResources()
    { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); }
};
std::shared_ptr<NaniteRasterFrameResources> NaniteRasterFrameWork;
std::vector<std::shared_ptr<NaniteRasterFrameResources>> NaniteRasterRetainedWork;
VkRenderPass NaniteRasterLoadPass=0;
void DestroyNaniteFrameRaster()
{
    DestroyNaniteHardware();
    NaniteRasterDecisions.reset(); NaniteRasterMainDepth={}; NaniteRasterPost={};
    NaniteRasterFrameWork.reset(); NaniteRasterRetainedWork.clear(); NaniteRasterPlan={};
    if (NaniteRasterLoadPass) DestroyRenderPassFunction(Device,NaniteRasterLoadPass,nullptr); NaniteRasterLoadPass=0;
    for (auto& pipeline:NaniteRasterFramePipelines) { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto& entry:NaniteRasterGraphics) if (entry.second) DestroyPipelineFunction(Device,entry.second,nullptr); NaniteRasterGraphics.clear();
    for (auto& module:NaniteRasterGraphicsModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    for (auto& layout:NaniteRasterFrameLayouts) { if (layout) DestroyPipelineLayoutFunction(Device,layout,nullptr); layout=0; }
    for (auto& set:NaniteRasterFrameSets) { if (set) DestroyDescriptorSetLayoutFunction(Device,set,nullptr); set=0; }
}
VkPipeline NaniteRasterPipeline(bool cull,bool mirrored,bool exportPass)
{
    const unsigned key=unsigned(cull)|unsigned(mirrored)<<1|unsigned(exportPass)<<2|unsigned(bool(NaniteRasterPost))<<3;
    const auto found=NaniteRasterGraphics.find(key); if (found!=NaniteRasterGraphics.end()) return found->second;
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (unsigned i=0;i<2;++i)
    { stages[i]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT,NaniteRasterGraphicsModules[(exportPass?2:0)+i],"main",nullptr}; }
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; raster.polygonMode=VK_POLYGON_MODE_FILL;
    raster.cullMode=cull&&!exportPass?CullBackFaces:VK_CULL_MODE_NONE; raster.frontFace=mirrored?VK_FRONT_FACE_CLOCKWISE:VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=1;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=depth.depthWriteEnable=VK_TRUE_VALUE;
    depth.depthCompareOp=exportPass?VK_COMPARE_OP_ALWAYS:NaniteRasterPost?VK_COMPARE_OP_LESS_OR_EQUAL:VK_COMPARE_OP_LESS; depth.maxDepthBounds=1;
    VkPipelineColorBlendAttachmentState colors[2]{};
    for (auto& color:colors) color.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=2; blend.pAttachments=colors;
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; desc.stageCount=2; desc.pStages=stages;
    desc.pVertexInputState=&vertices; desc.pInputAssemblyState=&assembly; desc.pViewportState=&viewport; desc.pRasterizationState=&raster;
    desc.pMultisampleState=&samples; desc.pDepthStencilState=&depth; desc.pColorBlendState=&blend; desc.pDynamicState=&dynamic;
    desc.layout=NaniteRasterFrameLayouts[1]; desc.renderPass=VisibilityRenderPass;
    VkPipeline pipeline=0; if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&desc,nullptr,&pipeline),"Create Nanite frame graphics pipeline")) return 0;
    NaniteRasterGraphics.emplace(key,pipeline); return pipeline;
}
bool PrepareNaniteFrameRaster(const VisibilityBufferDesc& frame,bool postPreparation=false)
{
    if (!postPreparation)
    { if (!PrepareNaniteRasterVisibility(frame)) return false; NaniteRasterRetainedWork.clear(); }
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 ||
        DeviceLimits.maxComputeWorkGroupInvocations<NaniteRasterThreads || DeviceLimits.maxComputeWorkGroupSize[0]<NaniteRasterThreads)
        return Fail("Vulkan hybrid raster requires compute and three storage buffers");
    if (!LoadDevice(NaniteGeometryCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(NaniteGeometryDispatch,"vkCmdDispatch")) return false;
    NaniteFrameRasterWork plan; if (!BuildNaniteFrameRasterWork(frame,CreateInfo.Width,CreateInfo.Height,plan,LastError)) return false;
    if (postPreparation) for (auto& job:plan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    for (const auto& job:plan.Dispatches)
        if (job.GroupsX>DeviceLimits.maxComputeWorkGroupCount[0] || job.GroupsY>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan hybrid raster exceeds dispatch limits");
    for (unsigned stage=0;stage<2;++stage)
    {
        if (!NaniteRasterFrameSets[stage])
        {
            VkDescriptorSetLayoutBinding bindings[6]{}; const unsigned count=6,images=stage?4:3;
            for (unsigned i=0;i<count;++i)
                bindings[i]={i,i==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(i<images || (!stage && i==5))?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,
                    stage?(i>=images?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT):VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=count; desc.pBindings=bindings;
            if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&NaniteRasterFrameSets[stage]),"Create Nanite frame raster descriptors")) return false;
        }
        if (!NaniteRasterFrameLayouts[stage])
        {
            VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&NaniteRasterFrameSets[stage];
            if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&NaniteRasterFrameLayouts[stage]),"Create Nanite frame raster layout")) return false;
        }
    }
    for (unsigned pass=0;pass<NaniteRasterFramePipelines.size();++pass) if (!NaniteRasterFramePipelines[pass])
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteFrameRasterCompute[pass],module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=NaniteRasterFrameLayouts[0]; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&desc,nullptr,&NaniteRasterFramePipelines[pass]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Nanite frame raster kernel")) return false;
    }
    for (unsigned i=0;i<NaniteRasterGraphicsModules.size();++i)
        if (!NaniteRasterGraphicsModules[i] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteFrameRasterGraphics[i],NaniteRasterGraphicsModules[i],LastError)) return false;
    for (const auto& draw:frame.Draws) if (!NaniteRasterPipeline(draw.Pipeline.BackFaceCulling,frame.Scene->Draws[draw.Range].Mirrored,false)) return false;
    if (!NaniteRasterPipeline(false,false,true)) return false;
    auto work=std::make_shared<NaniteRasterFrameResources>(*this);
    if (postPreparation) { work->Input=NaniteRasterFrameWork->Input; work->Output=NaniteRasterFrameWork->Output; }
    else if (!CreateNaniteFrameBuffer(plan.InputWords,nullptr,true,work->Input) || !CreateNaniteFrameBuffer(plan.OutputWords,nullptr,true,work->Output)) return false;
    const unsigned computes=unsigned(plan.Dispatches.size()),graphics=unsigned(frame.Draws.size())+1,total=computes+graphics;
    const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const auto stride=(sizeof(NaniteFrameRasterConstants)+alignment-1)/alignment*alignment;
    if (!CreateRawBuffer(stride*total,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,total},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,computes*3+graphics*3},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,total*2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=total; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Nanite frame raster pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(total,NaniteRasterFrameSets[0]); std::fill(layouts.begin()+computes,layouts.end(),NaniteRasterFrameSets[1]);
    std::vector<VkDescriptorSet> sets(total); VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=total; allocation.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocation,sets.data()),"Allocate Nanite frame raster sets")) return false;
    for (unsigned i=0;i<total;++i)
    {
        const bool graphicsPass=i>=computes; const unsigned imageStart=graphicsPass?4:3,count=6;
        VkDescriptorBufferInfo buffers[4]{},decisionView{}; VkDescriptorImageInfo images[2]{}; VkWriteDescriptorSet writes[6]{};
        auto* constants=static_cast<std::uint8_t*>(work->Uniform.Mapped)+i*stride;
        buffers[0]={work->Uniform.Buffer,i*stride,graphicsPass?sizeof(NaniteRasterDrawConstants):sizeof(NaniteFrameRasterConstants)};
        const auto describe=[](const RawBuffer& buffer) { return VkDescriptorBufferInfo{buffer.Buffer,0,buffer.Size}; };
        if (!graphicsPass)
        {
            const auto& job=plan.Dispatches[i]; std::memcpy(constants,&job.Constants,sizeof(job.Constants));
            const bool core=job.Pass<3; buffers[1]=describe(core?work->Input->Allocation:VisibilityGeometry); buffers[2]=describe(core?work->Output->Allocation:work->Input->Allocation);
            decisionView=describe(NaniteRasterDecisions?NaniteRasterDecisions->Allocation:work->Input->Allocation);
            images[0]={DeferredSampler,MainDepth.View,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            images[1]={DeferredSampler,VisibilityImages[0].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        else
        {
            const auto drawIndex=i-computes; const auto* draw=drawIndex<frame.Draws.size()?&frame.Draws[drawIndex]:nullptr;
            NaniteRasterDrawConstants value; if (!BuildNaniteRasterDrawConstants(frame,draw,CreateInfo.Width,CreateInfo.Height,value)) return Fail("Invalid Vulkan Nanite material");
            if (postPreparation) value.Raster.Quality.W|=NaniteRasterPreserveMain;
            std::memcpy(constants,&value,sizeof(value)); buffers[1]=describe(VisibilityGeometry); buffers[2]=describe(work->Input->Allocation); buffers[3]=describe(work->Output->Allocation);
            const TextureHandle handles[]{draw?draw->Pipeline.DiffuseTexture:TextureHandle{},draw?draw->Pipeline.OpacityTexture:TextureHandle{}};
            for (unsigned j=0;j<2;++j)
            {
                VkImageView view=DummyColor.View;
                if (handles[j])
                { const auto found=Textures.find(handles[j].Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8 || found->second.Image.Layout!=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return Fail("Invalid Vulkan Nanite texture"); view=found->second.Image.View; }
                images[j]={ShadowSampler,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }
        for (unsigned binding=0;binding<count;++binding)
        {
            auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=sets[i]; write.dstBinding=binding; write.descriptorCount=1;
            write.descriptorType=binding==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(binding<imageStart || (!graphicsPass && binding==5))?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (!graphicsPass && binding==5) write.pBufferInfo=&decisionView;
            else if (binding<imageStart) write.pBufferInfo=&buffers[binding]; else write.pImageInfo=&images[binding-imageStart];
        }
        UpdateDescriptorSets(Device,count,writes,0,nullptr);
    }
    work->ComputeSets.assign(sets.begin(),sets.begin()+computes); work->GraphicsSets.assign(sets.begin()+computes,sets.end());
    if (postPreparation) NaniteRasterRetainedWork.push_back(NaniteRasterFrameWork);
    NaniteRasterPlan=std::move(plan); NaniteRasterFrameWork=std::move(work); return true;
}
bool PrepareNanitePostRaster(const VisibilityBufferDesc& frame)
{
    if (!NaniteRasterLoadPass)
    {
        // Load/store differs from the ordinary clear pass, but attachment formats
        // and subpasses remain compatible with the existing framebuffer/pipelines.
        VkAttachmentDescription attachments[3]{};
        for (unsigned i=0;i<3;++i)
        {
            auto& a=attachments[i]; a.format=i==2?VK_FORMAT_D32_SFLOAT:i==0?VK_FORMAT_R32_UINT:VK_FORMAT_R32G32_SFLOAT;
            a.samples=1; a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout=a.finalLayout=i==2?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference colors[]{{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},{1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},depth{2,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; subpass.colorAttachmentCount=2; subpass.pColorAttachments=colors; subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo desc{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; desc.attachmentCount=3; desc.pAttachments=attachments;
        desc.subpassCount=1; desc.pSubpasses=&subpass; desc.dependencyCount=1; desc.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&desc,nullptr,&NaniteRasterLoadPass),"Create Nanite post visibility pass")) return false;
    }
    return PrepareNaniteFrameRaster(frame,true);
}
void DispatchNaniteFrameRaster(unsigned index)
{
    const auto& job=NaniteRasterPlan.Dispatches[index]; auto& work=*NaniteRasterFrameWork;
    auto& destination=job.Pass<3?work.Output->Allocation:work.Input->Allocation;
    const auto depthLayout=MainDepth.Layout,idLayout=VisibilityImages[0].Layout;
    if (job.Pass==5)
    { TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,VisibilityImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); }
    const VkFlags shaderStages=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|(MeshStagesEnabled?VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT:0))|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer=destination.Buffer; barrier.size=destination.Size;
    CmdPipelineBarrier(CommandBuffer,shaderStages,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteRasterFrameLayouts[0],0,1,&work.ComputeSets[index],0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteRasterFramePipelines[job.Pass]); NaniteGeometryDispatch(CommandBuffer,job.GroupsX,job.GroupsY,1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,shaderStages,0,0,nullptr,1,&barrier,0,nullptr);
    if (job.Pass==5) { TransitionImage(CommandBuffer,MainDepth,depthLayout); TransitionImage(CommandBuffer,VisibilityImages[0],idLayout); }
    ++PendingFrame.Statistics.NaniteRasterDispatchCount;
}
bool DrawNaniteRasterPhase(const VisibilityBufferDesc& frame)
{
    if (frame.NaniteVisibility) ++PendingFrame.Statistics.NaniteOcclusionPhases;
    auto& work=*NaniteRasterFrameWork;
    const auto begin=[&]()
    {
        for (auto& image:VisibilityImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue clear[3]{}; VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; pass.renderPass=NaniteRasterPost?NaniteRasterLoadPass:VisibilityRenderPass; pass.framebuffer=VisibilityFramebuffer;
        pass.renderArea={{0,0},{MainColor.Width,MainColor.Height}}; pass.clearValueCount=3; pass.pClearValues=clear; CmdBeginRenderPass(CommandBuffer,&pass,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1}; const VkRect2D scissor{{0,0},{MainColor.Width,MainColor.Height}};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor);
    };
    for (unsigned i=0;i<NaniteRasterPlan.HardwareBoundary;++i) DispatchNaniteFrameRaster(i);
    if (frame.GpuHardware)
    { if (!DrawNaniteHardware(frame)) return false; }
    else
    {
        begin();
        for (unsigned order=0;order<frame.Draws.size();++order)
        {
            const unsigned i=NaniteRasterPost?unsigned(frame.Draws.size())-1-order:order;
            const auto& draw=frame.Draws[i]; const auto& range=frame.Scene->Draws[draw.Range];
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,NaniteRasterPipeline(draw.Pipeline.BackFaceCulling,range.Mirrored,false));
            CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,NaniteRasterFrameLayouts[1],0,1,&work.GraphicsSets[i],0,nullptr);
            CmdDraw(CommandBuffer,range.TriangleCount*VisibilityTriangleCorners,1,0,0);
        }
        CmdEndRenderPass(CommandBuffer);
    }
    for (unsigned i=NaniteRasterPlan.HardwareBoundary;i<NaniteRasterPlan.Dispatches.size();++i) DispatchNaniteFrameRaster(i);
    begin(); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,NaniteRasterPipeline(false,false,true));
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,NaniteRasterFrameLayouts[1],0,1,&work.GraphicsSets.back(),0,nullptr);
    CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer); ++PendingFrame.Statistics.NaniteRasterExportCount; return true;
}

bool DrawNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (frame.Scene->HasNativeGeometry() && !(frame.GpuInstances?DrawNaniteInstances():DrawNaniteVisibilityGeometry())) return false;
    PendingFrame.Statistics.NaniteOcclusionQueries+=unsigned(frame.NaniteClusters.size());
    if (!DrawNaniteRasterPhase(frame)) return false;
    if (frame.NaniteVisibility)
    {
        if (!RetestNaniteRaster(frame)) return false;
        if (!PrepareNanitePostRaster(frame)) return false;
        if (!DrawNaniteRasterPhase(frame)) return false;
    }
    return true;
}
